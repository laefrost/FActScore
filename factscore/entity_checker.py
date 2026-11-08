"""The entity annotation pipeline as a drop-in fact checker for FactScorer.

FactScorer's other checkers take one atomic fact at a time. This one works on
the whole generation: a single model call (Claude with web search, see
``eval/entity``) extracts the entity spans - people, organisations, places,
dates, quantities, citations - verifies each one and labels it Supported /
Not Supported / Insufficient Information, and ``eval.entity`` locates every
span in the generation text. So there is no atomic fact step, and the "atoms"
FactScorer reports are the spans themselves. As in the original pipeline,
Insufficient Information counts as unsupported.

Nothing of the pipeline is reimplemented here: this module renders its prompt,
sends it through the LM (concurrently across generations, through the LM's
prompt cache) and reads the spans back.
"""

import logging
import os
import re
import sys

from concurrent.futures import ThreadPoolExecutor

# annotation_pipeline defaults: annotate_completion(max_searches=10, max_tokens=8192)
DEFAULT_MAX_SEARCHES = 10
DEFAULT_MAX_TOKENS = 8192
SUPPORTED_LABEL = "Supported"


def _do_import_entity():
    from eval import entity
    return entity


def _import_entity():
    """The eval.entity package, adding factscore's parent directory to sys.path if needed.

    Same arrangement as the SAFE checker: running from the repository finds
    `eval` next to `factscore`; an installed factscore has it in site-packages.
    """
    try:
        return _do_import_entity()
    except ImportError:
        pass

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    try:
        return _do_import_entity()
    except ImportError as e:
        raise ModuleNotFoundError(
            "fact_checker='entity' needs the `eval` package that ships with FActScore "
            "(and pydantic), and it is not importable from {!r}. Either reinstall factscore "
            "from the repository (pip install --force-reinstall --no-deps <path-to-FActScore>) "
            "or put the repository root on PYTHONPATH.".format(parent_dir)) from e


_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
_punkt = None


def sentence_spans(text):
    """(start, end) character offsets of the sentences of `text`."""
    global _punkt
    try:
        if _punkt is None:
            from nltk.tokenize import PunktTokenizer
            _punkt = PunktTokenizer("english")
        return list(_punkt.span_tokenize(text))
    except Exception:  # pylint: disable=broad-exception-caught
        # punkt_tab missing or an older nltk: a plain punctuation split
        return [(m.start(), m.end()) for m in _SENTENCE_RE.finditer(text) if m.group().strip()]


def sentence_at(text, spans, start, length):
    """The sentence of `text` that the characters [start, start+length) fall into."""
    end = start + max(length, 1)
    for s, e in spans:
        if s < end and e > start:
            return text[s:e].strip()
    return text[start:end].strip()


class EntityChecker:
    """Runs the entity annotation over whole generations."""

    def __init__(self, lm, max_searches=DEFAULT_MAX_SEARCHES, max_tokens=DEFAULT_MAX_TOKENS,
                 prompt_template=None, prompt_path=None, min_similarity=0.8, max_workers=None):
        entity = _import_entity()
        self._entity = entity
        self.lm = lm
        self.max_searches = max_searches
        self.max_tokens = max_tokens
        self.min_similarity = min_similarity
        if prompt_path is not None:
            prompt_template = entity.load_prompt_template(prompt_path)
        self.prompt_template = prompt_template  # None: eval/entity/entity_annotation.prompt
        # one call per generation, so the generations are what gets parallelized
        self.max_workers = max_workers or getattr(lm, 'max_concurrency', 1)

        # the search tool lives on the LM (factscore.anthropic_lm.AnthropicModel)
        if hasattr(lm, "web_search_max_uses"):
            lm.web_search_max_uses = max_searches
        elif max_searches:
            raise ValueError(
                "The entity checker verifies entities with web search, which needs a Claude "
                "backend: construct FactScorer with model_name='claude-...' (e.g. "
                "'claude-opus-5'). To label from the model's own knowledge instead, pass "
                "entity_max_searches=0 (the LM here is %s)." % type(lm).__name__)
        elif not hasattr(lm, "web_search_max_uses"):
            logging.warning("%s has no web search; entities are labelled from the model's own knowledge.",
                            type(lm).__name__)

    def annotate_one(self, instruction, completion):
        """One generation through the pipeline; the spans come back as dicts."""
        prompt = self._entity.format_prompt(instruction, completion, self.prompt_template)
        output = self.lm.generate(prompt, max_output_length=self.max_tokens)
        text = output[0] if isinstance(output, tuple) else output
        metadata = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        metadata = metadata if isinstance(metadata, dict) else {}

        result = {"instruction": instruction,
                  "spans": None,
                  "error": None,
                  "response_text": text if isinstance(text, str) else None,
                  "stop_reason": metadata.get("stop_reason"),
                  "num_web_searches": metadata.get("num_web_searches"),
                  "usage": metadata.get("usage")}

        if not isinstance(text, str) or not text.strip():
            result["error"] = "empty answer from the model"
            return result

        spans = self._entity.parse_annotations(text, completion, min_similarity=self.min_similarity)
        result["spans"] = [span.model_dump() for span in spans]
        return result

    def annotate(self, items):
        """Annotate many (instruction, completion) pairs, one result per item, in order.

        A failure (API error, refusal, unreadable answer) is reported in the
        item's result under "error" with "spans" None; it does not stop the batch.
        """
        items = list(items)
        if not items:
            return []

        def run(item):
            instruction, completion = item
            try:
                return self.annotate_one(instruction, completion)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.error("Entity annotation failed for %r: %s", instruction[:80], e)
                return {"instruction": instruction, "spans": None, "error": f"{type(e).__name__}: {e}",
                        "response_text": None, "stop_reason": None, "num_web_searches": None, "usage": None}

        workers = min(self.max_workers, len(items))
        lm = self.lm
        if workers <= 1:
            results = [run(item) for item in items]
        else:
            # one cache flush at the end instead of one per worker crossing the
            # backend's save interval
            suspended = hasattr(lm, 'suspend_autosave')
            if suspended:
                lm.suspend_autosave = True
            try:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    results = list(pool.map(run, items))
            finally:
                if suspended:
                    lm.suspend_autosave = False
        if hasattr(lm, 'save_cache'):
            lm.save_cache()
        return results
