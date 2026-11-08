"""Claude as a factscore LM, with Anthropic's server-side web search.

The entity fact checker (``factscore.entity_checker``) needs a model that can
look things up: its prompt tells the model to "use web search to verify" every
entity. That is Anthropic's ``web_search`` server tool - the searches run on
Anthropic's side, inside the one ``messages`` call, and the answer comes back
in the same response. Nothing about the LM interface changes: ``generate``
still maps a prompt to ``(text, metadata)`` and goes through the same pickle
cache as the other backends, so a rerun over the same generations is free.

The API key comes from ``ANTHROPIC_API_KEY``, or from an ``anthropic.key``
file (next to ``api.key``); failing both, the SDK looks for an
``ant auth login`` profile.
"""

import logging
import os
import re

from factscore.lm import LM

# Anthropic's web search tool has two versions: the 2026-02-09 one (with
# result filtering) on Claude Opus 4.6 / Sonnet 4.6 and later, the original
# on older models such as claude-sonnet-4-20250514, the model the annotation
# pipeline was written for.
WEB_SEARCH_TOOL = "web_search_20260209"
LEGACY_WEB_SEARCH_TOOL = "web_search_20250305"
_LEGACY_WEB_SEARCH_MODELS = re.compile(r"claude-(3|haiku|sonnet-4-(2025|5)|opus-4-(2025|1|5))")

# These reject temperature / top_p / top_k with a 400, so a temperature is only
# sent to the other models.
_NO_SAMPLING_MODELS = re.compile(r"claude-(opus-5|opus-4-[78]|sonnet-5|fable|mythos)")

# A True/False verdict is a classification, so for it the models are asked not
# to think: thinking tokens are billed as output and count against the small
# max_tokens of a verdict call, which an unconstrained reasoning pass can eat
# up entirely. How that is requested differs per model: Sonnet 5.5 rejects
# thinking "disabled" and takes "between_tools" instead; Opus 5.5 / Fable /
# Mythos cannot turn thinking off at all, so the lowest effort is the closest
# thing; every other model (Haiku 5.5, Opus 5, the 4.x family) accepts
# "disabled".
_BETWEEN_TOOLS_THINKING_MODELS = re.compile(r"claude-sonnet-5-5")
_ALWAYS_THINKING_MODELS = re.compile(r"claude-(opus-5-5|fable|mythos)")


def no_thinking_kwargs(model_name):
    """AnthropicModel keyword arguments that make `model_name` answer without a reasoning pass."""
    if _BETWEEN_TOOLS_THINKING_MODELS.search(model_name):
        return {"thinking": {"type": "between_tools"}}
    if _ALWAYS_THINKING_MODELS.search(model_name):
        return {"effort": "low"}
    return {"thinking": {"type": "disabled"}}


# Models whose safety classifiers may decline a request. For them a server-side
# fallback is requested, so a declined annotation is re-run on another model
# instead of leaving the generation unannotated.
_FALLBACK_MODELS = re.compile(r"claude-(opus-5|fable|mythos)")
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# A web search turn is a server-side loop; when it reaches the server's
# iteration limit the response stops with `pause_turn` and is resumed by
# sending it back as is.
MAX_CONTINUATIONS = 5


def default_web_search_tool_type(model_name):
    return LEGACY_WEB_SEARCH_TOOL if _LEGACY_WEB_SEARCH_MODELS.search(model_name) else WEB_SEARCH_TOOL


def read_api_key(key_path="anthropic.key"):
    """ANTHROPIC_API_KEY, else the key file; None lets the SDK resolve its own credentials."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key and key.strip():
        return key.strip()
    if key_path and os.path.exists(key_path):
        with open(key_path) as f:
            key = f.read().strip()
        if key:
            return key
    return None


def block_type(block):
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)


def response_text(response):
    """The text blocks of a message, joined; server tool blocks are skipped."""
    texts = []
    for block in getattr(response, "content", None) or []:
        if block_type(block) == "text":
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if text:
                texts.append(text)
    return "\n".join(texts)


def count_blocks(response, type_name):
    return sum(block_type(block) == type_name for block in getattr(response, "content", None) or [])


def to_plain(obj):
    """A pickle- and json-friendly copy of an SDK object."""
    for method in ("to_dict", "model_dump"):
        convert = getattr(obj, method, None)
        if convert is not None:
            try:
                return convert()
            except Exception:  # pylint: disable=broad-exception-caught
                continue
    if isinstance(obj, dict):
        return obj
    return str(obj)


class AnthropicModel(LM):

    # the client is thread-safe, so independent prompts can be in flight at once
    max_concurrency = 8

    def __init__(self, model_name, cache_file=None, key_path="anthropic.key", max_concurrency=None,
                 temp=None, web_search_max_uses=0, web_search_tool_type=None, effort=None,
                 thinking=None, refusal_fallbacks=None, max_continuations=MAX_CONTINUATIONS,
                 timeout=600.0, max_retries=6):
        """
        model_name: the API model id, e.g. "claude-opus-5" (a "retrieval+" prefix is dropped).
        temp: sampling temperature; None sends none (the 4.7+ models reject one).
        web_search_max_uses: searches the model may run per prompt; 0 attaches no search tool.
        effort: "low" | "medium" | "high" | "xhigh" | "max", None for the API default.
        thinking: a thinking config dict passed through as is, None for the model default.
        refusal_fallbacks: ask for a server-side fallback model on a refusal; None
            enables it on the models that have such refusals (Opus 5, Fable, Mythos).
        """
        self.model_name = model_name[len("retrieval+"):] if model_name.startswith("retrieval+") else model_name
        self.key_path = key_path
        self.temp = temp
        self.web_search_max_uses = web_search_max_uses
        self.web_search_tool_type = web_search_tool_type or default_web_search_tool_type(self.model_name)
        self.effort = effort
        self.thinking = thinking
        if refusal_fallbacks is None:
            refusal_fallbacks = _FALLBACK_MODELS.search(self.model_name) is not None
        self.refusal_fallbacks = refusal_fallbacks
        self.max_continuations = max_continuations
        self.timeout = timeout
        self.max_retries = max_retries
        self.save_interval = 100
        self.client = None
        self._warned_temperature = False
        if max_concurrency is not None:
            self.max_concurrency = max_concurrency
        super().__init__(cache_file)

    def load_model(self):
        import anthropic  # deferred: only a Claude run needs the SDK
        # retries cover rate limits, 5xx and connection errors with backoff
        self.client = anthropic.Anthropic(api_key=read_api_key(self.key_path),
                                          timeout=self.timeout, max_retries=self.max_retries)
        self.model = self.model_name

    def tools(self):
        if not self.web_search_max_uses:
            return []
        return [{"type": self.web_search_tool_type,
                 "name": "web_search",
                 "max_uses": int(self.web_search_max_uses)}]

    def request_kwargs(self, messages, max_output_length):
        kwargs = {"model": self.model_name,
                  "max_tokens": int(max_output_length),
                  "messages": messages}
        tools = self.tools()
        if tools:
            kwargs["tools"] = tools
        if self.temp is not None:
            if _NO_SAMPLING_MODELS.search(self.model_name):
                if not self._warned_temperature:
                    logging.warning("%s does not accept a temperature; temp=%s is not sent.",
                                    self.model_name, self.temp)
                    self._warned_temperature = True
            else:
                kwargs["temperature"] = self.temp
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if self.thinking is not None:
            kwargs["thinking"] = self.thinking
        if self.refusal_fallbacks:
            # "default" lets Anthropic pick the fallback by refusal category
            kwargs["extra_headers"] = {"anthropic-beta": FALLBACK_BETA}
            kwargs["extra_body"] = {"fallbacks": "default"}
        return kwargs

    def _create(self, kwargs):
        # streamed so that a long web search turn cannot run into the request timeout
        with self.client.messages.stream(**kwargs) as stream:
            return stream.get_final_message()

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128, response_format=None):
        # response_format is factscore's OpenAI json_schema for the True/False
        # verdict; it is not translated, the verdict is read from the text
        del response_format
        self.maybe_autosave(self.save_interval)

        messages = [{"role": "user", "content": prompt}]
        responses = [self._create(self.request_kwargs(messages, max_output_length))]
        while responses[-1].stop_reason == "pause_turn" and len(responses) <= self.max_continuations:
            # the paused turn is sent back unchanged and the server resumes it
            messages = messages + [{"role": "assistant", "content": responses[-1].content}]
            responses.append(self._create(self.request_kwargs(messages, max_output_length)))

        final = responses[-1]
        if final.stop_reason == "refusal":
            # not cached: raising leaves the prompt out of the cache, so a later
            # run asks again instead of replaying the refusal forever
            details = to_plain(getattr(final, "stop_details", None))
            raise RuntimeError(f"{self.model_name} refused the request: {details}")

        text = "\n".join(t for t in (response_text(r) for r in responses) if t)
        metadata = {
            "raw_response": to_plain(final),
            "output_text": text,
            "stop_reason": final.stop_reason,
            "model": getattr(final, "model", self.model_name),
            "usage": to_plain(getattr(final, "usage", None)),
            "num_web_searches": sum(count_blocks(r, "server_tool_use") for r in responses),
            "continuations": len(responses) - 1,
            # no token logprobs on this API, so no P(True) for the verdict path
            "p_true": None,
        }
        if final.stop_reason == "max_tokens":
            logging.warning("%s hit max_tokens=%d; the answer is cut off.", self.model_name, max_output_length)
        return text, metadata
