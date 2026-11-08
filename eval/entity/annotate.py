"""Prompting the annotator and locating its spans in the completion.

Ported from ``annotation_pipeline/annotate.py`` in hallucination_probes. The
original called Claude through ``safetytooling``'s ``InferenceAPI`` with the
web search tool attached; here the model call is a plain ``generate(prompt)``
callable supplied by the caller (see ``factscore.entity_checker``), so this
module is only the prompt, the JSON reading and the span positioning.
"""

import logging
import os
from typing import Callable, List, Optional

from eval.entity.data_models import AnnotatedSpan
from eval.entity.parsing import parse_json_items
from eval.entity.string_utils import try_matching_span_in_text

logger = logging.getLogger(__name__)

PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entity_annotation.prompt")


def load_prompt_template(path: str = PROMPT_PATH) -> str:
    """The annotation prompt, with its {instruction} and {completion} placeholders checked."""
    with open(path, encoding="utf-8") as f:
        template = f.read().strip()
    if '{instruction}' not in template or '{completion}' not in template:
        raise ValueError(f"The annotation prompt {path} needs both {{instruction}} and {{completion}} placeholders.")
    return template


ENTITY_ANNOTATION_PROMPT_TEMPLATE = load_prompt_template()


def format_prompt(instruction: str, completion: str, prompt_template: Optional[str] = None) -> str:
    """Format the user prompt with the text to analyze"""
    template = prompt_template or ENTITY_ANNOTATION_PROMPT_TEMPLATE
    return template.replace(
        "{instruction}", instruction
    ).replace(
        "{completion}", completion
    )


def assign_span_positions(spans: List[AnnotatedSpan], text: str, min_similarity: float = 0.8) -> List[AnnotatedSpan]:
    """
    Assign positions to spans and convert to the expected format.

    Args:
        spans: List of annotated spans from the model
        text: The original text

    Returns:
        List of spans with positions added
    """
    results = []
    cur_idx = 0
    used_positions = set()

    for span in spans:

        closest_match, matched_idx = try_matching_span_in_text(
            span.span,
            text,
            cur_idx=cur_idx,
            min_similarity=min_similarity
        )

        if closest_match is None:
            logger.warning(f"Could not locate span {repr(span.span)} in text (total_n_spans: {len(spans)}).\nKeeping it in the dataset but removing the label and index.\n(This is not an error and happens roughly once for every completion)")
            span.label = None
            span.index = None
            results.append(span)
            continue

        if matched_idx is not None and all(pos in used_positions for pos in range(matched_idx, matched_idx+len(closest_match))):
            logger.warning(f"Span {repr(span.span)} matched at same position as already-matched span {repr(text[matched_idx:matched_idx+len(closest_match)])}")
            continue

        if closest_match != span.span:
            logger.info(f"Span {repr(span.span)} matched to {repr(closest_match)}")

        span.index = matched_idx
        span.span = closest_match

        results.append(span)
        used_positions.update(range(matched_idx, matched_idx+len(closest_match)))
        cur_idx = max(cur_idx, matched_idx + len(closest_match))

    return results


def parse_annotations(response_text: str, completion: str, min_similarity: float = 0.8) -> List[AnnotatedSpan]:
    """The annotated spans in the model's answer, located in `completion`.

    Raises ValueError when the answer holds no readable JSON array.
    """
    annotated_spans = parse_json_items(
        response_text,
        AnnotatedSpan,
        allow_partial=True,  # try to parse the response even if it's cut off
    )

    annotated_spans = assign_span_positions(annotated_spans, completion, min_similarity=min_similarity)

    logger.info(f"Successfully annotated {len(annotated_spans)} spans")

    return annotated_spans


def annotate_completion(
    instruction: str,
    completion: str,
    generate: Callable[[str], str],
    prompt_template: Optional[str] = None,
    min_similarity: float = 0.8,
) -> List[AnnotatedSpan]:
    """
    Annotate spans in the provided text completion.

    Args:
        instruction: The instruction/prompt that was given to the model
        completion: The completion text to analyze for spans
        generate: Maps the rendered annotation prompt to the model's answer text.
            The model is expected to have web search available (the prompt asks
            it to verify every entity with a search).
        prompt_template: Replaces the default entity_annotation.prompt
        min_similarity: Threshold for locating spans that are not verbatim in the completion

    Returns:
        List of annotated spans with their positions
    """
    user_prompt = format_prompt(instruction, completion, prompt_template)
    response_text = generate(user_prompt)
    return parse_annotations(response_text, completion, min_similarity=min_similarity)
