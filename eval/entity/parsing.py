"""Reading the annotator's JSON answer.

Ported from ``utils/parsing.py`` in hallucination_probes. Two changes: when
the answer contains a closed ```json fence, the JSON is taken from the last
such fence rather than from the outermost brackets of the whole text (the
model's narration can contain brackets of its own, e.g. citation markers),
and ``parse_json_items`` validates the items one by one so that a single
malformed item does not discard the whole annotation.
"""

import json
import logging
import re
from typing import Any, List, Type, TypeVar

from pydantic import BaseModel, TypeAdapter

from eval.entity.string_utils import normalize_text

try:
    from pydantic_core import from_json as _from_json
except ImportError:  # pragma: no cover - pydantic < 2
    _from_json = None

T = TypeVar('T', bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)
_JSON_RE = re.compile(r'(\{.*\}|\[.*\])', flags=re.DOTALL)


def _loads(json_str: str, allow_partial: bool) -> Any:
    """json.loads that, with `allow_partial`, also reads a JSON document cut off at the end."""
    if _from_json is not None:
        try:
            return _from_json(json_str, allow_partial=allow_partial)
        except TypeError:
            # pydantic_core without allow_partial (added in pydantic 2.7)
            return _from_json(json_str)
    return json.loads(json_str)


def extract_json(llm_response: str) -> str:
    """
    The JSON object or array in an LLM response that may contain surrounding text.

    Normalizes the text to handle encoding issues, then takes the JSON from the
    last closed markdown fence that holds one; without such a fence, the fences
    are removed and the outermost brackets of the whole text are used.

    Raises:
        ValueError: If no JSON structure is found
    """
    cleaned_response = normalize_text(llm_response)

    for block in reversed(_FENCE_RE.findall(cleaned_response)):
        json_match = _JSON_RE.search(block)
        if json_match:
            return json_match.group(0).strip()

    # Remove markdown code fences (```json or ```)
    cleaned_response = re.sub(
        r"```(?:json)?", "",
        cleaned_response,
        flags=re.IGNORECASE
    ).replace("```", "").strip()

    # Find JSON structure (object or array)
    json_match = _JSON_RE.search(cleaned_response)
    if not json_match:
        raise ValueError(
            f"No valid JSON object or array found in response: {llm_response[:200]}..."
        )

    return json_match.group(0).strip()


def parse_and_validate_json(
    llm_response: str,
    schema: Any,
    allow_partial: bool = False
) -> Any:
    """
    Parse and validate JSON from an LLM response.

    Args:
        llm_response: Raw LLM response that may contain JSON with surrounding text
        schema: Pydantic model or type hint for validation (e.g., MyModel or List[MyModel])
        allow_partial: Whether to allow partial JSON parsing (useful for truncated responses)

    Returns:
        Parsed and validated object(s) according to the schema

    Raises:
        ValueError: If no valid JSON found or validation fails
    """
    json_str = extract_json(llm_response)

    try:
        parsed = _loads(json_str, allow_partial)
        return TypeAdapter(schema).validate_python(parsed)
    except Exception as e:
        raise ValueError(
            f"Error parsing/validating JSON: {e}\n"
            f"JSON string: {json_str[:200]}..."
        ) from e


def parse_json_items(
    llm_response: str,
    model: Type[T],
    allow_partial: bool = True
) -> List[T]:
    """
    The list of `model` items in an LLM response, skipping items that do not validate.

    A truncated answer (`allow_partial`) yields the items that were complete.

    Raises:
        ValueError: If no JSON is found, it cannot be parsed, or it is not a list
    """
    json_str = extract_json(llm_response)

    try:
        parsed = _loads(json_str, allow_partial)
    except Exception as e:
        raise ValueError(
            f"Error parsing JSON: {e}\n"
            f"JSON string: {json_str[:200]}..."
        ) from e

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON array of items, got {type(parsed).__name__}: {json_str[:200]}...")

    items = []
    skipped = []
    for i, item_dict in enumerate(parsed):
        try:
            items.append(model.model_validate(item_dict))
        except Exception as e:
            skipped.append((i, str(e).splitlines()[0]))

    if skipped:
        logging.warning("Skipped %d of %d annotated items that did not validate as %s: %s",
                        len(skipped), len(parsed), model.__name__, skipped[:3])

    return items


def validate_dicts_to_pydantic(
    dicts: List[dict],
    model: Type[T],
    skip_invalid: bool = False
) -> List[T]:
    """
    Validate a list of dictionaries against a Pydantic model.

    Args:
        dicts: List of dictionaries to validate
        model: Pydantic model class to validate against
        skip_invalid: If True, skip invalid items instead of raising an error

    Returns:
        List of validated Pydantic model instances

    Raises:
        ValueError: If skip_invalid is False and validation fails for any item
    """
    validated = []

    for i, item_dict in enumerate(dicts):
        try:
            validated_item = model.model_validate(item_dict)
            validated.append(validated_item)
        except Exception as e:
            if skip_invalid:
                # Silently skip invalid items
                continue
            else:
                raise ValueError(
                    f"Validation failed for item {i}: {e}"
                ) from e

    return validated
