"""Entity-level fact checking: the annotation pipeline from hallucination_probes.

Ported from ``annotation_pipeline/`` in
https://github.com/obalcells/hallucination_probes (Apache 2.0, see the LICENSE
file in this directory). The pipeline makes one model call per completion: the
model extracts the entity spans (people, organisations, locations, dates,
quantities, citations), verifies each with web search and labels it
``Supported`` / ``Not Supported`` / ``Insufficient Information``. The spans are
then located in the completion text (``annotate.assign_span_positions``).

Left out of the port: ``run.py`` (HuggingFace dataset loading and pushing, the
async driver) and the ``safetytooling`` inference client. The model call is
made by the caller - in FActScore by ``factscore.entity_checker`` through a
``factscore.anthropic_lm.AnthropicModel``, which carries Anthropic's web search
tool. The prompt is the original ``entity_annotation.prompt``, unchanged.
"""

from eval.entity.data_models import LABELS, AnnotatedSpan
from eval.entity.annotate import (ENTITY_ANNOTATION_PROMPT_TEMPLATE, annotate_completion,
                                  assign_span_positions, format_prompt, load_prompt_template,
                                  parse_annotations)
