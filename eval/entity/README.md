# Entity annotation pipeline (from hallucination_probes)

A port of the [`annotation_pipeline`](https://github.com/obalcells/hallucination_probes/tree/main/annotation_pipeline)
of *Real-Time Detection of Hallucinated Entities in Long-Form Generation*
(Obeso et al.), used in FActScore as the `entity` fact checker
(`--fact_checker entity`, see the top-level README).

## What it does

For one (instruction, completion) pair the annotator model - Claude with
Anthropic's web search tool - is asked once, with `entity_annotation.prompt`, to

1. extract the minimal spans of the entities in the completion: people,
   organisations, locations, dates and events, quantities, citations;
2. verify each one with web search;
3. label it `Supported`, `Not Supported` or `Insufficient Information`, with a
   short `verification_note`.

The answer is a JSON array of `{span, label, verification_note}` objects.
`annotate.assign_span_positions` then locates every span in the completion
(verbatim first, then by ROUGE-L similarity over a sliding window) and records
its character `index`. A span that cannot be located is kept but loses its
label. Downstream, `Insufficient Information` is treated like `Not Supported`.

## Files

| file | from | role |
| --- | --- | --- |
| `entity_annotation.prompt` | `annotation_pipeline/entity_annotation.prompt` | the prompt, unchanged |
| `data_models.py` | `annotation_pipeline/data_models.py` | `AnnotatedSpan` |
| `annotate.py` | `annotation_pipeline/annotate.py` | prompt rendering, span positioning |
| `parsing.py` | `utils/parsing.py` | reading the JSON answer |
| `string_utils.py` | `utils/string_utils.py` | fuzzy span matching (ROUGE-L computed locally, no `rouge_score` dependency) |

Not ported: `run.py` (HuggingFace dataset loading/pushing, async driver) and
the `safetytooling` inference client. The model call is made by the caller:
`factscore/entity_checker.py` renders the prompt and sends it through a
`factscore/anthropic_lm.py` `AnthropicModel`, which carries the web search
tool and FActScore's usual on-disk prompt cache.

## Cost

One model call per completion, with up to `max_searches` web searches
(10 by default, 15 in the original run script). Anthropic bills web search per
search on top of the tokens, and every search's results are added to the
context, so `max_searches` is the main cost knob. The original defaults were
`claude-sonnet-4-20250514`, `temperature=0`, `max_tokens=8192`.
