# FActScore

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![arxiv](https://img.shields.io/badge/arXiv-2305.14251-b31b1b.svg)](https://arxiv.org/abs/2305.14251)
[![PyPI version factscore](https://badge.fury.io/py/factscore.svg)](https://pypi.python.org/pypi/factscore/)
[![Downloads](https://pepy.tech/badge/factscore)](https://pepy.tech/project/factscore)

This is the official release accompanying our EMNLP 2023 paper, [FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation](https://arxiv.org/abs/2305.14251). FActScore is available as a PIP package as well.

If you find FActScore useful, please cite:
```
@inproceedings{ factscore,
    title={ {FActScore}: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation },
    author={ Min, Sewon and Krishna, Kalpesh and Lyu, Xinxi and Lewis, Mike and Yih, Wen-tau and Koh, Pang Wei and Iyyer, Mohit and Zettlemoyer, Luke and Hajishirzi, Hannaneh },
    year={ 2023 },
    booktitle = { EMNLP },
    url={ https://arxiv.org/abs/2305.14251 }
}
```

## Announcement
* **11/04/2023**: The data we release includes human annotations of factual precision reported in Section 3 of [the paper](https://arxiv.org/abs/2305.14251). If you want to download these human annotated data *only*, without other data, you can download it directly from [this Google Drive link](https://drive.google.com/drive/folders/1kFey69z8hGXScln01mVxrOhrqgM62X7I?usp=sharing). We are also releasing FActScore results of 12 different LMs reported in Section 4.3 of the paper, in case you want to obtain them without running the code. Please refer to [here](#factscore-results-of-the-unlabeled-data).

## Install
<!-- ```
conda create -n fs-env python=3.9
conda activate fs-env
pip install -r requirements.txt
``` -->

Make a new Python 3.7+ environment using `virtualenv` or `conda`.

```bash
pip install --upgrade factscore
python -m spacy download en_core_web_sm
```

## Download the data

```bash
python -m factscore.download_data --llama_7B_HF_path "llama-7B"
```

This command does the following.
1. Download the knowledge source and example data.
2. Take the LLAMA 7B model and reconstruct Inst-LLAMA. This requires having access to HuggingFace weights of the LLAMA-7B model, which are added to the `--llama_7B_HF_path` flag. Follow [this guide](https://huggingface.co/docs/transformers/main/model_doc/llama) in order to obtain those weights. Skip the `--llama_7B_HF_path` if you would only like to use the ChatGPT version of FActScore.

**Optional flags**:
- `--data_dir`: directory to store the knowledge source and example data. `.cache/factscore` by default.
- `--model_dir`: directory to store Inst-LLAMA weights. `.cache/factscore` by default.

**Troubleshooting**:
- If you get a `ERROR 429: Too Many Requests` error while downloading the DB file, please download the DB from [this Google Drive link](https://drive.google.com/drive/folders/1kFey69z8hGXScln01mVxrOhrqgM62X7I?usp=sharing) and place it under `--data_dir` (`.cache/factscore` by default).
- If everything else fails, consider downloading the files manually from [this link](https://drive.google.com/drive/folders/1kFey69z8hGXScln01mVxrOhrqgM62X7I?usp=sharing) and placing them in `--data_dir` and `--model_dir`, see [`factscore/download_data.py`](factscore/download_data.py) for more details.


## Running FActScore using a command line

We expect running FActScore costs about $1 of the API cost per 100 sentences. For instance, if you have 100 generations, each with 5 sentences on average, it costs $5 in total.

```bash
python -m factscore.factscorer --input_path {input_path} --model_name {estimator_name} --openai_key {openai_key}
```

- `--input_path` can be something like `data/unlabeled/InstructGPT.jsonl`. It should be a `.jsonl` format where each line contains `topic` (a topic entity that corresponds to the Wikipedia title) and `output` (a generation from the model).
- `--model_name`: `retrieval+ChatGPT` and `retrieval+llama+npm` (You can also use `retrieval+ChatGPT+npm` or `retrieval+llama` but we recommend the former two.)
  Claude verdict models: `retrieval+claude-haiku-5-5` (the price and size match
  for `retrieval+gpt-5.6-luna`) and `retrieval+claude-sonnet-5-5` (the step up in
  quality, about ten times the price). They need `ANTHROPIC_API_KEY` or
  `--anthropic_key` and no OpenAI key; see [Claude verdict models](#claude-verdict-models).
- `--openai_key`: File containing OpenAI API Key.

**Optional flags**:
- `--data_dir`: Directory containing knowledge source, etc. `.cache/factscore` by default.
- `--model_dir`: Directory containing Inst-LLAMA weights. Skip if your `model_name` doesn't include `llama`. `.cache/factscore` by default.
- `--cache_dir`: Directory containing cache from API/models. `.cache/factscore` by default.
- `--use_atomic_facts`: If specified, it uses model-generated atomic facts released as part of our data instead of running the atomic fact generator. This will allow reproducing our results with no (or little if it still uses ChatGPT) cost. You can't specify it if you are running new model generations.
- `--gamma`: A hyperparameter for length penalty. `10` by default. It penalizes the score if the number of facts is less than `gamma`. `10` roughly corresponds to 2 sentences, so would penalize if the generation has less than 2 sentences. Usually, this would not change the ranking between systems unless some systems generate overly short responses all the time (e.g., models trained on NLP datasets without long-form generation tasks may do so). If you would like to turn off the length penalty completely, specify `--gamma 0`.
- `--n_samples`: If specified, it runs the model on a subset of the data.
- `--verbose`: If specified, it shows the progress bar.
- `--print_rate_limit_error`: It specified, it prints out rate limit errors from OpenAI API.
- `--cost_estimate`: This flag decides the type of OpenAI API cost estimation that we provide before calling it. It can be `"consider_cache"` (default) or `"ignore_cache"`.
- `--abstain_detection`: This flag optionally enables automatic detection of abstained responses. By default this is disabled, but it is recommended to add your own function tailored to your model. The currently supported detectors are `"generic"` and `"perplexity_ai"`, and their implementations can be found in [`factscore/abstain_detection.py`](factscore/abstain_detection.py). There are two methods to add your own abstain function: a) clone our GitHub repository to install `factscore` locally (`pip install --editable .`), and then add your function to [`factscore/abstain_detection.py`](factscore/abstain_detection.py) directly; b) process your abstain detection outside our package, and use empty strings in the `output` key for the JSONL file used in `--input_path`.
- `--knowledge_source`: In case the default knowledge source (Wikipedia - 2023/04/01) will not be used, preprocess it using the [instructions below](#To-use-a-custom-knowledge-source), and then specify the knowledge_source name under this flag.

## To evaluate your own LM

There're two sets of prompt entities, `data/labeled/prompt_entities.txt` (183 entities) and `data/unlabeled/prompt_entities.txt` (500 entities). Each line contains the name of the person (which is also a corresponding Wikipedia title). You can use the labeled version if you want to be compatible with the data under `data/labeled` (Section 3 and Section 4.2 in the paper), and use the unlabeled version if you want to be compatible with the data under `data/unlabeled` (Section 4.3 in the paper).

You can prompt your LM with your own prompt (we used `Question: Tell me a bio of <entity>.`) and use the following code.

```python
from factscore.factscorer import FactScorer

fs = FactScorer(openai_key="...")

# topics: list of strings (human entities used to generate bios)
# generations: list of strings (model generations)
out = fs.get_score(topics, generations, gamma=10)
print (out["score"]) # FActScore
print (out["init_score"]) # FActScore w/o length penalty
print (out["respond_ratio"]) # % of responding (not abstaining from answering)
print (out["num_facts_per_response"]) # average number of atomic facts per response
```

Alternatively, you can create a .jsonl file, where each line has `topic` (entity name, exactly same as the one from `.txt` file) and `output` (generation from LM), and then use a command line [above](#Running-FActScore-using-a-command-line).

We recommend using (A) `FactScorer(model_name="retrieval+ChatGPT")` (default) or (B) `FactScorer(model_name="retrieval+llama+npm")`. They have 0.99 Pearson correlation. Here're results of a range of models, which you can easily reproduce through [these command lines](#Running-FActScore-using-a-command-line).

| Model | % respond | # facts | FActScore from (A) | FActScore from (B) |
|---|---|---|---|---|
| [GPT-4](https://arxiv.org/abs/2303.08774)                                         | 88.2 | 60.8 | 73.1 | 59.9 |
| [ChatGPT](https://openai.com/blog/chatgpt)                                        | 84.2 | 37.0 | 71.6 | 60.4 |
| [Alpaca 65B](https://crfm.stanford.edu/2023/03/13/alpaca.html)                    | 100.0 | 17.1 | 55.6 | 46.3 |
| [InstructGPT](https://openai.com/research/instruction-following)                  | 99.8 | 27.7 | 52.8 | 41.7 |
| [Alpaca 13B](https://crfm.stanford.edu/2023/03/13/alpaca.html)                    | 100.0 | 16.6 | 47.7 | 40.3 |
| [Vicuna 13B](https://lmsys.org/blog/2023-03-30-vicuna/)                           | 76.6 | 50.9 | 46.6 | 40.7 |
| [Alpaca 7B](https://crfm.stanford.edu/2023/03/13/alpaca.html)                     | 100.0 | 17.4 | 39.7 | 36.5 |
| [Vicuna 7B](https://lmsys.org/blog/2023-03-30-vicuna/)                            | 91.0 | 45.6 | 38.9 | 36.9 |
| [MPT Chat 7B](https://www.mosaicml.com/blog/mpt-7b)                               | 88.8 | 37.3 | 30.1 | 27.9 |
| [Oasst Pythia 12B](https://huggingface.co/OpenAssistant/oasst-sft-1-pythia-12b)   | 100.0 | 39.7 | 25.1 | 20.8 |
| [Dolly 12B](https://huggingface.co/databricks/dolly-v2-12b)                       | 100.0 | 24.6 | 21.7 | 17.1 |
| [StableLM tuned 7B](https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b)   | 66.6 | 38.0 | 17.3 | 16.3 |

`% respond` (% of responding instead of abstaining from answering) and `# facts` (# of atomic facts per valid response) indicate "factual recall" (how many pieces of information the model gives) and FActScore indicates "factual precision" (how accurate each piece of information the model gives is).

## To use a custom knowledge source

By default, FActScore uses Wikipedia dump from 2023/04/01. But you can also use your own knowledge source!

The knolwedge source should be ready in a `.jsonl` format, where each line is a dictionary containing `title` and `text`. `text` can either be a string or a list of strings (e.g., sections).

```python
from factscore.factscorer import FactScorer

fs = FactScorer()

# this will create a database using your file
# for English Wikipedia (18GB)), it takes ~8 hours
# once DB file is created, you can reuse it by only specifying `db_path`
fs.register_knowledge_source(name_of_your_knowledge_source,
                             data_path=path_to_jsonl_file,
                             db_path=path_to_output_db_file)

# now, when you compute a score, specify knowledge source to use
out = fs.get_score(topics, generations, knowledge_source=name_of_your_knowledge_source)
print (out["score"]) # FActScore
print (out["respond_ratio"]) # % of responding (not abstaining from answering)
print (out["num_facts_per_response"]) # average number of atomic facts per response
```

To see an example of constructing the ACL anthology knowledge source, see [`preprocessing/preprocess_acl.py`](preprocessing/preprocess_acl.py).

## FActScore results of the unlabeled data

You can easily reproduce FActScore results of 12 different LMs reported in Section 4.3 of [the paper](https://arxiv.org/abs/2305.14251) using this code. However, if you would like to obtain their predictions without running the code, you can download it from [this Google Drive link](https://drive.google.com/drive/folders/1kFey69z8hGXScln01mVxrOhrqgM62X7I?usp=sharing).

Each file corresponds to the subject LM (LM that generates responses that we are validating). Each line is a dictionary:
- `prompt`: the initial prompt fed into the LM
- `facts`: atomic facts decomposed by the model
- `LLAMA+NP_labels`: labels to facts, verified by LLAMA+NP
- `ChatGPT_labels`: labels to facts, verified by ChatGPT

Note that the number of lines may be less than 500, because it excludes the cases where the model abstains from responding (e.g., it says "I don't know"). You can do `# of lines / 500` to calculate the response ratio.

If you unzip the data and run the following code for verification, you will be able to get statistics that exactly match the statistics reported in the paper (Table 5 and Figure 3).
```python
dirname = "factscore-unlabeled-predictions"
for fn in os.listdir(dirname):
    chatgpt_fs = []
    llama_fs = []
    n_facts = []
    with open(os.path.join(dirname, fn)) as f:
        for line in f:
            dp = json.loads(line)
            n_facts.append(len(dp["facts"]))
            if "ChatGPT_Labels" in dp:
                chatgpt_fs.append(np.mean([l=="S" for l in dp["ChatGPT_Labels"]]))
            llama_fs.append(np.mean([l=="S" for l in dp["LLAMA+NP_Labels"]]))
    print ("Model=%s\t(%.1f%% responding, %.1f facts/response)\tFactScore=%.1f (ChatGPT)\t%.1f (LLAMA)" % (
        fn.split(".")[0], len(n_facts)*100/500, np.mean(n_facts), np.mean(chatgpt_fs)*100, np.mean(llama_fs)*100
    ))
```

## OpenAI Batch API and recovering a lost batch

With `--openai_batch_api` (or `FactScorer(use_openai_batch_api=True)`) the
True/False verdict prompts are submitted through the asynchronous Batch API in
chunks of `--openai_batch_size` requests. Each finished chunk is written to the
LM cache immediately, so a restarted run only resubmits unfinished chunks.

If the process loses contact with a batch that OpenAI goes on to complete,
download its output file from the dashboard (named like
`batch_<id>_output.jsonl`) and feed it back in instead of paying for the
prompts again:

```bash
python -m factscore.factscorer --input_path data/unlabeled/InstructGPT.jsonl \
    --model_name "retrieval+gpt-5.6-luna" --openai_batch_api \
    --batch_output_files batch_6a99708aa4c48190b42479cb03e60fa1_output.jsonl
```

or from Python, before the usual `get_score` call:

```python
fs = FactScorer(model_name="retrieval+gpt-5.6-luna", use_openai_batch_api=True)
fs.load_batch_outputs(["batch_6a99708aa4c48190b42479cb03e60fa1_output.jsonl"])
out = fs.get_score(topics=topics, generations=generations)
```

The output lines carry only a `custom_id`, not the prompt, so the prompts are
read from the batch's input file. By default it is fetched from OpenAI using
the batch id in the filename (or `batch_ids=[...]`); pass the downloaded input
JSONL(s) as `inputs=[...]` / `--batch_input_files` to stay offline. With only
a batch id, `fs.load_batch_outputs(batch_ids=["batch_..."])` fetches the
output file too. The verdicts go into the LM cache, so the following
`get_score` reuses them and submits a new batch only for prompts the recovered
batch did not answer.

## Fact checkers

`--fact_checker` (or `FactScorer(fact_checker=...)`) picks how the generation is
checked once it has been split into atomic facts:

- `factscore` (default): retrieve passages from the knowledge source and ask the
  LM for a True/False verdict per atomic fact.
- `safe`: [SAFE](eval/safe/README.md) - revise each atomic fact to be
  self-contained, drop the ones irrelevant to the prompt, and ground the rest
  in Google Search (needs a Serper key).
- `entity`: the [entity annotation pipeline](eval/entity/README.md) of
  [hallucination_probes](https://github.com/obalcells/hallucination_probes/tree/main/annotation_pipeline).
  It does not use atomic facts at all: one Claude call per generation, with
  Anthropic's web search tool, extracts the entity spans (people,
  organisations, locations, dates, quantities, citations), verifies each one
  and labels it `Supported`, `Not Supported` or `Insufficient Information`.
  The spans are then located in the generation text.

### Claude verdict models

The default `factscore` checker can ask a Claude model for the True/False
verdicts instead of an OpenAI one:

```bash
export ANTHROPIC_API_KEY=...   # or put the key in anthropic.key / --anthropic_key
python -m factscore.factscorer --input_path {input_path} --model_name retrieval+claude-haiku-5-5
```

```python
fs = FactScorer(model_name="retrieval+claude-haiku-5-5")   # or retrieval+claude-sonnet-5-5
```

- `retrieval+claude-haiku-5-5` is the equivalent of `retrieval+gpt-5.6-luna`:
  the small tier, at $0.10 / $0.50 per 1M input / output tokens against Luna's
  $0.20 / $1.20. `retrieval+claude-sonnet-5-5` ($2 / $10) is the step up.
  Both ids also work without the `retrieval+` prefix (the verdict is then
  given without retrieved passages, like `gpt-4o-mini` vs `retrieval+gpt-4o-mini`).
- The verdict is a classification, so the model is asked not to think
  (`thinking: disabled` on Haiku, `thinking: between_tools` on Sonnet 5.5,
  which rejects `disabled`; `anthropic_lm.no_thinking_kwargs` picks the form).
  The `entity` checker leaves thinking on.
- The Anthropic API returns no logprobs, so `p_true` is `None` and the verdict
  is parsed from the one-word answer, as for the OpenAI reasoning models.
- Verdicts are cached in `{cache_dir}/{model id}.pkl`, shared between the
  `retrieval+` and plain forms (the prompts differ, so the entries do not
  collide).
- `--openai_batch_api` / `use_openai_batch_api=True` sends the verdicts through
  Anthropic's [Message Batches API](https://platform.claude.com/docs/en/build-with-claude/batch-processing)
  instead: half price, results usually within an hour (24h at most). The
  `openai_batch_size` / `openai_max_active_batches` / `batch_poll_interval`
  settings apply as for OpenAI. Each ended batch is cached at once, so a
  restarted run resubmits only what is missing. `load_batch_outputs` (recovery
  of a lost OpenAI batch) has no Anthropic counterpart yet.

### The `entity` checker

```bash
export ANTHROPIC_API_KEY=...   # or put the key in anthropic.key
python -m factscore.factscorer --input_path data/unlabeled/InstructGPT.jsonl \
    --model_name claude-opus-5 --fact_checker entity --entity_max_searches 10
```

```python
fs = FactScorer(model_name="claude-opus-5", fact_checker="entity", entity_max_searches=10)
out = fs.get_score(topics=topics, generations=generations, questions=questions,
                   do_matching=True, gen_words=gen_words, word_tokens=word_tokens)
```

- `--model_name` must be a Claude model id (`claude-opus-5`, `claude-sonnet-5`,
  ... ; the original pipeline ran on `claude-sonnet-4-20250514`). The model
  is used through [`factscore/anthropic_lm.py`](factscore/anthropic_lm.py),
  which attaches the web search tool and caches answers in
  `{cache_dir}/{model_name}.pkl` like the other backends.
- **Atomic facts** (the default `factscore` checker, not the entity one) are
  generated by `claude-haiku-5-5` (`atomic_facts.ATOMIC_FACT_MODEL`) since
  OpenAI retired `gpt-3.5-turbo-instruct`. It needs the same Anthropic key
  (`ANTHROPIC_API_KEY` or `--anthropic_key`, default `anthropic.key`); the
  prompt carries an explicit "one fact per line, starting with `- `" instruction
  and answers are cached in `{cache_dir}/atomic_facts_claude-haiku-5-5.pkl`.
- `questions` (or, failing that, `topics`) is what the annotator is told the
  generation answers; the atomic fact generator is not run, so
  `--use_atomic_facts` / `atomic_facts` only decide which rows are skipped.
- **Cost**: one call per generation plus up to `--entity_max_searches` web
  searches (10 by default; 15 in the original run script). Anthropic bills
  each search on top of the tokens, and the search results are added to the
  context, so this is the main cost knob. `--entity_max_searches 0` labels from
  the model's own knowledge without searching.
- **Output**: each decision is one span: `atom` is the span text, `annotation`
  the pipeline's label, `verification_note` its reasoning, `span_index` the
  character offset in the generation. `is_supported` is `True` for
  `Supported`, `False` for `Not Supported` and `Insufficient Information` (as
  the pipeline treats them downstream), and `None` (excluded from the score,
  `parse_failed`) for a span that could not be located in the generation.
  With `do_matching`, located spans are mapped onto `gen_words` /
  `word_tokens` by character offset rather than by word matching. The
  per-generation entry in `scores` carries the model's raw answer, its stop
  reason and the number of searches under `entity`.
- On Claude Opus 5 / Fable models the request asks for a server-side fallback
  model (`fallbacks: "default"`) so that a safety refusal does not leave the
  generation unannotated; pass `refusal_fallbacks=False` to `AnthropicModel`
  to switch that off.
