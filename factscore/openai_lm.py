from factscore.lm import LM
import openai
import math
import sys
import time
import numpy as np
import logging
import json
import os
import re

from collections import namedtuple
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from factscore.lm import LM
from openai import OpenAI


# A True/False verdict is a classification, not something to think through, so
# the reasoning models are asked for no reasoning at all. Reasoning models bill
# hidden reasoning tokens as output, and at any effort above "none" a one-word
# verdict can cost hundreds of output tokens.
Route = namedtuple("Route", "needle api_model is_reasoning reasoning_effort")

# Order matters: the first substring hit wins, which is how the original
# if/elif chain resolved names like "retrieval+gpt-4o-mini".
MODEL_ROUTES = (
    Route("ChatGPT", "gpt-5-mini", True, None),
    Route("gpt-5.6-luna", "gpt-5.6-luna", True, "none"),
    Route("gpt-5.6-terra", "gpt-5.6-terra", True, "none"),
    Route("gpt-5-mini", "gpt-5-mini", True, None),
    Route("gpt-4o-mini", "gpt-4o-mini", False, None),
    Route("gpt-4.1-mini", "gpt-4.1-mini", False, None),
    Route("gpt-4.1-nano", "gpt-4.1-nano", False, None),
)

# Reasoning models also reject temperature and logprobs, so those are only sent
# to the others. max_output_tokens is withheld from them too: if reasoning does
# happen it can consume the whole budget and return an empty output_text.

MAX_RETRIES = 8
MAX_BACKOFF_SECONDS = 60


def resolve_route(model_name):
    for route in MODEL_ROUTES:
        if route.needle in model_name:
            return route
    raise NotImplementedError(f"Unknown model: {model_name}")


class OpenAIModel(LM):

    # the OpenAI client is thread-safe, so verdicts for independent atoms can be
    # requested in parallel
    max_concurrency = 8

    def __init__(self, model_name, cache_file=None, key_path="api.key", max_concurrency=None,
                 temp=0.0, reasoning_effort=None, use_batch_api=False,
                 batch_poll_interval=30.0, batch_completion_window="24h",
                 openai_batch_size=5000, openai_max_active_batches=4):
        self.model_name = model_name
        self.key_path = key_path
        # a True/False verdict is a classification; sampling only adds noise and
        # makes runs unreproducible
        self.temp = temp
        # overrides the route's default; ignored by non-reasoning models
        self.reasoning_effort = reasoning_effort
        self.save_interval = 100
        self.client = None
        self.use_batch_api = use_batch_api
        self.batch_poll_interval = batch_poll_interval
        self.batch_completion_window = batch_completion_window
        if openai_batch_size <= 0:
            raise ValueError("openai_batch_size must be positive")
        if openai_max_active_batches <= 0:
            raise ValueError("openai_max_active_batches must be positive")
        self.openai_batch_size = int(openai_batch_size)
        self.openai_max_active_batches = int(openai_max_active_batches)
        if max_concurrency is not None:
            self.max_concurrency = max_concurrency
        super().__init__(cache_file)

    def load_model(self):
        # API key is read automatically from OPENAI_API_KEY
        # Environment variable must be set
        self.client = OpenAI()
        self.model = self.model_name

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128, response_format = None):
        self.maybe_autosave(self.save_interval)

        if self.model_name == "InstructGPT":
            response = call_instruct_model(
                client=self.client,
                prompt=prompt,
                model_name="gpt-3.5-turbo-instruct",
                temp=self.temp,
                max_output_tokens=max_output_length
            )
            return response["output_text"], response

        route = resolve_route(self.model_name)
        response = call_chat_model(
            client=self.client,
            prompt=prompt,
            model_name=route.api_model,
            temp=self.temp,
            max_output_tokens=max_output_length,
            response_format=response_format,
            is_reasoning=route.is_reasoning,
            reasoning_effort=self.reasoning_effort or route.reasoning_effort,
        )
        return response["output_text"], response

    def generate_batch(self, prompts, sample_idx=0, max_sequence_length=2048,
                       max_output_length=128, response_format=None, max_workers=None):
        """Generate prompts concurrently or through the OpenAI Batch API.

        The Batch API path is synchronous from the caller's perspective: it
        uploads one JSONL file, waits for the batch to finish, downloads the
        result file, updates the normal LM cache, and returns outputs in the
        same order as ``prompts``.
        """
        if not self.use_batch_api:
            return super().generate_batch(
                prompts, sample_idx=sample_idx,
                max_sequence_length=max_sequence_length,
                max_output_length=max_output_length,
                response_format=response_format, max_workers=max_workers)

        if self.model_name == "InstructGPT":
            raise NotImplementedError("Batch API mode is implemented for /v1/responses only.")

        if self.model is None:
            self.load_model()

        def key_for(prompt):
            return f"{prompt.strip()}_{sample_idx}"

        outputs_by_key = {}
        uncached_by_key = {}
        with self.cache_lock:
            for prompt in prompts:
                key = key_for(prompt)
                if key in self.cache_dict:
                    outputs_by_key[key] = self.cache_dict[key]
                elif key not in uncached_by_key:
                    uncached_by_key[key] = prompt.strip()

        if uncached_by_key:
            keys = list(uncached_by_key)
            batch_prompts = [uncached_by_key[key] for key in keys]
            batch_outputs = self._generate_openai_batch(
                batch_prompts, sample_idx=sample_idx,
                max_output_length=max_output_length,
                response_format=response_format)
            for key, output in zip(keys, batch_outputs):
                outputs_by_key[key] = output

        return [outputs_by_key[key_for(prompt)] for prompt in prompts]

    def _generate_openai_batch(self, prompts, sample_idx=0, max_output_length=128,
                               response_format=None):
        """Run uncached prompts through multiple bounded OpenAI Batch jobs.

        At most ``openai_max_active_batches`` chunks are in flight at a time,
        counting both batches OpenAI is running and uploads still on their way
        there. Uploads run concurrently: each is a multi-megabyte request, and
        doing them one after another used to serialise the start of every run.
        Each completed chunk is written to the normal LM cache immediately, so
        a restarted run only resubmits prompts from unfinished chunks.
        """
        if not prompts:
            return []

        chunks = [
            (offset, prompts[offset:offset + self.openai_batch_size])
            for offset in range(0, len(prompts), self.openai_batch_size)
        ]
        outputs = [None] * len(prompts)
        pending = list(chunks)
        submitting = []
        active = {}

        with ThreadPoolExecutor(max_workers=self.openai_max_active_batches) as pool:
            while pending or submitting or active:
                while pending and len(submitting) + len(active) < self.openai_max_active_batches:
                    offset, chunk_prompts = pending.pop(0)
                    submitting.append(pool.submit(
                        self._submit_openai_batch_chunk, chunk_prompts, offset=offset,
                        max_output_length=max_output_length,
                        response_format=response_format))

                # With nothing to poll yet, block until the first upload lands
                # instead of sleeping a whole poll interval.
                if submitting and not active:
                    wait(submitting, return_when=FIRST_COMPLETED)
                still_submitting = []
                for future in submitting:
                    if future.done():
                        job = future.result()  # re-raises a failed upload
                        active[job["batch_id"]] = job
                    else:
                        still_submitting.append(future)
                submitting = still_submitting

                completed_any = False
                for batch_id, job in list(active.items()):
                    batch = call_with_retries(
                        lambda batch_id=batch_id: self.client.batches.retrieve(batch_id),
                        f"Batch API status for {batch_id}")
                    if batch.status not in {"completed", "failed", "expired", "cancelled"}:
                        continue

                    completed_any = True
                    del active[batch_id]
                    if batch.status != "completed":
                        raise RuntimeError(
                            f"OpenAI batch {batch_id} ended with status "
                            f"{batch.status}: {describe_batch_errors(batch)}")
                    chunk_outputs = self._download_openai_batch_chunk(batch, job)
                    offset = job["offset"]
                    outputs[offset:offset + len(chunk_outputs)] = chunk_outputs

                    # Checkpoint each completed chunk immediately. A restarted run
                    # therefore resubmits only prompts from unfinished/failed chunks.
                    with self.cache_lock:
                        for prompt, output in zip(
                                prompts[offset:offset + len(chunk_outputs)], chunk_outputs):
                            key = f"{prompt.strip()}_{sample_idx}"
                            if key not in self.cache_dict:
                                self.add_n += 1
                            self.cache_dict[key] = output
                    self.save_cache()

                    logging.info(
                        "Completed OpenAI batch %s (%d requests; %d/%d outputs ready)",
                        batch_id, len(chunk_outputs),
                        sum(output is not None for output in outputs), len(outputs))

                if active and not completed_any:
                    time.sleep(self.batch_poll_interval)

        missing = [index for index, output in enumerate(outputs) if output is None]
        if missing:
            raise RuntimeError(f"Missing {len(missing)} Batch API outputs: {missing[:5]}")
        return outputs

    def _submit_openai_batch_chunk(self, prompts, offset, max_output_length, response_format):
        route = resolve_route(self.model_name)
        lines = []
        for local_index, prompt in enumerate(prompts):
            body = build_responses_kwargs(
                prompt=prompt, model_name=route.api_model, temp=self.temp,
                max_output_tokens=max_output_length, response_format=response_format,
                is_reasoning=route.is_reasoning,
                reasoning_effort=self.reasoning_effort or route.reasoning_effort)
            lines.append(json.dumps({
                "custom_id": f"request-{offset + local_index}",
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }, ensure_ascii=False))
        # The JSONL is built in memory and handed to the SDK as bytes: nothing
        # to write, reopen and clean up on disk, and a retried upload trivially
        # resends the same payload. A 5000-request chunk is tens of megabytes,
        # well under the Batch API's 200 MB file limit.
        payload = ("\n".join(lines) + "\n").encode("utf-8")

        input_file = call_with_retries(
            lambda: self.client.files.create(
                file=(f"factscore-batch-{offset}.jsonl", payload), purpose="batch"),
            "Batch API input upload")
        # No status polling here: the file's `status` field is deprecated and a
        # batch-purpose upload is already "processed" when files.create returns.
        # The JSONL itself is validated by batches.create, whose errors
        # describe_batch_errors() surfaces on the batch object.
        if getattr(input_file, "status", None) == "error":
            raise RuntimeError(
                f"Batch input file {input_file.id} failed processing: "
                f"{getattr(input_file, 'status_details', None)}")

        batch = call_with_retries(
            lambda: self.client.batches.create(
                input_file_id=input_file.id, endpoint="/v1/responses",
                completion_window=self.batch_completion_window,
                metadata={"task": "factscore-verdicts", "offset": str(offset)}),
            "Batch API submission")
        logging.info(
            "Submitted OpenAI batch %s with %d requests (offset %d)",
            batch.id, len(prompts), offset)
        return {
            "batch_id": batch.id,
            "offset": offset,
            "size": len(prompts),
        }

    def _download_openai_batch_chunk(self, batch, job):
        if not batch.output_file_id:
            raise RuntimeError(f"OpenAI batch {batch.id} completed without an output file")

        raw_text = self._read_openai_file(batch.output_file_id,
                                          f"Batch API output for {batch.id}")
        results, failures = parse_batch_output(raw_text)
        if failures:
            raise RuntimeError(
                f"{len(failures)} request(s) failed in OpenAI batch {batch.id}: "
                f"{failures[:3]}")

        expected_ids = [
            f"request-{job['offset'] + local_index}"
            for local_index in range(job["size"])
        ]
        missing = [custom_id for custom_id in expected_ids if custom_id not in results]
        if missing:
            raise RuntimeError(
                f"Missing {len(missing)} result(s) from batch {batch.id}: {missing[:5]}")
        return [results[custom_id] for custom_id in expected_ids]

    def _read_openai_file(self, file_id, description):
        """The text of a file stored at OpenAI (a Batch input or output JSONL)."""
        if self.model is None:
            self.load_model()
        content = call_with_retries(lambda: self.client.files.content(file_id), description)
        raw_text = getattr(content, "text", None)
        if raw_text is None:
            raw = content.read() if hasattr(content, "read") else content.content
            raw_text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        return raw_text

    def load_batch_output(self, output=None, input=None, batch_id=None, sample_idx=0,
                          overwrite=False):
        """Import the results of an OpenAI Batch job into the LM cache.

        For runs where the process lost contact with a batch that OpenAI went
        on to complete: instead of resubmitting (and paying for) the same
        prompts, the finished verdicts are written into the cache under the
        prompts' cache keys, so the next ``generate_batch`` over the same
        prompts finds them there and only submits whatever is still missing.

        ``output`` is the batch's output JSONL - a path, or the already parsed
        records (an iterable of dicts, one per line). A result line carries only
        its ``custom_id``, not the prompt, so the prompts are read from the
        batch's input JSONL: ``input`` (a path or parsed records) when given,
        otherwise the input file is fetched from OpenAI by ``batch_id``, which
        defaults to the id in a dashboard-style filename such as
        ``batch_<id>_output.jsonl``. With a ``batch_id`` and no ``output`` the
        output file is fetched from OpenAI as well.

        Requests that failed inside the batch are skipped (and logged), so the
        normal run re-asks them. Prompts already in the cache keep their cached
        answer unless ``overwrite`` is set. Returns a summary dict.
        """
        if output is None and batch_id is None:
            raise ValueError("load_batch_output needs an output file or a batch_id")
        if batch_id is None and is_batch_file_path(output):
            batch_id = batch_id_from_filename(output)

        batch = None
        if output is None:
            batch = self._retrieve_batch(batch_id)
            if batch.status != "completed":
                raise RuntimeError(
                    f"OpenAI batch {batch_id} has status {batch.status}, not completed: "
                    f"{describe_batch_errors(batch)}")
            if not batch.output_file_id:
                raise RuntimeError(f"OpenAI batch {batch_id} has no output file")
            output = self._read_openai_file(batch.output_file_id,
                                            f"Batch API output for {batch_id}")
        results, failures = parse_batch_output(read_batch_records(output))

        if input is None:
            if batch_id is None:
                raise ValueError(
                    "Batch result lines do not contain the prompts: pass the batch's "
                    "input JSONL as `input`, or a `batch_id` (or an output path named "
                    "like batch_<id>_output.jsonl) so it can be fetched from OpenAI.")
            if batch is None:
                batch = self._retrieve_batch(batch_id)
            input = self._read_openai_file(batch.input_file_id,
                                           f"Batch API input for {batch_id}")
        prompts = parse_batch_input(read_batch_records(input))

        if self.model_name != "InstructGPT":
            expected_model = resolve_route(self.model_name).api_model
            other_models = {entry["model"] for entry in prompts.values()
                            if entry["model"] not in (None, expected_model)}
            if other_models:
                logging.warning("Batch %s was run with model(s) %s, this LM is %s",
                                batch_id, sorted(other_models), expected_model)

        unmatched = [custom_id for custom_id in results if custom_id not in prompts]
        if unmatched:
            raise RuntimeError(
                f"{len(unmatched)} result(s) have no prompt in the batch input "
                f"(wrong input file?): {unmatched[:5]}")

        loaded = kept = 0
        with self.cache_lock:
            for custom_id, generation in results.items():
                key = f"{prompts[custom_id]['prompt'].strip()}_{sample_idx}"
                if key in self.cache_dict and not overwrite:
                    kept += 1
                    continue
                if key not in self.cache_dict:
                    self.add_n += 1
                self.cache_dict[key] = generation
                loaded += 1
        self.save_cache()

        for failure in failures:
            logging.warning("Skipping failed request %s from batch %s: %s",
                            failure["custom_id"], batch_id,
                            failure["error"] or failure["response"])
        summary = {"batch_id": batch_id, "loaded": loaded, "already_cached": kept,
                   "failed": len(failures),
                   "unanswered": len([c for c in prompts if c not in results])}
        logging.info("Imported OpenAI batch %s: %s", batch_id, summary)
        return summary

    def _retrieve_batch(self, batch_id):
        if self.model is None:
            self.load_model()
        return call_with_retries(lambda: self.client.batches.retrieve(batch_id),
                                 f"Batch API status for {batch_id}")


def is_batch_file_path(source):
    """A path to a Batch JSONL rather than its text or records (text starts with '{')."""
    return isinstance(source, os.PathLike) or (
        isinstance(source, str) and not source.lstrip().startswith("{"))


def read_batch_records(source):
    """The records of a Batch JSONL: from a path, raw text, or parsed dicts."""
    if is_batch_file_path(source):
        with open(source, encoding="utf-8") as f:
            source = f.read()
    if isinstance(source, str):
        return [json.loads(line) for line in source.splitlines() if line.strip()]
    if isinstance(source, dict):
        return [source]
    return [json.loads(item) if isinstance(item, str) else item for item in source]


def parse_batch_output(records):
    """Split Batch output lines into {custom_id: (text, metadata)} and failures."""
    if isinstance(records, str):
        records = read_batch_records(records)
    results = {}
    failures = []
    for item in records:
        custom_id = item.get("custom_id")
        response = item.get("response")
        if item.get("error") or not response or response.get("status_code") != 200:
            failures.append({
                "custom_id": custom_id,
                "error": item.get("error"),
                "response": response,
            })
            continue
        results[custom_id] = response_body_to_generation(response["body"])
    return results, failures


def parse_batch_input(records):
    """{custom_id: {"prompt", "model"}} from the lines of a Batch input JSONL."""
    if isinstance(records, str):
        records = read_batch_records(records)
    prompts = {}
    for item in records:
        body = item.get("body") or {}
        messages = body.get("input")
        if isinstance(messages, str):
            prompt = messages
        elif messages:
            prompt = messages[0].get("content") if isinstance(messages[0], dict) else messages[0]
        else:
            # /v1/chat/completions style input
            chat = body.get("messages") or []
            prompt = chat[0].get("content") if chat else None
        if not isinstance(prompt, str):
            raise ValueError(f"Cannot read the prompt of request {item.get('custom_id')!r}")
        prompts[item.get("custom_id")] = {"prompt": prompt, "model": body.get("model")}
    return prompts


def batch_id_from_filename(path):
    """The batch id in a dashboard download name like batch_<id>_output.jsonl, else None."""
    match = re.match(r"(batch_[0-9A-Za-z]+)", os.path.basename(str(path)))
    return match.group(1) if match else None


def describe_batch_errors(batch):
    """The validation errors OpenAI attached to a batch, as one readable line.

    A bare "ended with status failed" says nothing about whether the JSONL was
    malformed, the model was rejected, or the account cannot run batches at all.
    That reason only lives in batch.errors, so it belongs in the exception.
    """
    data = getattr(getattr(batch, "errors", None), "data", None) or []
    messages = []
    for error in data:
        line = getattr(error, "line", None)
        where = f" (line {line})" if line is not None else ""
        messages.append(f"{getattr(error, 'code', None)}: "
                        f"{getattr(error, 'message', None)}{where}")
    return "; ".join(messages) or "no error detail returned"


def should_retry(error):
    """Rate limits, timeouts and 5xx are transient; a 400 will never succeed."""
    transient = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)
    if isinstance(error, transient):
        return True
    status = getattr(error, "status_code", None)
    return status is not None and status >= 500


def call_with_retries(send, description):
    """Run send(), backing off on transient errors and giving up eventually.

    The old loop retried every exception forever and never reset its counter, so
    a permanent 400 would spin at 2**n seconds without ever surfacing — with
    several requests in flight that is easy to mistake for a hung run.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return send()
        except Exception as e:
            if not should_retry(e) or attempt == MAX_RETRIES:
                logging.critical("API error on %s, giving up after %d attempt(s): %s",
                                 description, attempt, str(e))
                raise
            wait = min(2 ** attempt, MAX_BACKOFF_SECONDS)
            logging.error("API error: %s (%d/%d). Waiting %d sec",
                          str(e), attempt, MAX_RETRIES, wait)
            time.sleep(wait)


def verdict_word(token):
    """The token's text as a bare verdict word, or None if it isn't one."""
    cleaned = (token or "").strip().strip('"').strip().lower()
    return cleaned if cleaned in ("true", "false") else None


def verdict_probability(response):
    """P(True) read off the token logprobs, or None when they weren't returned.

    Scores the first token that is itself a verdict word — under the json_schema
    format that is the one inside {"verdict": ...} — and renormalises over the
    True/False alternatives at that position, ignoring the rest of the vocabulary.
    """
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            for entry in getattr(content, "logprobs", None) or []:
                if verdict_word(getattr(entry, "token", None)) is None:
                    continue
                mass = {"true": 0.0, "false": 0.0}
                alternatives = list(getattr(entry, "top_logprobs", None) or []) or [entry]
                for alternative in alternatives:
                    word = verdict_word(getattr(alternative, "token", None))
                    logprob = getattr(alternative, "logprob", None)
                    if word is not None and logprob is not None:
                        mass[word] += math.exp(logprob)
                total = mass["true"] + mass["false"]
                if total > 0:
                    return mass["true"] / total
    return None



def build_responses_kwargs(prompt, model_name, max_output_tokens=512, temp=0.0,
                            response_format=None, is_reasoning=False,
                            reasoning_effort="none", top_logprobs=5):
    """Build a /v1/responses request body usable online or in Batch JSONL."""
    kwargs = {
        "model": model_name,
        "input": [{"role": "user", "content": prompt}],
    }
    if response_format is not None:
        kwargs["text"] = {"format": response_format}
    if is_reasoning:
        if reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": reasoning_effort}
    else:
        kwargs["temperature"] = temp
        kwargs["max_output_tokens"] = max_output_tokens
        if top_logprobs:
            kwargs["top_logprobs"] = top_logprobs
    return kwargs


def response_body_to_generation(body):
    """Convert a Batch /v1/responses body to LM's (text, metadata) contract."""
    text_parts = []
    p_true = None
    for item in body.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") == "output_text" and content.get("text") is not None:
                text_parts.append(content["text"])
            if p_true is None:
                p_true = verdict_probability_dict(content.get("logprobs") or [])
    output_text = "".join(text_parts)
    return output_text, {"raw_response": body, "output_text": output_text, "p_true": p_true}


def verdict_probability_dict(entries):
    """Dictionary equivalent of verdict_probability() for Batch output JSON."""
    for entry in entries:
        if verdict_word(entry.get("token")) is None:
            continue
        mass = {"true": 0.0, "false": 0.0}
        alternatives = entry.get("top_logprobs") or [entry]
        for alternative in alternatives:
            word = verdict_word(alternative.get("token"))
            logprob = alternative.get("logprob")
            if word is not None and logprob is not None:
                mass[word] += math.exp(logprob)
        total = mass["true"] + mass["false"]
        if total > 0:
            return mass["true"] / total
    return None

def call_chat_model(
    client,
    prompt,
    model_name="gpt-4o-mini",  # Fixed model name
    max_output_tokens=512,
    temp=0.0,
    response_format = None,
    is_reasoning = False,
    reasoning_effort = "none",
    top_logprobs = 5,
):
    kwargs = build_responses_kwargs(
        prompt=prompt, model_name=model_name, max_output_tokens=max_output_tokens,
        temp=temp, response_format=response_format, is_reasoning=is_reasoning,
        reasoning_effort=reasoning_effort, top_logprobs=top_logprobs)

    response = call_with_retries(lambda: client.responses.create(**kwargs),
                                 f"{model_name} verdict")

    return {
        "raw_response": response,
        "output_text": response.output_text,  # Correct attribute path
        # None unless the model returned logprobs
        "p_true": verdict_probability(response),
    }

def call_instruct_model(
    client,
    prompt,
    model_name="gpt-3.5-turbo-instruct",
    max_output_tokens=512,
    temp=0.0,
):
    response = call_with_retries(
        lambda: client.completions.create(  # Correct method
            model=model_name,
            prompt=prompt,  # Correct parameter name
            max_tokens=max_output_tokens,  # Correct parameter name
            temperature=temp,
        ),
        f"{model_name} completion")

    return {
        "raw_response": response,
        "output_text": response.choices[0].text,  # Correct attribute path
        "p_true": None,
    }


# class OpenAIModel(LM):

#     def __init__(self, model_name, cache_file=None, key_path="api.key"):
#         self.model_name = model_name
#         self.key_path = key_path
#         self.temp = 0.7
#         self.save_interval = 100
#         super().__init__(cache_file)

#     def load_model(self):
#         # load api key
#         key_path = self.key_path
#         #assert os.path.exists(key_path), f"Please place your OpenAI APT Key in {key_path}."
#         #with open(key_path, 'r') as f:
#         #    api_key = f.readline()
#         openai.api_key = os.environ["OPENAI_API_KEY"]#api_key.strip()
#         self.model = self.model_name

#     def _generate(self, prompt, max_sequence_length=2048, max_output_length=128):
#         if self.add_n % self.save_interval == 0:
#             self.save_cache()
#         # return a tuple of string (generated text) and metadata (any format)
#         # This should be about generating a response from the prompt, no matter what the application is
#         if self.model_name == "ChatGPT":
#             # Construct the prompt send to ChatGPT
#             message = [{"role": "user", "content": prompt}]
#             # Call API
#             response = call_ChatGPT(message, temp=self.temp, max_len=max_sequence_length)
#             # Get the output from the response
#             output = response["choices"][0]["message"]["content"]
#             return output, response
#         elif self.model_name == "InstructGPT":
#             # Call API
#             response = call_GPT3(prompt, temp=self.temp)
#             # Get the output from the response
#             output = response["choices"][0]["text"]
#             return output, response
#         else:
#             raise NotImplementedError()

# def call_ChatGPT(message, model_name="gpt-3.5-turbo", max_len=1024, temp=0.7, verbose=False):
#     # call GPT-3 API until result is provided and then return it
#     response = None
#     received = False
#     num_rate_errors = 0
#     while not received:
#         try:
#             response = openai.ChatCompletion.create(model=model_name,
#                                                     messages=message,
#                                                     max_tokens=max_len,
#                                                     temperature=temp)
#             received = True
#         except:
#             # print(message)
#             num_rate_errors += 1
#             error = sys.exc_info()[0]
#             if error == openai.error.InvalidRequestError:
#                 # something is wrong: e.g. prompt too long
#                 logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{message}\n\n")
#                 assert False
            
#             logging.error("API error: %s (%d). Waiting %dsec" % (error, num_rate_errors, np.power(2, num_rate_errors)))
#             time.sleep(np.power(2, num_rate_errors))
#     return response


# def call_GPT3(prompt, model_name="gpt-3.5-turbo-instruct", max_len=512, temp=0.7, num_log_probs=0, echo=False, verbose=False):
#     # call GPT-3 API until result is provided and then return it
#     response = None
#     received = False
#     num_rate_errors = 0
#     while not received:
#         try:
#             response = openai.Completion.create(model=model_name,
#                                                 prompt=prompt,
#                                                 max_tokens=max_len,
#                                                 temperature=temp,
#                                                 logprobs=num_log_probs,
#                                                 echo=echo)
#             received = True
#         except:
#             error = sys.exc_info()[0]
#             num_rate_errors += 1
#             if error == openai.error.InvalidRequestError:
#                 # something is wrong: e.g. prompt too long
#                 logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
#                 assert False
#             logging.error("API error: %s (%d)" % (error, num_rate_errors))
#             time.sleep(np.power(2, num_rate_errors))
#     return response
