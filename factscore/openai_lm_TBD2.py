from factscore.lm import LM
import openai
import math
import sys
import time
import os
import numpy as np
import logging
import json
import tempfile

from collections import namedtuple
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
                 batch_poll_interval=30.0, batch_completion_window="24h"):
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
            batch_outputs = self._generate_openai_batch(
                [uncached_by_key[key] for key in keys],
                max_output_length=max_output_length,
                response_format=response_format)
            with self.cache_lock:
                for key, output in zip(keys, batch_outputs):
                    self.cache_dict[key] = output
                    outputs_by_key[key] = output
                    self.add_n += 1
            self.save_cache()

        return [outputs_by_key[key_for(prompt)] for prompt in prompts]

    def _generate_openai_batch(self, prompts, max_output_length=128, response_format=None):
        route = resolve_route(self.model_name)
        requests = []
        for index, prompt in enumerate(prompts):
            body = build_responses_kwargs(
                prompt=prompt, model_name=route.api_model, temp=self.temp,
                max_output_tokens=max_output_length, response_format=response_format,
                is_reasoning=route.is_reasoning,
                reasoning_effort=self.reasoning_effort or route.reasoning_effort)
            requests.append({
                "custom_id": f"request-{index}",
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            })

        path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                             encoding="utf-8", delete=False) as handle:
                path = handle.name
                for request in requests:
                    handle.write(json.dumps(request, ensure_ascii=False) + "\n")

            with open(path, "rb") as handle:
                input_file = call_with_retries(
                    lambda: self.client.files.create(file=handle, purpose="batch"),
                    "Batch API input upload")

            batch = call_with_retries(
                lambda: self.client.batches.create(
                    input_file_id=input_file.id, endpoint="/v1/responses",
                    completion_window=self.batch_completion_window,
                    metadata={"task": "factscore-verdicts"}),
                "Batch API submission")
            logging.info("Submitted OpenAI batch %s with %d requests", batch.id, len(prompts))

            terminal = {"completed", "failed", "expired", "cancelled"}
            while batch.status not in terminal:
                time.sleep(self.batch_poll_interval)
                batch = call_with_retries(
                    lambda: self.client.batches.retrieve(batch.id),
                    f"Batch API status for {batch.id}")

            if batch.status != "completed":
                raise RuntimeError(f"OpenAI batch {batch.id} ended with status {batch.status}")
            if not batch.output_file_id:
                raise RuntimeError(f"OpenAI batch {batch.id} completed without an output file")

            content = call_with_retries(
                lambda: self.client.files.content(batch.output_file_id),
                f"Batch API output for {batch.id}")
            raw_text = getattr(content, "text", None)
            if raw_text is None:
                raw = content.read() if hasattr(content, "read") else content.content
                raw_text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

            results = {}
            failures = []
            for line in raw_text.splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                custom_id = item.get("custom_id")
                response = item.get("response")
                if item.get("error") or not response or response.get("status_code") != 200:
                    failures.append({"custom_id": custom_id, "error": item.get("error"),
                                     "response": response})
                    continue
                results[custom_id] = response_body_to_generation(response["body"])

            if failures:
                raise RuntimeError(
                    f"{len(failures)} request(s) failed in OpenAI batch {batch.id}: "
                    f"{failures[:3]}")

            missing = [f"request-{i}" for i in range(len(prompts))
                       if f"request-{i}" not in results]
            if missing:
                raise RuntimeError(f"Missing {len(missing)} result(s) from batch {batch.id}: {missing[:5]}")
            return [results[f"request-{i}"] for i in range(len(prompts))]
        finally:
            if path and os.path.exists(path):
                os.remove(path)


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
