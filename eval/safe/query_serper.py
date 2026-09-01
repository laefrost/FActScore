# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Class for querying the Google Serper API."""

import atexit
import json
import os
import pickle
import random
import threading
import time
from typing import Any, Optional, Literal

import requests

_SERPER_URL = 'https://google.serper.dev'
NO_RESULT_MSG = 'No good Google Search result was found'

################################################################################
#                              RESPONSE CACHE
# Serper bills per request, not per result, so the only way to spend less is to
# send fewer requests. Responses are keyed on the exact request parameters and
# kept on disk, which makes two things free that were previously paid for at
# full price: rerunning an evaluation, and the pipeline-level retries in
# `factscore/safe_checker.py`, which re-issue every search for an atom whose
# verdict failed to parse.
#
# Mirrors the prompt cache in `factscore/lm.py`, including the load-merge-dump
# so that parallel processes do not clobber each other's entries. Set
# SERPER_CACHE=0 to bypass it, SERPER_CACHE_FILE to move it.
################################################################################
_DEFAULT_CACHE_FILE = os.path.join('.cache', 'factscore', 'serper.pkl')
_SAVE_INTERVAL = 20
_MISS = object()

_cache_lock = threading.RLock()
_cache_dict = None
_cache_adds = 0


def _cache_file() -> str:
  return os.environ.get('SERPER_CACHE_FILE', _DEFAULT_CACHE_FILE)


def _cache_enabled() -> bool:
  return os.environ.get('SERPER_CACHE', '1').lower() not in ('0', 'false', 'no')


def _load_cache(max_retries: int = 3) -> dict[str, Any]:
  """The cache file's contents; empty when it is missing or unreadable."""
  path = _cache_file()

  if not os.path.exists(path):
    return {}

  for attempt in range(max_retries):
    try:
      with open(path, 'rb') as f:
        return pickle.load(f)
    except Exception:  # pylint: disable=broad-exception-caught
      # most likely another process is mid-write; give it a moment
      if attempt < max_retries - 1:
        time.sleep(5)

  return {}


def save_cache() -> None:
  """Flush new entries, merging in whatever another process wrote meanwhile."""
  global _cache_adds

  with _cache_lock:
    if not _cache_adds or _cache_dict is None:
      return

    on_disk = _load_cache()
    on_disk.update(_cache_dict)
    _cache_dict.update(on_disk)

    path = _cache_file()
    directory = os.path.dirname(path)

    if directory:
      os.makedirs(directory, exist_ok=True)

    # write-then-rename, so a reader never picks up a half-written pickle
    tmp_path = f'{path}.tmp{os.getpid()}'

    with open(tmp_path, 'wb') as f:
      pickle.dump(_cache_dict, f)

    os.replace(tmp_path, path)
    _cache_adds = 0


def _cache_lookup(key: str) -> Any:
  global _cache_dict

  if not _cache_enabled():
    return _MISS

  with _cache_lock:
    if _cache_dict is None:
      _cache_dict = _load_cache()

    return _cache_dict.get(key, _MISS)


def _cache_store(key: str, value: Any) -> None:
  global _cache_dict, _cache_adds

  if not _cache_enabled():
    return

  with _cache_lock:
    if _cache_dict is None:
      _cache_dict = _load_cache()

    _cache_dict[key] = value
    _cache_adds += 1

    if _cache_adds >= _SAVE_INTERVAL:
      save_cache()


atexit.register(save_cache)


class SerperAPI:
  """Class for querying the Google Serper API."""

  def __init__(
      self,
      serper_api_key: str,
      gl: str = 'us',
      hl: str = 'en',
      k: int = 1,
      tbs: Optional[str] = None,
      search_type: Literal['news', 'search', 'places', 'images'] = 'search',
  ):
    self.serper_api_key = serper_api_key
    self.gl = gl
    self.hl = hl
    self.k = k
    self.tbs = tbs
    self.search_type = search_type
    self.result_key_for_type = {
        'news': 'news',
        'places': 'places',
        'images': 'images',
        'search': 'organic',
    }

  def run(self, query: str, **kwargs: Any) -> str:
    """Run query through GoogleSearch and parse result."""
    assert self.serper_api_key, 'Missing serper_api_key.'
    results = self._google_serper_api_results(
        query,
        gl=self.gl,
        hl=self.hl,
        num=self.k,
        tbs=self.tbs,
        search_type=self.search_type,
        **kwargs,
    )

    return self._parse_results(results)

  def _google_serper_api_results(
      self,
      search_term: str,
      search_type: str = 'search',
      max_retries: int = 20,
      **kwargs: Any,
  ) -> dict[Any, Any]:
    """Run query through Google Serper."""
    headers = {
        'X-API-KEY': self.serper_api_key or '',
        'Content-Type': 'application/json',
    }
    params = {
        'q': search_term,
        **{key: value for key, value in kwargs.items() if value is not None},
    }
    url = f'{_SERPER_URL}/{search_type}'
    # the full request signature, so a cached response is only ever reused for
    # an identical query (`num` included - a k=3 result cannot serve k=10)
    cache_key = json.dumps({'url': url, **params}, sort_keys=True, default=str)
    cached = _cache_lookup(cache_key)

    if cached is not _MISS:
      return cached

    response, num_fails, sleep_time = None, 0, 0

    while not response and num_fails < max_retries:
      try:
        response = requests.post(url, headers=headers, params=params)
      except AssertionError as e:
        raise e
      except Exception:  # pylint: disable=broad-exception-caught
        response = None
        num_fails += 1
        sleep_time = min(sleep_time * 2, 600)
        sleep_time = random.uniform(1, 10) if not sleep_time else sleep_time
        time.sleep(sleep_time)

    if not response:
      raise ValueError('Failed to get result from Google Serper API')

    response.raise_for_status()
    search_results = response.json()
    _cache_store(cache_key, search_results)
    return search_results

  def _parse_snippets(self, results: dict[Any, Any]) -> list[str]:
    """Parse results."""
    snippets = []

    if results.get('answerBox'):
      answer_box = results.get('answerBox', {})
      answer = answer_box.get('answer')
      snippet = answer_box.get('snippet')
      snippet_highlighted = answer_box.get('snippetHighlighted')

      if answer and isinstance(answer, str):
        snippets.append(answer)
      if snippet and isinstance(snippet, str):
        snippets.append(snippet.replace('\n', ' '))
      if snippet_highlighted:
        snippets.append(snippet_highlighted)

    if results.get('knowledgeGraph'):
      kg = results.get('knowledgeGraph', {})
      title = kg.get('title')
      entity_type = kg.get('type')
      description = kg.get('description')

      if entity_type:
        snippets.append(f'{title}: {entity_type}.')

      if description:
        snippets.append(description)

      for attribute, value in kg.get('attributes', {}).items():
        snippets.append(f'{title} {attribute}: {value}.')

    result_key = self.result_key_for_type[self.search_type]

    if result_key in results:
      for result in results[result_key][:self.k]:
        if 'snippet' in result:
          snippets.append(result['snippet'])

        for attribute, value in result.get('attributes', {}).items():
          snippets.append(f'{attribute}: {value}.')

    if not snippets:
      return [NO_RESULT_MSG]

    return snippets

  def _parse_results(self, results: dict[Any, Any]) -> str:
    return ' '.join(self._parse_snippets(results))
