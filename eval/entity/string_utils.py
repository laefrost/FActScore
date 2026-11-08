"""Locating the annotated spans in the completion text.

Ported from ``utils/string_utils.py`` in hallucination_probes. The one change:
that file scores candidate substrings with the ``rouge_score`` package's
ROUGE-L. The same measure - longest common subsequence over lower-cased
alphanumeric tokens, no stemming - is computed here by ``rouge_l`` so that the
dependency is not needed.
"""

import math
import re
from collections import namedtuple
from typing import Optional, Tuple

RougeScore = namedtuple("RougeScore", "precision recall fmeasure")

# rouge_score.tokenize.tokenize (use_stemmer=False): lower-case the text, then
# every maximal run of [a-z0-9] is a token
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())


def _lcs_length(a, b) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(target: str, prediction: str) -> RougeScore:
    """ROUGE-L of `prediction` against `target`, as rouge_score's RougeScorer(['rougeL']) computes it."""
    target_tokens = _tokenize(target)
    prediction_tokens = _tokenize(prediction)
    if not target_tokens or not prediction_tokens:
        return RougeScore(0.0, 0.0, 0.0)
    lcs = _lcs_length(target_tokens, prediction_tokens)
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(target_tokens)
    fmeasure = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return RougeScore(precision, recall, fmeasure)


def normalize_text(text: str) -> str:
    """
    Normalize text by removing control characters and standardizing Unicode.

    Args:
        text: Raw text that may contain control characters or Unicode variants

    Returns:
        Normalized text with standard characters
    """
    # Remove control characters except newline, tab, and carriage return
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

    # Replace Unicode spaces with regular spaces
    text = re.sub(r'[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000\uFEFF]', ' ', text)

    # Replace Unicode quotes with standard quotes
    text = re.sub(r'[\u201C\u201D\u2018\u2019\u201E\u201F\u2039\u203A\u00AB\u00BB]', '"', text)

    # Replace Unicode dashes with standard dash
    text = re.sub(r'[\u2013\u2014\u2015]', '-', text)

    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    return text


def normalize_for_matching(text: str) -> str:
    """Normalize text for matching - quotes, whitespace, punctuation."""
    # Step 1: Add spaces between letters and numbers (except after M/m/X/x for Roman numerals)
    text = re.sub(
        r'(?<=[^\W\d_])(?<![MmXx])(?=\d)|(?<=\d)(?=[^\W\d_])', ' ', text)

    # Step 2: Remove quotes and punctuation (but preserve decimals)
    # This combines quote removal with punctuation removal in one regex
    text = re.sub(r'[\'"`\u2018\u2019\u201C\u201D\u201B\u201E:;()\[\]\-\u2013\u2014]|[.,](?!(?<=\d[.,])\d)', ' ', text)

    # Step 3: Normalize whitespace and lowercase
    text = re.sub(r'\s+', ' ', text).strip().lower()

    return text


def _recall(query: str, match: str, normalize: bool) -> float:
    if normalize:
        return rouge_l(normalize_for_matching(query), normalize_for_matching(match)).recall
    return rouge_l(query, match).recall


def trim_match_edges(query: str, match: str, normalize_text: bool = False) -> str:
    """
    Trim unnecessary characters from the beginning and end of a match
    while maintaining or improving recall.
    """
    if not match or match == query:
        return match

    # Get initial recall score
    best_match = match
    best_recall = _recall(query, match, normalize_text)

    # Try trimming from the beginning
    for i in range(1, min(len(match) // 2, 20)):  # Don't trim more than half or 20 chars
        trimmed = match[i:]
        recall = _recall(query, trimmed, normalize_text)

        if recall >= best_recall:
            best_match = trimmed
            best_recall = recall
        else:
            break  # Stop if recall drops

    # Try trimming from the end
    match = best_match  # Start with best so far
    for i in range(1, min(len(match) // 2, 20)):
        trimmed = match[:-i]
        recall = _recall(query, trimmed, normalize_text)

        if recall >= best_recall:
            best_match = trimmed
            best_recall = recall
        else:
            break  # Stop if recall drops

    return best_match


def find_closest_match(query: str, text: str, window_margin: int = 10, min_similarity: float = 0.9, normalize_text: bool = False) -> Optional[str]:
    """
    Finds the substring of `text` that best matches `query` using ROUGE-L similarity.
    Returns the exact substring from `text`.

    :param query: The query string (e.g., the extracted factual claim).
    :param text: The larger text in which we want to find the best matching substring.
    :param window_margin: The +/- margin around the query length for the sliding window.
    :return: The substring from `text` that best matches `query`.
    """
    if query in text:
        return query

    # We'll attempt to match substrings in `text` that are around len(query).
    query_len = len(query)

    # Precompute the best match info
    best_substring = ""
    best_score = -math.inf

    if normalize_text:
        query = normalize_for_matching(query)

    # Because we want to allow some variation (extra words, punctuation, etc.),
    # we use a window size = query_len +/- window_margin
    min_len = max(1, query_len - window_margin)
    max_len = min(len(text), query_len + window_margin)

    # Slide over the text and compare substrings
    for start_idx in range(len(text)):
        # We'll try multiple substring lengths in [min_len, max_len]
        for length in range(min_len, max_len + 1):
            end_idx = start_idx + length
            if end_idx > len(text):
                break  # no need to go further if we exceed text length

            candidate_substring = text[start_idx:end_idx].strip()

            if normalize_text:
                scores = rouge_l(query, normalize_for_matching(candidate_substring))
            else:
                scores = rouge_l(query, candidate_substring)

            rouge_l_score = scores.fmeasure  # f-measure of ROUGE-L

            if rouge_l_score > best_score:
                best_score = rouge_l_score
                best_substring = candidate_substring

            if best_score > 0.95:
                # Trim edges if not perfect
                best_substring = trim_match_edges(query, best_substring, normalize_text)
                assert best_substring in text, f"Best substring {best_substring} not in text {text}"
                return best_substring

    if best_score >= min_similarity:
        # Trim edges before returning
        best_substring = trim_match_edges(query, best_substring, normalize_text)
        assert best_substring in text
        return best_substring
    else:
        if not normalize_text:
            return find_closest_match(query, text, window_margin, min_similarity, normalize_text=True)
        return None


def try_matching_span_in_text(span: str, text: str, cur_idx: int = 0, min_similarity: float = 0.8) -> Tuple[Optional[str], Optional[int]]:
    """
    Try to match a span to a substring in the text.

    Args:
        span: The span text to match
        text: The full text to search in
        cur_idx: Current index to start searching from
        min_similarity: Minimum similarity threshold

    Returns:
        Tuple of (matched_text, index) or (None, None) if no match found
    """
    # as in the original: find_closest_match runs with its own default threshold (0.9);
    # `min_similarity` is only carried into the retry from the start of the text
    closest_match: Optional[str] = find_closest_match(span, text[cur_idx:])

    if closest_match is not None:
        span_idx = text[cur_idx:].index(closest_match)
        assert span_idx != -1, f"Span {repr(span)} not found in text[cur_idx:]: {repr(text[cur_idx:])}"
        return closest_match, cur_idx + span_idx

    # try searching for the span in the previous part of the text
    if cur_idx > 0:
        return try_matching_span_in_text(span, text, cur_idx=0, min_similarity=min_similarity)

    return None, None
