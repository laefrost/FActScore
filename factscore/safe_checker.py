"""SAFE as a drop-in fact checker for FactScorer.

FactScorer's own checker retrieves passages from a knowledge source and asks the
LM for a True/False verdict on each atomic fact. SAFE (``eval/safe``) instead
makes the LM rewrite each fact to be self-contained, decide whether the fact is
even relevant to the prompt, and - for the relevant ones - ground it in Google
Search results over several query steps.

Nothing about that pipeline is reimplemented here: this module only wires
FactScorer's atoms and LM into ``eval.safe`` and runs the per-atom pipeline
concurrently. The atoms themselves still come from factscore's
``AtomicFactGenerator``; SAFE's own splitter (``eval/safe/get_atomic_facts.py``)
is not involved.
"""

import logging
import os
import sys

from concurrent.futures import ThreadPoolExecutor

# eval.safe.search_augmented_factuality_eval.IRRELEVANT_LABEL - taken as a
# literal because importing that module also pulls in SAFE's atomic fact
# splitter, which this integration does not use.
IRRELEVANT_LABEL = 'Irrelevant'

# eval.safe.search_augmented_factuality_eval._MAX_PIPELINE_RETRIES
MAX_PIPELINE_RETRIES = 3


def _import_safe():
    """The SAFE modules, adding the repo root to sys.path if it isn't there."""
    try:
        from eval.safe import classify_relevance, config, rate_atomic_fact
    except ImportError:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from eval.safe import classify_relevance, config, rate_atomic_fact
    return classify_relevance, rate_atomic_fact, config


class SafeChecker:
    """Runs the SAFE relevance + rating pipeline over individual atomic facts."""

    def __init__(self, lm, temperature=None, max_tokens=None, max_workers=None,
                 max_pipeline_retries=MAX_PIPELINE_RETRIES):
        classify_relevance, rate_atomic_fact, safe_config = _import_safe()
        from common import modeling, shared_config, utils

        self._classify_relevance = classify_relevance
        self._rate_atomic_fact = rate_atomic_fact
        self._utils = utils
        self.max_pipeline_retries = max_pipeline_retries
        # SAFE atoms are independent, but each one is a long serial chain of
        # calls (revise -> relevance -> up to max_steps searches -> answer), so
        # the atoms themselves are what gets parallelized.
        self.max_workers = max_workers or getattr(lm, 'max_concurrency', 1)

        self.rater = modeling.Model(
            lm=lm,
            temperature=safe_config.model_temp if temperature is None else temperature,
            max_tokens=safe_config.max_tokens if max_tokens is None else max_tokens)

        if not shared_config.serper_api_key:
            logging.warning(
                "SAFE grounds facts in Google Search but no Serper API key was found; "
                "set SERPER_API_KEY (or write the key to serper.key) unless "
                "rate_atomic_fact.call_search has been pointed at another search backend.")

    def check_atom(self, prompt, response, atom):
        """One atom through SAFE: revise, check relevance, then rate if relevant.

        Mirrors ``search_augmented_factuality_eval.classify_relevance_and_rate_single``;
        irrelevant atoms come back with ``is_relevant`` False and no rating.
        """
        is_relevant, self_contained_atom, relevance_data = self._classify_relevance.main(
            prompt, response, atomic_fact=atom, model=self.rater)

        result = {'atom': atom,
                  'self_contained_atom': self_contained_atom,
                  'is_relevant': is_relevant,
                  'relevance_data': relevance_data,
                  'annotation': IRRELEVANT_LABEL,
                  'is_supported': None,
                  'rate_response': None,
                  'search_data': None}

        if not is_relevant:  # no need to rate further
            return result

        rate_data, search_data = self._rate_atomic_fact.check_atomic_fact(
            atomic_fact=self_contained_atom, rater=self.rater)

        if not isinstance(rate_data, self._rate_atomic_fact.FinalAnswer):
            raise ValueError('No rate data found for atomic fact.')

        supported = self._rate_atomic_fact.SUPPORTED_LABEL
        result.update({'annotation': rate_data.answer,
                       'is_supported': rate_data.answer.lower() == supported.lower(),
                       'rate_response': rate_data.response,
                       'search_data': search_data})
        return result

    def check_atom_with_retries(self, prompt, response, atom):
        """check_atom, retried on failure; None once the retries are used up."""
        num_fails = 0

        while num_fails < self.max_pipeline_retries:
            try:
                return self.check_atom(prompt, response, atom)
            except Exception as e:  # pylint: disable=broad-exception-caught
                self._utils.maybe_print_error(e)
                num_fails += 1

        return None

    def check_atoms(self, items):
        """Check many (prompt, response, atom) triples, one result per item, in order."""
        items = list(items)
        if not items:
            return []

        def run(item):
            prompt, response, atom = item
            return self.check_atom_with_retries(prompt, response, atom)

        workers = min(self.max_workers, len(items))
        if workers <= 1:
            return [run(item) for item in items]

        # one cache flush at the end instead of one per worker crossing the
        # backend's save interval
        lm = self.rater.lm
        suspended = hasattr(lm, 'suspend_autosave')
        if suspended:
            lm.suspend_autosave = True
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(run, items))
        finally:
            if suspended:
                lm.suspend_autosave = False
        if hasattr(lm, 'save_cache'):
            lm.save_cache()

        return results
