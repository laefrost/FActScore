import argparse
import string
import json
import numpy as np
import os
import logging
from pydantic import BaseModel


from tqdm import tqdm
from factscore.abstain_detection import is_response_abstained
from factscore.atomic_facts import AtomicFactGenerator
from factscore.clm import CLM
from factscore.gpt_oss_lm import Oss
from factscore.hf_lm import HFInf
from factscore.npm import NPM
from factscore.openai_lm import OpenAIModel
from factscore.retrieval import DocDB, Retrieval
import re
from transformers import AutoTokenizer
import itertools
from collections import defaultdict
import re
import nltk
nltk.download("punkt_tab")  # one-time download
from nltk.tokenize import word_tokenize


# The verdict prompts were written for a completion model emitting a single
# token. Chat models answer in prose, so the output is constrained twice: with a
# json_schema response format for backends that support it (OpenAI), and with an
# explicit one-word instruction for those that ignore it.
VERDICT_INSTRUCTION = "Answer with exactly one word: True or False."

VERDICT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "fact_verdict",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"verdict": {"type": "string", "enum": ["True", "False"]}},
        "required": ["verdict"],
        "additionalProperties": False,
    },
}


# Surface forms of the verdict tokens, most vocabulary-specific first:
# "▁True" is SentencePiece (LLaMA-1/2), "ĠTrue" is byte-level BPE (GPT-2/o200k).
_TRUE_SURFACES = ("▁True", "ĠTrue", " True", "True")
_FALSE_SURFACES = ("▁False", "ĠFalse", " False", "False")


def _single_token_id(tokenizer, surfaces):
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for surface in surfaces:
        try:
            token_id = tokenizer.convert_tokens_to_ids(surface)
        except Exception:
            token_id = None
        if isinstance(token_id, int) and token_id >= 0 and token_id != unk_id:
            return token_id
        try:
            ids = tokenizer.encode(surface, add_special_tokens=False)
        except Exception:
            ids = []
        if len(ids) == 1 and ids[0] != unk_id:
            return ids[0]
    return None


def verdict_token_ids(lm):
    """(true_id, false_id) in lm's own vocabulary, or None if they can't be resolved.

    The ids used to be hardcoded to 5852 / 7700, which are ``_True`` / ``_False``
    in the LLaMA-1/2 SentencePiece vocabulary only; the same offsets are
    arbitrary tokens in any other vocabulary.
    """
    cached = getattr(lm, "_verdict_token_ids_cache", None)
    if cached is not None:
        return cached or None

    tokenizer = getattr(lm, "tokenizer", None)
    ids = None
    if tokenizer is not None:
        true_id = _single_token_id(tokenizer, _TRUE_SURFACES)
        false_id = _single_token_id(tokenizer, _FALSE_SURFACES)
        if true_id is not None and false_id is not None and true_id != false_id:
            ids = (true_id, false_id)

    if ids is None:
        logging.warning(
            "Could not resolve True/False token ids for %s; scoring from decoded text instead.",
            getattr(lm, "model_name", type(lm).__name__))
    lm._verdict_token_ids_cache = ids or False
    return ids


def parse_verdict(generated_answer):
    """Map a model answer onto True / False, or None when it cannot be read.

    Never guesses: an answer that is not an unambiguous verdict returns None so
    the caller can flag it instead of silently counting it as supported.
    """
    if generated_answer is None:
        return None

    text = generated_answer.strip()
    if not text:
        return None

    # constrained backends return {"verdict": "True"}
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        payload = None
    if isinstance(payload, dict):
        verdict = payload.get("verdict")
        if isinstance(verdict, str) and verdict.strip().lower() in ("true", "false"):
            return verdict.strip().lower() == "true"
        if isinstance(verdict, bool):
            return verdict
        return None

    words = text.lower().translate(str.maketrans("", "", string.punctuation)).split()
    if not words:
        return None
    if words[0] in ("true", "false"):
        return words[0] == "true"

    has_true = "true" in words
    has_false = "false" in words
    if has_true != has_false:
        return has_true
    return None


class FactScorer(object):

    def __init__(self,
                 model_name="retrieval+gpt-4o-mini",
                 data_dir=".cache/factscore",
                 model_dir=".cache/factscore",
                 cache_dir=".cache/factscore",
                 openai_key="api.key",
                 cost_estimate="consider_cache",
                 abstain_detection_type=None,
                 batch_size=256,
                 max_workers=None,
                 reasoning_effort=None):
        assert model_name in ["retrieval+llama", "retrieval+llama+npm", "retrieval+ChatGPT", "npm", 
                              "retrieval+ChatGPT+npm", "ChatGPT", "gpt-oss", "retrieval+gpt-oss-20b", 
                              "hf-inf", "retrieval+hf-inf", "gpt-4o-mini", "gpt-5-mini", "gpt-5.6-luna",
                              "retrieval+gpt-4o-mini", "retrieval+gpt-5-mini", "retrieval+gpt-5.6-luna"]
        self.model_name = model_name

        self.db = {}
        self.retrieval = {}
        self.npm = {}
        self.batch_size = batch_size # batch size for retrieval
        self.openai_key = openai_key
        self.abstain_detection_type = abstain_detection_type

        self.data_dir = data_dir
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.af_generator = None
        self.cost_estimate = cost_estimate

        if "llama" in model_name:
            self.lm = CLM("inst-llama-7B",
                          model_dir=os.path.join(model_dir, "inst-llama-7B"),
                          cache_file=os.path.join(cache_dir, "inst-llama-7B.pkl"))
        elif "ChatGPT" in model_name:
            self.lm = OpenAIModel("ChatGPT",
                                  cache_file=os.path.join(cache_dir, "ChatGPT.pkl"),
                                  key_path=openai_key)
        elif "oss" in model_name: 
            self.lm = Oss(model_name="gpt-oss-20b")
        elif "gpt" in model_name: 
            self.lm = OpenAIModel(model_name=model_name,
                                  cache_file=os.path.join(cache_dir, "ChatGPT.pkl"),
                                  key_path=openai_key)
            #self.backup_lm = OpenAIModel(model_name='')
        elif "hf" in model_name: 
            self.lm = HFInf(model_name="hf-inf", model_id="meta-llama/Llama-3.2-3B-Instruct", cache_file=os.path.join(cache_dir, "hf.pkl"))
        else:
            self.lm = None

        # non-reasoning, so verdicts come back with logprobs attached
        self.backup_lm = OpenAIModel(model_name='gpt-4o-mini',
                                  cache_file=os.path.join(cache_dir, "ChatGPT.pkl"),
                                  key_path=openai_key)

        # verdict requests for different atoms are independent, so they are sent
        # concurrently; lower this if the API account's rate limit is tight
        if max_workers is not None:
            for lm in (self.lm, self.backup_lm):
                if lm is not None:
                    lm.max_concurrency = max_workers

        # overrides the per-model default in openai_lm.MODEL_ROUTES; only the
        # reasoning models look at it
        if reasoning_effort is not None:
            for lm in (self.lm, self.backup_lm):
                if isinstance(lm, OpenAIModel):
                    lm.reasoning_effort = reasoning_effort

    def save_cache(self):
        if self.lm:
            self.lm.save_cache()
        if "npm" in self.model_name:
            for k, v in self.npm.items():
                v.save_cache()
        for k, v in self.retrieval.items():
            v.save_cache()

    def register_knowledge_source(self, name="enwiki-20230401", db_path=None, data_path=None):
        assert name not in self.retrieval, f"{name} already registered"
        if db_path is None:
            db_path = os.path.join(self.data_dir, f"{name}.db")

        if data_path is None:
            data_path = os.path.join(self.data_dir, f"{name}.jsonl")

        cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.json")
        embed_cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.pkl")

        self.db[name] = DocDB(db_path=db_path, data_path=data_path)
        self.retrieval[name] = Retrieval(self.db[name], cache_path, embed_cache_path, batch_size=self.batch_size)
        if "npm" in self.model_name:
            cache_path = os.path.join(self.cache_dir, f"bm25-{name}.json")
            embed_cache_path = os.path.join(self.cache_dir, f"bm25-{name}.pkl")
            self.npm[name] = NPM(Retrieval(self.db[name], cache_path, embed_cache_path, "bm25"),
                                 "npm-single",
                                 cache_file=os.path.join(self.cache_dir, f"npm-{name}.pkl"))


    def print_cost_estimates(self, total_words, task, model):
        # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        # Number of tokens are roughly 4/3 of the number of words
        total_tokens = total_words * 4.0 / 3

        # https://openai.com/pricing
        # if we use davinci-003, the cost is $0.02 per 1000 tokens
        # if we use gpt-3.5-turbo, the cost is $0.002 per 1000 tokens
        if model == "davinci-003":
            rate = 0.02
        elif model == "gpt-3.5-turbo":
            rate = 0.002

        total_cost = total_tokens * rate / 1000

        # print the total words, tokens, and cost along with rate
        logging.critical("Estimated OpenAI API cost for %s ($%.3f per 1000 tokens): $%.2f for %d words and %d tokens" % (task, rate, total_cost, total_words, total_tokens))

    def get_score(self,
                  topics,
                  generations,
                  true_answers = None, 
                  questions = None, 
                  gamma=10,
                  atomic_facts=None,
                  knowledge_source=None,
                  verbose=False, 
                  do_matching = False, 
                  gen_words = None, 
                  word_tokens = None, 
                  tokenizer_name = None):
        
        if knowledge_source is None:
            # use the default knowledge source
            knowledge_source = "enwiki-20230401"

        if knowledge_source not in  self.retrieval and self.model_name not in ["ChatGPT", "gpt-oss-20b", "hf-inf", "gpt-4o-mini", "gpt-5-mini"]:
            self.register_knowledge_source(knowledge_source)
        else:
            knowledge_source = None

        if type(topics)==type(generations)==str:
            topics = [topics]
            generations = [generations]
        else:
            assert type(topics)==type(generations)==list, "`topics` and `generations` should be lists."
            assert len(topics)==len(generations), "`topics` and `generations` should have the same length"

        if atomic_facts is not None:
            assert len(topics)==len(atomic_facts), "`topics` and `atomic_facts` should have the same length"
            corresponding_sentences = atomic_facts
        else:
            if self.af_generator is None:
                self.af_generator = AtomicFactGenerator(key_path=self.openai_key,
                                                        demon_dir=os.path.join(self.data_dir, "demos"),
                                                        gpt3_cache_file=os.path.join(self.cache_dir, "InstructGPT.pkl"))

            # estimate the total cost of atomic fact generation
            total_words = 0
            #for gen in generations:
            #    total_words += self.af_generator.run(gen, cost_estimate=self.cost_estimate)

            # self.print_cost_estimates(total_words, task="atomic fact generation", model="davinci-003")

            if verbose:
                topics = tqdm(topics)

            # list of all facts (list) across all generations and topics
            atomic_facts = []
            corresponding_sentences = []
            for topic, gen in zip(topics, generations):
                # optionally, first detect if the response is abstained
                response_abstained = is_response_abstained(gen, self.abstain_detection_type)
                if response_abstained:
                    atomic_facts.append(None)
                    # Maybe remove
                    corresponding_sentences.append(None)
                    continue
                # continue only when the response is not abstained
                curr_afs_og, _ = self.af_generator.run(gen)
                # list of facts in the respective generation
                curr_afs = [fact for _, facts in curr_afs_og for fact in facts]
                curr_sent = [sent for sent, facts in curr_afs_og for _ in facts] #[sent for sent, _ in curr_afs_og] #[sent for sents, _  in curr_afs_og for sent in sents]
                if len(curr_afs)==0:
                    atomic_facts.append(None)
                    corresponding_sentences.append(None)
                else:
                    atomic_facts.append(curr_afs)
                    corresponding_sentences.append(curr_sent)
                if len(atomic_facts) % 10 == 0:
                    self.af_generator.save_cache()

            assert len(atomic_facts)==len(topics)
            self.af_generator.save_cache()

        respond_ratio = np.mean([facts is not None for facts in atomic_facts])

        # if "ChatGPT" in self.model_name:
        #     # estimate the total cost of response generation
        #     total_words = 0
        #     for topic, generation, facts in zip(topics, generations, atomic_facts):
        #         if facts is not None:
        #             total_words += self._get_score(topic, generation, facts, knowledge_source, cost_estimate=self.cost_estimate)

        #     self.print_cost_estimates(total_words, task="factscore evaluation", model="gpt-3.5-turbo")

        if verbose:
            topics = tqdm(topics)

        scores = []
        score_list = []
        init_scores = []
        decisions = []
        
        if questions is None: 
            questions = [None] * len(generations)
        if true_answers is None: 
            true_answers = [None] * len(generations)
        if gen_words is None: 
            true_answers = [None] * len(generations)
        
        
        rows = list(zip(topics, generations, atomic_facts, corresponding_sentences,
                        true_answers, questions, gen_words, word_tokens))

        # Phase 1 (serial): render every verdict prompt up front. Retrieval has to
        # stay on one thread — DocDB holds a sqlite connection and Retrieval
        # mutates its cache unguarded.
        requests_per_row = []
        for topic, generation, facts, sentences, true_answer, question, gen_word, wt in rows:
            if facts is None:
                requests_per_row.append(None)
                continue
            requests, _ = self._build_score_requests(topic, generation, facts, sentences,
                                                     knowledge_source, question=question)
            requests_per_row.append(requests)
            if len(requests_per_row) % 10 == 0:
                self.save_cache()
        self.save_cache()

        # Phase 2 (parallel): one flat pool across all generations, so the
        # in-flight budget stays filled even when a single generation has few atoms.
        self._run_score_requests([request
                                  for requests in requests_per_row if requests
                                  for request in requests])

        # Phase 3 (serial): verdict parsing, npm and matching are all local work.
        for row, requests in zip(rows, requests_per_row):
            topic, generation, facts, sentences, true_answer, question, gen_word, wt = row
            if requests is None:
                decisions.append(None)
            else:
                decision = self._finalize_decisions(requests, knowledge_source, do_matching=do_matching, gen_words=gen_word, word_tokens = wt, tokenizer_name = tokenizer_name)
                # atoms whose verdict could not be parsed are excluded, not counted as supported
                verdicts = [d["is_supported"] for d in decision if d["is_supported"] is not None]
                num_unparsed = len(decision) - len(verdicts)
                score = np.mean(verdicts) if verdicts else np.nan
                sents = [d["sentence"] for d in decision]
                # if gamma:
                #     init_scores.append(score)
                #     penalty = 1.0 if len(facts)>gamma else np.exp(1-gamma/len(facts))
                #     score = penalty * score

                decisions.append(decision)
                scores.append(score)
                score_list.append({"facts" : facts, "sentence" : sents, "score" : score, "generation" : generation, "topic" : topic, "num_unparsed" : num_unparsed})
                if len(scores) % 10 == 0:
                    self.save_cache()
        self.save_cache()

        out = { 
                # scores across facts in generation
                "scores" : score_list, 
                # score for all generation
                "score": np.nanmean(scores) if scores else np.nan,
                "respond_ratio": respond_ratio,
                "decisions": decisions,
                "sentences" : corresponding_sentences,
                "num_facts_per_response": np.mean([len(d) for d in decisions if d is not None])}

        if gamma:
            out["init_score"] = np.mean(init_scores)
        
        return out

    def _get_score(self, topic, generation, atomic_facts, sentences, knowledge_source = None, cost_estimate=None, do_matching = False, question = None, gen_words = None, word_tokens = None, tokenizer_name = None):
        """Score one generation: build the prompts, dispatch them, read the verdicts.

        get_score() runs the same three phases itself, pooling the dispatch across
        every generation at once; this path is for single-generation callers.
        """
        requests, total_words = self._build_score_requests(
            topic, generation, atomic_facts, sentences, knowledge_source,
            cost_estimate=cost_estimate, question=question)

        if cost_estimate:
            return total_words

        self._run_score_requests(requests)
        return self._finalize_decisions(requests, knowledge_source, do_matching=do_matching,
                                        gen_words=gen_words, word_tokens=word_tokens,
                                        tokenizer_name=tokenizer_name)

    def _build_score_requests(self, topic, generation, atomic_facts, sentences, knowledge_source = None, cost_estimate=None, question = None):
        """One pending request per atom, with its prompt already rendered.

        Serial by design: this is the phase that hits retrieval. Returns
        (requests, total_words); the word count is only meaningful under
        cost_estimate, which short-circuits before any request is queued.
        """
        requests = []
        total_words = 0
        for atom, sent in zip(atomic_facts, sentences):
            atom = atom.strip()
            retrieval_failed = False
            prompt = None
            lm_used = None
            if self.lm:
                if knowledge_source != None:
                    try:
                        passages = self.retrieval[knowledge_source].get_passages(topic, atom, k=5)
                    except Exception as e:
                        print(repr(e))
                        retrieval_failed = True
                        passages = None

                    if retrieval_failed:
                        # no context available: fall back to a context-free judgement
                        prompt = f"""You are given a derived fact from the generated answer about {topic}.
                        \nDetermine if the derived fact is true or false.
                        \n{VERDICT_INSTRUCTION}
                        \nDerived Fact: {atom} True or False?\nOutput:"""
                    else:
                        definition = "Answer the question about {} based on the given context.\n\n".format(topic)
                        context = ""
                        print("---------------- len passages", len(passages))
                        for psg_idx, psg in enumerate(reversed(passages)):
                            context += "Title: {}\nText: {}\n\n".format(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
                        definition += context.strip()
                        if not definition[-1] in string.punctuation:
                            definition += "."
                        prompt = "{}\n\nInput: {} True or False?\n{}\nOutput:".format(
                            definition.strip(), atom.strip(), VERDICT_INSTRUCTION)

                    lm_used = self.lm
                else:
                    prompt = f"""You are given a generated answer, a derived fact from the generated answer to the question {question}.
                    \nDetermine if the derived fact is true or false. If you are unsure, do a websearch to determine the answers.
                    \n{VERDICT_INSTRUCTION}
                    \nGenerated answer: {generation}
                    \nDerived Fact: {atom} True or False?\nOutput:"""

                    lm_used = self.lm

                if cost_estimate:
                    if cost_estimate == "consider_cache" and (prompt.strip() + "_0") not in self.lm.cache_dict:
                        total_words += len(prompt.split())
                    elif cost_estimate == "ignore_cache":
                        total_words += len(prompt.split())
                    continue

            requests.append({"topic": topic,
                             "atom": atom,
                             "sentence": sent,
                             "prompt": prompt,
                             "lm": lm_used,
                             "retrieval_failed": retrieval_failed,
                             # filled in by _run_score_requests; stays None when
                             # there is no lm to ask
                             "output": None})

        return requests, total_words

    def _run_score_requests(self, requests):
        """Answer every pending prompt, one concurrent pool per backend."""
        by_backend = defaultdict(list)
        for request in requests:
            if request["prompt"] is not None:
                by_backend[id(request["lm"])].append(request)

        for grouped in by_backend.values():
            lm_used = grouped[0]["lm"]
            outputs = lm_used.generate_batch([request["prompt"] for request in grouped],
                                             response_format=VERDICT_RESPONSE_FORMAT)
            for request, output in zip(grouped, outputs):
                request["output"] = output

    def _finalize_decisions(self, requests, knowledge_source = None, do_matching = False, gen_words = None, word_tokens = None, tokenizer_name = None):
        """Turn answered requests into decision records. Local work only."""
        decisions = []
        for request in requests:
            atom = request["atom"]
            sent = request["sentence"]
            output = request["output"]
            parse_failed = False
            p_true = None

            if output is None:
                # no lm configured for this run
                is_supported = True
            else:
                lm_used = request["lm"]
                metadata = output[1]
                logits = metadata if isinstance(metadata, np.ndarray) else None
                verdict_ids = verdict_token_ids(lm_used) if logits is not None else None
                # API backends have no local vocabulary to index, so they report
                # P(True) directly instead of raw logits
                if isinstance(metadata, dict):
                    p_true = metadata.get("p_true")

                if verdict_ids is not None and max(verdict_ids) < logits.shape[0]:
                    # when logits are available and the vocabulary is known
                    true_id, false_id = verdict_ids
                    is_supported = bool(logits[true_id] > logits[false_id])
                elif p_true is not None:
                    is_supported = bool(p_true > 0.5)
                else:
                    # no logits, or a vocabulary we cannot index: read the decoded text
                    is_supported = parse_verdict(output[0])
                    if is_supported is None:
                        parse_failed = True
                        print("unparseable verdict: {!r}".format(output[0]))

            if is_supported and "npm" in self.model_name:
                npprob = self.npm[knowledge_source].get_probabilty(request["topic"], atom)
                is_supported = npprob > 0.3

            if do_matching:
                set_gen_words = set(word.lower() for word in gen_words)   # set = fast lookups, order untouched
                # maybe switch to atom?
                match_words = [w for w in word_tokenize(sent.lower()) if w in set_gen_words]
                matched_word_indices, token_indices, word_token_indices = self._match_string(sent, match_words, gen_words, word_tokens, tokenizer_name)
            else:
                match_words = gen_words
                matched_word_indices = list(range(len(gen_words)))
                token_indices = None
                word_token_indices = None

            decisions.append({"atom": atom,
                              # True / False, or None when the verdict was unparseable
                              "is_supported": is_supported,
                              # P(True) when the backend returned logprobs, else None
                              "p_true": p_true,
                              "retrieval_failed": request["retrieval_failed"],
                              "parse_failed": parse_failed,
                              "sentence" : sent,
                              "matched_words" : match_words,
                              "matched_word_indices" : matched_word_indices,
                              "matched_token_indices" : token_indices,
                              "matched_word_token_indices" : word_token_indices})

        return decisions

    def _match_string(self, sentence, matched_words, generated_words, word_tokens, tokenizer_name): 
        
        #tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # def find_sentence(generated_words, word_tokens, sentence): 
        #     print(sentence)
        #     sentence_indices, sentence_words, sentence_ids = None, None, None
        #     for e, element in enumerate(generated_words): 
        #         for i in range(e+1 , len(generated_words)+1): 
        #             ids_tokens = list(itertools.chain.from_iterable(word_tokens[e:i]))
        #             text_tokens = tokenizer.convert_ids_to_tokens(ids_tokens, skip_special_tokens=False)
        #             text = tokenizer.convert_tokens_to_string(text_tokens)
        #             #print(text, len(text), len(sentence), text in sentence)
        #             if text in sentence: 
        #                 sentence_indices = list(range(e, i))
        #                 sentence_words = generated_words[e:i]
        #                 sentence_ids = word_tokens[e:i]
        #                 if len(text) == len(sentence): 
        #                     return sentence_indices, sentence_words, sentence_ids
        #         return sentence_indices, sentence_words, sentence_ids
            
        # def index_nested(lst):
        #     counter = 0

        #     def walk(x):
        #         nonlocal counter
        #         if isinstance(x, list):
        #             return [walk(i) for i in x]

        #         idx = counter
        #         counter += 1
        #         return idx

        #     return walk(lst)

        
        # sentence_indices, sentence_words, sentence_ids = find_sentence(generated_words, word_tokens, sentence)
        
        # if sentence_indices is None: 
        #     return list(range(len(generated_words))), list(range(len(word_tokens)))
        
        # token_sentence_pos = index_nested(sentence_ids)

        # word_indices = defaultdict(list)
        # token_indices = defaultdict(list)
        # word_set = set(matched_words)
        # for word, index, token_idx in zip(sentence_words, sentence_indices, token_sentence_pos):
        #     for element in word_set: 
        #         if word in element:
        #             word_indices[word].append(index)
        #             token_indices[word].append(token_idx)

        # word_indices = dict(word_indices)
        # word_indices_list = word_indices.values()

        # token_indices = dict(token_indices)
        # token_indices_list = token_indices.values()
        

        # cart_prod = itertools.product(*word_indices_list)
        # cart_prod_tokens = itertools.product(*token_indices_list)
        # last_diff = float('inf')
        # final_indices = None
        # final_token_indices = None
        # for element, tokens in zip(cart_prod, cart_prod_tokens):
        #     diff = np.diff(element)
        #     if any(n < 0 for n in diff):
        #         continue
        #     if sum(diff) < last_diff: 
        #         last_diff = sum(diff)
        #         final_indices = element 
        #         final_token_indices = tokens
        
        # -----------------------------------------------
        # def expl(lst):
        #     counter = 0
        #     def walk(x):
        #         nonlocal counter
        #         if isinstance(x, list):
        #             return [walk(i) for i in x]

        #         idx = counter
        #         counter += 1
        #         return idx
        #     return walk(lst)
        
        # token_positions = expl(word_tokens)
        # print("token positions: ", token_positions)
                
        # def find_sentence(generated_words, word_tokens, sentence): 
        #     sentence_indices, sentence_words, sentence_ids = None, None, None
        #     for e, element in enumerate(generated_words): 
        #         for i in range(e+1 , len(generated_words)+1): 
        #             ids_tokens = list(itertools.chain.from_iterable(word_tokens[e:i]))
        #             text_tokens = tokenizer.convert_ids_to_tokens(ids_tokens, skip_special_tokens=False)
        #             text = tokenizer.convert_tokens_to_string(text_tokens)
        #             #print(text, len(text), len(sentence), text in sentence)
        #             if text in sentence: 
        #                 sentence_indices = list(range(e, i))
        #                 sentence_words = generated_words[e:i]
        #                 sentence_ids = token_positions[e:i]
        #                 if len(text) == len(sentence): 
        #                     return sentence_indices, sentence_words, sentence_ids
        #     return sentence_indices, sentence_words, sentence_ids
                    
                    
            
        # # def index_nested(sentence_indices, sentence_ids):
        # #     token_indices = list()
        # #     start_idx = sentence_indices[0]
        # #     for s, sentence_pos in enumerate(sentence_indices):
        # #         indices = list(range(start_idx, start_idx+len(sentence_ids[s])))
        # #         token_indices.append(indices)
        # #         start_idx = start_idx+len(sentence_ids[s])
                
        # #     return token_indices

        
        # sentence_indices, sentence_words, token_sentence_pos = find_sentence(generated_words, word_tokens, sentence)
        
        # if sentence_indices is None: 
        #     return list(range(len(generated_words))), None
        
        # word_indices = defaultdict(list)
        # token_indices = defaultdict(list)
        # # word_set = set(matched_words)
        # print(generated_words)
        # print("word set", matched_words)
        # def normalize(s):
        #     # Only strip trailing punctuation if string contains normal chars
        #     # and contains punctuation/special chars
        #     has_normal_chars = bool(re.search(r'[a-zA-Z0-9]', s))
        #     has_special_chars = bool(re.search(r'[^\w\s]', s))
            
        #     if not has_normal_chars or not has_special_chars:
        #         return s
        #     return re.sub(r'[^\w\s]+$', '', s).strip()
        
        # for word, index, token_idx in zip(sentence_words, sentence_indices, token_sentence_pos):
        #     word = word.strip()
        #     for element in matched_words: 
        #         if word == element or normalize(word) == normalize(element):
        #             print(word , element)
        #             word_indices[word].append(index)
        #             token_indices[word].append(token_idx)

        # word_indices = dict(word_indices)
        # word_indices_list = word_indices.values()

        # token_indices = dict(token_indices)
        # token_indices_list = token_indices.values()
        
        # print("sentence words", sentence_words)
        # print("word indices list: ", word_indices_list)

        # cart_prod = itertools.product(*word_indices_list)
        
        # cart_prod_tokens = itertools.product(*token_indices_list)
        # last_diff = float('inf')
        # final_indices = None
        # final_token_indices = None
        # for element, tokens in zip(cart_prod, cart_prod_tokens):
        #     diff = np.diff(element)
        #     if any(n < 0 for n in diff):
        #         continue
        #     if sum(diff) < last_diff: 
        #         last_diff = sum(diff)
        #         final_indices = element 
        #         final_token_indices = tokens
        # # print(fact['fact'])
        # print('final_indices',final_indices)
        # print('final_token_indices',final_token_indices)
        # print("-------------------------------------")
        
        def expl(lst):
            counter = 0
            def walk(x):
                nonlocal counter
                if isinstance(x, list):
                    return [walk(i) for i in x]

                idx = counter
                counter += 1
                return idx
            return walk(lst)
        
        token_positions = expl(word_tokens)
        
        word_to_indices = defaultdict(list)
        token_to_indices = defaultdict(list)
        def normalize(word):
            return re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", word.strip().lower())

        for i, w in enumerate(generated_words):
            word_to_indices[normalize(w)].append(i)
            token_to_indices[normalize(w)].append(token_positions[i])

        word_indices_list = [sorted(word_to_indices[normalize(w)]) for w in matched_words]
        token_indices_list = [sorted(token_to_indices[normalize(w)]) for w in matched_words]

        best_indices, best_token_indices = self._find_best_sequence(word_indices_list, token_indices_list)

        if best_indices is None:
            return {}, {}, {}

        # Key by (word, occurrence#) rather than just the word, since matched_words
        # can contain the same word more than once - a plain word key would silently
        # overwrite earlier occurrences.
        word_occurrence_counts = defaultdict(int)
        final_indices = {}
        final_token_indices = {}
        word_to_token_indices = defaultdict(list)
        for w, idx, tok in zip(matched_words, best_indices, best_token_indices):
            key = (w, word_occurrence_counts[w])
            word_occurrence_counts[w] += 1
            final_indices[key] = idx
            final_token_indices[key] = tok
            word_to_token_indices[w].append(tok)

        return final_indices, final_token_indices, dict(word_to_token_indices)
    
    def _find_best_sequence(self, word_indices_list, token_indices_list):
        # Layered shortest-path DP: pick one candidate per position (in order),
        # requiring indices to strictly increase from one position to the next.
        # This keeps the matched words in their natural left-to-right order in
        # the sentence, minimizes the total span, and - since no index can equal
        # an earlier one - guarantees every chosen index is used at most once.
        # Keeping only the best cost to reach each (depth, candidate) node -
        # instead of enumerating every path, as a naive search would - keeps
        # this polynomial instead of exponential in the number of candidates
        # per word.
        try:
            if not word_indices_list:
                return None, None

            costs = [0] * len(word_indices_list[0])
            backpointers = []

            for depth in range(1, len(word_indices_list)):
                prev_indices = word_indices_list[depth - 1]
                cur_indices = word_indices_list[depth]
                cur_costs = [float('inf')] * len(cur_indices)
                cur_back = [-1] * len(cur_indices)

                for j, idx in enumerate(cur_indices):
                    for k, prev_idx in enumerate(prev_indices):
                        if idx <= prev_idx or costs[k] == float('inf'):
                            continue
                        cost = costs[k] + (idx - prev_idx)
                        if cost < cur_costs[j]:
                            cur_costs[j] = cost
                            cur_back[j] = k

                backpointers.append(cur_back)
                costs = cur_costs

            if all(cost == float('inf') for cost in costs):
                return None, None

            best_j = min(range(len(costs)), key=lambda j: costs[j])

            best_indices = [None] * len(word_indices_list)
            best_tokens = [None] * len(word_indices_list)
            j = best_j
            for depth in range(len(word_indices_list) - 1, -1, -1):
                best_indices[depth] = word_indices_list[depth][j]
                best_tokens[depth] = token_indices_list[depth][j]
                j = backpointers[depth - 1][j] if depth > 0 else None

            return tuple(best_indices), tuple(best_tokens)
        except Exception as e:
            print("Exception", e)
            return (None, None)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',
                        type=str,
                        default="data/labeled/InstructGPT.jsonl")
    parser.add_argument('--model_name',
                        type=str,
                        default="retrieval+ChatGPT")
    parser.add_argument('--gamma',
                        type=int,
                        default=10,
                        help="hyperparameter for length penalty")

    parser.add_argument('--openai_key',
                        type=str,
                        default="api.key")
    parser.add_argument('--data_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--model_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--cache_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--knowledge_source',
                        type=str,
                        default=None)
    parser.add_argument('--cost_estimate',
                        type=str,
                        default="consider_cache",
                        choices=["consider_cache", "ignore_cache"])
    parser.add_argument('--abstain_detection_type',
                        type=str,
                        default=None,
                        choices=["perplexity_ai", "generic", "none"])
    parser.add_argument('--use_atomic_facts',
                        action="store_true")
    parser.add_argument('--verbose',
                        action="store_true",
                        help="for printing out the progress bar")    
    parser.add_argument('--print_rate_limit_error',
                        action="store_true",
                        help="for printing out rate limit error when using OpenAI keys")
    parser.add_argument('--n_samples',
                        type=int,
                        default=None)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.ERROR if args.print_rate_limit_error else logging.CRITICAL)

    fs = FactScorer(model_name=args.model_name,
                    data_dir=args.data_dir,
                    model_dir=args.model_dir,
                    cache_dir=args.cache_dir,
                    openai_key=args.openai_key,
                    cost_estimate=args.cost_estimate,
                    abstain_detection_type=args.abstain_detection_type)

    tot = 0
    topics, generations, atomic_facts = [], [], []
    with open(args.input_path) as f:
        for line in f:
            dp = json.loads(line)
            tot += 1
            if args.use_atomic_facts:
                assert "annotations" in dp, "You can specify `--use_atomic_facts` only when atomic facts are available in the input data already."
                if dp["annotations"] is None:
                    continue
                topics.append(dp["topic"])
                generations.append(dp["output"])
                atomic_facts.append([atom["text"] for sent in dp["annotations"] for atom in sent["model-atomic-facts"]])
            else:
                topics.append(dp["topic"])
                generations.append(dp["output"])
            if args.n_samples is not None and tot==args.n_samples:
                break
    out = fs.get_score(topics=topics,
                       generations=generations,
                       gamma=args.gamma,
                       atomic_facts=atomic_facts if args.use_atomic_facts else None,
                       knowledge_source=args.knowledge_source,
                       verbose=args.verbose)
    logging.critical("FActScore = %.1f%%" % (100*out["score"]))
    if "init_score" in out:
        logging.critical("FActScore w/o length penalty = %.1f%%" % (100*out["init_score"]))
    logging.critical("Respond ratio = %.1f%%" % (100*out["respond_ratio"]))
    logging.critical("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))

    # Save out as a json file
    with open(args.input_path.replace(".jsonl", f"_factscore_output.json"), 'w') as f:
        f.write(json.dumps(out) + "\n")

