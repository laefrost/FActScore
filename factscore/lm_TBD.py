import pickle
import os
import threading
import time

from concurrent.futures import ThreadPoolExecutor

class LM(object):

    # How many requests a backend tolerates in flight at once. API-backed
    # subclasses raise this; local (GPU) models keep 1, where extra threads only
    # contend for the same device.
    max_concurrency = 1

    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache_dict = self.load_cache()
        self.model = None
        self.add_n = 0
        # guards cache_dict / add_n, which generate_batch touches from workers
        self.cache_lock = threading.RLock()
        self.suspend_autosave = False

    def load_model(self):
        # load the model and put it as self.model
        raise NotImplementedError()

    def generate(self, prompt, sample_idx=0, max_sequence_length=2048, max_output_length=128, response_format = None):
        prompt = prompt.strip() # it's important not to end with a whitespace
        cache_key = f"{prompt}_{sample_idx}"

        with self.cache_lock:
            if cache_key in self.cache_dict:
                return self.cache_dict[cache_key]

            if self.model is None:
                self.load_model()

        if prompt.endswith(" True or False?\nAnswer:"):
            generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=1, response_format = response_format)
        else:
            generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=max_output_length, response_format = response_format)

        with self.cache_lock:
            self.cache_dict[cache_key] = generated
            self.add_n += 1
        return generated

    def generate_batch(self, prompts, sample_idx=0, max_sequence_length=2048,
                       max_output_length=128, response_format=None, max_workers=None):
        """generate() over many prompts, returning one result per prompt, in order.

        Prompts are independent of each other, so on API backends they are sent
        concurrently instead of one round trip at a time. Duplicates are issued
        once — generate() keys its cache on the prompt alone, so two identical
        prompts always shared an answer anyway.
        """
        if max_workers is None:
            max_workers = self.max_concurrency

        def cache_key(prompt):
            return f"{prompt.strip()}_{sample_idx}"

        position = {}
        unique_prompts = []
        for prompt in prompts:
            key = cache_key(prompt)
            if key not in position:
                position[key] = len(unique_prompts)
                unique_prompts.append(prompt)

        def run(prompt):
            return self.generate(prompt,
                                 sample_idx=sample_idx,
                                 max_sequence_length=max_sequence_length,
                                 max_output_length=max_output_length,
                                 response_format=response_format)

        workers = min(max_workers, len(unique_prompts))
        if workers <= 1:
            outputs = [run(prompt) for prompt in unique_prompts]
        else:
            # one pickle dump at the end rather than one per save_interval hit
            # inside the workers
            self.suspend_autosave = True
            try:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    outputs = list(pool.map(run, unique_prompts))
            finally:
                self.suspend_autosave = False
            self.save_cache()

        return [outputs[position[cache_key(prompt)]] for prompt in prompts]

    def maybe_autosave(self, interval):
        """Periodic cache flush for subclasses; a no-op inside generate_batch."""
        if self.suspend_autosave:
            return
        if self.add_n % interval == 0:
            self.save_cache()

    def save_cache(self):
        with self.cache_lock:
            if self.add_n == 0:
                return

            # load the latest cache first, since if there were other processes running in parallel, cache might have been updated
            for k, v in self.load_cache().items():
                self.cache_dict[k] = v

            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache_dict, f)

    def load_cache(self, allow_retry=True):
        if os.path.exists(self.cache_file):
            while True:
                try:
                    with open(self.cache_file, "rb") as f:
                        cache = pickle.load(f)
                    break
                except Exception:
                    if not allow_retry:
                        assert False
                    print ("Pickle Error: Retry in 5sec...")
                    time.sleep(5)        
        else:
            cache = {}
        return cache



