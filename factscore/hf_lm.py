from factscore.lm import LM
from huggingface_hub import InferenceClient
import sys
import time
import os
import numpy as np
import logging

class HFInf(LM):

    def __init__(self, model_name, model_id, cache_file=None):
        self.model_name = model_name
        self.model_id = model_id
        self.temp = 0.7
        self.save_interval = 100
        self.model = InferenceClient(model=model_id, api_key=os.getenv('HF_TOKEN'))
        super().__init__(cache_file)

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128):
        message = [{"role": "user", "content": prompt}]
        out = self.model.chat_completion(
            messages = message,
            temperature=self.temp,        
            top_p=0.9,
            )
        return out.choices[0].message["content"]


