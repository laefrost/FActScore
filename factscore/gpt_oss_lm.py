# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import time
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer

from factscore.utils import convert_model_to_int8_on_gpu
from factscore.lm import LM

class Oss(LM):
    def __init__(self, model_name, model_dir = "openai/gpt-oss-20b", cache_file=None):
        self.model_name = model_name
        self.model_dir = model_dir
        if cache_file:
            super().__init__(cache_file)

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir, 
            device_map="auto",
            torch_dtype="auto")
        # self.model = convert_model_to_int8_on_gpu(self.model, device='cuda')
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_dir)

    def _generate(self, prompts, max_sequence_length=2048, max_output_length=128,
                  end_if_newline=False, end_if_second_newline=False, verbose=False):
        is_single = type(prompts)==str
        if is_single:
            prompts = [prompts]
        if verbose:
            prompts = tqdm(prompts)

        generations = []
        scores = []
        
        for prompt in prompts:
            chat = [
                {"role": "user", "content": prompt}
            ]
            encoded = self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            ).to(self.model.device)
            
            if len(encoded['input_ids']) > max_sequence_length - max_output_length:
                curr_input_ids = encoded['input_ids'][-(max_sequence_length - max_output_length):]
            curr_input_ids = torch.LongTensor([encoded['input_ids']]).cuda()
            
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            with torch.no_grad():
                out = self.model.generate(
                    **encoded,  # Unpacks input_ids and attention_mask
                    do_sample=True,
                    top_p=0.9,
                    return_dict_in_generate=True,
                    output_scores=True, 
                    output_logits=True, 
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_length=curr_input_ids.shape[1]+max_output_length,
                )

            # Get prompt length
            # prompt_length = encoded['input_ids'].shape[1]
            
            full_seq = out.sequences[0] 
            # print(prompt_len, len(full_seq))
            gen_ids = full_seq[curr_input_ids:].tolist()
            gen = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            gen_scores = out["scores"][0][0].detach().cpu().numpy()
            
            if end_if_newline:
                gen = gen.split("\n")[0].strip()
            elif end_if_second_newline:
                gen = "\n".join(gen.split("\n")[:2]).strip()
                
            generations.append(gen)
            scores.append(gen_scores)

        assert len(generations)==len(prompts)==len(scores)
        if is_single:
            return generations[0], scores[0]
        
        return generations, scores

