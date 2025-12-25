from factscore.lm import LM
import openai
import sys
import time
import os
import numpy as np
import logging

from factscore.lm import LM
from openai import OpenAI
import sys
import time
import os
import numpy as np
import logging


class OpenAIModel(LM):

    def __init__(self, model_name, cache_file=None, key_path="api.key"):
        self.model_name = model_name
        self.key_path = key_path
        self.temp = 0.7
        self.save_interval = 100
        self.client = None
        super().__init__(cache_file)

    def load_model(self):
        # API key is read automatically from OPENAI_API_KEY
        # Environment variable must be set
        self.client = OpenAI()
        self.model = self.model_name

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128, response_format = None):
        if self.add_n % self.save_interval == 0:
            self.save_cache()

        if self.model_name == "ChatGPT":
            response = call_chat_model(
                client=self.client,
                prompt=prompt,
                model_name="gpt-4o-mini",
                temp=self.temp,
                max_output_tokens=max_output_length,
                response_format=response_format
            )
            output = response["output_text"]
            return output, response

        elif self.model_name == "InstructGPT":
            response = call_instruct_model(
                client=self.client,
                prompt=prompt,
                model_name="gpt-3.5-turbo-instruct",
                temp=self.temp,
                max_output_tokens=max_output_length
            )
            output = response["output_text"]
            return output, response

        else:
            raise NotImplementedError(f"Unknown model: {self.model_name}")


def call_chat_model(
    client,
    prompt,
    model_name="gpt-4o-mini",  # Fixed model name
    max_output_tokens=512,
    temp=0.7,
    response_format = None
):
    received = False
    num_rate_errors = 0
    response = None
    
    while not received:
        try:
            if response_format is None: 
                response = client.chat.completions.create(  # Correct method
                    model=model_name,
                    messages=[  # Correct parameter name
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=max_output_tokens,  # Correct parameter name
                    temperature=temp,
                )
                output_text = response.choices[0].message.content
            else: 
                # response = client.responses.create(
                #     model=model_name,
                #     input=[
                #         {"role": "user", "content": prompt}
                #     ],
                #     text = response_format
                # )
                # output_text = response.output_text
                response = client.chat.completions.create(
                model=model_name,
                messages=[
                    # {"role": "system", "content": "You are a helpful math tutor. Guide the user through the solution step by step."},
                    {"role": "user", "content": prompt}
                    ],
                    response_format=response_format,
                )
                output_text = response.choices[0].message.content
            received = True
        except Exception as e:
            num_rate_errors += 1
            logging.error(
                "API error: %s (%d). Waiting %d sec",
                str(e),
                num_rate_errors,
                2 ** num_rate_errors
            )
            time.sleep(2 ** num_rate_errors)
    
    return {
        "raw_response": response,
        "output_text": output_text  # Correct attribute path
    }

def call_instruct_model(
    client,
    prompt,
    model_name="gpt-3.5-turbo-instruct",
    max_output_tokens=512,
    temp=0.7,
):
    received = False
    num_rate_errors = 0
    response = None
    
    while not received:
        try:
            response = client.completions.create(  # Correct method
                model=model_name,
                prompt=prompt,  # Correct parameter name
                max_tokens=max_output_tokens,  # Correct parameter name
                temperature=temp,
            )
            received = True
        except Exception as e:
            num_rate_errors += 1
            logging.error(
                "API error: %s (%d). Waiting %d sec",
                str(e),
                num_rate_errors,
                2 ** num_rate_errors
            )
            time.sleep(2 ** num_rate_errors)
    
    return {
        "raw_response": response,
        "output_text": response.choices[0].text  # Correct attribute path
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
