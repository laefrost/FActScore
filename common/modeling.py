"""Language model interface expected by the SAFE code under ``eval/safe``.

SAFE only ever asks a model for text: ``model.generate(prompt, do_debug=...)``
returning a string. This wraps a factscore ``LM`` (``OpenAIModel``, ``CLM``,
``HFInf``, ...), whose ``generate`` returns a ``(text, metadata)`` pair, so the
same backend - and the same on-disk prompt cache - serves both the factscore and
the SAFE fact-checking path.
"""

import logging

_MODEL_NAME_PREFIXES = ('OPENAI:', 'ANTHROPIC:')


def _strip_provider(model_name):
    for prefix in _MODEL_NAME_PREFIXES:
        if model_name.startswith(prefix):
            return model_name[len(prefix):]
    return model_name


class Model:
    """Adapter exposing a factscore LM through SAFE's model interface."""

    def __init__(self,
                 model_name=None,
                 temperature=None,
                 max_tokens=512,
                 lm=None,
                 cache_file=None,
                 key_path='api.key'):
        if lm is None:
            if model_name is None:
                raise ValueError('Pass either an existing factscore `lm` or a `model_name`.')
            # deferred: only the from-scratch path needs the OpenAI client
            from factscore.openai_lm import OpenAIModel
            lm = OpenAIModel(model_name=_strip_provider(model_name),
                             cache_file=cache_file,
                             key_path=key_path)

        self.lm = lm
        self.model_name = getattr(lm, 'model_name', model_name)
        self.temperature = temperature
        self.max_tokens = max_tokens

        # factscore backends carry their decoding temperature on the instance;
        # backends without one (local CLM) keep their own default.
        if temperature is not None and hasattr(self.lm, 'temp'):
            self.lm.temp = temperature

    def generate(self, prompt, do_debug=False, max_tokens=None, **kwargs):
        """One completion, as a plain string ('' when the backend returns nothing)."""
        del kwargs  # SAFE passes nothing else; accepted so callers stay unchanged
        response = self.lm.generate(prompt, max_output_length=max_tokens or self.max_tokens)

        # factscore LMs return (text, metadata); tolerate bare-string backends
        text = response[0] if isinstance(response, tuple) else response
        text = text if isinstance(text, str) else ''

        if do_debug:
            print(f'PROMPT\n{prompt}\n\nRESPONSE\n{text}\n')

        return text

    def print_config(self) -> None:
        logging.critical('Model: %s, temperature: %s, max_tokens: %s',
                         self.model_name, self.temperature, self.max_tokens)
