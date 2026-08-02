"""Shared settings for the SAFE code under ``eval/safe``.

Only the fields SAFE actually reads are defined here. The Serper key is the one
that matters in practice: SAFE grounds every atomic fact in Google Search
results fetched through https://serper.dev.
"""

import os


def _read_key(env_var, *file_names):
    """Key from the environment, else from the first key file that exists."""
    key = os.environ.get(env_var)
    if key:
        return key.strip()

    search_dirs = [os.getcwd(), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
    for directory in search_dirs:
        for file_name in file_names:
            path = os.path.join(directory, file_name)
            if os.path.exists(path):
                with open(path) as f:
                    contents = f.read().strip()
                if contents:
                    return contents
    return ''


# SERPER_API_KEY, or a `serper.key` file in the working directory / repo root.
serper_api_key = _read_key('SERPER_API_KEY', 'serper.key', 'serper_api.key')

# factscore reads the OpenAI key from OPENAI_API_KEY itself; kept for parity
# with the fields the original shared_config exposes.
openai_api_key = _read_key('OPENAI_API_KEY', 'api.key')
anthropic_api_key = _read_key('ANTHROPIC_API_KEY', 'anthropic.key')

# Where run_eval writes its output.
path_to_result = os.environ.get('SAFE_RESULT_DIR', 'results/')

# `eval/safe/config.py` maps its short model name through this table. The
# FactScorer integration does not use it - there the rater is the LM that
# FactScorer was constructed with - but SAFE imports it at module load.
model_options = {
    'gpt_4_turbo': 'OPENAI:gpt-4-0125-preview',
    'gpt_4': 'OPENAI:gpt-4-0613',
    'gpt_4_32k': 'OPENAI:gpt-4-32k-0613',
    'gpt_35_turbo': 'OPENAI:gpt-3.5-turbo-0125',
    'gpt_35_turbo_16k': 'OPENAI:gpt-3.5-turbo-16k-0613',
    'claude_3_opus': 'ANTHROPIC:claude-3-opus-20240229',
    'claude_3_sonnet': 'ANTHROPIC:claude-3-sonnet-20240229',
    'claude_21': 'ANTHROPIC:claude-2.1',
    'claude_20': 'ANTHROPIC:claude-2.0',
    'claude_instant': 'ANTHROPIC:claude-instant-1.2',
}
