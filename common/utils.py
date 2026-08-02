"""Helpers used by the SAFE code under ``eval/safe``.

Same names and semantics as the ``common/utils.py`` of the original
long-form-factuality repository; SAFE relies on the exact parsing behaviour of
``extract_first_square_brackets`` and ``extract_first_code_block``, so those are
kept as they were.
"""

import json
import os
import re
from typing import Any

_RED = '\033[91m'
_CYAN = '\033[96m'
_RESET = '\033[0m'


def strip_string(s: str) -> str:
    return s.strip()


def extract_first_square_brackets(input_string: str) -> str:
    """Contents of the first [...] group, or '' when there is none."""
    raw_result = re.findall(r'\[.*?\]', input_string, flags=re.DOTALL)
    return raw_result[0][1:-1] if raw_result else ''


def extract_first_code_block(input_string: str, ignore_language: bool = False) -> str:
    """Contents of the first ``` code block, or '' when there is none."""
    if ignore_language:
        pattern = re.compile(r'```(?:\w+\n)?(.*?)```', re.DOTALL)
    else:
        pattern = re.compile(r'```(.*?)```', re.DOTALL)

    match = pattern.search(input_string)
    return strip_string(match.group(1)) if match else ''


def print_color(message: Any, color: str = '') -> None:
    print(f'{color}{message}{_RESET}' if color else str(message))


def maybe_print_error(message: Any, additional_info: str = '', verbose: bool = False) -> None:
    error = type(message).__name__ if isinstance(message, Exception) else 'ERROR'
    message = str(message)
    message = f'{error}: {message}'
    message += f'\n{additional_info}' if verbose and additional_info else ''
    print_color(message, _RED)


def print_info(message: str, add_punctuation: bool = True) -> None:
    if add_punctuation:
        message = f'{message}.' if message and message[-1] not in '.?!' else message
    print_color(message, _CYAN)


def print_divider() -> None:
    print('_' * 40)


def print_progress(message: str, current: int, total: int) -> None:
    print_info(f'{message}: {current}/{total}')


def read_json(filepath: str) -> dict[str, Any]:
    with open(filepath) as f:
        return json.load(f)


def save_json(filepath: str, data: dict[str, Any]) -> None:
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f)


def get_attributes(module: Any) -> dict[str, Any]:
    """Public, JSON-serializable module-level settings."""
    return {
        name: value
        for name, value in vars(module).items()
        if not name.startswith('_') and isinstance(value, (str, int, float, bool))
    }
