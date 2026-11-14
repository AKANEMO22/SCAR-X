"""Explicit bounded arithmetic tasks, not a general user-intent verifier.

An incidental expression in code, quoted text, a color question or a longer
instruction must not authorize a factual learning label. Unsupported wording
still uses ordinary conversation/review; it is not proved right or wrong.
The original SCAR-X research verifier remains unchanged.
"""
import re
import unicodedata

from ..learning.verifiers import integer_answer

VERIFIER_ID = 'integer-arithmetic-explicit-task-v2'
_TASK = re.compile(
    r"(?:(?:chào bạn|xin chào|hi|hello)\s*,\s*)?"
    r"(?:(?:hãy tính|tính(?: giúp tôi)?|calculate|compute|what is|what's|how much is|kết quả của)\s*:?\s+)?"
    r"(?P<a>[0-9]{1,2})\s*(?P<op>\+|-|−|cộng|trừ|plus|minus)\s*(?P<b>[0-9]{1,2})"
    r"\s*(?:(?:bằng|là) bao nhiêu|bằng mấy|=)?\s*[?.!]?"
)


def parse_integer_expression(text):
    if not isinstance(text, str) or not 1 <= len(text) <= 512:
        return None
    normalized = ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
    match = _TASK.fullmatch(normalized)
    if match is None:
        return None
    operation = '+' if match['op'] in ('+', 'cộng', 'plus') else '-'
    return int(match['a']), operation, int(match['b'])
