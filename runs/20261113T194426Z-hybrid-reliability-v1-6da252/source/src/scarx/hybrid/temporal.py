"""Explicit first/last evidence constraints, separate from learned relevance."""

import re
import unicodedata


def temporal_direction(query):
    value = unicodedata.normalize("NFC", query).casefold()
    first = re.search(r"\b(first|beginning|start|đầu)\b", value) is not None
    last = re.search(r"\b(last|ending|end|cuối)\b", value) is not None
    # Comparing both endpoints needs multiple events; do not collapse that query.
    return "first" if first and not last else "last" if last and not first else None
