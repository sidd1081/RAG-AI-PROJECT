import re
from typing import List

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> List[str]:
    """
    Lightweight, dependency-free sentence splitter. Not perfect on every
    edge case (abbreviations, decimals), but good enough as a chunk-boundary
    heuristic and avoids pulling in spaCy/nltk + their data downloads.
    """
    text = text.strip()
    if not text:
        return []

    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]
