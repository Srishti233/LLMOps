"""Dependency-free text metrics: ROUGE-L (F1 over LCS) and keyword matching."""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"\w+")


def tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> float:
    pred, ref = tokenize_words(prediction), tokenize_words(reference)
    lcs = _lcs_length(pred, ref)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def keyword_hits(response: str, keywords: list[str]) -> int:
    norm = _norm(response)
    return sum(1 for kw in keywords if _norm(kw) in norm)


def keyword_score(response: str, keywords: list[str]) -> tuple[float, bool]:
    """Return (coverage in [0,1], all_keywords_present)."""
    if not keywords:
        return 1.0, True
    hits = keyword_hits(response, keywords)
    return hits / len(keywords), hits == len(keywords)
