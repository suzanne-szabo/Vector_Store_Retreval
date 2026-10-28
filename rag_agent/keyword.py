"""Keyword search: a BM25 index over the stored chunks, built in memory at query time.

BM25 (Okapi) scores a chunk higher when it contains the query's words, especially words that
are rare across the corpus, while discounting long chunks. It complements vector search,
which matches meaning but can miss exact technical terms such as `JALR`, `callee-saved` or a
section heading.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

K1 = 1.5          # term-frequency saturation
B = 0.75          # length normalization

# Common English words that carry no search meaning. Kept short on purpose: programming words
# such as "if", "for", "in", "is", "not", "return" or "and" are meaningful in Python questions,
# so only words that are never the point of a query are dropped.
STOPWORDS = frozenset(
    "a an are as at be been by can could did do does from has have how i it its me my of on or "
    "our should that the their them then there these they this those to was we were what when "
    "where which who whom why will would you your about into than".split()
)

# A token is letters, digits and underscores, optionally joined by '-' or '.' inside the token:
# 'callee-saved', 'x10-x17', '__init__', '2.5', 'rv32i', 'jalr'.
_TOKEN = re.compile(r"[a-z0-9_]+(?:[-.][a-z0-9_]+)*")
_PART = re.compile(r"[-.]")


def tokenize(text: str) -> list[str]:
    """Lowercase tokens; a joined token also contributes its parts ('callee-saved' -> 'callee', 'saved')."""
    tokens: list[str] = []
    for token in _TOKEN.findall(text.lower()):
        if token in STOPWORDS:
            continue
        tokens.append(token)
        if _PART.search(token):
            tokens.extend(part for part in _PART.split(token) if part and part not in STOPWORDS)
    return tokens


@dataclass(frozen=True)
class KeywordHit:
    chunk_id: str
    score: float


class BM25Index:
    """An in-memory BM25 index over (chunk_id, text, metadata) records."""

    def __init__(self, records: Iterable[tuple[str, str, dict]], k1: float = K1, b: float = B) -> None:
        self._k1, self._b = k1, b
        self._ids: list[str] = []
        self._meta: list[dict] = []
        self._lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for chunk_id, text, metadata in records:
            index = len(self._ids)
            counts = Counter(tokenize(text))
            self._ids.append(chunk_id)
            self._meta.append(metadata)
            self._lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self._postings.setdefault(term, []).append((index, frequency))
        self._average = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0

    def __len__(self) -> int:
        return len(self._ids)

    def idf(self, term: str) -> float:
        """Inverse document frequency; always positive (the BM25+ style log(1 + ...))."""
        count = len(self._postings.get(term, ()))
        total = len(self._ids)
        return math.log(1 + (total - count + 0.5) / (count + 0.5))

    def _allowed(self, index: int, sources: set[str] | None, subject: str | None) -> bool:
        meta = self._meta[index]
        if sources is not None and meta.get("source_id") not in sources:
            return False
        return subject is None or meta.get("subject") == subject

    def search(self, query: str, k: int, sources: list[str] | None = None,
               subject: str | None = None) -> list[KeywordHit]:
        """The `k` best-scoring chunks that contain at least one query term, best first.

        Ties keep index order, so results are deterministic.
        """
        allowed_sources = set(sources) if sources is not None else None
        scores: dict[int, float] = {}
        for term in dict.fromkeys(tokenize(query)):          # unique terms, in query order
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for index, frequency in postings:
                if not self._allowed(index, allowed_sources, subject):
                    continue
                norm = self._k1 * (1 - self._b + self._b * self._lengths[index] / self._average)
                scores[index] = scores.get(index, 0.0) + idf * frequency * (self._k1 + 1) / (frequency + norm)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [KeywordHit(self._ids[index], score) for index, score in ranked]
