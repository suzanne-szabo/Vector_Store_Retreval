"""Read-only retrieval over the vector store: vector, keyword, or hybrid (fused) search.

Hybrid search runs a vector search and a BM25 keyword search with the same filters, then merges
the two ranked lists with Reciprocal Rank Fusion (RRF): each chunk scores 1 / (RRF_K + rank) in
each list it appears in, and the scores are added. A chunk ranked well by both rises to the top;
a chunk found by only one can still appear. Every returned passage records its rank in each list
and its cosine distance, so a trace shows why it was chosen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from langchain_core.embeddings import Embeddings

from rag_agent.keyword import BM25Index
from rag_agent.store import open_store

MODES = ("hybrid", "vector", "keyword")
DEFAULT_QUERY_INSTRUCTION = "Given a question, retrieve passages that answer it"   # frozen, architecture D19
RRF_K = 60            # the standard RRF constant; damps the advantage of rank 1 over rank 2
CANDIDATE_POOL = 30   # candidates taken from each list before fusion


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    source_id: str
    title: str
    page: int
    text: str
    distance: float | None = None      # cosine distance to the query; None in keyword mode
    vector_rank: int | None = None     # 1-based rank in the vector list, None if not in it
    keyword_rank: int | None = None    # 1-based rank in the keyword list, None if not in it


class Retriever(Protocol):
    def search(self, query: str, k: int, sources: list[str] | None = None,
               subject: str | None = None) -> list[Passage]: ...


def make_where(sources: list[str] | None, subject: str | None) -> dict | None:
    """The Chroma metadata filter for the given source ids and subject."""
    clauses = []
    if sources:
        clauses.append({"source_id": {"$in": list(sources)}})
    if subject:
        clauses.append({"subject": subject})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Merge ranked id lists; returns (id, score) best first.

    Ties are broken by the best rank the id reached in any list, then by the order of the
    lists, so the result is deterministic.
    """
    scores: dict[str, float] = {}
    best: dict[str, tuple[int, int]] = {}
    for list_index, ranking in enumerate(rankings):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            best[item] = min(best.get(item, (rank, list_index)), (rank, list_index))
    return sorted(scores.items(), key=lambda pair: (-pair[1], best[pair[0]]))


def with_instruction(query: str, instruction: str | None) -> str:
    """The text embedded for a query: Qwen3-Embedding's instruction format when one is set."""
    return f"Instruct: {instruction}\nQuery: {query}" if instruction else query


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 1.0 - dot / norm if norm else 1.0


class HybridRetriever:
    """Searches the persistent Chroma collection in `hybrid`, `vector` or `keyword` mode."""

    def __init__(self, db_path: Path, embeddings: Embeddings | None = None, mode: str = "hybrid",
                 query_instruction: str | None = None, pool: int = CANDIDATE_POOL) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        if not Path(db_path).exists():
            raise FileNotFoundError(
                f"No vector store at {db_path}; run: uv run python -m rag_agent.ingest --rebuild")
        self.mode = mode
        self.query_instruction = query_instruction or None
        self.pool = pool
        self._store = open_store(Path(db_path), embeddings)
        stored = self._store.get(include=["documents", "metadatas"])
        self._records = {
            chunk_id: (text, meta)
            for chunk_id, text, meta in zip(stored["ids"], stored["documents"], stored["metadatas"])
        }
        self._keyword = BM25Index((cid, text, meta) for cid, (text, meta) in self._records.items())

    def __len__(self) -> int:
        return len(self._records)

    def search(self, query: str, k: int, sources: list[str] | None = None,
               subject: str | None = None) -> list[Passage]:
        width = self.pool if self.mode == "hybrid" else k
        vector_ids: list[str] = []
        distances: dict[str, float] = {}
        query_vector = None
        if self.mode in ("hybrid", "vector"):
            query_vector = self._store.embeddings.embed_query(with_instruction(query, self.query_instruction))
            for document, distance in self._store.similarity_search_by_vector_with_relevance_scores(
                    query_vector, k=width, filter=make_where(sources, subject)):
                chunk_id = document.metadata["chunk_id"]
                vector_ids.append(chunk_id)
                distances[chunk_id] = float(distance)
        keyword_ids: list[str] = []
        if self.mode in ("hybrid", "keyword"):
            keyword_ids = [hit.chunk_id for hit in self._keyword.search(query, width, sources, subject)]

        if self.mode == "vector":
            chosen = vector_ids[:k]
        elif self.mode == "keyword":
            chosen = keyword_ids[:k]
        else:
            chosen = [chunk_id for chunk_id, _ in reciprocal_rank_fusion([vector_ids, keyword_ids])[:k]]

        if query_vector is not None:
            missing = [chunk_id for chunk_id in chosen if chunk_id not in distances]
            if missing:   # keyword-only hits: compute their distance so every passage shows one
                found = self._store.get(ids=missing, include=["embeddings"])
                for chunk_id, vector in zip(found["ids"], found["embeddings"]):
                    distances[chunk_id] = _cosine_distance(list(query_vector), list(vector))

        vector_rank = {chunk_id: rank for rank, chunk_id in enumerate(vector_ids, start=1)}
        keyword_rank = {chunk_id: rank for rank, chunk_id in enumerate(keyword_ids, start=1)}
        passages = []
        for chunk_id in chosen:
            text, meta = self._records[chunk_id]
            distance = distances.get(chunk_id)
            passages.append(Passage(
                chunk_id=chunk_id, source_id=meta["source_id"], title=meta["title"], page=int(meta["page"]),
                text=text, distance=None if distance is None else round(distance, 4),
                vector_rank=vector_rank.get(chunk_id), keyword_rank=keyword_rank.get(chunk_id),
            ))
        return passages
