"""Black-box tests for rag_agent.retrieval (vector, keyword and hybrid search), offline."""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from rag_agent.ingest import ingest
from rag_agent.retrieval import (
    RRF_K, HybridRetriever, make_where, reciprocal_rank_fusion, with_instruction,
)


class RecordingEmbedding(Embeddings):
    """A fake embedding that remembers every text it was asked to embed."""

    def __init__(self):
        self._inner = DeterministicFakeEmbedding(size=16)
        self.queries, self.documents = [], []

    def embed_query(self, text):
        self.queries.append(text)
        return self._inner.embed_query(text)

    def embed_documents(self, texts):
        self.documents.extend(texts)
        return self._inner.embed_documents(texts)


def recording():
    return RecordingEmbedding()


@pytest.fixture
def store(corpus):
    """The fixture corpus ingested into Chroma with fake embeddings; returns the db path."""
    ingest(registry_path=corpus["registry"], docs_dir=corpus["docs"], db_path=corpus["db"],
           manifest_path=corpus["manifest"], embeddings=DeterministicFakeEmbedding(size=16),
           rebuild=True, chunk_size=200, chunk_overlap=30)
    return corpus["db"]


# ---------------------------------------------------------------- pure functions

def test_make_where_for_every_combination():
    assert make_where(None, None) is None
    assert make_where(["a", "b"], None) == {"source_id": {"$in": ["a", "b"]}}
    assert make_where(None, "riscv") == {"subject": "riscv"}
    assert make_where(["a"], "riscv") == {"$and": [{"source_id": {"$in": ["a"]}}, {"subject": "riscv"}]}


def test_reciprocal_rank_fusion_adds_reciprocal_ranks():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]])
    assert [item for item, _ in fused] == ["a", "c", "b"]
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / (RRF_K + 1) + 1 / (RRF_K + 2))
    assert scores["c"] == pytest.approx(1 / (RRF_K + 3) + 1 / (RRF_K + 1))
    assert scores["b"] == pytest.approx(1 / (RRF_K + 2))


def test_fusion_ties_are_broken_by_best_rank_then_list_order():
    assert [item for item, _ in reciprocal_rank_fusion([["x"], ["y"]])] == ["x", "y"]


def test_query_instruction_format():
    assert with_instruction("what is ra", None) == "what is ra"
    assert with_instruction("what is ra", "Find the answer") == "Instruct: Find the answer\nQuery: what is ra"


# ---------------------------------------------------------------- the retriever

def test_missing_store_and_bad_mode_are_errors(tmp_path, store):
    with pytest.raises(FileNotFoundError, match="ingest --rebuild"):
        HybridRetriever(tmp_path / "nowhere", DeterministicFakeEmbedding(size=16))
    with pytest.raises(ValueError, match="mode"):
        HybridRetriever(store, DeterministicFakeEmbedding(size=16), mode="fuzzy")


def test_keyword_mode_finds_the_exact_term(store):
    retriever = HybridRetriever(store, DeterministicFakeEmbedding(size=16), mode="keyword")
    passages = retriever.search("return address", k=3)
    assert passages[0].page == 1 and "return address" in passages[0].text
    assert passages[0].keyword_rank == 1
    assert passages[0].vector_rank is None and passages[0].distance is None


def test_every_passage_carries_source_page_and_chunk_labels(store):
    retriever = HybridRetriever(store, DeterministicFakeEmbedding(size=16))
    passages = retriever.search("registers a0 through a7", k=4)
    assert 0 < len(passages) <= 4
    for passage in passages:
        assert len(passage.chunk_id) == 24
        assert passage.source_id in {"fixture-manual", "fixture-columns"}
        assert passage.title.startswith("Title of")
        assert passage.page >= 1 and passage.text
        assert passage.distance is not None                     # computed even for keyword-only hits
        assert passage.vector_rank is not None or passage.keyword_rank is not None


def test_hybrid_puts_a_strong_keyword_match_in_the_results(store):
    retriever = HybridRetriever(store, DeterministicFakeEmbedding(size=16))
    passages = retriever.search("preserved across calls", k=3)
    assert any("preserved across calls" in p.text for p in passages)


def test_vector_mode_reports_distances_in_rank_order(store):
    retriever = HybridRetriever(store, DeterministicFakeEmbedding(size=16), mode="vector")
    passages = retriever.search("registers", k=3)
    assert [p.vector_rank for p in passages] == [1, 2, 3]
    assert [p.distance for p in passages] == sorted(p.distance for p in passages)
    assert all(p.keyword_rank is None for p in passages)


@pytest.mark.parametrize("mode", ["hybrid", "vector", "keyword"])
def test_filters_and_k_are_honored_in_every_mode(store, mode):
    retriever = HybridRetriever(store, DeterministicFakeEmbedding(size=16), mode=mode)
    assert all(p.source_id == "fixture-columns"
               for p in retriever.search("LEFT RIGHT alpha", k=5, sources=["fixture-columns"]))
    assert all(p.source_id == "fixture-manual" for p in retriever.search("registers", k=5, subject="riscv"))
    assert len(retriever.search("registers", k=2)) <= 2
    assert retriever.search("registers", k=5, sources=["fixture-manual"], subject="pedagogy") == []


def test_query_instruction_applies_to_queries_only(store):
    embedding = recording()
    retriever = HybridRetriever(store, embedding, query_instruction="Find passages that answer it")
    retriever.search("what holds the return address", k=2)
    assert embedding.queries == ["Instruct: Find passages that answer it\nQuery: what holds the return address"]
    assert embedding.documents == []      # opening and searching never re-embeds documents
