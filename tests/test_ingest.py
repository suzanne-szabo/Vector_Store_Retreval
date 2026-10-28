"""Black-box tests for extraction, chunking and ingestion (no live services).

Covers the assignment's ingestion requirement: each chunk's metadata holds the source,
the one-based physical PDF page, and a stable chunk identifier.
"""

import json

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag_agent.chunk import METADATA_FIELDS, chunk_document, make_chunk_id, make_splitter
from rag_agent.clean import clean_document
from rag_agent.corpus import load_registry
from rag_agent.extract import extract_pages
from rag_agent.ingest import IngestError, ingest
from rag_agent.store import open_store


def fake_embeddings():
    return DeterministicFakeEmbedding(size=16)


def run(corpus, **kwargs):
    return ingest(registry_path=corpus["registry"], docs_dir=corpus["docs"], db_path=corpus["db"],
                  manifest_path=corpus["manifest"], embeddings=fake_embeddings(), **kwargs)


def stored(corpus):
    return open_store(corpus["db"], fake_embeddings()).get(include=["documents", "metadatas"])


# ---------------------------------------------------------------- extraction

def test_extraction_returns_one_entry_per_physical_page(corpus):
    pages = extract_pages(corpus["manual"])
    assert len(pages) == 5
    assert "return address" in pages[0]
    assert "a0 through a7" in pages[3]


def test_extraction_keeps_python_indentation(corpus):
    page = clean_document(extract_pages(corpus["manual"]))[1]
    assert "for i in range(10):\n    num = int(input('Enter a number: '))" in page
    assert "\n        count = count + 1" in page


def test_two_column_pages_are_read_one_column_at_a_time(corpus):
    for text in extract_pages(corpus["columns"], columns=2):
        assert text.index("LEFT three gamma") < text.index("RIGHT one delta")
        assert "\n" + "RIGHT one delta" in "\n" + text.replace("\n\n", "\n")   # column margin removed


def test_a_full_width_title_is_kept_whole_above_the_columns(corpus):
    first, second = extract_pages(corpus["columns"], columns=2)
    assert "A Centered Title Across Both Columns" in first
    assert first.index("A Centered Title") < first.index("LEFT one alpha")
    assert "Centered" not in second


# ---------------------------------------------------------------- chunk ids

def test_chunk_id_is_deterministic_and_text_sensitive():
    first = make_chunk_id("a" * 64, 3, 0, "some text")
    assert first == make_chunk_id("a" * 64, 3, 0, "some text")
    assert len(first) == 24
    assert first != make_chunk_id("a" * 64, 3, 0, "some text!")
    assert first != make_chunk_id("a" * 64, 4, 0, "some text")


def test_chunks_carry_metadata_and_one_based_physical_pages(corpus):
    source = load_registry(corpus["registry"])[0]
    pages = clean_document(extract_pages(corpus["manual"]))
    chunks = chunk_document(source, pages, make_splitter(200, 30))
    assert {chunk.metadata["page"] for chunk in chunks} == {1, 2, 4, 5}   # page 3 is blank and skipped
    for chunk in chunks:
        assert set(chunk.metadata) == set(METADATA_FIELDS)
        assert chunk.metadata["source_id"] == "fixture-manual"
        assert chunk.metadata["source_file"] == "manual.pdf"
        assert chunk.metadata["chunk_id"] == chunk.chunk_id
        assert "Fixture Manual" not in chunk.text           # running header removed
    assert any("x1 holds the return address" in c.text and c.metadata["page"] == 1 for c in chunks)


def test_chunk_ids_are_stable_across_builds(corpus):
    source = load_registry(corpus["registry"])[0]
    build = lambda: [c.chunk_id for c in chunk_document(
        source, clean_document(extract_pages(corpus["manual"])), make_splitter(200, 30))]
    assert build() == build()


def test_chunks_never_exceed_the_chunk_size(corpus):
    source = load_registry(corpus["registry"])[0]
    chunks = chunk_document(source, clean_document(extract_pages(corpus["manual"])), make_splitter(120, 20))
    assert chunks and all(len(chunk.text) <= 120 for chunk in chunks)


# ---------------------------------------------------------------- ingestion into Chroma

def test_ingest_stores_every_chunk_with_metadata(corpus):
    manifest = run(corpus, rebuild=True, chunk_size=200, chunk_overlap=30)
    records = stored(corpus)
    assert manifest["record_count"] == len(records["ids"]) > 0
    for chunk_id, meta in zip(records["ids"], records["metadatas"]):
        assert meta["chunk_id"] == chunk_id
        assert meta["page"] >= 1
        assert meta["source_id"] in {"fixture-manual", "fixture-columns"}
    assert manifest["embedding_dimension"] == 16
    assert manifest["chunking"] == {"chunk_size": 200, "chunk_overlap": 30}
    assert set(manifest["sources"]) == {"fixture-manual", "fixture-columns"}
    assert json.loads(corpus["manifest"].read_text())["index_version"] == manifest["index_version"]


def test_rebuilding_gives_the_same_ids(corpus):
    first = run(corpus, rebuild=True, chunk_size=200, chunk_overlap=30)["index_version"]
    second = run(corpus, rebuild=True, chunk_size=200, chunk_overlap=30)["index_version"]
    assert first == second


def test_ingesting_one_source_at_a_time_matches_a_full_build(corpus):
    full = run(corpus, rebuild=True, chunk_size=200, chunk_overlap=30)
    run(corpus, rebuild=True, only=["fixture-columns"], chunk_size=200, chunk_overlap=30)
    staged = run(corpus, only=["fixture-manual"], chunk_size=200, chunk_overlap=30)
    assert staged["index_version"] == full["index_version"]
    assert staged["record_count"] == full["record_count"]


def test_reingesting_a_source_replaces_instead_of_duplicating(corpus):
    first = run(corpus, rebuild=True, chunk_size=200, chunk_overlap=30)["record_count"]
    again = run(corpus, only=["fixture-manual"], chunk_size=200, chunk_overlap=30)["record_count"]
    assert again == first


def test_mixing_chunk_settings_without_rebuild_is_refused(corpus):
    run(corpus, rebuild=True, chunk_size=200, chunk_overlap=30)
    with pytest.raises(IngestError, match="--rebuild"):
        run(corpus, only=["fixture-manual"], chunk_size=300, chunk_overlap=30)


def test_dry_run_embeds_nothing(corpus, tmp_path):
    dump = tmp_path / "chunks.jsonl"
    summary = run(corpus, dry_run=True, chunk_size=200, chunk_overlap=30, dump_path=dump)
    assert summary["dry_run"] is True
    assert not corpus["db"].exists()
    rows = [json.loads(line) for line in dump.read_text().splitlines()]
    assert len(rows) == sum(summary["sources"].values())
