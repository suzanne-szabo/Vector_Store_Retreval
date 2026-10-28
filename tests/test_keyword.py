"""Black-box tests for rag_agent.keyword (BM25 keyword search)."""

from rag_agent.keyword import BM25Index, tokenize


def records(*items):
    """(chunk_id, text, metadata) records; metadata defaults to one source and subject."""
    out = []
    for item in items:
        chunk_id, text, *rest = item
        meta = rest[0] if rest else {"source_id": "src-a", "subject": "riscv"}
        out.append((chunk_id, text, meta))
    return out


def test_tokenize_keeps_technical_tokens_and_their_parts():
    tokens = tokenize("Registers x10-x17 are callee-saved; JALR and __init__ in section 2.5.")
    for expected in ["registers", "x10-x17", "x10", "x17", "callee-saved", "callee", "saved",
                     "jalr", "__init__", "section", "2.5", "2", "5"]:
        assert expected in tokens
    assert "are" not in tokens                     # stopword
    assert "and" in tokens and "in" in tokens       # Python keywords are kept


def test_only_chunks_containing_a_query_term_are_returned():
    index = BM25Index(records(("c1", "JALR writes pc+4 to rd"), ("c2", "ADDI adds an immediate")))
    assert [hit.chunk_id for hit in index.search("jalr", k=5)] == ["c1"]
    assert index.search("nonexistentword", k=5) == []


def test_rare_terms_outweigh_common_ones():
    index = BM25Index(records(
        ("common", "python tutor python tutor online python tutor"),
        ("rare", "limitations and ongoing work of the python tutor"),
        ("other1", "python tutor frontend"), ("other2", "python tutor backend"),
    ))
    assert index.search("limitations python tutor", k=1)[0].chunk_id == "rare"


def test_shorter_chunks_win_when_term_counts_are_equal():
    index = BM25Index(records(
        ("long", "callee-saved " + "filler " * 40),
        ("short", "callee-saved registers"),
    ))
    assert [hit.chunk_id for hit in index.search("callee-saved", k=2)] == ["short", "long"]


def test_source_and_subject_filters():
    index = BM25Index(records(
        ("a", "register convention", {"source_id": "riscv-psabi", "subject": "riscv"}),
        ("b", "register convention", {"source_id": "riscv-isa-2025", "subject": "riscv"}),
        ("c", "register convention", {"source_id": "heinold-python", "subject": "python"}),
    ))
    assert [h.chunk_id for h in index.search("register", 5, sources=["riscv-psabi"])] == ["a"]
    assert [h.chunk_id for h in index.search("register", 5, subject="python")] == ["c"]
    assert index.search("register", 5, sources=["riscv-psabi"], subject="python") == []


def test_k_limits_results_and_ties_keep_index_order():
    index = BM25Index(records(("a", "word"), ("b", "word"), ("c", "word")))
    assert [hit.chunk_id for hit in index.search("word", k=2)] == ["a", "b"]


def test_empty_index_returns_nothing():
    assert BM25Index([]).search("anything", k=3) == []
