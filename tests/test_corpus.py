"""Black-box tests for rag_agent.corpus."""

import pytest

from conftest import entry, write_registry
from rag_agent.config import DEFAULT_REGISTRY_PATH
from rag_agent.corpus import CorpusError, load_registry, resolve, select


def test_project_registry_lists_nine_sources():
    sources = load_registry(DEFAULT_REGISTRY_PATH)
    assert len(sources) == 9
    assert len({source.id for source in sources}) == 9
    assert {source.subject for source in sources} == {"python", "riscv", "pedagogy"}
    assert sum(source.pages for source in sources) == 2017


def test_resolve_accepts_matching_files(corpus):
    resolved = resolve(load_registry(corpus["registry"]), corpus["docs"])
    assert [source.id for source, _ in resolved] == ["fixture-manual", "fixture-columns"]


def test_resolve_rejects_a_changed_file(corpus):
    corpus["manual"].write_bytes(corpus["manual"].read_bytes() + b"\n% changed\n")
    with pytest.raises(CorpusError, match="SHA-256 mismatch"):
        resolve(load_registry(corpus["registry"]), corpus["docs"])


def test_resolve_rejects_a_missing_file(corpus):
    corpus["columns"].unlink()
    with pytest.raises(CorpusError, match="missing file"):
        resolve(load_registry(corpus["registry"]), corpus["docs"])


def test_select_unknown_id_is_an_error(corpus):
    with pytest.raises(CorpusError, match="Unknown source id"):
        select(load_registry(corpus["registry"]), ["no-such-source"])


def test_registry_rejects_duplicate_ids(tmp_path):
    sha = "0" * 64
    path = write_registry(tmp_path / "s.toml", [entry("dup", "a.pdf", sha, 1), entry("dup", "b.pdf", sha, 1)])
    with pytest.raises(CorpusError, match="Duplicate"):
        load_registry(path)
