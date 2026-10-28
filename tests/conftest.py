from __future__ import annotations

from pathlib import Path

import pytest

from pdf_fixture import make_pdf, manual_pages, two_column_page
from rag_agent.corpus import sha256_file


def write_registry(path: Path, entries: list[dict]) -> Path:
    blocks = []
    for entry in entries:
        lines = ["[[source]]"]
        for key, value in entry.items():
            lines.append(f"{key} = {value}" if isinstance(value, int) else f'{key} = "{value}"')
        blocks.append("\n".join(lines))
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return path


def entry(source_id: str, file: str, sha: str, pages: int, columns: int = 1, subject: str = "riscv") -> dict:
    return {
        "id": source_id, "file": file, "title": f"Title of {source_id}", "subject": subject,
        "version": "1.0", "role": "Test source.", "license": "test", "url": "https://example.org",
        "pages": pages, "sha256": sha, "columns": columns,
    }


@pytest.fixture
def corpus(tmp_path: Path) -> dict:
    """A documents folder with two fixture PDFs and a matching registry."""
    docs = tmp_path / "docs"
    docs.mkdir()
    manual = make_pdf(docs / "manual.pdf", manual_pages())
    columns = make_pdf(docs / "columns.pdf", two_column_page())
    registry = write_registry(tmp_path / "sources.toml", [
        entry("fixture-manual", "manual.pdf", sha256_file(manual), 5),
        entry("fixture-columns", "columns.pdf", sha256_file(columns), 2, columns=2, subject="pedagogy"),
    ])
    return {"docs": docs, "registry": registry, "manual": manual, "columns": columns,
            "db": tmp_path / "chroma_db", "manifest": tmp_path / "manifest.json"}
