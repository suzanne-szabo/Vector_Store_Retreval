"""Split cleaned pages into chunks with stable ids and full source metadata.

Each page is split on its own, so every chunk belongs to exactly one physical page.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_agent.corpus import Source

METADATA_FIELDS = (
    "chunk_id", "source_id", "source_file", "title", "subject",
    "version", "page", "start_index", "source_sha256",
)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict


def make_chunk_id(source_sha256: str, page: int, start_index: int, text: str) -> str:
    """24 hex characters of SHA-256 over file hash, page, offset and text.

    The same file, cleaning and chunk settings always give the same id; any change to
    the words gives a new id, so an old citation never points at different text.
    """
    identity = "|".join([source_sha256, str(page), str(start_index), text])
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Need chunk_size > 0 and 0 <= chunk_overlap < chunk_size")
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )


def chunk_document(source: Source, pages: list[str], splitter: RecursiveCharacterTextSplitter) -> list[Chunk]:
    """Chunk a document's cleaned pages; `pages[0]` is physical page 1, "" pages are skipped."""
    chunks: list[Chunk] = []
    for page_number, text in enumerate(pages, start=1):
        if not text:
            continue
        for piece in splitter.create_documents([text]):
            start = piece.metadata["start_index"]
            chunk_id = make_chunk_id(source.sha256, page_number, start, piece.page_content)
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=piece.page_content,
                metadata={
                    "chunk_id": chunk_id,
                    "source_id": source.id,
                    "source_file": source.file,
                    "title": source.title,
                    "subject": source.subject,
                    "version": source.version,
                    "page": page_number,
                    "start_index": start,
                    "source_sha256": source.sha256,
                },
            ))
    return chunks
