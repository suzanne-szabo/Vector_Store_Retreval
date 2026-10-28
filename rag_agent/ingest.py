"""Build the vector store: verify, extract, clean, chunk, embed, store, record.

Examples (run from the project folder):

    uv run python -m rag_agent.ingest --dry-run                    # no services; check chunks
    uv run python -m rag_agent.ingest --only guo-2013              # embed one source
    uv run python -m rag_agent.ingest --only riscv-psabi winans-rvalp
    uv run python -m rag_agent.ingest --rebuild                    # wipe and embed everything

Without --rebuild, each selected source's old chunks are replaced and other sources
already in the store are kept, so the corpus can be embedded a few sources at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.embeddings import Embeddings

from rag_agent import clean as clean_mod
from rag_agent.chunk import Chunk, chunk_document, make_splitter
from rag_agent.config import (
    COLLECTION_NAME, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_DB_PATH,
    DEFAULT_MANIFEST_PATH, DEFAULT_REGISTRY_PATH, STORE_BATCH_SIZE, documents_dir, embedding_model_name,
    setting,
)
from rag_agent.corpus import CorpusError, Source, load_registry, resolve, select
from rag_agent.extract import EXTRACTOR, extract_pages
from rag_agent.store import DISTANCE, open_store


class IngestError(Exception):
    """The requested build would mix incompatible settings or cannot proceed."""


@dataclass
class Prepared:
    """One source after extraction, cleaning and chunking (before embedding)."""

    source: Source
    pages_total: int
    pages_kept: int
    chunks: list[Chunk]
    removed_edges: Counter
    seconds: float


def build_settings(chunk_size: int, chunk_overlap: int) -> dict:
    """Everything that changes chunk text or vectors; mixing two settings in one store is refused."""
    return {
        "embedding_model": embedding_model_name(),
        "chunking": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        "extractor": EXTRACTOR,
        "cleaning": {
            "version": clean_mod.CLEANING_VERSION,
            "edge_depth": clean_mod.EDGE_DEPTH,
            "edge_repeat_pages": clean_mod.EDGE_REPEAT_PAGES,
            "min_page_chars": clean_mod.MIN_PAGE_CHARS,
        },
    }


def prepare(source: Source, path: Path, chunk_size: int, chunk_overlap: int) -> Prepared:
    started = time.monotonic()
    raw = extract_pages(path, columns=source.columns)
    removed: Counter = Counter()
    pages = clean_mod.clean_document(raw, removed)
    chunks = chunk_document(source, pages, make_splitter(chunk_size, chunk_overlap))
    return Prepared(source, len(raw), sum(1 for page in pages if page), chunks, removed,
                    time.monotonic() - started)


def _print_prepared(item: Prepared) -> None:
    sizes = [len(chunk.text) for chunk in item.chunks] or [0]
    print(f"[{item.source.id}] pages {item.pages_kept}/{item.pages_total} kept · "
          f"{len(item.chunks)} chunks · mean {sum(sizes) // len(sizes)} chars, max {max(sizes)} · "
          f"{item.seconds:.1f} s", flush=True)
    for signature, count in item.removed_edges.most_common(5):
        print(f"    removed header/footer ×{count}: {signature[:90]!r}")


def _read_manifest(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ingest(
    *,
    only: list[str] | None = None,
    rebuild: bool = False,
    dry_run: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    docs_dir: Path | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    embeddings: Embeddings | None = None,
    dump_path: Path | None = None,
) -> dict:
    """Run the pipeline; returns the manifest (or, for a dry run, a summary)."""
    settings = build_settings(chunk_size, chunk_overlap)
    sources = select(load_registry(registry_path), only)
    resolved = resolve(sources, docs_dir or documents_dir())

    previous = None if rebuild else _read_manifest(manifest_path)
    if not dry_run and previous is not None:
        old = {key: previous.get(key) for key in settings}
        if old != settings:
            raise IngestError(
                "The existing store was built with different settings; run with --rebuild.\n"
                f"  store:     {json.dumps(old)}\n  requested: {json.dumps(settings)}"
            )

    prepared: list[Prepared] = []
    for source, path in resolved:
        item = prepare(source, path, chunk_size, chunk_overlap)
        _print_prepared(item)
        prepared.append(item)

    if dump_path is not None:
        with open(dump_path, "w", encoding="utf-8") as handle:
            for item in prepared:
                for chunk in item.chunks:
                    handle.write(json.dumps({"metadata": chunk.metadata, "text": chunk.text}) + "\n")
        print(f"Wrote chunk dump to {dump_path}")

    if dry_run:
        total = sum(len(item.chunks) for item in prepared)
        print(f"Dry run: {total} chunks from {len(prepared)} source(s); nothing embedded.")
        return {"dry_run": True, "settings": settings,
                "sources": {item.source.id: len(item.chunks) for item in prepared}}

    store = open_store(db_path, embeddings)
    if rebuild:
        # Reset through Chroma rather than deleting the folder: Chroma caches open
        # databases per process, and a deleted folder leaves a stale read-only handle.
        store.reset_collection()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = previous or {"created_at": now, "sources": {}}

    for item in prepared:
        old_ids = store.get(where={"source_id": item.source.id}, include=[])["ids"]
        if old_ids:
            store.delete(ids=old_ids)
        started = time.monotonic()
        total = len(item.chunks)
        for offset in range(0, total, STORE_BATCH_SIZE):
            batch = item.chunks[offset:offset + STORE_BATCH_SIZE]
            store.add_texts(
                texts=[chunk.text for chunk in batch],
                metadatas=[chunk.metadata for chunk in batch],
                ids=[chunk.chunk_id for chunk in batch],
            )
            done = offset + len(batch)
            elapsed = time.monotonic() - started
            print(f"[{item.source.id}] embedded {done}/{total} chunks ({elapsed:.1f} s)", flush=True)
        manifest["sources"][item.source.id] = {
            "file": item.source.file,
            "sha256": item.source.sha256,
            "pages_total": item.pages_total,
            "pages_kept": item.pages_kept,
            "chunks": total,
            "embedded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "embed_seconds": round(time.monotonic() - started, 1),
        }

    all_ids = sorted(store.get(include=[])["ids"])
    sample = store.get(ids=all_ids[:1], include=["embeddings"])["embeddings"] if all_ids else []
    manifest.update(settings)
    manifest.update({
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "collection": COLLECTION_NAME,
        "distance": DISTANCE,
        "embedding_dimension": len(sample[0]) if len(sample) else None,
        "record_count": len(all_ids),
        "index_version": hashlib.sha256("\n".join(all_ids).encode("utf-8")).hexdigest()[:16],
    })
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Store now holds {len(all_ids)} chunks from {len(manifest['sources'])} source(s); "
          f"index_version {manifest['index_version']}. Manifest: {manifest_path}")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m rag_agent.ingest",
                                     description="Build the Chroma vector store from the corpus.")
    parser.add_argument("--only", nargs="+", metavar="SOURCE_ID", help="ingest only these source ids")
    parser.add_argument("--rebuild", action="store_true", help="delete the store and start over")
    parser.add_argument("--dry-run", action="store_true", help="extract, clean and chunk only; no embedding")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"characters per chunk (default {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                        help=f"characters shared by neighboring chunks (default {DEFAULT_CHUNK_OVERLAP})")
    parser.add_argument("--dump", type=Path, metavar="FILE", help="write every chunk (metadata + text) to this JSONL file")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, metavar="DIR",
                        help="vector store folder (default chroma_db/)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH, metavar="FILE",
                        help="build record to write (default manifest.json)")
    return parser.parse_args(argv)


def _service_error_message(error: Exception) -> str | None:
    """A short explanation for embedding-service failures, or None if it is not one."""
    try:
        import openai
    except ImportError:  # pragma: no cover
        return None
    url = setting("EMBEDDING_BASE_URL", "http://golem.cs.utahtech.edu:4000/v1")
    if isinstance(error, (openai.APITimeoutError, openai.APIConnectionError)):
        return (f"Could not reach the embedding service at {url} ({type(error).__name__}).\n"
                "  Check that you are on the course network or that Tailscale is connected. Over Tailscale\n"
                "  the host name may not resolve; try the gateway address in .env:\n"
                "  EMBEDDING_BASE_URL=http://100.64.0.1:4000/v1\n"
                "  Re-running the same command is safe: this source's chunks are replaced.")
    if isinstance(error, openai.AuthenticationError):
        return "The embedding service rejected the API key; check EMBEDDING_API_KEY in .env."
    if isinstance(error, openai.APIStatusError):
        return f"The embedding service returned HTTP {error.status_code}: {error.message}"
    return None


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        ingest(only=args.only, rebuild=args.rebuild, dry_run=args.dry_run,
               chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
               db_path=args.db, manifest_path=args.manifest, dump_path=args.dump)
    except (CorpusError, IngestError, ValueError, RuntimeError) as error:
        sys.exit(f"Error: {error}")
    except Exception as error:
        message = _service_error_message(error)
        if message is None:
            raise
        sys.exit(f"Error: {message}")


if __name__ == "__main__":
    main()
