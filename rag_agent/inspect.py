"""Look inside the vector store.

    uv run python -m rag_agent.inspect --manifest
    uv run python -m rag_agent.inspect --records --source riscv-psabi --page 7
    uv run python -m rag_agent.inspect --search "callee-saved registers" --source riscv-psabi
    uv run python -m rag_agent.inspect --search "callee-saved registers" --mode vector
"""

from __future__ import annotations

import argparse
import json
import sys

from rag_agent.config import DEFAULT_DB_PATH, DEFAULT_MANIFEST_PATH
from rag_agent.retrieval import DEFAULT_QUERY_INSTRUCTION, MODES, HybridRetriever, with_instruction
from rag_agent.store import open_store


class _NoEmbeddings:
    """Reading records needs no embedding service; only --search does."""

    def embed_documents(self, texts):  # pragma: no cover - never called for reads
        raise RuntimeError("embedding not available")

    def embed_query(self, text):  # pragma: no cover
        raise RuntimeError("embedding not available")


def _where(source: str | None, page: int | None) -> dict | None:
    clauses = []
    if source:
        clauses.append({"source_id": source})
    if page is not None:
        clauses.append({"page": page})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m rag_agent.inspect", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", action="store_true", help="print the build manifest")
    mode.add_argument("--records", action="store_true", help="print stored chunks")
    mode.add_argument("--search", metavar="QUERY",
                      help="search the corpus (hybrid and vector modes need the embedding service)")
    parser.add_argument("--source", metavar="SOURCE_ID", help="limit to one source id, such as riscv-psabi")
    parser.add_argument("--page", type=int, help="limit to one page (with --records)")
    parser.add_argument("--limit", type=int, default=5, help="how many results or records to show (default 5)")
    parser.add_argument("--instruction", metavar="TEXT", default=DEFAULT_QUERY_INSTRUCTION,
                        help="Qwen3 query instruction (default: the frozen one the agent uses; "
                             "pass \"\" to search without it)")
    parser.add_argument("--mode", choices=MODES, default="hybrid", help="search mode (with --search)")
    args = parser.parse_args(argv)

    if args.manifest:
        if not DEFAULT_MANIFEST_PATH.is_file():
            sys.exit("No manifest.json yet; run rag_agent.ingest first.")
        print(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
        return
    if not DEFAULT_DB_PATH.exists():
        sys.exit("No chroma_db/ yet; run rag_agent.ingest first.")

    if args.records:
        store = open_store(DEFAULT_DB_PATH, _NoEmbeddings())
        result = store.get(where=_where(args.source, args.page), limit=args.limit,
                           include=["documents", "metadatas"])
        for text, meta in zip(result["documents"], result["metadatas"]):
            print(json.dumps(meta))
            print(text)
            print("-" * 72)
        print(f"{len(result['documents'])} record(s) shown")
        return

    retriever = HybridRetriever(DEFAULT_DB_PATH, mode=args.mode, query_instruction=args.instruction)
    if args.mode != "keyword":
        print(f"Embedded query: {with_instruction(args.search, args.instruction)!r}")
    print(f"Mode: {args.mode} · {len(retriever)} chunks in the store\n")
    sources = [args.source] if args.source else None
    for rank, passage in enumerate(retriever.search(args.search, k=args.limit, sources=sources), start=1):
        distance = "—" if passage.distance is None else f"{passage.distance:.4f}"
        vector = passage.vector_rank or "—"
        keyword = passage.keyword_rank or "—"
        print(f"#{rank} distance {distance} · vector rank {vector} · keyword rank {keyword} · "
              f"{passage.source_id} p.{passage.page} · {passage.chunk_id}")
        print(passage.text[:600])
        print("-" * 72)


if __name__ == "__main__":
    main()
