"""The embedding client and the persistent Chroma collection."""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from rag_agent.config import COLLECTION_NAME, EMBED_BATCH_SIZE, embedding_model_name, require_api_key, setting

DISTANCE = "cosine"


def build_embeddings() -> Embeddings:
    """Qwen3-Embedding-8B through the course's OpenAI-compatible service."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=embedding_model_name(),
        base_url=setting("EMBEDDING_BASE_URL", "http://golem.cs.utahtech.edu:4000/v1"),
        api_key=require_api_key("EMBEDDING_API_KEY"),
        check_embedding_ctx_length=False,   # the service is not an OpenAI model; send text as-is
        chunk_size=EMBED_BATCH_SIZE,        # texts per request
        max_retries=2,
        request_timeout=180,
    )


def open_store(db_path: Path, embeddings: Embeddings | None = None) -> Chroma:
    """Open (or create) the persistent collection; cosine distance, so 0 = same direction."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(db_path),
        embedding_function=embeddings if embeddings is not None else build_embeddings(),
        collection_metadata={"hnsw:space": DISTANCE},
    )
