"""Shared configuration: paths, defaults, and settings read from `.env`."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_DIR / "chroma_db"
DEFAULT_MANIFEST_PATH = PROJECT_DIR / "manifest.json"
DEFAULT_REGISTRY_PATH = PROJECT_DIR / "corpus" / "sources.toml"
COLLECTION_NAME = "python_riscv_corpus"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
EMBED_BATCH_SIZE = 32          # texts per embedding request
STORE_BATCH_SIZE = 64          # chunks per Chroma add (and per progress line)

PLACEHOLDER_KEY = "student-should-use-their-own"

# Existing environment variables win over .env, so a shell export can override it.
load_dotenv(PROJECT_DIR / ".env", override=False)


def setting(name: str, default: str) -> str:
    """Read a setting, keeping the classroom defaults visible in code."""
    return os.getenv(name, default)


def require_api_key(name: str) -> str:
    """Return an API key, rejecting a missing value or the documented placeholder."""
    value = os.getenv(name, "")
    if not value or value == PLACEHOLDER_KEY:
        raise RuntimeError(
            f"Set {name} in .env to the API key assigned to you (see .env.example)."
        )
    return value


def documents_dir() -> Path:
    """The folder holding the corpus PDFs; relative paths are relative to the project."""
    raw = Path(setting("DOCUMENTS_DIR", "../U0-HW-09-materials/documents/personal_documents"))
    return (raw if raw.is_absolute() else PROJECT_DIR / raw).resolve()


def embedding_model_name() -> str:
    return setting("EMBEDDING_MODEL", "Qwen3-Embedding-8B")
