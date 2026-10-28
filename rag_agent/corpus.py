"""Source registry: which files make up the corpus, and checks that they are the right files."""

from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

SUBJECTS = ("python", "riscv", "pedagogy")
_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CorpusError(Exception):
    """The registry or the documents folder does not match what ingestion expects."""


@dataclass(frozen=True)
class Source:
    id: str
    file: str
    title: str
    subject: str
    version: str
    role: str
    license: str
    url: str
    pages: int
    sha256: str
    columns: int = 1


def load_registry(path: Path) -> list[Source]:
    """Read and validate `sources.toml`; returns sources in file order."""
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    entries = data.get("source", [])
    if not entries:
        raise CorpusError(f"No [[source]] entries in {path}")
    sources: list[Source] = []
    seen: set[str] = set()
    for entry in entries:
        try:
            source = Source(**entry)
        except TypeError as error:
            raise CorpusError(f"Bad registry entry {entry.get('id', '?')}: {error}") from None
        if not _ID.match(source.id):
            raise CorpusError(f"Bad source id {source.id!r}")
        if source.id in seen:
            raise CorpusError(f"Duplicate source id {source.id!r}")
        if source.subject not in SUBJECTS:
            raise CorpusError(f"{source.id}: subject must be one of {SUBJECTS}")
        if not _SHA256.match(source.sha256):
            raise CorpusError(f"{source.id}: sha256 must be 64 lowercase hex characters")
        if source.columns not in (1, 2):
            raise CorpusError(f"{source.id}: columns must be 1 or 2")
        seen.add(source.id)
        sources.append(source)
    return sources


def select(sources: list[Source], only: list[str] | None) -> list[Source]:
    """Keep the sources named in `only` (all when None); unknown ids are an error."""
    if not only:
        return list(sources)
    known = {source.id for source in sources}
    unknown = [name for name in only if name not in known]
    if unknown:
        raise CorpusError(f"Unknown source id(s): {', '.join(unknown)}. Known: {', '.join(sorted(known))}")
    wanted = set(only)
    return [source for source in sources if source.id in wanted]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(sources: list[Source], documents_dir: Path) -> list[tuple[Source, Path]]:
    """Find each source's file and verify its SHA-256; any problem stops the build."""
    resolved: list[tuple[Source, Path]] = []
    problems: list[str] = []
    for source in sources:
        path = documents_dir / source.file
        if not path.is_file():
            problems.append(f"{source.id}: missing file {path}")
            continue
        actual = sha256_file(path)
        if actual != source.sha256:
            problems.append(f"{source.id}: SHA-256 mismatch for {source.file} (expected {source.sha256[:16]}…, got {actual[:16]}…)")
            continue
        resolved.append((source, path))
    if problems:
        raise CorpusError("Corpus check failed:\n  " + "\n  ".join(problems))
    return resolved
