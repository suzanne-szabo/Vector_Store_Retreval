"""The composition root for one run (specs/controller.md §1, architecture §4.5).

A `Session` holds the run's question, retriever, source registry, configuration and the retrieved
set. Only `rag_agent.tools` adds to `retrieved` (architecture §5).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rag_agent.corpus import Source
from rag_agent.retrieval import Passage, Retriever

if TYPE_CHECKING:
    from rag_agent.controller import Config


@dataclass(frozen=True)
class Session:
    question: str
    config: Config
    retriever: Retriever
    registry: tuple[Source, ...]
    source_ids: frozenset[str]
    retrieved: dict[str, Passage] = field(default_factory=dict)   # insertion-ordered; tools only

    @classmethod
    def new(cls, question: str, config: Config, retriever: Retriever,
            registry: Iterable[Source]) -> Session:
        """A fresh session; an empty (or blank) question raises `ValueError`."""
        text = (question or "").strip()
        if not text:
            raise ValueError("The question must not be empty.")
        sources = tuple(registry)
        return cls(question=text, config=config, retriever=retriever, registry=sources,
                   source_ids=frozenset(source.id for source in sources))
