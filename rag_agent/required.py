"""The nine required queries: parse them, run them all, write traces and a summary (specs/agent.md §7).

    uv run python scripts/run_required.py [--only Q3 Q8] [--out-dir runs/<name>]

`queries/required-queries.md` holds sections `## Q<n>: <title>`, each with an
`<!-- expected: ANSWER -->` or `<!-- expected: INSUFFICIENT_EVIDENCE -->` line and the query text.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rag_agent.agent import RunResult, run, write_trace
from rag_agent.config import DEFAULT_MANIFEST_PATH, DEFAULT_REGISTRY_PATH, PROJECT_DIR
from rag_agent.controller import Config, TerminalReason

QUERIES_PATH = PROJECT_DIR / "queries" / "required-queries.md"
RUNS_DIR = PROJECT_DIR / "runs"
SUMMARY_HEADER = "| Query | Title | Expected | Actual | Match | Steps | Sources cited |"
EXPECTABLE = (TerminalReason.ANSWER, TerminalReason.INSUFFICIENT_EVIDENCE)

_HEADING = re.compile(r"^## (?P<rest>.*)$")
_QUERY_HEADING = re.compile(r"^Q(?P<number>\d+):\s*(?P<title>.+?)\s*$")
_LOOKS_LIKE_QUERY = re.compile(r"^Q\d*\s*:")       # 'Q:' (number missing) is parsed, then rejected
_EXPECTED = re.compile(r"^<!--\s*expected:\s*(?P<value>[A-Z_]+)\s*-->$")


@dataclass(frozen=True)
class RequiredQuery:
    number: int
    title: str
    expected: TerminalReason
    text: str


# ---------------------------------------------------------------- parsing

def _sections(lines: list[str]) -> list[tuple[str, list[str]]]:
    """(heading text, body lines) for every `## ` heading; text before the first one is dropped."""
    sections: list[tuple[str, list[str]]] = []
    for line in lines:
        heading = _HEADING.match(line)
        if heading:
            sections.append((heading.group("rest").strip(), []))
        elif sections:
            sections[-1][1].append(line)
    return sections


def _parse_section(heading: str, body: list[str]) -> RequiredQuery:
    match = _QUERY_HEADING.match(heading)
    if not match:
        raise ValueError(f"Bad query heading {heading!r}; expected 'Q<n>: <title>'")
    expected = None
    text_lines = []
    for raw in body:
        line = raw.strip()
        if not line:
            continue
        comment = _EXPECTED.match(line)
        if comment:
            value = comment.group("value")
            if value not in {reason.value for reason in EXPECTABLE}:
                raise ValueError(f"{heading}: expected outcome must be ANSWER or INSUFFICIENT_EVIDENCE")
            expected = TerminalReason(value)
        elif not line.startswith("<!--"):
            text_lines.append(line)
    if expected is None:
        raise ValueError(f"{heading}: no '<!-- expected: ... -->' line")
    if not text_lines:
        raise ValueError(f"{heading}: no query text")
    return RequiredQuery(number=int(match.group("number")), title=match.group("title"),
                         expected=expected, text=" ".join(text_lines))


def parse_queries(path: Path) -> list[RequiredQuery]:
    """The queries in file order; a missing expected outcome or a missing or duplicate number is an error."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    queries = [_parse_section(heading, body) for heading, body in _sections(lines)
               if _LOOKS_LIKE_QUERY.match(heading)]
    if not queries:
        raise ValueError(f"No '## Q<n>: <title>' sections in {path}")
    numbers = [query.number for query in queries]
    duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
    if duplicates:
        raise ValueError(f"Duplicate query number(s): {duplicates}")
    missing = sorted(set(range(1, max(numbers) + 1)) - set(numbers))
    if missing:
        raise ValueError(f"Missing query number(s): {missing}")
    return queries


# ---------------------------------------------------------------- running and the summary

def actual(result: RunResult) -> str:
    reason = result.terminal_reason.value
    return f"{reason}/{result.error_kind.value}" if result.error_kind is not None else reason


def cited_sources(result: RunResult) -> str:
    sources = list(dict.fromkeys(citation["source_id"] for citation in result.citations))
    return ", ".join(sources) if sources else "none"


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def summary_markdown(queries: list[RequiredQuery], results: list[RunResult], config: Config,
                     manifest: dict | None) -> str:
    manifest = manifest or {}
    chunking = manifest.get("chunking") or {}
    size, overlap = chunking.get("chunk_size"), chunking.get("chunk_overlap")
    settings = [
        ("Embedding model", manifest.get("embedding_model") or "unknown"),
        ("Chunk size and overlap", f"{size} / {overlap}" if size is not None and overlap is not None else "unknown"),
        ("Retrieval mode", config.retrieval_mode),
        ("Results per search (k)", config.default_k),
        ("Step limit", config.max_steps),
        ("Chat model", config.model_name or "unknown"),
        ("Query instruction", config.query_instruction or "none (off)"),
        ("index_version", manifest.get("index_version") or "unknown"),
    ]
    lines = ["# Required runs", "", "## Recorded configuration", ""]
    lines += [f"- {name}: {value}" for name, value in settings]
    lines += ["", "## Results", "", SUMMARY_HEADER, "|---|---|---|---|---|---|---|"]
    for query, result in zip(queries, results):
        match = "yes" if result.terminal_reason is query.expected else "no"
        lines.append(f"| Q{query.number} | {_cell(query.title)} | {query.expected.value} | {actual(result)} | "
                     f"{match} | {result.steps_used}/{result.max_steps} | {cited_sources(result)} |")
    return "\n".join(lines) + "\n"


def run_all(queries: list[RequiredQuery], config: Config, out_dir: Path, model=None, retriever=None,
            registry=None, manifest: dict | None = None) -> list[RunResult]:
    """Run each query once, in order, without stopping on failures; write traces and the summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for query in queries:
        result = run(query.text, config, model=model, retriever=retriever, registry=registry)
        write_trace(out_dir / f"Q{query.number}.jsonl", result, manifest)
        results.append(result)
    if manifest is not None:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(summary_markdown(queries, results, config, manifest), encoding="utf-8")
    return results


# ---------------------------------------------------------------- command line (scripts/run_required.py)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts/run_required.py",
                                     description="Run the required queries and write traces and a summary.")
    parser.add_argument("--only", nargs="+", metavar="QN", help="run only these queries, such as Q3 Q8")
    parser.add_argument("--out-dir", type=Path, metavar="DIR",
                        help="output folder (default: runs/<YYYY-MM-DD_HHMM>)")
    args = parser.parse_args(argv)

    queries = parse_queries(QUERIES_PATH)
    if args.only:
        wanted = {name.upper().lstrip("Q") for name in args.only}
        unknown = wanted - {str(q.number) for q in queries}
        if unknown:
            parser.error(f"unknown query: {', '.join(sorted(unknown))}")
        queries = [q for q in queries if str(q.number) in wanted]
    out_dir = args.out_dir or RUNS_DIR / datetime.now().strftime("%Y-%m-%d_%H%M")

    from rag_agent.corpus import load_registry

    config = Config()
    registry = load_registry(DEFAULT_REGISTRY_PATH)
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8")) if DEFAULT_MANIFEST_PATH.is_file() else None
    # Print the configuration first, so a wrong setting (for example an LLM_MODEL left in the shell,
    # which overrides .env) is visible before any query runs. Added 2026-10-27 after nine runs used
    # the wrong model.
    print(f"Chat model: {config.model_name} · retrieval: {config.retrieval_mode} · k = {config.default_k} · "
          f"step limit: {config.max_steps} · index: {(manifest or {}).get('index_version', 'unknown')}",
          flush=True)
    model, retriever = _build_services(config)
    results = run_all(queries, config, out_dir, model=model, retriever=retriever, registry=registry,
                      manifest=manifest)
    for query, result in zip(queries, results):
        print(f"Q{query.number}: {actual(result)} ({result.steps_used}/{result.max_steps} steps), "
              f"expected {query.expected.value}")
    print(f"Wrote {out_dir}")
    return 0


def _build_services(config: Config):
    """The real model and retriever, built once; on failure None, so each run records the error."""
    from rag_agent.config import DEFAULT_DB_PATH
    from rag_agent.llm import make_model
    from rag_agent.retrieval import HybridRetriever

    try:
        model = make_model(config)
    except Exception as error:
        print(f"Chat model unavailable: {type(error).__name__}: {error}")
        model = None
    try:
        retriever = HybridRetriever(DEFAULT_DB_PATH, mode=config.retrieval_mode,
                                    query_instruction=config.query_instruction)
    except Exception as error:
        print(f"Vector store unavailable: {type(error).__name__}: {error}")
        retriever = None
    return model, retriever
