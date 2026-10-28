"""Command line: ask one question, print the trace as it happens (specs/agent.md §5).

    uv run python -m rag_agent "What does the JALR instruction write to rd?"
    uv run python -m rag_agent "QUESTION" --max-steps 6 --k 3 --trace-out runs/q.jsonl

Exit status: 0 for ANSWER or INSUFFICIENT_EVIDENCE, 1 for ERROR or STEP_LIMIT, 2 for usage problems.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_agent.agent import RunResult, run, write_trace
from rag_agent.config import DEFAULT_DB_PATH, DEFAULT_MANIFEST_PATH
from rag_agent.controller import DEFAULT_QUERY_INSTRUCTION, Config, TerminalReason, TraceEntry
from rag_agent.retrieval import MODES
from rag_agent.tools import MAX_K

PROG = "python -m rag_agent"
NO_STORE = "No vector store found; run: uv run python -m rag_agent.ingest --rebuild"
EXAMPLES = f"""examples:
  uv run {PROG} "What does the RISC-V JALR instruction write to its destination register?"
  uv run {PROG} "How does Python's // operator differ from /?" --k 3 --trace-out runs/floor.jsonl
  uv run {PROG} "Which registers are callee-saved?" --mode keyword --quiet

exit status: 0 answer or insufficient evidence, 1 error or step limit, 2 usage problem"""
THOUGHT_CHARS = 200


def build_parser() -> argparse.ArgumentParser:
    defaults = Config()
    parser = argparse.ArgumentParser(
        prog=PROG, formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EXAMPLES,
        description="Answer one Python or RISC-V question from the document corpus, citing the "
                    "retrieved passages, or report that the corpus does not support an answer.")
    parser.add_argument("question", help="the question to answer (quote it)")
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps, metavar="N",
                        help=f"model/tool steps allowed before the run stops (default: {defaults.max_steps})")
    parser.add_argument("--k", type=int, default=defaults.default_k, metavar="N",
                        help=f"passages per search when the model does not choose, 1-{MAX_K} "
                             f"(default: {defaults.default_k})")
    parser.add_argument("--mode", choices=MODES, default=defaults.retrieval_mode,
                        help=f"retrieval mode (default: {defaults.retrieval_mode})")
    parser.add_argument("--query-instruction", default=DEFAULT_QUERY_INSTRUCTION, metavar="TEXT",
                        help="Qwen3 query instruction for embedded queries; an empty string turns it off "
                             f'(default: "{DEFAULT_QUERY_INSTRUCTION}")')
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, metavar="DIR",
                        help=f"vector store folder (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--trace-out", type=Path, metavar="FILE.jsonl",
                        help="also write the whole run as a JSONL trace (default: not written)")
    parser.add_argument("--quiet", action="store_true",
                        help="print only the result, not each step (default: print each step)")
    return parser


def _parse(argv: list[str] | None) -> argparse.Namespace | None:
    """Parsed arguments, or None for a usage error (argparse has already printed it)."""
    try:
        return build_parser().parse_args(argv)
    except SystemExit as stop:
        if stop.code in (0, None):       # --help
            raise
        return None


# ---------------------------------------------------------------- printing

def format_entry(entry: TraceEntry, max_steps: int) -> str:
    prefix = f"[step {entry.step}/{max_steps}]"
    if entry.kind == "thought":
        return f"{prefix} thought: {entry.message[:THOUGHT_CHARS]}"
    if entry.kind == "no_action":
        return f"{prefix} no tool call: {entry.message}"
    if entry.kind == "model_error":
        return f"{prefix} model error: {entry.message}"
    args = entry.args if isinstance(entry.args, str) else json.dumps(entry.args, ensure_ascii=False)
    if entry.accepted:
        status = (entry.observation or {}).get("data", {}).get("status", "ok")
        outcome = f"{status}: {entry.message}"
    elif entry.policy:
        outcome = f"REJECTED {entry.policy} {entry.reason}: {entry.message}"
    else:
        outcome = f"ERROR {entry.reason}: {entry.message}"
    return f"{prefix} {entry.tool} {args} -> {outcome}"


def format_result(result: RunResult) -> str:
    reason = result.terminal_reason.value
    if result.error_kind is not None:
        reason = f"{reason}/{result.error_kind.value}"
    lines = [f"{reason}: {result.message} ({result.steps_used}/{result.max_steps} steps)"]
    if result.terminal_reason in (TerminalReason.ANSWER, TerminalReason.INSUFFICIENT_EVIDENCE):
        lines += ["", result.answer or ""]
        if result.citations:
            lines += ["", "Citations:"]
            lines += [f"[{c['n']}] {c['source_id']}, page {c['page']} (chunk {c['chunk_id'][:8]}…): "
                      f"{c['supports']}" for c in result.citations]
    return "\n".join(lines)


def _load_manifest() -> dict | None:
    if not DEFAULT_MANIFEST_PATH.is_file():
        return None
    return json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None, *, model=None, retriever=None) -> int:
    args = _parse(argv)
    if args is None:
        return 2
    try:
        config = Config(max_steps=args.max_steps, default_k=args.k, retrieval_mode=args.mode,
                        query_instruction=args.query_instruction or None)
    except ValueError as error:
        print(f"{PROG}: error: {error}", file=sys.stderr)
        return 2
    if not args.question.strip():
        print(f"{PROG}: error: the question must not be empty", file=sys.stderr)
        return 2

    if retriever is None:
        if not args.db.exists():
            print(NO_STORE)
            return 2
        try:
            from rag_agent.retrieval import HybridRetriever

            retriever = HybridRetriever(args.db, mode=config.retrieval_mode,
                                        query_instruction=config.query_instruction)
        except Exception as error:
            print(f"Could not open the vector store: {type(error).__name__}: {error}")
            return 1

    on_entry = None if args.quiet else (lambda entry: print(format_entry(entry, config.max_steps), flush=True))
    result = run(args.question, config, model=model, retriever=retriever, on_entry=on_entry)
    if not args.quiet:
        print()
    print(format_result(result))
    if args.trace_out:
        write_trace(args.trace_out, result, _load_manifest())
        print(f"\nTrace written to {args.trace_out}")
    return 0 if result.terminal_reason in (TerminalReason.ANSWER, TerminalReason.INSUFFICIENT_EVIDENCE) else 1
