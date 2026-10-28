"""Check forced tool calling on the course chat service (specs/agent.md §3). Live only; not a test.

    uv run python scripts/check_llm.py > docs/llm-check-$(date +%F).txt

Sends one question with the two tools bound, once with tool_choice="auto" and once with
tool_choice="required", and prints whether exactly one well-formed tool call came back, its name
and arguments, and the latency.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # the project is not an installed package

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from rag_agent.config import DEFAULT_REGISTRY_PATH, setting  # noqa: E402
from rag_agent.controller import Config  # noqa: E402
from rag_agent.corpus import load_registry  # noqa: E402
from rag_agent.llm import DEFAULT_LLM_BASE_URL, make_model  # noqa: E402
from rag_agent.prompts import system_prompt  # noqa: E402
from rag_agent.tools import SCHEMAS, tool_definitions  # noqa: E402

QUESTION = "What does the RISC-V JALR instruction write to its destination register?"


def check(model, tool_choice: str, messages) -> None:
    bound = model.bind_tools(tool_definitions(), tool_choice=tool_choice, parallel_tool_calls=False)
    started = time.monotonic()
    try:
        reply = bound.invoke(messages)
    except Exception as error:
        print(f"tool_choice={tool_choice}: FAILED after {time.monotonic() - started:.1f} s: "
              f"{type(error).__name__}: {error}")
        return
    seconds = time.monotonic() - started
    calls, invalid = reply.tool_calls, reply.invalid_tool_calls
    well_formed = len(calls) == 1 and not invalid and calls[0]["name"] in SCHEMAS
    if well_formed:
        try:
            SCHEMAS[calls[0]["name"]].model_validate(calls[0]["args"])
        except Exception as error:
            well_formed = False
            print(f"  schema check failed: {error}")
    print(f"tool_choice={tool_choice}: exactly one well-formed call: {'yes' if well_formed else 'no'} "
          f"({len(calls)} parsed, {len(invalid)} invalid) in {seconds:.1f} s")
    for call in calls:
        print(f"  {call['name']} {json.dumps(call['args'], ensure_ascii=False)}")
    for call in invalid:
        print(f"  INVALID {call.get('name')} {call.get('args')!r}: {call.get('error')}")
    if reply.content:
        print(f"  text: {str(reply.content)[:300]}")


def main() -> int:
    config = Config()
    print(f"Chat model {config.model_name} at {setting('LLM_BASE_URL', DEFAULT_LLM_BASE_URL)}")
    model = make_model(config)
    messages = [SystemMessage(system_prompt(load_registry(DEFAULT_REGISTRY_PATH))), HumanMessage(QUESTION),
                HumanMessage("Step 1 of 8.\nRetrieved passages you may cite: none yet. "
                             "Call search_documents first.")]
    for tool_choice in ("auto", "required"):
        check(model, tool_choice, messages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
