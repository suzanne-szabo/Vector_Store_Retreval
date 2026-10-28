"""The chat-model factory (specs/agent.md §3, architecture D11).

`make_model` returns a `ChatOpenAI` for the course's OpenAI-compatible chat service, with
`temperature` from the config and `max_retries=0` (the controller does its own single retry).
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from rag_agent.config import require_api_key, setting
from rag_agent.controller import Config

DEFAULT_LLM_BASE_URL = "http://golem:4000/v1"
TIMEOUT_SECONDS = 180


def make_model(config: Config) -> ChatOpenAI:
    """The course chat model; raises `RuntimeError` when `LLM_API_KEY` is missing or a placeholder."""
    return ChatOpenAI(
        model=config.model_name,
        base_url=setting("LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
        api_key=require_api_key("LLM_API_KEY"),
        temperature=config.temperature,
        max_retries=0,
        timeout=TIMEOUT_SECONDS,
    )
