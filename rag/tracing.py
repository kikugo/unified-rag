"""Optional Langfuse tracing for retrieval and answering.

Off unless LANGFUSE_TRACING=1. When off, every helper here is a no-op context
manager, so the tests, the CLI and the app run exactly as before and nothing
talks to Langfuse. When on, the Langfuse client is created on first use from
LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL (read from .env
if they aren't in the environment).

Spans nest under whatever observation is current, so `retrieve` and `generate`
land inside the per-question trace the eval runner opens.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes

FLAG = "LANGFUSE_TRACING"
KEY_VARS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")

_client: Any = None


def enabled() -> bool:
    return os.getenv(FLAG) == "1"


def use_client(client: Any) -> None:
    """Swap in a client (tests pass a recorder); None goes back to the real one."""
    global _client
    _client = client


def require_keys() -> None:
    """Fail before any paid call when tracing is asked for but can't report anywhere."""
    load_dotenv()
    missing = [v for v in KEY_VARS if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"tracing needs {', '.join(missing)} in the environment or .env "
                           "(and LANGFUSE_BASE_URL for the region)")


def client() -> Any:
    global _client
    if _client is None:
        load_dotenv()
        _client = get_client()
    return _client


@contextmanager
def stage(name: str, **metadata: Any) -> Iterator[Any]:
    """A child span around one pipeline stage, or nothing when tracing is off."""
    if not enabled():
        yield None
        return
    with client().start_as_current_observation(as_type="span", name=name,
                                                metadata=metadata or None) as span:
        yield span


@contextmanager
def generation(name: str, model: str, input: Any = None) -> Iterator[Any]:
    """A generation span, so Langfuse shows the model, output and token usage."""
    if not enabled():
        yield None
        return
    with client().start_as_current_observation(as_type="generation", name=name, model=model,
                                                input=input) as gen:
        yield gen


def record_generation(gen: Any, output: str, usage: Any = None, input_tokens: int | None = None,
                      output_tokens: int | None = None) -> None:
    """Attach output and tokens to a generation span; a no-op for None.

    Takes either Gemini's `usage_metadata` or explicit token counts. Thinking
    tokens are billed as output, so they're counted as output here too.
    """
    if gen is None:
        return
    if usage is not None:
        input_tokens = usage.prompt_token_count or 0
        output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    details = None
    if input_tokens is not None:
        details = {"input": input_tokens, "output": output_tokens or 0}
    gen.update(output=output, usage_details=details)


@contextmanager
def attributes(tags: list[str], metadata: dict[str, str] | None = None) -> Iterator[None]:
    """Tags and metadata on every span opened inside, e.g. the config name and question id."""
    if not enabled():
        yield
        return
    with propagate_attributes(tags=tags, metadata=metadata):
        yield


def flush() -> None:
    if enabled() and _client is not None:
        _client.flush()
