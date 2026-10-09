"""Thin Gemini wrappers for the notes agent (single seam for tests to patch)."""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

_CLIENT = None
_CLIENT_LOCK = threading.Lock()


def client():
    """Shared Gemini client (created lazily)."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            from src.core.gemini_client import create_client

            _CLIENT = create_client()
        return _CLIENT


def _config(schema: Any = None, json_mode: bool = False, system: str | None = None):
    from google.genai import types

    kwargs: dict[str, Any] = {"temperature": 0.0}
    if system:
        kwargs["system_instruction"] = system
    if schema is not None or json_mode:
        kwargs["response_mime_type"] = "application/json"
    if schema is not None:
        kwargs["response_schema"] = schema
    return types.GenerateContentConfig(**kwargs)


def _generate(model: str, contents: Any, config: Any, retries: int = 2):
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return client().models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as exc:  # noqa: BLE001 - network/API errors are retried
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Gemini call failed after {retries + 1} attempts: {last}")


def generate_json(
    prompt: str | list,
    *,
    model: str,
    schema: Any = None,
    system: str | None = None,
) -> Any:
    """Run a JSON-mode generation and return the parsed payload."""
    response = _generate(model, prompt, _config(schema, json_mode=True, system=system))
    return parse_json(response.text or "")


def generate_text(prompt: str | list, *, model: str, system: str | None = None) -> str:
    """Run a plain text generation."""
    response = _generate(model, prompt, _config(system=system))
    return (response.text or "").strip()


def generate_with_tools(model: str, contents: list, tool, system: str):
    """One tool-enabled turn; returns the raw response for the caller's loop."""
    from google.genai import types

    config = types.GenerateContentConfig(
        temperature=0.0, tools=[tool], system_instruction=system
    )
    return _generate(model, contents, config)


def parse_json(text: str) -> Any:
    """Parse JSON from model text, tolerating code fences and leading prose."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(opener), cleaned.rfind(closer)
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Model did not return JSON: {text[:200]!r}")
