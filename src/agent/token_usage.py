from __future__ import annotations

from typing import Any


def extract_token_usage(message: Any) -> tuple[int, int]:
    """Normalize LangChain/provider usage metadata in one provider-isolation point."""
    usage = getattr(message, "usage_metadata", None) or {}
    response = getattr(message, "response_metadata", None) or {}
    provider_usage = response.get("token_usage") or response.get("usage") or {}
    input_tokens = _first_int(usage, "input_tokens", "prompt_tokens")
    output_tokens = _first_int(usage, "output_tokens", "completion_tokens")

    if not input_tokens:
        input_tokens = _first_int(provider_usage, "input_tokens", "prompt_tokens")
    if not output_tokens:
        output_tokens = _first_int(provider_usage, "output_tokens", "completion_tokens")
    return input_tokens, output_tokens


def _first_int(mapping: Any, *keys: str) -> int:
    if not isinstance(mapping, dict):
        return 0
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0

