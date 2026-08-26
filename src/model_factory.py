from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv

from src.config import ModelConfig


ModelBuilder = Callable[[ModelConfig, str], Any]
_PROVIDERS: dict[str, ModelBuilder] = {}


def register_provider(name: str, builder: ModelBuilder) -> None:
    _PROVIDERS[name] = builder


def create_model(config: ModelConfig, *, env_file: str = ".env") -> Any:
    load_dotenv(env_file, override=False)
    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"missing API key environment variable: {config.api_key_env}")
    try:
        builder = _PROVIDERS[config.provider]
    except KeyError as exc:
        raise ValueError(f"unsupported model provider: {config.provider}") from exc
    return builder(config, api_key)


def _build_nvidia(config: ModelConfig, api_key: str) -> Any:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    return ChatNVIDIA(
        model=config.model,
        temperature=config.temperature,
        api_key=api_key,
    )


register_provider("nvidia", _build_nvidia)
