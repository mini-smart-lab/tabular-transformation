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
    config = resolve_model_config(config)
    api_key = os.getenv(config.api_key_env, "")
    if config.provider not in ("brute_force",) and config.api_key_env not in ("", "NO_KEY_REQUIRED", "NONE") and not api_key:
        raise RuntimeError(f"missing API key environment variable: {config.api_key_env}")
    try:
        builder = _PROVIDERS[config.provider]
    except KeyError as exc:
        raise ValueError(f"unsupported model provider: {config.provider}") from exc
    return builder(config, api_key)


def resolve_model_config(config: ModelConfig) -> ModelConfig:
    """Apply non-secret environment overrides and return the effective run metadata."""
    load_dotenv(".env", override=False)
    return config.model_copy(update={
        "provider": os.getenv("BENCHMARK_MODEL_PROVIDER", config.provider),
        "model": os.getenv("BENCHMARK_MODEL", config.model),
        "api_key_env": os.getenv("BENCHMARK_API_KEY_ENV", config.api_key_env),
    })


def _build_nvidia(config: ModelConfig, api_key: str) -> Any:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    kwargs: dict[str, Any] = dict(
        model=config.model,
        temperature=config.temperature,
        api_key=api_key,
    )
    if config.seed is not None:
        kwargs["seed"] = config.seed
    if config.max_output_tokens is not None:
        kwargs["max_tokens"] = config.max_output_tokens
    return ChatNVIDIA(**kwargs)


def _build_openai(config: ModelConfig, api_key: str) -> Any:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError("provider 'openai' requires the optional langchain-openai package") from exc
    kwargs: dict[str, Any] = dict(
        model=config.model,
        temperature=config.temperature,
        api_key=api_key,
        max_tokens=config.max_output_tokens,
        max_retries=10,
    )
    if config.seed is not None:
        kwargs["seed"] = config.seed
    return ChatOpenAI(**kwargs)


def _build_brute_force(config: ModelConfig, api_key: str) -> Any:
    from src.agent.brute_force import BruteForceModel
    return BruteForceModel(config=config)


def _build_kaggle_proxy(config: ModelConfig, api_key: str) -> Any:
    try:
        from langchain_openai import ChatOpenAI
        import langchain_openai.chat_models.base as lc_openai_base
        from langchain_core.messages import AIMessage

        # Patch against null message choices returned by proxy endpoints
        if not getattr(lc_openai_base, "_safe_message_patch_applied", False):
            _orig_convert = lc_openai_base._convert_dict_to_message

            def _safe_convert(d: Any) -> Any:
                if d is None:
                    return AIMessage(content="[Empty message from proxy endpoint]")
                return _orig_convert(d)

            lc_openai_base._convert_dict_to_message = _safe_convert
            lc_openai_base._safe_message_patch_applied = True
    except ImportError as exc:
        raise RuntimeError("provider 'kaggle_proxy' requires langchain-openai package") from exc
    default_base = os.getenv("MODEL_PROXY_URL_OPENAPI")
    if not default_base:
        proxy_base = os.getenv("MODEL_PROXY_URL", "https://mp-staging.kaggle.net/models")
        default_base = f"{proxy_base.rstrip('/')}/openapi"
    base_url = default_base
    effective_key = api_key or os.getenv("MODEL_PROXY_API_KEY", "")
    kwargs: dict[str, Any] = dict(
        model=config.model,
        api_key=effective_key,
        base_url=base_url,
        max_tokens=config.max_output_tokens,
        max_retries=10,
    )
    if config.temperature is not None and "claude" not in config.model.lower():
        kwargs["temperature"] = config.temperature
    return ChatOpenAI(**kwargs)


register_provider("nvidia", _build_nvidia)
register_provider("openai", _build_openai)
register_provider("brute_force", _build_brute_force)
register_provider("kaggle_proxy", _build_kaggle_proxy)


