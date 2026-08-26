from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentBudgetConfig(StrictModel):
    max_total_tokens: int = Field(default=16_000, gt=0)
    max_tool_calls: int = Field(default=20, gt=0)
    max_total_execution_seconds: float = Field(default=60.0, gt=0)
    max_wall_time_seconds: float = Field(default=120.0, gt=0)


class SandboxConfig(StrictModel):
    execution_timeout_seconds: float = Field(default=10.0, gt=0)
    max_output_chars: int = Field(default=8_192, gt=0)


class BenchmarkConfig(StrictModel):
    numeric_atol: float = Field(default=1e-6, ge=0)
    numeric_rtol: float = Field(default=1e-5, ge=0)


class ModelConfig(StrictModel):
    provider: Literal["nvidia"] = "nvidia"
    model: str = "nvidia/nemotron-3-super-120b-a12b"
    temperature: float = Field(default=0.0, ge=0)
    api_key_env: str = "NVIDIA_API_KEY"
