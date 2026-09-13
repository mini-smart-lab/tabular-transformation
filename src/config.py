from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentBudgetConfig(StrictModel):
    max_total_tokens: int = Field(default=100_000, gt=0)
    max_tool_calls: int = Field(default=25, gt=0)
    max_total_execution_seconds: float = Field(default=120.0, gt=0)
    max_total_cpu_seconds: float = Field(default=120.0, gt=0)
    max_wall_time_seconds: float = Field(default=300.0, gt=0)


DEFAULT_LEVEL_BUDGETS: dict[int, AgentBudgetConfig] = {
    0: AgentBudgetConfig(
        max_total_tokens=40_000,
        max_tool_calls=8,
        max_total_execution_seconds=30.0,
        max_total_cpu_seconds=30.0,
        max_wall_time_seconds=120.0,
    ),
    1: AgentBudgetConfig(
        max_total_tokens=40_000,
        max_tool_calls=10,
        max_total_execution_seconds=30.0,
        max_total_cpu_seconds=30.0,
        max_wall_time_seconds=120.0,
    ),
    2: AgentBudgetConfig(
        max_total_tokens=60_000,
        max_tool_calls=14,
        max_total_execution_seconds=60.0,
        max_total_cpu_seconds=60.0,
        max_wall_time_seconds=180.0,
    ),
    3: AgentBudgetConfig(
        max_total_tokens=90_000,
        max_tool_calls=18,
        max_total_execution_seconds=90.0,
        max_total_cpu_seconds=90.0,
        max_wall_time_seconds=240.0,
    ),
    4: AgentBudgetConfig(
        max_total_tokens=120_000,
        max_tool_calls=24,
        max_total_execution_seconds=120.0,
        max_total_cpu_seconds=120.0,
        max_wall_time_seconds=300.0,
    ),
    5: AgentBudgetConfig(
        max_total_tokens=150_000,
        max_tool_calls=30,
        max_total_execution_seconds=150.0,
        max_total_cpu_seconds=150.0,
        max_wall_time_seconds=360.0,
    ),
}

# Backwards compatibility alias
LEVEL_BUDGETS = DEFAULT_LEVEL_BUDGETS


def get_level_budget(
    level: int,
    base_budget: AgentBudgetConfig | None = None,
    level_budgets: dict[int, AgentBudgetConfig] | None = None,
) -> AgentBudgetConfig:
    """Return level-appropriate budget, checking custom level_budgets first, then default level budgets, then base_budget."""
    if level_budgets and level in level_budgets:
        return level_budgets[level]
    if level in DEFAULT_LEVEL_BUDGETS:
        return DEFAULT_LEVEL_BUDGETS[level]
    return base_budget or AgentBudgetConfig()


class SandboxConfig(StrictModel):
    backend: str = "local"
    execution_timeout_seconds: float = Field(default=15.0, gt=0)
    max_output_chars: int = Field(default=8_192, gt=0)
    docker_image: str = "tabular-benchmark-sandbox:latest"
    docker_network: str = "none"
    docker_cpus: float = Field(default=1.0, gt=0)
    docker_memory: str = "1g"
    docker_pids_limit: int = Field(default=64, gt=0)


class BenchmarkConfig(StrictModel):
    numeric_atol: float = Field(default=1e-6, ge=0)
    numeric_rtol: float = Field(default=1e-5, ge=0)
    max_workers: int = Field(default=4, ge=1)


class ModelConfig(StrictModel):
    provider: str = "nvidia"
    model: str = "nvidia/nemotron-3-super-120b-a12b"
    temperature: float = Field(default=0.0, ge=0)
    seed: int | None = Field(default=42)
    api_key_env: str = "NVIDIA_API_KEY"
    max_output_tokens: int | None = Field(default=None, gt=0)


class HarnessConfig(StrictModel):
    budget: AgentBudgetConfig = Field(default_factory=AgentBudgetConfig)
    level_budgets: dict[int, AgentBudgetConfig] = Field(default_factory=lambda: dict(DEFAULT_LEVEL_BUDGETS))
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    operator_vocabulary: list[str] = Field(default_factory=lambda: [
        "LogTransformation",
        "Log1pTransformation",
        "SquareRootTransformation",
        "SquareTransformation",
        "CubeTransformation",
        "CubeRootTransformation",
        "ExpTransformation",
        "InverseTransformation",
        "StandardScalerTransformation",
        "MinMaxScalerTransformation",
        "SinTransformation",
        "CosTransformation",
        "TanhTransformation",
        "SigmoidTransformation",
        "AbsTransformation",
        "AddInteractionTransformation",
        "SubtractInteractionTransformation",
        "MultiplyInteractionTransformation",
        "DivideInteractionTransformation",
        "IndicatorZeroTransformation",
        "IndicatorMeanTransformation",
        "ClipZeroTransformation",
        "PiecewiseLogTransformation",
        "ConditionalInteractionTransformation",
    ])

