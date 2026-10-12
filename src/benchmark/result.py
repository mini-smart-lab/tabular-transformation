from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.agent.schemas import FinalSubmission


class ResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ColumnResult(ResultModel):
    column: str
    correct: bool
    is_transformed: bool = False
    error: str | None = None


class OperatorMetrics(ResultModel):
    precision: float
    recall: float
    f1: float
    exact_sequence_accuracy: float


class EvaluationResult(ResultModel):
    functional_column_accuracy: float
    transformed_column_accuracy: float | None = None
    columns: list[ColumnResult]
    operator_metrics: OperatorMetrics
    execution_status: str
    error: str | None = None


class BenchmarkResult(ResultModel):
    run_id: str
    task_id: str
    level: int | None = None
    source_dataset: str | None = None
    domain: str | None = None
    ablation_mode: str | None = None
    status: str
    termination_reason: str
    submission: FinalSubmission | None
    budget: dict[str, int | float]
    evaluation: EvaluationResult | None
    transform_path: str | None = None

