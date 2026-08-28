from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.agent.schemas import FinalSubmission


class ResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ColumnResult(ResultModel):
    column: str
    correct: bool
    error: str | None = None


class OperatorMetrics(ResultModel):
    precision: float
    recall: float
    f1: float
    exact_sequence_accuracy: float


class EvaluationResult(ResultModel):
    functional_column_accuracy: float
    columns: list[ColumnResult]
    operator_metrics: OperatorMetrics
    execution_status: str
    error: str | None = None


class BenchmarkResult(ResultModel):
    run_id: str
    task_id: str
    status: str
    termination_reason: str
    submission: FinalSubmission | None
    budget: dict[str, int | float]
    evaluation: EvaluationResult | None
    transform_path: str | None = None
