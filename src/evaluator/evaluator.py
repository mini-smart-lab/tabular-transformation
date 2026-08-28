from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from src.agent.schemas import FinalSubmission
from src.benchmark.result import EvaluationResult
from src.benchmark.task import BenchmarkTask
from src.config import BenchmarkConfig
from src.evaluator.column_metrics import compare_columns
from src.evaluator.operator_metrics import compute_operator_metrics
from src.evaluator.validation import validate_submission
from src.sandbox.base import Sandbox


class Evaluator:
    def __init__(
        self,
        sandbox_factory: Callable[[dict[str, object]], Sandbox],
        config: BenchmarkConfig | None = None,
    ) -> None:
        self.sandbox_factory = sandbox_factory
        self.config = config or BenchmarkConfig()

    def evaluate(self, task: BenchmarkTask, submission: FinalSubmission) -> EvaluationResult:
        validate_submission(submission)
        if task.test_after_path is None:
            raise ValueError("hidden test output has not been attached to this task")
        sandbox = self.sandbox_factory(task.evaluation_files())
        sandbox.start()
        try:
            # The hidden expected output is deliberately never mounted into this sandbox.
            code = _evaluation_program(submission.script)
            execution = sandbox.execute(code)
            metrics = compute_operator_metrics(
                submission.transformations, task.ground_truth_transformations
            )
            if execution.status != "success":
                return EvaluationResult(
                    functional_column_accuracy=0.0,
                    columns=[],
                    operator_metrics=metrics,
                    execution_status=execution.status,
                    error=execution.stderr or "transformation execution failed",
                )
            try:
                predicted = pd.read_csv(_bytes_path_adapter(sandbox.read_file("predicted.csv")))
                target = pd.read_csv(task.test_after_path)
                columns, accuracy = compare_columns(
                    predicted, target,
                    atol=self.config.numeric_atol, rtol=self.config.numeric_rtol,
                )
            except Exception as exc:
                return EvaluationResult(
                    functional_column_accuracy=0.0, columns=[], operator_metrics=metrics,
                    execution_status="error", error=f"invalid evaluator output: {exc}",
                )
            return EvaluationResult(
                functional_column_accuracy=accuracy,
                columns=columns,
                operator_metrics=metrics,
                execution_status="success",
            )
        finally:
            sandbox.close()


def _evaluation_program(script: str) -> str:
    return (
        script
        + "\n\nimport pandas as pd\n"
        + "_input_df = pd.read_csv('test_before.csv')\n"
        + "_predicted_df = transform(_input_df.copy())\n"
        + "if not isinstance(_predicted_df, pd.DataFrame):\n"
        + "    raise TypeError('transform(df) must return a pandas DataFrame')\n"
        + "_predicted_df.to_csv('predicted.csv', index=False)\n"
    )


def _bytes_path_adapter(content: bytes):
    from io import BytesIO

    return BytesIO(content)
