from pathlib import Path

import pandas as pd

from src.agent.schemas import FinalSubmission, Transformation
from src.benchmark.task import BenchmarkTask
from src.evaluator.evaluator import Evaluator
from src.sandbox.local_mock import LocalMockSandbox
from src.sandbox.models import ExecutionResult


def test_evaluator_column_accuracy_and_fresh_hidden_boundary(tmp_path: Path):
    before = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [1, 2, 3]})
    after = pd.DataFrame({"x1": [2.0, 4.0, 6.0], "x2": [1, 4, 9]})
    train_before = tmp_path / "before.csv"
    train_after = tmp_path / "after.csv"
    test_before = tmp_path / "test_before.csv"
    test_after = tmp_path / "test_after.csv"
    for path, frame in (
        (train_before, before),
        (train_after, after),
        (test_before, before),
        (test_after, after),
    ):
        frame.to_csv(path, index=False)

    task = BenchmarkTask(
        task_id="synthetic",
        before_path=train_before,
        after_path=train_after,
        test_before_path=test_before,
        test_after_path=test_after,
        ground_truth_transformations=[
            Transformation(column="x1", operators=["multiply_2"]),
            Transformation(column="x2", operators=["square"]),
        ],
    )
    predicted = after.copy()
    predicted.loc[0, "x2"] = 999
    mounted = []

    def factory(files):
        mounted.append(files)
        return LocalMockSandbox(
            [ExecutionResult(status="success")],
            {"predicted.csv": predicted.to_csv(index=False).encode()},
        )

    submission = FinalSubmission(
        script=(
            "def transform(df):\n"
            "    result = df.copy()\n"
            "    result['x1'] = result['x1'] * 2\n"
            "    result['x2'] = result['x2'] ** 2\n"
            "    return result\n"
        ),
        transformations=[
            Transformation(column="x1", operators=["multiply_2"]),
            Transformation(column="x2", operators=["square"]),
        ],
    )
    result = Evaluator(factory).evaluate(task, submission)
    assert result.functional_column_accuracy == 0.5
    assert result.operator_metrics.f1 == 1.0
    assert set(mounted[0]) == {"test_before.csv"}
    assert task.test_after_path not in mounted[0].values()


def test_evaluator_execution_error(tmp_path: Path):
    frame = pd.DataFrame({"x": [1]})
    paths = []
    for name in ("before", "after", "test_before", "test_after"):
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    task = BenchmarkTask(
        task_id="failure",
        before_path=paths[0], after_path=paths[1],
        test_before_path=paths[2], test_after_path=paths[3],
        ground_truth_transformations=[],
    )
    factory = lambda _: LocalMockSandbox(
        [ExecutionResult(status="error", stderr="boom")]
    )
    result = Evaluator(factory).evaluate(
        task, FinalSubmission(script="def transform(df):\n    return df\n", transformations=[])
    )
    assert result.functional_column_accuracy == 0
    assert result.error == "boom"
