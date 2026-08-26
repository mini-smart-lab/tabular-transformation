from pathlib import Path

import pandas as pd

from src.agent.schemas import FinalSubmission, Transformation
from src.benchmark.task import BenchmarkTask
from src.evaluator.evaluator import Evaluator
from src.sandbox.local_process import LocalProcessSandbox


def test_local_process_evaluator_end_to_end(tmp_path: Path):
    before = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [1, 2, 3]})
    after = pd.DataFrame({"x1": [2.0, 4.0, 6.0], "x2": [1, 4, 9]})
    paths = {}
    for name, frame in {
        "before": before,
        "after": after,
        "test_before": before,
        "test_after": after,
    }.items():
        paths[name] = tmp_path / f"{name}.csv"
        frame.to_csv(paths[name], index=False)
    task = BenchmarkTask(
        task_id="local-e2e",
        before_path=paths["before"],
        after_path=paths["after"],
        test_before_path=paths["test_before"],
        test_after_path=paths["test_after"],
        ground_truth_transformations=[
            Transformation(column="x1", operators=["multiply_2"]),
            Transformation(column="x2", operators=["square"]),
        ],
    )
    submission = FinalSubmission(
        script=(
            "def transform(df):\n"
            "    output = df.copy()\n"
            "    output['x1'] *= 2\n"
            "    output['x2'] **= 2\n"
            "    return output\n"
        ),
        transformations=task.ground_truth_transformations,
    )
    result = Evaluator(lambda files: LocalProcessSandbox(files)).evaluate(task, submission)
    assert result.execution_status == "success"
    assert result.functional_column_accuracy == 1.0
    assert result.operator_metrics.f1 == 1.0


def test_local_process_does_not_inherit_api_key(tmp_path: Path, monkeypatch):
    source = tmp_path / "visible.txt"
    source.write_text("visible", encoding="utf-8")
    monkeypatch.setenv("NVIDIA_API_KEY", "must-not-leak")
    sandbox = LocalProcessSandbox({"visible.txt": source})
    sandbox.start()
    try:
        result = sandbox.execute("import os\nprint(os.getenv('NVIDIA_API_KEY'))")
        assert result.status == "success"
        assert result.stdout.strip() == "None"
    finally:
        sandbox.close()
