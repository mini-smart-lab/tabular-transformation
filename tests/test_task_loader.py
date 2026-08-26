from pathlib import Path

import pandas as pd

from src.benchmark.task import load_task_directory, load_transformations


def test_load_existing_task_directory_without_split():
    task = load_task_directory("data/0")
    assert task.task_id == "0"
    assert task.before_path == Path("data/0/train_before.csv").resolve()
    assert task.after_path == Path("data/0/train_after.csv").resolve()
    assert len(pd.read_csv(task.before_path)) == 10_000
    assert len(pd.read_csv(task.after_path)) == 10_000
    assert task.test_before_path is None
    assert task.test_after_path is None
    assert task.ground_truth_transformations[0].column == "col_3"
    assert task.ground_truth_transformations[0].operators == ["SquareTransformation"]


def test_load_structured_and_multi_column_configs(tmp_path: Path):
    structured = tmp_path / "structured.yaml"
    structured.write_text(
        "transformations:\n  - column: x\n    operators: [LogTransformation, SquareTransformation]\n",
        encoding="utf-8",
    )
    assert load_transformations(structured)[0].operators == [
        "LogTransformation", "SquareTransformation"
    ]

    legacy = tmp_path / "legacy.yaml"
    legacy.write_text("columns: [x, y]\noperator: SquareTransformation\n", encoding="utf-8")
    assert [item.column for item in load_transformations(legacy)] == ["x", "y"]
