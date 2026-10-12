from pathlib import Path

import pandas as pd

from src.benchmark.task import load_task_directory, load_transformations


def test_load_existing_task_directory_without_split(tmp_path: Path):
    task_dir = tmp_path / "mock_0"
    task_dir.mkdir()
    df_before = pd.DataFrame({"col_3": [1.0, 2.0, 3.0], "col_4": [4.0, 5.0, 6.0]})
    df_after = pd.DataFrame({"col_3": [1.0, 4.0, 9.0], "col_4": [4.0, 5.0, 6.0]})
    df_before.to_csv(task_dir / "train_before.csv", index=False)
    df_after.to_csv(task_dir / "train_after.csv", index=False)
    (task_dir / "config.yaml").write_text("columns: col_3\noperator: SquareTransformation\n", encoding="utf-8")

    task = load_task_directory(task_dir)
    assert task.task_id == "mock_0"
    assert task.before_path == (task_dir / "train_before.csv").resolve()
    assert task.after_path == (task_dir / "train_after.csv").resolve()
    assert len(pd.read_csv(task.before_path)) == 3
    assert len(pd.read_csv(task.after_path)) == 3
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
