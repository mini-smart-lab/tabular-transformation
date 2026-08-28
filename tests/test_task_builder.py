from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.benchmark.task import load_task_directory
from src.benchmark.task_builder import applicable_pairs, build_atomic_task


def test_applicability_excludes_invalid_domains():
    frame = pd.DataFrame({"negative": [-4.0, -1.0], "zero": [0.0, 1.0], "positive": [1.0, 4.0]})
    pairs = set(applicable_pairs(frame))
    assert ("negative", "LogTransformation") not in pairs
    assert ("negative", "SquareRootTransformation") not in pairs
    assert ("zero", "LogTransformation") not in pairs
    assert ("zero", "SquareRootTransformation") in pairs
    assert ("positive", "LogTransformation") in pairs


def test_builder_creates_valid_hidden_task(tmp_path: Path):
    source = tmp_path / "source.csv"
    pd.DataFrame({"x": np.arange(1.0, 21.0), "y": np.arange(-10.0, 10.0)}).to_csv(source, index=False)
    output = tmp_path / "task"
    column, operator = build_atomic_task(
        source, output, seed=4, operator_names=["LogTransformation", "SquareRootTransformation"]
    )
    task = load_task_directory(output)
    assert task.test_before_path == (output / "hidden" / "test_before.csv").resolve()
    assert task.test_after_path == (output / "hidden" / "test_after.csv").resolve()
    assert task.ground_truth_transformations[0].column == column
    assert task.ground_truth_transformations[0].operators == [operator]
    assert set(task.discovery_files()) == {"before.csv", "after.csv"}
    assert set(task.evaluation_files()) == {"test_before.csv"}


def test_builder_rejects_impossible_operator(tmp_path: Path):
    source = tmp_path / "source.csv"
    pd.DataFrame({"x": [-2.0, -1.0]}).to_csv(source, index=False)
    with pytest.raises(ValueError, match="no requested operator"):
        build_atomic_task(source, tmp_path / "task", operator_names=["LogTransformation"])
