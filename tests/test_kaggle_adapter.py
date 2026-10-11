"""Tests for Kaggle Benchmarks adapter and exporter."""

import ast
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.benchmark.task_builder import build_task
from src.kaggle_adapter.exporter import export_suite_to_kbench, export_task_to_kbench, slugify_name


def test_slugify_name():
    assert slugify_name("My Task Name!") == "my-task-name"
    assert slugify_name("California_Housing_L1") == "california-housing-l1"


def test_export_task_to_kbench(tmp_path: Path):
    source_csv = tmp_path / "source.csv"
    pd.DataFrame({"x": np.arange(1.0, 31.0), "y": np.arange(10.0, 40.0)}).to_csv(source_csv, index=False)
    task_dir = tmp_path / "task_0"
    build_task(source_csv, task_dir, level=1, seed=42, source_dataset="test_ds")

    output_py = tmp_path / "kbench_task.py"
    res_path = export_task_to_kbench(task_dir, output_py, task_name="test-tab-task")
    assert res_path.is_file()

    content = res_path.read_text(encoding="utf-8")
    assert "@kbench.task(name=\"test-tab-task\")" in content
    assert ".run(kbench.llm)" in content
    assert "# %%" in content

    # Verify python syntax is 100% valid
    parsed = ast.parse(content)
    assert isinstance(parsed, ast.Module)


def test_export_suite_to_kbench(tmp_path: Path):
    tasks_dir = tmp_path / "tasks"
    source_csv = tmp_path / "source.csv"
    pd.DataFrame({"x": np.arange(1.0, 20.0), "y": np.arange(2.0, 21.0)}).to_csv(source_csv, index=False)

    for i in range(2):
        t_dir = tasks_dir / str(i)
        build_task(source_csv, t_dir, level=1, seed=i)

    output_dir = tmp_path / "exported_kbench"
    exported = export_suite_to_kbench(tasks_dir, output_dir, prefix="custom-bench")
    assert len(exported) == 2
    for exp in exported:
        assert exp.is_file()
        ast.parse(exp.read_text(encoding="utf-8"))
