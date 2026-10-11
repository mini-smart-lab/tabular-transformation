"""Unit tests for multi-dataset and multi-level task generation and pipeline."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from src.benchmark.task import load_task_directory
from src.benchmark.task_builder import build_multidataset_suite
from src.benchmark.aggregate import aggregate_results
from run_pipeline import parse_args


def test_build_multidataset_suite_balances_datasets_and_levels(tmp_path: Path):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    tasks_dir = tmp_path / "tasks"

    # Create 2 small source datasets
    df_a = pd.DataFrame({"a1": np.arange(1.0, 50.0), "a2": np.arange(2.0, 51.0)})
    df_b = pd.DataFrame({"b1": np.arange(10.0, 60.0), "b2": np.arange(20.0, 70.0)})
    df_a.to_csv(source_dir / "dataset_a.csv", index=False)
    df_b.to_csv(source_dir / "dataset_b.csv", index=False)

    manifest_payload = {
        "datasets": {
            "dataset_a": {"domain": "Economics"},
            "dataset_b": {"domain": "Healthcare"},
        }
    }
    (source_dir / "dataset_manifest.json").write_text(json.dumps(manifest_payload), encoding="utf-8")

    tasks = build_multidataset_suite(
        source_input=source_dir,
        output_directory=tasks_dir,
        levels=[1, 2],
        task_count=4,
        seed=100,
        train_fraction=0.7,
    )

    assert len(tasks) == 4
    datasets_assigned = [t["source_dataset"] for t in tasks]
    levels_assigned = [t["level"] for t in tasks]
    domains_assigned = [t["domain"] for t in tasks]

    # Balanced distribution: 2 tasks each
    assert datasets_assigned.count("dataset_a") == 2
    assert datasets_assigned.count("dataset_b") == 2
    assert levels_assigned.count(1) == 2
    assert levels_assigned.count(2) == 2
    assert domains_assigned.count("Economics") == 2
    assert domains_assigned.count("Healthcare") == 2

    # Check task 0 structure
    task0 = load_task_directory(tasks_dir / "0")
    assert task0.source_dataset in {"dataset_a", "dataset_b"}
    assert task0.level in {1, 2}
    assert task0.domain in {"Economics", "Healthcare"}
    assert task0.before_path.is_file()
    assert task0.after_path.is_file()
    assert task0.test_before_path.is_file()
    assert task0.test_after_path.is_file()


def test_build_multidataset_suite_filter(tmp_path: Path):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    tasks_dir = tmp_path / "tasks"

    df_a = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [5.0, 6.0, 7.0, 8.0]})
    df_b = pd.DataFrame({"p": [10.0, 20.0, 30.0, 40.0], "q": [50.0, 60.0, 70.0, 80.0]})
    df_a.to_csv(source_dir / "alpha.csv", index=False)
    df_b.to_csv(source_dir / "beta.csv", index=False)

    tasks = build_multidataset_suite(
        source_input=source_dir,
        output_directory=tasks_dir,
        levels=[1],
        task_count=2,
        dataset_filter=["alpha"],
    )

    assert len(tasks) == 2
    assert all(t["source_dataset"] == "alpha" for t in tasks)


def test_aggregate_results_with_metadata_fields(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    summary_csv = tmp_path / "summary.csv"

    sample_result = {
        "run_id": "run-001",
        "task_id": "0",
        "level": 2,
        "source_dataset": "california_housing",
        "domain": "Economics",
        "status": "submitted",
        "termination_reason": "solution_submitted",
        "budget": {"total_tokens": 1500, "tool_calls": 4},
        "evaluation": {
            "functional_column_accuracy": 0.95,
            "transformed_column_accuracy": 0.90,
            "operator_metrics": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_sequence_accuracy": 1.0},
        },
    }
    (results_dir / "run-001.json").write_text(json.dumps(sample_result), encoding="utf-8")

    df = aggregate_results(results_dir, summary_csv)
    assert len(df) == 1
    assert df.loc[0, "level"] == 2
    assert df.loc[0, "source_dataset"] == "california_housing"
    assert df.loc[0, "domain"] == "Economics"
    assert df.loc[0, "functional_column_accuracy"] == 0.95


def test_pipeline_cli_multi_dataset_args(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pipeline.py",
            "--source-dir", "data/source_datasets",
            "--levels", "0", "1", "2",
            "--datasets", "diabetes", "wine_quality",
        ],
    )
    args = parse_args()
    assert args.source_dir == Path("data/source_datasets")
    assert args.levels == [0, 1, 2]
    assert args.datasets == ["diabetes", "wine_quality"]
