import pandas as pd
import pytest

from run_pipeline import _build_report_data, _validate_source, parse_args


def test_pipeline_has_zero_argument_project_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_pipeline.py"])
    args = parse_args()
    assert args.source.as_posix() == "notebooks/processed_data.csv"
    assert args.model_config.as_posix() == "configs/nvidia.yaml"
    assert args.harness_config.as_posix() == "configs/harness.yaml"
    assert args.task_count == 24
    assert args.rows_per_task == 10_000
    assert args.workers is None


def test_pipeline_workers_cli_argument(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_pipeline.py", "--workers", "8"])
    args = parse_args()
    assert args.workers == 8



def test_pipeline_validates_numeric_complete_input():
    _validate_source(pd.DataFrame({"x": [1.0, 2.0], "y": [-1.0, 0.0]}))
    with pytest.raises(ValueError, match="numeric"):
        _validate_source(pd.DataFrame({"x": ["a", "b"]}))
    with pytest.raises(ValueError, match="missing"):
        _validate_source(pd.DataFrame({"x": [1.0, None]}))
    with pytest.raises(ValueError, match="infinite"):
        _validate_source(pd.DataFrame({"x": [1.0, float("inf")]}))


def test_pipeline_builds_aggregate_report_data():
    manifest = {
        "pipeline_run": "test",
        "source": "source.csv",
        "model": {"provider": "fake", "model": "fake-model"},
        "harness": {"sandbox": {"backend": "local"}},
    }
    summary = pd.DataFrame([{
        "status": "submitted", "functional_column_accuracy": 1.0,
        "operator_precision": 1.0, "operator_recall": 0.5,
        "operator_f1": 2 / 3, "exact_sequence_accuracy": 0.0,
        "total_tokens": 100, "tool_calls": 3, "cpu_time_seconds": 0.2,
    }])
    report = _build_report_data(manifest, summary, 1.5)
    assert report["mean_functional_column_accuracy"] == 1.0
    assert report["total_tokens"] == 100
    assert report["status_counts"] == {"submitted": 1}
