import json

from src.benchmark.aggregate import aggregate_results


def test_aggregate_flattens_research_metrics(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    payload = {
        "run_id": "r1", "task_id": "t1", "status": "submitted", "termination_reason": "submitted",
        "budget": {"total_tokens": 12, "tool_calls": 3},
        "evaluation": {
            "functional_column_accuracy": 0.5,
            "operator_metrics": {"precision": 1.0, "recall": 0.5, "f1": 2 / 3, "exact_sequence_accuracy": 0.0},
        },
        "transform_path": "transform.py",
    }
    (results / "r1.json").write_text(json.dumps(payload), encoding="utf-8")
    frame = aggregate_results(results, tmp_path / "summary.csv")
    assert len(frame) == 1
    assert frame.loc[0, "functional_column_accuracy"] == 0.5
    assert frame.loc[0, "operator_f1"] == 2 / 3
    assert (tmp_path / "summary.csv").is_file()
