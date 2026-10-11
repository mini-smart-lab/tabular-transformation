from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def aggregate_results(result_directory: Path | str, output_csv: Path | str) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(result_directory).glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        evaluation = result.get("evaluation") or {}
        operators = evaluation.get("operator_metrics") or {}
        budget = result.get("budget") or {}
        rows.append({
            "run_id": result.get("run_id"), "task_id": result.get("task_id"),
            "level": result.get("level"), "source_dataset": result.get("source_dataset"),
            "domain": result.get("domain"),
            "status": result.get("status"), "termination_reason": result.get("termination_reason"),
            "functional_column_accuracy": evaluation.get("functional_column_accuracy"),
            "transformed_column_accuracy": evaluation.get("transformed_column_accuracy"),
            "operator_precision": operators.get("precision"),
            "operator_recall": operators.get("recall"), "operator_f1": operators.get("f1"),
            "exact_sequence_accuracy": operators.get("exact_sequence_accuracy"),
            "input_tokens": budget.get("input_tokens"), "output_tokens": budget.get("output_tokens"),
            "total_tokens": budget.get("total_tokens"), "tool_calls": budget.get("tool_calls"),
            "execution_time_seconds": budget.get("execution_time_seconds"),
            "cpu_time_seconds": budget.get("cpu_time_seconds"),
            "wall_time_seconds": budget.get("wall_time_seconds"),
            "result_path": str(path.resolve()), "transform_path": result.get("transform_path"),
        })
    frame = pd.DataFrame(rows)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def _main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate benchmark result JSON files")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/summary.csv"))
    args = parser.parse_args()
    frame = aggregate_results(args.results, args.output)
    print(f"Wrote {len(frame)} runs to {args.output.resolve()}")


if __name__ == "__main__":
    _main()
