r"""Run the complete benchmark with project defaults: ``python run_pipeline.py``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
import numpy as np
import yaml

from src.benchmark.aggregate import aggregate_results
from src.benchmark.runner import BenchmarkRunner
from src.benchmark.task_builder import ALL_OPERATORS, build_task
from src.config import HarnessConfig, ModelConfig, get_level_budget
from src.model_factory import create_model, resolve_model_config
from src.sandbox.docker import DockerSandbox
from src.sandbox.local_process import LocalProcessSandbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate valid tasks, run discovery/evaluation, and export a research report."
    )
    parser.add_argument(
        "--source", type=Path, default=Path("notebooks/processed_data.csv"),
        help="Input numeric CSV (default: notebooks/processed_data.csv)",
    )
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/nvidia.yaml"),
        help="Model configuration (default: configs/nvidia.yaml)",
    )
    parser.add_argument("--harness-config", type=Path, default=Path("configs/harness.yaml"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("pipeline_runs"),
        help="Parent directory for isolated pipeline runs",
    )
    parser.add_argument("--run-name", help="Stable run name; default is a UTC timestamp")
    parser.add_argument("--task-count", type=int, default=24)
    parser.add_argument(
        "--level", type=int, default=1, choices=[0, 1, 2, 3, 4, 5],
        help="Difficulty level (0=atomic, 1=atomic unary, 2=composed unary, 3=interactions, 4=conditional if/else, 5=hybrid deep AST; default: 1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for task generation, data sampling, and model reproducibility (default: 42)",
    )
    parser.add_argument("--train-fraction", type=float, default=0.7)

    parser.add_argument(
        "--rows-per-task", type=int, default=10_000,
        help="Deterministically sample this many source rows per task; 0 uses all rows",
    )
    parser.add_argument(
        "--operators", nargs="+", choices=sorted(ALL_OPERATORS), default=sorted(ALL_OPERATORS)
    )
    parser.add_argument(
        "--fixed-budget", action="store_true",
        help="Use fixed budget from harness config instead of level-specific adaptive budget",
    )
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Validate input and generate tasks without creating an API client",
    )
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    started = time.monotonic()
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.task_count <= 0:
        raise ValueError("task-count must be positive")
    if args.rows_per_task < 0:
        raise ValueError("rows-per-task cannot be negative")

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = (args.output_root / run_name).resolve()
    if run_directory.exists() and any(run_directory.iterdir()):
        raise FileExistsError(
            f"run directory is not empty: {run_directory}; choose another --run-name"
        )

    paths = {
        "tasks": run_directory / "tasks",
        "logs": run_directory / "logs",
        "artifacts": run_directory / "artifacts",
        "results": run_directory / "results",
        "reports": run_directory / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    source_frame = pd.read_csv(source)
    _validate_source(source_frame)
    generated = []
    for index in range(args.task_count):
        task_directory = paths["tasks"] / str(index)
        task_frame = source_frame
        if args.rows_per_task and len(source_frame) > args.rows_per_task:
            task_frame = source_frame.sample(
                n=args.rows_per_task, random_state=args.seed + index, replace=False
            ).reset_index(drop=True)
        transforms = build_task(
            task_frame,
            task_directory,
            level=args.level,
            seed=args.seed + index,
            train_fraction=args.train_fraction,
            operator_names=args.operators,
        )
        generated.append({
            "task_id": str(index),
            "seed": args.seed + index,
            "level": args.level,
            "transformations": transforms,
        })

    model_config_raw = yaml.safe_load(args.model_config.read_text(encoding="utf-8")) or {}
    if args.seed is not None and "seed" not in model_config_raw:
        model_config_raw["seed"] = args.seed
    model_config = ModelConfig.model_validate(model_config_raw)
    model_config = resolve_model_config(model_config)

    harness_config = HarnessConfig.model_validate(
        yaml.safe_load(args.harness_config.read_text(encoding="utf-8"))
    )
    if not args.fixed_budget:
        harness_config.budget = get_level_budget(args.level, harness_config.budget)

    manifest: dict[str, Any] = {
        "pipeline_run": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_rows": len(source_frame),
        "source_columns": list(source_frame.columns),
        "task_count": args.task_count,
        "level": args.level,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "rows_per_task": min(args.rows_per_task or len(source_frame), len(source_frame)),
        "operators": args.operators,
        "model": model_config.model_dump(exclude={"api_key_env"}),
        "api_key_env": model_config.api_key_env,
        "harness": harness_config.model_dump(),
        "tasks": generated,
        "status": "prepared" if args.prepare_only else "running",
    }
    manifest_path = run_directory / "manifest.json"
    _write_json(manifest_path, manifest)

    print(f"Prepared {args.task_count} valid Level {args.level} tasks in {paths['tasks']}")
    if args.prepare_only:
        _write_preparation_report(paths["reports"] / "report.md", manifest)
        print(f"Manifest: {manifest_path}")
        print(f"Report: {paths['reports'] / 'report.md'}")
        return 0

    sandbox_class = (
        DockerSandbox if harness_config.sandbox.backend == "docker" else LocalProcessSandbox
    )
    sandbox_factory = lambda files: sandbox_class(files, harness_config.sandbox)
    runner = BenchmarkRunner(
        sandbox_factory,
        sandbox_factory,
        benchmark_config=harness_config.benchmark,
        sandbox_config=harness_config.sandbox,
        log_dir=paths["logs"],
        artifact_dir=paths["artifacts"],
        result_dir=paths["results"],
        operator_vocabulary=harness_config.operator_vocabulary,
    )
    model = create_model(model_config)
    results = []
    for index in range(args.task_count):
        task_directory = paths["tasks"] / str(index)
        print(f"[{index + 1}/{args.task_count}] Running task {index + 1}...")
        result = runner.run_directory(
            task_directory,
            model,
            harness_config.budget,
            provider_model=f"{model_config.provider}/{model_config.model}",
        )
        results.append(result)
        trans_acc = (
            result.evaluation.transformed_column_accuracy
            if result.evaluation and result.evaluation.transformed_column_accuracy is not None
            else (result.evaluation.functional_column_accuracy if result.evaluation else None)
        )
        metric = (
            f"trans_acc={trans_acc:.3f} (all={result.evaluation.functional_column_accuracy:.3f})"
            if result.evaluation else "no evaluation"
        )
        print(f"  {result.status}; {metric}; run_id={result.run_id}")

    summary_path = paths["reports"] / "summary.csv"
    summary = aggregate_results(paths["results"], summary_path)
    report_path = paths["reports"] / "report.md"
    report_json_path = paths["reports"] / "report.json"
    report = _build_report_data(manifest, summary, time.monotonic() - started)
    _write_json(report_json_path, report)
    report_path.write_text(_render_markdown_report(report, summary), encoding="utf-8")
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["successful_submissions"] = sum(r.submission is not None for r in results)
    _write_json(manifest_path, manifest)

    print("\nPipeline complete")
    print(f"Summary CSV: {summary_path}")
    print(f"Report: {report_path}")
    print(f"Machine-readable report: {report_json_path}")
    print(f"Raw results: {paths['results']}")
    return 0


def _validate_source(frame: pd.DataFrame) -> None:
    if len(frame) < 2:
        raise ValueError("source CSV must contain at least two rows")
    if frame.columns.empty:
        raise ValueError("source CSV has no columns")
    duplicate_columns = frame.columns[frame.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(f"duplicate source columns: {duplicate_columns}")
    non_numeric = [c for c in frame if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise ValueError(f"all source columns must be numeric: {non_numeric}")
    if frame.isna().any().any():
        raise ValueError("source CSV contains missing values")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("source CSV contains infinite values")


def _build_report_data(
    manifest: dict[str, Any], summary: pd.DataFrame, elapsed_seconds: float
) -> dict[str, Any]:
    task_count = len(summary)
    submitted_mask = summary["status"] == "submitted"
    evaluated = summary[submitted_mask]

    # 1. Benchmark scores across ALL tasks (Failed/Timeout tasks = 0.0)
    has_trans = "transformed_column_accuracy" in summary.columns
    trans_acc_series = summary["transformed_column_accuracy"] if has_trans else summary["functional_column_accuracy"]
    all_trans_acc = float(trans_acc_series.fillna(0.0).mean()) if task_count else 0.0
    all_func_acc = float(summary["functional_column_accuracy"].fillna(0.0).mean()) if task_count else 0.0
    all_op_f1 = float(summary["operator_f1"].fillna(0.0).mean()) if task_count else 0.0
    all_exact_seq = float(summary["exact_sequence_accuracy"].fillna(0.0).mean()) if task_count else 0.0

    # 2. Conditional scores on SUBMITTED tasks only
    submitted_trans_acc = _safe_mean(evaluated, "transformed_column_accuracy")
    if submitted_trans_acc is None:
        submitted_trans_acc = _safe_mean(evaluated, "functional_column_accuracy")

    return {
        "pipeline_run": manifest["pipeline_run"],
        "source": manifest["source"],
        "level": manifest.get("level", 1),
        "seed": manifest.get("seed", 42),
        "model": manifest["model"],
        "sandbox_backend": manifest["harness"]["sandbox"]["backend"],
        "task_count": task_count,
        "evaluated_task_count": int(len(evaluated)),
        "submitted_task_count": int(submitted_mask.sum()),
        "submission_rate": float(submitted_mask.sum() / task_count) if task_count else 0.0,
        # All-task benchmark metrics (penalized with 0.0 for failures)
        "benchmark_transformed_accuracy": all_trans_acc,
        "benchmark_functional_accuracy": all_func_acc,
        "benchmark_operator_f1": all_op_f1,
        "benchmark_exact_sequence_accuracy": all_exact_seq,
        # Backward-compatible keys
        "mean_transformed_column_accuracy": all_trans_acc,
        "mean_functional_column_accuracy": all_func_acc,
        "mean_operator_precision": _safe_mean(evaluated, "operator_precision"),
        "mean_operator_recall": _safe_mean(evaluated, "operator_recall"),
        "mean_operator_f1": all_op_f1,
        "mean_exact_sequence_accuracy": all_exact_seq,
        # Submitted-only metrics
        "submitted_transformed_accuracy": submitted_trans_acc,
        "submitted_functional_accuracy": _safe_mean(evaluated, "functional_column_accuracy"),
        "submitted_operator_precision": _safe_mean(evaluated, "operator_precision"),
        "submitted_operator_recall": _safe_mean(evaluated, "operator_recall"),
        "submitted_operator_f1": _safe_mean(evaluated, "operator_f1"),
        "submitted_exact_sequence_accuracy": _safe_mean(evaluated, "exact_sequence_accuracy"),
        "total_tokens": int(summary["total_tokens"].fillna(0).sum()),
        "total_tool_calls": int(summary["tool_calls"].fillna(0).sum()),
        "total_sandbox_cpu_seconds": float(summary["cpu_time_seconds"].fillna(0).sum()),
        "pipeline_wall_seconds": elapsed_seconds,
        "status_counts": {
            str(key): int(value) for key, value in summary["status"].value_counts().items()
        },
    }


def _safe_mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = [float(value) for value in frame[column].dropna().tolist()]
    return mean(values) if values else None


def _render_markdown_report(report: dict[str, Any], summary: pd.DataFrame) -> str:
    def metric(value: Any) -> str:
        return "N/A" if value is None or pd.isna(value) else f"{float(value):.4f}"

    def percent(value: Any) -> str:
        return "N/A" if value is None or pd.isna(value) else f"{float(value) * 100:.2f}%"

    def integer(value: Any) -> int:
        return 0 if value is None or pd.isna(value) else int(value)

    lines = [
        "# Tabular Transformation Benchmark Report",
        "",
        f"- Pipeline run: `{report['pipeline_run']}`",
        f"- Source: `{report['source']}`",
        f"- Level: {report.get('level', 1)}",
        f"- Seed: {report.get('seed', 42)}",
        f"- Model: `{report['model']['provider']}/{report['model']['model']}`",
        f"- Sandbox: `{report['sandbox_backend']}`",
        f"- Tasks: {report['task_count']} ({report['submitted_task_count']} submitted, {report['submission_rate'] * 100:.1f}% completion rate)",
        f"- Pipeline wall time: {report['pipeline_wall_seconds']:.2f} seconds",
        "",
        "## Benchmark Scores (All Tasks — Failed Tasks Penalized with 0.0)",
        "",
        "| Primary Metric | Score | Note |",
        "|---|---:|---|",
        f"| **Benchmark Transformed Accuracy** | **{metric(report.get('benchmark_transformed_accuracy'))}** | *Target columns only, failed tasks = 0.0 (Primary Score)* |",
        f"| **Benchmark Functional Accuracy** | {metric(report.get('benchmark_functional_accuracy'))} | *All columns, failed tasks = 0.0* |",
        f"| **Benchmark Operator F1** | {metric(report.get('benchmark_operator_f1'))} | *Symbolic recovery, failed tasks = 0.0* |",
        f"| **Benchmark Exact Sequence Acc** | {metric(report.get('benchmark_exact_sequence_accuracy'))} | *Exact chain match, failed tasks = 0.0* |",
        f"| **Task Submission Rate** | {percent(report.get('submission_rate'))} | *{report['submitted_task_count']}/{report['task_count']} tasks completed within budget* |",
        "",
        "## Conditional Metrics (Submitted Tasks Only)",
        "",
        "| Conditional Metric | Value |",
        "|---|---:|",
        f"| Transformed column accuracy | {metric(report.get('submitted_transformed_accuracy'))} |",
        f"| Functional column accuracy (overall) | {metric(report.get('submitted_functional_accuracy'))} |",
        f"| Operator precision | {metric(report.get('submitted_operator_precision'))} |",
        f"| Operator recall | {metric(report.get('submitted_operator_recall'))} |",
        f"| Operator F1 | {metric(report.get('submitted_operator_f1'))} |",
        f"| Exact operator sequence accuracy | {metric(report.get('submitted_exact_sequence_accuracy'))} |",
        f"| Total tokens | {report['total_tokens']} |",
        f"| Total tool calls | {report['total_tool_calls']} |",
        f"| Sandbox CPU seconds | {report['total_sandbox_cpu_seconds']:.3f} |",
        "",
        "## Per-task results",
        "",
        "| Task | Status | Transformed acc | Overall acc | Operator F1 | Tokens | Tool calls |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    has_trans = "transformed_column_accuracy" in summary.columns
    for _, row in summary.sort_values("task_id").iterrows():
        is_sub = row["status"] == "submitted"
        trans_acc = row["transformed_column_accuracy"] if has_trans else row["functional_column_accuracy"]
        trans_val = metric(trans_acc) if is_sub else "0.0000 (failed)"
        overall_val = metric(row["functional_column_accuracy"]) if is_sub else "0.0000 (failed)"
        f1_val = metric(row["operator_f1"]) if is_sub else "0.0000 (failed)"
        lines.append(
            f"| {row['task_id']} | {row['status']} | "
            f"{trans_val} | "
            f"{overall_val} | "
            f"{f1_val} | "
            f"{integer(row['total_tokens'])} | {integer(row['tool_calls'])} |"
        )
    lines.extend([
        "", "## Status counts", "",
        *[f"- {status}: {count}" for status, count in report["status_counts"].items()],
        "",
    ])
    return "\n".join(lines)


def _write_preparation_report(path: Path, manifest: dict[str, Any]) -> None:
    operator_counts: dict[str, int] = {}
    for task in manifest["tasks"]:
        for t in task.get("transformations", []):
            for op in t.get("operators", []):
                operator_counts[op] = operator_counts.get(op, 0) + 1
    lines = [
        "# Prepared Benchmark Dataset", "",
        f"- Source: `{manifest['source']}`",
        f"- Level: {manifest.get('level', 1)}",
        f"- Seed: {manifest.get('seed', 42)}",
        f"- Rows: {manifest['source_rows']}",
        f"- Columns: {len(manifest['source_columns'])}",
        f"- Generated tasks: {manifest['task_count']}",
        f"- Train fraction: {manifest['train_fraction']}",
        f"- Rows per task: {manifest['rows_per_task']}",
        "", "## Operator distribution", "",
        *[f"- {name}: {count}" for name, count in sorted(operator_counts.items())],
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Pipeline interrupted by user", file=sys.stderr)
        raise SystemExit(130)
