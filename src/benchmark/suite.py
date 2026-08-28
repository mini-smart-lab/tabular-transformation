from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.benchmark.aggregate import aggregate_results
from src.benchmark.runner import BenchmarkRunner
from src.config import HarnessConfig, ModelConfig
from src.model_factory import create_model, resolve_model_config
from src.sandbox.docker import DockerSandbox
from src.sandbox.local_process import LocalProcessSandbox


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run every benchmark task under one directory")
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--harness-config", type=Path, default=Path("configs/harness.yaml"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--result-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    model_config = ModelConfig.model_validate(yaml.safe_load(args.model_config.read_text(encoding="utf-8")))
    model_config = resolve_model_config(model_config)
    harness = HarnessConfig.model_validate(yaml.safe_load(args.harness_config.read_text(encoding="utf-8")))
    sandbox_class = DockerSandbox if harness.sandbox.backend == "docker" else LocalProcessSandbox
    factory = lambda files: sandbox_class(files, harness.sandbox)
    runner = BenchmarkRunner(
        factory, factory, benchmark_config=harness.benchmark, sandbox_config=harness.sandbox,
        log_dir=args.log_dir, artifact_dir=args.artifact_dir, result_dir=args.result_dir,
        operator_vocabulary=harness.operator_vocabulary,
    )
    model = create_model(model_config)
    task_directories = sorted(
        (path.parent for path in args.task_root.glob("*/config.yaml")),
        key=lambda path: (not path.name.isdigit(), int(path.name) if path.name.isdigit() else path.name),
    )
    if not task_directories:
        raise SystemExit(f"No task directories found under {args.task_root}")
    for index, task_directory in enumerate(task_directories, start=1):
        print(f"[{index}/{len(task_directories)}] {task_directory.name}")
        result = runner.run_directory(
            task_directory, model, harness.budget,
            provider_model=f"{model_config.provider}/{model_config.model}",
        )
        print(f"  {result.status} ({result.run_id})")
    summary = args.result_dir / "summary.csv"
    aggregate_results(args.result_dir, summary)
    print(f"Summary: {summary.resolve()}")


if __name__ == "__main__":
    _main()
