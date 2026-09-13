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
    parser.add_argument(
        "--workers", "--max-workers", type=int, default=None, dest="workers",
        help="Number of concurrent worker threads (default: from harness config benchmark.max_workers)",
    )
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
    task_directories = sorted(
        (path.parent for path in args.task_root.glob("*/config.yaml")),
        key=lambda path: (not path.name.isdigit(), int(path.name) if path.name.isdigit() else path.name),
    )
    if not task_directories:
        raise SystemExit(f"No task directories found under {args.task_root}")

    workers = max(1, args.workers if args.workers is not None else harness.benchmark.max_workers)
    print(f"Executing {len(task_directories)} tasks using {workers} concurrent worker(s)...")

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    lock = threading.Lock()

    def _run_task(index: int, task_dir: Path):
        task_model = create_model(model_config)
        res = runner.run_directory(
            task_dir, task_model, harness.budget,
            provider_model=f"{model_config.provider}/{model_config.model}",
        )
        with lock:
            print(f"[{index}/{len(task_directories)}] {task_dir.name}: {res.status} ({res.run_id})")
        return res

    if workers == 1:
        for index, task_directory in enumerate(task_directories, start=1):
            _run_task(index, task_directory)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_run_task, idx, tdir)
                for idx, tdir in enumerate(task_directories, start=1)
            ]
            for future in as_completed(futures):
                future.result()

    summary = args.result_dir / "summary.csv"
    aggregate_results(args.result_dir, summary)
    print(f"Summary: {summary.resolve()}")


if __name__ == "__main__":
    _main()
