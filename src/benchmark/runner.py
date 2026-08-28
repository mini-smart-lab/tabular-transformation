from __future__ import annotations

import argparse
import json
import logging
import uuid
from collections.abc import Callable
from pathlib import Path

import yaml

from src.agent.budget import BudgetManager
from src.agent.graph import DiscoveryGraph, normalize_terminal_state
from src.benchmark.result import BenchmarkResult
from src.benchmark.task import BenchmarkTask, load_task_directory
from src.config import (
    AgentBudgetConfig,
    BenchmarkConfig,
    HarnessConfig,
    ModelConfig,
    SandboxConfig,
)
from src.evaluator.evaluator import Evaluator
from src.logging_utils.episode_logger import EpisodeLogger
from src.model_factory import create_model, resolve_model_config
from src.sandbox.base import Sandbox
from src.sandbox.local_process import LocalProcessSandbox
from src.sandbox.docker import DockerSandbox


class BenchmarkRunner:
    def __init__(
        self,
        discovery_sandbox_factory: Callable[[dict[str, Path]], Sandbox],
        evaluation_sandbox_factory: Callable[[dict[str, Path]], Sandbox],
        *,
        benchmark_config: BenchmarkConfig | None = None,
        sandbox_config: SandboxConfig | None = None,
        log_dir: Path = Path("logs"),
        artifact_dir: Path = Path("artifacts"),
        result_dir: Path = Path("results"),
        operator_vocabulary: list[str] | None = None,
    ) -> None:
        self.discovery_sandbox_factory = discovery_sandbox_factory
        self.evaluation_sandbox_factory = evaluation_sandbox_factory
        self.benchmark_config = benchmark_config or BenchmarkConfig()
        self.sandbox_config = sandbox_config or SandboxConfig()
        self.log_dir = log_dir
        self.artifact_dir = artifact_dir
        self.result_dir = result_dir
        self.operator_vocabulary = operator_vocabulary or []

    def run(
        self,
        task: BenchmarkTask,
        model,
        budget: AgentBudgetConfig | None = None,
        *,
        provider_model: str = "unknown",
    ) -> BenchmarkResult:
        budget_manager = BudgetManager(budget or AgentBudgetConfig())
        run_id = uuid.uuid4().hex
        logger = EpisodeLogger(self.log_dir / f"{task.task_id}-{run_id}.jsonl")
        logger.record("episode_start", run_id=run_id, task_id=task.task_id, provider_model=provider_model)
        discovery = self.discovery_sandbox_factory(task.discovery_files())
        state = None
        discovery.start()
        try:
            episode = DiscoveryGraph(
                model,
                discovery,
                budget_manager,
                logger,
                sandbox_config=self.sandbox_config,
                operator_vocabulary=self.operator_vocabulary,
                provider_model=provider_model,
            )
            logger.record(
                "discovery_initialized",
                budget=budget_manager.config.model_dump(),
                sandbox=self.sandbox_config.model_dump(),
                operator_vocabulary=self.operator_vocabulary,
            )
            try:
                state = normalize_terminal_state(episode.invoke())
            except Exception as exc:
                logger.record("discovery_error", error_type=type(exc).__name__, error=str(exc))
                state = episode.initial_state()
                state["status"] = "failed"
                state["termination_reason"] = f"discovery_error:{type(exc).__name__}"
                state.update(episode._counter_update())
        finally:
            discovery.close()

        evaluation = None
        has_hidden_test = task.test_before_path is not None and task.test_after_path is not None
        if state["final_submission"] is not None and has_hidden_test:
            evaluator = Evaluator(self.evaluation_sandbox_factory, self.benchmark_config)
            evaluation = evaluator.evaluate(task, state["final_submission"])
            logger.record("evaluation_complete", **evaluation.model_dump())
        transform_path = None
        if state["final_submission"] is not None:
            artifact = self.artifact_dir / run_id / "transform.py"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(state["final_submission"].script.rstrip() + "\n", encoding="utf-8")
            transform_path = str(artifact.resolve())
        result = BenchmarkResult(
            run_id=run_id,
            task_id=task.task_id,
            status=state["status"],
            termination_reason=state["termination_reason"],
            submission=state["final_submission"],
            budget=budget_manager.snapshot(),
            evaluation=evaluation,
            transform_path=transform_path,
        )
        logger.record("episode_complete", **result.model_dump())
        self.result_dir.mkdir(parents=True, exist_ok=True)
        (self.result_dir / f"{run_id}.json").write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result

    def run_directory(
        self,
        task_directory: Path | str,
        model,
        budget: AgentBudgetConfig | None = None,
        *,
        test_before_path: Path | str | None = None,
        test_after_path: Path | str | None = None,
        provider_model: str = "unknown",
    ) -> BenchmarkResult:
        task = load_task_directory(
            task_directory,
            test_before_path=test_before_path,
            test_after_path=test_after_path,
        )
        return self.run(task, model, budget, provider_model=provider_model)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run one tabular transformation benchmark task")
    parser.add_argument(
        "--task",
        type=Path,
        required=True,
        help="Task directory containing train_before.csv, train_after.csv, and config.yaml",
    )
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--harness-config", type=Path, default=Path("configs/harness.yaml"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--result-dir", type=Path, default=Path("results"))
    parser.add_argument("--sandbox-backend", choices=["local", "docker"])
    parser.add_argument("--docker-image")
    parser.add_argument("--test-before", type=Path, help="Optional external hidden input CSV")
    parser.add_argument("--test-after", type=Path, help="Optional external hidden expected CSV")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    model_config = ModelConfig.model_validate(yaml.safe_load(args.model_config.read_text(encoding="utf-8")))
    model_config = resolve_model_config(model_config)
    harness_config = HarnessConfig.model_validate(
        yaml.safe_load(args.harness_config.read_text(encoding="utf-8")) if args.harness_config.is_file() else {}
    )
    updates = {}
    if args.sandbox_backend:
        updates["backend"] = args.sandbox_backend
    if args.docker_image:
        updates["docker_image"] = args.docker_image
    sandbox_config = harness_config.sandbox.model_copy(update=updates)
    sandbox_class = DockerSandbox if sandbox_config.backend == "docker" else LocalProcessSandbox
    sandbox_factory = lambda files: sandbox_class(files, sandbox_config)
    runner = BenchmarkRunner(
        sandbox_factory,
        sandbox_factory,
        sandbox_config=sandbox_config,
        log_dir=args.log_dir,
        artifact_dir=args.artifact_dir,
        result_dir=args.result_dir,
        benchmark_config=harness_config.benchmark,
        operator_vocabulary=harness_config.operator_vocabulary,
    )
    result = runner.run_directory(
        args.task,
        create_model(model_config), harness_config.budget,
        test_before_path=args.test_before,
        test_after_path=args.test_after,
        provider_model=f"{model_config.provider}/{model_config.model}",
    )
    print(f"Task: {result.task_id}")
    print(f"Status: {result.status}")
    print(f"Run ID: {result.run_id}")
    print(f"Result JSON: {(args.result_dir / (result.run_id + '.json')).resolve()}")
    if result.transform_path:
        print(f"Transform: {result.transform_path}")
    print(f"Tool calls: {result.budget['tool_calls']}")
    print(f"Tokens: {result.budget['total_tokens']}")
    print(f"Sandbox execution: {result.budget['execution_time_seconds']:.2f} sec")
    print(f"Sandbox CPU: {result.budget['cpu_time_seconds']:.2f} sec")
    if result.evaluation:
        metrics = result.evaluation.operator_metrics
        print(f"Functional column accuracy: {result.evaluation.functional_column_accuracy:.2f}")
        print(f"Operator precision: {metrics.precision:.2f}")
        print(f"Operator recall: {metrics.recall:.2f}")
        print(f"Operator F1: {metrics.f1:.2f}")
        print(f"Exact sequence accuracy: {metrics.exact_sequence_accuracy:.2f}")


if __name__ == "__main__":
    _main()
