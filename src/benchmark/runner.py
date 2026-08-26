from __future__ import annotations

import argparse
import logging
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
    ModelConfig,
    SandboxConfig,
)
from src.evaluator.evaluator import Evaluator
from src.logging_utils.episode_logger import EpisodeLogger
from src.model_factory import create_model
from src.sandbox.base import Sandbox
from src.sandbox.local_process import LocalProcessSandbox


class BenchmarkRunner:
    def __init__(
        self,
        discovery_sandbox_factory: Callable[[dict[str, Path]], Sandbox],
        evaluation_sandbox_factory: Callable[[dict[str, Path]], Sandbox],
        *,
        benchmark_config: BenchmarkConfig | None = None,
        sandbox_config: SandboxConfig | None = None,
        log_dir: Path = Path("logs"),
        operator_vocabulary: list[str] | None = None,
    ) -> None:
        self.discovery_sandbox_factory = discovery_sandbox_factory
        self.evaluation_sandbox_factory = evaluation_sandbox_factory
        self.benchmark_config = benchmark_config or BenchmarkConfig()
        self.sandbox_config = sandbox_config or SandboxConfig()
        self.log_dir = log_dir
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
        logger = EpisodeLogger(self.log_dir / f"{task.task_id}.jsonl")
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
            state = normalize_terminal_state(episode.invoke())
        finally:
            discovery.close()

        evaluation = None
        has_hidden_test = task.test_before_path is not None and task.test_after_path is not None
        if state["final_submission"] is not None and has_hidden_test:
            evaluator = Evaluator(self.evaluation_sandbox_factory, self.benchmark_config)
            evaluation = evaluator.evaluate(task, state["final_submission"])
        result = BenchmarkResult(
            task_id=task.task_id,
            status=state["status"],
            termination_reason=state["termination_reason"],
            submission=state["final_submission"],
            budget=budget_manager.snapshot(),
            evaluation=evaluation,
        )
        logger.record("episode_complete", **result.model_dump())
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
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--test-before", type=Path, help="Optional external hidden input CSV")
    parser.add_argument("--test-after", type=Path, help="Optional external hidden expected CSV")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    model_config = ModelConfig.model_validate(yaml.safe_load(args.model_config.read_text(encoding="utf-8")))
    sandbox_config = SandboxConfig()
    sandbox_factory = lambda files: LocalProcessSandbox(files, sandbox_config)
    runner = BenchmarkRunner(
        sandbox_factory,
        sandbox_factory,
        sandbox_config=sandbox_config,
        log_dir=args.log_dir,
        operator_vocabulary=["LogTransformation", "SquareRootTransformation", "SquareTransformation"],
    )
    result = runner.run_directory(
        args.task,
        create_model(model_config),
        test_before_path=args.test_before,
        test_after_path=args.test_after,
        provider_model=f"{model_config.provider}/{model_config.model}",
    )
    print(f"Task: {result.task_id}")
    print(f"Status: {result.status}")
    print(f"Tool calls: {result.budget['tool_calls']}")
    print(f"Tokens: {result.budget['total_tokens']}")
    print(f"Sandbox execution: {result.budget['execution_time_seconds']:.2f} sec")
    if result.evaluation:
        metrics = result.evaluation.operator_metrics
        print(f"Functional column accuracy: {result.evaluation.functional_column_accuracy:.2f}")
        print(f"Operator precision: {metrics.precision:.2f}")
        print(f"Operator recall: {metrics.recall:.2f}")
        print(f"Operator F1: {metrics.f1:.2f}")
        print(f"Exact sequence accuracy: {metrics.exact_sequence_accuracy:.2f}")


if __name__ == "__main__":
    _main()
