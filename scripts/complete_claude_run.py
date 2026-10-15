import sys
sys.path.insert(0, ".")
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
import dotenv
dotenv.load_dotenv(override=True)

from src.benchmark.aggregate import aggregate_results
from src.benchmark.runner import BenchmarkRunner
from src.benchmark.paper_tables import export_paper_tables
from src.config import HarnessConfig, ModelConfig, get_level_budget
from src.model_factory import create_model
from src.sandbox.local_process import LocalProcessSandbox

run_dir = Path("pipeline_runs/run_claude_sonnet_48tasks")
manifest_path = run_dir / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

results_dir = run_dir / "results"
logs_dir = run_dir / "logs"
artifacts_dir = run_dir / "artifacts"
tasks_dir = run_dir / "tasks"
reports_dir = run_dir / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

existing_task_ids = set()
for f in results_dir.glob("*.json"):
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        existing_task_ids.add(int(data["task_id"]))
    except Exception:
        pass

missing = [i for i in range(48) if i not in existing_task_ids]
print(f"Missing task IDs to execute: {missing}")

if not missing:
    print("All 48 tasks already complete!")
else:
    with open("configs/kaggle_claude_sonnet.yaml", "r", encoding="utf-8") as f:
        model_config_raw = yaml.safe_load(f)
    model_config = ModelConfig.model_validate(model_config_raw)

    with open("configs/harness.yaml", "r", encoding="utf-8") as f:
        harness_config = HarnessConfig.model_validate(yaml.safe_load(f))

    sandbox_factory = lambda files: LocalProcessSandbox(files, harness_config.sandbox)
    runner = BenchmarkRunner(
        sandbox_factory,
        sandbox_factory,
        benchmark_config=harness_config.benchmark,
        sandbox_config=harness_config.sandbox,
        log_dir=logs_dir,
        artifact_dir=artifacts_dir,
        result_dir=results_dir,
        operator_vocabulary=harness_config.operator_vocabulary,
    )

    tasks_meta = {int(t["task_id"]): t for t in manifest["tasks"]}

    def run_one_task(idx):
        t_meta = tasks_meta[idx]
        task_level = t_meta.get("level", 1)
        task_budget = get_level_budget(task_level, harness_config.budget, harness_config.level_budgets)
        task_model = create_model(model_config)
        task_directory = tasks_dir / str(idx)
        print(f"Starting missing Task {idx} (Level {task_level})...")
        res = runner.run_directory(
            task_directory,
            task_model,
            task_budget,
            provider_model=f"{model_config.provider}/{model_config.model}",
        )
        print(f"Finished Task {idx}: status={res.status}, trans_acc={res.evaluation.transformed_column_accuracy if res.evaluation else 'none'}")
        return idx, res

    with ThreadPoolExecutor(max_workers=min(4, len(missing))) as executor:
        futures = [executor.submit(run_one_task, idx) for idx in missing]
        for fut in as_completed(futures):
            fut.result()

print("\nAll missing tasks finished! Aggregating results...")
summary_path = reports_dir / "summary.csv"
summary_df = aggregate_results(results_dir, summary_path)
print(f"Aggregated {len(summary_df)} tasks to {summary_path}")

tables_dir = reports_dir / "tables"
tables = export_paper_tables(summary_path, tables_dir)
print(f"Exported tables to {tables_dir}")
