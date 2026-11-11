r"""Orchestrate 3-seed evaluation across models on data/benchmark_48tasks and compute Mean ± Std."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PYTHON = sys.executable

OUTPUT_ROOT = ROOT / "pipeline_runs_revision"
TASKS_DIR = ROOT / "data" / "benchmark_48tasks"

RUNS_CONFIG = [
    # Model 1: GPT-4o mini (Lightweight Frontier Model)
    ("configs/kaggle_gpt5_nano.yaml", "run_gpt5_nano_48tasks_seed43", 43),
    ("configs/kaggle_gpt5_nano.yaml", "run_gpt5_nano_48tasks_seed44", 44),
    # Model 2: Gemini 2.0 Flash (Frontier Multimodal Agent)
    ("configs/kaggle_gemini3_flash.yaml", "run_gemini3_flash_48tasks_seed43", 43),
    ("configs/kaggle_gemini3_flash.yaml", "run_gemini3_flash_48tasks_seed44", 44),
    # Model 3: Claude 3.5 Sonnet (Frontier Coding Agent)
    ("configs/kaggle_claude_sonnet.yaml", "run_claude_sonnet_48tasks_seed43", 43),
    ("configs/kaggle_claude_sonnet.yaml", "run_claude_sonnet_48tasks_seed44", 44),
]


def run_one_experiment(model_cfg: str, run_name: str, seed: int, workers: int = 4):
    target_dir = OUTPUT_ROOT / run_name
    summary_path = target_dir / "reports" / "summary.csv"
    results_dir = target_dir / "results"
    
    if summary_path.exists():
        try:
            df = pd.read_csv(summary_path)
            if len(df) == 48:
                print(f"[{run_name}] Already completed with 48 tasks, skipping.")
                return summary_path
        except Exception:
            pass

    if target_dir.exists():
        existing_tasks = set()
        if results_dir.exists():
            for f in results_dir.glob("*.json"):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    existing_tasks.add(int(data["task_id"]))
                except Exception:
                    pass
        missing = [i for i in range(48) if i not in existing_tasks]
        if missing:
            print(f"[{run_name}] Existing run dir has {len(existing_tasks)} valid tasks. Missing {len(missing)} tasks: {missing}. Retrying...")
            _retry_missing_tasks(target_dir, model_cfg, seed, missing)
            return summary_path
        elif len(existing_tasks) == 48:
            from src.benchmark.aggregate import aggregate_results
            aggregate_results(results_dir, summary_path)
            print(f"[{run_name}] All 48 tasks present. Aggregated to {summary_path}")
            return summary_path

    cmd = [
        PYTHON,
        str(ROOT / "run_pipeline.py"),
        "--tasks-dir", str(TASKS_DIR),
        "--model-config", str(ROOT / model_cfg),
        "--output-root", str(OUTPUT_ROOT),
        "--run-name", run_name,
        "--seed", str(seed),
        "--workers", str(workers),
    ]

    print(f"\n=======================================================")
    print(f"Launching run: {run_name} (Model: {model_cfg}, Seed: {seed})")
    print(f"Command: {' '.join(cmd)}")
    print(f"=======================================================")
    
    start_t = time.time()
    ret = subprocess.run(cmd)
    elapsed = time.time() - start_t
    print(f"Finished {run_name} in {elapsed:.1f}s with exit code {ret.returncode}")
    
    # Check if all 48 tasks were completed
    existing_tasks = set()
    if results_dir.exists():
        for f in results_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                existing_tasks.add(int(data["task_id"]))
            except Exception:
                pass
    
    missing = [i for i in range(48) if i not in existing_tasks]
    if missing:
        print(f"[{run_name}] Warning: {len(missing)} missing tasks: {missing}. Retrying missing tasks...")
        _retry_missing_tasks(target_dir, model_cfg, seed, missing)
        
    return summary_path


def _retry_missing_tasks(run_dir: Path, model_cfg: str, seed: int, missing: list[int]):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import yaml
    import dotenv
    dotenv.load_dotenv(override=True)
    from src.benchmark.runner import BenchmarkRunner
    from src.benchmark.aggregate import aggregate_results
    from src.config import HarnessConfig, ModelConfig, get_level_budget
    from src.model_factory import create_model
    from src.sandbox.local_process import LocalProcessSandbox

    with open(ROOT / model_cfg, "r", encoding="utf-8") as f:
        model_raw = yaml.safe_load(f)
    model_raw["seed"] = seed
    model_config = ModelConfig.model_validate(model_raw)

    with open(ROOT / "configs" / "harness.yaml", "r", encoding="utf-8") as f:
        harness_config = HarnessConfig.model_validate(yaml.safe_load(f))

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    tasks_meta = {int(t["task_id"]): t for t in manifest["tasks"]}

    sandbox_factory = lambda files: LocalProcessSandbox(files, harness_config.sandbox)
    runner = BenchmarkRunner(
        sandbox_factory,
        sandbox_factory,
        benchmark_config=harness_config.benchmark,
        sandbox_config=harness_config.sandbox,
        log_dir=run_dir / "logs",
        artifact_dir=run_dir / "artifacts",
        result_dir=run_dir / "results",
        operator_vocabulary=harness_config.operator_vocabulary,
    )

    def _run_one(idx):
        t_meta = tasks_meta[idx]
        task_lvl = t_meta.get("level", 1)
        budget = get_level_budget(task_lvl, harness_config.budget, harness_config.level_budgets)
        model = create_model(model_config)
        res = runner.run_directory(
            run_dir / "tasks" / str(idx),
            model,
            budget,
            provider_model=f"{model_config.provider}/{model_config.model}",
        )
        return idx, res

    with ThreadPoolExecutor(max_workers=min(4, len(missing))) as executor:
        futs = [executor.submit(_run_one, i) for i in missing]
        for fut in as_completed(futs):
            i, r = fut.result()
            print(f"Retried task {i}: status={r.status}")

    # Re-aggregate
    summary_path = run_dir / "reports" / "summary.csv"
    aggregate_results(run_dir / "results", summary_path)
    print(f"Re-aggregated results to {summary_path}")


def aggregate_all_seeds():
    runs_map = {
        "Claude 3.5 Sonnet": [
            ROOT / "pipeline_runs" / "run_claude_sonnet_48tasks" / "reports" / "summary.csv",
            OUTPUT_ROOT / "run_claude_sonnet_48tasks_seed43" / "reports" / "summary.csv",
        ],
        "Gemini 2.0 Flash": [
            ROOT / "pipeline_runs" / "run_gemini3_flash_48tasks" / "reports" / "summary.csv",
            OUTPUT_ROOT / "run_gemini3_flash_48tasks_seed43" / "reports" / "summary.csv",
        ],
        "GPT-4o mini": [
            ROOT / "pipeline_runs" / "run_gpt5_nano_48tasks" / "reports" / "summary.csv",
            OUTPUT_ROOT / "run_gpt5_nano_48tasks_seed43" / "reports" / "summary.csv",
        ],
    }

    metrics_records = []
    class_records = []

    for model_name, paths in runs_map.items():
        seed_stats = []
        for seed_idx, p in enumerate(paths, start=42):
            if not p.exists():
                print(f"Missing summary: {p}")
                continue
            df = pd.read_csv(p)
            n = len(df)
            sub = df[df["status"] == "submitted"]
            sub_rate = len(sub) / n if n else 0.0

            trans_col = "transformed_column_accuracy" if "transformed_column_accuracy" in df.columns else "functional_column_accuracy"
            cond_trans = float(sub[trans_col].fillna(0.0).mean()) if len(sub) else 0.0
            pen_trans = float(df[trans_col].fillna(0.0).mean()) if n else 0.0
            pen_func = float(df["functional_column_accuracy"].fillna(0.0).mean()) if n else 0.0
            cond_op_f1 = float(sub["operator_f1"].fillna(0.0).mean()) if len(sub) and "operator_f1" in sub.columns else 0.0

            seed_stats.append({
                "model": model_name,
                "seed": seed_idx,
                "sub_rate": sub_rate,
                "cond_trans": cond_trans,
                "pen_trans": pen_trans,
                "pen_func": pen_func,
                "cond_op_f1": cond_op_f1,
            })

            # Per class penalized transformed accuracy
            class_row = {"model": model_name, "seed": seed_idx}
            for lvl in range(6):
                subset = df[df["level"] == lvl]
                acc = float(subset[trans_col].fillna(0.0).mean()) if len(subset) else 0.0
                class_row[f"C{lvl}"] = acc
            class_records.append(class_row)

        metrics_records.extend(seed_stats)

    df_metrics = pd.DataFrame(metrics_records)
    df_classes = pd.DataFrame(class_records)

    df_metrics.to_csv(OUTPUT_ROOT / "multi_seed_metrics_raw.csv", index=False)
    df_classes.to_csv(OUTPUT_ROOT / "multi_seed_classes_raw.csv", index=False)

    print("\n================ MULTI-SEED SUMMARY (Table 1 Metrics) ================")
    summary_rows = []
    for model_name in runs_map.keys():
        sub_df = df_metrics[df_metrics["model"] == model_name]
        if sub_df.empty:
            continue
        row = {
            "Model": model_name,
            "Seeds": len(sub_df),
            "Sub. Rate": f"{sub_df['sub_rate'].mean()*100:.1f} ± {sub_df['sub_rate'].std()*100:.1f}%",
            "Trans. (Cond)": f"{sub_df['cond_trans'].mean():.3f} ± {sub_df['cond_trans'].std():.3f}",
            "Trans. (Pen)": f"{sub_df['pen_trans'].mean():.3f} ± {sub_df['pen_trans'].std():.3f}",
            "Func. (Pen)": f"{sub_df['pen_func'].mean():.3f} ± {sub_df['pen_func'].std():.3f}",
            "Op. F1 (Cond)": f"{sub_df['cond_op_f1'].mean():.3f} ± {sub_df['cond_op_f1'].std():.3f}",
        }
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUTPUT_ROOT / "multi_seed_table1_summary.csv", index=False)
    print(df_summary.to_string(index=False))

    print("\n================ MULTI-SEED CLASS BREAKDOWN (Table 4) ================")
    class_summary_rows = []
    for model_name in runs_map.keys():
        sub_c = df_classes[df_classes["model"] == model_name]
        if sub_c.empty:
            continue
        row = {"Model": model_name}
        for lvl in range(6):
            col = f"C{lvl}"
            m = sub_c[col].mean()
            s = sub_c[col].std()
            row[col] = f"{m:.3f} ± {s:.3f}"
        class_summary_rows.append(row)

    df_class_summary = pd.DataFrame(class_summary_rows)
    df_class_summary.to_csv(OUTPUT_ROOT / "multi_seed_table4_summary.csv", index=False)
    print(df_class_summary.to_string(index=False))


if __name__ == "__main__":
    for model_cfg, run_name, seed in RUNS_CONFIG:
        run_one_experiment(model_cfg, run_name, seed, workers=4)
    print("\nAll 6 runs finished! Aggregating multi-seed statistics...")
    aggregate_all_seeds()
