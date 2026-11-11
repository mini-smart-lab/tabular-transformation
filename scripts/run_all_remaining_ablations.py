r"""Automated runner for remaining ablation conditions and multi-model table export."""

import json
import subprocess
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

RUNS_DIR = ROOT / "pipeline_runs_revision"
TASKS_DIR = ROOT / "data" / "benchmark_48tasks"


def run_pipeline(model_config: str, run_name: str, ablation_flag: str, workers: int = 4):
    dest = RUNS_DIR / run_name
    summary_file = dest / "reports" / "summary.csv"
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        if len(df) == 48:
            print(f"[{run_name}] Already complete with 48 tasks, skipping.")
            return

    cmd = [
        PYTHON,
        str(ROOT / "run_pipeline.py"),
        "--tasks-dir", str(TASKS_DIR),
        "--model-config", str(ROOT / "configs" / model_config),
        "--output-root", str(RUNS_DIR),
        "--run-name", run_name,
        ablation_flag,
        "--workers", str(workers),
        "--seed", "42",
    ]
    print(f"\n=======================================================")
    print(f"Launching: {' '.join(cmd)}")
    print(f"=======================================================")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    # We execute oracle_ops for GPT and Claude
    run_pipeline("kaggle_gpt5_nano.yaml", "run_gpt5_nano_ablation_oracle_ops_48tasks", "--ablation-oracle-ops", workers=4)
    run_pipeline("kaggle_claude_sonnet.yaml", "run_claude_sonnet_ablation_oracle_ops_48tasks", "--ablation-oracle-ops", workers=4)
    print("\nAll remaining ablation runs complete!")
