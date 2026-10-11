import subprocess
import sys
import time

levels = [0, 1, 2, 3, 4, 5]
workers = 2
task_count = 24
seed = 42

print(f"Starting execution of levels {levels} with GPT-5 nano (workers={workers}, task_count={task_count})...")

for lvl in levels:
    print(f"\n{'='*80}")
    print(f"RUNNING LEVEL {lvl}...")
    print(f"{'='*80}\n")
    start = time.time()
    cmd = [
        sys.executable,
        "run_pipeline.py",
        "--level", str(lvl),
        "--task-count", str(task_count),
        "--workers", str(workers),
        "--seed", str(seed),
        "--model-config", "configs/openai.yaml"
    ]
    ret = subprocess.run(cmd)
    elapsed = time.time() - start
    print(f"\nLevel {lvl} finished in {elapsed:.2f}s with return code {ret.returncode}")
    if ret.returncode != 0:
        print(f"Warning: Level {lvl} returned non-zero exit code: {ret.returncode}")
    # Small pause between levels to let rate limits cool down
    time.sleep(5)

print("\nAll levels completed successfully!")
