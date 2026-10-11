"""Kaggle Benchmarks Task Exporter.

Converts internal Tabular Transformation Discovery tasks into standalone, reproducible
Python task scripts compliant with the `kaggle-benchmarks` SDK (`@kbench.task`).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def slugify_name(text: str) -> str:
    """Normalize string into a valid Kaggle task slug (lowercase, dashes)."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text.strip("-")


def to_python_identifier(slug: str) -> str:
    """Convert slug into a valid Python function identifier."""
    name = re.sub(r"[^\w]", "_", slug)
    if name and name[0].isdigit():
        name = f"task_{name}"
    return name


def export_task_to_kbench(
    task_dir: Path | str,
    output_py_file: Path | str,
    *,
    task_name: str | None = None,
    max_train_rows: int = 60,
    max_test_rows: int = 60,
) -> Path:
    """Export a single task directory to a self-contained Kaggle Benchmarks Python script."""
    task_path = Path(task_dir).resolve()
    out_file = Path(output_py_file).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load data and config
    train_before = pd.read_csv(task_path / "train_before.csv")
    train_after = pd.read_csv(task_path / "train_after.csv")
    test_before = pd.read_csv(task_path / "hidden" / "test_before.csv")
    test_after = pd.read_csv(task_path / "hidden" / "test_after.csv")

    config_path = task_path / "config.yaml"
    config: dict[str, Any] = {}
    if config_path.is_file():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    level = config.get("level", 1)
    source_ds = config.get("source_dataset", task_path.parent.name)
    transforms = config.get("transformations", [])
    transformed_cols = [t["column"] for t in transforms if "column" in t]

    # Sample rows for compact, fast self-contained execution
    train_sample_n = min(len(train_before), max_train_rows)
    test_sample_n = min(len(test_before), max_test_rows)

    tb_sample = train_before.iloc[:train_sample_n]
    ta_sample = train_after.iloc[:train_sample_n]
    test_b_sample = test_before.iloc[:test_sample_n]
    test_a_sample = test_after.iloc[:test_sample_n]

    tb_csv = tb_sample.to_csv(index=False)
    ta_csv = ta_sample.to_csv(index=False)
    test_b_csv = test_b_sample.to_csv(index=False)
    test_a_csv = test_a_sample.to_csv(index=False)

    # Slug and function name
    base_slug = task_name or f"tab-transform-{source_ds}-l{level}-{task_path.name}"
    slug = slugify_name(base_slug)
    func_name = to_python_identifier(slug)

    script_content = f'''# %%
"""Kaggle Benchmark Task: {slug}
Domain: Tabular Transformation Discovery
Level: {level}
Source: {source_ds}
"""
from typing import Any
import io
import re
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import kaggle_benchmarks as kbench

# %%
TRAIN_BEFORE_CSV = """{tb_csv}"""
TRAIN_AFTER_CSV = """{ta_csv}"""
TEST_BEFORE_CSV = """{test_b_csv}"""
TEST_AFTER_CSV = """{test_a_csv}"""
TRANSFORMED_COLUMNS = {transformed_cols!r}
LEVEL = {level}

# %%
@kbench.task(name="{slug}")
def {func_name}(llm) -> dict:
    prompt = f"""You are an expert tabular data scientist.
Below are two versions of a tabular dataset:
1. Training Input (`train_before.csv`)
2. Training Target (`train_after.csv`)

One or more columns have undergone mathematical transformations (e.g. logarithmic, scaling, powers, interactions, or conditional piecewise logic).

### Training Input (first rows):
```csv
{{TRAIN_BEFORE_CSV}}
```

### Training Target (first rows):
```csv
{{TRAIN_AFTER_CSV}}
```

Task:
Analyze the differences, deduce the exact mathematical transformations applied, and write a complete Python function:
```python
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # return the transformed copy of df
```
Provide your solution inside a single ```python ... ``` code block.
"""
    response = llm.prompt(prompt)

    # 1. Extract code block
    match = re.search(r"```python\\s*(.*?)\\s*```", response, re.DOTALL)
    code = match.group(1) if match else response

    # 2. Execute transform on hidden test set
    scope: dict[str, Any] = {{"pd": pd, "np": np}}
    try:
        exec(code, scope)
        transform_fn = scope.get("transform")
        if not callable(transform_fn):
            raise ValueError("No callable 'transform' function found in model output")
        
        test_in = pd.read_csv(io.StringIO(TEST_BEFORE_CSV))
        test_expected = pd.read_csv(io.StringIO(TEST_AFTER_CSV))
        predicted = transform_fn(test_in.copy())
    except Exception as exc:
        kbench.assertions.assert_true(False, expectation=f"Transform execution failed: {{exc}}")
        return {{"error": str(exc), "functional_column_accuracy": 0.0, "transformed_column_accuracy": 0.0}}

    # 3. Score functional column accuracy
    total_cols = len(test_expected.columns)
    matched_cols = 0
    matched_trans = 0

    for col in test_expected.columns:
        if col in predicted.columns:
            try:
                y_true = test_expected[col].to_numpy(dtype=float)
                y_pred = predicted[col].to_numpy(dtype=float)
                if np.allclose(y_true, y_pred, rtol=1e-3, atol=1e-3, equal_nan=True):
                    matched_cols += 1
                    if col in TRANSFORMED_COLUMNS:
                        matched_trans += 1
            except Exception:
                pass

    overall_acc = matched_cols / total_cols if total_cols else 0.0
    trans_total = len(TRANSFORMED_COLUMNS)
    trans_acc = matched_trans / trans_total if trans_total else overall_acc

    kbench.assertions.assert_true(
        trans_acc >= 0.5,
        expectation=f"Transformed column accuracy ({{trans_acc:.2f}}) should be >= 0.50",
    )

    return {{
        "level": LEVEL,
        "functional_column_accuracy": float(overall_acc),
        "transformed_column_accuracy": float(trans_acc),
        "passed": bool(trans_acc >= 0.9),
    }}

# %%
{func_name}.run(kbench.llm)
'''
    out_file.write_text(script_content, encoding="utf-8")
    return out_file


def export_suite_to_kbench(
    tasks_dir: Path | str,
    output_dir: Path | str,
    *,
    prefix: str = "tab-transform",
) -> list[Path]:
    """Export all task directories in a folder to Kaggle Benchmark scripts."""
    tasks_path = Path(tasks_dir).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    exported: list[Path] = []
    # Identify task subdirectories
    subdirs = [p for p in sorted(tasks_path.iterdir()) if p.is_dir() and (p / "train_before.csv").is_file()]

    for subdir in subdirs:
        py_filename = f"{prefix}_{subdir.name}.py"
        target_file = out_path / py_filename
        export_task_to_kbench(subdir, target_file, task_name=f"{prefix}-{subdir.name}")
        exported.append(target_file)

    return exported
