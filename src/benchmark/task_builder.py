from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from transformation.log import LogTransformation
from transformation.sqrt import SquareRootTransformation
from transformation.square import SquareTransformation


OPERATORS = {
    "LogTransformation": LogTransformation,
    "SquareRootTransformation": SquareRootTransformation,
    "SquareTransformation": SquareTransformation,
}


def applicable_pairs(frame: pd.DataFrame, operator_names: list[str] | None = None) -> list[tuple[str, str]]:
    names = operator_names or list(OPERATORS)
    unknown = set(names) - set(OPERATORS)
    if unknown:
        raise ValueError(f"unknown operators: {sorted(unknown)}")
    pairs: list[tuple[str, str]] = []
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = frame[column].to_numpy()
        for name in names:
            if OPERATORS[name]().is_applicable(values):
                pairs.append((column, name))
    return pairs


def build_atomic_task(
    source_csv: Path | str | pd.DataFrame,
    output_directory: Path | str,
    *,
    seed: int = 0,
    train_fraction: float = 0.7,
    operator_names: list[str] | None = None,
) -> tuple[str, str]:
    """Build a deterministic public train pair plus a host-only hidden test pair."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    frame = source_csv.copy() if isinstance(source_csv, pd.DataFrame) else pd.read_csv(source_csv)
    if len(frame) < 2:
        raise ValueError("at least two rows are required")
    non_numeric = [c for c in frame if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise ValueError(f"source data must be numeric; non-numeric columns: {non_numeric}")
    pairs = applicable_pairs(frame, operator_names)
    if not pairs:
        raise ValueError("no requested operator is applicable to any column")
    column, operator_name = random.Random(seed).choice(pairs)
    indices = np.random.default_rng(seed).permutation(len(frame))
    split = min(max(1, int(len(frame) * train_fraction)), len(frame) - 1)
    train_before = frame.iloc[indices[:split]].reset_index(drop=True)
    test_before = frame.iloc[indices[split:]].reset_index(drop=True)
    operator = OPERATORS[operator_name]()

    def transformed(source: pd.DataFrame) -> pd.DataFrame:
        result = source.copy()
        result[column] = operator.transform(result[column].to_numpy())
        if not np.isfinite(result.to_numpy(dtype=float)).all():
            raise ValueError("operator produced non-finite task output")
        return result

    output = Path(output_directory)
    hidden = output / "hidden"
    hidden.mkdir(parents=True, exist_ok=True)
    train_before.to_csv(output / "train_before.csv", index=False)
    transformed(train_before).to_csv(output / "train_after.csv", index=False)
    test_before.to_csv(hidden / "test_before.csv", index=False)
    transformed(test_before).to_csv(hidden / "test_after.csv", index=False)
    (output / "config.yaml").write_text(
        yaml.safe_dump({"transformations": [{"column": column, "operators": [operator_name]}]}, sort_keys=False),
        encoding="utf-8",
    )
    return column, operator_name


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build one valid atomic benchmark task")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--operators", nargs="*", choices=sorted(OPERATORS))
    args = parser.parse_args()
    column, operator = build_atomic_task(
        args.source, args.output, seed=args.seed,
        train_fraction=args.train_fraction, operator_names=args.operators,
    )
    print(f"Built task at {args.output}: {column} -> {operator}")


if __name__ == "__main__":
    _main()
