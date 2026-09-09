from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from transformation import (
    Transformation,
    LogTransformation,
    SquareRootTransformation,
    SquareTransformation,
    ExpTransformation,
    Log1pTransformation,
    CubeTransformation,
    CubeRootTransformation,
    InverseTransformation,
    StandardScalerTransformation,
    MinMaxScalerTransformation,
    SinTransformation,
    CosTransformation,
    TanhTransformation,
    SigmoidTransformation,
    AbsTransformation,
    AddInteractionTransformation,
    SubtractInteractionTransformation,
    MultiplyInteractionTransformation,
    DivideInteractionTransformation,
)

OPERATORS: dict[str, type[Transformation]] = {
    "LogTransformation": LogTransformation,
    "Log1pTransformation": Log1pTransformation,
    "SquareRootTransformation": SquareRootTransformation,
    "SquareTransformation": SquareTransformation,
    "CubeTransformation": CubeTransformation,
    "CubeRootTransformation": CubeRootTransformation,
    "ExpTransformation": ExpTransformation,
    "InverseTransformation": InverseTransformation,
    "StandardScalerTransformation": StandardScalerTransformation,
    "MinMaxScalerTransformation": MinMaxScalerTransformation,
    "SinTransformation": SinTransformation,
    "CosTransformation": CosTransformation,
    "TanhTransformation": TanhTransformation,
    "SigmoidTransformation": SigmoidTransformation,
    "AbsTransformation": AbsTransformation,
}

INTERACTION_OPERATORS: dict[str, type[Transformation]] = {
    "AddInteractionTransformation": AddInteractionTransformation,
    "SubtractInteractionTransformation": SubtractInteractionTransformation,
    "MultiplyInteractionTransformation": MultiplyInteractionTransformation,
    "DivideInteractionTransformation": DivideInteractionTransformation,
}

ALL_OPERATORS: dict[str, type[Transformation]] = {**OPERATORS, **INTERACTION_OPERATORS}


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


def applicable_interactions(
    frame: pd.DataFrame,
    candidate_targets: list[str],
    candidate_sources: list[str],
    interaction_names: list[str] | None = None,
) -> list[tuple[str, str, str, str]]:
    names = interaction_names or list(INTERACTION_OPERATORS)
    results: list[tuple[str, str, str, str]] = []
    for target in candidate_targets:
        for i, src_a in enumerate(candidate_sources):
            for j, src_b in enumerate(candidate_sources):
                if i >= j or src_a == target or src_b == target:
                    continue
                data_a = frame[src_a].to_numpy()
                data_b = frame[src_b].to_numpy()
                for op_name in names:
                    op = INTERACTION_OPERATORS[op_name]()
                    if hasattr(op, "is_applicable_columns") and op.is_applicable_columns(data_a, data_b):
                        results.append((target, src_a, src_b, op_name))
    return results


def build_atomic_task(
    source_csv: Path | str | pd.DataFrame,
    output_directory: Path | str,
    *,
    seed: int = 0,
    train_fraction: float = 0.7,
    operator_names: list[str] | None = None,
) -> tuple[str, str]:
    """Build a single atomic task (for backward compatibility)."""
    transforms = build_task(
        source_csv,
        output_directory,
        level=0,
        seed=seed,
        train_fraction=train_fraction,
        operator_names=operator_names,
    )
    first = transforms[0]
    return first["column"], first["operators"][0]


def build_task(
    source_csv: Path | str | pd.DataFrame,
    output_directory: Path | str,
    *,
    level: int = 1,
    seed: int = 0,
    train_fraction: float = 0.7,
    operator_names: list[str] | None = None,
    allow_chaining: bool = True,
) -> list[dict[str, Any]]:
    """Build a benchmark task matching specified difficulty level (0=atomic, 1 to 5)."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    frame = source_csv.copy() if isinstance(source_csv, pd.DataFrame) else pd.read_csv(source_csv)
    if len(frame) < 2:
        raise ValueError("at least two rows are required")
    non_numeric = [c for c in frame if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise ValueError(f"source data must be numeric; non-numeric columns: {non_numeric}")

    if operator_names is not None:
        unary_candidates = [op for op in operator_names if op in OPERATORS]
        if unary_candidates and not applicable_pairs(frame, unary_candidates):
            raise ValueError("no requested operator is applicable to any column")

    rng = random.Random(seed)
    columns = list(frame.columns)
    num_cols = len(columns)

    # Level specs: (num_independent, num_interactions, allow_chain)
    level_specs = {
        0: (1, 0, False),
        1: (2, 0, False),
        2: (3, 0, True),
        3: (4, 1, True),
        4: (5, 2, True),
        5: (6, 3, True),
    }
    req_indep, req_inter, req_chain = level_specs.get(level, (min(2, num_cols), 0, False))

    # Adjust requirements if dataset has few columns
    req_indep = min(req_indep, max(1, num_cols - 2 * req_inter - 1))
    if req_indep + req_inter > num_cols - 2 and req_inter > 0:
        req_inter = max(0, (num_cols - req_indep - 1) // 2)

    shuffled_cols = list(columns)
    rng.shuffle(shuffled_cols)

    indep_cols = shuffled_cols[:req_indep]
    remaining_cols = shuffled_cols[req_indep:]

    task_transforms: list[dict[str, Any]] = []

    # 1. Independent column transformations
    available_unary = operator_names or list(OPERATORS)
    for col in indep_cols:
        col_data = frame[col].to_numpy()
        valid_ops = [name for name in available_unary if name in OPERATORS and OPERATORS[name]().is_applicable(col_data)]
        if not valid_ops:
            if operator_names is not None:
                continue
            valid_ops = ["StandardScalerTransformation", "SquareTransformation"]
        
        chosen_op = rng.choice(valid_ops)
        ops_list = [chosen_op]

        # If chaining allowed and level >= 2, optionally chain 2 operators
        if allow_chaining and req_chain and rng.random() < 0.4:
            intermed = OPERATORS[chosen_op]().transform(col_data)
            second_valid = [
                name for name in available_unary
                if name in OPERATORS and name != chosen_op and OPERATORS[name]().is_applicable(intermed)
            ]
            if second_valid:
                second_op = rng.choice(second_valid)
                ops_list.append(second_op)

        task_transforms.append({
            "column": col,
            "operators": ops_list,
        })

    # 2. Inter-column interactions
    if req_inter > 0 and len(remaining_cols) >= 3:
        target_candidates = remaining_cols[:req_inter]
        source_pool = remaining_cols[req_inter:]
        if len(source_pool) < 2:
            source_pool = columns  # fallback pool

        for target in target_candidates:
            if len(source_pool) < 2:
                break
            src_a, src_b = rng.sample([c for c in source_pool if c != target], 2)
            interactions = applicable_interactions(
                frame, [target], [src_a, src_b], list(INTERACTION_OPERATORS)
            )
            if interactions:
                _, s_a, s_b, op_name = rng.choice(interactions)
                task_transforms.append({
                    "column": target,
                    "operators": [op_name],
                    "source_columns": [s_a, s_b],
                    "interaction": op_name.replace("InteractionTransformation", "").lower(),
                })

    if not task_transforms:
        # Fallback to atomic
        pairs = applicable_pairs(frame, operator_names)
        col, op = rng.choice(pairs)
        task_transforms.append({"column": col, "operators": [op]})

    # Generate train and test datasets
    indices = np.random.default_rng(seed).permutation(len(frame))
    split = min(max(1, int(len(frame) * train_fraction)), len(frame) - 1)
    train_before = frame.iloc[indices[:split]].reset_index(drop=True)
    test_before = frame.iloc[indices[split:]].reset_index(drop=True)

    def apply_transforms(source: pd.DataFrame) -> pd.DataFrame:
        result = source.copy()
        # Apply independent transformations first
        for item in task_transforms:
            col = item["column"]
            if item.get("source_columns"):
                continue
            val = result[col].to_numpy()
            for op_name in item["operators"]:
                op = OPERATORS[op_name]()
                val = op.transform(val)
            result[col] = val

        # Apply dependent interactions
        for item in task_transforms:
            col = item["column"]
            srcs = item.get("source_columns")
            if not srcs or len(srcs) != 2:
                continue
            src_a, src_b = srcs
            op_name = item["operators"][0]
            op = INTERACTION_OPERATORS[op_name]()
            result[col] = op.transform_columns(source[src_a].to_numpy(), source[src_b].to_numpy())

        if not np.isfinite(result.to_numpy(dtype=float)).all():
            raise ValueError("operators produced non-finite task output")
        return result

    output = Path(output_directory)
    hidden = output / "hidden"
    hidden.mkdir(parents=True, exist_ok=True)
    train_before.to_csv(output / "train_before.csv", index=False)
    apply_transforms(train_before).to_csv(output / "train_after.csv", index=False)
    test_before.to_csv(hidden / "test_before.csv", index=False)
    apply_transforms(test_before).to_csv(hidden / "test_after.csv", index=False)

    config_data = {
        "level": level,
        "transformations": task_transforms,
    }
    (output / "config.yaml").write_text(
        yaml.safe_dump(config_data, sort_keys=False),
        encoding="utf-8",
    )
    return task_transforms


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark tasks with difficulty levels")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", type=int, default=1, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--operators", nargs="*", choices=sorted(ALL_OPERATORS))
    args = parser.parse_args()
    transforms = build_task(
        args.source,
        args.output,
        level=args.level,
        seed=args.seed,
        train_fraction=args.train_fraction,
        operator_names=args.operators,
    )
    print(f"Built Level {args.level} task at {args.output} with {len(transforms)} transformations:")
    for t in transforms:
        print(f"  - {t['column']}: {t['operators']} (sources: {t.get('source_columns')})")


if __name__ == "__main__":
    _main()
