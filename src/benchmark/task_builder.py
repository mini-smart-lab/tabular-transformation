from __future__ import annotations

import argparse
import json
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
    IndicatorZeroTransformation,
    IndicatorMeanTransformation,
    ClipZeroTransformation,
    PiecewiseLogTransformation,
    ConditionalInteractionTransformation,
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
    "IndicatorZeroTransformation": IndicatorZeroTransformation,
    "IndicatorMeanTransformation": IndicatorMeanTransformation,
    "ClipZeroTransformation": ClipZeroTransformation,
    "PiecewiseLogTransformation": PiecewiseLogTransformation,
}

CONDITIONAL_UNARY_OPERATORS: list[str] = [
    "IndicatorZeroTransformation",
    "IndicatorMeanTransformation",
    "ClipZeroTransformation",
    "PiecewiseLogTransformation",
]

INTERACTION_OPERATORS: dict[str, type[Transformation]] = {
    "AddInteractionTransformation": AddInteractionTransformation,
    "SubtractInteractionTransformation": SubtractInteractionTransformation,
    "MultiplyInteractionTransformation": MultiplyInteractionTransformation,
    "DivideInteractionTransformation": DivideInteractionTransformation,
    "ConditionalInteractionTransformation": ConditionalInteractionTransformation,
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
    source_dataset: str | None = None,
    domain: str | None = None,
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

    # Level specifications:
    # Level 0: 1 indep, 0 inter, d=1 (atomic)
    # Level 1: 2 indep, 0 inter, d=1 (atomic unary)
    # Level 2: 3 indep, 0 inter, d=2 (composed unary f(g(x)))
    # Level 3: 3 indep, 1 inter, d=2 (interaction)
    # Level 4: 3 indep (with conditional ops), 1 inter (conditional/standard), d=2 (piecewise/if-else)
    # Level 5: 4 indep (composed+conditional), 2 inter, d=3+ (hybrid deep AST)
    level_specs = {
        0: (1, 0, False, 0.0),
        1: (2, 0, False, 0.0),
        2: (3, 0, True, 1.0),
        3: (3, 1, True, 0.5),
        4: (3, 1, True, 0.6),
        5: (4, 2, True, 0.8),
    }
    req_indep, req_inter, req_chain, chain_prob = level_specs.get(level, (min(2, num_cols), 0, False, 0.0))

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
    for col_idx, col in enumerate(indep_cols):
        col_data = frame[col].to_numpy()
        valid_ops = [
            name for name in available_unary
            if name in OPERATORS and OPERATORS[name]().is_applicable(col_data)
        ]
        if not valid_ops:
            if operator_names is not None:
                continue
            valid_ops = ["StandardScalerTransformation", "SquareTransformation"]

        # In level 4 and 5, prioritize conditional/piecewise unary operators for some columns
        if level in (4, 5) and col_idx == 0:
            cond_valid = [op for op in valid_ops if op in CONDITIONAL_UNARY_OPERATORS]
            chosen_op = rng.choice(cond_valid) if cond_valid else rng.choice(valid_ops)
        else:
            chosen_op = rng.choice(valid_ops)

        ops_list = [chosen_op]

        # Apply chaining if enabled
        should_chain = allow_chaining and req_chain and (rng.random() < chain_prob or (level == 2 and col_idx < 2))
        if should_chain:
            try:
                intermed = OPERATORS[chosen_op]().transform(col_data)
                second_valid = [
                    name for name in available_unary
                    if name in OPERATORS and name != chosen_op and OPERATORS[name]().is_applicable(intermed)
                ]
                if second_valid:
                    second_op = rng.choice(second_valid)
                    ops_list.append(second_op)

                    # In level 5, allow depth 3 for the first column
                    if level == 5 and col_idx == 0 and rng.random() < 0.5:
                        intermed2 = OPERATORS[second_op]().transform(intermed)
                        third_valid = [
                            name for name in available_unary
                            if name in OPERATORS and name != second_op and OPERATORS[name]().is_applicable(intermed2)
                        ]
                        if third_valid:
                            ops_list.append(rng.choice(third_valid))
            except Exception:
                pass

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

        for t_idx, target in enumerate(target_candidates):
            if len(source_pool) < 2:
                break
            src_a, src_b = rng.sample([c for c in source_pool if c != target], 2)
            
            # For Level 4 and 5, include conditional interaction
            cand_interaction_names = list(INTERACTION_OPERATORS)
            if level == 4 and t_idx == 0:
                # Prioritize conditional interaction
                cand_interaction_names = ["ConditionalInteractionTransformation"] + [
                    k for k in INTERACTION_OPERATORS if k != "ConditionalInteractionTransformation"
                ]

            interactions = applicable_interactions(
                frame, [target], [src_a, src_b], cand_interaction_names
            )
            if interactions:
                _, s_a, s_b, op_name = rng.choice(interactions)
                ops_list = [op_name]

                # In level 5, allow unary transform on top of interaction: f(x1 op x2)
                if level == 5 and rng.random() < 0.5:
                    op_inst = INTERACTION_OPERATORS[op_name]()
                    inter_val = op_inst.transform_columns(frame[s_a].to_numpy(), frame[s_b].to_numpy())
                    unary_on_inter = [
                        u for u in available_unary
                        if u in OPERATORS and OPERATORS[u]().is_applicable(inter_val)
                    ]
                    if unary_on_inter:
                        ops_list.append(rng.choice(unary_on_inter))

                task_transforms.append({
                    "column": target,
                    "operators": ops_list,
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

    # Fit operators strictly on train_before to eliminate data leakage.
    # Parameters (e.g. mean, std, min, ptp, thresholds) are recorded in task metadata.
    fitted_unaries: dict[str, list[Any]] = {}
    fitted_interactions: dict[str, tuple[Any, list[Any]]] = {}

    for item in task_transforms:
        col = item["column"]
        srcs = item.get("source_columns")
        if srcs and len(srcs) == 2:
            s_a, s_b = srcs
            op_name = item["operators"][0]
            op_inst = INTERACTION_OPERATORS[op_name]()
            op_inst.fit_columns(train_before[s_a].to_numpy(), train_before[s_b].to_numpy())
            inter_val = op_inst.transform_columns(train_before[s_a].to_numpy(), train_before[s_b].to_numpy())
            params_list = [op_inst.get_params()]
            unary_insts: list[Any] = []
            if len(item["operators"]) > 1:
                for u_name in item["operators"][1:]:
                    u_inst = OPERATORS[u_name]()
                    u_inst.fit(inter_val)
                    inter_val = u_inst.transform(inter_val)
                    params_list.append(u_inst.get_params())
                    unary_insts.append(u_inst)
            fitted_interactions[col] = (op_inst, unary_insts)
            item["parameters"] = params_list
        else:
            val = train_before[col].to_numpy()
            u_insts: list[Any] = []
            params_list = []
            for op_name in item["operators"]:
                op_inst = OPERATORS[op_name]()
                op_inst.fit(val)
                val = op_inst.transform(val)
                params_list.append(op_inst.get_params())
                u_insts.append(op_inst)
            fitted_unaries[col] = u_insts
            item["parameters"] = params_list

    def apply_fitted_transforms(source: pd.DataFrame) -> pd.DataFrame:
        result = source.copy()
        # 1. Independent column transformations
        for item in task_transforms:
            col = item["column"]
            if item.get("source_columns"):
                continue
            val = result[col].to_numpy()
            for op_inst in fitted_unaries[col]:
                val = op_inst.transform(val)
            result[col] = val

        # 2. Inter-column interactions
        for item in task_transforms:
            col = item["column"]
            srcs = item.get("source_columns")
            if not srcs or len(srcs) != 2:
                continue
            s_a, s_b = srcs
            op_inst, unary_insts = fitted_interactions[col]
            res = op_inst.transform_columns(source[s_a].to_numpy(), source[s_b].to_numpy())
            for u_inst in unary_insts:
                res = u_inst.transform(res)
            result[col] = res

        if not np.isfinite(result.to_numpy(dtype=float)).all():
            raise ValueError("operators produced non-finite task output")
        return result

    output = Path(output_directory)
    hidden = output / "hidden"
    hidden.mkdir(parents=True, exist_ok=True)
    train_before.to_csv(output / "train_before.csv", index=False)
    apply_fitted_transforms(train_before).to_csv(output / "train_after.csv", index=False)
    test_before.to_csv(hidden / "test_before.csv", index=False)
    apply_fitted_transforms(test_before).to_csv(hidden / "test_after.csv", index=False)

    config_data = {
        "level": level,
        "transformations": task_transforms,
    }
    if source_dataset:
        config_data["source_dataset"] = source_dataset
    if domain:
        config_data["domain"] = domain

    (output / "config.yaml").write_text(
        yaml.safe_dump(config_data, sort_keys=False),
        encoding="utf-8",
    )
    return task_transforms


def build_multidataset_suite(
    source_input: Path | str | list[Path | str],
    output_directory: Path | str,
    *,
    levels: list[int] | None = None,
    task_count: int = 24,
    seed: int = 42,
    train_fraction: float = 0.7,
    rows_per_task: int = 10_000,
    operator_names: list[str] | None = None,
    dataset_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build a balanced benchmark suite spanning multiple source datasets and difficulty levels."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_info: dict[str, dict[str, Any]] = {}
    csv_paths: list[Path] = []

    if isinstance(source_input, (list, tuple)):
        csv_paths = [Path(p).resolve() for p in source_input]
    else:
        src_p = Path(source_input).resolve()
        if src_p.is_dir():
            manifest_file = src_p / "dataset_manifest.json"
            if manifest_file.is_file():
                try:
                    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
                    manifest_info = manifest_data.get("datasets", {})
                except Exception:
                    manifest_info = {}
            csv_paths = sorted(src_p.glob("*.csv"))
        elif src_p.is_file():
            csv_paths = [src_p]
        else:
            raise FileNotFoundError(f"source input does not exist: {src_p}")

    if dataset_filter:
        filter_stems = {f.lower().removesuffix(".csv") for f in dataset_filter}
        csv_paths = [p for p in csv_paths if p.stem.lower() in filter_stems or p.name.lower() in filter_stems]

    if not csv_paths:
        raise ValueError("no valid CSV files found in source input")

    target_levels = levels if levels is not None else [0, 1, 2, 3, 4, 5]
    if not target_levels:
        target_levels = [1]

    datasets: list[dict[str, Any]] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) < 2:
            continue
        df = df[num_cols].dropna().reset_index(drop=True)
        ds_name = path.stem
        domain = manifest_info.get(ds_name, {}).get("domain", "General")
        datasets.append({
            "name": ds_name,
            "path": path,
            "domain": domain,
            "frame": df,
        })

    if not datasets:
        raise ValueError("none of the source datasets contain sufficient numeric data")

    generated: list[dict[str, Any]] = []
    num_datasets = len(datasets)
    num_levels = len(target_levels)

    for index in range(task_count):
        ds_item = datasets[index % num_datasets]
        lvl = target_levels[index % num_levels]
        task_dir = output_path / str(index)

        task_dir.mkdir(parents=True, exist_ok=True)

        df_source = ds_item["frame"]
        if rows_per_task > 0 and len(df_source) > rows_per_task:
            task_frame = df_source.sample(
                n=rows_per_task, random_state=seed + index, replace=False
            ).reset_index(drop=True)
        else:
            task_frame = df_source.copy()

        transforms = build_task(
            task_frame,
            task_dir,
            level=lvl,
            seed=seed + index,
            train_fraction=train_fraction,
            operator_names=operator_names,
            source_dataset=ds_item["name"],
            domain=ds_item["domain"],
        )

        generated.append({
            "task_id": str(index),
            "source_dataset": ds_item["name"],
            "domain": ds_item["domain"],
            "level": lvl,
            "seed": seed + index,
            "transformations": transforms,
        })

    return generated


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
