from __future__ import annotations

import numpy as np
import pandas as pd

from src.benchmark.result import ColumnResult


def compare_columns(
    predicted: pd.DataFrame,
    target: pd.DataFrame,
    *,
    transformed_columns: set[str] | None = None,
    atol: float,
    rtol: float,
) -> tuple[list[ColumnResult], float, float]:
    results: list[ColumnResult] = []
    transformed_set = set(transformed_columns) if transformed_columns else set()
    if len(predicted) != len(target):
        error = f"row count mismatch: predicted {len(predicted)}, expected {len(target)}"
        return [
            ColumnResult(column=c, correct=False, is_transformed=c in transformed_set, error=error)
            for c in target.columns
        ], 0.0, 0.0

    for column in target.columns:
        is_trans = column in transformed_set
        if column not in predicted.columns:
            results.append(ColumnResult(column=column, correct=False, is_transformed=is_trans, error="missing column"))
            continue
        pred = predicted[column].reset_index(drop=True)
        truth = target[column].reset_index(drop=True)
        try:
            if pd.api.types.is_numeric_dtype(truth.dtype):
                pred_numeric = pd.to_numeric(pred, errors="raise").to_numpy(dtype=float)
                truth_numeric = truth.to_numpy(dtype=float)
                correct = bool(
                    np.allclose(pred_numeric, truth_numeric, atol=atol, rtol=rtol, equal_nan=False)
                )
            else:
                # Object/string/category representations are compatible if their values agree.
                correct = bool(
                    not pred.isna().any()
                    and not truth.isna().any()
                    and pred.astype("string").equals(truth.astype("string"))
                )
            results.append(ColumnResult(column=column, correct=correct, is_transformed=is_trans))
        except Exception as exc:
            results.append(ColumnResult(column=column, correct=False, is_transformed=is_trans, error=str(exc)))

    overall_accuracy = sum(result.correct for result in results) / len(results) if results else 1.0
    if transformed_set:
        trans_results = [r for r in results if r.is_transformed]
        trans_accuracy = sum(r.correct for r in trans_results) / len(trans_results) if trans_results else 1.0
    else:
        trans_accuracy = overall_accuracy
    return results, overall_accuracy, trans_accuracy
