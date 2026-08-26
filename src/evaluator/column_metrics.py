from __future__ import annotations

import numpy as np
import pandas as pd

from src.benchmark.result import ColumnResult


def compare_columns(
    predicted: pd.DataFrame,
    target: pd.DataFrame,
    *,
    atol: float,
    rtol: float,
) -> tuple[list[ColumnResult], float]:
    results: list[ColumnResult] = []
    if len(predicted) != len(target):
        error = f"row count mismatch: predicted {len(predicted)}, expected {len(target)}"
        return [ColumnResult(column=c, correct=False, error=error) for c in target.columns], 0.0

    for column in target.columns:
        if column not in predicted.columns:
            results.append(ColumnResult(column=column, correct=False, error="missing column"))
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
            results.append(ColumnResult(column=column, correct=correct))
        except Exception as exc:
            results.append(ColumnResult(column=column, correct=False, error=str(exc)))

    accuracy = sum(result.correct for result in results) / len(results) if results else 1.0
    return results, accuracy
