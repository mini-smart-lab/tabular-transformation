from __future__ import annotations

import numpy as np
from transformation.base import Transformation
from transformation.interaction import MultiColumnTransformation


class IndicatorZeroTransformation(Transformation):
    """1.0 if x > 0 else 0.0."""

    def is_applicable(self, data) -> bool:
        arr = np.asarray(data, dtype=float)
        if not (arr.size and np.isfinite(arr).all()):
            return False
        # Ensure non-trivial (not all 0 or all 1)
        gt_zero = arr > 0
        return bool(gt_zero.any() and not gt_zero.all())

    def transform(self, data):
        arr = np.asarray(data, dtype=float)
        return (arr > 0).astype(float)

    def __str__(self):
        return "IndicatorZeroTransformation"


class IndicatorMeanTransformation(Transformation):
    """1.0 if x > mean(x) else 0.0."""

    def is_applicable(self, data) -> bool:
        arr = np.asarray(data, dtype=float)
        if not (arr.size and np.isfinite(arr).all()):
            return False
        mean_val = float(np.mean(arr))
        gt_mean = arr > mean_val
        return bool(gt_mean.any() and not gt_mean.all())

    def transform(self, data):
        arr = np.asarray(data, dtype=float)
        return (arr > np.mean(arr)).astype(float)

    def __str__(self):
        return "IndicatorMeanTransformation"


class ClipZeroTransformation(Transformation):
    """Piecewise ReLU: x if x > 0 else 0.0."""

    def is_applicable(self, data) -> bool:
        arr = np.asarray(data, dtype=float)
        if not (arr.size and np.isfinite(arr).all()):
            return False
        # Should have both positive and negative values to be meaningful
        return bool((arr > 0).any() and (arr < 0).any())

    def transform(self, data):
        arr = np.asarray(data, dtype=float)
        return np.maximum(arr, 0.0)

    def __str__(self):
        return "ClipZeroTransformation"


class PiecewiseLogTransformation(Transformation):
    """Piecewise log: log(x) if x > 1 else x."""

    def is_applicable(self, data) -> bool:
        arr = np.asarray(data, dtype=float)
        if not (arr.size and np.isfinite(arr).all()):
            return False
        # Check that some values are > 1 and some <= 1, and no negative values that would fail log
        return bool((arr > 1.0).any() and (arr >= 0).all())

    def transform(self, data):
        arr = np.asarray(data, dtype=float)
        result = arr.copy()
        mask = arr > 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            result[mask] = np.log(arr[mask])
        return result

    def __str__(self):
        return "PiecewiseLogTransformation"


class ConditionalInteractionTransformation(MultiColumnTransformation):
    """col_a if col_a > mean(col_a) else col_b."""

    def is_applicable_columns(self, data_a, data_b) -> bool:
        arr_a = np.asarray(data_a, dtype=float)
        arr_b = np.asarray(data_b, dtype=float)
        if not (
            arr_a.size == arr_b.size
            and np.isfinite(arr_a).all()
            and np.isfinite(arr_b).all()
        ):
            return False
        mean_a = float(np.mean(arr_a))
        mask = arr_a > mean_a
        return bool(mask.any() and not mask.all())

    def transform_columns(self, data_a, data_b):
        arr_a = np.asarray(data_a, dtype=float)
        arr_b = np.asarray(data_b, dtype=float)
        return np.where(arr_a > np.mean(arr_a), arr_a, arr_b)

    def __str__(self):
        return "ConditionalInteractionTransformation"
