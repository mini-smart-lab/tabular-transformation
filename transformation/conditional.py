from __future__ import annotations

from typing import Any
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
    """1.0 if x > mean(x) else 0.0.
    
    Mean threshold is fitted strictly on training data to prevent test leakage.
    """

    def __init__(self, mean_threshold: float | None = None):
        self.mean_threshold = mean_threshold

    def fit(self, data) -> "IndicatorMeanTransformation":
        arr = np.asarray(data, dtype=float)
        self.mean_threshold = float(np.mean(arr))
        return self

    def get_params(self) -> dict[str, Any]:
        return {"threshold": self.mean_threshold}

    def is_applicable(self, data) -> bool:
        arr = np.asarray(data, dtype=float)
        if not (arr.size and np.isfinite(arr).all()):
            return False
        mean_val = float(np.mean(arr))
        gt_mean = arr > mean_val
        return bool(gt_mean.any() and not gt_mean.all())

    def transform(self, data):
        arr = np.asarray(data, dtype=float)
        thresh = self.mean_threshold if self.mean_threshold is not None else float(np.mean(arr))
        return (arr > thresh).astype(float)

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
    """col_a if col_a > mean(col_a) else col_b.
    
    Condition threshold on col_a is fitted strictly on training data.
    """

    def __init__(self, threshold_a: float | None = None):
        self.threshold_a = threshold_a

    def fit_columns(self, data_a, data_b) -> "ConditionalInteractionTransformation":
        arr_a = np.asarray(data_a, dtype=float)
        self.threshold_a = float(np.mean(arr_a))
        return self

    def get_params(self) -> dict[str, Any]:
        return {"threshold_a": self.threshold_a}

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
        thresh = self.threshold_a if self.threshold_a is not None else float(np.mean(arr_a))
        return np.where(arr_a > thresh, arr_a, arr_b)

    def __str__(self):
        return "ConditionalInteractionTransformation"
