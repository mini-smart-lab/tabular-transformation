from __future__ import annotations

from typing import Any
import numpy as np
from transformation.base import Transformation


class StandardScalerTransformation(Transformation):
    """Standardize features by removing the mean and scaling to unit variance.
    
    Parameters (mean, std) are fitted strictly on training data to prevent test leakage.
    """

    def __init__(self, mean: float | None = None, std: float | None = None):
        self.mean = mean
        self.std = std

    def fit(self, data) -> "StandardScalerTransformation":
        arr = np.asarray(data, dtype=float)
        self.mean = float(np.mean(arr))
        std_val = float(np.std(arr))
        self.std = std_val if std_val > 1e-12 else 1.0
        return self

    def get_params(self) -> dict[str, Any]:
        return {"mean": self.mean, "std": self.std}

    def is_applicable(self, data) -> bool:
        arr = np.asarray(data, dtype=float)
        if not (arr.size > 1 and np.isfinite(arr).all()):
            return False
        return bool(np.std(arr) > 1e-6)

    def transform(self, data):
        arr = np.asarray(data, dtype=float)
        mean_val = self.mean if self.mean is not None else float(np.mean(arr))
        std_val = self.std if self.std is not None else float(np.std(arr))
        if std_val < 1e-12:
            return arr - mean_val
        return (arr - mean_val) / std_val

    def __str__(self):
        return "StandardScalerTransformation"


class MinMaxScalerTransformation(Transformation):
    """Transform features by scaling each feature to a given range [0, 1].
    
    Parameters (min, ptp) are fitted strictly on training data to prevent test leakage.
    """

    def __init__(self, min_val: float | None = None, ptp_val: float | None = None):
        self.min_val = min_val
        self.ptp_val = ptp_val

    def fit(self, data) -> "MinMaxScalerTransformation":
        arr = np.asarray(data, dtype=float)
        self.min_val = float(np.min(arr))
        ptp = float(np.ptp(arr))
        self.ptp_val = ptp if ptp > 1e-12 else 1.0
        return self

    def get_params(self) -> dict[str, Any]:
        return {"min": self.min_val, "ptp": self.ptp_val}

    def is_applicable(self, data) -> bool:
        arr = np.asarray(data, dtype=float)
        if not (arr.size > 1 and np.isfinite(arr).all()):
            return False
        ptp = np.ptp(arr)
        return bool(ptp > 1e-6)

    def transform(self, data):
        arr = np.asarray(data, dtype=float)
        min_v = self.min_val if self.min_val is not None else float(np.min(arr))
        ptp_v = self.ptp_val if self.ptp_val is not None else float(np.ptp(arr))
        if ptp_v < 1e-12:
            return np.zeros_like(arr)
        return (arr - min_v) / ptp_v

    def __str__(self):
        return "MinMaxScalerTransformation"
