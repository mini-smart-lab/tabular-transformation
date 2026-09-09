from transformation.base import Transformation
import numpy as np

class StandardScalerTransformation(Transformation):
    def is_applicable(self, data):
        arr = np.asarray(data, dtype=float)
        if not (arr.size > 1 and np.isfinite(arr).all()):
            return False
        return bool(np.std(arr) > 1e-6)

    def transform(self, data):
        arr = np.asarray(data, dtype=float)
        std = np.std(arr)
        if std < 1e-12:
            return arr - np.mean(arr)
        return (arr - np.mean(arr)) / std

    def __str__(self):
        return "StandardScalerTransformation"


class MinMaxScalerTransformation(Transformation):
    def is_applicable(self, data):
        arr = np.asarray(data, dtype=float)
        if not (arr.size > 1 and np.isfinite(arr).all()):
            return False
        ptp = np.ptp(arr)
        return bool(ptp > 1e-6)

    def transform(self, data):
        arr = np.asarray(data, dtype=float)
        ptp = np.ptp(arr)
        if ptp < 1e-12:
            return np.zeros_like(arr)
        return (arr - np.min(arr)) / ptp

    def __str__(self):
        return "MinMaxScalerTransformation"
