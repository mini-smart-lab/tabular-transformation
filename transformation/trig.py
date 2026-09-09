from transformation.base import Transformation
import numpy as np

class SinTransformation(Transformation):
    def is_applicable(self, data):
        arr = np.asarray(data, dtype=float)
        return bool(arr.size and np.isfinite(arr).all())

    def transform(self, data):
        return np.sin(data)

    def __str__(self):
        return "SinTransformation"


class CosTransformation(Transformation):
    def is_applicable(self, data):
        arr = np.asarray(data, dtype=float)
        return bool(arr.size and np.isfinite(arr).all())

    def transform(self, data):
        return np.cos(data)

    def __str__(self):
        return "CosTransformation"


class TanhTransformation(Transformation):
    def is_applicable(self, data):
        arr = np.asarray(data, dtype=float)
        return bool(arr.size and np.isfinite(arr).all())

    def transform(self, data):
        return np.tanh(data)

    def __str__(self):
        return "TanhTransformation"


class SigmoidTransformation(Transformation):
    def is_applicable(self, data):
        arr = np.asarray(data, dtype=float)
        return bool(arr.size and np.isfinite(arr).all())

    def transform(self, data):
        arr = np.asarray(data, dtype=float)
        # Numerically stable sigmoid
        return 1.0 / (1.0 + np.exp(-np.clip(arr, -500, 500)))

    def __str__(self):
        return "SigmoidTransformation"


class AbsTransformation(Transformation):
    def is_applicable(self, data):
        arr = np.asarray(data, dtype=float)
        return bool(arr.size and np.isfinite(arr).all())

    def transform(self, data):
        return np.abs(data)

    def __str__(self):
        return "AbsTransformation"
