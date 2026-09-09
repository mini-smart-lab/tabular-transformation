from transformation.base import Transformation
import numpy as np

class CubeTransformation(Transformation):
    def is_applicable(self, data):
        arr = np.asarray(data, dtype=float)
        # Prevent overflow when cubed
        return bool(arr.size and np.isfinite(arr).all() and (np.abs(arr) < 1e5).all())

    def transform(self, data):
        return np.power(data, 3)

    def __str__(self):
        return "CubeTransformation"


class CubeRootTransformation(Transformation):
    def is_applicable(self, data):
        arr = np.asarray(data, dtype=float)
        return bool(arr.size and np.isfinite(arr).all())

    def transform(self, data):
        return np.cbrt(data)

    def __str__(self):
        return "CubeRootTransformation"


class InverseTransformation(Transformation):
    def is_applicable(self, data):
        arr = np.asarray(data, dtype=float)
        return bool(arr.size and np.isfinite(arr).all() and (np.abs(arr) > 1e-5).all())

    def transform(self, data):
        return 1.0 / data

    def __str__(self):
        return "InverseTransformation"
