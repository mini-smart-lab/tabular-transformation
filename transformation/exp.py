from transformation.base import Transformation
import numpy as np

class ExpTransformation(Transformation):
    def is_applicable(self, data):
        arr = np.asarray(data, dtype=float)
        # Prevent overflow by restricting max value to 50
        return bool(arr.size and np.isfinite(arr).all() and (arr < 50.0).all())

    def transform(self, data):
        return np.exp(data)

    def __str__(self):
        return "ExpTransformation"
