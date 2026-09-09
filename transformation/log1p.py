from transformation.base import Transformation
import numpy as np

class Log1pTransformation(Transformation):
    def is_applicable(self, data):
        arr = np.asarray(data, dtype=float)
        return bool(arr.size and np.isfinite(arr).all() and (arr > -1.0).all())

    def transform(self, data):
        return np.log1p(data)

    def __str__(self):
        return "Log1pTransformation"
