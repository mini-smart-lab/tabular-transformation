from transformation.base import Transformation
import numpy as np

class SquareRootTransformation(Transformation):
    def is_applicable(self, data):
        return bool(np.asarray(data).size and np.isfinite(data).all() and (np.asarray(data) >= 0).all())

    def transform(self, data):
        return np.sqrt(data)

    def __str__(self):
        return "SquareRootTransformation"
