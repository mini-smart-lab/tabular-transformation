from transformation.base import Transformation
import numpy as np

class SquareTransformation(Transformation):
    def transform(self, data):
        return np.power(data, 2)

    def __str__(self):
        return "SquareTransformation"