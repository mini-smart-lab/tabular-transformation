from transformation.base import Transformation
import numpy as np

class SquareRootTransformation(Transformation):
    def transform(self, data):
        return np.sqrt(data)