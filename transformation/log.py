from transformation.base import Transformation
import numpy as np

class LogTransformation(Transformation):
    def transform(self, data):
        return np.log(data)