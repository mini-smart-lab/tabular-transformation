from transformation.base import Transformation
import numpy as np

class LogTransformation(Transformation):
    def transform(self, data):
        return np.log(data)

    def __str__(self):
        return "LogTransformation"