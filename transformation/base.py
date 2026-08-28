import abc

class Transformation(metaclass=abc.ABCMeta):
    def is_applicable(self, data):
        """Return whether applying this operator produces finite numeric values."""
        import numpy as np
        try:
            values = np.asarray(self.transform(data), dtype=float)
            return bool(values.size and np.isfinite(values).all())
        except (TypeError, ValueError, FloatingPointError, OverflowError):
            return False

    @abc.abstractmethod
    def transform(self, data):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass
