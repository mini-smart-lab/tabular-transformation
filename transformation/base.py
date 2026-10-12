import abc
from typing import Any


class Transformation(metaclass=abc.ABCMeta):
    def fit(self, data) -> "Transformation":
        """Fit statistical parameters (e.g. mean, std, min, ptp) on the training set. Default is no-op."""
        return self

    def fit_transform(self, data):
        """Fit parameters on data and transform it."""
        return self.fit(data).transform(data)

    def get_params(self) -> dict[str, Any]:
        """Return fitted or configured parameters."""
        return {}

    def set_params(self, **params) -> "Transformation":
        """Set parameters directly."""
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self

    def is_applicable(self, data) -> bool:
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
