from transformation.base import Transformation
import numpy as np

class MultiColumnTransformation(Transformation):
    def is_applicable_columns(self, data_a, data_b):
        """Check whether interaction between two columns produces valid finite output."""
        try:
            val = self.transform_columns(data_a, data_b)
            arr = np.asarray(val, dtype=float)
            return bool(arr.size and np.isfinite(arr).all())
        except (TypeError, ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
            return False

    def transform(self, data):
        # Fallback if 2D array or tuple passed
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return self.transform_columns(data[0], data[1])
        return data

    def transform_columns(self, data_a, data_b):
        raise NotImplementedError


class AddInteractionTransformation(MultiColumnTransformation):
    def is_applicable_columns(self, data_a, data_b):
        arr_a = np.asarray(data_a, dtype=float)
        arr_b = np.asarray(data_b, dtype=float)
        return bool(
            arr_a.size == arr_b.size
            and np.isfinite(arr_a).all()
            and np.isfinite(arr_b).all()
        )

    def transform_columns(self, data_a, data_b):
        return np.asarray(data_a, dtype=float) + np.asarray(data_b, dtype=float)

    def __str__(self):
        return "AddInteractionTransformation"


class SubtractInteractionTransformation(MultiColumnTransformation):
    def is_applicable_columns(self, data_a, data_b):
        arr_a = np.asarray(data_a, dtype=float)
        arr_b = np.asarray(data_b, dtype=float)
        return bool(
            arr_a.size == arr_b.size
            and np.isfinite(arr_a).all()
            and np.isfinite(arr_b).all()
        )

    def transform_columns(self, data_a, data_b):
        return np.asarray(data_a, dtype=float) - np.asarray(data_b, dtype=float)

    def __str__(self):
        return "SubtractInteractionTransformation"


class MultiplyInteractionTransformation(MultiColumnTransformation):
    def is_applicable_columns(self, data_a, data_b):
        arr_a = np.asarray(data_a, dtype=float)
        arr_b = np.asarray(data_b, dtype=float)
        if not (arr_a.size == arr_b.size and np.isfinite(arr_a).all() and np.isfinite(arr_b).all()):
            return False
        # Avoid overflow
        product = arr_a * arr_b
        return bool(np.isfinite(product).all() and (np.abs(product) < 1e12).all())

    def transform_columns(self, data_a, data_b):
        return np.asarray(data_a, dtype=float) * np.asarray(data_b, dtype=float)

    def __str__(self):
        return "MultiplyInteractionTransformation"


class DivideInteractionTransformation(MultiColumnTransformation):
    def is_applicable_columns(self, data_a, data_b):
        arr_a = np.asarray(data_a, dtype=float)
        arr_b = np.asarray(data_b, dtype=float)
        if not (arr_a.size == arr_b.size and np.isfinite(arr_a).all() and np.isfinite(arr_b).all()):
            return False
        # Disallow near-zero divisor
        return bool((np.abs(arr_b) > 1e-4).all())

    def transform_columns(self, data_a, data_b):
        return np.asarray(data_a, dtype=float) / np.asarray(data_b, dtype=float)

    def __str__(self):
        return "DivideInteractionTransformation"
