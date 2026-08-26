import pytest

from src.agent.schemas import Transformation
from src.evaluator.operator_metrics import compute_operator_metrics


def t(column, operators):
    return Transformation(column=column, operators=operators)


def test_exact_sequence():
    metrics = compute_operator_metrics([t("x", ["log", "std"])], [t("x", ["log", "std"])])
    assert metrics.precision == metrics.recall == metrics.f1 == 1
    assert metrics.exact_sequence_accuracy == 1


@pytest.mark.parametrize(
    "prediction,precision,recall",
    [
        ([t("x", ["log"])], 1.0, 0.5),
        ([t("x", ["log", "std", "sqrt"])], 2 / 3, 1.0),
        ([t("x", ["std", "log"])], 1.0, 1.0),
        ([t("y", ["log", "std"])], 0.0, 0.0),
    ],
)
def test_non_exact_cases(prediction, precision, recall):
    metrics = compute_operator_metrics(prediction, [t("x", ["log", "std"])])
    assert metrics.precision == pytest.approx(precision)
    assert metrics.recall == pytest.approx(recall)
    assert metrics.exact_sequence_accuracy == 0
