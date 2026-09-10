from __future__ import annotations

from collections import Counter

from src.agent.schemas import Transformation
from src.benchmark.result import OperatorMetrics


_OPERATOR_ALIASES: dict[str, str] = {
    "log": "LogTransformation",
    "logtransformation": "LogTransformation",
    "np.log": "LogTransformation",
    "log1p": "Log1pTransformation",
    "log1ptransformation": "Log1pTransformation",
    "np.log1p": "Log1pTransformation",
    "sqrt": "SquareRootTransformation",
    "squareroot": "SquareRootTransformation",
    "squareroottransformation": "SquareRootTransformation",
    "np.sqrt": "SquareRootTransformation",
    "square": "SquareTransformation",
    "squaretransformation": "SquareTransformation",
    "cube": "CubeTransformation",
    "cubetransformation": "CubeTransformation",
    "cbrt": "CubeRootTransformation",
    "cuberoottransformation": "CubeRootTransformation",
    "np.cbrt": "CubeRootTransformation",
    "exp": "ExpTransformation",
    "exptransformation": "ExpTransformation",
    "np.exp": "ExpTransformation",
    "inv": "InverseTransformation",
    "inverse": "InverseTransformation",
    "inversetransformation": "InverseTransformation",
    "standard_scaler": "StandardScalerTransformation",
    "standardscaler": "StandardScalerTransformation",
    "standardscalertransformation": "StandardScalerTransformation",
    "min_max_scaler": "MinMaxScalerTransformation",
    "minmaxscaler": "MinMaxScalerTransformation",
    "minmaxscalertransformation": "MinMaxScalerTransformation",
    "sin": "SinTransformation",
    "sintransformation": "SinTransformation",
    "np.sin": "SinTransformation",
    "cos": "CosTransformation",
    "costransformation": "CosTransformation",
    "np.cos": "CosTransformation",
    "tanh": "TanhTransformation",
    "tanhtransformation": "TanhTransformation",
    "np.tanh": "TanhTransformation",
    "sigmoid": "SigmoidTransformation",
    "sigmoidtransformation": "SigmoidTransformation",
    "abs": "AbsTransformation",
    "abstransformation": "AbsTransformation",
    "np.abs": "AbsTransformation",
    "add": "AddInteractionTransformation",
    "addinteraction": "AddInteractionTransformation",
    "addinteractiontransformation": "AddInteractionTransformation",
    "subtract": "SubtractInteractionTransformation",
    "subtractinteraction": "SubtractInteractionTransformation",
    "subtractinteractiontransformation": "SubtractInteractionTransformation",
    "multiply": "MultiplyInteractionTransformation",
    "multiplyinteraction": "MultiplyInteractionTransformation",
    "multiplyinteractiontransformation": "MultiplyInteractionTransformation",
    "divide": "DivideInteractionTransformation",
    "divideinteraction": "DivideInteractionTransformation",
    "divideinteractiontransformation": "DivideInteractionTransformation",
    "indicator_zero": "IndicatorZeroTransformation",
    "indicatorzero": "IndicatorZeroTransformation",
    "indicatorzerotransformation": "IndicatorZeroTransformation",
    "indicator_mean": "IndicatorMeanTransformation",
    "indicatormean": "IndicatorMeanTransformation",
    "indicatormeantransformation": "IndicatorMeanTransformation",
    "clip_zero": "ClipZeroTransformation",
    "clipzero": "ClipZeroTransformation",
    "clipzerotransformation": "ClipZeroTransformation",
    "relu": "ClipZeroTransformation",
    "piecewise_log": "PiecewiseLogTransformation",
    "piecewiselog": "PiecewiseLogTransformation",
    "piecewiselogtransformation": "PiecewiseLogTransformation",
    "conditional_interaction": "ConditionalInteractionTransformation",
    "conditionalinteraction": "ConditionalInteractionTransformation",
    "conditionalinteractiontransformation": "ConditionalInteractionTransformation",
}



def canonicalize_operator(op: str) -> str:
    cleaned = op.strip()
    return _OPERATOR_ALIASES.get(cleaned.lower(), _OPERATOR_ALIASES.get(cleaned, cleaned))


def compute_operator_metrics(
    predicted: list[Transformation], ground_truth: list[Transformation]
) -> OperatorMetrics:
    pred_map = {item.column: [canonicalize_operator(op) for op in item.operators] for item in predicted}
    truth_map = {item.column: [canonicalize_operator(op) for op in item.operators] for item in ground_truth}

    true_positive = 0
    predicted_total = 0
    truth_total = 0
    for column in pred_map.keys() | truth_map.keys():
        pred_counts = Counter(pred_map.get(column, []))
        truth_counts = Counter(truth_map.get(column, []))
        true_positive += sum((pred_counts & truth_counts).values())
        predicted_total += sum(pred_counts.values())
        truth_total += sum(truth_counts.values())

    precision = true_positive / predicted_total if predicted_total else float(truth_total == 0)
    recall = true_positive / truth_total if truth_total else float(predicted_total == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    columns = pred_map.keys() | truth_map.keys()
    exact = (
        sum(pred_map.get(column) == truth_map.get(column) for column in columns) / len(columns)
        if columns
        else 1.0
    )
    return OperatorMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        exact_sequence_accuracy=exact,
    )
