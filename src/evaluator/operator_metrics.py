from __future__ import annotations

from collections import Counter

from src.agent.schemas import Transformation
from src.benchmark.result import OperatorMetrics


def compute_operator_metrics(
    predicted: list[Transformation], ground_truth: list[Transformation]
) -> OperatorMetrics:
    pred_map = {item.column: item.operators for item in predicted}
    truth_map = {item.column: item.operators for item in ground_truth}

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
