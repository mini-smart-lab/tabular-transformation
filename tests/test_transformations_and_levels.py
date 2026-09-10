from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from transformation import (
    ExpTransformation,
    Log1pTransformation,
    CubeTransformation,
    CubeRootTransformation,
    InverseTransformation,
    StandardScalerTransformation,
    MinMaxScalerTransformation,
    SinTransformation,
    CosTransformation,
    TanhTransformation,
    SigmoidTransformation,
    AbsTransformation,
    AddInteractionTransformation,
    SubtractInteractionTransformation,
    MultiplyInteractionTransformation,
    DivideInteractionTransformation,
)
from src.benchmark.task_builder import build_task, ALL_OPERATORS
from src.benchmark.task import load_task_directory
from src.evaluator.column_metrics import compare_columns
from src.evaluator.operator_metrics import canonicalize_operator, compute_operator_metrics
from src.agent.schemas import Transformation


def test_unary_transformations():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    
    # Exp
    exp_op = ExpTransformation()
    assert exp_op.is_applicable(data)
    assert np.allclose(exp_op.transform(data), np.exp(data))

    # Log1p
    log1p_op = Log1pTransformation()
    assert log1p_op.is_applicable(data)
    assert np.allclose(log1p_op.transform(data), np.log1p(data))

    # Cube & CubeRoot
    cube_op = CubeTransformation()
    cbrt_op = CubeRootTransformation()
    assert np.allclose(cube_op.transform(data), [1.0, 8.0, 27.0, 64.0])
    assert np.allclose(cbrt_op.transform(cube_op.transform(data)), data)

    # Inverse
    inv_op = InverseTransformation()
    assert inv_op.is_applicable(data)
    assert np.allclose(inv_op.transform(data), 1.0 / data)
    assert not inv_op.is_applicable(np.array([0.0, 1.0]))

    # Scaling
    std_op = StandardScalerTransformation()
    assert std_op.is_applicable(data)
    std_res = std_op.transform(data)
    assert pytest.approx(np.mean(std_res)) == 0.0
    assert pytest.approx(np.std(std_res)) == 1.0

    minmax_op = MinMaxScalerTransformation()
    assert minmax_op.is_applicable(data)
    mm_res = minmax_op.transform(data)
    assert np.min(mm_res) == 0.0
    assert np.max(mm_res) == 1.0

    # Trig and activations
    sin_op = SinTransformation()
    cos_op = CosTransformation()
    tanh_op = TanhTransformation()
    sig_op = SigmoidTransformation()
    abs_op = AbsTransformation()
    assert np.allclose(sin_op.transform(data), np.sin(data))
    assert np.allclose(cos_op.transform(data), np.cos(data))
    assert np.allclose(tanh_op.transform(data), np.tanh(data))
    assert np.allclose(sig_op.transform(data), 1.0 / (1.0 + np.exp(-data)))
    assert np.allclose(abs_op.transform(np.array([-1.0, 2.0])), [1.0, 2.0])


def test_interaction_transformations():
    a = np.array([2.0, 4.0, 6.0])
    b = np.array([1.0, 2.0, 3.0])

    add_op = AddInteractionTransformation()
    sub_op = SubtractInteractionTransformation()
    mul_op = MultiplyInteractionTransformation()
    div_op = DivideInteractionTransformation()

    assert add_op.is_applicable_columns(a, b)
    assert np.allclose(add_op.transform_columns(a, b), [3.0, 6.0, 9.0])
    assert np.allclose(sub_op.transform_columns(a, b), [1.0, 2.0, 3.0])
    assert np.allclose(mul_op.transform_columns(a, b), [2.0, 8.0, 18.0])
    assert np.allclose(div_op.transform_columns(a, b), [2.0, 2.0, 2.0])

    zero_b = np.array([0.0, 1.0, 2.0])
    assert not div_op.is_applicable_columns(a, zero_b)


def test_difficulty_levels_generation(tmp_path: Path):
    # 8 columns synthetic dataset
    df = pd.DataFrame({
        f"col_{i}": np.random.default_rng(i).uniform(1.0, 50.0, size=50)
        for i in range(8)
    })
    source = tmp_path / "source.csv"
    df.to_csv(source, index=False)

    for level in range(1, 6):
        task_dir = tmp_path / f"level_{level}"
        transforms = build_task(source, task_dir, level=level, seed=42 + level)
        assert len(transforms) >= 2
        task = load_task_directory(task_dir)
        assert task.test_before_path.is_file()
        assert task.test_after_path.is_file()
        assert len(task.ground_truth_transformations) == len(transforms)


def test_transformed_column_accuracy():
    target = pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [10.0, 20.0, 30.0],
        "c": [100.0, 200.0, 300.0],
        "d": [1000.0, 2000.0, 3000.0],
    })
    # Say columns b and c were transformed.
    predicted = target.copy()
    # Model got b right, but failed on c.
    predicted["c"] = [999.0, 999.0, 999.0]

    cols, overall_acc, trans_acc = compare_columns(
        predicted,
        target,
        transformed_columns={"b", "c"},
        atol=1e-6,
        rtol=1e-5,
    )
    # Overall: 3 out of 4 columns correct = 0.75
    assert overall_acc == 0.75
    # Transformed: 1 out of 2 transformed columns correct = 0.50
    assert trans_acc == 0.50


def test_canonical_operator_normalization():
    assert canonicalize_operator("log") == "LogTransformation"
    assert canonicalize_operator("np.log1p") == "Log1pTransformation"
    assert canonicalize_operator("StandardScaler") == "StandardScalerTransformation"
    assert canonicalize_operator("sin") == "SinTransformation"
    assert canonicalize_operator("add") == "AddInteractionTransformation"

    pred = [Transformation(column="col_1", operators=["log"])]
    truth = [Transformation(column="col_1", operators=["LogTransformation"])]
    metrics = compute_operator_metrics(pred, truth)
    assert metrics.f1 == 1.0
    assert metrics.exact_sequence_accuracy == 1.0


def test_conditional_transformations():
    from transformation import (
        IndicatorZeroTransformation,
        IndicatorMeanTransformation,
        ClipZeroTransformation,
        PiecewiseLogTransformation,
        ConditionalInteractionTransformation,
    )
    from src.config import get_level_budget, LEVEL_BUDGETS

    # IndicatorZero
    data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    ind_zero = IndicatorZeroTransformation()
    assert ind_zero.is_applicable(data)
    assert np.allclose(ind_zero.transform(data), [0.0, 0.0, 0.0, 1.0, 1.0])

    # IndicatorMean
    ind_mean = IndicatorMeanTransformation()
    assert ind_mean.is_applicable(data)
    # mean is 0.0, so > 0.0 gives [0, 0, 0, 1, 1]
    assert np.allclose(ind_mean.transform(data), [0.0, 0.0, 0.0, 1.0, 1.0])

    # ClipZero (ReLU)
    clip_op = ClipZeroTransformation()
    assert clip_op.is_applicable(data)
    assert np.allclose(clip_op.transform(data), [0.0, 0.0, 0.0, 1.0, 2.0])

    # PiecewiseLog
    pos_data = np.array([0.5, 1.0, 2.0, np.e])
    pw_log = PiecewiseLogTransformation()
    assert pw_log.is_applicable(pos_data)
    assert np.allclose(pw_log.transform(pos_data), [0.5, 1.0, np.log(2.0), 1.0])

    # ConditionalInteraction
    col_a = np.array([10.0, -5.0, 20.0, -10.0])  # mean = 3.75
    col_b = np.array([1.0, 2.0, 3.0, 4.0])
    cond_inter = ConditionalInteractionTransformation()
    assert cond_inter.is_applicable_columns(col_a, col_b)
    # mask: col_a > 3.75 -> [True, False, True, False]
    # out: [10.0, 2.0, 20.0, 4.0]
    assert np.allclose(cond_inter.transform_columns(col_a, col_b), [10.0, 2.0, 20.0, 4.0])

    # Test Level Budgets
    for lvl in range(6):
        budget = get_level_budget(lvl)
        assert budget.max_total_tokens >= 40_000
        assert budget.max_tool_calls >= 8

