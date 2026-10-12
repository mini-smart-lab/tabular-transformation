from pathlib import Path
import pandas as pd
import pytest

from src.benchmark.paper_tables import (
    generate_complexity_latex,
    generate_domain_latex,
    generate_ablation_latex,
    export_paper_tables,
)


def test_paper_tables_generation(tmp_path):
    df = pd.DataFrame([
        {
            "run_id": "r1",
            "task_id": "0",
            "level": 0,
            "source_dataset": "california_housing",
            "domain": "Real Estate / Housing",
            "ablation_mode": "standard",
            "transformed_column_accuracy": 1.0,
            "functional_column_accuracy": 1.0,
            "operator_f1": 1.0,
            "exact_sequence_accuracy": 1.0,
            "total_tokens": 12000,
            "tool_calls": 3,
        },
        {
            "run_id": "r2",
            "task_id": "1",
            "level": 1,
            "source_dataset": "diabetes",
            "domain": "Healthcare / Endocrinology",
            "ablation_mode": "no_tools",
            "transformed_column_accuracy": 0.5,
            "functional_column_accuracy": 0.8,
            "operator_f1": 0.67,
            "exact_sequence_accuracy": 0.5,
            "total_tokens": 5000,
            "tool_calls": 1,
        },
        {
            "run_id": "r3",
            "task_id": "2",
            "level": 2,
            "source_dataset": "wine_quality",
            "domain": "Food Science / Chemistry",
            "ablation_mode": "oracle_ops",
            "transformed_column_accuracy": 1.0,
            "functional_column_accuracy": 1.0,
            "operator_f1": 1.0,
            "exact_sequence_accuracy": 1.0,
            "total_tokens": 8000,
            "tool_calls": 2,
        },
    ])

    summary_csv = tmp_path / "summary.csv"
    df.to_csv(summary_csv, index=False)

    comp_tex = generate_complexity_latex(df)
    assert "\\begin{table*}" in comp_tex
    assert "Class 0: Atomic Unary" in comp_tex
    assert "Class 1: Independent Unary" in comp_tex

    domain_tex = generate_domain_latex(df)
    assert "\\begin{table*}" in domain_tex
    assert "Real Estate / Housing" in domain_tex
    assert "california_housing" in domain_tex

    ablation_tex = generate_ablation_latex(df)
    assert "\\begin{table}" in ablation_tex
    assert "Standard (Full Two-Tier Agent)" in ablation_tex
    assert "Direct Generation (No Execution Tools)" in ablation_tex
    assert "Oracle Operators (Code Synthesis Only)" in ablation_tex

    out_dir = tmp_path / "tables"
    exported = export_paper_tables(summary_csv, out_dir)
    assert len(exported) == 3
    for name, path in exported.items():
        assert path.is_file()
        assert path.stat().st_size > 0
