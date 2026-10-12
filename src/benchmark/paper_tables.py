from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


COMPLEXITY_CLASS_INFO = {
    0: ("Class 0: Atomic Unary", "Single unary transformation on 1 column"),
    1: ("Class 1: Independent Unary", "Parallel unary transformations across columns"),
    2: ("Class 2: Composed Chains", "Chained unary pipelines $f(g(x))$"),
    3: ("Class 3: Arithmetic Interactions", "Bivariate interactions $x_a \\oplus x_b$"),
    4: ("Class 4: Piecewise / Conditional", "Indicator masks and threshold branching"),
    5: ("Class 5: Deep Hybrid ASTs", "Composite interactions + chaining + conditions"),
}


def generate_complexity_latex(summary_df: pd.DataFrame) -> str:
    """Generate LaTeX table breaking down performance across Structural Complexity Classes."""
    if "level" not in summary_df.columns:
        return "% Missing 'level' column in summary"

    grouped = summary_df.groupby("level")
    rows = []
    for lvl in sorted(summary_df["level"].dropna().unique()):
        lvl_int = int(lvl)
        subset = summary_df[summary_df["level"] == lvl]
        name, desc = COMPLEXITY_CLASS_INFO.get(lvl_int, (f"Class {lvl_int}", "Custom"))
        count = len(subset)
        trans_acc = subset["transformed_column_accuracy"].fillna(0.0).mean()
        func_acc = subset["functional_column_accuracy"].fillna(0.0).mean()
        f1 = subset["operator_f1"].fillna(0.0).mean()
        seq_acc = subset["exact_sequence_accuracy"].fillna(0.0).mean()
        tokens = subset["total_tokens"].dropna().mean() if "total_tokens" in subset else 0
        calls = subset["tool_calls"].dropna().mean() if "tool_calls" in subset else 0

        rows.append(
            f"{name} & {count} & {trans_acc:.3f} & {func_acc:.3f} & {f1:.3f} & {seq_acc:.3f} & {int(tokens):,d} & {calls:.1f} \\\\"
        )

    latex = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Performance Breakdown across Structural Complexity Classes (Failed/Budget-Exceeded Episodes Penalized with 0.0).}",
        "\\label{tab:complexity_breakdown}",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "\\textbf{Complexity Class} & \\textbf{Tasks} & \\textbf{Transformed Acc.} & \\textbf{Functional Acc.} & \\textbf{Operator F1} & \\textbf{Seq. Acc.} & \\textbf{Avg. Tokens} & \\textbf{Avg. Calls} \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ]
    return "\n".join(latex)


def generate_domain_latex(summary_df: pd.DataFrame) -> str:
    """Generate LaTeX table breaking down performance across domains and source datasets."""
    if "domain" not in summary_df.columns:
        return "% Missing 'domain' column in summary"

    rows = []
    for domain in sorted(summary_df["domain"].dropna().unique()):
        subset = summary_df[summary_df["domain"] == domain]
        datasets = ", ".join(sorted(subset["source_dataset"].dropna().unique()))
        count = len(subset)
        trans_acc = subset["transformed_column_accuracy"].fillna(0.0).mean()
        func_acc = subset["functional_column_accuracy"].fillna(0.0).mean()
        f1 = subset["operator_f1"].fillna(0.0).mean()

        rows.append(
            f"{domain} & {datasets} & {count} & {trans_acc:.3f} & {func_acc:.3f} & {f1:.3f} \\\\"
        )

    latex = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Multi-Domain Benchmark Evaluation: Functional and Symbolic Recovery across Domains.}",
        "\\label{tab:domain_breakdown}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "\\textbf{Domain} & \\textbf{Source Datasets} & \\textbf{Tasks} & \\textbf{Transformed Acc.} & \\textbf{Functional Acc.} & \\textbf{Operator F1} \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ]
    return "\n".join(latex)


def generate_ablation_latex(summary_df: pd.DataFrame) -> str:
    """Generate LaTeX table comparing ablation settings (Standard vs No-Tools vs Oracle-Ops)."""
    ablation_col = "ablation_mode" if "ablation_mode" in summary_df.columns else None
    if not ablation_col:
        return "% Missing 'ablation_mode' column in summary"

    mode_labels = {
        "standard": "Standard (Full Two-Tier Agent)",
        "no_tools": "Direct Generation (No Execution Tools)",
        "oracle_ops": "Oracle Operators (Code Synthesis Only)",
    }

    rows = []
    for mode in sorted(summary_df[ablation_col].dropna().unique()):
        subset = summary_df[summary_df[ablation_col] == mode]
        label = mode_labels.get(mode, mode.replace("_", " ").title())
        count = len(subset)
        trans_acc = subset["transformed_column_accuracy"].fillna(0.0).mean()
        func_acc = subset["functional_column_accuracy"].fillna(0.0).mean()
        f1 = subset["operator_f1"].fillna(0.0).mean()
        seq_acc = subset["exact_sequence_accuracy"].fillna(0.0).mean()

        rows.append(
            f"{label} & {count} & {trans_acc:.3f} & {func_acc:.3f} & {f1:.3f} & {seq_acc:.3f} \\\\"
        )

    latex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Ablation Study: Disentangling Discovery Reasoning from Iterative Execution and Code Synthesis.}",
        "\\label{tab:ablation_results}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "\\textbf{Ablation Setting} & \\textbf{Tasks} & \\textbf{Transformed Acc.} & \\textbf{Functional Acc.} & \\textbf{Operator F1} & \\textbf{Seq. Acc.} \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(latex)


def export_paper_tables(summary_csv: Path | str, output_directory: Path | str) -> dict[str, Path]:
    summary_path = Path(summary_csv)
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)
    outputs: dict[str, Path] = {}

    table_comp = out_dir / "table_complexity_breakdown.tex"
    table_comp.write_text(generate_complexity_latex(df), encoding="utf-8")
    outputs["complexity"] = table_comp

    table_domain = out_dir / "table_domain_breakdown.tex"
    table_domain.write_text(generate_domain_latex(df), encoding="utf-8")
    outputs["domain"] = table_domain

    table_ablation = out_dir / "table_ablation_study.tex"
    table_ablation.write_text(generate_ablation_latex(df), encoding="utf-8")
    outputs["ablation"] = table_ablation

    return outputs


def _main() -> None:
    parser = argparse.ArgumentParser(description="Export paper-ready LaTeX tables from benchmark summary")
    parser.add_argument("--summary", type=Path, default=Path("results/summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/tables"))
    args = parser.parse_args()

    files = export_paper_tables(args.summary, args.output_dir)
    for name, path in files.items():
        print(f"Generated {name} table: {path.resolve()}")


if __name__ == "__main__":
    _main()
