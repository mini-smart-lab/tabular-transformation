r"""Generate publication-ready Multi-Model Ablation Study Table for ICIIT 2027 paper."""

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

def compute_metrics(csv_path: Path):
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    n_tasks = len(df)
    submitted = df[df["status"] == "submitted"]
    sub_count = len(submitted)
    sub_rate = sub_count / n_tasks if n_tasks else 0.0

    # Penalized Transformed Accuracy: failed tasks count as 0.0
    has_trans = "transformed_column_accuracy" in df.columns
    trans_col = "transformed_column_accuracy" if has_trans else "functional_column_accuracy"
    pen_trans = float(df[trans_col].fillna(0.0).mean()) if n_tasks else 0.0

    # Conditional metrics on submitted tasks (runtime errors in Tier-2 evaluate to 0.0)
    cond_trans = float(submitted[trans_col].fillna(0.0).mean()) if sub_count else 0.0
    cond_func = float(submitted["functional_column_accuracy"].fillna(0.0).mean()) if sub_count else 0.0
    cond_f1 = float(submitted["operator_f1"].fillna(0.0).mean()) if sub_count and "operator_f1" in submitted.columns else 0.0
    
    seq_col = "exact_sequence_accuracy" if "exact_sequence_accuracy" in submitted.columns else None
    cond_seq = float(submitted[seq_col].fillna(0.0).mean()) if sub_count and seq_col else 0.0

    return {
        "n_tasks": n_tasks,
        "sub_count": sub_count,
        "sub_rate": sub_rate,
        "cond_trans": cond_trans,
        "pen_trans": pen_trans,
        "cond_func": cond_func,
        "cond_f1": cond_f1,
        "cond_seq": cond_seq,
    }

def build_latex_table(runs, condition_labels):
    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Multi-Model Ablation Study across Gemini 2.0 Flash, Claude 3.5 Sonnet, and GPT-4o mini: Disentangling Discovery Hypothesis Search from Interactive Sandbox Feedback and Code Synthesis ($N=48$). We report Submissions, Conditional Transformed Accuracy ($\text{Acc}_{\text{cond}}$), Penalized Transformed Accuracy ($\text{Acc}_{\text{pen}} = \text{Submission Rate} \times \text{Acc}_{\text{cond}}$), Functional Accuracy, Operator F1, and Column Exact Sequence Accuracy conditioned on submitted tasks.}",
        r"\label{tab:ablation_study}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Ablation Condition} & \textbf{Submissions} & \textbf{Trans. (Cond)} & \textbf{Trans. (Pen)} & \textbf{Func. (Cond)} & \textbf{Op. F1 (Cond)} & \textbf{Seq. (Cond)} \\",
        r"\midrule",
    ]

    for model_name, conds in runs.items():
        first = True
        for cond_key, path in conds.items():
            metrics = compute_metrics(path)
            c_label = condition_labels[cond_key]
            if metrics is None:
                continue
            sub_str = f"{metrics['sub_count']}/{metrics['n_tasks']} ({metrics['sub_rate']*100:.1f}\%)"
            m_col = f"\\textbf{{{model_name}}}" if first else ""
            latex.append(
                f"{m_col} & {c_label} & {sub_str} & {metrics['cond_trans']:.3f} & {metrics['pen_trans']:.3f} & {metrics['cond_func']:.3f} & {metrics['cond_f1']:.3f} & {metrics['cond_seq']:.3f} \\\\"
            )
            first = False
        latex.append(r"\midrule")

    # Remove trailing midrule and add bottomrule
    if latex[-1] == r"\midrule":
        latex.pop()
    latex.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ])
    return "\n".join(latex)

def main():
    runs = {
        "Gemini 2.0 Flash": {
            "no_tools": ROOT / "pipeline_runs" / "run_gemini3_flash_ablation_no_tools_48tasks" / "reports" / "summary.csv",
            "full_agent": ROOT / "pipeline_runs" / "run_gemini3_flash_48tasks" / "reports" / "summary.csv",
            "oracle_ops": ROOT / "pipeline_runs" / "run_gemini3_flash_ablation_oracle_ops_48tasks" / "reports" / "summary.csv",
        },
        "Claude 3.5 Sonnet": {
            "no_tools": ROOT / "pipeline_runs_revision" / "run_claude_sonnet_ablation_no_tools_48tasks" / "reports" / "summary.csv",
            "full_agent": ROOT / "pipeline_runs" / "run_claude_sonnet_48tasks" / "reports" / "summary.csv",
            "oracle_ops": ROOT / "pipeline_runs_revision" / "run_claude_sonnet_ablation_oracle_ops_48tasks" / "reports" / "summary.csv",
        },
        "GPT-4o mini": {
            "no_tools": ROOT / "pipeline_runs_revision" / "run_gpt5_nano_ablation_no_tools_48tasks" / "reports" / "summary.csv",
            "full_agent": ROOT / "pipeline_runs" / "run_gpt5_nano_48tasks" / "reports" / "summary.csv",
            "oracle_ops": ROOT / "pipeline_runs_revision" / "run_gpt5_nano_ablation_oracle_ops_48tasks" / "reports" / "summary.csv",
        },
    }

    condition_labels = {
        "no_tools": "Direct Symbolic Gen. (No Tools)",
        "full_agent": "Standard Interactive Discovery (Full)",
        "oracle_ops": "Oracle Operators Revealed",
    }

    print(f"{'Model':<20} | {'Condition':<35} | {'Submissions':<14} | {'Trans(C)':<8} | {'Trans(P)':<8} | {'Func(C)':<8} | {'Op F1':<8} | {'Seq(C)':<8}")
    print("-" * 120)

    for model_name, conds in runs.items():
        first = True
        for cond_key, path in conds.items():
            metrics = compute_metrics(path)
            c_label = condition_labels[cond_key]
            if metrics is None:
                print(f"{model_name:<20} | {c_label:<35} | PENDING...")
                continue
            sub_str = f"{metrics['sub_count']}/{metrics['n_tasks']} ({metrics['sub_rate']*100:.1f}%)"
            print(f"{model_name if first else '':<20} | {c_label:<35} | {sub_str:<14} | {metrics['cond_trans']:<8.3f} | {metrics['pen_trans']:<8.3f} | {metrics['cond_func']:<8.3f} | {metrics['cond_f1']:<8.3f} | {metrics['cond_seq']:<8.3f}")
            first = False

    latex_code = build_latex_table(runs, condition_labels)
    out_table = ROOT / "paper" / "iciit" / "draft" / "tables" / "table_ablation_study.tex"
    out_table.write_text(latex_code, encoding="utf-8")
    print(f"\nWritten LaTeX table to: {out_table}")

if __name__ == "__main__":
    main()
