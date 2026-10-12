r"""Generate publication-ready LaTeX tables for the research manuscript.

Reads summary CSVs from the completed benchmark runs across Frontier LLM,
Lightweight LLM, Baseline, and Ablation conditions.
"""

from pathlib import Path
import pandas as pd


def load_runs() -> dict[str, pd.DataFrame]:
    runs = {
        "brute_force": Path("pipeline_runs/run_brute_force_baseline_24tasks/reports/summary.csv"),
        "gpt5_nano": Path("pipeline_runs/run_gpt5_nano_24tasks/reports/summary.csv"),
        "gemini3_flash": Path("pipeline_runs/run_gemini3_flash_v2_24tasks/reports/summary.csv"),
        "ablation_no_tools": Path("pipeline_runs/run_gemini3_flash_ablation_no_tools_24tasks/reports/summary.csv"),
        "ablation_oracle_ops": Path("pipeline_runs/run_gemini3_flash_ablation_oracle_ops_24tasks/reports/summary.csv"),
    }
    dfs = {}
    for k, p in runs.items():
        if p.exists():
            dfs[k] = pd.read_csv(p)
    return dfs


def build_table_model_comparison(dfs: dict[str, pd.DataFrame]) -> str:
    """Table 1: Main comparative evaluation between Frontier, Lightweight, and Classical Baselines."""
    models = [
        ("Gemini 3 Flash (Google)", "Frontier Multimodal Agent", "gemini3_flash"),
        ("GPT-5.4 Nano (OpenAI)", "Lightweight Edge Model", "gpt5_nano"),
        ("Symbolic Enumeration", "Combinatorial Brute Force", "brute_force"),
    ]
    rows = []
    for display_name, m_type, key in models:
        df = dfs.get(key)
        if df is None:
            continue
        n_tasks = len(df)
        sub_pct = (df["status"] == "submitted").mean() * 100
        trans_acc = df["transformed_column_accuracy"].mean()
        func_acc = df["functional_column_accuracy"].mean()
        f1 = df["operator_f1"].mean()
        
        # Per level accuracy
        lvl_accs = []
        for lvl in range(6):
            subset = df[df["level"] == lvl]
            mean_acc = subset["transformed_column_accuracy"].mean()
            lvl_accs.append(f"{mean_acc:.3f}" if pd.notna(mean_acc) else "N/A")
            
        rows.append(
            f"{display_name} & {m_type} & {sub_pct:.1f}\\% & {trans_acc:.3f} & {func_acc:.3f} & {f1:.3f} & "
            + " & ".join(lvl_accs)
            + r" \\"
        )

    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Comparative Evaluation of Frontier Agent, Lightweight Model, and Classical Baseline across 6 Structural Complexity Classes ($N=24$ Tasks, 8 Domains).}",
        r"\label{tab:main_model_comparison}",
        r"\begin{tabular}{llcccccccccc}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Paradigm}} & \textbf{Sub.} & \textbf{Trans.} & \textbf{Func.} & \textbf{Op.} & \multicolumn{6}{c}{\textbf{Transformed Col. Accuracy by Complexity Class}} \\",
        r"\cmidrule(lr){7-12}",
        r" & & \textbf{Rate} & \textbf{Acc.} & \textbf{Acc.} & \textbf{F1} & \textbf{C0} & \textbf{C1} & \textbf{C2} & \textbf{C3} & \textbf{C4} & \textbf{C5} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(latex)


def build_table_ablation(dfs: dict[str, pd.DataFrame]) -> str:
    """Table 2: Ablation Study on Gemini 3 Flash disentangling discovery reasoning from tool execution."""
    modes = [
        ("Direct Symbolic Generation (No Tools)", "ablation_no_tools", "Pure LLM parametric knowledge; no sandbox code execution"),
        ("Standard Interactive Discovery (Full Agent)", "gemini3_flash", "Two-tier sandbox with iterative hypothesis testing via run_python"),
        ("Oracle Operators Revealed (Synthesis Only)", "ablation_oracle_ops", "Ground-truth operator names revealed; isolates code synthesis"),
    ]
    rows = []
    for label, key, desc in modes:
        df = dfs.get(key)
        if df is None:
            continue
        n_tasks = len(df)
        sub_cnt = sum(df["status"] == "submitted")
        trans_acc = df["transformed_column_accuracy"].mean()
        func_acc = df["functional_column_accuracy"].mean()
        f1 = df["operator_f1"].mean()
        seq_acc = df["exact_sequence_accuracy"].mean() if "exact_sequence_accuracy" in df else 0.0
        
        sub_row = f"{label} & {sub_cnt}/{n_tasks} & {trans_acc:.3f} & {func_acc:.3f} & {f1:.3f} & {seq_acc:.3f} \\\\"
        desc_row = f"\\multicolumn{{6}}{{p{{0.92\\textwidth}}}}{{\\footnotesize \\textit{{{desc}}}}} \\\\"
        rows.append(f"{sub_row}\n{desc_row}")

    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Ablation Study on Gemini 3 Flash: Disentangling Discovery Hypothesis Search from Interactive Sandbox Feedback and Code Synthesis ($N=24$).}",
        r"\label{tab:ablation_study}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Ablation Condition} & \textbf{Submissions} & \textbf{Transformed Acc.} & \textbf{Functional Acc.} & \textbf{Operator F1} & \textbf{Exact Seq.} \\",
        r"\midrule",
        "\n\\midrule\n".join(rows),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(latex)


def build_table_domain_breakdown(dfs: dict[str, pd.DataFrame]) -> str:
    """Table 3: Multi-Domain Robustness across the 8 real-world domains."""
    df = dfs.get("gemini3_flash")
    if df is None:
        return ""
    
    rows = []
    for domain, group in sorted(df.groupby("domain")):
        ds_names = ", ".join(sorted(group["source_dataset"].unique())).replace("_", r"\_")
        n = len(group)
        sub_cnt = sum(group["status"] == "submitted")
        t_acc = group["transformed_column_accuracy"].mean()
        f_acc = group["functional_column_accuracy"].mean()
        f1 = group["operator_f1"].mean()
        
        t_acc_str = f"{t_acc:.3f}" if pd.notna(t_acc) else "--"
        f_acc_str = f"{f_acc:.3f}" if pd.notna(f_acc) else "--"
        f1_str = f"{f1:.3f}" if pd.notna(f1) else "--"
        
        rows.append(
            f"{domain} & {ds_names} & {n} & {sub_cnt}/{n} & {t_acc_str} & {f_acc_str} & {f1_str} \\\\"
        )
    
    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Multi-Domain Benchmark Evaluation on Gemini 3 Flash Across 8 Real-World Datasets.}",
        r"\label{tab:domain_breakdown}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Domain} & \textbf{Source Datasets} & \textbf{Tasks} & \textbf{Submissions} & \textbf{Transformed Acc.} & \textbf{Functional Acc.} & \textbf{Operator F1} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(latex)


def main():
    tables_dir = Path("paper/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    dfs = load_runs()
    
    t1 = build_table_model_comparison(dfs)
    (tables_dir / "table_main_model_comparison.tex").write_text(t1, encoding="utf-8")
    print(f"Exported: {tables_dir / 'table_main_model_comparison.tex'}")
    
    t2 = build_table_ablation(dfs)
    (tables_dir / "table_ablation_study.tex").write_text(t2, encoding="utf-8")
    print(f"Exported: {tables_dir / 'table_ablation_study.tex'}")
    
    t3 = build_table_domain_breakdown(dfs)
    (tables_dir / "table_domain_breakdown.tex").write_text(t3, encoding="utf-8")
    print(f"Exported: {tables_dir / 'table_domain_breakdown.tex'}")


if __name__ == "__main__":
    main()
