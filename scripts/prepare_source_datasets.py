"""Prepare a diverse multi-domain tabular dataset suite for Tabular Transformation Discovery Harness."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml,
    load_breast_cancer,
    load_diabetes,
    load_wine,
)


def sanitize_column_name(col: str) -> str:
    """Normalize column name to lowercase alphanumeric snake_case."""
    col = str(col).strip().lower()
    col = re.sub(r"[^\w\s]", "_", col)
    col = re.sub(r"[\s_]+", "_", col)
    return col.strip("_")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all columns are numeric, snake_cased, with no nulls."""
    df = df.copy()
    # Rename columns cleanly
    df.columns = [sanitize_column_name(c) for c in df.columns]

    # Deduplicate column names if needed
    seen: dict[str, int] = {}
    new_cols: list[str] = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols

    # Keep only numeric columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df = df[numeric_cols]

    # Drop any NaN rows or impute median if few
    df = df.dropna().reset_index(drop=True)
    return df


def compute_dataset_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary statistics including skewness and memory."""
    skewness = df.skew(numeric_only=True).to_dict()
    stats = {
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "columns": list(df.columns),
        "mean_skewness": float(np.mean(list(skewness.values()))) if skewness else 0.0,
        "max_skewness": float(np.max(np.abs(list(skewness.values())))) if skewness else 0.0,
    }
    return stats


def main() -> None:
    output_dir = Path("data/source_datasets").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {output_dir}")

    manifest: dict[str, Any] = {
        "description": "Multi-domain benchmark dataset suite for Tabular Transformation Discovery",
        "created_at": pd.Timestamp.now().isoformat(),
        "datasets": {},
    }

    # 1. California Housing (Real Estate / Economics)
    print("Fetching 1/8: california_housing...")
    cal_data = fetch_california_housing(as_frame=True)
    df_cal = clean_dataframe(cal_data.frame)
    cal_path = output_dir / "california_housing.csv"
    df_cal.to_csv(cal_path, index=False)
    manifest["datasets"]["california_housing"] = {
        "filename": "california_housing.csv",
        "domain": "Real Estate / Economics",
        "description": "Census data on California housing blocks with continuous economic & geospatial features",
        **compute_dataset_stats(df_cal),
    }

    # 2. Diabetes (Biomedical / Clinical)
    print("Fetching 2/8: diabetes...")
    diab_data = load_diabetes(as_frame=True)
    df_diab = clean_dataframe(diab_data.frame)
    diab_path = output_dir / "diabetes.csv"
    df_diab.to_csv(diab_path, index=False)
    manifest["datasets"]["diabetes"] = {
        "filename": "diabetes.csv",
        "domain": "Biomedical / Clinical Medicine",
        "description": "Ten baseline blood serum, physiological, and demographic measurements of diabetes patients",
        **compute_dataset_stats(df_diab),
    }

    # 3. Breast Cancer (Oncology / Pathology)
    print("Fetching 3/8: breast_cancer...")
    bc_data = load_breast_cancer(as_frame=True)
    df_bc = clean_dataframe(bc_data.frame)
    bc_path = output_dir / "breast_cancer.csv"
    df_bc.to_csv(bc_path, index=False)
    manifest["datasets"]["breast_cancer"] = {
        "filename": "breast_cancer.csv",
        "domain": "Oncology / Pathology",
        "description": "Wisconsin diagnostic breast cancer database with 30 collinear morphologic cell features",
        **compute_dataset_stats(df_bc),
    }

    # 4. Wine Quality (Food Chemistry / Agriculture)
    print("Fetching 4/8: wine_quality...")
    wine_data = load_wine(as_frame=True)
    df_wine = clean_dataframe(wine_data.frame)
    wine_path = output_dir / "wine_quality.csv"
    df_wine.to_csv(wine_path, index=False)
    manifest["datasets"]["wine_quality"] = {
        "filename": "wine_quality.csv",
        "domain": "Food Chemistry / Agriculture",
        "description": "Chemical analysis of Italian wines grown across three cultivars",
        **compute_dataset_stats(df_wine),
    }

    # 5. Banknote Authentication (Signal Processing / Security)
    print("Fetching 5/8: banknote_authentication...")
    bn_data = fetch_openml(name="banknote-authentication", as_frame=True, version=1, parser="auto")
    df_bn = clean_dataframe(bn_data.frame)
    bn_path = output_dir / "banknote_authentication.csv"
    df_bn.to_csv(bn_path, index=False)
    manifest["datasets"]["banknote_authentication"] = {
        "filename": "banknote_authentication.csv",
        "domain": "Signal Processing / Security",
        "description": "Continuous wavelet transform moments extracted from banknote evaluation images",
        **compute_dataset_stats(df_bn),
    }

    # 6. Airfoil Self-Noise (Aeronautics / Fluid Acoustics)
    print("Fetching 6/8: airfoil_self_noise...")
    air_data = fetch_openml(name="airfoil_self_noise", as_frame=True, version=1, parser="auto")
    df_air = clean_dataframe(air_data.frame)
    air_path = output_dir / "airfoil_self_noise.csv"
    df_air.to_csv(air_path, index=False)
    manifest["datasets"]["airfoil_self_noise"] = {
        "filename": "airfoil_self_noise.csv",
        "domain": "Aeronautics / Fluid Acoustics",
        "description": "NASA aerodynamic wind tunnel noise measurements with non-linear frequency and angle dynamics",
        **compute_dataset_stats(df_air),
    }

    # 7. Concrete Compressive Strength (Civil Engineering / Materials)
    print("Fetching 7/8: concrete_strength...")
    cnc_data = fetch_openml(name="concrete_compressive_strength", as_frame=True, version=7, parser="auto")
    df_cnc = clean_dataframe(cnc_data.frame)
    cnc_path = output_dir / "concrete_strength.csv"
    df_cnc.to_csv(cnc_path, index=False)
    manifest["datasets"]["concrete_strength"] = {
        "filename": "concrete_strength.csv",
        "domain": "Civil Engineering / Materials Science",
        "description": "Compressive strength of high-performance concrete mixtures vs constituent ingredient proportions",
        **compute_dataset_stats(df_cnc),
    }

    # 8. CPU Small (Computer Systems / Telemetry)
    print("Fetching 8/8: cpu_small...")
    cpu_data = fetch_openml(name="cpu_small", as_frame=True, version=1, parser="auto")
    df_cpu = clean_dataframe(cpu_data.frame)
    cpu_path = output_dir / "cpu_small.csv"
    df_cpu.to_csv(cpu_path, index=False)
    manifest["datasets"]["cpu_small"] = {
        "filename": "cpu_small.csv",
        "domain": "Computer Systems / Telemetry",
        "description": "Operating system process and hardware activity measurements in high-load server environments",
        **compute_dataset_stats(df_cpu),
    }

    # Include original baseline processed_data if present
    baseline_source = Path("notebooks/processed_data.csv")
    if baseline_source.is_file():
        print("Adding baseline synthetic reference: processed_data...")
        df_base = clean_dataframe(pd.read_csv(baseline_source))
        base_path = output_dir / "synthetic_baseline.csv"
        df_base.to_csv(base_path, index=False)
        manifest["datasets"]["synthetic_baseline"] = {
            "filename": "synthetic_baseline.csv",
            "domain": "Synthetic / Quantitative Finance",
            "description": "Original synthetic tabular dataset with 10 numeric features used in pilot studies",
            **compute_dataset_stats(df_base),
        }

    # Write manifest JSON
    manifest_path = output_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAll datasets downloaded and saved to: {output_dir}")
    print(f"Manifest created at: {manifest_path}")


if __name__ == "__main__":
    main()
