# Multi-Domain Source Datasets Suite

This directory contains standardized, purely numeric tabular datasets spanning 8 diverse real-world domains + 1 baseline reference dataset. These datasets serve as the source distributions for generating Level 0–5 discovery tasks in the Tabular Transformation Discovery Harness.

## Dataset Inventory

| Dataset | Domain | Rows | Features | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| `california_housing.csv` | Real Estate / Economics | 20,640 | 9 | Heavy-tailed distributions, geospatial and economic attributes |
| `diabetes.csv` | Biomedical / Clinical | 442 | 11 | Normalized physiological markers, disease progression target |
| `breast_cancer.csv` | Oncology / Pathology | 569 | 31 | High-dimensional, strongly correlated morphological cell nuclei features |
| `wine_quality.csv` | Food Chemistry / Agriculture | 178 | 14 | Physicochemical and spectroscopic attributes |
| `banknote_authentication.csv` | Signal Processing / Security | 1,372 | 5 | Continuous wavelet moments (variance, skewness, curtosis, entropy) |
| `airfoil_self_noise.csv` | Aeronautics / Acoustics | 1,503 | 6 | NASA aeroacoustic aerodynamic measurements, non-linear physical dynamics |
| `concrete_strength.csv` | Civil Engineering / Materials | 1,030 | 9 | Highly non-linear mixture proportions vs compressive strength |
| `cpu_small.csv` | Computer Systems / Telemetry | 8,192 | 13 | High-frequency CPU and memory operating system telemetry |
| `synthetic_baseline.csv` | Quantitative Finance / Synthetic | 247,435 | 10 | Original baseline dataset used in preliminary single-domain pilot runs |

## Metadata

See `dataset_manifest.json` for per-column names, summary statistics, and skewness metrics.
To regenerate or update these datasets, run:
```bash
python scripts/prepare_source_datasets.py
```
