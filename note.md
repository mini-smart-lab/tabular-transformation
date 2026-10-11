# Research Project Notes & Progress Log
**Project:** Two-Tier Tabular Transformation Discovery Harness  
**Repository:** `tabular-transformation`  
**Last Updated:** 2026-10-11  

---

## 1. Context & Reviewer Feedback Tracking
- **Reviewer Feedback:** Manuscript scale was evaluated as *too small* (previously evaluated only on a single synthetic 10-column dataset with ~24 tasks).
- **Core Action Items:**
  1. [x] **Archive Legacy Runs:** Safely backup all previous 2026-09 single-dataset runs in `archive/legacy_runs_202609/`.
  2. [x] **Install Kaggle Benchmarks Skill:** Integrated `write-kaggle-benchmarks` (`.agent/skills/write-kaggle-benchmarks/`) and installed SDK (`kaggle-benchmarks` 0.6.1) to leverage Kaggle Model Proxy (0 API cost for frontier LLMs).
  3. [x] **Phase 1: Multi-Domain Source Datasets:** Curated 8 real-world datasets across diverse domains + 1 baseline dataset in `data/source_datasets/`.
  4. [x] **Phase 2: Task Engine & Pipeline Upgrade:** Extended `src/benchmark/task_builder.py` and `run_pipeline.py` with `build_multidataset_suite`, multi-dataset CLI flags (`--source-dir`, `--levels`, `--datasets`), and per-task level-specific adaptive budgets.
  5. [x] **Phase 3: Kaggle Benchmarks Bridge:** Implemented `src/kaggle_adapter/exporter.py` (`export_task_to_kbench`, `export_suite_to_kbench`) converting internal tasks to `@kbench.task` standard format with `# %%` cell markers, ground-truth assertion scoring, and standalone `.py` task files. Verified via end-to-end pilot run on multi-dataset suite.
  6. [ ] **Phase 4: Academic Aggregation & LaTeX Tables:** Multi-dimensional metrics (Domain x Level x Model) with automated paper-ready table generation.
  7. [ ] **Phase 5: Pilot Run & Full Evaluation:** Pilot run across all domains and levels.

---

## 2. Multi-Domain Dataset Suite & Academic Citations

All cleaned, standardized numeric CSV files and detailed column metadata reside in `data/source_datasets/` (see `data/source_datasets/dataset_manifest.json`).

### Dataset References for Manuscript Citation:

1. **California Housing (`california_housing.csv`)**
   - **Domain:** Real Estate / Economics
   - **Characteristics:** 20,640 rows, 9 columns, heavy-tailed distribution, spatial collinearity.
   - **Academic Reference:** Pace, R. Kelley, and Ronald Barry. *"Sparse spatial autoregressions."* Statistics & Probability Letters 33.3 (1997): 291-297.
   - **URL:** [OpenML Dataset 43939](https://www.openml.org/d/43939) | [Scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

2. **Diabetes Baseline (`diabetes.csv`)**
   - **Domain:** Biomedical / Clinical Medicine
   - **Characteristics:** 442 rows, 11 columns, physiological and blood serum measurements.
   - **Academic Reference:** Efron, Bradley, Trevor Hastie, Iain Johnstone, and Robert Tibshirani. *"Least angle regression."* The Annals of Statistics 32.2 (2004): 407-499.
   - **URL:** [OpenML Dataset 503](https://www.openml.org/d/503) | [Scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)

3. **Breast Cancer Wisconsin Diagnostic (`breast_cancer.csv`)**
   - **Domain:** Oncology / Pathology
   - **Characteristics:** 569 rows, 31 columns, high-dimensional collinear morphological features.
   - **Academic Reference:** Street, W. Nick, William H. Wolberg, and Olvi L. Mangasarian. *"Nuclear feature extraction for breast tumor diagnosis."* Biomedical Image Processing and Biomedical Visualization, Vol. 1905. SPIE, 1993.
   - **URL:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | [OpenML Dataset 15](https://www.openml.org/d/15)

4. **Wine Cultivars Physicochemical (`wine_quality.csv`)**
   - **Domain:** Food Chemistry / Agriculture
   - **Characteristics:** 178 rows, 14 columns, non-linear chemical proportions across 3 cultivars.
   - **Academic Reference:** Aeberhard, Stefan, Danny Coomans, and Olivier de Vel. *"Comparative analysis of statistical pattern recognition methods in high dimensional settings."* Pattern Recognition 27.8 (1994): 1065-1077.
   - **URL:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/109/wine) | [OpenML Dataset 187](https://www.openml.org/d/187)

5. **Banknote Authentication (`banknote_authentication.csv`)**
   - **Domain:** Signal Processing / Security
   - **Characteristics:** 1,372 rows, 5 columns, continuous wavelet transform statistical moments.
   - **Academic Reference:** Lohweg, Volker, et al. *"Banknote authentication with mobile devices."* Electronic Imaging (2013).
   - **URL:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication) | [OpenML Dataset 1462](https://www.openml.org/d/1462)

6. **Airfoil Self-Noise (`airfoil_self_noise.csv`)**
   - **Domain:** Aeronautics / Fluid Acoustics
   - **Characteristics:** 1,503 rows, 6 columns, non-linear aerodynamic blade acoustics.
   - **Academic Reference:** Brooks, Thomas F., D. Stuart Pope, and Michael A. Marcolini. *"Airfoil self-noise and prediction."* NASA Reference Publication 1218 (1989).
   - **URL:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) | [OpenML Dataset 43919](https://www.openml.org/d/43919)

7. **Concrete Compressive Strength (`concrete_strength.csv`)**
   - **Domain:** Civil Engineering / Materials Science
   - **Characteristics:** 1,030 rows, 9 columns, physical mixture constituents vs material strength.
   - **Academic Reference:** Yeh, I-Cheng. *"Modeling of strength of high-performance concrete using artificial neural networks."* Cement and Concrete Research 28.12 (1998): 1797-1808.
   - **URL:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) | [OpenML Dataset 44959](https://www.openml.org/d/44959)

8. **CPU Small OS Activity (`cpu_small.csv`)**
   - **Domain:** Computer Systems / Telemetry
   - **Characteristics:** 8,192 rows, 13 columns, operating system process scheduling and memory activity.
   - **Academic Reference:** Delve Benchmark Collection, Department of Computer Science, University of Toronto.
   - **URL:** [OpenML Dataset 197](https://www.openml.org/d/197)

9. **Synthetic Baseline (`synthetic_baseline.csv`)**
   - **Domain:** Quantitative Finance / Synthetic
   - **Characteristics:** 247,435 rows, 10 columns, reference pilot dataset from `notebooks/processed_data.csv`.

---

## 3. Important Implementation Notes
- **Testing Guardrail:** Run `.venv\Scripts\python.exe -m pytest -q` before and after every modification. Ensure 100% offline pass.
- **Data Isolation:** Hidden test sets (`hidden/test_*.csv`) must never be leaked or mounted to Tier 1 discovery sandboxes.
- **Ablation Study Reminder (from prior notes):**
  - Run ablation on agent components (without code interpreter vs with code interpreter).
  - Verify multimodal / plot reading tool effect if chart generation is added.
- **Kaggle CLI Benchmarks:**
  - Authenticate proxy credentials via `kaggle b auth -y`.
  - Skill docs located in `.agent/skills/write-kaggle-benchmarks/SKILL.md`.
