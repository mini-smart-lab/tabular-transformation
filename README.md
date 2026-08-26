# Tabular Transformation Discovery Harness

This repository implements a two-tier LangGraph benchmark. Tier 1 gives a tool-calling
LLM only `before.csv`, `after.csv`, `run_python`, and `submit_solution`. Tier 2 evaluates
the submitted `transform(df)` in a fresh sandbox without any LLM call or discovery state.

## Setup

Create `.env` with `NVIDIA_API_KEY`, then install dependencies:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run tests:

```powershell
.venv\Scripts\python.exe -m pytest
```

Each task is a directory with the repository's existing layout:

```text
data/0/
  train_before.csv
  train_after.csv
  config.yaml
```

The full `train_before.csv` and `train_after.csv` pair is exposed to the discovery
sandbox without splitting. Run discovery for a task directory directly:

```powershell
.venv\Scripts\python.exe -m src.benchmark.runner --task data\0 --model-config configs\nvidia.yaml
```

This produces and logs the submitted script but skips functional evaluation. Attach a
separate hidden pair when it is available elsewhere:

```powershell
.venv\Scripts\python.exe -m src.benchmark.runner --task data\0 `
  --model-config configs\nvidia.yaml `
  --test-before path\to\hidden_before.csv `
  --test-after path\to\hidden_after.csv
```

The hidden expected CSV is read only by the host evaluator and is never mounted into
either sandbox.

The included `LocalProcessSandbox` is a development adapter that isolates execution in
a child process and scrubs host environment variables. It is not a security boundary;
production must inject the planned Docker-backed implementation through the same
`Sandbox` protocol. Raw trajectories are stored as JSONL under `logs/`.
