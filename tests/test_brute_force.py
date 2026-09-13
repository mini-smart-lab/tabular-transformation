from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.agent.budget import BudgetManager
from src.agent.graph import DiscoveryGraph, normalize_terminal_state
from src.config import AgentBudgetConfig, ModelConfig
from src.logging_utils.episode_logger import EpisodeLogger
from src.model_factory import create_model
from src.sandbox.local_process import LocalProcessSandbox


def test_brute_force_model_factory():
    config = ModelConfig(
        provider="brute_force",
        model="baseline-brute-force",
        api_key_env="NO_KEY_REQUIRED",
    )
    model = create_model(config)
    assert model is not None


def test_brute_force_solves_unary_and_interaction_task(tmp_path: Path):
    # Create before and after data with Level 3 transformations:
    # col_1: Square
    # col_2: Log1p
    # col_3: col_1 + col_2 (AddInteraction)
    # col_4: unchanged
    np.random.seed(42)
    n_rows = 100
    x1 = np.random.uniform(1.0, 5.0, size=n_rows)
    x2 = np.random.uniform(0.1, 2.0, size=n_rows)
    x4 = np.random.uniform(10.0, 20.0, size=n_rows)
    x3 = x1 + x2

    df_before = pd.DataFrame({
        "col_1": x1,
        "col_2": x2,
        "col_3": np.random.uniform(0.0, 1.0, size=n_rows),
        "col_4": x4,
    })
    df_after = pd.DataFrame({
        "col_1": np.square(x1),
        "col_2": np.log1p(x2),
        "col_3": x3,
        "col_4": x4,
    })

    before_path = tmp_path / "before.csv"
    after_path = tmp_path / "after.csv"
    df_before.to_csv(before_path, index=False)
    df_after.to_csv(after_path, index=False)

    sandbox = LocalProcessSandbox({"before.csv": before_path, "after.csv": after_path})
    sandbox.start()

    budget = BudgetManager(AgentBudgetConfig(max_tool_calls=5, max_wall_time_seconds=30.0))
    model_config = ModelConfig(
        provider="brute_force",
        model="baseline-brute-force",
        api_key_env="NO_KEY_REQUIRED",
    )
    model = create_model(model_config)

    episode = DiscoveryGraph(
        model, sandbox, budget, EpisodeLogger(tmp_path / "ep.jsonl"),
    )
    state = normalize_terminal_state(episode.invoke())
    sandbox.close()

    assert state["status"] == "submitted"
    assert state["final_submission"] is not None

    # Test the submitted script execution on before data
    local_ns = {}
    exec(state["final_submission"].script, local_ns)
    transform_fn = local_ns["transform"]
    predicted_df = transform_fn(df_before)

    assert np.allclose(predicted_df["col_1"], df_after["col_1"], atol=1e-5)
    assert np.allclose(predicted_df["col_2"], df_after["col_2"], atol=1e-5)
    assert np.allclose(predicted_df["col_3"], df_after["col_3"], atol=1e-5)
    assert np.allclose(predicted_df["col_4"], df_after["col_4"], atol=1e-5)
