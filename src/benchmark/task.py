from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from src.agent.schemas import Transformation


class BenchmarkTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    before_path: Path
    after_path: Path
    test_before_path: Path | None = None
    test_after_path: Path | None = None
    ground_truth_transformations: list[Transformation]
    level: int | None = None
    source_dataset: str | None = None
    domain: str | None = None

    def discovery_files(self) -> dict[str, Path]:
        """Only this mapping may be handed to the Tier-1 sandbox."""
        return {"before.csv": self.before_path, "after.csv": self.after_path}

    def evaluation_files(self) -> dict[str, Path]:
        """Only hidden input is mounted; expected output stays with the evaluator."""
        if self.test_before_path is None:
            raise ValueError("hidden test input has not been attached to this task")
        return {"test_before.csv": self.test_before_path}


def load_task_directory(
    task_directory: Path | str,
    *,
    test_before_path: Path | str | None = None,
    test_after_path: Path | str | None = None,
) -> BenchmarkTask:
    """Load one existing `data/<id>` directory without modifying or splitting rows."""
    directory = Path(task_directory).resolve()
    before_source = directory / "train_before.csv"
    after_source = directory / "train_after.csv"
    config_source = directory / "config.yaml"
    missing = [path.name for path in (before_source, after_source, config_source) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"task directory {directory} is missing: {', '.join(missing)}")

    before = pd.read_csv(before_source)
    after = pd.read_csv(after_source)
    _validate_pair(before, after)
    raw_config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    raw_config = raw_config if isinstance(raw_config, dict) else {}
    level = raw_config.get("level")
    source_dataset = raw_config.get("source_dataset")
    domain = raw_config.get("domain")

    transformations = load_transformations(config_source)
    unknown = {item.column for item in transformations} - set(before.columns)
    if unknown:
        raise ValueError(f"config references unknown columns: {sorted(unknown)}")
    if (test_before_path is None) != (test_after_path is None):
        raise ValueError("test_before_path and test_after_path must be supplied together")
    bundled_before = directory / "hidden" / "test_before.csv"
    bundled_after = directory / "hidden" / "test_after.csv"
    if test_before_path is None and bundled_before.is_file() and bundled_after.is_file():
        test_before_path, test_after_path = bundled_before, bundled_after
    hidden_before = Path(test_before_path).resolve() if test_before_path is not None else None
    hidden_after = Path(test_after_path).resolve() if test_after_path is not None else None
    if hidden_before is not None and not hidden_before.is_file():
        raise FileNotFoundError(hidden_before)
    if hidden_after is not None and not hidden_after.is_file():
        raise FileNotFoundError(hidden_after)
    return BenchmarkTask(
        task_id=directory.name,
        before_path=before_source,
        after_path=after_source,
        test_before_path=hidden_before,
        test_after_path=hidden_after,
        ground_truth_transformations=transformations,
        level=level,
        source_dataset=source_dataset,
        domain=domain,
    )


def load_transformations(config_path: Path | str) -> list[Transformation]:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("task config must be a YAML mapping")
    if "transformations" in raw:
        items = raw["transformations"]
        if not isinstance(items, list):
            raise ValueError("transformations must be a list")
        return [Transformation.model_validate(item) for item in items]

    columns = raw.get("columns")
    operator = raw.get("operator")
    if isinstance(columns, str):
        operators = operator if isinstance(operator, list) else [operator]
        return [Transformation(column=columns, operators=operators)]
    if isinstance(columns, list) and all(isinstance(column, str) for column in columns):
        if not isinstance(operator, str):
            raise ValueError("a multi-column legacy config requires one shared string operator")
        return [Transformation(column=column, operators=[operator]) for column in columns]
    raise ValueError("config requires `columns` and `operator`, or `transformations`")


def _validate_pair(before: pd.DataFrame, after: pd.DataFrame) -> None:
    if len(before) != len(after):
        raise ValueError("train_before.csv and train_after.csv must have equal row counts")
    if list(before.columns) != list(after.columns):
        raise ValueError("train_before.csv and train_after.csv must have identical ordered columns")
