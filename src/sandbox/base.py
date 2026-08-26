from pathlib import Path
from typing import Protocol

from src.sandbox.models import ExecutionResult


class Sandbox(Protocol):
    """Boundary implemented later by the hardened Docker runtime."""

    def start(self) -> None: ...
    def execute(self, code: str) -> ExecutionResult: ...
    def read_file(self, relative_path: str) -> bytes: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...


def validate_mounts(files: dict[str, Path]) -> None:
    for name, source in files.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"unsafe sandbox path: {name}")
        if not Path(source).is_file():
            raise FileNotFoundError(source)
