from __future__ import annotations

from collections import deque

from src.sandbox.models import ExecutionResult


class LocalMockSandbox:
    """Safe deterministic fake: it never evaluates the submitted Python code."""

    def __init__(
        self,
        results: list[ExecutionResult] | None = None,
        files: dict[str, bytes] | None = None,
    ) -> None:
        self.results = deque(results or [])
        self.files = dict(files or {})
        self.executed_code: list[str] = []
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True
        self.closed = False

    def execute(self, code: str) -> ExecutionResult:
        if not self.started or self.closed:
            raise RuntimeError("sandbox is not running")
        self.executed_code.append(code)
        if not self.results:
            return ExecutionResult(status="success")
        return self.results.popleft()

    def read_file(self, relative_path: str) -> bytes:
        return self.files[relative_path]

    def reset(self) -> None:
        self.executed_code.clear()

    def close(self) -> None:
        self.closed = True
