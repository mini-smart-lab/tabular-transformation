from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from src.config import SandboxConfig
from src.sandbox.base import validate_mounts
from src.sandbox.models import ExecutionResult


class LocalProcessSandbox:
    """Development-only subprocess adapter; replace with Docker in production."""

    def __init__(self, files: dict[str, Path], config: SandboxConfig | None = None) -> None:
        validate_mounts(files)
        self.files = {name: Path(path) for name, path in files.items()}
        self.config = config or SandboxConfig()
        self.workspace: Path | None = None

    def start(self) -> None:
        if self.workspace is not None:
            raise RuntimeError("sandbox already started")
        self.workspace = Path(tempfile.mkdtemp(prefix="tabular-sandbox-"))
        for name, source in self.files.items():
            destination = self.workspace / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

    def execute(self, code: str) -> ExecutionResult:
        workspace = self._require_workspace()
        program = workspace / "__execution__.py"
        cpu_file = workspace / "__cpu_time__.txt"
        cpu_file.unlink(missing_ok=True)
        program.write_text(_instrument(code), encoding="utf-8")
        started = time.monotonic()
        # API keys and other host secrets are not inherited by generated code.
        child_env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        }
        try:
            completed = subprocess.run(
                [sys.executable, "-I", program.name],
                cwd=workspace,
                env=child_env,
                capture_output=True,
                text=True,
                timeout=self.config.execution_timeout_seconds,
                check=False,
            )
            elapsed = time.monotonic() - started
            return ExecutionResult(
                status="success" if completed.returncode == 0 else "error",
                stdout=self._truncate(completed.stdout),
                stderr=self._truncate(completed.stderr),
                execution_time_seconds=elapsed,
                cpu_time_seconds=_read_cpu_time(cpu_file),
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                status="timeout",
                stdout=self._truncate(_to_text(exc.stdout)),
                stderr=self._truncate(_to_text(exc.stderr) or "execution timed out"),
                execution_time_seconds=time.monotonic() - started,
                cpu_time_seconds=_read_cpu_time(cpu_file),
            )

    def read_file(self, relative_path: str) -> bytes:
        workspace = self._require_workspace().resolve()
        target = (workspace / relative_path).resolve()
        if workspace not in target.parents:
            raise ValueError("path escapes sandbox workspace")
        return target.read_bytes()

    def reset(self) -> None:
        self.close()
        self.start()

    def close(self) -> None:
        if self.workspace is not None:
            shutil.rmtree(self.workspace, ignore_errors=True)
            self.workspace = None

    def _require_workspace(self) -> Path:
        if self.workspace is None:
            raise RuntimeError("sandbox is not running")
        return self.workspace

    def _truncate(self, value: str) -> str:
        limit = self.config.max_output_chars
        if len(value) <= limit:
            return value
        return value[:limit] + f"\n...[truncated {len(value) - limit} chars]"


def _to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _instrument(code: str) -> str:
    return (
        "import time as _benchmark_time\n"
        "_benchmark_cpu_start = _benchmark_time.process_time()\n"
        "try:\n"
        f"    exec(compile({code!r}, '<agent-code>', 'exec'), {{'__name__': '__main__'}})\n"
        "finally:\n"
        "    with open('__cpu_time__.txt', 'w', encoding='ascii') as _benchmark_handle:\n"
        "        _benchmark_handle.write(str(_benchmark_time.process_time() - _benchmark_cpu_start))\n"
    )


def _read_cpu_time(path: Path) -> float:
    try:
        return max(0.0, float(path.read_text(encoding="ascii")))
    except (OSError, ValueError):
        return 0.0
