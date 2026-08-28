from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from src.config import SandboxConfig
from src.sandbox.base import validate_mounts
from src.sandbox.local_process import _instrument, _read_cpu_time, _to_text
from src.sandbox.models import ExecutionResult


class DockerSandbox:
    """Docker CLI adapter. Image construction and host policy remain owner-configurable."""

    def __init__(self, files: dict[str, Path], config: SandboxConfig | None = None) -> None:
        validate_mounts(files)
        self.files = {name: Path(path) for name, path in files.items()}
        self.config = config or SandboxConfig(backend="docker")
        self.workspace: Path | None = None
        self.container_name: str | None = None

    def start(self) -> None:
        if self.workspace is not None:
            raise RuntimeError("sandbox already started")
        self.workspace = Path(tempfile.mkdtemp(prefix="tabular-docker-sandbox-"))
        for name, source in self.files.items():
            destination = self.workspace / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        self.container_name = f"tabular-benchmark-{uuid.uuid4().hex[:12]}"
        mount = f"{self.workspace.resolve()}:/workspace"
        command = [
            "docker", "run", "--detach", "--rm", "--name", self.container_name,
            "--network", self.config.docker_network,
            "--cpus", str(self.config.docker_cpus),
            "--memory", self.config.docker_memory,
            "--pids-limit", str(self.config.docker_pids_limit),
            "--mount", f"type=bind,source={self.workspace.resolve()},target=/workspace",
            "--workdir", "/workspace", self.config.docker_image,
            "sh", "-c", "tail -f /dev/null",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            self.close()
            raise RuntimeError(f"failed to start Docker sandbox: {completed.stderr.strip()}")

    def execute(self, code: str) -> ExecutionResult:
        workspace = self._require_workspace()
        cpu_file = workspace / "__cpu_time__.txt"
        cpu_file.unlink(missing_ok=True)
        (workspace / "__execution__.py").write_text(_instrument(code), encoding="utf-8")
        started = time.monotonic()
        try:
            completed = subprocess.run(
                ["docker", "exec", self._require_container(), "python", "-I", "__execution__.py"],
                capture_output=True, text=True, timeout=self.config.execution_timeout_seconds,
                check=False,
            )
            elapsed = time.monotonic() - started
            return ExecutionResult(
                status="success" if completed.returncode == 0 else "error",
                stdout=self._truncate(completed.stdout), stderr=self._truncate(completed.stderr),
                execution_time_seconds=elapsed, cpu_time_seconds=_read_cpu_time(cpu_file),
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                status="timeout", stdout=self._truncate(_to_text(exc.stdout)),
                stderr=self._truncate(_to_text(exc.stderr) or "execution timed out"),
                execution_time_seconds=time.monotonic() - started,
                cpu_time_seconds=_read_cpu_time(cpu_file),
            )

    def read_file(self, relative_path: str) -> bytes:
        root = self._require_workspace().resolve()
        target = (root / relative_path).resolve()
        if root not in target.parents:
            raise ValueError("path escapes sandbox workspace")
        return target.read_bytes()

    def reset(self) -> None:
        self.close()
        self.start()

    def close(self) -> None:
        if self.container_name:
            subprocess.run(["docker", "rm", "--force", self.container_name], capture_output=True)
            self.container_name = None
        if self.workspace:
            shutil.rmtree(self.workspace, ignore_errors=True)
            self.workspace = None

    def _require_workspace(self) -> Path:
        if self.workspace is None:
            raise RuntimeError("sandbox is not running")
        return self.workspace

    def _require_container(self) -> str:
        if self.container_name is None:
            raise RuntimeError("sandbox is not running")
        return self.container_name

    def _truncate(self, text: str) -> str:
        limit = self.config.max_output_chars
        return text if len(text) <= limit else text[:limit] + f"\n...[truncated {len(text)-limit} chars]"
