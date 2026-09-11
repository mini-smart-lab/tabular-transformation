from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.agent.budget import BudgetManager
from src.agent.schemas import FinalSubmission, Transformation
from src.config import SandboxConfig
from src.logging_utils.episode_logger import EpisodeLogger
from src.sandbox.base import Sandbox


class RunPythonInput(BaseModel):
    code: str = Field(min_length=1)


class ReadImageInput(BaseModel):
    path: str = Field(min_length=1, description="Relative path to the image file (e.g. 'scatter.png') saved in the sandbox workspace.")


class SubmitSolutionInput(BaseModel):
    script: str = Field(min_length=1)
    transformations: list[Transformation]


class AgentTools:
    def __init__(
        self,
        sandbox: Sandbox,
        budget: BudgetManager,
        logger: EpisodeLogger,
        sandbox_config: SandboxConfig,
    ) -> None:
        self.sandbox = sandbox
        self.budget = budget
        self.logger = logger
        self.sandbox_config = sandbox_config
        self.final_submission: FinalSubmission | None = None
        self.langchain_tools = [
            StructuredTool.from_function(
                self.run_python,
                name="run_python",
                description="Execute general Python code in the discovery sandbox.",
                args_schema=RunPythonInput,
            ),
            StructuredTool.from_function(
                self.read_image,
                name="read_image",
                description="Read an image or plot file (PNG, JPEG, SVG) saved in the sandbox workspace to inspect visual charts and relationships.",
                args_schema=ReadImageInput,
            ),
            StructuredTool.from_function(
                self.submit_solution,
                name="submit_solution",
                description="Submit the final self-contained transform(df) script and operators.",
                args_schema=SubmitSolutionInput,
            ),
        ]
        self.by_name = {tool.name: tool for tool in self.langchain_tools}

    def run_python(self, code: str) -> dict[str, Any]:
        if not self.budget.can_run_python():
            return {"status": "error", "stderr": "budget exhausted; execution denied"}
        result = self.sandbox.execute(code)
        self.budget.record_execution_time(result.execution_time_seconds)
        self.budget.record_cpu_time(result.cpu_time_seconds)
        payload = result.model_dump()
        # Enforce truncation even when a third-party Sandbox implementation forgets it.
        for stream in ("stdout", "stderr"):
            payload[stream] = payload[stream][: self.sandbox_config.max_output_chars]
        self.logger.record(
            "python_execution",
            experiment_index=self.budget.tool_calls,
            code=code,
            **payload,
            cumulative_execution_time=self.budget.execution_time_seconds,
        )
        return payload

    def read_image(self, path: str) -> dict[str, Any]:
        if not self.budget.can_run_python():
            return {"status": "error", "message": "budget exhausted; execution denied"}
        try:
            image_bytes = self.sandbox.read_file(path)
        except Exception as exc:
            return {"status": "error", "message": f"Failed to read image '{path}': {exc}"}

        ext = Path(path).suffix.lower()
        mime_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
        }.get(ext, "image/png")

        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64_data}"

        self.logger.record(
            "read_image",
            experiment_index=self.budget.tool_calls,
            path=path,
            size_bytes=len(image_bytes),
            mime_type=mime_type,
        )
        return {
            "status": "success",
            "content_type": "image",
            "path": path,
            "size_bytes": len(image_bytes),
            "mime_type": mime_type,
            "data_url": data_url,
        }

    def submit_solution(
        self, script: str, transformations: list[Transformation]
    ) -> dict[str, Any]:
        self.final_submission = FinalSubmission(script=script, transformations=transformations)
        self.logger.record("submission", **self.final_submission.model_dump())
        return {"accepted": True, "message": "Submission stored; episode terminated."}
