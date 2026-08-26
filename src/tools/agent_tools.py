from __future__ import annotations

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
        self.budget.record_tool_call()
        result = self.sandbox.execute(code)
        self.budget.record_execution_time(result.execution_time_seconds)
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

    def submit_solution(
        self, script: str, transformations: list[Transformation]
    ) -> dict[str, Any]:
        self.final_submission = FinalSubmission(script=script, transformations=transformations)
        self.logger.record("submission", **self.final_submission.model_dump())
        return {"accepted": True, "message": "Submission stored; episode terminated."}
