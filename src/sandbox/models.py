from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "error", "timeout"]
    stdout: str = ""
    stderr: str = ""
    execution_time_seconds: float = Field(default=0.0, ge=0)
    cpu_time_seconds: float = Field(default=0.0, ge=0)
