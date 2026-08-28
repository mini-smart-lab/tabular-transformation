from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.agent.schemas import FinalSubmission


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    input_tokens: int
    output_tokens: int
    token_used: int
    tool_calls: int
    execution_time_seconds: float
    cpu_time_seconds: float
    started_at: float
    final_submission: FinalSubmission | None
    status: Literal["running", "submitted", "budget_exceeded", "failed"]
    termination_reason: str
    turn_index: int
