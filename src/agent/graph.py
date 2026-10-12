from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from src.agent.budget import BudgetManager
from src.agent.prompts import build_system_prompt
from src.agent.state import AgentState
from src.agent.token_usage import extract_token_usage
from src.config import SandboxConfig
from src.logging_utils.episode_logger import EpisodeLogger
from src.sandbox.base import Sandbox
from src.tools.agent_tools import AgentTools


class DiscoveryGraph:
    def __init__(
        self,
        model: Any,
        sandbox: Sandbox,
        budget: BudgetManager,
        logger: EpisodeLogger,
        *,
        sandbox_config: SandboxConfig | None = None,
        operator_vocabulary: list[str] | None = None,
        provider_model: str = "unknown",
        enable_code_execution: bool = True,
        oracle_transformations: list[Any] | None = None,
    ) -> None:
        self.budget = budget
        self.logger = logger
        self.provider_model = provider_model
        resolved_sandbox_config = sandbox_config or SandboxConfig()
        self.tools = AgentTools(
            sandbox,
            budget,
            logger,
            resolved_sandbox_config,
            enable_code_execution=enable_code_execution,
        )
        self.model = model.bind_tools(self.tools.langchain_tools)
        self.system_message = SystemMessage(
            content=build_system_prompt(
                operator_vocabulary=operator_vocabulary or [],
                budget_config=budget.config,
                sandbox_config=resolved_sandbox_config,
                oracle_transformations=oracle_transformations,
                no_tools=not enable_code_execution,
            )
        )
        self.graph = self._build()


    def _build(self):
        builder = StateGraph(AgentState)
        builder.add_node("llm", self._model_node)
        builder.add_node("tools", self._tool_node)
        builder.add_edge(START, "llm")
        builder.add_conditional_edges("llm", self._route_after_model, {"tools": "tools", "end": END})
        builder.add_conditional_edges("tools", self._route_after_tools, {"llm": "llm", "end": END})
        return builder.compile()

    def initial_state(self) -> AgentState:
        return {
            "messages": [HumanMessage(content="Begin discovery using before.csv and after.csv.")],
            "input_tokens": 0,
            "output_tokens": 0,
            "token_used": 0,
            "tool_calls": 0,
            "execution_time_seconds": 0.0,
            "cpu_time_seconds": 0.0,
            "started_at": self.budget.started_at,
            "final_submission": None,
            "status": "running",
            "termination_reason": "",
            "turn_index": 0,
        }

    def invoke(self) -> AgentState:
        # The graph's own recursion guard is finite and derived from the external call budget.
        recursion_limit = max(10, self.budget.config.max_tool_calls * 3 + 5)
        return self.graph.invoke(self.initial_state(), {"recursion_limit": recursion_limit})

    def _model_node(self, state: AgentState) -> dict[str, Any]:
        budget_status = self.budget.check()
        if budget_status.exhausted:
            return self._budget_update(budget_status.reason)
        response = self.model.invoke([self.system_message, *state["messages"]])
        input_tokens, output_tokens = extract_token_usage(response)
        self.budget.record_llm_usage(input_tokens, output_tokens)
        turn = state["turn_index"] + 1
        snapshot = self.budget.snapshot()
        self.logger.record(
            "llm_turn",
            turn_index=turn,
            provider_model=self.provider_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cumulative_tokens=snapshot["total_tokens"],
            response_content=response.content,
            tool_calls=getattr(response, "tool_calls", []),
        )
        return {
            "messages": [response],
            "turn_index": turn,
            **self._counter_update(),
        }

    def _tool_node(self, state: AgentState) -> dict[str, Any]:
        message = state["messages"][-1]
        observations: list[ToolMessage] = []
        status: str = "running"
        reason = ""
        for call in getattr(message, "tool_calls", []):
            name = call.get("name", "")
            self.budget.record_tool_call()
            try:
                if self.budget.tool_calls > self.budget.config.max_tool_calls:
                    result: Any = {"status": "error", "stderr": "budget exhausted"}
                elif name not in self.tools.by_name:
                    result = {"status": "error", "stderr": f"unknown tool: {name}"}
                else:
                    result = self.tools.by_name[name].invoke(call.get("args", {}))
            except Exception as exc:
                result = {"status": "error", "stderr": f"invalid tool call: {exc}"}

            if isinstance(result, dict) and result.get("content_type") == "image" and result.get("status") == "success":
                tool_content: Any = [
                    {"type": "text", "text": f"Image file '{result.get('path')}' loaded ({result.get('size_bytes', 0)} bytes) [Budget: {self.budget.tool_calls}/{self.budget.config.max_tool_calls} calls, ~{self.budget.total_tokens:,}/{self.budget.config.max_total_tokens:,} tokens used]:"},
                    {"type": "image_url", "image_url": {"url": result.get("data_url")}},
                ]
            else:
                if isinstance(result, dict):
                    result_payload = dict(result)
                    result_payload["budget_status"] = (
                        f"{self.budget.tool_calls}/{self.budget.config.max_tool_calls} tool calls used, "
                        f"~{self.budget.total_tokens:,}/{self.budget.config.max_total_tokens:,} tokens consumed"
                    )
                    tool_content = json.dumps(result_payload, default=str)
                else:
                    tool_content = json.dumps(result, default=str)

            observations.append(
                ToolMessage(
                    content=tool_content,
                    tool_call_id=call.get("id", name or "unknown"),
                    name=name or "unknown",
                )
            )
            if self.tools.final_submission is not None:
                status = "submitted"
                reason = "submitted"
                break

        if status != "submitted":
            exhausted = self.budget.check()
            if exhausted.exhausted:
                status = "budget_exceeded"
                reason = f"budget_exceeded:{exhausted.reason}"
        update: dict[str, Any] = {
            "messages": observations,
            "status": status,
            "termination_reason": reason,
            "final_submission": self.tools.final_submission,
            **self._counter_update(),
        }
        return update

    def _route_after_model(self, state: AgentState) -> Literal["tools", "end"]:
        if state["status"] != "running":
            return "end"
        last = state["messages"][-1]
        if getattr(last, "tool_calls", []):
            return "tools"
        return "end"

    def _route_after_tools(self, state: AgentState) -> Literal["llm", "end"]:
        return "llm" if state["status"] == "running" else "end"

    def _counter_update(self) -> dict[str, int | float]:
        return {
            "input_tokens": self.budget.input_tokens,
            "output_tokens": self.budget.output_tokens,
            "token_used": self.budget.total_tokens,
            "tool_calls": self.budget.tool_calls,
            "execution_time_seconds": self.budget.execution_time_seconds,
            "cpu_time_seconds": self.budget.cpu_time_seconds,
        }

    def _budget_update(self, reason: str | None) -> dict[str, Any]:
        return {
            "status": "budget_exceeded",
            "termination_reason": f"budget_exceeded:{reason}",
            **self._counter_update(),
        }


def normalize_terminal_state(state: AgentState) -> AgentState:
    """Treat a tool-less model response as failure instead of a valid final answer."""
    if state["status"] == "running":
        state = dict(state)
        state["status"] = "failed"
        state["termination_reason"] = "model_stopped_without_submission"
    return state  # type: ignore[return-value]
