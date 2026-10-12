from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from src.config import AgentBudgetConfig


BudgetReason = Literal["tokens", "tool_calls", "execution_time", "cpu_time", "wall_time"]


@dataclass(frozen=True)
class BudgetStatus:
    exhausted: bool
    reason: BudgetReason | None = None


class BudgetManager:
    def __init__(self, config: AgentBudgetConfig, *, clock=time.monotonic) -> None:
        self.config = config
        self._clock = clock
        self.started_at = clock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.tool_calls = 0
        self.execution_time_seconds = 0.0
        self.cpu_time_seconds = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def wall_time_seconds(self) -> float:
        return max(0.0, self._clock() - self.started_at)

    def record_llm_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += max(0, input_tokens)
        self.output_tokens += max(0, output_tokens)

    def record_tool_call(self) -> None:
        self.tool_calls += 1

    def record_execution_time(self, seconds: float) -> None:
        self.execution_time_seconds += max(0.0, seconds)

    def record_cpu_time(self, seconds: float) -> None:
        self.cpu_time_seconds += max(0.0, seconds)

    def check(self) -> BudgetStatus:
        checks: list[tuple[bool, BudgetReason]] = [
            (self.total_tokens >= self.config.max_total_tokens, "tokens"),
            (self.tool_calls >= self.config.max_tool_calls, "tool_calls"),
            (
                self.execution_time_seconds >= self.config.max_total_execution_seconds,
                "execution_time",
            ),
            (self.cpu_time_seconds >= self.config.max_total_cpu_seconds, "cpu_time"),
            (self.wall_time_seconds >= self.config.max_wall_time_seconds, "wall_time"),
        ]
        for exhausted, reason in checks:
            if exhausted:
                return BudgetStatus(True, reason)
        return BudgetStatus(False)

    def can_run_python(self) -> bool:
        return not self.check().exhausted

    def consumption_ratio(self) -> float:
        token_ratio = self.total_tokens / max(1, self.config.max_total_tokens)
        calls_ratio = self.tool_calls / max(1, self.config.max_tool_calls)
        time_ratio = self.wall_time_seconds / max(1, self.config.max_wall_time_seconds)
        return max(token_ratio, calls_ratio, time_ratio)

    def get_budget_status_message(self) -> str:
        pct = self.consumption_ratio() * 100
        msg = (
            f"Budget consumed: {pct:.1f}% "
            f"({self.tool_calls}/{self.config.max_tool_calls} tool calls, "
            f"~{self.total_tokens:,}/{self.config.max_total_tokens:,} tokens, "
            f"{self.wall_time_seconds:.1f}s/{self.config.max_wall_time_seconds:.1f}s wall time)"
        )
        if pct >= 70.0:
            msg += (
                f"\n[CRITICAL BUDGET ALERT - {pct:.1f}% EXPENDED]: You are approaching hard termination! "
                "Per your instructions, prioritize submitting your best discovered transform(df) script IMMEDIATELY "
                "via `submit_solution`. Do NOT run more exploratory python calls or the episode will abort with 0.0 penalty!"
            )
        return msg

    def snapshot(self) -> dict[str, int | float]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
            "execution_time_seconds": self.execution_time_seconds,
            "cpu_time_seconds": self.cpu_time_seconds,
            "wall_time_seconds": self.wall_time_seconds,
        }

