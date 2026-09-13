from langchain_core.messages import AIMessage

from src.agent.budget import BudgetManager
from src.agent.graph import DiscoveryGraph, normalize_terminal_state
from src.config import AgentBudgetConfig
from src.logging_utils.episode_logger import EpisodeLogger
from src.sandbox.local_mock import LocalMockSandbox
from src.sandbox.models import ExecutionResult


SCRIPT = "def transform(df):\n    result = df.copy()\n    result['x'] = result['x'] ** 2\n    return result\n"


class MockToolCallingModel:
    def __init__(self):
        self.index = 0
        self.responses = [
            AIMessage(content="", tool_calls=[{"name": "run_python", "args": {"code": "print(1)"}, "id": "1"}]),
            AIMessage(content="", tool_calls=[{"name": "run_python", "args": {"code": "print(2)"}, "id": "2"}]),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "submit_solution",
                    "args": {
                        "script": SCRIPT,
                        "transformations": [{"column": "x", "operators": ["square"]}],
                    },
                    "id": "3",
                }],
            ),
        ]

    def bind_tools(self, tools):
        assert set(tool.name for tool in tools) == {"run_python", "read_image", "submit_solution"}
        return self

    def invoke(self, messages):
        response = self.responses[self.index]
        self.index += 1
        response.usage_metadata = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
        return response


def test_graph_run_run_submit(tmp_path):
    sandbox = LocalMockSandbox([
        ExecutionResult(status="success", stdout="1\n", execution_time_seconds=0.1),
        ExecutionResult(status="success", stdout="2\n", execution_time_seconds=0.2),
    ])
    sandbox.start()
    budget = BudgetManager(AgentBudgetConfig())
    episode = DiscoveryGraph(
        MockToolCallingModel(), sandbox, budget, EpisodeLogger(tmp_path / "episode.jsonl")
    )
    state = normalize_terminal_state(episode.invoke())
    assert state["status"] == "submitted"
    assert state["final_submission"].transformations[0].operators == ["square"]
    assert state["tool_calls"] == 3
    assert state["token_used"] == 36
    assert len(sandbox.executed_code) == 2


def test_graph_read_image(tmp_path):
    class MockImageModel:
        def __init__(self):
            self.index = 0
            self.responses = [
                AIMessage(content="", tool_calls=[{"name": "read_image", "args": {"path": "scatter.png"}, "id": "img1"}]),
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "submit_solution",
                        "args": {
                            "script": SCRIPT,
                            "transformations": [{"column": "x", "operators": ["square"]}],
                        },
                        "id": "sub1",
                    }],
                ),
            ]

        def bind_tools(self, tools):
            assert "read_image" in [t.name for t in tools]
            return self

        def invoke(self, messages):
            response = self.responses[self.index]
            self.index += 1
            response.usage_metadata = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
            return response

    sandbox = LocalMockSandbox(files={"scatter.png": b"\x89PNG\r\n\x1a\nfake_image_bytes"})
    sandbox.start()
    budget = BudgetManager(AgentBudgetConfig())
    episode = DiscoveryGraph(
        MockImageModel(), sandbox, budget, EpisodeLogger(tmp_path / "episode.jsonl")
    )
    state = normalize_terminal_state(episode.invoke())
    assert state["status"] == "submitted"
    assert state["tool_calls"] == 2


def test_graph_budget_stops_without_extra_llm_call():
    model = MockToolCallingModel()
    sandbox = LocalMockSandbox([ExecutionResult(status="success")])
    sandbox.start()
    episode = DiscoveryGraph(
        model,
        sandbox,
        BudgetManager(AgentBudgetConfig(max_tool_calls=1)),
        EpisodeLogger(),
    )
    state = normalize_terminal_state(episode.invoke())
    assert state["status"] == "budget_exceeded"
    assert model.index == 1


def test_budget_prompt_and_tool_feedback(tmp_path):
    budget_config = AgentBudgetConfig(
        max_total_tokens=77_777,
        max_tool_calls=9,
        max_total_execution_seconds=45.0,
        max_wall_time_seconds=99.0,
    )
    budget = BudgetManager(budget_config)
    sandbox = LocalMockSandbox([
        ExecutionResult(status="success", stdout="result_1\n", execution_time_seconds=0.1)
    ])
    sandbox.start()

    class InspectModel:
        def __init__(self):
            self.passed_system_message = None
            self.received_tool_message = None
            self.turn = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            if self.turn == 0:
                self.passed_system_message = messages[0].content
                self.turn += 1
                resp = AIMessage(
                    content="",
                    tool_calls=[{"name": "run_python", "args": {"code": "print(1)"}, "id": "call1"}],
                )
                resp.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
                return resp
            else:
                self.received_tool_message = messages[-1].content
                resp = AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "submit_solution",
                        "args": {
                            "script": SCRIPT,
                            "transformations": [{"column": "x", "operators": ["square"]}],
                        },
                        "id": "call2",
                    }],
                )
                resp.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
                return resp

    model = InspectModel()
    episode = DiscoveryGraph(
        model, sandbox, budget, EpisodeLogger(tmp_path / "ep.jsonl"),
    )
    state = normalize_terminal_state(episode.invoke())
    assert state["status"] == "submitted"
    assert "77,777 tokens" in model.passed_system_message
    assert "9 calls" in model.passed_system_message
    assert "budget_status" in model.received_tool_message

