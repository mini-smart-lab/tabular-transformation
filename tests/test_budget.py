from src.agent.budget import BudgetManager
from src.config import AgentBudgetConfig


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


def test_budget_counters_accumulate():
    manager = BudgetManager(AgentBudgetConfig(), clock=Clock())
    manager.record_llm_usage(10, 4)
    manager.record_llm_usage(3, 2)
    manager.record_tool_call()
    manager.record_execution_time(1.25)
    assert manager.snapshot()["total_tokens"] == 19
    assert manager.tool_calls == 1
    assert manager.execution_time_seconds == 1.25


def test_each_hard_limit_triggers():
    clock = Clock()
    cases = [
        ("tokens", lambda b: b.record_llm_usage(5, 0)),
        ("tool_calls", lambda b: b.record_tool_call()),
        ("execution_time", lambda b: b.record_execution_time(5)),
        ("cpu_time", lambda b: b.record_cpu_time(5)),
    ]
    for reason, action in cases:
        manager = BudgetManager(
            AgentBudgetConfig(
                max_total_tokens=5,
                max_tool_calls=1,
                max_total_execution_seconds=5,
                max_wall_time_seconds=5,
                max_total_cpu_seconds=5,
            ),
            clock=clock,
        )
        action(manager)
        assert manager.check().reason == reason

    manager = BudgetManager(AgentBudgetConfig(max_wall_time_seconds=5), clock=clock)
    clock.value = 6
    assert manager.check().reason == "wall_time"


def test_custom_level_budgets_in_harness_config():
    import yaml
    from src.config import HarnessConfig, get_level_budget

    yaml_text = """
    level_budgets:
      3:
        max_total_tokens: 99999
        max_tool_calls: 33
        max_total_execution_seconds: 111
        max_total_cpu_seconds: 111
        max_wall_time_seconds: 222
    """
    config = HarnessConfig.model_validate(yaml.safe_load(yaml_text))
    assert 3 in config.level_budgets
    assert config.level_budgets[3].max_total_tokens == 99999

    budget = get_level_budget(3, level_budgets=config.level_budgets)
    assert budget.max_total_tokens == 99999
    assert budget.max_tool_calls == 33

    # level not in custom dict falls back to defaults
    default_b1 = get_level_budget(1, level_budgets=config.level_budgets)
    assert default_b1.max_total_tokens == 40000

