from src.sandbox.local_mock import LocalMockSandbox
from src.sandbox.models import ExecutionResult


def test_mock_is_deterministic_and_never_executes_code():
    sandbox = LocalMockSandbox([ExecutionResult(status="success", stdout="expected")])
    sandbox.start()
    result = sandbox.execute("raise RuntimeError('would fail if evaluated')")
    assert result.stdout == "expected"
    assert sandbox.executed_code
    sandbox.close()
