from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


SEARCH_SCRIPT_TEMPLATE = r'''import json
import time
import numpy as np
import pandas as pd

start_time = time.monotonic()
max_search_seconds = 12.0

a = pd.read_csv("before.csv")
b = pd.read_csv("after.csv")

numeric_cols = list(a.columns)

# Define canonical unary functions and their code generators
unary_ops = {
    "LogTransformation": (lambda x: np.log(np.where(x > 0, x, np.nan)), lambda col: f"np.log(df['{col}'])"),
    "Log1pTransformation": (lambda x: np.log1p(np.where(x > -1, x, np.nan)), lambda col: f"np.log1p(df['{col}'])"),
    "SquareRootTransformation": (lambda x: np.sqrt(np.where(x >= 0, x, np.nan)), lambda col: f"np.sqrt(np.maximum(df['{col}'], 0.0))"),
    "SquareTransformation": (lambda x: np.square(x), lambda col: f"np.square(df['{col}'])"),
    "CubeTransformation": (lambda x: np.power(x, 3), lambda col: f"np.power(df['{col}'], 3)"),
    "CubeRootTransformation": (lambda x: np.cbrt(x), lambda col: f"np.cbrt(df['{col}'])"),
    "ExpTransformation": (lambda x: np.exp(np.clip(x, -50, 50)), lambda col: f"np.exp(np.clip(df['{col}'], -50, 50))"),
    "InverseTransformation": (lambda x: 1.0 / np.where(x != 0, x, np.nan), lambda col: f"1.0 / np.where(df['{col}'] == 0, 1e-12, df['{col}'])"),
    "StandardScalerTransformation": (lambda x: (x - np.mean(x)) / (np.std(x, ddof=0) if np.std(x, ddof=0) != 0 else 1.0), lambda col: f"(df['{col}'] - df['{col}'].mean()) / df['{col}'].std(ddof=0)"),
    "MinMaxScalerTransformation": (lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) if np.max(x) != np.min(x) else 1.0), lambda col: f"(df['{col}'] - df['{col}'].min()) / (df['{col}'].max() - df['{col}'].min())"),
    "SinTransformation": (lambda x: np.sin(x), lambda col: f"np.sin(df['{col}'])"),
    "CosTransformation": (lambda x: np.cos(x), lambda col: f"np.cos(df['{col}'])"),
    "TanhTransformation": (lambda x: np.tanh(x), lambda col: f"np.tanh(df['{col}'])"),
    "SigmoidTransformation": (lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))), lambda col: f"1.0 / (1.0 + np.exp(-np.clip(df['{col}'], -50, 50)))"),
    "AbsTransformation": (lambda x: np.abs(x), lambda col: f"np.abs(df['{col}'])"),
    "IndicatorZeroTransformation": (lambda x: (x > 0.0).astype(float), lambda col: f"(df['{col}'] > 0.0).astype(float)"),
    "IndicatorMeanTransformation": (lambda x: (x > np.mean(x)).astype(float), lambda col: f"(df['{col}'] > df['{col}'].mean()).astype(float)"),
    "ClipZeroTransformation": (lambda x: np.maximum(x, 0.0), lambda col: f"np.maximum(df['{col}'], 0.0)"),
    "PiecewiseLogTransformation": (lambda x: np.where(x > 1.0, np.log(np.where(x > 0, x, 1e-12)), x), lambda col: f"np.where(df['{col}'] > 1.0, np.log(np.maximum(df['{col}'], 1e-12)), df['{col}'])"),
}

discovered = {}
transformations_list = []

# Step 1: Identify identical columns
for col in b.columns:
    if col in a.columns and np.allclose(a[col].values, b[col].values, equal_nan=True):
        discovered[col] = ([], None)

remaining_cols = [c for c in b.columns if c not in discovered]

# Step 2: Level 1 - Single Unary Search
for col in remaining_cols:
    if time.monotonic() - start_time > max_search_seconds:
        break
    x = a[col].values.astype(float)
    y = b[col].values.astype(float)
    for op_name, (fn, code_fn) in unary_ops.items():
        try:
            pred = fn(x)
            if np.allclose(pred, y, atol=1e-5, equal_nan=True):
                code_expr = code_fn(col)
                discovered[col] = ([op_name], f"out['{col}'] = {code_expr}")
                transformations_list.append({"column": col, "operators": [op_name]})
                break
        except Exception:
            pass

remaining_cols = [c for c in b.columns if c not in discovered]

# Step 3: Level 2 - Depth 2 Composed Unary Search: op1(op2(x))
for col in remaining_cols:
    if time.monotonic() - start_time > max_search_seconds:
        break
    x = a[col].values.astype(float)
    y = b[col].values.astype(float)
    found = False
    for op1_name, (fn1, code_fn1) in unary_ops.items():
        if found or time.monotonic() - start_time > max_search_seconds:
            break
        for op2_name, (fn2, code_fn2) in unary_ops.items():
            try:
                pred = fn2(fn1(x))
                if np.allclose(pred, y, atol=1e-5, equal_nan=True):
                    inner_expr = code_fn1(col)
                    # replace the inner df[col] with the expression for op2
                    # simple format for depth-2
                    discovered[col] = ([op1_name, op2_name], f"out['{col}'] = {op2_name}__call__")
                    # For safety, let's create a clean nested function call
                    # We can use numpy functions for standard ops
                    # Or write a clean composed line:
                    if op2_name == "SquareTransformation":
                        outer_line = f"np.square({inner_expr})"
                    elif op2_name == "SquareRootTransformation":
                        outer_line = f"np.sqrt(np.maximum({inner_expr}, 0.0))"
                    elif op2_name == "CubeTransformation":
                        outer_line = f"np.power({inner_expr}, 3)"
                    elif op2_name == "CubeRootTransformation":
                        outer_line = f"np.cbrt({inner_expr})"
                    elif op2_name == "LogTransformation":
                        outer_line = f"np.log(np.maximum({inner_expr}, 1e-12))"
                    elif op2_name == "Log1pTransformation":
                        outer_line = f"np.log1p({inner_expr})"
                    elif op2_name == "ExpTransformation":
                        outer_line = f"np.exp(np.clip({inner_expr}, -50, 50))"
                    elif op2_name == "SinTransformation":
                        outer_line = f"np.sin({inner_expr})"
                    elif op2_name == "CosTransformation":
                        outer_line = f"np.cos({inner_expr})"
                    elif op2_name == "TanhTransformation":
                        outer_line = f"np.tanh({inner_expr})"
                    elif op2_name == "SigmoidTransformation":
                        outer_line = f"1.0 / (1.0 + np.exp(-np.clip({inner_expr}, -50, 50)))"
                    elif op2_name == "AbsTransformation":
                        outer_line = f"np.abs({inner_expr})"
                    elif op2_name == "ClipZeroTransformation":
                        outer_line = f"np.maximum({inner_expr}, 0.0)"
                    else:
                        outer_line = f"{inner_expr}"
                    discovered[col] = ([op1_name, op2_name], f"out['{col}'] = {outer_line}")
                    transformations_list.append({"column": col, "operators": [op1_name, op2_name]})
                    found = True
                    break
            except Exception:
                pass

remaining_cols = [c for c in b.columns if c not in discovered]

# Step 4: Level 3 & 4 - Inter-Column Pairwise & Conditional Interaction Search
for col in remaining_cols:
    if time.monotonic() - start_time > max_search_seconds:
        break
    y = b[col].values.astype(float)
    found = False
    for c1 in numeric_cols:
        if found or time.monotonic() - start_time > max_search_seconds:
            break
        x1 = a[c1].values.astype(float)
        for c2 in numeric_cols:
            if found or time.monotonic() - start_time > max_search_seconds:
                break
            x2 = a[c2].values.astype(float)
            # Add
            try:
                if np.allclose(x1 + x2, y, atol=1e-5, equal_nan=True):
                    discovered[col] = (["AddInteractionTransformation"], f"out['{col}'] = df['{c1}'] + df['{c2}']")
                    transformations_list.append({"column": col, "operators": ["AddInteractionTransformation"], "source_columns": [c1, c2], "interaction": "add"})
                    found = True
                    break
            except Exception:
                pass
            # Subtract
            try:
                if np.allclose(x1 - x2, y, atol=1e-5, equal_nan=True):
                    discovered[col] = (["SubtractInteractionTransformation"], f"out['{col}'] = df['{c1}'] - df['{c2}']")
                    transformations_list.append({"column": col, "operators": ["SubtractInteractionTransformation"], "source_columns": [c1, c2], "interaction": "subtract"})
                    found = True
                    break
            except Exception:
                pass
            # Multiply
            try:
                if np.allclose(x1 * x2, y, atol=1e-5, equal_nan=True):
                    discovered[col] = (["MultiplyInteractionTransformation"], f"out['{col}'] = df['{c1}'] * df['{c2}']")
                    transformations_list.append({"column": col, "operators": ["MultiplyInteractionTransformation"], "source_columns": [c1, c2], "interaction": "multiply"})
                    found = True
                    break
            except Exception:
                pass
            # Divide
            try:
                if np.allclose(x1 / np.where(x2 != 0, x2, np.nan), y, atol=1e-5, equal_nan=True):
                    discovered[col] = (["DivideInteractionTransformation"], f"out['{col}'] = df['{c1}'] / np.where(df['{c2}'] == 0, 1e-12, df['{c2}'])")
                    transformations_list.append({"column": col, "operators": ["DivideInteractionTransformation"], "source_columns": [c1, c2], "interaction": "divide"})
                    found = True
                    break
            except Exception:
                pass
            # Conditional Interaction: where(c1 > mean(c1), c1, c2)
            try:
                cond_pred = np.where(x1 > np.mean(x1), x1, x2)
                if np.allclose(cond_pred, y, atol=1e-5, equal_nan=True):
                    discovered[col] = (["ConditionalInteractionTransformation"], f"out['{col}'] = np.where(df['{c1}'] > df['{c1}'].mean(), df['{c1}'], df['{c2}'])")
                    transformations_list.append({"column": col, "operators": ["ConditionalInteractionTransformation"], "source_columns": [c1, c2], "interaction": "conditional"})
                    found = True
                    break
            except Exception:
                pass

# Assemble transform(df) script
script_lines = [
    "import numpy as np",
    "import pandas as pd",
    "",
    "def transform(df: pd.DataFrame) -> pd.DataFrame:",
    "    out = df.copy()",
]

for col, (ops, line) in discovered.items():
    if line:
        script_lines.append(f"    {line}")

script_lines.append("    return out[df.columns]")
final_script = "\n".join(script_lines) + "\n"

output_payload = {
    "status": "success",
    "script": final_script,
    "transformations": transformations_list,
}
print("__BRUTE_FORCE_RESULT__" + json.dumps(output_payload) + "__BRUTE_FORCE_RESULT__")
'''


class BruteForceModel:
    """Non-LLM Baseline model performing systematic symbolic/grid search within time limits."""

    def __init__(self, config: Any = None, timeout_seconds: float = 12.0) -> None:
        self.config = config
        self.timeout_seconds = timeout_seconds
        self.tools: list[Any] = []
        self.turn = 0

    def bind_tools(self, tools: list[Any]) -> BruteForceModel:
        self.tools = tools
        return self

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        # Check if we have received tool execution results
        last_tool_message = None
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                last_tool_message = msg
                break

        if last_tool_message is None:
            # Turn 0: Initiate the search script execution
            self.turn += 1
            return AIMessage(
                content="Initiating systematic brute force symbolic search on before.csv and after.csv.",
                tool_calls=[{
                    "name": "run_python",
                    "args": {"code": SEARCH_SCRIPT_TEMPLATE},
                    "id": "brute_force_search_call",
                }],
                usage_metadata={"input_tokens": 50, "output_tokens": 50, "total_tokens": 100},
            )
        else:
            # Turn 1: Parse search results from tool execution and submit
            content_str = str(last_tool_message.content)
            stdout_text = content_str
            try:
                parsed_tool_data = json.loads(content_str)
                if isinstance(parsed_tool_data, dict) and "stdout" in parsed_tool_data:
                    stdout_text = parsed_tool_data["stdout"]
            except Exception:
                pass

            script = "import numpy as np\nimport pandas as pd\n\ndef transform(df: pd.DataFrame) -> pd.DataFrame:\n    return df.copy()\n"
            transformations: list[dict[str, Any]] = []

            if "__BRUTE_FORCE_RESULT__" in stdout_text:
                try:
                    payload_raw = stdout_text.split("__BRUTE_FORCE_RESULT__")[1]
                    payload = json.loads(payload_raw)
                    script = payload.get("script", script)
                    transformations = payload.get("transformations", [])
                except Exception:
                    pass

            self.turn += 1
            return AIMessage(
                content="Brute force search complete. Submitting discovered transformations.",
                tool_calls=[{
                    "name": "submit_solution",
                    "args": {
                        "script": script,
                        "transformations": transformations,
                    },
                    "id": "brute_force_submit_call",
                }],
                usage_metadata={"input_tokens": 50, "output_tokens": 50, "total_tokens": 100},
            )
