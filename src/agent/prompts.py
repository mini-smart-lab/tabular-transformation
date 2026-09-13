from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """You are discovering hidden tabular transformations between before.csv and after.csv.

The sandbox contains before.csv and after.csv with anonymized columns. The data was
transformed using canonical operators from this benchmark vocabulary: {operators}.
Transformations can be:
- Independent unary operations (single, chained/nested, piecewise, indicator, or thresholding)
- Dependent multi-column interactions (such as arithmetic operations or conditional selection across columns)

HARD BUDGET CONSTRAINTS FOR THIS EPISODE:
- Max Total Tokens (Input + Output): {max_tokens} tokens
- Max Tool Calls: {max_tool_calls} calls
- Max Episode Wall-Time: {max_wall_time}s (execution timeout: {execution_timeout}s per code run)
- Max CPU Time: {max_cpu_time}s

Follow this systematic, token-efficient workflow to discover the exact transformations within your budget:

1. Fast Column Screening & Batch Testing:
   - Immediately identify unchanged columns: `[c for c in b.columns if np.allclose(a[c], b[c], equal_nan=True)]`.
   - For changed columns, test canonical operators from the vocabulary in a single vectorized Python script.
   - Test unary operators: Log, Log1p, Sqrt, Square, Cube, CubeRoot, Exp, Inverse, Abs, StandardScaler, MinMaxScaler, Sin, Cos, Tanh, Sigmoid, IndicatorZero, IndicatorMean, ClipZero, PiecewiseLog.
   - For chained functions (e.g. `f(g(x))`), test compositions systematically over the vocabulary candidates.

2. Interaction & Conditional Checks:
   - For multi-column interactions, test pairwise operators (Add, Subtract, Multiply, Divide) and conditional branches (`np.where(A > threshold, A, B)` or `np.where(condition, A, 0)`).

3. Visual Inspection with Charts:
   - When mathematical relations or piecewise thresholds are ambiguous from numbers, do NOT dump long arrays. Instead, save a scatter/residual plot using matplotlib (e.g. `plt.scatter(a[col_x], b[col_y]); plt.savefig('plot.png')`) and call `read_image(path='plot.png')` to inspect the relationship visually.

4. Token & Context Conservation (CRITICAL TO AVOID BUDGET_EXCEEDED):
   - Keep Python stdout EXTREMELY minimal and concise (under 10 lines per run).
   - STRICTLY FORBIDDEN: Do NOT print entire DataFrames, full tables, `.to_string()`, `.to_dict()`, `.describe()`, or long numpy arrays. Every character printed into stdout inflates all future turns and exhausts your token budget.
   - Print ONLY summary scalars: matched column names, best operator name, max absolute error (`np.max(np.abs(pred - actual))`), and boolean flags (`allclose: True/False`).
   - If manual inspection is necessary, inspect at most 2 rows (e.g. `print(df.head(2).to_dict())`).

5. Immediate Submission & Risk Management:
   - Plan your exploration to complete within your call budget ({max_tool_calls} calls).
   - As soon as candidate operators reproduce all columns on before.csv within tolerance (`atol=1e-5`), call `submit_solution` immediately. Do not waste extra turns.
   - RISK MANAGEMENT: If you have consumed > 70% of your tool calls or token budget, do NOT continue guessing indefinitely. Prioritize submitting your best discovered `transform(df)` immediately. A partially solved submission earns credit for the correct columns, whereas exceeding the budget terminates the episode with an automatic 0.0 penalty.
   - The submission script must define exactly `def transform(df: pd.DataFrame) -> pd.DataFrame:`, return a DataFrame with matching shape and column names, and be fully self-contained (no reading files, no network).
   - Report canonical, ordered operator names per transformed column.
"""


def build_system_prompt(
    operator_vocabulary: list[str] | None = None,
    budget_config: Any = None,
    sandbox_config: Any = None,
) -> str:
    operators = ", ".join(operator_vocabulary) if operator_vocabulary else "(not disclosed)"
    max_tokens = f"{budget_config.max_total_tokens:,}" if budget_config else "100,000"
    max_tool_calls = str(budget_config.max_tool_calls) if budget_config else "25"
    max_wall_time = str(int(budget_config.max_wall_time_seconds)) if budget_config else "300"
    max_cpu_time = str(int(budget_config.max_total_cpu_seconds)) if budget_config else "120"
    execution_timeout = str(int(sandbox_config.execution_timeout_seconds)) if sandbox_config else "15"

    return SYSTEM_PROMPT.format(
        operators=operators,
        max_tokens=max_tokens,
        max_tool_calls=max_tool_calls,
        max_wall_time=max_wall_time,
        max_cpu_time=max_cpu_time,
        execution_timeout=execution_timeout,
    )
