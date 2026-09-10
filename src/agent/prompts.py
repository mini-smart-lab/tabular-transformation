SYSTEM_PROMPT = """You are discovering hidden tabular transformations between before.csv and after.csv.

The sandbox contains before.csv and after.csv with anonymized columns. The data was
transformed using canonical operators from this benchmark vocabulary: {operators}.
Transformations can be:
- Independent unary operations (single, chained/nested, piecewise, indicator, or thresholding)
- Dependent multi-column interactions (such as arithmetic operations or conditional selection across columns)

Infer transformations that generalize to unseen rows. Use run_python for focused,
systematic experiments. Write vectorized checks in Python to verify hypotheses efficiently
within your token and tool budgets. Python calls run in isolated executions, so each experiment
should load any data it needs from the visible CSV files.

Finish only by calling submit_solution. Its script must be self-contained, define
exactly `transform(df)`, return a pandas DataFrame with the same rows and expected
columns, and must not read files, use session variables, networks, or external APIs.
Report canonical, ordered operator names per transformed column.
"""


def build_system_prompt(operator_vocabulary: list[str]) -> str:
    operators = ", ".join(operator_vocabulary) if operator_vocabulary else "(not disclosed)"
    return SYSTEM_PROMPT.format(operators=operators)
