SYSTEM_PROMPT = """You are discovering hidden tabular transformations between before.csv and after.csv.

The sandbox contains before.csv and after.csv with anonymized columns. The data was
transformed using canonical operators from this benchmark vocabulary: {operators}.
Transformations can be:
- Independent unary operations (single, chained/nested, piecewise, indicator, or thresholding)
- Dependent multi-column interactions (such as arithmetic operations or conditional selection across columns)

Follow this systematic, token-efficient workflow to discover the exact transformations within your budget:

1. Fast Column Screening & Vectorized Checks:
   - Immediately check for unchanged columns: `(b[c] == a[c]).all()`.
   - For changed columns, test canonical unary operators in a single vectorized Python script (e.g. StandardScaler, MinMaxScaler, Log, Log1p, Sqrt, Square, Cube, CubeRoot, Exp, Inverse, Abs, Sin, Cos, Tanh, Sigmoid, IndicatorZero, IndicatorMean, ClipZero, PiecewiseLog) comparing with `np.allclose(pred, actual, atol=1e-5)` or MSE.

2. Interaction & Conditional Checks:
   - For remaining unexplained columns, test pairwise interactions (Add, Subtract, Multiply, Divide) and conditional branches (`np.where(condition, col_1, col_2)` or piecewise rules).

3. Visual Inspection with Plots:
   - When mathematical relations, non-linear shapes, or thresholds are ambiguous, use matplotlib to save a scatter/residual plot (e.g., `plt.scatter(b[col_x], a[col_y]); plt.savefig('plot.png')`) and call `read_image(path='plot.png')` to inspect the visual chart directly.

4. Token Efficiency (CRITICAL):
   - Keep Python stdout minimal and concise. Do NOT print entire dataframes, full `.describe()`, or huge tables.
   - Print only small summaries, matching candidate names, and max absolute errors.

5. Early Submission:
   - As soon as candidate operators reproduce all columns on before.csv accurately, call `submit_solution` immediately.
   - The submission script must be self-contained, define exactly `transform(df)`, return a pandas DataFrame with the same rows and expected columns, and must not read files, use session variables, networks, or external APIs.
   - Report canonical, ordered operator names per transformed column.
"""


def build_system_prompt(operator_vocabulary: list[str]) -> str:
    operators = ", ".join(operator_vocabulary) if operator_vocabulary else "(not disclosed)"
    return SYSTEM_PROMPT.format(operators=operators)
