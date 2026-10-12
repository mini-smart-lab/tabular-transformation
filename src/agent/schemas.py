from __future__ import annotations

import ast
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_ALLOWED_IMPORT_ROOTS = {"math", "numpy", "pandas"}
_BLOCKED_CALLS = {"compile", "eval", "exec", "globals", "input", "locals", "open", "__import__"}
_BLOCKED_IO_ATTRIBUTES = {
    "read_csv", "read_excel", "read_feather", "read_hdf", "read_html", "read_json",
    "read_parquet", "read_pickle", "read_sql", "to_csv", "to_excel", "to_feather",
    "to_hdf", "to_json", "to_parquet", "to_pickle",
}


class Transformation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str = Field(min_length=1)
    operators: list[str]
    source_columns: list[str] | None = None
    interaction: str | None = None
    parameters: list[dict[str, Any]] | None = None

    @field_validator("operators")
    @classmethod
    def operators_must_be_nonempty_names(cls, value: list[str]) -> list[str]:
        if not value or any(not item.strip() for item in value):
            raise ValueError("operators must contain at least one non-empty canonical name")
        return value


class FinalSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    script: str = Field(min_length=1)
    transformations: list[Transformation]

    @field_validator("script")
    @classmethod
    def validate_entry_point(cls, script: str) -> str:
        try:
            tree = ast.parse(script)
        except SyntaxError as exc:
            raise ValueError(f"script is not valid Python: {exc.msg}") from exc
        transforms = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "transform"
        ]
        if len(transforms) != 1 or isinstance(transforms[0], ast.AsyncFunctionDef):
            raise ValueError("script must define exactly one synchronous transform(df) function")
        args = transforms[0].args
        if len(args.args) != 1 or args.vararg or args.kwarg or args.kwonlyargs:
            raise ValueError("transform must accept exactly one positional argument")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                if any(name.split(".")[0] not in _ALLOWED_IMPORT_ROOTS for name in names):
                    raise ValueError("only pandas, numpy, and math imports are allowed")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in _BLOCKED_CALLS:
                    raise ValueError(f"call to {node.func.id} is not allowed")
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise ValueError("dunder attribute access is not allowed")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in _BLOCKED_IO_ATTRIBUTES:
                    raise ValueError("final transform script may not perform file or URL I/O")
        public_functions = [
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
        ]
        if public_functions != ["transform"]:
            raise ValueError("transform must be the only public top-level function")
        return script

    @model_validator(mode="after")
    def unique_columns(self) -> "FinalSubmission":
        columns = [item.column for item in self.transformations]
        if len(columns) != len(set(columns)):
            raise ValueError("each transformed column may be reported only once")
        return self


class SubmissionAcknowledgement(BaseModel):
    accepted: bool
    message: str
