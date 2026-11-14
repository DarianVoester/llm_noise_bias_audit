"""Sandboxed execution of user-editable analysis code.

Layers (see SECURITY.md): AST validation -> restricted globals -> fresh Python
interpreter (``python -I src/sandbox_child.py``, i.e. a spawned process that shares no
memory with the Streamlit server) with an allowlisted environment -> audit hook (no
network, no processes, file reads only inside the Python installation) -> resource limits
-> 5 s timer around the user code plus a hard kill -> validated, JSON-only payload.

This is defense in depth, NOT a hard security boundary. The container is.
"""

from __future__ import annotations

import ast
import json
import os
import traceback
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.config import SANDBOX_CPU_SECONDS, SANDBOX_MEMORY_BYTES, SANDBOX_TIMEOUT_S
from src.schemas import decode_value
from src.security import sandbox_environ

FORBIDDEN_NAMES = frozenset({
    "__import__", "open", "exec", "eval", "compile", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "hasattr", "input", "breakpoint", "help", "memoryview",
    "type", "object", "super", "dir", "id", "classmethod", "staticmethod", "property",
    "exit", "quit", "copyright", "credits", "license", "__builtins__", "__loader__",
    "__spec__", "os", "sys", "subprocess", "importlib", "builtins", "io", "pickle",
    "shutil", "socket", "ctypes",
})

FORBIDDEN_ATTRS = frozenset({
    "format", "format_map", "load", "loads", "save", "savez", "savez_compressed", "savetxt",
    "loadtxt", "genfromtxt", "fromfile", "tofile", "memmap", "eval", "query", "system",
    "popen", "pipe", "apply_along_axis", "frompyfunc", "vectorize", "ctypeslib", "lib",
    "testing", "f2py", "distutils", "io", "api", "plotting", "write_html", "write_image",
    "write_json", "show", "open", "mro", "gi_frame", "gi_code", "f_globals", "f_locals",
    "f_builtins", "co_code", "func_globals", "tb_frame", "cr_frame", "ag_frame", "style",
    "fromregex", "DataSource", "ExcelWriter", "HDFStore", "set_option", "options",
    # names of dangerous modules that might be reachable as attributes of injected modules
    "os", "sys", "subprocess", "builtins", "importlib", "shutil", "socket", "pickle",
    "ctypes", "pathlib", "threading", "multiprocessing", "signal", "platform", "tempfile",
    "urllib", "http", "requests", "glob", "codecs", "marshal", "gc", "inspect", "runpy",
    "code", "pty", "webbrowser", "warnings", "logging", "weakref", "functools", "operator",
    "copyreg", "types", "abc", "contextlib", "atexit", "resource", "mmap", "select",
    "asyncio", "concurrent", "zipfile", "tarfile", "gzip", "bz2", "lzma", "sqlite3",
    "dbm", "shelve", "json", "csv", "xml", "html", "email", "ssl", "hashlib", "uuid",
    "secrets", "time", "datetime", "locale", "textwrap", "string", "re", "ExcelFile",
    "compat", "util", "utils", "core", "base", "internals", "config", "basedatatypes",
    # dangerous builtins must not be reachable as attributes either (``mod.getattr``) ...
    "getattr", "setattr", "delattr", "hasattr", "exec", "compile", "globals", "locals",
    "vars", "breakpoint", "input",
    # ... nor helpers that some library module happens to re-export at module level
    "import_module", "Popen", "check_output", "check_call", "environ", "getenv", "putenv",
    "listdir", "scandir", "unlink", "rmtree", "urlopen", "dump", "test", "nw",
    "option_context", "reset_option", "get_option", "show_versions", "show_config",
})

# Methods that accept a *string* naming another method (e.g. ``df.agg("to_pickle")``).
# String arguments to these must come from SAFE_FUNC_NAMES.
DISPATCH_METHODS = frozenset({
    "agg", "aggregate", "apply", "transform", "pipe", "resample",
    "rolling", "expanding", "ewm",
})
SAFE_FUNC_NAMES = frozenset({
    "mean", "median", "std", "var", "sum", "count", "min", "max", "size", "nunique",
    "first", "last", "prod", "sem", "quantile", "idxmax", "idxmin", "any", "all", "cumsum",
    "cumprod", "cummax", "cummin", "rank", "abs", "skew", "kurt", "mode", "unique",
})
FORBIDDEN_ATTR_PREFIXES = ("_", "read_", "to_", "from_")
ALLOWED_TO_ATTRS = frozenset({"to_dict", "to_list", "to_numpy", "to_numeric", "to_frame",
                               "to_records", "to_plotly_json"})

FORBIDDEN_NODES: tuple[type[ast.AST], ...] = (
    ast.Import, ast.ImportFrom, ast.While, ast.Global, ast.Nonlocal, ast.ClassDef,
    ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await, ast.Yield, ast.YieldFrom,
)

SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float",
    "frozenset", "int", "isinstance", "len", "list", "map", "max", "min", "pow", "print",
    "range", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    "ArithmeticError", "Exception", "IndexError", "KeyError", "TypeError", "ValueError",
    "ZeroDivisionError", "True", "False", "None",
)

MAX_CODE_CHARS = 50_000
MAX_RESULT_BYTES = 20 * 1024 * 1024


@dataclass
class Violation:
    line: int
    message: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.message}"


class _Validator(ast.NodeVisitor):
    def __init__(self, defined_functions: set[str]) -> None:
        self.violations: list[Violation] = []
        self.defined_functions = defined_functions
        # ids of attribute nodes that are the callee of a call, e.g. ``df.agg`` in ``df.agg(..)``
        self._called_attrs: set[int] = set()

    def _add(self, node: ast.AST, msg: str) -> None:
        self.violations.append(Violation(getattr(node, "lineno", 0), msg))

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, FORBIDDEN_NODES):
            self._add(node, f"'{type(node).__name__}' statements are not allowed "
                            "(no imports, while-loops, classes, globals or async code)")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        # a lone "_" is the conventional throw-away variable and is harmless
        if node.id in FORBIDDEN_NAMES or (node.id.startswith("_") and node.id != "_"):
            self._add(node, f"name '{node.id}' is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        a = node.attr
        if a in FORBIDDEN_ATTRS or (a.startswith(FORBIDDEN_ATTR_PREFIXES)
                                    and a not in ALLOWED_TO_ATTRS):
            self._add(node, f"attribute '.{a}' is not allowed")
        elif a in DISPATCH_METHODS and id(node) not in self._called_attrs:
            # ``f = df.agg; f("to_pickle")`` would slip past the argument check in visit_Call
            self._add(node, f"'.{a}' must be called directly, e.g. df.{a}(...); it cannot be "
                            "stored in a variable or passed on")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name.startswith("_"):
            self._add(node, f"function name '{node.name}' is not allowed")
        if node.decorator_list:
            self._add(node, "decorators are not allowed")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and "__" in node.value:
            self._add(node, "string constants containing '__' are not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            self._called_attrs.add(id(func))
        if isinstance(func, ast.Attribute) and func.attr in DISPATCH_METHODS:
            for arg in [*node.args, *(k.value for k in node.keywords)]:
                self._check_dispatch_arg(arg, func.attr)
        self.generic_visit(node)

    def _check_dispatch_arg(self, arg: ast.AST, method: str) -> None:
        """Only literal safe names, lambdas, attributes (np.mean) or numbers may be passed."""
        if isinstance(arg, ast.Constant):
            if isinstance(arg.value, str) and arg.value not in SAFE_FUNC_NAMES:
                self._add(arg, f"'.{method}(\"{arg.value}\")' is not allowed; "
                               f"use one of: {', '.join(sorted(SAFE_FUNC_NAMES))}")
        elif (isinstance(arg, ast.Tuple) and len(arg.elts) == 2
              and isinstance(arg.elts[0], ast.Constant)):
            # pandas named aggregation: agg(new_col=("source_col", "func"))
            self._check_dispatch_arg(arg.elts[1], method)
        elif isinstance(arg, (ast.List, ast.Tuple, ast.Set)):
            for elt in arg.elts:
                self._check_dispatch_arg(elt, method)
        elif isinstance(arg, ast.Dict):
            for v in arg.values:
                self._check_dispatch_arg(v, method)
        elif isinstance(arg, (ast.Lambda, ast.Attribute)):
            return
        elif isinstance(arg, ast.Name) and arg.id in self.defined_functions | set(
                SAFE_BUILTIN_NAMES):
            return
        else:
            self._add(arg, f"arguments to '.{method}()' must be a literal function name, a "
                           "lambda, a function defined in this script, or e.g. np.mean")


def validate_code(code: str) -> list[Violation]:
    """Statically check user code. Empty list means the code passed validation."""
    if len(code) > MAX_CODE_CHARS:
        return [Violation(0, f"code is longer than {MAX_CODE_CHARS} characters")]
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return [Violation(exc.lineno or 0, f"syntax error: {exc.msg}")]
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    v = _Validator(defined)
    v.visit(tree)
    has_evaluate = any(isinstance(n, ast.FunctionDef) and n.name == "evaluate"
                       for n in tree.body)
    if not has_evaluate:
        v.violations.append(Violation(0, "code must define a top-level function "
                                         "'evaluate(df_results, ground_truth=None)'"))
    return v.violations


# --- Execution -------------------------------------------------------------

@dataclass
class EvaluationResult:
    ok: bool
    summary_metrics: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)  # Plotly figure JSON strings
    warnings: list[str] = field(default_factory=list)
    interpretation: list[str] = field(default_factory=list)
    error: str | None = None
    violations: list[Violation] = field(default_factory=list)
    logs: str = ""
    duration_ms: float = 0.0


_CHILD_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_child.py")
_RESULT_MARKER = "@@SANDBOX_RESULT@@"
_SANDBOX_LIBRARIES = ("numpy", "pandas", "scipy", "plotly")


def _library_paths() -> list[str]:
    """site-packages directories that hold the sandbox's scientific libraries.

    The child runs with ``python -I`` (isolated mode), which also drops the *user*
    site-packages (``~/.local/...``). If the libraries were installed there (e.g. a
    ``pip install --user`` in a Codespace), the child could not import numpy. Passing
    exactly these directories keeps isolation (no PYTHONPATH, no cwd, no user code dirs)
    while making the analysis work in every installation layout.
    """
    import importlib.util

    paths: list[str] = []
    for name in _SANDBOX_LIBRARIES:
        spec = importlib.util.find_spec(name)
        locations = list(spec.submodule_search_locations or []) if spec else []
        for loc in locations[:1]:
            parent = os.path.dirname(os.path.abspath(loc))
            if parent not in paths:
                paths.append(parent)
    return paths


def _decode_table(obj: Any) -> pd.DataFrame | None:
    if isinstance(obj, dict) and "__dataframe__" in obj:
        d = obj["__dataframe__"]
        return pd.DataFrame(d.get("data", []), columns=d.get("columns"), index=d.get("index"))
    return None


def _unpack(result: dict[str, Any]) -> EvaluationResult:
    """Convert the child's JSON payload into an EvaluationResult (validating its shape)."""
    res = EvaluationResult(ok=True)
    summary = result.get("summary_metrics", {})
    res.summary_metrics = summary if isinstance(summary, dict) else {"value": summary}
    tables_raw = result.get("tables", {})
    if "per_model_table" in result:
        tables_raw = {"per_model_table": result["per_model_table"], **(tables_raw or {})}
    if isinstance(tables_raw, dict):
        for name, obj in tables_raw.items():
            table = _decode_table(obj)
            if table is not None:
                res.tables[str(name)] = table
    figs = result.get("figures", [])
    if isinstance(figs, dict):
        figs = list(figs.values())
    for f in figs if isinstance(figs, list) else []:
        if isinstance(f, dict) and isinstance(f.get("__figure__"), str):
            res.figures.append(f["__figure__"])
    for key in ("warnings", "interpretation"):
        val = result.get(key, [])
        items = [val] if isinstance(val, str) else val if isinstance(val, list) else []
        setattr(res, key, [str(x) for x in items])
    return res


def with_decoded_values(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``parsed_value`` (float, str or list) because the sandbox has no ``json`` module."""
    if "parsed_value_num" not in df.columns:
        return df
    out = df.copy()
    nums = out["parsed_value_num"].tolist()
    jsons = out["parsed_value_json"].tolist() if "parsed_value_json" in out.columns \
        else [None] * len(out)
    out["parsed_value"] = [decode_value(n, j) for n, j in zip(nums, jsons, strict=True)]
    return out


def run_sandboxed(
    code: str,
    df: pd.DataFrame,
    ground_truth: float | None = None,
    timeout_s: float = SANDBOX_TIMEOUT_S,
    mem_bytes: int = SANDBOX_MEMORY_BYTES,
    cpu_seconds: int = SANDBOX_CPU_SECONDS,
) -> EvaluationResult:
    """Validate and execute ``evaluate(df_results, ground_truth)`` in an isolated process."""
    import subprocess
    import sys

    violations = validate_code(code)
    if violations:
        return EvaluationResult(ok=False, violations=violations,
                                error="Code rejected by the security validator.")

    df = with_decoded_values(df)
    job = {
        "code": code,
        "df": json.loads(df.to_json(orient="split", date_format="iso", default_handler=str)),
        "ground_truth": ground_truth,
        "builtins": list(SAFE_BUILTIN_NAMES),
        "lib_paths": _library_paths(),
        "mem_bytes": mem_bytes,
        "cpu_seconds": cpu_seconds,
        "timeout_s": timeout_s,
    }
    env = sandbox_environ()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Grace period on top of the in-child timer covers interpreter + library start-up.
    hard_timeout = timeout_s + 25.0
    try:
        proc = subprocess.run(
            [sys.executable, "-I", _CHILD_SCRIPT],
            input=json.dumps(job),
            capture_output=True,
            text=True,
            timeout=hard_timeout,
            env=env,
            cwd=os.path.dirname(_CHILD_SCRIPT),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return EvaluationResult(ok=False, error=f"Timeout: analysis killed after "
                                                f"{hard_timeout:.0f} s.")
    stdout = proc.stdout or ""
    if _RESULT_MARKER not in stdout:
        err = (proc.stderr or "").strip().splitlines()[-3:]
        reason = "killed (CPU or memory limit exceeded)" if proc.returncode and \
            proc.returncode < 0 else f"exit code {proc.returncode}"
        return EvaluationResult(ok=False, error=f"Analysis process {reason}. "
                                                + " | ".join(err)[:500])
    payload_text = stdout.split(_RESULT_MARKER, 1)[1]
    if len(payload_text) > MAX_RESULT_BYTES:
        return EvaluationResult(ok=False, error="Result too large (> 20 MB).")
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        return EvaluationResult(ok=False, error=f"Invalid result payload: {exc}")
    if not payload.get("ok"):
        return EvaluationResult(ok=False, error=str(payload.get("error")),
                                logs=str(payload.get("logs", "")),
                                duration_ms=float(payload.get("duration_ms", 0.0)))
    try:
        res = _unpack(payload["result"])
    except Exception as exc:  # malformed but JSON-valid output
        return EvaluationResult(ok=False, error=f"Unexpected result structure: {exc}",
                                logs=traceback.format_exc(limit=1))
    res.logs = str(payload.get("logs", ""))
    res.duration_ms = float(payload.get("duration_ms", 0.0))
    return res

