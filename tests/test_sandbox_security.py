"""Security tests for the analysis sandbox (AST validator + isolated process)."""

# Note: tests/test_sandbox_paths.py covers the library-path handover for user installs.

import os

import pandas as pd
import pytest

from src.evaluator import run_sandboxed, validate_code

DF = pd.DataFrame({"model": ["a", "a", "b", "b"], "parsed_value_num": [1.0, 2.0, 3.0, 5.0]})


def wrap(body: str) -> str:
    return "def evaluate(df_results, ground_truth=None):\n" + "\n".join(
        "    " + line for line in body.splitlines()) + "\n"


STATIC_ATTACKS = {
    "import": "import os\ndef evaluate(df_results, ground_truth=None):\n    return {}",
    "from_import": "from os import system\ndef evaluate(df_results, ground_truth=None):\n    return {}",
    "dunder_import": wrap("__import__('subprocess')\nreturn {}"),
    "while_true": wrap("while True:\n    pass"),
    "class_bases": wrap("return {'x': ().__class__.__bases__}"),
    "subclasses": wrap("return {'x': [].__class__.__mro__[1].__subclasses__()}"),
    "format_escape": wrap("return {'x': '{0.__class__}'.format(1)}"),
    "format_map": wrap("return {'x': '{a}'.format_map({'a': 1})}"),
    "read_pickle": wrap("return {'x': pd.read_pickle('x')}"),
    "read_csv": wrap("return {'x': pd.read_csv('/etc/passwd')}"),
    "to_csv": wrap("df_results.to_csv('x')\nreturn {}"),
    "to_pickle": wrap("df_results.to_pickle('x')\nreturn {}"),
    "np_load": wrap("return {'x': np.load('x', allow_pickle=True)}"),
    "np_fromfile": wrap("return {'x': np.fromfile('/etc/passwd')}"),
    "df_eval": wrap("return {'x': df_results.eval('1+1')}"),
    "df_query": wrap("return {'x': df_results.query('model == \"a\"')}"),
    "open": wrap("return {'x': open('/etc/passwd').read()}"),
    "getattr": wrap("return {'x': getattr(np, 'load')}"),
    "eval": wrap("return {'x': eval('1+1')}"),
    "exec": wrap("exec('x=1')\nreturn {}"),
    "compile": wrap("return {'x': compile('1', 'f', 'eval')}"),
    "globals": wrap("return {'x': globals()}"),
    "vars": wrap("return {'x': vars(np)}"),
    "type_trick": wrap("return {'x': type(1)}"),
    "agg_string_dispatch": wrap("return {'x': df_results.agg('to_pickle', 'x')}"),
    "apply_name_dispatch": wrap("f = open\nreturn {'x': df_results.apply(f)}"),
    "np_module_escape": wrap("return {'x': np.os.system('id')}"),
    "pandas_io": wrap("return {'x': pd.io.common}"),
    "plotly_write": wrap("px.scatter(x=[1]).write_html('/tmp/x.html')\nreturn {}"),
    "dunder_string": wrap("return {'x': '__class__'}"),
    "global_stmt": "x = 1\ndef evaluate(df_results, ground_truth=None):\n    global x\n    return {}",
    "class_def": "class A:\n    pass\ndef evaluate(df_results, ground_truth=None):\n    return {}",
    "async_def": "async def evaluate(df_results, ground_truth=None):\n    return {}",
    "decorator": "@np.vectorize\ndef evaluate(df_results, ground_truth=None):\n    return {}",
    "missing_evaluate": "def something_else():\n    return {}",
    "fstring_dunder": wrap("return {'x': f'{df_results.__class__}'}"),
    "scipy_os": wrap("return {'x': stats.os}"),
    "plotly_io": wrap("return {'x': go.io}"),
    "np_ctypeslib": wrap("return {'x': np.ctypeslib}"),
    "np_lib": wrap("return {'x': np.lib.npyio}"),
    "alias_reader": wrap("f = pd.read_csv\nreturn {}"),
    "lambda_default_open": wrap("f = (lambda o=open: o)\nreturn {}"),
    "module_dict": wrap("return {'x': math.__dict__}"),
    "pd_eval": wrap("return {'x': pd.eval('1+1')}"),
    "apply_eval_string": wrap("return {'x': df_results.apply('eval')}"),
    # dispatch methods stored or passed on instead of being called directly
    "alias_dispatch": wrap("f = df_results.agg\nreturn {'x': f('to_pickle', 'x')}"),
    "subscript_dispatch": wrap("return {'x': [df_results.apply][0]('eval', 0, '1+1')}"),
    "lambda_dispatch": wrap("return {'x': (lambda f: f('to_csv'))(df_results.agg)}"),
    # dangerous builtins / helpers reached as attributes of a module
    "attr_getattr": wrap("return {'x': math.getattr(np, 'lo' + 'ad')}"),
    "px_data_import_module": wrap("m = px.data.import_module('subprocess')\nreturn {}"),
    "attr_environ": wrap("return {'x': dict(np.environ)}"),
    "attr_popen": wrap("return {'x': np.Popen(['id'])}"),
    "np_test_runner": wrap("np.test()\nreturn {}"),
    "pd_option_context": wrap("pd.option_context('plotting.backend', 'x')\nreturn {}"),
}


@pytest.mark.parametrize("name", sorted(STATIC_ATTACKS))
def test_static_attacks_are_blocked(name):
    assert validate_code(STATIC_ATTACKS[name]), f"{name} was not blocked"


@pytest.mark.parametrize("name", ["import", "dunder_import", "while_true", "read_pickle", "open"])
def test_blocked_code_is_never_executed(name):
    res = run_sandboxed(STATIC_ATTACKS[name], DF)
    assert not res.ok and res.violations


def test_legitimate_code_passes_and_runs():
    code = wrap(
        "g = df_results.groupby('model')['parsed_value_num'].agg(['mean', 'std']).reset_index()\n"
        "fig = px.box(df_results, x='model', y='parsed_value_num')\n"
        "u = stats.mannwhitneyu([1, 2], [3, 5])\n"
        "return {'summary_metrics': {'mean': float(np.mean(df_results['parsed_value_num'])),\n"
        "        'p': float(u.pvalue), 'sqrt2': math.sqrt(2) ** 2},\n"
        "        'per_model_table': g, 'figures': [fig], 'warnings': ['demo']}"
    )
    assert validate_code(code) == []
    res = run_sandboxed(code, DF)
    assert res.ok, res.error
    assert res.summary_metrics["mean"] == pytest.approx(2.75)
    assert "per_model_table" in res.tables and len(res.figures) == 1
    assert res.warnings == ["demo"]


def test_infinite_for_loop_times_out():
    code = wrap("s = 0\nfor i in range(10**12):\n    s += i\nreturn {'s': s}")
    res = run_sandboxed(code, DF, timeout_s=1.0)
    assert not res.ok and "Timeout" in (res.error or "")


def test_memory_bomb_is_stopped():
    code = wrap("x = np.ones(400_000_000)\nreturn {'n': float(x.sum())}")
    res = run_sandboxed(code, DF, mem_bytes=256 * 1024 * 1024)
    assert not res.ok and "Memory" in (res.error or "") or "killed" in (res.error or "")


def test_secrets_are_not_visible_in_child(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-should-not-leak-1234567890")
    # Even with a (hypothetical) validator bypass, the child environment is scrubbed.
    # We verify the parent-side environment construction directly:
    from src.security import scrubbed_environ

    assert "OPENAI_API_KEY" not in scrubbed_environ()
    assert os.environ["OPENAI_API_KEY"].startswith("sk-test")


def test_sandbox_environ_is_an_allowlist(monkeypatch):
    from src.security import sandbox_environ

    monkeypatch.setenv("GH_PAT", "x")  # no KEY/TOKEN/SECRET in the name
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pw@host/db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-should-not-leak-1234567890")
    monkeypatch.setenv("PATH", os.environ.get("PATH", "/usr/bin"))
    env = sandbox_environ()
    assert not {"GH_PAT", "DATABASE_URL", "OPENAI_API_KEY"} & set(env)
    assert "PATH" in env


# What user code could try after a (hypothetical) bypass of the AST validator. The audit
# hook of the child process must refuse each of them.
HOOK_ATTACKS = {
    "read_outside": "open(sys.argv[1]).read()",
    "write_file": "open(os.path.join(os.path.dirname(os.__file__), 'x.tmp'), 'w')",
    "os_open_write": "os.open(sys.argv[1], os.O_WRONLY)",
    "listdir_outside": "os.listdir(os.path.dirname(sys.argv[1]))",
    "socket": "import socket\nsocket.socket()",
    "subprocess": "import subprocess\nsubprocess.run([sys.executable, '-c', 'pass'])",
    "os_system": "os.system('echo hi')",
    "ctypes": "import ctypes",
    "remove": "os.remove(sys.argv[1])",
}


@pytest.mark.parametrize("name", sorted(HOOK_ATTACKS))
def test_audit_hook_blocks(name, tmp_path):
    import subprocess
    import sys

    secret = tmp_path / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")
    child = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    script = (
        "import os, sys\n"
        f"sys.path.insert(0, {child!r})\n"
        "import sandbox_child as sc\n"
        "sc._install_audit_hook(sc._read_roots([]))\n"
        "try:\n"
        + "".join(f"    {line}\n" for line in HOOK_ATTACKS[name].splitlines())
        + "except (PermissionError, ImportError) as exc:\n"
        "    print('BLOCKED', exc)\n"
        "else:\n"
        "    print('ALLOWED')\n"
    )
    proc = subprocess.run([sys.executable, "-c", script, str(secret)], capture_output=True,
                          text=True, timeout=60, check=False)
    assert "BLOCKED" in proc.stdout and "sandbox" in proc.stdout, proc.stdout + proc.stderr
    assert secret.read_text(encoding="utf-8") == "top secret"


def test_audit_hook_allows_library_reads():
    """Lazy imports of the scientific stack keep working under the hook."""
    code = wrap(
        "t = df_results.pivot_table(index='model', values='parsed_value_num', aggfunc='mean')\n"
        "r = stats.spearmanr([1, 2, 3], [1, 3, 2])\n"
        "fig = go.Figure(go.Bar(x=[1], y=[2]))\n"
        "return {'summary_metrics': {'r': float(r.statistic)}, 'tables': {'t': t},\n"
        "        'figures': [fig]}"
    )
    res = run_sandboxed(code, DF)
    assert res.ok, res.error


def test_runtime_error_reports_line():
    res = run_sandboxed(wrap("x = 1 / 0\nreturn {}"), DF)
    assert not res.ok and "ZeroDivisionError" in res.error and "line 2" in res.error


def test_non_dict_return_rejected():
    res = run_sandboxed(wrap("return 42"), DF)
    assert not res.ok and "dict" in res.error
