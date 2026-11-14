"""Child-process entry point for the analysis sandbox.

Started by ``src.evaluator.run_sandboxed`` as ``python -I sandbox_child.py`` with an
allowlisted environment. Reads a JSON job from stdin, runs the (already AST-validated)
user code under resource limits and an audit hook, and writes a JSON result to stdout.

This file is intentionally self-contained (it does not import ``src``).
"""

from __future__ import annotations

import io
import json
import math
import signal
import sys
import time
import traceback

RESULT_MARKER = "@@SANDBOX_RESULT@@"


def _set_limits(mem_bytes: int, cpu_seconds: int) -> None:
    try:
        import resource
    except ImportError:  # non-POSIX platform: rely on the parent's wall-clock timeout
        return
    used_cpu = int(time.process_time()) + 1
    resource.setrlimit(resource.RLIMIT_CPU, (used_cpu + cpu_seconds, used_cpu + cpu_seconds + 1))
    try:
        with open("/proc/self/statm") as fh:
            vm_bytes = int(fh.read().split()[0]) * resource.getpagesize()
    except OSError:
        vm_bytes = 0
    limit = vm_bytes + mem_bytes
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))  # no file writes
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))  # no new processes
    except (ValueError, OSError):
        pass


# Audit events that are refused outright once the user code runs (prefix match).
_DENIED_EVENTS = (
    "socket.", "subprocess.", "ctypes.", "os.system", "os.exec", "os.spawn", "os.posix_spawn",
    "os.fork", "os.kill", "os.startfile", "os.putenv", "os.unsetenv", "os.remove", "os.rename",
    "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.link", "os.symlink", "os.truncate",
    "os.chdir", "os.add_dll_directory", "shutil.", "tempfile.", "webbrowser.", "urllib.",
    "http.client.", "ftplib.", "smtplib.", "mmap.", "pty.", "fcntl.", "winreg.", "msvcrt.",
    "sqlite3.", "pickle.find_class", "signal.pthread_kill", "sys.settrace", "sys.setprofile",
)
# Modules that may not be loaded for the first time while the user code runs.
_DENIED_IMPORTS = frozenset({
    "ctypes", "_ctypes", "cffi", "subprocess", "_posixsubprocess", "_winapi",
    "multiprocessing", "socket", "_socket", "ssl", "_ssl", "asyncio", "pty", "webbrowser",
})
_PATH_EVENTS = frozenset({"os.listdir", "os.scandir", "os.walk", "os.fwalk", "glob.glob"})


def _read_roots(lib_paths: list) -> tuple[str, ...]:
    """Directories the user code may read: the Python installation and its libraries."""
    import os
    import sysconfig

    candidates = [sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix,
                  os.path.dirname(os.__file__), sysconfig.get_paths().get("purelib"),
                  sysconfig.get_paths().get("platlib"), "/usr/share/zoneinfo",
                  *[p for p in lib_paths if isinstance(p, str)]]
    roots = {os.path.normcase(os.path.realpath(c)) for c in candidates if c}
    return tuple(sorted(roots))


def _install_audit_hook(read_roots: tuple[str, ...]) -> None:
    """Second line of defence behind the AST validator (it cannot be removed again).

    Even if user code gets hold of a dangerous function, the interpreter refuses network
    access, new processes, file writes and file reads outside the Python installation
    (so not ``/proc/<pid>/environ``, the repository, or ``models/models.toml``).
    """
    import os

    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
    loaded = sys.modules

    def inside(path) -> bool:  # noqa: ANN001
        if not isinstance(path, (str, bytes, os.PathLike)):
            return False  # file descriptors and other objects are not needed
        try:
            p = os.path.normcase(os.path.realpath(os.fsdecode(path)))
        except (OSError, ValueError):
            return False
        return any(p == r or p.startswith(r.rstrip("\\/") + os.sep) for r in read_roots)

    def hook(event: str, args: tuple) -> None:
        if event == "open":
            path, mode, flags = (*args, None, None)[:3]
            writing = bool(isinstance(mode, str) and set(mode) & set("wax+")) or \
                bool(isinstance(flags, int) and flags & write_flags)
            if writing or not inside(path):
                raise PermissionError("sandbox: file access is not allowed")
        elif event in _PATH_EVENTS:
            if not inside(args[0] if args and args[0] is not None else "."):
                raise PermissionError("sandbox: directory access is not allowed")
        elif event == "import":
            name = str(args[0]) if args else ""
            if name.split(".", 1)[0] in _DENIED_IMPORTS and name not in loaded:
                raise ImportError(f"sandbox: import of '{name}' is not allowed")
        elif event.startswith(_DENIED_EVENTS):
            raise PermissionError(f"sandbox: '{event}' is not allowed")

    sys.addaudithook(hook)


def _to_jsonable(obj, pd, np, go):  # noqa: ANN001 - internal helper
    if isinstance(obj, pd.DataFrame):
        return {"__dataframe__": json.loads(obj.to_json(orient="split", default_handler=str))}
    if isinstance(obj, pd.Series):
        return _to_jsonable(obj.to_frame(), pd, np, go)
    if isinstance(obj, go.Figure):
        return {"__figure__": obj.to_json()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v, pd, np, go) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v, pd, np, go) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist(), pd, np, go)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if obj is None or isinstance(obj, (int, str)):
        return obj
    return str(obj)


def main() -> None:
    real_stdout = sys.stdout
    job = json.loads(sys.stdin.read())

    # Library locations from the parent (needed when numpy & co. live in the user
    # site-packages, which isolated mode skips). Appended, so the interpreter's own
    # site-packages keep priority. Only directories holding the scientific stack are sent.
    for path in job.get("lib_paths", []):
        if isinstance(path, str) and path not in sys.path:
            sys.path.append(path)

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from scipy import stats

    df = pd.read_json(io.StringIO(json.dumps(job["df"])), orient="split", dtype=False,
                      convert_dates=False)
    logs: list[str] = []

    def safe_print(*args, **_kwargs):  # noqa: ANN002, ANN003
        if sum(len(x) for x in logs) < 100_000:
            logs.append(" ".join(str(a) for a in args))

    import builtins as _b

    allowed = {name: getattr(_b, name) for name in job["builtins"] if hasattr(_b, name)}
    allowed["print"] = safe_print

    # C extensions (numpy/pandas) sometimes import lazily; CPython then resolves
    # ``__import__`` from the *calling* frame's builtins, i.e. the user's restricted
    # globals. User code itself can never name ``__import__`` (the AST validator
    # rejects every identifier starting with "_"), so this guarded hook is only reached
    # by library internals. It serves already-loaded modules and submodules of the
    # pre-injected scientific packages only.
    real_import = _b.__import__
    safe_roots = {"numpy", "pandas", "scipy", "plotly", "math"}

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002, ANN001
        root = name.split(".", 1)[0]
        if level == 0 and (name in sys.modules or root in safe_roots):
            return real_import(name, globals, locals, fromlist, level)
        raise ImportError(f"import of '{name}' is not allowed in the sandbox")

    allowed["__import__"] = guarded_import
    env = {"__builtins__": allowed, "np": np, "pd": pd, "stats": stats, "px": px, "go": go,
           "math": math}

    _set_limits(int(job["mem_bytes"]), int(job["cpu_seconds"]))
    timeout_s = float(job["timeout_s"])

    def _on_alarm(_signum, _frame):  # noqa: ANN001
        raise TimeoutError(f"analysis exceeded the {timeout_s:g} s time limit")

    # Windows has no SIGALRM; there the parent's hard wall-clock timeout is the only timer.
    has_alarm = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")

    def _timer(seconds: float) -> None:
        if has_alarm:
            signal.setitimer(signal.ITIMER_REAL, seconds)

    if has_alarm:
        signal.signal(signal.SIGALRM, _on_alarm)
    _install_audit_hook(_read_roots(job.get("lib_paths", [])))
    _timer(timeout_s)

    out: dict = {"ok": False}
    t0 = time.perf_counter()
    try:
        exec(compile(job["code"], "<analysis>", "exec"), env)  # noqa: S102 - validated code
        result = env["evaluate"](df, job.get("ground_truth"))
        _timer(0)
        if not isinstance(result, dict):
            raise TypeError("evaluate() must return a dict")
        out = {"ok": True, "result": _to_jsonable(result, pd, np, go)}
    except MemoryError:
        out = {"ok": False, "error": "MemoryError: the analysis exceeded the memory limit"}
    except BaseException as exc:  # noqa: BLE001 - report every failure to the parent
        tb = traceback.extract_tb(exc.__traceback__)
        line = next((f.lineno for f in reversed(tb) if f.filename == "<analysis>"), None)
        where = f" (line {line})" if line else ""
        out = {"ok": False, "error": f"{type(exc).__name__}{where}: {exc}"}
    finally:
        _timer(0)
    out["logs"] = "\n".join(logs)
    out["duration_ms"] = (time.perf_counter() - t0) * 1000
    real_stdout.write(RESULT_MARKER + json.dumps(out, allow_nan=False, default=str))
    real_stdout.flush()


if __name__ == "__main__":
    main()
