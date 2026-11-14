"""The sandbox child must find numpy & co. even when they live in the user site-packages."""

import os
import site

from src.evaluator import _library_paths, run_sandboxed

CODE = "def evaluate(df_results, ground_truth=None):\n    return {'summary_metrics': {'m': float(np.mean([1, 2, 3]))}}\n"


def test_library_paths_point_to_site_packages_only():
    paths = _library_paths()
    assert paths, "numpy/pandas/scipy/plotly must be importable in the test environment"
    for p in paths:
        assert os.path.isdir(os.path.join(p, "numpy")) or any(
            os.path.isdir(os.path.join(p, lib)) for lib in ("pandas", "scipy", "plotly"))
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert all(not os.path.abspath(p).startswith(os.path.join(repo, "src")) for p in paths)


def test_sandbox_runs_numpy_code():
    res = run_sandboxed(CODE, __import__("pandas").DataFrame({"x": [1]}))
    assert res.ok, res.error
    assert res.summary_metrics["m"] == 2.0


def test_user_site_is_covered_when_libraries_live_there():
    user_site = site.getusersitepackages()
    numpy_dir = os.path.dirname(os.path.dirname(__import__("numpy").__file__))
    if os.path.abspath(numpy_dir) == os.path.abspath(user_site):
        assert user_site in _library_paths()
