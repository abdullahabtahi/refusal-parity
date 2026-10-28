"""The package layout: every module imports without side effects, and shared paths resolve to the repository."""

import importlib
import pkgutil
import subprocess

import pytest

import refusal_parity
from refusal_parity.common import DATA, ROOT, RUNS

MODULES = sorted(m.name for m in pkgutil.walk_packages(refusal_parity.__path__, "refusal_parity."))
OPTIONAL = {"refusal_parity.scoring.gemini": "google.genai"}  # needs the `remote` extra (pyproject.toml)


def test_every_stage_is_a_subpackage():
    stages = {m.split(".")[1] for m in MODULES if m.count(".") >= 2}
    assert {"data", "activations", "steering", "generation", "scoring", "analysis", "circuits", "reporting"} <= stages


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_without_running(module, monkeypatch):
    """Importing must not parse arguments, load models or write files: scripts keep their work in main()."""
    if module in OPTIONAL:
        pytest.importorskip(OPTIONAL[module])
    monkeypatch.setattr("sys.argv", ["pytest"])  # a module that parses arguments at import would exit here
    importlib.import_module(module)


def test_paths_resolve_to_the_repository():
    assert (ROOT / "pyproject.toml").exists()
    assert DATA == ROOT / "data" and RUNS == ROOT / "runs"


def test_no_source_file_is_git_ignored():
    """An unanchored ignore pattern (`data/`) once hid the whole refusal_parity/data subpackage from the repository."""
    sources = [str(p.relative_to(ROOT)) for d in ("src", "tests", "scripts") for p in (ROOT / d).rglob("*.py")]
    sources = [p for p in sources if "__pycache__" not in p]
    run = subprocess.run(["git", "check-ignore", "--no-index", *sources], cwd=ROOT, capture_output=True, text=True)
    if run.returncode == 128:
        pytest.skip("not a git checkout")
    assert run.stdout.split() == [], f"git-ignored source files: {run.stdout.split()}"
