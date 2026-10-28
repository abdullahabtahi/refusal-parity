"""The package layout: every module imports without side effects, and shared paths resolve to the repository."""

import importlib
import pkgutil

import pytest

import refusal_parity
from refusal_parity.common import DATA, ROOT, RUNS

MODULES = sorted(m.name for m in pkgutil.walk_packages(refusal_parity.__path__, "refusal_parity."))


def test_every_stage_is_a_subpackage():
    stages = {m.split(".")[1] for m in MODULES if m.count(".") >= 2}
    assert {"data", "activations", "steering", "generation", "scoring", "analysis", "circuits", "reporting"} <= stages


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_without_running(module, monkeypatch):
    """Importing must not parse arguments, load models or write files: scripts keep their work in main()."""
    monkeypatch.setattr("sys.argv", ["pytest"])  # a module that parses arguments at import would exit here
    importlib.import_module(module)


def test_paths_resolve_to_the_repository():
    assert (ROOT / "pyproject.toml").exists()
    assert DATA == ROOT / "data" and RUNS == ROOT / "runs"
