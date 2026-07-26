"""Smoke tests for package scaffolding."""

import re
import tomllib
from pathlib import Path

import med_langchain_memory

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_version_is_semver() -> None:
    """Package exposes a semantic version string."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", med_langchain_memory.__version__)


def test_version_matches_pyproject() -> None:
    """__version__ stays in sync with pyproject.toml (boundary: drift check)."""
    pyproject = PROJECT_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert data["project"]["version"] == med_langchain_memory.__version__


def test_py_typed_marker_exists() -> None:
    """PEP 561 marker ships with the package."""
    pkg_dir = Path(med_langchain_memory.__file__).resolve().parent
    assert (pkg_dir / "py.typed").is_file()
