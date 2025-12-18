"""End-to-end integration tests for all file converters.

These tests run against real fixture files under ``test/integration_tests/data``
"""

# pylint: disable=redefined-outer-name
# pylint: disable=import-error

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from PyRa_light.io.loader import get_converter, load_file


pytestmark = pytest.mark.integration

DATA_ROOT = Path(__file__).resolve().parent / "data"


def _expected_format(path: Path) -> str | None:
    """
    Map a fixture file to an expected ConvertedFile.fileformat (lowercase).

    Returns None for unknown/unsupported extensions (those are ignored by this suite).
    """
    suffix = path.suffix.lower()

    if suffix.startswith(".") and suffix[1:].isdigit():
        return "opus"

    mapping = {
        ".opus": "opus",
        ".spc": "spc",
        ".wdf": "wdf",
        ".wip": "witec",
        ".dx": "jcamp",
        ".jdx": "jcamp",
        ".jcamp": "jcamp",
    }
    return mapping.get(suffix)


def _discover_cases(data_root: Path) -> list[Any]:
    """Discover real fixture files and build parametrized test cases."""
    if not data_root.exists():
        return []

    cases: list[Any] = []
    for path in sorted(p for p in data_root.rglob("*") if p.is_file()):
        expected = _expected_format(path)
        if expected is None:
            continue
        cases.append(pytest.param(path, expected, id=f"{expected}:{path.name}"))
    return cases


CASES = _discover_cases(DATA_ROOT)


def test_integration_fixtures_present() -> None:
    """
    Assert the integration fixture directory exists and contains at least one supported file.

    This prevents a silent “0 collected” situation when the data folder is missing.
    """
    if not DATA_ROOT.exists():
        pytest.skip(f"Integration data directory not found: {DATA_ROOT}")
    assert CASES, f"No supported integration fixture files found under: {DATA_ROOT}"


@pytest.mark.parametrize(("file_path", "expected_format"), CASES)
def test_get_converter_selects_a_converter(file_path: Path, expected_format: str) -> None:
    """
    get_converter() should return a concrete converter instance for supported fixtures.

    This validates registry wiring independently from conversion.
    """
    try:
        converter = get_converter(file_path)
    except ImportError as exc:
        pytest.skip(
            f"Optional dependency missing for {expected_format} ({file_path.name}): {exc}"
        )

    # Keep this minimal: the converter must at least be callable for conversion.
    assert hasattr(converter, "convert")
    assert callable(converter.convert)


@pytest.mark.parametrize(("file_path", "expected_format"), CASES)
def test_load_file_converts_and_returns_sane_structure(
    file_path: Path,
    expected_format: str,
) -> None:
    """
    load_file() should convert a real fixture file end-to-end into a sensible ConvertedFile.

    The checks are intentionally format-agnostic: at least one measurement must be present,
    and x/y arrays must be 1D, same length, finite, and non-trivial in size.
    """
    try:
        converted = load_file(file_path)
    except ImportError as exc:
        # Optional dependencies for some formats may not be installed everywhere.
        pytest.skip(f"Optional dependency missing for {file_path.name}: {exc}")

    fileformat = str(getattr(converted, "fileformat", "")).lower()
    assert fileformat
    assert expected_format in fileformat

    filename = str(getattr(converted, "filename", ""))
    assert filename  # non-empty string

    converted_path = Path(str(getattr(converted, "path")))
    assert converted_path.resolve() == file_path.resolve()

    measurements = getattr(converted, "measurements", None)
    assert isinstance(measurements, list)
    assert len(measurements) >= 1

    for meas in measurements[:2]:
        x = np.asarray(getattr(meas, "x"))
        y = np.asarray(getattr(meas, "y"))

        assert x.ndim == 1
        assert y.ndim == 1
        assert x.size == y.size
        assert x.size > 10

        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(y))

        dx = np.diff(x)
        assert np.all(dx >= 0) or np.all(dx <= 0)

    data = converted.to_dict()
    assert isinstance(data, dict)
    assert data  # non-empty

    assert any(k.lower() == "fileformat" for k in data)
