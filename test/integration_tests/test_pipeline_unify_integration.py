"""
Integration tests for the unification pipeline (Converted* -> Ra).
"""

# pylint: disable=redefined-outer-name
# pylint: disable=import-error

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from PyRa_light.core import Ra
from PyRa_light.io.loader import load_file


pytestmark = pytest.mark.integration

DATA_ROOT = Path(__file__).resolve().parent / "data"


def _expected_format(path: Path) -> str | None:
    """Map fixture files to the expected fileformat token (lowercase)."""
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


def _select_representative_cases(cases: list[Any]) -> list[Any]:
    """
    Select one representative file per format to keep the integration suite fast.

    This still provides coverage of each converter+unify path, but avoids running
    unification on every file in the data directory.
    """
    chosen: dict[str, Any] = {}
    for case in cases:
        file_path, expected_format = case.values  # type: ignore[attr-defined]
        if expected_format not in chosen:
            chosen[expected_format] = case
    return [chosen[k] for k in sorted(chosen)]


CASES = _discover_cases(DATA_ROOT)
REPRESENTATIVE_CASES = _select_representative_cases(CASES)


def test_unify_integration_fixtures_present() -> None:
    """
    Assert fixture directory exists and contains at least one supported file.

    Prevents a silent “0 collected” situation when data is missing.
    """
    if not DATA_ROOT.exists():
        pytest.skip(f"Integration data directory not found: {DATA_ROOT}")
    assert CASES, f"No supported integration fixture files found under: {DATA_ROOT}"


@pytest.mark.parametrize(("file_path", "expected_format"), REPRESENTATIVE_CASES)
def test_load_file_and_build_ra_from_first_measurement(
    file_path: Path,
    expected_format: str,
) -> None:
    """
    Real file -> ConvertedFile -> first ConvertedMeasurement -> Ra.

    Validates the converter+loader output can be unified into a sane Ra object.
    """
    try:
        converted = load_file(file_path)
    except ImportError as exc:
        pytest.skip(f"Optional dependency missing for {expected_format} ({file_path.name}): {exc}")

    measurements = getattr(converted, "measurements", None)
    assert isinstance(measurements, list)
    assert measurements, "ConvertedFile should contain at least one measurement"

    meas0 = measurements[0]

    if not hasattr(Ra, "from_converted_measurement"):
        pytest.skip("Ra.from_converted_measurement is not available in this version")

    ra = Ra.from_converted_measurement(  # type: ignore[attr-defined]
        meas0,
        filename=getattr(converted, "filename", None),
    )

    # Structural sanity
    x = np.asarray(getattr(ra, "x"))
    y = np.asarray(getattr(ra, "y"))

    assert x.ndim == 1
    assert y.ndim == 1
    assert x.size == y.size
    assert x.size > 10
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))

    filename = str(getattr(ra, "filename", ""))
    assert filename
    assert Path(filename).name == filename

    data = ra.to_dict()
    assert isinstance(data, dict)
    assert data


@pytest.mark.parametrize(("file_path", "expected_format"), REPRESENTATIVE_CASES)
def test_load_file_and_build_ras_from_converted_file(
    file_path: Path,
    expected_format: str,
) -> None:
    """
    Real file -> ConvertedFile -> list[Ra] via Ra.from_converted_file (if available).

    This validates the bulk-unification convenience API and its basic invariants.
    """
    try:
        converted = load_file(file_path)
    except ImportError as exc:
        pytest.skip(f"Optional dependency missing for {expected_format} ({file_path.name}): {exc}")

    if not hasattr(Ra, "from_converted_file"):
        pytest.skip("Ra.from_converted_file is not available in this version")

    ras = Ra.from_converted_file(converted)  # type: ignore[attr-defined]
    assert isinstance(ras, list)
    assert ras, "Expected at least one Ra from ConvertedFile"

    measurements = getattr(converted, "measurements", None)
    if isinstance(measurements, list):
        assert len(ras) == len(measurements)

    ra0 = ras[0]
    x = np.asarray(getattr(ra0, "x"))
    y = np.asarray(getattr(ra0, "y"))
    assert x.ndim == 1
    assert y.ndim == 1
    assert x.size == y.size
    assert x.size > 10
