"""Integration tests for the unification pipeline (Converted* -> Ra).

These tests run against representative real fixture files under
``test/integration_tests/data`` and validate that:

- the loader produces a sane ``ConvertedFile`` with at least one measurement
- unification APIs can build one or more ``Ra`` objects with valid 1D arrays
"""

# pylint: disable=redefined-outer-name
# pylint: disable=import-error

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable

import numpy as np
import pytest

from PyRa_light.core import Ra
from PyRa_light.core.converted import ConvertedFile


pytestmark = pytest.mark.integration


def test_unify_integration_fixtures_present(
    integration_data_root: Path,
    integration_cases: list[tuple[Path, str]],
) -> None:
    """Assert the integration data folder exists and contains at least one supported file."""
    if not integration_data_root.exists():
        pytest.skip(f"Integration data directory not found: {integration_data_root}")
    assert integration_cases, (
        f"No supported integration fixture files found under: {integration_data_root}"
    )


def test_load_file_and_build_ra_from_first_measurement(
    representative_cases: list[tuple[Path, str]],
    load_converted_file: Callable[[Path], ConvertedFile | None],
) -> None:
    """Real file -> ConvertedFile -> first ConvertedMeasurement -> Ra."""
    if not hasattr(Ra, "from_converted_measurement"):
        pytest.skip("Ra.from_converted_measurement is not available in this version")

    tested = 0

    for file_path, _expected_format in representative_cases:
        converted = load_converted_file(file_path)
        if converted is None:
            continue

        assert isinstance(converted, ConvertedFile)
        assert converted.measurements, "ConvertedFile should contain at least one measurement"

        ra = Ra.from_converted_measurement(  # type: ignore[attr-defined]
            converted.measurements[0],
            filename=converted.filename,
        )

        x = np.asarray(ra.x)
        y = np.asarray(ra.y)

        assert x.ndim == 1
        assert y.ndim == 1
        assert x.size == y.size
        assert x.size > 10
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(y))

        assert ra.filename
        assert Path(ra.filename).name == ra.filename

        data = ra.to_dict()
        assert isinstance(data, dict)
        assert data

        tested += 1

    if tested == 0:
        pytest.skip("No representative fixtures could be unified (missing optional dependencies?)")


def test_load_file_and_build_ras_from_converted_file(
    representative_cases: list[tuple[Path, str]],
    load_converted_file: Callable[[Path], ConvertedFile | None],
) -> None:
    """Real file -> ConvertedFile -> list[Ra] via Ra.from_converted_file (if available)."""
    if not hasattr(Ra, "from_converted_file"):
        pytest.skip("Ra.from_converted_file is not available in this version")

    tested = 0

    for file_path, _expected_format in representative_cases:
        converted = load_converted_file(file_path)
        if converted is None:
            continue

        assert isinstance(converted, ConvertedFile)

        ras = Ra.from_converted_file(converted)  # type: ignore[attr-defined]

        assert isinstance(ras, list)
        assert ras, "Expected at least one Ra from ConvertedFile"
        assert len(ras) == len(converted.measurements)

        ra0 = ras[0]
        x = np.asarray(ra0.x)
        y = np.asarray(ra0.y)

        assert x.ndim == 1
        assert y.ndim == 1
        assert x.size == y.size
        assert x.size > 10

        # Bulk path should still produce JSON-serializable dict output.
        data = ra0.to_dict()
        assert isinstance(data, dict)
        assert data

        tested += 1

    if tested == 0:
        pytest.skip("No fixtures could be bulk-unified (missing optional dependencies?)")
