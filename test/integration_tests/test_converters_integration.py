"""End-to-end integration tests for the IO loader."""

# pylint: disable=redefined-outer-name
# pylint: disable=import-error

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from PyRa_light.io.loader import get_converter, load_file
from PyRa_light.io.converters_base import BaseConverter
from PyRa_light.core.converted import ConvertedFile, ConvertedMeasurement


pytestmark = pytest.mark.integration


def test_integration_fixtures_present(
    integration_data_root: Path,
    integration_cases: list[tuple[Path, str]],
) -> None:
    """Assert the integration data folder exists and contains at least one supported file."""
    if not integration_data_root.exists():
        pytest.skip(f"Integration data directory not found: {integration_data_root}")
    assert integration_cases, (
        f"No supported integration fixture files found under: {integration_data_root}"
    )


def test_get_converter_selects_a_converter(
    representative_cases: list[tuple[Path, str]],
) -> None:
    """Ensure the registry selects a concrete converter for supported fixtures."""
    tested = 0

    for file_path, _expected in representative_cases:
        try:
            converter = get_converter(file_path)
        except ImportError:
            # Optional dependencies vary across environments; try the remaining fixtures.
            continue

        assert isinstance(converter, BaseConverter)
        tested += 1

    if tested == 0:
        pytest.skip(
            "No representative converters could be instantiated (missing optional dependencies?)"
        )


def test_load_file_converts_and_returns_sane_structure(
    representative_cases: list[tuple[Path, str]],
) -> None:
    """Convert representative fixtures end-to-end and validate basic output structure."""
    tested = 0

    for file_path, expected in representative_cases:
        try:
            converted = load_file(file_path)
        except ImportError:
            # Optional dependencies vary across environments; try the remaining fixtures.
            continue

        assert isinstance(converted, ConvertedFile)

        fileformat = str(converted.fileformat).lower()
        assert fileformat
        assert expected in fileformat

        assert converted.filename

        assert Path(converted.path).resolve() == file_path.resolve()

        assert converted.measurements

        for meas in converted.measurements[:2]:
            assert isinstance(meas, ConvertedMeasurement)
            x = np.asarray(meas.x)
            y = np.asarray(meas.y)

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
        assert data
        assert any(k.lower() == "fileformat" for k in data)

        tested += 1

    if tested == 0:
        pytest.skip(
            "No representative fixtures could be converted (missing optional dependencies?)"
        )
