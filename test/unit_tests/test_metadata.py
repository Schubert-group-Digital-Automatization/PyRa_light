"""
Tests for metadata normalization and unification.
"""
# pylint: disable=redefined-outer-name
# pylint: disable=import-error
from __future__ import annotations

from typing import Any

import pytest

from PyRa_light.core.metadata import MetadataField, unify_metadata


def test_unify_metadata_empty_input_populates_all_fields_with_none() -> None:
    """An empty input dict should still produce all known unified keys."""

    unified = unify_metadata({})

    assert len(unified) == len(MetadataField)
    for field in MetadataField:
        key = field.value
        assert key in unified
        assert unified[key] is None


def test_unify_metadata_smoke_converts_numeric_and_bool_fields() -> None:
    """A small smoke test for the most common numeric/bool normalizations."""

    raw: dict[str, Any] = {
        "excitation wavelength [nm]": "532",
        "integration time [s]": "1.5",
        "accumulations": "3",
        "auto_background": "On",
    }

    unified = unify_metadata(raw)

    assert unified["excitation_wavelength_nm"] == pytest.approx(532.0)
    assert unified["integration_time_s"] == pytest.approx(1.5)
    assert unified["accumulations"] == 3.0
    assert unified["background_corrected"] is True


def test_unify_metadata_wavenumber_to_nm_conversion() -> None:
    """A laser wavenumber should be converted to an excitation wavelength in nm."""

    wn = 18796.0  # ~532 nm
    raw: dict[str, Any] = {
        "raman laser wavenumber": wn,
    }

    unified = unify_metadata(raw)
    value = unified["excitation_wavelength_nm"]

    assert value is not None
    assert value == pytest.approx(1e7 / wn, rel=1e-6)


def test_unify_metadata_sample_name_and_datetime_combination() -> None:
    """Sample name and separate acquisition date/time fields should be combined cleanly."""

    raw: dict[str, Any] = {
        "sample name": "Test sample",
        "acquisition date": "2025-06-13",
        "acquisition time": "14:30:00",
    }

    unified = unify_metadata(raw)

    assert unified["sample_name"] == "Test sample"

    dt = unified["measurement_datetime"]
    assert isinstance(dt, str)
    assert "2025-06-13" in dt
    assert "14:30" in dt


@pytest.mark.parametrize(
    "key",
    [
        "excitation_wavelength_nm",
        "excitation_nm",
        "laser_wavelength_nm",
        "excitation wavelength [nm]",
        "raman laser wavelength",
        "source setting",
        "channel_2_laser_wavelength",
        "laser_wavelength",
        "laser wavelength",
    ],
)
def test_excitation_wavelength_aliases(key: str) -> None:
    """All supported excitation-wavelength aliases should map to the same unified key."""
    raw: dict[str, Any] = {key: "532"}
    unified = unify_metadata(raw)
    assert unified["excitation_wavelength_nm"] == pytest.approx(532.0)


@pytest.mark.parametrize(
    "key, value, unified_key, expected",
    [
        # INTEGRATION_TIME_S
        ("integration_time_s", "1.5", "integration_time_s", 1.5),
        ("integration_time", "1.5", "integration_time_s", 1.5),
        ("integration time [s]", "1.5", "integration_time_s", 1.5),
        # ACCUMULATIONS
        ("accumulations", "3", "accumulations", 3.0),
        ("num_accumulations", "3", "accumulations", 3.0),
        ("accumulation_count", "3", "accumulations", 3.0),
        ("number of accumulations", "3", "accumulations", 3.0),
        ("number of scans", "3", "accumulations", 3.0),
        # BACKGROUND_CORRECTED
        ("auto_background", "On", "background_corrected", True),
        ("auto_bg", "yes", "background_corrected", True),
        ("automatic background", "true", "background_corrected", True),
        ("automatic fluorescence rejection", "1", "background_corrected", True),
        ("background_subtracted", "0", "background_corrected", False),
        ("background subtracted", "false", "background_corrected", False),
        ("dark_subtracted", "no", "background_corrected", False),
        ("background corrected", "0", "background_corrected", False),
    ],
)
def test_numeric_and_bool_aliases(
    key: str, value: Any, unified_key: str, expected: Any
) -> None:
    """Numeric and boolean aliases should normalize into the expected unified fields."""

    raw: dict[str, Any] = {key: value}
    unified = unify_metadata(raw)

    if isinstance(expected, (int, float)):
        assert unified[unified_key] == pytest.approx(expected)
    else:
        assert unified[unified_key] is expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("true", True),
        ("True", True),
        ("YES", True),
        ("0", False),
        ("false", False),
        ("No", False),
        (1, True),
        (0, False),
    ],
)
def test_unify_metadata_bool_parsing_for_background_corrected(
    value: Any, expected: bool
) -> None:
    """Boolean-like inputs should be parsed consistently for background correction."""

    raw: dict[str, Any] = {
        "auto_background": value,
    }

    unified = unify_metadata(raw)
    assert unified["background_corrected"] is expected


def test_unify_metadata_unknown_keys_are_ignored() -> None:
    """Unknown vendor keys should be ignored and must not create new unified keys."""

    raw: dict[str, Any] = {
        "CompletelyUnknownVendorKey": "servus!",
    }

    unified = unify_metadata(raw)

    valid_keys = {field.value for field in MetadataField}
    assert set(unified.keys()) == valid_keys
    assert unified["sample_name"] is None
    assert unified["excitation_wavelength_nm"] is None
    assert "servus!" not in unified.values()
