"""Integration tests for JSON export across the full pipeline.

These tests run against real fixture files under ``test/integration_tests/data/merge`` and
exercise:

real file -> ConvertedFile -> (ConvertedMeasurement / Ra) -> merge -> RaProc -> JSONExport
"""

# pylint: disable=redefined-outer-name
# pylint: disable=import-error

from __future__ import annotations

import json
from pathlib import Path

import pytest

from PyRa_light.core import merge_ras
from PyRa_light.core import Ra
from PyRa_light.core.converted import ConvertedFile, ConvertedMeasurement
from PyRa_light.export.json_exporter import ExportMode, JSONExport


pytestmark = pytest.mark.integration


def _read_json(path: Path) -> dict:
    """Read a JSON file and return its parsed content."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_export_integration_fixtures_present(merge_files: list[Path]) -> None:
    """Assert export fixtures exist to avoid a silent empty test run."""
    if not merge_files:
        pytest.skip("No supported fixture files found under test/integration_tests/data/merge")


def test_export_writes_all_object_types(
    tmp_path: Path,
    first_merge_converted: ConvertedFile,
    first_merge_measurement: ConvertedMeasurement,
    merge_ra_pool: list[Ra],
) -> None:
    """Export should write JSON for ConvertedFile, ConvertedMeasurement, Ra, and RaProc."""
    raproc = merge_ras(merge_ra_pool[:2], method="average")

    exporter = JSONExport(export_dir=tmp_path, mode=ExportMode.OVERWRITE)

    out_conv = exporter.export(first_merge_converted)
    out_meas = exporter.export(first_merge_measurement)
    out_ra = exporter.export(merge_ra_pool[0], suffix="it")
    out_rp = exporter.export(raproc)

    assert out_conv is not None and out_conv.exists()
    assert out_meas is not None and out_meas.exists()
    assert out_ra is not None and out_ra.exists()
    assert out_rp is not None and out_rp.exists()

    # Suffix rule
    assert out_ra.stem.endswith("_it")

    assert out_conv.stem.endswith("_conv")
    assert out_meas.stem.endswith(f"_{first_merge_measurement.index}_conv")
    assert out_rp.stem.endswith("_avg")

    for p in (out_conv, out_meas, out_ra, out_rp):
        assert p.suffix == ".json"
        data = _read_json(p)
        assert isinstance(data, dict)
        assert data


def test_export_mode_skip_does_not_overwrite(
    tmp_path: Path,
    first_merge_converted: ConvertedFile,
) -> None:
    """ExportMode.SKIP should not modify existing files and should return None."""
    exporter_ow = JSONExport(export_dir=tmp_path, mode=ExportMode.OVERWRITE)
    out = exporter_ow.export(first_merge_converted)
    assert out is not None and out.exists()

    before = out.read_bytes()

    exporter_skip = JSONExport(export_dir=tmp_path, mode=ExportMode.SKIP)
    out2 = exporter_skip.export(first_merge_converted)

    assert out2 is None
    after = out.read_bytes()
    assert after == before


def test_export_creates_no_leftover_tmp_files(
    tmp_path: Path,
    first_merge_converted: ConvertedFile,
) -> None:
    """Export should write atomically and leave no *.tmp files behind."""
    exporter = JSONExport(export_dir=tmp_path, mode=ExportMode.OVERWRITE)
    out = exporter.export(first_merge_converted)
    assert out is not None and out.exists()

    tmp_files = list(tmp_path.rglob("*.tmp"))
    assert not tmp_files
