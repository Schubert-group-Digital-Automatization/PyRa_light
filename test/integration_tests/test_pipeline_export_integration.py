"""Integration tests for export across the full pipeline.

These tests run against real fixture files under ``test/integration_tests/data/merge`` and
exercise:

file -> ConvertedFile -> (ConvertedMeasurement / Ra) -> merge -> RaProc -> JSONExport / CSVExport
"""

# pylint: disable=redefined-outer-name
# pylint: disable=import-error

from __future__ import annotations

import json
import csv
from pathlib import Path

import pytest

from PyRa_light.core import merge_ras
from PyRa_light.core import Ra
from PyRa_light.core.converted import ConvertedFile, ConvertedMeasurement
from PyRa_light.export.json_exporter import ExportMode, JSONExport
from PyRa_light.export.csv_exporter import CSVExport


pytestmark = pytest.mark.integration


def _read_json(path: Path) -> dict:
    """Read a JSON file and return its parsed content."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv_rows(path: Path) -> list[list[str]]:
    """Read a semicolon-delimited CSV file and return rows."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        return [list(row) for row in reader]


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
    assert out_ra.stem.startswith(f"{Path(merge_ra_pool[0].filename).stem}_{merge_ra_pool[0].index}_ra")

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


# -----------------------------------------------------------------------------
# CSVExport integration tests (Ra-only 3-CSV bundle)
# -----------------------------------------------------------------------------


def test_csv_export_writes_triplet_for_ra(
    tmp_path: Path,
    merge_ra_pool: list[Ra],
) -> None:
    """CSVExport should write *_x.csv, *_y.csv, and *_meta.csv for a single Ra."""
    ra = merge_ra_pool[0]

    exporter = CSVExport(export_dir=tmp_path, mode=ExportMode.OVERWRITE)
    out = exporter.export(ra, suffix="it")

    assert out is not None
    x_path, y_path, meta_path = out

    assert x_path.exists()
    assert y_path.exists()
    assert meta_path.exists()

    assert x_path.name.endswith("_it_x.csv")
    assert y_path.name.endswith("_it_y.csv")
    assert meta_path.name.endswith("_it_meta.csv")

    expected_prefix = f"{Path(ra.filename).stem}_{ra.index}_it"
    assert x_path.stem.startswith(expected_prefix)
    assert y_path.stem.startswith(expected_prefix)
    assert meta_path.stem.startswith(expected_prefix)

    x_rows = _read_csv_rows(x_path)
    y_rows = _read_csv_rows(y_path)
    meta_rows = _read_csv_rows(meta_path)

    # x/y files are wide: a single row where the first cell is the export id (<stem>_x / <stem>_y).
    assert len(x_rows) == 1
    assert len(y_rows) == 1
    assert x_rows[0][0] == x_path.stem
    assert y_rows[0][0] == y_path.stem

    assert len(x_rows[0]) == len(list(ra.x)) + 1
    assert len(y_rows[0]) == len(list(ra.y)) + 1

    # meta file: header row + single value row
    assert len(meta_rows) == 2
    headers, values = meta_rows
    assert len(headers) == len(values)
    assert "filename" in headers
    assert "sample_name" in headers


def test_csv_export_mode_skip_does_not_overwrite_bundle(
    tmp_path: Path,
    merge_ra_pool: list[Ra],
) -> None:
    """ExportMode.SKIP should keep the existing CSV bundle unchanged and return None."""
    ra = merge_ra_pool[0]

    exporter_ow = CSVExport(export_dir=tmp_path, mode=ExportMode.OVERWRITE)
    out = exporter_ow.export(ra)
    assert out is not None
    x_path, y_path, meta_path = out

    before_x = x_path.read_bytes()
    before_y = y_path.read_bytes()
    before_m = meta_path.read_bytes()

    exporter_skip = CSVExport(export_dir=tmp_path, mode=ExportMode.SKIP)
    out2 = exporter_skip.export(ra)

    assert out2 is None
    assert x_path.read_bytes() == before_x
    assert y_path.read_bytes() == before_y
    assert meta_path.read_bytes() == before_m


def test_csv_export_creates_no_leftover_tmp_files(
    tmp_path: Path,
    merge_ra_pool: list[Ra],
) -> None:
    """CSVExport should write atomically and leave no *.tmp files behind."""
    exporter = CSVExport(export_dir=tmp_path, mode=ExportMode.OVERWRITE)
    out = exporter.export(merge_ra_pool[0])
    assert out is not None

    tmp_files = list(tmp_path.rglob("*.tmp"))
    assert not tmp_files


def test_csv_export_rejects_non_ra_objects(
    tmp_path: Path,
    first_merge_converted: ConvertedFile,
) -> None:
    """CSVExport is intentionally Ra-only and should reject other object types."""
    exporter = CSVExport(export_dir=tmp_path, mode=ExportMode.OVERWRITE)

    with pytest.raises(TypeError):
        exporter.export(first_merge_converted)  # type: ignore[arg-type]
