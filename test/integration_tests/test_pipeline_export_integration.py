"""Integration tests for JSON export across the full pipeline.

These tests run against real fixture files under ``test/integration_tests/data/merge`` and
exercise the end-to-end pipeline:

real file -> ConvertedFile -> (ConvertedMeasurement / Ra) -> merge -> RaProc -> JSONExport

Assertions are intentionally minimal and focus on:
- exporter wiring and naming rules
- correct write/skip semantics
- JSON structure presence (not exact vendor metadata)
"""

# pylint: disable=redefined-outer-name
# pylint: disable=import-error

from __future__ import annotations

import json
from pathlib import Path

import pytest

from PyRa_light.core import Ra, merge_ras
from PyRa_light.export.json_exporter import ExportMode, JSONExport
from PyRa_light.io.loader import load_file


pytestmark = pytest.mark.integration

DATA_ROOT = Path(__file__).resolve().parent / "data" / "merge"


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


def _discover_merge_files(data_root: Path) -> list[Path]:
    """Return a stable, sorted list of merge fixture files."""
    if not data_root.exists():
        return []

    files = [
        p for p in data_root.rglob("*") if p.is_file() and _expected_format(p) is not None
    ]
    return sorted(files)


MERGE_FILES = _discover_merge_files(DATA_ROOT)


def test_export_integration_fixtures_present() -> None:
    """Assert export fixtures exist to avoid a silent empty test run."""
    if not DATA_ROOT.exists():
        pytest.skip(f"Export integration data directory not found: {DATA_ROOT}")
    assert MERGE_FILES, f"No supported export fixture files found under: {DATA_ROOT}"


def _load_one_converted_file() -> object:
    """Load a single real fixture file into a ConvertedFile-like object."""
    if not MERGE_FILES:
        pytest.skip("No merge fixture files available")

    file_path = MERGE_FILES[0]
    try:
        converted = load_file(file_path)
    except ImportError as exc:
        pytest.skip(f"Optional dependency missing for {file_path.name}: {exc}")

    converted_path = Path(str(getattr(converted, "path")))
    assert converted_path.resolve() == file_path.resolve()
    return converted


def _ras_from_converted(converted: object) -> list[Ra]:
    """Build one Ra per measurement from a ConvertedFile-like object."""
    if not hasattr(Ra, "from_converted_measurement"):
        pytest.skip("Ra.from_converted_measurement is not available in this version")

    measurements = getattr(converted, "measurements", None)
    assert isinstance(measurements, list)
    assert measurements

    ras: list[Ra] = []
    for meas in measurements:
        ra = Ra.from_converted_measurement(  # type: ignore[attr-defined]
            meas,
            filename=getattr(converted, "filename", None),
        )
        ras.append(ra)

    return ras


def _read_json(path: Path) -> dict:
    """Read a JSON file and return its parsed content."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_export_writes_all_object_types(tmp_path: Path) -> None:
    """Export should write JSON for ConvertedFile, ConvertedMeasurement, Ra, and RaProc."""
    converted = _load_one_converted_file()
    measurements = getattr(converted, "measurements", None)
    assert isinstance(measurements, list) and measurements

    meas0 = measurements[0]
    ras = _ras_from_converted(converted)
    assert ras

    ras_pool: list[Ra] = list(ras)
    for file_path in MERGE_FILES[1:]:
        if len(ras_pool) >= 2:
            break
        try:
            other_converted = load_file(file_path)
        except ImportError as exc:
            pytest.skip(f"Optional dependency missing for {file_path.name}: {exc}")
        ras_pool.extend(_ras_from_converted(other_converted))

    assert len(ras_pool) >= 2, "Merge fixtures should provide at least two spectra"
    raproc = merge_ras(ras_pool[:2], method="average")

    exporter = JSONExport(export_dir=tmp_path, mode=ExportMode.OVERWRITE)

    out_conv = exporter.export(converted)
    out_meas = exporter.export(meas0)
    out_ra = exporter.export(ras_pool[0], suffix="it")
    out_rp = exporter.export(raproc)

    assert out_conv is not None and out_conv.exists()
    assert out_meas is not None and out_meas.exists()
    assert out_ra is not None and out_ra.exists()
    assert out_rp is not None and out_rp.exists()

    assert out_ra.stem.endswith("_it")

    assert out_conv.stem.endswith("_conv")
    assert out_meas.stem.endswith(f"_{meas0.index}_conv")
    assert out_rp.stem.endswith("_avg")

    for p in (out_conv, out_meas, out_ra, out_rp):
        data = _read_json(p)
        assert isinstance(data, dict)
        assert data


def test_export_mode_skip_does_not_overwrite(tmp_path: Path) -> None:
    """ExportMode.SKIP should not modify existing files and should return None."""
    converted = _load_one_converted_file()

    exporter_ow = JSONExport(export_dir=tmp_path, mode=ExportMode.OVERWRITE)
    out = exporter_ow.export(converted)
    assert out is not None and out.exists()

    before = out.read_bytes()

    exporter_skip = JSONExport(export_dir=tmp_path, mode=ExportMode.SKIP)
    out2 = exporter_skip.export(converted)

    assert out2 is None
    after = out.read_bytes()
    assert after == before


def test_export_creates_no_leftover_tmp_files(tmp_path: Path) -> None:
    """Export should write atomically and leave no *.tmp files behind."""
    converted = _load_one_converted_file()

    exporter = JSONExport(export_dir=tmp_path, mode=ExportMode.OVERWRITE)
    out = exporter.export(converted)
    assert out is not None and out.exists()

    tmp_files = list(tmp_path.rglob("*.tmp"))
    assert not tmp_files
