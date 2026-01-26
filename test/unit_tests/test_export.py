"""Tests for JSON and CSV exporting."""

# pylint: disable=redefined-outer-name
# pylint: disable=import-error

from __future__ import annotations

import json
import csv
from pathlib import Path

import numpy as np
import pytest

from PyRa_light.export.csv_exporter import CSVExport
from PyRa_light.export.json_exporter import ExportMode, JSONExport
from PyRa_light.core import ConvertedFile, ConvertedMeasurement, Ra, RaProc


# ------------------------------------------------
# Test data fixtures
# ------------------------------------------------


@pytest.fixture()
def meas() -> ConvertedMeasurement:
    """Fixture: one in-memory `ConvertedMeasurement` with small x/y arrays."""

    x = np.asarray([100.0, 200.0, 300.0], dtype=np.float64)
    y = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)

    return ConvertedMeasurement(
        path=Path("dummy.spc"),
        index=1,
        fileformat="spc",
        x=x,
        y=y,
        x_label="wavenumber [1/cm]",
        y_label="intensity [a.u.]",
        raw_metadata={"instrument": "test"},
        other_spectra={},
    )


@pytest.fixture()
def converted_file(meas: ConvertedMeasurement) -> ConvertedFile:
    """Fixture: one in-memory `ConvertedFile` containing exactly one measurement."""

    return ConvertedFile(
        path=Path("dummy.spc"),
        fileformat="spc",
        filename="dummy.spc",
        measurements=[meas],
        raw={},
    )


@pytest.fixture()
def ra(meas: ConvertedMeasurement) -> Ra:
    """Fixture: one `Ra` created from a `ConvertedMeasurement` (index is propagated)."""

    return Ra.from_converted_measurement(meas, filename="dummy.spc")


@pytest.fixture()
def raproc() -> RaProc:
    """Fixture: one `RaProc` with deterministic `merged_sample_name` and processing info."""

    x = np.asarray([100.0, 200.0, 300.0], dtype=np.float64)
    y = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)

    return RaProc(
        merged_sample_name="testmerged",
        filenames=["a.spc", "b.spc"],
        sample_names=["A", "B"],
        x_processed=x,
        y_processed=y,
        processing={
            "method": "average",
            "n_spectra": 2,
            "merge_timestamp": "1970-01-01T00:00:00Z",
            "source_indices": [0, 1],
        },
        unified_meta={"sample_type": "sample"},
        metadata_user={"group_comment": "merged set"},
    )


def _read_csv_rows(path: Path) -> list[list[str]]:
    """Read a semicolon-delimited CSV file and return all rows."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        return [list(row) for row in reader]


# ------------------------------------------------
# export (ConvertedFile and ConvertedMeasurement)
# ------------------------------------------------


def test_export_convertedfile_creates_json(tmp_path: Path, converted_file: ConvertedFile) -> None:
    """Assert ConvertedFile export: creates `<filename.stem>_conv.json` with key fields."""

    exporter = JSONExport(export_dir=tmp_path)
    out = exporter.export(converted_file)

    assert out is not None
    assert out.exists()
    assert out.suffix == ".json"

    expected_name = f"{Path(converted_file.filename).stem}_conv.json"
    assert out.name == expected_name

    with out.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict)
    assert data.get("fileformat") == converted_file.fileformat
    assert data.get("filename") == converted_file.filename
    assert "measurements" in data
    assert isinstance(data["measurements"], list)


def test_export_convertedmeasurement_creates_json(
    tmp_path: Path,
    meas: ConvertedMeasurement,
) -> None:
    """Assert ConvertedMeasurement export: creates `<path.stem>_<index>_conv.json` with x/y."""

    exporter = JSONExport(export_dir=tmp_path)
    out = exporter.export(meas)

    assert out is not None
    assert out.exists()
    assert out.suffix == ".json"

    expected_name = f"{meas.path.stem}_{meas.index}_conv.json"
    assert out.name == expected_name

    with out.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert "x" in data
    assert "y" in data
    assert data.get("index") == meas.index
    assert data.get("fileformat") == meas.fileformat


# ------------------------------------------------
# export (Ra)
# ------------------------------------------------


def test_export_ra_includes_metadata(
    tmp_path: Path,
    ra: Ra,
) -> None:
    """Assert Ra export: naming uses `<filename.stem>_<index>_ra` and suffix appends to the stem."""

    exporter = JSONExport(export_dir=tmp_path)
    out = exporter.export(ra)

    out2 = exporter.export(ra, suffix="second")

    assert out is not None
    assert out.name == f"{Path(ra.filename).stem}_{ra.index}_ra.json"

    assert out2 is not None
    assert out2.name == f"{Path(ra.filename).stem}_{ra.index}_ra_second.json"

    with out.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert "x" in data
    assert "y" in data
    assert "unified_meta" in data
    assert "metadata_user" in data


def test_export_ra_fallback_when_index_missing(tmp_path: Path, meas: ConvertedMeasurement) -> None:
    """Assert Ra export fallback: if `index` is None, the stem is `<filename.stem>_ra.json`."""

    ra = Ra.from_converted_measurement(meas, filename="dummy.spc")
    ra.index = None

    exporter = JSONExport(export_dir=tmp_path)
    out = exporter.export(ra)

    assert out is not None
    assert out.name == "dummy_ra.json"


# ------------------------------------------------
# export (RaProc)
# ------------------------------------------------


@pytest.mark.parametrize(
    "method, expected_name",
    [
        ("accumulate", "testmerged_acc.json"),
        ("average", "testmerged_avg.json"),
        ("median", "testmerged_med.json"),
        ("unknown", "testmerged_proc.json"),
    ],
)
def test_export_raproc_naming_and_skip(
    tmp_path: Path,
    raproc: RaProc,
    method: str,
    expected_name: str,
) -> None:
    """Assert RaProc export: method->tag naming and `SKIP` returns None on existing file."""

    raproc.processing["method"] = method

    exporter = JSONExport(export_dir=tmp_path)
    out = exporter.export(raproc)

    assert out is not None
    assert out.exists()
    assert out.name == expected_name

    exporter_skip = JSONExport(export_dir=tmp_path, mode=ExportMode.SKIP)
    out2 = exporter_skip.export(raproc)
    assert out2 is None


# ------------------------------------------------
# export (miscellaneous)
# ------------------------------------------------


def test_export_output_dir_override(tmp_path: Path, meas: ConvertedMeasurement) -> None:
    """Assert output_dir override: creates nested dirs and writes the JSON there."""

    exporter = JSONExport(export_dir=tmp_path)

    subdir = tmp_path / "nested" / "out"
    out = exporter.export(meas, output_dir=subdir)

    assert out is not None
    assert out.exists()
    assert out.parent == subdir


def test_export_output_dir_must_be_directory(tmp_path: Path, meas: ConvertedMeasurement) -> None:
    """Assert output_dir validation: ValueError if output_dir exists and is a file."""

    exporter = JSONExport(export_dir=tmp_path)

    not_a_dir = tmp_path / "not_a_dir"
    not_a_dir.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError):
        exporter.export(meas, output_dir=not_a_dir)


def test_export_unsupported_type_raises(tmp_path: Path) -> None:
    """Assert type safety: exporting unsupported objects raises TypeError."""

    exporter = JSONExport(export_dir=tmp_path)

    with pytest.raises(TypeError):
        exporter.export(object())


def test_export_leaves_no_tmp_files(tmp_path: Path, meas: ConvertedMeasurement) -> None:
    """Assert atomic write: no `*.tmp` remains after a successful export."""

    exporter = JSONExport(export_dir=tmp_path)
    out = exporter.export(meas)

    assert out is not None
    assert not list(out.parent.glob("*.tmp"))


def test_export_overwrite_semantics(tmp_path: Path, converted_file: ConvertedFile) -> None:
    """Assert OVERWRITE: existing JSON is replaced (rewritten) when exporting again."""

    exporter = JSONExport(export_dir=tmp_path, mode=ExportMode.OVERWRITE)

    out1 = exporter.export(converted_file)
    assert out1 is not None

    with out1.open("r", encoding="utf-8") as f:
        data1 = json.load(f)

    out1.write_text("{}", encoding="utf-8")

    out2 = exporter.export(converted_file)
    assert out2 is not None
    assert out2 == out1

    with out2.open("r", encoding="utf-8") as f:
        data2 = json.load(f)

    assert data2 == data1


# ------------------------------------------------
# CSVExport (Ra-only triplet CSV bundle)
# ------------------------------------------------


def test_csv_export_writes_triplet(tmp_path: Path, ra: Ra) -> None:
    """Assert CSVExport export: writes *_x.csv, *_y.csv and *_meta.csv for a single Ra."""

    exporter = CSVExport(export_dir=tmp_path)
    out = exporter.export(ra)

    assert out is not None
    x_path, y_path, meta_path = out

    assert x_path.exists()
    assert y_path.exists()
    assert meta_path.exists()

    assert x_path.name.endswith("_x.csv")
    assert y_path.name.endswith("_y.csv")
    assert meta_path.name.endswith("_meta.csv")

    # x/y files are wide: a single row where the first cell is the label.
    assert len(x_rows := _read_csv_rows(x_path)) == 1
    assert len(y_rows := _read_csv_rows(y_path)) == 1

    assert x_rows[0][0] == "x"
    assert y_rows[0][0] == "y"

    assert len(x_rows[0]) == len(list(ra.x)) + 1
    assert len(y_rows[0]) == len(list(ra.y)) + 1

    meta_rows = _read_csv_rows(meta_path)

    # metadata: header row + single data row
    assert len(meta_rows) == 2
    headers, values = meta_rows
    assert len(headers) == len(values)
    assert "filename" in headers
    assert "sample_name" in headers

    expected_stem = f"{Path(ra.filename).stem}_{ra.index}"
    assert x_path.stem.startswith(expected_stem)
    assert y_path.stem.startswith(expected_stem)
    assert meta_path.stem.startswith(expected_stem)


def test_csv_export_suffix_appends_to_stem(tmp_path: Path, ra: Ra) -> None:
    """Assert suffix: it is inserted before the *_x/_y/_meta endings."""

    exporter = CSVExport(export_dir=tmp_path)
    out = exporter.export(ra, suffix="second")

    assert out is not None
    x_path, y_path, meta_path = out

    assert x_path.stem.endswith("_second_x")
    assert y_path.stem.endswith("_second_y")
    assert meta_path.stem.endswith("_second_meta")


def test_csv_export_mode_skip_returns_none(tmp_path: Path, ra: Ra) -> None:
    """Assert SKIP: if any target exists, the whole bundle is skipped and returns None."""

    exporter = CSVExport(export_dir=tmp_path, mode=ExportMode.OVERWRITE)
    out = exporter.export(ra)
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


def test_csv_export_output_dir_override(tmp_path: Path, ra: Ra) -> None:
    """Assert output_dir override: writes the CSV bundle into the provided folder."""

    exporter = CSVExport(export_dir=tmp_path)
    subdir = tmp_path / "nested" / "out"

    out = exporter.export(ra, output_dir=subdir)

    assert out is not None
    x_path, y_path, meta_path = out

    assert x_path.parent == subdir
    assert y_path.parent == subdir
    assert meta_path.parent == subdir


def test_csv_export_output_dir_must_be_directory(tmp_path: Path, ra: Ra) -> None:
    """Assert output_dir validation: ValueError if output_dir exists and is a file."""

    exporter = CSVExport(export_dir=tmp_path)

    not_a_dir = tmp_path / "not_a_dir"
    not_a_dir.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError):
        exporter.export(ra, output_dir=not_a_dir)


def test_csv_export_rejects_non_ra_objects(tmp_path: Path, converted_file: ConvertedFile) -> None:
    """Assert type safety: CSVExport is Ra-only and raises TypeError for other types."""

    exporter = CSVExport(export_dir=tmp_path)

    with pytest.raises(TypeError):
        exporter.export(converted_file)  # type: ignore[arg-type]


def test_csv_export_leaves_no_tmp_files(tmp_path: Path, ra: Ra) -> None:
    """Assert atomic write: no `*.tmp` remains after a successful CSV export."""

    exporter = CSVExport(export_dir=tmp_path)
    out = exporter.export(ra)

    assert out is not None
    assert not list(tmp_path.glob("*.tmp"))
