"""Unit tests for core models."""

# pylint: disable=redefined-outer-name
# pylint: disable=import-error

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from PyRa_light.core import ConvertedFile, ConvertedMeasurement, Ra, RaProc


# ------------------------------------------------
# Fixtures: minimal in-memory objects
# ------------------------------------------------


@pytest.fixture()
def meas1() -> ConvertedMeasurement:
    """Fixture: a small, valid ConvertedMeasurement."""

    x = np.asarray([100.0, 200.0, 300.0], dtype=np.float64)
    y = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)

    return ConvertedMeasurement(
        path=Path("dummy.spc"),
        index=1,
        fileformat="spc",
        x=x,
        y=y,
        x_label="wavenumber [1/cm]",
        y_label="intensity [a.u]",
        raw_metadata={},
        other_spectra={},
    )


@pytest.fixture()
def meas2() -> ConvertedMeasurement:
    """Fixture: a second ConvertedMeasurement with the same x-grid."""

    x = np.asarray([100.0, 200.0, 300.0], dtype=np.float64)
    y = np.asarray([2.0, 3.0, 4.0], dtype=np.float64)

    return ConvertedMeasurement(
        path=Path("dummy.spc"),
        index=2,
        fileformat="spc",
        x=x,
        y=y,
        x_label="wavenumber [1/cm]",
        y_label="intensity [a.u]",
        raw_metadata={"instrument": "test"},
        other_spectra={},
    )


@pytest.fixture()
def converted_file(meas1: ConvertedMeasurement, meas2: ConvertedMeasurement) -> ConvertedFile:
    """Fixture: a ConvertedFile containing two measurements."""

    return ConvertedFile(
        path=Path("dummy.spc"),
        fileformat="spc",
        filename="dummy.spc",
        measurements=[meas1, meas2],
        raw={},
    )


@pytest.fixture()
def ra(meas1: ConvertedMeasurement) -> Ra:
    """Fixture: a Ra constructed from a ConvertedMeasurement."""

    return Ra.from_converted_measurement(meas1, filename="dummy.spc")


@pytest.fixture()
def raproc() -> RaProc:
    """Fixture: a minimal RaProc instance (constructed directly)."""
    x = np.asarray([100.0, 200.0, 300.0], dtype=np.float64)
    y = np.asarray([1.5, 2.5, 3.5], dtype=np.float64)

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
        unified_meta={},
        metadata_user={},
    )


# ------------------------------------------------
# ConvertedFile / ConvertedMeasurement
# ------------------------------------------------


def test_converted_file_from_dict_parses_measurements() -> None:
    """
    Builds a `ConvertedFile` from a converter-style dict 
    and checks that the measurements are created correctly. 
    This verifies that indices and basic file attributes survive the round-trip.
    """

    path = Path("dummy.spc")
    data = {
        "Fileformat": "spc",
        "Filename": "dummy.spc",
        "Measurements": {
            "1": {
                "Raman": {
                    "x": [1.0, 2.0, 3.0],
                    "y": [10.0, 11.0, 12.0],
                    "x_label": "wavenumber [1/cm]",
                },
                "Metadata": {"raw_metadata": {"k": "v"}},
                "Other_spectra": {},
            },
            "2": {
                "Raman": {
                    "x": [1.0, 2.0, 3.0],
                    "y": [20.0, 21.0, 22.0],
                    "x_label": "wavenumber [1/cm]",
                },
                "Metadata": {"raw_metadata": {}},
                "Other_spectra": {},
            },
        },
    }

    cf = ConvertedFile.from_dict(data=data, path=path)

    assert cf.path == path
    assert cf.fileformat == "spc"
    assert cf.filename == "dummy.spc"
    assert len(cf.measurements) == 2
    assert {m.index for m in cf.measurements} == {1, 2}


def test_converted_file_basic_structure(converted_file: ConvertedFile) -> None:
    """Checks that a `ConvertedFile` stores the expected basic fields and a list of measurements."""

    assert isinstance(converted_file.path, Path)
    assert isinstance(converted_file.fileformat, str)
    assert converted_file.fileformat
    assert isinstance(converted_file.filename, str)
    assert converted_file.filename

    assert isinstance(converted_file.measurements, list)
    assert len(converted_file.measurements) == 2


def test_converted_measurement_arrays(meas1: ConvertedMeasurement) -> None:
    """
    Verifies that a `ConvertedMeasurement` ends up with matching 1D numpy arrays for x and y. 
    It also checks that the measurement has a valid path and fileformat.
    """

    assert isinstance(meas1.path, Path)
    assert isinstance(meas1.fileformat, str)
    assert meas1.fileformat

    assert isinstance(meas1.x, np.ndarray)
    assert isinstance(meas1.y, np.ndarray)
    assert meas1.x.ndim == 1
    assert meas1.y.ndim == 1
    assert meas1.x.shape == meas1.y.shape
    assert meas1.x.size > 0


def test_converted_measurement_inject_meta(meas1: ConvertedMeasurement) -> None:
    """
    Peforms `inject_meta` on a `ConvertedMeasurement` with and without overwriting existing keys. Th
    e test makes sure metadata updates behave predictably.
    """

    assert "test_key" not in meas1.raw_metadata

    meas1.inject_meta({"test_key": "value1"})
    assert meas1.raw_metadata["test_key"] == "value1"

    meas1.inject_meta({"test_key": "value2"}, overwrite=False)
    assert meas1.raw_metadata["test_key"] == "value1"

    meas1.inject_meta({"test_key": "value3"}, overwrite=True)
    assert meas1.raw_metadata["test_key"] == "value3"


def test_converted_measurement_n_points(meas1: ConvertedMeasurement) -> None:
    """Checks that `n_points` matches the number of x-values in the spectrum."""

    assert meas1.n_points == meas1.x.size


def test_converted_measurement_to_dict(meas1: ConvertedMeasurement) -> None:
    """
    Converts a `ConvertedMeasurement` to a dict and verifies if it is JSON-friendly. 
    In particular, x and y should be plain lists rather than numpy arrays.
    """

    data = meas1.to_dict()

    assert isinstance(data, dict)
    assert data["path"] == str(meas1.path)
    assert data["index"] == meas1.index
    assert data["fileformat"] == meas1.fileformat

    assert isinstance(data["x"], list)
    assert isinstance(data["y"], list)
    assert len(data["x"]) == meas1.x.size
    assert len(data["y"]) == meas1.y.size

    assert data["raw_metadata"] == meas1.raw_metadata
    assert data["other_spectra"] == meas1.other_spectra


@pytest.mark.parametrize(
    "x,y",
    [
        ([1.0, 2.0, 3.0], [1.0, 2.0]),
        ([[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]]),
    ],
)
def test_converted_measurement_invalid_arrays_raise(x: object, y: object) -> None:
    """Feeds invalid x/y inputs into `ConvertedMeasurement` and expects it to fail"""

    with pytest.raises(ValueError):
        ConvertedMeasurement(
            path=Path("dummy.spc"),
            index=1,
            fileformat="spc",
            x=x,  # type: ignore[arg-type]
            y=y,  # type: ignore[arg-type]
            x_label="wavenumber [1/cm]",
            y_label="intensity [a.u]",
            raw_metadata={},
            other_spectra={},
        )


def test_converted_measurement_inject_meta_type_error(meas1: ConvertedMeasurement) -> None:
    """Makes sure `inject_meta` refuses non-dictionary input"""

    with pytest.raises(TypeError):
        meas1.inject_meta("not a dict")  # type: ignore[arg-type]


def test_converted_file_to_dict(converted_file: ConvertedFile) -> None:
    """Converts a `ConvertedFile` to a dict and checks that measurements are embedded as dicts"""

    data = converted_file.to_dict()

    assert isinstance(data, dict)
    assert data["path"] == str(converted_file.path)
    assert data["fileformat"] == converted_file.fileformat
    assert data["filename"] == converted_file.filename

    assert isinstance(data["measurements"], list)
    assert len(data["measurements"]) == len(converted_file.measurements)
    assert all(isinstance(m, dict) for m in data["measurements"])


def test_converted_file_filename_normalization() -> None:
    """
    Creates `ConvertedFile` instances with path-like filenames 
    and checks that the stored filename is basename-only. 
    It also checks the fallback to `path.name` when the filename is empty.
    """

    cf = ConvertedFile(
        path=Path("/tmp/a/b/c.spc"),
        fileformat="spc",
        filename="/tmp/a/b/c.spc",
        measurements=[],
        raw={},
    )
    assert cf.filename == "c.spc"

    cf2 = ConvertedFile(
        path=Path("/tmp/a/b/d.spc"),
        fileformat="spc",
        filename="",
        measurements=[],
        raw={},
    )
    assert cf2.filename == "d.spc"


# ------------------------------------------------
# Ra
# ------------------------------------------------


def test_ra_from_converted_measurement(meas1: ConvertedMeasurement) -> None:
    """
    Builds a `Ra` from a `ConvertedMeasurement` and checks that x/y are aligned 1D arrays. 
    It also verifies that basic metadata containers exist.
    """

    ra = Ra.from_converted_measurement(meas1, filename="dummy.spc")

    assert isinstance(ra, Ra)
    assert ra.filename == "dummy.spc"

    assert isinstance(ra.x, np.ndarray)
    assert isinstance(ra.y, np.ndarray)
    assert ra.x.ndim == 1
    assert ra.y.ndim == 1
    assert ra.x.shape == ra.y.shape
    assert ra.x.size > 0

    assert isinstance(ra.unified_meta, dict)
    assert isinstance(ra.metadata_user, dict)


def test_ra_from_converted_file(converted_file: ConvertedFile) -> None:
    """
    Builds a list of `Ra` objects from a `ConvertedFile` 
    and ensures you get one spectrum per measurement.
    """

    ras = Ra.from_converted_file(converted_file)

    assert isinstance(ras, list)
    assert len(ras) == len(converted_file.measurements)
    assert all(isinstance(r, Ra) for r in ras)


def test_ra_inject_and_edit_meta(ra: Ra) -> None:
    """
    Checks that user_metadata and unified_meta are written to the right places. 
    It also checks if it raises an error for unknown fields.
    """

    ra.inject_meta({"note": "nice spectrum"})
    assert ra.metadata_user.get("note") == "nice spectrum"
    assert "note" not in ra.unified_meta

    ra.inject_meta({"laser_wavelength_nm": 520.0}, overwrite=True)
    assert ra.unified_meta.get("laser_wavelength_nm") == 520.0

    ra.edit_meta(excitation_wavelength_nm=532.0)
    assert ra.unified_meta.get("excitation_wavelength_nm") == 532.0

    with pytest.raises(KeyError):
        ra.edit_meta(non_existing_field="foo")

    ra.unified_meta["sample_name"] = "original"
    with pytest.raises(ValueError):
        ra.edit_meta(overwrite=False, sample_name="changed")


def test_ra_to_dict(ra: Ra) -> None:
    """Converts a `Ra` to a dict and checks that arrays and metadata are JSON-compatible."""

    data = ra.to_dict()

    assert isinstance(data, dict)
    assert data["filename"] == ra.filename
    assert data["sample_name"] == ra.sample_name

    assert isinstance(data["x"], list)
    assert isinstance(data["y"], list)

    assert isinstance(data["unified_meta"], dict)
    assert isinstance(data["metadata_user"], dict)


@pytest.mark.parametrize(
    "x,y",
    [
        ([1.0, 2.0], [1.0, 2.0, 3.0]),
        ([[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]]),
    ],
)
def test_ra_invalid_arrays_raise(x: object, y: object) -> None:
    """Tries to construct a `Ra` with invalid x/y inputs and expects an error."""

    with pytest.raises(ValueError):
        Ra(
            filename="dummy.spc",
            x=x,  # type: ignore[arg-type]
            y=y,  # type: ignore[arg-type]
            unified_meta={},
            metadata_user={},
        )


def test_ra_unified_meta_type_error() -> None:
    """Creates a `Ra` with an invalid unified_meta type and checks that it is rejected."""
    with pytest.raises(TypeError):
        Ra(
            filename="dummy.spc",
            x = np.asarray([100.0, 200.0, 300.0], dtype=np.float64),
            y = np.asarray([1.5, 2.5, 3.5], dtype=np.float64),
            unified_meta="not a dict",  # type: ignore[arg-type]
            metadata_user={},
        )


def test_ra_filename_normalization(meas1: ConvertedMeasurement) -> None:
    """
    Passes a path-like filename into Ra construction 
    and checks that it is normalized to the basename.
    """

    ra2 = Ra.from_converted_measurement(meas1, filename="/tmp/a/b/c.spc")
    assert ra2.filename == "c.spc"


def test_ra_inject_meta_type_error(ra: Ra) -> None:
    """Verifies that `Ra.inject_meta` requires a dict and raises on other input types."""

    with pytest.raises(TypeError):
        ra.inject_meta("not a dict")  # type: ignore[arg-type]


# ------------------------------------------------
# RaProc
# ------------------------------------------------


def test_raproc_basic_invariants(raproc: RaProc) -> None:
    """
    Checks the core shape and invariants of a `RaProc`. 
    Filenames/sample names must line up, and processed x/y must be matching 1D arrays.
    """

    assert isinstance(raproc, RaProc)

    assert len(raproc.filenames) == len(raproc.sample_names)
    assert raproc.processing["n_spectra"] == len(raproc.filenames)

    assert isinstance(raproc.x_processed, np.ndarray)
    assert isinstance(raproc.y_processed, np.ndarray)
    assert raproc.x_processed.ndim == 1
    assert raproc.y_processed.ndim == 1
    assert raproc.x_processed.shape == raproc.y_processed.shape
    assert raproc.x_processed.size > 0


def test_raproc_inject_and_edit_meta(raproc: RaProc) -> None:
    """Does metadata updates on `RaProc` and verifies the expected failures."""

    raproc.inject_meta({"group_comment": "merged"}, overwrite=False)
    assert raproc.metadata_user.get("group_comment") == "merged"
    assert "group_comment" not in raproc.unified_meta

    raproc.inject_meta({"sample_type": "sample"}, overwrite=True)
    assert raproc.unified_meta.get("sample_type") == "sample"

    raproc.edit_meta(sample_type="background")
    assert raproc.unified_meta.get("sample_type") == "background"

    with pytest.raises(KeyError):
        raproc.edit_meta(foo_bar="baz")

    raproc.unified_meta["instrument_manufacturer"] = "Bruker"
    with pytest.raises(ValueError):
        raproc.edit_meta(overwrite=False, instrument_manufacturer="Renishaw")


def test_raproc_to_dict(raproc: RaProc) -> None:
    """Converts a `RaProc` to a dict and checks that the result is JSON-compartible."""

    data = raproc.to_dict()

    assert isinstance(data, dict)
    assert data["merged_sample_name"] == raproc.merged_sample_name
    assert data["filenames"] == raproc.filenames
    assert data["sample_names"] == raproc.sample_names

    assert isinstance(data["x_processed"], list)
    assert isinstance(data["y_processed"], list)

    assert isinstance(data["processing"], dict)
    assert isinstance(data["unified_meta"], dict)
    assert isinstance(data["metadata_user"], dict)


@pytest.mark.parametrize(
    "x_processed,y_processed",
    [
        ([1.0, 2.0, 3.0], [1.0, 2.0]),
        ([[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]]),
    ],
)
def test_raproc_invalid_arrays_raise(x_processed: object, y_processed: object) -> None:
    """Tries to build `RaProc` instances with invalid processed arrays and expects errors."""

    with pytest.raises(ValueError):
        RaProc(
            merged_sample_name="testmerged",
            filenames=["a.spc", "b.spc"],
            sample_names=["A", "B"],
            x_processed=x_processed,  # type: ignore[arg-type]
            y_processed=y_processed,  # type: ignore[arg-type]
            processing={
                "method": "average",
                "n_spectra": 2,
                "merge_timestamp": "2025-01-01T00:00:00Z",
                "source_indices": [0, 1],
            },
            unified_meta={},
            metadata_user={},
        )


def test_raproc_inject_meta_type_error(raproc: RaProc) -> None:
    """Checks that `RaProc.inject_meta` refuses non-dict input."""

    with pytest.raises(TypeError):
        raproc.inject_meta("not a dict")  # type: ignore[arg-type]
