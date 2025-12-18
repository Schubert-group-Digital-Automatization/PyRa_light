"""Unit tests for Raman spectrum merge logic.

These tests cover `merge_ras` in a purely in-memory way (no loader, no files).
They focus on numeric correctness, basic metadata bookkeeping, and error paths.
"""

# pylint: disable=redefined-outer-name
# pylint: disable=import-error

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from PyRa_light.core import Ra, RaProc, merge_ras


def _make_ra(
    *,
    filename: str,
    x: np.ndarray,
    y: np.ndarray,
    sample_name: str,
    unified_meta: dict | None = None,
) -> Ra:
    """Create a small Ra instance with deterministic fields for unit tests."""

    ra = Ra(
        filename=filename,
        x=x,
        y=y,
        unified_meta={} if unified_meta is None else unified_meta,
        metadata_user={},
    )
    ra.sample_name = sample_name
    return ra


# ------------------------------------------------
# Fixtures: small in-memory Ra objects
# ------------------------------------------------


@pytest.fixture()
def ras_same_grid() -> list[Ra]:
    """Two small Ra objects sharing the same x-grid."""

    x = np.asarray([100.0, 200.0, 300.0], dtype=np.float64)

    r1 = _make_ra(
        filename="a.spc",
        x=x,
        y=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        sample_name="A",
    )

    r2 = _make_ra(
        filename="b.spc",
        x=x,
        y=np.asarray([2.0, 3.0, 4.0], dtype=np.float64),
        sample_name="B",
    )

    return [r1, r2]


@pytest.fixture()
def ras_three_grid() -> list[Ra]:
    """Three Ra objects sharing an x-grid, with values chosen for stable statistics."""

    x = np.asarray([100.0, 200.0, 300.0], dtype=np.float64)

    r1 = _make_ra(
        filename="a.spc",
        x=x,
        y=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        sample_name="S1",
    )

    r2 = _make_ra(
        filename="b.spc",
        x=x,
        y=np.asarray([5.0, 6.0, 7.0], dtype=np.float64),
        sample_name="S2",
    )

    r3 = _make_ra(
        filename="c.spc",
        x=x,
        y=np.asarray([9.0, 10.0, 11.0], dtype=np.float64),
        sample_name="S3",
    )

    return [r1, r2, r3]


@pytest.fixture()
def ra_single() -> Ra:
    """One small Ra instance (used for input-size validation)."""
    x = np.asarray([100.0, 200.0, 300.0], dtype=np.float64)
    return _make_ra(
        filename="a.spc",
        x=x,
        y=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        sample_name="A",
    )


@pytest.fixture()
def ras_with_paths() -> list[Ra]:
    """Two Ra objects where filenames contain path components."""

    x = np.asarray([100.0, 200.0, 300.0], dtype=np.float64)

    r1 = _make_ra(
        filename=str(Path("/tmp/a/b/c.spc")),
        x=x,
        y=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        sample_name="TC001_MKY_P1S1",
    )

    r2 = _make_ra(
        filename=str(Path("/tmp/a/b/d.spc")),
        x=x,
        y=np.asarray([2.0, 3.0, 4.0], dtype=np.float64),
        sample_name="TC001_MKY_P1S2",
    )

    return [r1, r2]


@pytest.fixture()
def ras_mismatched_grid(ras_same_grid: list[Ra]) -> list[Ra]:
    """Two Ra objects where the second has a different x-grid."""

    r1, _ = ras_same_grid

    x2 = np.asarray([101.0, 201.0, 301.0], dtype=np.float64)
    r2 = _make_ra(
        filename="b.spc",
        x=x2,
        y=np.asarray([2.0, 3.0, 4.0], dtype=np.float64),
        sample_name="B",
    )

    return [r1, r2]


# ------------------------------------------------
# merge_ras: numeric correctness and bookkeeping
# ------------------------------------------------


@pytest.mark.parametrize(
    "method, expected",
    [
        ("average", np.asarray([1.5, 2.5, 3.5], dtype=np.float64)),
        ("median", np.asarray([1.5, 2.5, 3.5], dtype=np.float64)),
        ("accumulate", np.asarray([3.0, 5.0, 7.0], dtype=np.float64)),
    ],
)
def test_merge_ras_methods_compute_expected_y(
    ras_same_grid: list[Ra],
    method: str,
    expected: np.ndarray,
) -> None:
    """Merging produces the expected processed y-values for each supported method."""

    rp = merge_ras(ras_same_grid, method=method)

    assert isinstance(rp, RaProc)
    np.testing.assert_allclose(rp.x_processed, ras_same_grid[0].x)
    np.testing.assert_allclose(rp.y_processed, expected)

    assert rp.processing.get("method") == method
    assert rp.processing.get("n_spectra") == len(ras_same_grid)

    # Required ProcessingInfo fields
    assert isinstance(rp.processing.get("merge_timestamp"), str)
    src = rp.processing.get("source_indices")
    assert isinstance(src, list)
    assert len(src) == len(ras_same_grid)

    assert rp.filenames == [r.filename for r in ras_same_grid]
    assert rp.sample_names == [r.sample_name for r in ras_same_grid]


@pytest.mark.parametrize(
    "method, expect_err, expect_sem, expect_mad, expect_q",
    [
        ("average", True, True, False, True),
        ("median", True, False, True, True),
        ("accumulate", False, False, False, False),
    ],
)
def test_merge_ras_optional_outputs_depend_on_method(
    ras_same_grid: list[Ra],
    method: str,
    expect_err: bool,
    expect_sem: bool,
    expect_mad: bool,
    expect_q: bool,
) -> None:
    """The merge method controls which dispersion/quantile arrays are produced."""

    rp = merge_ras(ras_same_grid, method=method)

    if expect_err:
        assert rp.y_err is not None
        assert rp.y_err.shape == rp.y_processed.shape
    else:
        assert rp.y_err is None

    if expect_sem:
        assert rp.y_sem is not None
        assert rp.y_sem.shape == rp.y_processed.shape
    else:
        assert rp.y_sem is None

    if expect_mad:
        assert rp.y_mad is not None
        assert rp.y_mad.shape == rp.y_processed.shape
    else:
        assert rp.y_mad is None

    if expect_q:
        assert rp.y_q25 is not None
        assert rp.y_q75 is not None
        assert rp.y_q25.shape == rp.y_processed.shape
        assert rp.y_q75.shape == rp.y_processed.shape
    else:
        assert rp.y_q25 is None
        assert rp.y_q75 is None


def test_merge_ras_average_dispersion_values_are_correct(ras_three_grid: list[Ra]) -> None:
    """Average merge should compute std (ddof=1), SEM, and IQR consistently."""

    rp = merge_ras(ras_three_grid, method="average")

    stack = np.stack([ra.y for ra in ras_three_grid], axis=0)
    expected_err = np.std(stack, axis=0, ddof=1)
    expected_sem = expected_err / np.sqrt(stack.shape[0])
    expected_q25 = np.quantile(stack, 0.25, axis=0)
    expected_q75 = np.quantile(stack, 0.75, axis=0)

    assert rp.y_err is not None
    assert rp.y_sem is not None
    assert rp.y_q25 is not None
    assert rp.y_q75 is not None
    assert rp.y_mad is None

    np.testing.assert_allclose(rp.y_err, expected_err)
    np.testing.assert_allclose(rp.y_sem, expected_sem)
    np.testing.assert_allclose(rp.y_q25, expected_q25)
    np.testing.assert_allclose(rp.y_q75, expected_q75)


def test_merge_ras_median_dispersion_values_are_correct(ras_three_grid: list[Ra]) -> None:
    """Median merge should compute std (ddof=1), MAD, and IQR consistently."""

    rp = merge_ras(ras_three_grid, method="median")

    stack = np.stack([ra.y for ra in ras_three_grid], axis=0)
    expected_err = np.std(stack, axis=0, ddof=1)
    expected_q25 = np.quantile(stack, 0.25, axis=0)
    expected_q75 = np.quantile(stack, 0.75, axis=0)

    median = np.median(stack, axis=0)
    expected_mad = np.median(np.abs(stack - median), axis=0)

    assert rp.y_err is not None
    assert rp.y_mad is not None
    assert rp.y_q25 is not None
    assert rp.y_q75 is not None
    assert rp.y_sem is None

    np.testing.assert_allclose(rp.y_err, expected_err)
    np.testing.assert_allclose(rp.y_mad, expected_mad)
    np.testing.assert_allclose(rp.y_q25, expected_q25)
    np.testing.assert_allclose(rp.y_q75, expected_q75)


def test_merge_ras_outputs_are_float64_and_x_is_copied(ras_same_grid: list[Ra]) -> None:
    """Merged arrays should be float64.
    The processed x-grid should be a copy and must not alias the input x array.
    """

    rp = merge_ras(ras_same_grid, method="average")

    assert rp.y_processed.dtype == np.float64
    assert rp.x_processed.dtype == np.float64

    original_first = ras_same_grid[0].x[0]
    rp.x_processed[0] = rp.x_processed[0] + 1.0
    assert ras_same_grid[0].x[0] == original_first


def test_merge_ras_filenames_are_basename_only(ras_with_paths: list[Ra]) -> None:
    """Merged filenames should be basename-only, even if inputs contain directories."""

    rp = merge_ras(ras_with_paths, method="average")
    assert rp.filenames == ["c.spc", "d.spc"]


def test_merge_ras_common_prefix_is_trimmed_to_boundary(ras_with_paths: list[Ra]) -> None:
    """Merged sample name uses the common prefix trimmed to the last '_' boundary."""

    rp = merge_ras(ras_with_paths, method="average")
    assert rp.merged_sample_name == "TC001_MKY"
    assert rp.unified_meta.get("sample_name") == "TC001_MKY"


def test_merge_ras_common_prefix_without_delimiter_is_not_trimmed() -> None:
    """If there is no delimiter in the common prefix, it is kept as-is."""

    x = np.asarray([100.0, 200.0, 300.0], dtype=np.float64)
    r1 = _make_ra(
        filename="a.spc",
        x=x,
        y=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        sample_name="ABC",
    )

    r2 = _make_ra(
        filename="b.spc",
        x=x,
        y=np.asarray([2.0, 3.0, 4.0], dtype=np.float64),
        sample_name="ABD",
    )

    rp = merge_ras([r1, r2], method="average")
    assert rp.merged_sample_name == "AB"


def test_merge_ras_merges_selected_metadata_fields() -> None:
    """A few representative metadata fields should be merged using their rules."""

    x = np.asarray([100.0, 200.0, 300.0], dtype=np.float64)

    r1 = _make_ra(
        filename="a.spc",
        x=x,
        y=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        sample_name="S1",
        unified_meta={
            "excitation_wavelength_nm": 500.0,
            "instrument_manufacturer": "Bruker",
            "measurement_datetime": "2025-01-01 10:00",
        },
    )

    r2 = _make_ra(
        filename="b.spc",
        x=x,
        y=np.asarray([2.0, 3.0, 4.0], dtype=np.float64),
        sample_name="S2",
        unified_meta={
            "excitation_wavelength_nm": 700.0,
            "instrument_manufacturer": "Bruker",
            "measurement_datetime": "2025-01-01 11:00",
        },
    )

    rp = merge_ras([r1, r2], method="average")

    # numeric fields are averaged
    assert rp.unified_meta.get("excitation_wavelength_nm") == pytest.approx(600.0)

    # common string fields are kept
    assert rp.unified_meta.get("instrument_manufacturer") == "Bruker"

    # values across spectra are collected
    dt = rp.unified_meta.get("measurement_datetime")
    assert isinstance(dt, str)
    assert "2025-01-01" in dt
    assert ";" in dt


def test_merge_ras_unknown_method_raises(ras_same_grid: list[Ra]) -> None:
    """Unknown merge methods are treated as invalid input and should raise ValueError."""

    with pytest.raises(ValueError):
        merge_ras(ras_same_grid, method="does_not_exist")


# ------------------------------------------------
# merge_ras: error paths
# ------------------------------------------------


def test_merge_ras_single_input_raises(ra_single: Ra) -> None:
    """A merge needs at least two spectra, so a single input should raise."""

    with pytest.raises(ValueError):
        merge_ras([ra_single], method="average")


def test_merge_ras_empty_input_raises() -> None:
    """An empty input list is not mergeable and should fail."""

    with pytest.raises(ValueError):
        merge_ras([], method="average")


def test_merge_ras_mismatched_x_raises(ras_mismatched_grid: list[Ra]) -> None:
    """Merging should raise when the input spectra do not share the same x-grid."""

    with pytest.raises(ValueError):
        merge_ras(ras_mismatched_grid, method="average")


def test_merge_ras_mismatched_x_can_be_skipped(ras_mismatched_grid: list[Ra]) -> None:
    """When x-grid validation is disabled, merge should proceed using the first x-grid."""

    rp = merge_ras(ras_mismatched_grid, method="average", validate_x_grid=False)
    assert isinstance(rp, RaProc)
    np.testing.assert_allclose(rp.x_processed, ras_mismatched_grid[0].x)
