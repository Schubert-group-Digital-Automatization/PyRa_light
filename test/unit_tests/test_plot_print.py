"""Plot/print tests for PyRa_light.

This module unit-tests the public functions in :mod:`PyRa_light.plot_print`.
It also includes a small smoke suite for the thin plotting wrappers on the core
models that delegate into :func:`PyRa_light.plot_print.plot_raman_xy`.

Notes
-----
The matplotlib backend is forced to ``Agg`` so tests run headless.
"""

# pylint: disable=redefined-outer-name
# pylint: disable=import-error
# pylint: disable=wrong-import-position

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from PyRa_light import plot_print as pp
from PyRa_light.core import ConvertedFile, ConvertedMeasurement, RaProc, Ra




# ------------------------------------------------
# Fixtures
# ------------------------------------------------


@pytest.fixture()
def xy() -> tuple[np.ndarray, np.ndarray]:
    """A small, deterministic x/y pair suitable for plotting tests."""
    x_vals = np.linspace(100.0, 2000.0, 200)
    y_vals = np.sin(x_vals / 100.0)
    return x_vals, y_vals


def _close_fig(fig) -> None:
    """Close a matplotlib figure to avoid leaking GUI resources during tests."""

    plt.close(fig)


def _patch_show(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Patch `matplotlib.pyplot.show` and return a small call counter."""
    called = {"n": 0}

    def _fake_show() -> None:
        called["n"] += 1

    monkeypatch.setattr(plt, "show", _fake_show)
    return called


# ------------------------------------------------
# format_metadata_tables / print_metadata_tables
# ------------------------------------------------


def test_format_metadata_tables_returns_string() -> None:
    """Formatting metadata tables returns a readable string with a trailing newline."""
    tables = [("Test table", [["a", 1, "x"], ["b", 2, "y"]])]

    formatted = pp.format_metadata_tables(tables, tablefmt="github")

    assert isinstance(formatted, str)
    assert "Test table" in formatted
    assert "a" in formatted
    assert "b" in formatted
    assert formatted.endswith("\n")


def test_print_metadata_tables_return_str() -> None:
    """When return_str=True, the function should return the formatted output."""
    tables = [("T", [["k", "v", "src"]])]

    out = pp.print_metadata_tables(tables, return_str=True)

    assert isinstance(out, str)
    assert "T" in out
    assert "k" in out
    assert "v" in out


def test_print_metadata_tables_prints(capsys: pytest.CaptureFixture[str]) -> None:
    """When return_str=False, the function prints to stdout and returns None."""
    tables = [("T", [["k", "v", "src"]])]

    out = pp.print_metadata_tables(tables, return_str=False)

    assert out is None
    captured = capsys.readouterr().out
    assert "T" in captured
    assert "k" in captured
    assert "v" in captured


# ------------------------------------------------
# plot_raman_xy
# ------------------------------------------------


def test_plot_raman_xy_basic(xy: tuple[np.ndarray, np.ndarray]) -> None:
    """Basic plotting produces a single line with the original x/y data."""
    x_vals, y_vals = xy

    fig, ax = pp.plot_raman_xy(x_vals, y_vals, title="basic", show=False)

    assert fig is not None
    assert ax is not None
    assert len(ax.lines) == 1
    np.testing.assert_allclose(ax.lines[0].get_xdata(), x_vals) # type: ignore
    np.testing.assert_allclose(ax.lines[0].get_ydata(), y_vals) # type: ignore

    _close_fig(fig)


def test_plot_raman_xy_uses_existing_axes(xy: tuple[np.ndarray, np.ndarray]) -> None:
    """Passing an existing Axes should plot into it and return the same object."""
    x_vals, y_vals = xy

    fig, ax = plt.subplots()
    fig_out, ax_out = pp.plot_raman_xy(x_vals, y_vals, ax=ax, title="existing", show=False)

    assert fig_out is fig
    assert ax_out is ax
    assert len(ax.lines) == 1

    _close_fig(fig)


def test_plot_raman_xy_calls_show_when_requested(
    monkeypatch: pytest.MonkeyPatch,
    xy: tuple[np.ndarray, np.ndarray],
) -> None:
    """When show=True, the helper calls plt.show exactly once."""
    called = _patch_show(monkeypatch)

    x_vals, y_vals = xy
    fig, _ = pp.plot_raman_xy(x_vals, y_vals, title="show", show=True)

    assert called["n"] == 1
    _close_fig(fig)


def test_plot_raman_xy_with_band(xy: tuple[np.ndarray, np.ndarray]) -> None:
    """Providing y_low/y_high adds a filled band in addition to the line."""
    x_vals, y_vals = xy
    y_low = y_vals - 0.1
    y_high = y_vals + 0.1

    fig, ax = pp.plot_raman_xy(
        x_vals,
        y_vals,
        y_low=y_low,
        y_high=y_high,
        band_alpha=0.2,
        show=False,
    )

    assert len(ax.lines) == 1
    assert len(ax.collections) >= 1

    _close_fig(fig)


def test_plot_raman_xy_partial_band_is_ignored(xy: tuple[np.ndarray, np.ndarray]) -> None:
    """A band is only drawn when both y_low and y_high are provided."""
    x_vals, y_vals = xy

    fig, ax = pp.plot_raman_xy(x_vals, y_vals, y_low=y_vals - 0.1, show=False)
    assert len(ax.lines) == 1
    assert len(ax.collections) == 0
    _close_fig(fig)

    fig, ax = pp.plot_raman_xy(x_vals, y_vals, y_high=y_vals + 0.1, show=False)
    assert len(ax.lines) == 1
    assert len(ax.collections) == 0
    _close_fig(fig)


# ------------------------------------------------
# Thin wrapper smoke tests
# ------------------------------------------------


def _dummy_converted_measurement(
    index: int,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
) -> ConvertedMeasurement:
    """Create a small ConvertedMeasurement suitable for wrapper smoke tests."""
    return ConvertedMeasurement(
        path=Path("dummy.opus"),
        index=index,
        fileformat="opus",
        x=x_vals,
        y=y_vals,
        x_label="wavenumber [1/cm]",
        y_label="intensity [a.u.]",
        raw_metadata={},
        other_spectra={},
    )


@pytest.mark.smoke
def test_converted_measurement_plot_raman(
    monkeypatch: pytest.MonkeyPatch,
    xy: tuple[np.ndarray, np.ndarray],
) -> None:
    """ConvertedMeasurement.plot_raman should delegate to the shared plotting helper."""
    called = _patch_show(monkeypatch)

    x_vals, y_vals = xy
    meas = _dummy_converted_measurement(index=1, x_vals=x_vals, y_vals=y_vals)

    fig, ax = meas.plot_raman(show=False)

    assert len(ax.lines) == 1
    assert called["n"] == 0

    _close_fig(fig)


@pytest.mark.smoke
def test_converted_file_plot_raman_overlay_and_single(
    monkeypatch: pytest.MonkeyPatch,
    xy: tuple[np.ndarray, np.ndarray],
) -> None:
    """ConvertedFile.plot_raman should overlay all spectra or plot a single index."""
    called = _patch_show(monkeypatch)

    x_vals, y_vals = xy
    meas1 = _dummy_converted_measurement(index=1, x_vals=x_vals, y_vals=y_vals)
    meas2 = _dummy_converted_measurement(index=2, x_vals=x_vals, y_vals=y_vals * 2)

    converted = ConvertedFile(
        path=Path("dummy.opus"),
        fileformat="opus",
        filename="dummy.opus",
        measurements=[meas1, meas2],
        raw={},
    )

    fig, ax = converted.plot_raman(show=False)
    assert len(ax.lines) == 2 # pyright: ignore[reportOptionalMemberAccess]
    _close_fig(fig)

    fig, ax = converted.plot_raman(index=2, show=False)
    assert len(ax.lines) == 1 # pyright: ignore[reportOptionalMemberAccess]
    _close_fig(fig)

    fig, _ = converted.plot_raman(show=True)
    assert called["n"] == 1
    _close_fig(fig)


@pytest.mark.smoke
def test_raproc_plot_raman_bands(
    monkeypatch: pytest.MonkeyPatch,
    xy: tuple[np.ndarray, np.ndarray],
) -> None:
    """RaProc.plot_raman should support band selection (y_err and IQR)."""
    called = _patch_show(monkeypatch)

    x_vals, y_vals = xy
    y_err = np.full_like(y_vals, 0.05)
    y_q25 = y_vals - 0.1
    y_q75 = y_vals + 0.1

    rp = RaProc(
        filenames=["a", "b"],
        sample_names=["s1", "s2"],
        merged_sample_name="TC001_MKY",
        x_processed=x_vals,
        y_processed=y_vals,
        y_err=y_err,
        y_q25=y_q25,
        y_q75=y_q75,
        processing={
            "method": "average",
            "n_spectra": 2,
            "merge_timestamp": "1970-01-01T00:00:00Z",
            "source_indices": [0, 1],
        },
        unified_meta={},
        metadata_user={},
    )

    fig, ax = rp.plot_raman(band="y_err", show=False)
    assert len(ax.lines) == 1
    assert len(ax.collections) >= 1
    _close_fig(fig)

    fig, ax = rp.plot_raman(band="iqr", show=False)
    assert len(ax.lines) == 1
    assert len(ax.collections) >= 1
    _close_fig(fig)

    fig, _ = rp.plot_raman(band="y_err", show=True)
    assert called["n"] == 1
    _close_fig(fig)


@pytest.mark.smoke
def test_converted_measurement_print_metadata_returns_string(
    xy: tuple[np.ndarray, np.ndarray],
) -> None:
    """ConvertedMeasurement.print_metadata should return a non-empty string when requested."""
    x_vals, y_vals = xy
    meas = _dummy_converted_measurement(index=1, x_vals=x_vals, y_vals=y_vals)

    out = meas.print_metadata(return_str=True)

    assert isinstance(out, str)
    assert out.strip() != ""


@pytest.mark.smoke
def test_converted_file_print_metadata_prints(
    capsys: pytest.CaptureFixture[str],
    xy: tuple[np.ndarray, np.ndarray],
) -> None:
    """ConvertedFile.print_metadata should print to stdout when return_str=False."""
    x_vals, y_vals = xy
    meas = _dummy_converted_measurement(index=1, x_vals=x_vals, y_vals=y_vals)

    converted = ConvertedFile(
        path=Path("dummy.opus"),
        fileformat="opus",
        filename="dummy.opus",
        measurements=[meas],
        raw={},
    )

    out = converted.print_metadata(return_str=False)

    assert out is None
    captured = capsys.readouterr().out
    assert captured.strip() != ""

@pytest.mark.smoke
def test_ra_print_metadata_returns_string(xy: tuple[np.ndarray, np.ndarray]) -> None:
    """Ra.print_metadata should return a non-empty string when requested."""
    x_vals, y_vals = xy

    ra = Ra(
        filename="dummy.spc",
        x=x_vals,
        y=y_vals,
        unified_meta={},
        metadata_user={},
    )

    out = ra.print_metadata(return_str=True)

    assert isinstance(out, str)
    assert out.strip() != ""

@pytest.mark.smoke
def test_raproc_print_metadata_returns_string(xy: tuple[np.ndarray, np.ndarray]) -> None:
    """RaProc.print_metadata should return a non-empty string when requested."""
    x_vals, y_vals = xy

    rp = RaProc(
        filenames=["a", "b"],
        sample_names=["s1", "s2"],
        merged_sample_name="TC001_MKY",
        x_processed=x_vals,
        y_processed=y_vals,
        processing={
            "method": "average",
            "n_spectra": 2,
            "merge_timestamp": "1970-01-01T00:00:00Z",
            "source_indices": [0, 1],
        },
        unified_meta={},
        metadata_user={},
    )

    out = rp.print_metadata(return_str=True)

    assert isinstance(out, str)
    assert out.strip() != ""
