"""Integration tests for the merge pipeline (real files -> Ra -> merge_ras -> RaProc).

These tests run against real fixture files under ``test/integration_tests/data/merge`` and
validate that unified Ra objects produced from converted measurements can be merged
into a sane RaProc object for each supported merge method.
"""

# pylint: disable=redefined-outer-name
# pylint: disable=import-error

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from PyRa_light.core import Ra, merge_ras
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


def test_merge_integration_fixtures_present() -> None:
    """Assert merge fixtures exist to avoid a silent empty test run."""
    if not DATA_ROOT.exists():
        pytest.skip(f"Merge integration data directory not found: {DATA_ROOT}")
    assert MERGE_FILES, f"No supported merge fixture files found under: {DATA_ROOT}"


def _ras_from_file(file_path: Path) -> list[Ra]:
    """Load a real file and build one Ra per ConvertedMeasurement."""
    converted = load_file(file_path)

    converted_path = Path(str(getattr(converted, "path")))
    assert converted_path.resolve() == file_path.resolve()

    measurements = getattr(converted, "measurements", None)
    assert isinstance(measurements, list)
    assert measurements, "ConvertedFile should contain at least one measurement"

    ras: list[Ra] = []
    for meas in measurements:
        ra = Ra.from_converted_measurement(  # type: ignore[attr-defined]
            meas,
            filename=getattr(converted, "filename", None),
        )
        ras.append(ra)

    return ras


@pytest.mark.parametrize("method", ["average", "median", "accumulate"])
def test_merge_pipeline_builds_raproc(method: str) -> None:
    """Build Ra from real files and validate merge outputs are structurally sane."""
    if not MERGE_FILES:
        pytest.skip("No merge fixture files available")

    if not hasattr(Ra, "from_converted_measurement"):
        pytest.skip("Ra.from_converted_measurement is not available in this version")

    ras: list[Ra] = []

    for file_path in MERGE_FILES:
        try:
            ras.extend(_ras_from_file(file_path))
        except ImportError as exc:
            pytest.skip(f"Optional dependency missing for {file_path.name}: {exc}")

    assert len(ras) >= 2, "Need at least two Ra objects for merge integration testing"

    raproc = merge_ras(ras, method=method)

    x = np.asarray(getattr(raproc, "x_processed"))
    y = np.asarray(getattr(raproc, "y_processed"))

    assert x.ndim == 1
    assert y.ndim == 1
    assert x.size == y.size
    assert x.size > 10
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))

    y_err = getattr(raproc, "y_err", None)
    y_sem = getattr(raproc, "y_sem", None)
    y_mad = getattr(raproc, "y_mad", None)
    y_q25 = getattr(raproc, "y_q25", None)
    y_q75 = getattr(raproc, "y_q75", None)

    if method == "average":
        assert y_err is not None
        assert y_sem is not None
        assert y_mad is None
        assert y_q25 is not None
        assert y_q75 is not None
    elif method == "median":
        assert y_err is not None
        assert y_mad is not None
        assert y_sem is None
        assert y_q25 is not None
        assert y_q75 is not None
    else:
        assert y_err is None
        assert y_sem is None
        assert y_mad is None
        assert y_q25 is None
        assert y_q75 is None

    data = raproc.to_dict()
    assert isinstance(data, dict)
    assert data


def test_merge_pipeline_rejects_mismatched_x_grids() -> None:
    """Merge should raise on mismatched x grids when validation is enabled."""
    if not MERGE_FILES:
        pytest.skip("No merge fixture files available")

    if not hasattr(Ra, "from_converted_measurement"):
        pytest.skip("Ra.from_converted_measurement is not available in this version")

    file_path = MERGE_FILES[0]

    try:
        ras = _ras_from_file(file_path)
    except ImportError as exc:
        pytest.skip(f"Optional dependency missing for {file_path.name}: {exc}")

    ra0 = ras[0]
    x0 = np.asarray(getattr(ra0, "x"))
    y0 = np.asarray(getattr(ra0, "y"))

    ra_bad = Ra(
        filename=getattr(ra0, "filename", "dummy.spc"),
        x=x0 + 1.0,
        y=y0,
        unified_meta=getattr(ra0, "unified_meta", {}),
        metadata_user=getattr(ra0, "metadata_user", {}),
    )

    with pytest.raises(ValueError):
        merge_ras([ra0, ra_bad], method="average", validate_x_grid=True)
