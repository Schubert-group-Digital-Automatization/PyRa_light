"""Integration tests for the merge pipeline (real files -> Ra -> merge_ras -> RaProc).

These tests run against real fixture files under ``test/integration_tests/data/merge`` and
validate that unified ``Ra`` objects can be merged into a structurally sane ``RaProc``
object for each supported merge method.
"""

# pylint: disable=redefined-outer-name
# pylint: disable=import-error

from __future__ import annotations

import numpy as np
import pytest

from PyRa_light.core import Ra, merge_ras


pytestmark = pytest.mark.integration

# Limit the number of spectra merged to keep the integration suite fast and deterministic.
_MAX_MERGE_SPECTRA = 10


@pytest.mark.parametrize("method", ["average", "median", "accumulate"])
def test_merge_pipeline_builds_raproc(method: str, merge_ra_pool: list[Ra]) -> None:
    """Merge real Ra objects and validate the output is structurally sane."""
    ras = merge_ra_pool[:_MAX_MERGE_SPECTRA]
    assert len(ras) >= 2

    raproc = merge_ras(ras, method=method)

    x = np.asarray(raproc.x_processed)
    y = np.asarray(raproc.y_processed)

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


def test_merge_pipeline_rejects_mismatched_x_grids(merge_ra_pool: list[Ra]) -> None:
    """Merge should raise on mismatched x grids when validation is enabled."""
    ra0 = merge_ra_pool[0]

    x0 = np.asarray(ra0.x)
    y0 = np.asarray(ra0.y)

    ra_bad = Ra(
        filename=ra0.filename,
        x=x0 + 1.0,
        y=y0,
        unified_meta=ra0.unified_meta,
        metadata_user=ra0.metadata_user,
    )

    with pytest.raises(ValueError):
        merge_ras([ra0, ra_bad], method="average", validate_x_grid=True)
