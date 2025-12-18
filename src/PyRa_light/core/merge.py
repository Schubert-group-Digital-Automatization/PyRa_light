"""core.merge: Here lives the merge logic"""
from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Sequence
from datetime import datetime
from pathlib import Path
import logging
import os
import re

import numpy as np

from .ra import Ra
from .raproc import RaProc, ProcessingInfo

from .metadata import MetadataDict

logger = logging.getLogger(__name__)
_DELIMS = re.compile(r"[_\s-]")

# ------------------------------------------------------------------------- #
# MergeMethod
# ------------------------------------------------------------------------- #

class MergeMethod(Enum):
    """
    Methods for merging spectra.

    Attributes
    ----------
    AVERAGE : MergeMethod
        Compute the mean across spectra.
    ACCUMULATE : MergeMethod
        Sum spectra values.
    MEDIAN : MergeMethod
        Take the median across spectra.
    """
    AVERAGE = "average"
    ACCUMULATE = "accumulate"
    MEDIAN = "median"

    @classmethod
    def from_string(cls, method: str) -> MergeMethod:
        """
        Parse a string into a MergeMethod enum.

        Parameters
        ----------
        method : str
            String representation of the merge method.

        Returns
        -------
        MergeMethod
            Corresponding enum member.

        Raises
        ------
        ValueError
            If the input is not a valid method name.
        """
        method = method.strip().lower()
        try:
            return cls(method)
        except ValueError:
            valid = ", ".join(m.value for m in cls)
            raise ValueError(
                f"Invalid merge method '{method}'. Must be one of: {valid}"
            ) from None

# ------------------------------------------------------------------------- #
# MetadataMergeStrategy
# ------------------------------------------------------------------------- #

class MetadataMergeStrategy(Enum):
    """
    Strategies for combining metadata fields.

    Attributes
    ----------
    KEEP_FIRST : MetadataMergeStrategy
        Use the value from the first spectrum.
    KEEP_COMMON : MetadataMergeStrategy
        Keep the value only if all spectra share the same one.
    COLLECT_ALL : MetadataMergeStrategy
        Concatenate unique values into a combined representation.
    AVERAGE_NUMERIC : MetadataMergeStrategy
        Average numeric values where possible.
    """
    KEEP_FIRST = auto()       # Use value from first spectrum
    KEEP_COMMON = auto()      # Only keep if all values identical
    COLLECT_ALL = auto()      # Collect all unique values
    AVERAGE_NUMERIC = auto()  # Average numeric values

METADATA_MERGE_RULES = {
    # numeric values
    "excitation_wavelength_nm": MetadataMergeStrategy.AVERAGE_NUMERIC,
    "integration_time_s": MetadataMergeStrategy.AVERAGE_NUMERIC,
    "laser_power_mw": MetadataMergeStrategy.AVERAGE_NUMERIC,    
    "environment_temperature_K": MetadataMergeStrategy.AVERAGE_NUMERIC,
    "detector_temperature_K": MetadataMergeStrategy.AVERAGE_NUMERIC,
    "spectral_resolution_cm_1": MetadataMergeStrategy.AVERAGE_NUMERIC,
    "accumulations": MetadataMergeStrategy.AVERAGE_NUMERIC,
    "environment_pressure_kPa": MetadataMergeStrategy.AVERAGE_NUMERIC,
    "objective_magnification": MetadataMergeStrategy.AVERAGE_NUMERIC,
    "objective_aperture": MetadataMergeStrategy.AVERAGE_NUMERIC,
    "spot_size_um": MetadataMergeStrategy.AVERAGE_NUMERIC,
    "confocal_aperture_um": MetadataMergeStrategy.AVERAGE_NUMERIC,
    "spectral_center_rel_1_per_cm": MetadataMergeStrategy.AVERAGE_NUMERIC,

    # instrument config
    "instrument_manufacturer": MetadataMergeStrategy.KEEP_COMMON,
    "instrument_model": MetadataMergeStrategy.KEEP_COMMON,
    "microscope_type": MetadataMergeStrategy.KEEP_COMMON,
    "detector_type": MetadataMergeStrategy.KEEP_COMMON,
    "laser_type": MetadataMergeStrategy.KEEP_COMMON,
    "grating": MetadataMergeStrategy.KEEP_COMMON,
    "objective_name": MetadataMergeStrategy.KEEP_COMMON,
    "sample_type": MetadataMergeStrategy.KEEP_COMMON,
    "scattering_configuration_porto": MetadataMergeStrategy.KEEP_COMMON,
    "polarization_state": MetadataMergeStrategy.KEEP_COMMON,
    "environment_atmosphere": MetadataMergeStrategy.KEEP_COMMON,
    "objective_type": MetadataMergeStrategy.KEEP_COMMON,
    "raman_shift_min_cm_1": MetadataMergeStrategy.KEEP_COMMON,
    "raman_shift_max_cm_1": MetadataMergeStrategy.KEEP_COMMON,

    # datetimes
    "measurement_datetime": MetadataMergeStrategy.COLLECT_ALL,

    # processing
    "background_correction_method": MetadataMergeStrategy.COLLECT_ALL,
    "baseline_correction_method": MetadataMergeStrategy.COLLECT_ALL,
    "normalization_method": MetadataMergeStrategy.COLLECT_ALL,
    "smoothing_method": MetadataMergeStrategy.COLLECT_ALL,
    "raman_scattering_correction_method": MetadataMergeStrategy.COLLECT_ALL,
    "cosmic_ray_removal_method": MetadataMergeStrategy.COLLECT_ALL,
}

def merge_ras(
    ras: Sequence[Ra],
    method: str | MergeMethod = MergeMethod.AVERAGE,
    validate_x_grid: bool = True,
) -> RaProc:
    """
    Merge a list of :class:`Ra` objects into a single :class:`RaProc`.

    Parameters
    ----------
    ras : Sequence[Ra]
        List of spectra to merge (at least two).
    method : str or MergeMethod, optional
        Merge method (``"average"``, ``"accumulate"``, or ``"median"``).
        By default :class:`MergeMethod.AVERAGE`.
    validate_x_grid : bool, optional
        If ``True``, check that all spectra share the same x grid.
        Default is ``True``.

    Returns
    -------
    RaProc
        Merged spectrum instance.

    Raises
    ------
    ValueError
        If fewer than two spectra are provided, the x grids do not
        match when validation is enabled, or an invalid method string
        is passed.
    """
    if len(ras) < 2:
        raise ValueError(f"Need at least 2 spectra to merge, got {len(ras)}")

    if isinstance(method, str):
        method = MergeMethod.from_string(method)

    x_ref = ras[0].x
    if validate_x_grid:
        for i, ra in enumerate(ras[1:], 1):
            if not np.array_equal(ra.x, x_ref):
                raise ValueError(
                    f"Spectrum {i} has different x grid than reference. "
                    "All spectra must share the same x axis."
                )

    (
        y_processed,
        y_err,
        y_sem,
        y_mad,
        y_q25,
        y_q75,
    ) = _compute_merged_values(ras, method)

    unified_meta = _merge_metadata(ras)

    filenames = [str(Path(ra.filename).name) for ra in ras]
    sample_names = [ra.sample_name for ra in ras]
    merged_sample_name = _common_prefix_to_boundary(sample_names)

    unified_meta["sample_name"] = merged_sample_name

    x_unit = ras[0].x_unit
    y_unit = ras[0].y_unit

    processing = ProcessingInfo(
        method=method.value,
        n_spectra=len(ras),
        merge_timestamp=datetime.now().isoformat(),
        source_indices=list(range(len(ras))),
    )

    logger.info(
        "Merged %d spectra using %s method",
        len(ras),
        method.value,
    )

    return RaProc(
        filenames=filenames,
        sample_names=sample_names,
        merged_sample_name=merged_sample_name,
        x_processed=x_ref.copy(),
        y_processed=y_processed,
        y_err=y_err,
        y_sem=y_sem,
        y_mad=y_mad,
        y_q25=y_q25,
        y_q75=y_q75,
        x_unit=x_unit,
        y_unit=y_unit,
        unified_meta=unified_meta,
        processing=processing,
    )


def _compute_merged_values(
    ras: Sequence[Ra],
    method: MergeMethod,
) -> tuple[
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """
    Compute merged intensity and error arrays from a set of spectra.

    Parameters
    ----------
    ras : Sequence[Ra]
        Spectra to merge.
    method : MergeMethod
        Merge method to use (average, accumulate, or median).

    Returns
    -------
        Tuple of processed intensities and additional
        error/dispersion arrays: standard deviation
        (``y_err``), standard error of the mean (``y_sem``),
        median absolute deviation (``y_mad``), and 25th/75th
        percentile bands (``y_q25``, ``y_q75``). For accumulation,
        all optional arrays are ``None``.
    """
    y_stack = np.stack([ra.y for ra in ras], axis=0)
    n = y_stack.shape[0]

    y_err: Optional[np.ndarray] = None
    y_sem: Optional[np.ndarray] = None
    y_mad: Optional[np.ndarray] = None
    y_q25: Optional[np.ndarray] = None
    y_q75: Optional[np.ndarray] = None

    if method == MergeMethod.ACCUMULATE:
        y_processed = y_stack.sum(axis=0)
        logger.debug("Accumulated %d spectra", n)
    elif method == MergeMethod.MEDIAN:
        y_processed = np.median(y_stack, axis=0)
        # STD
        y_err = y_stack.std(axis=0, ddof=1) if n > 1 else None
        # MAD (only for median)
        if n > 1:
            y_mad = np.median(np.abs(y_stack - y_processed), axis=0)
        # Quartile bands
        y_q25 = np.quantile(y_stack, 0.25, axis=0)
        y_q75 = np.quantile(y_stack, 0.75, axis=0)
        logger.debug("Computed median across %d spectra", n)
    else:
        # Default to average
        y_processed = y_stack.mean(axis=0)
        # STD
        y_err = y_stack.std(axis=0, ddof=1) if n > 1 else None
        # SEM (only for averaging)
        if y_err is not None and n > 1:
            y_sem = y_err / np.sqrt(float(n))
        # Quartile bands
        y_q25 = np.quantile(y_stack, 0.25, axis=0)
        y_q75 = np.quantile(y_stack, 0.75, axis=0)
        logger.debug("Averaged %d spectra", n)

    y_processed = y_processed.astype(np.float64)

    return y_processed, y_err, y_sem, y_mad, y_q25, y_q75


def _merge_metadata(ras: Sequence[Ra]) -> MetadataDict:
    """
    Merge metadata from multiple :class:`Ra` objects.

    Parameters
    ----------
    ras : Sequence[Ra]
        Spectra whose unified metadata should be combined.

    Returns
    -------
    MetadataDict
        Merged metadata mapping according to the merge rules.
    """
    if not ras:
        return {}

    merged: MetadataDict = {}
    all_keys = set().union(*(ra.unified_meta.keys() for ra in ras))

    for key in all_keys:
        values = [ra.unified_meta.get(key) for ra in ras]
        values = [v for v in values if v is not None]

        if not values:
            continue

        strategy = METADATA_MERGE_RULES.get(key, MetadataMergeStrategy.KEEP_COMMON)

        if strategy == MetadataMergeStrategy.KEEP_FIRST:
            merged[key] = values[0]

        elif strategy == MetadataMergeStrategy.KEEP_COMMON:
            if all(v == values[0] for v in values):
                merged[key] = values[0]

        elif strategy == MetadataMergeStrategy.COLLECT_ALL:
            unique_values = list(dict.fromkeys(map(str, values)))
            if len(unique_values) == 1:
                merged[key] = values[0]
            else:
                merged[key] = "; ".join(unique_values)

        elif strategy == MetadataMergeStrategy.AVERAGE_NUMERIC:
            try:
                numeric_values = [float(v) for v in values if v is not None]
                if numeric_values:
                    merged[key] = float(np.mean(numeric_values))
            except (TypeError, ValueError):
                if len(set(map(str, values))) == 1:
                    merged[key] = values[0]

    return merged


def _common_prefix_to_boundary(strings: Sequence[object]) -> str:
    """Return the common prefix, trimmed back to the last delimiter boundary.

    Notes
    -----
    This uses :func:`os.path.commonprefix`, which is character-based.
    The resulting prefix is then cut back to the last occurrence of a delimiter
    (underscore, whitespace, or dash) so it does not end mid-token.

    Parameters
    ----------
    strings : Sequence[object]
        Input values. Non-string values are converted to strings. ``None`` and
        empty/whitespace-only values are ignored.

    Returns
    -------
    str
        Common prefix trimmed to the last delimiter boundary. If no delimiter is
        present in the common prefix, the full common prefix is returned.
    """
    striped = [str(s).strip() for s in strings if s is not None and str(s).strip()]
    if not striped:
        return ""

    cp = os.path.commonprefix(striped)

    last = max((m.start() for m in _DELIMS.finditer(cp)), default=-1)
    return cp[:last] if last != -1 else cp
