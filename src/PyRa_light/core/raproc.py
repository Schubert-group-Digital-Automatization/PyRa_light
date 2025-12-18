"""core.raproc: here lays `RaProc`, `ProcessingInfo` and some smaller helpers & co."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, TypedDict

import numpy as np

from .metadata import MetadataDict, MetadataField
from ..plot_print.plot_print import print_metadata_tables, plot_raman_xy

logger = logging.getLogger(__name__)

# Constants
DEFAULT_X_UNIT = "wavenumber [1/cm]"
DEFAULT_Y_UNIT = "intensity [a.u.]"


# ------------------------------------------------------------------------- #
# ProcessingInfo for RaProc
# ------------------------------------------------------------------------- #

class ProcessingInfo(TypedDict):
    """
    Processing metadata for a :class:`RaProc` merge.

    Keys
    ----
    method : str
        Merge method used (``"average"``, ``"accumulate"``, or ``"median"``).
    n_spectra : int
        Number of spectra merged.
    merge_timestamp : str
        ISO 8601 timestamp of the merge operation.
    source_indices : list[int]
        Indices of source spectra in the original list.
    """
    method: str
    n_spectra: int
    merge_timestamp: str
    source_indices: list[int]

# ------------------------------------------------------------------------- #
# RaProc
# ------------------------------------------------------------------------- #

@dataclass(slots=True)
class RaProc:
    """
    Processed Raman spectrum created by merging multiple :class:`Ra` objects.

    Attributes
    ----------
    filenames : list[str]
        Original filenames that contributed to this spectrum.
    sample_names : list[str]
        Sample names corresponding to each filename.
    merged_sample_name: str
        Merged sample names
    x_processed : np.ndarray
        Common x-axis values (wavenumber or wavelength).
    y_processed : np.ndarray
        Merged intensity values (summed or averaged).
    y_err : np.ndarray or None
        Standard deviation of intensities across spectra (for
        averaging and median merging; ``None`` for accumulation).
    y_sem : np.ndarray or None
        Standard error of the mean (only for averaging; ``None`` for
        accumulation and median merging).
    y_mad : np.ndarray or None
        Median absolute deviation across spectra (only for median
        merging; ``None`` for accumulation and averaging).
    y_q25 : np.ndarray or None
        25th percentile (first quartile) across spectra,
        or ``None`` if not computed.
    y_q75 : np.ndarray or None
        75th percentile (third quartile) across spectra,
        or ``None`` if not computed.
    x_unit : str
        Unit for the x-axis.
    y_unit : str
        Unit for the y-axis.
    unified_meta : MetadataDict
        Merged metadata from all input spectra.
    metadata_user : MetadataDict
        By user injected, unstandardized metadata if `overwrite=False`.
    processing : ProcessingInfo
        Information about the merge process.
    """

    filenames: list[str]
    sample_names: list[str]
    merged_sample_name: str
    x_processed: np.ndarray
    y_processed: np.ndarray
    y_err: Optional[np.ndarray] = None
    y_sem: Optional[np.ndarray] = None
    y_mad: Optional[np.ndarray] = None
    y_q25: Optional[np.ndarray] = None
    y_q75: Optional[np.ndarray] = None
    x_unit: str = DEFAULT_X_UNIT
    y_unit: str = DEFAULT_Y_UNIT
    unified_meta: MetadataDict = field(default_factory=dict)
    metadata_user: dict[str, Any] = field(default_factory=dict)
    processing: ProcessingInfo = field(
        default_factory=lambda: ProcessingInfo(
            method="unknown",
            n_spectra=0,
            merge_timestamp=datetime.now().isoformat(),
            source_indices=[],
        )
    )

    def __post_init__(self) -> None:
        """
        Validate and normalize inputs after initialization.

        Ensures matching lengths of filename and sample lists, compatible
        array shapes, and float64 dtypes for all numeric arrays.
        """

        if len(self.filenames) != len(self.sample_names):
            raise ValueError(
                f"filenames ({len(self.filenames)}) and sample_names "
                f"({len(self.sample_names)}) must have same length"
            )

        # Ensure 1D float64 numpy arrays
        self.x_processed = np.asarray(self.x_processed, dtype=np.float64)
        self.y_processed = np.asarray(self.y_processed, dtype=np.float64)

        if self.x_processed.ndim != 1 or self.y_processed.ndim != 1:
            raise ValueError(
                "x_processed and y_processed must be 1D arrays; got "
                f"ndim={self.x_processed.ndim} and {self.y_processed.ndim}"
            )

        if self.x_processed.shape != self.y_processed.shape:
            raise ValueError(
                "x_processed and y_processed must have same shape: "
                f"{self.x_processed.shape} != {self.y_processed.shape}"
            )

        if self.y_err is not None:
            self.y_err = np.asarray(self.y_err, dtype=np.float64)
            if self.y_err.ndim != 1 or self.y_err.shape != self.y_processed.shape:
                raise ValueError(
                    f"y_err {self.y_err.shape} must match y_processed "
                    f"{self.y_processed.shape}"
                )

        if self.y_sem is not None:
            self.y_sem = np.asarray(self.y_sem, dtype=np.float64)
            if self.y_sem.ndim != 1 or self.y_sem.shape != self.y_processed.shape:
                raise ValueError(
                    f"y_sem {self.y_sem.shape} must match y_processed "
                    f"{self.y_processed.shape}"
                )

        if self.y_mad is not None:
            self.y_mad = np.asarray(self.y_mad, dtype=np.float64)
            if self.y_mad.ndim != 1 or self.y_mad.shape != self.y_processed.shape:
                raise ValueError(
                    f"y_mad {self.y_mad.shape} must match y_processed "
                    f"{self.y_processed.shape}"
                )

        if self.y_q25 is not None:
            self.y_q25 = np.asarray(self.y_q25, dtype=np.float64)
            if self.y_q25.ndim != 1 or self.y_q25.shape != self.y_processed.shape:
                raise ValueError(
                    f"y_q25 {self.y_q25.shape} must match y_processed "
                    f"{self.y_processed.shape}"
                )

        if self.y_q75 is not None:
            self.y_q75 = np.asarray(self.y_q75, dtype=np.float64)
            if self.y_q75.ndim != 1 or self.y_q75.shape != self.y_processed.shape:
                raise ValueError(
                    f"y_q75 {self.y_q75.shape} must match y_processed "
                    f"{self.y_processed.shape}"
                )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this :class:`RaProc` instance to a JSON-compatible dict.

        Returns
        -------
        dict[str, Any]
            Serialized representation of the processed spectrum, with
            arrays converted to lists.
        """
        return {
            "class": "RaProc",
            "filenames": self.filenames,
            "sample_names": self.sample_names,
            "merged_sample_name": self.merged_sample_name,
            "x_processed": self.x_processed.tolist(),
            "y_processed": self.y_processed.tolist(),
            "y_err": None if self.y_err is None else self.y_err.tolist(),
            "y_sem": None if self.y_sem is None else self.y_sem.tolist(),
            "y_mad": None if self.y_mad is None else self.y_mad.tolist(),
            "y_q25": None if self.y_q25 is None else self.y_q25.tolist(),
            "y_q75": None if self.y_q75 is None else self.y_q75.tolist(),
            "x_unit": self.x_unit,
            "y_unit": self.y_unit,
            "unified_meta": self.unified_meta,
            "metadata_user": self.metadata_user,
            "processing": dict(self.processing),
        }

    def inject_meta(self, meta: dict[str, Any], overwrite: bool = False) -> None:
        """
        Inject additional metadata into this processed spectrum.

        Parameters
        ----------
        meta : dict[str, Any]
            Dictionary of metadata entries to inject.
        overwrite : bool, optional
            If False (default), entries are stored in ``metadata_user`` and
            do not change the unified merged metadata. If True, the provided
            keys are written directly into ``unified_meta``, potentially
            overwriting existing values.
        """
        if not isinstance(meta, dict):
            raise TypeError(f"meta must be a dict, got {type(meta)!r}")

        if overwrite:
            for key, value in meta.items():
                self.unified_meta[key] = value
        else:
            self.metadata_user.update(meta)

    def edit_meta(self, overwrite: bool = True, **kwargs: Any) -> None:
        """
        Edit accepted/predefined fields in the unified and merged metadata.

        Only keys corresponding to known :class:`MetadataField` values are
        accepted. Unknown keys raise a KeyError.

        Parameters
        ----------
        overwrite : bool, optional
            If False, an error is raised when attempting to change a field
            that already has a non-None value. If True (default), existing
            values are replaced.
        **kwargs : Any
            Canonical metadata fields to set, e.g.
            ``edit_meta(laser_wavelength_nm=532.0)``.
        """
        allowed_keys = {field.value for field in MetadataField}

        for key, value in kwargs.items():
            if key not in allowed_keys:
                raise KeyError(
                    f"Unknown metadata field '{key}'. "
                    f"Allowed fields are: {sorted(allowed_keys)}"
                )

            if not overwrite and self.unified_meta.get(key) is not None:
                raise ValueError(
                    f"Metadata field '{key}' is already set; "
                    f"use overwrite=True to replace it."
                )

            self.unified_meta[key] = value

    def print_metadata(
        self,
        tablefmt: str = "github",
        return_str: bool = False,
    ) -> str | None:
        """Print key attributes and metadata as a single table.

        Parameters
        ----------
        tablefmt : str, optional
            Tabulate table format. Default is ``"github"``.
        return_str : bool, optional
            If True, return the rendered output instead of printing it.

        Returns
        -------
        str | None
            Rendered output if ``return_str=True``, otherwise None.
        """

        rows: list[list[Any]] = [
            ["method", self.processing.get("method", "unknown"), "processing"],
            ["n_spectra", self.processing.get("n_spectra", len(self.filenames)), "processing"],
            ["merge_timestamp", self.processing.get("merge_timestamp", ""), "processing"],
            ["source_indices", self.processing.get("source_indices", []), "processing"],
            ["filenames", self.filenames, "attribute"],
            ["sample_names", self.sample_names, "attribute"],
            ["merged_sample_name", self.merged_sample_name, "attribute"],
            ["x_unit", self.x_unit, "attribute"],
            ["y_unit", self.y_unit, "attribute"],
        ]

        rows += [[k, v, "unified_meta"] for k, v in self.unified_meta.items()]
        rows += [[k, v, "metadata_user"] for k, v in self.metadata_user.items()]

        return print_metadata_tables(
            tables=[("RaProc metadata", rows)],
            tablefmt=tablefmt,
            return_str=return_str,
        )

    def plot_raman(
        self,
        ax=None,
        show: bool = True,
        title: str | None = None,
        band: str | None = None,
        band_alpha: float = 0.2,
        **plot_kwargs,
    ):
        """Plot the processed Raman spectrum.

        Parameters
        ----------
        ax : matplotlib.axes.Axes | None, optional
            Target axes to draw into. If ``None`` (default), a new figure and axes
            are created.
        show : bool, optional
            If ``True`` (default), display the figure via ``matplotlib.pyplot.show``.
        title : str | None, optional
            Plot title. If ``None`` (default), uses ``merged_sample_name``.
        band : str | None, optional
            Optional filled band to plot on top of the curve. Allowed values are
            ``None`` (default), ``"y_err"``, ``"y_sem"``, ``"y_mad"``, and ``"iqr"``.

            - ``"y_err"``, ``"y_sem"``, ``"y_mad"`` plot ``y_processed ± band``.
            - ``"iqr"`` plots the range ``[y_q25, y_q75]``.
        band_alpha : float, optional
            Alpha value for the filled band. Default is 0.2.
        **plot_kwargs
            Additional keyword arguments forwarded to ``Axes.plot``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure containing the plot.
        ax : matplotlib.axes.Axes
            The axes the spectrum was drawn into.
        """

        if title is None:
            title = self.merged_sample_name

        y_low = None
        y_high = None

        if band is not None:
            if band in {"y_err", "y_sem", "y_mad"}:
                arr = getattr(self, band, None)
                if arr is None:
                    raise ValueError(f"Band '{band}' is not available (value is None).")
                y_low = self.y_processed - arr
                y_high = self.y_processed + arr
            elif band == "iqr":
                if self.y_q25 is None or self.y_q75 is None:
                    raise ValueError("Band 'iqr' requires y_q25 and y_q75 to be present.")
                y_low = self.y_q25
                y_high = self.y_q75
            else:
                raise ValueError(
                    "Unknown band. Use one of None, 'y_err', 'y_sem', 'y_mad', 'iqr'."
                )

        return plot_raman_xy(
            self.x_processed,
            self.y_processed,
            x_label=self.x_unit,
            y_label=self.y_unit,
            title=title,
            ax=ax,
            show=show,
            y_low=y_low,
            y_high=y_high,
            band_alpha=band_alpha,
            **plot_kwargs,
        )
    def __repr__(self) -> str:
        """Return a concise string representation."""
        method = self.processing.get("method", "unknown")
        n_spectra = self.processing.get("n_spectra", len(self.filenames))

        if self.x_processed.size == 0:
            x_range = "n/a"
        else:
            x_range = f"[{self.x_processed.min():.1f}, {self.x_processed.max():.1f}]"

        return (
            f"RaProc(method='{method}', "
            f"n_spectra={n_spectra}, "
            f"points={self.x_processed.size}, "
            f"x_range={x_range})"
        )
