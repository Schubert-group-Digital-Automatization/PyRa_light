"""core.Ra: here lays `Ra`"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import numpy as np

from .metadata import MetadataDict, MetadataField, unify_metadata
from .converted import ConvertedFile, ConvertedMeasurement
from ..plot_print.plot_print import print_metadata_tables, plot_raman_xy

logger = logging.getLogger(__name__)

DEFAULT_X_UNIT = "wavenumber [1/cm]"
DEFAULT_Y_UNIT = "intensity [a.u.]"


@dataclass
class Ra:
    """
    Uniform Raman spectrum container.

    Attributes
    ----------
    x : np.ndarray
        Raman shift values.
    y : np.ndarray
        Intensity values.
    filename : str
        Source file name.
    sample_name : str
        Sample identifier.
    unified_meta : MetadataDict
        Standardized metadata mapping & by user added metadata.
    x_unit : str
        Unit for x-axis.
    y_unit : str
        Unit for y-axis.
    metadata_user : MetadataDict
        By user injected, unstandardized metadata if `overwrite=False`.
    """
    x: np.ndarray
    y: np.ndarray
    filename: str
    sample_name: str = ""
    unified_meta: MetadataDict = field(default_factory=dict)
    x_unit: str = DEFAULT_X_UNIT
    y_unit: str = DEFAULT_Y_UNIT
    metadata_user: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate and normalize Ra data after initialization.

        Ensures that ``x`` and ``y`` are NumPy arrays of matching shape,
        maps and unifies metadata, extracts the sample name if missing,
        and removes the redundant sample name entry from unified metadata.
        """
        # Ensure 1D float64 numpy arrays
        self.x = np.asarray(self.x, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)

        if self.x.ndim != 1 or self.y.ndim != 1:
            raise ValueError(
                f"x and y must be 1D arrays; got ndim={self.x.ndim} and {self.y.ndim}"
            )

        # Validate array shapes
        if self.x.shape != self.y.shape:
            raise ValueError(
                "x and y arrays must have same shape: "
                f"{self.x.shape} != {self.y.shape}"
            )

        if not isinstance(self.unified_meta, dict):
            raise TypeError(
                f"unified_meta must be a dict, got {type(self.unified_meta)!r}"
            )

        self.filename = Path(self.filename).name

        self.unified_meta = unify_metadata(self.unified_meta)

        if not self.sample_name:
            sn = self.unified_meta.get(MetadataField.SAMPLE_NAME.value)
            if sn not in (None, ""):
                self.sample_name = str(sn)

        # Keep sample_name only on the attribute, not duplicated in unified_meta
        self.unified_meta.pop(MetadataField.SAMPLE_NAME.value, None)

    @classmethod
    def from_converted_measurement(
        cls,
        measurement: ConvertedMeasurement,
        filename: str | None = None,
    ) -> Ra:
        """
        Create a :class:`Ra` instance from a single converted measurement.

        Parameters
        ----------
        measurement : ConvertedMeasurement
            Pre-unified measurement object produced by the converter layer.
        filename : str or None, optional
            Filename to associate with this spectrum. If None, the name of
            ``measurement.path`` is used.

        Returns
        -------
        Ra
            Unified Raman spectrum built from the measurement.
        """
        if filename is None:
            filename = measurement.path.name

        return cls(
            x=measurement.x,
            y=measurement.y,
            filename=filename,
            unified_meta=measurement.raw_metadata,
            x_unit=measurement.x_label or DEFAULT_X_UNIT,
            y_unit=measurement.y_label or DEFAULT_Y_UNIT,
        )

    @classmethod
    def from_converted_file(cls, converted: ConvertedFile) -> list[Ra]:
        """
        Create :class:`Ra` instances from a :class:`ConvertedFile`.

        Parameters
        ----------
        converted : ConvertedFile
            Converted representation of a Raman file containing one or more
            measurements.

        Returns
        -------
        list[Ra]
            List of :class:`Ra` objects, one per measurement in the file.
        """
        ra_list: list[Ra] = []
        for m in converted.measurements:
            ra_list.append(
                cls.from_converted_measurement(
                    m,
                    filename=converted.filename,
                )
            )
        return ra_list

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this :class:`Ra` instance to a plain dictionary.

        Returns
        -------
        dict[str, Any]
            JSON-compatible representation of the spectrum and its
            metadata, including numeric arrays converted to lists.
        """
        return {
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "filename": self.filename,
            "sample_name": self.sample_name,
            "unified_meta": self.unified_meta,
            "x_unit": self.x_unit,
            "y_unit": self.y_unit,
            "metadata_user": self.metadata_user,
        }

    def inject_meta(self, meta: dict[str, Any], overwrite: bool = False) -> None:
        """
        Inject additional metadata into (user or unified) metadata.

        Parameters
        ----------
        meta : dict[str, Any]
            Dictionary of metadata entries to inject.
        overwrite : bool, optional
            If False (default), entries are stored in ``metadata_user`` and
            do not change the unified metadata. If True, the provided keys
            are written directly into ``unified_meta``, potentially
            overwriting existing values.
        """
        if not isinstance(meta, dict):
            raise TypeError(f"meta must be a dict, got {type(meta)!r}")

        if overwrite:
            self.unified_meta.update(meta)
        else:
            self.metadata_user.update(meta)

    def edit_meta(self, overwrite: bool = True, **kwargs: Any) -> None:
        """
        Edit predefined/accepted, unified metadata fields in a controlled way.

        Only keys corresponding to known :class:`MetadataField` values are
        accepted. Unknown keys raise a KeyError.

        Parameters
        ----------
        overwrite : bool, optional
            If False, an error is raised when attempting to change a field
            that already has a non-None value. If True (default), existing
            values are replaced.
        **kwargs : Any
            predefined/accepted metadata fields to set, e.g.
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
        """
        Print key attributes and metadata as a single table.

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

        rows = [
            ["filename", self.filename, "attribute"],
            ["sample_name", self.sample_name, "attribute"],
        ]
        rows += [[k, v, "unified_meta"] for k, v in self.unified_meta.items()]
        rows += [[k, v, "metadata_user"] for k, v in self.metadata_user.items()]

        return print_metadata_tables(
            tables=[("Ra metadata", rows)],
            tablefmt=tablefmt,
            return_str=return_str,
        )

    def plot_raman(
        self,
        ax=None,
        show: bool = True,
        title: str | None = None,
        **plot_kwargs,
    ):
        """Plot the Raman spectrum.

        This is a convenience wrapper around :func:`PyRa_light.plot_print.plot_print.plot_raman_xy`.
        It plots ``self.x`` versus ``self.y`` and applies the stored axis units.

        Parameters
        ----------
        ax : matplotlib.axes.Axes | None, optional
            Target axes to draw into. If ``None`` (default), a new figure and axes
            are created.
        show : bool, optional
            If ``True`` (default), display the figure via ``matplotlib.pyplot.show``.
        title : str | None, optional
            Plot title. If ``None`` (default), uses ``sample_name`` if available,
            otherwise ``filename``.
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
            title = self.sample_name or self.filename

        return plot_raman_xy(
            self.x,
            self.y,
            x_label=self.x_unit,
            y_label=self.y_unit,
            title=title,
            ax=ax,
            show=show,
            **plot_kwargs,
        )

    def __repr__(self) -> str:
        """
        Return a concise string representation of the spectrum.

        Returns
        -------
        str
            Human-readable representation including filename, sample
            name, number of points and x-range.
        """
        if self.x.size == 0:
            x_range = "n/a"
        else:
            x_range = f"[{self.x.min():.1f}, {self.x.max():.1f}]"

        return (
            f"Ra(filename='{self.filename}', "
            f"sample_name='{self.sample_name}', "
            f"points={self.x.size}, "
            f"x_range={x_range})"
        )
