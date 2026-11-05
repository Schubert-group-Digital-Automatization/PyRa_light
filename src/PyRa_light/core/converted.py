"""core.converted: here lives `ConvertedFile`, `ConvertedMeasurements`"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..plot_print import plot_raman_xy, print_metadata_tables
from ..plot_print.availability import (
    require_plot_raman_xy,
    require_print_metadata_tables,
)


@dataclass
class ConvertedFile:
    """
    Standardized representation of a converted Raman file.

    This groups one or more :class:`ConvertedMeasurement` instances that all
    originate from the same physical file, along with basic file-level
    metadata and the file/global-dict if available.

    Parameters
    ----------
    path : Path
        Path to the original source file.
    fileformat : str
        Short identifier for the file format (e.g. "spc", "witec").
    filename : str
        File name (usually ``path.name``) stored in the converter output.
    measurements : list[ConvertedMeasurement]
        List of all measurements extracted from the file.
    raw : dict[str, Any], optional
        Optional copy of the original converter dict as returned by the
        underlying converter implementation.
    """

    path: Path
    fileformat: str
    filename: str
    measurements: list[ConvertedMeasurement]
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize file attributes after initialization.
        """
        self.path = Path(self.path)
        if self.filename:
            self.filename = Path(self.filename).name
        else:
            self.filename = self.path.name

    def to_dict(self) -> dict[str, Any]:
        """Convert this object into a JSON-serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation suitable for JSON export.
        """
        return {
            "path": str(self.path),
            "fileformat": self.fileformat,
            "filename": self.filename,
            "measurements": [m.to_dict() for m in self.measurements],
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: Path) -> ConvertedFile:
        """
        Construct a ConvertedFile from a legacy converter dict.

        Parameters
        ----------
        data : dict[str, Any]
            Standardized converter output as currently returned by
            ``BaseConverter.convert``.
        path : Path
            Path to the original source file.

        Returns
        -------
        ConvertedFile
            Parsed file-level representation with measurements.
        """
        fileformat = data.get("Fileformat", "")
        filename = data.get("Filename", path.name)

        measurements_data = data.get("Measurements", {})
        measurements: list[ConvertedMeasurement] = []

        for key, entry in measurements_data.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                idx = 0

            raman = entry.get("Raman", {})
            metadata = entry.get("Metadata", {})
            other = entry.get("Other_spectra", {})

            measurements.append(
                ConvertedMeasurement(
                    path=path,
                    index=idx,
                    fileformat=fileformat,
                    x=raman.get("x", []),
                    y=raman.get("y", []),
                    x_label=raman.get("x_label", ""),
                    raw_metadata=metadata.get("raw_metadata", {}),
                    other_spectra=other,
                )
            )

        return cls(
            path=path,
            fileformat=fileformat,
            filename=filename,
            measurements=measurements,
            raw=data,
        )

    @property
    def n_measurements(self) -> int:
        """
        Number of measurements contained in this file.

        Returns
        -------
        int
            Length of the ``measurements`` list.
        """
        return len(self.measurements)

    def print_metadata(
        self,
        tablefmt: str = "github",
        return_str: bool = False,
    ) -> str | None:
        """Print file info and raw metadata tables.

        This prints:
        - one table with file-level attributes,
        - one table with the file-level raw converter dict,
        - one table per measurement containing that measurement's raw metadata.

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
        tables: list[tuple[str, list[list[Any]]]] = []

        info_rows: list[list[Any]] = [
            ["fileformat", self.fileformat, "attribute"],
            ["filename", self.filename, "attribute"],
            ["n_measurements", self.n_measurements, "attribute"],
        ]
        tables.append(("ConvertedFile info", info_rows))

        if self.raw:
            raw_rows = [[k, v, "raw_file_metadata"] for k, v in self.raw.items()]
            tables.append(("ConvertedFile raw", raw_rows))

        for m in self.measurements:
            rows: list[list[Any]] = [
                ["name", m.path.name, "measurement"],
                ["fileformat", m.fileformat, "measurement"],
                ["index", m.index, "measurement"],
            ]
            rows += [[k, v, "raw_measurement_metadata"] for k, v in m.raw_metadata.items()]
            tables.append((f"ConvertedMeasurement[{m.index}] metadata", rows))

        printer = require_print_metadata_tables(print_metadata_tables)
        return printer(tables=tables, tablefmt=tablefmt, return_str=return_str)

    def plot_raman(
        self,
        index: int | None = None,
        ax=None,
        show: bool = True,
        title: str | None = None,
        **plot_kwargs,
    ):
        """Plot one or all measurements from this file.

        Parameters
        ----------
        index : int | None, optional
            Measurement index to plot (matches ``ConvertedMeasurement.index``).
            If ``None`` (default), all measurements are overlaid.
        ax : matplotlib.axes.Axes | None, optional
            Target axes to draw into. If ``None`` (default), a new figure and axes
            are created.
        show : bool, optional
            If ``True`` (default), show the figure via ``matplotlib.pyplot.show``.
        title : str | None, optional
            Plot title. If ``None`` (default), uses ``filename``.
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
            title = self.filename

        if not self.measurements:
            raise ValueError("No measurements to plot.")

        if index is None:
            measurements = self.measurements
        else:
            matches = [m for m in self.measurements if m.index == index]
            if not matches:
                raise ValueError(f"No measurement with index={index} in this file.")
            measurements = [matches[0]]

        fig = None
        for i, m in enumerate(measurements):
            this_title = title if i == 0 else ""
            this_show = show if i == (len(measurements) - 1) else False

            x_label = m.x_label or "wavenumber [1/cm]"
            y_label = m.y_label or "intensity [a.u.]"

            plotter = require_plot_raman_xy(plot_raman_xy)
            fig, ax = plotter(
                m.x,
                m.y,
                x_label=x_label,
                y_label=y_label,
                title=this_title,
                ax=ax,
                show=this_show,
                **plot_kwargs,
            )

        return fig, ax


# ------------------------------------------------------------------------- #
# ConvertedMeasurement
# ------------------------------------------------------------------------- #


@dataclass
class ConvertedMeasurement:
    """
    One Raman measurement extracted from a converted file.

    This represents the "pre-unified" spectrum: raw x/y data plus the
    metadata that came from the instrument / converter, without any
    Ra-level unification applied yet.

    Parameters
    ----------
    path : Path
        Path to the original source file this measurement came from.
    index : int
        Index of this measurement within the file (e.g. 1, 2, 3, ...).
    fileformat : str
        Short identifier for the file format (e.g. "spc", "witec").
    x : array-like of float
        Abscissa values (e.g. Raman shift in cm^-1).
    y : array-like of float
        Ordinate values (e.g. intensity).
    x_label : str
        Label for the x-axis (e.g. "Raman shift / cm^-1").
    y_label : str or None, optional
        Label for the y-axis, if available.
    raw_metadata : dict[str, Any], optional
        Raw per-measurement metadata as provided by the converter.
        Typically this is the content of the "raw_metadata" block.
    other_spectra : dict[str, Any], optional
        Additional spectra associated with this measurement, taken
        verbatim from the converter output (e.g. background, white
        reference, fluorescence channels).
    """

    path: Path
    index: int
    fileformat: str

    x: np.ndarray
    y: np.ndarray
    x_label: str
    y_label: str | None = None

    raw_metadata: dict[str, Any] = field(default_factory=dict)
    other_spectra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Normalize x/y data and perform basic consistency checks.
        """
        if not isinstance(self.x, np.ndarray):
            self.x = np.asarray(self.x, dtype=np.float64)
        else:
            self.x = self.x.astype(np.float64, copy=False)

        if not isinstance(self.y, np.ndarray):
            self.y = np.asarray(self.y, dtype=np.float64)
        else:
            self.y = self.y.astype(np.float64, copy=False)

        if self.x.shape != self.y.shape:
            raise ValueError(
                f"x and y must have the same shape: {self.x.shape} != {self.y.shape}"
            )
        if self.x.ndim != 1:
            raise ValueError(
                f"x and y must be 1D arrays; got ndim={self.x.ndim}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert this object into a JSON-serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation suitable for JSON export.
        """
        return {
            "path": str(self.path),
            "index": self.index,
            "fileformat": self.fileformat,
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "x_label": self.x_label,
            "y_label": self.y_label,
            "raw_metadata": self.raw_metadata,
            "other_spectra": self.other_spectra,
        }

    @property
    def n_points(self) -> int:
        """
        Number of data points in this measurement.

        Returns
        -------
        int
            Length of the x/y arrays.
        """
        return int(self.x.size)

    def inject_meta(self, meta: dict[str, Any], overwrite: bool = False) -> None:
        """
        Inject additional key-value pairs into this measurement's raw metadata.
        Make sure you know what you do!

        Parameters
        ----------
        meta : dict[str, Any]
            Dictionary of metadata entries to inject.
        overwrite : bool, optional
            If False (default), existing keys in ``raw_metadata`` are left
            unchanged and only new keys are added. If True, existing keys
            in ``raw_metadata`` are overwritten with the provided values.
        """
        if not isinstance(meta, dict):
            raise TypeError(f"meta must be a dict, got {type(meta)!r}")

        if overwrite:
            self.raw_metadata.update(meta)
        else:
            for key, value in meta.items():
                if key not in self.raw_metadata:
                    self.raw_metadata[key] = value

    def print_metadata(
        self,
        tablefmt: str = "github",
        return_str: bool = False,
    ) -> str | None:
        """Print measurement info and raw metadata as a single table.

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
            ["name", self.path.name, "attribute"],
            ["fileformat", self.fileformat, "attribute"],
            ["index", self.index, "attribute"],
        ]
        rows += [[k, v, "raw_measurement_metadata"] for k, v in self.raw_metadata.items()]

        printer = require_print_metadata_tables(print_metadata_tables)
        return printer(
            tables=[("ConvertedMeasurement metadata", rows)],
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
        """Plot this measurement as a Raman spectrum.

        Parameters
        ----------
        ax : matplotlib.axes.Axes | None, optional
            Target axes to draw into. If ``None`` (default), a new figure and axes
            are created.
        show : bool, optional
            If ``True`` (default), show the figure via ``matplotlib.pyplot.show``.
        title : str | None, optional
            Plot title. If ``None`` (default), uses ``"<filename> [#<index>]"``.
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
            title = f"{self.path.name} [#{self.index}]"

        x_label = self.x_label or "wavenumber [1/cm]"
        y_label = self.y_label or "intensity [a.u.]"

        plotter = require_plot_raman_xy(plot_raman_xy)
        return plotter(
            self.x,
            self.y,
            x_label=x_label,
            y_label=y_label,
            title=title,
            ax=ax,
            show=show,
            **plot_kwargs,
        )
