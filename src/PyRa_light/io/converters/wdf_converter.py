"""
core.wdf_converter

Converter for Renishaw WiRE .wdf Raman spectroscopy files.

Defines WdfConverter, which reads .wdf files using the py-wdf-reader
package, extracts spectral data, mapping coordinates, and metadata into
a standardized dictionary format.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from renishawWiRE import WDFReader
from renishawWiRE.utils import convert_attr_name

from ..converters_base import BaseConverter


# Constants for excluded attributes
EXCLUDED_ATTRS = frozenset(
    ['file_obj', 'spectra', 'xdata']
)


class WdfConverter(BaseConverter):
    """
    Converter for Renishaw WiRE ``.wdf`` Raman spectroscopy files.

    This converter wraps the :class:`renishawWiRE.WDFReader` API and transforms
    the spectra and metadata into the internal PyRa dictionary format. All
    spectra contained in the file are exposed as individual measurements.
    """

    @property
    def format_name(self) -> str:
        """Name of this converter format."""
        return "WDF"

    @property
    def supported_extensions(self) -> list[str]:
        """File extensions this converter can read."""
        return [".wdf"]

    def convert(self, file_path: Path) -> dict[str, Any]:
        """
        Read a Renishaw WiRE ``.wdf`` file and extract spectra and metadata.

        The file is parsed via :class:`renishawWiRE.WDFReader`. All spectra
        (including mapping data, which are flattened into individual spectra)
        are collected and returned as separate measurements in the PyRa
        internal format.

        Parameters
        ----------
        file_path : Path
            Path to the Renishaw WiRE ``.wdf`` file to be converted.

        Returns
        -------
        dict[str, Any]
            Standardized output dictionary following the PyRa internal format.
            The top-level keys include at least:

            - ``"Measurements"``: mapping from measurement IDs (as strings)
              to measurement data dictionaries. Each measurement contains:
                * ``"Raman"``: x/y spectral data and x-axis label.
                * ``"Metadata"``: a ``"raw_metadata"`` sub-dictionary with
                  attributes extracted from :class:`WDFReader`.
                * ``"Other_spectra"``: reserved for additional spectral
                  content (empty for this converter).
            - ``"Fileformat"``: name of the file format (``"WDF"``).
            - ``"Filename"``: base name of the original ``.wdf`` file.

        Raises
        ------
        Error
            If the file cannot be opened or read by :class:`WDFReader`.
        ValueError
            If no spectral data is present, or if the spectra have an
            unsupported dimensionality, or if mandatory axis/unit
            information is missing.
        """
        self.logger.debug("Converting WDF file: %s", file_path)
        path = Path(file_path)

        # ------------------------------------------------------------------------- #
        # Read File
        # ------------------------------------------------------------------------- #

        try:
            reader = WDFReader(str(path))
        except IOError as e:
            self.logger.error(
                "Failed to read WDF file %s: %s",
                path.name, e, exc_info=True,
            )
            raise


        # ------------------------------------------------------------------------- #
        # Extracting metadata
        # ------------------------------------------------------------------------- #

        raw_metadata = self._extract_metadata(reader)
        self.logger.debug("Extracted %d metadata fields", len(raw_metadata))

        # ------------------------------------------------------------------------- #
        # Extracting metadata
        # ------------------------------------------------------------------------- #

        x_vals = self._extract_x_data(reader)

        unit = getattr(reader, "xlist_unit", None) or getattr(
            reader, "spectral_unit", None
        )

        if unit is None:
            raise ValueError("WDF file has no spectral unit information.")
        x_label = unit.name

        spectra = np.asarray(reader.spectra)

        if spectra.ndim == 1:
            spectra_2d = spectra.reshape(1, -1)
        elif spectra.ndim == 2:
            spectra_2d = spectra
        elif spectra.ndim == 3:
            h, w, pps = spectra.shape
            spectra_2d = spectra.reshape(h * w, pps)
        else:
            raise ValueError(
                f"Unexpected spectra dimensions for file {path.name}: "
                f"{spectra.shape!r}"
            )

        count = spectra_2d.shape[0]

        if count == 0:
            raise ValueError(f"No spectral data found in {path.name}")

        self.logger.debug("Extracting %d spectra from %s", count, path.name)

        # ------------------------------------------------------------------------- #
        # Build output dictionary
        # ------------------------------------------------------------------------- #

        measurements = {}
        for idx in range(count):
            self.logger.debug("Processing spectrum %d/%d", idx + 1, count)

            y_vals = spectra_2d[idx]
            y_list = y_vals.tolist()

            measurements[str(idx + 1)] = {
                "Raman": {
                    "x": x_vals,
                    "y": y_list,
                    "x_label": x_label,
                },
                "Metadata": {
                    "raw_metadata": raw_metadata,
                },
                "Other_spectra": {},
            }

        output = self._create_output(measurements, path.name)
        self.logger.info(
            "Converted %s with %d spectra",
            path.name,
            len(measurements),
        )

        return output

    # ------------------------------------------------------------------------- #
    # Private Helpers
    # ------------------------------------------------------------------------- #


    def _extract_metadata(self, reader: WDFReader) -> dict[str, Any]:
        """
        Extract metadata from a :class:`WDFReader` instance.

        All non-private, non-excluded, non-callable attributes of the reader
        are inspected and converted into a plain-Python dictionary. Enum-like
        values are stored via their ``.name`` attribute, NumPy arrays are
        converted to lists, and all other values are stored as-is.

        Parameters
        ----------
        reader : WDFReader
            The WDF reader instance from which to extract metadata.

        Returns
        -------
        dict[str, Any]
            Dictionary of extracted metadata suitable for JSON-serializable
            storage under the ``"raw_metadata"`` key.
        """
        raw_metadata = {}

        for attr_name in dir(reader):
            # Skip private and excluded attributes
            if attr_name.startswith("_") or attr_name in EXCLUDED_ATTRS:
                continue

            try:
                val = getattr(reader, attr_name)
            except AttributeError as e:
                self.logger.debug("Skipping attribute %s: %s", attr_name, e)
                continue

            # Skip callable attributes
            if callable(val):
                continue

            key = convert_attr_name(attr_name)

            if hasattr(val, "name"):
                raw_metadata[key] = val.name
            elif isinstance(val, np.ndarray):
                raw_metadata[key] = val.tolist()
            else:
                raw_metadata[key] = val

        return raw_metadata

    def _extract_x_data(self, reader: WDFReader) -> list[float]:
        """
        Extract x-axis data from a :class:`WDFReader` instance.

        Parameters
        ----------
        reader : WDFReader
            The WDF reader instance providing the ``xdata`` attribute.

        Returns
        -------
        list[float]
            X-axis values as a list of floats.

        Raises
        ------
        ValueError
            If no x-axis data is available in the WDF file.
        """
        x_data = getattr(reader, "xdata", None)

        if x_data is None:
            raise ValueError("WDF file has no xdata defined.")

        try:
            return x_data.tolist()
        except AttributeError:
            return list(x_data)
