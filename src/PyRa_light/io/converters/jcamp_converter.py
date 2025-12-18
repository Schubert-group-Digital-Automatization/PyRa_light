"""
Converters for JCAMP-DX (.jdx/.dx) spectral files.

This module defines :class:`JcampConverter` for reading JCAMP-DX files, 
extracting spectral data and associated metadata, 
and returning a standardized dictionary in the internal PyRa
format.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jcamp import jcamp_readfile

from ..converters_base import BaseConverter


class JcampConverter(BaseConverter):
    """
    Converter for JCAMP-DX (.jdx/.dx) spectral files.

    The converter reads JCAMP-DX files using :func:`jcamp.jcamp_readfile`,
    parses x and y arrays and header metadata, and returns a standardized
    dictionary in the PyRa internal format. Compound JCAMP "LINK" files are
    handled by exposing each child spectrum in ``children`` as its own
    measurement entry.
    """

    @property
    def format_name(self) -> str:
        """Name of this converter format."""
        return "JCAMP"

    @property
    def supported_extensions(self) -> list[str]:
        """File extensions this converter can read."""
        return [".jdx", ".dx"]

    def convert(self, file_path: Path) -> dict[str, Any]:
        """
        Read a JCAMP-DX file and extract spectra and metadata.

        Parameters
        ----------
        file_path : Path
            Path to the JCAMP-DX file (``.jdx`` or ``.dx``) to be converted.

        Returns
        -------
        dict[str, Any]
            Standardized output dictionary following the PyRa internal
            format. The top-level keys include at least:

            - ``"Measurements"``: mapping from measurement IDs to
              measurement data. For compound JCAMP "LINK" files, each
              child block becomes one measurement.
            - ``"Fileformat"``: name of the file format
              (``"JCAMP"`` for this converter).
            - ``"Filename"``: base name of the original JCAMP file.

        Raises
        ------
        Exception
            Any exception raised by :func:`jcamp.jcamp_readfile` when the
            file cannot be read or the JCAMP content is invalid.
        ValueError
            If no valid spectral data blocks could be identified in the
            file (for example, when no ``x``/``y`` data are present).
        """
        # ------------------------------------------------------------------------- #
        # Read File
        # ------------------------------------------------------------------------- #

        self.logger.debug("Converting JCAMP file: %s", file_path)
        path = Path(file_path)

        try:
            jcamp_data = jcamp_readfile(str(path))
        except Exception as e:
            self.logger.error(
                "Failed to read JCAMP file %s: %s",
                file_path,
                e,
                exc_info=True,
            )
            raise

        data_type = str(
            jcamp_data.get("data type") or jcamp_data.get("datatype")
        ).lower()
        if data_type and "raman" not in data_type:
            self.logger.warning(
                "JCAMP file %s has DATA TYPE '%s'! Might be not a Raman file.",
                path.name,
                data_type,
            )

        # ------------------------------------------------------------------------- #
        # Extracting spectra and metadata and building output...
        # ------------------------------------------------------------------------- #

        blocks = jcamp_data.get("children", [jcamp_data])
        self.logger.debug("Found %d spectra blocks in %s", len(blocks), path.name)

        measurements: dict[str, dict[str, Any]] = {}
        for idx, block in enumerate(blocks, start=1):
            self.logger.debug("Processing block %d of %d", idx, len(blocks))

            x = block.get("x", []).tolist()
            y = block.get("y", []).tolist()

            raw_metadata = {k: v for k, v in block.items() if k not in ("x", "y")}

            measurements[str(idx)] = {
                "Raman": {
                    "x": x,
                    "y": y,
                    "x_label": block.get("xunits", "wavenumber [1/cm]"),
                },
                "Metadata": {
                    "raw_metadata": raw_metadata,
                },
                "Other_spectra": {},
            }

        if not measurements:
            raise ValueError(f"No valid spectral data found in {path.name}")

        output = self._create_output(measurements, path.name)
        self.logger.info("Converted %s with %d measurements", path.name, len(measurements))

        return output
