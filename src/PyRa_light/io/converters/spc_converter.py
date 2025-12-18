"""
Converter for SPC (.spc) files.

This module defines :class:`SpcConverter` for reading SPC files, 
extracting spectral data and associated metadata, 
and returning a standardized dictionary in the internal
PyRa format.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import spc_spectra

from ..converters_base import BaseConverter


class SpcConverter(BaseConverter):
    """
    Converter for SPC (.spc) files.

    The converter reads a SPC file using `spc_spectra`,
    combines the global x-axis with per-subfile y-series, extracts
    metadata from the header and log dictionary, and returns a
    standardized dictionary in the PyRa internal format.
    """

    @property
    def format_name(self) -> str:
        """Name of this converter format."""
        return "SPC"

    @property
    def supported_extensions(self) -> list[str]:
        """File extensions this converter can read."""
        return [".spc"]

    def convert(self, file_path: Path) -> dict[str, Any]:
        """
        Read a SPC file and extract spectra and metadata.

        Parameters
        ----------
        file_path : Path
            Path to the SPC file to be converted.

        Returns
        -------
        dict[str, Any]
            Standardized output dictionary following the PyRa internal
            format. The top-level keys include at least:

            - ``"Measurements"``: mapping from measurement IDs to
              measurement data (one entry per SPC subfile).
            - ``"Fileformat"``: name of the file format
              (``"SPC"`` for this converter).
            - ``"Filename"``: base name of the original SPC file.

        Raises
        ------
        Exception
            Any exception raised by :class:`spc_spectra.File` when the
            file cannot be read or the SPC content is invalid.
        ValueError
            If the file contains no valid spectral subfiles.
        """

    # ------------------------------------------------------------------------- #
    # Read File
    # ------------------------------------------------------------------------- #

        self.logger.debug("Starting SPC conversion: %s", file_path)
        path = Path(file_path)

        try:
            spc_file = spc_spectra.File(str(path))
        except IOError as e:
            self.logger.error(
                "Failed to open SPC file %s: %s",
                path.name, e, exc_info=True,
            )
            raise


    # ------------------------------------------------------------------------- #
    # Extracting Metadata
    # ------------------------------------------------------------------------- #

        raw_metadata = self._extract_metadata(spc_file)
        self.logger.debug("Extracted %d raw metadata entries", len(raw_metadata))

    # ------------------------------------------------------------------------- #
    # Extracting Data
    # ------------------------------------------------------------------------- #

        if hasattr(spc_file.x, "tolist"):
            x_values = spc_file.x.tolist()  # type: ignore[assignment]
        else:
            x_values = list(spc_file.x)

        # Check for valid data
        if not spc_file.sub:
            raise ValueError(f"No spectral data found in {path.name}")

        # Process each subfile
        measurements = {}
        for idx, subfile in enumerate(spc_file.sub, start=1):
            if hasattr(subfile.y, "tolist"):
                y_vals = subfile.y.tolist()
            else:
                y_vals = list(subfile.y)

            # Extract subfile-specific metadata
            z = getattr(subfile, "subtime", idx)
            z_end = getattr(subfile, "subnext", None)
            w = getattr(subfile, "subwlevel", None)

    # ------------------------------------------------------------------------- #
    # Building output dictionary
    # ------------------------------------------------------------------------- #

            measurements[str(idx)] = {
                "Raman": {
                    "x": x_values,
                    "y": y_vals,
                    "x_label": spc_file.xlabel,
                },
                "Metadata": {
                    "z": z,
                    "z_end": z_end,
                    "w": w,
                    "raw_metadata": raw_metadata,
                },
                "Other_spectra": {},
            }

        output = self._create_output(measurements, path.name)
        self.logger.info("Converted %s with %d spectra", path.name, len(measurements))

        return output

    # ------------------------------------------------------------------------- #
    # Private helpers
    # ------------------------------------------------------------------------- #

    def _extract_metadata(self, spc_file: Any) -> dict[str, Any]:
        """
        Extract and decode metadata from an SPC file object.

        Parameters
        ----------
        spc_file : Any
            SPC file object returned by :class:`spc_spectra.File`.

        Returns
        -------
        dict[str, Any]
            Dictionary of extracted metadata with decoded values. It
            includes at least:

            - ``"x_label"`` / ``"y_label"`` / ``"z_label"``: axis
              labels provided by the SPC file.
            - ``"comment"`` / ``"raw_comment"``: decoded textual
              comments.
            - ``"experiment_type"``: experiment or acquisition type.
            - ``"num_subfiles"``: number of spectral subfiles.
            - ``"num_points"``: number of points per spectrum.
            - ``"first_x"`` / ``"last_x"``: x-axis range.
            - ``"acquisition_date"`` and ``"acquisition_time"``:
              optional date and time, if available.
            - additional key-value pairs decoded from
              ``spc_file.log_dict`` when present.
        """

        raw_metadata = {
            "x_label": spc_file.xlabel,
            "y_label": spc_file.ylabel,
            "z_label": spc_file.zlabel,
            "comment": self._decode_metadata_value(spc_file.cmnt),
            "raw_comment": self._decode_metadata_value(spc_file.fcmnt),
            "experiment_type": self._decode_metadata_value(spc_file.exp_type),
            "num_subfiles": len(spc_file.sub),
            "num_points": spc_file.fnpts,
            "first_x": spc_file.ffirst,
            "last_x": spc_file.flast,
        }

        # Add date/time if available
        if (
            hasattr(spc_file, "year")
            and hasattr(spc_file, "month")
            and hasattr(spc_file, "day")
        ):
            raw_metadata["acquisition_date"] = (
                f"{spc_file.year:04d}-{spc_file.month:02d}-{spc_file.day:02d}"
            )

        if hasattr(spc_file, "hour") and hasattr(spc_file, "minute"):
            raw_metadata["acquisition_time"] = (
                f"{spc_file.hour:02d}:{spc_file.minute:02d}"
            )

        # Process log dictionary (a weird thingy where I am not sure if it gets converted out-of-the-box)
        if hasattr(spc_file, "log_dict") and isinstance(spc_file.log_dict, dict):
            for k, v in spc_file.log_dict.items():
                key = self._decode_metadata_value(k)
                raw_metadata[key] = self._decode_metadata_value(v)

        return raw_metadata

    def _decode_metadata_value(self, value: Any) -> str:
        """
        Convert raw metadata values into clean Unicode strings.

        Parameters
        ----------
        value : Any
            Raw metadata value (bytes, bytearray, str, or other
            Python object) to be normalized.

        Returns
        -------
        str
            Cleaned string representation with null characters
            removed and surrounding whitespace stripped. If ``value``
            is ``None``, an empty string is returned.
        """
        # Handle actual bytes
        if isinstance(value, (bytes, bytearray)):
            try:
                decoded_str = value.decode("utf-8", errors="replace")
            except ValueError as e:
                self.logger.warning("Byte-decode failed, falling back: %s", e)
                decoded_str = value.decode(errors="ignore")

        # Handle string literal of bytes
        elif isinstance(value, str) and value.startswith("b'"):
            try:
                byte_seq = ast.literal_eval(value)
                if isinstance(byte_seq, bytes):
                    decoded_str = byte_seq.decode("utf-8", errors="replace")
                else:
                    decoded_str = str(value)
            except (ValueError, SyntaxError) as e:
                self.logger.warning("Literal eval of byte-string failed: %s", e)
                decoded_str = value
        else:
            return str(value) if value is not None else ""

        decoded_str = decoded_str.replace("\x00", "").strip()
        return decoded_str


__all__ = ["SpcConverter"]
