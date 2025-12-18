"""
core.opus_converter

Converter for Bruker OPUS (.0, .1, .2, etc.) files.

This module defines OpusConverter, which reads OPUS files, extracts spectra
and metadata, and returns a standardized output dictionary for PyRa.

Raman block selection (if there are more than one)
---------------------
If multiple spectral blocks with Raman-like keys are present, the converter
applies the following selection rule:

1. Prefer a block with the exact key ``"ra"`` if available.
2. Otherwise, select the first block whose key starts with ``"ra"`` (case-
insensitive).
3. If no such block exists, a :class:`ValueError` is raised.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from brukeropus import read_opus
from brukeropus.file import get_param_label

from ..converters_base import BaseConverter


class OpusConverter(BaseConverter):
    """
    Converter for Bruker OPUS files.

    This converter reads Bruker OPUS files, extracts spectra and associated
    metadata, and returns a standardized dictionary representation suitable
    for further downstream processing within PyRa.

    The main entry point is :meth:`convert`, which accepts a file path and
    produces a nested mapping containing the Raman spectrum, additional
    spectral blocks, and raw metadata.
    """

    @property
    def format_name(self) -> str:
        """Name of this converter format."""
        return "OPUS"

    @property
    def supported_extensions(self) -> list[str]:
        return [f".{i}" for i in range(10)]

    def convert(self, file_path: Path | str) -> dict[str, Any]:
        """
        Read a Bruker OPUS file and extract spectra and metadata.

        Parameters
        ----------
        file_path : Path or str
            Path to the OPUS file to be converted. 

        Returns
        -------
        dict[str, Any]
            Standardized output dictionary following the PyRa internal format.
            The top-level keys include at least:

            - ``"Measurements"``: mapping from measurement IDs to measurement data.
            - ``"Fileformat"``: name of the file format (``"OPUS"`` for this converter).
            - ``"Filename"``: base name of the original OPUS file.

        Raises
        ------
        Exception
            Any exception raised by :func:`brukeropus.read_opus`, for example
            :class:`OSError` when the file cannot be read or a parsing error for
            invalid OPUS content.
        ValueError
            If no Raman spectral block could be identified in the file.
        """

        # ------------------------------------------------------------------------- #
        # Read File
        # ------------------------------------------------------------------------- #

        self.logger.debug("Starting OPUS conversion: %s", file_path)
        path = Path(file_path)

        try:
            opusfile = read_opus(str(path))
        except IOError as e:
            self.logger.error(
                "Failed to read OPUS file %s: %s",
                path.name, e, exc_info=True,
            )
            raise

        # ------------------------------------------------------------------------- #
        # Extracting Parameters & Labels
        # ------------------------------------------------------------------------- #

        params = (
            dict(opusfile.params.items())    # type: ignore[arg-type]
            if opusfile.params
            else {}
        )
        self.logger.debug("Extracted %d metadata entries from %s", len(params), path.name)

        # Labels for params
        params_labels: dict[str, str] = {}
        for key in params:
            label = get_param_label(str(key))
            if label:
                params_labels[key] = label

        # Extract reference parameters
        rf_params = (
            dict(opusfile.rf_params.items())    # type: ignore[arg-type]
            if opusfile.rf_params
            else {}
        )
        self.logger.debug(
            "Extracted %d reference metadata entries from %s",
            len(rf_params),
            path.name,
        )

        # Labels for rf_params
        rf_params_labels: dict[str, str] = {}
        for key in rf_params:
            label = get_param_label(str(key))
            if label:
                rf_params_labels[key] = label

        # History and Reports
        history = getattr(opusfile, "history", None)
        reports = getattr(opusfile, "reports", None)

        # Construct raw_metadata dictionary
        raw_metadata = {
            (params_labels.get(k) or k): v
            for k, v in params.items()
        }
        if rf_params:
            raw_metadata["reference_params"] = {
                (rf_params_labels.get(k) or k): v
                for k, v in rf_params.items()
            }
        if history is not None:
            raw_metadata["file_history"] = history
        if reports:
            raw_metadata["reports"] = reports

        # ------------------------------------------------------------------------- #
        # Extracting spectral/data blocks
        # ------------------------------------------------------------------------- #

        spectra: dict[str, dict[str, Any]] = {}
        data_objects: dict[str, Any] = {}
        self.logger.debug("Scanning for spectral blocks")

        for data in opusfile.iter_all_data():
            key = data.key
            dxu = data.dxu
            spectra[key] = self._extract_spectrum(data, key, dxu)
            data_objects[key] = data

        # Find Raman block: prefer exact 'ra', otherwise first key starting with 'ra'
        raman_key = (
            "ra"
            if "ra" in spectra
            else next((k for k in spectra if k.lower().startswith("ra")), None)
        )
        self.logger.info("Identified Raman block: %s", raman_key)

        if not raman_key:
            self.logger.warning(
                "No Raman spectral block found in '%s'. Available keys: %s",
                path.name,
                list(spectra.keys()),
            )
            raise ValueError(
                f"No Raman spectral block found in file '{path.name}'"
            )


        # Bruker apparently safes also other (filled but not relevant) data blocks.
        raman_spec = spectra.pop(raman_key)
        other_spectra = spectra

        # ------------------------------------------------------------------------- #
        # Build output dictionary
        # ------------------------------------------------------------------------- #

        measurements = {
            "1": {
                "Raman": {
                    "x": raman_spec["x"],
                    "y": raman_spec["y"],
                    "x_label": raman_spec.get("x_label", None),
                },
                "Metadata": {
                    "sample_name": params.get("snm"),
                    "raw_metadata": raw_metadata,
                },
                "Other_spectra": other_spectra,
            }
        }

        raman_data = data_objects.get(raman_key)
        if raman_data is not None:
            raw_metadata["measurement_datetime"] = raman_data.dat    # type: ignore[arg-type]

        output = self._create_output(measurements, path.name)
        self.logger.info("Converted %s with %d measurement(s)", path.name, len(measurements))

        return output

    # ------------------------------------------------------------------------- #
    # Private Helpers
    # ------------------------------------------------------------------------- #

    def _extract_spectrum(self, block: Any, block_id: str, dxu: Any) -> dict[str, Any]:
        """
        Extract a single spectral block into a simple dictionary.

        Parameters
        ----------
        block : Any
            Spectral data object as returned by :mod:`brukeropus`, typically
            a ``Data`` or ``DataSeries`` instance providing ``x``/``wn`` and
            ``y`` attributes.
        block_id : str
            Identifier for the spectral block. This is used as a key in the
            spectra mapping and stored as the ``"blocktype"`` field.
        dxu : Any
            OPUS unit code describing the original x-axis units. This is used
            to derive a human-readable x-axis label when no wavenumber axis
            (``wn``) is available.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the extracted spectral data and basic
            metadata. The dictionary includes the following keys:

            - ``"blockname"``: human-readable label for the block, if available.
            - ``"blocktype"``: the passed ``block_id``.
            - ``"x_label"``: label for the x-axis.
            - ``"y_label"``: label for the y-axis (intensity).
            - ``"x"``: list of x-axis values.
            - ``"y"``: list of y-axis values.
        """
        # Choose x-axis: prefer wavenumber axis if available,
        # otherwise use stored x with DXU-based label
        x_source = getattr(block, "wn", None)
        if x_source is None:
            x_source = block.x
            x_label = self._get_x_unit(dxu)
        else:
            x_label = "wavenumber [1/cm]"

        x_data = x_source.tolist()
        y_data = block.y.tolist()

        return {
            "blockname": getattr(block, "label", block_id),
            "blocktype": block_id,
            "x_label": x_label,
            "y_label": "intensity [a.u.]",
            "x": x_data,
            "y": y_data,
        }

    def _get_x_unit(self, dxu: Any) -> str:
        """
        Map an OPUS DXU code to a human-readable x-axis unit string.

        Parameters
        ----------
        dxu : Any
            OPUS DXU code describing the stored x-axis units, for example
            ``"WN"``, ``"WL"``, ``"FREQ"``, or ``"NM"``.

        Returns
        -------
        str
            Human-readable unit string corresponding to the given DXU code.
            For unknown codes, the original value is included with
            ``"[a.u.]"`` as a generic unit.
        """
        unit_map = {
            'WN': 'wavenumber [1/cm]',
            'WL': 'wavelength [µm]',
            'FREQ': 'frequency [Hz]',
            'NM': 'wavelength [nm]',
        }
        return unit_map.get(str(dxu).upper(), f'{dxu} [a.u.]')
