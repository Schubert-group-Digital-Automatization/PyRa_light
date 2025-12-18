"""Abstract base classes and typed structures for Raman file converters."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypedDict

# ------------------------------------------------------------------------- #
# UnsupportedFormatError
# ------------------------------------------------------------------------- #

class UnsupportedFormatError(ValueError):
    """Raised when the file extension is not supported by any converter."""

# ------------------------------------------------------------------------- #
# RamanData
# ------------------------------------------------------------------------- #

class RamanData(TypedDict):
    """Structure for Raman spectral data."""
    x: list[float]
    y: list[float]
    x_label: str

# ------------------------------------------------------------------------- #
# MeasurementData
# ------------------------------------------------------------------------- #

class MeasurementData(TypedDict):
    """Structure for a single measurement."""
    Raman: RamanData
    Metadata: dict[str, Any]
    Other_spectra: dict[str, Any]

# ------------------------------------------------------------------------- #
# ConvertedData
# ------------------------------------------------------------------------- #

class ConvertedData(TypedDict):
    """Structure for converted file output."""
    Measurements: dict[str, MeasurementData]
    Fileformat: str
    Filename: str

# ------------------------------------------------------------------------- #
# BaseConverter
# ------------------------------------------------------------------------- #

class BaseConverter(ABC):
    """
    Abstract base class for Raman file converters.

    All converters must implement the required properties and methods
    to ensure a consistent interface for file conversion and metadata
    handling.

    Attributes
    ----------
    logger : logging.Logger
        Logger instance associated with this converter class.
    """

    def __init__(self) -> None:
        """
        Initialize the converter with default state.

        Sets up the logger for this converter instance.
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    @abstractmethod
    def format_name(self) -> str:
        """
        Return the human-readable name of this converter format.

        Returns
        -------
        str
            Format name.
        """

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """
        Return the file extensions this converter can read.

        Returns
        -------
        list[str]
            List of supported extensions including dots
            (for example, ``[".wdf"]``).
        """

    @abstractmethod
    def convert(self, file_path: Path) -> dict[str, Any]:
        """
        Convert a file into the standardized PyRa dictionary format.

        Parameters
        ----------
        file_path : Path
            Path to the file to convert.

        Returns
        -------
        dict[str, Any]
            Standardized output dictionary with structure::

                {
                    "Measurements": {
                        "<idx>": {
                            "Raman": {
                                "x": list[float],
                                "y": list[float],
                                "x_label": str,
                            },
                            "Metadata": {
                                "raw_metadata": dict[str, Any],
                                ...
                            },
                            "Other_spectra": dict[str, Any],
                        },
                        ...
                    },
                    "Fileformat": str,
                    "Filename": str,
                }

        Raises
        ------
        Exception
            If the file cannot be opened or read by an underlying
            library in a concrete converter implementation.
        ValueError
            If the file contains no valid spectral data according
            to the concrete converter.
        """

    def _create_output(
        self,
        measurements: dict[str, Any],
        filename: str
    ) -> dict[str, Any]:
        """
        Create a standardized output dictionary.

        Parameters
        ----------
        measurements : dict[str, Any]
            Measurement data indexed by string keys. Each entry
            should follow the :class:`MeasurementData` structure.
        filename : str
            Name of the source file used for conversion.

        Returns
        -------
        dict[str, Any]
            Standardized output dictionary conforming to the
            :class:`ConvertedData` structure.
        """
        output = {
            'Measurements': measurements,
            'Fileformat': self.format_name,
            'Filename': filename
        }
        return output

    def __repr__(self) -> str:
        """
        Return a concise string representation of the converter.

        The representation includes the converter class name,
        its declared format name, and supported extensions.
        """
        return (
            f"{self.__class__.__name__}("
            f"format='{self.format_name}', "
            f"extensions={self.supported_extensions})"
        )
