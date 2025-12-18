"""
utils.registry

Registry mapping file extensions to converter instances.

This module defines ConverterRegistry for looking up converters by file
extension and get_default_registry() for accessing a shared, cached instance.
"""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache

from ..io.converters import JcampConverter
from ..io.converters import OpusConverter
from ..io.converters import SpcConverter
from ..io.converters import WdfConverter
from ..io.converters import WitecConverter
from ..io.converters_base import UnsupportedFormatError, BaseConverter

# ------------------------------------------------------------------------- #
# ConverterRegistry
# ------------------------------------------------------------------------- #

class ConverterRegistry:
    """
    ConverterRegistry

    Tracks which converter handles each file extension.

    Methods
    -------
    get_converter(path: Path) -> Any
        Return the converter for a given file, or raise if unsupported.
    can_read(path: Path) -> bool
        Check if any converter can read the given file.
    list_formats() -> dict[str, list[str]]
        List supported format names and their extensions.
    """

    def __init__(self) -> None:
        """
        Initialize the registry.

        Instantiates each converter and maps its supported extensions
        (in lowercase) to the converter instance.

        Attributes
        ----------
        _converters : dict[str, Any]
            Mapping from file extension to converter instance.
        """
        self._converters: dict[str, BaseConverter] = {}
        for Converter in (
            OpusConverter,
            WdfConverter,
            SpcConverter,
            WitecConverter,
            JcampConverter,
        ):
            inst = Converter()
            for ext in inst.supported_extensions:
                self._converters[ext.lower()] = inst

    def get_converter(self, path: Path) -> BaseConverter:
        """
        Return the converter for a given file.

        Parameters
        ----------
        path : Path
            Path to a file.

        Returns
        -------
        Any
            The converter for the file's extension.

        Raises
        ------
        UnsupportedFormatError
            If no converter is found for the extension.
        """
        ext = path.suffix.lower()
        try:
            return self._converters[ext]
        except KeyError as e:
            raise UnsupportedFormatError(f"No converter registered for extension '{ext}'") from e

    def can_read(self, path: Path) -> bool:
        """
        Check if any converter can read a file.

        Parameters
        ----------
        path : Path
            Path to a file.

        Returns
        -------
        bool
            True if a converter exists, False otherwise.
        """
        try:
            self.get_converter(path)
            return True
        except UnsupportedFormatError:
            return False

    def list_formats(self) -> dict[str, list[str]]:
        """
        List all supported formats and their extensions.

        Returns
        -------
        dict[str, list[str]]
            Map from format names to their extensions.
        """
        formats: dict[str, list[str]] = {}
        for ext, conv in self._converters.items():
            name = conv.format_name
            formats.setdefault(name, []).append(ext)
        return formats


@lru_cache(maxsize=1)
def get_default_registry() -> ConverterRegistry:
    """
    Provide the shared ConverterRegistry for the application.

    Returns
    -------
    ConverterRegistry
        A cached registry instance mapping file extensions to converter instances.
        The registry is created once and reused on subsequent calls.
    """
    return ConverterRegistry()
