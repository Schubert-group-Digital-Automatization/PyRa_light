"""
Core utilities to select and run Raman file converters.

Select and use the correct proprietary Raman file converter based on file
extension. Provides:

- ``get_converter(path)`` to obtain a converter instance.
- ``load_file(path)`` to convert a file into a :class:`ConvertedFile` with
  one or more measurements.
"""

from __future__ import annotations
import logging

from pathlib import Path

from .converters_base import BaseConverter, UnsupportedFormatError
from ..core.converted import ConvertedFile
from ..utils.registry import get_default_registry

logger = logging.getLogger(__name__)


def get_converter(path: Path | str) -> BaseConverter:
    """
    Return a converter instance for the given file based on its extension.

    Parameters
    ----------
    path : Path | str
        Path to the proprietary Raman file to convert.

    Returns
    -------
    BaseConverter
        An instance of the appropriate converter.

    Raises
    ------
    FileNotFoundError
        If the provided path does not exist.
    UnsupportedFormatError
        If the file extension is not recognized.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    logger.debug("Selecting converter for file: %s (suffix=%s)", p, p.suffix.lower())

    try:
        converter = get_default_registry().get_converter(p)
        if not isinstance(converter, BaseConverter):
            raise TypeError(f"Converter must inherit from BaseConverter, got {type(converter)}")
        logger.debug("Selected converter: %s", converter.__class__.__name__)
        return converter
    except ValueError as e:
        raise UnsupportedFormatError(str(e)) from e


def load_file(path: Path | str) -> ConvertedFile:
    """
    Load and convert a proprietary Raman file into the PyRa format.

    Parameters
    ----------
    path : Path | str
        Path to the file to load and convert.

    Returns
    -------
    ConvertedFile
        Standardized representation of the converted file including one or
        more measurements, file-level metadata, and (optionally) the raw
        converter dict.

    Raises
    ------
    FileNotFoundError
        If the provided path does not exist.
    UnsupportedFormatError
        If the file extension is not recognized.
    ValueError
        If the converter fails internally.
    """
    p = Path(path)
    logger.info("Loading Raman file: %s", p)
    converter = get_converter(p)
    logger.debug("Converting with %s", converter.__class__.__name__)

    try:
        data = converter.convert(p)
    except Exception:
        logger.exception("Conversion failed for %s using %s", p, converter.__class__.__name__)
        raise

    n_meas = len(data.get("Measurements", {}) or {})
    logger.info("Conversion succeeded for %s (%d measurement(s))", p, n_meas)
    return ConvertedFile.from_dict(data, path=p)


__all__ = ["get_converter", "load_file"]
