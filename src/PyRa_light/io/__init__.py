"""
I/O utilities for PyRa_light.

This subpackage provides:

- Base classes and type definitions for Raman file converters.
- Loader utilities to select and run the appropriate converter
  based on file extension.
"""

from __future__ import annotations

from .converters_base import (
    BaseConverter,
    ConvertedData,
    MeasurementData,
    RamanData,
    UnsupportedFormatError,
)
from .loader import get_converter, load_file

__all__ = [
    "BaseConverter",
    "RamanData",
    "MeasurementData",
    "ConvertedData",
    "UnsupportedFormatError",
    "get_converter",
    "load_file",
]