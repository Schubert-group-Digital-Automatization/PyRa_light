"""
Top-level package for PyRa_light.

Provides high-level access to core data models, I/O utilities, and export
functions for Raman data.
"""

from __future__ import annotations

from .core import (
    ConvertedFile,
    ConvertedMeasurement,
    ProcessingInfo,
    Ra,
    RaProc,
    MetadataDict,
    MetadataField,
    unify_metadata,
    MergeMethod,
    MetadataMergeStrategy,
    merge_ras,
)
from .io import (
    BaseConverter,
    RamanData,
    MeasurementData,
    ConvertedData,
    UnsupportedFormatError,
    get_converter,
    load_file,
)
from .export import CSVExport, ExportMode, JSONExport, ensure_dir

__all__ = [
    # core
    "ConvertedFile",
    "ConvertedMeasurement",
    "ProcessingInfo",
    "Ra",
    "RaProc",
    "MetadataDict",
    "MetadataField",
    "unify_metadata",
    "MergeMethod",
    "MetadataMergeStrategy",
    "merge_ras",
    # io
    "BaseConverter",
    "RamanData",
    "MeasurementData",
    "ConvertedData",
    "UnsupportedFormatError",
    "get_converter",
    "load_file",
    # export
    "ExportMode",
    "CSVExport",
    "JSONExport",
    "ensure_dir",
]
