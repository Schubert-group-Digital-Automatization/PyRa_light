"""
Core data models, metadata handling, and merge logic for PyRa_light.

This subpackage provides:

- Unified data containers for Raman spectra (:class:`Ra`, :class:`RaProc`)
- Intermediate converter models (:class:`ConvertedFile`, :class:`ConvertedMeasurement`)
- Metadata normalization (:func:`unify_metadata`, :class:`MetadataField`)
- High-level merge functionality (:func:`merge_ras`, :class:`MergeMethod`)
"""

from __future__ import annotations

from .converted import (
    ConvertedFile,
    ConvertedMeasurement,
)
from .ra import (
    Ra,
)
from .raproc import (
    ProcessingInfo,
    RaProc,
)
from .metadata import (
    MetadataDict,
    MetadataField,
    unify_metadata,
)

from .merge import (
    MergeMethod,
    MetadataMergeStrategy,
    merge_ras,
)

__all__ = [
    # models
    "ConvertedFile",
    "ConvertedMeasurement",
    "ProcessingInfo",
    "Ra",
    "RaProc",
    # metadata
    "MetadataDict",
    "MetadataField",
    "unify_metadata",
    # merge
    "MergeMethod",
    "MetadataMergeStrategy",
    "merge_ras",
]
