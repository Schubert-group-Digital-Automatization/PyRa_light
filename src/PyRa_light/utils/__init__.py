"""
Utility helpers for PyRa_light.

This subpackage currently provides:

- A registry for mapping file extensions to converter instances.
"""

from __future__ import annotations

from .registry import ConverterRegistry, get_default_registry

__all__ = [
    "ConverterRegistry",
    "get_default_registry",
]