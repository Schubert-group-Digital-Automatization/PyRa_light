"""Optional plotting and printing dependency behavior tests."""

from __future__ import annotations

import builtins
import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest


@contextmanager
def _fresh_pyra_import_blocking(*blocked_packages: str) -> Iterator[None]:
    """Import PyRa_light freshly while selected external packages are missing.

    Parameters
    ----------
    *blocked_packages : str
        Top-level external package names to make unavailable during import.

    Yields
    ------
    None
        Control while the import blocker and fresh module state are active.
    """
    original_import = builtins.__import__
    isolated_roots = ("PyRa_light", *blocked_packages)
    saved_modules: dict[str, ModuleType] = {}

    for module_name in list(sys.modules):
        if _matches_any_root(module_name, isolated_roots):
            saved_modules[module_name] = sys.modules.pop(module_name)

    def _guarded_import(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        """Raise ModuleNotFoundError for blocked top-level packages."""
        if level == 0:
            for blocked_package in blocked_packages:
                if name == blocked_package or name.startswith(f"{blocked_package}."):
                    raise ModuleNotFoundError(
                        f"No module named '{blocked_package}'",
                        name=blocked_package,
                    )
        return original_import(name, globals_, locals_, fromlist, level)

    builtins.__import__ = _guarded_import
    try:
        yield
    finally:
        builtins.__import__ = original_import
        for module_name in list(sys.modules):
            if _matches_any_root(module_name, isolated_roots):
                sys.modules.pop(module_name, None)
        sys.modules.update(saved_modules)


def _matches_any_root(module_name: str, roots: tuple[str, ...]) -> bool:
    """Return whether a module name is a root module or child module.

    Parameters
    ----------
    module_name : str
        Fully qualified module name from ``sys.modules``.
    roots : tuple[str, ...]
        Root module names to match.

    Returns
    -------
    bool
        True when ``module_name`` is one of the roots or below one.
    """
    return any(module_name == root or module_name.startswith(f"{root}.") for root in roots)


def _dummy_xy() -> tuple[np.ndarray, np.ndarray]:
    """Create a minimal x/y pair for core object construction.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Matching one-dimensional Raman x/y arrays.
    """
    return np.array([100.0, 200.0]), np.array([1.0, 2.0])


def test_print_helpers_are_unavailable_without_tabulate() -> None:
    """Missing tabulate keeps imports working and guards print helpers."""
    with _fresh_pyra_import_blocking("tabulate"):
        plot_print = importlib.import_module("PyRa_light.plot_print")
        legacy = importlib.import_module("PyRa_light.plot_print.plot_print")

        assert plot_print.format_metadata_tables is None
        assert plot_print.print_metadata_tables is None
        assert legacy.format_metadata_tables is None
        assert "format_metadata_tables" not in plot_print.__all__
        assert "print_metadata_tables" not in plot_print.__all__

        from PyRa_light.core import ConvertedMeasurement

        x_vals, y_vals = _dummy_xy()
        measurement = ConvertedMeasurement(
            path=Path("dummy.spc"),
            index=1,
            fileformat="spc",
            x=x_vals,
            y=y_vals,
            x_label="wavenumber [1/cm]",
        )

        with pytest.raises(ImportError, match=r"PyRa_light\[print\]"):
            measurement.print_metadata(return_str=True)


def test_plot_helper_is_unavailable_without_matplotlib() -> None:
    """Missing matplotlib keeps imports working and guards plot helpers."""
    with _fresh_pyra_import_blocking("matplotlib"):
        plot_print = importlib.import_module("PyRa_light.plot_print")
        legacy = importlib.import_module("PyRa_light.plot_print.plot_print")

        assert plot_print.plot_raman_xy is None
        assert legacy.plot_raman_xy is None
        assert "plot_raman_xy" not in plot_print.__all__

        from PyRa_light.core import ConvertedMeasurement

        x_vals, y_vals = _dummy_xy()
        measurement = ConvertedMeasurement(
            path=Path("dummy.spc"),
            index=1,
            fileformat="spc",
            x=x_vals,
            y=y_vals,
            x_label="wavenumber [1/cm]",
        )

        with pytest.raises(ImportError, match=r"PyRa_light\[plot\]"):
            measurement.plot_raman(show=False)
