"""Availability checks for optional plotting and printing helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_Helper = Callable[..., Any]

_PRINTING_ERROR = (
    "Metadata printing requires optional dependency 'tabulate'. "
    "Install PyRa_light[print]."
)
_PLOTTING_ERROR = (
    "Raman plotting requires optional dependency 'matplotlib'. "
    "Install PyRa_light[plot]."
)


def require_print_metadata_tables(helper: _Helper | None) -> _Helper:
    """Return the metadata printing helper or raise a dependency error.

    Parameters
    ----------
    helper : Callable[..., Any] or None
        Imported metadata printing helper. ``None`` means the optional
        printing dependency is unavailable.

    Returns
    -------
    Callable[..., Any]
        The available metadata printing helper.

    Raises
    ------
    ImportError
        If the optional ``tabulate`` dependency is not installed.
    """
    if helper is None:
        raise ImportError(_PRINTING_ERROR)
    return helper


def require_plot_raman_xy(helper: _Helper | None) -> _Helper:
    """Return the Raman plotting helper or raise a dependency error.

    Parameters
    ----------
    helper : Callable[..., Any] or None
        Imported Raman plotting helper. ``None`` means the optional plotting
        dependency is unavailable.

    Returns
    -------
    Callable[..., Any]
        The available Raman plotting helper.

    Raises
    ------
    ImportError
        If the optional ``matplotlib`` dependency is not installed.
    """
    if helper is None:
        raise ImportError(_PLOTTING_ERROR)
    return helper
