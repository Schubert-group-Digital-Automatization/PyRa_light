"""Plotting and printing utilities."""

from __future__ import annotations

from typing import Any, Iterable, Sequence
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np


def format_metadata_tables(
    tables: Sequence[tuple[str, Iterable[Sequence[Any]]]],
    tablefmt: str = "github",
) -> str:
    """Format one or more metadata tables into a single string.

    Parameters
    ----------
    tables : Sequence[tuple[str, Iterable[Sequence[Any]]]]
        List of (title, rows) pairs.
    tablefmt : str, optional
        Tabulate table format.

    Returns
    -------
    str
        Rendered tables as a single string.
    """
    parts: Sequence[str] = []
    for title, rows in tables:
        parts.append(title)
        parts.append(
            tabulate(
                list(rows),
                headers=["key", "value", "source"],
                tablefmt=tablefmt,
            )
        )
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def print_metadata_tables(
    tables: Sequence[tuple[str, Iterable[Sequence[Any]]]],
    tablefmt: str = "github",
    return_str: bool = False,
) -> str | None:
    """Print one or more metadata tables to console.

    Parameters
    ----------
    tables : Sequence[tuple[str, Iterable[Sequence[Any]]]]
        List of (title, rows) pairs.
    tablefmt : str, optional
        Tabulate table format.
    return_str : bool, optional
        If True, return the rendered string instead of printing it.

    Returns
    -------
    str | None
        Rendered output if ``return_str=True``, else None.
    """
    out = format_metadata_tables(tables=tables, tablefmt=tablefmt)
    if return_str:
        return out
    print(out, end="")
    return None


def plot_raman_xy(
    x,
    y,
    *,
    x_label: str = "wavenumber [1/cm]",
    y_label: str = "intensity [a.u.]",
    title: str | None = None,
    ax=None,
    show: bool = True,
    y_low=None,
    y_high=None,
    band_alpha: float = 0.2,
    **plot_kwargs,
):
    """Plot a Raman spectrum from x/y arrays.

    Parameters
    ----------
    x, y : array-like
        Data arrays.
    x_label, y_label : str, optional
        Axis labels.
    title : str | None, optional
        Optional title.
    ax : matplotlib.axes.Axes | None, optional
        Target axes to draw into. If ``None`` (default), a new figure and axes
        are created.
    show : bool, optional
        If ``True`` (default), show the figure via ``matplotlib.pyplot.show``.
    y_low, y_high : array-like | None, optional
        If both are provided, draw a filled band between ``y_low`` and ``y_high``.
    band_alpha : float, optional
        Alpha value for the filled band. Default is 0.2.
    **plot_kwargs
        Additional keyword arguments forwarded to ``Axes.plot``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure containing the plot.
    ax : matplotlib.axes.Axes
        The axes the spectrum was drawn into.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if y_low is not None:
        y_low = np.asarray(y_low)
    if y_high is not None:
        y_high = np.asarray(y_high)

    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    else:
        fig = ax.figure

    ax.plot(x, y, **plot_kwargs)

    if (y_low is not None) and (y_high is not None):
        line_color = ax.lines[-1].get_color() if ax.lines else None
        ax.fill_between(x, y_low, y_high, alpha=band_alpha, color=line_color)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if title:
        ax.set_title(title)

    ax.grid(True, alpha=0.3)


    if show:
        plt.show()

    return fig, ax
