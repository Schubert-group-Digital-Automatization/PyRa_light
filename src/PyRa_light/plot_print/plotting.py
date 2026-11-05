"""Raman plotting helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as _plt
except ModuleNotFoundError as exc:
    if exc.name != "matplotlib":
        raise
    _plt = None

if _plt is None:
    plot_raman_xy = None
else:

    def plot_raman_xy(
        x: Any,
        y: Any,
        *,
        x_label: str = "wavenumber [1/cm]",
        y_label: str = "intensity [a.u.]",
        title: str | None = None,
        ax: Any = None,
        show: bool = True,
        y_low: Any = None,
        y_high: Any = None,
        band_alpha: float = 0.2,
        **plot_kwargs: Any,
    ) -> tuple[Any, Any]:
        """Plot a Raman spectrum from x/y arrays.

        Parameters
        ----------
        x : Any
            X-axis data array.
        y : Any
            Y-axis data array.
        x_label : str, optional
            X-axis label.
        y_label : str, optional
            Y-axis label.
        title : str or None, optional
            Optional plot title.
        ax : Any, optional
            Target matplotlib axes. A new figure and axes are created when
            omitted.
        show : bool, optional
            If True, show the figure through ``matplotlib.pyplot.show``.
        y_low : Any, optional
            Lower bound for an optional filled band.
        y_high : Any, optional
            Upper bound for an optional filled band.
        band_alpha : float, optional
            Alpha value for the filled band.
        **plot_kwargs : Any
            Additional keyword arguments forwarded to ``Axes.plot``.

        Returns
        -------
        tuple[Any, Any]
            The matplotlib figure and axes.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        if y_low is not None:
            y_low = np.asarray(y_low)
        if y_high is not None:
            y_high = np.asarray(y_high)

        if ax is None:
            fig, ax = _plt.subplots(layout="constrained")
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
            _plt.show()

        return fig, ax

__all__ = ["plot_raman_xy"]
