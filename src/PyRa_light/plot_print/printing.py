"""Metadata table formatting and printing helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

try:
    from tabulate import tabulate as _tabulate
except ModuleNotFoundError as exc:
    if exc.name != "tabulate":
        raise
    _tabulate = None

if _tabulate is None:
    format_metadata_tables = None
    print_metadata_tables = None
else:

    def format_metadata_tables(
        tables: Sequence[tuple[str, Iterable[Sequence[Any]]]],
        tablefmt: str = "github",
    ) -> str:
        """Format one or more metadata tables into a single string.

        Parameters
        ----------
        tables : Sequence[tuple[str, Iterable[Sequence[Any]]]]
            List of ``(title, rows)`` pairs.
        tablefmt : str, optional
            Tabulate table format.

        Returns
        -------
        str
            Rendered tables as a single string.
        """
        parts: list[str] = []
        for title, rows in tables:
            parts.append(title)
            parts.append(
                _tabulate(
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
            List of ``(title, rows)`` pairs.
        tablefmt : str, optional
            Tabulate table format.
        return_str : bool, optional
            If True, return the rendered string instead of printing it.

        Returns
        -------
        str or None
            Rendered output if ``return_str=True``, else None.
        """
        out = format_metadata_tables(tables=tables, tablefmt=tablefmt)
        if return_str:
            return out
        print(out, end="")
        return None

__all__ = [
    "format_metadata_tables",
    "print_metadata_tables",
]
