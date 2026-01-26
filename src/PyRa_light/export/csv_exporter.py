"""CSV exporter for unified Raman spectra.

This exporter was created to be compliant with Bocklitz' RamanMetrix:
- It supports only :class:`Ra`-objects.
- One export call writes three CSV files: ``*_x.csv``, ``*_y.csv``, ``*_meta.csv``.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from ..core.metadata import MetadataField
from ..core.ra import Ra
from .json_exporter import ExportMode, ensure_dir

logger = logging.getLogger(__name__)

_CSV_EXT = "csv"
_DELIMITER = ";"


class CSVExport:
    """Export :class:`Ra` objects to three CSVs."""

    def __init__(self, export_dir: Path, mode: ExportMode = ExportMode.OVERWRITE) -> None:
        """Create a CSV exporter writing into a directory.

        Parameters
        ----------
        export_dir : Path
            Target directory for exports. The directory is created if it does
            not exist.
        mode : ExportMode, optional
            Export policy when any target file already exists.

        Raises
        ------
        ValueError
            If `export_dir` exists but is not a directory.
        """
        self.export_dir = Path(export_dir)
        if self.export_dir.exists() and not self.export_dir.is_dir():
            raise ValueError(f"export_dir must be a directory, got: {self.export_dir}")
        ensure_dir(self.export_dir)
        self.mode = mode

    def export(
        self,
        ra: Ra,
        output_dir: str | Path | None = None,
        *,
        suffix: str | None = None,
    ) -> tuple[Path, Path, Path] | None:
        """Export a single :class:`Ra` to three CSV files.

        Files written are:
        - ``<stem>_x.csv``: one column named ``x``
        - ``<stem>_y.csv``: one column named ``y``
        - ``<stem>_meta.csv``: one-row wide table (columns are metadata keys)

        Parameters
        ----------
        ra : Ra
            Object to export. Must be an instance of :class:`Ra`.
        output_dir : str or Path or None, optional
            Output directory. If None, `self.export_dir` is used.
        suffix : str or None, optional
            Free-form suffix appended as ``_{suffix}`` to the generated stem.
            If None or an empty string, no suffix is added.

        Returns
        -------
        tuple[Path, Path, Path] or None
            Paths to ``(x_csv, y_csv, meta_csv)``. 

        Raises
        ------
        ValueError
            If `output_dir` exists and is not a directory.
        OSError
            If writing any CSV file fails.
        """
        if not isinstance(ra, Ra):
            raise TypeError(
                "CSVExport.export supports only Ra objects; "
                f"got {type(ra).__name__}"
            )
        out_dir = self.export_dir if output_dir is None else Path(output_dir)
        if out_dir.exists() and not out_dir.is_dir():
            raise ValueError(f"output_dir must be a directory, got: {out_dir}")
        ensure_dir(out_dir)

        stem = self._filename_stem(ra, suffix=suffix)
        x_path = out_dir / f"{stem}_x.{_CSV_EXT}"
        y_path = out_dir / f"{stem}_y.{_CSV_EXT}"
        meta_path = out_dir / f"{stem}_meta.{_CSV_EXT}"

        bundle_exists = x_path.exists() or y_path.exists() or meta_path.exists()
        if self.mode == ExportMode.SKIP and bundle_exists:
            logger.debug(
                "Skipping existing CSV bundle (SKIP mode): %s", stem
            )
            return None

        x_values = list(ra.x)
        y_values = list(ra.y)
        if len(x_values) != len(y_values):
            raise ValueError(
                f"x/y length mismatch for export: {len(x_values)} vs {len(y_values)}"
            )

        meta = self._build_meta(ra)

        self._write_vector_csv(x_values, header=f"{stem}_x", path=x_path)
        self._write_vector_csv(y_values, header=f"{stem}_y", path=y_path)
        self._write_meta_csv(meta, path=meta_path)

        logger.debug("Wrote CSV bundle: %s", stem)
        return (x_path, y_path, meta_path)

    @staticmethod
    def _apply_suffix(stem: str, suffix: str | None) -> str:
        """Append a free-form suffix to a stem.

        Parameters
        ----------
        stem : str
            Base filename stem.
        suffix : str or None
            If provided and non-empty, append ``_{suffix}``.

        Returns
        -------
        str
            Stem with suffix applied.
        """
        if suffix is None or suffix == "":
            return stem
        return f"{stem}_{suffix}"

    def _filename_stem(self, ra: Ra, *, suffix: str | None) -> str:
        """Compute the filename stem for a :class:`~PyRa_light.core.ra.Ra`.

        Rules
        -----
        The stem is derived from the original filename and the measurement index:

        - ``<Path(ra.filename).stem>_<ra.index>``

        This exporter is Ra-only, so no additional `_ra` tag is added.

        Parameters
        ----------
        ra : Ra
            Spectrum to export.
        suffix : str or None
            Optional suffix applied via :meth:`_apply_suffix`.

        Returns
        -------
        str
            Filename stem without extension.

        Raises
        ------
        ValueError
            If `ra.index` is None.
        """
        if ra.index is None:
            raise ValueError("Ra.index is required for CSV export naming")

        base = Path(ra.filename).stem
        stem = f"{base}_{ra.index}"
        return self._apply_suffix(stem, suffix)

    @staticmethod
    def _stringify(value: object) -> str:
        """Convert values to a CSV-safe string representation."""
        if value is None:
            return ""
        return str(value)

    def _build_meta(self, ra: Ra) -> dict[str, str]:
        """Build a wide metadata dict (columns=keys) for the meta CSV.

        Column order is deterministic:
        1) core columns: filename, sample_name, index
        2) unified_meta fields in MetadataField enum order

        Returns
        -------
        dict[str, str]
            Mapping of column name to stringified value.
        """
        meta: dict[str, str] = {
            "filename": self._stringify(ra.filename),
            "sample_name": self._stringify(ra.sample_name),
            "index": self._stringify(ra.index),
        }

        unified = ra.unified_meta or {}
        for field in MetadataField:
            key = field.value
            if key in meta:
                continue
            meta[key] = self._stringify(unified.get(key))

        return meta

    @staticmethod
    def _atomic_write_csv(rows: list[list[str]], path: Path) -> None:
        """Write CSV rows atomically using a temporary file."""
        temp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with temp_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter=_DELIMITER)
                writer.writerows(rows)
            temp_path.replace(path)
        except OSError as err:
            if temp_path.exists():
                temp_path.unlink()
            raise OSError(f"Failed to write {path}: {err}") from err

    def _write_vector_csv(self, values: list[object], *, header: str, path: Path) -> None:
        """Write a wide vector CSV (x or y) with the label in the first cell."""
        row = [header] + [self._stringify(v) for v in values]
        self._atomic_write_csv([row], path)

    def _write_meta_csv(self, meta: dict[str, str], *, path: Path) -> None:
        """Write the wide, one-row metadata CSV."""
        headers = list(meta.keys())
        row = [meta[h] for h in headers]
        self._atomic_write_csv([headers, row], path)
