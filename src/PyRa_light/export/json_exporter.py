"""Minimal folder-only JSON exporter."""

from __future__ import annotations

import json
import logging
from enum import Enum, auto
from pathlib import Path
from typing import Any

from ..core.converted import ConvertedFile, ConvertedMeasurement
from ..core.ra import Ra
from ..core.raproc import RaProc

JSON_EXT = "json"

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    """Ensure a directory exists.

    Parameters
    ----------
    path : Path
        Directory path to create if missing. Parent directories are created
        as needed.

    Returns
    -------
    None
        This function is called for its side effects.

    Raises
    ------
    OSError
        If the directory cannot be created (e.g., permissions issues) or if a
        non-directory object blocks the path.
    """
    path.mkdir(parents=True, exist_ok=True)


class ExportMode(Enum):
    """Policy for handling existing files during export."""
    OVERWRITE = auto()
    SKIP = auto()


class JSONExport:
    """Export objects implementing `.to_dict()` to JSON files in a directory."""

    def __init__(self, export_dir: Path, mode: ExportMode = ExportMode.OVERWRITE):
        """Create a JSON exporter writing into a directory.

        Parameters
        ----------
        export_dir : Path
            Target directory for exports. The directory is created if it does
            not exist.
        mode : ExportMode, optional
            Export policy when the target file already exists.

        Returns
        -------
        None
            The exporter instance is initialized in-place.

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
        obj: Any,
        output_dir: str | Path | None = None,
        *,
        suffix: str | None = None,
    ) -> Path | None:
        """Export one object to a JSON file.

        The exporter writes a single JSON file per call using the object's
        `.to_dict()` method. Only output directories are accepted.

        Parameters
        ----------
        obj : Any
            Exportable object. Must implement `.to_dict()` and be one of:
            :class:`ConvertedFile`, :class:`ConvertedMeasurement`, :class:`Ra`,
            or :class:`RaProc`.
        output_dir : str or Path or None, optional
            Output directory. If None, `self.export_dir` is used.
        suffix : str or None, optional
            Free-form suffix appended as `_{suffix}` to the generated stem.
            If None or an empty string, no suffix is added.

        Returns
        -------
        Path or None
            Path to the written JSON file. If `mode` is :attr:`ExportMode.SKIP`
            and the target already exists, returns None without writing.

        Raises
        ------
        ValueError
            If `output_dir` exists and is not a directory.
        TypeError
            If `obj` is of an unsupported type.
        OSError
            If writing the JSON file fails.
        """
        out_dir = self.export_dir if output_dir is None else Path(output_dir)

        if out_dir.exists() and not out_dir.is_dir():
            raise ValueError(f"output_dir must be a directory, got: {out_dir}")

        ensure_dir(out_dir)

        stem = self._filename_gen(obj, suffix=suffix)
        out_path = out_dir / f"{stem}.{JSON_EXT}"

        if out_path.exists() and self.mode == ExportMode.SKIP:
            logger.debug("Skipping existing file (SKIP mode): %s", out_path)
            return None

        self._write_json(obj.to_dict(), out_path)
        return out_path

    @staticmethod
    def _apply_suffix(stem: str, suffix: str | None) -> str:
        """Apply a free-form suffix to a filename stem.

        Parameters
        ----------
        stem : str
            Base stem.
        suffix : str or None
            Suffix to append. If None or an empty string, the stem is returned
            unchanged.

        Returns
        -------
        str
            Stem with `_{suffix}` appended if a suffix is provided.
        """
        if suffix is None or suffix == "":
            return stem
        return f"{stem}_{suffix}"

    def _filename_gen(self, obj: Any, *, suffix: str | None) -> str:
        """Generate the output filename for the exportable object.

        The rules are type-dependent and follow the exporter specification:

        - ConvertedFile: `<base>_conv` (base from `filename` stem, fallback to
          `path` stem)
        - ConvertedMeasurement: `<path.stem>_<index>_conv`
        - Ra: `<filename.stem>_<index>_ra` (fallback to `<filename.stem>_ra` if `index` is None)
        - RaProc: `<merged_sample_name>_<tag>` where tag is derived from
          `processing['method']`

        Parameters
        ----------
        obj : Any
            Exportable object.
        suffix : str or None
            Optional suffix applied via `_apply_suffix`.

        Returns
        -------
        str
            Filename stem without extension.

        Raises
        ------
        TypeError
            If `obj` is of an unsupported type.
        """
        #ConvertedFile: Path(converted.filename).stem + "_conv" + optional suffix
        if isinstance(obj, ConvertedFile):
            base = Path((getattr(obj, "filename", "") or "")).stem
            if not base:
                base = Path(obj.path).stem  # fallback only to avoid failure
            stem = f"{base}_conv"
            return self._apply_suffix(stem, suffix)

        #ConvertedMeasurement: Path(obj.path).stem + obj.index + "_conv" + optional suffix
        if isinstance(obj, ConvertedMeasurement):
            base = Path(obj.path).stem
            stem = f"{base}_{obj.index}_conv"
            return self._apply_suffix(stem, suffix)

        # Ra: Path(ra.filename).stem + "_" + ra.index + "_ra" + optional suffix
        # Fallback (only if index is None): Path(ra.filename).stem + "_ra"
        if isinstance(obj, Ra):
            base = Path(obj.filename).stem
            if obj.index is None:
                stem = f"{base}_ra"
            else:
                stem = f"{base}_{obj.index}_ra"
            return self._apply_suffix(stem, suffix)

        #RaProc: raproc.merged_sample_name + optional suffix else "merged" + optional suffix
        if isinstance(obj, RaProc):
            base = getattr(obj, "merged_sample_name", None) or "merged"
            method_raw: Any = None
            try:
                method_raw = obj.processing.get("method")
            except (AttributeError, TypeError):
                method_raw = None

            method = str(method_raw).lower() if method_raw is not None else ""

            tag = {
                "average": "avg",
                "median": "med",
                "accumulate": "acc",
            }.get(method, "proc")

            stem = f"{base}_{tag}"
            return self._apply_suffix(stem, suffix)

        raise TypeError(f"Unsupported export type: {type(obj)!r}")

    @staticmethod
    def _write_json(data: dict[str, Any], path: Path) -> None:
        """Write a JSON file atomically.

        The data are written to a temporary file with a `.tmp` suffix and then
        moved into place using :meth:`Path.replace`.

        Parameters
        ----------
        data : dict[str, Any]
            JSON-serializable data to write.
        path : Path
            Final output file path.

        Returns
        -------
        None
            This function is called for its side effects.

        Raises
        ------
        OSError
            If writing fails or the temporary file cannot be moved into place.
        TypeError
            If `data` contains values that are not JSON-serializable (after
            applying `default=str`).
        ValueError
            If serialization fails due to invalid values.
        """
        temp_path = path.with_suffix(".tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, default=str)

            temp_path.replace(path)
            logger.debug("Wrote JSON file: %s", path)
        except (OSError, TypeError, ValueError) as err:
            if temp_path.exists():
                temp_path.unlink()
            raise OSError(f"Failed to write {path}: {err}") from err
