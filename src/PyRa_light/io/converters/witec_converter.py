"""
Converters for WITec .wip files.

This module defines :class:`WitecConverter` for reading WITec Project
(.wip) files, extracting spectra and associated metadata, and returning a
standardized dictionary in the internal PyRa format.

A word of warning: .wip files can contain several measurements! 
If you use them like this make sure that you save the metadata and the spectra with a caption!

The package can also match based on positional data or IDs,
but those matching methods are prone to errors.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from photonicdata_files_wip import TDGraph, TDText, WIPFile

from ..converters_base import BaseConverter


class WitecConverter(BaseConverter):
    """
    Converter for WITec .wip files.

    The converter reads a WITec Project (.wip) file via
    photonicdata_files_wip.WIPFile, extracts spectral and text
    metadata blocks, matches them using caption, position, or
    nearest-ID fallback, and returns the measurement data with
    associated metadata in the PyRa internal dict format.
    """

    @property
    def format_name(self) -> str:
        """Name of this converter format."""
        return "WITEC"

    @property
    def supported_extensions(self) -> list[str]:
        """File extensions this converter can read."""
        return [".wip"]

    def convert(self, file_path: Path) -> dict[str, Any]:
        """
        Read a WITec .wip file and extract spectra and metadata.

        Parameters
        ----------
        file_path : Path
            Path to the WIP file to be converted.

        Returns
        -------
        dict[str, Any]
            Standardized output dictionary following the PyRa internal
            format. The top-level keys include at least:

            - ``"Measurements"``: mapping from measurement IDs to
              measurement data.
            - ``"Fileformat"``: name of the file format
              (``"WITEC"`` for this converter).
            - ``"Filename"``: base name of the original WIP file.

        Raises
        ------
        Exception
            Any exception raised by :class:`photonicdata_files_wip.WIPFile`
            when the file cannot be read or the WIP content is invalid.
        ValueError
            If no valid TDGraph spectra blocks could be identified in
            the file.
        """

        # ------------------------------------------------------------------------- #
        # Read File
        # ------------------------------------------------------------------------- #

        self.logger.debug("Starting WITEC conversion: %s", file_path)
        path = Path(file_path)

        try:
            wipfile = WIPFile(str(path))
        except IOError as e:
            self.logger.error(
                "Failed to open WIP file %s: %s", path.name, e, exc_info=True
            )
            raise

        # ------------------------------------------------------------------------- #
        # Extracting (Spectra) Blocks and matching them with metadatblocks
        # ------------------------------------------------------------------------- #

        blocks: list[Any] = wipfile.data_td(
            keep=["TDGraph", "TDText"])  # pyright: ignore[reportAssignmentType]

        graphs = [
            block
            for block in blocks
            if isinstance(block, TDGraph)
            and block.is_exportable
            and block.is_valid_spectra
        ]

        self.logger.debug("Found %d exportable TDGraph blocks", len(graphs))
        if not graphs:
            raise ValueError(f"No valid spectra found in file '{path.name}'")

        tdgraph_entries: dict[str, dict[str, Any]] = {}
        for graph in graphs:
            self.logger.debug("Processing graph datanumber %s", graph.datanumber)
            spectra_list = self._extract_spectra(graph, wipfile)
            metadata = self._match_metadata(graph, wipfile, blocks)
            tdgraph_entries[str(graph.datanumber)] = {
                "graph_caption": wipfile.caption(graph.datanumber),
                "Spectra": spectra_list,
                "Metadata": metadata,
            }

        # ------------------------------------------------------------------------- #
        # Building output dictionary
        # ------------------------------------------------------------------------- #

        measurements: dict[str, dict[str, Any]] = {}
        for i, entry in enumerate(tdgraph_entries.values(), start=1):
            spectra = entry["Spectra"]
            raman_block = spectra[0]

            other_spectra: dict[str, Any] = {}
            if len(spectra) > 1:
                other_spectra["Raman_extra"] = spectra[1:]
                self.logger.debug("Found %d spectra within one TDGraph", len(other_spectra))

            measurements[str(i)] = {
                "Raman": {
                    "x": raman_block.get("x", []),
                    "y": raman_block.get("y", []),
                    "x_label": raman_block.get("x_label", "wavenumber [1/cm]"),
                },
                "Metadata": entry["Metadata"],
                "Other_spectra": other_spectra,
            }

        output = self._create_output(measurements, path.name)
        self.logger.info(
            "Converted %s with %d measurements", path.name, len(measurements)
        )

        return output

    # ------------------------------------------------------------------------- #
    # Private Helpers
    # ------------------------------------------------------------------------- #

    def _extract_spectra(
        self,
        graph: TDGraph,
        wipfile: WIPFile,
    ) -> list[dict[str, Any]]:
        """
        Extract spectral data from a TDGraph block into PyRa format.

        Parameters
        ----------
        graph : TDGraph
            Graph block containing the spectral matrix and associated
            Raman shift axis.
        wipfile : WIPFile
            WIPFile instance used to resolve the caption for the given
            graph datanumber.

        Returns
        -------
        list[dict[str, Any]]
            List of spectra dictionaries. Each entry contains at least
            the keys:

            - ``"x"``: list of Raman shift values.
            - ``"y"``: list of intensity values.
            - ``"x_label"``: x-axis label (wavenumber).
            - ``"y_label"``: y-axis label (intensity).
            - ``"label"``: human-readable label derived from the graph
              caption and spectrum index.
        """
        x_vals = graph.raman_shift.tolist()
        caption = wipfile.caption(graph.datanumber)
        spectra_list: list[dict[str, Any]] = []

        for idx, y_vals in enumerate(graph.spectra_matrix, start=1):
            spec = {
                "x": x_vals,
                "y": y_vals.tolist(),
                "x_label": "wavenumber [1/cm]",
                "y_label": "intensity [a.u.]",
                "label": f"{caption} [{idx}]",
            }
            spectra_list.append(spec)

        return spectra_list

    def _match_metadata(
        self,
        graph: TDGraph,
        wipfile: WIPFile,
        all_blocks: list[Any],
    ) -> dict[str, Any]:
        """
        Match a TDGraph block to its associated text metadata. This is kinda odd.
        The main problem is that you can safe several measurments in a single file!
        If the measurements and the metadata are not saved with a caption the matching gets weird.

        Parameters
        ----------
        graph : TDGraph
            Graph block for which metadata should be located.
        wipfile : WIPFile
            WIPFile instance providing access to captions, data blocks
            and related-data utilities.
        all_blocks : list[Any]
            List of TDGraph and TDText objects returned by
            :meth:`WIPFile.data_td` for this file.

        Returns
        -------
        dict[str, Any]
            Metadata dictionary describing the match and containing the
            parsed text metadata. It includes at least:

            - ``"filename"``: base name of the originating WIP file.
            - ``"match_type"``: strategy used for matching
              (``"caption"``, ``"position"``, ``"ID-fallback"`` or
              ``"none"``).
            - ``"ID_text"``: datanumber of the matched TDText block
              (if available).
            - ``"ID_graph"``: datanumber of the TDGraph block.
            - ``"Caption_text"``: caption of the matched TDText block
              (if available).
            - ``"Caption_graph"``: caption of the TDGraph block.
            - ``"raw_metadata"``: dictionary returned by
              ``TDText.content(format="dict")`` containing the raw
              textual metadata.
        """
        caption_graph = wipfile.caption(graph.datanumber)

        # 1. Matching based on captions. Best-practice!!
        caption_matches = wipfile.find_related(
            graph.datanumber,
            by="caption",
            keep=["TDText"],
            sort_order="auto",
            soft_matches=True,
        )

        if caption_matches:
            text_datanum = caption_matches[0]
            match_type = "caption"
        else:
            # 2. If that fails, matching by position overlap (via photonicdata_files_wip)
            position_matches = wipfile.find_related(
                graph.datanumber,
                by="position",
                keep=["TDText"],
                sort_order="auto",
            )

            if position_matches:
                text_datanum = position_matches[0]
                match_type = "position"
            else:
                # 3. As a last resort, match by nearest datanumber (ID). This is bad-practice!!
                all_text_dns = [
                    block.datanumber for block in all_blocks if isinstance(block, TDText)
                ]

                if not all_text_dns:
                    self.logger.warning(
                        "No TDText blocks found for metadata matching"
                    )
                    filename = (
                        wipfile.filename
                        if wipfile.filename is not None
                        else "unknown.wip"
                    )
                    return {
                        "filename": Path(filename).name,
                        "match_type": "none",
                        "Caption_graph": caption_graph,
                        "raw_metadata": {},
                    }

                text_datanum = min(
                    all_text_dns, key=lambda n: abs(n - graph.datanumber)
                )
                match_type = "ID-fallback"

        text_obj = cast(TDText, wipfile.data_td(text_datanum))

        metadata = {
            "filename": Path(wipfile.filename).name,  # type: ignore[arg-type]
            "match_type": match_type,
            "ID_text": text_obj.datanumber,
            "ID_graph": graph.datanumber,
            "Caption_text": wipfile.caption(text_datanum),
            "Caption_graph": caption_graph,
            "raw_metadata": text_obj.content(format="dict"),
        }

        self.logger.debug(
            "Matched graph %d to text %d using %s method",
            graph.datanumber,
            text_obj.datanumber,
            match_type,
        )

        return metadata
