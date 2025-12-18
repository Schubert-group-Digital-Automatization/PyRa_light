"""Unit tests for io.loader."""

# pylint: disable=redefined-outer-name
# pylint: disable=import-error

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

from PyRa_light.io import loader
from PyRa_light.io.converters_base import UnsupportedFormatError


def _patch_registry(
    monkeypatch: pytest.MonkeyPatch,
    get_converter: Callable[[Path], object],
) -> None:
    """Replace the default registry with a small stub for this test.

    The stub exposes only `get_converter` so tests can drive specific branches.
    """
    registry = SimpleNamespace(get_converter=get_converter)
    monkeypatch.setattr(loader, "get_default_registry", lambda: registry)


def _raise_no_converter(_: Path) -> object:
    """Raises ValueError to simulate a registry lookup failure for an unsupported suffix."""
    raise ValueError("no converter for suffix")


def test_get_converter_missing_file_raises(tmp_path: Path) -> None:
    """Missing paths should fail before any registry logic is touched."""
    missing_path = tmp_path / "does_not_exist.spc"
    with pytest.raises(FileNotFoundError):
        loader.get_converter(missing_path)


def test_get_converter_unknown_suffix_maps_to_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registry ValueError should be exposed as UnsupportedFormatError."""
    file_path = tmp_path / "file.unknown"
    file_path.write_text("x", encoding="utf-8")

    _patch_registry(monkeypatch, _raise_no_converter)

    with pytest.raises(UnsupportedFormatError, match="no converter"):
        loader.get_converter(file_path)


def test_get_converter_rejects_non_baseconverter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the registry returns the wrong type, loader should raise TypeError."""
    file_path = tmp_path / "file.spc"
    file_path.write_text("x", encoding="utf-8")

    _patch_registry(monkeypatch, lambda _: object())

    with pytest.raises(TypeError):
        loader.get_converter(file_path)


def test_load_file_returns_from_dict_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On success, load_file should call convert() and return ConvertedFile.from_dict()."""
    file_path = tmp_path / "file.spc"
    file_path.write_text("x", encoding="utf-8")

    class DummyConverter(loader.BaseConverter):
        """Minimal BaseConverter implementation for loader unit tests."""

        @property
        def format_name(self) -> str:
            """Human-readable format name used by the converter registry."""
            return "dummy"

        @property
        def supported_extensions(self) -> tuple[str, ...]: # pyright: ignore[reportIncompatibleMethodOverride]
            """File extensions supported by this converter."""
            return (".spc",)

        def convert(self, path: Path) -> dict: # pyright: ignore[reportIncompatibleMethodOverride]
            """Return a minimal converter-style dict for the given file.

            This keeps the test focused on loader wiring rather than file parsing.
            """
            assert path == file_path
            return {"Fileformat": "spc", "Filename": "file.spc", "Measurements": {}}

    _patch_registry(monkeypatch, lambda _: DummyConverter())

    out = loader.load_file(file_path)

    assert isinstance(out, loader.ConvertedFile)
    assert out.path == file_path
    assert out.fileformat == "spc"
    assert out.filename == "file.spc"
    assert out.measurements == []


def test_load_file_propagates_converter_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Converter exceptions should not be swallowed by the loader."""
    file_path = tmp_path / "file.spc"
    file_path.write_text("x", encoding="utf-8")

    class ExplodingConverter(loader.BaseConverter):
        """Converter stub that raises to verify exception propagation."""

        @property
        def format_name(self) -> str:
            """Human-readable format name used by the converter registry."""
            return "dummy"

        @property
        def supported_extensions(self) -> tuple[str, ...]: # pyright: ignore[reportIncompatibleMethodOverride]
            """File extensions supported by this converter."""
            return (".spc",)

        def convert(self, _: Path) -> dict:   # pyright: ignore[reportIncompatibleMethodOverride]
            """Raise a deterministic error to ensure the loader does not swallow exceptions."""
            raise RuntimeError("boom")

    _patch_registry(monkeypatch, lambda _: ExplodingConverter())

    with pytest.raises(RuntimeError, match="boom"):
        loader.load_file(file_path)
