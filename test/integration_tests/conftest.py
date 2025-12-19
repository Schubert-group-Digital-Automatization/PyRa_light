"""
Shared fixtures and helpers for integration tests.

The integration suite runs against real fixture files under
`test/integration_tests/data/`. This conftest keeps test modules focused on
assertions by centralizing:

- fixture discovery (supported files, representative cases)
- robust loading (continue on optional-dependency ImportError)
- small pipeline helpers (ConvertedFile -> list[Ra])
"""

# pylint: disable=redefined-outer-name
# pylint: disable=import-error

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable, Iterable

import pytest

from PyRa_light.core import Ra
from PyRa_light.core.converted import ConvertedFile, ConvertedMeasurement
from PyRa_light.io.loader import load_file


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

INTEGRATION_ROOT = Path(__file__).resolve().parent
DATA_ROOT = INTEGRATION_ROOT / "data"
MERGE_DATA_ROOT = DATA_ROOT / "merge"

# The `data/merge/` folder is curated to contain a small set of mergeable examples!
_MAX_MERGE_FILES = 15

# ---------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------


def expected_format(path: Path) -> str | None:
    """
    Return a normalized format for supported fixture files.
    """
    suffix = path.suffix.lower()

    if suffix.startswith(".") and suffix[1:].isdigit():
        return "opus"

    mapping = {
        ".opus": "opus",
        ".spc": "spc",
        ".wdf": "wdf",
        ".wip": "witec",
        ".dx": "jcamp",
        ".jdx": "jcamp",
        ".jcamp": "jcamp",
    }
    return mapping.get(suffix)


def discover_supported_files(root: Path) -> list[Path]:
    """Return a stable, sorted list of supported fixture files under a root."""
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if p.is_file() and expected_format(p) is not None]
    return sorted(files)


def discover_cases(root: Path) -> list[tuple[Path, str]]:
    """Return (file_path, format) pairs for supported fixture files under a root."""
    cases: list[tuple[Path, str]] = []
    for p in discover_supported_files(root):
        fmt = expected_format(p)
        if fmt is not None:
            cases.append((p, fmt))
    return cases


def select_one_per_format(cases: Iterable[tuple[Path, str]]) -> list[tuple[Path, str]]:
    """Select one representative (path, fmt) case per format to keep runs fast."""
    chosen: dict[str, tuple[Path, str]] = {}
    for path, fmt in cases:
        if fmt not in chosen:
            chosen[fmt] = (path, fmt)
    return [chosen[k] for k in sorted(chosen)]


# ---------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------


def load_converted(file_path: Path) -> ConvertedFile | None:
    """
    Load a fixture file via the public loader.

    Returns None if an optional dependency for this particular file/format
    is missing (ImportError), so tests can continue with other fixtures.
    """
    try:
        converted = load_file(file_path)
    except ImportError:
        return None

    converted_path = Path(str(converted.path))
    assert converted_path.resolve() == file_path.resolve()
    return converted


def ras_from_converted(converted: ConvertedFile) -> list[Ra]:
    """
    Build one Ra per ConvertedMeasurement from a ConvertedFile.
    """
    if not hasattr(Ra, "from_converted_measurement"):
        raise RuntimeError("Ra.from_converted_measurement is not available")

    ras: list[Ra] = []
    for meas in converted.measurements:
        ra = Ra.from_converted_measurement(  # type: ignore[attr-defined]
            meas,
            filename=getattr(converted, "filename", None),
        )
        ras.append(ra)
    return ras


# ---------------------------------------------------------------------
# Public fixtures
# ---------------------------------------------------------------------

@pytest.fixture(scope="session")
def load_converted_file() -> Callable[[Path], ConvertedFile | None]:
    """
    Return a helper that loads a fixture file via the public loader.
    The helper returns None when optional dependencies are missing for a given
    fixture.
    """
    return load_converted


@pytest.fixture(scope="session")
def integration_data_root() -> Path:
    """Root directory containing integration fixture files."""
    return DATA_ROOT


@pytest.fixture(scope="session")
def integration_cases(integration_data_root: Path) -> list[tuple[Path, str]]:
    """All supported fixture cases under the integration data directory."""
    return discover_cases(integration_data_root)


@pytest.fixture(scope="session")
def representative_cases(integration_cases: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    """One representative case per supported format."""
    return select_one_per_format(integration_cases)


@pytest.fixture(scope="session")
def merge_files() -> list[Path]:
    """Supported fixture files under the merge data directory."""
    files = discover_supported_files(MERGE_DATA_ROOT)
    return files[:_MAX_MERGE_FILES]


@pytest.fixture(scope="session")
def first_merge_converted(merge_files: list[Path]) -> ConvertedFile:
    """
    First successfully loadable ConvertedFile from the merge fixture folder.

    Skips if the merge data folder is missing or nothing can be loaded due to
    missing optional dependencies.
    """
    if not MERGE_DATA_ROOT.exists():
        pytest.skip(f"Merge integration data directory not found: {MERGE_DATA_ROOT}")

    for p in merge_files:
        converted = load_converted(p)
        if converted is not None:
            return converted

    pytest.skip("No merge fixtures could be loaded (missing optional dependencies?)")


@pytest.fixture(scope="session")
def first_merge_measurement(first_merge_converted: ConvertedFile) -> ConvertedMeasurement:
    """First ConvertedMeasurement from the first merge ConvertedFile."""
    assert first_merge_converted.measurements
    return first_merge_converted.measurements[0]


@pytest.fixture(scope="session")
def merge_ra_pool(merge_files: list[Path]) -> list[Ra]:
    """
    List of Ra objects built from all loadable merge fixtures.

    Skips if unification is unavailable or fewer than 2 spectra can be built.
    """
    if not hasattr(Ra, "from_converted_measurement"):
        pytest.skip("Ra.from_converted_measurement is not available in this version")

    ras: list[Ra] = []

    for p in merge_files:
        converted = load_converted(p)
        if converted is None:
            continue
        ras.extend(ras_from_converted(converted))

    if len(ras) < 2:
        pytest.skip("Need at least two mergable spectra to run merge integration tests")

    return ras
