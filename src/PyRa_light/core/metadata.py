"""Metadata normalization and unification for Raman spectra."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

MetaValue = str | int | float | bool | None
MetadataDict = dict[str, MetaValue]
ConverterFunc = Callable[[Any], Optional[MetaValue]]

WAVENUMBER_TO_NM_FACTOR = 1e7
MS_TO_S_FACTOR = 1000.0
C_TO_K_FACTOR = 273.15

# ------------------------------------------------------------------------- #
# MetadataField
# ------------------------------------------------------------------------- #

class MetadataField(Enum):
    """Define standardized metadata field names.

    This enumeration lists all the keys used for normalized metadata
    fields in Raman spectra.
    """

    # Identification
    SAMPLE_NAME = "sample_name"
    SAMPLE_TYPE = "sample_type"  # calibration, background, sample
    MEASUREMENT_DATETIME = "measurement_datetime"

    # Instrument Configuration
    INSTRUMENT_MANUFACTURER = "instrument_manufacturer"
    INSTRUMENT_MODEL = "instrument_model"
    EXCITATION_WAVELENGTH_NM = "excitation_wavelength_nm"
    MICROSCOPE_TYPE = "microscope_type"
    DETECTOR_TYPE = "detector_type"
    LASER_TYPE = "laser_type"

    # Acquisition
    INTEGRATION_TIME_S = "integration_time_s"
    ACCUMULATIONS = "accumulations"
    LASER_POWER_MW = "laser_power_mw"
    SCATTERING_CONFIGURATION_PORTO = "scattering_configuration_porto"
    POLARIZATION_STATE = "polarization_state"
    SPECTRAL_RESOLUTION_CM_1 = "spectral_resolution_cm_1"

    # Environment
    ENV_TEMPERATURE_K = "environment_temperature_K"
    ENV_PRESSURE_KPA = "environment_pressure_kPa"
    ENV_ATMOSPHERE = "environment_atmosphere"
    DETECTOR_TEMPERATURE_K = "detector_temperature_K"

    # Optics
    OBJECTIVE_NAME = "objective_name"
    OBJECTIVE_TYPE = "objective_type"
    OBJECTIVE_MAGNIFICATION = "objective_magnification"
    OBJECTIVE_APERTURE_UM = "objective_aperture"
    SPOT_SIZE_UM = "spot_size_um"
    CONFOCAL_APERTURE_UM = "confocal_aperture_um"
    GRATING_GROOVES_MM = "grating"


    # Spectra
    SPECTRAL_CENTER_REL_1_PER_CM = "spectral_center_rel_1_per_cm"
    RAMAN_SHIFT_MIN_CM_1 = "raman_shift_min_cm_1"
    RAMAN_SHIFT_MAX_CM_1 = "raman_shift_max_cm_1"

    # Corrections
    BACKGROUND_CORRECTED = "background_corrected"
    RAMAN_SCATTERING_CORRECTED = "raman_scattering_corrected"
    SIGNAL_NORMALIZED = "normalized"
    BASELINE_CORRECTED = "baseline_corrected"
    COSMIC_RAY_REMOVAL = "cosmic_ray_removal"
    SMOOTHING = "smoothing"
    BACKGROUND_CORRECTION_METHOD = "background_correction_method"
    RAMAN_SCATTERING_CORRECTION_METHOD = "raman_scattering_correction_method"
    SIGNAL_NORMALIZATION_METHOD = "normalization_method"
    BASELINE_CORRECTION_METHOD = "baseline_correction_method"
    COSMIC_RAY_REMOVAL_METHOD = "cosmic_ray_removal_method"
    SMOOTHING_METHOD = "smoothing_method"


def _to_int(value: Any) -> Optional[int]:
    """Convert a value to integer if possible.

    Parameters
    ----------
    value : Any
        The input value to convert.

    Returns
    -------
    int or None
        Integer representation of value, or None if conversion fails.
    """
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None

    if value == "":
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        logger.debug("Failed to convert %s to int", value)
        return None

def _to_float(value: Any) -> Optional[float]:
    """Convert a value to float if possible.

    Parameters
    ----------
    value : Any
        The input value to convert.

    Returns
    -------
    float or None
        Float representation of value, or None if conversion fails.
    """
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None

    if value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        logger.debug("Failed to convert %s to float", value)
        return None


def _to_bool(value: Any) -> Optional[bool]:
    """Convert a value to boolean for recognized literals.

    Parameters
    ----------
    value : Any
        The input value to convert.

    Returns
    -------
    bool or None
        Boolean representation for recognized true/false strings,
        numeric 1/0, and bools, or None if input is None, empty, or
        unrecognized.
    """
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None

    if value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        val = value.lower()
        if val in ("true", "yes", "1", "on"):
            return True
        if val in ("false", "no", "0", "off"):
            return False
        return None
    return None


def _wavenumber_to_nm(value: Any) -> Optional[float]:
    """Convert wavenumber (1/cm) to wavelength in nanometers.

    Parameters
    ----------
    value : Any

    Returns
    -------
    float or None
    """
    wn = _to_float(value)
    if wn is not None and wn > 0:
        return round(WAVENUMBER_TO_NM_FACTOR / wn, 3)
    return None


def _ms_to_s(value: Any) -> Optional[float]:
    """Convert milliseconds to seconds.

    Parameters
    ----------
    value : Any

    Returns
    -------
    float or None
    """
    ms = _to_float(value)
    if ms is not None:
        return round(ms / MS_TO_S_FACTOR, 3)
    return None

def _c_to_k(value: Any) -> Optional[float]:
    """Convert Celcius to Kelvin.

    Parameters
    ----------
    value : Any

    Returns
    -------
    float or None
    """
    c = _to_float(value)
    if c is not None:
        return round(c + C_TO_K_FACTOR, 2)
    return None

# ------------------------------------------------------------------------- #
# FieldMapping
# ------------------------------------------------------------------------- #

@dataclass
class FieldMapping:
    """Store mapping configuration for a metadata field.

    Attributes
    ----------
    field : MetadataField
        The standardized metadata field.
    aliases : list[str]
        List of vendor-specific or alternative keys mapping to this field.
    converter : Optional[ConverterFunc], optional
        Function to convert raw values to standardized form, by default None.
    """

    field: MetadataField
    aliases: list[str]
    converter: Optional[ConverterFunc] = None

    def convert(self, value: Any) -> Optional[MetaValue]:
        """Convert a raw metadata value using the mapping's converter.

        Parameters
        ----------
        value : Any
            Raw metadata value.

        Returns
        -------
        MetaValue or None
            Converted metadata value, or None if empty.
        """
        if self.converter and value not in (None, ""):
            return self.converter(value)
        return value if value not in (None, "") else None

# ------------------------------------------------------------------------- #
# MetadataMapper
# ------------------------------------------------------------------------- #

class MetadataMapper:
    """Normalize and map raw metadata keys to standardized fields.

    Builds an alias-to-field mapping and applies conversions to unify
    metadata from various vendor formats.
    """

    def __init__(self):
        """Initialize the mapper and build alias lookup index.

        Populates a dict mapping lowercase alias strings to FieldMapping.
        Later mappings override earlier ones for alias conflicts.
        """
        self._alias_to_mapping: dict[str, FieldMapping] = {}
        for mapping in FIELD_MAPPINGS:
            for alias in mapping.aliases:
                self._alias_to_mapping[alias.lower()] = mapping

    def map_metadata(self, raw_meta: dict[str, Any]) -> MetadataDict:
        """Normalize raw metadata keys to standardized field names.

        Parameters
        ----------
        raw_meta : dict[str, Any]
            Original metadata dictionary from converter output.

        Returns
        -------
        MetadataDict
            Mapped metadata with standardized keys.
        """
        mapped: MetadataDict = {field.value: None for field in MetadataField}

        for raw_key, raw_value in raw_meta.items():
            mapping = self._alias_to_mapping.get(raw_key.lower())
            if mapping:
                converted_value = mapping.convert(raw_value)
                if converted_value is not None:
                    # Only update if we don't already have a value
                    if mapped[mapping.field.value] is None:
                        mapped[mapping.field.value] = converted_value

        self._handle_datetime_combinations(raw_meta, mapped)

        return mapped

    def _handle_datetime_combinations(
        self,
        raw_meta: dict[str, Any],
        mapped: MetadataDict,
    ) -> None:
        """Combine separate date and time fields if measurement_datetime missing.

        Parameters
        ----------
        raw_meta : dict[str, Any]
            Raw metadata dictionary.
        mapped : MetadataDict
            Current mapped metadata to update.
        """
        if mapped[MetadataField.MEASUREMENT_DATETIME.value] is not None:
            return

        lower_meta: dict[str, Any] = {}
        for key, value in raw_meta.items():
            if isinstance(key, str):
                lower_meta[key.lower()] = value

        date_time_pairs = [
            ("start date", "start time"),
            ("start_date", "start_time"),
            ("acquisition date", "acquisition time"),
            ("acquisition_date", "acquisition_time"),
        ]

        for date_key, time_key in date_time_pairs:
            date_str = lower_meta.get(date_key, "") or ""
            time_str = lower_meta.get(time_key, "") or ""
            combined = f"{date_str} {time_str}".strip()
            if combined:
                mapped[MetadataField.MEASUREMENT_DATETIME.value] = combined
                return


def unify_metadata(raw_meta: dict[str, Any]) -> MetadataDict:
    """Unify raw metadata into a standardized format.

    This is the main entry point for metadata normalization.

    Parameters
    ----------
    raw_meta : dict[str, Any]
        Raw metadata dictionary from converter output.

    Returns
    -------
    MetadataDict
        Unified metadata with standardized keys and converted values.
    """
    mapper = MetadataMapper()
    return mapper.map_metadata(raw_meta)


# ============================================================================= #
# ============================================================================= #
#                          F I E L D   M A P P I N G S
# ============================================================================= #
# ============================================================================= #

FIELD_MAPPINGS = [

    # ------------------------------------------------------------------------- #
    # Identification
    # ------------------------------------------------------------------------- #

    FieldMapping(
        MetadataField.SAMPLE_NAME,
        [
            "sample_name",
            "sample",
            "sample name",
            "caption_graph",
            "caption_text",
            "caption",
            "label",
            "title",
            "sample id",
            "sample_id",
        ],
    ),

    FieldMapping(
        MetadataField.SAMPLE_TYPE,
        [
            "sample_type",
            "sample type",
            "type",
            "measurement_type",
            "measurement type",
        ],
    ),

    # ------------------------------------------------------------------------- #
    # Timing
    # ------------------------------------------------------------------------- #

    FieldMapping(
        MetadataField.MEASUREMENT_DATETIME,
        [
            "measurement_datetime",
            "measurement datetime",
            "datetime",
            "date_time",
            "measurement_date",
            "measurement time",
            "measurement_time",
            "acquisition_date_and_time",
            "acquisition_date_time",
            "acquisition timestamp",
            "acquisition_timestamp",
        ],
    ),

    # ------------------------------------------------------------------------- #
    # Instrument configuration
    # ------------------------------------------------------------------------- #

    FieldMapping(
        MetadataField.INSTRUMENT_MANUFACTURER,
        [
            "instrument_manufacturer",
            "instrument manufacturer",
            "manufacturer",
            "manufacturer_name",
            "spectrometer manufacturer",
            "instrument vendor",
            "vendor",
        ],
    ),

    FieldMapping(
        MetadataField.INSTRUMENT_MODEL,
        [
            "instrument_model",
            "instrument model",
            "model",
            "model_name",
            "model name",
            "device model",
            "instrument",
        ],
    ),

    FieldMapping(
        MetadataField.EXCITATION_WAVELENGTH_NM,
        [
            "excitation_wavelength_nm",
            "excitation_nm",
            "laser_wavelength_nm",
            "excitation wavelength [nm]",
            "raman laser wavelength",
            "source setting",
            "channel_2_laser_wavelength",
            "laser_wavelength",
            "laser wavelength",
            "source wavelength",
        ],
        converter=_to_float,
    ),

    FieldMapping(
        MetadataField.EXCITATION_WAVELENGTH_NM,
        [
            "raman laser wavenumber",
            "excitation_wavelength_wn",
            "excitation_wn",
            "laser_wavelength_wn",
            "excitation wavelength [cm-1]",
            "excitation wavenumber",
            "excitation wavenumber [cm-1]",
            "laser wavenumber",
            "source wavenumber",
            "wavenumber",
        ],
        converter=_wavenumber_to_nm,
    ),

    FieldMapping(
        MetadataField.MICROSCOPE_TYPE,
        [
            "microscope_type",
            "microscope type",
            "microscope",
            "microscope configuration",
            "microscope_config",
            "microscope setup",
            "microscope_setup",
            "measurement channel",
            "measurement_channel",
        ],
    ),

    FieldMapping(
        MetadataField.DETECTOR_TYPE,
        [
            "detector_type",
            "detector type",
            "detector",
            "detector model",
            "detector_model",
            "sensor",
            "sensor_type",
        ],
    ),

    FieldMapping(
        MetadataField.LASER_TYPE,
        [
            "laser_type",
            "laser type",
            "laser source type",
            "source type",
            "source_type",
            "laser source",
            "laser_source",
            "laser",
            "excitation source",
            "excitation_source",
        ],
    ),

    # ------------------------------------------------------------------------- #
    # Acquisition
    # ------------------------------------------------------------------------- #

    FieldMapping(
        MetadataField.INTEGRATION_TIME_S,
        [
            "integration_time_s",
            "integration_time",  # another gamble...might be removed
            "integration time",
            "integration time [s]",
            "integration time (s)",
            "exposure time (s)",
            "exposure time [s]",
            "acquisition time (s)",
            "acquisition time [s]",
            "exposure length (s)",
            "exposure length [s]",
        ],
        converter=_to_float,
    ),

    FieldMapping(
        MetadataField.INTEGRATION_TIME_S,
        [
            "integration_time_ms",
            "integration time [ms]",
            "integration time (ms)",
            "exposure time (ms)",
            "exposure time [ms]",
            "acquisition time (ms)",
            "acquisition time [ms]",
            "exposure length (ms)",
            "exposure length [ms]",
        ],
        converter=_ms_to_s,
    ),

    FieldMapping(
        MetadataField.ACCUMULATIONS,
        [
            "accumulations",
            "num_accumulations",
            "number_accumulations",
            "accumulation_count",
            "averages",
            "accumulation number",
            "accumulation_number",
            "number of accumulations",
            "number of scans",
            "number of sample scans",
            "number_of_scans",
            "number_scans",
        ],
        converter=_to_int,
    ),

    FieldMapping(
        MetadataField.LASER_POWER_MW,
        [
            "laser_power_mw",
            "laser_power",
            "raman laser power in mw",
            "raman nominal laser power in mw",
            "nominal laser power",
            "laser power a sample",
            "power at sample",
            "laser power (mW)",
            "laser power [mW]",
            "laser power mW",
        ],
        converter=_to_float,
    ),

    FieldMapping(
        MetadataField.SCATTERING_CONFIGURATION_PORTO,
        [
            "scattering_configuration_porto",
            "scattering configuration porto",
            "porto notation",
            "porto",
            "porto_notation",
            "scattering configuration",
            "scattering_configuration",
            "scattering geometry",
            "scattering_geometry",
            "geometry",
        ],
    ),

    FieldMapping(
        MetadataField.POLARIZATION_STATE,
        [
            "polarization_state",
            "polarization state",
            "polarization",
            "polarisation_state",
            "polarisation state",
            "polarisation",
            "pol_state",
            "pol",
            "polarization configuration",
            "polarisation configuration",
        ],
    ),

    FieldMapping(
        MetadataField.SPECTRAL_RESOLUTION_CM_1,
        [
            "spectral_resolution_cm_1",
            "spectral resolution [cm-1]",
            "spectral_resolution",
            "resolution [rel. 1/cm]",
            "spectral resolution",
            "spectral_res",
            "resolution (cm-1)",
            "resolution_cm_1",
            "resolving_power",
            "spectral resolving power",
        ],
        converter=_to_float,
    ),

    # ------------------------------------------------------------------------- #
    # Environment
    # ------------------------------------------------------------------------- #

    FieldMapping(
        MetadataField.ENV_TEMPERATURE_K,
        [
            "environment_temperature_K",
            "sample_temperature_K",
            "sample temperature [K]",
            "temperature [K]",
            "temp [K]",
            "stage_temperature_K",
            "stage temperature [K]",
            "temperature_K",
            "temp_K",
        ],
        converter=_to_float,
    ),

    FieldMapping(
        MetadataField.ENV_TEMPERATURE_K,
        [
            "environment_temperature_C",
            "environment_temperature",
            "sample_temperature_C",
            "sample temperature [°C]",
            "temperature [°C]",
            "temp [C]",
            "stage_temperature_C",
            "stage temperature [C]",
            "temperature_C",
            "temp_C",
        ],
        converter=_c_to_k,
    ),

    FieldMapping(
        MetadataField.ENV_PRESSURE_KPA,
        [
            "environment_pressure_kPa",
            "environment pressure [kPa]",
            "sample_pressure_kPa",
            "pressure_kPa",
            "pressure [kPa]",
            "sample pressure [kPa]",
            "gas_pressure_kPa",
            "chamber_pressure_kPa",
            "cell_pressure_kPa",
            "pressure",
        ],
        converter=_to_float,
    ),

    FieldMapping(
        MetadataField.ENV_ATMOSPHERE,
        [
            "environment_atmosphere",
            "atmosphere",
            "environment",
            "ambient atmosphere",
            "ambient",
        ],
    ),

    FieldMapping(
        MetadataField.DETECTOR_TEMPERATURE_K,
        [
            "detector_temperature_K",
            "detector temperature [K]",
            "temperature [°K]",
            "sensor temperature [°K]",
            "sensor temperature [K]",
            "ccd temperature [K]",
            "ccd_temp_K",
            "detector temp [K]",
            "detector temp K",
            "sensor temp K",
        ],
        converter=_to_float,
    ),

    FieldMapping(
        MetadataField.DETECTOR_TEMPERATURE_K,
        [
            "ccd_temp_c",
            "ccd_temperature_c",
            "detector_temperature",  # that's a gamble if it's in °C! Most vendors do so tho!
            "temperature [°c]",
            "ccd temperature",  # that's a gamble too!
            "detector_temperature_c",
            "sensor temperature [°c]",
            "sensor_temp_c",
            "sensor temp [°c]",
            "detector temp [°c]",
        ],
        converter=_c_to_k,
    ),

    # ------------------------------------------------------------------------- #
    # Optics
    # ------------------------------------------------------------------------- #

    FieldMapping(
        MetadataField.OBJECTIVE_NAME,
        [
            "objective_name",
            "objective",
            "objective name",
            "objective_label",
            "objective label",
            "objective description",
            "objective_lens",
            "lens name",
            "lens_label",
        ],
    ),

    FieldMapping(
        MetadataField.OBJECTIVE_TYPE,
        [
            "objective_type",
            "objective type",
            "objective_class",
            "objective class",
            "lens type",
            "lens_type",
            "immersion type",
            "immersion_type",
        ],
    ),

    FieldMapping(
        MetadataField.OBJECTIVE_MAGNIFICATION,
        [
            "objective_magnification",
            "objective_mag",
            "objective magnification",
            "magnification",
            "mag",
            "objective_x",
            "magnification_x",
            "mag_x",
        ],
        converter=_to_float,
    ),

    FieldMapping(
        MetadataField.OBJECTIVE_APERTURE_UM,
        [
            "objective_aperture",
            "objective aperture [µm]",
            "objective_aperture_um",
            "numerical_aperture",
            "aperture",
            "aperture_um",
            "aperture [um]",
            "objective aperture",
            "objective aperture [um]",
            "objective aperture (um)",
            "aperture setting",
            "aperture_setting",
        ],
    ),

    FieldMapping(
        MetadataField.SPOT_SIZE_UM,
        [
            "spot_size_um",
            "spot size [µm]",
            "spot size",
            "spot diameter",
            "spot_diameter_um",
            "laser_spot_size",
            "laser spot [µm]",
            "spot [µm]",
            "spot (µm)",
            "beam_size_um",
        ],
        converter=_to_float,
    ),

    FieldMapping(
        MetadataField.CONFOCAL_APERTURE_UM,
        [
            "confocal_aperture_um",
            "confocal aperture [µm]",
            "confocal_aperture",
            "confocal aperture [um]",
            "confocal aperture (um)",
            "pinhole size",
            "pinhole_size",
            "confocal pinhole",
            "pinhole diameter [µm]",
            "pinhole_diameter_um",
        ],
        converter=_to_float,
    ),

    FieldMapping(
        MetadataField.GRATING_GROOVES_MM,
        [
            "grating_part_number",
            "grating name",
            "grating grooves/mm",
            "grating_grooves_per_mm",
            "grooves per mm",
            "grooves_per_mm",
            "lines_per_mm",
            "grating_pitch_density",
            "grating (g/mm)",
            "grating [g/mm]",
        ],
    ),

    # ------------------------------------------------------------------------- #
    # Spectra
    # ------------------------------------------------------------------------- #

    FieldMapping(
        MetadataField.SPECTRAL_CENTER_REL_1_PER_CM,
        [
            "spectral_center_rel_1_per_cm",
            "spectral_center_cm",
            "spectral_center",
            "spectral center [rel. 1/cm]",
            "center wavenumber",
            "center_raman_shift",
            "center [cm-1]",
            "central wavenumber",
            "spectral_center_wavenumber",
            "center position [1/cm]",
        ],
        converter=_to_float,
    ),

    FieldMapping(
        MetadataField.RAMAN_SHIFT_MIN_CM_1,
        [
            "raman_shift_min_cm_1",
            "raman shift min [cm-1]",
            "raman_shift_min",
            "min raman shift",
            "raman shift min (cm-1)",
            "min shift [cm-1]",
            "shift_min_cm_1",
            "min_raman_shift",
            "raman_shift_minimum",
            "minimum_raman_shift",
        ],
        converter=_to_float,
    ),

    FieldMapping(
        MetadataField.RAMAN_SHIFT_MAX_CM_1,
        [
            "raman_shift_max_cm_1",
            "raman shift max [cm-1]",
            "raman_shift_max",
            "max raman shift",
            "raman shift max (cm-1)",
            "max shift [cm-1]",
            "shift_max_cm_1",
            "max_raman_shift",
            "raman_shift_maximum",
            "maximum_raman_shift",
        ],
        converter=_to_float,
    ),

    # ------------------------------------------------------------------------- #
    # Corrections and processing
    # ------------------------------------------------------------------------- #

    FieldMapping(
        MetadataField.BACKGROUND_CORRECTED,
        [
            "auto_background",
            "auto_bg",
            "automatic background",
            "background_subtracted",
            "background subtracted",
            "dark_subtracted",
            "background corrected",
            "raman background corrected",
        ],
        converter=_to_bool,
    ),

    FieldMapping(
        MetadataField.BACKGROUND_CORRECTION_METHOD,
        [
            "background_correction_method",
            "background correction method",
            "background_correction_algorithm",
            "background correction algorithm",
            "background_subtraction_method",
            "background subtraction method",
            "background_subtraction_algorithm",
            "background subtraction algorithm",
            "background removal method",
            "background_removal_method",
            "background removal algorithm",
            "background_removal_algorithm",
        ],
    ),

    FieldMapping(
        MetadataField.RAMAN_SCATTERING_CORRECTED,
        [
            "raman_scattering_corrected",
            "raman_corrected",
            "raman scattering corrected",
            "raman corrected intensity",
            "raman scattering corr",
            "raman_scattering_corr",
            "raman intensity corrected",
        ],
        converter=_to_bool,
    ),

    FieldMapping(
        MetadataField.RAMAN_SCATTERING_CORRECTION_METHOD,
        [
            "raman_scattering_correction_method",
            "raman scattering correction method",
            "raman_scattering_correction_algorithm",
            "raman scattering correction algorithm",
            "raman_correction_method",
            "raman correction method",
            "raman_correction_algorithm",
            "raman correction algorithm",
        ],
    ),

    FieldMapping(
        MetadataField.SIGNAL_NORMALIZED,
        [
            "normalized",
            "signal_normalized",
            "intensity_normalized",
            "normalized intensity",
            "normalised",
            "normalised intensity",
            "signal normalised",
        ],
        converter=_to_bool,
    ),

    FieldMapping(
        MetadataField.SIGNAL_NORMALIZATION_METHOD,
        [
            "normalization_method",
            "normalization method",
            "normalisation_method",
            "normalisation method",
            "signal_normalization_method",
            "signal normalization method",
            "intensity_normalization_method",
            "intensity normalization method",
            "normalization_algorithm",
            "normalisation_algorithm",
            "normalization algorithm",
            "normalisation algorithm",
        ],
    ),

    FieldMapping(
        MetadataField.BASELINE_CORRECTED,
        [
            "baseline_corrected",
            "baseline corrected",
            "baseline subtraction",
            "baseline_subtracted",
            "baseline removed",
            "baseline-corrected",
            "baseline_removed",
        ],
        converter=_to_bool,
    ),

    FieldMapping(
        MetadataField.BASELINE_CORRECTION_METHOD,
        [
            "baseline_correction_method",
            "baseline-correction_algorithm",
            "baseline_subtraction_method",
            "baseline_subtracted_algorithm",
            "baseline correction method",
            "baseline correction algorithm",
            "baseline subtraction method",
            "baseline subtracted algorithm",
        ],
    ),

    FieldMapping(
        MetadataField.COSMIC_RAY_REMOVAL,
        [
            "cosmic_ray_removal",
            "cosmic ray removal",
            "cosmic spike removal",
            "cosmic ray correction",
            "cosmic rays removed",
            "cosmic ray filter",
            "cosmic ray cleaned",
            "cosmic_ray_cleaned",
            "cosmic_ray_filter",
            "cosmic ray",
        ],
        converter=_to_bool,
    ),

    FieldMapping(
        MetadataField.COSMIC_RAY_REMOVAL_METHOD,
        [
            "cosmic_ray_removal_method",
            "cosmic ray removal method",
            "cosmic ray correction method",
            "cosmic_ray_correction_method",
            "cosmic ray filter method",
            "cosmic_ray_filter_method",
            "cosmic_ray_removal_algorithm",
            "cosmic ray removal algorithm",
            "cosmic ray correction algorithm",
            "cosmic_ray_correction_algorithm",
            "cosmic ray filter algorithm",
            "cosmic_ray_filter_algorithm",
        ],
    ),

    FieldMapping(
        MetadataField.SMOOTHING,
        [
            "smoothing",
            "smoothed",
            "smoothing applied",
            "smoothing filter",
            "filtering",
            "filter",
            "filtered",
        ],
        converter=_to_bool,
    ),

    FieldMapping(
        MetadataField.SMOOTHING_METHOD,
        [
            "filtering_method",
            "filtering method",
            "filter_method",
            "filter method",
            "smoothing method",
            "smoothing_method",
            "smoothing algorithm",
            "smoothing_algorithm",
            "filtering algorithm",
            "filtering_algorithm",
            "filter algorithm",
            "filter_algorithm",
        ],
    ),
]
