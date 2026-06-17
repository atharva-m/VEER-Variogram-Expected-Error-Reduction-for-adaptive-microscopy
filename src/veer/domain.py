"""Validated v2 domain contracts with v1-compatible defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, model_validator


PolicyName = Literal[
    "uniform",
    "random",
    "gradient",
    "uncertainty",
    "balance",
    "balance_v3",
    "balance_v3_attention",
    "balance_v3_residual_attention",
    "balance_v3_scheduled",
    "balance_v3_gated",
    "uncertainty_distance_sequential",
    "uncertainty_distance_one_anchor",
    "uncertainty_lookahead",
    "uncertainty_calibrated_guarded",
    "uncertainty_neural_guarded_selector_only",
    "uncertainty_neural_guarded_full_system",
    "bayesian_variance_reduction",
    "bayesian_guarded_lookahead",
    "bayesian_residual_roi_summary",
    "bayesian_residual_subtile",
    "bayesian_subtile_evidence_gated",
    "bayesian_morphology_fantasy_guarded",
    "bayesian_pareto_eivr_4x4_mean_tau090",
    "bayesian_pareto_eivr_4x4_mean_tau085",
    "bayesian_pareto_eivr_4x4_texture_tau090",
    "bayesian_pareto_eivr_4x4_texture_tau085",
    "bayesian_pareto_eivr_8x8_mean_tau090",
    "bayesian_pareto_eivr_8x8_mean_tau085",
    "bayesian_pareto_eivr_8x8_texture_tau090",
    "bayesian_pareto_eivr_8x8_texture_tau085",
    "bayesian_pareto_additive_eivr_4x4_mean_tau090_alpha1",
    "bayesian_pareto_additive_eivr_4x4_mean_tau090_alpha2",
    "bayesian_pareto_additive_eivr_4x4_mean_tau090_alpha5",
    "bayesian_pareto_additive_eivr_4x4_mean_tau090_alpha10",
    "oracle_composite_gain",
]
TaskMode = Literal["interface_imaging", "multiobjective_mapping", "corrosion_morphology_reconstruction"]
ObjectiveName = Literal[
    "interface", "gradient", "segregation", "inclusion", "clustering", "anomaly"
]
ActionType = Literal[
    "coarse_initial",
    "fine_tile",
    "repeat_tile",
    "raster_tile",
    "high_resolution_tile",
    "high_statistics_tile",
    "point_roi",
    "stop",
]


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScenarioConfig(DomainModel):
    width_nm: float = Field(gt=0)
    height_nm: float = Field(gt=0)
    grid_size: int = Field(ge=8)
    grid_shape: tuple[int, int] | None = None
    elements: list[str] = Field(min_length=1)
    fuel_composition: dict[str, float] | None = None
    cladding_composition: dict[str, float] | None = None
    rate_range: tuple[float, float]
    interface_center_fraction_range: tuple[float, float]
    curvature_fraction_range: tuple[float, float]
    slope_fraction_range: tuple[float, float]
    layer_thickness_nm_range: tuple[float, float]
    oxygen_enrichment_range: tuple[float, float] = (0.0, 0.0)
    texture_amplitude: float = Field(default=0.05, ge=0)
    multiobjective_patterns: bool = False
    inclusion_count_range: tuple[int, int] = (1, 3)
    inclusion_radius_nm_range: tuple[float, float] = (100.0, 260.0)
    anomaly_amplitude_range: tuple[float, float] = (0.15, 0.35)
    segregation_element: str = "O"

    @model_validator(mode="after")
    def validate_scenario(self) -> "ScenarioConfig":
        if len(set(self.elements)) != len(self.elements):
            raise ValueError("elements must be unique")
        if (self.fuel_composition is None) != (self.cladding_composition is None):
            raise ValueError("fuel_composition and cladding_composition must be supplied together")
        for name, composition in (
            ("fuel_composition", self.fuel_composition),
            ("cladding_composition", self.cladding_composition),
        ):
            if composition is None:
                continue
            if set(composition) != set(self.elements):
                raise ValueError(f"{name} must specify every configured element")
            if any(value < 0 for value in composition.values()):
                raise ValueError(f"{name} values must be non-negative")
            if not np.isclose(sum(composition.values()), 1.0, atol=1e-5):
                raise ValueError(f"{name} values must sum to 1")
        for name in (
            "rate_range",
            "interface_center_fraction_range",
            "curvature_fraction_range",
            "slope_fraction_range",
            "layer_thickness_nm_range",
            "oxygen_enrichment_range",
            "inclusion_radius_nm_range",
            "anomaly_amplitude_range",
        ):
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(f"{name} must be ordered low to high")
        if self.rate_range[0] <= 0 or self.layer_thickness_nm_range[0] <= 0:
            raise ValueError("rate and layer thickness ranges must be positive")
        if self.segregation_element not in self.elements:
            raise ValueError("segregation_element must be a configured element")
        if self.inclusion_count_range[0] < 0 or self.inclusion_count_range[0] > self.inclusion_count_range[1]:
            raise ValueError("inclusion_count_range must be ordered and non-negative")
        if self.grid_shape is not None and (self.grid_shape[0] < 8 or self.grid_shape[1] < 8):
            raise ValueError("scenario grid_shape dimensions must each be at least 8")
        return self

    @property
    def grid_rows(self) -> int:
        return self.grid_shape[0] if self.grid_shape else self.grid_size

    @property
    def grid_columns(self) -> int:
        return self.grid_shape[1] if self.grid_shape else self.grid_size


class InstrumentConfig(DomainModel):
    coarse_step_nm: float = Field(gt=0)
    fine_step_nm: float = Field(gt=0)
    coarse_y_step_nm: float | None = Field(default=None, gt=0)
    fine_y_step_nm: float | None = Field(default=None, gt=0)
    repeat_y_step_nm: float | None = Field(default=None, gt=0)
    tile_size_nm: float = Field(gt=0)
    coarse_dwell_ms: float = Field(gt=0)
    fine_dwell_ms: float = Field(gt=0)
    repeat_dwell_ms: float = Field(gt=0)
    allowed_step_sizes_nm: list[float] = Field(min_length=1)
    allowed_dwell_times_ms: list[float] = Field(min_length=1)
    psf_width_by_step_nm: dict[int, float]
    sensitivity: dict[str, float]
    background_rate: dict[str, float]
    action_overhead_ms: float = Field(ge=0)
    line_overhead_ms: float = Field(ge=0)
    pixel_overhead_ms: float = Field(ge=0)
    dose_coefficient: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_instrument(self) -> "InstrumentConfig":
        if self.fine_step_nm >= self.coarse_step_nm:
            raise ValueError("fine_step_nm must be smaller than coarse_step_nm")
        if self.repeat_dwell_ms <= self.fine_dwell_ms:
            raise ValueError("repeat_dwell_ms must exceed fine_dwell_ms")
        for value in (self.coarse_step_nm, self.fine_step_nm):
            if value not in self.allowed_step_sizes_nm:
                raise ValueError("coarse and fine steps must be allowed")
            if int(value) not in self.psf_width_by_step_nm:
                raise ValueError("a PSF width must exist for each active step size")
        for value in (self.coarse_y_step_nm, self.fine_y_step_nm, self.repeat_y_step_nm):
            if value is not None and value not in self.allowed_step_sizes_nm:
                raise ValueError("configured y step sizes must be allowed")
        for value in (self.coarse_dwell_ms, self.fine_dwell_ms, self.repeat_dwell_ms):
            if value not in self.allowed_dwell_times_ms:
                raise ValueError("configured dwell times must be allowed")
        if any(value <= 0 for value in self.sensitivity.values()):
            raise ValueError("sensitivities must be positive")
        if any(value < 0 for value in self.background_rate.values()):
            raise ValueError("background rates must be non-negative")
        return self


class Budget(DomainModel):
    max_scan_time_s: float = Field(gt=0)
    max_dose_proxy: float = Field(gt=0)


class ModelConfig(DomainModel):
    length_scale_fraction: float = Field(default=0.12, gt=0, le=1)
    max_training_points: int = Field(default=512, ge=16)
    max_observations: int = Field(default=1500, ge=16)


class TaskConfig(DomainModel):
    mode: TaskMode = "interface_imaging"
    data_semantics: Literal["counts", "uncalibrated_counts", "intensity_proxy"] = "counts"
    label_status: Literal["labeled", "derived", "unannotated"] = "derived"


class MorphologyConfig(DomainModel):
    reference_method: Literal["frozen_unsupervised"] = "frozen_unsupervised"
    state_model: Literal["spatial_gmm", "kmeans", "otsu"] = "spatial_gmm"
    front_extraction: Literal["boundary_contour"] = "boundary_contour"
    penetration_axis: Literal["x", "y"] = "x"
    surface_side: Literal["left", "right", "top", "bottom"] = "left"
    smoothing_sigma_px: float = Field(default=1.0, ge=0)
    minimum_altered_fraction: float = Field(default=0.005, ge=0, le=0.5)
    max_state_fit_points: int = Field(default=12000, ge=100)


class FoldConfig(DomainModel):
    outer_test_ranges: list[tuple[int, int]] = Field(
        default_factory=lambda: [(1, 53), (54, 106), (107, 159), (160, 212), (213, 265)]
    )
    outer_guard_slices: int = Field(default=3, ge=0)
    validation_slices: int = Field(default=4, ge=1)
    validation_guard_slices: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> "FoldConfig":
        if any(first <= 0 or last < first for first, last in self.outer_test_ranges):
            raise ValueError("outer test ranges must be positive inclusive ranges")
        return self


class AcquisitionConfig(DomainModel):
    roi_size_px: tuple[int, int] = Field(default=(64, 64))
    pilot_rois: int = Field(default=4, ge=2)
    total_rois: int = Field(default=17, ge=3)
    front_weight: float = Field(default=0.50, ge=0.0)
    penetration_d95_weight: float = Field(default=0.50, ge=0.0)
    rmse_regression_limit_fraction: float = Field(default=0.02, ge=0.0)
    excluded_channels: list[str] = Field(default_factory=lambda: ["CPS"])
    folds: FoldConfig = Field(default_factory=FoldConfig)
    # Deployment-oriented routing knobs. Defaults reproduce the validated
    # single-step, travel-blind behavior exactly (batch_size=1, travel cost=0).
    batch_size: int = Field(default=1, ge=1)
    travel_cost_ms_per_nm: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_acquisition(self) -> "AcquisitionConfig":
        if self.roi_size_px[0] <= 0 or self.roi_size_px[1] <= 0:
            raise ValueError("ROI dimensions must be positive")
        if self.total_rois <= self.pilot_rois:
            raise ValueError("total_rois must exceed pilot_rois")
        if self.front_weight + self.penetration_d95_weight <= 0:
            raise ValueError("at least one morphology-composite weight must be positive")
        return self


class VariogramConfig(DomainModel):
    residual_filter_sigma_px: float = Field(default=1.0, gt=0.0)
    latent_components: int = Field(default=4, ge=1)
    kernel_catalog: list[tuple[float, float]] = Field(
        default_factory=lambda: [
            (0.025, 0.025),
            (0.050, 0.050),
            (0.075, 0.075),
            (0.050, 0.100),
            (0.100, 0.050),
        ]
    )
    temper_reference_subtiles: int = Field(default=64, ge=1)
    robust_iqr_epsilon: float = Field(default=1.0e-6, gt=0.0)
    scaled_feature_clip: float = Field(default=8.0, gt=0.0)
    alpha_floor: float = Field(default=1.0e-6, gt=0.0)
    jitter: float = Field(default=1.0e-8, gt=0.0)
    front_bandwidth_nm: float = Field(default=1600.0, gt=0.0)
    front_gate_movement_fraction: float = Field(default=0.01, gt=0.0)
    nested_length_scale_grid: list[float] = Field(
        default_factory=lambda: [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
    )
    nested_bin_edges: list[float] = Field(
        default_factory=lambda: [
            0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.45, 0.7, 1.0,
        ]
    )
    nested_minimum_bin_pairs: int = Field(default=5, ge=2)
    trailing_window_iterations: int = Field(default=6, ge=1)

    @model_validator(mode="after")
    def validate_variogram(self) -> "VariogramConfig":
        if not self.kernel_catalog or any(
            not 0.0 < value <= 1.0 for pair in self.kernel_catalog for value in pair
        ):
            raise ValueError("variogram kernel length scales must be in (0, 1]")
        if not self.nested_length_scale_grid or any(
            not 0.0 < value <= 1.0 for value in self.nested_length_scale_grid
        ):
            raise ValueError("nested length-scale grid values must be in (0, 1]")
        edges = self.nested_bin_edges
        if len(edges) < 3 or any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)) or edges[0] <= 0.0:
            raise ValueError("nested bin edges must be positive and strictly increasing")
        return self


class SpectrumWindow(DomainModel):
    peak_range: tuple[float, float]
    background_ranges: list[tuple[float, float]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ranges(self) -> "SpectrumWindow":
        if self.peak_range[0] >= self.peak_range[1]:
            raise ValueError("spectral peak range must be increasing")
        if any(low >= high for low, high in self.background_ranges):
            raise ValueError("spectral background ranges must be increasing")
        return self


class ReplayCapabilities(DomainModel):
    native_step_nm: float | None = Field(default=None, gt=0)
    native_y_step_nm: float | None = Field(default=None, gt=0)
    native_dwell_ms: float | None = Field(default=None, gt=0)
    available_step_sizes_nm: list[float] = Field(default_factory=list)
    available_y_step_sizes_nm: list[float] = Field(default_factory=list)
    available_dwell_times_ms: list[float] = Field(default_factory=list)
    supported_action_types: list[ActionType] = Field(
        default_factory=lambda: ["coarse_initial", "raster_tile", "point_roi"]
    )
    has_repeat_measurements: bool = False
    supports_statistical_thinning: bool = True


class MapAlignmentConfig(DomainModel):
    method: Literal["strict", "configured_crop", "resample_to_reference"] = "strict"
    reference_element: str | None = None
    crops: dict[str, tuple[int, int, int, int]] = Field(default_factory=dict)
    assume_shared_extent: bool = False
    interpolation: Literal["nearest", "linear"] = "nearest"

    @model_validator(mode="after")
    def validate_alignment(self) -> "MapAlignmentConfig":
        for name, (row0, row1, column0, column1) in self.crops.items():
            if row0 < 0 or column0 < 0 or row1 <= row0 or column1 <= column0:
                raise ValueError(f"map crop for {name} must define a non-empty positive region")
        if self.method == "resample_to_reference" and not self.assume_shared_extent:
            raise ValueError(
                "resample_to_reference requires assume_shared_extent=true; "
                "resolution mismatches cannot be aligned implicitly"
            )
        return self


class DatasetConfig(DomainModel):
    mode: Literal["synthetic", "replay"] = "synthetic"
    adapter: Literal[
        "generic_element_map",
        "element_map_images",
        "binary_element_map",
        "ornl_usid_h5",
        "standardized_zarr",
    ] | None = None
    source: Path | None = None
    element_map_sources: dict[str, Path] = Field(default_factory=dict)
    map_alignment: MapAlignmentConfig = Field(default_factory=MapAlignmentConfig)
    value_semantics: Literal["counts", "uncalibrated_counts", "intensity_proxy"] = "counts"
    intensity_proxy_variance: float = Field(default=1.0, gt=0)
    binary_dtype: Literal["uint32_le", "uint16_le", "float32_le"] = "uint32_le"
    binary_dimensions_from_header: bool = True
    binary_shape: tuple[int, int] | None = None
    binary_data_offset_values: int = Field(default=2, ge=0)
    dataset_path: str | None = None
    energy_path: str | None = None
    grid_shape: tuple[int, int] | None = None
    spatial_crop_indices: tuple[int, int, int, int] | None = None
    retain_spectrum: bool = True
    x_step_nm: float | None = Field(default=None, gt=0)
    y_step_nm: float | None = Field(default=None, gt=0)
    dwell_ms: float | None = Field(default=None, gt=0)
    spectral_windows: dict[str, SpectrumWindow] = Field(default_factory=dict)
    capabilities: ReplayCapabilities = Field(default_factory=ReplayCapabilities)

    @model_validator(mode="after")
    def validate_source(self) -> "DatasetConfig":
        if self.mode == "replay" and self.adapter is None:
            raise ValueError("replay datasets require an adapter")
        if self.adapter == "element_map_images" and not self.element_map_sources:
            raise ValueError("element-map image ingestion requires element_map_sources")
        if self.adapter == "binary_element_map" and not self.element_map_sources:
            raise ValueError("binary element-map ingestion requires element_map_sources")
        if (
            self.adapter == "binary_element_map"
            and not self.binary_dimensions_from_header
            and self.binary_shape is None
        ):
            raise ValueError("binary element maps without header dimensions require binary_shape")
        if self.spatial_crop_indices is not None:
            row0, row1, column0, column1 = self.spatial_crop_indices
            if row0 < 0 or column0 < 0 or row1 <= row0 or column1 <= column0:
                raise ValueError("spatial_crop_indices must define a non-empty positive crop")
        return self


class ObjectiveConfig(DomainModel):
    weights: dict[ObjectiveName, float] = Field(default_factory=lambda: {"interface": 1.0})
    thresholds: dict[ObjectiveName, float] = Field(default_factory=dict)
    normalization: Literal["max", "robust"] = "max"

    @model_validator(mode="after")
    def validate_weights(self) -> "ObjectiveConfig":
        if not self.weights or all(value == 0 for value in self.weights.values()):
            raise ValueError("at least one objective weight must be positive")
        if any(value < 0 for value in self.weights.values()):
            raise ValueError("objective weights must be non-negative")
        return self

    @property
    def enabled(self) -> list[str]:
        return [name for name, weight in self.weights.items() if weight > 0]


class ActionPreset(DomainModel):
    preset_id: str
    action_type: ActionType
    step_size_nm: float = Field(gt=0)
    y_step_size_nm: float | None = Field(default=None, gt=0)
    dwell_time_ms: float = Field(gt=0)
    tile_size_nm: float | None = Field(default=None, gt=0)
    tile_height_nm: float | None = Field(default=None, gt=0)
    roi_size_nm: float | None = Field(default=None, gt=0)
    roi_height_nm: float | None = Field(default=None, gt=0)
    spatial_resolution_proxy: float | None = Field(default=None, gt=0)
    count_statistics_proxy: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_geometry(self) -> "ActionPreset":
        if self.action_type == "point_roi" and self.roi_size_nm is None:
            raise ValueError("point ROI presets require roi_size_nm")
        if self.action_type not in ("point_roi", "stop") and self.tile_size_nm is None:
            raise ValueError("raster presets require tile_size_nm")
        return self


class SafeOperatingLimits(DomainModel):
    approved_preset_ids: list[str] = Field(default_factory=list)
    excluded_regions_nm: list[tuple[float, float, float, float]] = Field(default_factory=list)


class QualityConfig(DomainModel):
    minimum_signal_to_background: float = Field(default=0.0, ge=0)
    maximum_relative_count_error: float = Field(default=1.0, gt=0)
    penalty_strength: float = Field(default=0.0, ge=0)
    hard_exclude_invalid: bool = True


class BenchmarkConfig(DomainModel):
    seeds: int = Field(default=30, ge=1)
    policies: list[PolicyName] = Field(
        default_factory=lambda: ["uniform", "random", "gradient", "uncertainty", "balance"]
    )


class RunConfig(DomainModel):
    schema_version: int = Field(default=2, ge=1)
    scenario: ScenarioConfig
    instrument: InstrumentConfig
    budget: Budget
    model: ModelConfig = Field(default_factory=ModelConfig)
    acquisition: AcquisitionConfig = Field(default_factory=AcquisitionConfig)
    variogram: VariogramConfig = Field(default_factory=VariogramConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    morphology: MorphologyConfig = Field(default_factory=MorphologyConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    objectives: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    action_presets: list[ActionPreset] = Field(default_factory=list)
    safety: SafeOperatingLimits = Field(default_factory=SafeOperatingLimits)
    quality: QualityConfig = Field(default_factory=QualityConfig)

    @model_validator(mode="after")
    def validate_elements(self) -> "RunConfig":
        expected = set(self.scenario.elements)
        if set(self.instrument.sensitivity) != expected:
            raise ValueError("instrument sensitivities must match scenario elements")
        if set(self.instrument.background_rate) != expected:
            raise ValueError("instrument backgrounds must match scenario elements")
        if self.dataset.adapter in ("element_map_images", "binary_element_map") and set(
            self.dataset.element_map_sources
        ) != expected:
            raise ValueError("element_map_sources must match scenario elements")
        if (
            "interface" in self.objectives.enabled
            or "segregation" in self.objectives.enabled
        ) and len(self.scenario.elements) < 2:
            raise ValueError("interface and segregation objectives require at least two elements")
        requires_endmembers = self.dataset.mode == "synthetic" or "interface" in self.objectives.enabled
        if requires_endmembers and (
            self.scenario.fuel_composition is None or self.scenario.cladding_composition is None
        ):
            raise ValueError("interface/synthetic experiments require fuel and cladding compositions")
        if (
            "interface" in self.objectives.enabled
            and self.scenario.fuel_composition == self.scenario.cladding_composition
        ):
            raise ValueError("interface objectives require distinct endmember compositions")
        if self.dataset.mode == "synthetic" and len(self.scenario.elements) < 2:
            raise ValueError("synthetic interface experiments require at least two elements")
        if self.instrument.tile_size_nm > min(
            self.scenario.width_nm, self.scenario.height_nm
        ):
            raise ValueError("tile size cannot exceed specimen field")
        preset_ids = [preset.preset_id for preset in self.action_presets]
        if len(set(preset_ids)) != len(preset_ids):
            raise ValueError("action preset ids must be unique")
        for preset in self.action_presets:
            if preset.step_size_nm not in self.instrument.allowed_step_sizes_nm:
                raise ValueError(f"preset {preset.preset_id} uses an unapproved step size")
            if (
                preset.y_step_size_nm is not None
                and preset.y_step_size_nm not in self.instrument.allowed_step_sizes_nm
            ):
                raise ValueError(f"preset {preset.preset_id} uses an unapproved y step size")
            if preset.dwell_time_ms not in self.instrument.allowed_dwell_times_ms:
                raise ValueError(f"preset {preset.preset_id} uses an unapproved dwell time")
        if self.safety.approved_preset_ids and not set(
            self.safety.approved_preset_ids
        ).issubset(preset_ids):
            raise ValueError("safe operating preset ids must refer to configured action presets")
        return self

    def resolved_action_presets(self) -> list[ActionPreset]:
        """Return configured v2 presets or v1-compatible defaults."""
        if self.action_presets:
            return self.action_presets
        instrument = self.instrument
        return [
            ActionPreset(
                preset_id="fine_tile",
                action_type="fine_tile",
                tile_size_nm=instrument.tile_size_nm,
                tile_height_nm=instrument.tile_size_nm,
                step_size_nm=instrument.fine_step_nm,
                y_step_size_nm=instrument.fine_y_step_nm,
                dwell_time_ms=instrument.fine_dwell_ms,
                spatial_resolution_proxy=instrument.fine_step_nm,
                count_statistics_proxy=instrument.fine_dwell_ms,
            ),
            ActionPreset(
                preset_id="repeat_tile",
                action_type="repeat_tile",
                tile_size_nm=instrument.tile_size_nm,
                tile_height_nm=instrument.tile_size_nm,
                step_size_nm=instrument.fine_step_nm,
                y_step_size_nm=instrument.repeat_y_step_nm or instrument.fine_y_step_nm,
                dwell_time_ms=instrument.repeat_dwell_ms,
                spatial_resolution_proxy=instrument.fine_step_nm,
                count_statistics_proxy=instrument.repeat_dwell_ms,
            ),
        ]


class MeasurementAction(DomainModel):
    action_id: str
    action_type: ActionType
    preset_id: str | None = None
    bounds_nm: tuple[float, float, float, float] | None = None
    roi_center_nm: tuple[float, float] | None = None
    roi_size_nm: float | None = Field(default=None, gt=0)
    step_size_nm: float | None = Field(default=None, gt=0)
    y_step_size_nm: float | None = Field(default=None, gt=0)
    dwell_time_ms: float | None = Field(default=None, gt=0)
    spatial_resolution_proxy: float | None = Field(default=None, gt=0)
    count_statistics_proxy: float | None = Field(default=None, gt=0)
    estimated_time_s: float = Field(default=0.0, ge=0)
    estimated_dose: float = Field(default=0.0, ge=0)
    pixel_count: int = Field(default=0, ge=0)
    supported_in_current_dataset: bool = True
    objective_gain_by_type: dict[str, float] = Field(default_factory=dict)
    quality_penalty: float = Field(default=0.0, ge=0)
    total_utility: float = 0.0
    constraint_status: Literal["pending", "valid", "invalid", "unsupported"] = "pending"

    @model_validator(mode="after")
    def validate_action(self) -> "MeasurementAction":
        if self.action_type == "stop":
            return self
        if self.bounds_nm is None or self.step_size_nm is None or self.dwell_time_ms is None:
            raise ValueError("measurement actions require bounds, step size, and dwell time")
        x0, x1, y0, y1 = self.bounds_nm
        if x1 <= x0 or y1 <= y0:
            raise ValueError("action bounds must have positive area")
        return self


class ConstraintViolation(DomainModel):
    action_id: str
    reason: str
    category: Literal["constraint", "quality", "capability"] = "constraint"


class MetricRecord(DomainModel):
    policy: str
    seed: int
    iteration: int
    scan_time_s: float
    dose_proxy: float
    interface_mean_distance_nm: float | None
    interface_p95_distance_nm: float | None
    rate_rmse: float
    normalized_channel_rmse: float
    composition_rmse: float | None
    negative_log_likelihood: float
    coverage_95: float
    weighted_pattern_score: float | None = None
    mean_quality_score: float | None = None
    invalid_candidate_count: int = 0
    unsupported_candidate_count: int = 0


class ObjectiveMetricRecord(DomainModel):
    policy: str
    seed: int
    iteration: int
    objective: str
    scan_time_s: float
    dose_proxy: float
    average_precision: float | None = None
    mean_interest: float
    mean_uncertainty: float


@dataclass(frozen=True)
class ObservationBatch:
    action: MeasurementAction
    data: xr.Dataset


@dataclass
class Recommendation:
    action: MeasurementAction
    utility: float
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[ConstraintViolation] = field(default_factory=list)
    unsupported: list[ConstraintViolation] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def as_record(self, iteration: int) -> dict[str, Any]:
        return {
            "iteration": iteration,
            "action_id": self.action.action_id,
            "action_type": self.action.action_type,
            "preset_id": self.action.preset_id,
            "utility": self.utility,
            "estimated_time_s": self.action.estimated_time_s,
            "estimated_dose": self.action.estimated_dose,
            "bounds_nm": str(self.action.bounds_nm),
            "objective_gain_by_type": str(self.action.objective_gain_by_type),
            "quality_penalty": self.action.quality_penalty,
            "reasons": "; ".join(self.reasons),
            "alternatives": str(self.alternatives),
            "rejected_count": len(self.rejected),
            "unsupported_count": len(self.unsupported),
        }


@dataclass
class ExperimentState:
    """State visible to policies; no simulated or replay reference truth is stored here."""

    observations: xr.Dataset | None = None
    consumed_time_s: float = 0.0
    consumed_dose: float = 0.0
    actions: list[MeasurementAction] = field(default_factory=list)
    decision_trace: list[dict[str, Any]] = field(default_factory=list)
    prediction: xr.Dataset | None = None
    quality_products: xr.Dataset | None = None
    pattern_products: xr.Dataset | None = None
    rejected_actions: list[dict[str, Any]] = field(default_factory=list)
    unsupported_proposals: list[dict[str, Any]] = field(default_factory=list)

    @property
    def observation_count(self) -> int:
        return 0 if self.observations is None else int(self.observations.sizes["observation"])

    def add_batch(self, batch: ObservationBatch) -> None:
        if self.observations is None:
            self.observations = batch.data
        else:
            self.observations = xr.concat(
                [self.observations, batch.data],
                dim="observation",
                data_vars="all",
                coords="minimal",
                compat="override",
            )
        self.actions.append(batch.action)
        self.consumed_time_s += batch.action.estimated_time_s
        self.consumed_dose += batch.action.estimated_dose
