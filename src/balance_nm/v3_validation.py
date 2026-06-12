"""V3 stack replay for morphology-front reconstruction without v2 ROI-max metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.ndimage import label

from .data import ingest_dataset
from .domain import RunConfig
from .io import load_config, write_config
from .v3_morphology import (
    dense_signal_from_observations,
    front_distance_metrics,
    front_from_probability,
    normalized_reconstruction_rmse,
    penetration_stat,
    pseudo_reference_from_dense_signal,
    reconstruct_from_observed_mask,
    robust_scale_signal,
)

V3PolicyName = Literal[
    "balance_v3",
    "balance_v3_attention",
    "balance_v3_residual_attention",
    "balance_v3_scheduled",
    "balance_v3_gated",
    "gradient",
    "uncertainty",
    "uniform",
    "random",
]
DEFAULT_V3_POLICIES: list[V3PolicyName] = [
    "uncertainty",
    "balance_v3_residual_attention",
    "balance_v3_attention",
    "balance_v3",
    "balance_v3_scheduled",
    "balance_v3_gated",
    "gradient",
    "uniform",
    "random",
]
V3_WEIGHT_KEYS = ("front_entropy", "penetration_variance", "reconstruction_uncertainty")


@dataclass
class V3SliceResult:
    config: RunConfig
    slice_id: str
    policy: str
    seed: int
    candidates: pd.DataFrame
    trace: pd.DataFrame
    metrics: pd.DataFrame
    reference: xr.Dataset
    final_prediction: xr.Dataset


def _axis_origins(size: int, span: int) -> list[int]:
    if span > size:
        raise ValueError("v3 ROI dimensions cannot exceed the dense map")
    origins = list(range(0, size - span + 1, span))
    if origins[-1] != size - span:
        origins.append(size - span)
    return origins


def build_v3_roi_catalog(x: np.ndarray, y: np.ndarray, roi_size_px: tuple[int, int]) -> pd.DataFrame:
    rows, columns = roi_size_px
    records = []
    for row0 in _axis_origins(len(y), rows):
        for column0 in _axis_origins(len(x), columns):
            row1, column1 = row0 + rows, column0 + columns
            records.append(
                {
                    "roi_id": f"r{row0:04d}_c{column0:04d}",
                    "row0": row0,
                    "row1": row1,
                    "column0": column0,
                    "column1": column1,
                    "center_x_nm": float((x[column0] + x[column1 - 1]) / 2.0),
                    "center_y_nm": float((y[row0] + y[row1 - 1]) / 2.0),
                    "pixel_count": int((row1 - row0) * (column1 - column0)),
                }
            )
    return pd.DataFrame(records)


def v3_config_for_slice(
    template: RunConfig, slice_id: str, element_sources: dict[str, Path] | None = None
) -> RunConfig:
    raw = template.model_dump(mode="python")
    if element_sources is not None:
        raw["dataset"]["element_map_sources"] = {
            element: str(element_sources[element]) for element in template.scenario.elements
        }
    else:
        for element, path in raw["dataset"]["element_map_sources"].items():
            raw["dataset"]["element_map_sources"][element] = (
                str(path).replace("slice_006", f"slice_{slice_id}").replace("006_0", f"{slice_id}_0")
            )
    raw["schema_version"] = max(int(raw.get("schema_version", 2)), 3)
    raw["dataset"]["spatial_crop_indices"] = None
    raw["dataset"]["value_semantics"] = "intensity_proxy"
    raw.setdefault("task", {})
    raw["task"].update(
        {
            "mode": "corrosion_morphology_reconstruction",
            "data_semantics": "intensity_proxy",
            "label_status": "unannotated",
        }
    )
    return RunConfig.model_validate(raw)


def sources_from_manifest(
    manifest_path: Path, slice_ids: list[str], elements: list[str]
) -> dict[str, dict[str, Path]]:
    manifest = pd.read_csv(manifest_path, dtype={"slice": str})
    manifest["slice"] = manifest["slice"].str.zfill(3)
    manifest["channel_upper"] = manifest["channel"].str.upper()
    sources: dict[str, dict[str, Path]] = {}
    for slice_id in slice_ids:
        subset = manifest[manifest["slice"] == slice_id]
        sources[slice_id] = {}
        for element in elements:
            match = subset[subset["channel_upper"] == element.upper()]
            if len(match) != 1:
                raise ValueError(f"manifest must contain exactly one {element} map for slice {slice_id}")
            path = Path(str(match.iloc[0]["local_path"]))
            if not path.exists():
                raise FileNotFoundError(path)
            sources[slice_id][element] = path
    return sources


def _raster_cost(config: RunConfig, roi: pd.Series) -> tuple[float, float]:
    rows = int(roi["row1"] - roi["row0"])
    pixels = int(roi["pixel_count"])
    dwell = config.dataset.dwell_ms or config.instrument.fine_dwell_ms
    time_ms = (
        config.instrument.action_overhead_ms
        + config.instrument.line_overhead_ms * rows
        + pixels * (dwell + config.instrument.pixel_overhead_ms)
    )
    dose = config.instrument.dose_coefficient * pixels * dwell
    return float(time_ms / 1000.0), float(dose)


def _normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    maximum = float(np.nanmax(values)) if values.size else 0.0
    if maximum <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return np.nan_to_num(values / maximum, nan=0.0)


def _penetration_variance_map(prediction: xr.Dataset) -> np.ndarray:
    variance = prediction["penetration_depth_variance_nm2"]
    if variance.dims == ("y",):
        return np.repeat(variance.values[:, None], prediction.sizes["x"], axis=1)
    return np.repeat(variance.values[None, :], prediction.sizes["y"], axis=0)


def _gradient_map(prediction: xr.Dataset) -> np.ndarray:
    signal = robust_scale_signal(prediction["mean_intensity"].values)
    gradient = np.zeros(signal.shape[1:], dtype=float)
    for channel in signal:
        dy, dx = np.gradient(channel)
        gradient += np.hypot(dx, dy)
    return gradient


def _weight_value(weights: dict[str, float], key: str) -> float:
    return float(weights.get(key, 0.0))


def _scheduled_weights(config: RunConfig, adaptive_query_index: int) -> dict[str, float]:
    denominator = max(config.acquisition_v3.adaptive_rois - 1, 1)
    progress = float(np.clip(adaptive_query_index / denominator, 0.0, 1.0))
    early = config.acquisition_v3.scheduled_weights.early
    late = config.acquisition_v3.scheduled_weights.late
    return {
        key: (1.0 - progress) * _weight_value(early, key) + progress * _weight_value(late, key)
        for key in V3_WEIGHT_KEYS
    }


def _front_continuity_score(front_mask: np.ndarray) -> float:
    if not np.any(front_mask):
        return 0.0
    labels, count = label(front_mask)
    if count == 0:
        return 0.0
    component_sizes = np.bincount(labels.ravel())[1:]
    if component_sizes.size == 0:
        return 0.0
    return float(component_sizes.max() / max(front_mask.sum(), 1))


def _morphology_reliability(prediction: xr.Dataset, observed_mask: np.ndarray, config: RunConfig) -> dict[str, float | bool]:
    shape = prediction["reconstruction_uncertainty"].shape
    altered = (
        prediction["altered_region_probability"].values
        if "altered_region_probability" in prediction
        else np.zeros(shape, dtype=float)
    )
    front = (
        prediction["alteration_front_probability"].values
        if "alteration_front_probability" in prediction
        else np.zeros(shape, dtype=float)
    )
    observed_area = float(np.mean(observed_mask))
    state_separation = float(np.percentile(altered, 90.0) - np.percentile(altered, 10.0))
    front_mask = front > 0.25
    front_support = float(np.mean(front_mask))
    continuity = _front_continuity_score(front_mask)
    gate = config.acquisition_v3.morphology_gate
    passed = (
        observed_area >= gate.minimum_observed_area_fraction
        and state_separation >= gate.minimum_state_separation_score
        and gate.minimum_front_support_fraction <= front_support <= gate.maximum_front_support_fraction
    )
    observed_ratio = observed_area / max(gate.minimum_observed_area_fraction, 1e-12)
    separation_ratio = state_separation / max(gate.minimum_state_separation_score, 1e-12)
    support_low_ratio = front_support / max(gate.minimum_front_support_fraction, 1e-12)
    support_high_ratio = (
        gate.maximum_front_support_fraction / max(front_support, 1e-12)
        if front_support > gate.maximum_front_support_fraction
        else 1.0
    )
    reliability = float(
        np.clip(
            min(observed_ratio, separation_ratio, support_low_ratio, support_high_ratio) * max(continuity, 1e-6),
            0.0,
            1.0,
        )
    )
    return {
        "morphology_gate_passed": bool(passed),
        "morphology_reliability_score": reliability,
        "state_separation_score": state_separation,
        "front_support_fraction": front_support,
        "front_continuity_score": continuity,
        "observed_area_fraction": observed_area,
    }


def _attention_maps(
    prediction: xr.Dataset,
    config: RunConfig,
    diagnostics: dict[str, float | bool] | None = None,
) -> dict[str, np.ndarray | float | bool]:
    """Build deterministic spatial attention from reconstructed multichannel morphology."""

    attention_config = config.acquisition_v3.attention
    shape = prediction["reconstruction_uncertainty"].shape
    if "mean_intensity" in prediction:
        signal = robust_scale_signal(prediction["mean_intensity"].values)
        contrast = _normalize_map(_gradient_map(prediction))
        if signal.shape[0] > 1:
            diversity = _normalize_map(np.std(signal, axis=0))
        else:
            diversity = np.zeros(shape, dtype=float)
    else:
        contrast = np.zeros(shape, dtype=float)
        diversity = np.zeros(shape, dtype=float)
    gate_passed = bool((diagnostics or {}).get("morphology_gate_passed", False))
    use_front = (not attention_config.require_morphology_gate_for_front) or gate_passed
    if use_front and "alteration_front_probability" in prediction:
        front = _normalize_map(prediction["alteration_front_probability"].values)
    else:
        front = np.zeros(shape, dtype=float)
    raw = (
        attention_config.contrast_weight * contrast
        + attention_config.channel_diversity_weight * diversity
        + attention_config.front_weight * front
    )
    attention = attention_config.floor + (1.0 - attention_config.floor) * _normalize_map(raw)
    attention = np.clip(np.nan_to_num(attention, nan=attention_config.floor), attention_config.floor, 1.0)
    return {
        "contrast_attention": contrast,
        "channel_diversity_attention": diversity,
        "front_attention": front,
        "simple_attention": attention,
        "attention_floor": float(attention_config.floor),
        "attention_gate_used": bool(use_front),
    }


def _attention_diagnostics(attention: dict[str, np.ndarray | float | bool]) -> dict[str, float | bool]:
    attention_map = attention["simple_attention"]
    return {
        "mean_attention": float(np.mean(attention_map)),
        "max_attention": float(np.max(attention_map)),
        "attention_floor": float(attention["attention_floor"]),
        "mean_contrast_attention": float(np.mean(attention["contrast_attention"])),
        "mean_channel_diversity_attention": float(
            np.mean(attention["channel_diversity_attention"])
        ),
        "mean_front_attention": float(np.mean(attention["front_attention"])),
        "attention_gate_used": bool(attention["attention_gate_used"]),
    }


def _empty_residual_attention_diagnostics() -> dict[str, float | bool]:
    return {
        "base_uncertainty_score": np.nan,
        "best_uncertainty_score": np.nan,
        "selected_base_uncertainty_ratio": np.nan,
        "residual_attention_score": np.nan,
        "residual_attention_center": np.nan,
        "residual_bonus_fraction": np.nan,
        "near_tie_candidate_count": np.nan,
        "residual_overrode_uncertainty_argmax": False,
    }


def _balance_weights_and_diagnostics(
    policy: str,
    prediction: xr.Dataset,
    config: RunConfig,
    observed_mask: np.ndarray,
    adaptive_query_index: int,
) -> tuple[dict[str, float], dict[str, float | bool]]:
    diagnostics = _morphology_reliability(prediction, observed_mask, config)
    if policy == "balance_v3_scheduled":
        weights = _scheduled_weights(config, adaptive_query_index)
    elif policy == "balance_v3_gated":
        if diagnostics["morphology_gate_passed"]:
            weights = {key: _weight_value(config.acquisition_v3.utility_weights, key) for key in V3_WEIGHT_KEYS}
        else:
            weights = {
                "front_entropy": 0.0,
                "penetration_variance": 0.0,
                "reconstruction_uncertainty": 1.0,
            }
    else:
        weights = {key: _weight_value(config.acquisition_v3.utility_weights, key) for key in V3_WEIGHT_KEYS}
    diagnostics.update(
        {
            "effective_front_entropy_weight": weights["front_entropy"],
            "effective_penetration_variance_weight": weights["penetration_variance"],
            "effective_reconstruction_uncertainty_weight": weights["reconstruction_uncertainty"],
            "mean_attention": np.nan,
            "max_attention": np.nan,
            "attention_floor": np.nan,
            "mean_contrast_attention": np.nan,
            "mean_channel_diversity_attention": np.nan,
            "mean_front_attention": np.nan,
            "attention_gate_used": False,
            **_empty_residual_attention_diagnostics(),
        }
    )
    return weights, diagnostics


def _empty_policy_diagnostics(observed_mask: np.ndarray) -> dict[str, float | bool]:
    return {
        "effective_front_entropy_weight": np.nan,
        "effective_penetration_variance_weight": np.nan,
        "effective_reconstruction_uncertainty_weight": np.nan,
        "morphology_gate_passed": False,
        "morphology_reliability_score": np.nan,
        "state_separation_score": np.nan,
        "front_support_fraction": np.nan,
        "front_continuity_score": np.nan,
        "observed_area_fraction": float(np.mean(observed_mask)),
        "mean_attention": np.nan,
        "max_attention": np.nan,
        "attention_floor": np.nan,
        "mean_contrast_attention": np.nan,
        "mean_channel_diversity_attention": np.nan,
        "mean_front_attention": np.nan,
        "attention_gate_used": False,
        **_empty_residual_attention_diagnostics(),
    }


def _policy_utility_map(
    policy: str,
    prediction: xr.Dataset,
    config: RunConfig,
    observed_mask: np.ndarray,
    adaptive_query_index: int = 0,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    if policy in {"balance_v3", "balance_v3_scheduled", "balance_v3_gated"}:
        weights, diagnostics = _balance_weights_and_diagnostics(
            policy, prediction, config, observed_mask, adaptive_query_index
        )
        front = _normalize_map(prediction["front_entropy"].values)
        penetration = _normalize_map(_penetration_variance_map(prediction))
        reconstruction = _normalize_map(prediction["reconstruction_uncertainty"].values)
        return (
            weights.get("front_entropy", 0.0) * front
            + weights.get("penetration_variance", 0.0) * penetration
            + weights.get("reconstruction_uncertainty", 0.0) * reconstruction
        ), diagnostics
    if policy == "balance_v3_attention":
        diagnostics = _morphology_reliability(prediction, observed_mask, config)
        diagnostics.update(
            {
                "effective_front_entropy_weight": 0.0,
                "effective_penetration_variance_weight": 0.0,
                "effective_reconstruction_uncertainty_weight": 1.0,
            }
        )
        attention = _attention_maps(prediction, config, diagnostics)
        attention_map = attention["simple_attention"]
        diagnostics.update(_attention_diagnostics(attention))
        diagnostics.update(_empty_residual_attention_diagnostics())
        reconstruction = _normalize_map(prediction["reconstruction_uncertainty"].values)
        return reconstruction * attention_map, diagnostics
    if policy == "gradient":
        return _normalize_map(_gradient_map(prediction)), _empty_policy_diagnostics(observed_mask)
    if policy == "uncertainty":
        diagnostics = _empty_policy_diagnostics(observed_mask)
        diagnostics.update(
            {
                "effective_front_entropy_weight": 0.0,
                "effective_penetration_variance_weight": 0.0,
                "effective_reconstruction_uncertainty_weight": 1.0,
            }
        )
        return _normalize_map(prediction["reconstruction_uncertainty"].values), diagnostics
    raise ValueError(f"policy {policy} has no score map")


def _candidate_score_from_map(utility_map: np.ndarray, roi: pd.Series, config: RunConfig) -> float:
    row0, row1 = int(roi["row0"]), int(roi["row1"])
    column0, column1 = int(roi["column0"]), int(roi["column1"])
    time_s, _ = _raster_cost(config, roi)
    return float(np.sum(utility_map[row0:row1, column0:column1]) / max(time_s, 1e-12))


def _candidate_mean_from_map(values: np.ndarray, roi: pd.Series) -> float:
    row0, row1 = int(roi["row0"]), int(roi["row1"])
    column0, column1 = int(roi["column0"]), int(roi["column1"])
    return float(np.mean(values[row0:row1, column0:column1]))


def _residual_attention_candidate_scores(
    candidates: pd.DataFrame,
    prediction: xr.Dataset,
    config: RunConfig,
    observed_mask: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float | bool]]:
    """Rank uncertainty-near-tied ROIs with a bounded positive attention residual."""

    diagnostics = _morphology_reliability(prediction, observed_mask, config)
    diagnostics.update(
        {
            "effective_front_entropy_weight": 0.0,
            "effective_penetration_variance_weight": 0.0,
            "effective_reconstruction_uncertainty_weight": 1.0,
        }
    )
    attention = _attention_maps(prediction, config, diagnostics)
    diagnostics.update(_attention_diagnostics(attention))
    reconstruction = _normalize_map(prediction["reconstruction_uncertainty"].values)
    base_scores = np.asarray(
        [_candidate_score_from_map(reconstruction, row, config) for _, row in candidates.iterrows()]
    )
    attention_scores = np.asarray(
        [_candidate_mean_from_map(attention["simple_attention"], row) for _, row in candidates.iterrows()]
    )
    residual_config = config.acquisition_v3.attention.residual
    best_base = float(np.max(base_scores))
    near_tie = base_scores >= residual_config.near_tie_ratio * best_base
    near_tie_attention = attention_scores[near_tie]
    center = float(np.median(near_tie_attention))
    denominator = max(float(np.max(near_tie_attention)) - center, 1e-12)
    scaled_attention = np.clip((attention_scores - center) / denominator, 0.0, 1.0)
    bonuses = np.where(
        near_tie,
        residual_config.maximum_bonus_fraction * scaled_attention,
        0.0,
    )
    utilities = base_scores * (1.0 + bonuses)
    selected_index = int(np.argmax(utilities))
    base_argmax_index = int(np.argmax(base_scores))
    selected_ratio = base_scores[selected_index] / best_base if best_base > 1e-12 else 1.0
    diagnostics.update(
        {
            "base_uncertainty_score": float(base_scores[selected_index]),
            "best_uncertainty_score": best_base,
            "selected_base_uncertainty_ratio": float(selected_ratio),
            "residual_attention_score": float(attention_scores[selected_index]),
            "residual_attention_center": center,
            "residual_bonus_fraction": float(bonuses[selected_index]),
            "near_tie_candidate_count": float(np.sum(near_tie)),
            "residual_overrode_uncertainty_argmax": bool(selected_index != base_argmax_index),
        }
    )
    scores = candidates[["roi_id"]].reset_index(drop=True).copy()
    scores["base_uncertainty_score"] = base_scores
    scores["residual_attention_score"] = attention_scores
    scores["near_tie_eligible"] = near_tie
    scores["residual_bonus_fraction"] = bonuses
    scores["residual_utility"] = utilities
    return scores, diagnostics


def _candidate_score(policy: str, prediction: xr.Dataset, roi: pd.Series, config: RunConfig) -> float:
    observed_mask = np.zeros(prediction["reconstruction_uncertainty"].shape, dtype=bool)
    utility_map, _ = _policy_utility_map(policy, prediction, config, observed_mask)
    return _candidate_score_from_map(utility_map, roi, config)


def _select_candidate(
    policy: str,
    candidates: pd.DataFrame,
    prediction: xr.Dataset,
    queried_ids: set[str],
    config: RunConfig,
    rng: np.random.Generator,
    observed_mask: np.ndarray,
    adaptive_query_index: int,
) -> tuple[pd.Series, float, str, dict[str, float | bool]]:
    eligible = candidates[~candidates["roi_id"].isin(queried_ids)]
    if eligible.empty:
        raise ValueError("no feasible v3 ROI candidates remain")
    if policy == "uniform":
        selected = eligible.sort_values(["row0", "column0"]).iloc[0]
        return selected, np.nan, "uniform_order", _empty_policy_diagnostics(observed_mask)
    if policy == "random":
        selected = eligible.iloc[int(rng.integers(0, len(eligible)))]
        return selected, np.nan, "random_selection", _empty_policy_diagnostics(observed_mask)
    if policy == "balance_v3_residual_attention":
        scores, diagnostics = _residual_attention_candidate_scores(
            eligible, prediction, config, observed_mask
        )
        selected_index = int(np.argmax(scores["residual_utility"].values))
        selected = eligible.iloc[selected_index]
        return (
            selected,
            float(scores.iloc[selected_index]["residual_utility"]),
            "balance_v3_residual_attention_uncertainty_reconstruction",
            diagnostics,
        )
    utility_map, diagnostics = _policy_utility_map(
        policy, prediction, config, observed_mask, adaptive_query_index
    )
    scores = np.asarray([_candidate_score_from_map(utility_map, row, config) for _, row in eligible.iterrows()])
    selected = eligible.iloc[int(np.argmax(scores))]
    return selected, float(np.max(scores)), f"{policy}_uncertainty_reconstruction", diagnostics


def _neighbor_candidates(candidates: pd.DataFrame, anchor: pd.Series, queried_ids: set[str]) -> list[pd.Series]:
    rows = sorted(candidates["row0"].unique().tolist())
    columns = sorted(candidates["column0"].unique().tolist())
    row_index = rows.index(int(anchor["row0"]))
    column_index = columns.index(int(anchor["column0"]))
    selected: list[pd.Series] = []
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            if row_offset == 0 and column_offset == 0:
                continue
            candidate_row = row_index + row_offset
            candidate_column = column_index + column_offset
            if not (0 <= candidate_row < len(rows) and 0 <= candidate_column < len(columns)):
                continue
            match = candidates[
                (candidates["row0"] == rows[candidate_row])
                & (candidates["column0"] == columns[candidate_column])
            ]
            if not match.empty and str(match.iloc[0]["roi_id"]) not in queried_ids:
                selected.append(match.iloc[0])
    return selected


def _absolute_stat_error(reference: float, predicted: float, fallback: float) -> float:
    if np.isfinite(reference) and np.isfinite(predicted):
        return float(abs(predicted - reference))
    if not np.isfinite(reference) and not np.isfinite(predicted):
        return 0.0
    return fallback


def _availability_status(reference_present: bool, predicted_present: bool) -> str:
    if reference_present and predicted_present:
        return "matched_present"
    if not reference_present and not predicted_present:
        return "matched_absent"
    return "missed_reference_feature" if reference_present else "spurious_predicted_feature"


def _score_prediction(
    config: RunConfig,
    policy: str,
    seed: int,
    slice_id: str,
    iteration: int,
    stage: str,
    observed_mask: np.ndarray,
    dense_signal: np.ndarray,
    reference: xr.Dataset,
    prediction: xr.Dataset,
    consumed_time_s: float,
    consumed_dose: float,
) -> dict[str, object]:
    x = prediction.coords["x"].values.astype(float)
    y = prediction.coords["y"].values.astype(float)
    predicted_front, predicted_penetration = front_from_probability(
        prediction["altered_region_probability"].values, x, y, config
    )
    distance = front_distance_metrics(reference["pseudo_front"].values, predicted_front, x, y)
    reference_penetration = reference["pseudo_penetration_depth_nm"].values
    field_span = float(x[-1] - x[0]) if config.morphology.penetration_axis == "x" else float(y[-1] - y[0])
    reference_d95 = penetration_stat(reference_penetration, "d95")
    predicted_d95 = penetration_stat(predicted_penetration, "d95")
    reference_dmax = penetration_stat(reference_penetration, "dmax")
    predicted_dmax = penetration_stat(predicted_penetration, "dmax")
    reference_penetration_present = bool(np.isfinite(reference_d95))
    predicted_penetration_present = bool(np.isfinite(predicted_d95))
    penetration_evaluable = reference_penetration_present and predicted_penetration_present
    return {
        "slice": slice_id,
        "policy": policy,
        "seed": seed,
        "iteration": iteration,
        "stage": stage,
        "scan_time_s": consumed_time_s,
        "dose_proxy": consumed_dose,
        "selected_area_fraction": float(observed_mask.mean()),
        "front_mean_symmetric_distance_nm": distance.mean_symmetric_distance_nm,
        "front_hausdorff_distance_nm": distance.hausdorff_distance_nm,
        "front_reference_present": distance.reference_front_present,
        "front_predicted_present": distance.predicted_front_present,
        "front_availability_status": distance.availability_status,
        "front_detection_correct": distance.detection_correct,
        "front_localization_evaluable": (
            distance.reference_front_present and distance.predicted_front_present
        ),
        "front_localization_mean_symmetric_distance_nm": (
            distance.localization_mean_symmetric_distance_nm
        ),
        "front_localization_hausdorff_distance_nm": distance.localization_hausdorff_distance_nm,
        "penetration_d95_absolute_error_nm": _absolute_stat_error(reference_d95, predicted_d95, field_span),
        "penetration_dmax_absolute_error_nm": _absolute_stat_error(reference_dmax, predicted_dmax, field_span),
        "penetration_reference_present": reference_penetration_present,
        "penetration_predicted_present": predicted_penetration_present,
        "penetration_availability_status": _availability_status(
            reference_penetration_present, predicted_penetration_present
        ),
        "penetration_detection_correct": reference_penetration_present == predicted_penetration_present,
        "penetration_localization_evaluable": penetration_evaluable,
        "penetration_d95_localization_absolute_error_nm": (
            abs(predicted_d95 - reference_d95) if penetration_evaluable else np.nan
        ),
        "penetration_dmax_localization_absolute_error_nm": (
            abs(predicted_dmax - reference_dmax) if penetration_evaluable else np.nan
        ),
        "mean_front_entropy": float(prediction["front_entropy"].mean()),
        "mean_reconstruction_uncertainty": float(prediction["reconstruction_uncertainty"].mean()),
        "normalized_reconstruction_rmse": normalized_reconstruction_rmse(
            dense_signal, prediction["mean_intensity"].values
        ),
    }


def run_v3_slice_replay(
    config: RunConfig,
    dense_signal: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    channels: list[str],
    slice_id: str,
    policy: V3PolicyName,
    seed: int = 0,
) -> V3SliceResult:
    """Run one leakage-free v3 reconstruction replay over a dense slice."""

    if policy not in DEFAULT_V3_POLICIES:
        raise ValueError(f"v3 policy must be one of {DEFAULT_V3_POLICIES}")
    candidates = build_v3_roi_catalog(x, y, config.acquisition_v3.roi_size_px)
    if config.acquisition_v3.pilot_rois > len(candidates):
        raise ValueError("pilot_rois exceeds the v3 ROI catalog")
    rng = np.random.default_rng(seed)
    observed_mask = np.zeros(dense_signal.shape[1:], dtype=bool)
    queried_ids: set[str] = set()
    trace_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    consumed_time_s = 0.0
    consumed_dose = 0.0
    reference = pseudo_reference_from_dense_signal(dense_signal, x, y, channels, config)
    prediction = reconstruct_from_observed_mask(dense_signal, observed_mask, x, y, channels, config)

    def reveal(
        roi: pd.Series,
        stage: str,
        utility: float = np.nan,
        anchor_id: str | None = None,
        diagnostics: dict[str, float | bool] | None = None,
    ) -> None:
        nonlocal prediction, consumed_time_s, consumed_dose
        diagnostics = diagnostics or _empty_policy_diagnostics(observed_mask)
        row0, row1 = int(roi["row0"]), int(roi["row1"])
        column0, column1 = int(roi["column0"]), int(roi["column1"])
        observed_mask[row0:row1, column0:column1] = True
        time_s, dose = _raster_cost(config, roi)
        consumed_time_s += time_s
        consumed_dose += dose
        queried_ids.add(str(roi["roi_id"]))
        prediction = reconstruct_from_observed_mask(dense_signal, observed_mask, x, y, channels, config)
        trace_rows.append(
            {
                "slice": slice_id,
                "policy": policy,
                "query_index": len(trace_rows) + 1,
                "stage": stage,
                "roi_id": str(roi["roi_id"]),
                "row0": row0,
                "row1": row1,
                "column0": column0,
                "column1": column1,
                "selection_utility": utility,
                "confirmation_anchor_roi_id": anchor_id,
                "estimated_time_s": time_s,
                "estimated_dose": dose,
                "effective_front_entropy_weight": diagnostics["effective_front_entropy_weight"],
                "effective_penetration_variance_weight": diagnostics[
                    "effective_penetration_variance_weight"
                ],
                "effective_reconstruction_uncertainty_weight": diagnostics[
                    "effective_reconstruction_uncertainty_weight"
                ],
                "morphology_gate_passed": diagnostics["morphology_gate_passed"],
                "morphology_reliability_score": diagnostics["morphology_reliability_score"],
                "state_separation_score": diagnostics["state_separation_score"],
                "front_support_fraction": diagnostics["front_support_fraction"],
                "front_continuity_score": diagnostics["front_continuity_score"],
                "observed_area_fraction": diagnostics["observed_area_fraction"],
                "mean_attention": diagnostics["mean_attention"],
                "max_attention": diagnostics["max_attention"],
                "attention_floor": diagnostics["attention_floor"],
                "mean_contrast_attention": diagnostics["mean_contrast_attention"],
                "mean_channel_diversity_attention": diagnostics[
                    "mean_channel_diversity_attention"
                ],
                "mean_front_attention": diagnostics["mean_front_attention"],
                "attention_gate_used": diagnostics["attention_gate_used"],
                "base_uncertainty_score": diagnostics["base_uncertainty_score"],
                "best_uncertainty_score": diagnostics["best_uncertainty_score"],
                "selected_base_uncertainty_ratio": diagnostics[
                    "selected_base_uncertainty_ratio"
                ],
                "residual_attention_score": diagnostics["residual_attention_score"],
                "residual_attention_center": diagnostics["residual_attention_center"],
                "residual_bonus_fraction": diagnostics["residual_bonus_fraction"],
                "near_tie_candidate_count": diagnostics["near_tie_candidate_count"],
                "residual_overrode_uncertainty_argmax": diagnostics[
                    "residual_overrode_uncertainty_argmax"
                ],
            }
        )
        metric_rows.append(
            _score_prediction(
                config,
                policy,
                seed,
                slice_id,
                len(metric_rows) + 1,
                stage,
                observed_mask,
                dense_signal,
                reference,
                prediction,
                consumed_time_s,
                consumed_dose,
            )
        )

    pilot_indices = rng.choice(len(candidates), size=config.acquisition_v3.pilot_rois, replace=False)
    for index in pilot_indices:
        reveal(candidates.iloc[int(index)], "random_pilot")
    for adaptive_index in range(config.acquisition_v3.adaptive_rois):
        if len(queried_ids) >= len(candidates):
            break
        selected, utility, stage, diagnostics = _select_candidate(
            policy,
            candidates,
            prediction,
            queried_ids,
            config,
            rng,
            observed_mask,
            adaptive_index,
        )
        if np.isfinite(utility) and utility < config.acquisition_v3.stop_min_utility:
            break
        reveal(selected, stage, utility, diagnostics=diagnostics)
    if config.acquisition_v3.neighbor_confirmation != "disabled":
        trace = pd.DataFrame(trace_rows)
        anchor_pool = trace[trace["stage"] != "random_pilot"]
        if anchor_pool.empty:
            anchor_pool = trace
        anchor_count = (
            1
            if config.acquisition_v3.neighbor_confirmation == "one_anchor"
            else config.acquisition_v3.neighbor_anchors
        )
        for _, anchor in (
            anchor_pool.sort_values("selection_utility", ascending=False, na_position="last")
            .drop_duplicates("roi_id")
            .head(anchor_count)
            .iterrows()
        ):
            anchor_roi = candidates[candidates["roi_id"] == anchor["roi_id"]].iloc[0]
            anchor_diagnostics = {
                "effective_front_entropy_weight": anchor["effective_front_entropy_weight"],
                "effective_penetration_variance_weight": anchor[
                    "effective_penetration_variance_weight"
                ],
                "effective_reconstruction_uncertainty_weight": anchor[
                    "effective_reconstruction_uncertainty_weight"
                ],
                "morphology_gate_passed": bool(anchor["morphology_gate_passed"]),
                "morphology_reliability_score": anchor["morphology_reliability_score"],
                "state_separation_score": anchor["state_separation_score"],
                "front_support_fraction": anchor["front_support_fraction"],
                "front_continuity_score": anchor["front_continuity_score"],
                "observed_area_fraction": anchor["observed_area_fraction"],
                "mean_attention": anchor.get("mean_attention", np.nan),
                "max_attention": anchor.get("max_attention", np.nan),
                "attention_floor": anchor.get("attention_floor", np.nan),
                "mean_contrast_attention": anchor.get("mean_contrast_attention", np.nan),
                "mean_channel_diversity_attention": anchor.get(
                    "mean_channel_diversity_attention", np.nan
                ),
                "mean_front_attention": anchor.get("mean_front_attention", np.nan),
                "attention_gate_used": bool(anchor.get("attention_gate_used", False)),
                "base_uncertainty_score": anchor.get("base_uncertainty_score", np.nan),
                "best_uncertainty_score": anchor.get("best_uncertainty_score", np.nan),
                "selected_base_uncertainty_ratio": anchor.get(
                    "selected_base_uncertainty_ratio", np.nan
                ),
                "residual_attention_score": anchor.get("residual_attention_score", np.nan),
                "residual_attention_center": anchor.get("residual_attention_center", np.nan),
                "residual_bonus_fraction": anchor.get("residual_bonus_fraction", np.nan),
                "near_tie_candidate_count": anchor.get("near_tie_candidate_count", np.nan),
                "residual_overrode_uncertainty_argmax": bool(
                    anchor.get("residual_overrode_uncertainty_argmax", False)
                ),
            }
            for neighbor in _neighbor_candidates(candidates, anchor_roi, queried_ids):
                reveal(
                    neighbor,
                    "neighbor_confirmation_roi",
                    anchor_id=str(anchor["roi_id"]),
                    diagnostics=anchor_diagnostics,
                )
    return V3SliceResult(
        config=config,
        slice_id=slice_id,
        policy=policy,
        seed=seed,
        candidates=candidates,
        trace=pd.DataFrame(trace_rows),
        metrics=pd.DataFrame(metric_rows),
        reference=reference,
        final_prediction=prediction,
    )


def _metric_auc(frame: pd.DataFrame, metric: str) -> float:
    ordered = frame.sort_values("selected_area_fraction")
    x = ordered["selected_area_fraction"].to_numpy(float)
    y = ordered[metric].to_numpy(float)
    if len(x) == 0:
        return np.nan
    if len(x) == 1:
        return float(x[0] * y[0])
    return float(np.trapezoid(y, x))


def summarize_v3_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    per_slice_rows = []
    for (policy, slice_id), frame in metrics.groupby(["policy", "slice"], sort=False):
        final = frame.sort_values("iteration").iloc[-1]
        per_slice_rows.append(
            {
                "policy": policy,
                "slice": slice_id,
                "final_selected_area_fraction": final["selected_area_fraction"],
                "final_front_mean_symmetric_distance_nm": final["front_mean_symmetric_distance_nm"],
                "final_front_hausdorff_distance_nm": final["front_hausdorff_distance_nm"],
                "final_penetration_d95_absolute_error_nm": final["penetration_d95_absolute_error_nm"],
                "final_penetration_dmax_absolute_error_nm": final["penetration_dmax_absolute_error_nm"],
                "front_entropy_auc_vs_cost": _metric_auc(frame, "mean_front_entropy"),
                "reconstruction_rmse_auc_vs_cost": _metric_auc(frame, "normalized_reconstruction_rmse"),
            }
        )
    per_slice = pd.DataFrame(per_slice_rows)
    summary_rows = []
    for policy, frame in per_slice.groupby("policy", sort=False):
        summary_rows.append(
            {
                "policy": policy,
                "slices": len(frame),
                "mean_final_selected_area_fraction": frame["final_selected_area_fraction"].mean(),
                "mean_final_front_mean_symmetric_distance_nm": frame[
                    "final_front_mean_symmetric_distance_nm"
                ].mean(),
                "median_final_front_mean_symmetric_distance_nm": frame[
                    "final_front_mean_symmetric_distance_nm"
                ].median(),
                "mean_final_front_hausdorff_distance_nm": frame[
                    "final_front_hausdorff_distance_nm"
                ].mean(),
                "mean_final_penetration_d95_absolute_error_nm": frame[
                    "final_penetration_d95_absolute_error_nm"
                ].mean(),
                "mean_final_penetration_dmax_absolute_error_nm": frame[
                    "final_penetration_dmax_absolute_error_nm"
                ].mean(),
                "mean_front_entropy_auc_vs_cost": frame["front_entropy_auc_vs_cost"].mean(),
                "mean_reconstruction_rmse_auc_vs_cost": frame["reconstruction_rmse_auc_vs_cost"].mean(),
            }
        )
    return pd.DataFrame(summary_rows).sort_values(
        ["mean_final_front_mean_symmetric_distance_nm", "mean_final_selected_area_fraction"]
    )


def summarize_v3_stratified_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Separate morphology-proxy detection outcomes from conditional localization error."""

    required = {
        "front_availability_status",
        "front_reference_present",
        "front_predicted_present",
        "front_detection_correct",
        "front_localization_mean_symmetric_distance_nm",
        "front_localization_hausdorff_distance_nm",
        "penetration_availability_status",
        "penetration_reference_present",
        "penetration_predicted_present",
        "penetration_detection_correct",
        "penetration_d95_localization_absolute_error_nm",
        "penetration_dmax_localization_absolute_error_nm",
    }
    if metrics.empty or not required.issubset(metrics.columns):
        return pd.DataFrame()
    final = metrics.sort_values("iteration").groupby(["policy", "slice"], sort=False).tail(1)
    rows = []
    for policy, frame in final.groupby("policy", sort=False):
        front_reference_present = frame["front_reference_present"].astype(bool)
        front_predicted_present = frame["front_predicted_present"].astype(bool)
        penetration_reference_present = frame["penetration_reference_present"].astype(bool)
        penetration_predicted_present = frame["penetration_predicted_present"].astype(bool)
        rows.append(
            {
                "policy": policy,
                "slices": len(frame),
                "mean_final_selected_area_fraction": frame["selected_area_fraction"].mean(),
                "front_reference_present_slices": int(front_reference_present.sum()),
                "front_reference_absent_slices": int((~front_reference_present).sum()),
                "front_matched_present_slices": int(
                    (frame["front_availability_status"] == "matched_present").sum()
                ),
                "front_matched_absent_slices": int(
                    (frame["front_availability_status"] == "matched_absent").sum()
                ),
                "front_missed_slices": int(
                    (frame["front_availability_status"] == "missed_front").sum()
                ),
                "front_spurious_slices": int(
                    (frame["front_availability_status"] == "spurious_front").sum()
                ),
                "front_detection_accuracy": frame["front_detection_correct"].astype(bool).mean(),
                "front_detection_recall": (
                    front_predicted_present[front_reference_present].mean()
                    if front_reference_present.any()
                    else np.nan
                ),
                "front_detection_specificity": (
                    (~front_predicted_present[~front_reference_present]).mean()
                    if (~front_reference_present).any()
                    else np.nan
                ),
                "front_localization_evaluable_slices": int(
                    frame["front_localization_mean_symmetric_distance_nm"].notna().sum()
                ),
                "front_exact_match_slices": int(
                    (frame["front_localization_mean_symmetric_distance_nm"] == 0.0).sum()
                ),
                "front_exact_match_rate": (
                    frame["front_localization_mean_symmetric_distance_nm"] == 0.0
                ).mean(),
                "mean_front_localization_distance_nm": frame[
                    "front_localization_mean_symmetric_distance_nm"
                ].mean(),
                "median_front_localization_distance_nm": frame[
                    "front_localization_mean_symmetric_distance_nm"
                ].median(),
                "p90_front_localization_distance_nm": frame[
                    "front_localization_mean_symmetric_distance_nm"
                ].quantile(0.90),
                "mean_front_localization_hausdorff_distance_nm": frame[
                    "front_localization_hausdorff_distance_nm"
                ].mean(),
                "penetration_reference_present_slices": int(penetration_reference_present.sum()),
                "penetration_reference_absent_slices": int((~penetration_reference_present).sum()),
                "penetration_matched_present_slices": int(
                    (frame["penetration_availability_status"] == "matched_present").sum()
                ),
                "penetration_matched_absent_slices": int(
                    (frame["penetration_availability_status"] == "matched_absent").sum()
                ),
                "penetration_missed_slices": int(
                    (frame["penetration_availability_status"] == "missed_reference_feature").sum()
                ),
                "penetration_spurious_slices": int(
                    (frame["penetration_availability_status"] == "spurious_predicted_feature").sum()
                ),
                "penetration_detection_accuracy": frame["penetration_detection_correct"].astype(bool).mean(),
                "penetration_detection_recall": (
                    penetration_predicted_present[penetration_reference_present].mean()
                    if penetration_reference_present.any()
                    else np.nan
                ),
                "penetration_detection_specificity": (
                    (~penetration_predicted_present[~penetration_reference_present]).mean()
                    if (~penetration_reference_present).any()
                    else np.nan
                ),
                "penetration_localization_evaluable_slices": int(
                    frame["penetration_d95_localization_absolute_error_nm"].notna().sum()
                ),
                "mean_penetration_d95_localization_error_nm": frame[
                    "penetration_d95_localization_absolute_error_nm"
                ].mean(),
                "median_penetration_d95_localization_error_nm": frame[
                    "penetration_d95_localization_absolute_error_nm"
                ].median(),
                "p90_penetration_d95_localization_error_nm": frame[
                    "penetration_d95_localization_absolute_error_nm"
                ].quantile(0.90),
                "mean_penetration_dmax_localization_error_nm": frame[
                    "penetration_dmax_localization_absolute_error_nm"
                ].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["front_detection_accuracy", "mean_front_localization_distance_nm"],
        ascending=[False, True],
    )


def summarize_v3_slice_blocks(metrics: pd.DataFrame, block_size: int = 25) -> pd.DataFrame:
    """Summarize audited final metrics over contiguous stack blocks."""

    if metrics.empty or "front_localization_mean_symmetric_distance_nm" not in metrics:
        return pd.DataFrame()
    final = metrics.sort_values("iteration").groupby(["policy", "slice"], sort=False).tail(1).copy()
    slice_number = final["slice"].astype(str).astype(int)
    final["slice_block_start"] = ((slice_number - 1) // block_size) * block_size + 1
    final["slice_block_end"] = final["slice_block_start"] + block_size - 1
    rows = []
    for (policy, block_start, block_end), frame in final.groupby(
        ["policy", "slice_block_start", "slice_block_end"], sort=False
    ):
        rows.append(
            {
                "policy": policy,
                "slice_block": f"{int(block_start):03d}:{int(block_end):03d}",
                "slices": len(frame),
                "front_exact_match_slices": int(
                    (frame["front_localization_mean_symmetric_distance_nm"] == 0.0).sum()
                ),
                "front_exact_match_rate": (
                    frame["front_localization_mean_symmetric_distance_nm"] == 0.0
                ).mean(),
                "mean_front_localization_distance_nm": frame[
                    "front_localization_mean_symmetric_distance_nm"
                ].mean(),
                "median_front_localization_distance_nm": frame[
                    "front_localization_mean_symmetric_distance_nm"
                ].median(),
                "mean_penetration_d95_localization_error_nm": frame[
                    "penetration_d95_localization_absolute_error_nm"
                ].mean(),
                "mean_final_selected_area_fraction": frame["selected_area_fraction"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["policy", "slice_block"])


def summarize_residual_attention_diagnostics(trace: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    residual_stage = "balance_v3_residual_attention_uncertainty_reconstruction"
    residual_trace = trace[
        (trace["policy"] == "balance_v3_residual_attention") & (trace["stage"] == residual_stage)
    ].copy()
    if residual_trace.empty:
        return pd.DataFrame()
    residual_trace["residual_overrode_uncertainty_argmax"] = (
        residual_trace["residual_overrode_uncertainty_argmax"].astype(str).str.lower() == "true"
    )
    residual_trace["morphology_gate_passed"] = (
        residual_trace["morphology_gate_passed"].astype(str).str.lower() == "true"
    )
    residual_metrics = metrics[metrics["policy"] == "balance_v3_residual_attention"].copy()
    residual_metrics = residual_metrics.sort_values(["slice", "iteration"])
    residual_metrics["immediate_front_distance_change_nm"] = residual_metrics.groupby("slice")[
        "front_mean_symmetric_distance_nm"
    ].diff()
    residual_trace = residual_trace.merge(
        residual_metrics[
            ["slice", "policy", "iteration", "immediate_front_distance_change_nm"]
        ],
        left_on=["slice", "policy", "query_index"],
        right_on=["slice", "policy", "iteration"],
        how="left",
    )
    groups = [
        ("all_adaptive_queries", residual_trace),
        (
            "overrode_uncertainty_argmax",
            residual_trace[residual_trace["residual_overrode_uncertainty_argmax"]],
        ),
        (
            "kept_uncertainty_argmax",
            residual_trace[~residual_trace["residual_overrode_uncertainty_argmax"]],
        ),
    ]
    rows = []
    for group, frame in groups:
        if frame.empty:
            continue
        rows.append(
            {
                "group": group,
                "adaptive_queries": len(frame),
                "slices": frame["slice"].nunique(),
                "override_frequency": frame["residual_overrode_uncertainty_argmax"].mean(),
                "mean_selected_uncertainty_ratio": frame["selected_base_uncertainty_ratio"].mean(),
                "minimum_selected_uncertainty_ratio": frame["selected_base_uncertainty_ratio"].min(),
                "mean_residual_bonus_fraction": frame["residual_bonus_fraction"].mean(),
                "maximum_residual_bonus_fraction": frame["residual_bonus_fraction"].max(),
                "mean_near_tie_candidate_count": frame["near_tie_candidate_count"].mean(),
                "morphology_gate_pass_rate": frame["morphology_gate_passed"].mean(),
                "mean_immediate_front_distance_change_nm": frame[
                    "immediate_front_distance_change_nm"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def _comparison_table_vs_uncertainty(
    metrics: pd.DataFrame, slice_ids: list[str] | None = None
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    if slice_ids is not None:
        metrics = metrics[metrics["slice"].astype(str).str.zfill(3).isin(slice_ids)]
        if metrics.empty:
            return pd.DataFrame()
    final = metrics.sort_values("iteration").groupby(["slice", "policy"], sort=False).tail(1)
    if "uncertainty" not in set(final["policy"]):
        return pd.DataFrame()
    front = final.pivot(index="slice", columns="policy", values="front_mean_symmetric_distance_nm")
    area = final.pivot(index="slice", columns="policy", values="selected_area_fraction")
    if "uncertainty" not in front:
        return pd.DataFrame()
    rows: list[dict[str, float | int | str]] = []
    for policy in sorted(column for column in front.columns if column != "uncertainty"):
        common = front[[policy, "uncertainty"]].dropna()
        if common.empty:
            continue
        area_common = area.reindex(common.index)
        front_delta = float((common[policy] - common["uncertainty"]).mean())
        win_rate = float((common[policy] < common["uncertainty"]).mean())
        area_delta = float((area_common[policy] - area_common["uncertainty"]).mean())
        rows.append(
            {
                "policy": policy,
                "compared_slices": len(common),
                "mean_front_distance_delta_vs_uncertainty_nm": front_delta,
                "front_distance_win_rate_vs_uncertainty": win_rate,
                "mean_selected_area_delta_vs_uncertainty_pp": 100.0 * area_delta,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_front_distance_delta_vs_uncertainty_nm")


def _comparison_vs_uncertainty(metrics: pd.DataFrame, slice_ids: list[str] | None = None) -> str:
    comparison = _comparison_table_vs_uncertainty(metrics, slice_ids)
    if comparison.empty:
        return "no completed policy overlap with uncertainty yet"
    return "; ".join(
        f"{row['policy']}: delta_front={row['mean_front_distance_delta_vs_uncertainty_nm']:+.1f} nm, "
        f"win_rate={100.0 * row['front_distance_win_rate_vs_uncertainty']:.1f}%, "
        f"delta_area={row['mean_selected_area_delta_vs_uncertainty_pp']:+.2f} pp"
        for _, row in comparison.iterrows()
    )


def _milestone_comparisons_vs_uncertainty(metrics: pd.DataFrame, slice_ids: list[str]) -> pd.DataFrame:
    milestones = list(range(10, len(slice_ids) + 1, 10))
    if not milestones or milestones[-1] != len(slice_ids):
        milestones.append(len(slice_ids))
    frames = []
    for milestone in milestones:
        comparison = _comparison_table_vs_uncertainty(metrics, slice_ids[:milestone])
        if not comparison.empty:
            comparison.insert(0, "evaluated_slices", milestone)
            frames.append(comparison)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _plot_v3_result(result: V3SliceResult, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    channel0 = result.reference["dense_intensity"].isel(channel=0).values
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    panels = [
        (channel0, f"Dense {str(result.reference.channel.values[0])}", "gray"),
        (result.final_prediction["reconstruction_uncertainty"].values, "Reconstruction uncertainty", "viridis"),
        (result.reference["pseudo_altered_region"].values, "Pseudo altered region", "magma"),
        (result.final_prediction["altered_region_probability"].values, "Predicted altered probability", "magma"),
        (result.final_prediction["front_entropy"].values, "Front entropy", "inferno"),
        (result.reference["pseudo_front"].values, "Pseudo front and queried ROIs", "gray"),
    ]
    extent = [
        float(result.reference.x.min()),
        float(result.reference.x.max()),
        float(result.reference.y.min()),
        float(result.reference.y.max()),
    ]
    for axis, (values, title, cmap) in zip(axes.ravel(), panels):
        image = axis.imshow(values, origin="lower", extent=extent, aspect="auto", cmap=cmap)
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046)
    roi_axis = axes.ravel()[-1]
    x_values = result.reference.x.values
    y_values = result.reference.y.values
    for _, row in result.trace.iterrows():
        x0 = x_values[int(row["column0"])]
        y0 = y_values[int(row["row0"])]
        width = x_values[int(row["column1"]) - 1] - x0
        height = y_values[int(row["row1"]) - 1] - y0
        roi_axis.add_patch(
            plt.Rectangle((x0, y0), width, height, fill=False, edgecolor="cyan", linewidth=1)
        )
    figure.suptitle(f"V3 {result.policy} slice {result.slice_id}")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _observed_mask_from_trace(trace: pd.DataFrame, shape: tuple[int, int]) -> np.ndarray:
    observed_mask = np.zeros(shape, dtype=bool)
    for _, row in trace.iterrows():
        observed_mask[
            int(row["row0"]) : int(row["row1"]),
            int(row["column0"]) : int(row["column1"]),
        ] = True
    return observed_mask


def audit_v3_stack_from_trace(
    template_path: Path,
    slice_ids: list[str],
    input_dir: Path,
    output: Path,
    manifest_path: Path | None = None,
    policies: list[V3PolicyName] | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute final stratified metrics from stored ROI traces without rerunning full loops."""

    template = load_config(template_path)
    policies = policies or DEFAULT_V3_POLICIES
    unsupported = set(policies) - set(DEFAULT_V3_POLICIES)
    if unsupported:
        raise ValueError(f"unsupported v3 policies: {sorted(unsupported)}")
    trace_path = input_dir / "v3_acquisition_trace.csv"
    if not trace_path.exists():
        trace_path = input_dir / "v3_acquisition_trace.partial.csv"
    if not trace_path.exists():
        raise FileNotFoundError("v3 acquisition trace artifact was not found")
    trace = pd.read_csv(trace_path, dtype={"slice": str}, low_memory=False)
    trace["slice"] = trace["slice"].str.zfill(3)
    manifest_sources = (
        sources_from_manifest(manifest_path, slice_ids, template.scenario.elements)
        if manifest_path is not None
        else None
    )
    rows: list[dict[str, object]] = []
    repaired_trace_frames: list[pd.DataFrame] = []
    for index, slice_id in enumerate(slice_ids, start=1):
        config = v3_config_for_slice(
            template, slice_id, manifest_sources[slice_id] if manifest_sources else None
        )
        source, _ = ingest_dataset(config)
        dense_signal, x, y, channels = dense_signal_from_observations(config, source)
        reference = pseudo_reference_from_dense_signal(dense_signal, x, y, channels, config)
        for policy in policies:
            policy_trace = trace[(trace["slice"] == slice_id) & (trace["policy"] == policy)].copy()
            trace_replayed = False
            if policy_trace.empty:
                replay = run_v3_slice_replay(
                    config, dense_signal, x, y, channels, slice_id, policy, seed=seed
                )
                policy_trace = replay.trace
                repaired_trace_frames.append(policy_trace)
                trace_replayed = True
            policy_trace = policy_trace.sort_values("query_index")
            observed_mask = _observed_mask_from_trace(policy_trace, dense_signal.shape[1:])
            prediction = reconstruct_from_observed_mask(
                dense_signal, observed_mask, x, y, channels, config
            )
            row = _score_prediction(
                config,
                policy,
                seed,
                slice_id,
                len(policy_trace),
                "audited_final_trace_reconstruction",
                observed_mask,
                dense_signal,
                reference,
                prediction,
                float(policy_trace["estimated_time_s"].sum()),
                float(policy_trace["estimated_dose"].sum()),
            )
            row["query_count"] = len(policy_trace)
            row["trace_replayed_during_audit"] = trace_replayed
            rows.append(row)
        if index % 10 == 0 or index == len(slice_ids):
            print(f"Audited v3 front availability on {index}/{len(slice_ids)} slices.")
    audited = pd.DataFrame(rows)
    summary = summarize_v3_stratified_metrics(audited)
    output.mkdir(parents=True, exist_ok=True)
    audited.to_csv(output / "v3_audited_final_metrics_by_slice.csv", index=False)
    summary.to_csv(output / "v3_audited_stratified_summary.csv", index=False)
    summarize_v3_slice_blocks(audited).to_csv(
        output / "v3_audited_slice_block_summary.csv", index=False
    )
    repaired = (
        pd.concat(repaired_trace_frames, ignore_index=True)
        if repaired_trace_frames
        else pd.DataFrame(columns=trace.columns)
    )
    repaired.to_csv(output / "v3_audited_replayed_missing_trace_rows.csv", index=False)
    protocol = {
        "schema": "balance_nm_v3_stratified_front_availability_audit",
        "template_config": str(template_path),
        "source_trace": str(trace_path),
        "download_manifest": str(manifest_path) if manifest_path is not None else None,
        "slice_count": len(slice_ids),
        "slice_ids": slice_ids,
        "seed": seed,
        "policies": policies,
        "replayed_missing_policy_slice_pairs": int(
            repaired[["slice", "policy"]].drop_duplicates().shape[0]
        ) if not repaired.empty else 0,
        "metric_semantics": {
            "front_detection": "reports matched-present, matched-absent, missed, and spurious alteration-front proxies separately",
            "front_localization": "reported only when both frozen pseudo-reference and reconstruction contain a front",
            "penetration_localization": "reported only when both frozen pseudo-reference and reconstruction contain a finite penetration proxy",
            "front_semantics": "morphology-defined alteration-front proxy; not expert-labeled corrosion truth",
        },
    }
    with (output / "v3_audited_protocol.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(protocol, handle, sort_keys=False)
    return audited, summary


def run_v3_stack_validation(
    template_path: Path,
    slice_ids: list[str],
    output: Path,
    manifest_path: Path | None = None,
    policies: list[V3PolicyName] | None = None,
    seed: int = 0,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    template = load_config(template_path)
    policies = policies or DEFAULT_V3_POLICIES
    unsupported = set(policies) - set(DEFAULT_V3_POLICIES)
    if unsupported:
        raise ValueError(f"unsupported v3 policies: {sorted(unsupported)}")
    manifest_sources = (
        sources_from_manifest(manifest_path, slice_ids, template.scenario.elements)
        if manifest_path is not None
        else None
    )
    output.mkdir(parents=True, exist_ok=True)
    write_config(template, output / "resolved_template_config.yaml")
    metrics_frames: list[pd.DataFrame] = []
    trace_frames: list[pd.DataFrame] = []
    completed: set[tuple[str, str]] = set()
    metrics_partial = output / "v3_metrics_by_iteration.partial.csv"
    trace_partial = output / "v3_acquisition_trace.partial.csv"
    if resume and metrics_partial.exists():
        existing_metrics = pd.read_csv(metrics_partial, dtype={"slice": str})
        if not existing_metrics.empty:
            existing_metrics["slice"] = existing_metrics["slice"].str.zfill(3)
            metrics_frames.append(existing_metrics)
            completed = set(
                existing_metrics[["slice", "policy"]].drop_duplicates().itertuples(index=False, name=None)
            )
        if trace_partial.exists():
            existing_trace = pd.read_csv(trace_partial, dtype={"slice": str}, low_memory=False)
            if not existing_trace.empty:
                existing_trace["slice"] = existing_trace["slice"].str.zfill(3)
                trace_frames.append(existing_trace)
    first_result: V3SliceResult | None = None
    for index, slice_id in enumerate(slice_ids, start=1):
        config = v3_config_for_slice(
            template, slice_id, manifest_sources[slice_id] if manifest_sources else None
        )
        source, _ = ingest_dataset(config)
        dense_signal, x, y, channels = dense_signal_from_observations(config, source)
        for policy in policies:
            if (slice_id, policy) in completed:
                continue
            result = run_v3_slice_replay(config, dense_signal, x, y, channels, slice_id, policy, seed=seed)
            metrics_frames.append(result.metrics)
            trace_frames.append(result.trace)
            if first_result is None:
                first_result = result
        if index % 5 == 0 or index == len(slice_ids):
            if metrics_frames:
                pd.concat(metrics_frames, ignore_index=True).to_csv(metrics_partial, index=False)
            if trace_frames:
                pd.concat(trace_frames, ignore_index=True).to_csv(trace_partial, index=False)
        if (index % 10 == 0 or index == len(slice_ids)) and metrics_frames:
            comparison = _comparison_vs_uncertainty(
                pd.concat(metrics_frames, ignore_index=True), slice_ids[:index]
            )
            print(f"Validated v3 morphology reconstruction on {index}/{len(slice_ids)} slices.")
            print(f"Comparison vs uncertainty after {index} slices: {comparison}")
    metrics = pd.concat(metrics_frames, ignore_index=True)
    trace = pd.concat(trace_frames, ignore_index=True)
    summary = summarize_v3_metrics(metrics)
    metrics.to_csv(output / "v3_metrics_by_iteration.csv", index=False)
    trace.to_csv(output / "v3_acquisition_trace.csv", index=False)
    summary.to_csv(output / "v3_summary.csv", index=False)
    summarize_v3_stratified_metrics(metrics).to_csv(output / "v3_stratified_summary.csv", index=False)
    summarize_residual_attention_diagnostics(trace, metrics).to_csv(
        output / "v3_residual_attention_diagnostics.csv", index=False
    )
    _milestone_comparisons_vs_uncertainty(metrics, slice_ids).to_csv(
        output / "v3_comparison_vs_uncertainty_every_10_slices.csv", index=False
    )
    protocol = {
        "schema": "balance_nm_v3_corrosion_morphology_reconstruction",
        "template_config": str(template_path),
        "download_manifest": str(manifest_path) if manifest_path is not None else None,
        "slice_count": len(slice_ids),
        "slice_ids": slice_ids,
        "seed": seed,
        "policies": policies,
        "excluded_legacy_v2_surfaces": [
            "balance",
            "balance_no_confirmation",
            "balance_one_anchor",
            "balance_two_anchors",
            "reference_equivalent_roi",
            "recommended_regret_fraction",
        ],
        "front_semantics": "morphology-defined alteration-front proxy; not expert-labeled corrosion truth",
    }
    with (output / "v3_protocol.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(protocol, handle, sort_keys=False)
    if first_result is not None:
        first_result.reference.to_zarr(output / "example_v3_pseudo_reference.zarr", mode="w")
        first_result.final_prediction.to_zarr(output / "example_v3_prediction.zarr", mode="w")
        _plot_v3_result(first_result, output / "example_v3_reconstruction.png")
    return metrics, summary
