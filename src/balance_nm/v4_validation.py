"""Uncertainty-first adaptive raster reconstruction for multichannel corrosion replay."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Callable

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import distance_transform_edt
from scipy.stats import t
from sklearn.ensemble import HistGradientBoostingRegressor

from .data import ingest_dataset
from .domain import RunConfig
from .io import load_config, write_config
from .v3_morphology import (
    dense_signal_from_observations,
    front_distance_metrics,
    front_from_probability,
    morphology_products_from_signal,
    normalized_reconstruction_rmse,
    penetration_stat,
    pseudo_reference_from_dense_signal,
    reconstruct_from_observed_mask,
)
from .v3_validation import (
    _neighbor_candidates,
    _raster_cost,
    build_v3_roi_catalog,
    sources_from_manifest,
    v3_config_for_slice,
)
from .v4_neural import NeuralReconstructionEnsemble, nearest_signal_and_distance, torch_available, train_neural_ensemble

V4_POLICIES = [
    "uncertainty_distance_sequential",
    "uncertainty_distance_one_anchor",
    "uncertainty_lookahead",
    "uncertainty_calibrated_guarded",
    "uncertainty_neural_guarded_selector_only",
    "uncertainty_neural_guarded_full_system",
    "uniform",
    "random",
    "oracle_composite_gain",
]
V4_NEURAL_POLICIES = {
    "uncertainty_neural_guarded_selector_only",
    "uncertainty_neural_guarded_full_system",
}
V4_DEPLOYABLE_POLICIES = set(V4_POLICIES) - {"oracle_composite_gain"}


@dataclass(frozen=True)
class V4Fold:
    fold_id: str
    test_slices: list[str]
    validation_slices: list[str]
    training_slices: list[str]
    excluded_guard_slices: list[str]


@dataclass
class GuardedCalibrator:
    model: HistGradientBoostingRegressor
    feature_columns: list[str]
    shortlist_ratio: float
    learned_weight: float
    training_rows: int
    training_slices: list[str]
    validation_slices: list[str]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(frame[self.feature_columns])


@dataclass
class V4SliceResult:
    metrics: pd.DataFrame
    candidate_trace: pd.DataFrame


def _slice_width_nm(x: np.ndarray) -> float:
    if len(x) <= 1:
        return 1.0
    return float(x[-1] - x[0] + np.median(np.diff(x)))


def _distance_pixels(observed_mask: np.ndarray) -> np.ndarray:
    if not np.any(observed_mask):
        return np.full(observed_mask.shape, float(np.hypot(*observed_mask.shape)) / 2.0)
    return distance_transform_edt(~observed_mask)


def _normalized_distance(distance_pixels: np.ndarray) -> np.ndarray:
    maximum = max(float(np.hypot(*distance_pixels.shape)) / 2.0, 1.0)
    return np.clip(distance_pixels / maximum, 0.0, 1.0)


def _rectangle_distance_pixels(shape: tuple[int, int], roi: pd.Series) -> np.ndarray:
    rows, columns = shape
    row0, row1 = int(roi["row0"]), int(roi["row1"])
    column0, column1 = int(roi["column0"]), int(roi["column1"])
    row_values = np.arange(rows)
    column_values = np.arange(columns)
    row_distance = np.maximum(np.maximum(row0 - row_values, 0), row_values - (row1 - 1))
    column_distance = np.maximum(
        np.maximum(column0 - column_values, 0), column_values - (column1 - 1)
    )
    return np.hypot(row_distance[:, None], column_distance[None, :])


def lookahead_coverage_gain(distance_pixels: np.ndarray, roi: pd.Series) -> float:
    """Exact integrated distance-uncertainty reduction for revealing a rectangular ROI."""

    rectangle_distance = _rectangle_distance_pixels(distance_pixels.shape, roi)
    updated = np.minimum(distance_pixels, rectangle_distance)
    maximum = max(float(np.hypot(*distance_pixels.shape)) / 2.0, 1.0)
    return float(np.sum(distance_pixels - updated) / maximum)


def _sanitize_channel(channel: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", channel.lower()).strip("_")


def _candidate_features(
    candidates: pd.DataFrame,
    prediction,
    observed_mask: np.ndarray,
    distance_pixels: np.ndarray,
    channels: list[str],
    config: RunConfig,
    query_index: int,
    neural_variance: np.ndarray | None = None,
) -> pd.DataFrame:
    means = prediction["mean_intensity"].values
    normalized_distance = _normalized_distance(distance_pixels)
    records = []
    for _, roi in candidates.iterrows():
        row0, row1 = int(roi["row0"]), int(roi["row1"])
        column0, column1 = int(roi["column0"]), int(roi["column1"])
        cost, _ = _raster_cost(config, roi)
        region_uncertainty = normalized_distance[row0:row1, column0:column1]
        region_signal = means[:, row0:row1, column0:column1]
        expanded = means[
            :,
            max(row0 - 1, 0) : min(row1 + 1, means.shape[1]),
            max(column0 - 1, 0) : min(column1 + 1, means.shape[2]),
        ]
        gain = lookahead_coverage_gain(distance_pixels, roi)
        record = {
            "roi_id": str(roi["roi_id"]),
            "coverage_gain": gain,
            "coverage_gain_per_cost": gain / max(cost, 1.0e-12),
            "uncertainty_mean": float(np.mean(region_uncertainty)),
            "uncertainty_max": float(np.max(region_uncertainty)),
            "uncertainty_q25": float(np.quantile(region_uncertainty, 0.25)),
            "uncertainty_q75": float(np.quantile(region_uncertainty, 0.75)),
            "newly_covered_fraction": float(np.mean(~observed_mask[row0:row1, column0:column1])),
            "nearest_observation_distance_mean": float(np.mean(distance_pixels[row0:row1, column0:column1])),
            "acquisition_progress": float(query_index / max(config.acquisition_v4.total_rois - 1, 1)),
            "normalized_center_x": float(roi["center_x_nm"] / max(config.scenario.width_nm, 1.0)),
            "normalized_center_y": float(roi["center_y_nm"] / max(config.scenario.height_nm, 1.0)),
            "signal_mean_mean": float(np.mean(region_signal)),
            "signal_mean_std": float(np.std(region_signal.mean(axis=(1, 2)))),
            "neighborhood_dispersion_mean": float(np.mean(np.std(expanded, axis=(1, 2)))),
            "neighborhood_dispersion_max": float(np.max(np.std(expanded, axis=(1, 2)))),
            "neural_disagreement_mean": (
                float(np.mean(neural_variance[:, row0:row1, column0:column1]))
                if neural_variance is not None
                else 0.0
            ),
        }
        for channel_index, channel in enumerate(channels):
            name = _sanitize_channel(channel)
            record[f"channel_mean__{name}"] = float(np.mean(region_signal[channel_index]))
            record[f"channel_dispersion__{name}"] = float(np.std(expanded[channel_index]))
        records.append(record)
    return pd.DataFrame(records)


def _prediction_from_neural(
    mean_signal: np.ndarray,
    variance_signal: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    channels: list[str],
    config: RunConfig,
):
    uncertainty = np.sqrt(np.mean(np.maximum(variance_signal, 0.0), axis=0))
    maximum = max(float(np.max(uncertainty)), 1.0e-12)
    uncertainty = np.clip(uncertainty / maximum, 0.0, 1.0)
    import xarray as xr

    prediction = xr.Dataset(
        {
            "mean_intensity": (("channel", "y", "x"), mean_signal),
            "epistemic_uncertainty": (("channel", "y", "x"), variance_signal),
            "reconstruction_uncertainty": (("y", "x"), uncertainty),
        },
        coords={"channel": channels, "x": x, "y": y},
    )
    return xr.merge(
        [
            prediction,
            morphology_products_from_signal(
                mean_signal, x, y, channels, config, reconstruction_uncertainty=uncertainty
            ),
        ]
    )


def morphology_composite_error(
    reference,
    prediction,
    x: np.ndarray,
    y: np.ndarray,
    config: RunConfig,
) -> float:
    predicted_front, predicted_penetration = front_from_probability(
        prediction["altered_region_probability"].values, x, y, config
    )
    front = front_distance_metrics(reference["pseudo_front"].values, predicted_front, x, y)
    reference_d95 = penetration_stat(reference["pseudo_penetration_depth_nm"].values, "d95")
    predicted_d95 = penetration_stat(predicted_penetration, "d95")
    width = _slice_width_nm(x)
    if np.isfinite(reference_d95) and np.isfinite(predicted_d95):
        penetration_error = abs(predicted_d95 - reference_d95)
    elif not np.isfinite(reference_d95) and not np.isfinite(predicted_d95):
        penetration_error = 0.0
    else:
        penetration_error = width
    weights = config.acquisition_v4
    total_weight = weights.front_weight + weights.penetration_d95_weight
    return float(
        (
            weights.front_weight * front.mean_symmetric_distance_nm / width
            + weights.penetration_d95_weight * penetration_error / width
        )
        / total_weight
    )


def _score_v4_prediction(
    config: RunConfig,
    fold_id: str,
    policy: str,
    slice_id: str,
    iteration: int,
    stage: str,
    observed_mask: np.ndarray,
    dense_signal: np.ndarray,
    reference,
    prediction,
    scan_time_s: float,
    dose_proxy: float,
) -> dict[str, object]:
    x = prediction.coords["x"].values.astype(float)
    y = prediction.coords["y"].values.astype(float)
    predicted_front, predicted_penetration = front_from_probability(
        prediction["altered_region_probability"].values, x, y, config
    )
    front = front_distance_metrics(reference["pseudo_front"].values, predicted_front, x, y)
    reference_d95 = penetration_stat(reference["pseudo_penetration_depth_nm"].values, "d95")
    predicted_d95 = penetration_stat(predicted_penetration, "d95")
    reference_dmax = penetration_stat(reference["pseudo_penetration_depth_nm"].values, "dmax")
    predicted_dmax = penetration_stat(predicted_penetration, "dmax")
    width = _slice_width_nm(x)

    def absolute_error(first: float, second: float) -> float:
        if np.isfinite(first) and np.isfinite(second):
            return float(abs(first - second))
        if not np.isfinite(first) and not np.isfinite(second):
            return 0.0
        return width

    return {
        "fold": fold_id,
        "slice": slice_id,
        "policy": policy,
        "iteration": iteration,
        "stage": stage,
        "query_count": iteration,
        "scan_time_s": scan_time_s,
        "dose_proxy": dose_proxy,
        "selected_area_fraction": float(observed_mask.mean()),
        "morphology_composite_error": morphology_composite_error(reference, prediction, x, y, config),
        "front_mean_symmetric_distance_nm": front.mean_symmetric_distance_nm,
        "front_hausdorff_distance_nm": front.hausdorff_distance_nm,
        "front_availability_status": front.availability_status,
        "front_detection_correct": front.detection_correct,
        "penetration_d95_absolute_error_nm": absolute_error(reference_d95, predicted_d95),
        "penetration_dmax_absolute_error_nm": absolute_error(reference_dmax, predicted_dmax),
        "normalized_reconstruction_rmse": normalized_reconstruction_rmse(
            dense_signal, prediction["mean_intensity"].values
        ),
    }


def _hypothetical_composite_gain(
    dense_signal: np.ndarray,
    observed_mask: np.ndarray,
    roi: pd.Series,
    x: np.ndarray,
    y: np.ndarray,
    channels: list[str],
    config: RunConfig,
    reference,
    current_error: float,
) -> float:
    hypothetical = observed_mask.copy()
    hypothetical[int(roi["row0"]) : int(roi["row1"]), int(roi["column0"]) : int(roi["column1"])] = True
    prediction = reconstruct_from_observed_mask(dense_signal, hypothetical, x, y, channels, config)
    return float(current_error - morphology_composite_error(reference, prediction, x, y, config))


def _guarded_scores(
    features: pd.DataFrame,
    calibrator: GuardedCalibrator | None,
) -> pd.DataFrame:
    scored = features.copy()
    best_coverage = max(float(scored["coverage_gain_per_cost"].max()), 1.0e-12)
    scored["normalized_coverage_gain"] = scored["coverage_gain_per_cost"] / best_coverage
    if calibrator is None:
        scored["predicted_composite_gain"] = 0.0
        scored["shortlist_eligible"] = True
        scored["guarded_utility"] = scored["normalized_coverage_gain"]
        return scored
    predicted = calibrator.predict(scored)
    scored["predicted_composite_gain"] = predicted
    positive = predicted - min(float(np.min(predicted)), 0.0)
    maximum = max(float(np.max(positive)), 1.0e-12)
    scored["normalized_predicted_composite_gain"] = positive / maximum
    scored["shortlist_eligible"] = (
        scored["coverage_gain_per_cost"] >= calibrator.shortlist_ratio * best_coverage
    )
    scored["guarded_utility"] = np.where(
        scored["shortlist_eligible"],
        (1.0 - calibrator.learned_weight) * scored["normalized_coverage_gain"]
        + calibrator.learned_weight * scored["normalized_predicted_composite_gain"],
        -np.inf,
    )
    return scored


def _select_by_policy(
    policy: str,
    candidates: pd.DataFrame,
    queried_ids: set[str],
    prediction,
    observed_mask: np.ndarray,
    distance_pixels: np.ndarray,
    channels: list[str],
    config: RunConfig,
    rng: np.random.Generator,
    query_index: int,
    calibrator: GuardedCalibrator | None,
    neural_variance: np.ndarray | None,
    dense_signal: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    reference,
) -> tuple[pd.Series, pd.DataFrame]:
    eligible = candidates[~candidates["roi_id"].isin(queried_ids)].copy()
    if eligible.empty:
        raise ValueError("no feasible v4 ROI candidates remain")
    features = _candidate_features(
        eligible, prediction, observed_mask, distance_pixels, channels, config, query_index, neural_variance
    )
    if policy == "uniform":
        chosen = eligible.sort_values(["row0", "column0"]).iloc[0]
        features["selection_utility"] = np.nan
    elif policy == "random":
        chosen = eligible.iloc[int(rng.integers(0, len(eligible)))]
        features["selection_utility"] = np.nan
    elif policy in {"uncertainty_distance_sequential", "uncertainty_distance_one_anchor"}:
        features["selection_utility"] = features["uncertainty_mean"]
        chosen = eligible.iloc[int(np.argmax(features["selection_utility"].values))]
    elif policy == "uncertainty_lookahead":
        features["selection_utility"] = features["coverage_gain_per_cost"]
        chosen = eligible.iloc[int(np.argmax(features["selection_utility"].values))]
    elif policy in {
        "uncertainty_calibrated_guarded",
        "uncertainty_neural_guarded_selector_only",
        "uncertainty_neural_guarded_full_system",
    }:
        features = _guarded_scores(features, calibrator)
        features["selection_utility"] = features["guarded_utility"]
        chosen = eligible.iloc[int(np.argmax(features["selection_utility"].values))]
    elif policy == "oracle_composite_gain":
        current_error = morphology_composite_error(reference, prediction, x, y, config)
        features["oracle_composite_gain"] = [
            _hypothetical_composite_gain(
                dense_signal, observed_mask, roi, x, y, channels, config, reference, current_error
            )
            for _, roi in eligible.iterrows()
        ]
        features["selection_utility"] = features["oracle_composite_gain"]
        chosen = eligible.iloc[int(np.argmax(features["selection_utility"].values))]
    else:
        raise ValueError(f"unsupported v4 policy: {policy}")
    features["selected"] = features["roi_id"] == str(chosen["roi_id"])
    return chosen, features


def run_v4_slice_replay(
    config: RunConfig,
    dense_signal: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    channels: list[str],
    slice_id: str,
    policy: str,
    fold_id: str = "smoke",
    seed: int = 0,
    calibrator: GuardedCalibrator | None = None,
    neural: NeuralReconstructionEnsemble | None = None,
    checkpoint_callback: Callable[[pd.DataFrame, pd.DataFrame], None] | None = None,
) -> V4SliceResult:
    """Run one fixed-budget v4 replay while keeping dense values hidden from deployable selectors."""

    if policy not in V4_POLICIES:
        raise ValueError(f"unsupported v4 policy: {policy}")
    if policy in V4_NEURAL_POLICIES and neural is None:
        raise ValueError(f"{policy} requires a trained neural ensemble")
    candidates = build_v3_roi_catalog(x, y, config.acquisition_v4.roi_size_px)
    if config.acquisition_v4.total_rois > len(candidates):
        raise ValueError("v4 total_rois exceeds ROI catalog size")
    rng = np.random.default_rng(seed)
    observed_mask = np.zeros(dense_signal.shape[1:], dtype=bool)
    queried_ids: set[str] = set()
    metrics: list[dict[str, object]] = []
    trace: list[pd.DataFrame] = []
    consumed_time_s = 0.0
    consumed_dose = 0.0
    reference = pseudo_reference_from_dense_signal(dense_signal, x, y, channels, config)
    prediction = reconstruct_from_observed_mask(dense_signal, observed_mask, x, y, channels, config)
    adaptive_records: list[tuple[pd.Series, float]] = []
    anchor_queue: list[pd.Series] = []

    def reveal(roi: pd.Series, stage: str) -> None:
        nonlocal consumed_time_s, consumed_dose, prediction
        row0, row1 = int(roi["row0"]), int(roi["row1"])
        column0, column1 = int(roi["column0"]), int(roi["column1"])
        observed_mask[row0:row1, column0:column1] = True
        queried_ids.add(str(roi["roi_id"]))
        time_s, dose = _raster_cost(config, roi)
        consumed_time_s += time_s
        consumed_dose += dose
        nearest_prediction = reconstruct_from_observed_mask(dense_signal, observed_mask, x, y, channels, config)
        if policy == "uncertainty_neural_guarded_full_system":
            neural_mean, neural_variance = neural.predict(dense_signal, observed_mask)
            prediction = _prediction_from_neural(neural_mean, neural_variance, x, y, channels, config)
        else:
            prediction = nearest_prediction
        score = _score_v4_prediction(
            config,
            fold_id,
            policy,
            slice_id,
            len(metrics) + 1,
            stage,
            observed_mask,
            dense_signal,
            reference,
            prediction,
            consumed_time_s,
            consumed_dose,
        )
        score["roi_id"] = str(roi["roi_id"])
        metrics.append(score)
        if checkpoint_callback is not None:
            checkpoint_callback(
                pd.DataFrame(metrics),
                pd.concat(trace, ignore_index=True) if trace else pd.DataFrame(),
            )

    pilots = rng.choice(len(candidates), size=config.acquisition_v4.pilot_rois, replace=False)
    for index in pilots:
        reveal(candidates.iloc[int(index)], "random_pilot")
    while len(metrics) < config.acquisition_v4.total_rois:
        if (
            policy == "uncertainty_distance_one_anchor"
            and len(adaptive_records) >= config.acquisition_v4.historical_adaptive_rois_before_anchor
        ):
            if not anchor_queue and adaptive_records:
                anchor = max(adaptive_records, key=lambda item: item[1])[0]
                anchor_queue = _neighbor_candidates(candidates, anchor, queried_ids)
            if anchor_queue:
                reveal(anchor_queue.pop(0), "historical_one_anchor_neighbor")
                continue
        distance_pixels = _distance_pixels(observed_mask)
        neural_variance = None
        selection_prediction = reconstruct_from_observed_mask(
            dense_signal, observed_mask, x, y, channels, config
        )
        if policy in V4_NEURAL_POLICIES:
            _, neural_variance = neural.predict(dense_signal, observed_mask)
        selected, candidate_scores = _select_by_policy(
            policy,
            candidates,
            queried_ids,
            selection_prediction,
            observed_mask,
            distance_pixels,
            channels,
            config,
            rng,
            len(metrics),
            calibrator,
            neural_variance,
            dense_signal,
            x,
            y,
            reference,
        )
        candidate_scores.insert(0, "fold", fold_id)
        candidate_scores.insert(1, "slice", slice_id)
        candidate_scores.insert(2, "policy", policy)
        candidate_scores.insert(3, "query_index", len(metrics) + 1)
        trace.append(candidate_scores)
        selected_utility = float(candidate_scores.loc[candidate_scores["selected"], "selection_utility"].iloc[0])
        reveal(selected, f"{policy}_adaptive")
        adaptive_records.append((selected, selected_utility))
    return V4SliceResult(
        metrics=pd.DataFrame(metrics),
        candidate_trace=pd.concat(trace, ignore_index=True) if trace else pd.DataFrame(),
    )


def _contiguous_runs(numbers: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for number in sorted(numbers):
        if not runs or number != runs[-1][-1] + 1:
            runs.append([number])
        else:
            runs[-1].append(number)
    return runs


def build_v4_folds(slice_ids: list[str], config: RunConfig) -> list[V4Fold]:
    """Build deterministic blocked folds with outer and validation guard regions."""

    available = sorted({int(value) for value in slice_ids})
    available_set = set(available)
    folds = []
    settings = config.acquisition_v4.folds
    for index, (first, last) in enumerate(settings.outer_test_ranges, start=1):
        test = [value for value in available if first <= value <= last]
        if not test:
            continue
        outer_guard = {
            value
            for value in available
            if value not in test
            and min(abs(value - test[0]), abs(value - test[-1])) <= settings.outer_guard_slices
        }
        remaining = sorted(available_set - set(test) - outer_guard)
        longest = max(_contiguous_runs(remaining), key=lambda run: (len(run), -run[0]))
        validation_count = min(
            settings.validation_slices,
            config.acquisition_v4.calibrator.max_validation_slices,
            len(longest),
        )
        validation = longest[:validation_count]
        validation_guard = {
            value
            for value in remaining
            if value not in validation
            and min(abs(value - validation[0]), abs(value - validation[-1]))
            <= settings.validation_guard_slices
        }
        training = sorted(set(remaining) - set(validation) - validation_guard)
        folds.append(
            V4Fold(
                fold_id=f"fold_{index}",
                test_slices=[f"{value:03d}" for value in test],
                validation_slices=[f"{value:03d}" for value in validation],
                training_slices=[f"{value:03d}" for value in training],
                excluded_guard_slices=[f"{value:03d}" for value in sorted(outer_guard | validation_guard)],
            )
        )
    return folds


def _sample_slice_ids(slice_ids: list[str], maximum: int) -> list[str]:
    if len(slice_ids) <= maximum:
        return slice_ids
    indices = np.linspace(0, len(slice_ids) - 1, maximum, dtype=int)
    return [slice_ids[index] for index in indices]


def _collect_calibrator_examples(
    config: RunConfig,
    slice_ids: list[str],
    loader: Callable[[str], tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]],
    seed: int,
    neural: NeuralReconstructionEnsemble | None = None,
) -> pd.DataFrame:
    rows = []
    settings = config.acquisition_v4.calibrator
    for slice_id in _sample_slice_ids(slice_ids, settings.max_training_slices):
        dense_signal, x, y, channels = loader(slice_id)
        reference = pseudo_reference_from_dense_signal(dense_signal, x, y, channels, config)
        catalog = build_v3_roi_catalog(x, y, config.acquisition_v4.roi_size_px)
        rng = np.random.default_rng(seed)
        observed_mask = np.zeros(dense_signal.shape[1:], dtype=bool)
        queried: set[str] = set()
        for index in rng.choice(len(catalog), size=config.acquisition_v4.pilot_rois, replace=False):
            roi = catalog.iloc[int(index)]
            observed_mask[int(roi["row0"]) : int(roi["row1"]), int(roi["column0"]) : int(roi["column1"])] = True
            queried.add(str(roi["roi_id"]))
        for _ in range(settings.max_training_states_per_slice):
            prediction = reconstruct_from_observed_mask(dense_signal, observed_mask, x, y, channels, config)
            distance_pixels = _distance_pixels(observed_mask)
            eligible = catalog[~catalog["roi_id"].isin(queried)].copy()
            neural_variance = neural.predict(dense_signal, observed_mask)[1] if neural is not None else None
            features = _candidate_features(
                eligible,
                prediction,
                observed_mask,
                distance_pixels,
                channels,
                config,
                len(queried),
                neural_variance,
            ).sort_values("coverage_gain_per_cost", ascending=False)
            sample_indices = np.linspace(
                0, len(features) - 1, min(settings.max_candidates_per_state, len(features)), dtype=int
            )
            sampled = features.iloc[sample_indices].copy()
            current_error = morphology_composite_error(reference, prediction, x, y, config)
            by_id = eligible.set_index("roi_id")
            sampled["composite_gain_label"] = [
                _hypothetical_composite_gain(
                    dense_signal,
                    observed_mask,
                    by_id.loc[roi_id],
                    x,
                    y,
                    channels,
                    config,
                    reference,
                    current_error,
                )
                for roi_id in sampled["roi_id"]
            ]
            sampled["training_slice"] = slice_id
            rows.append(sampled)
            selected = by_id.loc[str(features.iloc[0]["roi_id"])]
            observed_mask[
                int(selected["row0"]) : int(selected["row1"]),
                int(selected["column0"]) : int(selected["column1"]),
            ] = True
            queried.add(str(selected.name))
    return pd.concat(rows, ignore_index=True)


def _fit_calibrator(
    config: RunConfig,
    fold: V4Fold,
    loader: Callable[[str], tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]],
    seed: int,
    neural: NeuralReconstructionEnsemble | None = None,
) -> GuardedCalibrator:
    examples = _collect_calibrator_examples(config, fold.training_slices, loader, seed, neural)
    excluded = {"roi_id", "composite_gain_label", "training_slice"}
    features = [column for column in examples.columns if column not in excluded]
    model = HistGradientBoostingRegressor(max_iter=120, random_state=seed)
    model.fit(examples[features], examples["composite_gain_label"])
    best: tuple[float, float, float] | None = None
    validation_ids = _sample_slice_ids(
        fold.validation_slices, config.acquisition_v4.calibrator.max_validation_slices
    )
    for ratio in sorted(config.acquisition_v4.calibrator.shortlist_ratios, reverse=True):
        for weight in sorted(config.acquisition_v4.calibrator.learned_weights):
            calibrator = GuardedCalibrator(
                model, features, ratio, weight, len(examples), fold.training_slices, fold.validation_slices
            )
            errors = []
            for slice_id in validation_ids:
                dense_signal, x, y, channels = loader(slice_id)
                policy = (
                    "uncertainty_neural_guarded_selector_only"
                    if neural is not None
                    else "uncertainty_calibrated_guarded"
                )
                result = run_v4_slice_replay(
                    config,
                    dense_signal,
                    x,
                    y,
                    channels,
                    slice_id,
                    policy,
                    fold.fold_id,
                    seed,
                    calibrator,
                    neural,
                )
                errors.append(float(result.metrics.iloc[-1]["morphology_composite_error"]))
            candidate = (float(np.mean(errors)), -ratio, weight)
            if best is None or candidate < best:
                best = candidate
    return GuardedCalibrator(
        model=model,
        feature_columns=features,
        shortlist_ratio=-best[1],
        learned_weight=best[2],
        training_rows=len(examples),
        training_slices=fold.training_slices,
        validation_slices=fold.validation_slices,
    )


def _mean_ci(values: pd.Series) -> tuple[float, float, float]:
    clean = values.dropna().to_numpy(float)
    if clean.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(clean))
    if clean.size == 1:
        return mean, np.nan, np.nan
    margin = float(t.ppf(0.975, clean.size - 1) * np.std(clean, ddof=1) / np.sqrt(clean.size))
    return mean, mean - margin, mean + margin


def _auc(frame: pd.DataFrame, metric: str) -> float:
    ordered = frame.sort_values("scan_time_s")
    return float(np.trapezoid(ordered[metric].to_numpy(float), ordered["scan_time_s"].to_numpy(float)))


def summarize_v4_metrics(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final = metrics.sort_values("iteration").groupby(["fold", "slice", "policy"], sort=False).tail(1)
    rows = []
    for policy, frame in final.groupby("policy", sort=False):
        rows.append(
            {
                "policy": policy,
                "slices": frame["slice"].nunique(),
                "mean_morphology_composite_error": frame["morphology_composite_error"].mean(),
                "mean_front_distance_nm": frame["front_mean_symmetric_distance_nm"].mean(),
                "mean_penetration_d95_error_nm": frame["penetration_d95_absolute_error_nm"].mean(),
                "mean_penetration_dmax_error_nm": frame["penetration_dmax_absolute_error_nm"].mean(),
                "mean_normalized_reconstruction_rmse": frame["normalized_reconstruction_rmse"].mean(),
                "mean_selected_area_fraction": frame["selected_area_fraction"].mean(),
                "mean_scan_time_s": frame["scan_time_s"].mean(),
            }
        )
    summary = pd.DataFrame(rows).sort_values("mean_morphology_composite_error")
    curves = (
        metrics.groupby(["policy", "query_count"], as_index=False)
        .agg(
            mean_morphology_composite_error=("morphology_composite_error", "mean"),
            mean_front_distance_nm=("front_mean_symmetric_distance_nm", "mean"),
            mean_penetration_d95_error_nm=("penetration_d95_absolute_error_nm", "mean"),
            mean_reconstruction_rmse=("normalized_reconstruction_rmse", "mean"),
            mean_scan_time_s=("scan_time_s", "mean"),
            mean_selected_area_fraction=("selected_area_fraction", "mean"),
        )
    )
    auc_rows = []
    for (fold, slice_id, policy), frame in metrics.groupby(["fold", "slice", "policy"], sort=False):
        auc_rows.append(
            {
                "fold": fold,
                "slice": slice_id,
                "policy": policy,
                "composite_error_auc_vs_cost": _auc(frame, "morphology_composite_error"),
            }
        )
    auc = pd.DataFrame(auc_rows)
    return summary, curves, auc


def paired_v4_comparisons(metrics: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    final = metrics.sort_values("iteration").groupby(["fold", "slice", "policy"], sort=False).tail(1)
    baseline_candidates = final[
        final["policy"].isin(
            ["uncertainty_distance_sequential", "uncertainty_distance_one_anchor"]
        )
    ]
    baseline_name = (
        baseline_candidates.groupby("policy")["morphology_composite_error"].mean().idxmin()
    )
    baseline = final[final["policy"] == baseline_name].set_index(["fold", "slice"])
    rows = []
    for policy in sorted(set(final["policy"]) - {baseline_name, "oracle_composite_gain"}):
        candidate = final[final["policy"] == policy].set_index(["fold", "slice"])
        joined = candidate.join(baseline, lsuffix="_candidate", rsuffix="_baseline", how="inner")
        composite_delta = (
            joined["morphology_composite_error_candidate"]
            - joined["morphology_composite_error_baseline"]
        )
        rmse_delta_fraction = (
            joined["normalized_reconstruction_rmse_candidate"]
            / joined["normalized_reconstruction_rmse_baseline"]
            - 1.0
        )
        mean, low, high = _mean_ci(composite_delta)
        rows.append(
            {
                "policy": policy,
                "baseline": baseline_name,
                "paired_slices": len(joined),
                "mean_composite_error_delta": mean,
                "composite_error_delta_ci95_low": low,
                "composite_error_delta_ci95_high": high,
                "composite_error_win_rate": float((composite_delta < 0).mean()),
                "mean_rmse_regression_fraction": float(rmse_delta_fraction.mean()),
                "equal_mean_scan_cost": bool(
                    np.allclose(
                        joined["scan_time_s_candidate"], joined["scan_time_s_baseline"], atol=1e-9
                    )
                ),
                "promoted": bool(
                    high < 0.0
                    and float(rmse_delta_fraction.mean())
                    <= config.acquisition_v4.rmse_regression_limit_fraction
                    and np.allclose(
                        joined["scan_time_s_candidate"], joined["scan_time_s_baseline"], atol=1e-9
                    )
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_composite_error_delta")


def _manifest_slice_ids(manifest_path: Path) -> list[str]:
    manifest = pd.read_csv(manifest_path, dtype={"slice": str})
    return sorted(manifest["slice"].str.zfill(3).unique().tolist())


def _checkpoint_part_path(directory: Path, fold: str, slice_id: str, policy: str) -> Path:
    return directory / f"{fold}__{slice_id}__{policy}.csv"


def _write_checkpoint_part(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _read_checkpoint_frames(directory: Path) -> list[pd.DataFrame]:
    if not directory.exists():
        return []
    return [
        pd.read_csv(path, dtype={"slice": str}, low_memory=False)
        for path in sorted(directory.glob("*.csv"))
    ]


def _deduplicate_metrics(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["slice"] = combined["slice"].astype(str).str.zfill(3)
    return combined.drop_duplicates(["fold", "slice", "policy", "iteration"], keep="last")


def _deduplicate_trace(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["slice"] = combined["slice"].astype(str).str.zfill(3)
    return combined.drop_duplicates(["fold", "slice", "policy", "query_index", "roi_id"], keep="last")


def run_v4_uncertainty_stack_validation(
    template_path: Path,
    output: Path,
    manifest_path: Path,
    fold_specification: str = "all",
    slice_ids: list[str] | None = None,
    policies: list[str] | None = None,
    seed: int = 0,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run out-of-fold v4 uncertainty validation with resumable fixed-budget replays."""

    template = load_config(template_path)
    all_slice_ids = _manifest_slice_ids(manifest_path)
    requested = set(slice_ids or all_slice_ids)
    folds = build_v4_folds(all_slice_ids, template)
    if fold_specification != "all":
        folds = [fold for fold in folds if fold.fold_id == f"fold_{int(fold_specification)}"]
    policies = policies or V4_POLICIES
    unknown = set(policies) - set(V4_POLICIES)
    if unknown:
        raise ValueError(f"unsupported v4 policies: {sorted(unknown)}")
    runnable = list(policies)
    skipped: dict[str, str] = {}
    if not template.acquisition_v4.neural.enabled:
        for policy in sorted(V4_NEURAL_POLICIES & set(runnable)):
            runnable.remove(policy)
            skipped[policy] = "neural training is disabled by acquisition_v4.neural.enabled"
    elif not torch_available():
        for policy in sorted(V4_NEURAL_POLICIES & set(runnable)):
            runnable.remove(policy)
            skipped[policy] = "optional torch learned runtime is not installed"
    sources = sources_from_manifest(manifest_path, all_slice_ids, template.scenario.elements)
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}

    def load_slice(slice_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        if slice_id not in cache:
            config = v3_config_for_slice(template, slice_id, sources[slice_id])
            observations, _ = ingest_dataset(config)
            cache[slice_id] = dense_signal_from_observations(config, observations)
        return cache[slice_id]

    output.mkdir(parents=True, exist_ok=True)
    write_config(template, output / "resolved_template_config.yaml")
    metrics_path = output / "v4_metrics_by_iteration.partial.csv"
    trace_path = output / "v4_candidate_trace.partial.csv"
    metrics_parts = output / "v4_metrics_parts"
    trace_parts = output / "v4_candidate_trace_parts"
    metrics_frames: list[pd.DataFrame] = []
    trace_frames: list[pd.DataFrame] = []
    completed: set[tuple[str, str, str]] = set()
    if resume and metrics_path.exists():
        existing = pd.read_csv(metrics_path, dtype={"slice": str}, low_memory=False)
        existing["slice"] = existing["slice"].str.zfill(3)
        metrics_frames.append(existing)
        if trace_path.exists() and trace_path.stat().st_size:
            trace_frames.append(pd.read_csv(trace_path, dtype={"slice": str}, low_memory=False))
    metrics_frames.extend(_read_checkpoint_frames(metrics_parts))
    trace_frames.extend(_read_checkpoint_frames(trace_parts))
    existing_metrics = _deduplicate_metrics(metrics_frames)
    existing_trace = _deduplicate_trace(trace_frames)
    if not existing_metrics.empty:
        metric_complete = set(
            existing_metrics[
                existing_metrics["query_count"] >= template.acquisition_v4.total_rois
            ][["fold", "slice", "policy"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        trace_complete = set(
            existing_trace[["fold", "slice", "policy"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        completed = metric_complete & trace_complete
    oracle_ids = set(_sample_slice_ids(all_slice_ids, template.acquisition_v4.oracle_sample_slices))
    model_manifest: list[dict[str, object]] = []
    processed = 0
    for fold in folds:
        tests = [slice_id for slice_id in fold.test_slices if slice_id in requested]
        if not tests:
            continue
        needs_calibrator = any(
            policy in {"uncertainty_calibrated_guarded", *V4_NEURAL_POLICIES}
            and any((fold.fold_id, slice_id, policy) not in completed for slice_id in tests)
            for policy in runnable
        )
        calibrator = None
        neural = None
        neural_calibrator = None
        if needs_calibrator and template.acquisition_v4.calibrator.enabled:
            calibrator = _fit_calibrator(template, fold, load_slice, seed)
            model_manifest.append(
                {
                    "fold": fold.fold_id,
                    "model": "hist_gradient_boosting_guarded_gain",
                    "training_rows": calibrator.training_rows,
                    "shortlist_ratio": calibrator.shortlist_ratio,
                    "learned_weight": calibrator.learned_weight,
                }
            )
        if any(policy in V4_NEURAL_POLICIES for policy in runnable):
            signals = [
                load_slice(slice_id)[0]
                for slice_id in _sample_slice_ids(
                    fold.training_slices, template.acquisition_v4.neural.max_training_slices
                )
            ]
            neural = train_neural_ensemble(signals, template.acquisition_v4.neural, seed)
            neural_calibrator = _fit_calibrator(template, fold, load_slice, seed, neural)
            model_manifest.append(
                {
                    "fold": fold.fold_id,
                    "model": "compact_unet_ensemble_guarded_gain",
                    "ensemble_size": template.acquisition_v4.neural.ensemble_size,
                    "training_rows": neural_calibrator.training_rows,
                    "shortlist_ratio": neural_calibrator.shortlist_ratio,
                    "learned_weight": neural_calibrator.learned_weight,
                }
            )
        for slice_id in tests:
            config = v3_config_for_slice(template, slice_id, sources[slice_id])
            dense_signal, x, y, channels = load_slice(slice_id)
            for policy in runnable:
                if policy == "oracle_composite_gain" and slice_id not in oracle_ids:
                    continue
                if (fold.fold_id, slice_id, policy) in completed:
                    continue
                model = neural_calibrator if policy in V4_NEURAL_POLICIES else calibrator
                result = run_v4_slice_replay(
                    config,
                    dense_signal,
                    x,
                    y,
                    channels,
                    slice_id,
                    policy,
                    fold.fold_id,
                    seed,
                    model,
                    neural,
                    checkpoint_callback=lambda metric_frame, trace_frame, fold_id=fold.fold_id, current_slice=slice_id, current_policy=policy: (
                        _write_checkpoint_part(
                            metric_frame,
                            _checkpoint_part_path(
                                metrics_parts, fold_id, current_slice, current_policy
                            ),
                        ),
                        (
                            _write_checkpoint_part(
                                trace_frame,
                                _checkpoint_part_path(
                                    trace_parts, fold_id, current_slice, current_policy
                                ),
                            )
                            if not trace_frame.empty
                            else None
                        ),
                    ),
                )
                metrics_frames.append(result.metrics)
                if not result.candidate_trace.empty:
                    trace_frames.append(result.candidate_trace)
                _write_checkpoint_part(
                    result.metrics,
                    _checkpoint_part_path(metrics_parts, fold.fold_id, slice_id, policy),
                )
                if not result.candidate_trace.empty:
                    _write_checkpoint_part(
                        result.candidate_trace,
                        _checkpoint_part_path(trace_parts, fold.fold_id, slice_id, policy),
                    )
            processed += 1
            if processed % 10 == 0:
                print(f"Validated v4 uncertainty reconstruction on {processed} requested slices.")
    metrics = _deduplicate_metrics(metrics_frames)
    trace = _deduplicate_trace(trace_frames)
    summary, curves, auc = summarize_v4_metrics(metrics)
    final = metrics.sort_values("iteration").groupby(["fold", "slice", "policy"], sort=False).tail(1)
    comparisons = paired_v4_comparisons(metrics, template)
    final.to_csv(output / "v4_final_metrics_by_slice.csv", index=False)
    metrics.to_csv(output / "v4_metrics_by_iteration.csv", index=False)
    trace.to_csv(output / "v4_candidate_trace.csv", index=False)
    summary.to_csv(output / "v4_oof_summary.csv", index=False)
    curves.to_csv(output / "v4_error_vs_cost_curves.csv", index=False)
    auc.to_csv(output / "v4_composite_error_auc_vs_cost.csv", index=False)
    comparisons.to_csv(output / "v4_paired_comparisons.csv", index=False)
    final[final["policy"] == "oracle_composite_gain"].to_csv(
        output / "v4_oracle_headroom_summary.csv", index=False
    )
    blocks = final.copy()
    blocks["slice_block"] = blocks["slice"].astype(int).map(
        lambda value: f"{((value - 1) // 25) * 25 + 1:03d}:{((value - 1) // 25) * 25 + 25:03d}"
    )
    blocks.groupby(["policy", "slice_block"], as_index=False).agg(
        slices=("slice", "nunique"),
        mean_morphology_composite_error=("morphology_composite_error", "mean"),
        mean_front_distance_nm=("front_mean_symmetric_distance_nm", "mean"),
        mean_penetration_d95_error_nm=("penetration_d95_absolute_error_nm", "mean"),
    ).to_csv(output / "v4_slice_block_summary.csv", index=False)
    protocol = {
        "schema": "balance_nm_v4_uncertainty_first_adaptive_reconstruction",
        "template_config": str(template_path),
        "manifest": str(manifest_path),
        "seed": seed,
        "requested_slices": sorted(requested),
        "runnable_policies": runnable,
        "skipped_policies": skipped,
        "folds": [fold.__dict__ for fold in folds],
        "dense_truth_policy": "hidden from deployable selectors; evaluation-only replay reference",
        "front_semantics": "frozen unsupervised alteration-front proxy; not expert-labeled corrosion truth",
    }
    with (output / "v4_fold_protocol.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(protocol, handle, sort_keys=False)
    with (output / "v4_model_manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"models": model_manifest, "skipped_policies": skipped}, handle, sort_keys=False)
    return metrics, summary
