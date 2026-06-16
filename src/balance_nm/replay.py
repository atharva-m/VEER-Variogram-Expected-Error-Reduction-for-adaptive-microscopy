"""Replay plumbing: ROI catalog, raster cost, folds, scoring, and checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.stats import t

from .domain import RunConfig
from .morphology import (
    front_distance_metrics,
    front_from_probability,
    normalized_reconstruction_rmse,
    penetration_stat,
)


@dataclass(frozen=True)
class Fold:
    fold_id: str
    test_slices: list[str]
    validation_slices: list[str]
    training_slices: list[str]
    excluded_guard_slices: list[str]


def _axis_origins(size: int, span: int) -> list[int]:
    if span > size:
        raise ValueError("ROI dimensions cannot exceed the dense map")
    origins = list(range(0, size - span + 1, span))
    if origins[-1] != size - span:
        origins.append(size - span)
    return origins


def build_roi_catalog(x: np.ndarray, y: np.ndarray, roi_size_px: tuple[int, int]) -> pd.DataFrame:
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


def config_for_slice(
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


def raster_cost(config: RunConfig, roi: pd.Series) -> tuple[float, float]:
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


def slice_width_nm(x: np.ndarray) -> float:
    if len(x) <= 1:
        return 1.0
    return float(x[-1] - x[0] + np.median(np.diff(x)))


def distance_pixels(observed_mask: np.ndarray) -> np.ndarray:
    if not np.any(observed_mask):
        return np.full(observed_mask.shape, float(np.hypot(*observed_mask.shape)) / 2.0)
    return distance_transform_edt(~observed_mask)


def rectangle_distance_pixels(shape: tuple[int, int], roi: pd.Series) -> np.ndarray:
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


def lookahead_coverage_gain(distance_field: np.ndarray, roi: pd.Series) -> float:
    """Exact integrated distance-uncertainty reduction for revealing a rectangular ROI."""

    rectangle_distance = rectangle_distance_pixels(distance_field.shape, roi)
    updated = np.minimum(distance_field, rectangle_distance)
    maximum = max(float(np.hypot(*distance_field.shape)) / 2.0, 1.0)
    return float(np.sum(distance_field - updated) / maximum)


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
    width = slice_width_nm(x)
    if np.isfinite(reference_d95) and np.isfinite(predicted_d95):
        penetration_error = abs(predicted_d95 - reference_d95)
    elif not np.isfinite(reference_d95) and not np.isfinite(predicted_d95):
        penetration_error = 0.0
    else:
        penetration_error = width
    weights = config.acquisition
    total_weight = weights.front_weight + weights.penetration_d95_weight
    return float(
        (
            weights.front_weight * front.mean_symmetric_distance_nm / width
            + weights.penetration_d95_weight * penetration_error / width
        )
        / total_weight
    )


def score_prediction(
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
    width = slice_width_nm(x)

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


def _contiguous_runs(numbers: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for number in sorted(numbers):
        if not runs or number != runs[-1][-1] + 1:
            runs.append([number])
        else:
            runs[-1].append(number)
    return runs


def build_folds(slice_ids: list[str], config: RunConfig) -> list[Fold]:
    """Build deterministic blocked folds with outer and validation guard regions."""

    available = sorted({int(value) for value in slice_ids})
    available_set = set(available)
    folds = []
    settings = config.acquisition.folds
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
        validation_count = min(settings.validation_slices, len(longest))
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
            Fold(
                fold_id=f"fold_{index}",
                test_slices=[f"{value:03d}" for value in test],
                validation_slices=[f"{value:03d}" for value in validation],
                training_slices=[f"{value:03d}" for value in training],
                excluded_guard_slices=[f"{value:03d}" for value in sorted(outer_guard | validation_guard)],
            )
        )
    return folds


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


def summarize_metrics(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def paired_morphology_comparisons(metrics: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    final = metrics.sort_values("iteration").groupby(["fold", "slice", "policy"], sort=False).tail(1)
    baseline = final[final["policy"] == "uncertainty_lookahead"].set_index(["fold", "slice"])
    rows = []
    for policy in sorted(set(final["policy"]) - {"uncertainty_lookahead", "oracle_composite_gain"}):
        candidate = final[final["policy"] == policy].set_index(["fold", "slice"])
        joined = candidate.join(baseline, lsuffix="_candidate", rsuffix="_baseline", how="inner")
        delta = (
            joined["morphology_composite_error_candidate"]
            - joined["morphology_composite_error_baseline"]
        )
        rmse_delta = (
            joined["normalized_reconstruction_rmse_candidate"]
            / joined["normalized_reconstruction_rmse_baseline"]
            - 1.0
        )
        mean, low, high = _mean_ci(delta)
        equal_cost = bool(
            np.allclose(joined["scan_time_s_candidate"], joined["scan_time_s_baseline"], atol=1.0e-9)
        )
        leave_one_out = (
            max(float((delta.sum() - value) / (len(delta) - 1)) for value in delta)
            if len(delta) > 1
            else np.nan
        )
        fold_means = delta.groupby(level="fold").mean()
        nonpositive_folds = int((fold_means <= 0.0).sum())
        rmse_regression = float(rmse_delta.mean())
        maximum_regression = float(delta.max())
        limit = config.acquisition.rmse_regression_limit_fraction
        rows.append(
            {
                "policy": policy,
                "baseline": "uncertainty_lookahead",
                "paired_slices": len(joined),
                "mean_composite_error_delta": mean,
                "median_composite_error_delta": float(delta.median()),
                "composite_error_delta_ci95_low": low,
                "composite_error_delta_ci95_high": high,
                "composite_error_win_rate": float((delta < 0.0).mean()),
                "leave_one_slice_out_worst_mean_delta": leave_one_out,
                "nonpositive_fold_means": nonpositive_folds,
                "mean_rmse_regression_fraction": rmse_regression,
                "maximum_slice_composite_error_regression": maximum_regression,
                "equal_mean_scan_cost": equal_cost,
                "passes_ten_slice_gate": bool(
                    mean <= 0.0
                    and rmse_regression <= limit
                    and maximum_regression <= 0.02
                    and equal_cost
                ),
                "passes_thirty_slice_gate": bool(
                    mean < 0.0
                    and float(delta.median()) <= 0.0
                    and leave_one_out <= 0.0
                    and nonpositive_folds >= 3
                    and rmse_regression <= limit
                    and maximum_regression <= 0.02
                    and equal_cost
                ),
                "promoted": bool(
                    high < 0.0 and rmse_regression <= limit and equal_cost
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_composite_error_delta") if rows else pd.DataFrame()


def manifest_slice_ids(manifest_path: Path) -> list[str]:
    manifest = pd.read_csv(manifest_path, dtype={"slice": str})
    return sorted(manifest["slice"].str.zfill(3).unique().tolist())


def checkpoint_part_path(directory: Path, fold: str, slice_id: str, policy: str) -> Path:
    return directory / f"{fold}__{slice_id}__{policy}.csv"


def write_checkpoint_part(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def read_checkpoint_frames(directory: Path) -> list[pd.DataFrame]:
    if not directory.exists():
        return []
    return [
        pd.read_csv(path, dtype={"slice": str}, low_memory=False)
        for path in sorted(directory.glob("*.csv"))
    ]


def deduplicate_metrics(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["slice"] = combined["slice"].astype(str).str.zfill(3)
    return combined.drop_duplicates(["fold", "slice", "policy", "iteration"], keep="last")


def deduplicate_trace(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["slice"] = combined["slice"].astype(str).str.zfill(3)
    return combined.drop_duplicates(["fold", "slice", "policy", "query_index", "roi_id"], keep="last")
