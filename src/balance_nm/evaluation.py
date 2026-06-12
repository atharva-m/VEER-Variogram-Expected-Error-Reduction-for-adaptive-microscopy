"""Metrics for multi-objective discovery, reconstruction, and uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.metrics import average_precision_score

from .domain import ExperimentState, MetricRecord, ObjectiveMetricRecord, RunConfig


def predicted_interface_mask(phase_probability: np.ndarray) -> np.ndarray:
    phase = phase_probability >= 0.5
    boundary = phase ^ binary_erosion(phase)
    boundary[[0, -1], :] = False
    boundary[:, [0, -1]] = False
    return boundary


def interface_distances_nm(
    true_mask: np.ndarray, predicted_mask: np.ndarray, dy_nm: float, dx_nm: float
) -> np.ndarray:
    if not np.any(true_mask) or not np.any(predicted_mask):
        return np.array([float(np.hypot(true_mask.shape[0] * dy_nm, true_mask.shape[1] * dx_nm))])
    distance_to_true = distance_transform_edt(~true_mask, sampling=(dy_nm, dx_nm))
    distance_to_prediction = distance_transform_edt(~predicted_mask, sampling=(dy_nm, dx_nm))
    return np.concatenate([distance_to_true[predicted_mask], distance_to_prediction[true_mask]])


def continuous_interface_x_nm(phase_probability: np.ndarray, x_nm: np.ndarray) -> np.ndarray:
    """Interpolate each row's first 0.5 contour crossing for subpixel error metrics."""

    crossings = np.full(phase_probability.shape[0], np.nan)
    for row, probability in enumerate(phase_probability):
        shifted = probability - 0.5
        indices = np.flatnonzero(shifted[:-1] * shifted[1:] <= 0)
        if not indices.size:
            continue
        index = int(indices[np.argmin(np.abs(shifted[indices]))])
        p0, p1 = probability[index], probability[index + 1]
        fraction = 0.5 if np.isclose(p0, p1) else float((0.5 - p0) / (p1 - p0))
        crossings[row] = x_nm[index] + np.clip(fraction, 0.0, 1.0) * (x_nm[index + 1] - x_nm[index])
    return crossings


def continuous_interface_distances_nm(
    true_x_nm: np.ndarray, predicted_x_nm: np.ndarray, width_nm: float
) -> np.ndarray:
    valid = np.isfinite(true_x_nm) & np.isfinite(predicted_x_nm)
    if not np.any(valid):
        return np.array([width_nm])
    distances = np.abs(true_x_nm[valid] - predicted_x_nm[valid])
    missing = int(np.count_nonzero(np.isfinite(true_x_nm) & ~np.isfinite(predicted_x_nm)))
    if missing:
        distances = np.concatenate([distances, np.full(missing, width_nm)])
    return distances


class Evaluator:
    def __init__(self, config: RunConfig, policy: str, seed: int):
        self.config = config
        self.policy = policy
        self.seed = seed

    def score(
        self,
        reference_sample: xr.Dataset,
        state: ExperimentState,
        prediction: xr.Dataset,
        iteration: int,
    ) -> MetricRecord:
        x = prediction.coords["x"].values
        score_interface = self.config.dataset.mode == "synthetic" or "interface" in self.config.objectives.enabled
        if score_interface and "true_interface_x_nm" in reference_sample:
            true_x = reference_sample["true_interface_x_nm"].values
            predicted_x = continuous_interface_x_nm(prediction["phase_probability"].values, x)
            distances = continuous_interface_distances_nm(true_x, predicted_x, float(x[-1] - x[0]))
        elif score_interface:
            dx = float(np.mean(np.diff(x)))
            y = prediction.coords["y"].values
            dy = float(np.mean(np.diff(y)))
            distances = interface_distances_nm(
                reference_sample["true_interface"].values.astype(bool),
                predicted_interface_mask(prediction["phase_probability"].values),
                dy,
                dx,
            )
        else:
            distances = None
        true_rate = reference_sample["true_rate"].values
        true_composition = true_rate / np.maximum(true_rate.sum(axis=0, keepdims=True), 1e-12)
        predicted_composition = prediction["composition"].values
        rate_rmse = float(np.sqrt(np.mean((true_rate - prediction["mean_rate"].values) ** 2)))
        channel_scale = np.maximum(
            np.nanpercentile(true_rate, 95, axis=(1, 2))
            - np.nanpercentile(true_rate, 5, axis=(1, 2)),
            1.0,
        )
        normalized_channel_rmse = float(
            np.sqrt(
                np.mean(
                    ((true_rate - prediction["mean_rate"].values) / channel_scale[:, None, None])
                    ** 2
                )
            )
        )
        composition_rmse = (
            float(np.sqrt(np.mean((true_composition - predicted_composition) ** 2)))
            if len(self.config.scenario.elements) > 1
            and self.config.dataset.value_semantics == "counts"
            else None
        )
        mean_rate = prediction["mean_rate"].values
        variance_name = "predictive_variance" if "predictive_variance" in prediction else "variance_rate"
        variance = np.maximum(prediction[variance_name].values, 1e-10)
        residual = true_rate - mean_rate
        nll = float(np.mean(0.5 * (np.log(2 * np.pi * variance) + residual**2 / variance)))
        coverage = float(np.mean(np.abs(residual) <= 1.96 * np.sqrt(variance)))
        objective_rows = self.score_objectives(reference_sample, state, prediction, iteration)
        weighted_score = 0.0
        total_weight = 0.0
        for row in objective_rows:
            if row.average_precision is not None:
                weight = self.config.objectives.weights.get(row.objective, 0.0)
                weighted_score += weight * row.average_precision
                total_weight += weight
        mean_quality = float(prediction["quality_score"].mean()) if "quality_score" in prediction else None
        return MetricRecord(
            policy=self.policy,
            seed=self.seed,
            iteration=iteration,
            scan_time_s=state.consumed_time_s,
            dose_proxy=state.consumed_dose,
            interface_mean_distance_nm=float(np.mean(distances)) if distances is not None else None,
            interface_p95_distance_nm=float(np.percentile(distances, 95)) if distances is not None else None,
            rate_rmse=rate_rmse,
            normalized_channel_rmse=normalized_channel_rmse,
            composition_rmse=composition_rmse,
            negative_log_likelihood=nll,
            coverage_95=coverage,
            weighted_pattern_score=weighted_score / total_weight if total_weight else None,
            mean_quality_score=mean_quality,
            invalid_candidate_count=len(state.rejected_actions),
            unsupported_candidate_count=len(state.unsupported_proposals),
        )

    def score_objectives(
        self,
        reference_sample: xr.Dataset,
        state: ExperimentState,
        prediction: xr.Dataset,
        iteration: int,
    ) -> list[ObjectiveMetricRecord]:
        if "pattern_interest" not in prediction:
            return []
        records = []
        for objective in self.config.objectives.enabled:
            values = prediction["pattern_probability"].sel(objective=objective).values
            average_precision = None
            if "true_pattern_mask" in reference_sample and objective in reference_sample.coords["objective"].values:
                truth = reference_sample["true_pattern_mask"].sel(objective=objective).values.astype(bool).ravel()
                if truth.any() and not truth.all():
                    average_precision = float(average_precision_score(truth, values.ravel()))
            records.append(
                ObjectiveMetricRecord(
                    policy=self.policy,
                    seed=self.seed,
                    iteration=iteration,
                    objective=objective,
                    scan_time_s=state.consumed_time_s,
                    dose_proxy=state.consumed_dose,
                    average_precision=average_precision,
                    mean_interest=float(prediction["pattern_interest"].sel(objective=objective).mean()),
                    mean_uncertainty=float(prediction["pattern_uncertainty"].sel(objective=objective).mean()),
                )
            )
        return records


def summarize_benchmark(metrics: pd.DataFrame) -> pd.DataFrame:
    run_rows = []
    for (policy, seed), run in metrics.groupby(["policy", "seed"], sort=False):
        run = run.sort_values("scan_time_s")
        x = run["scan_time_s"].to_numpy()
        y = run["interface_mean_distance_nm"].to_numpy()
        if len(x) > 1 and x[-1] > x[0]:
            integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            cost_auc = float(integrate(y, x) / (x[-1] - x[0]))
        else:
            cost_auc = float(y[-1])
        final = run.iloc[-1]
        run_rows.append(
            {
                "policy": policy,
                "seed": seed,
                "interface_error_cost_auc": cost_auc,
                "final_interface_mean_distance_nm": final["interface_mean_distance_nm"],
                "final_interface_p95_distance_nm": final["interface_p95_distance_nm"],
                "final_composition_rmse": final["composition_rmse"],
                "final_rate_rmse": final["rate_rmse"],
                "final_normalized_channel_rmse": final["normalized_channel_rmse"],
                "final_coverage_95": final["coverage_95"],
                "final_scan_time_s": final["scan_time_s"],
                "final_dose_proxy": final["dose_proxy"],
            }
        )
    runs = pd.DataFrame(run_rows)
    summary = (
        runs.groupby("policy")
        .agg(
            runs=("seed", "count"),
            interface_error_cost_auc_mean=("interface_error_cost_auc", "mean"),
            interface_error_cost_auc_std=("interface_error_cost_auc", "std"),
            final_interface_mean_distance_nm_mean=("final_interface_mean_distance_nm", "mean"),
            final_interface_mean_distance_nm_std=("final_interface_mean_distance_nm", "std"),
            final_composition_rmse_mean=("final_composition_rmse", "mean"),
            final_rate_rmse_mean=("final_rate_rmse", "mean"),
            final_normalized_channel_rmse_mean=("final_normalized_channel_rmse", "mean"),
            final_coverage_95_mean=("final_coverage_95", "mean"),
            final_scan_time_s_mean=("final_scan_time_s", "mean"),
            final_dose_proxy_mean=("final_dose_proxy", "mean"),
        )
        .reset_index()
    )
    return summary.merge(
        runs.groupby("policy")["interface_error_cost_auc"]
        .quantile([0.025, 0.975])
        .unstack()
        .rename(columns={0.025: "interface_error_cost_auc_q025", 0.975: "interface_error_cost_auc_q975"})
        .reset_index(),
        on="policy",
    )
