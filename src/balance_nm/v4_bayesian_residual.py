"""Bayesian residual lookahead with ROI-summary and morphology-preserving subtile models."""

from __future__ import annotations

from dataclasses import dataclass
import json
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA

from .domain import RunConfig
from .v3_validation import _raster_cost
from .v4_bayesian import BayesianROIObservation, observation_from_revealed_roi
from .v4_validation import lookahead_coverage_gain


@dataclass(frozen=True)
class BayesianSubtileObservation:
    roi_id: str
    subtile_id: str
    center_x_nm: float
    center_y_nm: float
    acquisition_sequence: int
    channel_mean: np.ndarray
    residual_noise_proxy: np.ndarray
    pixel_count: int


@dataclass
class BayesianResidualPosterior:
    model_mode: str
    kernel_labels: list[str]
    posterior_weights: np.ndarray
    integrated_variance: float
    candidate_scores: pd.DataFrame
    effective_components: int


@dataclass
class _KernelPosterior:
    label: str
    log_marginal_likelihood: float
    covariances: np.ndarray
    candidate_noise: np.ndarray
    integrated_variance: float


def anisotropic_matern_3_2_kernel(
    first: np.ndarray,
    second: np.ndarray,
    length_scales: tuple[float, float],
) -> np.ndarray:
    """Unit-amplitude anisotropic Matern-3/2 covariance."""

    scales = np.maximum(np.asarray(length_scales, dtype=float), 1.0e-12)
    distance = np.linalg.norm(
        (first[:, None, :] - second[None, :, :]) / scales[None, None, :],
        axis=2,
    )
    scaled = np.sqrt(3.0) * distance
    return (1.0 + scaled) * np.exp(-scaled)


def normalized_kernel_weights(log_marginal_likelihoods: np.ndarray) -> np.ndarray:
    """Return stable posterior model weights for uniform prior kernel hypotheses."""

    values = np.asarray(log_marginal_likelihoods, dtype=float)
    if values.ndim != 1 or not values.size:
        raise ValueError("kernel model averaging requires at least one log likelihood")
    shifted = values - np.max(values)
    weights = np.exp(np.clip(shifted, -745.0, 0.0))
    return weights / np.sum(weights)


def _normalized_points(points_nm: np.ndarray, config: RunConfig) -> np.ndarray:
    return np.column_stack(
        [
            points_nm[:, 0] / max(config.scenario.width_nm, 1.0),
            points_nm[:, 1] / max(config.scenario.height_nm, 1.0),
        ]
    )


def _robust_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.median(values, axis=0)
    low, high = np.percentile(values, [5.0, 95.0], axis=0)
    span = np.maximum(high - low, 1.0)
    return (values - median[None, :]) / span[None, :], median, span


def _fit_kernel_posterior(
    label: str,
    training_points: np.ndarray,
    targets: np.ndarray,
    training_noise: np.ndarray,
    evaluation_points: np.ndarray,
    length_scales: tuple[float, float],
    jitter: float,
) -> _KernelPosterior:
    base_training = anisotropic_matern_3_2_kernel(
        training_points, training_points, length_scales
    )
    evaluation_training = anisotropic_matern_3_2_kernel(
        evaluation_points, training_points, length_scales
    )
    evaluation_covariance = anisotropic_matern_3_2_kernel(
        evaluation_points, evaluation_points, length_scales
    )
    covariances = []
    log_likelihood = 0.0
    for output in range(targets.shape[1]):
        covariance = base_training + np.diag(training_noise[:, output] + jitter)
        cholesky = np.linalg.cholesky(covariance)
        alpha = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, targets[:, output]))
        log_likelihood += float(
            -0.5 * targets[:, output] @ alpha
            - np.sum(np.log(np.diag(cholesky)))
            - 0.5 * len(training_points) * np.log(2.0 * np.pi)
        )
        projected = np.linalg.solve(cholesky, evaluation_training.T)
        posterior = evaluation_covariance - projected.T @ projected
        covariances.append(0.5 * (posterior + posterior.T))
    stacked = np.stack(covariances)
    variance = np.maximum(np.diagonal(stacked, axis1=1, axis2=2), 0.0)
    return _KernelPosterior(
        label=label,
        log_marginal_likelihood=log_likelihood,
        covariances=stacked,
        candidate_noise=np.maximum(np.median(training_noise, axis=0), 1.0e-12),
        integrated_variance=float(np.mean(variance)),
    )


def batch_expected_variance_reduction(
    covariances: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_noise: np.ndarray,
) -> float:
    """Compute mean integrated variance reduction from a prospective observation batch."""

    reductions = []
    for output, covariance in enumerate(covariances):
        cross = covariance[:, candidate_indices]
        observed = covariance[np.ix_(candidate_indices, candidate_indices)]
        observed = observed + np.eye(len(candidate_indices)) * candidate_noise[output]
        solved = np.linalg.solve(observed, cross.T).T
        reductions.append(np.sum(cross * solved, axis=1))
    return float(np.mean(np.maximum(np.stack(reductions), 0.0)))


def direct_batch_expected_variance_reduction(
    covariances: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_noise: np.ndarray,
) -> float:
    """Compute the same batch reduction by explicitly forming conditioned covariances."""

    before = float(
        np.mean(np.maximum(np.diagonal(covariances, axis1=1, axis2=2), 0.0))
    )
    conditioned = []
    for output, covariance in enumerate(covariances):
        cross = covariance[:, candidate_indices]
        observed = covariance[np.ix_(candidate_indices, candidate_indices)]
        observed = observed + np.eye(len(candidate_indices)) * candidate_noise[output]
        conditioned.append(covariance - cross @ np.linalg.solve(observed, cross.T))
    after = float(
        np.mean(
            np.maximum(np.diagonal(np.stack(conditioned), axis1=1, axis2=2), 0.0)
        )
    )
    return before - after


def residual_rank_candidates(
    scores: pd.DataFrame,
    config: RunConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    """Apply a capped Bayesian bonus only within the near-optimal geometry shortlist."""

    settings = config.acquisition_v4.bayesian_residual
    ranked = scores.copy()
    maximum_geometry = max(float(ranked["geometry_coverage_gain"].max()), 1.0e-12)
    ranked["normalized_geometry_gain"] = ranked["geometry_coverage_gain"] / maximum_geometry
    ranked["geometry_shortlist_eligible"] = (
        ranked["normalized_geometry_gain"] >= settings.geometry_shortlist_ratio
    )
    eligible = ranked["geometry_shortlist_eligible"]
    bayesian = ranked["model_averaged_fractional_EIVR"] / np.maximum(
        ranked["estimated_raster_cost_s"], 1.0e-12
    )
    center = float(np.median(bayesian[eligible]))
    upper = float(np.max(bayesian[eligible]))
    denominator = upper - center
    if denominator <= settings.tie_tolerance:
        residual = np.zeros(len(ranked), dtype=float)
    else:
        residual = np.clip((bayesian - center) / denominator, 0.0, 1.0)
    ranked["residual_center"] = center
    ranked["residual_bonus_fraction"] = np.where(
        eligible, settings.maximum_bonus_fraction * residual, 0.0
    )
    ranked["selection_utility"] = np.where(
        eligible,
        ranked["normalized_geometry_gain"] * (1.0 + ranked["residual_bonus_fraction"]),
        -np.inf,
    )
    maximum_utility = float(ranked["selection_utility"].max())
    tied = ranked[
        ranked["selection_utility"] >= maximum_utility - settings.tie_tolerance
    ]
    selected = tied.sort_values(
        ["geometry_coverage_gain", "row0", "column0"],
        ascending=[False, True, True],
    ).iloc[0]
    ranked["selected"] = ranked["roi_id"] == str(selected["roi_id"])
    return selected, ranked


def _score_kernel_candidates(
    kernels: list[_KernelPosterior],
    weights: np.ndarray,
    catalog: pd.DataFrame,
    evaluation_groups: dict[str, np.ndarray],
    distance_pixels: np.ndarray,
    config: RunConfig,
) -> pd.DataFrame:
    rows = []
    weighted_integrated = float(
        np.sum([weight * kernel.integrated_variance for weight, kernel in zip(weights, kernels)])
    )
    for _, roi in catalog.iterrows():
        reductions = []
        fractional = []
        indices = evaluation_groups[str(roi["roi_id"])]
        for kernel in kernels:
            reduction = batch_expected_variance_reduction(
                kernel.covariances, indices, kernel.candidate_noise
            )
            reductions.append(reduction)
            fractional.append(reduction / max(kernel.integrated_variance, 1.0e-12))
        cost, _ = _raster_cost(config, roi)
        rows.append(
            {
                "roi_id": str(roi["roi_id"]),
                "geometry_coverage_gain": lookahead_coverage_gain(distance_pixels, roi),
                "estimated_raster_cost_s": cost,
                "fractional_EIVR_by_kernel": json.dumps(dict(zip([k.label for k in kernels], fractional))),
                "model_averaged_fractional_EIVR": float(np.dot(weights, fractional)),
                "integrated_current_posterior_variance": weighted_integrated,
            }
        )
    return pd.DataFrame(rows)


def fit_roi_summary_ensemble(
    nodes: list[BayesianROIObservation],
    catalog: pd.DataFrame,
    config: RunConfig,
    distance_pixels: np.ndarray,
) -> BayesianResidualPosterior:
    """Fit model-averaged ROI-summary GPs and score every raster candidate."""

    if not nodes:
        raise ValueError("ROI-summary ensemble requires revealed nodes")
    settings = config.acquisition_v4.bayesian_residual.roi_summary
    raw = np.stack([node.channel_mean for node in nodes])
    targets, _, spans = _robust_scale(raw)
    within = np.stack([node.channel_variance for node in nodes])
    pixels = np.asarray([node.pixel_count for node in nodes], dtype=float)[:, None]
    training_noise = np.maximum(
        within / np.maximum(pixels, 1.0) / spans[None, :] ** 2,
        settings.alpha_floor,
    )
    catalog_nm = catalog[["center_x_nm", "center_y_nm"]].to_numpy(float)
    training_nm = np.asarray([[node.center_x_nm, node.center_y_nm] for node in nodes])
    training_points = _normalized_points(training_nm, config)
    evaluation_points = _normalized_points(catalog_nm, config)
    kernels = [
        _fit_kernel_posterior(
            f"{length_scale:.6g}",
            training_points,
            targets,
            training_noise,
            evaluation_points,
            (length_scale, length_scale),
            settings.jitter,
        )
        for length_scale in settings.length_scale_catalog
    ]
    weights = normalized_kernel_weights(
        np.asarray([kernel.log_marginal_likelihood for kernel in kernels])
    )
    groups = {
        str(roi_id): np.asarray([index], dtype=int)
        for index, roi_id in enumerate(catalog["roi_id"].astype(str))
    }
    scores = _score_kernel_candidates(kernels, weights, catalog, groups, distance_pixels, config)
    scores["revealed"] = scores["roi_id"].isin({node.roi_id for node in nodes})
    return BayesianResidualPosterior(
        model_mode="roi_summary",
        kernel_labels=[kernel.label for kernel in kernels],
        posterior_weights=weights,
        integrated_variance=float(
            np.dot(weights, [kernel.integrated_variance for kernel in kernels])
        ),
        candidate_scores=scores,
        effective_components=raw.shape[1],
    )


def subtile_geometry_for_roi(
    roi: pd.Series,
    x: np.ndarray,
    y: np.ndarray,
    grid_shape: tuple[int, int],
) -> pd.DataFrame:
    """Split one raster ROI into deterministic local subtile geometries."""

    row0, row1 = int(roi["row0"]), int(roi["row1"])
    column0, column1 = int(roi["column0"]), int(roi["column1"])
    if row1 - row0 < grid_shape[0] or column1 - column0 < grid_shape[1]:
        raise ValueError("subtile grid exceeds ROI pixel dimensions")
    row_bounds = np.linspace(row0, row1, grid_shape[0] + 1, dtype=int)
    column_bounds = np.linspace(column0, column1, grid_shape[1] + 1, dtype=int)
    rows = []
    for row_index, (subrow0, subrow1) in enumerate(zip(row_bounds[:-1], row_bounds[1:])):
        for column_index, (subcolumn0, subcolumn1) in enumerate(
            zip(column_bounds[:-1], column_bounds[1:])
        ):
            rows.append(
                {
                    "roi_id": str(roi["roi_id"]),
                    "subtile_id": f"{roi['roi_id']}__s{row_index:02d}_c{column_index:02d}",
                    "row0": int(subrow0),
                    "row1": int(subrow1),
                    "column0": int(subcolumn0),
                    "column1": int(subcolumn1),
                    "center_x_nm": float((x[subcolumn0] + x[subcolumn1 - 1]) / 2.0),
                    "center_y_nm": float((y[subrow0] + y[subrow1 - 1]) / 2.0),
                    "pixel_count": int((subrow1 - subrow0) * (subcolumn1 - subcolumn0)),
                }
            )
    return pd.DataFrame(rows)


def build_subtile_catalog(
    catalog: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    grid_shape: tuple[int, int],
) -> pd.DataFrame:
    frames = [subtile_geometry_for_roi(roi, x, y, grid_shape) for _, roi in catalog.iterrows()]
    return pd.concat(frames, ignore_index=True)


def local_residual_mad_noise(values: np.ndarray, sigma_px: float) -> np.ndarray:
    """Estimate mean-observation noise without treating broad structural contrast as noise."""

    smoothed = gaussian_filter(values.astype(float), sigma=(0.0, sigma_px, sigma_px))
    residual = values - smoothed
    centered = residual - np.median(residual, axis=(1, 2), keepdims=True)
    mad = np.median(np.abs(centered), axis=(1, 2))
    residual_variance = (1.4826 * mad) ** 2
    pixels = max(values.shape[1] * values.shape[2], 1)
    return residual_variance / pixels


def subtile_observations_from_revealed_roi(
    roi: pd.Series,
    dense_signal: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    acquisition_sequence: int,
    config: RunConfig,
) -> list[BayesianSubtileObservation]:
    settings = config.acquisition_v4.bayesian_residual.subtile
    geometry = subtile_geometry_for_roi(roi, x, y, settings.grid_shape)
    observations = []
    for _, subtile in geometry.iterrows():
        values = dense_signal[
            :,
            int(subtile["row0"]) : int(subtile["row1"]),
            int(subtile["column0"]) : int(subtile["column1"]),
        ]
        observations.append(
            BayesianSubtileObservation(
                roi_id=str(roi["roi_id"]),
                subtile_id=str(subtile["subtile_id"]),
                center_x_nm=float(subtile["center_x_nm"]),
                center_y_nm=float(subtile["center_y_nm"]),
                acquisition_sequence=acquisition_sequence,
                channel_mean=np.mean(values, axis=(1, 2)),
                residual_noise_proxy=local_residual_mad_noise(
                    values, settings.residual_filter_sigma_px
                ),
                pixel_count=int(subtile["pixel_count"]),
            )
        )
    return observations


def fit_subtile_ensemble(
    nodes: list[BayesianSubtileObservation],
    catalog: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    config: RunConfig,
    distance_pixels: np.ndarray,
) -> BayesianResidualPosterior:
    """Fit model-averaged latent subtile GPs and score every raster candidate by batch EIVR."""

    if len(nodes) < 2:
        raise ValueError("subtile ensemble requires at least two revealed subtiles")
    settings = config.acquisition_v4.bayesian_residual.subtile
    subtile_catalog = build_subtile_catalog(catalog, x, y, settings.grid_shape)
    raw = np.stack([node.channel_mean for node in nodes])
    scaled, _, spans = _robust_scale(raw)
    effective_components = min(settings.latent_components, raw.shape[1], len(nodes) - 1)
    embedding = PCA(n_components=effective_components, svd_solver="full")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
        )
        targets = embedding.fit_transform(scaled)
    channel_noise = np.stack([node.residual_noise_proxy for node in nodes])
    scaled_noise = channel_noise / spans[None, :] ** 2
    training_noise = np.maximum(
        scaled_noise @ (embedding.components_ ** 2).T,
        settings.alpha_floor,
    )
    training_nm = np.asarray([[node.center_x_nm, node.center_y_nm] for node in nodes])
    evaluation_nm = subtile_catalog[["center_x_nm", "center_y_nm"]].to_numpy(float)
    training_points = _normalized_points(training_nm, config)
    evaluation_points = _normalized_points(evaluation_nm, config)
    kernels = [
        _fit_kernel_posterior(
            f"{length_x:.6g},{length_y:.6g}",
            training_points,
            targets,
            training_noise,
            evaluation_points,
            (length_x, length_y),
            settings.jitter,
        )
        for length_x, length_y in settings.kernel_catalog
    ]
    weights = normalized_kernel_weights(
        np.asarray([kernel.log_marginal_likelihood for kernel in kernels])
    )
    groups = {
        roi_id: frame.index.to_numpy(int)
        for roi_id, frame in subtile_catalog.groupby("roi_id", sort=False)
    }
    scores = _score_kernel_candidates(kernels, weights, catalog, groups, distance_pixels, config)
    scores["revealed"] = scores["roi_id"].isin({node.roi_id for node in nodes})
    return BayesianResidualPosterior(
        model_mode="subtile",
        kernel_labels=[kernel.label for kernel in kernels],
        posterior_weights=weights,
        integrated_variance=float(
            np.dot(weights, [kernel.integrated_variance for kernel in kernels])
        ),
        candidate_scores=scores,
        effective_components=effective_components,
    )


def roi_summary_observation_from_reveal(
    roi: pd.Series, dense_signal: np.ndarray, acquisition_sequence: int
) -> BayesianROIObservation:
    return observation_from_revealed_roi(roi, dense_signal, acquisition_sequence)
