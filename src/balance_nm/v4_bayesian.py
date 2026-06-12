"""ROI-summary Gaussian-process posterior and Bayesian lookahead utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from .domain import RunConfig
from .v3_validation import _raster_cost
from .v4_validation import lookahead_coverage_gain


@dataclass(frozen=True)
class BayesianROIObservation:
    roi_id: str
    center_x_nm: float
    center_y_nm: float
    acquisition_sequence: int
    channel_mean: np.ndarray
    channel_variance: np.ndarray
    pixel_count: int


@dataclass
class BayesianPosterior:
    length_scale_fraction: float
    catalog_roi_ids: list[str]
    catalog_points: np.ndarray
    channel_mean: np.ndarray
    channel_variance: np.ndarray
    channel_covariance: np.ndarray
    channel_noise: np.ndarray
    integrated_variance: float
    candidate_scores: pd.DataFrame


def observation_from_revealed_roi(
    roi: pd.Series,
    dense_signal: np.ndarray,
    acquisition_sequence: int,
) -> BayesianROIObservation:
    """Summarize one raster tile using only pixels that have just been revealed."""

    row0, row1 = int(roi["row0"]), int(roi["row1"])
    column0, column1 = int(roi["column0"]), int(roi["column1"])
    values = dense_signal[:, row0:row1, column0:column1]
    return BayesianROIObservation(
        roi_id=str(roi["roi_id"]),
        center_x_nm=float(roi["center_x_nm"]),
        center_y_nm=float(roi["center_y_nm"]),
        acquisition_sequence=acquisition_sequence,
        channel_mean=np.mean(values, axis=(1, 2)),
        channel_variance=np.var(values, axis=(1, 2)),
        pixel_count=int(roi["pixel_count"]),
    )


def _normalized_points(catalog: pd.DataFrame, config: RunConfig) -> np.ndarray:
    return np.column_stack(
        [
            catalog["center_x_nm"].to_numpy(float) / max(config.scenario.width_nm, 1.0),
            catalog["center_y_nm"].to_numpy(float) / max(config.scenario.height_nm, 1.0),
        ]
    )


def matern_3_2_kernel(first: np.ndarray, second: np.ndarray, length_scale: float) -> np.ndarray:
    """Unit-amplitude Matérn-3/2 covariance."""

    distance = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    scaled = np.sqrt(3.0) * distance / max(length_scale, 1.0e-12)
    return (1.0 + scaled) * np.exp(-scaled)


def _revealed_channel_scale(nodes: list[BayesianROIObservation]) -> tuple[np.ndarray, np.ndarray]:
    means = np.stack([node.channel_mean for node in nodes])
    median = np.median(means, axis=0)
    low, high = np.percentile(means, [5.0, 95.0], axis=0)
    return median, np.maximum(high - low, 1.0)


def _channel_training_noise(
    nodes: list[BayesianROIObservation],
    spans: np.ndarray,
    alpha_floor: float,
) -> np.ndarray:
    within = np.stack([node.channel_variance for node in nodes])
    pixels = np.asarray([node.pixel_count for node in nodes], dtype=float)[:, None]
    return np.maximum(within / np.maximum(pixels, 1.0) / spans[None, :] ** 2, alpha_floor)


def _posterior_channel(
    training_points: np.ndarray,
    normalized_values: np.ndarray,
    training_noise: np.ndarray,
    catalog_points: np.ndarray,
    length_scale: float,
    jitter: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_covariance = matern_3_2_kernel(training_points, training_points, length_scale)
    train_covariance = train_covariance + np.diag(training_noise + jitter)
    catalog_train = matern_3_2_kernel(catalog_points, training_points, length_scale)
    inverse = np.linalg.inv(train_covariance)
    posterior_mean = catalog_train @ inverse @ normalized_values
    catalog_covariance = matern_3_2_kernel(catalog_points, catalog_points, length_scale)
    posterior_covariance = catalog_covariance - catalog_train @ inverse @ catalog_train.T
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
    posterior_variance = np.maximum(np.diag(posterior_covariance), 0.0)
    return posterior_mean, posterior_variance, posterior_covariance


def fit_bayesian_roi_posterior(
    nodes: list[BayesianROIObservation],
    catalog: pd.DataFrame,
    config: RunConfig,
    length_scale_fraction: float,
    distance_pixels: np.ndarray,
) -> BayesianPosterior:
    """Fit independent channel posteriors and score every catalog ROI by EIVR."""

    if not nodes:
        raise ValueError("Bayesian ROI posterior requires at least one revealed ROI")
    settings = config.acquisition_v4.bayesian
    catalog_points = _normalized_points(catalog, config)
    training_points = np.asarray(
        [
            [
                node.center_x_nm / max(config.scenario.width_nm, 1.0),
                node.center_y_nm / max(config.scenario.height_nm, 1.0),
            ]
            for node in nodes
        ]
    )
    medians, spans = _revealed_channel_scale(nodes)
    values = np.stack([node.channel_mean for node in nodes])
    normalized_values = (values - medians[None, :]) / spans[None, :]
    training_noise = _channel_training_noise(nodes, spans, settings.alpha_floor)
    posterior_means = []
    posterior_variances = []
    posterior_covariances = []
    candidate_noise = np.median(training_noise, axis=0)
    for channel in range(values.shape[1]):
        mean, variance, covariance = _posterior_channel(
            training_points,
            normalized_values[:, channel],
            training_noise[:, channel],
            catalog_points,
            length_scale_fraction,
            settings.jitter,
        )
        posterior_means.append(mean)
        posterior_variances.append(variance)
        posterior_covariances.append(covariance)
    means = np.stack(posterior_means)
    variances = np.stack(posterior_variances)
    integrated_variance = float(np.mean(variances))
    revealed = {node.roi_id for node in nodes}
    rows = []
    for candidate_index, (_, roi) in enumerate(catalog.iterrows()):
        reductions = []
        for channel, covariance in enumerate(posterior_covariances):
            reduction = covariance[:, candidate_index] ** 2 / max(
                float(covariance[candidate_index, candidate_index] + candidate_noise[channel]),
                1.0e-12,
            )
            reductions.append(np.maximum(reduction, 0.0))
        expected_reduction = float(np.mean(np.stack(reductions)))
        fractional = expected_reduction / max(integrated_variance, 1.0e-12)
        cost, _ = _raster_cost(config, roi)
        rows.append(
            {
                "roi_id": str(roi["roi_id"]),
                "geometry_coverage_gain": lookahead_coverage_gain(distance_pixels, roi),
                "integrated_current_posterior_variance": integrated_variance,
                "expected_variance_reduction": expected_reduction,
                "fractional_expected_variance_reduction": fractional,
                "estimated_candidate_noise_by_channel": json.dumps(candidate_noise.tolist()),
                "bayesian_utility": fractional / max(cost, 1.0e-12),
                "revealed": str(roi["roi_id"]) in revealed,
            }
        )
    return BayesianPosterior(
        length_scale_fraction=length_scale_fraction,
        catalog_roi_ids=catalog["roi_id"].astype(str).tolist(),
        catalog_points=catalog_points,
        channel_mean=means,
        channel_variance=variances,
        channel_covariance=np.stack(posterior_covariances),
        channel_noise=candidate_noise,
        integrated_variance=integrated_variance,
        candidate_scores=pd.DataFrame(rows),
    )


def direct_expected_variance_reduction(
    posterior: BayesianPosterior,
    candidate_index: int,
) -> float:
    """Recompute integrated variance reduction by direct Gaussian conditioning."""

    conditioned = []
    for channel, covariance in enumerate(posterior.channel_covariance):
        column = covariance[:, candidate_index]
        denominator = max(
            float(covariance[candidate_index, candidate_index] + posterior.channel_noise[channel]),
            1.0e-12,
        )
        conditioned.append(np.diag(covariance - np.outer(column, column) / denominator))
    conditioned_variance = float(np.mean(np.maximum(np.stack(conditioned), 0.0)))
    return posterior.integrated_variance - conditioned_variance
