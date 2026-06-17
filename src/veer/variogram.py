"""Calibrated model-averaged Matern-3/2 variogram estimation for replay."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import nnls
from sklearn.decomposition import PCA

from .domain import RunConfig
from .features import SubtileObservation, robust_scale_feature_tensor
from .features import anisotropic_matern_3_2_kernel


@dataclass(frozen=True)
class VariogramPosterior:
    kernel_labels: list[str]
    kernel_length_scales: list[tuple[float, float]]
    log_marginal_likelihoods: np.ndarray
    weights: np.ndarray
    untempered_weights: np.ndarray
    temper: float
    sill: float
    component_eigenvalues: np.ndarray
    effective_components: int
    subtile_count: int


def matern_3_2_correlation(distance: np.ndarray) -> np.ndarray:
    """Matern-3/2 correlation for distances already in length-scale units."""

    scaled = np.sqrt(3.0) * np.asarray(distance, dtype=float)
    return (1.0 + scaled) * np.exp(-scaled)


def normalized_training_points(points_nm: np.ndarray, config: RunConfig) -> np.ndarray:
    return np.column_stack(
        [
            points_nm[:, 0] / max(config.scenario.width_nm, 1.0),
            points_nm[:, 1] / max(config.scenario.height_nm, 1.0),
        ]
    )


def stable_cholesky(matrix: np.ndarray, jitter: float) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    identity = np.eye(symmetric.shape[0])
    for multiplier in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        try:
            return np.linalg.cholesky(symmetric + identity * jitter * multiplier)
        except np.linalg.LinAlgError:
            continue
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, jitter)
    repaired = (eigenvectors * clipped[None, :]) @ eigenvectors.T
    return np.linalg.cholesky(0.5 * (repaired + repaired.T) + identity * jitter)


def tempered_model_weights(log_marginal_likelihoods: np.ndarray, temper: float) -> np.ndarray:
    """Softmax model weights with likelihood tempering to resist weight collapse."""

    values = np.asarray(log_marginal_likelihoods, dtype=float) / max(temper, 1.0)
    shifted = values - np.max(values)
    weights = np.exp(np.clip(shifted, -745.0, 0.0))
    return weights / np.sum(weights)


def _summed_log_marginal_likelihood(
    correlation: np.ndarray,
    targets: np.ndarray,
    noise: np.ndarray,
    jitter: float,
) -> float:
    total = 0.0
    count = correlation.shape[0]
    for component in range(targets.shape[1]):
        covariance = correlation + np.diag(noise[:, component] + jitter)
        factor = stable_cholesky(covariance, jitter)
        solved = np.linalg.solve(factor, targets[:, component])
        total += float(
            -0.5 * solved @ solved
            - np.sum(np.log(np.diag(factor)))
            - 0.5 * count * np.log(2.0 * np.pi)
        )
    return total


def fit_variogram_posterior(
    observations: list[SubtileObservation],
    config: RunConfig,
) -> VariogramPosterior:
    """Fit the revealed-only model-averaged variogram on standardized latents."""

    if len(observations) < 2:
        raise ValueError("v5 variogram posterior requires at least two revealed subtiles")
    settings = config.variogram
    raw = np.stack([node.feature_values for node in observations])
    scaled, _, feature_iqr = robust_scale_feature_tensor(
        raw,
        epsilon=settings.robust_iqr_epsilon,
        clip=settings.scaled_feature_clip,
    )
    flattened = scaled.reshape(len(observations), -1)
    effective_components = min(
        settings.latent_components, flattened.shape[1], len(observations) - 1
    )
    if effective_components < 1:
        raise ValueError("v5 variogram posterior needs at least one latent component")
    embedding = PCA(n_components=effective_components, svd_solver="full")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
        )
        scores = embedding.fit_transform(flattened)
    eigenvalues = np.maximum(embedding.explained_variance_, settings.robust_iqr_epsilon)
    targets = scores / np.sqrt(eigenvalues)[None, :]

    mean_noise = np.stack([node.mean_noise for node in observations])
    scaled_noise = mean_noise / np.maximum(
        feature_iqr[:, 0][None, :] ** 2, settings.robust_iqr_epsilon
    )
    latent_noise = scaled_noise @ (embedding.components_**2).T
    standardized_noise = np.maximum(latent_noise / eigenvalues[None, :], settings.alpha_floor)

    points_nm = np.asarray(
        [[node.center_x_nm, node.center_y_nm] for node in observations], dtype=float
    )
    points = normalized_training_points(points_nm, config)
    labels = []
    likelihoods = []
    for length_x, length_y in settings.kernel_catalog:
        correlation = anisotropic_matern_3_2_kernel(points, points, (length_x, length_y))
        labels.append(f"{length_x:.6g},{length_y:.6g}")
        likelihoods.append(
            _summed_log_marginal_likelihood(
                correlation, targets, standardized_noise, settings.jitter
            )
        )
    likelihood_array = np.asarray(likelihoods, dtype=float)
    temper = max(1.0, len(observations) / settings.temper_reference_subtiles)
    return VariogramPosterior(
        kernel_labels=labels,
        kernel_length_scales=[tuple(pair) for pair in settings.kernel_catalog],
        log_marginal_likelihoods=likelihood_array,
        weights=tempered_model_weights(likelihood_array, temper),
        untempered_weights=tempered_model_weights(likelihood_array, 1.0),
        temper=temper,
        sill=float(np.sum(eigenvalues)),
        component_eigenvalues=eigenvalues,
        effective_components=effective_components,
        subtile_count=len(observations),
    )


@dataclass(frozen=True)
class NestedVariogramFit:
    nugget: float
    matern_amplitude: float
    linear_slope: float
    length_scale: float
    weighted_sse: float
    bin_distances: np.ndarray
    bin_semivariances: np.ndarray
    bin_pair_counts: np.ndarray
    subtile_count: int


def gamma_nested(distance: np.ndarray, fit: NestedVariogramFit) -> np.ndarray:
    """Evaluate the nested semivariogram (nugget excluded; it cancels in differences)."""

    distance = np.asarray(distance, dtype=float)
    bounded = fit.matern_amplitude * (
        1.0 - matern_3_2_correlation(distance / max(fit.length_scale, 1.0e-12))
    )
    return bounded + fit.linear_slope * distance


def _binned_semivariogram(
    observations: list[SubtileObservation],
    config: RunConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    settings = config.variogram
    raw = np.stack([node.feature_values for node in observations])
    scaled, _, _ = robust_scale_feature_tensor(
        raw,
        epsilon=settings.robust_iqr_epsilon,
        clip=settings.scaled_feature_clip,
    )
    z = scaled.reshape(len(observations), -1)
    points_nm = np.asarray(
        [[node.center_x_nm, node.center_y_nm] for node in observations], dtype=float
    )
    points = normalized_training_points(points_nm, config)
    first, second = np.triu_indices(len(observations), k=1)
    distances = np.linalg.norm(points[first] - points[second], axis=1)
    semivariances = 0.5 * np.mean((z[first] - z[second]) ** 2, axis=1)
    edges = [0.0, *settings.nested_bin_edges]
    bin_distances = []
    bin_values = []
    bin_counts = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (distances > low) & (distances <= high)
        count = int(np.sum(mask))
        if count < settings.nested_minimum_bin_pairs:
            continue
        bin_distances.append(float(np.mean(distances[mask])))
        bin_values.append(float(np.mean(semivariances[mask])))
        bin_counts.append(count)
    return (
        np.asarray(bin_distances, dtype=float),
        np.asarray(bin_values, dtype=float),
        np.asarray(bin_counts, dtype=float),
    )


def fit_nested_variogram(
    observations: list[SubtileObservation],
    config: RunConfig,
) -> NestedVariogramFit:
    """Fit gamma(d) = c0 + c1 * (1 - rho(d; l)) + c2 * d by Cressie-weighted NNLS.

    The unbounded linear component represents the intrinsic long-range growth
    observed in the empirical variograms of these maps; its expected-error
    reduction equals the deterministic coverage gain, so the baseline policy is
    the data-selectable special case c1 = 0.
    """

    if len(observations) < 2:
        raise ValueError("v5.1 nested variogram requires at least two revealed subtiles")
    settings = config.variogram
    bin_distances, bin_values, bin_counts = _binned_semivariogram(observations, config)
    if bin_distances.size == 0:
        raise ValueError("v5.1 nested variogram has no usable distance bins")
    weights = bin_counts / np.maximum(bin_values, settings.robust_iqr_epsilon) ** 2
    root = np.sqrt(weights)
    if bin_distances.size < 3:
        slope_numerator = float(np.sum(weights * bin_distances * bin_values))
        slope_denominator = max(float(np.sum(weights * bin_distances**2)), 1.0e-12)
        slope = max(slope_numerator / slope_denominator, 0.0)
        residual = root * (bin_values - slope * bin_distances)
        return NestedVariogramFit(
            nugget=0.0,
            matern_amplitude=0.0,
            linear_slope=slope,
            length_scale=float(settings.nested_length_scale_grid[0]),
            weighted_sse=float(residual @ residual),
            bin_distances=bin_distances,
            bin_semivariances=bin_values,
            bin_pair_counts=bin_counts,
            subtile_count=len(observations),
        )
    best: tuple[float, np.ndarray, float] | None = None
    target = root * bin_values
    for length_scale in settings.nested_length_scale_grid:
        design = np.column_stack(
            [
                np.ones_like(bin_distances),
                1.0 - matern_3_2_correlation(bin_distances / length_scale),
                bin_distances,
            ]
        )
        coefficients, residual_norm = nnls(root[:, None] * design, target)
        sse = float(residual_norm**2)
        if best is None or sse < best[2]:
            best = (length_scale, coefficients, sse)
    length_scale, coefficients, sse = best
    return NestedVariogramFit(
        nugget=float(coefficients[0]),
        matern_amplitude=float(coefficients[1]),
        linear_slope=float(coefficients[2]),
        length_scale=float(length_scale),
        weighted_sse=sse,
        bin_distances=bin_distances,
        bin_semivariances=bin_values,
        bin_pair_counts=bin_counts,
        subtile_count=len(observations),
    )
