"""Evidence-gated Bayesian subtile lookahead and posterior-fantasy morphology scoring."""

from __future__ import annotations

from dataclasses import dataclass
import json
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from sklearn.decomposition import PCA

from .domain import RunConfig
from .v3_morphology import front_from_probability, morphology_products_from_signal
from .v3_validation import _raster_cost
from .v4_bayesian_residual import (
    BayesianSubtileObservation,
    anisotropic_matern_3_2_kernel,
    batch_expected_variance_reduction,
    build_subtile_catalog,
    normalized_kernel_weights,
)
from .v4_validation import lookahead_coverage_gain


@dataclass
class MorphologyKernelPosterior:
    label: str
    log_marginal_likelihood: float
    subtile_mean: np.ndarray
    subtile_covariance: np.ndarray
    coarse_mean: np.ndarray
    coarse_covariance: np.ndarray
    coarse_subtile_cross_covariance: np.ndarray
    candidate_noise: np.ndarray
    integrated_variance: float


@dataclass
class BayesianMorphologyPosterior:
    kernel_labels: list[str]
    posterior_weights: np.ndarray
    kernels: list[MorphologyKernelPosterior]
    candidate_scores: pd.DataFrame
    subtile_catalog: pd.DataFrame
    evaluation_groups: dict[str, np.ndarray]
    effective_components: int
    scaling_median: np.ndarray
    scaling_span: np.ndarray
    pca_components: np.ndarray
    pca_mean: np.ndarray
    coarse_x: np.ndarray
    coarse_y: np.ndarray
    integrated_variance: float


@dataclass(frozen=True)
class MorphologyTaskSummary:
    task_uncertainty: float
    front_uncertainty: float
    penetration_uncertainty: float
    reconstruction_uncertainty: float
    front_presence_probability: float
    mean_front_support_fraction: float
    state_assignment_confidence: float


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


def _stable_cholesky(matrix: np.ndarray, jitter: float) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    identity = np.eye(symmetric.shape[0])
    for multiplier in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        try:
            return np.linalg.cholesky(symmetric + identity * jitter * multiplier)
        except np.linalg.LinAlgError:
            continue
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, jitter)
    return np.linalg.cholesky((eigenvectors * clipped[None, :]) @ eigenvectors.T)


def _fit_kernel(
    label: str,
    training_points: np.ndarray,
    targets: np.ndarray,
    training_noise: np.ndarray,
    subtile_points: np.ndarray,
    coarse_points: np.ndarray,
    length_scales: tuple[float, float],
    jitter: float,
) -> MorphologyKernelPosterior:
    training_covariance = anisotropic_matern_3_2_kernel(
        training_points, training_points, length_scales
    )
    subtile_training = anisotropic_matern_3_2_kernel(
        subtile_points, training_points, length_scales
    )
    coarse_training = anisotropic_matern_3_2_kernel(
        coarse_points, training_points, length_scales
    )
    subtile_covariance = anisotropic_matern_3_2_kernel(
        subtile_points, subtile_points, length_scales
    )
    coarse_covariance = anisotropic_matern_3_2_kernel(
        coarse_points, coarse_points, length_scales
    )
    coarse_subtile = anisotropic_matern_3_2_kernel(
        coarse_points, subtile_points, length_scales
    )
    subtile_means = []
    subtile_covariances = []
    coarse_means = []
    coarse_covariances = []
    coarse_subtile_cross = []
    log_likelihood = 0.0
    for component in range(targets.shape[1]):
        covariance = training_covariance + np.diag(training_noise[:, component] + jitter)
        cholesky = _stable_cholesky(covariance, jitter)
        alpha = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, targets[:, component]))
        log_likelihood += float(
            -0.5 * targets[:, component] @ alpha
            - np.sum(np.log(np.diag(cholesky)))
            - 0.5 * len(training_points) * np.log(2.0 * np.pi)
        )
        projected_subtile = np.linalg.solve(cholesky, subtile_training.T)
        projected_coarse = np.linalg.solve(cholesky, coarse_training.T)
        subtile_means.append(subtile_training @ alpha)
        coarse_means.append(coarse_training @ alpha)
        subtile_posterior = subtile_covariance - projected_subtile.T @ projected_subtile
        coarse_posterior = coarse_covariance - projected_coarse.T @ projected_coarse
        subtile_covariances.append(0.5 * (subtile_posterior + subtile_posterior.T))
        coarse_covariances.append(0.5 * (coarse_posterior + coarse_posterior.T))
        coarse_subtile_cross.append(coarse_subtile - projected_coarse.T @ projected_subtile)
    stacked_subtile_covariance = np.stack(subtile_covariances)
    integrated_variance = float(
        np.mean(
            np.maximum(
                np.diagonal(stacked_subtile_covariance, axis1=1, axis2=2),
                0.0,
            )
        )
    )
    return MorphologyKernelPosterior(
        label=label,
        log_marginal_likelihood=log_likelihood,
        subtile_mean=np.stack(subtile_means),
        subtile_covariance=stacked_subtile_covariance,
        coarse_mean=np.stack(coarse_means),
        coarse_covariance=np.stack(coarse_covariances),
        coarse_subtile_cross_covariance=np.stack(coarse_subtile_cross),
        candidate_noise=np.maximum(np.median(training_noise, axis=0), 1.0e-12),
        integrated_variance=integrated_variance,
    )


def fit_bayesian_morphology_posterior(
    nodes: list[BayesianSubtileObservation],
    catalog: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    config: RunConfig,
    distance_pixels: np.ndarray,
) -> BayesianMorphologyPosterior:
    """Fit the revealed-only latent subtile ensemble used by both v4.3 policies."""

    if len(nodes) < 2:
        raise ValueError("v4.3 morphology posterior requires at least two revealed subtiles")
    settings = config.acquisition_v4.bayesian_residual.subtile
    morphology_settings = config.acquisition_v4.bayesian_morphology.fantasy
    subtile_catalog = build_subtile_catalog(catalog, x, y, settings.grid_shape)
    raw = np.stack([node.channel_mean for node in nodes])
    scaled, scaling_median, scaling_span = _robust_scale(raw)
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
    scaled_noise = channel_noise / scaling_span[None, :] ** 2
    training_noise = np.maximum(
        scaled_noise @ (embedding.components_ ** 2).T,
        settings.alpha_floor,
    )
    training_nm = np.asarray([[node.center_x_nm, node.center_y_nm] for node in nodes])
    subtile_nm = subtile_catalog[["center_x_nm", "center_y_nm"]].to_numpy(float)
    coarse_x = np.linspace(float(x[0]), float(x[-1]), morphology_settings.morphology_grid_shape[1])
    coarse_y = np.linspace(float(y[0]), float(y[-1]), morphology_settings.morphology_grid_shape[0])
    coarse_x_mesh, coarse_y_mesh = np.meshgrid(coarse_x, coarse_y)
    coarse_nm = np.column_stack([coarse_x_mesh.ravel(), coarse_y_mesh.ravel()])
    training_points = _normalized_points(training_nm, config)
    subtile_points = _normalized_points(subtile_nm, config)
    coarse_points = _normalized_points(coarse_nm, config)
    kernels = [
        _fit_kernel(
            f"{length_x:.6g},{length_y:.6g}",
            training_points,
            targets,
            training_noise,
            subtile_points,
            coarse_points,
            (length_x, length_y),
            settings.jitter,
        )
        for length_x, length_y in settings.kernel_catalog
    ]
    weights = normalized_kernel_weights(
        np.asarray([kernel.log_marginal_likelihood for kernel in kernels])
    )
    groups = {
        str(roi_id): frame.index.to_numpy(int)
        for roi_id, frame in subtile_catalog.groupby("roi_id", sort=False)
    }
    rows = []
    weighted_integrated = float(
        np.dot(weights, [kernel.integrated_variance for kernel in kernels])
    )
    for _, roi in catalog.iterrows():
        indices = groups[str(roi["roi_id"])]
        fractional = []
        for kernel in kernels:
            reduction = batch_expected_variance_reduction(
                kernel.subtile_covariance,
                indices,
                kernel.candidate_noise,
            )
            fractional.append(reduction / max(kernel.integrated_variance, 1.0e-12))
        cost, _ = _raster_cost(config, roi)
        rows.append(
            {
                "roi_id": str(roi["roi_id"]),
                "geometry_coverage_gain": lookahead_coverage_gain(distance_pixels, roi),
                "estimated_raster_cost_s": cost,
                "EIVR_by_kernel": json.dumps(dict(zip([kernel.label for kernel in kernels], fractional))),
                "model_averaged_fractional_EIVR": float(np.dot(weights, fractional)),
                "integrated_current_posterior_variance": weighted_integrated,
            }
        )
    candidate_scores = pd.DataFrame(rows)
    candidate_scores["revealed"] = candidate_scores["roi_id"].isin({node.roi_id for node in nodes})
    return BayesianMorphologyPosterior(
        kernel_labels=[kernel.label for kernel in kernels],
        posterior_weights=weights,
        kernels=kernels,
        candidate_scores=candidate_scores,
        subtile_catalog=subtile_catalog,
        evaluation_groups=groups,
        effective_components=effective_components,
        scaling_median=scaling_median,
        scaling_span=scaling_span,
        pca_components=embedding.components_,
        pca_mean=embedding.mean_,
        coarse_x=coarse_x,
        coarse_y=coarse_y,
        integrated_variance=weighted_integrated,
    )


def _kernel_values(row: pd.Series, labels: list[str]) -> np.ndarray:
    values = row["EIVR_by_kernel"]
    if isinstance(values, str):
        values = json.loads(values)
    return np.asarray([float(values[label]) for label in labels])


def evidence_gated_rank_candidates(
    scores: pd.DataFrame,
    posterior: BayesianMorphologyPosterior,
    config: RunConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    """Permit an EIVR override only when its absolute evidence is credible."""

    settings = config.acquisition_v4.bayesian_morphology.evidence_gate
    ranked = scores.copy()
    maximum_geometry = max(float(ranked["geometry_coverage_gain"].max()), 1.0e-12)
    ranked["normalized_geometry_gain"] = ranked["geometry_coverage_gain"] / maximum_geometry
    geometry_best = ranked.sort_values(
        ["geometry_coverage_gain", "row0", "column0"],
        ascending=[False, True, True],
    ).iloc[0]
    best_values = _kernel_values(geometry_best, posterior.kernel_labels)
    best_eivr = max(float(np.dot(posterior.posterior_weights, best_values)), 1.0e-12)
    diagnostics = []
    for _, row in ranked.iterrows():
        deltas = _kernel_values(row, posterior.kernel_labels) - best_values
        mean_delta = float(np.dot(posterior.posterior_weights, deltas))
        standard_deviation = float(
            np.sqrt(np.dot(posterior.posterior_weights, (deltas - mean_delta) ** 2))
        )
        lower_bound = mean_delta - settings.lcb_standard_deviations * standard_deviation
        relative = lower_bound / best_eivr
        support = float(np.dot(posterior.posterior_weights, deltas > 0.0))
        passed = bool(
            float(row["geometry_coverage_gain"]) / maximum_geometry
            >= settings.geometry_shortlist_ratio
            and lower_bound > 0.0
            and relative >= settings.minimum_relative_eivr_lcb
            and support >= settings.minimum_kernel_support
        )
        diagnostics.append((mean_delta, standard_deviation, lower_bound, relative, support, passed))
    values = np.asarray(diagnostics, dtype=object)
    ranked["EIVR_delta_mean"] = values[:, 0].astype(float)
    ranked["EIVR_delta_std"] = values[:, 1].astype(float)
    ranked["EIVR_LCB"] = values[:, 2].astype(float)
    ranked["relative_EIVR_LCB"] = values[:, 3].astype(float)
    ranked["kernel_support"] = values[:, 4].astype(float)
    ranked["evidence_gate_passed"] = values[:, 5].astype(bool)
    ranked["evidence_bonus_fraction"] = np.where(
        ranked["evidence_gate_passed"],
        np.minimum(settings.maximum_bonus_fraction, ranked["relative_EIVR_LCB"]),
        0.0,
    )
    ranked["selection_utility"] = ranked["normalized_geometry_gain"] * (
        1.0 + ranked["evidence_bonus_fraction"]
    )
    maximum_utility = float(ranked["selection_utility"].max())
    tied = ranked[ranked["selection_utility"] >= maximum_utility - settings.tie_tolerance]
    selected = tied.sort_values(
        ["geometry_coverage_gain", "row0", "column0"],
        ascending=[False, True, True],
    ).iloc[0]
    ranked["selected"] = ranked["roi_id"] == str(selected["roi_id"])
    ranked["fallback_reason"] = np.where(
        ranked["evidence_gate_passed"], "", "evidence_gate_not_satisfied"
    )
    return selected, ranked


def _weighted_moments(
    posterior: BayesianMorphologyPosterior,
    attribute_mean: str,
    attribute_covariance: str,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.stack([getattr(kernel, attribute_mean) for kernel in posterior.kernels])
    covariances = np.stack([getattr(kernel, attribute_covariance) for kernel in posterior.kernels])
    weighted_mean = np.tensordot(posterior.posterior_weights, means, axes=(0, 0))
    covariance = np.tensordot(posterior.posterior_weights, covariances, axes=(0, 0))
    for kernel_weight, kernel_mean in zip(posterior.posterior_weights, means):
        delta = kernel_mean - weighted_mean
        covariance += kernel_weight * np.einsum("ci,cj->cij", delta, delta)
    return weighted_mean, covariance


def decode_latent_fields(
    latent_samples: np.ndarray,
    posterior: BayesianMorphologyPosterior,
) -> np.ndarray:
    """Decode sample, component, point latent fields into coarse channel images."""

    samples, _, points = latent_samples.shape
    rows, columns = len(posterior.coarse_y), len(posterior.coarse_x)
    if points != rows * columns:
        raise ValueError("latent sample point count does not match the configured morphology grid")
    decoded = []
    for latent in latent_samples:
        scaled = latent.T @ posterior.pca_components + posterior.pca_mean[None, :]
        raw = scaled * posterior.scaling_span[None, :] + posterior.scaling_median[None, :]
        decoded.append(raw.T.reshape(raw.shape[1], rows, columns))
    return np.stack(decoded)


def _draw_latent_samples(
    mean: np.ndarray,
    covariance: np.ndarray,
    standard_normal: np.ndarray,
    jitter: float,
) -> np.ndarray:
    draws = []
    for component in range(mean.shape[0]):
        cholesky = _stable_cholesky(covariance[component], jitter)
        draws.append(mean[component][None, :] + standard_normal[:, component, :] @ cholesky.T)
    return np.stack(draws, axis=1)


def _binary_entropy(probability: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(probability, 1.0e-6, 1.0 - 1.0e-6)
    return -(clipped * np.log2(clipped) + (1.0 - clipped) * np.log2(1.0 - clipped))


def summarize_morphology_samples(
    decoded_samples: np.ndarray,
    posterior: BayesianMorphologyPosterior,
    config: RunConfig,
    reconstruction_uncertainty: float,
) -> MorphologyTaskSummary:
    """Summarize posterior morphology dispersion without consulting dense reference pixels."""

    fronts = []
    penetrations = []
    confidences = []
    for signal in decoded_samples:
        products = morphology_products_from_signal(
            signal,
            posterior.coarse_x,
            posterior.coarse_y,
            [f"channel_{index}" for index in range(signal.shape[0])],
            config,
        )
        probability = products["altered_region_probability"].values
        front, penetration = front_from_probability(
            probability, posterior.coarse_x, posterior.coarse_y, config
        )
        fronts.append(front)
        penetrations.append(penetration)
        confidences.append(float(np.mean(2.0 * np.abs(probability - 0.5))))
    front_masks = np.stack(fronts).astype(float)
    front_probability = np.mean(front_masks, axis=0)
    front_presence = np.any(front_masks > 0.5, axis=(1, 2))
    front_presence_probability = float(np.mean(front_presence))
    front_uncertainty = float(
        0.75 * np.mean(_binary_entropy(front_probability))
        + 0.25 * _binary_entropy(front_presence_probability)
    )
    field_width = max(float(posterior.coarse_x[-1] - posterior.coarse_x[0]), 1.0)
    depth = np.nan_to_num(np.stack(penetrations), nan=0.0)
    penetration_uncertainty = float(np.mean(np.var(depth, axis=0)) / field_width**2)
    weights = config.acquisition_v4.bayesian_morphology.fantasy.utility_weights
    total_weight = (
        weights.front_uncertainty
        + weights.penetration_uncertainty
        + weights.reconstruction_uncertainty
    )
    task_uncertainty = float(
        (
            weights.front_uncertainty * front_uncertainty
            + weights.penetration_uncertainty * penetration_uncertainty
            + weights.reconstruction_uncertainty * reconstruction_uncertainty
        )
        / total_weight
    )
    return MorphologyTaskSummary(
        task_uncertainty=task_uncertainty,
        front_uncertainty=front_uncertainty,
        penetration_uncertainty=penetration_uncertainty,
        reconstruction_uncertainty=float(reconstruction_uncertainty),
        front_presence_probability=front_presence_probability,
        mean_front_support_fraction=float(np.mean(front_masks)),
        state_assignment_confidence=float(np.mean(confidences)),
    )


def _current_task_summary(
    posterior: BayesianMorphologyPosterior,
    config: RunConfig,
    query_index: int,
) -> tuple[MorphologyTaskSummary, np.ndarray, np.ndarray]:
    settings = config.acquisition_v4.bayesian_morphology.fantasy
    mean, covariance = _weighted_moments(posterior, "coarse_mean", "coarse_covariance")
    rng = np.random.default_rng(settings.random_seed + 1009 * query_index)
    standard_normal = rng.standard_normal(
        (settings.current_posterior_samples, posterior.effective_components, mean.shape[1])
    )
    latent = _draw_latent_samples(
        mean,
        covariance,
        standard_normal,
        config.acquisition_v4.bayesian_residual.subtile.jitter,
    )
    decoded = decode_latent_fields(latent, posterior)
    reconstruction_uncertainty = float(
        np.mean(np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 0.0))
    )
    return (
        summarize_morphology_samples(decoded, posterior, config, reconstruction_uncertainty),
        mean,
        covariance,
    )


def morphology_reliability_passed(
    summary: MorphologyTaskSummary,
    observed_area_fraction: float,
    config: RunConfig,
) -> bool:
    gate = config.acquisition_v4.bayesian_morphology.fantasy.reliability_gate
    return bool(
        observed_area_fraction >= gate.minimum_observed_area_fraction
        and summary.front_presence_probability >= gate.minimum_front_presence_probability
        and gate.minimum_front_support_fraction
        <= summary.mean_front_support_fraction
        <= gate.maximum_front_support_fraction
        and summary.state_assignment_confidence >= gate.minimum_state_assignment_confidence
    )


def _weighted_candidate_moments(
    posterior: BayesianMorphologyPosterior,
    candidate_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weights = posterior.posterior_weights
    candidate_means = np.stack(
        [kernel.subtile_mean[:, candidate_indices] for kernel in posterior.kernels]
    )
    candidate_covariances = np.stack(
        [
            kernel.subtile_covariance[:, candidate_indices][:, :, candidate_indices]
            for kernel in posterior.kernels
        ]
    )
    coarse_means = np.stack([kernel.coarse_mean for kernel in posterior.kernels])
    coarse_covariances = np.stack([kernel.coarse_covariance for kernel in posterior.kernels])
    crosses = np.stack(
        [kernel.coarse_subtile_cross_covariance[:, :, candidate_indices] for kernel in posterior.kernels]
    )
    candidate_mean = np.tensordot(weights, candidate_means, axes=(0, 0))
    candidate_covariance = np.tensordot(weights, candidate_covariances, axes=(0, 0))
    coarse_mean = np.tensordot(weights, coarse_means, axes=(0, 0))
    coarse_covariance = np.tensordot(weights, coarse_covariances, axes=(0, 0))
    cross = np.tensordot(weights, crosses, axes=(0, 0))
    for weight, kernel_candidate_mean, kernel_coarse_mean in zip(
        weights, candidate_means, coarse_means
    ):
        candidate_delta = kernel_candidate_mean - candidate_mean
        coarse_delta = kernel_coarse_mean - coarse_mean
        candidate_covariance += weight * np.einsum("ci,cj->cij", candidate_delta, candidate_delta)
        coarse_covariance += weight * np.einsum("ci,cj->cij", coarse_delta, coarse_delta)
        cross += weight * np.einsum("ci,cj->cij", coarse_delta, candidate_delta)
    candidate_noise = np.tensordot(
        weights,
        np.stack([kernel.candidate_noise for kernel in posterior.kernels]),
        axes=(0, 0),
    )
    return candidate_mean, candidate_covariance, coarse_mean, coarse_covariance, cross, candidate_noise


def _candidate_task_gains(
    posterior: BayesianMorphologyPosterior,
    roi_id: str,
    current_summary: MorphologyTaskSummary,
    config: RunConfig,
    query_index: int,
) -> np.ndarray:
    settings = config.acquisition_v4.bayesian_morphology.fantasy
    indices = posterior.evaluation_groups[roi_id]
    (
        candidate_mean,
        candidate_covariance,
        coarse_mean,
        coarse_covariance,
        cross,
        candidate_noise,
    ) = _weighted_candidate_moments(posterior, indices)
    rng = np.random.default_rng(settings.random_seed + 7919 * query_index)
    fantasy_standard_normal = rng.standard_normal(
        (settings.fantasies_per_candidate, posterior.effective_components, len(indices))
    )
    conditional_standard_normal = rng.standard_normal(
        (
            settings.conditional_samples_per_fantasy,
            posterior.effective_components,
            coarse_mean.shape[1],
        )
    )
    fantasy_values = []
    conditioned = []
    jitter = config.acquisition_v4.bayesian_residual.subtile.jitter
    for component in range(posterior.effective_components):
        observation_covariance = candidate_covariance[component] + np.eye(len(indices)) * (
            candidate_noise[component] + jitter
        )
        observation_cholesky = _stable_cholesky(observation_covariance, jitter)
        fantasy_values.append(
            candidate_mean[component][None, :]
            + fantasy_standard_normal[:, component, :] @ observation_cholesky.T
        )
        gain = np.linalg.solve(
            observation_cholesky.T,
            np.linalg.solve(observation_cholesky, cross[component].T),
        ).T
        conditional_covariance = coarse_covariance[component] - gain @ cross[component].T
        conditioned.append((gain, 0.5 * (conditional_covariance + conditional_covariance.T)))
    fantasies = np.stack(fantasy_values, axis=1)
    gains = []
    for fantasy in fantasies:
        conditional_means = []
        conditional_covariances = []
        for component, (gain, conditional_covariance) in enumerate(conditioned):
            conditional_means.append(
                coarse_mean[component] + gain @ (fantasy[component] - candidate_mean[component])
            )
            conditional_covariances.append(conditional_covariance)
        conditional_mean = np.stack(conditional_means)
        conditional_covariance = np.stack(conditional_covariances)
        latent = _draw_latent_samples(
            conditional_mean,
            conditional_covariance,
            conditional_standard_normal,
            jitter,
        )
        decoded = decode_latent_fields(latent, posterior)
        reconstruction_uncertainty = float(
            np.mean(
                np.maximum(
                    np.diagonal(conditional_covariance, axis1=1, axis2=2),
                    0.0,
                )
            )
        )
        summary = summarize_morphology_samples(
            decoded, posterior, config, reconstruction_uncertainty
        )
        gains.append(current_summary.task_uncertainty - summary.task_uncertainty)
    return np.asarray(gains)


def task_gain_statistics(
    gains: np.ndarray,
    lcb_standard_errors: float,
) -> tuple[float, float, float]:
    """Return mean, standard error, and conservative lower bound for fantasy gains."""

    values = np.asarray(gains, dtype=float)
    if values.ndim != 1 or not values.size:
        raise ValueError("task gain statistics require at least one fantasy outcome")
    standard_error = (
        float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    )
    mean_gain = float(np.mean(values))
    return mean_gain, standard_error, mean_gain - lcb_standard_errors * standard_error


def morphology_fantasy_rank_candidates(
    scores: pd.DataFrame,
    posterior: BayesianMorphologyPosterior,
    observed_area_fraction: float,
    config: RunConfig,
    query_index: int,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Rank CPU-bounded posterior fantasies, falling back to evidence-gated EIVR."""

    evidence_selected, evidence_scores = evidence_gated_rank_candidates(scores, posterior, config)
    settings = config.acquisition_v4.bayesian_morphology.fantasy
    current_summary, _, _ = _current_task_summary(posterior, config, query_index)
    reliable = morphology_reliability_passed(
        current_summary, observed_area_fraction, config
    )
    reliability = {
        "query_index": query_index,
        "observed_area_fraction": observed_area_fraction,
        "front_presence_probability": current_summary.front_presence_probability,
        "mean_front_support_fraction": current_summary.mean_front_support_fraction,
        "state_assignment_confidence": current_summary.state_assignment_confidence,
        "current_task_uncertainty": current_summary.task_uncertainty,
        "front_uncertainty": current_summary.front_uncertainty,
        "penetration_uncertainty": current_summary.penetration_uncertainty,
        "reconstruction_uncertainty": current_summary.reconstruction_uncertainty,
        "morphology_reliability_passed": reliable,
    }
    ranked = evidence_scores.copy()
    ranked["task_gain_mean"] = np.nan
    ranked["task_gain_standard_error"] = np.nan
    ranked["task_gain_LCB"] = np.nan
    ranked["morphology_bonus_fraction"] = 0.0
    ranked["morphology_reliability_passed"] = reliable
    ranked["morphology_shortlist_eligible"] = (
        ranked["normalized_geometry_gain"] >= settings.geometry_shortlist_ratio
    )
    ranked["fantasy_evaluated"] = False
    ranked["selected"] = ranked["roi_id"] == str(evidence_selected["roi_id"])
    if not settings.enabled or not reliable:
        ranked["fallback_reason"] = (
            "fantasy_disabled" if not settings.enabled else "morphology_reliability_gate_failed"
        )
        return evidence_selected, ranked, pd.DataFrame(), reliability
    shortlist = ranked[ranked["morphology_shortlist_eligible"]].sort_values(
        ["geometry_coverage_gain", "row0", "column0"],
        ascending=[False, True, True],
    )
    shortlist = shortlist.head(settings.maximum_shortlist_candidates)
    if str(evidence_selected["roi_id"]) not in set(shortlist["roi_id"]):
        shortlist = pd.concat(
            [
                shortlist,
                ranked[ranked["roi_id"] == str(evidence_selected["roi_id"])],
            ],
            ignore_index=True,
        )
    fantasy_rows = []
    for _, candidate in shortlist.iterrows():
        roi_id = str(candidate["roi_id"])
        gains = _candidate_task_gains(posterior, roi_id, current_summary, config, query_index)
        mean_gain, standard_error, lower_bound = task_gain_statistics(
            gains, settings.task_gain_lcb_standard_errors
        )
        bonus = (
            min(
                settings.maximum_bonus_fraction,
                lower_bound / max(current_summary.task_uncertainty, 1.0e-12),
            )
            if lower_bound > 0.0
            else 0.0
        )
        row_index = ranked.index[ranked["roi_id"] == roi_id][0]
        ranked.loc[row_index, "task_gain_mean"] = mean_gain
        ranked.loc[row_index, "task_gain_standard_error"] = standard_error
        ranked.loc[row_index, "task_gain_LCB"] = lower_bound
        ranked.loc[row_index, "morphology_bonus_fraction"] = bonus
        ranked.loc[row_index, "fantasy_evaluated"] = True
        for fantasy_index, gain in enumerate(gains):
            fantasy_rows.append(
                {
                    "query_index": query_index,
                    "roi_id": roi_id,
                    "fantasy_index": fantasy_index,
                    "task_gain": float(gain),
                    "task_gain_mean": mean_gain,
                    "task_gain_standard_error": standard_error,
                    "task_gain_LCB": lower_bound,
                }
            )
    ranked["morphology_utility"] = ranked["normalized_geometry_gain"] * (
        1.0 + ranked["morphology_bonus_fraction"]
    )
    evidence_utility = float(
        ranked.loc[
            ranked["roi_id"] == str(evidence_selected["roi_id"]), "selection_utility"
        ].iloc[0]
    )
    eligible = ranked[
        ranked["fantasy_evaluated"]
        & (ranked["task_gain_LCB"] > 0.0)
        & ranked["morphology_shortlist_eligible"]
        & (ranked["morphology_utility"] > evidence_utility + config.acquisition_v4.bayesian_morphology.evidence_gate.tie_tolerance)
    ]
    if eligible.empty:
        selected = evidence_selected
        ranked["fallback_reason"] = "no_positive_morphology_override"
    else:
        maximum_utility = float(eligible["morphology_utility"].max())
        tied = eligible[
            eligible["morphology_utility"]
            >= maximum_utility
            - config.acquisition_v4.bayesian_morphology.evidence_gate.tie_tolerance
        ]
        selected = tied.sort_values(
            ["geometry_coverage_gain", "row0", "column0"],
            ascending=[False, True, True],
        ).iloc[0]
        ranked["fallback_reason"] = ""
    ranked["selected"] = ranked["roi_id"] == str(selected["roi_id"])
    return selected, ranked, pd.DataFrame(fantasy_rows), reliability


def gp_prediction_from_posterior(
    posterior: BayesianMorphologyPosterior,
    x: np.ndarray,
    y: np.ndarray,
    channels: list[str],
    config: RunConfig,
) -> xr.Dataset:
    """Decode the model-averaged GP mean as a diagnostic-only full-grid prediction."""

    coarse_mean, coarse_covariance = _weighted_moments(
        posterior, "coarse_mean", "coarse_covariance"
    )
    decoded = decode_latent_fields(coarse_mean[None, :, :], posterior)[0]
    y_mesh, x_mesh = np.meshgrid(y, x, indexing="ij")
    target = np.column_stack([y_mesh.ravel(), x_mesh.ravel()])
    full = []
    for channel in decoded:
        interpolator = RegularGridInterpolator(
            (posterior.coarse_y, posterior.coarse_x),
            channel,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        full.append(interpolator(target).reshape(len(y), len(x)))
    mean_signal = np.stack(full)
    latent_variance = np.mean(
        np.maximum(np.diagonal(coarse_covariance, axis1=1, axis2=2), 0.0),
        axis=0,
    ).reshape(len(posterior.coarse_y), len(posterior.coarse_x))
    uncertainty_interpolator = RegularGridInterpolator(
        (posterior.coarse_y, posterior.coarse_x),
        latent_variance,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    uncertainty = uncertainty_interpolator(target).reshape(len(y), len(x))
    uncertainty = np.clip(uncertainty / max(float(np.max(uncertainty)), 1.0e-12), 0.0, 1.0)
    prediction = xr.Dataset(
        {
            "mean_intensity": (("channel", "y", "x"), mean_signal),
            "epistemic_uncertainty": (
                ("channel", "y", "x"),
                np.broadcast_to(uncertainty, mean_signal.shape),
            ),
            "reconstruction_uncertainty": (("y", "x"), uncertainty),
        },
        coords={"channel": channels, "x": x, "y": y},
    )
    return xr.merge(
        [
            prediction,
            morphology_products_from_signal(
                mean_signal,
                x,
                y,
                channels,
                config,
                reconstruction_uncertainty=uncertainty,
            ),
        ]
    )
