"""Pareto-gated Bayesian EIVR selectors for v4.4 uncertainty-first replay."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA

from .domain import RunConfig
from .v3_morphology import morphology_products_from_signal
from .v3_validation import _raster_cost
from .v4_bayesian_residual import (
    anisotropic_matern_3_2_kernel,
    build_subtile_catalog,
    normalized_kernel_weights,
    subtile_geometry_for_roi,
)
from .v4_validation import lookahead_coverage_gain

PARETO_POLICY_PATTERN = re.compile(
    r"^bayesian_pareto_eivr_(?P<grid>4x4|8x8)_(?P<features>mean|texture)_tau(?P<tau>090|085)$"
)
PARETO_ADDITIVE_POLICY_PATTERN = re.compile(
    r"^bayesian_pareto_additive_eivr_4x4_mean_tau090_alpha(?P<alpha>1|2|5|10)$"
)


@dataclass(frozen=True)
class ParetoPolicySpec:
    policy: str
    grid_shape: tuple[int, int]
    feature_mode: str
    geometry_threshold: float


@dataclass(frozen=True)
class ParetoAdditivePolicySpec:
    policy: str
    pareto_spec: ParetoPolicySpec
    exchange_rate: float


@dataclass(frozen=True)
class ParetoSubtileObservation:
    roi_id: str
    subtile_id: str
    center_x_nm: float
    center_y_nm: float
    acquisition_sequence: int
    feature_values: np.ndarray
    mean_noise: np.ndarray
    channel_mean: np.ndarray
    pixel_count: int


@dataclass
class ParetoKernelPosterior:
    label: str
    length_scales: tuple[float, float]
    log_marginal_likelihood: float
    cholesky: np.ndarray
    alpha: np.ndarray
    evaluation_projection: np.ndarray
    candidate_noise: np.ndarray
    integrated_variance: float


@dataclass
class ParetoPosterior:
    spec: ParetoPolicySpec
    kernel_labels: list[str]
    posterior_weights: np.ndarray
    kernels: list[ParetoKernelPosterior]
    subtile_catalog: pd.DataFrame
    evaluation_groups: dict[str, np.ndarray]
    training_points: np.ndarray
    evaluation_points: np.ndarray
    effective_components: int
    feature_families: list[str]
    feature_median: np.ndarray
    feature_iqr: np.ndarray
    pca_components: np.ndarray
    pca_mean: np.ndarray
    integrated_variance: float
    retained_training_count: int
    total_revealed_subtiles: int


def parse_pareto_policy(policy: str) -> ParetoPolicySpec:
    """Parse a v4.4 policy name into its deterministic selector settings."""

    match = PARETO_POLICY_PATTERN.match(policy)
    if match is None:
        raise ValueError(f"unsupported v4.4 Pareto policy: {policy}")
    grid = (4, 4) if match.group("grid") == "4x4" else (8, 8)
    threshold = 0.90 if match.group("tau") == "090" else 0.85
    return ParetoPolicySpec(
        policy=policy,
        grid_shape=grid,
        feature_mode=match.group("features"),
        geometry_threshold=threshold,
    )


def parse_pareto_additive_policy(policy: str) -> ParetoAdditivePolicySpec:
    """Parse a v4.5 additive Pareto policy name."""

    match = PARETO_ADDITIVE_POLICY_PATTERN.match(policy)
    if match is None:
        raise ValueError(f"unsupported v4.5 additive Pareto policy: {policy}")
    return ParetoAdditivePolicySpec(
        policy=policy,
        pareto_spec=ParetoPolicySpec(
            policy=policy,
            grid_shape=(4, 4),
            feature_mode="mean",
            geometry_threshold=0.90,
        ),
        exchange_rate=float(match.group("alpha")),
    )


def feature_families_for_mode(feature_mode: str) -> list[str]:
    if feature_mode == "mean":
        return ["mean"]
    if feature_mode == "texture":
        return ["mean", "residual_mad", "gradient_mean", "gradient_p95", "contrast"]
    raise ValueError(f"unsupported v4.4 feature mode: {feature_mode}")


def _normalized_points(points_nm: np.ndarray, config: RunConfig) -> np.ndarray:
    return np.column_stack(
        [
            points_nm[:, 0] / max(config.scenario.width_nm, 1.0),
            points_nm[:, 1] / max(config.scenario.height_nm, 1.0),
        ]
    )


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
    repaired = (eigenvectors * clipped[None, :]) @ eigenvectors.T
    return np.linalg.cholesky(0.5 * (repaired + repaired.T) + identity * jitter)


def _residual_mad(values: np.ndarray, sigma_px: float) -> tuple[np.ndarray, np.ndarray]:
    smoothed = gaussian_filter(values.astype(float), sigma=(0.0, sigma_px, sigma_px))
    residual = values - smoothed
    centered = residual - np.median(residual, axis=(1, 2), keepdims=True)
    mad = 1.4826 * np.median(np.abs(centered), axis=(1, 2))
    pixels = max(values.shape[1] * values.shape[2], 1)
    mean_noise = (mad**2) / pixels
    return mad, mean_noise


def _feature_tensor(values: np.ndarray, feature_mode: str, sigma_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-channel subtile features and mean-intensity noise."""

    mean = np.mean(values, axis=(1, 2))
    residual_mad, mean_noise = _residual_mad(values, sigma_px)
    if feature_mode == "mean":
        return mean[:, None], mean_noise
    dy = np.gradient(values, axis=1)
    dx = np.gradient(values, axis=2)
    gradient = np.hypot(dx, dy)
    features = np.stack(
        [
            mean,
            residual_mad,
            np.mean(gradient, axis=(1, 2)),
            np.percentile(gradient, 95.0, axis=(1, 2)),
            np.percentile(values, 95.0, axis=(1, 2)) - np.percentile(values, 5.0, axis=(1, 2)),
        ],
        axis=1,
    )
    return features, mean_noise


def pareto_subtile_observations_from_revealed_roi(
    roi: pd.Series,
    dense_signal: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    acquisition_sequence: int,
    config: RunConfig,
    spec: ParetoPolicySpec,
) -> list[ParetoSubtileObservation]:
    """Convert one revealed raster tile into v4.4 mean or texture subtile observations."""

    settings = config.acquisition_v4.bayesian_pareto
    geometry = subtile_geometry_for_roi(roi, x, y, spec.grid_shape)
    observations = []
    for _, subtile in geometry.iterrows():
        values = dense_signal[
            :,
            int(subtile["row0"]) : int(subtile["row1"]),
            int(subtile["column0"]) : int(subtile["column1"]),
        ]
        feature_values, mean_noise = _feature_tensor(
            values, spec.feature_mode, settings.residual_filter_sigma_px
        )
        observations.append(
            ParetoSubtileObservation(
                roi_id=str(roi["roi_id"]),
                subtile_id=str(subtile["subtile_id"]),
                center_x_nm=float(subtile["center_x_nm"]),
                center_y_nm=float(subtile["center_y_nm"]),
                acquisition_sequence=acquisition_sequence,
                feature_values=feature_values,
                mean_noise=mean_noise,
                channel_mean=np.mean(values, axis=(1, 2)),
                pixel_count=int(subtile["pixel_count"]),
            )
        )
    return observations


def robust_scale_feature_tensor(
    values: np.ndarray,
    epsilon: float = 1.0e-6,
    clip: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale revealed features independently by channel and feature family."""

    if values.ndim != 3:
        raise ValueError("feature tensor must have shape (node, channel, feature)")
    median = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
    iqr = np.maximum(q75 - q25, epsilon)
    scaled = np.clip((values - median[None, :, :]) / iqr[None, :, :], -clip, clip)
    return scaled, median, iqr


def select_training_subtile_indices(
    observations: list[ParetoSubtileObservation],
    spec: ParetoPolicySpec,
    max_training_subtiles: int,
) -> np.ndarray:
    """Deterministically cap 8x8 training subtiles using coordinates only."""

    count = len(observations)
    if spec.grid_shape != (8, 8) or count <= max_training_subtiles:
        return np.arange(count, dtype=int)
    latest_sequence = max(node.acquisition_sequence for node in observations)
    recent = [
        index
        for index, node in enumerate(observations)
        if node.acquisition_sequence == latest_sequence
    ]
    selected = list(recent[:max_training_subtiles])
    remaining_slots = max_training_subtiles - len(selected)
    if remaining_slots <= 0:
        return np.asarray(selected[:max_training_subtiles], dtype=int)
    candidates = [
        index
        for index, node in enumerate(observations)
        if node.acquisition_sequence != latest_sequence
    ]
    if not selected and candidates:
        selected.append(
            min(
                candidates,
                key=lambda index: (
                    observations[index].acquisition_sequence,
                    observations[index].center_y_nm,
                    observations[index].center_x_nm,
                    observations[index].subtile_id,
                ),
            )
        )
        candidates.remove(selected[-1])
    centers = np.asarray([[node.center_x_nm, node.center_y_nm] for node in observations], dtype=float)
    while candidates and len(selected) < max_training_subtiles:
        selected_centers = centers[np.asarray(selected, dtype=int)]
        best = max(
            candidates,
            key=lambda index: (
                float(np.min(np.linalg.norm(centers[index][None, :] - selected_centers, axis=1))),
                -observations[index].acquisition_sequence,
                -observations[index].center_y_nm,
                -observations[index].center_x_nm,
                observations[index].subtile_id,
            ),
        )
        selected.append(best)
        candidates.remove(best)
    return np.asarray(selected, dtype=int)


def _fit_kernel(
    label: str,
    length_scales: tuple[float, float],
    training_points: np.ndarray,
    targets: np.ndarray,
    training_noise: np.ndarray,
    evaluation_points: np.ndarray,
    jitter: float,
) -> ParetoKernelPosterior:
    base_training = anisotropic_matern_3_2_kernel(
        training_points, training_points, length_scales
    )
    evaluation_training = anisotropic_matern_3_2_kernel(
        evaluation_points, training_points, length_scales
    )
    cholesky = []
    alpha = []
    projections = []
    variances = []
    log_likelihood = 0.0
    for component in range(targets.shape[1]):
        covariance = base_training + np.diag(training_noise[:, component] + jitter)
        factor = _stable_cholesky(covariance, jitter)
        solved = np.linalg.solve(factor, targets[:, component])
        weights = np.linalg.solve(factor.T, solved)
        log_likelihood += float(
            -0.5 * targets[:, component] @ weights
            - np.sum(np.log(np.diag(factor)))
            - 0.5 * len(training_points) * np.log(2.0 * np.pi)
        )
        projected = np.linalg.solve(factor, evaluation_training.T)
        variances.append(np.maximum(1.0 - np.sum(projected * projected, axis=0), 0.0))
        cholesky.append(factor)
        alpha.append(weights)
        projections.append(projected)
    return ParetoKernelPosterior(
        label=label,
        length_scales=length_scales,
        log_marginal_likelihood=log_likelihood,
        cholesky=np.stack(cholesky),
        alpha=np.stack(alpha),
        evaluation_projection=np.stack(projections),
        candidate_noise=np.maximum(np.median(training_noise, axis=0), 1.0e-12),
        integrated_variance=float(np.mean(np.stack(variances))),
    )


def _posterior_covariance_block(
    kernel: ParetoKernelPosterior,
    component: int,
    training_points: np.ndarray,
    first_points: np.ndarray,
    second_points: np.ndarray,
) -> np.ndarray:
    prior = anisotropic_matern_3_2_kernel(first_points, second_points, kernel.length_scales)
    first_training = anisotropic_matern_3_2_kernel(first_points, training_points, kernel.length_scales)
    second_training = anisotropic_matern_3_2_kernel(second_points, training_points, kernel.length_scales)
    first_projected = np.linalg.solve(kernel.cholesky[component], first_training.T)
    second_projected = np.linalg.solve(kernel.cholesky[component], second_training.T)
    return prior - first_projected.T @ second_projected


def _candidate_eivr(
    kernel: ParetoKernelPosterior,
    candidate_indices: np.ndarray,
    posterior: ParetoPosterior,
) -> float:
    candidate_points = posterior.evaluation_points[candidate_indices]
    prior_cross = anisotropic_matern_3_2_kernel(
        posterior.evaluation_points, candidate_points, kernel.length_scales
    )
    prior_observed = anisotropic_matern_3_2_kernel(
        candidate_points, candidate_points, kernel.length_scales
    )
    reductions = []
    for component in range(posterior.effective_components):
        projected_evaluation = kernel.evaluation_projection[component]
        projected_candidate = projected_evaluation[:, candidate_indices]
        cross = prior_cross - projected_evaluation.T @ projected_candidate
        observed = prior_observed - projected_candidate.T @ projected_candidate
        observed = observed + np.eye(len(candidate_indices)) * kernel.candidate_noise[component]
        factor = _stable_cholesky(observed, 1.0e-10)
        solved = np.linalg.solve(factor.T, np.linalg.solve(factor, cross.T)).T
        reductions.append(np.sum(cross * solved, axis=1))
    return float(np.mean(np.maximum(np.stack(reductions), 0.0)))


def fit_pareto_subtile_posterior(
    observations: list[ParetoSubtileObservation],
    catalog: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    config: RunConfig,
    spec: ParetoPolicySpec,
) -> ParetoPosterior:
    """Fit the revealed-only model-averaged latent GP posterior for v4.4."""

    if len(observations) < 2:
        raise ValueError("v4.4 Pareto posterior requires at least two revealed subtiles")
    settings = config.acquisition_v4.bayesian_pareto
    retained = select_training_subtile_indices(
        observations, spec, settings.max_training_subtiles_8x8
    )
    nodes = [observations[int(index)] for index in retained]
    raw = np.stack([node.feature_values for node in nodes])
    scaled, feature_median, feature_iqr = robust_scale_feature_tensor(
        raw,
        epsilon=settings.robust_iqr_epsilon,
        clip=settings.scaled_feature_clip,
    )
    flattened = scaled.reshape(len(nodes), -1)
    effective_components = min(settings.latent_components, flattened.shape[1], len(nodes) - 1)
    if effective_components < 1:
        raise ValueError("v4.4 Pareto posterior needs at least one latent component")
    embedding = PCA(n_components=effective_components, svd_solver="full")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
        )
        targets = embedding.fit_transform(flattened)

    mean_noise = np.stack([node.mean_noise for node in nodes])
    channels = raw.shape[1]
    features = raw.shape[2]
    scaled_noise = np.empty_like(raw, dtype=float)
    scaled_noise[:, :, 0] = mean_noise / np.maximum(feature_iqr[:, 0][None, :] ** 2, settings.robust_iqr_epsilon)
    if features > 1:
        mean_noise_floor = np.maximum(np.median(scaled_noise[:, :, 0], axis=0), settings.alpha_floor)
        for feature_index in range(1, features):
            scaled_noise[:, :, feature_index] = mean_noise_floor[None, :]
    flattened_noise = scaled_noise.reshape(len(nodes), channels * features)
    training_noise = np.maximum(
        flattened_noise @ (embedding.components_**2).T,
        settings.alpha_floor,
    )

    subtile_catalog = build_subtile_catalog(catalog, x, y, spec.grid_shape)
    training_nm = np.asarray([[node.center_x_nm, node.center_y_nm] for node in nodes], dtype=float)
    evaluation_nm = subtile_catalog[["center_x_nm", "center_y_nm"]].to_numpy(float)
    training_points = _normalized_points(training_nm, config)
    evaluation_points = _normalized_points(evaluation_nm, config)
    kernels = [
        _fit_kernel(
            f"{length_x:.6g},{length_y:.6g}",
            (length_x, length_y),
            training_points,
            targets,
            training_noise,
            evaluation_points,
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
    integrated = float(np.dot(weights, [kernel.integrated_variance for kernel in kernels]))
    return ParetoPosterior(
        spec=spec,
        kernel_labels=[kernel.label for kernel in kernels],
        posterior_weights=weights,
        kernels=kernels,
        subtile_catalog=subtile_catalog,
        evaluation_groups=groups,
        training_points=training_points,
        evaluation_points=evaluation_points,
        effective_components=effective_components,
        feature_families=feature_families_for_mode(spec.feature_mode),
        feature_median=feature_median,
        feature_iqr=feature_iqr,
        pca_components=embedding.components_,
        pca_mean=embedding.mean_,
        integrated_variance=integrated,
        retained_training_count=len(nodes),
        total_revealed_subtiles=len(observations),
    )


def pareto_geometry_scores(
    catalog: pd.DataFrame,
    queried_ids: set[str],
    distance_pixels: np.ndarray,
    config: RunConfig,
    threshold: float,
) -> pd.DataFrame:
    """Compute the cheap geometry shortlist before any Bayesian EIVR work."""

    scores = catalog[~catalog["roi_id"].isin(queried_ids)].copy()
    if scores.empty:
        raise ValueError("no feasible v4.4 raster candidates remain")
    scores["geometry_gain"] = [
        lookahead_coverage_gain(distance_pixels, roi) for _, roi in scores.iterrows()
    ]
    scores["estimated_raster_cost_s"] = [
        _raster_cost(config, roi)[0] for _, roi in scores.iterrows()
    ]
    scores["geometry_gain_per_cost"] = scores["geometry_gain"] / np.maximum(
        scores["estimated_raster_cost_s"], 1.0e-12
    )
    best = max(float(scores["geometry_gain_per_cost"].max()), 1.0e-12)
    scores["normalized_geometry_gain"] = scores["geometry_gain_per_cost"] / best
    scores["geometry_threshold"] = threshold
    scores["geometry_shortlist_eligible"] = scores["normalized_geometry_gain"] >= threshold
    best_index = scores.sort_values(
        ["geometry_gain_per_cost", "row0", "column0"],
        ascending=[False, True, True],
    ).index[0]
    scores.loc[best_index, "geometry_shortlist_eligible"] = True
    scores["geometry_argmax"] = False
    scores.loc[best_index, "geometry_argmax"] = True
    scores["shortlist_size"] = int(scores["geometry_shortlist_eligible"].sum())
    return scores


def score_shortlisted_eivr(
    geometry_scores: pd.DataFrame,
    posterior: ParetoPosterior,
    config: RunConfig,
) -> pd.DataFrame:
    """Evaluate Bayesian EIVR only for shortlisted candidates plus the geometry argmax."""

    scored = geometry_scores.copy()
    shortlisted = scored["geometry_shortlist_eligible"].astype(bool)
    scored["eivr_evaluated"] = shortlisted
    scored["EIVR_by_kernel"] = "{}"
    scored["model_averaged_fractional_EIVR"] = np.nan
    for index, row in scored[shortlisted].iterrows():
        roi_id = str(row["roi_id"])
        reductions = []
        fractional = []
        candidate_indices = posterior.evaluation_groups[roi_id]
        for kernel in posterior.kernels:
            reduction = _candidate_eivr(kernel, candidate_indices, posterior)
            reductions.append(reduction)
            fractional.append(reduction / max(kernel.integrated_variance, 1.0e-12))
        scored.at[index, "EIVR_by_kernel"] = json.dumps(
            dict(zip(posterior.kernel_labels, fractional))
        )
        scored.at[index, "model_averaged_fractional_EIVR"] = float(
            np.dot(posterior.posterior_weights, fractional)
        )
    return scored


def pareto_rank_candidates(
    scores: pd.DataFrame,
    posterior: ParetoPosterior,
    config: RunConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    """Rank shortlisted candidates by geometry safety plus uncapped Bayesian evidence."""

    settings = config.acquisition_v4.bayesian_pareto
    ranked = scores.copy()
    ranked["EIVR_LCB"] = np.nan
    ranked["kernel_support"] = np.nan
    ranked["relative_evidence"] = 0.0
    ranked["evidence_eligible"] = False
    ranked["selection_utility"] = np.where(
        ranked["geometry_shortlist_eligible"], ranked["normalized_geometry_gain"], -np.inf
    )
    geometry = ranked[ranked["geometry_argmax"]].iloc[0]
    baseline_by_kernel = json.loads(str(geometry["EIVR_by_kernel"]))
    baseline_eivr = max(float(geometry["model_averaged_fractional_EIVR"]), 1.0e-12)
    for index, row in ranked[ranked["eivr_evaluated"]].iterrows():
        by_kernel = json.loads(str(row["EIVR_by_kernel"]))
        deltas = np.asarray(
            [
                float(by_kernel[label]) - float(baseline_by_kernel[label])
                for label in posterior.kernel_labels
            ],
            dtype=float,
        )
        mean_delta = float(np.dot(posterior.posterior_weights, deltas))
        std_delta = float(
            np.sqrt(np.dot(posterior.posterior_weights, (deltas - mean_delta) ** 2))
        )
        lcb = mean_delta - settings.lcb_standard_deviations * std_delta
        support = float(np.sum(posterior.posterior_weights[deltas > 0.0]))
        eligible = lcb > 0.0 and support >= settings.minimum_kernel_support
        relative = max(0.0, lcb / baseline_eivr) if eligible else 0.0
        ranked.at[index, "EIVR_LCB"] = lcb
        ranked.at[index, "kernel_support"] = support
        ranked.at[index, "relative_evidence"] = relative
        ranked.at[index, "evidence_eligible"] = eligible
        ranked.at[index, "selection_utility"] = float(
            row["normalized_geometry_gain"] * (1.0 + relative)
        )
    maximum = float(ranked["selection_utility"].max())
    tolerance = 1.0e-12
    tied = ranked[ranked["selection_utility"] >= maximum - tolerance]
    selected = tied.sort_values(
        ["selection_utility", "geometry_gain_per_cost", "row0", "column0"],
        ascending=[False, False, True, True],
    ).iloc[0]
    ranked["selected"] = ranked["roi_id"] == str(selected["roi_id"])
    return selected, ranked


def pareto_additive_rank_candidates(
    scores: pd.DataFrame,
    posterior: ParetoPosterior,
    config: RunConfig,
    exchange_rate: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Rank shortlisted candidates with additive geometry-EIVR exchange."""

    settings = config.acquisition_v4.bayesian_pareto_additive
    ranked = scores.copy()
    ranked["EIVR_LCB"] = np.nan
    ranked["kernel_support"] = np.nan
    ranked["evidence_eligible"] = False
    ranked["additive_exchange_rate"] = float(exchange_rate)
    ranked["additive_bonus"] = 0.0
    ranked["relative_evidence"] = 0.0
    ranked["selection_utility"] = np.where(
        ranked["geometry_shortlist_eligible"], ranked["normalized_geometry_gain"], -np.inf
    )
    geometry = ranked[ranked["geometry_argmax"]].iloc[0]
    geometry_roi_id = str(geometry["roi_id"])
    baseline_by_kernel = json.loads(str(geometry["EIVR_by_kernel"]))
    for index, row in ranked[ranked["eivr_evaluated"]].iterrows():
        by_kernel = json.loads(str(row["EIVR_by_kernel"]))
        deltas = np.asarray(
            [
                float(by_kernel[label]) - float(baseline_by_kernel[label])
                for label in posterior.kernel_labels
            ],
            dtype=float,
        )
        mean_delta = float(np.dot(posterior.posterior_weights, deltas))
        std_delta = float(
            np.sqrt(np.dot(posterior.posterior_weights, (deltas - mean_delta) ** 2))
        )
        lcb = mean_delta - settings.lcb_standard_deviations * std_delta
        support = float(np.sum(posterior.posterior_weights[deltas > 0.0]))
        eligible = (
            bool(row["geometry_shortlist_eligible"])
            and lcb > 0.0
            and support >= settings.minimum_kernel_support
        )
        bonus = exchange_rate * lcb if eligible else 0.0
        ranked.at[index, "EIVR_LCB"] = lcb
        ranked.at[index, "kernel_support"] = support
        ranked.at[index, "evidence_eligible"] = eligible
        ranked.at[index, "additive_bonus"] = bonus
        ranked.at[index, "selection_utility"] = float(
            row["normalized_geometry_gain"] + bonus
        )
    maximum = float(ranked["selection_utility"].max())
    tolerance = 1.0e-12
    tied = ranked[ranked["selection_utility"] >= maximum - tolerance]
    selected = tied.sort_values(
        ["selection_utility", "geometry_gain_per_cost", "row0", "column0"],
        ascending=[False, False, True, True],
    ).iloc[0]
    selected_roi_id = str(selected["roi_id"])
    ranked["selected"] = ranked["roi_id"] == selected_roi_id
    ranked["geometry_argmax_roi_id"] = geometry_roi_id
    ranked["selected_roi_id"] = selected_roi_id
    ranked["selected_was_geometry_argmax"] = selected_roi_id == geometry_roi_id
    return selected, ranked


def pareto_select_candidate(
    catalog: pd.DataFrame,
    queried_ids: set[str],
    distance_pixels: np.ndarray,
    posterior: ParetoPosterior,
    config: RunConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    geometry = pareto_geometry_scores(
        catalog, queried_ids, distance_pixels, config, posterior.spec.geometry_threshold
    )
    scored = score_shortlisted_eivr(geometry, posterior, config)
    return pareto_rank_candidates(scored, posterior, config)


def pareto_additive_select_candidate(
    catalog: pd.DataFrame,
    queried_ids: set[str],
    distance_pixels: np.ndarray,
    posterior: ParetoPosterior,
    config: RunConfig,
    exchange_rate: float,
) -> tuple[pd.Series, pd.DataFrame]:
    threshold = config.acquisition_v4.bayesian_pareto_additive.geometry_shortlist_ratio
    geometry = pareto_geometry_scores(catalog, queried_ids, distance_pixels, config, threshold)
    scored = score_shortlisted_eivr(geometry, posterior, config)
    return pareto_additive_rank_candidates(scored, posterior, config, exchange_rate)


def gp_prediction_from_pareto_posterior(
    posterior: ParetoPosterior,
    x: np.ndarray,
    y: np.ndarray,
    channels: list[str],
    config: RunConfig,
) -> xr.Dataset:
    """Decode the model-averaged GP mean as endpoint-only diagnostic reconstruction."""

    settings = config.acquisition_v4.bayesian_pareto.gp_reconstruction
    yy, xx = np.meshgrid(y, x, indexing="ij")
    pixel_nm = np.column_stack([xx.ravel(), yy.ravel()])
    pixel_points = _normalized_points(pixel_nm, config)
    chunk = settings.chunk_pixels
    latent_mean = np.zeros((len(pixel_points), posterior.effective_components), dtype=float)
    latent_var = np.zeros(len(pixel_points), dtype=float)
    for start in range(0, len(pixel_points), chunk):
        stop = min(start + chunk, len(pixel_points))
        points = pixel_points[start:stop]
        chunk_mean = np.zeros((len(points), posterior.effective_components), dtype=float)
        chunk_var = np.zeros(len(points), dtype=float)
        for weight, kernel in zip(posterior.posterior_weights, posterior.kernels):
            training_cross = anisotropic_matern_3_2_kernel(
                points, posterior.training_points, kernel.length_scales
            )
            kernel_var_components = []
            for component in range(posterior.effective_components):
                mean = training_cross @ kernel.alpha[component]
                projected = np.linalg.solve(kernel.cholesky[component], training_cross.T)
                variance = np.maximum(1.0 - np.sum(projected * projected, axis=0), 0.0)
                chunk_mean[:, component] += weight * mean
                kernel_var_components.append(variance)
            chunk_var += weight * np.mean(np.stack(kernel_var_components), axis=0)
        latent_mean[start:stop] = chunk_mean
        latent_var[start:stop] = chunk_var
    decoded_flat = latent_mean @ posterior.pca_components + posterior.pca_mean[None, :]
    channel_count = len(channels)
    feature_count = len(posterior.feature_families)
    decoded = decoded_flat.reshape(len(pixel_points), channel_count, feature_count)
    mean_feature = decoded[:, :, 0]
    mean_signal_flat = mean_feature * posterior.feature_iqr[:, 0][None, :] + posterior.feature_median[:, 0][None, :]
    mean_signal = mean_signal_flat.T.reshape(channel_count, len(y), len(x))
    uncertainty = latent_var.reshape(len(y), len(x))
    uncertainty = np.clip(
        uncertainty / max(float(np.max(uncertainty)), 1.0e-12),
        0.0,
        1.0,
    )
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
