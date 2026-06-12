"""V3 morphology reconstruction products for unannotated corrosion map replay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt, gaussian_filter, label
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
import warnings

from .domain import RunConfig


@dataclass(frozen=True)
class FrontMetrics:
    mean_symmetric_distance_nm: float
    hausdorff_distance_nm: float
    localization_mean_symmetric_distance_nm: float
    localization_hausdorff_distance_nm: float
    reference_front_present: bool
    predicted_front_present: bool
    availability_status: str
    detection_correct: bool


def dense_signal_from_observations(config: RunConfig, observations: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Convert standardized dense observations into channel, y, x intensity proxies."""

    x = np.unique(observations["x_nm"].values.astype(float))
    y = np.unique(observations["y_nm"].values.astype(float))
    elements = observations.coords["element"].values.astype(str).tolist()
    values = observations["counts"].values.reshape(len(y), len(x), len(elements))
    signal = values.transpose(2, 0, 1).astype(float)
    if config.dataset.value_semantics == "counts":
        dwell = observations["dwell_ms"].values.reshape(len(y), len(x)).astype(float)
        signal = signal / np.maximum(dwell[None, :, :], 1e-12)
    return signal, x, y, elements


def robust_scale_signal(signal: np.ndarray) -> np.ndarray:
    """Map each channel to a robust [0, 1] intensity feature."""

    signal = np.asarray(signal, dtype=float)
    features = np.zeros_like(signal, dtype=float)
    for index, channel in enumerate(signal):
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            continue
        low, high = np.percentile(finite, [1.0, 99.0])
        span = high - low
        if span <= 1e-12:
            continue
        features[index] = np.clip((channel - low) / span, 0.0, 1.0)
    return features


def _first_component(features: np.ndarray) -> np.ndarray:
    points = features.reshape(features.shape[0], -1).T
    centered = points - points.mean(axis=0, keepdims=True)
    if np.max(np.abs(centered)) <= 1e-12:
        return np.zeros(features.shape[1:], dtype=float)
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    _, vectors = np.linalg.eigh(covariance)
    component = centered @ vectors[:, -1]
    component = component.reshape(features.shape[1:])
    low, high = np.percentile(component, [1.0, 99.0])
    if high - low <= 1e-12:
        return np.zeros_like(component)
    return np.clip((component - low) / (high - low), 0.0, 1.0)


def _surface_mask(shape: tuple[int, int], config: RunConfig) -> np.ndarray:
    rows, columns = shape
    mask = np.zeros(shape, dtype=bool)
    side = config.morphology.surface_side
    if side in ("left", "right"):
        width = max(1, int(np.ceil(0.08 * columns)))
        if side == "left":
            mask[:, :width] = True
        else:
            mask[:, -width:] = True
    else:
        height = max(1, int(np.ceil(0.08 * rows)))
        if side == "top":
            mask[:height, :] = True
        else:
            mask[-height:, :] = True
    return mask


def _connected_to_surface(mask: np.ndarray, config: RunConfig) -> np.ndarray:
    if not np.any(mask):
        return mask
    labels, count = label(mask)
    if count == 0:
        return mask
    surface_labels = np.unique(labels[_surface_mask(mask.shape, config)])
    surface_labels = surface_labels[surface_labels > 0]
    if surface_labels.size == 0:
        return mask
    return np.isin(labels, surface_labels)


def _cluster_probabilities(features: np.ndarray, config: RunConfig) -> np.ndarray:
    rows, columns = features.shape[1:]
    smoothed = np.stack(
        [
            gaussian_filter(channel, sigma=config.morphology.smoothing_sigma_px)
            if config.morphology.smoothing_sigma_px > 0
            else channel
            for channel in features
        ]
    )
    embedding = _first_component(smoothed)
    points = embedding.reshape(-1, 1)
    if np.unique(points, axis=0).shape[0] < 2:
        return np.zeros((2, rows, columns), dtype=float)
    if points.shape[0] > config.morphology.max_state_fit_points:
        fit_indices = np.linspace(
            0, points.shape[0] - 1, config.morphology.max_state_fit_points, dtype=int
        )
        fit_points = points[fit_indices]
    else:
        fit_points = points
    method = config.morphology.state_model
    if method == "otsu":
        flat_embedding = embedding.ravel()
        threshold = float(np.median(flat_embedding))
        altered = (flat_embedding >= threshold).astype(float)
        return np.stack([1.0 - altered, altered]).reshape(2, rows, columns)
    if method == "kmeans":
        model = KMeans(n_clusters=2, random_state=0, n_init=10).fit(fit_points)
        labels = model.predict(points)
        probabilities = np.zeros((points.shape[0], 2), dtype=float)
        probabilities[np.arange(points.shape[0]), labels] = 1.0
        return probabilities.T.reshape(2, rows, columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mixture = GaussianMixture(n_components=2, covariance_type="full", reg_covar=1e-6, random_state=0)
        mixture.fit(fit_points)
        probabilities = mixture.predict_proba(points)
    return probabilities.T.reshape(2, rows, columns)


def altered_probability(signal: np.ndarray, config: RunConfig) -> tuple[np.ndarray, np.ndarray]:
    """Estimate surface-connected altered-region probability from multichannel morphology."""

    features = robust_scale_signal(signal)
    if np.max(features) <= 1e-12:
        state_probability = np.zeros((2, signal.shape[1], signal.shape[2]), dtype=float)
        state_probability[0] = 1.0
        return np.zeros(signal.shape[1:], dtype=float), state_probability
    raw_state_probability = _cluster_probabilities(features, config)
    surface = _surface_mask(features.shape[1:], config)
    surface_scores = raw_state_probability[:, surface].mean(axis=1)
    altered_label = int(np.argmax(surface_scores))
    probability = np.clip(raw_state_probability[altered_label], 0.0, 1.0)
    connected = _connected_to_surface(probability >= 0.5, config)
    if connected.mean() < config.morphology.minimum_altered_fraction:
        probability = np.zeros_like(probability)
    else:
        probability = np.where(connected, probability, np.minimum(probability, 0.49))
    state_probability = np.stack([1.0 - probability, probability])
    return probability, state_probability


def front_from_probability(
    probability: np.ndarray, x: np.ndarray, y: np.ndarray, config: RunConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a row/column-wise alteration-front proxy and penetration distribution."""

    mask = _connected_to_surface(probability >= 0.5, config)
    front = np.zeros_like(mask, dtype=bool)
    if config.morphology.penetration_axis == "x":
        depth = np.full(mask.shape[0], np.nan, dtype=float)
        for row in range(mask.shape[0]):
            columns = np.flatnonzero(mask[row])
            if columns.size == 0:
                continue
            column = int(columns.max() if config.morphology.surface_side == "left" else columns.min())
            front[row, column] = True
            if config.morphology.surface_side == "right":
                depth[row] = float(x[-1] - x[column])
            else:
                depth[row] = float(x[column] - x[0])
        return front, depth
    depth = np.full(mask.shape[1], np.nan, dtype=float)
    for column in range(mask.shape[1]):
        rows = np.flatnonzero(mask[:, column])
        if rows.size == 0:
            continue
        row = int(rows.max() if config.morphology.surface_side == "top" else rows.min())
        front[row, column] = True
        if config.morphology.surface_side == "bottom":
            depth[column] = float(y[-1] - y[row])
        else:
            depth[column] = float(y[row] - y[0])
    return front, depth


def _binary_entropy(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    entropy = -(clipped * np.log2(clipped) + (1.0 - clipped) * np.log2(1.0 - clipped))
    return np.clip(entropy, 0.0, 1.0)


def morphology_products_from_signal(
    signal: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    channels: list[str],
    config: RunConfig,
    reconstruction_uncertainty: np.ndarray | None = None,
) -> xr.Dataset:
    """Build v3 morphology-proxy maps from a reconstructed multichannel signal."""

    probability, state_probability = altered_probability(signal, config)
    front, penetration = front_from_probability(probability, x, y, config)
    dy, dx = np.gradient(probability)
    front_probability = np.hypot(dx, dy)
    if float(front_probability.max()) > 0:
        front_probability = front_probability / float(front_probability.max())
    entropy_input = probability
    if reconstruction_uncertainty is not None:
        uncertainty = np.clip(reconstruction_uncertainty, 0.0, 1.0)
        entropy_input = np.clip((1.0 - uncertainty) * probability + uncertainty * 0.5, 0.0, 1.0)
        front_probability = np.maximum(front_probability, uncertainty * front_probability.max())
    front_entropy = _binary_entropy(entropy_input)
    if reconstruction_uncertainty is not None:
        front_entropy = np.maximum(front_entropy, np.clip(reconstruction_uncertainty, 0.0, 1.0) * 0.5)
    if config.morphology.penetration_axis == "x":
        penetration_dim = "y"
        penetration_coord = y
        field_span = float(x[-1] - x[0]) if len(x) > 1 else 1.0
        row_uncertainty = (
            np.nanmean(reconstruction_uncertainty, axis=1)
            if reconstruction_uncertainty is not None
            else np.zeros_like(penetration)
        )
    else:
        penetration_dim = "x"
        penetration_coord = x
        field_span = float(y[-1] - y[0]) if len(y) > 1 else 1.0
        row_uncertainty = (
            np.nanmean(reconstruction_uncertainty, axis=0)
            if reconstruction_uncertainty is not None
            else np.zeros_like(penetration)
        )
    penetration_variance = (np.nan_to_num(row_uncertainty, nan=1.0) * field_span) ** 2
    return xr.Dataset(
        {
            "altered_region_probability": (("y", "x"), probability),
            "alteration_front_probability": (("y", "x"), front_probability),
            "front_entropy": (("y", "x"), front_entropy),
            "morphology_state_probability": (
                ("state", "y", "x"),
                state_probability,
            ),
            "penetration_depth_mean_nm": ((penetration_dim,), penetration),
            "penetration_depth_distribution": ((penetration_dim,), penetration),
            "penetration_depth_variance_nm2": ((penetration_dim,), penetration_variance),
        },
        coords={
            "x": x,
            "y": y,
            "channel": channels,
            "state": ["unaltered_proxy", "altered_proxy"],
        },
    )


def reconstruct_from_observed_mask(
    dense_signal: np.ndarray,
    observed_mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    channels: list[str],
    config: RunConfig,
) -> xr.Dataset:
    """Nearest-observation reconstruction visible to v3 policies during replay."""

    observed_mask = observed_mask.astype(bool)
    rows, columns = observed_mask.shape
    if observed_mask.shape != dense_signal.shape[1:]:
        raise ValueError("observed mask must match dense signal spatial shape")
    if not np.any(observed_mask):
        means = np.zeros_like(dense_signal, dtype=float)
        reconstruction_uncertainty = np.ones((rows, columns), dtype=float)
    else:
        _, nearest_indices = distance_transform_edt(~observed_mask, return_indices=True)
        nearest_y, nearest_x = nearest_indices
        means = np.empty_like(dense_signal, dtype=float)
        for channel in range(dense_signal.shape[0]):
            means[channel] = dense_signal[channel, nearest_y, nearest_x]
        distances = distance_transform_edt(~observed_mask)
        max_distance = max(float(np.hypot(rows, columns)) / 2.0, 1.0)
        reconstruction_uncertainty = np.clip(distances / max_distance, 0.0, 1.0)
        reconstruction_uncertainty[observed_mask] = 0.0
    prediction = xr.Dataset(
        {
            "mean_intensity": (("channel", "y", "x"), means),
            "epistemic_uncertainty": (
                ("channel", "y", "x"),
                np.broadcast_to(reconstruction_uncertainty, means.shape),
            ),
            "reconstruction_uncertainty": (("y", "x"), reconstruction_uncertainty),
        },
        coords={"channel": channels, "x": x, "y": y},
        attrs={
            "task_mode": "corrosion_morphology_reconstruction",
            "data_semantics": "intensity_proxy",
            "label_status": "unannotated",
        },
    )
    return xr.merge(
        [
            prediction,
            morphology_products_from_signal(
                means, x, y, channels, config, reconstruction_uncertainty=reconstruction_uncertainty
            ),
        ]
    )


def pseudo_reference_from_dense_signal(
    dense_signal: np.ndarray, x: np.ndarray, y: np.ndarray, channels: list[str], config: RunConfig
) -> xr.Dataset:
    """Freeze an evaluation-only morphology proxy from the withheld dense map."""

    products = morphology_products_from_signal(dense_signal, x, y, channels, config)
    pseudo_front, penetration = front_from_probability(
        products["altered_region_probability"].values, x, y, config
    )
    reference = xr.Dataset(
        {
            "dense_intensity": (("channel", "y", "x"), dense_signal),
            "pseudo_altered_region": (
                ("y", "x"),
                products["altered_region_probability"].values >= 0.5,
            ),
            "pseudo_front": (("y", "x"), pseudo_front),
            "pseudo_penetration_depth_nm": (
                ("y" if config.morphology.penetration_axis == "x" else "x",),
                penetration,
            ),
        },
        coords={"channel": channels, "x": x, "y": y},
        attrs={
            "role": "evaluation_only_frozen_unsupervised_reference",
            "reference_method": config.morphology.reference_method,
            "front_semantics": "morphology_defined_alteration_front_proxy",
        },
    )
    return xr.merge([reference, products])


def front_distance_metrics(reference_front: np.ndarray, predicted_front: np.ndarray, x: np.ndarray, y: np.ndarray) -> FrontMetrics:
    """Symmetric distance from predicted to reference alteration-front pixels."""

    reference = np.asarray(reference_front, dtype=bool)
    predicted = np.asarray(predicted_front, dtype=bool)
    reference_present = bool(np.any(reference))
    predicted_present = bool(np.any(predicted))
    if not reference_present and not predicted_present:
        return FrontMetrics(0.0, 0.0, np.nan, np.nan, False, False, "matched_absent", True)
    dy = float(np.median(np.diff(y))) if len(y) > 1 else 1.0
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    fallback = float(np.hypot(dx * max(reference.shape[1] - 1, 1), dy * max(reference.shape[0] - 1, 1)))
    if not reference_present:
        return FrontMetrics(fallback, fallback, np.nan, np.nan, False, True, "spurious_front", False)
    if not predicted_present:
        return FrontMetrics(fallback, fallback, np.nan, np.nan, True, False, "missed_front", False)
    distance_to_predicted = distance_transform_edt(~predicted, sampling=(dy, dx))
    distance_to_reference = distance_transform_edt(~reference, sampling=(dy, dx))
    reference_distances = distance_to_predicted[reference]
    predicted_distances = distance_to_reference[predicted]
    combined = np.concatenate([reference_distances, predicted_distances])
    mean_distance = float(np.mean(combined))
    hausdorff_distance = float(np.max(combined))
    return FrontMetrics(
        mean_distance,
        hausdorff_distance,
        mean_distance,
        hausdorff_distance,
        True,
        True,
        "matched_present",
        True,
    )


def penetration_stat(values: np.ndarray, statistic: str) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan
    if statistic == "d95":
        return float(np.percentile(finite, 95.0))
    if statistic == "dmax":
        return float(np.max(finite))
    raise ValueError(f"unknown penetration statistic: {statistic}")


def normalized_reconstruction_rmse(reference_signal: np.ndarray, predicted_signal: np.ndarray) -> float:
    errors = []
    for reference, predicted in zip(reference_signal, predicted_signal):
        low, high = np.percentile(reference[np.isfinite(reference)], [1.0, 99.0])
        scale = max(float(high - low), 1.0)
        errors.append(np.mean(((predicted - reference) / scale) ** 2))
    return float(np.sqrt(np.mean(errors)))
