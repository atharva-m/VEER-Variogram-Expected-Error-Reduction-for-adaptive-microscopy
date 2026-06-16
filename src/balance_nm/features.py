"""Subtile feature extraction and the anisotropic Matern-3/2 kernel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .domain import RunConfig


@dataclass(frozen=True)
class SubtileSpec:
    grid_shape: tuple[int, int]
    feature_mode: str


@dataclass(frozen=True)
class SubtileObservation:
    roi_id: str
    subtile_id: str
    center_x_nm: float
    center_y_nm: float
    acquisition_sequence: int
    feature_values: np.ndarray
    mean_noise: np.ndarray
    channel_mean: np.ndarray
    pixel_count: int


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


def subtile_observations_from_revealed_roi(
    roi: pd.Series,
    dense_signal: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    acquisition_sequence: int,
    config: RunConfig,
    spec: SubtileSpec,
) -> list[SubtileObservation]:
    """Convert one revealed raster tile into mean or texture subtile observations."""

    sigma_px = config.variogram.residual_filter_sigma_px
    geometry = subtile_geometry_for_roi(roi, x, y, spec.grid_shape)
    observations = []
    for _, subtile in geometry.iterrows():
        values = dense_signal[
            :,
            int(subtile["row0"]) : int(subtile["row1"]),
            int(subtile["column0"]) : int(subtile["column1"]),
        ]
        feature_values, mean_noise = _feature_tensor(values, spec.feature_mode, sigma_px)
        observations.append(
            SubtileObservation(
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
