"""Interpretable multi-element pattern-interest and uncertainty maps."""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from .domain import RunConfig


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values, 0.0)
    scale = float(np.nanpercentile(values, 99))
    return np.clip(values / max(scale, 1e-12), 0.0, 1.0)


class PatternAnalyzer:
    def __init__(self, config: RunConfig):
        self.config = config

    def analyze(self, prediction: xr.Dataset, quality: xr.Dataset) -> xr.Dataset:
        features = prediction["feature_signal"].values
        epistemic = prediction["epistemic_variance"].sum("element").values
        q = quality["quality_score"].values
        interest = []
        probability = []
        uncertainty = []
        for objective in self.config.objectives.enabled:
            values = self._detector(objective, features, prediction)
            values = _normalize(values)
            threshold = self.config.objectives.thresholds.get(objective, 0.5)
            interest.append(values * (0.25 + 0.75 * q))
            probability.append(np.clip(values, 0.0, 1.0))
            uncertainty.append(_normalize(epistemic) * np.maximum(values, threshold / 4.0))
        objectives = self.config.objectives.enabled
        coords = {"objective": objectives, "x": prediction.coords["x"], "y": prediction.coords["y"]}
        return xr.Dataset(
            {
                "pattern_interest": (("objective", "y", "x"), np.stack(interest)),
                "pattern_probability": (("objective", "y", "x"), np.stack(probability)),
                "pattern_uncertainty": (("objective", "y", "x"), np.stack(uncertainty)),
            },
            coords=coords,
        )

    def _detector(
        self, objective: str, features: np.ndarray, prediction: xr.Dataset
    ) -> np.ndarray:
        if objective == "interface":
            return prediction["interface_weight"].values
        if objective == "gradient":
            score = np.zeros(features.shape[1:])
            for channel in features:
                dy, dx = np.gradient(channel)
                score += np.hypot(dx, dy)
            return score
        local = np.stack([gaussian_filter(channel, sigma=2.0) for channel in features])
        residual = np.sqrt(np.sum((features - local) ** 2, axis=0))
        if objective == "segregation":
            element = self.config.scenario.segregation_element
            index = self.config.scenario.elements.index(element)
            return np.abs(features[index] - local[index]) * prediction["interface_weight"].values
        if objective == "inclusion":
            return gaussian_filter(residual, sigma=0.7)
        if objective == "anomaly":
            median = np.median(residual)
            mad = np.median(np.abs(residual - median)) + 1e-12
            return np.maximum((residual - median) / mad, 0.0)
        if objective == "clustering":
            points = features.reshape(features.shape[0], -1).T
            if np.unique(points, axis=0).shape[0] < 2:
                return np.zeros(features.shape[1:], dtype=float)
            mixture = GaussianMixture(n_components=2, random_state=0, covariance_type="full")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                membership = mixture.fit_predict(points)
            if np.unique(membership).size < 2:
                return np.zeros(features.shape[1:], dtype=float)
            probability = mixture.predict_proba(points).max(axis=1)
            labels = membership.reshape(features.shape[1:])
            boundary = np.zeros_like(labels, dtype=float)
            boundary[:, 1:] += labels[:, 1:] != labels[:, :-1]
            boundary[1:, :] += labels[1:, :] != labels[:-1, :]
            return boundary + (1.0 - probability.reshape(labels.shape))
        raise ValueError(f"unknown pattern objective: {objective}")
