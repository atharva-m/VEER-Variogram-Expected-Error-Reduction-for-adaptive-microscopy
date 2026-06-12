"""Data-quality maps used to avoid invalid or low-value recommendations."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .domain import RunConfig


class QualityAnalyzer:
    def __init__(self, config: RunConfig):
        self.config = config

    def compute(
        self,
        prediction: xr.Dataset,
        observations: xr.Dataset | None = None,
        known_validity_source: xr.Dataset | None = None,
    ) -> xr.Dataset:
        dwell = self.config.instrument.fine_dwell_ms
        signal = np.zeros(prediction["mean_rate"].shape[1:], dtype=float)
        background = np.zeros_like(signal)
        for element in self.config.scenario.elements:
            signal += (
                prediction["mean_rate"].sel(element=element).values
                * self.config.instrument.sensitivity[element]
                * dwell
            )
            background += self.config.instrument.background_rate[element] * dwell
        signal_to_background = signal / np.maximum(background, 1e-12)
        relative_count_error = 1.0 / np.sqrt(np.maximum(signal + background, 1.0))
        quality_score = np.clip(
            signal_to_background / (signal_to_background + 1.0)
            * (1.0 - relative_count_error),
            0.0,
            1.0,
        )
        invalid = (
            (signal_to_background < self.config.quality.minimum_signal_to_background)
            | (relative_count_error > self.config.quality.maximum_relative_count_error)
        )
        artifact = np.zeros_like(invalid)
        validity = known_validity_source if known_validity_source is not None else observations
        if validity is not None and "valid_observation" in validity:
            bad = ~validity["valid_observation"].values.astype(bool)
            x = prediction.coords["x"].values
            y = prediction.coords["y"].values
            for x_nm, y_nm in zip(validity["x_nm"].values[bad], validity["y_nm"].values[bad]):
                x_index = int(np.argmin(np.abs(x - x_nm)))
                y_index = int(np.argmin(np.abs(y - y_nm)))
                artifact[y_index, x_index] = True
            invalid |= artifact
        quality_score[invalid] = 0.0
        coords = {"x": prediction.coords["x"], "y": prediction.coords["y"]}
        return xr.Dataset(
            {
                "signal_to_background": (("y", "x"), signal_to_background),
                "relative_count_error": (("y", "x"), relative_count_error),
                "artifact_flag": (("y", "x"), artifact),
                "invalid_region": (("y", "x"), invalid),
                "quality_score": (("y", "x"), quality_score),
            },
            coords=coords,
        )
