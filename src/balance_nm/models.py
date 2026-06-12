"""Count-aware independent Gaussian Process surrogate models."""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from .domain import MeasurementAction, RunConfig


class CountAwareIndependentGP:
    """Independent per-element GPs with dwell-derived observation variance."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.models: dict[str, GaussianProcessRegressor] = {}
        self.target_offsets: dict[str, float] = {}
        self.target_scales: dict[str, float] = {}
        self._last_prediction: xr.Dataset | None = None

    def _aleatoric_variance(
        self, element: str, rate: np.ndarray, dwell: np.ndarray | float
    ) -> np.ndarray:
        if (
            self.config.dataset.mode == "replay"
            and self.config.dataset.value_semantics == "intensity_proxy"
        ):
            return np.full_like(
                rate, self.config.dataset.intensity_proxy_variance, dtype=float
            )
        sensitivity = self.config.instrument.sensitivity[element]
        background = self.config.instrument.background_rate[element]
        return np.maximum(sensitivity * rate + background, 1e-8) / (
            np.asarray(dwell, dtype=float) * sensitivity**2
        )

    def _rate_from_counts(
        self, element: str, counts: np.ndarray, dwell: np.ndarray
    ) -> np.ndarray:
        sensitivity = self.config.instrument.sensitivity[element]
        if (
            self.config.dataset.mode == "replay"
            and self.config.dataset.value_semantics == "intensity_proxy"
        ):
            return np.maximum(counts / sensitivity, 0.0)
        background = self.config.instrument.background_rate[element]
        return np.maximum((counts / dwell - background) / sensitivity, 0.0)

    def _scaled_coordinates(self, x_nm: np.ndarray, y_nm: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                x_nm / self.config.scenario.width_nm,
                y_nm / self.config.scenario.height_nm,
            ]
        )

    def _training_indices(self, size: int) -> np.ndarray:
        limit = self.config.model.max_training_points
        if size <= limit:
            return np.arange(size)
        return np.linspace(0, size - 1, limit, dtype=int)

    def fit(self, observations: xr.Dataset) -> None:
        if observations is None or observations.sizes.get("observation", 0) == 0:
            raise ValueError("at least one observation is required to fit the surrogate")
        indices = self._training_indices(observations.sizes["observation"])
        x_train = self._scaled_coordinates(
            observations["x_nm"].values[indices], observations["y_nm"].values[indices]
        )
        dwell = observations["dwell_ms"].values[indices]
        self.models = {}
        self.target_offsets = {}
        self.target_scales = {}
        for element in self.config.scenario.elements:
            counts = observations["counts"].sel(element=element).values[indices].astype(float)
            rate = self._rate_from_counts(element, counts, dwell)
            observation_variance = self._aleatoric_variance(element, rate, dwell)
            offset = float(np.median(rate))
            scale = max(float(np.nanpercentile(rate, 95) - np.nanpercentile(rate, 5)), 1.0)
            normalized_rate = (rate - offset) / scale
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
                length_scale=self.config.model.length_scale_fraction,
                length_scale_bounds="fixed",
                nu=1.5,
            )
            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=np.maximum(observation_variance / scale**2, 1e-8),
                normalize_y=False,
                optimizer=None,
                random_state=0,
            )
            model.fit(x_train, normalized_rate)
            self.models[element] = model
            self.target_offsets[element] = offset
            self.target_scales[element] = scale

    def evaluation_grid(self) -> tuple[np.ndarray, np.ndarray]:
        scenario = self.config.scenario
        x = (np.arange(scenario.grid_columns) + 0.5) * scenario.width_nm / scenario.grid_columns
        y = (np.arange(scenario.grid_rows) + 0.5) * scenario.height_nm / scenario.grid_rows
        return x, y

    def predict(self, grid: xr.Dataset | None = None) -> xr.Dataset:
        if not self.models:
            raise ValueError("fit must be called before predict")
        if grid is None:
            x, y = self.evaluation_grid()
        else:
            x, y = grid.coords["x"].values, grid.coords["y"].values
        xx, yy = np.meshgrid(x, y)
        query = self._scaled_coordinates(xx.ravel(), yy.ravel())
        means = []
        variances = []
        for element in self.config.scenario.elements:
            mean, std = self.models[element].predict(query, return_std=True)
            scale = self.target_scales[element]
            offset = self.target_offsets[element]
            means.append(np.maximum(mean.reshape(yy.shape) * scale + offset, 0.0))
            variances.append(np.maximum(std.reshape(yy.shape) ** 2 * scale**2, 1e-12))
        mean_rate = np.stack(means)
        variance_rate = np.stack(variances)
        aleatoric_variance = []
        dwell = self.config.instrument.fine_dwell_ms
        for index, element in enumerate(self.config.scenario.elements):
            aleatoric_variance.append(
                self._aleatoric_variance(element, mean_rate[index], dwell)
            )
        aleatoric_variance = np.stack(aleatoric_variance)
        total = np.maximum(mean_rate.sum(axis=0), 1e-9)
        composition = mean_rate / total[np.newaxis, :, :]
        uncalibrated_features = (
            self.config.dataset.mode == "replay"
            and self.config.dataset.value_semantics in ("uncalibrated_counts", "intensity_proxy")
        )
        if len(self.config.scenario.elements) > 1 and not uncalibrated_features:
            feature_signal = composition
            feature_signal_basis = "composition_fraction"
        else:
            feature_signal = np.empty_like(mean_rate)
            for index in range(mean_rate.shape[0]):
                low = float(np.nanpercentile(mean_rate[index], 1))
                high = float(np.nanpercentile(mean_rate[index], 99))
                feature_signal[index] = np.clip(
                    (mean_rate[index] - low) / max(high - low, 1e-9), 0.0, 1.0
                )
            feature_signal_basis = (
                "normalized_uncalibrated_count_channels"
                if self.config.dataset.value_semantics == "uncalibrated_counts"
                else "normalized_rate"
            )
        fuel = np.array(
            [self.config.scenario.fuel_composition[element] for element in self.config.scenario.elements]
        )
        cladding = np.array(
            [
                self.config.scenario.cladding_composition[element]
                for element in self.config.scenario.elements
            ]
        )
        direction = cladding - fuel
        direction_norm = float(np.dot(direction, direction))
        if direction_norm <= 1e-12:
            phase_probability = np.zeros(mean_rate.shape[1:], dtype=float)
            interface_weight = np.zeros_like(phase_probability)
        else:
            midpoint = (fuel + cladding) / 2.0
            discriminant = np.einsum("eyx,e->yx", composition - midpoint[:, None, None], direction)
            discriminant /= direction_norm
            variance_discriminant = np.einsum(
                "eyx,e->yx", variance_rate / total[np.newaxis, :, :] ** 2, direction**2
            ) / (direction_norm**2)
            z_score = discriminant / np.sqrt(np.maximum(variance_discriminant, 1e-8))
            phase_probability = norm.cdf(z_score)
            interface_weight = 4.0 * phase_probability * (1.0 - phase_probability)
        prediction = xr.Dataset(
            data_vars={
                "mean_rate": (("element", "y", "x"), mean_rate),
                "variance_rate": (("element", "y", "x"), variance_rate),
                "epistemic_variance": (("element", "y", "x"), variance_rate),
                "aleatoric_variance": (("element", "y", "x"), aleatoric_variance),
                "predictive_variance": (
                    ("element", "y", "x"),
                    variance_rate + aleatoric_variance,
                ),
                "composition": (("element", "y", "x"), composition),
                "feature_signal": (("element", "y", "x"), feature_signal),
                "phase_probability": (("y", "x"), phase_probability),
                "interface_weight": (("y", "x"), interface_weight),
            },
            coords={"element": self.config.scenario.elements, "x": x, "y": y},
            attrs={
                "feature_signal_basis": feature_signal_basis,
                "composition_basis": (
                    "uncalibrated_channel_fraction_not_quantitative_composition"
                    if uncalibrated_features
                    else "composition_fraction"
                ),
            },
        )
        self._last_prediction = prediction
        return prediction

    def expected_variance_reduction(
        self, action: MeasurementAction, grid: xr.Dataset | None = None
    ) -> xr.DataArray:
        """Approximate local variance reduction before measuring a candidate tile."""

        prediction = grid if grid is not None else self._last_prediction
        if prediction is None:
            raise ValueError("prediction is required before scoring actions")
        if action.bounds_nm is None or action.dwell_time_ms is None:
            return xr.zeros_like(prediction["variance_rate"])
        x0, x1, y0, y1 = action.bounds_nm
        x = prediction.coords["x"].values
        y = prediction.coords["y"].values
        region = (
            (x[np.newaxis, :] >= x0)
            & (x[np.newaxis, :] < x1)
            & (y[:, np.newaxis] >= y0)
            & (y[:, np.newaxis] < y1)
        )
        reductions = []
        for element in self.config.scenario.elements:
            mean = prediction["mean_rate"].sel(element=element).values
            variance = prediction["variance_rate"].sel(element=element).values
            observation_variance = self._aleatoric_variance(
                element, mean, float(action.dwell_time_ms)
            )
            local_reduction = variance**2 / np.maximum(
                variance + observation_variance, 1e-12
            )
            reductions.append(np.where(region, local_reduction, 0.0))
        return xr.DataArray(
            np.stack(reductions),
            dims=("element", "y", "x"),
            coords={
                "element": self.config.scenario.elements,
                "x": prediction.coords["x"],
                "y": prediction.coords["y"],
            },
            name="expected_variance_reduction",
        )
