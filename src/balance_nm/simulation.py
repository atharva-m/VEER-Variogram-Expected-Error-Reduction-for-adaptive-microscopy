"""Parameterized interface specimens and a virtual SEM-EDS microscope."""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.special import expit

from .domain import MeasurementAction, ObservationBatch, RunConfig


def generate_hidden_sample(config: RunConfig, rng: np.random.Generator) -> xr.Dataset:
    """Generate a randomized U-Zr-O-style interface field hidden from policies."""

    scenario = config.scenario
    rows = scenario.grid_rows
    columns = scenario.grid_columns
    x = (np.arange(columns) + 0.5) * scenario.width_nm / columns
    y = (np.arange(rows) + 0.5) * scenario.height_nm / rows
    xx, yy = np.meshgrid(x, y)
    y_fraction = yy / scenario.height_nm - 0.5

    center = rng.uniform(*scenario.interface_center_fraction_range) * scenario.width_nm
    curvature = rng.uniform(*scenario.curvature_fraction_range) * scenario.width_nm
    slope = rng.uniform(*scenario.slope_fraction_range) * scenario.width_nm
    phase_shift = rng.uniform(0, 2 * np.pi)
    interface_x = center + slope * y_fraction + curvature * np.sin(
        2 * np.pi * yy / scenario.height_nm + phase_shift
    )
    layer_thickness = rng.uniform(*scenario.layer_thickness_nm_range)
    signed_distance = xx - interface_x
    clad_fraction = expit(signed_distance / max(layer_thickness / 4.0, 1.0))

    texture = gaussian_filter(
        rng.normal(size=(rows, columns)), sigma=max(max(rows, columns) / 14.0, 1.0)
    )
    texture /= max(float(np.std(texture)), 1e-8)
    texture_factor = np.clip(1 + scenario.texture_amplitude * texture, 0.6, 1.4)
    total_rate = rng.uniform(*scenario.rate_range) * texture_factor

    enrichment = rng.uniform(*scenario.oxygen_enrichment_range) * np.exp(
        -(signed_distance / max(layer_thickness, 1.0)) ** 2
    )
    rates = []
    for element in scenario.elements:
        fraction = (
            scenario.fuel_composition[element] * (1 - clad_fraction)
            + scenario.cladding_composition[element] * clad_fraction
        )
        if element == "O":
            fraction = fraction + enrichment
        rates.append(total_rate * fraction)
    true_rate = np.stack(rates)
    inclusion_mask = np.zeros_like(clad_fraction, dtype=bool)
    anomaly_mask = np.zeros_like(clad_fraction, dtype=bool)
    if scenario.multiobjective_patterns:
        inclusion_count = int(rng.integers(scenario.inclusion_count_range[0], scenario.inclusion_count_range[1] + 1))
        for _ in range(inclusion_count):
            center_x = rng.uniform(0.15, 0.85) * scenario.width_nm
            center_y = rng.uniform(0.10, 0.90) * scenario.height_nm
            radius = rng.uniform(*scenario.inclusion_radius_nm_range)
            region = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
            inclusion_mask |= region
        if np.any(inclusion_mask):
            target = 1 if len(scenario.elements) > 1 else 0
            true_rate[target, inclusion_mask] *= 2.2
            true_rate[0, inclusion_mask] *= 0.55
        anomaly_x = rng.uniform(0.2, 0.8) * scenario.width_nm
        anomaly_y = rng.uniform(0.2, 0.8) * scenario.height_nm
        anomaly_radius = rng.uniform(*scenario.inclusion_radius_nm_range)
        anomaly_mask = (xx - anomaly_x) ** 2 + (yy - anomaly_y) ** 2 <= anomaly_radius**2
        true_rate[-1, anomaly_mask] *= 1.0 + rng.uniform(*scenario.anomaly_amplitude_range)
    true_rate /= np.maximum(true_rate.sum(axis=0, keepdims=True), 1e-10) / total_rate

    true_phase = clad_fraction >= 0.5
    true_interface = np.zeros_like(true_phase, dtype=bool)
    closest_x_index = np.argmin(np.abs(x[np.newaxis, :] - interface_x[:, :1]), axis=1)
    true_interface[np.arange(rows), closest_x_index] = True
    true_composition = true_rate / np.maximum(true_rate.sum(axis=0, keepdims=True), 1e-10)
    gradient_score = np.zeros_like(clad_fraction)
    for channel in true_composition:
        gradient_y, gradient_x = np.gradient(channel)
        gradient_score += np.hypot(gradient_x, gradient_y)
    segregation_mask = enrichment > np.percentile(enrichment, 75)
    gradient_mask = gradient_score >= np.percentile(gradient_score, 90)
    truth_by_objective = np.stack(
        [
            true_interface,
            gradient_mask,
            segregation_mask,
            inclusion_mask,
            true_interface,
            anomaly_mask,
        ]
    )
    objectives = ["interface", "gradient", "segregation", "inclusion", "clustering", "anomaly"]

    return xr.Dataset(
        data_vars={
            "true_rate": (("element", "y", "x"), true_rate),
            "true_phase": (("y", "x"), true_phase),
            "true_interface": (("y", "x"), true_interface),
            "signed_distance_nm": (("y", "x"), signed_distance),
            "true_interface_x_nm": (("y",), interface_x[:, 0]),
            "true_pattern_mask": (("objective", "y", "x"), truth_by_objective),
        },
        coords={"element": scenario.elements, "objective": objectives, "x": x, "y": y},
        attrs={"layer_thickness_nm": float(layer_thickness)},
    )


class VirtualMicroscope:
    """Produces observable Poisson SEM-EDS count measurements from hidden truth."""

    def __init__(self, config: RunConfig):
        self.config = config

    def initial_action(self) -> MeasurementAction:
        scenario = self.config.scenario
        instrument = self.config.instrument
        action = MeasurementAction(
            action_id="coarse_initial",
            action_type="coarse_initial",
            bounds_nm=(0.0, scenario.width_nm, 0.0, scenario.height_nm),
            step_size_nm=instrument.coarse_step_nm,
            y_step_size_nm=instrument.coarse_y_step_nm,
            dwell_time_ms=instrument.coarse_dwell_ms,
        )
        return self.with_cost(action)

    def action_positions(self, action: MeasurementAction) -> tuple[np.ndarray, np.ndarray]:
        if action.bounds_nm is None or action.step_size_nm is None:
            return np.array([]), np.array([])
        x0, x1, y0, y1 = action.bounds_nm
        step_x = action.step_size_nm
        step_y = action.y_step_size_nm or step_x
        xs = np.arange(x0 + step_x / 2.0, x1, step_x)
        ys = np.arange(y0 + step_y / 2.0, y1, step_y)
        return xs, ys

    def with_cost(self, action: MeasurementAction) -> MeasurementAction:
        if action.action_type == "stop":
            return action
        xs, ys = self.action_positions(action)
        instrument = self.config.instrument
        pixels = int(xs.size * ys.size)
        elapsed_ms = (
            instrument.action_overhead_ms
            + instrument.line_overhead_ms * ys.size
            + pixels * (float(action.dwell_time_ms) + instrument.pixel_overhead_ms)
        )
        dose = instrument.dose_coefficient * pixels * float(action.dwell_time_ms)
        return action.model_copy(
            update={
                "pixel_count": pixels,
                "estimated_time_s": elapsed_ms / 1000.0,
                "estimated_dose": dose,
            }
        )

    def acquire(
        self,
        hidden_sample: xr.Dataset,
        action: MeasurementAction,
        rng: np.random.Generator,
    ) -> ObservationBatch:
        if action.action_type == "stop":
            raise ValueError("stop actions cannot be acquired")
        action = self.with_cost(action)
        xs, ys = self.action_positions(action)
        if not xs.size or not ys.size:
            raise ValueError("measurement action produces no pixels")
        sample_x, sample_y = np.meshgrid(xs, ys)
        grid_x = hidden_sample.coords["x"].values
        grid_y = hidden_sample.coords["y"].values
        dx = float(np.mean(np.diff(grid_x)))
        dy = float(np.mean(np.diff(grid_y)))
        query = np.vstack(
            [
                (sample_y.ravel() - grid_y[0]) / dy,
                (sample_x.ravel() - grid_x[0]) / dx,
            ]
        )
        sigma_nm = self.config.instrument.psf_width_by_step_nm[int(action.step_size_nm)]
        sigma_pixels = (sigma_nm / dy, sigma_nm / dx)
        counts_by_element = []
        for element in self.config.scenario.elements:
            latent = hidden_sample["true_rate"].sel(element=element).values
            blurred = gaussian_filter(latent, sigma=sigma_pixels, mode="nearest")
            sampled_rate = map_coordinates(blurred, query, order=1, mode="nearest")
            sensitivity = self.config.instrument.sensitivity[element]
            background = self.config.instrument.background_rate[element]
            expected = float(action.dwell_time_ms) * (
                sensitivity * sampled_rate + background
            )
            counts_by_element.append(rng.poisson(np.maximum(expected, 1e-8)))
        counts = np.stack(counts_by_element, axis=1)
        observations = counts.shape[0]
        dataset = xr.Dataset(
            data_vars={"counts": (("observation", "element"), counts.astype(np.int64))},
            coords={
                "observation": np.arange(observations),
                "element": np.asarray(self.config.scenario.elements, dtype=str),
                "x_nm": ("observation", sample_x.ravel()),
                "y_nm": ("observation", sample_y.ravel()),
                "step_nm": ("observation", np.full(observations, action.step_size_nm)),
                "y_step_nm": (
                    "observation",
                    np.full(observations, action.y_step_size_nm or action.step_size_nm),
                ),
                "dwell_ms": ("observation", np.full(observations, action.dwell_time_ms)),
                "action_id": (
                    "observation",
                    np.full(observations, action.action_id, dtype="U64"),
                ),
                "action_type": (
                    "observation",
                    np.full(observations, action.action_type, dtype="U32"),
                ),
                "acquisition_id": (
                    "observation",
                    np.full(observations, action.action_id, dtype="U64"),
                ),
            },
        )
        dataset["valid_observation"] = ("observation", np.ones(observations, dtype=bool))
        return ObservationBatch(action=action, data=dataset)
