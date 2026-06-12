"""Capability-gated retrospective measurement replay from dense SEM-EDS data."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .domain import MeasurementAction, ObservationBatch, ReplayCapabilities, RunConfig
from .simulation import VirtualMicroscope


class ReplayMicroscope(VirtualMicroscope):
    """Reveal only source observations supported by a dense acquired dataset."""

    def __init__(self, config: RunConfig, source: xr.Dataset, capabilities: ReplayCapabilities):
        super().__init__(config)
        self.source = source
        self.capabilities = capabilities
        self.revealed_source_indices: set[int] = set()

    def acquire(
        self,
        reference_sample: xr.Dataset,
        action: MeasurementAction,
        rng: np.random.Generator,
    ) -> ObservationBatch:
        if action.action_type == "stop":
            raise ValueError("stop actions cannot be acquired")
        action = self.with_cost(action)
        xs, ys = self.action_positions(action)
        if not xs.size or not ys.size:
            raise ValueError("replay measurement action produces no pixels")
        target_x, target_y = np.meshgrid(xs, ys)
        source_x = self.source["x_nm"].values
        source_y = self.source["y_nm"].values
        selected = []
        for x_nm, y_nm in zip(target_x.ravel(), target_y.ravel()):
            distance = (source_x - x_nm) ** 2 + (source_y - y_nm) ** 2
            selected.append(int(np.argmin(distance)))
        selected = np.asarray(list(dict.fromkeys(selected)), dtype=int)
        selected = np.asarray(
            [index for index in selected if index not in self.revealed_source_indices],
            dtype=int,
        )
        self.revealed_source_indices.update(selected.tolist())
        source_batch = self.source.isel(observation=selected).copy().load()
        original_dwell = source_batch["dwell_ms"].values.astype(float)
        requested_dwell = float(action.dwell_time_ms)
        if np.any(requested_dwell > original_dwell + 1e-9):
            raise ValueError("replay cannot create count statistics exceeding source acquisition")
        fraction = np.clip(requested_dwell / original_dwell, 0.0, 1.0)
        counts = source_batch["counts"].values.astype(np.int64)
        if np.any(fraction < 1.0):
            if self.source.attrs.get("value_semantics", "counts") == "intensity_proxy":
                raise ValueError("intensity-proxy replay cannot simulate reduced count statistics")
            counts = rng.binomial(counts, fraction[:, None])
        result = xr.Dataset(
            {"counts": (("observation", "element"), counts)},
            coords={
                "observation": np.arange(selected.size),
                "element": source_batch.coords["element"].values,
                "x_nm": ("observation", source_batch["x_nm"].values),
                "y_nm": ("observation", source_batch["y_nm"].values),
                "step_nm": ("observation", np.full(selected.size, action.step_size_nm)),
                "y_step_nm": (
                    "observation",
                    np.full(selected.size, action.y_step_size_nm or action.step_size_nm),
                ),
                "dwell_ms": ("observation", np.full(selected.size, action.dwell_time_ms)),
                "action_id": ("observation", np.full(selected.size, action.action_id, dtype="U64")),
                "action_type": ("observation", np.full(selected.size, action.action_type, dtype="U32")),
                "acquisition_id": ("observation", np.full(selected.size, action.action_id, dtype="U64")),
            },
        )
        result["valid_observation"] = (
            "observation",
            source_batch["valid_observation"].values.astype(bool),
        )
        return ObservationBatch(action=action, data=result)


def reference_from_dense_observations(config: RunConfig, observations: xr.Dataset) -> xr.Dataset:
    """Build evaluation-only dense-rate reference products for retrospective replay."""

    x = np.unique(observations["x_nm"].values)
    y = np.unique(observations["y_nm"].values)
    if x.size * y.size != observations.sizes["observation"]:
        raise ValueError("replay reference source must form a complete regular spatial grid")
    elements = config.scenario.elements
    true_rate = np.zeros((len(elements), y.size, x.size), dtype=float)
    for element_index, element in enumerate(elements):
        counts = observations["counts"].sel(element=element).values.astype(float)
        dwell = observations["dwell_ms"].values.astype(float)
        sensitivity = config.instrument.sensitivity[element]
        background = config.instrument.background_rate[element]
        if config.dataset.value_semantics == "intensity_proxy":
            rates = np.maximum(counts / sensitivity, 0.0)
        else:
            rates = np.maximum((counts / dwell - background) / sensitivity, 0.0)
        for rate, x_nm, y_nm in zip(rates, observations["x_nm"].values, observations["y_nm"].values):
            column = int(np.argmin(np.abs(x - x_nm)))
            row = int(np.argmin(np.abs(y - y_nm)))
            true_rate[element_index, row, column] = rate
    composition = true_rate / np.maximum(true_rate.sum(axis=0, keepdims=True), 1e-12)
    fuel = np.array([config.scenario.fuel_composition[element] for element in elements])
    cladding = np.array([config.scenario.cladding_composition[element] for element in elements])
    direction = cladding - fuel
    if float(np.dot(direction, direction)) <= 1e-12:
        phase = np.zeros((y.size, x.size), dtype=bool)
    else:
        midpoint = (fuel + cladding) / 2.0
        discriminant = np.einsum("eyx,e->yx", composition - midpoint[:, None, None], direction)
        phase = discriminant >= 0
    interface = np.zeros_like(phase, dtype=bool)
    interface_x = np.full(y.size, np.nan)
    for row in range(y.size):
        crossing = np.flatnonzero(np.diff(phase[row].astype(int)) != 0)
        if crossing.size:
            column = int(crossing[0])
            interface[row, column] = True
            interface_x[row] = (x[column] + x[column + 1]) / 2.0
    return xr.Dataset(
        {
            "true_rate": (("element", "y", "x"), true_rate),
            "true_phase": (("y", "x"), phase),
            "true_interface": (("y", "x"), interface),
            "true_interface_x_nm": (("y",), interface_x),
        },
        coords={
            "element": elements,
            "x": x,
            "y": y,
        },
        attrs={
            "reference_kind": "dense_replay_reference",
            "interface_kind": "derived_not_labeled"
            if float(np.dot(direction, direction)) > 1e-12
            else "not_configured",
            "value_semantics": observations.attrs.get("value_semantics", "counts"),
        },
    )
