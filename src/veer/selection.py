"""Front-weighted variogram expected-error-reduction (VEER) selection."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import distance_transform_edt

from .domain import RunConfig
from .morphology import front_from_probability
from .replay import raster_cost
from .variogram import (
    NestedVariogramFit,
    VariogramPosterior,
    gamma_nested,
    matern_3_2_correlation,
)

VEER_POLICY_PATTERN = re.compile(r"^variogram_eer_4x4_mean_kappa(?P<kappa>\d+)$")
GATED_POLICY_PATTERN = re.compile(r"^gated_veer_4x4_mean_kappa(?P<kappa>\d+)$")
NESTED_POLICY_PATTERN = re.compile(r"^nested_veer_4x4_mean_kappa(?P<kappa>\d+)$")
NESTED_BAND_POLICY_PATTERN = re.compile(r"^nested_band_veer_4x4_mean_kappa(?P<kappa>\d+)$")


@dataclass(frozen=True)
class VeerPolicySpec:
    policy: str
    front_kappa: float
    gated: bool = False


@dataclass(frozen=True)
class NestedPolicySpec:
    policy: str
    front_kappa: float
    weight_mode: str = "probability"


def parse_veer_policy(policy: str) -> VeerPolicySpec:
    """Parse a v5 VEER policy name into its front-weighting strength."""

    match = VEER_POLICY_PATTERN.match(policy)
    if match is not None:
        return VeerPolicySpec(policy=policy, front_kappa=float(match.group("kappa")))
    match = GATED_POLICY_PATTERN.match(policy)
    if match is not None:
        return VeerPolicySpec(
            policy=policy, front_kappa=float(match.group("kappa")), gated=True
        )
    raise ValueError(f"unsupported v5 VEER policy: {policy}")


def predicted_depth_profile(
    prediction: xr.Dataset,
    x: np.ndarray,
    y: np.ndarray,
    config: RunConfig,
) -> np.ndarray:
    """Per-row predicted penetration depth from the current reconstruction."""

    _, depth = front_from_probability(
        prediction["altered_region_probability"].values, x, y, config
    )
    return depth


def front_movement_fraction(
    previous_depth: np.ndarray,
    current_depth: np.ndarray,
    width_nm: float,
) -> float:
    """Mean per-row movement of the predicted front, as a fraction of slice width.

    Rows where the front appears or disappears between reveals count as a full
    slice-width move; rows absent in both count as zero.
    """

    previous = np.asarray(previous_depth, dtype=float)
    current = np.asarray(current_depth, dtype=float)
    width = max(width_nm, 1.0)
    both = np.isfinite(previous) & np.isfinite(current)
    flipped = np.isfinite(previous) != np.isfinite(current)
    values = np.zeros(previous.shape, dtype=float)
    values[both] = np.abs(previous[both] - current[both])
    values[flipped] = width
    return float(np.mean(values) / width)


def parse_nested_policy(policy: str) -> NestedPolicySpec:
    """Parse a v5.1 nested-variogram policy name into kappa and weight mode."""

    match = NESTED_POLICY_PATTERN.match(policy)
    if match is not None:
        return NestedPolicySpec(
            policy=policy, front_kappa=float(match.group("kappa")), weight_mode="probability"
        )
    match = NESTED_BAND_POLICY_PATTERN.match(policy)
    if match is not None:
        return NestedPolicySpec(
            policy=policy, front_kappa=float(match.group("kappa")), weight_mode="band"
        )
    raise ValueError(f"unsupported v5.1 nested VEER policy: {policy}")


def _normalized_steps(x: np.ndarray, y: np.ndarray, config: RunConfig) -> tuple[float, float]:
    step_x = float(x[1] - x[0]) if len(x) > 1 else 1.0
    step_y = float(y[1] - y[0]) if len(y) > 1 else 1.0
    return (
        step_x / max(config.scenario.width_nm, 1.0),
        step_y / max(config.scenario.height_nm, 1.0),
    )


def kernel_distance_field(
    observed_mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    config: RunConfig,
    length_scales: tuple[float, float],
) -> np.ndarray:
    """Distance to the nearest observed pixel under one kernel's anisotropic metric."""

    step_x_norm, step_y_norm = _normalized_steps(x, y, config)
    sampling = (step_y_norm / length_scales[1], step_x_norm / length_scales[0])
    if not np.any(observed_mask):
        rows, columns = observed_mask.shape
        half_diagonal = float(np.hypot(rows * sampling[0], columns * sampling[1])) / 2.0
        return np.full(observed_mask.shape, half_diagonal)
    return distance_transform_edt(~observed_mask, sampling=sampling)


def kernel_rectangle_distance(
    shape: tuple[int, int],
    roi: pd.Series,
    x: np.ndarray,
    y: np.ndarray,
    config: RunConfig,
    length_scales: tuple[float, float],
) -> np.ndarray:
    """Analytic distance to a candidate rectangle under one kernel's metric."""

    rows, columns = shape
    step_x_norm, step_y_norm = _normalized_steps(x, y, config)
    row0, row1 = int(roi["row0"]), int(roi["row1"])
    column0, column1 = int(roi["column0"]), int(roi["column1"])
    row_values = np.arange(rows)
    column_values = np.arange(columns)
    row_distance = np.maximum(np.maximum(row0 - row_values, 0), row_values - (row1 - 1))
    column_distance = np.maximum(
        np.maximum(column0 - column_values, 0), column_values - (column1 - 1)
    )
    row_scaled = row_distance * (step_y_norm / length_scales[1])
    column_scaled = column_distance * (step_x_norm / length_scales[0])
    return np.hypot(row_scaled[:, None], column_scaled[None, :])


def front_relevance_weights(
    prediction: xr.Dataset,
    x: np.ndarray,
    y: np.ndarray,
    config: RunConfig,
    front_kappa: float,
) -> np.ndarray:
    """Pixel weights emphasizing the band around the currently predicted front."""

    shape = (len(y), len(x))
    if front_kappa <= 0.0:
        return np.ones(shape)
    probability = prediction["altered_region_probability"].values
    front, _ = front_from_probability(probability, x, y, config)
    if not np.any(front):
        return np.ones(shape)
    step_x = float(x[1] - x[0]) if len(x) > 1 else 1.0
    step_y = float(y[1] - y[0]) if len(y) > 1 else 1.0
    distance_nm = distance_transform_edt(~front, sampling=(step_y, step_x))
    bandwidth = config.variogram.front_bandwidth_nm
    band = np.exp(-0.5 * (distance_nm / bandwidth) ** 2)
    return 1.0 + front_kappa * band


def front_probability_weights(
    prediction: xr.Dataset,
    front_kappa: float,
) -> np.ndarray:
    """Pixel weights from the uncertainty-inflated front-probability field.

    The prediction's `alteration_front_probability` is the normalized gradient
    of the altered-region probability, maxed with the nearest-observation
    reconstruction uncertainty (which grows linearly with gap depth). Weighting
    by it therefore concentrates on sharp predicted fronts while also growing
    with distance into unsampled voids, so the weights cannot lock onto a
    wrong early front estimate.
    """

    field = prediction["alteration_front_probability"].values.astype(float)
    if front_kappa <= 0.0:
        return np.ones_like(field)
    return 1.0 + front_kappa * np.clip(field, 0.0, 1.0)


def _cached_rectangle_distance(
    cache: dict | None,
    roi: pd.Series,
    shape: tuple[int, int],
    x: np.ndarray,
    y: np.ndarray,
    config: RunConfig,
    length_scales: tuple[float, float],
) -> np.ndarray:
    """Rectangle distances depend only on geometry, so memoize them per replay."""

    if cache is None:
        return kernel_rectangle_distance(shape, roi, x, y, config, length_scales)
    key = (str(roi["roi_id"]), length_scales)
    rectangle = cache.get(key)
    if rectangle is None:
        rectangle = kernel_rectangle_distance(shape, roi, x, y, config, length_scales)
        cache[key] = rectangle
    return rectangle


def _travel_costs(
    records: list[dict],
    raster_costs: np.ndarray,
    last_center: tuple[float, float] | None,
    travel_cost_ms_per_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage-travel distance (nm) and routing-aware total cost (s) per candidate.

    With travel_cost_ms_per_nm == 0 (the default) the total cost equals the
    dwell/dose raster cost exactly, so selection is unchanged. A positive value
    folds Euclidean stage travel from the last revealed tile into the cost,
    discouraging the policy from thrashing the stage across the specimen.
    """

    if last_center is None:
        travel = np.zeros(len(records), dtype=float)
    else:
        travel = np.asarray(
            [
                float(np.hypot(roi["center_x_nm"] - last_center[0], roi["center_y_nm"] - last_center[1]))
                for roi in records
            ],
            dtype=float,
        )
    total = raster_costs + travel_cost_ms_per_nm * travel / 1000.0
    return travel, total


def nested_veer_candidate_scores(
    catalog: pd.DataFrame,
    queried_ids: set[str],
    observed_mask: np.ndarray,
    fit: NestedVariogramFit,
    pixel_weights: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    config: RunConfig,
    rectangle_cache: dict | None = None,
    last_center: tuple[float, float] | None = None,
    travel_cost_ms_per_nm: float = 0.0,
) -> pd.DataFrame:
    """Score candidates by expected error reduction under the nested variogram.

    The linear component contributes `linear_slope * (D - min(D, R))`, which is
    exactly the deterministic coverage gain, so the uncertainty-lookahead
    baseline is the special case matern_amplitude = 0.
    """

    candidates = catalog[~catalog["roi_id"].isin(queried_ids)].copy()
    if candidates.empty:
        raise ValueError("no feasible v5.1 raster candidates remain")
    weight_total = max(float(np.sum(pixel_weights)), 1.0e-12)
    distance = kernel_distance_field(observed_mask, x, y, config, (1.0, 1.0))
    gamma_current = gamma_nested(distance, fit)
    records = candidates.to_dict("records")
    reductions = np.zeros(len(records), dtype=float)
    raster_costs = np.zeros(len(records), dtype=float)
    for position, roi in enumerate(records):
        rectangle = _cached_rectangle_distance(
            rectangle_cache, roi, observed_mask.shape, x, y, config, (1.0, 1.0)
        )
        raster_costs[position] = raster_cost(config, roi)[0]
        closer = rectangle < distance
        if not np.any(closer):
            continue
        gain = gamma_current[closer] - gamma_nested(rectangle[closer], fit)
        reductions[position] = float(np.sum(pixel_weights[closer] * gain))
    travel, total_cost = _travel_costs(records, raster_costs, last_center, travel_cost_ms_per_nm)
    candidates["expected_error_reduction"] = reductions / weight_total
    candidates["estimated_raster_cost_s"] = raster_costs
    candidates["travel_distance_nm"] = travel
    candidates["total_cost_s"] = total_cost
    candidates["eer_per_cost"] = candidates["expected_error_reduction"] / np.maximum(
        candidates["total_cost_s"], 1.0e-12
    )
    candidates["selection_utility"] = candidates["eer_per_cost"]
    candidates["nested_length_scale"] = fit.length_scale
    candidates["nested_matern_amplitude"] = fit.matern_amplitude
    candidates["nested_linear_slope"] = fit.linear_slope
    candidates["subtile_count"] = fit.subtile_count
    return candidates


def nested_veer_select_candidate(
    catalog: pd.DataFrame,
    queried_ids: set[str],
    observed_mask: np.ndarray,
    fit: NestedVariogramFit,
    pixel_weights: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    config: RunConfig,
    rectangle_cache: dict | None = None,
    last_center: tuple[float, float] | None = None,
    travel_cost_ms_per_nm: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame]:
    scored = nested_veer_candidate_scores(
        catalog, queried_ids, observed_mask, fit, pixel_weights, x, y, config,
        rectangle_cache, last_center, travel_cost_ms_per_nm,
    )
    selected = scored.sort_values(
        ["selection_utility", "row0", "column0"],
        ascending=[False, True, True],
    ).iloc[0]
    scored["selected"] = scored["roi_id"] == str(selected["roi_id"])
    return selected, scored


def veer_candidate_scores(
    catalog: pd.DataFrame,
    queried_ids: set[str],
    observed_mask: np.ndarray,
    posterior: VariogramPosterior,
    pixel_weights: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    config: RunConfig,
    rectangle_cache: dict | None = None,
    last_center: tuple[float, float] | None = None,
    travel_cost_ms_per_nm: float = 0.0,
) -> pd.DataFrame:
    """Score candidates by model-averaged front-weighted expected error reduction."""

    candidates = catalog[~catalog["roi_id"].isin(queried_ids)].copy()
    if candidates.empty:
        raise ValueError("no feasible v5 raster candidates remain")
    weight_total = max(float(np.sum(pixel_weights)), 1.0e-12)
    records = candidates.to_dict("records")
    reductions = np.zeros(len(records), dtype=float)
    for weight, length_scales in zip(posterior.weights, posterior.kernel_length_scales):
        distance = kernel_distance_field(observed_mask, x, y, config, length_scales)
        current_correlation = matern_3_2_correlation(distance)
        for position, roi in enumerate(records):
            rectangle = _cached_rectangle_distance(
                rectangle_cache, roi, observed_mask.shape, x, y, config, length_scales
            )
            closer = rectangle < distance
            if not np.any(closer):
                continue
            gain = matern_3_2_correlation(rectangle[closer]) - current_correlation[closer]
            reductions[position] += weight * float(
                np.sum(pixel_weights[closer] * gain)
            )
    raster_costs = np.asarray([raster_cost(config, roi)[0] for roi in records], dtype=float)
    travel, total_cost = _travel_costs(records, raster_costs, last_center, travel_cost_ms_per_nm)
    candidates["expected_error_reduction"] = (
        posterior.sill * reductions / weight_total
    )
    candidates["estimated_raster_cost_s"] = raster_costs
    candidates["travel_distance_nm"] = travel
    candidates["total_cost_s"] = total_cost
    candidates["eer_per_cost"] = candidates["expected_error_reduction"] / np.maximum(
        candidates["total_cost_s"], 1.0e-12
    )
    candidates["selection_utility"] = candidates["eer_per_cost"]
    candidates["variogram_sill"] = posterior.sill
    candidates["variogram_temper"] = posterior.temper
    candidates["subtile_count"] = posterior.subtile_count
    return candidates


def veer_select_candidate(
    catalog: pd.DataFrame,
    queried_ids: set[str],
    observed_mask: np.ndarray,
    posterior: VariogramPosterior,
    pixel_weights: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    config: RunConfig,
    rectangle_cache: dict | None = None,
    last_center: tuple[float, float] | None = None,
    travel_cost_ms_per_nm: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame]:
    scored = veer_candidate_scores(
        catalog, queried_ids, observed_mask, posterior, pixel_weights, x, y, config,
        rectangle_cache, last_center, travel_cost_ms_per_nm,
    )
    selected = scored.sort_values(
        ["selection_utility", "row0", "column0"],
        ascending=[False, True, True],
    ).iloc[0]
    scored["selected"] = scored["roi_id"] == str(selected["roi_id"])
    return selected, scored


def nearest_neighbor_route(
    centers: list[tuple[float, float]], start: tuple[float, float] | None
) -> list[int]:
    """Greedy nearest-neighbor visiting order through `centers` from `start`."""

    remaining = list(range(len(centers)))
    order: list[int] = []
    current = start
    while remaining:
        if current is None:
            nxt = remaining[0]
        else:
            nxt = min(
                remaining,
                key=lambda i: (centers[i][0] - current[0]) ** 2 + (centers[i][1] - current[1]) ** 2,
            )
        order.append(nxt)
        remaining.remove(nxt)
        current = centers[nxt]
    return order


def plan_fantasized_batch(
    score_fn,
    observed_mask: np.ndarray,
    queried_ids: set[str],
    catalog_size: int,
    batch_size: int,
    last_center: tuple[float, float] | None,
) -> list[tuple[pd.Series, pd.DataFrame]]:
    """Select a non-redundant batch by the Kriging-believer heuristic, then route it.

    Because the variogram expected-error reduction depends only on observation
    *coordinates* (not measured values), each provisional pick can be folded into
    a working mask and the batch re-scored without revealing any data. The chosen
    tiles are then ordered by a nearest-neighbor route to minimize stage travel.
    """

    working = observed_mask.copy()
    working_queried = set(queried_ids)
    picked: list[tuple[pd.Series, pd.DataFrame]] = []
    for _ in range(batch_size):
        if len(working_queried) >= catalog_size:
            break
        scored = score_fn(working, working_queried)
        selected = scored.sort_values(
            ["selection_utility", "row0", "column0"],
            ascending=[False, True, True],
        ).iloc[0]
        scored = scored.copy()
        scored["selected"] = scored["roi_id"] == str(selected["roi_id"])
        picked.append((selected, scored))
        working[
            int(selected["row0"]) : int(selected["row1"]),
            int(selected["column0"]) : int(selected["column1"]),
        ] = True
        working_queried.add(str(selected["roi_id"]))
    centers = [(float(s["center_x_nm"]), float(s["center_y_nm"])) for s, _ in picked]
    order = nearest_neighbor_route(centers, last_center)
    return [picked[i] for i in order]
