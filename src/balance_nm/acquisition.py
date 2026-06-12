"""Preset-based safe acquisition and multi-objective recommendation policies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import xarray as xr

from .domain import (
    ConstraintViolation,
    ExperimentState,
    MeasurementAction,
    Recommendation,
    ReplayCapabilities,
    RunConfig,
)
from .models import CountAwareIndependentGP
from .simulation import VirtualMicroscope


def _region_mask(action: MeasurementAction, dataset: xr.Dataset) -> np.ndarray:
    x0, x1, y0, y1 = action.bounds_nm or (0.0, 0.0, 0.0, 0.0)
    x = dataset.coords["x"].values
    y = dataset.coords["y"].values
    return (
        (x[np.newaxis, :] >= x0)
        & (x[np.newaxis, :] < x1)
        & (y[:, np.newaxis] >= y0)
        & (y[:, np.newaxis] < y1)
    )


class ActionValidator:
    def __init__(
        self, config: RunConfig, capabilities: ReplayCapabilities | None = None
    ):
        self.config = config
        self.capabilities = capabilities

    def violations(
        self,
        action: MeasurementAction,
        state: ExperimentState,
        quality: xr.Dataset | None = None,
    ) -> list[ConstraintViolation]:
        if action.action_type == "stop":
            return []
        reasons: list[ConstraintViolation] = []
        scenario = self.config.scenario
        instrument = self.config.instrument

        def reject(reason: str, category: str = "constraint") -> None:
            reasons.append(ConstraintViolation(action_id=action.action_id, reason=reason, category=category))

        if action.bounds_nm is None:
            reject("missing bounds")
        else:
            x0, x1, y0, y1 = action.bounds_nm
            if x0 < 0 or y0 < 0 or x1 > scenario.width_nm or y1 > scenario.height_nm:
                reject("outside valid specimen coordinates")
            for region in self.config.safety.excluded_regions_nm:
                rx0, rx1, ry0, ry1 = region
                if x0 < rx1 and x1 > rx0 and y0 < ry1 and y1 > ry0:
                    reject("overlaps an excluded operating region")
        if action.step_size_nm not in instrument.allowed_step_sizes_nm:
            reject("step size is not allowed")
        if action.y_step_size_nm is not None and action.y_step_size_nm not in instrument.allowed_step_sizes_nm:
            reject("y step size is not allowed")
        if action.dwell_time_ms not in instrument.allowed_dwell_times_ms:
            reject("dwell time is not allowed")
        approved = self.config.safety.approved_preset_ids
        if approved and action.action_type != "coarse_initial" and action.preset_id not in approved:
            reject("action preset is not approved")
        if state.consumed_time_s + action.estimated_time_s > self.config.budget.max_scan_time_s:
            reject("scan-time budget exceeded")
        if state.consumed_dose + action.estimated_dose > self.config.budget.max_dose_proxy:
            reject("dose-proxy budget exceeded")
        if state.observation_count + action.pixel_count > self.config.model.max_observations:
            reject("maximum accumulated observation count exceeded")
        if quality is not None and self.config.quality.hard_exclude_invalid:
            region = _region_mask(action, quality)
            if region.any() and bool(quality["invalid_region"].values[region].any()):
                reject("overlaps a hard-invalid data-quality region", "quality")
        if self.config.dataset.mode == "replay" and self.capabilities and action.action_type != "coarse_initial":
            capability_reason = self._unsupported_reason(action)
            if capability_reason:
                reject(capability_reason, "capability")
        return reasons

    def _unsupported_reason(self, action: MeasurementAction) -> str | None:
        capability = self.capabilities
        if capability is None:
            return None
        if action.action_type not in capability.supported_action_types:
            return "action type is unsupported by replay dataset capabilities"
        if capability.native_step_nm is not None and float(action.step_size_nm) < capability.native_step_nm:
            return "requested spatial resolution is finer than available replay data"
        requested_y_step = float(action.y_step_size_nm or action.step_size_nm)
        if capability.native_y_step_nm is not None and requested_y_step < capability.native_y_step_nm:
            return "requested y spatial resolution is finer than available replay data"
        if capability.native_dwell_ms is not None and float(action.dwell_time_ms) > capability.native_dwell_ms:
            return "requested count statistics exceed available replay acquisition"
        if (
            capability.native_dwell_ms is not None
            and not capability.supports_statistical_thinning
            and not np.isclose(float(action.dwell_time_ms), capability.native_dwell_ms)
        ):
            return "replay source provides intensity proxies, not thinnable count statistics"
        return None


class CandidateGenerator:
    def __init__(
        self,
        config: RunConfig,
        microscope: VirtualMicroscope,
        capabilities: ReplayCapabilities | None = None,
    ):
        self.config = config
        self.microscope = microscope
        self.validator = ActionValidator(config, capabilities)
        self.last_unsupported: list[ConstraintViolation] = []
        self.last_unsupported_records: list[dict] = []
        self.last_rejected_records: list[dict] = []

    def _tile_bounds(
        self, width_nm: float, height_nm: float | None = None
    ) -> list[tuple[float, float, float, float]]:
        scenario = self.config.scenario
        height_nm = height_nm or width_nm
        bounds = []
        for y0 in np.arange(0.0, scenario.height_nm, height_nm):
            for x0 in np.arange(0.0, scenario.width_nm, width_nm):
                bounds.append(
                    (
                        float(x0),
                        float(min(x0 + width_nm, scenario.width_nm)),
                        float(y0),
                        float(min(y0 + height_nm, scenario.height_nm)),
                    )
                )
        return bounds

    def _point_regions(
        self, preset, prediction: xr.Dataset | None
    ) -> list[tuple[float, float, float, float]]:
        width = float(preset.roi_size_nm)
        height = float(preset.roi_height_nm or width)
        scenario = self.config.scenario
        if prediction is None or "pattern_interest" not in prediction:
            return [(scenario.width_nm / 2 - width / 2, scenario.width_nm / 2 + width / 2,
                     scenario.height_nm / 2 - height / 2, scenario.height_nm / 2 + height / 2)]
        values = prediction["pattern_interest"].max("objective").values.copy()
        regions = []
        x = prediction.coords["x"].values
        y = prediction.coords["y"].values
        for _ in range(min(8, values.size)):
            row, column = np.unravel_index(np.argmax(values), values.shape)
            center_x, center_y = float(x[column]), float(y[row])
            half_x = width / 2.0
            half_y = height / 2.0
            regions.append(
                (
                    max(0.0, center_x - half_x),
                    min(scenario.width_nm, center_x + half_x),
                    max(0.0, center_y - half_y),
                    min(scenario.height_nm, center_y + half_y),
                )
            )
            values[max(0, row - 2): row + 3, max(0, column - 2): column + 3] = -np.inf
        return regions

    def generate(
        self,
        state: ExperimentState,
        prediction: xr.Dataset | None = None,
        quality: xr.Dataset | None = None,
    ) -> tuple[list[MeasurementAction], list[ConstraintViolation]]:
        presets = self.config.resolved_action_presets()
        completed = {(action.preset_id, action.bounds_nm) for action in state.actions}
        legacy_fine_bounds = {
            action.bounds_nm for action in state.actions if action.action_type == "fine_tile"
        }
        candidates: list[MeasurementAction] = []
        for preset in presets:
            if not self.config.action_presets and preset.action_type == "repeat_tile":
                regions = sorted(legacy_fine_bounds)
            elif preset.action_type == "point_roi":
                regions = self._point_regions(preset, prediction)
            else:
                regions = self._tile_bounds(float(preset.tile_size_nm), preset.tile_height_nm)
            for index, bounds in enumerate(regions):
                if (preset.preset_id, bounds) in completed:
                    continue
                if not self.config.action_presets and preset.action_type == "fine_tile" and bounds in legacy_fine_bounds:
                    continue
                center = ((bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2)
                candidates.append(
                    MeasurementAction(
                        action_id=f"{preset.preset_id}_{index:04d}",
                        preset_id=preset.preset_id,
                        action_type=preset.action_type,
                        bounds_nm=bounds,
                        roi_center_nm=center if preset.action_type == "point_roi" else None,
                        roi_size_nm=preset.roi_size_nm,
                        step_size_nm=preset.step_size_nm,
                        y_step_size_nm=preset.y_step_size_nm,
                        dwell_time_ms=preset.dwell_time_ms,
                        spatial_resolution_proxy=preset.spatial_resolution_proxy,
                        count_statistics_proxy=preset.count_statistics_proxy,
                    )
                )
        feasible: list[MeasurementAction] = []
        rejected: list[ConstraintViolation] = []
        self.last_unsupported = []
        self.last_unsupported_records = []
        self.last_rejected_records = []
        for candidate in candidates:
            costed = self.microscope.with_cost(candidate)
            violations = self.validator.violations(costed, state, quality)
            capability = [item for item in violations if item.category == "capability"]
            ordinary = [item for item in violations if item.category != "capability"]
            if capability:
                self.last_unsupported.extend(capability)
                self.last_unsupported_records.append(
                    {
                        **costed.model_copy(
                            update={
                                "supported_in_current_dataset": False,
                                "constraint_status": "unsupported",
                            }
                        ).model_dump(mode="json"),
                        "violations": "; ".join(item.reason for item in capability),
                    }
                )
            elif ordinary:
                rejected.extend(ordinary)
                self.last_rejected_records.append(
                    {
                        **costed.model_copy(update={"constraint_status": "invalid"}).model_dump(mode="json"),
                        "violations": "; ".join(item.reason for item in ordinary),
                    }
                )
            else:
                feasible.append(costed.model_copy(update={"constraint_status": "valid"}))
        return feasible, rejected


class Policy(ABC):
    name: str

    def __init__(
        self,
        config: RunConfig,
        surrogate: CountAwareIndependentGP,
        rng: np.random.Generator | None = None,
    ):
        self.config = config
        self.surrogate = surrogate
        self.rng = rng or np.random.default_rng(0)

    def recommend(
        self,
        state: ExperimentState,
        prediction: xr.Dataset,
        candidates: list[MeasurementAction],
        rejected: list[ConstraintViolation] | None = None,
        unsupported: list[ConstraintViolation] | None = None,
    ) -> Recommendation:
        rejected = rejected or []
        unsupported = unsupported or []
        if not candidates:
            return self._stop(rejected, unsupported, "No feasible measurement action remains.")
        scored = []
        for candidate in candidates:
            utility, details, penalty = self.score_details(candidate, prediction)
            scored.append(
                (
                    candidate.model_copy(
                        update={
                            "objective_gain_by_type": details,
                            "quality_penalty": penalty,
                            "total_utility": utility,
                        }
                    ),
                    utility,
                )
            )
        scored.sort(key=lambda value: value[1], reverse=True)
        selected, utility = scored[0]
        if utility < self.config.acquisition.minimum_utility:
            return self._stop(rejected, unsupported, "All feasible action utilities fall below threshold.")
        alternatives = [
            {
                "action_id": action.action_id,
                "preset_id": action.preset_id,
                "action_type": action.action_type,
                "utility": float(score),
                "estimated_time_s": action.estimated_time_s,
                "objective_gain_by_type": action.objective_gain_by_type,
            }
            for action, score in scored[1:6]
        ]
        return Recommendation(
            action=selected,
            utility=float(utility),
            alternatives=alternatives,
            rejected=rejected,
            unsupported=unsupported,
            reasons=self.reasons(selected),
        )

    def _stop(
        self,
        rejected: list[ConstraintViolation],
        unsupported: list[ConstraintViolation],
        reason: str,
    ) -> Recommendation:
        return Recommendation(
            action=MeasurementAction(action_id="stop", action_type="stop"),
            utility=0.0,
            rejected=rejected,
            unsupported=unsupported,
            reasons=[reason],
        )

    def score_details(
        self, action: MeasurementAction, prediction: xr.Dataset
    ) -> tuple[float, dict[str, float], float]:
        return self.score(action, prediction), {}, 0.0

    def reasons(self, action: MeasurementAction) -> list[str]:
        return [f"Selected by {self.name} acquisition policy.", "Action passed operating constraints."]

    @abstractmethod
    def score(self, action: MeasurementAction, prediction: xr.Dataset) -> float:
        raise NotImplementedError


class UniformPolicy(Policy):
    name = "uniform"

    def score(self, action: MeasurementAction, prediction: xr.Dataset) -> float:
        type_order = {
            "fine_tile": 5,
            "raster_tile": 5,
            "high_resolution_tile": 4,
            "high_statistics_tile": 3,
            "point_roi": 2,
            "repeat_tile": 1,
        }
        index = int(action.action_id.rsplit("_", 1)[-1])
        return type_order.get(action.action_type, 0) - index / 10000.0


class RandomPolicy(Policy):
    name = "random"

    def score(self, action: MeasurementAction, prediction: xr.Dataset) -> float:
        return float(self.rng.random())


class GradientPolicy(Policy):
    name = "gradient"

    def score(self, action: MeasurementAction, prediction: xr.Dataset) -> float:
        features = prediction["feature_signal"].values
        gradient = np.zeros(features.shape[1:])
        for channel in features:
            dy, dx = np.gradient(channel)
            gradient += np.hypot(dx, dy)
        return float(gradient[_region_mask(action, prediction)].sum()) / max(action.estimated_time_s, 1e-12)

    def reasons(self, action: MeasurementAction) -> list[str]:
        return ["High predicted elemental-signal gradient per scan time.", "Action passed operating constraints."]


class UncertaintyPolicy(Policy):
    name = "uncertainty"

    def score(self, action: MeasurementAction, prediction: xr.Dataset) -> float:
        reduction = self.surrogate.expected_variance_reduction(action, prediction)
        return float(reduction.sum()) / max(action.estimated_time_s, 1e-12)

    def reasons(self, action: MeasurementAction) -> list[str]:
        return ["High expected posterior variance reduction per scan time.", "Action passed operating constraints."]


class BalancePolicy(Policy):
    name = "balance"

    def score_details(
        self, action: MeasurementAction, prediction: xr.Dataset
    ) -> tuple[float, dict[str, float], float]:
        reduction_by_element = self.surrogate.expected_variance_reduction(action, prediction)
        if (
            self.config.dataset.mode == "replay"
            and self.config.dataset.value_semantics in ("uncalibrated_counts", "intensity_proxy")
        ):
            scales = []
            for element in self.config.scenario.elements:
                signal = prediction["mean_rate"].sel(element=element).values
                spread = max(float(np.nanpercentile(signal, 95) - np.nanpercentile(signal, 5)), 1.0)
                scales.append(spread**2)
            scale = xr.DataArray(
                np.asarray(scales), dims=("element",), coords={"element": prediction.coords["element"]}
            )
            reduction = (reduction_by_element / scale).sum("element")
        else:
            reduction = reduction_by_element.sum("element")
        if "pattern_interest" in prediction:
            gains: dict[str, float] = {}
            weighted_gain = 0.0
            for objective, weight in self.config.objectives.weights.items():
                if weight <= 0 or objective not in prediction.coords["objective"].values:
                    continue
                interest = prediction["pattern_interest"].sel(objective=objective)
                gain = float((reduction * interest).sum())
                gains[objective] = gain
                weighted_gain += weight * gain
        else:
            gain = float((reduction * prediction["interface_weight"]).sum())
            gains = {"interface": gain}
            weighted_gain = gain
        region = _region_mask(action, prediction)
        quality_value = (
            float(prediction["quality_score"].values[region].mean())
            if "quality_score" in prediction and region.any()
            else 1.0
        )
        penalty = self.config.quality.penalty_strength * (1.0 - quality_value)
        utility = weighted_gain * quality_value / max(action.estimated_time_s, 1e-12) - penalty
        return utility, gains, penalty

    def score(self, action: MeasurementAction, prediction: xr.Dataset) -> float:
        return self.score_details(action, prediction)[0]

    def reasons(self, action: MeasurementAction) -> list[str]:
        objectives = ", ".join(action.objective_gain_by_type) or "interface"
        return [
            f"High quality-adjusted expected information gain for: {objectives}.",
            "Action passed approved-preset, operating-limit, and dataset-capability checks.",
        ]


def create_policy(
    name: str,
    config: RunConfig,
    surrogate: CountAwareIndependentGP,
    rng: np.random.Generator,
) -> Policy:
    policies = {
        "uniform": UniformPolicy,
        "random": RandomPolicy,
        "gradient": GradientPolicy,
        "uncertainty": UncertaintyPolicy,
        "balance": BalancePolicy,
    }
    try:
        return policies[name](config, surrogate, rng)
    except KeyError as exc:
        raise ValueError(f"unknown acquisition policy: {name}") from exc
