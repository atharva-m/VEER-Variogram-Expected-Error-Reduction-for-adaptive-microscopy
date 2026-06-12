"""Adaptive spatial graph refinement for v3 corrosion-morphology replay."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import Delaunay, QhullError
from scipy.stats import t
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from .data import ingest_dataset
from .domain import RunConfig
from .io import load_config, write_config
from .v3_morphology import (
    dense_signal_from_observations,
    pseudo_reference_from_dense_signal,
    reconstruct_from_observed_mask,
)
from .v3_validation import (
    _neighbor_candidates,
    _normalize_map,
    _raster_cost,
    _score_prediction,
    build_v3_roi_catalog,
    sources_from_manifest,
    summarize_v3_slice_blocks,
    summarize_v3_stratified_metrics,
    v3_config_for_slice,
)

GraphPolicyName = Literal["uncertainty", "graph_gap", "graph_uncertainty", "graph_hybrid"]
GraphProviderName = Literal["distance", "roi_gp"]
GraphConfirmationMode = Literal["disabled", "one_anchor"]


@dataclass(frozen=True)
class GraphArm:
    policy: GraphPolicyName
    provider: GraphProviderName
    confirmation_mode: GraphConfirmationMode

    @property
    def arm_id(self) -> str:
        return f"{self.policy}__{self.provider}__{self.confirmation_mode}"


@dataclass
class GraphNode:
    roi_id: str
    center_x_nm: float
    center_y_nm: float
    acquisition_sequence: int
    acquisition_stage: str
    channel_mean: np.ndarray
    channel_variance: np.ndarray
    pixel_count: int
    parent_edge_id: str | None = None


@dataclass
class GraphSliceResult:
    arm: GraphArm
    metrics: pd.DataFrame
    node_trace: pd.DataFrame
    edge_trace: pd.DataFrame
    fallback_events: pd.DataFrame


def default_graph_arms(config: RunConfig) -> list[GraphArm]:
    policies: list[GraphPolicyName] = ["uncertainty", *config.acquisition_v3.graph.policies]
    return [
        GraphArm(policy, provider, confirmation)
        for policy in policies
        for provider in config.acquisition_v3.graph.uncertainty_providers
        for confirmation in config.acquisition_v3.graph.confirmation_modes
    ]


def _complete_edges(nodes: list[GraphNode]) -> list[tuple[int, int]]:
    return list(combinations(range(len(nodes)), 2))


def active_spatial_edges(nodes: list[GraphNode]) -> tuple[list[tuple[int, int]], str, bool]:
    """Return deterministic Delaunay edges or a complete-graph fallback."""

    if len(nodes) < 3:
        return _complete_edges(nodes), "complete", True
    points = np.asarray([[node.center_x_nm, node.center_y_nm] for node in nodes], dtype=float)
    if np.linalg.matrix_rank(points - points[0]) < 2:
        return _complete_edges(nodes), "complete", True
    try:
        triangulation = Delaunay(points)
    except QhullError:
        return _complete_edges(nodes), "complete", True
    edges: set[tuple[int, int]] = set()
    for simplex in triangulation.simplices:
        for first, second in combinations(sorted(int(value) for value in simplex), 2):
            edges.add((first, second))
    return sorted(edges), "delaunay", False


def _node_from_roi(
    roi: pd.Series,
    dense_signal: np.ndarray,
    acquisition_sequence: int,
    acquisition_stage: str,
    parent_edge_id: str | None = None,
) -> GraphNode:
    row0, row1 = int(roi["row0"]), int(roi["row1"])
    column0, column1 = int(roi["column0"]), int(roi["column1"])
    revealed = dense_signal[:, row0:row1, column0:column1]
    return GraphNode(
        roi_id=str(roi["roi_id"]),
        center_x_nm=float(roi["center_x_nm"]),
        center_y_nm=float(roi["center_y_nm"]),
        acquisition_sequence=acquisition_sequence,
        acquisition_stage=acquisition_stage,
        channel_mean=np.nanmean(revealed, axis=(1, 2)),
        channel_variance=np.nanvar(revealed, axis=(1, 2)),
        pixel_count=int(roi["pixel_count"]),
        parent_edge_id=parent_edge_id,
    )


def _node_record(
    node: GraphNode,
    arm: GraphArm,
    slice_id: str,
    channels: list[str],
) -> dict[str, object]:
    return {
        "slice": slice_id,
        "arm_id": arm.arm_id,
        "policy": arm.policy,
        "provider": arm.provider,
        "confirmation_mode": arm.confirmation_mode,
        "roi_id": node.roi_id,
        "center_x_nm": node.center_x_nm,
        "center_y_nm": node.center_y_nm,
        "acquisition_sequence": node.acquisition_sequence,
        "acquisition_stage": node.acquisition_stage,
        "channel_mean_json": json.dumps(dict(zip(channels, node.channel_mean.tolist()))),
        "channel_variance_json": json.dumps(dict(zip(channels, node.channel_variance.tolist()))),
        "pixel_count": node.pixel_count,
        "parent_edge_id": node.parent_edge_id,
    }


def _roi_mean(values: np.ndarray, roi: pd.Series) -> float:
    return float(
        np.mean(
            values[
                int(roi["row0"]) : int(roi["row1"]),
                int(roi["column0"]) : int(roi["column1"]),
            ]
        )
    )


class DistanceGraphUncertaintyProvider:
    name: GraphProviderName = "distance"

    def __init__(self) -> None:
        self._scores = pd.Series(dtype=float)

    def fit(
        self, nodes: list[GraphNode], candidates: pd.DataFrame, prediction, config: RunConfig
    ) -> None:
        del nodes, config
        uncertainty = _normalize_map(prediction["reconstruction_uncertainty"].values)
        self._scores = pd.Series(
            [_roi_mean(uncertainty, row) for _, row in candidates.iterrows()],
            index=candidates["roi_id"].astype(str).tolist(),
            dtype=float,
        )

    def score_candidates(self, candidates: pd.DataFrame) -> pd.Series:
        return candidates["roi_id"].astype(str).map(self._scores).astype(float)


class ROIGPGraphUncertaintyProvider:
    name: GraphProviderName = "roi_gp"

    def __init__(self) -> None:
        self._scores = pd.Series(dtype=float)

    def fit(
        self, nodes: list[GraphNode], candidates: pd.DataFrame, prediction, config: RunConfig
    ) -> None:
        del prediction
        if not nodes:
            raise ValueError("ROI-GP uncertainty requires at least one revealed graph node")
        x_train = np.asarray(
            [
                [
                    node.center_x_nm / config.scenario.width_nm,
                    node.center_y_nm / config.scenario.height_nm,
                ]
                for node in nodes
            ],
            dtype=float,
        )
        x_query = np.column_stack(
            [
                candidates["center_x_nm"].to_numpy(float) / config.scenario.width_nm,
                candidates["center_y_nm"].to_numpy(float) / config.scenario.height_nm,
            ]
        )
        means = np.stack([node.channel_mean for node in nodes])
        within_variances = np.stack([node.channel_variance for node in nodes])
        pixel_counts = np.asarray([node.pixel_count for node in nodes], dtype=float)
        variances = []
        gp_config = config.acquisition_v3.graph.gp
        for channel in range(means.shape[1]):
            values = means[:, channel]
            low, high = np.nanpercentile(values, [5.0, 95.0])
            scale = max(float(high - low), 1.0)
            normalized = (values - float(np.nanmedian(values))) / scale
            alpha = np.maximum(
                within_variances[:, channel] / np.maximum(pixel_counts, 1.0) / scale**2,
                gp_config.alpha_floor,
            )
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
                length_scale=gp_config.length_scale_fraction,
                length_scale_bounds="fixed",
                nu=1.5,
            )
            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                normalize_y=False,
                optimizer=None,
                random_state=0,
            )
            model.fit(x_train, normalized)
            _, std = model.predict(x_query, return_std=True)
            variances.append(std**2)
        score = np.mean(np.stack(variances), axis=0)
        maximum = max(float(np.max(score)), 1.0e-12)
        self._scores = pd.Series(
            score / maximum,
            index=candidates["roi_id"].astype(str).tolist(),
            dtype=float,
        )

    def score_candidates(self, candidates: pd.DataFrame) -> pd.Series:
        return candidates["roi_id"].astype(str).map(self._scores).astype(float)


def _uncertainty_provider(name: GraphProviderName):
    if name == "distance":
        return DistanceGraphUncertaintyProvider()
    if name == "roi_gp":
        return ROIGPGraphUncertaintyProvider()
    raise ValueError(f"unsupported graph uncertainty provider: {name}")


def _point_to_segment_distance(
    points: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    direction = end - start
    denominator = float(np.dot(direction, direction))
    if denominator <= 1.0e-12:
        return np.linalg.norm(points - start, axis=1)
    progress = np.clip(((points - start) @ direction) / denominator, 0.0, 1.0)
    projections = start + progress[:, None] * direction
    return np.linalg.norm(points - projections, axis=1)


def _corridor_candidates(
    candidates: pd.DataFrame,
    first: GraphNode,
    second: GraphNode,
    corridor_half_width_nm: float,
) -> pd.DataFrame:
    points = candidates[["center_x_nm", "center_y_nm"]].to_numpy(float)
    distances = _point_to_segment_distance(
        points,
        np.asarray([first.center_x_nm, first.center_y_nm]),
        np.asarray([second.center_x_nm, second.center_y_nm]),
    )
    corridor = candidates.copy()
    corridor["distance_to_edge_nm"] = distances
    midpoint = np.asarray(
        [(first.center_x_nm + second.center_x_nm) / 2.0, (first.center_y_nm + second.center_y_nm) / 2.0]
    )
    corridor["distance_to_edge_midpoint_nm"] = np.linalg.norm(points - midpoint, axis=1)
    return corridor[corridor["distance_to_edge_nm"] <= corridor_half_width_nm].copy()


def _choose_corridor_candidate(candidates: pd.DataFrame) -> pd.Series:
    return candidates.sort_values(
        ["provider_uncertainty", "distance_to_edge_midpoint_nm", "row0", "column0"],
        ascending=[False, True, True, True],
    ).iloc[0]


def _endpoint_contrast(first: GraphNode, second: GraphNode, nodes: list[GraphNode]) -> float:
    values = np.stack([node.channel_mean for node in nodes])
    span = np.maximum(np.nanpercentile(values, 95.0, axis=0) - np.nanpercentile(values, 5.0, axis=0), 1.0)
    difference = (first.channel_mean - second.channel_mean) / span
    return float(np.clip(np.linalg.norm(difference) / np.sqrt(max(len(difference), 1)), 0.0, 1.0))


def _selection_diagnostics(
    nodes: list[GraphNode],
    eligible: pd.DataFrame,
    prediction,
    config: RunConfig,
    arm: GraphArm,
    slice_id: str,
    adaptive_index: int,
) -> tuple[pd.Series, float, list[dict[str, object]], dict[str, object] | None]:
    provider = _uncertainty_provider(arm.provider)
    provider.fit(nodes, eligible, prediction, config)
    candidate_scores = eligible.copy()
    candidate_scores["provider_uncertainty"] = provider.score_candidates(eligible).to_numpy(float)
    if arm.policy == "uncertainty":
        selected = candidate_scores.sort_values(
            ["provider_uncertainty", "row0", "column0"],
            ascending=[False, True, True],
        ).iloc[0]
        return selected, float(selected["provider_uncertainty"]), [], None

    edges, topology, fallback_used = active_spatial_edges(nodes)
    maximum_edge_length = max(
        (
            float(
                np.hypot(
                    nodes[first].center_x_nm - nodes[second].center_x_nm,
                    nodes[first].center_y_nm - nodes[second].center_y_nm,
                )
            )
            for first, second in edges
        ),
        default=1.0,
    )
    roi_width = float(np.median(eligible["column1"] - eligible["column0"])) * (
        config.dataset.x_step_nm or config.instrument.fine_step_nm
    )
    roi_height = float(np.median(eligible["row1"] - eligible["row0"])) * (
        config.dataset.y_step_nm or config.instrument.fine_y_step_nm or config.instrument.fine_step_nm
    )
    corridor_half_width = (
        config.acquisition_v3.graph.corridor_half_width_roi_diagonals
        * float(np.hypot(roi_width, roi_height))
    )
    records: list[dict[str, object]] = []
    scored: list[tuple[float, str, pd.Series, dict[str, object]]] = []
    for first_index, second_index in edges:
        first, second = nodes[first_index], nodes[second_index]
        endpoint_ids = sorted([first.roi_id, second.roi_id])
        edge_id = f"{endpoint_ids[0]}--{endpoint_ids[1]}"
        length = float(
            np.hypot(first.center_x_nm - second.center_x_nm, first.center_y_nm - second.center_y_nm)
        )
        gap = length / max(maximum_edge_length, 1.0e-12)
        corridor = _corridor_candidates(eligible, first, second, corridor_half_width)
        if corridor.empty:
            continue
        corridor["provider_uncertainty"] = corridor["roi_id"].map(
            candidate_scores.set_index("roi_id")["provider_uncertainty"]
        )
        selected = _choose_corridor_candidate(corridor)
        uncertainty = float(corridor["provider_uncertainty"].mean())
        contrast = _endpoint_contrast(first, second, nodes)
        cost, _ = _raster_cost(config, selected)
        if arm.policy == "graph_gap":
            edge_score = gap / max(cost, 1.0e-12)
        elif arm.policy == "graph_uncertainty":
            edge_score = gap * uncertainty / max(cost, 1.0e-12)
        else:
            weights = config.acquisition_v3.graph.hybrid_weights
            edge_score = (
                weights.spatial_gap * gap
                + weights.uncertainty * uncertainty
                + weights.endpoint_contrast * contrast
            ) / max(cost, 1.0e-12)
        record = {
            "slice": slice_id,
            "arm_id": arm.arm_id,
            "policy": arm.policy,
            "provider": arm.provider,
            "confirmation_mode": arm.confirmation_mode,
            "record_type": "active_scored_edge",
            "adaptive_query_index": adaptive_index,
            "active_topology": topology,
            "triangulation_fallback_used": fallback_used,
            "node_count": len(nodes),
            "active_edge_count": len(edges),
            "selected_edge_id": edge_id,
            "endpoint_roi_ids": "|".join(endpoint_ids),
            "edge_length_nm": length,
            "normalized_gap": gap,
            "corridor_candidate_count": len(corridor),
            "provider_uncertainty": uncertainty,
            "endpoint_contrast": contrast,
            "edge_score": edge_score,
            "selected_roi_id": str(selected["roi_id"]),
            "selected_roi_uncertainty": float(selected["provider_uncertainty"]),
            "selected": False,
            "fallback_reason": None,
        }
        records.append(record)
        scored.append((edge_score, edge_id, selected, record))
    if scored:
        _, _, selected, selected_record = sorted(scored, key=lambda item: (-item[0], item[1]))[0]
        selected_record["selected"] = True
        fallback = None
        if fallback_used:
            fallback = {
                "slice": slice_id,
                "arm_id": arm.arm_id,
                "policy": arm.policy,
                "provider": arm.provider,
                "confirmation_mode": arm.confirmation_mode,
                "adaptive_query_index": adaptive_index,
                "fallback_reason": "delaunay_unavailable_complete_graph_used",
                "selected_roi_id": str(selected["roi_id"]),
                "selected_roi_uncertainty": float(selected["provider_uncertainty"]),
            }
        return selected, float(selected_record["edge_score"]), records, fallback
    selected = candidate_scores.sort_values(
        ["provider_uncertainty", "row0", "column0"],
        ascending=[False, True, True],
    ).iloc[0]
    fallback = {
        "slice": slice_id,
        "arm_id": arm.arm_id,
        "policy": arm.policy,
        "provider": arm.provider,
        "confirmation_mode": arm.confirmation_mode,
        "adaptive_query_index": adaptive_index,
        "fallback_reason": "all_active_edge_corridors_exhausted",
        "selected_roi_id": str(selected["roi_id"]),
        "selected_roi_uncertainty": float(selected["provider_uncertainty"]),
    }
    return selected, float(selected["provider_uncertainty"]), records, fallback


def _pilot_history_edge_records(nodes: list[GraphNode], arm: GraphArm, slice_id: str) -> list[dict[str, object]]:
    records = []
    for first_index, second_index in _complete_edges(nodes):
        first, second = nodes[first_index], nodes[second_index]
        endpoint_ids = sorted([first.roi_id, second.roi_id])
        records.append(
            {
                "slice": slice_id,
                "arm_id": arm.arm_id,
                "policy": arm.policy,
                "provider": arm.provider,
                "confirmation_mode": arm.confirmation_mode,
                "record_type": "historical_pilot_edge",
                "adaptive_query_index": np.nan,
                "active_topology": "complete",
                "triangulation_fallback_used": False,
                "node_count": len(nodes),
                "active_edge_count": len(_complete_edges(nodes)),
                "selected_edge_id": f"{endpoint_ids[0]}--{endpoint_ids[1]}",
                "endpoint_roi_ids": "|".join(endpoint_ids),
                "edge_length_nm": float(
                    np.hypot(
                        first.center_x_nm - second.center_x_nm,
                        first.center_y_nm - second.center_y_nm,
                    )
                ),
                "normalized_gap": np.nan,
                "corridor_candidate_count": np.nan,
                "provider_uncertainty": np.nan,
                "endpoint_contrast": np.nan,
                "edge_score": np.nan,
                "selected_roi_id": None,
                "selected_roi_uncertainty": np.nan,
                "selected": False,
                "fallback_reason": None,
            }
        )
    return records


def run_v3_graph_slice_replay(
    config: RunConfig,
    dense_signal: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    channels: list[str],
    slice_id: str,
    arm: GraphArm,
    seed: int = 0,
) -> GraphSliceResult:
    """Run one leakage-free graph refinement replay over a dense slice."""

    candidates = build_v3_roi_catalog(x, y, config.acquisition_v3.roi_size_px)
    rng = np.random.default_rng(seed)
    observed_mask = np.zeros(dense_signal.shape[1:], dtype=bool)
    queried_ids: set[str] = set()
    nodes: list[GraphNode] = []
    node_records: list[dict[str, object]] = []
    edge_records: list[dict[str, object]] = []
    fallback_records: list[dict[str, object]] = []
    metric_records: list[dict[str, object]] = []
    consumed_time_s = 0.0
    consumed_dose = 0.0
    reference = pseudo_reference_from_dense_signal(dense_signal, x, y, channels, config)
    prediction = reconstruct_from_observed_mask(dense_signal, observed_mask, x, y, channels, config)

    def reveal(
        roi: pd.Series,
        stage: str,
        utility: float = np.nan,
        parent_edge_id: str | None = None,
    ) -> None:
        nonlocal consumed_time_s, consumed_dose, prediction
        row0, row1 = int(roi["row0"]), int(roi["row1"])
        column0, column1 = int(roi["column0"]), int(roi["column1"])
        observed_mask[row0:row1, column0:column1] = True
        queried_ids.add(str(roi["roi_id"]))
        time_s, dose = _raster_cost(config, roi)
        consumed_time_s += time_s
        consumed_dose += dose
        node = _node_from_roi(roi, dense_signal, len(nodes) + 1, stage, parent_edge_id)
        nodes.append(node)
        node_records.append(_node_record(node, arm, slice_id, channels))
        prediction = reconstruct_from_observed_mask(dense_signal, observed_mask, x, y, channels, config)
        score = _score_prediction(
            config,
            arm.arm_id,
            seed,
            slice_id,
            len(metric_records) + 1,
            stage,
            observed_mask,
            dense_signal,
            reference,
            prediction,
            consumed_time_s,
            consumed_dose,
        )
        score.update(
            {
                "arm_id": arm.arm_id,
                "graph_policy": arm.policy,
                "uncertainty_provider": arm.provider,
                "confirmation_mode": arm.confirmation_mode,
                "selection_utility": utility,
                "roi_id": str(roi["roi_id"]),
                "parent_edge_id": parent_edge_id,
                "query_count": len(nodes),
            }
        )
        metric_records.append(score)

    pilot_indices = rng.choice(len(candidates), size=config.acquisition_v3.pilot_rois, replace=False)
    for index in pilot_indices:
        reveal(candidates.iloc[int(index)], "random_pilot")
    edge_records.extend(_pilot_history_edge_records(nodes, arm, slice_id))
    for adaptive_index in range(config.acquisition_v3.adaptive_rois):
        eligible = candidates[~candidates["roi_id"].isin(queried_ids)].copy()
        if eligible.empty:
            break
        selected, utility, iteration_edges, fallback = _selection_diagnostics(
            nodes, eligible, prediction, config, arm, slice_id, adaptive_index
        )
        edge_records.extend(iteration_edges)
        if fallback is not None:
            fallback_records.append(fallback)
        selected_edge_id = None
        if iteration_edges:
            matching = [record for record in iteration_edges if record["selected"]]
            if matching:
                selected_edge_id = str(matching[0]["selected_edge_id"])
        reveal(selected, f"{arm.policy}_{arm.provider}_adaptive", utility, selected_edge_id)
    if arm.confirmation_mode == "one_anchor":
        adaptive_nodes = [
            node for node in nodes if node.acquisition_stage.endswith("_adaptive")
        ]
        if adaptive_nodes:
            anchor = max(
                adaptive_nodes,
                key=lambda node: next(
                    row["selection_utility"]
                    for row in metric_records
                    if row["roi_id"] == node.roi_id and row["stage"] == node.acquisition_stage
                ),
            )
            anchor_roi = candidates[candidates["roi_id"] == anchor.roi_id].iloc[0]
            for neighbor in _neighbor_candidates(candidates, anchor_roi, queried_ids):
                reveal(neighbor, "neighbor_confirmation_roi", parent_edge_id=anchor.parent_edge_id)
    return GraphSliceResult(
        arm=arm,
        metrics=pd.DataFrame(metric_records),
        node_trace=pd.DataFrame(node_records),
        edge_trace=pd.DataFrame(edge_records),
        fallback_events=pd.DataFrame(fallback_records),
    )


def _mean_ci(values: pd.Series) -> tuple[float, float, float]:
    clean = values.dropna().to_numpy(float)
    if clean.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(clean))
    if clean.size == 1:
        return mean, np.nan, np.nan
    margin = float(t.ppf(0.975, clean.size - 1) * np.std(clean, ddof=1) / np.sqrt(clean.size))
    return mean, mean - margin, mean + margin


def _final_graph_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.sort_values("iteration").groupby(["slice", "arm_id"], sort=False).tail(1)


def summarize_graph_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    final = _final_graph_metrics(metrics)
    summary = summarize_v3_stratified_metrics(final.rename(columns={"arm_id": "graph_arm_id"}))
    summary = summary.rename(columns={"policy": "arm_id"})
    arm_columns = final[
        ["arm_id", "graph_policy", "uncertainty_provider", "confirmation_mode"]
    ].drop_duplicates()
    return summary.merge(arm_columns, on="arm_id", how="left")


def graph_comparisons_vs_matching_uncertainty(metrics: pd.DataFrame) -> pd.DataFrame:
    final = _final_graph_metrics(metrics)
    rows = []
    for (provider, confirmation), frame in final.groupby(
        ["uncertainty_provider", "confirmation_mode"], sort=False
    ):
        baseline = frame[frame["graph_policy"] == "uncertainty"].set_index("slice")
        for policy in ("graph_gap", "graph_uncertainty", "graph_hybrid"):
            candidate = frame[frame["graph_policy"] == policy].set_index("slice")
            joined = candidate.join(baseline, lsuffix="_candidate", rsuffix="_baseline", how="inner")
            front_delta = (
                joined["front_localization_mean_symmetric_distance_nm_candidate"]
                - joined["front_localization_mean_symmetric_distance_nm_baseline"]
            )
            d95_delta = (
                joined["penetration_d95_localization_absolute_error_nm_candidate"]
                - joined["penetration_d95_localization_absolute_error_nm_baseline"]
            )
            area_delta = (
                joined["selected_area_fraction_candidate"] - joined["selected_area_fraction_baseline"]
            )
            mean_front, front_low, front_high = _mean_ci(front_delta)
            mean_d95, d95_low, d95_high = _mean_ci(d95_delta)
            rows.append(
                {
                    "graph_policy": policy,
                    "uncertainty_provider": provider,
                    "confirmation_mode": confirmation,
                    "compared_slices": len(joined),
                    "mean_front_localization_delta_nm": mean_front,
                    "front_localization_delta_ci95_low_nm": front_low,
                    "front_localization_delta_ci95_high_nm": front_high,
                    "front_localization_win_rate": float((front_delta < 0).mean()),
                    "mean_penetration_d95_delta_nm": mean_d95,
                    "penetration_d95_delta_ci95_low_nm": d95_low,
                    "penetration_d95_delta_ci95_high_nm": d95_high,
                    "penetration_d95_win_rate": float((d95_delta < 0).mean()),
                    "mean_selected_area_delta_pp": float(100.0 * area_delta.mean()),
                }
            )
    return pd.DataFrame(rows).sort_values("mean_front_localization_delta_nm")


def graph_milestone_comparisons(metrics: pd.DataFrame, slice_ids: list[str]) -> pd.DataFrame:
    milestones = list(range(10, len(slice_ids) + 1, 10))
    if not milestones or milestones[-1] != len(slice_ids):
        milestones.append(len(slice_ids))
    frames = []
    for milestone in milestones:
        frame = metrics[metrics["slice"].astype(str).str.zfill(3).isin(slice_ids[:milestone])]
        comparison = graph_comparisons_vs_matching_uncertainty(frame)
        comparison.insert(0, "evaluated_slices", milestone)
        frames.append(comparison)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_v3_graph_stack_validation(
    template_path: Path,
    slice_ids: list[str],
    output: Path,
    manifest_path: Path | None = None,
    seed: int = 0,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run paired graph-refinement arms over a dense multichannel stack."""

    template = load_config(template_path)
    arms = default_graph_arms(template)
    manifest_sources = (
        sources_from_manifest(manifest_path, slice_ids, template.scenario.elements)
        if manifest_path is not None
        else None
    )
    output.mkdir(parents=True, exist_ok=True)
    write_config(template, output / "resolved_template_config.yaml")
    artifacts = {
        "metrics": output / "v3_graph_metrics_by_iteration.partial.csv",
        "nodes": output / "v3_graph_node_trace.partial.csv",
        "edges": output / "v3_graph_edge_trace.partial.csv",
        "fallbacks": output / "v3_graph_fallback_events.partial.csv",
    }
    frames: dict[str, list[pd.DataFrame]] = {key: [] for key in artifacts}
    completed: set[tuple[str, str]] = set()
    if resume and artifacts["metrics"].exists():
        existing = pd.read_csv(artifacts["metrics"], dtype={"slice": str}, low_memory=False)
        existing["slice"] = existing["slice"].str.zfill(3)
        frames["metrics"].append(existing)
        completed = set(existing[["slice", "arm_id"]].drop_duplicates().itertuples(index=False, name=None))
        for key in ("nodes", "edges", "fallbacks"):
            if artifacts[key].exists() and artifacts[key].stat().st_size:
                frame = pd.read_csv(artifacts[key], dtype={"slice": str}, low_memory=False)
                if not frame.empty:
                    frame["slice"] = frame["slice"].str.zfill(3)
                    frames[key].append(frame)
    for slice_index, slice_id in enumerate(slice_ids, start=1):
        config = v3_config_for_slice(
            template, slice_id, manifest_sources[slice_id] if manifest_sources else None
        )
        source, _ = ingest_dataset(config)
        dense_signal, x, y, channels = dense_signal_from_observations(config, source)
        for arm in arms:
            if (slice_id, arm.arm_id) in completed:
                continue
            result = run_v3_graph_slice_replay(config, dense_signal, x, y, channels, slice_id, arm, seed)
            frames["metrics"].append(result.metrics)
            frames["nodes"].append(result.node_trace)
            frames["edges"].append(result.edge_trace)
            if not result.fallback_events.empty:
                frames["fallbacks"].append(result.fallback_events)
        if slice_index % 5 == 0 or slice_index == len(slice_ids):
            for key, path in artifacts.items():
                if frames[key]:
                    pd.concat(frames[key], ignore_index=True).to_csv(path, index=False)
        if slice_index % 10 == 0 or slice_index == len(slice_ids):
            print(f"Validated v3 graph refinement on {slice_index}/{len(slice_ids)} slices.")
    combined = {
        key: pd.concat(value, ignore_index=True) if value else pd.DataFrame()
        for key, value in frames.items()
    }
    metrics = combined["metrics"]
    summary = summarize_graph_metrics(metrics)
    comparisons = graph_comparisons_vs_matching_uncertainty(metrics)
    blocks = summarize_v3_slice_blocks(metrics.rename(columns={"arm_id": "graph_arm_id"}))
    blocks = blocks.rename(columns={"policy": "arm_id"})
    for key, frame in combined.items():
        name = {
            "metrics": "v3_graph_metrics_by_iteration.csv",
            "nodes": "v3_graph_node_trace.csv",
            "edges": "v3_graph_edge_trace.csv",
            "fallbacks": "v3_graph_fallback_events.csv",
        }[key]
        frame.to_csv(output / name, index=False)
    summary.to_csv(output / "v3_graph_summary.csv", index=False)
    comparisons.to_csv(output / "v3_graph_comparison_vs_matching_uncertainty.csv", index=False)
    graph_milestone_comparisons(metrics, slice_ids).to_csv(
        output / "v3_graph_comparison_every_10_slices.csv", index=False
    )
    blocks.to_csv(output / "v3_graph_slice_block_summary.csv", index=False)
    protocol = {
        "schema": "balance_nm_v3_adaptive_spatial_graph_refinement",
        "template_config": str(template_path),
        "download_manifest": str(manifest_path) if manifest_path is not None else None,
        "slice_count": len(slice_ids),
        "slice_ids": slice_ids,
        "seed": seed,
        "arms": [arm.arm_id for arm in arms],
        "front_semantics": "morphology-defined alteration-front proxy; not expert-labeled corrosion truth",
        "reconstruction_semantics": "all arms evaluated with the same nearest-observation v3 reconstruction",
    }
    with (output / "v3_graph_protocol.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(protocol, handle, sort_keys=False)
    return metrics, summary
