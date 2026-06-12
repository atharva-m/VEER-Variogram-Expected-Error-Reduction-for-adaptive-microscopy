"""Bayesian guarded ROI-summary variance-reduction validation for Alloy 617 replay."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml
from scipy.stats import t

from .data import ingest_dataset
from .domain import RunConfig
from .io import load_config, write_config
from .v3_morphology import (
    dense_signal_from_observations,
    pseudo_reference_from_dense_signal,
    reconstruct_from_observed_mask,
)
from .v3_validation import _raster_cost, build_v3_roi_catalog, sources_from_manifest, v3_config_for_slice
from .v4_bayesian import (
    BayesianROIObservation,
    fit_bayesian_roi_posterior,
    observation_from_revealed_roi,
)
from .v4_validation import (
    _checkpoint_part_path,
    _deduplicate_metrics,
    _deduplicate_trace,
    _distance_pixels,
    _hypothetical_composite_gain,
    _manifest_slice_ids,
    _normalized_distance,
    _read_checkpoint_frames,
    _sample_slice_ids,
    _score_v4_prediction,
    _write_checkpoint_part,
    build_v4_folds,
    lookahead_coverage_gain,
    morphology_composite_error,
    summarize_v4_metrics,
)

V4_BAYESIAN_POLICIES = [
    "uncertainty_distance_sequential",
    "uncertainty_lookahead",
    "bayesian_variance_reduction",
    "bayesian_guarded_lookahead",
    "uniform",
    "random",
    "oracle_composite_gain",
]
V4_BAYESIAN_PRIMARY_POLICIES = {"bayesian_variance_reduction", "bayesian_guarded_lookahead"}


@dataclass
class V4BayesianSliceResult:
    metrics: pd.DataFrame
    candidate_trace: pd.DataFrame
    posterior_trace: pd.DataFrame
    node_trace: pd.DataFrame


def _roi_mean(values: np.ndarray, roi: pd.Series) -> float:
    return float(
        np.mean(
            values[
                int(roi["row0"]) : int(roi["row1"]),
                int(roi["column0"]) : int(roi["column1"]),
            ]
        )
    )


def _posterior_record(
    fold_id: str,
    slice_id: str,
    policy: str,
    iteration: int,
    posterior,
    channels: list[str],
) -> dict[str, object]:
    return {
        "fold": fold_id,
        "slice": slice_id,
        "policy": policy,
        "iteration": iteration,
        "length_scale_fraction": posterior.length_scale_fraction,
        "integrated_posterior_variance": posterior.integrated_variance,
        "candidate_noise_by_channel": json.dumps(dict(zip(channels, posterior.channel_noise.tolist()))),
        "posterior_mean_by_channel_roi": json.dumps(posterior.channel_mean.tolist()),
        "posterior_variance_by_channel_roi": json.dumps(posterior.channel_variance.tolist()),
    }


def _node_record(
    fold_id: str,
    slice_id: str,
    policy: str,
    node: BayesianROIObservation,
    channels: list[str],
) -> dict[str, object]:
    return {
        "fold": fold_id,
        "slice": slice_id,
        "policy": policy,
        "roi_id": node.roi_id,
        "acquisition_sequence": node.acquisition_sequence,
        "center_x_nm": node.center_x_nm,
        "center_y_nm": node.center_y_nm,
        "pixel_count": node.pixel_count,
        "channel_mean": json.dumps(dict(zip(channels, node.channel_mean.tolist()))),
        "channel_variance": json.dumps(dict(zip(channels, node.channel_variance.tolist()))),
    }


def _deterministic_bayesian_choice(
    scored: pd.DataFrame,
    tie_tolerance: float,
) -> pd.Series:
    maximum = float(scored["bayesian_utility"].max())
    tied = scored[scored["bayesian_utility"] >= maximum - tie_tolerance]
    return tied.sort_values(
        ["geometry_coverage_gain", "row0", "column0"],
        ascending=[False, True, True],
    ).iloc[0]


def _score_candidates(
    policy: str,
    catalog: pd.DataFrame,
    queried_ids: set[str],
    distance_pixels: np.ndarray,
    posterior,
    config: RunConfig,
    rng: np.random.Generator,
    dense_signal: np.ndarray,
    observed_mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    channels: list[str],
    reference,
    shared_prediction,
) -> tuple[pd.Series, pd.DataFrame]:
    eligible = catalog[~catalog["roi_id"].isin(queried_ids)].copy()
    if eligible.empty:
        raise ValueError("no feasible Bayesian ROI candidates remain")
    scores = eligible.merge(posterior.candidate_scores, on="roi_id", how="left", validate="one_to_one")
    maximum_geometry = max(float(scores["geometry_coverage_gain"].max()), 1.0e-12)
    scores["geometry_shortlist_eligible"] = (
        scores["geometry_coverage_gain"]
        >= config.acquisition_v4.bayesian.geometry_shortlist_ratio * maximum_geometry
    )
    if policy == "uncertainty_distance_sequential":
        normalized_distance = _normalized_distance(distance_pixels)
        scores["selection_utility"] = [
            _roi_mean(normalized_distance, roi) for _, roi in scores.iterrows()
        ]
        selected = scores.sort_values(
            ["selection_utility", "row0", "column0"], ascending=[False, True, True]
        ).iloc[0]
    elif policy == "uncertainty_lookahead":
        scores["selection_utility"] = scores["geometry_coverage_gain"]
        selected = scores.sort_values(
            ["selection_utility", "row0", "column0"], ascending=[False, True, True]
        ).iloc[0]
    elif policy == "bayesian_variance_reduction":
        scores["geometry_shortlist_eligible"] = True
        scores["selection_utility"] = scores["bayesian_utility"]
        selected = _deterministic_bayesian_choice(
            scores, config.acquisition_v4.bayesian.tie_tolerance
        )
    elif policy == "bayesian_guarded_lookahead":
        scores["selection_utility"] = np.where(
            scores["geometry_shortlist_eligible"], scores["bayesian_utility"], -np.inf
        )
        selected = _deterministic_bayesian_choice(
            scores[scores["geometry_shortlist_eligible"]],
            config.acquisition_v4.bayesian.tie_tolerance,
        )
    elif policy == "uniform":
        scores["selection_utility"] = np.nan
        selected = scores.sort_values(["row0", "column0"]).iloc[0]
    elif policy == "random":
        scores["selection_utility"] = np.nan
        selected = scores.iloc[int(rng.integers(0, len(scores)))]
    elif policy == "oracle_composite_gain":
        current_error = morphology_composite_error(reference, shared_prediction, x, y, config)
        scores["oracle_composite_gain"] = [
            _hypothetical_composite_gain(
                dense_signal,
                observed_mask,
                roi,
                x,
                y,
                channels,
                config,
                reference,
                current_error,
            )
            for _, roi in scores.iterrows()
        ]
        scores["selection_utility"] = scores["oracle_composite_gain"]
        selected = scores.sort_values(
            ["selection_utility", "row0", "column0"], ascending=[False, True, True]
        ).iloc[0]
    else:
        raise ValueError(f"unsupported Bayesian v4.1 policy: {policy}")
    scores["selected"] = scores["roi_id"] == str(selected["roi_id"])
    return selected, scores


def run_v4_bayesian_slice_replay(
    config: RunConfig,
    dense_signal: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    channels: list[str],
    slice_id: str,
    policy: str,
    length_scale_fraction: float,
    fold_id: str = "smoke",
    seed: int = 0,
    checkpoint_callback: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], None]
    | None = None,
) -> V4BayesianSliceResult:
    """Run one fixed-budget shared-evaluator Bayesian raster replay."""

    if policy not in V4_BAYESIAN_POLICIES:
        raise ValueError(f"unsupported Bayesian v4.1 policy: {policy}")
    catalog = build_v3_roi_catalog(x, y, config.acquisition_v4.roi_size_px)
    if config.acquisition_v4.total_rois > len(catalog):
        raise ValueError("v4 total_rois exceeds ROI catalog size")
    rng = np.random.default_rng(seed)
    observed_mask = np.zeros(dense_signal.shape[1:], dtype=bool)
    queried_ids: set[str] = set()
    nodes: list[BayesianROIObservation] = []
    metrics: list[dict[str, object]] = []
    candidate_frames: list[pd.DataFrame] = []
    posterior_rows: list[dict[str, object]] = []
    node_rows: list[dict[str, object]] = []
    consumed_time_s = 0.0
    consumed_dose = 0.0
    reference = pseudo_reference_from_dense_signal(dense_signal, x, y, channels, config)
    prediction = reconstruct_from_observed_mask(dense_signal, observed_mask, x, y, channels, config)

    def checkpoint() -> None:
        if checkpoint_callback is not None:
            checkpoint_callback(
                pd.DataFrame(metrics),
                pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame(),
                pd.DataFrame(posterior_rows),
                pd.DataFrame(node_rows),
            )

    def reveal(roi: pd.Series, stage: str) -> None:
        nonlocal consumed_time_s, consumed_dose, prediction
        row0, row1 = int(roi["row0"]), int(roi["row1"])
        column0, column1 = int(roi["column0"]), int(roi["column1"])
        observed_mask[row0:row1, column0:column1] = True
        queried_ids.add(str(roi["roi_id"]))
        time_s, dose = _raster_cost(config, roi)
        consumed_time_s += time_s
        consumed_dose += dose
        node = observation_from_revealed_roi(roi, dense_signal, len(nodes) + 1)
        nodes.append(node)
        node_rows.append(_node_record(fold_id, slice_id, policy, node, channels))
        prediction = reconstruct_from_observed_mask(dense_signal, observed_mask, x, y, channels, config)
        posterior = fit_bayesian_roi_posterior(
            nodes, catalog, config, length_scale_fraction, _distance_pixels(observed_mask)
        )
        posterior_rows.append(
            _posterior_record(fold_id, slice_id, policy, len(nodes), posterior, channels)
        )
        score = _score_v4_prediction(
            config,
            fold_id,
            policy,
            slice_id,
            len(metrics) + 1,
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
                "roi_id": str(roi["roi_id"]),
                "length_scale_fraction": length_scale_fraction,
                "integrated_posterior_variance": posterior.integrated_variance,
            }
        )
        metrics.append(score)
        checkpoint()

    for index in rng.choice(len(catalog), size=config.acquisition_v4.pilot_rois, replace=False):
        reveal(catalog.iloc[int(index)], "random_pilot")
    while len(metrics) < config.acquisition_v4.total_rois:
        distance_pixels = _distance_pixels(observed_mask)
        posterior = fit_bayesian_roi_posterior(
            nodes, catalog, config, length_scale_fraction, distance_pixels
        )
        selected, scored = _score_candidates(
            policy,
            catalog,
            queried_ids,
            distance_pixels,
            posterior,
            config,
            rng,
            dense_signal,
            observed_mask,
            x,
            y,
            channels,
            reference,
            prediction,
        )
        scored.insert(0, "fold", fold_id)
        scored.insert(1, "slice", slice_id)
        scored.insert(2, "policy", policy)
        scored.insert(3, "query_index", len(metrics) + 1)
        scored["length_scale_fraction"] = length_scale_fraction
        candidate_frames.append(scored)
        reveal(selected, f"{policy}_adaptive")
    return V4BayesianSliceResult(
        metrics=pd.DataFrame(metrics),
        candidate_trace=pd.concat(candidate_frames, ignore_index=True)
        if candidate_frames
        else pd.DataFrame(),
        posterior_trace=pd.DataFrame(posterior_rows),
        node_trace=pd.DataFrame(node_rows),
    )


def select_bayesian_length_scale(
    config: RunConfig,
    fold,
    loader: Callable[[str], tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]],
    seed: int = 0,
) -> tuple[float, pd.DataFrame]:
    """Select one frozen fold-level prior from validation slices only."""

    validation_ids = _sample_slice_ids(
        fold.validation_slices, config.acquisition_v4.bayesian.max_validation_slices
    )
    rows = []
    for length_scale in config.acquisition_v4.bayesian.length_scale_catalog:
        for slice_id in validation_ids:
            dense_signal, x, y, channels = loader(slice_id)
            result = run_v4_bayesian_slice_replay(
                config,
                dense_signal,
                x,
                y,
                channels,
                slice_id,
                "bayesian_guarded_lookahead",
                length_scale,
                fold.fold_id,
                seed,
            )
            final = result.metrics.iloc[-1]
            rows.append(
                {
                    "fold": fold.fold_id,
                    "validation_slice": slice_id,
                    "length_scale_fraction": length_scale,
                    "morphology_composite_error": final["morphology_composite_error"],
                    "normalized_reconstruction_rmse": final["normalized_reconstruction_rmse"],
                }
            )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["fold", "length_scale_fraction"], as_index=False)
        .agg(
            mean_validation_morphology_composite_error=("morphology_composite_error", "mean"),
            mean_validation_reconstruction_rmse=("normalized_reconstruction_rmse", "mean"),
        )
        .sort_values(
            [
                "mean_validation_morphology_composite_error",
                "mean_validation_reconstruction_rmse",
                "length_scale_fraction",
            ]
        )
    )
    selected = float(summary.iloc[0]["length_scale_fraction"])
    frame = frame.merge(summary, on=["fold", "length_scale_fraction"], how="left")
    frame["selected"] = frame["length_scale_fraction"] == selected
    return selected, frame


def _mean_ci(values: pd.Series) -> tuple[float, float, float]:
    clean = values.dropna().to_numpy(float)
    if clean.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(clean))
    if clean.size == 1:
        return mean, np.nan, np.nan
    margin = float(t.ppf(0.975, clean.size - 1) * np.std(clean, ddof=1) / np.sqrt(clean.size))
    return mean, mean - margin, mean + margin


def paired_bayesian_comparisons(metrics: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    final = metrics.sort_values("iteration").groupby(["fold", "slice", "policy"], sort=False).tail(1)
    deterministic = final[
        final["policy"].isin(["uncertainty_distance_sequential", "uncertainty_lookahead"])
    ]
    baseline = deterministic.groupby("policy")["morphology_composite_error"].mean().idxmin()
    baseline_frame = final[final["policy"] == baseline].set_index(["fold", "slice"])
    rows = []
    for policy in sorted(set(final["policy"]) - {baseline, "oracle_composite_gain"}):
        candidate = final[final["policy"] == policy].set_index(["fold", "slice"])
        joined = candidate.join(baseline_frame, lsuffix="_candidate", rsuffix="_baseline", how="inner")
        delta = joined["morphology_composite_error_candidate"] - joined["morphology_composite_error_baseline"]
        rmse_delta = (
            joined["normalized_reconstruction_rmse_candidate"]
            / joined["normalized_reconstruction_rmse_baseline"]
            - 1.0
        )
        mean, low, high = _mean_ci(delta)
        equal_cost = bool(
            np.allclose(joined["scan_time_s_candidate"], joined["scan_time_s_baseline"], atol=1e-9)
        )
        rows.append(
            {
                "policy": policy,
                "baseline": baseline,
                "paired_slices": len(joined),
                "mean_composite_error_delta": mean,
                "composite_error_delta_ci95_low": low,
                "composite_error_delta_ci95_high": high,
                "composite_error_win_rate": float((delta < 0.0).mean()),
                "mean_rmse_regression_fraction": float(rmse_delta.mean()),
                "equal_mean_scan_cost": equal_cost,
                "promoted": bool(
                    policy in V4_BAYESIAN_PRIMARY_POLICIES
                    and high < 0.0
                    and float(rmse_delta.mean())
                    <= config.acquisition_v4.rmse_regression_limit_fraction
                    and equal_cost
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_composite_error_delta")


def _deduplicate_posterior(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["slice"] = combined["slice"].astype(str).str.zfill(3)
    return combined.drop_duplicates(["fold", "slice", "policy", "iteration"], keep="last")


def _deduplicate_nodes(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["slice"] = combined["slice"].astype(str).str.zfill(3)
    return combined.drop_duplicates(
        ["fold", "slice", "policy", "acquisition_sequence"], keep="last"
    )


def _cached_length_scale(
    prior: pd.DataFrame,
    fold,
    config: RunConfig,
) -> tuple[float, pd.DataFrame] | None:
    if prior.empty:
        return None
    expected_slices = set(
        _sample_slice_ids(
            fold.validation_slices, config.acquisition_v4.bayesian.max_validation_slices
        )
    )
    expected_scales = set(config.acquisition_v4.bayesian.length_scale_catalog)
    frame = prior[prior["fold"] == fold.fold_id].copy()
    if (
        set(frame["validation_slice"].astype(str).str.zfill(3)) != expected_slices
        or set(frame["length_scale_fraction"].astype(float)) != expected_scales
    ):
        return None
    selected = frame[frame["selected"].astype(str).str.lower() == "true"]
    selected_scales = selected["length_scale_fraction"].astype(float).unique()
    if len(selected_scales) != 1:
        return None
    return float(selected_scales[0]), frame


def run_v4_bayesian_stack_validation(
    template_path: Path,
    output: Path,
    manifest_path: Path,
    fold_specification: str = "all",
    slice_ids: list[str] | None = None,
    policies: list[str] | None = None,
    seed: int = 0,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run blocked, resumable Bayesian variance-reduction validation."""

    template = load_config(template_path)
    all_slices = _manifest_slice_ids(manifest_path)
    requested = set(slice_ids or all_slices)
    folds = build_v4_folds(all_slices, template)
    if fold_specification != "all":
        folds = [fold for fold in folds if fold.fold_id == f"fold_{int(fold_specification)}"]
    policies = policies or V4_BAYESIAN_POLICIES
    unknown = set(policies) - set(V4_BAYESIAN_POLICIES)
    if unknown:
        raise ValueError(f"unsupported Bayesian v4.1 policies: {sorted(unknown)}")
    sources = sources_from_manifest(manifest_path, all_slices, template.scenario.elements)
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}

    def load_slice(slice_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        if slice_id not in cache:
            config = v3_config_for_slice(template, slice_id, sources[slice_id])
            observations, _ = ingest_dataset(config)
            cache[slice_id] = dense_signal_from_observations(config, observations)
        return cache[slice_id]

    output.mkdir(parents=True, exist_ok=True)
    write_config(template, output / "resolved_template_config.yaml")
    part_directories = {
        "metrics": output / "v4_bayesian_metrics_parts",
        "candidate": output / "v4_bayesian_candidate_trace_parts",
        "posterior": output / "v4_bayesian_posterior_trace_parts",
        "nodes": output / "v4_bayesian_node_trace_parts",
    }
    frames = {
        "metrics": _read_checkpoint_frames(part_directories["metrics"]) if resume else [],
        "candidate": _read_checkpoint_frames(part_directories["candidate"]) if resume else [],
        "posterior": _read_checkpoint_frames(part_directories["posterior"]) if resume else [],
        "nodes": _read_checkpoint_frames(part_directories["nodes"]) if resume else [],
    }
    metrics = _deduplicate_metrics(frames["metrics"])
    candidates = _deduplicate_trace(frames["candidate"])
    posterior = _deduplicate_posterior(frames["posterior"])
    nodes = _deduplicate_nodes(frames["nodes"])
    metric_complete = (
        set(
            metrics[metrics["query_count"] >= template.acquisition_v4.total_rois][
                ["fold", "slice", "policy"]
            ]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if not metrics.empty
        else set()
    )
    candidate_complete = (
        set(
            candidates[["fold", "slice", "policy"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if not candidates.empty
        else set()
    )
    completed = metric_complete & candidate_complete
    oracle_ids = set(_sample_slice_ids(all_slices, template.acquisition_v4.oracle_sample_slices))
    prior_path = output / "v4_bayesian_prior_selection.csv"
    cached_prior = (
        pd.read_csv(prior_path, dtype={"validation_slice": str})
        if resume and prior_path.exists() and prior_path.stat().st_size
        else pd.DataFrame()
    )
    prior_frames = []
    processed = 0
    for fold in folds:
        tests = [slice_id for slice_id in fold.test_slices if slice_id in requested]
        if not tests:
            continue
        cached = _cached_length_scale(cached_prior, fold, template)
        if cached is None:
            length_scale, prior = select_bayesian_length_scale(template, fold, load_slice, seed)
        else:
            length_scale, prior = cached
        prior_frames.append(prior)
        cached_prior = pd.concat([cached_prior, prior], ignore_index=True).drop_duplicates(
            ["fold", "validation_slice", "length_scale_fraction"], keep="last"
        )
        cached_prior.to_csv(prior_path, index=False)
        for slice_id in tests:
            config = v3_config_for_slice(template, slice_id, sources[slice_id])
            dense_signal, x, y, channels = load_slice(slice_id)
            for policy in policies:
                if policy == "oracle_composite_gain" and slice_id not in oracle_ids:
                    continue
                if (fold.fold_id, slice_id, policy) in completed:
                    continue

                def checkpoint(metric_frame, candidate_frame, posterior_frame, node_frame):
                    values = {
                        "metrics": metric_frame,
                        "candidate": candidate_frame,
                        "posterior": posterior_frame,
                        "nodes": node_frame,
                    }
                    for name, frame in values.items():
                        if not frame.empty:
                            _write_checkpoint_part(
                                frame,
                                _checkpoint_part_path(
                                    part_directories[name], fold.fold_id, slice_id, policy
                                ),
                            )

                result = run_v4_bayesian_slice_replay(
                    config,
                    dense_signal,
                    x,
                    y,
                    channels,
                    slice_id,
                    policy,
                    length_scale,
                    fold.fold_id,
                    seed,
                    checkpoint,
                )
                frames["metrics"].append(result.metrics)
                frames["candidate"].append(result.candidate_trace)
                frames["posterior"].append(result.posterior_trace)
                frames["nodes"].append(result.node_trace)
            processed += 1
            if processed % 10 == 0:
                print(f"Validated Bayesian v4.1 reconstruction on {processed} requested slices.")
    metrics = _deduplicate_metrics(frames["metrics"])
    candidates = _deduplicate_trace(frames["candidate"])
    posterior = _deduplicate_posterior(frames["posterior"])
    nodes = _deduplicate_nodes(frames["nodes"])
    summary, curves, auc = summarize_v4_metrics(metrics)
    final = metrics.sort_values("iteration").groupby(["fold", "slice", "policy"], sort=False).tail(1)
    comparisons = paired_bayesian_comparisons(metrics, template)
    prior = cached_prior
    metrics.to_csv(output / "v4_bayesian_metrics_by_iteration.csv", index=False)
    final.to_csv(output / "v4_bayesian_final_metrics_by_slice.csv", index=False)
    summary.to_csv(output / "v4_bayesian_oof_summary.csv", index=False)
    comparisons.to_csv(output / "v4_bayesian_paired_comparisons.csv", index=False)
    curves.to_csv(output / "v4_bayesian_error_vs_cost_curves.csv", index=False)
    auc.to_csv(output / "v4_bayesian_composite_error_auc_vs_cost.csv", index=False)
    candidates.to_csv(output / "v4_bayesian_candidate_trace.csv", index=False)
    posterior.to_csv(output / "v4_bayesian_posterior_trace.csv", index=False)
    nodes.to_csv(output / "v4_bayesian_node_trace.csv", index=False)
    prior.to_csv(output / "v4_bayesian_prior_selection.csv", index=False)
    final[final["policy"] == "oracle_composite_gain"].to_csv(
        output / "v4_bayesian_oracle_headroom_summary.csv", index=False
    )
    variance_auc = []
    for (fold_id, slice_id, policy), frame in metrics.groupby(["fold", "slice", "policy"]):
        ordered = frame.sort_values("scan_time_s")
        variance_auc.append(
            {
                "fold": fold_id,
                "slice": slice_id,
                "policy": policy,
                "integrated_posterior_variance_auc_vs_cost": float(
                    np.trapezoid(
                        ordered["integrated_posterior_variance"].to_numpy(float),
                        ordered["scan_time_s"].to_numpy(float),
                    )
                ),
            }
        )
    pd.DataFrame(variance_auc).to_csv(
        output / "v4_bayesian_integrated_variance_auc_vs_cost.csv", index=False
    )
    protocol = {
        "schema": "balance_nm_v4_1_bayesian_guarded_variance_reduction",
        "template_config": str(template_path),
        "manifest": str(manifest_path),
        "seed": seed,
        "requested_slices": sorted(requested),
        "policies": policies,
        "folds": [fold.__dict__ for fold in folds],
        "dense_truth_policy": "hidden from deployable selectors; evaluation-only replay reference",
        "bayesian_prior_policy": "fold validation catalog selection; frozen during test replay",
        "shared_evaluator": "nearest-observation reconstruction for every primary comparison arm",
    }
    with (output / "v4_bayesian_fold_protocol.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(protocol, handle, sort_keys=False)
    return metrics, summary
