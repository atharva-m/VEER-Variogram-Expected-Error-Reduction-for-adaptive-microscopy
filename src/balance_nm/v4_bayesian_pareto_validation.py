"""V4.4 Pareto-gated Bayesian EIVR validation for Alloy 617 replay."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from .data import ingest_dataset
from .domain import RunConfig
from .io import load_config, write_config
from .v3_morphology import (
    dense_signal_from_observations,
    pseudo_reference_from_dense_signal,
    reconstruct_from_observed_mask,
)
from .v3_validation import _raster_cost, build_v3_roi_catalog, sources_from_manifest, v3_config_for_slice
from .v4_bayesian_morphology_validation import paired_morphology_comparisons
from .v4_bayesian_pareto import (
    ParetoPosterior,
    ParetoSubtileObservation,
    fit_pareto_subtile_posterior,
    gp_prediction_from_pareto_posterior,
    pareto_select_candidate,
    pareto_subtile_observations_from_revealed_roi,
    parse_pareto_policy,
)
from .v4_validation import (
    _checkpoint_part_path,
    _deduplicate_metrics,
    _deduplicate_trace,
    _distance_pixels,
    _hypothetical_composite_gain,
    _manifest_slice_ids,
    _read_checkpoint_frames,
    _sample_slice_ids,
    _score_v4_prediction,
    _write_checkpoint_part,
    build_v4_folds,
    lookahead_coverage_gain,
    morphology_composite_error,
    summarize_v4_metrics,
)

V4_BAYESIAN_PARETO_ARMS = [
    "bayesian_pareto_eivr_4x4_mean_tau090",
    "bayesian_pareto_eivr_4x4_mean_tau085",
    "bayesian_pareto_eivr_4x4_texture_tau090",
    "bayesian_pareto_eivr_4x4_texture_tau085",
    "bayesian_pareto_eivr_8x8_mean_tau090",
    "bayesian_pareto_eivr_8x8_mean_tau085",
    "bayesian_pareto_eivr_8x8_texture_tau090",
    "bayesian_pareto_eivr_8x8_texture_tau085",
]
V4_BAYESIAN_PARETO_POLICIES = [
    "uncertainty_lookahead",
    *V4_BAYESIAN_PARETO_ARMS,
    "uniform",
    "random",
    "oracle_composite_gain",
]


@dataclass
class V4BayesianParetoSliceResult:
    metrics: pd.DataFrame
    candidate_trace: pd.DataFrame
    kernel_weight_trace: pd.DataFrame
    subtile_trace: pd.DataFrame
    gp_diagnostics: pd.DataFrame


def _kernel_records(
    fold_id: str,
    slice_id: str,
    policy: str,
    iteration: int,
    posterior: ParetoPosterior,
) -> list[dict[str, object]]:
    return [
        {
            "fold": fold_id,
            "slice": slice_id,
            "policy": policy,
            "iteration": iteration,
            "model_mode": f"{posterior.spec.grid_shape[0]}x{posterior.spec.grid_shape[1]}_{posterior.spec.feature_mode}",
            "kernel": label,
            "posterior_weight": float(weight),
            "integrated_posterior_variance": posterior.integrated_variance,
            "effective_components": posterior.effective_components,
            "retained_training_subtiles": posterior.retained_training_count,
            "total_revealed_subtiles": posterior.total_revealed_subtiles,
        }
        for label, weight in zip(posterior.kernel_labels, posterior.posterior_weights)
    ]


def _subtile_records(
    fold_id: str,
    slice_id: str,
    policy: str,
    observations: list[ParetoSubtileObservation],
    channels: list[str],
    feature_families: list[str],
) -> list[dict[str, object]]:
    rows = []
    for node in observations:
        feature_payload = {
            channel: dict(zip(feature_families, node.feature_values[index].tolist()))
            for index, channel in enumerate(channels)
        }
        rows.append(
            {
                "fold": fold_id,
                "slice": slice_id,
                "policy": policy,
                "roi_id": node.roi_id,
                "subtile_id": node.subtile_id,
                "acquisition_sequence": node.acquisition_sequence,
                "center_x_nm": node.center_x_nm,
                "center_y_nm": node.center_y_nm,
                "pixel_count": node.pixel_count,
                "feature_families": json.dumps(feature_families),
                "channel_mean": json.dumps(dict(zip(channels, node.channel_mean.tolist()))),
                "mean_noise": json.dumps(dict(zip(channels, node.mean_noise.tolist()))),
                "feature_values": json.dumps(feature_payload),
            }
        )
    return rows


def _base_candidate_scores(
    catalog: pd.DataFrame,
    queried_ids: set[str],
    distance_pixels: np.ndarray,
    config: RunConfig,
) -> pd.DataFrame:
    scores = catalog[~catalog["roi_id"].isin(queried_ids)].copy()
    if scores.empty:
        raise ValueError("no feasible v4.4 raster candidates remain")
    scores["geometry_gain"] = [
        lookahead_coverage_gain(distance_pixels, roi) for _, roi in scores.iterrows()
    ]
    scores["estimated_raster_cost_s"] = [
        _raster_cost(config, roi)[0] for _, roi in scores.iterrows()
    ]
    scores["geometry_gain_per_cost"] = scores["geometry_gain"] / np.maximum(
        scores["estimated_raster_cost_s"], 1.0e-12
    )
    maximum = max(float(scores["geometry_gain_per_cost"].max()), 1.0e-12)
    scores["normalized_geometry_gain"] = scores["geometry_gain_per_cost"] / maximum
    scores["geometry_threshold"] = np.nan
    scores["geometry_shortlist_eligible"] = True
    scores["geometry_argmax"] = scores["geometry_gain_per_cost"] == scores["geometry_gain_per_cost"].max()
    scores["shortlist_size"] = len(scores)
    scores["eivr_evaluated"] = False
    scores["EIVR_by_kernel"] = "{}"
    scores["model_averaged_fractional_EIVR"] = np.nan
    scores["EIVR_LCB"] = np.nan
    scores["kernel_support"] = np.nan
    scores["relative_evidence"] = 0.0
    scores["evidence_eligible"] = False
    return scores


def _score_candidates(
    policy: str,
    catalog: pd.DataFrame,
    queried_ids: set[str],
    observed_mask: np.ndarray,
    posterior: ParetoPosterior | None,
    config: RunConfig,
    rng: np.random.Generator,
    dense_signal: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    channels: list[str],
    reference,
    prediction,
) -> tuple[pd.Series, pd.DataFrame]:
    distance_pixels = _distance_pixels(observed_mask)
    if policy in V4_BAYESIAN_PARETO_ARMS:
        if posterior is None:
            raise ValueError("v4.4 Bayesian Pareto policies require a fitted posterior")
        selected, scores = pareto_select_candidate(
            catalog, queried_ids, distance_pixels, posterior, config
        )
    else:
        scores = _base_candidate_scores(catalog, queried_ids, distance_pixels, config)
        if policy == "uncertainty_lookahead":
            scores["selection_utility"] = scores["geometry_gain_per_cost"]
            selected = scores.sort_values(
                ["selection_utility", "row0", "column0"],
                ascending=[False, True, True],
            ).iloc[0]
        elif policy == "uniform":
            scores["selection_utility"] = np.nan
            selected = scores.sort_values(["row0", "column0"]).iloc[0]
        elif policy == "random":
            scores["selection_utility"] = np.nan
            selected = scores.iloc[int(rng.integers(0, len(scores)))]
        elif policy == "oracle_composite_gain":
            current_error = morphology_composite_error(reference, prediction, x, y, config)
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
                ["selection_utility", "row0", "column0"],
                ascending=[False, True, True],
            ).iloc[0]
        else:
            raise ValueError(f"unsupported v4.4 Pareto policy: {policy}")
        scores["selected"] = scores["roi_id"] == str(selected["roi_id"])
    return selected, scores


def run_v4_bayesian_pareto_slice_replay(
    config: RunConfig,
    dense_signal: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    channels: list[str],
    slice_id: str,
    policy: str,
    fold_id: str = "smoke",
    seed: int = 0,
    checkpoint_callback: Callable[..., None] | None = None,
) -> V4BayesianParetoSliceResult:
    """Run one fixed-budget v4.4 Pareto-gated Bayesian replay."""

    if policy not in V4_BAYESIAN_PARETO_POLICIES:
        raise ValueError(f"unsupported v4.4 Pareto policy: {policy}")
    spec = parse_pareto_policy(policy) if policy in V4_BAYESIAN_PARETO_ARMS else None
    catalog = build_v3_roi_catalog(x, y, config.acquisition_v4.roi_size_px)
    if config.acquisition_v4.total_rois > len(catalog):
        raise ValueError("v4 total_rois exceeds ROI catalog size")
    rng = np.random.default_rng(seed)
    observed_mask = np.zeros(dense_signal.shape[1:], dtype=bool)
    queried_ids: set[str] = set()
    subtile_nodes: list[ParetoSubtileObservation] = []
    metrics: list[dict[str, object]] = []
    candidate_frames: list[pd.DataFrame] = []
    kernel_rows: list[dict[str, object]] = []
    subtile_rows: list[dict[str, object]] = []
    gp_rows: list[dict[str, object]] = []
    consumed_time_s = 0.0
    consumed_dose = 0.0
    posterior: ParetoPosterior | None = None
    reference = pseudo_reference_from_dense_signal(dense_signal, x, y, channels, config)
    prediction = reconstruct_from_observed_mask(dense_signal, observed_mask, x, y, channels, config)

    def frames() -> tuple[pd.DataFrame, ...]:
        return (
            pd.DataFrame(metrics),
            pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame(),
            pd.DataFrame(kernel_rows),
            pd.DataFrame(subtile_rows),
            pd.DataFrame(gp_rows),
        )

    def checkpoint() -> None:
        if checkpoint_callback is not None:
            checkpoint_callback(*frames())

    def reveal(roi: pd.Series, stage: str) -> None:
        nonlocal consumed_time_s, consumed_dose, prediction, posterior
        row0, row1 = int(roi["row0"]), int(roi["row1"])
        column0, column1 = int(roi["column0"]), int(roi["column1"])
        observed_mask[row0:row1, column0:column1] = True
        queried_ids.add(str(roi["roi_id"]))
        time_s, dose = _raster_cost(config, roi)
        consumed_time_s += time_s
        consumed_dose += dose
        if spec is not None:
            observations = pareto_subtile_observations_from_revealed_roi(
                roi, dense_signal, x, y, len(metrics) + 1, config, spec
            )
            subtile_nodes.extend(observations)
            subtile_rows.extend(
                _subtile_records(
                    fold_id,
                    slice_id,
                    policy,
                    observations,
                    channels,
                    posterior.feature_families if posterior is not None else spec_feature_families(spec),
                )
            )
        prediction = reconstruct_from_observed_mask(dense_signal, observed_mask, x, y, channels, config)
        if spec is not None:
            posterior = fit_pareto_subtile_posterior(
                subtile_nodes, catalog, x, y, config, spec
            )
            kernel_rows.extend(
                _kernel_records(fold_id, slice_id, policy, len(metrics) + 1, posterior)
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
                "integrated_posterior_variance": (
                    posterior.integrated_variance if posterior is not None else np.nan
                ),
            }
        )
        metrics.append(score)
        if (
            posterior is not None
            and config.acquisition_v4.bayesian_pareto.gp_reconstruction.enabled
            and len(metrics) == config.acquisition_v4.total_rois
        ):
            gp_prediction = gp_prediction_from_pareto_posterior(
                posterior, x, y, channels, config
            )
            diagnostic = _score_v4_prediction(
                config,
                fold_id,
                policy,
                slice_id,
                len(metrics),
                "diagnostic_only_gp_prediction",
                observed_mask,
                dense_signal,
                reference,
                gp_prediction,
                consumed_time_s,
                consumed_dose,
            )
            gp_rows.append(
                {
                    "fold": fold_id,
                    "slice": slice_id,
                    "policy": policy,
                    "iteration": len(metrics),
                    "gp_reconstruction_rmse_diagnostic": diagnostic[
                        "normalized_reconstruction_rmse"
                    ],
                    "gp_front_distance_diagnostic": diagnostic[
                        "front_mean_symmetric_distance_nm"
                    ],
                    "gp_penetration_d95_error_diagnostic": diagnostic[
                        "penetration_d95_absolute_error_nm"
                    ],
                }
            )
        checkpoint()

    pilots = rng.choice(len(catalog), size=config.acquisition_v4.pilot_rois, replace=False)
    for index in pilots:
        reveal(catalog.iloc[int(index)], "random_pilot")
    while len(metrics) < config.acquisition_v4.total_rois:
        selected, scored = _score_candidates(
            policy,
            catalog,
            queried_ids,
            observed_mask,
            posterior,
            config,
            rng,
            dense_signal,
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
        candidate_frames.append(scored)
        reveal(selected, f"{policy}_adaptive")
    return V4BayesianParetoSliceResult(*frames())


def spec_feature_families(spec) -> list[str]:
    return ["mean"] if spec.feature_mode == "mean" else [
        "mean",
        "residual_mad",
        "gradient_mean",
        "gradient_p95",
        "contrast",
    ]


def _deduplicate_kernel(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["slice"] = combined["slice"].astype(str).str.zfill(3)
    return combined.drop_duplicates(["fold", "slice", "policy", "iteration", "kernel"], keep="last")


def _deduplicate_subtiles(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["slice"] = combined["slice"].astype(str).str.zfill(3)
    return combined.drop_duplicates(
        ["fold", "slice", "policy", "acquisition_sequence", "subtile_id"], keep="last"
    )


def _deduplicate_gp(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["slice"] = combined["slice"].astype(str).str.zfill(3)
    deduplicated = combined.drop_duplicates(
        ["fold", "slice", "policy", "iteration"], keep="last"
    )
    return (
        deduplicated.sort_values("iteration")
        .groupby(["fold", "slice", "policy"], sort=False)
        .tail(1)
    )


def _path_diagnostics(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    selected = candidates[candidates["selected"]].sort_values(["slice", "query_index"])
    paths = (
        selected.groupby(["fold", "slice", "policy"], as_index=False)["roi_id"]
        .agg("|".join)
        .rename(columns={"roi_id": "adaptive_roi_path"})
    )
    return (
        paths.groupby("policy", as_index=False)
        .agg(slices=("slice", "nunique"), unique_adaptive_paths=("adaptive_roi_path", "nunique"))
    )


def run_v4_bayesian_pareto_stack_validation(
    template_path: Path,
    output: Path,
    manifest_path: Path,
    fold_specification: str = "all",
    slice_ids: list[str] | None = None,
    policies: list[str] | None = None,
    seed: int = 0,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run blocked, resumable v4.4 Pareto-gated Bayesian validation."""

    template = load_config(template_path)
    all_slices = _manifest_slice_ids(manifest_path)
    requested = set(slice_ids or all_slices)
    folds = build_v4_folds(all_slices, template)
    if fold_specification != "all":
        folds = [fold for fold in folds if fold.fold_id == f"fold_{int(fold_specification)}"]
    policies = policies or ["uncertainty_lookahead", *V4_BAYESIAN_PARETO_ARMS]
    unknown = set(policies) - set(V4_BAYESIAN_PARETO_POLICIES)
    if unknown:
        raise ValueError(f"unsupported v4.4 Pareto policies: {sorted(unknown)}")
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
    directories = {
        "metrics": output / "v4_bayesian_pareto_metrics_parts",
        "candidate": output / "v4_bayesian_pareto_candidate_parts",
        "kernel": output / "v4_bayesian_pareto_kernel_parts",
        "subtile": output / "v4_bayesian_pareto_subtile_parts",
        "gp": output / "v4_bayesian_pareto_gp_parts",
    }
    frames = {
        name: _read_checkpoint_frames(directory) if resume else []
        for name, directory in directories.items()
    }
    metrics = _deduplicate_metrics(frames["metrics"])
    candidates = _deduplicate_trace(frames["candidate"])
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
        set(candidates[["fold", "slice", "policy"]].drop_duplicates().itertuples(index=False, name=None))
        if not candidates.empty
        else set()
    )
    completed = metric_complete & candidate_complete
    oracle_ids = set(_sample_slice_ids(all_slices, template.acquisition_v4.oracle_sample_slices))
    processed = 0
    for fold in folds:
        tests = [slice_id for slice_id in fold.test_slices if slice_id in requested]
        for slice_id in tests:
            config = v3_config_for_slice(template, slice_id, sources[slice_id])
            dense_signal, x, y, channels = load_slice(slice_id)
            for policy in policies:
                if policy == "oracle_composite_gain" and slice_id not in oracle_ids:
                    continue
                if (fold.fold_id, slice_id, policy) in completed:
                    continue

                def checkpoint(*values):
                    for name, frame in zip(directories, values):
                        if not frame.empty:
                            _write_checkpoint_part(
                                frame,
                                _checkpoint_part_path(
                                    directories[name], fold.fold_id, slice_id, policy
                                ),
                            )

                result = run_v4_bayesian_pareto_slice_replay(
                    config,
                    dense_signal,
                    x,
                    y,
                    channels,
                    slice_id,
                    policy,
                    fold.fold_id,
                    seed,
                    checkpoint,
                )
                result_frames = {
                    "metrics": result.metrics,
                    "candidate": result.candidate_trace,
                    "kernel": result.kernel_weight_trace,
                    "subtile": result.subtile_trace,
                    "gp": result.gp_diagnostics,
                }
                for name, frame in result_frames.items():
                    if not frame.empty:
                        frames[name].append(frame)
            processed += 1
            if processed % 10 == 0:
                print(f"Validated v4.4 Pareto Bayesian reconstruction on {processed} requested slices.")
    metrics = _deduplicate_metrics(frames["metrics"])
    candidates = _deduplicate_trace(frames["candidate"])
    kernels = _deduplicate_kernel(frames["kernel"])
    subtiles = _deduplicate_subtiles(frames["subtile"])
    gp = _deduplicate_gp(frames["gp"])
    summary, curves, auc = summarize_v4_metrics(metrics)
    diagnostics = _path_diagnostics(candidates)
    if not diagnostics.empty:
        summary = summary.merge(diagnostics, on=["policy", "slices"], how="left")
    final = metrics.sort_values("iteration").groupby(["fold", "slice", "policy"], sort=False).tail(1)
    comparisons = paired_morphology_comparisons(metrics, template)
    metrics.to_csv(output / "v4_bayesian_pareto_metrics_by_iteration.csv", index=False)
    final.to_csv(output / "v4_bayesian_pareto_final_metrics_by_slice.csv", index=False)
    summary.to_csv(output / "v4_bayesian_pareto_oof_summary.csv", index=False)
    comparisons.to_csv(output / "v4_bayesian_pareto_paired_comparisons.csv", index=False)
    curves.to_csv(output / "v4_bayesian_pareto_error_vs_cost_curves.csv", index=False)
    auc.to_csv(output / "v4_bayesian_pareto_composite_error_auc_vs_cost.csv", index=False)
    candidates.to_csv(output / "v4_bayesian_pareto_candidate_trace.csv", index=False)
    kernels.to_csv(output / "v4_bayesian_pareto_kernel_weight_trace.csv", index=False)
    subtiles.to_csv(output / "v4_bayesian_pareto_subtile_trace.csv", index=False)
    gp.to_csv(output / "v4_bayesian_pareto_gp_diagnostics.csv", index=False)
    final[final["policy"] == "oracle_composite_gain"].to_csv(
        output / "v4_bayesian_pareto_oracle_headroom_summary.csv", index=False
    )
    blocks = final.copy()
    blocks["slice_block"] = blocks["slice"].astype(int).map(
        lambda value: f"{((value - 1) // 25) * 25 + 1:03d}:{((value - 1) // 25) * 25 + 25:03d}"
    )
    blocks.groupby(["policy", "slice_block"], as_index=False).agg(
        slices=("slice", "nunique"),
        mean_morphology_composite_error=("morphology_composite_error", "mean"),
        mean_front_distance_nm=("front_mean_symmetric_distance_nm", "mean"),
        mean_penetration_d95_error_nm=("penetration_d95_absolute_error_nm", "mean"),
    ).to_csv(output / "v4_bayesian_pareto_slice_block_summary.csv", index=False)
    protocol = {
        "schema": "balance_nm_v4_4_pareto_gated_bayesian_eivr",
        "template_config": str(template_path),
        "manifest": str(manifest_path),
        "seed": seed,
        "requested_slices": sorted(requested),
        "policies": policies,
        "folds": [fold.__dict__ for fold in folds],
        "dense_truth_policy": "hidden from deployable selectors; evaluation-only replay reference",
        "shared_evaluator": "nearest-observation reconstruction for every primary comparison arm",
        "bayesian_order": "geometry shortlist first; EIVR only for shortlisted candidates plus geometry argmax",
        "gp_diagnostics": "endpoint diagnostic only; excluded from promotion decisions",
        "front_semantics": "frozen unsupervised alteration-front proxy; not expert-labeled corrosion truth",
        "historical_ablation_command": "validate-v4-bayesian-morphology-stack",
    }
    with (output / "v4_bayesian_pareto_fold_protocol.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(protocol, handle, sort_keys=False)
    return metrics, summary
