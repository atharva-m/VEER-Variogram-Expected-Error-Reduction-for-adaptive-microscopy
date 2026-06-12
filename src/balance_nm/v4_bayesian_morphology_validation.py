"""V4.3 evidence-gated Bayesian morphology lookahead validation for Alloy 617 replay."""

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
from .v4_bayesian_morphology import (
    BayesianMorphologyPosterior,
    evidence_gated_rank_candidates,
    fit_bayesian_morphology_posterior,
    gp_prediction_from_posterior,
    morphology_fantasy_rank_candidates,
)
from .v4_bayesian_residual import (
    BayesianSubtileObservation,
    subtile_observations_from_revealed_roi,
)
from .v4_bayesian_residual_validation import _deduplicate_kernel, _deduplicate_subtiles, _subtile_records
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

V4_BAYESIAN_MORPHOLOGY_POLICIES = [
    "uncertainty_lookahead",
    "bayesian_subtile_evidence_gated",
    "bayesian_morphology_fantasy_guarded",
    "uniform",
    "random",
    "oracle_composite_gain",
]
V4_BAYESIAN_MORPHOLOGY_PRIMARY_POLICIES = {
    "bayesian_subtile_evidence_gated",
    "bayesian_morphology_fantasy_guarded",
}


@dataclass
class V4BayesianMorphologySliceResult:
    metrics: pd.DataFrame
    candidate_trace: pd.DataFrame
    kernel_weight_trace: pd.DataFrame
    subtile_trace: pd.DataFrame
    fantasy_trace: pd.DataFrame
    reliability_trace: pd.DataFrame
    gp_diagnostics: pd.DataFrame


def _kernel_records(
    fold_id: str,
    slice_id: str,
    policy: str,
    iteration: int,
    posterior: BayesianMorphologyPosterior,
) -> list[dict[str, object]]:
    return [
        {
            "fold": fold_id,
            "slice": slice_id,
            "policy": policy,
            "iteration": iteration,
            "kernel": label,
            "posterior_weight": float(weight),
            "integrated_posterior_variance": posterior.integrated_variance,
            "effective_components": posterior.effective_components,
        }
        for label, weight in zip(posterior.kernel_labels, posterior.posterior_weights)
    ]


def _base_candidate_scores(
    catalog: pd.DataFrame,
    queried_ids: set[str],
    distance_pixels: np.ndarray,
    config: RunConfig,
) -> pd.DataFrame:
    eligible = catalog[~catalog["roi_id"].isin(queried_ids)].copy()
    if eligible.empty:
        raise ValueError("no feasible v4.3 raster candidates remain")
    eligible["geometry_coverage_gain"] = [
        lookahead_coverage_gain(distance_pixels, roi) for _, roi in eligible.iterrows()
    ]
    eligible["estimated_raster_cost_s"] = [
        _raster_cost(config, roi)[0] for _, roi in eligible.iterrows()
    ]
    maximum = max(float(eligible["geometry_coverage_gain"].max()), 1.0e-12)
    eligible["normalized_geometry_gain"] = eligible["geometry_coverage_gain"] / maximum
    eligible["EIVR_by_kernel"] = "{}"
    eligible["model_averaged_fractional_EIVR"] = np.nan
    eligible["EIVR_LCB"] = np.nan
    eligible["kernel_support"] = np.nan
    eligible["evidence_gate_passed"] = False
    eligible["task_gain_mean"] = np.nan
    eligible["task_gain_LCB"] = np.nan
    eligible["morphology_reliability_passed"] = False
    eligible["fallback_reason"] = ""
    return eligible


def _score_candidates(
    policy: str,
    catalog: pd.DataFrame,
    queried_ids: set[str],
    observed_mask: np.ndarray,
    posterior: BayesianMorphologyPosterior | None,
    config: RunConfig,
    rng: np.random.Generator,
    dense_signal: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    channels: list[str],
    reference,
    prediction,
    query_index: int,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, dict[str, object] | None]:
    distance_pixels = _distance_pixels(observed_mask)
    if posterior is None:
        scores = _base_candidate_scores(catalog, queried_ids, distance_pixels, config)
    else:
        eligible = catalog[~catalog["roi_id"].isin(queried_ids)].copy()
        if eligible.empty:
            raise ValueError("no feasible v4.3 raster candidates remain")
        scores = eligible.merge(
            posterior.candidate_scores,
            on="roi_id",
            how="left",
            validate="one_to_one",
        )
    fantasies = pd.DataFrame()
    reliability = None
    if policy == "bayesian_subtile_evidence_gated":
        if posterior is None:
            raise ValueError("evidence-gated Bayesian policy requires a fitted posterior")
        selected, scores = evidence_gated_rank_candidates(scores, posterior, config)
    elif policy == "bayesian_morphology_fantasy_guarded":
        if posterior is None:
            raise ValueError("morphology-fantasy Bayesian policy requires a fitted posterior")
        selected, scores, fantasies, reliability = morphology_fantasy_rank_candidates(
            scores,
            posterior,
            float(observed_mask.mean()),
            config,
            query_index,
        )
    elif policy == "uncertainty_lookahead":
        scores["selection_utility"] = scores["geometry_coverage_gain"]
        selected = scores.sort_values(
            ["selection_utility", "row0", "column0"],
            ascending=[False, True, True],
        ).iloc[0]
        scores["selected"] = scores["roi_id"] == str(selected["roi_id"])
    elif policy == "uniform":
        scores["selection_utility"] = np.nan
        selected = scores.sort_values(["row0", "column0"]).iloc[0]
        scores["selected"] = scores["roi_id"] == str(selected["roi_id"])
    elif policy == "random":
        scores["selection_utility"] = np.nan
        selected = scores.iloc[int(rng.integers(0, len(scores)))]
        scores["selected"] = scores["roi_id"] == str(selected["roi_id"])
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
        scores["selected"] = scores["roi_id"] == str(selected["roi_id"])
    else:
        raise ValueError(f"unsupported Bayesian morphology policy: {policy}")
    return selected, scores, fantasies, reliability


def run_v4_bayesian_morphology_slice_replay(
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
) -> V4BayesianMorphologySliceResult:
    """Run one fixed-budget v4.3 replay with shared evaluation and optional fantasy scoring."""

    if policy not in V4_BAYESIAN_MORPHOLOGY_POLICIES:
        raise ValueError(f"unsupported Bayesian morphology policy: {policy}")
    catalog = build_v3_roi_catalog(x, y, config.acquisition_v4.roi_size_px)
    if config.acquisition_v4.total_rois > len(catalog):
        raise ValueError("v4 total_rois exceeds ROI catalog size")
    rng = np.random.default_rng(seed)
    observed_mask = np.zeros(dense_signal.shape[1:], dtype=bool)
    queried_ids: set[str] = set()
    subtile_nodes: list[BayesianSubtileObservation] = []
    metrics: list[dict[str, object]] = []
    candidate_frames: list[pd.DataFrame] = []
    kernel_rows: list[dict[str, object]] = []
    subtile_rows: list[dict[str, object]] = []
    fantasy_frames: list[pd.DataFrame] = []
    reliability_rows: list[dict[str, object]] = []
    gp_rows: list[dict[str, object]] = []
    consumed_time_s = 0.0
    consumed_dose = 0.0
    posterior = None
    reference = pseudo_reference_from_dense_signal(dense_signal, x, y, channels, config)
    prediction = reconstruct_from_observed_mask(dense_signal, observed_mask, x, y, channels, config)

    def frames() -> tuple[pd.DataFrame, ...]:
        return (
            pd.DataFrame(metrics),
            pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame(),
            pd.DataFrame(kernel_rows),
            pd.DataFrame(subtile_rows),
            pd.concat(fantasy_frames, ignore_index=True) if fantasy_frames else pd.DataFrame(),
            pd.DataFrame(reliability_rows),
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
        if policy in V4_BAYESIAN_MORPHOLOGY_PRIMARY_POLICIES:
            observations = subtile_observations_from_revealed_roi(
                roi, dense_signal, x, y, len(metrics) + 1, config
            )
            subtile_nodes.extend(observations)
            subtile_rows.extend(_subtile_records(fold_id, slice_id, policy, observations, channels))
        prediction = reconstruct_from_observed_mask(
            dense_signal, observed_mask, x, y, channels, config
        )
        if (
            policy in V4_BAYESIAN_MORPHOLOGY_PRIMARY_POLICIES
            and len(metrics) + 1 >= config.acquisition_v4.pilot_rois
        ):
            posterior = fit_bayesian_morphology_posterior(
                subtile_nodes,
                catalog,
                x,
                y,
                config,
                _distance_pixels(observed_mask),
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
        if posterior is not None and len(metrics) == config.acquisition_v4.total_rois:
            gp_prediction = gp_prediction_from_posterior(posterior, x, y, channels, config)
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

    for index in rng.choice(len(catalog), size=config.acquisition_v4.pilot_rois, replace=False):
        reveal(catalog.iloc[int(index)], "random_pilot")
    while len(metrics) < config.acquisition_v4.total_rois:
        selected, scored, fantasies, reliability = _score_candidates(
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
            len(metrics) + 1,
        )
        scored.insert(0, "fold", fold_id)
        scored.insert(1, "slice", slice_id)
        scored.insert(2, "policy", policy)
        scored.insert(3, "query_index", len(metrics) + 1)
        candidate_frames.append(scored)
        if not fantasies.empty:
            fantasies.insert(0, "fold", fold_id)
            fantasies.insert(1, "slice", slice_id)
            fantasies.insert(2, "policy", policy)
            fantasy_frames.append(fantasies)
        if reliability is not None:
            reliability_rows.append(
                {"fold": fold_id, "slice": slice_id, "policy": policy, **reliability}
            )
        reveal(selected, f"{policy}_adaptive")
    result_frames = frames()
    return V4BayesianMorphologySliceResult(*result_frames)


def _mean_ci(values: pd.Series) -> tuple[float, float, float]:
    clean = values.dropna().to_numpy(float)
    if clean.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(clean))
    if clean.size == 1:
        return mean, np.nan, np.nan
    margin = float(t.ppf(0.975, clean.size - 1) * np.std(clean, ddof=1) / np.sqrt(clean.size))
    return mean, mean - margin, mean + margin


def paired_morphology_comparisons(metrics: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    final = metrics.sort_values("iteration").groupby(["fold", "slice", "policy"], sort=False).tail(1)
    baseline = final[final["policy"] == "uncertainty_lookahead"].set_index(["fold", "slice"])
    rows = []
    for policy in sorted(set(final["policy"]) - {"uncertainty_lookahead", "oracle_composite_gain"}):
        candidate = final[final["policy"] == policy].set_index(["fold", "slice"])
        joined = candidate.join(baseline, lsuffix="_candidate", rsuffix="_baseline", how="inner")
        delta = (
            joined["morphology_composite_error_candidate"]
            - joined["morphology_composite_error_baseline"]
        )
        rmse_delta = (
            joined["normalized_reconstruction_rmse_candidate"]
            / joined["normalized_reconstruction_rmse_baseline"]
            - 1.0
        )
        mean, low, high = _mean_ci(delta)
        equal_cost = bool(
            np.allclose(joined["scan_time_s_candidate"], joined["scan_time_s_baseline"], atol=1.0e-9)
        )
        leave_one_out = (
            max(float((delta.sum() - value) / (len(delta) - 1)) for value in delta)
            if len(delta) > 1
            else np.nan
        )
        fold_means = delta.groupby(level="fold").mean()
        nonpositive_folds = int((fold_means <= 0.0).sum())
        rmse_regression = float(rmse_delta.mean())
        maximum_regression = float(delta.max())
        rows.append(
            {
                "policy": policy,
                "baseline": "uncertainty_lookahead",
                "paired_slices": len(joined),
                "mean_composite_error_delta": mean,
                "median_composite_error_delta": float(delta.median()),
                "composite_error_delta_ci95_low": low,
                "composite_error_delta_ci95_high": high,
                "composite_error_win_rate": float((delta < 0.0).mean()),
                "leave_one_slice_out_worst_mean_delta": leave_one_out,
                "nonpositive_fold_means": nonpositive_folds,
                "mean_rmse_regression_fraction": rmse_regression,
                "maximum_slice_composite_error_regression": maximum_regression,
                "equal_mean_scan_cost": equal_cost,
                "passes_ten_slice_gate": bool(
                    mean <= 0.0
                    and rmse_regression <= config.acquisition_v4.rmse_regression_limit_fraction
                    and maximum_regression <= 0.02
                    and equal_cost
                ),
                "passes_thirty_slice_gate": bool(
                    mean < 0.0
                    and float(delta.median()) <= 0.0
                    and leave_one_out <= 0.0
                    and nonpositive_folds >= 3
                    and rmse_regression <= config.acquisition_v4.rmse_regression_limit_fraction
                    and maximum_regression <= 0.02
                    and equal_cost
                ),
                "promoted": bool(
                    high < 0.0
                    and rmse_regression <= config.acquisition_v4.rmse_regression_limit_fraction
                    and equal_cost
                ),
            }
        )
    return (
        pd.DataFrame(rows).sort_values("mean_composite_error_delta")
        if rows
        else pd.DataFrame()
    )


def _deduplicate_fantasies(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["slice"] = combined["slice"].astype(str).str.zfill(3)
    return combined.drop_duplicates(
        ["fold", "slice", "policy", "query_index", "roi_id", "fantasy_index"],
        keep="last",
    )


def _deduplicate_reliability(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["slice"] = combined["slice"].astype(str).str.zfill(3)
    return combined.drop_duplicates(["fold", "slice", "policy", "query_index"], keep="last")


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


def run_v4_bayesian_morphology_stack_validation(
    template_path: Path,
    output: Path,
    manifest_path: Path,
    fold_specification: str = "all",
    slice_ids: list[str] | None = None,
    policies: list[str] | None = None,
    seed: int = 0,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run resumable v4.3 morphology-aware Bayesian validation."""

    template = load_config(template_path)
    all_slices = _manifest_slice_ids(manifest_path)
    requested = set(slice_ids or all_slices)
    folds = build_v4_folds(all_slices, template)
    if fold_specification != "all":
        folds = [fold for fold in folds if fold.fold_id == f"fold_{int(fold_specification)}"]
    policies = policies or V4_BAYESIAN_MORPHOLOGY_POLICIES
    unknown = set(policies) - set(V4_BAYESIAN_MORPHOLOGY_POLICIES)
    if unknown:
        raise ValueError(f"unsupported Bayesian morphology policies: {sorted(unknown)}")
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
        "metrics": output / "v4_bayesian_morphology_metrics_parts",
        "candidate": output / "v4_bayesian_morphology_candidate_parts",
        "kernel": output / "v4_bayesian_morphology_kernel_parts",
        "subtile": output / "v4_bayesian_morphology_subtile_parts",
        "fantasy": output / "v4_bayesian_morphology_fantasy_parts",
        "reliability": output / "v4_bayesian_morphology_reliability_parts",
        "gp": output / "v4_bayesian_morphology_gp_parts",
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

                result = run_v4_bayesian_morphology_slice_replay(
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
                    "fantasy": result.fantasy_trace,
                    "reliability": result.reliability_trace,
                    "gp": result.gp_diagnostics,
                }
                for name, frame in result_frames.items():
                    if not frame.empty:
                        frames[name].append(frame)
            processed += 1
            if processed % 10 == 0:
                print(f"Validated Bayesian morphology v4.3 reconstruction on {processed} requested slices.")
    metrics = _deduplicate_metrics(frames["metrics"])
    candidates = _deduplicate_trace(frames["candidate"])
    kernels = _deduplicate_kernel(frames["kernel"])
    subtiles = _deduplicate_subtiles(frames["subtile"])
    fantasies = _deduplicate_fantasies(frames["fantasy"])
    reliability = _deduplicate_reliability(frames["reliability"])
    gp = _deduplicate_gp(frames["gp"])
    summary, curves, auc = summarize_v4_metrics(metrics)
    final = metrics.sort_values("iteration").groupby(["fold", "slice", "policy"], sort=False).tail(1)
    comparisons = paired_morphology_comparisons(metrics, template)
    metrics.to_csv(output / "v4_bayesian_morphology_metrics_by_iteration.csv", index=False)
    final.to_csv(output / "v4_bayesian_morphology_final_metrics_by_slice.csv", index=False)
    summary.to_csv(output / "v4_bayesian_morphology_oof_summary.csv", index=False)
    comparisons.to_csv(output / "v4_bayesian_morphology_paired_comparisons.csv", index=False)
    curves.to_csv(output / "v4_bayesian_morphology_error_vs_cost_curves.csv", index=False)
    auc.to_csv(output / "v4_bayesian_morphology_composite_error_auc_vs_cost.csv", index=False)
    candidates.to_csv(output / "v4_bayesian_morphology_candidate_trace.csv", index=False)
    kernels.to_csv(output / "v4_bayesian_morphology_kernel_weight_trace.csv", index=False)
    subtiles.to_csv(output / "v4_bayesian_morphology_subtile_trace.csv", index=False)
    fantasies.to_csv(output / "v4_bayesian_morphology_fantasy_trace.csv", index=False)
    reliability.to_csv(output / "v4_bayesian_morphology_reliability_trace.csv", index=False)
    gp.to_csv(output / "v4_bayesian_morphology_gp_diagnostics.csv", index=False)
    final[final["policy"] == "oracle_composite_gain"].to_csv(
        output / "v4_bayesian_morphology_oracle_headroom_summary.csv", index=False
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
    ).to_csv(output / "v4_bayesian_morphology_slice_block_summary.csv", index=False)
    protocol = {
        "schema": "balance_nm_v4_3_evidence_gated_bayesian_morphology_lookahead",
        "template_config": str(template_path),
        "manifest": str(manifest_path),
        "seed": seed,
        "requested_slices": sorted(requested),
        "policies": policies,
        "folds": [fold.__dict__ for fold in folds],
        "dense_truth_policy": "hidden from deployable selectors; evaluation-only replay reference",
        "shared_evaluator": "nearest-observation reconstruction for every primary comparison arm",
        "gp_diagnostics": "diagnostic only; excluded from promotion decisions",
        "front_semantics": "frozen unsupervised alteration-front proxy; not expert-labeled corrosion truth",
        "historical_ablation_command": "validate-v4-bayesian-residual-stack",
    }
    with (output / "v4_bayesian_morphology_fold_protocol.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(protocol, handle, sort_keys=False)
    return metrics, summary
