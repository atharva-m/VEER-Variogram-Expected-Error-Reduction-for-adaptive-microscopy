"""V4.5 additive Pareto Bayesian EIVR validation for Alloy 617 replay."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import label

from .data import ingest_dataset
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
    pareto_additive_select_candidate,
    pareto_select_candidate,
    pareto_subtile_observations_from_revealed_roi,
    parse_pareto_additive_policy,
    parse_pareto_policy,
)
from .v4_bayesian_pareto_validation import (
    V4BayesianParetoSliceResult,
    _base_candidate_scores,
    _deduplicate_gp,
    _deduplicate_kernel,
    _deduplicate_subtiles,
    _kernel_records,
    _subtile_records,
    spec_feature_families,
)
from .v4_validation import (
    _checkpoint_part_path,
    _deduplicate_metrics,
    _deduplicate_trace,
    _distance_pixels,
    _manifest_slice_ids,
    _read_checkpoint_frames,
    _score_v4_prediction,
    _write_checkpoint_part,
    build_v4_folds,
    summarize_v4_metrics,
)

V4_BAYESIAN_ADDITIVE_ARMS = [
    "bayesian_pareto_additive_eivr_4x4_mean_tau090_alpha1",
    "bayesian_pareto_additive_eivr_4x4_mean_tau090_alpha2",
    "bayesian_pareto_additive_eivr_4x4_mean_tau090_alpha5",
    "bayesian_pareto_additive_eivr_4x4_mean_tau090_alpha10",
]
V4_BAYESIAN_ADDITIVE_POLICIES = [
    "uncertainty_lookahead",
    "bayesian_pareto_eivr_4x4_mean_tau090",
    *V4_BAYESIAN_ADDITIVE_ARMS,
]


def _policy_spec(policy: str):
    if policy == "bayesian_pareto_eivr_4x4_mean_tau090":
        return parse_pareto_policy(policy)
    if policy in V4_BAYESIAN_ADDITIVE_ARMS:
        return parse_pareto_additive_policy(policy).pareto_spec
    return None


def _score_candidates(
    policy: str,
    catalog: pd.DataFrame,
    queried_ids: set[str],
    observed_mask: np.ndarray,
    posterior: ParetoPosterior | None,
    config,
    rng: np.random.Generator,
) -> tuple[pd.Series, pd.DataFrame]:
    distance_pixels = _distance_pixels(observed_mask)
    if policy == "bayesian_pareto_eivr_4x4_mean_tau090":
        if posterior is None:
            raise ValueError("v4.5 historical Pareto comparator requires a fitted posterior")
        return pareto_select_candidate(catalog, queried_ids, distance_pixels, posterior, config)
    if policy in V4_BAYESIAN_ADDITIVE_ARMS:
        if posterior is None:
            raise ValueError("v4.5 additive Pareto policies require a fitted posterior")
        additive = parse_pareto_additive_policy(policy)
        return pareto_additive_select_candidate(
            catalog,
            queried_ids,
            distance_pixels,
            posterior,
            config,
            additive.exchange_rate,
        )
    scores = _base_candidate_scores(catalog, queried_ids, distance_pixels, config)
    if policy == "uncertainty_lookahead":
        scores["selection_utility"] = scores["geometry_gain_per_cost"]
        selected = scores.sort_values(
            ["selection_utility", "row0", "column0"],
            ascending=[False, True, True],
        ).iloc[0]
        scores["selected"] = scores["roi_id"] == str(selected["roi_id"])
        return selected, scores
    raise ValueError(f"unsupported v4.5 additive policy: {policy}")


def run_v4_bayesian_additive_slice_replay(
    config,
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
    """Run one fixed-budget v4.5 additive Pareto replay."""

    if policy not in V4_BAYESIAN_ADDITIVE_POLICIES:
        raise ValueError(f"unsupported v4.5 additive policy: {policy}")
    spec = _policy_spec(policy)
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
            posterior = fit_pareto_subtile_posterior(subtile_nodes, catalog, x, y, config, spec)
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
            gp_prediction = gp_prediction_from_pareto_posterior(posterior, x, y, channels, config)
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
        selected, scored = _score_candidates(
            policy, catalog, queried_ids, observed_mask, posterior, config, rng
        )
        scored.insert(0, "fold", fold_id)
        scored.insert(1, "slice", slice_id)
        scored.insert(2, "policy", policy)
        scored.insert(3, "query_index", len(metrics) + 1)
        candidate_frames.append(scored)
        reveal(selected, f"{policy}_adaptive")
    return V4BayesianParetoSliceResult(*frames())


def _front_runs(depth: np.ndarray) -> list[int]:
    finite = np.isfinite(depth)
    runs = []
    start = None
    for index, value in enumerate(finite):
        if value and start is None:
            start = index
        if (not value or index == len(finite) - 1) and start is not None:
            stop = index if not value else index + 1
            runs.append(stop - start)
            start = None
    return runs


def _pseudo_reference_descriptors(slice_id: str, dense_signal, x, y, channels, config) -> dict[str, object]:
    reference = pseudo_reference_from_dense_signal(dense_signal, x, y, channels, config)
    mask = reference["pseudo_altered_region"].values.astype(bool)
    front = reference["pseudo_front"].values.astype(bool)
    depth = reference["pseudo_penetration_depth_nm"].values.astype(float)
    labels, count = label(mask)
    areas = [int(np.sum(labels == label_id)) for label_id in range(1, count + 1)]
    finite = depth[np.isfinite(depth)]
    runs = _front_runs(depth)
    depth_std = float(np.nanstd(depth)) if finite.size else np.nan
    d95 = float(np.nanpercentile(depth, 95.0)) if finite.size else np.nan
    dmax = float(np.nanmax(depth)) if finite.size else np.nan
    return {
        "slice": slice_id,
        "altered_fraction": float(mask.mean()),
        "front_fraction_rows": float(np.isfinite(depth).mean()),
        "front_pixels": int(front.sum()),
        "component_count": int(count),
        "largest_component_share": float(max(areas) / sum(areas)) if areas and sum(areas) else 0.0,
        "small_component_count": int(sum(1 for area in areas if area < 0.01 * mask.size)),
        "d95_nm": d95,
        "dmax_nm": dmax,
        "depth_std_nm": depth_std,
        "front_run_count": len(runs),
        "longest_front_run_rows": max(runs) if runs else 0,
        "is_saturated_flat_front": bool(
            finite.size
            and np.isfinite(depth_std)
            and depth_std <= 1.0e-9
            and np.isfinite(d95)
            and np.isfinite(dmax)
            and abs(d95 - dmax) <= 1.0e-9
            and np.isfinite(depth).mean() >= 0.99
        ),
    }


def _flat_front_audit(final: pd.DataFrame, loader, template) -> pd.DataFrame:
    if final.empty:
        return pd.DataFrame()
    baseline = final[final["policy"] == "uncertainty_lookahead"].set_index(["fold", "slice"])
    rows = []
    for (fold, slice_id), base_row in baseline.iterrows():
        dense_signal, x, y, channels = loader(str(slice_id).zfill(3))
        config = v3_config_for_slice(template, str(slice_id).zfill(3), loader.sources[str(slice_id).zfill(3)])
        row = {"fold": fold, **_pseudo_reference_descriptors(str(slice_id).zfill(3), dense_signal, x, y, channels, config)}
        for _, candidate in final[
            (final["fold"] == fold)
            & (final["slice"].astype(str).str.zfill(3) == str(slice_id).zfill(3))
            & (final["policy"] != "uncertainty_lookahead")
        ].iterrows():
            delta = float(candidate["morphology_composite_error"]) - float(
                base_row["morphology_composite_error"]
            )
            safe_name = str(candidate["policy"]).replace("bayesian_pareto_", "")
            row[f"delta__{safe_name}"] = delta
            row[f"heavy_regression__{safe_name}"] = bool(delta > 0.02)
        rows.append(row)
    return pd.DataFrame(rows)


def run_v4_bayesian_additive_stack_validation(
    template_path: Path,
    output: Path,
    manifest_path: Path,
    fold_specification: str = "all",
    slice_ids: list[str] | None = None,
    policies: list[str] | None = None,
    seed: int = 0,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run blocked, resumable v4.5 additive Pareto validation."""

    template = load_config(template_path)
    all_slices = _manifest_slice_ids(manifest_path)
    requested = set(slice_ids or all_slices)
    folds = build_v4_folds(all_slices, template)
    if fold_specification != "all":
        folds = [fold for fold in folds if fold.fold_id == f"fold_{int(fold_specification)}"]
    policies = policies or V4_BAYESIAN_ADDITIVE_POLICIES
    unknown = set(policies) - set(V4_BAYESIAN_ADDITIVE_POLICIES)
    if unknown:
        raise ValueError(f"unsupported v4.5 additive policies: {sorted(unknown)}")
    sources = sources_from_manifest(manifest_path, all_slices, template.scenario.elements)
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}

    def load_slice(slice_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        if slice_id not in cache:
            config = v3_config_for_slice(template, slice_id, sources[slice_id])
            observations, _ = ingest_dataset(config)
            cache[slice_id] = dense_signal_from_observations(config, observations)
        return cache[slice_id]

    load_slice.sources = sources  # type: ignore[attr-defined]
    output.mkdir(parents=True, exist_ok=True)
    write_config(template, output / "resolved_template_config.yaml")
    directories = {
        "metrics": output / "v4_bayesian_additive_metrics_parts",
        "candidate": output / "v4_bayesian_additive_candidate_parts",
        "kernel": output / "v4_bayesian_additive_kernel_parts",
        "subtile": output / "v4_bayesian_additive_subtile_parts",
        "gp": output / "v4_bayesian_additive_gp_parts",
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
    processed = 0
    for fold in folds:
        tests = [slice_id for slice_id in fold.test_slices if slice_id in requested]
        for slice_id in tests:
            config = v3_config_for_slice(template, slice_id, sources[slice_id])
            dense_signal, x, y, channels = load_slice(slice_id)
            for policy in policies:
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

                result = run_v4_bayesian_additive_slice_replay(
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
                print(f"Validated v4.5 additive Pareto Bayesian reconstruction on {processed} requested slices.")
    metrics = _deduplicate_metrics(frames["metrics"])
    candidates = _deduplicate_trace(frames["candidate"])
    kernels = _deduplicate_kernel(frames["kernel"])
    subtiles = _deduplicate_subtiles(frames["subtile"])
    gp = _deduplicate_gp(frames["gp"])
    summary, curves, auc = summarize_v4_metrics(metrics)
    final = metrics.sort_values("iteration").groupby(["fold", "slice", "policy"], sort=False).tail(1)
    comparisons = paired_morphology_comparisons(metrics, template)
    flat_front_audit = _flat_front_audit(final, load_slice, template)
    metrics.to_csv(output / "v4_bayesian_additive_metrics_by_iteration.csv", index=False)
    final.to_csv(output / "v4_bayesian_additive_final_metrics_by_slice.csv", index=False)
    summary.to_csv(output / "v4_bayesian_additive_oof_summary.csv", index=False)
    comparisons.to_csv(output / "v4_bayesian_additive_paired_comparisons.csv", index=False)
    curves.to_csv(output / "v4_bayesian_additive_error_vs_cost_curves.csv", index=False)
    auc.to_csv(output / "v4_bayesian_additive_composite_error_auc_vs_cost.csv", index=False)
    candidates.to_csv(output / "v4_bayesian_additive_candidate_trace.csv", index=False)
    kernels.to_csv(output / "v4_bayesian_additive_kernel_weight_trace.csv", index=False)
    subtiles.to_csv(output / "v4_bayesian_additive_subtile_trace.csv", index=False)
    gp.to_csv(output / "v4_bayesian_additive_gp_diagnostics.csv", index=False)
    flat_front_audit.to_csv(output / "v4_bayesian_additive_flat_front_audit.csv", index=False)
    protocol = {
        "schema": "balance_nm_v4_5_additive_pareto_bayesian_eivr",
        "template_config": str(template_path),
        "manifest": str(manifest_path),
        "seed": seed,
        "requested_slices": sorted(requested),
        "policies": policies,
        "folds": [fold.__dict__ for fold in folds],
        "dense_truth_policy": "hidden from deployable selectors; evaluation-only replay reference",
        "shared_evaluator": "nearest-observation reconstruction for every primary comparison arm",
        "bayesian_order": "geometry shortlist first; EIVR only for shortlisted candidates plus geometry argmax",
        "additive_utility": "utility = normalized_geometry_gain + alpha * EIVR_LCB",
        "gp_diagnostics": "endpoint diagnostic only; excluded from promotion decisions",
        "front_semantics": "frozen unsupervised alteration-front proxy; not expert-labeled corrosion truth",
        "historical_ablation_command": "validate-v4-bayesian-pareto-stack",
    }
    with (output / "v4_bayesian_additive_fold_protocol.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(protocol, handle, sort_keys=False)
    return metrics, summary
