"""Variogram expected-error-reduction (VEER) acquisition validation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from .data import ingest_dataset
from .io import load_config, write_config
from .morphology import (
    dense_signal_from_observations,
    pseudo_reference_from_dense_signal,
    reconstruct_from_observed_mask,
)
from .replay import (
    raster_cost,
    build_roi_catalog,
    sources_from_manifest,
    config_for_slice,
)
from .replay import paired_morphology_comparisons
from .features import (
    SubtileSpec,
    SubtileObservation,
    subtile_observations_from_revealed_roi,
)
from .replay import (
    checkpoint_part_path,
    deduplicate_metrics,
    deduplicate_trace,
    distance_pixels,
    manifest_slice_ids,
    read_checkpoint_frames,
    score_prediction,
    write_checkpoint_part,
    build_folds,
    lookahead_coverage_gain,
    summarize_metrics,
)
from .selection import (
    NestedPolicySpec,
    VeerPolicySpec,
    front_movement_fraction,
    front_probability_weights,
    front_relevance_weights,
    nested_veer_select_candidate,
    parse_nested_policy,
    parse_veer_policy,
    predicted_depth_profile,
    veer_select_candidate,
)
from .variogram import (
    NestedVariogramFit,
    VariogramPosterior,
    fit_nested_variogram,
    fit_variogram_posterior,
)

VEER_ARMS = [
    "variogram_eer_4x4_mean_kappa0",
    "variogram_eer_4x4_mean_kappa2",
    "variogram_eer_4x4_mean_kappa5",
]
NESTED_ARMS = [
    "nested_veer_4x4_mean_kappa0",
    "nested_veer_4x4_mean_kappa2",
    "nested_veer_4x4_mean_kappa5",
    "nested_veer_4x4_mean_kappa10",
    "nested_band_veer_4x4_mean_kappa2",
    "nested_band_veer_4x4_mean_kappa5",
]
GATED_ARMS = [
    "gated_veer_4x4_mean_kappa5",
    "gated_veer_4x4_mean_kappa10",
]
VEER_POLICIES = [
    "uncertainty_lookahead",
    *VEER_ARMS,
    *NESTED_ARMS,
    *GATED_ARMS,
]


@dataclass
class VeerSliceResult:
    metrics: pd.DataFrame
    candidate_trace: pd.DataFrame
    variogram_trace: pd.DataFrame


def _veer_spec(policy: str) -> VeerPolicySpec | None:
    if policy in VEER_ARMS or policy in GATED_ARMS:
        return parse_veer_policy(policy)
    return None


def _nested_spec(policy: str) -> NestedPolicySpec | None:
    if policy in NESTED_ARMS:
        return parse_nested_policy(policy)
    return None


def _subtile_spec(policy: str) -> SubtileSpec:
    return SubtileSpec(grid_shape=(4, 4), feature_mode="mean")


def _variogram_records(
    fold_id: str,
    slice_id: str,
    policy: str,
    iteration: int,
    posterior: VariogramPosterior,
) -> list[dict[str, object]]:
    return [
        {
            "fold": fold_id,
            "slice": slice_id,
            "policy": policy,
            "iteration": iteration,
            "kernel": label,
            "length_scale_x": scales[0],
            "length_scale_y": scales[1],
            "log_marginal_likelihood": float(likelihood),
            "weight": float(weight),
            "untempered_weight": float(untempered),
            "temper": posterior.temper,
            "sill": posterior.sill,
            "effective_components": posterior.effective_components,
            "subtile_count": posterior.subtile_count,
        }
        for label, scales, likelihood, weight, untempered in zip(
            posterior.kernel_labels,
            posterior.kernel_length_scales,
            posterior.log_marginal_likelihoods,
            posterior.weights,
            posterior.untempered_weights,
        )
    ]


def _nested_records(
    fold_id: str,
    slice_id: str,
    policy: str,
    iteration: int,
    fit: NestedVariogramFit,
) -> list[dict[str, object]]:
    return [
        {
            "fold": fold_id,
            "slice": slice_id,
            "policy": policy,
            "iteration": iteration,
            "kernel": "nested_wls",
            "length_scale_x": fit.length_scale,
            "length_scale_y": fit.length_scale,
            "weight": 1.0,
            "untempered_weight": 1.0,
            "nugget": fit.nugget,
            "matern_amplitude": fit.matern_amplitude,
            "linear_slope": fit.linear_slope,
            "weighted_sse": fit.weighted_sse,
            "bins_used": int(fit.bin_distances.size),
            "subtile_count": fit.subtile_count,
        }
    ]


def trailing_endpoint_summary(metrics: pd.DataFrame, window: int) -> pd.DataFrame:
    """Pre-registered co-primary endpoint: trailing-median composite error.

    The per-reveal front extraction is discontinuous (single reveals can move
    the extracted front by tens of micrometers in either direction), so the
    final-iteration composite carries metric noise of the same order as the
    policy effects under study. The median over the final `window` reveals is
    a robust functional of the same trajectory.
    """

    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for (fold, slice_id, policy), group in metrics.groupby(
        ["fold", "slice", "policy"], sort=False
    ):
        tail = group.sort_values("iteration").tail(window)
        rows.append(
            {
                "fold": fold,
                "slice": slice_id,
                "policy": policy,
                "trailing_median_composite": float(tail["morphology_composite_error"].median()),
                "trailing_std_composite": float(tail["morphology_composite_error"].std(ddof=0)),
                "final_composite": float(
                    group.sort_values("iteration")["morphology_composite_error"].iloc[-1]
                ),
                "iterations_in_window": len(tail),
            }
        )
    frame = pd.DataFrame(rows)
    baseline = (
        frame[frame["policy"] == "uncertainty_lookahead"]
        .set_index(["fold", "slice"])["trailing_median_composite"]
    )
    frame["trailing_median_delta_vs_uncertainty"] = [
        row["trailing_median_composite"] - baseline.get((row["fold"], row["slice"]), np.nan)
        for _, row in frame.iterrows()
    ]
    return frame


def _deduplicate_variogram(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["slice"] = combined["slice"].astype(str).str.zfill(3)
    return combined.drop_duplicates(
        ["fold", "slice", "policy", "iteration", "kernel"], keep="last"
    )


def run_veer_slice_replay(
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
) -> VeerSliceResult:
    """Run one fixed-budget VEER replay with per-slice paired pilots."""

    if policy not in VEER_POLICIES:
        raise ValueError(f"unsupported VEER policy: {policy}")
    spec = _veer_spec(policy)
    nested = _nested_spec(policy)
    needs_subtiles = spec is not None or nested is not None
    catalog = build_roi_catalog(x, y, config.acquisition.roi_size_px)
    if config.acquisition.total_rois > len(catalog):
        raise ValueError("v5 total_rois exceeds ROI catalog size")
    rng = np.random.default_rng([seed, int(slice_id)])
    rectangle_cache: dict = {}
    observed_mask = np.zeros(dense_signal.shape[1:], dtype=bool)
    queried_ids: set[str] = set()
    subtile_nodes: list[SubtileObservation] = []
    metrics: list[dict[str, object]] = []
    candidate_frames: list[pd.DataFrame] = []
    variogram_rows: list[dict[str, object]] = []
    consumed_time_s = 0.0
    consumed_dose = 0.0
    posterior: VariogramPosterior | None = None
    nested_fit: NestedVariogramFit | None = None
    previous_depth: np.ndarray | None = None
    front_movement: float | None = None
    reference = pseudo_reference_from_dense_signal(dense_signal, x, y, channels, config)
    prediction = reconstruct_from_observed_mask(dense_signal, observed_mask, x, y, channels, config)

    def frames() -> tuple[pd.DataFrame, ...]:
        return (
            pd.DataFrame(metrics),
            pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame(),
            pd.DataFrame(variogram_rows),
        )

    def checkpoint() -> None:
        if checkpoint_callback is not None:
            checkpoint_callback(*frames())

    def reveal(roi: pd.Series, stage: str) -> None:
        nonlocal consumed_time_s, consumed_dose, prediction, posterior, nested_fit
        nonlocal previous_depth, front_movement
        row0, row1 = int(roi["row0"]), int(roi["row1"])
        column0, column1 = int(roi["column0"]), int(roi["column1"])
        observed_mask[row0:row1, column0:column1] = True
        queried_ids.add(str(roi["roi_id"]))
        time_s, dose = raster_cost(config, roi)
        consumed_time_s += time_s
        consumed_dose += dose
        if needs_subtiles:
            subtile_nodes.extend(
                subtile_observations_from_revealed_roi(
                    roi, dense_signal, x, y, len(metrics) + 1, config, _subtile_spec(policy)
                )
            )
        if spec is not None:
            posterior = fit_variogram_posterior(subtile_nodes, config)
            variogram_rows.extend(
                _variogram_records(fold_id, slice_id, policy, len(metrics) + 1, posterior)
            )
        if nested is not None:
            nested_fit = fit_nested_variogram(subtile_nodes, config)
            variogram_rows.extend(
                _nested_records(fold_id, slice_id, policy, len(metrics) + 1, nested_fit)
            )
        prediction = reconstruct_from_observed_mask(dense_signal, observed_mask, x, y, channels, config)
        if spec is not None and spec.gated:
            depth = predicted_depth_profile(prediction, x, y, config)
            if previous_depth is not None:
                front_movement = front_movement_fraction(
                    previous_depth, depth, config.scenario.width_nm
                )
            previous_depth = depth
        score = score_prediction(
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
                "variogram_sill": posterior.sill if posterior is not None else np.nan,
            }
        )
        metrics.append(score)
        checkpoint()

    def select() -> tuple[pd.Series, pd.DataFrame]:
        if nested is not None:
            if nested_fit is None:
                raise ValueError("v5.1 nested policies require a fitted nested variogram")
            if nested.weight_mode == "band":
                pixel_weights = front_relevance_weights(
                    prediction, x, y, config, nested.front_kappa
                )
            else:
                pixel_weights = front_probability_weights(prediction, nested.front_kappa)
            selected, scored = nested_veer_select_candidate(
                catalog, queried_ids, observed_mask, nested_fit, pixel_weights, x, y, config,
                rectangle_cache,
            )
            scored["front_kappa"] = nested.front_kappa
            return selected, scored
        if spec is not None:
            if posterior is None:
                raise ValueError("VEER policies require a fitted variogram posterior")
            effective_kappa = spec.front_kappa
            if spec.gated:
                instability = (
                    1.0
                    if front_movement is None
                    else min(
                        1.0,
                        front_movement / config.variogram.front_gate_movement_fraction,
                    )
                )
                effective_kappa = spec.front_kappa * instability
            pixel_weights = front_relevance_weights(prediction, x, y, config, effective_kappa)
            selected, scored = veer_select_candidate(
                catalog, queried_ids, observed_mask, posterior, pixel_weights, x, y, config,
                rectangle_cache,
            )
            scored["front_kappa"] = spec.front_kappa
            if spec.gated:
                scored["front_kappa_effective"] = effective_kappa
                scored["front_movement_fraction"] = (
                    np.nan if front_movement is None else front_movement
                )
            return selected, scored
        distance_field = distance_pixels(observed_mask)
        scores = catalog[~catalog["roi_id"].isin(queried_ids)].copy()
        if scores.empty:
            raise ValueError("no feasible raster candidates remain")
        scores["geometry_gain"] = [
            lookahead_coverage_gain(distance_field, roi) for _, roi in scores.iterrows()
        ]
        scores["estimated_raster_cost_s"] = [
            raster_cost(config, roi)[0] for _, roi in scores.iterrows()
        ]
        scores["geometry_gain_per_cost"] = scores["geometry_gain"] / np.maximum(
            scores["estimated_raster_cost_s"], 1.0e-12
        )
        scores["selection_utility"] = scores["geometry_gain_per_cost"]
        selected = scores.sort_values(
            ["selection_utility", "row0", "column0"],
            ascending=[False, True, True],
        ).iloc[0]
        scores["selected"] = scores["roi_id"] == str(selected["roi_id"])
        return selected, scores

    for index in rng.choice(len(catalog), size=config.acquisition.pilot_rois, replace=False):
        reveal(catalog.iloc[int(index)], "random_pilot")
    while len(metrics) < config.acquisition.total_rois:
        selected, scored = select()
        scored.insert(0, "fold", fold_id)
        scored.insert(1, "slice", slice_id)
        scored.insert(2, "policy", policy)
        scored.insert(3, "query_index", len(metrics) + 1)
        candidate_frames.append(scored)
        reveal(selected, f"{policy}_adaptive")
    return VeerSliceResult(*frames())


_BLAS_THREAD_VARIABLES = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]


def _replay_worker(
    template_path: str,
    manifest_path: str,
    fold_id: str,
    slice_id: str,
    policy: str,
    seed: int,
    directories: dict[str, str],
) -> tuple[str, str, str]:
    """Run one slice-policy replay in a worker process, writing checkpoint parts."""

    template = load_config(Path(template_path))
    all_slices = manifest_slice_ids(Path(manifest_path))
    sources = sources_from_manifest(Path(manifest_path), all_slices, template.scenario.elements)
    config = config_for_slice(template, slice_id, sources[slice_id])
    observations, _ = ingest_dataset(config)
    dense_signal, x, y, channels = dense_signal_from_observations(config, observations)

    def checkpoint(*values) -> None:
        for name, frame in zip(directories, values):
            if not frame.empty:
                write_checkpoint_part(
                    frame,
                    checkpoint_part_path(Path(directories[name]), fold_id, slice_id, policy),
                )

    run_veer_slice_replay(
        config, dense_signal, x, y, channels, slice_id, policy, fold_id, seed, checkpoint
    )
    return fold_id, slice_id, policy


def run_veer_stack_validation(
    template_path: Path,
    output: Path,
    manifest_path: Path,
    fold_specification: str = "all",
    slice_ids: list[str] | None = None,
    policies: list[str] | None = None,
    seed: int = 0,
    resume: bool = True,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run blocked, resumable VEER validation, optionally across worker processes."""

    template = load_config(template_path)
    all_slices = manifest_slice_ids(manifest_path)
    requested = set(slice_ids or all_slices)
    folds = build_folds(all_slices, template)
    if fold_specification != "all":
        folds = [fold for fold in folds if fold.fold_id == f"fold_{int(fold_specification)}"]
    policies = policies or VEER_POLICIES
    unknown = set(policies) - set(VEER_POLICIES)
    if unknown:
        raise ValueError(f"unsupported VEER policies: {sorted(unknown)}")
    sources = sources_from_manifest(manifest_path, all_slices, template.scenario.elements)
    output.mkdir(parents=True, exist_ok=True)
    write_config(template, output / "resolved_template_config.yaml")
    directories = {
        "metrics": output / "veer_metrics_parts",
        "candidate": output / "veer_candidate_parts",
        "variogram": output / "veer_variogram_parts",
    }
    frames = {
        name: read_checkpoint_frames(directory) if resume else []
        for name, directory in directories.items()
    }
    metrics = deduplicate_metrics(frames["metrics"])
    candidates = deduplicate_trace(frames["candidate"])
    metric_complete = (
        set(
            metrics[metrics["query_count"] >= template.acquisition.total_rois][
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
    if workers > 1:
        tasks = [
            (fold.fold_id, slice_id, policy)
            for fold in folds
            for slice_id in fold.test_slices
            if slice_id in requested
            for policy in policies
            if (fold.fold_id, slice_id, policy) not in completed
        ]
        saved_environment = {key: os.environ.get(key) for key in _BLAS_THREAD_VARIABLES}
        for key in _BLAS_THREAD_VARIABLES:
            os.environ[key] = "1"
        try:
            if tasks:
                with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
                    futures = [
                        pool.submit(
                            _replay_worker,
                            str(template_path),
                            str(manifest_path),
                            fold_id,
                            slice_id,
                            policy,
                            seed,
                            {name: str(directory) for name, directory in directories.items()},
                        )
                        for fold_id, slice_id, policy in tasks
                    ]
                    done = 0
                    for future in as_completed(futures):
                        future.result()
                        done += 1
                        if done % 10 == 0:
                            print(f"Validated {done}/{len(tasks)} VEER replays.")
        finally:
            for key, value in saved_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        frames = {
            name: read_checkpoint_frames(directory)
            for name, directory in directories.items()
        }
    else:
        processed = 0
        for fold in folds:
            tests = [slice_id for slice_id in fold.test_slices if slice_id in requested]
            for slice_id in tests:
                config = config_for_slice(template, slice_id, sources[slice_id])
                pending = [
                    policy
                    for policy in policies
                    if (fold.fold_id, slice_id, policy) not in completed
                ]
                if pending:
                    observations, _ = ingest_dataset(config)
                    dense_signal, x, y, channels = dense_signal_from_observations(config, observations)
                    for policy in pending:

                        def checkpoint(*values):
                            for name, frame in zip(directories, values):
                                if not frame.empty:
                                    write_checkpoint_part(
                                        frame,
                                        checkpoint_part_path(
                                            directories[name], fold.fold_id, slice_id, policy
                                        ),
                                    )

                        result = run_veer_slice_replay(
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
                            "variogram": result.variogram_trace,
                        }
                        for name, frame in result_frames.items():
                            if not frame.empty:
                                frames[name].append(frame)
                processed += 1
                if processed % 10 == 0:
                    print(f"Validated VEER reconstruction on {processed} requested slices.")
    metrics = deduplicate_metrics(frames["metrics"])
    candidates = deduplicate_trace(frames["candidate"])
    variograms = _deduplicate_variogram(frames["variogram"])
    summary, curves, auc = summarize_metrics(metrics)
    final = metrics.sort_values("iteration").groupby(["fold", "slice", "policy"], sort=False).tail(1)
    comparisons = paired_morphology_comparisons(metrics, template)
    trailing = trailing_endpoint_summary(
        metrics, template.variogram.trailing_window_iterations
    )
    metrics.to_csv(output / "veer_metrics_by_iteration.csv", index=False)
    final.to_csv(output / "veer_final_metrics_by_slice.csv", index=False)
    summary.to_csv(output / "veer_oof_summary.csv", index=False)
    comparisons.to_csv(output / "veer_paired_comparisons.csv", index=False)
    curves.to_csv(output / "veer_error_vs_cost_curves.csv", index=False)
    auc.to_csv(output / "veer_composite_error_auc_vs_cost.csv", index=False)
    candidates.to_csv(output / "veer_candidate_trace.csv", index=False)
    variograms.to_csv(output / "veer_variogram_trace.csv", index=False)
    trailing.to_csv(output / "veer_trailing_summary.csv", index=False)
    protocol = {
        "schema": "balance_nm_variogram_expected_error_reduction",
        "template_config": str(template_path),
        "manifest": str(manifest_path),
        "seed": seed,
        "requested_slices": sorted(requested),
        "policies": policies,
        "folds": [fold.__dict__ for fold in folds],
        "dense_truth_policy": "hidden from deployable selectors; evaluation-only replay reference",
        "shared_evaluator": "nearest-observation reconstruction for every primary comparison arm",
        "objective": (
            "expected nearest-observation squared-error reduction per cost under a "
            "revealed-only model-averaged anisotropic Matern-3/2 variogram"
        ),
        "calibration": (
            "latent PCA scores standardized to unit variance; eigenvalue-weighted sill; "
            "likelihood-tempered kernel model averaging"
        ),
        "front_weighting": "w = 1 + kappa * exp(-0.5 * (front_distance_nm / bandwidth_nm)^2)",
        "nested_variogram": (
            "v5.1 arms fit gamma(d) = c0 + c1*(1 - matern32(d/l)) + c2*d by "
            "Cressie-weighted NNLS on the binned empirical semivariogram of revealed "
            "subtiles; the linear term equals the deterministic coverage gain, so the "
            "baseline is the data-selectable special case c1 = 0"
        ),
        "nested_front_weighting": (
            "v5.1 arms weight pixels by 1 + kappa * alteration_front_probability, the "
            "uncertainty-inflated front field of the shared evaluator's own prediction"
        ),
        "pilot_seeding": (
            "default_rng([seed, slice]); paired across policies within a slice, "
            "varying across slices"
        ),
        "baseline_equivalence": "gamma(d) = d with uniform weights reproduces uncertainty_lookahead",
        "co_primary_endpoint": (
            "trailing-median composite over the final "
            f"{template.variogram.trailing_window_iterations} reveals; pre-registered "
            "2026-06-11 after policy-agnostic endpoint-volatility diagnostics on the "
            "10-slice smoke, before any 30-slice gate run"
        ),
        "front_semantics": "frozen unsupervised alteration-front proxy; not expert-labeled corrosion truth",
    }
    with (output / "veer_fold_protocol.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(protocol, handle, sort_keys=False)
    return metrics, summary
