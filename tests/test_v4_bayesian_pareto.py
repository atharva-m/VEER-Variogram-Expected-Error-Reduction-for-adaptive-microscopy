from pathlib import Path
import json

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from typer.testing import CliRunner

from balance_nm.cli import app
from balance_nm.domain import RunConfig
from balance_nm.io import write_config
from balance_nm.v3_morphology import normalized_reconstruction_rmse, reconstruct_from_observed_mask
from balance_nm.v3_validation import build_v3_roi_catalog
from balance_nm.v4_bayesian_pareto import (
    ParetoPosterior,
    ParetoPolicySpec,
    ParetoSubtileObservation,
    fit_pareto_subtile_posterior,
    pareto_additive_rank_candidates,
    pareto_geometry_scores,
    pareto_rank_candidates,
    pareto_subtile_observations_from_revealed_roi,
    parse_pareto_additive_policy,
    robust_scale_feature_tensor,
    score_shortlisted_eivr,
    select_training_subtile_indices,
)
from balance_nm.v4_bayesian_additive_validation import (
    _pseudo_reference_descriptors,
    run_v4_bayesian_additive_slice_replay,
)
from balance_nm.v4_bayesian_pareto_validation import run_v4_bayesian_pareto_slice_replay
from balance_nm.v4_validation import _distance_pixels


def _write_dense_source(path: Path, values: np.ndarray) -> None:
    x = (np.arange(values.shape[2]) + 0.5) * 100.0
    y = (np.arange(values.shape[1]) + 0.5) * 100.0
    xr.Dataset(
        {"counts": (("element", "y", "x"), values)},
        coords={"element": ["Cr", "Ni"], "x": x, "y": y},
    ).to_zarr(path, mode="w")


def _signal() -> np.ndarray:
    values = np.zeros((2, 24, 24), dtype=float)
    values[0, :, :12] = 100.0
    values[1, :, 12:] = 80.0
    values[0, 4:12, 4:12] += 25.0
    values[1, 12:20, 12:20] += 40.0
    return values


def _grid():
    x = (np.arange(24) + 0.5) * 100.0
    y = (np.arange(24) + 0.5) * 100.0
    return x, y


def _config(small_config, source: Path, *, total_rois: int = 4) -> RunConfig:
    raw = small_config.model_dump(mode="python")
    raw["schema_version"] = 4
    raw["scenario"].update({"elements": ["Cr", "Ni"], "segregation_element": "Cr"})
    raw["scenario"].pop("fuel_composition", None)
    raw["scenario"].pop("cladding_composition", None)
    raw["instrument"].update(
        {
            "sensitivity": {"Cr": 1.0, "Ni": 1.0},
            "background_rate": {"Cr": 0.0, "Ni": 0.0},
        }
    )
    raw["dataset"] = {
        "mode": "replay",
        "adapter": "generic_element_map",
        "source": source,
        "value_semantics": "intensity_proxy",
        "x_step_nm": 100.0,
        "y_step_nm": 100.0,
        "dwell_ms": 1.0,
    }
    raw["task"] = {
        "mode": "corrosion_morphology_reconstruction",
        "data_semantics": "intensity_proxy",
        "label_status": "unannotated",
    }
    raw["morphology"] = {
        "reference_method": "frozen_unsupervised",
        "state_model": "spatial_gmm",
        "front_extraction": "boundary_contour",
        "penetration_axis": "x",
        "surface_side": "left",
        "smoothing_sigma_px": 0.0,
        "minimum_altered_fraction": 0.0,
    }
    raw["acquisition_v4"] = {
        "roi_size_px": [8, 8],
        "pilot_rois": 2,
        "total_rois": total_rois,
        "excluded_channels": ["CPS"],
        "oracle_sample_slices": 1,
        "bayesian_pareto": {
            "geometry_shortlist_ratios": [0.90, 0.85],
            "minimum_kernel_support": 0.90,
            "latent_components": 2,
            "kernel_catalog": [[0.1, 0.1], [0.2, 0.2]],
            "gp_reconstruction": {"enabled": True, "chunk_pixels": 128},
        },
        "folds": {
            "outer_test_ranges": [[1, 1]],
            "outer_guard_slices": 0,
            "validation_slices": 1,
            "validation_guard_slices": 0,
        },
    }
    raw["objectives"] = {"weights": {"gradient": 1.0}}
    return RunConfig.model_validate(raw)


def _observed_mask(catalog: pd.DataFrame, indices: list[int]) -> np.ndarray:
    mask = np.zeros((24, 24), dtype=bool)
    for index in indices:
        roi = catalog.iloc[index]
        mask[int(roi["row0"]) : int(roi["row1"]), int(roi["column0"]) : int(roi["column1"])] = True
    return mask


def _pareto_posterior(signal: np.ndarray, config: RunConfig, spec: ParetoPolicySpec, indices: list[int]):
    x, y = _grid()
    catalog = build_v3_roi_catalog(x, y, (8, 8))
    nodes = []
    for sequence, index in enumerate(indices, start=1):
        nodes.extend(
            pareto_subtile_observations_from_revealed_roi(
                catalog.iloc[index], signal, x, y, sequence, config, spec
            )
        )
    return fit_pareto_subtile_posterior(nodes, catalog, x, y, config, spec), catalog


def _dummy_posterior(weights: np.ndarray | None = None) -> ParetoPosterior:
    return ParetoPosterior(
        spec=ParetoPolicySpec("dummy", (4, 4), "mean", 0.90),
        kernel_labels=["k1", "k2"],
        posterior_weights=np.asarray([0.5, 0.5]) if weights is None else weights,
        kernels=[],
        subtile_catalog=pd.DataFrame(),
        evaluation_groups={},
        training_points=np.empty((0, 2)),
        evaluation_points=np.empty((0, 2)),
        effective_components=1,
        feature_families=["mean"],
        feature_median=np.zeros((1, 1)),
        feature_iqr=np.ones((1, 1)),
        pca_components=np.ones((1, 1)),
        pca_mean=np.zeros(1),
        integrated_variance=1.0,
        retained_training_count=0,
        total_revealed_subtiles=0,
    )


def _rank_scores(rows: list[tuple[str, float, tuple[float, float]]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "roi_id": roi_id,
                "row0": index,
                "column0": index,
                "geometry_gain": geometry,
                "geometry_gain_per_cost": geometry,
                "normalized_geometry_gain": geometry / 100.0,
                "geometry_shortlist_eligible": True,
                "geometry_argmax": index == 0,
                "shortlist_size": len(rows),
                "eivr_evaluated": True,
                "EIVR_by_kernel": json.dumps({"k1": eivr[0], "k2": eivr[1]}),
                "model_averaged_fractional_EIVR": float(np.mean(eivr)),
            }
            for index, (roi_id, geometry, eivr) in enumerate(rows)
        ]
    )
    frame["geometry_threshold"] = 0.90
    return frame


def test_geometry_shortlist_is_computed_before_eivr_and_limits_evaluation(small_config, tmp_path: Path):
    source = tmp_path / "dense.zarr"
    signal = _signal()
    _write_dense_source(source, signal)
    config = _config(small_config, source)
    spec = ParetoPolicySpec("bayesian_pareto_eivr_4x4_mean_tau090", (4, 4), "mean", 0.90)
    posterior, catalog = _pareto_posterior(signal, config, spec, [0, 4])
    mask = _observed_mask(catalog, [0, 4])
    distance = _distance_pixels(mask)
    loose = pareto_geometry_scores(catalog, {"r0000_c0000", "r0008_c0008"}, distance, config, 0.85)
    medium = pareto_geometry_scores(catalog, {"r0000_c0000", "r0008_c0008"}, distance, config, 0.90)
    strict = pareto_geometry_scores(catalog, {"r0000_c0000", "r0008_c0008"}, distance, config, 0.995)
    assert loose["geometry_shortlist_eligible"].sum() >= medium["geometry_shortlist_eligible"].sum()
    assert medium["geometry_shortlist_eligible"].sum() >= strict["geometry_shortlist_eligible"].sum()
    scored = score_shortlisted_eivr(medium, posterior, config)
    assert set(scored[scored["eivr_evaluated"]]["roi_id"]) == set(
        medium[medium["geometry_shortlist_eligible"]]["roi_id"]
    )
    assert set(scored[~scored["eivr_evaluated"]]["EIVR_by_kernel"]) == {"{}"}


def test_dynamic_utility_can_override_geometry_winner_when_evidence_is_strong(small_config, tmp_path: Path):
    source = tmp_path / "dense.zarr"
    _write_dense_source(source, _signal())
    config = _config(small_config, source)
    selected, scored = pareto_rank_candidates(
        _rank_scores([("geometry", 100.0, (1.0, 1.0)), ("bayes", 90.0, (2.0, 2.0))]),
        _dummy_posterior(),
        config,
    )
    assert str(selected["roi_id"]) == "bayes"
    assert float(selected["relative_evidence"]) > 0.0
    assert float(selected["selection_utility"]) > 1.0
    assert bool(scored.loc[scored["roi_id"] == "bayes", "evidence_eligible"].iloc[0])


def test_negative_lcb_and_low_kernel_support_fall_back_to_geometry(small_config, tmp_path: Path):
    source = tmp_path / "dense.zarr"
    _write_dense_source(source, _signal())
    config = _config(small_config, source)
    negative, _ = pareto_rank_candidates(
        _rank_scores([("geometry", 100.0, (1.0, 1.0)), ("negative", 95.0, (0.8, 0.8))]),
        _dummy_posterior(),
        config,
    )
    assert str(negative["roi_id"]) == "geometry"
    split, scored = pareto_rank_candidates(
        _rank_scores([("geometry", 100.0, (1.0, 1.0)), ("split", 95.0, (2.0, 0.8))]),
        _dummy_posterior(),
        config,
    )
    assert str(split["roi_id"]) == "geometry"
    assert float(scored.loc[scored["roi_id"] == "split", "kernel_support"].iloc[0]) < 0.90


def test_additive_utility_does_not_divide_by_geometry_winner_eivr(small_config, tmp_path: Path):
    source = tmp_path / "dense.zarr"
    _write_dense_source(source, _signal())
    config = _config(small_config, source)
    selected, scored = pareto_additive_rank_candidates(
        _rank_scores([("geometry", 100.0, (0.001, 0.001)), ("small", 90.0, (0.002, 0.002))]),
        _dummy_posterior(),
        config,
        exchange_rate=5.0,
    )
    assert str(selected["roi_id"]) == "geometry"
    small = scored[scored["roi_id"] == "small"].iloc[0]
    assert np.isclose(float(small["additive_bonus"]), 0.005)
    assert float(small["selection_utility"]) < 1.0
    assert "relative_evidence" in scored.columns
    assert float(small["relative_evidence"]) == 0.0


def test_additive_alpha_zero_reproduces_geometry_and_large_alpha_can_override(
    small_config, tmp_path: Path
):
    source = tmp_path / "dense.zarr"
    _write_dense_source(source, _signal())
    config = _config(small_config, source)
    scores = _rank_scores([("geometry", 100.0, (1.0, 1.0)), ("bayes", 90.0, (1.012, 1.012))])
    zero, _ = pareto_additive_rank_candidates(scores, _dummy_posterior(), config, exchange_rate=0.0)
    assert str(zero["roi_id"]) == "geometry"
    alpha5, scored5 = pareto_additive_rank_candidates(scores, _dummy_posterior(), config, exchange_rate=5.0)
    assert str(alpha5["roi_id"]) == "geometry"
    assert float(scored5.loc[scored5["roi_id"] == "bayes", "selection_utility"].iloc[0]) < 1.0
    alpha10, scored10 = pareto_additive_rank_candidates(scores, _dummy_posterior(), config, exchange_rate=10.0)
    assert str(alpha10["roi_id"]) == "bayes"
    assert float(scored10.loc[scored10["roi_id"] == "bayes", "additive_bonus"].iloc[0]) > 0.10


def test_additive_blocks_negative_lcb_and_low_kernel_support(small_config, tmp_path: Path):
    source = tmp_path / "dense.zarr"
    _write_dense_source(source, _signal())
    config = _config(small_config, source)
    negative, _ = pareto_additive_rank_candidates(
        _rank_scores([("geometry", 100.0, (1.0, 1.0)), ("negative", 95.0, (0.9, 0.9))]),
        _dummy_posterior(),
        config,
        exchange_rate=10.0,
    )
    assert str(negative["roi_id"]) == "geometry"
    split, scored = pareto_additive_rank_candidates(
        _rank_scores([("geometry", 100.0, (1.0, 1.0)), ("split", 95.0, (1.3, 0.9))]),
        _dummy_posterior(),
        config,
        exchange_rate=10.0,
    )
    assert str(split["roi_id"]) == "geometry"
    assert not bool(scored.loc[scored["roi_id"] == "split", "evidence_eligible"].iloc[0])
    assert float(scored.loc[scored["roi_id"] == "split", "additive_bonus"].iloc[0]) == 0.0


def test_texture_feature_scaling_is_revealed_only_per_channel_and_feature_family():
    raw = np.zeros((9, 2, 5), dtype=float)
    raw[:, :, 0] = np.linspace(0.0, 1_000_000.0, 9)[:, None]
    raw[:, :, 1] = np.linspace(0.0, 1.0, 9)[:, None]
    raw[:, :, 2] = np.linspace(5.0, 6.0, 9)[:, None]
    raw[:, :, 3] = np.linspace(-2.0, 2.0, 9)[:, None]
    raw[:, :, 4] = np.linspace(100.0, 102.0, 9)[:, None]
    scaled, median, iqr = robust_scale_feature_tensor(raw, epsilon=1.0e-6, clip=8.0)
    assert median.shape == (2, 5)
    assert iqr.shape == (2, 5)
    assert np.all(np.abs(scaled) <= 8.0)
    raw_variance_ratio = np.var(raw[:, 0, 0]) / np.var(raw[:, 0, 1])
    scaled_variance_ratio = np.var(scaled[:, 0, 0]) / np.var(scaled[:, 0, 1])
    assert raw_variance_ratio > 1.0e10
    assert scaled_variance_ratio < 2.0


def test_8x8_training_cap_is_deterministic_and_uses_no_feature_values():
    spec = ParetoPolicySpec("bayesian_pareto_eivr_8x8_texture_tau090", (8, 8), "texture", 0.90)
    nodes = []
    changed = []
    for index in range(520):
        sequence = index // 64 + 1
        feature = np.full((2, 5), float(index))
        nodes.append(
            ParetoSubtileObservation(
                roi_id=f"r{sequence}",
                subtile_id=f"s{index}",
                center_x_nm=float((index * 37) % 1000),
                center_y_nm=float((index * 91) % 1000),
                acquisition_sequence=sequence,
                feature_values=feature,
                mean_noise=np.ones(2),
                channel_mean=np.ones(2),
                pixel_count=1,
            )
        )
        changed.append(
            ParetoSubtileObservation(
                roi_id=f"r{sequence}",
                subtile_id=f"s{index}",
                center_x_nm=float((index * 37) % 1000),
                center_y_nm=float((index * 91) % 1000),
                acquisition_sequence=sequence,
                feature_values=feature * 999.0,
                mean_noise=np.ones(2) * 999.0,
                channel_mean=np.ones(2) * 999.0,
                pixel_count=1,
            )
        )
    first = select_training_subtile_indices(nodes, spec, 384)
    second = select_training_subtile_indices(changed, spec, 384)
    assert len(first) == 384
    assert np.array_equal(first, second)
    latest = max(node.acquisition_sequence for node in nodes)
    latest_indices = {i for i, node in enumerate(nodes) if node.acquisition_sequence == latest}
    assert latest_indices <= set(first.tolist())


def test_pareto_policies_share_pilots_budget_and_shared_evaluator(small_config, tmp_path: Path):
    source = tmp_path / "dense.zarr"
    signal = _signal()
    _write_dense_source(source, signal)
    config = _config(small_config, source)
    x, y = _grid()
    policies = ["uncertainty_lookahead", "bayesian_pareto_eivr_4x4_mean_tau090"]
    results = {
        policy: run_v4_bayesian_pareto_slice_replay(
            config, signal, x, y, ["Cr", "Ni"], "001", policy, seed=3
        )
        for policy in policies
    }
    pilots = [
        tuple(result.metrics[result.metrics["stage"] == "random_pilot"]["roi_id"])
        for result in results.values()
    ]
    assert pilots[0] == pilots[1]
    assert all(len(result.metrics) == config.acquisition_v4.total_rois for result in results.values())
    assert results["bayesian_pareto_eivr_4x4_mean_tau090"].gp_diagnostics.shape[0] == 1
    catalog = build_v3_roi_catalog(x, y, (8, 8)).set_index("roi_id")
    for result in results.values():
        mask = np.zeros((24, 24), dtype=bool)
        for roi_id in result.metrics["roi_id"]:
            roi = catalog.loc[roi_id]
            mask[int(roi["row0"]) : int(roi["row1"]), int(roi["column0"]) : int(roi["column1"])] = True
        prediction = reconstruct_from_observed_mask(signal, mask, x, y, ["Cr", "Ni"], config)
        expected = normalized_reconstruction_rmse(signal, prediction["mean_intensity"].values)
        assert np.isclose(result.metrics.iloc[-1]["normalized_reconstruction_rmse"], expected)


def test_additive_policies_share_pilots_budget_and_shared_evaluator(small_config, tmp_path: Path):
    source = tmp_path / "dense.zarr"
    signal = _signal()
    _write_dense_source(source, signal)
    config = _config(small_config, source)
    x, y = _grid()
    policies = [
        "uncertainty_lookahead",
        "bayesian_pareto_additive_eivr_4x4_mean_tau090_alpha5",
    ]
    results = {
        policy: run_v4_bayesian_additive_slice_replay(
            config, signal, x, y, ["Cr", "Ni"], "001", policy, seed=3
        )
        for policy in policies
    }
    pilots = [
        tuple(result.metrics[result.metrics["stage"] == "random_pilot"]["roi_id"])
        for result in results.values()
    ]
    assert pilots[0] == pilots[1]
    assert all(len(result.metrics) == config.acquisition_v4.total_rois for result in results.values())
    columns = set(results["bayesian_pareto_additive_eivr_4x4_mean_tau090_alpha5"].candidate_trace.columns)
    assert {"additive_exchange_rate", "additive_bonus", "selected_was_geometry_argmax"} <= columns
    catalog = build_v3_roi_catalog(x, y, (8, 8)).set_index("roi_id")
    for result in results.values():
        mask = np.zeros((24, 24), dtype=bool)
        for roi_id in result.metrics["roi_id"]:
            roi = catalog.loc[roi_id]
            mask[int(roi["row0"]) : int(roi["row1"]), int(roi["column0"]) : int(roi["column1"])] = True
        prediction = reconstruct_from_observed_mask(signal, mask, x, y, ["Cr", "Ni"], config)
        expected = normalized_reconstruction_rmse(signal, prediction["mean_intensity"].values)
        assert np.isclose(result.metrics.iloc[-1]["normalized_reconstruction_rmse"], expected)


def test_additive_flat_front_descriptor_identifies_saturated_reference(small_config, tmp_path: Path):
    source = tmp_path / "dense.zarr"
    values = np.zeros((2, 24, 24), dtype=float)
    values[0, :, :] = 100.0
    values[1, :, :] = np.linspace(0.0, 50.0, 24)[None, :]
    _write_dense_source(source, values)
    config = _config(small_config, source)
    config = config.model_copy(
        update={
            "morphology": config.morphology.model_copy(
                update={"minimum_altered_fraction": 0.0, "smoothing_sigma_px": 0.0}
            )
        }
    )
    x, y = _grid()
    descriptors = _pseudo_reference_descriptors("001", values, x, y, ["Cr", "Ni"], config)
    assert descriptors["front_fraction_rows"] == 1.0
    assert descriptors["depth_std_nm"] == 0.0
    assert descriptors["is_saturated_flat_front"]


def test_pareto_cli_resume_does_not_duplicate_work_or_emit_historical_fields(
    small_config, tmp_path: Path
):
    source = tmp_path / "dense.zarr"
    _write_dense_source(source, _signal())
    config = _config(small_config, source)
    config_path = tmp_path / "v4_bayesian_pareto.yaml"
    write_config(config, config_path)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"slice": f"{slice_id:03d}", "channel": channel, "local_path": str(source)}
            for slice_id in (1, 2, 3)
            for channel in ("Cr", "Ni")
        ]
    ).to_csv(manifest, index=False)
    output = tmp_path / "v4_bayesian_pareto_out"
    arguments = [
        "validate-v4-bayesian-pareto-stack",
        "--config",
        str(config_path),
        "--manifest",
        str(manifest),
        "--fold",
        "1",
        "--slices",
        "001:001",
        "--policies",
        "uncertainty_lookahead,bayesian_pareto_eivr_4x4_mean_tau090",
        "--out",
        str(output),
    ]
    runner = CliRunner()
    first = runner.invoke(app, arguments)
    assert first.exit_code == 0, first.output
    first_metrics = pd.read_csv(output / "v4_bayesian_pareto_metrics_by_iteration.csv")
    second = runner.invoke(app, arguments)
    assert second.exit_code == 0, second.output
    second_metrics = pd.read_csv(output / "v4_bayesian_pareto_metrics_by_iteration.csv")
    assert len(first_metrics) == len(second_metrics)
    assert (output / "v4_bayesian_pareto_gp_diagnostics.csv").exists()
    columns = set(pd.read_csv(output / "v4_bayesian_pareto_candidate_trace.csv").columns)
    assert {"geometry_threshold", "shortlist_size", "EIVR_LCB", "relative_evidence"} <= columns
    assert not {"roi_max_score", "attention_score", "graph_edge_score", "predicted_composite_gain"} & columns
    protocol = yaml.safe_load(
        (output / "v4_bayesian_pareto_fold_protocol.yaml").read_text(encoding="utf-8")
    )
    assert protocol["bayesian_order"].startswith("geometry shortlist first")


def test_additive_cli_resume_does_not_duplicate_work_or_emit_historical_fields(
    small_config, tmp_path: Path
):
    source = tmp_path / "dense.zarr"
    _write_dense_source(source, _signal())
    config = _config(small_config, source)
    config_path = tmp_path / "v4_bayesian_additive.yaml"
    write_config(config, config_path)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"slice": f"{slice_id:03d}", "channel": channel, "local_path": str(source)}
            for slice_id in (1, 2, 3)
            for channel in ("Cr", "Ni")
        ]
    ).to_csv(manifest, index=False)
    output = tmp_path / "v4_bayesian_additive_out"
    arguments = [
        "validate-v4-bayesian-additive-stack",
        "--config",
        str(config_path),
        "--manifest",
        str(manifest),
        "--fold",
        "1",
        "--slices",
        "001:001",
        "--policies",
        "uncertainty_lookahead,bayesian_pareto_additive_eivr_4x4_mean_tau090_alpha5",
        "--out",
        str(output),
    ]
    runner = CliRunner()
    first = runner.invoke(app, arguments)
    assert first.exit_code == 0, first.output
    first_metrics = pd.read_csv(output / "v4_bayesian_additive_metrics_by_iteration.csv")
    second = runner.invoke(app, arguments)
    assert second.exit_code == 0, second.output
    second_metrics = pd.read_csv(output / "v4_bayesian_additive_metrics_by_iteration.csv")
    assert len(first_metrics) == len(second_metrics)
    assert (output / "v4_bayesian_additive_flat_front_audit.csv").exists()
    columns = set(pd.read_csv(output / "v4_bayesian_additive_candidate_trace.csv").columns)
    assert {"additive_bonus", "additive_exchange_rate", "selected_was_geometry_argmax"} <= columns
    assert not {"roi_max_score", "attention_score", "graph_edge_score", "predicted_composite_gain"} & columns
    protocol = yaml.safe_load(
        (output / "v4_bayesian_additive_fold_protocol.yaml").read_text(encoding="utf-8")
    )
    assert protocol["additive_utility"].startswith("utility =")
