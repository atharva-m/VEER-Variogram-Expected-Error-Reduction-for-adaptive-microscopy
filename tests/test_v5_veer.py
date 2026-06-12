from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from typer.testing import CliRunner

from balance_nm.cli import app
from balance_nm.domain import RunConfig
from balance_nm.io import write_config
from balance_nm.v3_morphology import normalized_reconstruction_rmse, reconstruct_from_observed_mask
from balance_nm.v3_validation import build_v3_roi_catalog
from balance_nm.v4_bayesian_pareto import ParetoSubtileObservation
from balance_nm.v4_validation import _distance_pixels, lookahead_coverage_gain
from balance_nm.v5 import (
    NestedVariogramFit,
    fit_nested_variogram,
    fit_variogram_posterior,
    front_movement_fraction,
    front_probability_weights,
    front_relevance_weights,
    gamma_nested,
    kernel_distance_field,
    kernel_rectangle_distance,
    nested_veer_select_candidate,
    parse_nested_policy,
    parse_veer_policy,
    run_v5_veer_slice_replay,
)


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
        "folds": {
            "outer_test_ranges": [[1, 1]],
            "outer_guard_slices": 0,
            "validation_slices": 1,
            "validation_guard_slices": 0,
        },
    }
    raw["acquisition_v5"] = {
        "latent_components": 2,
        "kernel_catalog": [[0.1, 0.1], [0.2, 0.2]],
        "temper_reference_subtiles": 16,
        "front_bandwidth_nm": 400.0,
    }
    raw["objectives"] = {"weights": {"gradient": 1.0}}
    return RunConfig.model_validate(raw)


def _observations(scale: float = 1.0) -> list[ParetoSubtileObservation]:
    rng = np.random.default_rng(7)
    nodes = []
    for index in range(24):
        center_x = float((index * 137) % 2000) + 100.0
        center_y = float((index * 211) % 2000) + 100.0
        base = np.asarray(
            [np.sin(center_x / 500.0), np.cos(center_y / 500.0)], dtype=float
        )
        feature = (base + 0.05 * rng.standard_normal(2)) * scale
        nodes.append(
            ParetoSubtileObservation(
                roi_id=f"r{index // 4}",
                subtile_id=f"s{index}",
                center_x_nm=center_x,
                center_y_nm=center_y,
                acquisition_sequence=index // 4 + 1,
                feature_values=feature[:, None],
                mean_noise=np.full(2, 0.01) * scale**2,
                channel_mean=feature,
                pixel_count=4,
            )
        )
    return nodes


def test_parse_veer_policy_and_rejects_unknown():
    assert parse_veer_policy("variogram_eer_4x4_mean_kappa0").front_kappa == 0.0
    assert parse_veer_policy("variogram_eer_4x4_mean_kappa5").front_kappa == 5.0
    with pytest.raises(ValueError):
        parse_veer_policy("variogram_eer_4x4_mean_alpha5")


def test_variogram_weights_are_scale_invariant_and_tempering_flattens(small_config, tmp_path: Path):
    source = tmp_path / "dense.zarr"
    _write_dense_source(source, _signal())
    config = _config(small_config, source)
    first = fit_variogram_posterior(_observations(scale=1.0), config)
    second = fit_variogram_posterior(_observations(scale=1000.0), config)
    assert np.allclose(first.weights, second.weights, atol=1.0e-8)
    assert np.allclose(first.log_marginal_likelihoods, second.log_marginal_likelihoods, atol=1.0e-6)
    assert first.temper == pytest.approx(24 / 16)
    assert np.max(first.weights) <= np.max(first.untempered_weights) + 1.0e-12
    assert np.isclose(np.sum(first.weights), 1.0)
    assert first.sill > 0.0


def test_kernel_rectangle_distance_matches_anisotropic_edt(small_config, tmp_path: Path):
    source = tmp_path / "dense.zarr"
    _write_dense_source(source, _signal())
    config = _config(small_config, source)
    x, y = _grid()
    catalog = build_v3_roi_catalog(x, y, (8, 8))
    roi = catalog.iloc[4]
    mask = np.zeros((24, 24), dtype=bool)
    mask[int(roi["row0"]) : int(roi["row1"]), int(roi["column0"]) : int(roi["column1"])] = True
    for scales in [(0.1, 0.1), (0.05, 0.2)]:
        field = kernel_distance_field(mask, x, y, config, scales)
        analytic = kernel_rectangle_distance(mask.shape, roi, x, y, config, scales)
        assert np.allclose(field, analytic, atol=1.0e-9)


def test_front_weights_peak_at_predicted_front(small_config, tmp_path: Path):
    source = tmp_path / "dense.zarr"
    signal = _signal()
    _write_dense_source(source, signal)
    config = _config(small_config, source)
    x, y = _grid()
    full_mask = np.ones((24, 24), dtype=bool)
    prediction = reconstruct_from_observed_mask(signal, full_mask, x, y, ["Cr", "Ni"], config)
    uniform = front_relevance_weights(prediction, x, y, config, 0.0)
    assert np.allclose(uniform, 1.0)
    weighted = front_relevance_weights(prediction, x, y, config, 5.0)
    assert weighted.shape == (24, 24)
    assert float(weighted.max()) == pytest.approx(6.0, abs=0.2)
    front_columns = weighted.argmax(axis=1)
    assert np.all((front_columns >= 8) & (front_columns <= 16))
    assert float(weighted[:, 0].max()) < 1.5


def test_veer_policies_share_pilots_budget_and_shared_evaluator(small_config, tmp_path: Path):
    source = tmp_path / "dense.zarr"
    signal = _signal()
    _write_dense_source(source, signal)
    config = _config(small_config, source)
    x, y = _grid()
    policies = ["uncertainty_lookahead", "variogram_eer_4x4_mean_kappa0"]
    results = {
        policy: run_v5_veer_slice_replay(
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
    veer = results["variogram_eer_4x4_mean_kappa0"]
    assert not veer.variogram_trace.empty
    weight_sums = veer.variogram_trace.groupby("iteration")["weight"].sum()
    assert np.allclose(weight_sums, 1.0)
    columns = set(veer.candidate_trace.columns)
    assert {"expected_error_reduction", "eer_per_cost", "selection_utility", "selected"} <= columns
    assert (veer.candidate_trace["expected_error_reduction"] >= 0.0).all()
    catalog = build_v3_roi_catalog(x, y, (8, 8)).set_index("roi_id")
    for result in results.values():
        mask = np.zeros((24, 24), dtype=bool)
        for roi_id in result.metrics["roi_id"]:
            roi = catalog.loc[roi_id]
            mask[int(roi["row0"]) : int(roi["row1"]), int(roi["column0"]) : int(roi["column1"])] = True
        prediction = reconstruct_from_observed_mask(signal, mask, x, y, ["Cr", "Ni"], config)
        expected = normalized_reconstruction_rmse(signal, prediction["mean_intensity"].values)
        assert np.isclose(result.metrics.iloc[-1]["normalized_reconstruction_rmse"], expected)


def test_parse_nested_policy_and_rejects_unknown():
    assert parse_nested_policy("nested_veer_4x4_mean_kappa0").front_kappa == 0.0
    assert parse_nested_policy("nested_veer_4x4_mean_kappa10").front_kappa == 10.0
    assert parse_nested_policy("nested_veer_4x4_mean_kappa5").weight_mode == "probability"
    band = parse_nested_policy("nested_band_veer_4x4_mean_kappa5")
    assert band.front_kappa == 5.0
    assert band.weight_mode == "band"
    with pytest.raises(ValueError):
        parse_nested_policy("variogram_eer_4x4_mean_kappa5")


def test_nested_fit_recovers_unbounded_growth_on_brownian_features(small_config, tmp_path: Path):
    source = tmp_path / "dense.zarr"
    _write_dense_source(source, _signal())
    config = _config(small_config, source)
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.standard_normal((64, 2)), axis=0)
    nodes = [
        ParetoSubtileObservation(
            roi_id=f"r{i // 16}",
            subtile_id=f"s{i}",
            center_x_nm=float(i * 25.0),
            center_y_nm=800.0,
            acquisition_sequence=i // 16 + 1,
            feature_values=walk[i][:, None],
            mean_noise=np.full(2, 1.0e-4),
            channel_mean=walk[i],
            pixel_count=4,
        )
        for i in range(64)
    ]
    fit = fit_nested_variogram(nodes, config)
    assert fit.nugget >= 0.0 and fit.matern_amplitude >= 0.0
    assert fit.linear_slope > 0.05
    grid = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
    values = gamma_nested(grid, fit)
    assert np.all(np.diff(values) > 0.0)
    long_range_growth = values[-1] - values[-2]
    assert long_range_growth / values[-1] > 0.05


def test_nested_linear_only_fit_matches_baseline_selection(small_config, tmp_path: Path):
    source = tmp_path / "dense.zarr"
    _write_dense_source(source, _signal())
    config = _config(small_config, source)
    x, y = _grid()
    catalog = build_v3_roi_catalog(x, y, (8, 8))
    linear_fit = NestedVariogramFit(
        nugget=0.0,
        matern_amplitude=0.0,
        linear_slope=1.0,
        length_scale=0.1,
        weighted_sse=0.0,
        bin_distances=np.empty(0),
        bin_semivariances=np.empty(0),
        bin_pair_counts=np.empty(0),
        subtile_count=0,
    )
    mask = np.zeros((24, 24), dtype=bool)
    queried: set[str] = set()
    weights = np.ones((24, 24))
    for index in [0, 4]:
        roi = catalog.iloc[index]
        mask[int(roi["row0"]) : int(roi["row1"]), int(roi["column0"]) : int(roi["column1"])] = True
        queried.add(str(roi["roi_id"]))
    for _ in range(3):
        candidates = catalog[~catalog["roi_id"].isin(queried)].copy()
        distance = _distance_pixels(mask)
        candidates["gain"] = [
            lookahead_coverage_gain(distance, roi) for _, roi in candidates.iterrows()
        ]
        baseline = candidates.sort_values(
            ["gain", "row0", "column0"], ascending=[False, True, True]
        ).iloc[0]
        selected, _ = nested_veer_select_candidate(
            catalog, queried, mask, linear_fit, weights, x, y, config
        )
        assert str(selected["roi_id"]) == str(baseline["roi_id"])
        mask[
            int(selected["row0"]) : int(selected["row1"]),
            int(selected["column0"]) : int(selected["column1"]),
        ] = True
        queried.add(str(selected["roi_id"]))


def test_front_probability_weights_inflate_unsampled_regions(small_config, tmp_path: Path):
    source = tmp_path / "dense.zarr"
    signal = _signal()
    _write_dense_source(source, signal)
    config = _config(small_config, source)
    x, y = _grid()
    mask = np.zeros((24, 24), dtype=bool)
    mask[:, :8] = True
    prediction = reconstruct_from_observed_mask(signal, mask, x, y, ["Cr", "Ni"], config)
    uniform = front_probability_weights(prediction, 0.0)
    assert np.allclose(uniform, 1.0)
    weighted = front_probability_weights(prediction, 5.0)
    assert weighted.shape == (24, 24)
    assert np.all(weighted >= 1.0) and np.all(weighted <= 6.0)
    far_unsampled = float(weighted[:, 20:].mean())
    well_sampled = float(weighted[:, :6].mean())
    assert far_unsampled > well_sampled


def test_nested_policies_share_pilots_budget_and_traces(small_config, tmp_path: Path):
    source = tmp_path / "dense.zarr"
    signal = _signal()
    _write_dense_source(source, signal)
    config = _config(small_config, source)
    x, y = _grid()
    policies = ["uncertainty_lookahead", "nested_veer_4x4_mean_kappa5"]
    results = {
        policy: run_v5_veer_slice_replay(
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
    nested = results["nested_veer_4x4_mean_kappa5"]
    assert not nested.variogram_trace.empty
    assert set(nested.variogram_trace["kernel"]) == {"nested_wls"}
    assert (nested.variogram_trace["linear_slope"] >= 0.0).all()
    columns = set(nested.candidate_trace.columns)
    assert {"expected_error_reduction", "nested_linear_slope", "front_kappa", "selected"} <= columns
    assert (nested.candidate_trace["expected_error_reduction"] >= 0.0).all()
    catalog = build_v3_roi_catalog(x, y, (8, 8)).set_index("roi_id")
    for result in results.values():
        mask = np.zeros((24, 24), dtype=bool)
        for roi_id in result.metrics["roi_id"]:
            roi = catalog.loc[roi_id]
            mask[int(roi["row0"]) : int(roi["row1"]), int(roi["column0"]) : int(roi["column1"])] = True
        prediction = reconstruct_from_observed_mask(signal, mask, x, y, ["Cr", "Ni"], config)
        expected = normalized_reconstruction_rmse(signal, prediction["mean_intensity"].values)
        assert np.isclose(result.metrics.iloc[-1]["normalized_reconstruction_rmse"], expected)


def test_pilots_pair_across_policies_and_vary_across_slices(small_config, tmp_path: Path):
    source = tmp_path / "dense.zarr"
    signal = _signal()
    _write_dense_source(source, signal)
    config = _config(small_config, source)
    x, y = _grid()

    def pilots(slice_id: str, policy: str) -> tuple[str, ...]:
        result = run_v5_veer_slice_replay(
            config, signal, x, y, ["Cr", "Ni"], slice_id, policy, seed=3
        )
        return tuple(result.metrics[result.metrics["stage"] == "random_pilot"]["roi_id"])

    assert pilots("001", "uncertainty_lookahead") == pilots("001", "variogram_eer_4x4_mean_kappa5")
    assert pilots("001", "uncertainty_lookahead") != pilots("002", "uncertainty_lookahead")


def test_parse_gated_policy_and_movement_fraction():
    gated = parse_veer_policy("gated_veer_4x4_mean_kappa5")
    assert gated.front_kappa == 5.0 and gated.gated
    assert not parse_veer_policy("variogram_eer_4x4_mean_kappa5").gated
    width = 1000.0
    static = np.array([100.0, 200.0, np.nan])
    assert front_movement_fraction(static, static, width) == 0.0
    moved = np.array([150.0, 200.0, np.nan])
    assert front_movement_fraction(static, moved, width) == pytest.approx(50.0 / 3 / width)
    appeared = np.array([100.0, 200.0, 300.0])
    assert front_movement_fraction(static, appeared, width) == pytest.approx(1.0 / 3)


def test_gated_replay_shuts_off_front_weighting_when_front_is_static(small_config, tmp_path: Path):
    source = tmp_path / "dense.zarr"
    signal = _signal()
    _write_dense_source(source, signal)
    config = _config(small_config, source, total_rois=6)
    x, y = _grid()
    result = run_v5_veer_slice_replay(
        config, signal, x, y, ["Cr", "Ni"], "001", "gated_veer_4x4_mean_kappa5", seed=3
    )
    assert len(result.metrics) == 6
    trace = result.candidate_trace
    assert {"front_kappa", "front_kappa_effective", "front_movement_fraction"} <= set(trace.columns)
    assert (trace["front_kappa"] == 5.0).all()
    assert (trace["front_kappa_effective"] <= 5.0 + 1e-12).all()
    assert (trace["front_kappa_effective"] >= 0.0).all()
    selected = trace[trace["selected"] == True]
    static = selected[selected["front_movement_fraction"] == 0.0]
    if not static.empty:
        assert (static["front_kappa_effective"] == 0.0).all()


def test_parallel_workers_produce_identical_results_to_serial(small_config, tmp_path: Path):
    from balance_nm.v5 import run_v5_veer_stack_validation

    source = tmp_path / "dense.zarr"
    _write_dense_source(source, _signal())
    config = _config(small_config, source)
    config_path = tmp_path / "v5_veer.yaml"
    write_config(config, config_path)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"slice": f"{slice_id:03d}", "channel": channel, "local_path": str(source)}
            for slice_id in (1, 2, 3)
            for channel in ("Cr", "Ni")
        ]
    ).to_csv(manifest, index=False)
    policies = ["uncertainty_lookahead", "nested_veer_4x4_mean_kappa0"]
    serial_metrics, _ = run_v5_veer_stack_validation(
        config_path, tmp_path / "serial", manifest, "1", ["001"], policies, seed=0, workers=1
    )
    parallel_metrics, _ = run_v5_veer_stack_validation(
        config_path, tmp_path / "parallel", manifest, "1", ["001"], policies, seed=0, workers=2
    )
    key = ["fold", "slice", "policy", "iteration"]
    serial_sorted = serial_metrics.sort_values(key).reset_index(drop=True)
    parallel_sorted = parallel_metrics.sort_values(key).reset_index(drop=True)
    assert list(serial_sorted["roi_id"]) == list(parallel_sorted["roi_id"])
    assert np.allclose(
        serial_sorted["morphology_composite_error"].astype(float),
        parallel_sorted["morphology_composite_error"].astype(float),
        equal_nan=True,
    )
    assert np.allclose(
        serial_sorted["normalized_reconstruction_rmse"].astype(float),
        parallel_sorted["normalized_reconstruction_rmse"].astype(float),
    )


def test_veer_cli_resume_does_not_duplicate_work_or_emit_v4_fields(small_config, tmp_path: Path):
    source = tmp_path / "dense.zarr"
    _write_dense_source(source, _signal())
    config = _config(small_config, source)
    config_path = tmp_path / "v5_veer.yaml"
    write_config(config, config_path)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"slice": f"{slice_id:03d}", "channel": channel, "local_path": str(source)}
            for slice_id in (1, 2, 3)
            for channel in ("Cr", "Ni")
        ]
    ).to_csv(manifest, index=False)
    output = tmp_path / "v5_veer_out"
    arguments = [
        "validate-v5-veer-stack",
        "--config",
        str(config_path),
        "--manifest",
        str(manifest),
        "--fold",
        "1",
        "--slices",
        "001:001",
        "--policies",
        "uncertainty_lookahead,variogram_eer_4x4_mean_kappa0,nested_veer_4x4_mean_kappa5",
        "--out",
        str(output),
    ]
    runner = CliRunner()
    first = runner.invoke(app, arguments)
    assert first.exit_code == 0, first.output
    first_metrics = pd.read_csv(output / "v5_veer_metrics_by_iteration.csv")
    second = runner.invoke(app, arguments)
    assert second.exit_code == 0, second.output
    second_metrics = pd.read_csv(output / "v5_veer_metrics_by_iteration.csv")
    assert len(first_metrics) == len(second_metrics)
    assert (output / "v5_veer_variogram_trace.csv").exists()
    columns = set(pd.read_csv(output / "v5_veer_candidate_trace.csv").columns)
    assert {"expected_error_reduction", "eer_per_cost", "front_kappa"} <= columns
    assert not {"EIVR_LCB", "additive_bonus", "relative_evidence", "geometry_shortlist_eligible"} & columns
    trailing = pd.read_csv(output / "v5_veer_trailing_summary.csv")
    assert {"trailing_median_composite", "trailing_median_delta_vs_uncertainty"} <= set(trailing.columns)
    assert set(trailing["policy"]) == {
        "uncertainty_lookahead",
        "variogram_eer_4x4_mean_kappa0",
        "nested_veer_4x4_mean_kappa5",
    }
    protocol = yaml.safe_load(
        (output / "v5_veer_fold_protocol.yaml").read_text(encoding="utf-8")
    )
    assert protocol["objective"].startswith("expected nearest-observation")
    assert protocol["pilot_seeding"].startswith("default_rng([seed, slice])")
    assert protocol["nested_variogram"].startswith("v5.1 arms fit gamma")
    assert protocol["co_primary_endpoint"].startswith("trailing-median")
