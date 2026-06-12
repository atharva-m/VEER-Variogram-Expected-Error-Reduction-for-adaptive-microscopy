from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from scipy.ndimage import distance_transform_edt
from typer.testing import CliRunner

from balance_nm.cli import app
from balance_nm.domain import RunConfig
from balance_nm.io import load_config, write_config
from balance_nm.v3_morphology import normalized_reconstruction_rmse, reconstruct_from_observed_mask
from balance_nm.v3_validation import build_v3_roi_catalog
from balance_nm.v4_neural import nearest_signal_and_distance, torch_available, train_neural_ensemble
from balance_nm.v4_validation import (
    _candidate_features,
    _distance_pixels,
    build_v4_folds,
    lookahead_coverage_gain,
    run_v4_slice_replay,
)


def _write_dense_source(path: Path, values: np.ndarray) -> None:
    x = (np.arange(values.shape[2]) + 0.5) * 100.0
    y = (np.arange(values.shape[1]) + 0.5) * 100.0
    xr.Dataset(
        {"counts": (("element", "y", "x"), values)},
        coords={"element": ["Cr", "Ni"], "x": x, "y": y},
    ).to_zarr(path, mode="w")


def _v4_config(small_config, source: Path, *, total_rois: int = 5) -> RunConfig:
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
        "historical_adaptive_rois_before_anchor": 1,
        "front_weight": 0.5,
        "penetration_d95_weight": 0.5,
        "excluded_channels": ["CPS"],
        "oracle_sample_slices": 1,
        "calibrator": {
            "enabled": False,
            "max_training_slices": 1,
            "max_validation_slices": 1,
            "max_training_states_per_slice": 1,
            "max_candidates_per_state": 2,
        },
        "neural": {
            "enabled": False,
            "ensemble_size": 1,
            "depth": 1,
            "base_channels": 4,
            "epochs": 1,
            "early_stop_patience": 1,
            "batch_size": 1,
            "training_masks_per_slice": 1,
            "max_training_slices": 1,
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


def _signal() -> np.ndarray:
    values = np.zeros((2, 24, 24), dtype=float)
    values[0, :, :12] = 100.0
    values[1, :, 12:] = 80.0
    return values


def test_lookahead_gain_matches_explicit_distance_transform():
    observed = np.zeros((24, 24), dtype=bool)
    observed[:8, :8] = True
    x = (np.arange(24) + 0.5) * 100.0
    y = (np.arange(24) + 0.5) * 100.0
    roi = build_v3_roi_catalog(x, y, (8, 8)).iloc[4]
    before = distance_transform_edt(~observed)
    hypothetical = observed.copy()
    hypothetical[int(roi["row0"]) : int(roi["row1"]), int(roi["column0"]) : int(roi["column1"])] = True
    after = distance_transform_edt(~hypothetical)
    expected = float(np.sum(before - after) / (np.hypot(*observed.shape) / 2.0))
    assert np.isclose(lookahead_coverage_gain(before, roi), expected)


def test_candidate_features_do_not_read_hidden_values(small_config, tmp_path: Path):
    source = tmp_path / "dense.zarr"
    _write_dense_source(source, _signal())
    config = _v4_config(small_config, source)
    x = (np.arange(24) + 0.5) * 100.0
    y = (np.arange(24) + 0.5) * 100.0
    observed = np.zeros((24, 24), dtype=bool)
    observed[:8, :8] = True
    first = _signal()
    second = first.copy()
    second[:, 8:, 8:] = 9999.0
    catalog = build_v3_roi_catalog(x, y, (8, 8))
    candidates = catalog[catalog["roi_id"] != "r0000_c0000"]
    first_prediction = reconstruct_from_observed_mask(first, observed, x, y, ["Cr", "Ni"], config)
    second_prediction = reconstruct_from_observed_mask(second, observed, x, y, ["Cr", "Ni"], config)
    first_features = _candidate_features(
        candidates, first_prediction, observed, _distance_pixels(observed), ["Cr", "Ni"], config, 2
    )
    second_features = _candidate_features(
        candidates, second_prediction, observed, _distance_pixels(observed), ["Cr", "Ni"], config, 2
    )
    pd.testing.assert_frame_equal(first_features, second_features)


def test_v4_policies_share_pilots_budget_and_only_historical_arm_forces_neighbors(
    small_config, tmp_path: Path
):
    source = tmp_path / "dense.zarr"
    signal = _signal()
    _write_dense_source(source, signal)
    config = _v4_config(small_config, source)
    x = (np.arange(24) + 0.5) * 100.0
    y = (np.arange(24) + 0.5) * 100.0
    results = {
        policy: run_v4_slice_replay(config, signal, x, y, ["Cr", "Ni"], "001", policy, seed=3)
        for policy in [
            "uncertainty_distance_sequential",
            "uncertainty_distance_one_anchor",
            "uncertainty_lookahead",
        ]
    }
    pilots = [
        tuple(result.metrics[result.metrics["stage"] == "random_pilot"]["roi_id"])
        for result in results.values()
    ]
    assert pilots[0] == pilots[1] == pilots[2]
    assert all(len(result.metrics) == config.acquisition_v4.total_rois for result in results.values())
    assert all(result.metrics.iloc[-1]["query_count"] == config.acquisition_v4.total_rois for result in results.values())
    assert "historical_one_anchor_neighbor" in set(results["uncertainty_distance_one_anchor"].metrics["stage"])
    assert "historical_one_anchor_neighbor" not in set(results["uncertainty_distance_sequential"].metrics["stage"])
    assert "historical_one_anchor_neighbor" not in set(results["uncertainty_lookahead"].metrics["stage"])


def test_blocked_folds_exclude_outer_and_validation_guards():
    config = load_config(Path(__file__).parents[1] / "configs" / "alloy617_v4_uncertainty.yaml")
    folds = build_v4_folds([f"{number:03d}" for number in range(1, 266)], config)
    assert len(folds) == 5
    for fold in folds:
        assert set(fold.test_slices).isdisjoint(fold.training_slices)
        assert set(fold.validation_slices).isdisjoint(fold.training_slices)
        assert set(fold.excluded_guard_slices).isdisjoint(fold.training_slices)


def test_cps_is_excluded_from_v4_element_inputs():
    config = load_config(Path(__file__).parents[1] / "configs" / "alloy617_v4_uncertainty.yaml")
    assert "CPS" not in config.scenario.elements
    assert config.acquisition_v4.excluded_channels == ["CPS"]


class _FakeNeural:
    def predict(self, visible_signal: np.ndarray, observed_mask: np.ndarray):
        nearest, _ = nearest_signal_and_distance(visible_signal, observed_mask)
        return nearest, np.ones_like(nearest)


def test_neural_selector_only_uses_shared_nearest_reconstruction_evaluator(small_config, tmp_path: Path):
    source = tmp_path / "dense.zarr"
    signal = _signal()
    _write_dense_source(source, signal)
    config = _v4_config(small_config, source)
    x = (np.arange(24) + 0.5) * 100.0
    y = (np.arange(24) + 0.5) * 100.0
    result = run_v4_slice_replay(
        config,
        signal,
        x,
        y,
        ["Cr", "Ni"],
        "001",
        "uncertainty_neural_guarded_selector_only",
        seed=2,
        neural=_FakeNeural(),
    )
    mask = np.zeros((24, 24), dtype=bool)
    catalog = build_v3_roi_catalog(x, y, (8, 8)).set_index("roi_id")
    for roi_id in result.metrics["roi_id"]:
        roi = catalog.loc[roi_id]
        mask[int(roi["row0"]) : int(roi["row1"]), int(roi["column0"]) : int(roi["column1"])] = True
    prediction = reconstruct_from_observed_mask(signal, mask, x, y, ["Cr", "Ni"], config)
    expected = normalized_reconstruction_rmse(signal, prediction["mean_intensity"].values)
    assert np.isclose(result.metrics.iloc[-1]["normalized_reconstruction_rmse"], expected)


@pytest.mark.skipif(not torch_available(), reason="optional torch learned runtime is not installed")
def test_tiny_neural_ensemble_prediction_shapes():
    from balance_nm.domain import AcquisitionV4NeuralConfig

    config = AcquisitionV4NeuralConfig(
        ensemble_size=1,
        depth=1,
        base_channels=4,
        epochs=1,
        early_stop_patience=1,
        batch_size=1,
        training_masks_per_slice=1,
        max_training_slices=1,
    )
    signal = _signal()[:, :8, :8]
    ensemble = train_neural_ensemble([signal], config, seed=0)
    mask = np.zeros((8, 8), dtype=bool)
    mask[:4, :4] = True
    mean, variance = ensemble.predict(signal, mask)
    assert mean.shape == variance.shape == signal.shape


def test_v4_cli_resume_does_not_duplicate_completed_work(small_config, tmp_path: Path):
    source = tmp_path / "dense.zarr"
    _write_dense_source(source, _signal())
    config = _v4_config(small_config, source)
    config_path = tmp_path / "v4.yaml"
    write_config(config, config_path)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"slice": f"{slice_id:03d}", "channel": channel, "local_path": str(source)}
            for slice_id in (1, 2, 3)
            for channel in ("Cr", "Ni")
        ]
    ).to_csv(manifest, index=False)
    output = tmp_path / "v4_out"
    arguments = [
        "validate-v4-uncertainty-stack",
        "--config",
        str(config_path),
        "--manifest",
        str(manifest),
        "--fold",
        "1",
        "--slices",
        "001:001",
        "--policies",
        "uncertainty_distance_sequential,uncertainty_distance_one_anchor,uncertainty_lookahead,uniform,random",
        "--out",
        str(output),
    ]
    runner = CliRunner()
    first = runner.invoke(app, arguments)
    assert first.exit_code == 0, first.output
    first_metrics = pd.read_csv(output / "v4_metrics_by_iteration.csv")
    second = runner.invoke(app, arguments)
    assert second.exit_code == 0, second.output
    second_metrics = pd.read_csv(output / "v4_metrics_by_iteration.csv")
    assert len(first_metrics) == len(second_metrics)
    assert "morphology_composite_error" in second_metrics.columns
    assert (output / "v4_error_vs_cost_curves.csv").exists()
    assert (output / "v4_paired_comparisons.csv").exists()
    protocol = yaml.safe_load((output / "v4_fold_protocol.yaml").read_text(encoding="utf-8"))
    assert protocol["dense_truth_policy"].startswith("hidden from deployable selectors")


def test_v4_query_checkpoint_callback_receives_every_reveal(small_config, tmp_path: Path):
    source = tmp_path / "dense.zarr"
    signal = _signal()
    _write_dense_source(source, signal)
    config = _v4_config(small_config, source)
    x = (np.arange(24) + 0.5) * 100.0
    y = (np.arange(24) + 0.5) * 100.0
    checkpoints: list[tuple[int, int]] = []
    run_v4_slice_replay(
        config,
        signal,
        x,
        y,
        ["Cr", "Ni"],
        "001",
        "uncertainty_lookahead",
        checkpoint_callback=lambda metrics, trace: checkpoints.append((len(metrics), len(trace))),
    )
    assert [count for count, _ in checkpoints] == [1, 2, 3, 4, 5]
    assert checkpoints[-1][1] > 0
