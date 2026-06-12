"""Configuration and artifact persistence helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .domain import RunConfig
from .experiment import RunResult
from .visualization import plot_run, plot_v2_products


def load_config(path: Path) -> RunConfig:
    with path.open("r", encoding="utf-8") as handle:
        return RunConfig.model_validate(yaml.safe_load(handle))


def write_config(config: RunConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.model_dump(mode="json"), handle, sort_keys=False)


def save_run_artifacts(result: RunResult, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_config(result.config, output / "resolved_config.yaml")
    reference_name = (
        "evaluation_reference.zarr"
        if result.config.dataset.mode == "replay"
        else "hidden_truth.zarr"
    )
    result.hidden_sample.to_zarr(output / reference_name, mode="w")
    if result.state.observations is not None:
        result.state.observations.to_zarr(output / "observations.zarr", mode="w")
        source = result.standardized_dataset if result.standardized_dataset is not None else result.state.observations
        source.to_zarr(output / "standardized_dataset.zarr", mode="w")
    result.final_prediction.to_zarr(output / "prediction.zarr", mode="w")
    if result.state.quality_products is not None:
        result.state.quality_products.to_zarr(output / "quality_products.zarr", mode="w")
    if result.state.pattern_products is not None:
        result.state.pattern_products.to_zarr(output / "pattern_products.zarr", mode="w")
    pd.DataFrame([action.model_dump(mode="json") for action in result.state.actions]).to_csv(
        output / "actions.csv", index=False
    )
    pd.DataFrame(result.state.decision_trace).to_csv(output / "decision_trace.csv", index=False)
    pd.DataFrame(result.state.decision_trace).to_csv(output / "ranked_actions.csv", index=False)
    audit_columns = [
        "iteration",
        "action_id",
        "preset_id",
        "action_type",
        "bounds_nm",
        "step_size_nm",
        "y_step_size_nm",
        "dwell_time_ms",
        "estimated_time_s",
        "estimated_dose",
        "constraint_status",
        "violations",
    ]
    rejected = pd.DataFrame(result.state.rejected_actions)
    unsupported = pd.DataFrame(result.state.unsupported_proposals)
    if rejected.empty:
        rejected = pd.DataFrame(columns=audit_columns)
    if unsupported.empty:
        unsupported = pd.DataFrame(columns=audit_columns)
    rejected.to_csv(output / "rejected_actions.csv", index=False)
    unsupported.to_csv(output / "unsupported_proposals.csv", index=False)
    pd.DataFrame([record.model_dump() for record in result.metrics]).to_csv(
        output / "metrics.csv", index=False
    )
    pd.DataFrame([record.model_dump() for record in result.objective_metrics]).to_csv(
        output / "objective_metrics.csv", index=False
    )
    if result.capabilities is not None:
        with (output / "capability_manifest.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(result.capabilities.model_dump(mode="json"), handle, sort_keys=False)
    plot_run(
        result.hidden_sample,
        result.state,
        result.final_prediction,
        result.last_recommendation,
        output / "diagnostic_maps.png",
    )
    plot_v2_products(result.final_prediction, output / "pattern_quality_maps.png")
