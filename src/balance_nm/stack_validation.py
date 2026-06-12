"""Scalable retrospective ROI validation on dense multichannel map stacks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .domain import RunConfig
from .io import load_config
from .roi_search import (
    _full_reference_rate,
    _full_source,
    _morphology_interest_score,
    _morphology_objective_score,
    _neighbor_candidates,
    _outer_selection,
    _score_reference_rois,
    _source_coordinates_without_values,
    build_roi_catalog,
)


@dataclass(frozen=True)
class ValidationArm:
    name: str
    selection_strategy: str
    confirm_neighbors: bool
    neighbor_anchors: int = 1


DEFAULT_ARMS = [
    ValidationArm("balance_no_confirmation", "balance", False),
    ValidationArm("balance_one_anchor", "balance", True, 1),
    ValidationArm("balance_two_anchors", "balance", True, 2),
    ValidationArm("gradient", "gradient", False),
    ValidationArm("uncertainty", "uncertainty", False),
    ValidationArm("uniform", "uniform", False),
    ValidationArm("random", "random", False),
]


def config_for_slice(
    template: RunConfig, slice_id: str, element_sources: dict[str, Path] | None = None
) -> RunConfig:
    raw = template.model_dump(mode="python")
    if element_sources is not None:
        raw["dataset"]["element_map_sources"] = {
            element: str(element_sources[element]) for element in template.scenario.elements
        }
    else:
        for element, path in raw["dataset"]["element_map_sources"].items():
            raw["dataset"]["element_map_sources"][element] = (
                str(path).replace("slice_006", f"slice_{slice_id}").replace("006_0", f"{slice_id}_0")
            )
    raw["dataset"]["spatial_crop_indices"] = None
    raw["dataset"]["value_semantics"] = "intensity_proxy"
    return RunConfig.model_validate(raw)


def sources_from_manifest(
    manifest_path: Path, slice_ids: list[str], elements: list[str]
) -> dict[str, dict[str, Path]]:
    manifest = pd.read_csv(manifest_path, dtype={"slice": str})
    manifest["slice"] = manifest["slice"].str.zfill(3)
    manifest["channel_upper"] = manifest["channel"].str.upper()
    sources: dict[str, dict[str, Path]] = {}
    for slice_id in slice_ids:
        subset = manifest[manifest["slice"] == slice_id]
        sources[slice_id] = {}
        for element in elements:
            match = subset[subset["channel_upper"] == element.upper()]
            if len(match) != 1:
                raise ValueError(
                    f"manifest must contain exactly one {element} map for slice {slice_id}"
                )
            path = Path(str(match.iloc[0]["local_path"]))
            if not path.exists():
                raise FileNotFoundError(path)
            sources[slice_id][element] = path
    return sources


def _raster_cost(config: RunConfig, roi: pd.Series) -> tuple[float, float]:
    rows = int(roi["row1"] - roi["row0"])
    columns = int(roi["column1"] - roi["column0"])
    pixels = rows * columns
    time_ms = (
        config.instrument.action_overhead_ms
        + config.instrument.line_overhead_ms * rows
        + pixels * (config.dataset.dwell_ms + config.instrument.pixel_overhead_ms)
    )
    dose = config.instrument.dose_coefficient * pixels * config.dataset.dwell_ms
    return time_ms / 1000.0, dose


def evaluate_direct_raster_arm(
    config: RunConfig,
    reference_signal: np.ndarray,
    candidate_evaluation: pd.DataFrame,
    arm: ValidationArm,
    seed: int = 0,
    pilot_rois: int = 4,
    adaptive_rois: int = 8,
    equivalence_fraction: float = 0.99,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Apply an ROI strategy; only queried raster tiles expose morphology scores."""

    catalog = candidate_evaluation.drop(
        columns=["evaluation_only_reference_score"], errors="ignore"
    ).copy()
    rng = np.random.default_rng(seed)
    pilot_indices = rng.choice(len(catalog), size=pilot_rois, replace=False)
    catalog["pilot_order"] = np.nan
    for order, index in enumerate(pilot_indices, start=1):
        catalog.loc[int(index), "pilot_order"] = order
    ordered_pilots = catalog.dropna(subset=["pilot_order"]).sort_values("pilot_order")
    queried_ids: set[str] = set()
    records: list[dict[str, object]] = []

    def query(
        selected: pd.Series,
        stage: str,
        selection_score: float = np.nan,
        predicted_mean: float = np.nan,
        predicted_std: float = np.nan,
        anchor_id: str | None = None,
    ) -> None:
        tile = reference_signal[
            :,
            int(selected["row0"]) : int(selected["row1"]),
            int(selected["column0"]) : int(selected["column1"]),
        ]
        time_s, dose = _raster_cost(config, selected)
        records.append(
            {
                "query_index": len(records) + 1,
                "stage": stage,
                **selected.to_dict(),
                "selection_expected_improvement": selection_score,
                "selection_posterior_mean": predicted_mean,
                "selection_posterior_std": predicted_std,
                "confirmation_anchor_roi_id": anchor_id,
                "observed_interest_score": _morphology_interest_score(config, tile),
                "observed_gradient_score": _morphology_objective_score(config, tile, "gradient"),
                "scan_time_s": time_s,
                "dose_proxy": dose,
            }
        )
        queried_ids.add(str(selected["roi_id"]))

    for index in range(pilot_rois + adaptive_rois):
        if index < pilot_rois:
            query(ordered_pilots.iloc[index], "random_pilot")
            continue
        selected, score, mean, std, stage = _outer_selection(
            arm.selection_strategy, catalog, records, queried_ids, rng
        )
        query(selected, stage, score, mean, std)

    anchors: list[str] = []
    if arm.confirm_neighbors:
        base = pd.DataFrame(records)
        selected = base[base["stage"] != "random_pilot"]
        pool = selected if not selected.empty else base
        for _, anchor in (
            pool.sort_values("observed_interest_score", ascending=False)
            .drop_duplicates("roi_id")
            .head(arm.neighbor_anchors)
            .iterrows()
        ):
            anchor_id = str(anchor["roi_id"])
            anchors.append(anchor_id)
            for neighbor in _neighbor_candidates(catalog, anchor, queried_ids):
                query(neighbor, "neighbor_confirmation", anchor_id=anchor_id)

    queried = pd.DataFrame(records).merge(
        candidate_evaluation[["roi_id", "evaluation_only_reference_score"]],
        on="roi_id",
        how="left",
    )
    target = candidate_evaluation.loc[candidate_evaluation["evaluation_only_reference_score"].idxmax()]
    maximum = float(target["evaluation_only_reference_score"])
    threshold = equivalence_fraction * maximum
    queried["reference_equivalent_roi"] = queried["evaluation_only_reference_score"] >= threshold
    recommended = queried.loc[queried["observed_interest_score"].idxmax()]
    best_queried = queried.loc[queried["evaluation_only_reference_score"].idxmax()]
    pilots = queried[queried["stage"] == "random_pilot"]
    best_pilot = pilots.loc[pilots["evaluation_only_reference_score"].idxmax()]
    summary: dict[str, object] = {
        "arm": arm.name,
        "selection_strategy": arm.selection_strategy,
        "measurement_mode": "direct_dense_follow_on_raster",
        "input_signal_semantics": "nonquantitative_multichannel_intensity_proxy",
        "seed": seed,
        "pilot_rois": pilot_rois,
        "adaptive_rois": adaptive_rois,
        "confirm_neighbors": arm.confirm_neighbors,
        "neighbor_anchors": arm.neighbor_anchors if arm.confirm_neighbors else 0,
        "confirmation_anchor_roi_ids": ";".join(anchors),
        "total_queried_rois": len(queried),
        "candidate_count": len(catalog),
        "scan_time_s": float(queried["scan_time_s"].sum()),
        "dose_proxy": float(queried["dose_proxy"].sum()),
        "scan_fraction_of_exhaustive": float(len(queried) / len(catalog)),
        "reference_roi_id": str(target["roi_id"]),
        "reference_roi_score": maximum,
        "equivalence_fraction": equivalence_fraction,
        "recommended_roi_id": str(recommended["roi_id"]),
        "recommended_score": float(recommended["evaluation_only_reference_score"]),
        "recommended_regret_fraction": float((maximum - recommended["evaluation_only_reference_score"]) / maximum),
        "recommended_reference_equivalent": bool(recommended["reference_equivalent_roi"]),
        "best_queried_regret_fraction": float((maximum - best_queried["evaluation_only_reference_score"]) / maximum),
        "best_pilot_regret_fraction": float((maximum - best_pilot["evaluation_only_reference_score"]) / maximum),
    }
    return summary, queried


def wilson_interval(hits: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if count == 0:
        return np.nan, np.nan
    proportion = hits / count
    denominator = 1.0 + z**2 / count
    center = (proportion + z**2 / (2.0 * count)) / denominator
    spread = z * np.sqrt(proportion * (1.0 - proportion) / count + z**2 / (4.0 * count**2)) / denominator
    return float(center - spread), float(center + spread)


def run_stack_validation(
    template_path: Path,
    slice_ids: list[str],
    output: Path,
    manifest_path: Path | None = None,
    arms: list[ValidationArm] | None = None,
    seed: int = 0,
    pilot_rois: int = 4,
    adaptive_rois: int = 8,
    equivalence_fraction: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    template = load_config(template_path)
    arms = arms or DEFAULT_ARMS
    manifest_sources = (
        sources_from_manifest(manifest_path, slice_ids, template.scenario.elements)
        if manifest_path is not None
        else None
    )
    summary_rows: list[dict[str, object]] = []
    output.mkdir(parents=True, exist_ok=True)
    for index, slice_id in enumerate(slice_ids, start=1):
        config = config_for_slice(
            template, slice_id, manifest_sources[slice_id] if manifest_sources else None
        )
        x, y = _source_coordinates_without_values(config)
        candidates = build_roi_catalog(config, x, y)
        source = _full_source(config)
        reference_signal, _, _ = _full_reference_rate(config, source)
        evaluated = _score_reference_rois(config, candidates, reference_signal)
        for arm in arms:
            summary, _ = evaluate_direct_raster_arm(
                config,
                reference_signal,
                evaluated,
                arm,
                seed=seed,
                pilot_rois=pilot_rois,
                adaptive_rois=adaptive_rois,
                equivalence_fraction=equivalence_fraction,
            )
            summary_rows.append({"slice": slice_id, **summary})
        if index % 10 == 0 or index == len(slice_ids):
            pd.DataFrame(summary_rows).to_csv(output / "full_stack_by_slice.partial.csv", index=False)
            print(f"Validated {index}/{len(slice_ids)} slices.")
    detail = pd.DataFrame(summary_rows)
    aggregate_rows = []
    for arm, frame in detail.groupby("arm", sort=False):
        hits = int(frame["recommended_reference_equivalent"].sum())
        lower, upper = wilson_interval(hits, len(frame))
        aggregate_rows.append(
            {
                "arm": arm,
                "slices": len(frame),
                "reference_equivalent_hits": hits,
                "hit_rate": hits / len(frame),
                "hit_rate_wilson_95_lower": lower,
                "hit_rate_wilson_95_upper": upper,
                "mean_queried_rois": frame["total_queried_rois"].mean(),
                "mean_scan_fraction_of_exhaustive": frame["scan_fraction_of_exhaustive"].mean(),
                "mean_recommended_regret_fraction": frame["recommended_regret_fraction"].mean(),
                "median_recommended_regret_fraction": frame["recommended_regret_fraction"].median(),
                "p95_recommended_regret_fraction": frame["recommended_regret_fraction"].quantile(0.95),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        ["mean_recommended_regret_fraction", "mean_scan_fraction_of_exhaustive"]
    )
    detail.to_csv(output / "full_stack_by_slice.csv", index=False)
    aggregate.to_csv(output / "full_stack_summary.csv", index=False)
    protocol = {
        "template_config": str(template_path),
        "download_manifest": str(manifest_path) if manifest_path is not None else None,
        "measurement_mode": "direct_dense_follow_on_raster",
        "morphology_reference_definition": "ROI score at least 99% of withheld dense-map maximum",
        "slice_count": len(slice_ids),
        "slice_ids": slice_ids,
        "seed": seed,
        "pilot_rois": pilot_rois,
        "adaptive_rois": adaptive_rois,
        "equivalence_fraction": equivalence_fraction,
        "arms": [arm.__dict__ for arm in arms],
        "claim_limit": "Score-equivalence is not posterior certainty or expert-confirmed corrosion relevance.",
    }
    with (output / "full_stack_protocol.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(protocol, stream, sort_keys=False)
    return detail, aggregate
