"""Leakage-free retrospective search over pre-registered elemental-map ROIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from .data import ingest_dataset
from .domain import RunConfig
from .experiment import RunResult, run_replay_experiment


@dataclass
class RoiSearchResult:
    config: RunConfig
    policy: str
    seed: int
    candidates: pd.DataFrame
    queried: pd.DataFrame
    summary: dict[str, object]
    reference_interest: xr.DataArray


def _axis_origins(size: int, span: int) -> list[int]:
    if span > size:
        raise ValueError("ROI dimensions cannot exceed the full replay map")
    origins = list(range(0, size - span + 1, span))
    if origins[-1] != size - span:
        origins.append(size - span)
    return origins


def _template_roi_shape(config: RunConfig) -> tuple[int, int]:
    crop = config.dataset.spatial_crop_indices
    if crop is not None:
        row0, row1, column0, column1 = crop
        return row1 - row0, column1 - column0
    return config.scenario.grid_rows, config.scenario.grid_columns


def _full_source(config: RunConfig) -> xr.Dataset:
    if config.dataset.mode != "replay":
        raise ValueError("ROI search requires a replay dataset configuration")
    if config.dataset.adapter not in ("binary_element_map", "element_map_images"):
        raise ValueError("ROI search currently supports dense binary or image element-map adapters")
    full_config = config.model_copy(
        update={
            "dataset": config.dataset.model_copy(update={"spatial_crop_indices": None})
        }
    )
    source, _ = ingest_dataset(full_config)
    return source


def _source_coordinates_without_values(config: RunConfig) -> tuple[np.ndarray, np.ndarray]:
    """Read only source geometry needed to pre-register ROIs before acquisition."""

    dataset = config.dataset
    if dataset.map_alignment.method != "strict" or dataset.map_alignment.crops:
        raise ValueError("ROI search requires strict native-grid source geometry")
    source = Path(next(iter(dataset.element_map_sources.values())))
    if dataset.adapter == "binary_element_map":
        if dataset.binary_dimensions_from_header:
            dtype = {
                "uint32_le": "<u4",
                "uint16_le": "<u2",
                "float32_le": "<f4",
            }[dataset.binary_dtype]
            header = np.fromfile(source, dtype=dtype, count=2)
            if header.size != 2:
                raise ValueError("binary ROI-search source has no dimension header")
            columns, rows = int(header[0]), int(header[1])
        else:
            rows, columns = dataset.binary_shape
    elif dataset.adapter == "element_map_images":
        with Image.open(source) as image:
            columns, rows = image.size
    else:
        raise ValueError("ROI search currently supports dense binary or image element-map adapters")
    step_x = dataset.x_step_nm or config.instrument.fine_step_nm
    step_y = dataset.y_step_nm or step_x
    return (np.arange(columns) + 0.5) * step_x, (np.arange(rows) + 0.5) * step_y


def build_roi_catalog(config: RunConfig, x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Build an auditable candidate catalog independently of element values."""

    roi_rows, roi_columns = _template_roi_shape(config)
    records = []
    for row0 in _axis_origins(len(y), roi_rows):
        for column0 in _axis_origins(len(x), roi_columns):
            row1, column1 = row0 + roi_rows, column0 + roi_columns
            records.append(
                {
                    "roi_id": f"r{row0:04d}_c{column0:04d}",
                    "row0": row0,
                    "row1": row1,
                    "column0": column0,
                    "column1": column1,
                    "center_x_nm": float((x[column0] + x[column1 - 1]) / 2.0),
                    "center_y_nm": float((y[row0] + y[row1 - 1]) / 2.0),
                }
            )
    return pd.DataFrame(records)


def _roi_config(config: RunConfig, row: pd.Series) -> RunConfig:
    crop = (
        int(row["row0"]),
        int(row["row1"]),
        int(row["column0"]),
        int(row["column1"]),
    )
    return config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"spatial_crop_indices": crop})}
    )


def _morphology_evidence_maps(config: RunConfig, signal: np.ndarray) -> dict[str, np.ndarray]:
    """Map spatial morphology evidence after monotonic signal compression."""

    features = 2.0 * np.sqrt(np.maximum(signal, 0.0) + 3.0 / 8.0)
    maps: dict[str, np.ndarray] = {}
    gradient = np.zeros(features.shape[1:], dtype=float)
    for channel in features:
        dy, dx = np.gradient(channel)
        gradient += np.hypot(dx, dy)
    maps["gradient"] = gradient
    local = np.stack([gaussian_filter(channel, sigma=2.0) for channel in features])
    residual = np.sqrt(np.sum((features - local) ** 2, axis=0))
    maps["anomaly"] = residual
    maps["inclusion"] = gaussian_filter(residual, sigma=0.7)
    if "clustering" in config.objectives.enabled:
        points = features.reshape(features.shape[0], -1).T
        if np.unique(points, axis=0).shape[0] < 2:
            maps["clustering"] = np.zeros(features.shape[1:], dtype=float)
        else:
            mixture = GaussianMixture(n_components=2, random_state=0, covariance_type="full")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                membership = mixture.fit_predict(points)
            if np.unique(membership).size < 2:
                maps["clustering"] = np.zeros(features.shape[1:], dtype=float)
                return maps
            membership_probability = mixture.predict_proba(points).max(axis=1)
            labels = membership.reshape(features.shape[1:])
            boundary = np.zeros_like(labels, dtype=float)
            boundary[:, 1:] += labels[:, 1:] != labels[:, :-1]
            boundary[1:, :] += labels[1:, :] != labels[:-1, :]
            maps["clustering"] = boundary + (1.0 - membership_probability.reshape(labels.shape))
    return maps


def _morphology_interest_score(config: RunConfig, signal: np.ndarray) -> float:
    """Comparable ROI score from multichannel spatial patterns."""

    evidence = _morphology_evidence_maps(config, signal)
    total_weight = 0.0
    total_score = 0.0
    for objective, weight in config.objectives.weights.items():
        if weight <= 0 or objective not in evidence:
            continue
        values = evidence[objective].ravel()
        count = max(1, int(np.ceil(0.1 * values.size)))
        high_values = np.partition(values, values.size - count)[-count:]
        total_score += weight * float(np.mean(high_values))
        total_weight += weight
    return total_score / max(total_weight, 1e-12)


def _morphology_objective_score(config: RunConfig, signal: np.ndarray, objective: str) -> float:
    """Comparable single-objective ROI evidence score used by selection baselines."""

    evidence = _morphology_evidence_maps(config, signal)
    if objective not in evidence:
        return 0.0
    values = evidence[objective].ravel()
    count = max(1, int(np.ceil(0.1 * values.size)))
    return float(np.mean(np.partition(values, values.size - count)[-count:]))


def _normalized_coordinates(candidates: pd.DataFrame) -> np.ndarray:
    xy = candidates[["center_x_nm", "center_y_nm"]].to_numpy(float)
    scale = np.maximum(xy.max(axis=0) - xy.min(axis=0), 1.0)
    return (xy - xy.min(axis=0)) / scale


def _spatial_gp_prediction(
    candidates: pd.DataFrame, records: list[dict], score_key: str
) -> tuple[np.ndarray, np.ndarray]:
    coords = _normalized_coordinates(candidates)
    index_by_id = {roi_id: index for index, roi_id in enumerate(candidates["roi_id"])}
    observed_indices = np.asarray([index_by_id[record["roi_id"]] for record in records], dtype=int)
    observed_scores = np.asarray([record[score_key] for record in records], dtype=float)
    model = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0, constant_value_bounds="fixed")
        * Matern(length_scale=0.3, length_scale_bounds="fixed", nu=1.5),
        alpha=1e-5,
        normalize_y=True,
        optimizer=None,
        random_state=0,
    )
    model.fit(coords[observed_indices], observed_scores)
    return model.predict(coords, return_std=True)


def _expected_improvement_selection(
    candidates: pd.DataFrame,
    records: list[dict],
    excluded_ids: set[str],
    score_key: str = "observed_interest_score",
) -> tuple[pd.Series, float, float, float]:
    index_by_id = {roi_id: index for index, roi_id in enumerate(candidates["roi_id"])}
    observed_scores = np.asarray([record[score_key] for record in records], dtype=float)
    mean, std = _spatial_gp_prediction(candidates, records, score_key)
    improvement = mean - float(observed_scores.max()) - 0.001
    z = improvement / np.maximum(std, 1e-12)
    expected_improvement = improvement * norm.cdf(z) + std * norm.pdf(z)
    for roi_id in excluded_ids:
        expected_improvement[index_by_id[roi_id]] = -np.inf
    selected_index = int(np.argmax(expected_improvement))
    return (
        candidates.iloc[selected_index],
        float(expected_improvement[selected_index]),
        float(mean[selected_index]),
        float(std[selected_index]),
    )


def _uncertainty_selection(
    candidates: pd.DataFrame, records: list[dict], excluded_ids: set[str]
) -> tuple[pd.Series, float, float, float]:
    index_by_id = {roi_id: index for index, roi_id in enumerate(candidates["roi_id"])}
    mean, std = _spatial_gp_prediction(candidates, records, "observed_interest_score")
    utility = std.copy()
    for roi_id in excluded_ids:
        utility[index_by_id[roi_id]] = -np.inf
    selected_index = int(np.argmax(utility))
    return (
        candidates.iloc[selected_index],
        float(utility[selected_index]),
        float(mean[selected_index]),
        float(std[selected_index]),
    )


def _outer_selection(
    strategy: str,
    candidates: pd.DataFrame,
    records: list[dict],
    excluded_ids: set[str],
    rng: np.random.Generator,
) -> tuple[pd.Series, float, float, float, str]:
    if strategy == "balance":
        selected = _expected_improvement_selection(candidates, records, excluded_ids)
        return (*selected, "bayesian_expected_improvement")
    if strategy == "gradient":
        selected = _expected_improvement_selection(
            candidates, records, excluded_ids, score_key="observed_gradient_score"
        )
        return (*selected, "gradient_expected_improvement")
    if strategy == "uncertainty":
        selected = _uncertainty_selection(candidates, records, excluded_ids)
        return (*selected, "uncertainty_exploration")
    eligible = candidates[~candidates["roi_id"].isin(excluded_ids)]
    if strategy == "uniform":
        selected = eligible.sort_values(["row0", "column0"]).iloc[0]
        return selected, np.nan, np.nan, np.nan, "uniform_order"
    if strategy == "random":
        selected = eligible.iloc[int(rng.integers(0, len(eligible)))]
        return selected, np.nan, np.nan, np.nan, "random_selection"
    raise ValueError(f"unknown ROI selection strategy: {strategy}")


def _neighbor_candidates(
    candidates: pd.DataFrame, anchor: pd.Series, excluded_ids: set[str]
) -> list[pd.Series]:
    """Return the unqueried eight-cell neighborhood around an anchor ROI."""

    rows = sorted(candidates["row0"].unique().tolist())
    columns = sorted(candidates["column0"].unique().tolist())
    row_index = rows.index(int(anchor["row0"]))
    column_index = columns.index(int(anchor["column0"]))
    selected: list[pd.Series] = []
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            if row_offset == 0 and column_offset == 0:
                continue
            candidate_row = row_index + row_offset
            candidate_column = column_index + column_offset
            if not (0 <= candidate_row < len(rows) and 0 <= candidate_column < len(columns)):
                continue
            match = candidates[
                (candidates["row0"] == rows[candidate_row])
                & (candidates["column0"] == columns[candidate_column])
            ]
            if match.empty:
                continue
            neighbor = match.iloc[0]
            if str(neighbor["roi_id"]) not in excluded_ids:
                selected.append(neighbor)
    return selected


def _full_reference_rate(config: RunConfig, full_source: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.unique(full_source["x_nm"].values)
    y = np.unique(full_source["y_nm"].values)
    counts = full_source["counts"].values.reshape(len(y), len(x), -1).transpose(2, 0, 1).astype(float)
    dwell = full_source["dwell_ms"].values.reshape(len(y), len(x)).astype(float)
    rate = np.empty_like(counts)
    for index, element in enumerate(config.scenario.elements):
        sensitivity = config.instrument.sensitivity[element]
        background = config.instrument.background_rate[element]
        rate[index] = np.maximum((counts[index] / dwell - background) / sensitivity, 0.0)
    return rate, x, y


def _full_reference_interest(
    config: RunConfig, rate: np.ndarray, x: np.ndarray, y: np.ndarray
) -> xr.DataArray:
    """Construct a display surface from withheld dense maps after ROI selection."""

    maps = _morphology_evidence_maps(config, rate)
    weighted = np.zeros((len(y), len(x)), dtype=float)
    total_weight = 0.0
    for objective, weight in config.objectives.weights.items():
        if weight > 0 and objective in maps:
            weighted += weight * maps[objective]
            total_weight += weight
    return xr.DataArray(
        weighted / max(total_weight, 1e-12),
        dims=("y", "x"),
        coords={"x": x, "y": y},
        name="reference_interest",
        attrs={"role": "post_selection_evaluation_only"},
    )


def _score_reference_rois(
    config: RunConfig, candidates: pd.DataFrame, reference_rate: np.ndarray
) -> pd.DataFrame:
    result = candidates.copy()
    scores = []
    for _, row in result.iterrows():
        values = reference_rate[
            :,
            int(row["row0"]) : int(row["row1"]),
            int(row["column0"]) : int(row["column1"]),
        ]
        scores.append(_morphology_interest_score(config, values))
    result["evaluation_only_reference_score"] = scores
    return result


def run_roi_search(
    config: RunConfig,
    policy: str,
    seed: int,
    pilot_rois: int,
    bayesian_rois: int,
    selection_strategy: str = "balance",
    equivalence_fraction: float = 0.99,
    confirm_neighbors: bool = True,
    neighbor_anchors: int = 1,
    save_run: Callable[[RunResult, Path], None] | None = None,
    run_output: Path | None = None,
) -> RoiSearchResult:
    """Search ROI locations using observations only; apply dense reference after selection."""

    if pilot_rois < 2:
        raise ValueError("ROI search needs at least two random pilot ROIs")
    if not 0.0 < equivalence_fraction <= 1.0:
        raise ValueError("ROI equivalence fraction must be in (0, 1]")
    if neighbor_anchors < 1:
        raise ValueError("neighbor_anchors must be at least one")
    if selection_strategy not in {"balance", "gradient", "uncertainty", "uniform", "random"}:
        raise ValueError(f"unknown ROI selection strategy: {selection_strategy}")
    x, y = _source_coordinates_without_values(config)
    candidates = build_roi_catalog(config, x, y)
    total_queries = pilot_rois + bayesian_rois
    if total_queries > len(candidates):
        raise ValueError("requested ROI queries exceed the pre-registered candidate catalog")
    rng = np.random.default_rng(seed)
    pilot_indices = rng.choice(len(candidates), size=pilot_rois, replace=False)
    candidates = candidates.copy()
    candidates["pilot_order"] = np.nan
    for order, index in enumerate(pilot_indices, start=1):
        candidates.loc[int(index), "pilot_order"] = order

    records: list[dict] = []
    queried_ids: set[str] = set()
    ordered_pilots = candidates.dropna(subset=["pilot_order"]).sort_values("pilot_order")

    def query(
        selected: pd.Series,
        stage: str,
        selection_score: float = np.nan,
        predicted_mean: float = np.nan,
        predicted_std: float = np.nan,
        confirmation_anchor_roi_id: str | None = None,
    ) -> None:
        sequence = len(records)
        roi_config = _roi_config(config, selected)
        source, capabilities = ingest_dataset(roi_config)
        result = run_replay_experiment(roi_config, policy, source, capabilities, seed + sequence)
        observed_score = _morphology_interest_score(
            config, result.final_prediction["mean_rate"].values
        )
        observed_gradient_score = _morphology_objective_score(
            config, result.final_prediction["mean_rate"].values, "gradient"
        )
        record = {
            "query_index": sequence + 1,
            "stage": stage,
            **selected.to_dict(),
            "selection_expected_improvement": selection_score,
            "selection_posterior_mean": predicted_mean,
            "selection_posterior_std": predicted_std,
            "confirmation_anchor_roi_id": confirmation_anchor_roi_id,
            "observed_interest_score": observed_score,
            "observed_gradient_score": observed_gradient_score,
            "scan_time_s": result.metrics[-1].scan_time_s,
            "normalized_channel_rmse": result.metrics[-1].normalized_channel_rmse,
        }
        records.append(record)
        queried_ids.add(str(selected["roi_id"]))
        if save_run is not None and run_output is not None:
            save_run(result, run_output / "runs" / f"query_{sequence + 1:02d}_{selected['roi_id']}")

    for sequence in range(total_queries):
        if sequence < pilot_rois:
            selected = ordered_pilots.iloc[sequence]
            stage = "random_pilot"
            query(selected, stage)
        else:
            selected, selection_score, predicted_mean, predicted_std, stage = _outer_selection(
                selection_strategy, candidates, records, queried_ids, rng
            )
            query(selected, stage, selection_score, predicted_mean, predicted_std)

    confirmation_anchor_ids: list[str] = []
    if confirm_neighbors:
        base_records = pd.DataFrame(records)
        selected_records = base_records[
            base_records["stage"] != "random_pilot"
        ]
        anchor_pool = selected_records if not selected_records.empty else base_records
        ranked = anchor_pool.sort_values(
            "observed_interest_score", ascending=False
        )
        for _, anchor in ranked.drop_duplicates("roi_id").head(neighbor_anchors).iterrows():
            anchor_id = str(anchor["roi_id"])
            confirmation_anchor_ids.append(anchor_id)
            for neighbor in _neighbor_candidates(candidates, anchor, queried_ids):
                query(
                    neighbor,
                    "neighbor_confirmation",
                    confirmation_anchor_roi_id=anchor_id,
                )

    # Source values and the evaluation-only reference are loaded only after all choices are made.
    full_source = _full_source(config)
    reference_rate, x, y = _full_reference_rate(config, full_source)
    reference_interest = _full_reference_interest(config, reference_rate, x, y)
    candidate_evaluation = _score_reference_rois(config, candidates, reference_rate)
    target = candidate_evaluation.loc[
        candidate_evaluation["evaluation_only_reference_score"].idxmax()
    ]
    queried = pd.DataFrame(records).merge(
        candidate_evaluation[["roi_id", "evaluation_only_reference_score"]],
        on="roi_id",
        how="left",
    )
    queried["distance_to_reference_roi_nm"] = np.hypot(
        queried["center_x_nm"] - float(target["center_x_nm"]),
        queried["center_y_nm"] - float(target["center_y_nm"]),
    )
    reference_max = float(target["evaluation_only_reference_score"])
    equivalent_threshold = equivalence_fraction * reference_max
    candidate_evaluation["reference_equivalent_roi"] = (
        candidate_evaluation["evaluation_only_reference_score"] >= equivalent_threshold
    )
    queried["reference_equivalent_roi"] = (
        queried["evaluation_only_reference_score"] >= equivalent_threshold
    )
    best_observed = queried.loc[queried["observed_interest_score"].idxmax()]
    nearest_queried = queried.loc[queried["distance_to_reference_roi_nm"].idxmin()]
    best_evaluated_query = queried.loc[queried["evaluation_only_reference_score"].idxmax()]
    equivalent_queries = queried[queried["reference_equivalent_roi"]]
    pilots = queried[queried["stage"] == "random_pilot"]
    best_pilot = pilots.loc[pilots["evaluation_only_reference_score"].idxmax()]
    recommended_equivalent = bool(best_observed["reference_equivalent_roi"])
    if recommended_equivalent:
        status = "recommended_roi_is_near_optimal"
    elif not equivalent_queries.empty:
        status = "near_optimal_roi_queried_but_not_resolved"
    else:
        status = "near_optimal_roi_not_queried"
    total_scan_time_s = float(queried["scan_time_s"].sum())
    estimated_exhaustive_time_s = float(
        len(candidates) * queried["scan_time_s"].mean()
    )
    summary: dict[str, object] = {
        "seed": seed,
        "policy": policy,
        "within_roi_policy": policy,
        "selection_strategy": selection_strategy,
        "pilot_rois": pilot_rois,
        "bayesian_rois": bayesian_rois,
        "confirm_neighbors": confirm_neighbors,
        "neighbor_anchors": neighbor_anchors if confirm_neighbors else 0,
        "confirmation_anchor_basis": (
            f"{selection_strategy}_selected_roi"
            if confirm_neighbors and bayesian_rois > 0
            else "queried_fallback_no_bayesian_stage"
            if confirm_neighbors
            else "disabled"
        ),
        "confirmation_anchor_roi_ids": confirmation_anchor_ids,
        "neighbor_confirmation_rois": int((queried["stage"] == "neighbor_confirmation").sum()),
        "total_queried_rois": len(queried),
        "candidate_count": len(candidates),
        "queried_roi_fraction": float(len(queried) / len(candidates)),
        "total_scan_time_s": total_scan_time_s,
        "estimated_exhaustive_same_protocol_time_s": estimated_exhaustive_time_s,
        "estimated_time_fraction_vs_exhaustive_same_protocol": float(
            total_scan_time_s / max(estimated_exhaustive_time_s, 1e-12)
        ),
        "reference_roi_id": str(target["roi_id"]),
        "reference_roi_score": float(target["evaluation_only_reference_score"]),
        "equivalence_fraction": equivalence_fraction,
        "equivalence_score_threshold": equivalent_threshold,
        "equivalent_roi_count": int(candidate_evaluation["reference_equivalent_roi"].sum()),
        "best_observed_roi_id": str(best_observed["roi_id"]),
        "best_observed_distance_to_reference_nm": float(best_observed["distance_to_reference_roi_nm"]),
        "recommended_roi_reference_score": float(best_observed["evaluation_only_reference_score"]),
        "recommended_roi_regret_fraction": float(
            (reference_max - best_observed["evaluation_only_reference_score"])
            / max(reference_max, 1e-12)
        ),
        "recommended_roi_reference_equivalent": recommended_equivalent,
        "best_queried_evaluation_roi_id": str(best_evaluated_query["roi_id"]),
        "best_queried_reference_score": float(best_evaluated_query["evaluation_only_reference_score"]),
        "best_queried_regret_fraction": float(
            (reference_max - best_evaluated_query["evaluation_only_reference_score"])
            / max(reference_max, 1e-12)
        ),
        "best_pilot_roi_id": str(best_pilot["roi_id"]),
        "best_pilot_reference_score": float(best_pilot["evaluation_only_reference_score"]),
        "best_pilot_regret_fraction": float(
            (reference_max - best_pilot["evaluation_only_reference_score"])
            / max(reference_max, 1e-12)
        ),
        "bayesian_reference_score_improvement_over_pilots": float(
            best_evaluated_query["evaluation_only_reference_score"]
            - best_pilot["evaluation_only_reference_score"]
        ),
        "nearest_queried_roi_id": str(nearest_queried["roi_id"]),
        "nearest_queried_distance_to_reference_nm": float(nearest_queried["distance_to_reference_roi_nm"]),
        "reference_roi_queried": bool(str(target["roi_id"]) in queried_ids),
        "equivalent_roi_queried": bool(not equivalent_queries.empty),
        "first_equivalent_query_index": (
            int(equivalent_queries["query_index"].min())
            if not equivalent_queries.empty
            else -1
        ),
        "selection_status": status,
    }
    return RoiSearchResult(
        config=config,
        policy=policy,
        seed=seed,
        candidates=candidate_evaluation,
        queried=queried,
        summary=summary,
        reference_interest=reference_interest,
    )


def save_roi_search_artifacts(result: RoiSearchResult, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result.config.model_dump(mode="json"), handle, sort_keys=False)
    result.candidates.drop(
        columns=["evaluation_only_reference_score", "reference_equivalent_roi"]
    ).to_csv(
        output / "pre_registered_roi_catalog.csv", index=False
    )
    result.candidates.to_csv(output / "pre_registered_roi_catalog_evaluated.csv", index=False)
    result.queried.to_csv(output / "queried_rois.csv", index=False)
    result.reference_interest.to_dataset().to_zarr(output / "evaluation_reference_interest.zarr", mode="w")
    with (output / "roi_search_summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result.summary, handle, sort_keys=False)
    target_id = str(result.summary["reference_roi_id"])
    target = result.candidates[result.candidates["roi_id"] == target_id].iloc[0]
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = axis.imshow(
        result.reference_interest.values,
        origin="lower",
        extent=[
            float(result.reference_interest.x.min()),
            float(result.reference_interest.x.max()),
            float(result.reference_interest.y.min()),
            float(result.reference_interest.y.max()),
        ],
        cmap="magma",
        aspect="auto",
    )
    for _, row in result.queried.iterrows():
        x0 = float(result.reference_interest.x.values[int(row["column0"])])
        y0 = float(result.reference_interest.y.values[int(row["row0"])])
        width = float(result.reference_interest.x.values[int(row["column1"]) - 1] - x0)
        height = float(result.reference_interest.y.values[int(row["row1"]) - 1] - y0)
        axis.add_patch(
            plt.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor="lime" if bool(row["reference_equivalent_roi"]) else "cyan",
                linewidth=2 if bool(row["reference_equivalent_roi"]) else 1,
            )
        )
    x0 = float(result.reference_interest.x.values[int(target["column0"])])
    y0 = float(result.reference_interest.y.values[int(target["row0"])])
    width = float(result.reference_interest.x.values[int(target["column1"]) - 1] - x0)
    height = float(result.reference_interest.y.values[int(target["row1"]) - 1] - y0)
    axis.add_patch(
        plt.Rectangle(
            (x0, y0),
            width,
            height,
            fill=False,
            edgecolor="white",
            linewidth=2,
            linestyle="--",
        )
    )
    axis.set_title("Evaluation-only reference interest and queried ROI coverage")
    axis.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="cyan", label="queried"),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fill=False,
                edgecolor="lime",
                linewidth=2,
                label="queried within 99% of maximum",
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fill=False,
                edgecolor="white",
                linewidth=2,
                linestyle="--",
                label="evaluation-only maximum",
            ),
        ],
        loc="upper left",
        fontsize="small",
    )
    axis.set_xlabel("x (nm)")
    axis.set_ylabel("y (nm)")
    figure.colorbar(image, ax=axis, label="reference interest proxy")
    figure.savefig(output / "roi_search_reference_evaluation.png", dpi=180)
    plt.close(figure)
