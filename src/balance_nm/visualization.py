"""Publication-oriented diagnostic figures for experiment and benchmark artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from .domain import ExperimentState, Recommendation


def plot_run(
    hidden_sample,
    state: ExperimentState,
    prediction,
    recommendation: Recommendation,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    element = str(prediction.coords["element"].values[0])
    extent = [
        float(prediction.coords["x"].min()),
        float(prediction.coords["x"].max()),
        float(prediction.coords["y"].min()),
        float(prediction.coords["y"].max()),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    has_interface = bool(prediction["interface_weight"].values.any())
    if has_interface:
        maps = [
            (hidden_sample["true_rate"].sel(element=element), f"Truth: {element} rate", "viridis"),
            (prediction["mean_rate"].sel(element=element), f"Posterior mean: {element}", "viridis"),
            (prediction["variance_rate"].sum("element"), "Total posterior variance", "magma"),
            (hidden_sample["true_phase"], "True cladding phase", "coolwarm"),
            (prediction["phase_probability"], "Predicted phase probability", "coolwarm"),
            (prediction["interface_weight"], "Interface acquisition weight", "inferno"),
        ]
    else:
        maps = [
            (hidden_sample["true_rate"].sel(element=element), f"Reference: {element} signal", "viridis"),
            (prediction["mean_rate"].sel(element=element), f"Posterior mean: {element}", "viridis"),
            (prediction["variance_rate"].sum("element"), "Total posterior variance", "magma"),
            (prediction["feature_signal"].sel(element=element), "Normalized signal feature", "viridis"),
            (prediction["quality_score"], "Data quality score", "cividis"),
            (prediction["pattern_interest"].max("objective"), "Maximum pattern interest", "inferno"),
        ]
    for axis, (data, title, cmap) in zip(axes.ravel(), maps):
        image = axis.imshow(data.values, origin="lower", extent=extent, cmap=cmap, aspect="equal")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046)
    if state.observations is not None:
        axes[0, 1].scatter(
            state.observations["x_nm"].values,
            state.observations["y_nm"].values,
            s=1,
            c="white",
            alpha=0.35,
        )
    if recommendation.action.bounds_nm is not None and recommendation.action.action_type != "stop":
        x0, x1, y0, y1 = recommendation.action.bounds_nm
        axes[1, 2].add_patch(
            Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", linewidth=2)
        )
    axes[1, 2].set_title(f"Next: {recommendation.action.action_type}")
    for axis in axes.ravel():
        axis.set_xlabel("x (nm)")
        axis.set_ylabel("y (nm)")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_benchmark_summary(summary: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = summary.sort_values("interface_error_cost_auc_mean")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].bar(
        ordered["policy"],
        ordered["interface_error_cost_auc_mean"],
        color="#376795",
    )
    axes[0].set_ylabel("Interface error-cost AUC (nm)")
    axes[0].set_title("Acquisition efficiency")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(
        ordered["policy"],
        ordered["final_interface_mean_distance_nm_mean"],
        color="#9b4d40",
    )
    axes[1].set_ylabel("Final mean interface error (nm)")
    axes[1].set_title("Final localization")
    axes[1].tick_params(axis="x", rotation=30)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_v2_products(prediction, output: Path) -> None:
    if "pattern_interest" not in prediction or "quality_score" not in prediction:
        return
    objectives = [str(value) for value in prediction.coords["objective"].values]
    columns = min(3, max(1, len(objectives)))
    rows = int(np.ceil((len(objectives) + 1) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    image = axes[0].imshow(prediction["quality_score"].values, origin="lower", cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("Data quality score")
    figure.colorbar(image, ax=axes[0], fraction=0.046)
    for axis, objective in zip(axes[1:], objectives):
        image = axis.imshow(
            prediction["pattern_interest"].sel(objective=objective).values,
            origin="lower",
            cmap="inferno",
            vmin=0,
            vmax=1,
        )
        axis.set_title(f"Interest: {objective}")
        figure.colorbar(image, ax=axis, fraction=0.046)
    for axis in axes[len(objectives) + 1:]:
        axis.axis("off")
    figure.savefig(output, dpi=160)
    plt.close(figure)
