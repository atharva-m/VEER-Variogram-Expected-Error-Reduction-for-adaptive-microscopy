"""Render a Truth | Baseline | VEER acquisition animation as a GIF.

Left panel: the dense slice with its frozen unsupervised proxy front.
Middle/right panels: the baseline and VEER policies revealing tiles in
slow motion, with each policy's predicted front converging onto the proxy.

Usage:
    python scripts/render_acquisition_gif.py --slice 032 --out assets/veer_acquisition.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from veer.data import ingest_dataset
from veer.io import load_config
from veer.morphology import (
    dense_signal_from_observations,
    front_from_probability,
    pseudo_reference_from_dense_signal,
    reconstruct_from_observed_mask,
)
from veer.replay import build_roi_catalog, config_for_slice, manifest_slice_ids, sources_from_manifest
from veer import run_veer_slice_replay

ROOT = Path(__file__).resolve().parents[1]
PROXY = "#d83b2f"   # proxy / truth front
BASE = "#3266ad"    # baseline predicted front
VEER = "#1d9e75"    # VEER predicted front


def _robust_image(channel: np.ndarray) -> np.ndarray:
    finite = channel[np.isfinite(channel)]
    low, high = np.percentile(finite, [1.0, 99.0]) if finite.size else (0.0, 1.0)
    return np.clip((channel - low) / max(high - low, 1e-9), 0.0, 1.0)


def _front_points(front_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.where(np.asarray(front_mask, dtype=bool))
    return cols, rows


def _reveal_order(result) -> list[str]:
    return list(result.metrics.sort_values("iteration")["roi_id"])


def _states(result, dense_signal, x, y, channels, config, catalog):
    """Per-step observed mask, predicted front, composite error, area fraction."""
    by_id = catalog.set_index("roi_id")
    order = _reveal_order(result)
    errors = list(result.metrics.sort_values("iteration")["morphology_composite_error"])
    mask = np.zeros(dense_signal.shape[1:], dtype=bool)
    states = []
    for k, roi_id in enumerate(order):
        roi = by_id.loc[roi_id]
        mask[int(roi["row0"]): int(roi["row1"]), int(roi["column0"]): int(roi["column1"])] = True
        prediction = reconstruct_from_observed_mask(dense_signal, mask, x, y, channels, config)
        front, _ = front_from_probability(prediction["altered_region_probability"].values, x, y, config)
        rects = [
            (int(by_id.loc[r]["column0"]), int(by_id.loc[r]["row0"]),
             int(by_id.loc[r]["column1"] - by_id.loc[r]["column0"]),
             int(by_id.loc[r]["row1"] - by_id.loc[r]["row0"]))
            for r in order[: k + 1]
        ]
        states.append(
            {
                "rects": rects,
                "front": _front_points(front),
                "error": float(errors[k]),
                "area": float(mask.mean()),
                "tiles": k + 1,
            }
        )
    return states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/alloy617_veer.yaml"))
    parser.add_argument("--manifest", default=str(ROOT / "data/alloy617_nrds/full_stack_download_manifest.csv"))
    parser.add_argument("--slice", default="032")
    parser.add_argument("--baseline", default="uncertainty_lookahead")
    parser.add_argument("--veer", default="gated_veer_4x4_mean_kappa5")
    parser.add_argument("--channel", default="Cr")
    parser.add_argument("--out", default=str(ROOT / "assets/veer_acquisition.gif"))
    parser.add_argument("--fps", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    template = load_config(Path(args.config))
    manifest = Path(args.manifest)
    sources = sources_from_manifest(manifest, manifest_slice_ids(manifest), template.scenario.elements)
    config = config_for_slice(template, args.slice, sources[args.slice])
    observations, _ = ingest_dataset(config)
    dense_signal, x, y, channels = dense_signal_from_observations(config, observations)
    catalog = build_roi_catalog(x, y, config.acquisition.roi_size_px)

    reference = pseudo_reference_from_dense_signal(dense_signal, x, y, channels, config)
    proxy_cols, proxy_rows = _front_points(reference["pseudo_front"].values)
    background = _robust_image(dense_signal[channels.index(args.channel)])

    base_states = _states(
        run_veer_slice_replay(config, dense_signal, x, y, channels, args.slice, args.baseline, seed=args.seed),
        dense_signal, x, y, channels, config, catalog,
    )
    veer_states = _states(
        run_veer_slice_replay(config, dense_signal, x, y, channels, args.slice, args.veer, seed=args.seed),
        dense_signal, x, y, channels, config, catalog,
    )
    steps = len(base_states)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9), dpi=72)
    fig.patch.set_facecolor("white")

    def draw_panel(ax, title, image, dim, state):
        ax.clear()
        ax.imshow(image, cmap="gray", origin="upper", vmin=0, vmax=1, alpha=1.0 if dim is None else 0.45)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=12)
        # proxy front (truth) on every panel, faint on policy panels
        ax.plot(proxy_cols, proxy_rows, ".", color=PROXY,
                ms=2.6 if dim is None else 2.0,
                alpha=1.0 if dim is None else 0.6, zorder=5)
        if state is not None:
            for (cx, cy, w, h) in state["rects"]:
                ax.add_patch(Rectangle((cx - 0.5, cy - 0.5), w, h, facecolor=dim, edgecolor="none", alpha=0.30))
            fc, fr = state["front"]
            ax.plot(fc, fr, ".", color=dim, ms=2.6, zorder=6)
            ax.set_xlabel(
                f"{state['tiles']} tiles · {100 * state['area']:.0f}% scanned · error {state['error']:.3f}",
                fontsize=10,
            )

    def update(frame):
        k = min(frame, steps - 1)
        draw_panel(axes[0], "Truth: slice + proxy front", background, None, None)
        axes[0].set_xlabel("frozen unsupervised front (target)", fontsize=10)
        draw_panel(axes[1], "Baseline: coverage", background, BASE, base_states[k])
        draw_panel(axes[2], "VEER: variogram + front weighting", background, VEER, veer_states[k])
        fig.suptitle("Adaptive corrosion-front acquisition  ·  slice " + args.slice, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return []

    frames = list(range(steps)) + [steps - 1] * 4  # hold the final frame
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=animation.PillowWriter(fps=args.fps))
    update(steps - 1)
    final_png = out.with_name(out.stem + "_final.png")
    fig.savefig(str(final_png), dpi=110, bbox_inches="tight")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")
    print(f"wrote {final_png}")


if __name__ == "__main__":
    main()
