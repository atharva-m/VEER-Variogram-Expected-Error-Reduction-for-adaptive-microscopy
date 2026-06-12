"""Command-line interface for BALANCE-NM synthetic experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from .data import ingest_dataset
from .experiment import run_benchmark, run_experiment, run_replay_experiment
from .io import load_config, save_run_artifacts, write_config
from .roi_search import run_roi_search, save_roi_search_artifacts
from .stack_validation import run_stack_validation
from .v3_graph import run_v3_graph_stack_validation
from .v3_validation import (
    DEFAULT_V3_POLICIES,
    audit_v3_stack_from_trace,
    run_v3_stack_validation,
)
from .v4_validation import V4_POLICIES, run_v4_uncertainty_stack_validation
from .v4_bayesian_validation import (
    V4_BAYESIAN_POLICIES,
    run_v4_bayesian_stack_validation,
)
from .v4_bayesian_residual_validation import (
    V4_BAYESIAN_RESIDUAL_POLICIES,
    run_v4_bayesian_residual_stack_validation,
)
from .v4_bayesian_morphology_validation import (
    V4_BAYESIAN_MORPHOLOGY_POLICIES,
    run_v4_bayesian_morphology_stack_validation,
)
from .v4_bayesian_pareto_validation import (
    V4_BAYESIAN_PARETO_POLICIES,
    run_v4_bayesian_pareto_stack_validation,
)
from .v4_bayesian_additive_validation import (
    V4_BAYESIAN_ADDITIVE_POLICIES,
    run_v4_bayesian_additive_stack_validation,
)
from .v5 import V5_VEER_POLICIES, run_v5_veer_stack_validation
from .visualization import plot_benchmark_summary

app = typer.Typer(help="Bayesian adaptive SEM-EDS elemental mapping experiments and replay.")


def _parse_slice_ids(specification: str) -> list[str]:
    """Parse comma-separated slice numbers and inclusive ranges."""

    slice_ids: list[str] = []
    for part in specification.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            first, last = (int(value) for value in part.split(":", maxsplit=1))
            step = 1 if last >= first else -1
            slice_ids.extend(f"{number:03d}" for number in range(first, last + step, step))
        else:
            slice_ids.append(f"{int(part):03d}")
    seen: set[str] = set()
    unique = []
    for slice_id in slice_ids:
        if slice_id not in seen:
            unique.append(slice_id)
            seen.add(slice_id)
    if not unique:
        raise typer.BadParameter("slice specification did not contain any slices")
    return unique


@app.command()
def run(
    config: Path = typer.Option(..., exists=True, readable=True, help="Experiment YAML configuration."),
    policy: str = typer.Option("balance", help="One of: uniform, random, gradient, uncertainty, balance."),
    seed: int = typer.Option(0, min=0, help="Paired specimen and measurement seed."),
    out: Path = typer.Option(..., help="Output directory for run artifacts."),
) -> None:
    """Execute one offline adaptive acquisition experiment."""

    configuration = load_config(config)
    if configuration.dataset.mode == "replay":
        raise typer.BadParameter("Use the replay command for dataset.mode = replay configurations.")
    result = run_experiment(configuration, policy, seed)
    save_run_artifacts(result, out)
    final = result.metrics[-1]
    typer.echo(
        f"Saved {policy} run to {out}. Final interface mean error: "
        f"{final.interface_mean_distance_nm:.2f} nm at {final.scan_time_s:.3f} s."
    )


@app.command()
def ingest(
    config: Path = typer.Option(..., exists=True, readable=True, help="Real-data ingestion YAML configuration."),
    out: Path = typer.Option(..., help="Output standardized Zarr dataset path."),
) -> None:
    """Import a SEM-EDS dataset into the standardized replay contract."""

    configuration = load_config(config)
    dataset, capabilities = ingest_dataset(configuration)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_zarr(out, mode="w")
    write_config(configuration, out.parent / f"{out.stem}_resolved_config.yaml")
    import yaml

    with (out.parent / f"{out.stem}_capability_manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(capabilities.model_dump(mode="json"), handle, sort_keys=False)
    typer.echo(f"Saved standardized dataset with {dataset.sizes['observation']} observations to {out}.")


@app.command()
def replay(
    config: Path = typer.Option(..., exists=True, readable=True, help="Replay experiment YAML configuration."),
    policy: str = typer.Option("balance", help="Acquisition policy."),
    seed: int = typer.Option(0, min=0, help="Replay thinning and policy seed."),
    out: Path = typer.Option(..., help="Output directory for replay artifacts."),
) -> None:
    """Execute capability-gated retrospective acquisition replay."""

    configuration = load_config(config)
    source, capabilities = ingest_dataset(configuration)
    result = run_replay_experiment(configuration, policy, source, capabilities, seed)
    save_run_artifacts(result, out)
    final = result.metrics[-1]
    if final.interface_mean_distance_nm is None:
        outcome = "Interface accuracy not scored because this replay has no enabled labeled interface objective."
    else:
        outcome = f"Final interface mean error: {final.interface_mean_distance_nm:.2f} nm."
    typer.echo(f"Saved replay to {out}. {outcome} Scan time: {final.scan_time_s:.3f} s.")


@app.command("roi-search")
def roi_search(
    config: Path = typer.Option(..., exists=True, readable=True, help="Dense replay YAML template."),
    policy: str = typer.Option("balance", help="Within-ROI adaptive acquisition policy."),
    selection_strategy: str = typer.Option(
        "balance",
        "--selection-strategy",
        help="Outer ROI selection: balance, gradient, uncertainty, uniform, or random.",
    ),
    seed: int = typer.Option(0, min=0, help="Pre-registered random pilot seed."),
    pilot_rois: int = typer.Option(4, min=2, help="Number of seeded random pilot ROIs."),
    bayesian_rois: int = typer.Option(4, min=0, help="Adaptive outer-policy ROI queries after pilots."),
    equivalence_fraction: float = typer.Option(
        0.99, min=0.0, max=1.0, help="Reference-score fraction defining a near-optimal ROI."
    ),
    confirm_neighbors: bool = typer.Option(
        True,
        "--confirm-neighbors/--no-confirm-neighbors",
        help="Query unmeasured eight-cell neighbors of the highest selected-information ROI.",
    ),
    neighbor_anchors: int = typer.Option(
        1, min=1, help="Number of high-information anchor ROIs to confirm by neighbors."
    ),
    out: Path = typer.Option(..., help="Output directory for search artifacts."),
) -> None:
    """Search ROI locations without using dense-map reference information for selection."""

    configuration = load_config(config)
    result = run_roi_search(
        configuration,
        policy,
        seed,
        pilot_rois,
        bayesian_rois,
        selection_strategy=selection_strategy,
        equivalence_fraction=equivalence_fraction,
        confirm_neighbors=confirm_neighbors,
        neighbor_anchors=neighbor_anchors,
        save_run=save_run_artifacts,
        run_output=out,
    )
    save_roi_search_artifacts(result, out)
    summary = result.summary
    typer.echo(
        f"Saved {selection_strategy} ROI search to {out}. Queried {summary['total_queried_rois']} of "
        f"{summary['candidate_count']} pre-registered ROIs; nearest queried region is "
        f"{summary['nearest_queried_distance_to_reference_nm']:.1f} nm from the "
        f"single maximum, with best reference regret "
        f"{100 * summary['best_queried_regret_fraction']:.2f}% "
        f"({summary['selection_status']})."
    )


@app.command()
def benchmark(
    config: Path = typer.Option(..., exists=True, readable=True, help="Benchmark YAML configuration."),
    out: Path = typer.Option(..., help="Output directory for paired benchmark artifacts."),
) -> None:
    """Run paired randomized specimens for every configured policy."""

    configuration = load_config(config)
    out.mkdir(parents=True, exist_ok=True)
    write_config(configuration, out / "resolved_config.yaml")
    metrics, summary = run_benchmark(configuration, out, save_run_artifacts)
    metrics.to_csv(out / "metrics.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    plot_benchmark_summary(summary, out / "benchmark_summary.png")
    typer.echo(f"Saved benchmark with {configuration.benchmark.seeds} paired seeds to {out}.")


@app.command()
def report(
    input: Path = typer.Option(..., exists=True, readable=True, help="Benchmark artifact directory."),
    out: Path = typer.Option(..., help="Report output directory."),
) -> None:
    """Regenerate compact tables and figures from benchmark artifacts."""

    summary_path = input / "summary.csv"
    if not summary_path.exists():
        raise typer.BadParameter(f"benchmark summary not found: {summary_path}")
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_path)
    summary.to_csv(out / "summary.csv", index=False)
    plot_benchmark_summary(summary, out / "benchmark_summary.png")
    markdown = summary.round(3).to_markdown(index=False)
    (out / "summary.md").write_text(
        "# BALANCE-NM Benchmark Summary\n\n" + markdown + "\n",
        encoding="utf-8",
    )
    typer.echo(f"Saved report artifacts to {out}.")


@app.command("validate-stack")
def validate_stack(
    config: Path = typer.Option(..., exists=True, readable=True, help="Dense replay YAML template."),
    manifest: Path | None = typer.Option(
        None, exists=True, readable=True, help="Downloaded stack manifest mapping slices and channels to files."
    ),
    slices: str = typer.Option("001:265", help="Inclusive slice range as start:end."),
    seed: int = typer.Option(0, min=0, help="Paired initial ROI-selection seed."),
    out: Path = typer.Option(..., help="Output directory for stack validation artifacts."),
) -> None:
    """Validate direct-raster ROI selection over a dense multichannel slice stack."""

    slice_ids = _parse_slice_ids(slices)
    _, aggregate = run_stack_validation(
        config, slice_ids, out, manifest_path=manifest, seed=seed
    )
    best = aggregate.iloc[0]
    typer.echo(
        f"Validated {len(slice_ids)} slices. Best mean-regret arm: {best['arm']} with "
        f"{100 * best['hit_rate']:.1f}% reference-equivalent recommendations."
    )


@app.command("validate-v3-stack")
def validate_v3_stack(
    config: Path = typer.Option(..., exists=True, readable=True, help="V3 dense replay YAML template."),
    manifest: Path | None = typer.Option(
        None, exists=True, readable=True, help="Downloaded stack manifest mapping slices and channels to files."
    ),
    slices: str = typer.Option("001:265", help="Inclusive slice range as start:end."),
    policies: str = typer.Option(
        ",".join(DEFAULT_V3_POLICIES),
        help=(
            "Comma-separated v3 policies: uncertainty,balance_v3_residual_attention,"
            "balance_v3_attention,balance_v3,balance_v3_scheduled,balance_v3_gated,"
            "gradient,uniform,random."
        ),
    ),
    seed: int = typer.Option(0, min=0, help="Paired initial ROI-selection seed."),
    out: Path = typer.Option(..., help="Output directory for v3 morphology-reconstruction artifacts."),
) -> None:
    """Validate v3 corrosion-morphology reconstruction over a dense slice stack."""

    slice_ids = _parse_slice_ids(slices)
    policy_names = [policy.strip() for policy in policies.split(",") if policy.strip()]
    _, summary = run_v3_stack_validation(
        config,
        slice_ids,
        out,
        manifest_path=manifest,
        policies=policy_names,
        seed=seed,
    )
    best = summary.iloc[0]
    typer.echo(
        f"Validated v3 morphology reconstruction on {len(slice_ids)} slices. Best mean front-distance policy: "
        f"{best['policy']} at {best['mean_final_front_mean_symmetric_distance_nm']:.2f} nm."
    )


@app.command("audit-v3-stack")
def audit_v3_stack(
    config: Path = typer.Option(..., exists=True, readable=True, help="V3 dense replay YAML template."),
    manifest: Path | None = typer.Option(
        None, exists=True, readable=True, help="Downloaded stack manifest mapping slices and channels to files."
    ),
    slices: str = typer.Option("001:265", help="Inclusive slice range as start:end."),
    policies: str = typer.Option(
        ",".join(DEFAULT_V3_POLICIES),
        help="Comma-separated v3 policies to audit from stored ROI traces.",
    ),
    seed: int = typer.Option(0, min=0, help="Paired seed used to replay any missing trace rows."),
    input_dir: Path = typer.Option(
        ..., "--input", exists=True, file_okay=False, help="Existing v3 validation artifact directory."
    ),
    out: Path = typer.Option(..., help="Output directory for audited stratified metrics."),
) -> None:
    """Audit front availability and conditional localization from stored v3 ROI traces."""

    slice_ids = _parse_slice_ids(slices)
    policy_names = [policy.strip() for policy in policies.split(",") if policy.strip()]
    _, summary = audit_v3_stack_from_trace(
        config,
        slice_ids,
        input_dir,
        out,
        manifest_path=manifest,
        policies=policy_names,
        seed=seed,
    )
    best = summary.iloc[0]
    typer.echo(
        f"Audited {len(slice_ids)} slices. Best front-detection policy: {best['policy']} at "
        f"{100 * best['front_detection_accuracy']:.1f}% accuracy."
    )


@app.command("validate-v3-graph-stack")
def validate_v3_graph_stack(
    config: Path = typer.Option(..., exists=True, readable=True, help="V3 graph replay YAML template."),
    manifest: Path | None = typer.Option(
        None, exists=True, readable=True, help="Downloaded stack manifest mapping slices and channels to files."
    ),
    slices: str = typer.Option("001:265", help="Inclusive slice range as start:end."),
    seed: int = typer.Option(0, min=0, help="Paired initial ROI-selection seed."),
    out: Path = typer.Option(..., help="Output directory for v3 graph-refinement artifacts."),
) -> None:
    """Validate adaptive spatial graph refinement over a dense slice stack."""

    slice_ids = _parse_slice_ids(slices)
    _, summary = run_v3_graph_stack_validation(
        config,
        slice_ids,
        out,
        manifest_path=manifest,
        seed=seed,
    )
    best = summary.iloc[0]
    typer.echo(
        f"Validated v3 graph refinement on {len(slice_ids)} slices. Best conditional front-localization arm: "
        f"{best['arm_id']} at {best['mean_front_localization_distance_nm']:.2f} nm."
    )


@app.command("validate-v4-uncertainty-stack")
def validate_v4_uncertainty_stack(
    config: Path = typer.Option(..., exists=True, readable=True, help="V4 uncertainty replay YAML template."),
    manifest: Path = typer.Option(
        ..., exists=True, readable=True, help="Downloaded stack manifest mapping slices and channels to files."
    ),
    fold: str = typer.Option("all", help="Outer blocked fold number (1-5) or all."),
    slices: str | None = typer.Option(
        None, help="Optional inclusive slice ranges for a smoke subset, for example 001:010."
    ),
    policies: str = typer.Option(
        ",".join(V4_POLICIES),
        help="Comma-separated v4 uncertainty policies.",
    ),
    seed: int = typer.Option(0, min=0, help="Paired pilot-ROI seed."),
    out: Path = typer.Option(..., help="Output directory for v4 uncertainty artifacts."),
) -> None:
    """Validate uncertainty-first adaptive reconstruction over blocked stack folds."""

    slice_ids = _parse_slice_ids(slices) if slices else None
    policy_names = [policy.strip() for policy in policies.split(",") if policy.strip()]
    _, summary = run_v4_uncertainty_stack_validation(
        config,
        out,
        manifest,
        fold_specification=fold,
        slice_ids=slice_ids,
        policies=policy_names,
        seed=seed,
    )
    best = summary.iloc[0]
    typer.echo(
        f"Validated v4 uncertainty reconstruction. Best endpoint composite policy: "
        f"{best['policy']} at {best['mean_morphology_composite_error']:.5f}."
    )


@app.command("validate-v4-bayesian-stack")
def validate_v4_bayesian_stack(
    config: Path = typer.Option(..., exists=True, readable=True, help="V4.1 Bayesian replay YAML template."),
    manifest: Path = typer.Option(
        ..., exists=True, readable=True, help="Downloaded stack manifest mapping slices and channels to files."
    ),
    fold: str = typer.Option("all", help="Outer blocked fold number (1-5) or all."),
    slices: str | None = typer.Option(
        None, help="Optional inclusive slice ranges for staged smoke validation."
    ),
    policies: str = typer.Option(
        ",".join(V4_BAYESIAN_POLICIES),
        help="Comma-separated Bayesian v4.1 policies.",
    ),
    seed: int = typer.Option(0, min=0, help="Paired pilot-ROI seed."),
    out: Path = typer.Option(..., help="Output directory for Bayesian v4.1 artifacts."),
) -> None:
    """Validate Bayesian guarded variance-reduction raster selection."""

    slice_ids = _parse_slice_ids(slices) if slices else None
    policy_names = [policy.strip() for policy in policies.split(",") if policy.strip()]
    _, summary = run_v4_bayesian_stack_validation(
        config,
        out,
        manifest,
        fold_specification=fold,
        slice_ids=slice_ids,
        policies=policy_names,
        seed=seed,
    )
    best = summary.iloc[0]
    typer.echo(
        f"Validated Bayesian v4.1 reconstruction. Best endpoint composite policy: "
        f"{best['policy']} at {best['mean_morphology_composite_error']:.5f}."
    )


@app.command("validate-v4-bayesian-residual-stack")
def validate_v4_bayesian_residual_stack(
    config: Path = typer.Option(..., exists=True, readable=True, help="V4.2 Bayesian residual replay YAML template."),
    manifest: Path = typer.Option(
        ..., exists=True, readable=True, help="Downloaded stack manifest mapping slices and channels to files."
    ),
    fold: str = typer.Option("all", help="Outer blocked fold number (1-5) or all."),
    slices: str | None = typer.Option(
        None, help="Optional inclusive slice ranges for staged smoke validation."
    ),
    policies: str = typer.Option(
        ",".join(V4_BAYESIAN_RESIDUAL_POLICIES),
        help="Comma-separated Bayesian residual v4.2 policies.",
    ),
    seed: int = typer.Option(0, min=0, help="Paired pilot-ROI seed."),
    out: Path = typer.Option(..., help="Output directory for Bayesian residual v4.2 artifacts."),
) -> None:
    """Validate Bayesian residual and subtile raster selection."""

    slice_ids = _parse_slice_ids(slices) if slices else None
    policy_names = [policy.strip() for policy in policies.split(",") if policy.strip()]
    _, summary = run_v4_bayesian_residual_stack_validation(
        config,
        out,
        manifest,
        fold_specification=fold,
        slice_ids=slice_ids,
        policies=policy_names,
        seed=seed,
    )
    best = summary.iloc[0]
    typer.echo(
        f"Validated Bayesian residual v4.2 reconstruction. Best endpoint composite policy: "
        f"{best['policy']} at {best['mean_morphology_composite_error']:.5f}."
    )


@app.command("validate-v4-bayesian-morphology-stack")
def validate_v4_bayesian_morphology_stack(
    config: Path = typer.Option(..., exists=True, readable=True, help="V4.3 Bayesian morphology replay YAML template."),
    manifest: Path = typer.Option(
        ..., exists=True, readable=True, help="Downloaded stack manifest mapping slices and channels to files."
    ),
    fold: str = typer.Option("all", help="Outer blocked fold number (1-5) or all."),
    slices: str | None = typer.Option(
        None, help="Optional inclusive slice ranges for staged smoke validation."
    ),
    policies: str = typer.Option(
        ",".join(V4_BAYESIAN_MORPHOLOGY_POLICIES),
        help="Comma-separated evidence-gated Bayesian morphology v4.3 policies.",
    ),
    seed: int = typer.Option(0, min=0, help="Paired pilot-ROI seed."),
    out: Path = typer.Option(..., help="Output directory for Bayesian morphology v4.3 artifacts."),
) -> None:
    """Validate evidence-gated and posterior-fantasy Bayesian raster selection."""

    slice_ids = _parse_slice_ids(slices) if slices else None
    policy_names = [policy.strip() for policy in policies.split(",") if policy.strip()]
    _, summary = run_v4_bayesian_morphology_stack_validation(
        config,
        out,
        manifest,
        fold_specification=fold,
        slice_ids=slice_ids,
        policies=policy_names,
        seed=seed,
    )
    best = summary.iloc[0]
    typer.echo(
        f"Validated Bayesian morphology v4.3 reconstruction. Best endpoint composite policy: "
        f"{best['policy']} at {best['mean_morphology_composite_error']:.5f}."
    )


@app.command("validate-v4-bayesian-pareto-stack")
def validate_v4_bayesian_pareto_stack(
    config: Path = typer.Option(..., exists=True, readable=True, help="V4.4 Pareto Bayesian replay YAML template."),
    manifest: Path = typer.Option(
        ..., exists=True, readable=True, help="Downloaded stack manifest mapping slices and channels to files."
    ),
    fold: str = typer.Option("all", help="Outer blocked fold number (1-5) or all."),
    slices: str | None = typer.Option(
        None, help="Optional inclusive slice ranges for staged smoke validation."
    ),
    policies: str = typer.Option(
        ",".join(V4_BAYESIAN_PARETO_POLICIES),
        help="Comma-separated Pareto-gated Bayesian v4.4 policies.",
    ),
    seed: int = typer.Option(0, min=0, help="Paired pilot-ROI seed."),
    out: Path = typer.Option(..., help="Output directory for Pareto Bayesian v4.4 artifacts."),
) -> None:
    """Validate Pareto-gated Bayesian EIVR raster selection."""

    slice_ids = _parse_slice_ids(slices) if slices else None
    policy_names = [policy.strip() for policy in policies.split(",") if policy.strip()]
    _, summary = run_v4_bayesian_pareto_stack_validation(
        config,
        out,
        manifest,
        fold_specification=fold,
        slice_ids=slice_ids,
        policies=policy_names,
        seed=seed,
    )
    best = summary.iloc[0]
    typer.echo(
        f"Validated Pareto Bayesian v4.4 reconstruction. Best endpoint composite policy: "
        f"{best['policy']} at {best['mean_morphology_composite_error']:.5f}."
    )


@app.command("validate-v4-bayesian-additive-stack")
def validate_v4_bayesian_additive_stack(
    config: Path = typer.Option(..., exists=True, readable=True, help="V4.5 additive Bayesian replay YAML template."),
    manifest: Path = typer.Option(
        ..., exists=True, readable=True, help="Downloaded stack manifest mapping slices and channels to files."
    ),
    fold: str = typer.Option("all", help="Outer blocked fold number (1-5) or all."),
    slices: str | None = typer.Option(
        None, help="Optional inclusive slice ranges for staged smoke validation."
    ),
    policies: str = typer.Option(
        ",".join(V4_BAYESIAN_ADDITIVE_POLICIES),
        help="Comma-separated additive Pareto Bayesian v4.5 policies.",
    ),
    seed: int = typer.Option(0, min=0, help="Paired pilot-ROI seed."),
    out: Path = typer.Option(..., help="Output directory for additive Pareto Bayesian v4.5 artifacts."),
) -> None:
    """Validate additive Pareto Bayesian EIVR raster selection."""

    slice_ids = _parse_slice_ids(slices) if slices else None
    policy_names = [policy.strip() for policy in policies.split(",") if policy.strip()]
    _, summary = run_v4_bayesian_additive_stack_validation(
        config,
        out,
        manifest,
        fold_specification=fold,
        slice_ids=slice_ids,
        policies=policy_names,
        seed=seed,
    )
    best = summary.iloc[0]
    typer.echo(
        f"Validated additive Pareto Bayesian v4.5 reconstruction. Best endpoint composite policy: "
        f"{best['policy']} at {best['mean_morphology_composite_error']:.5f}."
    )


@app.command("validate-v5-veer-stack")
def validate_v5_veer_stack(
    config: Path = typer.Option(..., exists=True, readable=True, help="V5 VEER replay YAML template."),
    manifest: Path = typer.Option(
        ..., exists=True, readable=True, help="Downloaded stack manifest mapping slices and channels to files."
    ),
    fold: str = typer.Option("all", help="Outer blocked fold number (1-5) or all."),
    slices: str | None = typer.Option(
        None, help="Optional inclusive slice ranges for staged smoke validation."
    ),
    policies: str = typer.Option(
        ",".join(V5_VEER_POLICIES),
        help="Comma-separated v5 variogram expected-error-reduction policies.",
    ),
    seed: int = typer.Option(0, min=0, help="Pilot-ROI seed mixed per slice."),
    workers: int = typer.Option(
        1, min=1, help="Parallel replay worker processes; results are identical to serial."
    ),
    out: Path = typer.Option(..., help="Output directory for v5 VEER artifacts."),
) -> None:
    """Validate variogram expected-error-reduction raster selection."""

    slice_ids = _parse_slice_ids(slices) if slices else None
    policy_names = [policy.strip() for policy in policies.split(",") if policy.strip()]
    _, summary = run_v5_veer_stack_validation(
        config,
        out,
        manifest,
        fold_specification=fold,
        slice_ids=slice_ids,
        policies=policy_names,
        seed=seed,
        workers=workers,
    )
    best = summary.iloc[0]
    typer.echo(
        f"Validated v5 VEER reconstruction. Best endpoint composite policy: "
        f"{best['policy']} at {best['mean_morphology_composite_error']:.5f}."
    )


if __name__ == "__main__":
    app()
