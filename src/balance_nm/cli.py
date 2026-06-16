"""Command-line interface for BALANCE-NM variogram expected-error-reduction replay."""

from __future__ import annotations

from pathlib import Path

import typer

from .validation import VEER_POLICIES, run_veer_stack_validation

app = typer.Typer(help="Variogram expected-error-reduction (VEER) adaptive raster replay.")


@app.callback()
def main() -> None:
    """VEER adaptive raster selection for SEM-EDS corrosion-morphology replay."""


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


@app.command("validate-veer-stack")
def validate_veer_stack(
    config: Path = typer.Option(..., exists=True, readable=True, help="VEER replay YAML template."),
    manifest: Path = typer.Option(
        ..., exists=True, readable=True, help="Downloaded stack manifest mapping slices and channels to files."
    ),
    fold: str = typer.Option("all", help="Outer blocked fold number (1-5) or all."),
    slices: str | None = typer.Option(
        None, help="Optional inclusive slice ranges for staged smoke validation."
    ),
    policies: str = typer.Option(
        ",".join(VEER_POLICIES),
        help="Comma-separated VEER policies.",
    ),
    seed: int = typer.Option(0, min=0, help="Pilot-ROI seed mixed per slice."),
    workers: int = typer.Option(
        1, min=1, help="Parallel replay worker processes; results are identical to serial."
    ),
    out: Path = typer.Option(..., help="Output directory for VEER artifacts."),
) -> None:
    """Validate variogram expected-error-reduction raster selection."""

    slice_ids = _parse_slice_ids(slices) if slices else None
    policy_names = [policy.strip() for policy in policies.split(",") if policy.strip()]
    _, summary = run_veer_stack_validation(
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
        f"Validated VEER reconstruction. Best endpoint composite policy: "
        f"{best['policy']} at {best['mean_morphology_composite_error']:.5f}."
    )


if __name__ == "__main__":
    app()
