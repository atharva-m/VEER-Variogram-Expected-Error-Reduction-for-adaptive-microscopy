# BALANCE-NM

BALANCE-NM is a retrospective replay framework for **uncertainty-guided
adaptive raster selection** on unannotated multichannel Alloy 617
corrosion-morphology maps. It answers one question:

> Can a learned-variogram expected-error-reduction acquisition policy improve
> reconstruction of an alteration-front proxy under a fixed raster budget,
> relative to a strong deterministic coverage baseline?

Dense elemental maps are hidden from the acquisition policy and used only for
reveal and evaluation. The method (VEER — Variogram Expected Error Reduction)
selects which raster tiles to scan so that a frozen, unsupervised
alteration-front and penetration-depth proxy is reconstructed as accurately
as possible from a fraction of the field.

## Result

On a frozen 235-slice held-out cohort (never used for design or selection),
the confirmed policy `gated_veer_4x4_mean_kappa5` improves mean
morphology-composite error by **~14–17%** over the `uncertainty_lookahead`
baseline at **equal scan cost** (95% CI excluding zero on both the
final-iteration and the pre-registered trailing-median endpoint, with RMSE
also improved). The improvement is a mean/risk redistribution, not per-slice
dominance — see [docs/DESIGN.md](docs/DESIGN.md) for the full method,
endpoints, gates, and the honest per-slice risk profile.

## Method

The shared evaluator is nearest-observation reconstruction, so expected
squared error at a pixel is approximately the semivariogram of the signal at
the distance to its nearest sample. VEER therefore selects the tile that
maximizes front-weighted expected-error reduction per cost under a
revealed-only, model-averaged variogram:

```text
utility(r) = sum_p w_p * [ gamma_hat(d_p) - gamma_hat(d_p_after_r) ] / cost(r)
```

The `uncertainty_lookahead` baseline is the special case `gamma_hat(d) = d`
with uniform weights. Successive refinements (a nested unbounded variogram
fitted by weighted least squares, and movement-gated front weighting) are
documented in [docs/DESIGN.md](docs/DESIGN.md).

## Layout

```text
src/balance_nm/
  domain.py        # pydantic config models (RunConfig, AcquisitionConfig, VariogramConfig)
  data.py          # dataset ingestion (binary element maps, zarr)
  io.py            # config load/save
  morphology.py    # frozen unsupervised front/penetration proxy + nearest-obs evaluator
  features.py      # subtile feature extraction + anisotropic Matern-3/2 kernel
  replay.py        # ROI catalog, raster cost, folds, scoring, checkpoints
  variogram.py     # calibrated model-averaged + nested WLS variogram estimation
  selection.py     # VEER candidate scoring and front weighting
  validation.py    # resumable, parallel stack validation and endpoints
  cli.py           # `balance-nm validate-veer-stack`
configs/alloy617_veer.yaml
tests/test_veer.py
docs/DESIGN.md
```

## Install and test

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest tests -q
```

(On Windows, pytest's default temp directory may have restrictive ACLs; if
so, add `--basetemp=.pytest_tmp`.)

## Run

```powershell
.\.venv\Scripts\python.exe -m balance_nm validate-veer-stack `
  --config configs\alloy617_veer.yaml `
  --manifest data\alloy617_nrds\full_stack_download_manifest.csv `
  --fold all `
  --policies uncertainty_lookahead,gated_veer_4x4_mean_kappa5 `
  --workers 12 `
  --out results\veer
```

Add `--slices 001,011,021,...` for a staged smoke. `--workers N` runs
slice/policy replays across processes with results identical to serial.
Runs are resumable: re-running the same command continues from the last
completed checkpoint.

## Policies

```text
uncertainty_lookahead              # deterministic coverage baseline (gamma = d)
variogram_eer_4x4_mean_kappa{0,2,5}      # learned variogram, front-band weighting
nested_veer_4x4_mean_kappa{0,2,5,10}     # nested WLS variogram (no-sill model)
nested_band_veer_4x4_mean_kappa{2,5}     # nested variogram + Gaussian-band weights
gated_veer_4x4_mean_kappa{5,10}          # + movement-gated front weighting (confirmed: kappa5)
```

## Data

The retained dataset is the downloaded INL NRDS Alloy 617 EDS stack
(`data/alloy617_nrds/`, CC BY 4.0; see that directory's README and
`scripts/download_nrds_alloy617_eds_stack.py`). The bulk binary maps are
not version-controlled; the manifests and provenance files are.

## Scientific caveats

- The alteration front is a frozen unsupervised proxy, not expert truth.
- Results are retrospective replay on dense maps, not live microscope control.
- All comparisons use the same nearest-observation evaluator to isolate the
  acquisition-policy effect.
- Evidence is from one specimen (265 spatially correlated serial sections);
  generalization requires additional specimens.
