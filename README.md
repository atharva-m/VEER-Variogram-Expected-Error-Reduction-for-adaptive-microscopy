# BALANCE-NM

BALANCE-NM is now focused on one research question:

> Can Bayesian expected-information lookahead improve uncertainty-guided adaptive raster selection for unannotated multichannel Alloy 617 corrosion-morphology maps?

The current project is a retrospective, offline replay framework. Dense
elemental maps are hidden from acquisition policies and used only for reveal
and evaluation. The target is not calibrated composition, live microscope
control, or expert-labeled corrosion segmentation. The target is efficient
reconstruction of a frozen unsupervised alteration-front proxy and
penetration-depth proxy under a fixed raster budget.

## Current Scope

The active comparison set is intentionally small:

```text
baselines
uncertainty_lookahead
bayesian_pareto_eivr_4x4_mean_tau090          # v4.4 ratio utility
bayesian_pareto_additive_eivr_*_alpha{1,2,5,10} # v4.5 additive utility
```

Earlier v1/v2/v3, attention, graph, neural, fantasy-morphology, residual, and
ROI-max experiments are historical ablations and are no longer part of the
reported primary goal.

Some helper modules still have older names because v4.4/v4.5 reuse shared ROI
catalog, morphology proxy, and metric code. Those helpers are implementation
plumbing, not active project claims.

## Data

The retained dataset is the downloaded Alloy 617 NRDS EDS stack:

```text
data/alloy617_nrds/
```

The maps are treated as multichannel intensity proxies. `CPS` is excluded by
default. The frozen alteration-front proxy is unsupervised and should be
described as a proxy unless validated by a domain expert.

## Objective And Metric

Each policy receives:

```text
4 paired random pilot ROIs
17 total raster ROIs
same 72-ROI catalog
same nearest-observation primary evaluator
same scan-time and dose accounting
```

The primary endpoint is:

```text
morphology_composite_error =
    0.50 * front_mean_symmetric_distance_nm / slice_width_nm
  + 0.50 * penetration_d95_absolute_error_nm / slice_width_nm
```

Secondary checks include normalized reconstruction RMSE, selected area
fraction, scan time, dose proxy, maximum slice regression, fold behavior, and
paired confidence intervals.

## Method Summary

### Uncertainty Lookahead

`uncertainty_lookahead` is the strongest deterministic baseline. It computes
the geometric coverage gain from adding each candidate ROI mask without
revealing hidden values:

```text
coverage_gain(r) =
    integrated current distance uncertainty
  - integrated distance uncertainty after adding ROI r
```

It selects the ROI with the largest coverage gain per cost.

### V4.4 Pareto-Gated Bayesian EIVR

`bayesian_pareto_eivr_4x4_mean_tau090` keeps deterministic geometry as a
safety backbone:

```text
G(r) = geometry_gain(r) / geometry_gain(g*)
S_tau = {r : G(r) >= 0.90}
```

Only candidates in `S_tau` receive Bayesian scoring. Revealed ROIs are split
into `4 x 4` subtiles. A revealed-only robust scaling and PCA embedding feeds
model-averaged anisotropic Matern-3/2 Gaussian processes. Candidate utility in
v4.4 used a ratio-style evidence multiplier:

```text
utility(r) = G(r) * [1 + EIVR_LCB(r) / B(g*)]
```

This improved the mean on 30 slices but produced a heavy regression tail on
some flat or saturated-front slices.

### V4.5 Additive Pareto Bayesian EIVR

V4.5 keeps the v4.4 representation and geometry-first shortlist, but replaces
the volatile ratio with an additive exchange-rate rule:

```text
eligible(r) =
    G(r) >= 0.90
    EIVR_LCB(r) > 0
    kernel_support(r) >= 0.90

utility(r) = G(r) + alpha * EIVR_LCB(r)
```

This avoids dividing by a near-zero Bayesian score for the geometry winner.
The intent is to preserve the mean improvement while trimming severe
regressions on flat-front slices. Current tested exchange rates are:

```text
alpha = 1, 2, 5, 10
```

`alpha5` is the preferred current candidate because it ties `alpha10` on the
10-slice smoke result while being less aggressive.

## Current Progress

### V4.4 30-Slice Smoke

Artifact folder:

```text
results/alloy617_v4_bayesian_pareto_smoke_030/
```

| Policy | Slices | Mean Composite Error | Delta vs Uncertainty | Mean RMSE | Max Slice Regression | Status |
|---|---:|---:|---:|---:|---:|---|
| `bayesian_pareto_eivr_4x4_mean_tau090` | 30 | 0.08853 | -0.01603 | 0.26119 | +0.02986 | mean improved, gate failed |
| `bayesian_pareto_eivr_4x4_mean_tau085` | 30 | 0.08867 | -0.01589 | 0.26173 | +0.02642 | mean improved, gate failed |
| `uncertainty_lookahead` | 30 | 0.10456 | baseline | 0.28761 | 0 | baseline |

Interpretation: v4.4 beat uncertainty on mean composite error, but it did not
pass the 30-slice gate because the confidence interval crossed zero and the
maximum slice regression exceeded the `+0.02` guardrail.

### V4.5 10-Slice Smoke

Artifact folder:

```text
results/alloy617_v4_bayesian_additive_smoke_010/
```

| Policy | Slices | Mean Composite Error | Delta vs Uncertainty | Win Rate | RMSE Delta | Max Slice Regression | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| `bayesian_pareto_additive_eivr_4x4_mean_tau090_alpha5` | 10 | 0.13082 | -0.00539 (-3.96%) | 0.60 | +0.58% | +0.01582 | passes 10-slice gate |
| `bayesian_pareto_additive_eivr_4x4_mean_tau090_alpha10` | 10 | 0.13082 | -0.00539 (-3.96%) | 0.60 | +0.58% | +0.01582 | passes 10-slice gate |
| `uncertainty_lookahead` | 10 | 0.13621 | baseline | - | 0 | 0 | baseline |
| `bayesian_pareto_additive_eivr_4x4_mean_tau090_alpha1` | 10 | 0.13621 | 0.00000 | 0.00 | 0 | 0 | behaves like geometry |
| `bayesian_pareto_additive_eivr_4x4_mean_tau090_alpha2` | 10 | 0.13707 | +0.00086 | 0.50 | +0.39% | +0.01582 | not advanced |
| `bayesian_pareto_eivr_4x4_mean_tau090` | 10 | 0.13810 | +0.00189 | 0.30 | +0.31% | +0.02193 | v4.4 comparator |

Target slice behavior:

```text
Slice 021:
  v4.4 ratio regression: +0.02193
  v4.5 alpha5/10:        -0.00114

Slice 075:
  v4.4 ratio regression: +0.02170
  v4.5 alpha5/10:        +0.01582

Slice 085:
  v4.5 alpha5/10:        -0.00609
```

Interpretation: v4.5 successfully trims the worst v4.4 regression tail in the
10-slice smoke and keeps RMSE regression under the `2%` limit. It is promising
but not yet promoted because it has not completed the 30-slice gate or full
stack validation.

## Validation Gates

The next decision point is the frozen 30-slice cohort:

```text
001,011,021,032,042,053,
054,064,075,085,096,106,
107,117,128,138,149,159,
160,170,181,191,202,212,
213,223,234,244,255,265
```

V4.5 should advance only if it satisfies:

```text
mean composite-error delta < 0
median composite-error delta <= 0
leave-one-slice-out worst mean delta <= 0
at least 3 of 5 fold means <= 0
RMSE regression <= 2%
equal scan cost
maximum slice composite regression <= 0.02
```

Full-stack promotion requires a paired mean composite-error improvement with a
95% confidence interval excluding zero, equal scan cost, and no more than 2%
RMSE regression.

## Commands

Install and test:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest tests\test_v4_uncertainty.py tests\test_v4_bayesian_pareto.py -q
```

Run uncertainty baseline:

```powershell
.\.venv\Scripts\python.exe -m balance_nm validate-v4-uncertainty-stack `
  --config configs\alloy617_v4_uncertainty.yaml `
  --manifest data\alloy617_nrds\full_stack_download_manifest.csv `
  --fold all `
  --out results\alloy617_v4_uncertainty
```

Run v4.4 Pareto Bayesian EIVR:

```powershell
.\.venv\Scripts\python.exe -m balance_nm validate-v4-bayesian-pareto-stack `
  --config configs\alloy617_v4_bayesian_pareto.yaml `
  --manifest data\alloy617_nrds\full_stack_download_manifest.csv `
  --fold all `
  --out results\alloy617_v4_bayesian_pareto
```

Run v4.5 additive Pareto Bayesian EIVR on the 10-slice smoke:

```powershell
.\.venv\Scripts\python.exe -m balance_nm validate-v4-bayesian-additive-stack `
  --config configs\alloy617_v4_bayesian_additive.yaml `
  --manifest data\alloy617_nrds\full_stack_download_manifest.csv `
  --fold all `
  --slices 001,011,021,032,042,053,054,064,075,085 `
  --policies uncertainty_lookahead,bayesian_pareto_eivr_4x4_mean_tau090,bayesian_pareto_additive_eivr_4x4_mean_tau090_alpha1,bayesian_pareto_additive_eivr_4x4_mean_tau090_alpha2,bayesian_pareto_additive_eivr_4x4_mean_tau090_alpha5,bayesian_pareto_additive_eivr_4x4_mean_tau090_alpha10 `
  --out results\alloy617_v4_bayesian_additive_smoke_010
```

Run v4.5 on the frozen 30-slice gate:

```powershell
.\.venv\Scripts\python.exe -m balance_nm validate-v4-bayesian-additive-stack `
  --config configs\alloy617_v4_bayesian_additive.yaml `
  --manifest data\alloy617_nrds\full_stack_download_manifest.csv `
  --fold all `
  --slices 001,011,021,032,042,053,054,064,075,085,096,106,107,117,128,138,149,159,160,170,181,191,202,212,213,223,234,244,255,265 `
  --policies uncertainty_lookahead,bayesian_pareto_eivr_4x4_mean_tau090,bayesian_pareto_additive_eivr_4x4_mean_tau090_alpha5 `
  --out results\alloy617_v4_bayesian_additive_smoke_030
```

## Repository State

The cleaned workspace keeps only current configs, current results, and current
tests:

```text
configs/
  alloy617_v4_uncertainty.yaml
  alloy617_v4_bayesian_pareto.yaml
  alloy617_v4_bayesian_additive.yaml

results/
  alloy617_v4_uncertainty_smoke/
  alloy617_v4_bayesian_pareto_smoke_001/
  alloy617_v4_bayesian_pareto_smoke_030/
  alloy617_v4_bayesian_additive_smoke_001/
  alloy617_v4_bayesian_additive_smoke_010/

tests/
  test_v4_uncertainty.py
  test_v4_bayesian_pareto.py
```

## Scientific Caveats

- The alteration front is a frozen unsupervised proxy, not expert truth.
- Reported improvements are retrospective replay results on downloaded dense
  maps, not live microscope performance.
- The current primary comparison uses the same nearest-observation evaluator
  for every policy to isolate acquisition-policy effects.
- GP-kriging diagnostics are useful but are not promotion criteria.
- Current v4.5 evidence is promising at 10 slices, but the method is not
  promoted until it passes the 30-slice and full-stack gates.
