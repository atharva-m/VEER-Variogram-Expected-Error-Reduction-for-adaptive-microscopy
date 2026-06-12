# V5: Variogram Expected Error Reduction (VEER)

V5 replaces the v4.5 two-term utility (`G(r) + alpha * EIVR_LCB(r)`) with a
single acquisition objective expressed in the units that the evaluator
actually measures: expected squared nearest-observation reconstruction error.
The exchange rate `alpha`, the LCB-positivity gate, the kernel-support gate,
and the geometry shortlist all disappear because there is no longer a second
score to reconcile with the first.

## 1. Why a variogram unifies geometry and evidence

Every primary comparison arm is evaluated with the same nearest-observation
reconstruction. For that evaluator, the expected squared error at pixel `p`
is, to first order, the semivariogram of the underlying signal evaluated at
the distance from `p` to its nearest observed pixel:

```text
E[ (f(p) - f(nearest(p)))^2 ] ~= 2 * gamma(d_p)
```

The v4 `uncertainty_lookahead` baseline scores a candidate ROI by the
integrated reduction in distance-to-nearest-observation. That is exactly the
acquisition rule above under the *assumption* `gamma(d) = d`: error grows
linearly with distance, identically everywhere, for every slice.

The v4.4/v4.5 Bayesian machinery exists because that assumption ignores what
the revealed data say about spatial structure. But v4.5 injects the evidence
as a separate additive bonus, which forces the tuned exchange rate `alpha`
and the eligibility gates. V5 instead lets the evidence *bend the
distance-error curve*: the revealed subtiles determine a model-averaged
Matern-3/2 semivariogram `gamma_hat`, and the candidate utility becomes

```text
utility(r) = sum_p w_p * [ gamma_hat(d_p) - gamma_hat(d_p_after_r) ] / cost(r)
```

Short inferred length scales make `gamma_hat` steep (signal decorrelates
quickly, so dense coverage pays); long length scales make it flat near zero
(signal is smooth, so reach matters more than density). The Bayesian evidence
and the geometric coverage argument are now the same number.

Setting `gamma_hat(d) = d` and `w_p = 1` recovers `uncertainty_lookahead`
exactly, which keeps the ablation story clean.

## 2. Estimation: calibrated, model-averaged Matern-3/2 variogram

Inputs are the same revealed-only 4x4 mean subtile observations used by
v4.4/v4.5 (shared plumbing: `pareto_subtile_observations_from_revealed_roi`),
so the representation is identical across versions and any improvement is
attributable to the objective, not the features.

Per refit (after every reveal):

1. Robust-scale the `(n, channel, 1)` feature tensor by revealed-only
   median/IQR per channel (shared `robust_scale_feature_tensor`).
2. PCA to `m = min(latent_components, channels, n - 1)` components with
   eigenvalues `lambda_j`.
3. **Standardize each latent component to unit variance** before any
   likelihood computation. This fixes the v4.4/v4.5 miscalibration where raw
   PCA scores (variance `lambda_j`, spanning orders of magnitude) were fit
   under a fixed unit-amplitude prior, distorting marginal likelihoods and
   posterior variances.
4. Propagate the per-subtile mean-intensity noise through the same scaling
   (`/ IQR^2`), the PCA rotation (`@ components^2`), and the standardization
   (`/ lambda_j`), flooring at `alpha_floor`.
5. For each kernel hypothesis `k` in the catalog of anisotropic length-scale
   pairs, compute the log marginal likelihood summed over standardized
   components (unit-amplitude prior is now approximately correct).
6. **Temper the model weights**: `w_k ∝ exp((LML_k - max LML) / T)` with
   `T = max(1, n_subtiles / temper_reference_subtiles)`. Untempered softmax
   weights collapse onto one catalog kernel as soon as `n` grows (LML
   differences scale linearly in `n`), which silently destroyed the model
   averaging in v4.4/v4.5. Tempering keeps the averaged variogram an honest
   mixture at realistic subtile counts.
7. The feature-space sill is `sigma^2 = sum_j lambda_j` (variance captured by
   the retained components, in robust-scaled units). The model-averaged
   semivariogram shape is `1 - rho_k(d)` with `rho_k` the Matern-3/2
   correlation under kernel `k`'s anisotropic metric.

The nugget (noise) enters the likelihood fit but cancels in utility
differences, so it does not appear in selection.

## 3. Anisotropy without approximation

Each catalog kernel has its own metric `(l_x, l_y)` in normalized field
coordinates. Rather than collapsing to a scalar distance, v5 evaluates each
kernel's variogram under its own metric:

- `D_k(p)` = distance from `p` to the nearest observed pixel under metric
  `k`, computed with an anisotropic Euclidean distance transform
  (`scipy.ndimage.distance_transform_edt` with `sampling = (step_y_norm /
  l_y, step_x_norm / l_x)`). One EDT per kernel per reveal.
- `R_k(p; r)` = distance from `p` to candidate rectangle `r` under the same
  metric, computed analytically from clamped axis distances.

The model-averaged expected-error reduction for candidate `r` is

```text
EER(r) = sigma^2 * sum_k w_k * sum_p w_p * [ (1 - rho(D_k(p))) - (1 - rho(min(D_k(p), R_k(p; r)))) ]
       = sigma^2 * sum_k w_k * sum_p w_p * [ rho(min(D_k(p), R_k(p; r))) - rho(D_k(p)) ]
```

normalized by `sum_p w_p` and divided by raster cost. Because `rho` is
monotone decreasing, `EER(r) >= 0` always; no clipping or eligibility gates
are needed.

## 4. Goal-oriented front weighting

The endpoint (front symmetric distance + penetration-d95 error) is decided in
a narrow band around the alteration front, but an unweighted integral spends
budget on bulk regions. V5 weights pixels by proximity to the *currently
predicted* front — a deployable quantity computed from the same
nearest-observation reconstruction every policy already maintains:

```text
w_p = 1 + kappa * exp(-0.5 * (dist_nm(p, predicted_front) / h)^2)
```

with bandwidth `h = front_bandwidth_nm` (default 1600 nm, one subtile width)
and `kappa` carried in the policy name. `kappa = 0` is the pure variogram
policy. If no front is predicted yet (early reveals, flat slices), the
weights are uniform, so the policy degrades gracefully to coverage — the
same safety property the v4.5 geometry backbone provided, but obtained
structurally instead of through gates.

## 5. Policies

```text
uncertainty_lookahead              # shared deterministic baseline (gamma = d)
variogram_eer_4x4_mean_kappa0      # v5.0: ML Matern variogram, uniform weights
variogram_eer_4x4_mean_kappa2      # v5.0: + moderate front weighting (band weight 3x)
variogram_eer_4x4_mean_kappa5      # v5.0: + strong front weighting (band weight 6x)
nested_veer_4x4_mean_kappa0        # v5.1: nested WLS variogram, uniform weights
nested_veer_4x4_mean_kappa2        # v5.1: + probability-field front weighting
nested_veer_4x4_mean_kappa5        # v5.1: + stronger front weighting
nested_veer_4x4_mean_kappa10       # v5.1: + strongest front weighting
nested_band_veer_4x4_mean_kappa2   # v5.1: nested variogram + Gaussian-band weights
nested_band_veer_4x4_mean_kappa5   # v5.1: nested variogram + Gaussian-band weights
```

## 6. Replay protocol

Identical raster protocol to v4.5: same 72-ROI catalog, 4 random pilot ROIs,
17 total ROIs, same scan-time and dose accounting, same nearest-observation
evaluator and frozen morphology proxies, same blocked folds and resumable
checkpoints.

One deliberate protocol fix: pilot ROIs are seeded with
`default_rng([seed, int(slice_id)])` instead of `default_rng(seed)`. Pilots
remain paired across policies within a slice (the paired comparison is
preserved) but now vary across slices. Under the v4 scheme every slice in the
study shared one fixed 4-ROI pilot layout, so the across-slice confidence
intervals conditioned on a single spatial draw. Because the baseline is
re-run inside the v5 driver, all v5 comparisons remain internally paired;
numbers are not directly comparable to v4-era artifact folders.

## 7. Validation gates

Unchanged from the project standard. Advancement on the frozen 30-slice
cohort requires:

```text
mean composite-error delta < 0
median composite-error delta <= 0
leave-one-slice-out worst mean delta <= 0
at least 3 of 5 fold means <= 0
RMSE regression <= 2%
equal scan cost
maximum slice composite regression <= 0.02
```

## 8. Commands

```powershell
.\.venv\Scripts\python.exe -m balance_nm validate-v5-veer-stack `
  --config configs\alloy617_v5_veer.yaml `
  --manifest data\alloy617_nrds\full_stack_download_manifest.csv `
  --fold all `
  --slices 001,011,021,032,042,053,054,064,075,085 `
  --out results\alloy617_v5_veer_smoke_010
```

## 9. Status: first real-slice smoke (slice 001, fold 1)

Artifact folder: `results/alloy617_v5_veer_smoke_001/`. Pipeline verified
end-to-end on real data: equal scan cost across arms, paired pilots, all
traces written, resume works.

| Policy | Composite Error | RMSE | Front Distance (nm) |
|---|---:|---:|---:|
| `uncertainty_lookahead` | 0.01678 | 0.21607 | 1457 |
| `variogram_eer_4x4_mean_kappa5` | 0.17704 | 0.21954 | 14639 |
| `variogram_eer_4x4_mean_kappa0` | 0.22401 | 0.22117 | 18496 |

One slice with fresh pilot seeds is not a verdict (the baseline's 0.0168 is
far below v4-era slice means, so this slice draw is unusually easy for
coverage), but the trace exposes a real mechanism that the next iteration
must address:

**Saturation indifference.** The marginal likelihood decisively selects
`l = 0.05` (weights collapse to one kernel even after tempering at
`T = 4.25`; the LML gaps are hundreds of nats). At that length scale the
Matern-3/2 correlation at one ROI width is 0.089 — the variogram saturates
within a single ROI. Consequences visible in the candidate trace: the top-10
candidates' EER values sit within 10-30% of each other at every adaptive
iteration. Under a saturated variogram, every pixel deeper than ~2 length
scales into a void carries identical expected error, so the objective is
nearly indifferent between splitting the largest unexplored void and
covering a moderate gap — whereas linear `gamma(d) = d` scores the deepest
void superlinearly (integrated distance over a 1-D gap of width `L` grows as
`L^2`). The morphology endpoint behaves like the latter: front localization
error in a row is bounded by the unsampled span that brackets the front, so
large contiguous gaps are quadratically painful in exactly the way the
stationary squared-error model does not see.

**Implication.** The stationary intensity variogram is an honest model of
mean squared reconstruction error, and it may still help on slices where
coverage is not the binding constraint (10-slice smoke pending). But the
endpoint is front *geometry*, and its uncertainty lives in front-position
space, not intensity space. The principled v5.1 objective is therefore
expected reduction of front-position variance directly: for each row, under
a locally uniform prior the front-column variance is `L^2 / 12` where `L` is
the width of the unsampled span bracketing the current front estimate.
Minimizing `sum_rows L^2 / 12` is "uncertainty lookahead in front-position
space" — it restores void-splitting in the penetration direction, needs no
exchange rate (units are nm^2 of front position, the endpoint's own units),
and reduces to a goal-oriented geometry rule with the variogram EER as a
natural tie-breaker among near-equal gap splits.

## 10. V5.1: nested WLS variogram and probability-field front weights

The 10-slice smoke (`results/alloy617_v5_veer_smoke_010/`) produced three
diagnostics that define v5.1:

1. **The maps have no sill.** Empirical semivariograms computed from the
   dense data grow essentially linearly out to half the field width on every
   slice checked (001: 0.28 to 2.43 over d = 0.01 to 0.6; 032: 0.73 to 7.82;
   075: 1.16 to 11.41). These are intrinsic, Brownian-like fields. The v5.0
   stationary Matern catalog (l <= 0.1) saturates by d ~ 0.2, so maximum
   likelihood — dominated by abundant short-range pairs — discards the
   long-range growth, which is precisely the void-splitting information. The
   baseline's hard-coded `gamma(d) = d` is a *better* long-range model of
   these maps than the fitted GP, which is why it is so strong.
2. **The final-iteration endpoint is metrically unstable.** Single reveals
   move the extracted front by 15-20 um in either direction for every policy
   (the baseline went 0 -> 21,494 nm on slice 032 as data was *added*).
   Late-window composite volatility (~0.05 std per slice) is the same order
   as the policy effects under study. Under a trailing-median readout
   (final 6 reveals), kappa5's mean delta improves to -0.0459, its maximum
   regression collapses from +0.160 to +0.0098 (inside the guardrail), and
   all kappa arms beat the baseline monotonically in kappa.
3. **Model averaging is decorative.** The winning kernel carries > 0.91
   weight on 9 of 10 slices; tempering at T ~ 4 is cosmetic against
   hundreds-of-nats likelihood gaps.

V5.1 therefore replaces the ML Matern machinery with a **nested unbounded
variogram fitted by method of moments**:

```text
gamma_hat(d) = c0 + c1 * (1 - rho_Matern32(d / l)) + c2 * d      c0, c1, c2 >= 0
```

fitted by Cressie-weighted (`w_b = N_b / gamma_b^2`) non-negative least
squares on the binned empirical semivariogram of the revealed robust-scaled
subtile features, with a 1-D scan over `l`. With fewer than three usable
bins the fit degrades to a slope-through-origin linear variogram. The nugget
`c0` is estimated but cancels in EER differences.

Properties:

- The linear term's expected-error reduction is exactly `c2` times the
  deterministic coverage gain, so `uncertainty_lookahead` is the
  data-selectable special case `c1 = 0`, and the exchange rate between
  coverage and short-range refinement is the estimated, per-slice,
  per-iteration quantity `c1 / c2` — completing the v4.5 alpha-elimination
  argument with a model class the data actually support.
- No Cholesky factorizations, marginal likelihoods, PCA noise propagation,
  or model averaging; the fit is deterministic and O(n^2) in subtiles.
- Selection uses a single isotropic normalized-coordinate distance field per
  iteration instead of one anisotropic field per catalog kernel.

V5.1 also replaces the hard-front Gaussian band weights with the
**uncertainty-inflated front-probability field** the shared evaluator's own
prediction already computes:

```text
w_p = 1 + kappa * alteration_front_probability(p)
```

This field is the normalized gradient of the altered-region probability,
maxed with the nearest-observation reconstruction uncertainty — which grows
linearly with gap depth. The weights therefore concentrate on sharp
predicted fronts *and* grow into unsampled voids, so front weighting can no
longer lock onto a wrong early front estimate, and void preference is
restored inside the weighting as well as inside the variogram.

Finally, v5.1 **pre-registers the trailing-median composite (final 6
reveals) as a co-primary endpoint** alongside the final-iteration composite,
written per slice to `v5_veer_trailing_summary.csv`. This is registered on
2026-06-11, on the basis of the policy-agnostic volatility diagnostic above
and before any 30-slice gate run, to avoid post-hoc metric selection.

### V5.1 10-slice smoke results (merged into `results/alloy617_v5_veer_smoke_010/`)

Trailing-median co-primary, delta vs `uncertainty_lookahead` across the 10
smoke slices:

| Arm | Mean | Median | Max regression | Wins |
|---|---:|---:|---:|---:|
| `nested_veer_kappa0` | -0.0509 | -0.0054 | +0.0367 | 6/10 |
| `variogram_eer_kappa5` (v5.0) | -0.0459 | -0.0302 | +0.0098 | 6/10 |
| `nested_band_veer_kappa2` | -0.0416 | -0.0100 | +0.0179 | 7/10 |
| `nested_veer_kappa2` | -0.0043 | -0.0083 | +0.3700 | 8/10 |

Interpretation:

- The nested variogram alone (`kappa0`) achieves the best mean in the study
  and converts the v5.0 slice-001 failure (+0.160 final-iteration) into the
  largest single win (-0.179): the unbounded linear term restored
  void-splitting as designed.
- The probability-field weights are **rejected**: their blowup slices move
  chaotically with kappa (+0.30 to +0.37 regressions), consistent with the
  |grad probability| field containing artificial gradients at the
  nearest-observation mosaic's tile seams, which the weights then chase.
- The hybrid `nested_band_veer_kappa2` (nested variogram + the robust v5.0
  Gaussian-band weights) is the only v5.1 arm inside the 0.02 guardrail on
  the co-primary, with 7/10 wins.
- Only `variogram_eer_kappa5` is strong on *both* endpoints (final-iteration
  mean -0.0387; trailing -0.0459); the v5.1 arms look good only under the
  robust endpoint, which is further evidence of how much final-iteration
  noise distorts arm ranking.
- Caveat: these 10 slices are the first third of the frozen 30-slice cohort,
  and arm selection happened on them. The 30-slice gate should be reported
  both with and without the 10 tuning slices.

### Frozen 30-slice gate results (`results/alloy617_v5_veer_gate_030/`)

Run 2026-06-12 with 12 worker processes (120 replays, ~12 minutes). Deltas
vs `uncertainty_lookahead` on the pre-registered trailing-median co-primary:

| Arm | Mean (95% CI) | Median | LOSO worst | Folds <= 0 | Wins | Max regression | RMSE reg. (excl. 265) |
|---|---|---:|---:|---:|---:|---:|---:|
| `variogram_eer_kappa5` | -0.0439 +/- 0.0315 | -0.0259 | -0.0337 | 5/5 | 20/30 | +0.1603 | -2.0% |
| `nested_veer_kappa0` | -0.0315 +/- 0.0252 | -0.0004 | -0.0250 | 5/5 | 17/30 | +0.0654 | -0.2% |
| `nested_band_veer_kappa2` | -0.0344 +/- 0.0381 | -0.0101 | -0.0245 | 4/5 | 20/30 | +0.2108 | -0.6% |

Verdict: **no arm promotes under the strict gates** — every arm fails the
per-slice max-regression guardrail (<= +0.02). `variogram_eer_kappa5` passes
every other criterion, including a paired 95% CI excluding zero (the
full-stack promotion bar, met for the first time in the project), holds on
the 20 held-out slices (-0.0430), and improves RMSE. Its result also holds
on the legacy final-iteration endpoint (mean -0.0248, median -0.0100, 4/5
folds) except for the same guardrail.

The blockers are localized, not systematic:

- **Slice 265 is degenerate**: every policy including the baseline has
  normalized RMSE > 1.0 (baseline 1.13) while the composite endpoint is
  exactly 0.0 for all arms — the intensity normalization blows up on a
  near-flat field and the RMSE gate measures noise there. The nested arms'
  headline +12% RMSE regression is entirely this slice (-0.2% / -0.6%
  without it). Slice 265 needs a data-quality audit; any exclusion rule must
  be pre-registered before a full-stack run.
- **`variogram_eer_kappa5`'s entire tail is slices 106 and 107** (adjacent,
  fold 2): +0.0598 and +0.1603, both "easy" slices where the baseline is
  near-perfect (composite 0.0065 on 107) and any adaptive deviation reads as
  a large relative regression. Excluding them the mean is -0.0549 and all
  remaining slices are within the guardrail. Tail control on easy slices is
  the single remaining obstacle, as it was for v4.4/v4.5.

### Pre-registered degenerate-slice rule (registered 2026-06-12)

The slice 265 audit confirmed a blank field: all 15 channels have zero IQR,
99.8% of pixels are zero, and the frozen reference contains **no altered
region and no front** (`altered_fraction = 0`, `front_fraction_rows = 0`,
`d95 = NaN`). The composite endpoint is vacuously 0 for every policy and the
normalized RMSE exceeds 1.0 for every policy including the baseline, so both
gate metrics are ill-conditioned there.

Rule, registered before any full-stack run: **slices whose frozen reference
contains no front (`front_fraction_rows == 0`) are excluded from endpoint
aggregation.** The criterion is computed from the frozen evaluation-only
reference, identically for all policies, so it cannot leak acquisition
behavior. Gate results are reported with and without such slices.

## 11. V5.2: movement-gated front weighting

The slice 106/107 diagnosis identified the precise tail mechanism: the
predicted front was **stable but wrong** for reveals 6-14 (front error
constant at ~15.3 um while the baseline's coverage sampling disambiguated
the segmentation by reveal 8-9). Band weights around a static front estimate
concentrate samples where they cannot change the estimate; the
disambiguating information comes from coverage.

V5.2 therefore gates the front-weighting strength by the **movement of the
predicted front between consecutive reveals**:

```text
movement_t = mean_rows( |depth_t - depth_(t-1)| ) / slice_width
             (rows where the front appears/disappears count as a full width)
kappa_eff  = kappa * min(1, movement_t / front_gate_movement_fraction)
```

with `front_gate_movement_fraction = 0.01` (1% of slice width). A static
front estimate — converged *or* stuck — sends `kappa_eff -> 0` and the
policy reverts to uniform-weight variogram EER (coverage-seeking); a moving
estimate keeps the focus on the front band. Both failure directions are
covered: easy slices shut the gate off early (no deviation downside), and
stable-but-wrong lock-ins shut it off too (coverage breaks the deadlock).
The movement signal is deployable (computed from the policy-visible
reconstruction only) and is traced per iteration
(`front_movement_fraction`, `front_kappa_effective`).

Policies:

```text
gated_veer_4x4_mean_kappa5         # v5.2: ML variogram + movement-gated band weights
gated_veer_4x4_mean_kappa10        # v5.2: higher base kappa (gate shrinks it adaptively)
```

Caveat: v5.2 was designed from gate-cohort diagnostics (slices 106/107), so
its 30-slice numbers are screening, not confirmation. Promotion requires the
untouched remainder of the full stack.

### V5.2 30-slice screening results (2026-06-12)

`gated_veer_4x4_mean_kappa5` **passes all seven gate criteria** on the
pre-registered trailing-median co-primary — the first full gate pass in the
project — and does so with and without the degenerate-slice rule:

| Criterion | All 30 | Excl. 265 | Gate |
|---|---:|---:|---|
| Mean delta (95% CI) | -0.0478 +/- 0.0270 | -0.0494 +/- 0.0277 | < 0, CI excludes 0: PASS |
| Median delta | -0.0168 | -0.0168 | <= 0: PASS |
| LOSO worst mean | -0.0386 | -0.0400 | <= 0: PASS |
| Fold means <= 0 | 5/5 | 5/5 | >= 3/5: PASS |
| RMSE regression | -2.0% | -2.0% | <= 2%: PASS |
| Scan cost | equal | equal | PASS |
| Max slice regression | **+0.0118** | +0.0118 | <= +0.02: PASS |

The movement gate fixed exactly the diagnosed failure — slices 106/107 went
from +0.0598/+0.1603 (ungated) to **+0.0016/+0.0020** — while *improving*
the mean over the ungated arm (-0.0478 vs -0.0439): robustness was not
bought with performance. Wins: 19/30.

Qualifications: (1) on the legacy final-iteration endpoint the guardrail
still fails (max +0.168; the endpoint's single-reveal volatility is why the
trailing co-primary exists); (2) `kappa10` under the same gate has a +0.155
tail, so the mechanism is kappa-sensitive — kappa5 is the candidate, not the
family; (3) the cohort has now been used for both arm selection and gate
design, so **promotion requires the untouched full-stack remainder**
(gated_kappa5 vs baseline, all gates as registered, degenerate-reference
slices excluded per the pre-registered rule).

### Full-stack confirmation (2026-06-12, `results/alloy617_v5_veer_confirm_full_stack/`)

`gated_veer_4x4_mean_kappa5` vs `uncertainty_lookahead` on the **235
held-out slices** never used for any design or selection decision (470
replays; degenerate-reference rule applied — zero exclusions triggered;
every slice `matched_present`).

**The registered full-stack promotion criteria are met on both endpoints:**

| | Final-iteration (legacy primary) | Trailing-median (co-primary) |
|---|---|---|
| Mean delta (95% CI) | **-0.0239 [-0.0377, -0.0102]** | **-0.0189 [-0.0355, -0.0023]** |
| Relative improvement | 17.4% | 13.6% |
| RMSE regression | **-1.65%** (improved) | — |
| Equal scan cost | yes | yes |
| Median delta | -0.0063 | +0.0010 |
| Wins | 133/235 | 116/235 |
| Improvements / regressions beyond 0.02 | 99 / 59 | 96 / 83 |
| Large effects (abs delta > 0.10), wins : losses | 53 : 22 | 57 : 30 |

**Honest characterization.** The mean improvement is real, confirmed out of
sample with confidence intervals excluding zero on both endpoints, at equal
cost and with better reconstruction RMSE. But the method is a **risk
redistribution, not a uniform improvement**: large wins outnumber large
losses roughly 2:1, the median slice is near a wash, and the 30-slice
guardrail did not generalize (max out-of-sample regression +0.42 to +0.47;
25-35% of slices regress by more than +0.02). The screening cohort's tail
estimate (+0.0118 max) badly understated true tails — n = 30 cannot bound a
heavy-tailed per-slice distribution, and a large share of the per-slice
spread is the endpoint's own single-reveal volatility (~0.05 std).

Promotion verdict: **pass per the registered criteria**, with the risk
profile above attached as a mandatory caveat. Deployment-style claims should
be phrased as "improves expected morphology error by ~14-17% at equal dose,
with per-slice variance" — not as a per-slice dominance claim.

### Large-loss inspection (22 slices with final delta > +0.10)

- **15 sustained / 4 volatility / 3 intermediate.** Four losses are pure
  final-reveal metric noise (slice 256: final +0.42 but trailing -0.107 — a
  *win* on the robust endpoint); fifteen are genuine sustained failures.
- **Losses are the easy-slice tax.** Baseline median composite on loss
  slices is 0.018 vs 0.215 on large-win slices: the method wins where
  coverage fails and loses where coverage was already sufficient.
- **The movement gate never fired on them.** Mean front movement on the
  sustained losses is 0.08-0.24 — 8 to 24 times the 0.01 gate threshold —
  keeping `kappa_eff` at 2.9-5.0 throughout. These are not 106/107-style
  static lock-ins but the opposite: the predicted front *churns* (the GMM
  segmentation flips reveal-to-reveal under focused sampling), and the
  monotone gate reads churn as a front being refined.
- **Movement alone cannot separate wins from losses** — large wins also
  churn (slice 047: -0.234 with movement 0.26). A band-pass gate (shut
  kappa off above a churn ceiling) is the natural v5.3 hypothesis, but it
  cannot be honestly validated within this stack: **no untouched data
  remains.** Further iteration requires a new specimen/stack or strict
  nested cross-validation from scratch.

## 12. Known approximations and limitations

- `gamma(d_p)` treats nearest-observation error as a pure function of
  distance to the single nearest sample; it ignores screening from multiple
  nearby samples (a kriging-variance refinement, deliberately out of scope).
- The variogram is stationary within a slice. Front weighting reintroduces
  spatial focus on the endpoint-relevant band; a locally varying variogram is
  the natural v5.x extension if the stationary fit underperforms on
  heterogeneous slices.
- Subtile featurization reuses the v4 plumbing, including
  `acquisition_v4.bayesian_pareto.residual_filter_sigma_px` for the noise
  estimator. V5-specific knobs live under `acquisition_v5`.
- The standardized unit-amplitude prior is a first-order calibration; full
  amplitude estimation per component is omitted because only the *shape* of
  `gamma_hat` affects the argmax (the sill is a common factor within an
  iteration).
