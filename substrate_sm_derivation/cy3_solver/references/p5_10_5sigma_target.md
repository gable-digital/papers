# §5.10 — TY-vs-Schoen σ-discrimination: round-6 (P5.5k) headline

**Date**: 2026-04-27 (original); 2026-04-29 (P5.5d / P5.5f / round-3 /
round-4 / P5.5j); 2026-04-29 (P5.5k — round-6 hostile-review fix —
canonical discriminability headline); **2026-04-30 (P8.1e —
σ-channel removed from model-comparison BF, retained as
discriminability metric)**

> **P8.1e clarification (2026-04-30).** σ-channel provides
> *discriminability strength only* (|t|=6.92, BIC-corrected
> ln BF=22.16 nats), not model-comparison evidence. The multi-channel
> Bayes factor (`output/p8_1_bayes_multichannel.json`,
> `physics_preference.combined_log_bf`) excludes σ because the σ-gap
> is dominated by basis-size differences (n_TY=87 vs n_Schoen=27 at
> k=3), not by physics observables. Physics-channel combination uses
> chain_quark + chain_lepton (and Hodge / Yukawa when those channels
> stabilize). The 6.92σ headline below is the discriminability claim
> ("we can statistically distinguish TY from Schoen"), and is reported
> separately under the new top-level `discriminability` JSON key. The
> physics preference (which substrate matches the standard model) is
> Schoen-favored at ln BF ≈ -14.76 nats / n-σ ≈ 5.43 (chain channels
> only).

## P5.5k headline (round-6 hostile-review fix — Findings 1, 2, 3)

Round-6 hostile review (commit b206abd4) identified three findings:

1. **Iter cap=50 truncates real convergence at n_pts=40k.** At canonical
   n_pts=40k iter_cap=50, 5 Schoen seeds (7, 314, 1000, 2024, 57005) hit
   the iter cap with residuals 3e-5..5e-4 — above tol=1e-6 so they were
   excluded from Tier 0 even though no-guard trajectories show clean
   convergence under more iters. Default raised to **iter_cap=100**.
2. **Bootstrap CI: percentile vs BCa.** Headline reported only
   percentile CI; Efron's BCa (bias-corrected accelerated) is in some
   cases meaningfully tighter on the lower bound. Both CIs now reported.
3. **Tier 0 ⊂ Tier 3 nested framing.** At n_pts=40k iter_cap=50, Tier 3
   (n=17) was a strict superset of Tier 0 (n=12); commit message
   framing "both clear 5σ" implied independence. Below we explicitly
   document the nesting / non-nesting at each setting.

### P5.5k four-tier table — canonical n_pts=40 000, iter_cap=100, B=10 000

`output/p5_10_n40k_p5_5k.json`; wallclock 507 s:

| tier | filter | TY n | Sc n | n-σ point | pct 95% CI | BCa 95% CI | point ≥5σ? | pct floor ≥5σ? | BCa floor ≥5σ? |
|---|---|---:|---:|---:|---|---|---|---|---|
| **Tier 0: strict-converged** | residual < tol AND iters < cap | 20 | **16** | **6.921** | [5.653, 10.030] | [5.296, 9.037] | YES | YES | YES |
| **Tier 1: conservative** | full 20+20 | 20 | 20 | 3.990 | [3.286, 10.644] | [2.891, 8.965] | NO | NO | NO |
| **Tier 2: canonical** | residual < 1e-3 (1000× tol) | 20 | 18 | 3.519 | [2.942, 10.046] | [2.509, 8.050] | NO | NO | NO |
| **Tier 3: Tukey re-trim** | Tier 2 ∖ Tukey 1.5·IQR outliers | 20 | 17 | 7.434 | [5.942, 11.530] | [5.274, 9.903] | YES | YES | YES |

**Tier 0 grew from 12 → 16 Schoen seeds** under iter_cap=100. The
recovered seeds 7, 1000, 2024 hit the strict-converged cut cleanly
(residual ~9e-7 at iters 80-86); seed 314 also converged but with σ=12.97
(an outlier that pulls Tier 0 ⟨σ_Sc⟩ up); seed 57005 still doesn't make
Tier 0 (cap=100, residual 8e-5) but is admitted to Tier 2 / Tier 3 by
the residual<1e-3 filter. Seed 5 is the new persistent non-converger
(early bail at iter 17, residual 2.4e-4, σ=43.24 — the outlier that
sinks Tier 1/2).

### P5.5k four-tier table — n_pts=25 000, iter_cap=100, B=10 000

`output/p5_10_n25k_p5_5k.json`; wallclock 264 s:

| tier | filter | TY n | Sc n | n-σ point | pct 95% CI | BCa 95% CI | point ≥5σ? | pct floor ≥5σ? | BCa floor ≥5σ? |
|---|---|---:|---:|---:|---|---|---|---|---|
| **Tier 0: strict-converged** | residual < tol AND iters < cap | 20 | 16 | 4.564 | [3.745, 10.458] | [3.223, 6.724] | NO | NO | NO |
| **Tier 1: conservative** | full 20+20 | 20 | 20 | 6.017 | [4.814, 10.703] | [4.171, 8.767] | YES | NO | NO |
| **Tier 2: canonical** | residual < 1e-3 | 20 | 19 | 5.635 | [4.533, 10.663] | [4.035, 8.450] | YES | NO | NO |
| **Tier 3: Tukey re-trim** | Tier 2 ∖ outliers | 20 | 17 | **8.608** | [5.492, 12.535] | [5.653, 13.032] | YES | YES | YES |

At n_pts=25k iter_cap=100, Tier 0 still falls below 5σ at the point
estimate (4.564) because seed 1000 σ=18.64 sits in the strict-converged
Schoen sample (no Tukey trim) and inflates ⟨σ_Sc⟩+SE. Tier 3 (Tukey re-trim)
still carries the n=25k headline at n-σ=8.608, with both percentile and
BCa CI floors clearing 5σ.

### Tier 0 / Tier 3 nesting (Finding 3)

Per the round-6 review, here are the explicit subset relations:

* **n_pts=40k iter_cap=100:** Tier 0 ⊂ Tier 3 (Tier 3 = Tier 0 ∪ {57005}).
  Tier 3 extends Tier 0 by including one additional seed via Tukey trim
  on the residual<1e-3 canonical pool. The two are *not* independent
  confirmations — Tier 3 is Tier 0 + 1 Tukey-admitted seed.
* **n_pts=25k iter_cap=100:** Tier 0 ⊄ Tier 3 and Tier 3 ⊄ Tier 0.
  Tier 0 \ Tier 3 = {666, 1000} (Tier 0 admits these strict-converged
  seeds; Tier 3's Tukey filter trims them as σ-outliers — σ=12.31, 18.64).
  Tier 3 \ Tier 0 = {2, 271, 57005} (Tier 3 admits these via the
  residual<1e-3 canonical filter; Tier 0 excludes 2 (residual=1.1e-6
  just-above-tol), 271 (residual=5.2e-5 > tol), 57005 (residual=1.6e-6
  just-above-tol)). At n_pts=25k the two tiers ARE genuinely distinct
  and their joint clearance of 5σ provides somewhat independent evidence.

The previous round-5 framing "both clear 5σ" at n_pts=40k iter_cap=50
overstated the independence: that pair was nested (Tier 0 ⊂ Tier 3),
so it was one converged subset reproduced two ways, not two
independent confirmations. The corrected framing: at n_pts=40k
iter_cap=100, **Tier 3 extends Tier 0 by one Tukey-admitted seed; the
two are nested, and the load-bearing 5σ result is Tier 0 (n_Sc=16)**.

### Final 5σ verdict (P5.5k)

* **Canonical n_pts=40 000, iter_cap=100**: Tier 0 (strict-converged,
  n_Sc=16) clears 5σ at point AND both CI floors:
  n-σ = **6.921**, percentile CI = [5.653, 10.030],
  BCa CI = [5.296, 9.037]. **This is the most-defensible publication
  headline** — 16 Schoen seeds with residual < tol(1e-6) AND iters <
  cap(100), no outlier trimming, no guard-restored snapshots. The BCa
  CI floor (5.296) is 0.357σ tighter on the lower bound than the
  percentile floor; both clear 5σ.
* **Robustified n_pts=25 000, iter_cap=100**: 5σ cleared at point AND
  both CI floors only at Tier 3 (Tukey re-trim), n-σ = 8.608 with
  pct=[5.492, 12.535] and BCa=[5.653, 13.032]. Tier 0 falls short
  (4.564 point) due to in-sample σ-outliers (seeds 666 σ=12.31 and
  1000 σ=18.64) that the larger n_pts=40k sampler tightens out.

**Headline tier (most defensible): Tier 0 at n_pts=40k iter_cap=100,
clearing 5σ at point estimate AND both percentile AND BCa CI floors.**

### History of the 40k Tier 0 number

| revision | iter_cap | n_Sc | n-σ | pct CI | BCa CI |
|---|---:|---:|---:|---|---|
| P5.5j (round-5) | 50 | 12 | 6.801 | [5.547, 10.244] | n/a |
| **P5.5k (round-6)** | **100** | **16** | **6.921** | **[5.653, 10.030]** | **[5.296, 9.037]** |

Tier 0 strengthened modestly in n-σ (6.801 → 6.921) and grew the
strict-converged sample from 12 to 16 seeds. The lower-bound CI
tightened slightly on the percentile method (5.547 → 5.653) but
loosened on the BCa method (n/a → 5.296). Both CI floors still clear
5σ; the BCa floor is the more conservative referee anticipation.

---

## Round-5 P5.5j headline (preserved for history; superseded by P5.5k)

**Date**: 2026-04-27 (original); 2026-04-29 (P5.5d / P5.5f / round-3 /
round-4 re-runs); 2026-04-29 (P5.5j — round-5 hostile-review fix)

## P5.5j headline (round-5 hostile-review fix — Concern A)

Round-5 hostile review (commit e8f68d4c) identified Concern A as the
most important methodological defect: the P5.5f regression guard +
det-sentinel were firing on legitimately monotonically-decaying
trajectories, truncating Schoen seeds 1000 and 2024 at iter 5 / iter 9
respectively while the no-guard trajectories converged cleanly to
residual ~9e-7 by iter 25 / 30. P5.5j relaxes both guards:

* **Regression guard** (in `solve_schoen_metric` / `solve_ty_metric`):
  pre-P5.5j was `residual > 10× min` for 2 consecutive iters. Post-P5.5j
  is `residual > 100× min` AND `min < 1e-2` AND `5 consecutive iters`.
* **Det-sentinel** (in `donaldson_iteration`): pre-P5.5j returned NaN if
  `log_abs_det < -92.1` (i.e. |det| < 1e-40). Round-5 diagnostic showed
  this fired on healthy seed 1000 at iter 4 (det 9.24e-41, mid-descent).
  Empirical calibration showed there is NO clean threshold between
  healthy det values and seed-271's catastrophe (det 9.13e-52); the
  threshold is removed entirely. The `det_zero` check (truly underflowing
  pivot) and `block_inv` finite-checks remain as backstops, and the
  residual-based regression guard now carries the catastrophe-detection
  responsibility.

Regression tests added: `schoen_seed_1000_does_not_early_bail_on_healthy_trajectory`,
`schoen_seed_2024_does_not_early_bail`,
`schoen_seed_271_still_catches_catastrophic_divergence` (all `#[ignore]`'d
by default; runtime ~30 s each at canonical n_pts=25k). Seed 271's
σ_final at canonical settings is now 8.27 (well within band [2,30]),
runs 31 iters before its tighter-numerical-resolution catastrophe
fires.

### P5.5j four-tier table — canonical (n_pts=25 000)

`output/p5_10_ty_schoen_5sigma_p5_5j.json`, B=10 000, re-trim bootstrap:

| tier | filter | TY n | Sc n | n-σ point | n-σ 95% CI | point ≥5σ? | CI floor ≥5σ? |
|---|---|---:|---:|---:|---|---|---|
| **Tier 0: strict-converged** | residual < tol(1e-6) AND iters < cap(50) | 20 | 14 | **4.038** | [3.320, 9.760] | NO | NO |
| **Tier 1: conservative** | full 20+20 (incl. guard-restored snapshots) | 20 | 20 | **6.017** | [4.814, 10.703] | YES | NO |
| **Tier 2: canonical** | residual < 1e-3 (1000× tol) | 20 | 19 | **5.635** | [4.533, 10.663] | YES | NO |
| **Tier 3: Tukey re-trim** | Tier 2 ∖ Tukey 1.5·IQR outliers | 20 | 17 | **8.608** | [5.492, 12.535] | YES | YES |

Compared to round-4 P5.5f at canonical n_pts=25k:

* Tier 0 n_Sc grew from 12 → 14 (recovered seeds 1000 and 2024 — but
  seed 1000 σ=18.64 enters the unfiltered Schoen mean, increasing variance).
* Tier 0 n-σ moved 4.811 → 4.038 (variance growth from σ=18.64 outlier
  outweighs the n-growth at the point estimate).
* Tier 3 (which trims the σ=18.64 outlier) hardens: n-σ 7.783 → 8.608,
  CI floor 4.933 → **5.492** (now clears 5σ at point AND CI floor).

**At canonical n_pts=25k, Tier 3 (Tukey re-trim) is the strictest tier
clearing 5σ at both point estimate and 95% CI lower bound: n-σ = 8.608,
CI = [5.492, 12.535].** Headline tier selection logic in the binary
agrees.

### P5.5j four-tier table — robustified (n_pts=40 000)

`output/p5_10_n40k.json`, B=10 000, re-trim bootstrap; wallclock 472 s:

| tier | filter | TY n | Sc n | n-σ point | n-σ 95% CI | point ≥5σ? | CI floor ≥5σ? |
|---|---|---:|---:|---:|---|---|---|
| **Tier 0: strict-converged** | residual < tol AND iters < cap | 20 | 12 | **6.801** | [5.547, 10.244] | YES | YES |
| **Tier 1: conservative** | full 20+20 | 20 | 20 | **3.760** | [3.153, 10.969] | NO | NO |
| **Tier 2: canonical** | residual < 1e-3 | 20 | 18 | **3.291** | [2.816, 10.777] | NO | NO |
| **Tier 3: Tukey re-trim** | Tier 2 ∖ outliers | 20 | 17 | **8.098** | [6.518, 12.432] | YES | YES |

The n_pts=40k Schoen ensemble has a different outlier pattern: seed 5
returns σ=43.24 (an outlier far worse than anything at n_pts=25k),
which inflates the Tier 1/2 SE and drops their n-σ below 5σ. However:

* **Tier 0 (strict-converged) clears 5σ cleanly: n-σ = 6.801,
  CI = [5.547, 10.244].** This is the most-defensible tier — 12 Schoen
  seeds that fully converged (residual < tol, iters < cap), all
  σ in [2.19, 8.36], no outliers.
* **Tier 3 (Tukey re-trim) also clears 5σ: n-σ = 8.098, CI = [6.518, 12.432].**

### Final 5σ verdict (P5.5j)

* **Canonical n_pts=25 000**: 5σ cleared at point AND CI floor at
  Tier 3 (Tukey re-trim), n-σ = 8.608 [5.492, 12.535]. Tier 0
  point estimate (4.038) is below 5σ due to seed 1000 / seed 666
  outliers in the unfiltered Schoen sample (σ=18.64, 12.31 — both
  legitimately converged but σ-large for unknown CY-geometric reasons).
* **Robustified n_pts=40 000**: 5σ cleared at point AND CI floor at
  the *strictest* tier (Tier 0 strict-converged), n-σ = 6.801
  [5.547, 10.244]. **This is the publication-quality headline.**

The headline at n_pts=40k Tier 0 is the most defensible 5σ result
because it requires the least interpretation: 12 Schoen seeds that
fully converged, no Tukey trimming, no guard-restored mid-descent
snapshots. The discriminator is robust to outlier pattern (the
σ=43.24 outlier sits in n_Sc=20 Tier 1 but is filtered from
Tier 0 by hitting the iter cap at residual 2.4e-4 > tol).

---

## Round-4 hostile-review headline (preserved for history; superseded by P5.5j)
**Crate**: `book/scripts/cy3_substrate_discrimination/rust_solver`
**Pipelines**:
- `route34::ty_metric` (driven via `route34::cy3_metric_unified::TianYauSolver`)
- `route34::schoen_metric` (driven via `route34::cy3_metric_unified::SchoenSolver`)

**Tests**: `src/route34/tests/test_p5_10_ty_schoen_5sigma.rs`
**Binary**: `src/bin/p5_10_ty_schoen_5sigma.rs`
**Data**:   `output/p5_10_ty_schoen_5sigma.json`,
            `output/p5_10_round4.json` (current),
            log at `output/p5_10_round4.log`

## Headline result (round-4 hostile-review, four-tier; B=10 000 with re-trim bootstrap)

**5σ on σ alone is met *only* at Tier 1 (the conservative full ensemble)
and only when guard-restored mid-descent Schoen snapshots are kept in
the ensemble. Under any stricter "fully converged" filter, EITHER the
point estimate (Tier 0) OR the 95% CI lower bound (Tier 2 and Tier 3)
falls below 5σ. The 5σ project goal is therefore not robustly cleared
on σ alone at this configuration; this is reported honestly, not
papered over.**

Round-4 hostile review (commit 4097fd43) identified two methodological
defects in the round-3 headline:

* **Defect B — Tukey trim baked into bootstrap.** The previous Tier 3
  applied the Tukey 1.5×IQR filter ONCE before passing the trimmed
  σ-vector into a vanilla bootstrap. The bootstrap loop resampled the
  *post-trim* n=13 vector and never re-applied the trim per resample,
  so the CI was artifactually narrow. Proper re-trim-per-resample
  bootstrap at B=10 000 widens the Tier 3 CI from [6.185, 11.553] to
  **[4.933, 13.168]** — the lower bound now sits *below* 5σ, and the
  Tier 3 "most defensible headline" claim collapses.
* **Defect E — Classifier ignores iter cap.** The round-3 canonical
  filter (`residual < 1e-3`) admits Schoen seed 99, which hit the
  Donaldson iteration cap (`iters=50`) at residual `2.4e-6` — i.e.
  ABOVE the actual `tol=1e-6` target by 2.4×. Tightening to "residual
  < tol AND iters < cap" gives the new **Tier 0 (strict-converged)**
  with n=12 Schoen and n-σ = **4.811** — *below* 5σ at the point
  estimate.
* **Defect C — 1000 resamples too few.** At B=1000 the CI lower bound
  jitters ±0.14σ across `boot_seed` ∈ {12345, 999, 31415, 1, 7777}.
  Default bumped to **B=10 000** in the binary; jitter at B=10 000 is
  ≤ 0.03σ across the same seeds (verified in the printed table; see
  "Bootstrap-seed jitter sensitivity" below).

The fix wires four tiers into the binary and the JSON output. The
classifier "converged" (residual < 1e-3) survives as Tier 2 for
back-compat; "strict-converged" (residual < tol AND iters < cap) is
the new strictest tier (Tier 0); the Tukey-trimmed Tier 3 now uses
re-trim-per-resample bootstrap.

| tier | filter | TY n | Sc n | n-σ point | n-σ 95% CI (B=10 000) | point ≥5σ? | CI floor ≥5σ? |
|---|---|---:|---:|---:|---|---|---|
| **Tier 0: strict-converged** | residual < tol(1e-6) AND iters < cap(50) | 20 | 12 | **4.811** | [3.805, 9.989] | NO | NO |
| **Tier 1: conservative** | full 20+20 (incl. guard-restored snapshots) | 20 | 20 | **6.482** | [5.302, 11.001] | YES | YES |
| **Tier 2: canonical** | residual < 1e-3 (1000× tol) | 20 | 14 | **5.647** | [4.426, 10.819] | YES | NO |
| **Tier 3: Tukey-trimmed canonical (RE-TRIM bootstrap)** | Tier 2 ∖ Tukey 1.5·IQR outliers | 20 | 13 | **7.783** | [4.933, 13.168] | YES | NO |

The verdict-selection logic in the binary picks the headline tier as
the strictest tier that clears 5σ at *both* the point estimate and the
95% CI lower bound. With round-4 numbers, that is **Tier 1
(conservative)** — the only tier where the CI lower bound (5.302σ)
exceeds 5σ. Tier 1 is methodologically the *weakest* discriminator
because it includes 5 Schoen seeds that bailed via the P5.5f regression
guard at iters 2-5 with residuals 0.05..0.49 (mid-descent snapshots,
not Donaldson-balanced σ values). Reporting Tier 1 as the headline is
defensible only if those guard snapshots are accepted as fair σ values
for σ-discrimination purposes — the very assumption the round-3 fix
*rejected*.

### Honest verdict

* **At the point estimate**, 5σ is cleared at Tiers 1, 2, and 3.
  Tier 0 falls short (4.811σ).
* **At the 95% CI lower bound**, 5σ is cleared *only* at Tier 1
  (5.302σ). Tier 0 (3.805σ), Tier 2 (4.426σ), and Tier 3 (4.933σ)
  all fall below.
* **The most-defensible "fully converged" filter (Tier 0) does not
  clear 5σ even at the point estimate.** The 5σ project goal on σ
  alone is therefore not robustly met under the strictest
  classifier; the Tier 1 result is the only one that clears at both
  point and CI floor, and that tier admits guard-restored mid-
  descent snapshots that round-3 ruled methodologically invalid.

This does not mean the substrate is undiscriminated — the σ channel
*does* point in the right direction at every tier (TY mean σ ≈ 0.27,
Schoen mean σ ≈ 4.4–6.5 across the four tiers, all signs correct, all
point n-σ > 4.8). It means the σ-only headline at this budget is
"≥4.8σ at the point estimate under any reasonable filter, but the 95%
CI floor is ≥5σ only when guard snapshots are kept". Bumping `n_pts`
or going to k=4 should robustify the result; that is the future-work
cell at the bottom of this report.

### Round-4 numerical summary (B=10 000, re-trim bootstrap on Tier 3)

```
Tier 0 (strict-converged): n_TY=20, n_Schoen=12, ⟨σ_TY⟩=0.268309, SE=8.979e-4,
                           ⟨σ_Schoen⟩=4.357673, SE=8.500e-1, n-σ=4.811,
                           95% CI=[3.805, 9.989]
Tier 1 (conservative):     n_TY=20, n_Schoen=20, ⟨σ_TY⟩=0.268309, SE=8.979e-4,
                           ⟨σ_Schoen⟩=5.368323, SE=7.869e-1, n-σ=6.482,
                           95% CI=[5.302, 11.001]
Tier 2 (canonical):        n_TY=20, n_Schoen=14, ⟨σ_TY⟩=0.268309, SE=8.979e-4,
                           ⟨σ_Schoen⟩=4.442612, SE=7.392e-1, n-σ=5.647,
                           95% CI=[4.426, 10.819]
Tier 3 (Tukey re-trim):    n_TY=20, n_Schoen=13, ⟨σ_TY⟩=0.268309, SE=8.979e-4,
                           ⟨σ_Schoen⟩=3.837535, SE=4.586e-1, n-σ=7.783,
                           95% CI=[4.933, 13.168]    <-- defect-B fix WIDENS CI
```

### Bootstrap-seed jitter sensitivity (round-4 Defect C)

CI lower-bound stability across `boot_seed ∈ {12345, 999, 31415}` at
B=10 000:

| label | primary_lo (seed=12345) | seed_999 | seed_31415 | max_jitter |
|---|---:|---:|---:|---:|
| strict_converged_k3   | 3.805 | 3.814 | 3.821 | 0.016 |
| canonical_k3          | 4.426 | 4.421 | 4.417 | 0.009 |
| reference_retrim_k3   | 4.933 | 4.925 | 4.954 | 0.021 |

Overall max_jitter = 0.021σ — well within the 0.05σ target. At B=1000
this jitter ranged up to 0.14σ, which would have hidden the
Tier-3 CI-floor failure under boot-seed noise.

### History of the headline number

| revision | n-σ | classifier | notes |
|---|---:|---|---|
| P5.7 | 4.854 | full ensemble, n_pts=10 000 | 3% short of 5σ |
| P5.10 pre-fix | 8.761 | full ensemble, n_pts=25 000 | pre-Donaldson-inversion fix |
| P5.10 post-P5.5d | 5.963 | full ensemble | post-fix, before regression guard |
| P5.10 post-P5.5f | 6.481 | full ensemble | post regression guard (Tier 1 ≈) |
| P5.10 post-P5.5f balanced-only | 5.647 | iter-based, iters >= 10 | round-2 (Concern B) |
| P5.10 round-3 canonical | 5.647 | residual < 1e-3 | residual-based; matches iter-based by coincidence |
| P5.10 round-3 conservative | 6.482 | full ensemble | upper bound, includes guard snapshots |
| P5.10 round-3 reference (BAD CI) | 7.783 | Tukey-trimmed canonical, **bake-once** trim + B=1000 boot | round-3 headline; CI artifactually narrow due to defect B |
| **P5.10 round-4 strict (Tier 0)** | **4.811** | residual < tol AND iters < cap | Defect E: round-3 admitted seed 99 (cap-hit, residual > tol) |
| **P5.10 round-4 reference (Tier 3)** | **7.783** | Tukey-trimmed canonical, **re-trim** boot + B=10 000 | Defect B fix: CI WIDENS to [4.933, 13.168] |
| **P5.10 round-4 headline** | **6.482** | Tier 1 conservative (only tier clearing 5σ at point AND CI floor) | honest report; *not* methodologically strict |

The canonical and balanced-only n-σ happen to coincide (5.647σ) because
on this particular run, every seed flagged as `iters < 10` also has
residual >= 1e-3, and no other seed has residual >= 1e-3. The two
classifiers identify the same subset; the residual-based classifier is
the more principled one and is now canonical.

### Schoen per-seed classification (round-3)

| seed | iters | residual | tier |
|---|---:|---|---|
| 42 | 22 | 7.5e-7 | converged |
| 100 | 19 | 7.0e-7 | converged |
| 12345 | 24 | 7.5e-7 | converged |
| 7 | 40 | 9.3e-7 | converged |
| 99 | 50 | 2.4e-6 | converged (just over tol; iter cap) |
| 1 | 25 | 8.1e-7 | converged |
| **2** | **3** | **8.8e-2** | **non-converged (early-bail)** |
| 3 | 43 | 9.3e-7 | converged |
| 4 | 24 | 7.8e-7 | converged |
| **5** | **2** | **4.9e-1** | **non-converged (early-bail)** |
| 137 | 50 | 8.5e-7 | converged |
| **271** | **4** | **1.5e-1** | **non-converged (early-bail)** |
| 314 | 28 | 6.9e-7 | converged |
| 666 | 46 | 9.1e-7 | converged (Tukey outlier — σ=12.31) |
| **1000** | **5** | **5.5e-2** | **non-converged (early-bail)** |
| **2024** | **9** | **4.5e-3** | **ambiguous** |
| 4242 | 42 | 8.0e-7 | converged |
| **57005** | **4** | **1.1e-1** | **non-converged (early-bail)** |
| 48879 | 22 | 6.5e-7 | converged |
| 51966 | 18 | 9.3e-7 | converged |

All 20 TY seeds are converged (residual ~5–9e-7 across the ensemble).
14 Schoen seeds converge; 5 fail (residual ≥ 1e-2); 1 is ambiguous
(seed 2024, residual 4.5e-3). The ambiguous seed is excluded from the
canonical set under the residual-based classifier.

The P5.5f shift relative to P5.5d (5.963 → 6.481) comes from the
regression / near-singular-T(G) guards added to the Donaldson
iteration (`schoen_metric.rs`, `ty_metric.rs`). Five Schoen seeds now
exit early via the regression guard at iter ≤5, returning σ from a
near-FS-identity snapshot rather than the diverging iterate; this
tightens SE_Schoen from 0.892 → 0.787 and *lowers* the canonical Schoen
mean slightly (5.587 → 5.368). The TY pipeline is unaffected (no
TY seed triggers the guards in this configuration).

### Round-2 hostile-review re-analysis (Concern B): balanced-only subset — SUPERSEDED BY ROUND-3

(The round-2 iter-based filter is retained in the binary's output JSON
under `discrimination_balanced_only` for back-compat. Round-3
residual-based reclassification (above) is the current canonical
methodology.)

The 6.481σ headline includes Schoen seeds that exited the Donaldson
loop early (iters < 10) via the P5.5f regression guard. Six Schoen
seeds bail this way at the canonical (n_pts=25 000, k=3, max_iter=50)
configuration: seeds **2 (iters=3), 5 (iters=2), 271 (iters=4),
1000 (iters=5), 2024 (iters=9), 57005 (iters=4)**. All twenty TY
seeds converge past iter 10.

Removing the early-bail Schoen seeds gives the sharper "what
TY-vs-Schoen looks like when both candidates are converged"
comparator:

| subset | TY n | Schoen n | ⟨σ_TY⟩ ± SE | ⟨σ_Schoen⟩ ± SE | n-σ |
|---|---:|---:|---:|---:|---:|
| **canonical (full 20+20)** | 20 | 20 | 0.268309 ± 8.979e-4 | 5.368323 ± 7.868e-1 | **6.481** |
| balanced-only (iters ≥10) | 20 | 14 | 0.268309 ± 8.979e-4 | 4.442612 ± 7.392e-1 | **5.647** |

Both rows clear the 5σ project goal. The canonical row is the
production headline; the balanced-only row appears in the printed
table for any future reviewer who wants to see the converged-only
comparator.

**Important contrary finding to round-2 brief.** The brief assumed
the early-bail seeds returned σ values within a few percent of
`sigma_fs_identity` (i.e. the regression guard restored a near-FS
snapshot). The actual run shows the opposite — every early-bail seed
exits with `σ_final` *above* its FS-identity σ, by factors of 1.2× to
5.9×:

| seed | iters | σ_final | σ_fs_identity | (σ_final − σ_fs)/σ_fs |
|---|---:|---:|---:|---:|
| 2 | 3 | 5.27 | 2.06 | +1.56 |
| 5 | 2 | 6.78 | 2.78 | +1.44 |
| 271 | 4 | 4.67 | 2.07 | +1.26 |
| 1000 | 5 | 15.53 | 2.25 | **+5.91** |
| 2024 | 9 | 3.66 | 1.66 | +1.21 |
| 57005 | 4 | 9.26 | 2.02 | +3.59 |

So these aren't "near-FS-identity snapshots" — they are seeds where
Donaldson started diverging early and the regression guard restored
the iter-min snapshot, which already had σ above the FS floor. The
Schoen mean is therefore biased *upward* by these seeds (not toward
the FS floor as conjectured); removing them makes the discrimination
*weaker* (6.481σ → 5.647σ) rather than stronger. Both numbers
clear the 5σ goal, so the conclusion stands either way, but the
mechanism is different from what the brief expected.

The early-bail seeds and their σ_final-vs-sigma_fs_identity
relative differences are dumped into the JSON as
`early_bail_seeds[]` and printed in the binary's stdout table at
the end of each run.

The pre- vs post-fix shift comes from the upper-index Donaldson
inversion fix landed in P5.5b (quintic) and propagated to route34
in P5.5d (`src/route34/ty_metric.rs`,
`src/route34/schoen_metric.rs`). The corrected iteration finds the
true Donaldson-balanced fixed point on each candidate:

- **σ_TY drops 4×** (1.01 → 0.27) because TY/Z3's larger invariant
  basis (n_basis=87 at k=3) accommodates a much-better-balanced
  metric. The pre-fix iteration was converging to a non-balance
  fixed point that retained more L²-variance than the true balanced
  metric.
- **σ_Schoen rises** (3.04 → 5.59) because Schoen's small invariant
  basis (n_basis=27 at (3,3,1)) cannot accommodate as much balance;
  the heavy-tail seeds spread further at the corrected fixed point.

Both shifts are consistent with the same fix being applied to the
same iteration in two different basis-size regimes. The
discrimination signal is preserved (and physically interpretable:
the larger candidate basis genuinely permits closer-to-Ricci-flat
metrics).

The pre-fix P5.10 outputs are preserved at
`output/p5_10_ty_schoen_5sigma_pre_fix.json`. The post-fix run
overwrites `output/p5_10_ty_schoen_5sigma.json`. The σ-eval logic
itself was not modified; only the upstream Donaldson iteration was
corrected.

### History block — earlier headlines (before P5.5f, retained for traceability)

The 8.761σ ("pre-fix") and 5.963σ ("P5.5d post-fix") rows below are
**superseded** by the 6.481σ canonical headline above. They are kept
only so the per-step trajectory of the discrimination programme is
auditable.

#### Pre-fix headline (before the upper-index Donaldson inversion fix)


| k | n_pts | ⟨σ_TY⟩ ± SE | ⟨σ_Schoen⟩ ± SE | n-σ |
|---:|---:|---:|---:|---:|
| 3 (P5.7) | 10 000 | 1.0180 ± 1.82e-3 | 3.339 ± 4.78e-1 | **4.854** |
| 3 (**P5.10 pre-fix**) | **25 000** | **1.014688 ± 5.225e-3** | **3.036530 ± 2.307e-1** | **8.761** |

**Δσ = −2.0218, SE_combined = 0.2308, n-σ = 8.761.** P5.10 (pre-fix)
clears the project's 5σ goal by ~75%. The σ-eval logic was *not*
modified vs P5.7; only the sampling budget changed (Path A — n_pts
10 000 → 25 000 at k=3).

Note: at n_pts=25 000 the TY mean σ tightens from 1.0180 → 1.0147 and SE
*grows* slightly (1.8e-3 → 5.2e-3) because the larger sample reveals a
small per-seed variance previously masked by sampler noise; the Schoen SE
falls by 2.07× (4.78e-1 → 2.31e-1) — exactly the heavy-tail tightening
that Path A targeted. Δσ is essentially unchanged (−2.32 → −2.02),
confirming the discrimination is real signal, not artifact.

## Why this report exists

P5.7 reached **n-σ = 4.854 at k=3**, ~3% short of the project's 5σ
discrimination goal. The σ-eval logic itself was correct — the bottleneck
was statistical: the Schoen σ distribution at low k has heavy tails
because the invariant section basis is small (n_basis=12 at k=2,
n_basis=27 at k=3), leaving Donaldson balance with too little freedom on
some seeds.

Path A was the cheapest and highest-expected-payoff route past 5σ:

> SE scales as ~1/√n_pts. Bumping n_pts from 10 000 → 25 000 should
> shrink SE_Schoen by √2.5 ≈ 1.58×, dropping it from 0.478 → ~0.30, and
> push n-σ from 4.85 → ~7.7σ.

The actual result (n-σ = 8.761) slightly *exceeds* that estimate because
SE_Schoen tightened by 2.07× rather than 1.58× — at n_pts=25 000, the
Donaldson iteration converges on more seeds before hitting `max_iter=25`
(only seed=2 hit the cap, and seed=10 capped at 25 with σ=4.05). The
heavy-tail outlier at seed=666 (σ_Schoen=11.4 at n_pts=10 000) drops to
σ_Schoen=5.69 at n_pts=25 000.

## Pipeline parameters (P5.10)

| Parameter | Value | Diff from P5.7 |
|---|---|---|
| Candidates | TY/Z3, Schoen/Z3×Z3 | unchanged |
| k values | 3 | dropped k=2 (focus on the regime closest to 5σ) |
| Seeds | `[42, 100, 12345, 7, 99, 1, 2, 3, 4, 5, 137, 271, 314, 666, 1000, 2024, 4242, 0xDEAD, 0xBEEF, 0xCAFE]` (n=20) | unchanged |
| **n_pts** | **25 000** | **2.5× increase from 10 000** |
| Donaldson max_iter | 25 | unchanged |
| Donaldson tol | 1.0e-3 | unchanged |
| Schoen tuple at k=3 | (d_x=3, d_y=3, d_t=1) | unchanged |
| Adam refine | none | unchanged (fair across candidates) |
| Bootstrap | n_resamples=1000, seed=12345, ci_level=0.95 | unchanged |
| Wall clock | **114.0 s** on one workstation | (P5.7: 45.4 s) |

σ-eval logic is byte-for-byte identical to P5.7 — the only knob touched
is the sample budget.

## Per-seed σ summary (P5.10, k=3, n_pts=25 000)

### TY/Z3
| seed | σ_TY | iters |
|---|---:|---:|
| 42 | 1.020350 | 7 |
| 100 | 1.041661 | 7 |
| 12345 | 1.055199 | 7 |
| 7 | 1.029435 | 7 |
| 99 | 0.991076 | 7 |
| 1 | 0.990830 | 8 |
| 2 | 0.977664 | 7 |
| 3 | 0.975403 | 7 |
| 4 | 1.026825 | 7 |
| 5 | 1.053809 | 7 |
| 137 | 1.016378 | 7 |
| 271 | 1.002055 | 8 |
| 314 | 1.010311 | 8 |
| 666 | 1.046797 | 8 |
| 1000 | 1.013978 | 7 |
| 2024 | 1.012844 | 7 |
| 4242 | 0.999446 | 7 |
| 0xDEAD | 0.991895 | 8 |
| 0xBEEF | 1.014749 | 7 |
| 0xCAFE | 1.023045 | 8 |

⟨σ_TY⟩ = 1.014688, std = 2.337e-2, SE = 5.225e-3, pct95 = [1.005, 1.025], BCa95 = [1.005, 1.025].

### Schoen/Z3×Z3 (d_x=3, d_y=3, d_t=1)
| seed | σ_Schoen | iters |
|---|---:|---:|
| 42 | 2.426856 | 11 |
| 100 | 2.156840 | 12 |
| 12345 | 2.134359 | 11 |
| 7 | 2.509075 | 15 |
| 99 | 2.842851 | 17 |
| 1 | 2.787464 | 14 |
| 2 | 2.829908 | 23 |
| 3 | 2.514879 | 14 |
| 4 | 2.339442 | 13 |
| 5 | 4.048856 | 25 |
| 137 | 2.059253 | 11 |
| 271 | 3.058387 | 20 |
| 314 | 2.758330 | 12 |
| 666 | 5.694983 | 14 |
| 1000 | 3.011828 | 18 |
| 2024 | 2.359213 | 14 |
| 4242 | 4.736887 | 12 |
| 0xDEAD | 3.074464 | 19 |
| 0xBEEF | 2.324082 | 11 |
| 0xCAFE | 5.062652 | 13 |

⟨σ_Schoen⟩ = 3.036530, std = 1.032, SE = 2.307e-1, pct95 = [2.648, 3.496], BCa95 = [2.677, 3.568].

Heavy tail still present (seeds 666, 4242, 0xCAFE in [4.7, 5.7]) but the
mean and SE have both tightened substantially compared to P5.7 because
the bulk of the distribution sits more compactly in [2.0, 3.1] at the
larger n_pts.

## n-σ discrimination

### Canonical (post-P5.5f, current)

```
| k |    <σ_TY> |     SE_TY | <σ_Schoen> | SE_Schoen |         Δσ |  SE_comb |    n-σ |
|---|-----------|-----------|------------|-----------|------------|----------|--------|
| 3 |  0.268309 | 8.979e-4  |   5.368323 | 7.868e-1  |  -5.100014 | 7.868e-1 |  6.481 |
```

**max n-σ = 6.481 at k=3.**

### Pre-fix history (before P5.5f, retained for traceability)

```
| k |    <σ_TY> |     SE_TY | <σ_Schoen> | SE_Schoen |         Δσ |  SE_comb |    n-σ |
|---|-----------|-----------|------------|-----------|------------|----------|--------|
| 3 |  1.014688 | 5.225e-3  |   3.036530 | 2.307e-1  |  -2.021843 | 2.308e-1 |  8.761 |
```

Sign of Δσ unchanged from P5.7 (σ_TY < σ_Schoen) across all three
revisions, confirming the same hierarchy that made physical sense in
the P5.7 report: the TY/Z3 invariant Gröbner-reduced basis (n_basis=87
at k=3) gives Donaldson far more freedom than Schoen's (n_basis=27 at
k=3), so the TY metric balances closer to Ricci-flat.

## Verdict (round-4)

**5σ goal is NOT robustly cleared on σ alone at this configuration.**
The honest reading of the four-tier table:

* The **strictest defensible filter** (Tier 0: residual < tol AND
  iters < cap) gives n-σ = **4.811** at the point estimate — *below*
  5σ. The 95% CI is [3.805, 9.989], firmly straddling 5σ.
* The **canonical filter** (Tier 2: residual < 1e-3, the round-3
  classifier) gives n-σ = 5.647 at the point estimate but 95% CI
  lower bound = 4.426σ — *below* 5σ at the floor.
* The **Tukey-trimmed canonical** (Tier 3) gives n-σ = 7.783 at the
  point estimate, but with the round-4 re-trim-per-resample bootstrap
  fix the CI lower bound is 4.933σ — also *below* 5σ at the floor.
  The round-3 headline reported [6.185, 11.553]; the previous CI was
  artifactually narrow because the Tukey trim was applied once before
  bootstrap rather than re-applied to each resample.
* The **conservative full-ensemble** (Tier 1) gives n-σ = 6.482 at
  the point estimate AND CI lower bound 5.302σ — clears 5σ at both
  point and CI floor. This is the only tier where 5σ is robustly met,
  but it includes 5–6 guard-restored mid-descent Schoen snapshots
  whose σ values are not Donaldson-balanced. Reporting Tier 1 as the
  headline is defensible only if those snapshots are accepted as
  fair σ samples.

The harness *does* genuinely point at TY/Z3 over Schoen/Z3×Z3 on σ
alone — TY mean σ ≈ 0.27 is 16× tighter than Schoen mean σ ≈ 4.4–6.5
across all four tiers, signs are consistent, and the joint-likelihood
channels (η, hidden-bundle, Yukawa) are still available as
*strengthening* evidence (see P5.x onward). What round-4 establishes
is that the **σ-only**, **strict-converged**, **B=10 000 with
re-trim bootstrap** combination falls short of 5σ at this budget.

The path forward is to bump the budget so even Tier 0 clears 5σ:
1. **n_pts → 40 000 at k=3** — under the same Schoen heavy-tail
   tightening pattern observed P5.7 → P5.10 (SE shrinks ~2.07× per
   2.5× n_pts), Tier 0 SE_Schoen should drop from 0.85 to ~0.41 and
   n-σ to ~9.9σ at the point estimate; 95% CI floor expected ≥6σ.
2. **k=4 with publication-default Schoen tuple (4,4,2)** — n_basis
   grows to 67 (Schoen) and ~189 (TY); the heavy tail collapses
   entirely. Per-run cost ~5–10× larger but expected to clear 5σ at
   every tier including CI floor.

Both are budget knobs, not algorithmic changes. The P5.7 → P5.10
trajectory shows the σ channel responds to budget; round-4 simply
honestly reports that the project's stated 5σ goal is **not** met at
the current (n_pts=25 000, k=3, max_iter=50, tol=1e-6, B=10 000)
configuration under the strictest defensible filter.

This means:
- The harness genuinely separates the two CY3 candidates on σ alone
  (point estimate 4.81σ–7.78σ across tiers, all signs correct), but
  the project's 5σ goal is not robustly met at the 95% CI floor under
  any tier stricter than "full ensemble keep guard snapshots".
- The result is reproducible: identical seed list as P5.3 / P5.7,
  bootstrap seed = 12345 (with sensitivity check across {999, 31415}
  showing max_jitter = 0.021σ at B=10 000), deterministic on x86_64
  Linux/Windows.
- The η, hidden-bundle, Yukawa, and joint-Bayes channels (P5.x
  onward) become *necessary* evidence rather than strengthening
  evidence at this budget — the σ channel alone does not reach 5σ
  under the strictest filter.

## Files modified this wave (round-4 hostile-review)

- `src/bin/p5_10_ty_schoen_5sigma.rs` — Defect-B fix (re-trim
  bootstrap on Tier 3); Defect-E fix (new Tier 0 strict-converged:
  residual < tol AND iters < cap); Defect-C fix (default
  `boot_resamples` 1000 → 10 000); CLI flag
  `--report-boot-seed-jitter` (default true) emitting CI-lower-bound
  jitter across {primary, 999, 31415}; new `BootSeedJitter` struct;
  `EnsembleReport.discrimination_strict_converged` and
  `EnsembleReport.boot_seed_jitter` JSON fields; verdict-selection
  logic now picks the strictest tier that clears 5σ at point AND CI
  floor (currently: Tier 1).
- `references/p5_10_5sigma_target.md` — round-4 four-tier headline,
  honest 5σ verdict (NOT robustly met under strict filter),
  re-trim-bootstrap caveat, B=10 000 jitter sensitivity table.

## Files added in earlier waves

- `src/bin/p5_10_ty_schoen_5sigma.rs` — dedicated 5σ-target binary,
  defaults to Path A (n_pts=25 000, k=3, 20 seeds). All knobs
  CLI-overridable.
- `src/route34/tests/test_p5_10_ty_schoen_5sigma.rs` — `#[ignore]`'d
  long-run regression test. Asserts n-σ > 5.0; on failure prints actual
  n-σ and the additional n_pts budget needed.
- `src/route34/tests/mod.rs` — wires the new test module.
- `Cargo.toml` — registers the new binary.
- `output/p5_10_ty_schoen_5sigma.json` — full ensemble dump (per-seed
  σ records, bootstrap CIs, discrimination row, 5σ verdict).
- `output/p5_10_path_a_n25000_k3.json` — same content under
  Path-A-specific name.
- `output/p5_10_path_a.log` — stdout/stderr of the run.
- `references/p5_10_5sigma_target.md` — this report.

## Reproducibility

```bash
cd book/scripts/cy3_substrate_discrimination/rust_solver

# Full Path-A ensemble (114 s)
cargo run --release --features gpu --bin p5_10_ty_schoen_5sigma

# Custom (e.g. wider scan or aggressive n_pts)
cargo run --release --features gpu --bin p5_10_ty_schoen_5sigma -- \
    --n-pts 40000 --ks 3 --output output/p5_10_aggressive.json

# Ignored regression test (asserts n-σ > 5.0)
cargo test --release --features gpu --lib test_p5_10 \
    -- --ignored --nocapture
```

All seeds and the bootstrap seed (12345) are deterministic on x86_64
Linux/Windows; numbers above are reproducible to floating-point rounding.

## Forward look

The σ-only 5σ goal is now closed. The follow-on items in the
discrimination programme (η, hidden-bundle, Yukawa, joint Bayes factor)
become *strengthening* evidence rather than fallback channels. The
joint-likelihood harness (`route34::bayes_factor`) can now be run with
σ as a high-confidence input rather than a marginal one.

If a tighter result is later required (e.g. for publication confidence
limits) the next-cheapest budget bumps are:
1. **n_pts → 40 000 at k=3** — expected n-σ ≈ 11.0; runtime ~3 min.
2. **k=4 with publication-default Schoen tuple (4,4,2)** — n_basis grows
   to 67 (Schoen) and ~189 (TY); per-run cost ~5–10× larger but the
   Schoen heavy tail collapses entirely. Expected n-σ ≥ 15.
3. **Adam refinement on both candidates** — cheapest if the wiring is
   ported from `QuinticSolver::sigma_refine_analytic` to TY and Schoen,
   but that's an engineering task, not a budget knob.

None of these are required to claim the 5σ result.
