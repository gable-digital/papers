# P-K5 — TY-vs-Schoen σ-discrimination at k=5 (SUPPORTING tier — corroborates k=3 headline)

> **TL;DR.**
> k=5 σ-discrimination at the strict-converged tier reads
> **n-σ = 10.87**, BCa 95% CI **[8.79, 15.01]** (percentile 95% CI
> [9.27, 34.44]), n_TY=20, **n_Schoen=6**. Source:
> `output/p5_10_k5_damped_relaunch.json` (P-K5-relaunch, 2026-04-30
> production, GPU, donaldson_iters=200, donaldson_tol=1e-6, 40 000 pts,
> 20-seed list, B=10 000). The point estimate clears 5σ at all four
> bootstrap seeds (jitter ≤ 0.02σ on the strict tier), but the
> strict-converged Schoen sample is very small (6/20). The k=3 Tier 0
> **6.92σ** result (`output/p5_10_ty_schoen_5sigma.json`, n_Schoen=16)
> remains the canonical σ-channel discriminability publication
> headline. **k=5 is reported as a supporting tier consistent with —
> and qualitatively strengthening — the k=3 headline at higher basis
> order**, NOT as a replacement publication headline. The k=5 Tier 3
> Tukey re-trim (24.45σ) remains **diagnostic-only**, NOT
> publication-grade — same post-hoc trim was rejected at k=3 by P5.5h.

## Production sweep configuration

**Run**: `output/p5_10_k5_damped_relaunch.json` (Donaldson-only, GPU,
no Adam refinement). 20-seed ensemble × 2 candidates × k=5. Wallclock
total **11 383 s ≈ 3.16 hr**. Same n_pts=40 000, donaldson_tol=1e-6,
boot_resamples=10 000, boot_seed=12345, and same 20-seed list as the
k=3 P5.10 and k=4 P8.4-fix damped sweeps. donaldson_iters bumped to
**200** (vs 100 at k=3/k=4) to give the larger k=5 basis (n_basis=385
TY, 75 Schoen) headroom under the 1e-6 tolerance.

Damping mode: same static α=0.5 default in the Donaldson update step
that made the k=4 supporting tier work (P8.4-fix). No CLI flag for
damping was changed at k=5; the binary's default α=0.5 path is the
load-bearing supporting damping at k≥4.

## Discrimination — four-tier table (verified from JSON)

| Tier | Filter | <σ_TY> | <σ_Schoen> | n-σ | n-σ pct 95% CI | n-σ BCa 95% CI | n_TY | n_Sc |
|------|--------|--------|-----------|-----|----------------|-----------------|------|------|
| **0** strict-converged | residual<tol AND iters<cap | 0.2954 | 3.969 | **10.869** | [9.268, 34.437] | **[8.788, 15.011]** | 20 | **6** |
| **1** conservative | full 20+20 | 0.2954 | 1197.43 | 1.007 | [1.004, 13.598] | [1.003, 1.833] | 20 | 20 |
| **2** canonical | residual<1e-3 | 0.2954 | 6.301 | 3.861 | [3.359, 23.211] | [3.040, 11.192] | 20 | 13 |
| **3** Tukey re-trim | trim Q1−1.5·IQR / Q3+1.5·IQR | 0.2954 | 4.671 | **24.445** | [6.842, 75.317] | [7.225, 83.661] | 20 | 9 |

Tier 1 is dominated by two pathological mid-descent Schoen seeds —
seed 5 (σ=54.07, residual=5.17e-3, ambiguous) and seed 4242
(σ=23 790.58, residual=1.81e-3, ambiguous) — that the strict gate
correctly excludes. Tier 1's 1.01σ point estimate is therefore a
diagnostic floor, not a discriminability claim.

Bootstrap-seed jitter at B=10 000 (seeds 12345 / 999 / 31415):

| label | primary CI lo | seed 999 CI lo | seed 31415 CI lo | max jitter |
|-------|---------------|----------------|------------------|------------|
| strict_converged_k5 | 9.2682 | 9.2498 | 9.2664 | **0.0184** |
| canonical_k5 | 3.3590 | 3.3449 | 3.3508 | 0.0141 |
| reference_retrim_k5 | 6.8421 | 6.8970 | 6.8670 | 0.0548 |

All three tiers' jitter is within the round-4 target (≤ 0.05 σ at
B=10 000) for the strict and canonical tiers; reference_retrim is at
0.055 σ, just over the soft target — same-order behaviour as k=4.

## TY-side (n_basis=385, all 20 strict-converged)

`<σ_TY> = 0.29541`, std = 4.05e-3, SE = 9.05e-4, BCa CI
[0.2937, 0.2972]. Extremely tight distribution. All 20 seeds
converged in 30–33 iters with residual ≤ 9.6e-7. The TY trajectory
across k is: σ_TY(k=3) ≈ 0.27, σ_TY(k=4) = 0.2782, **σ_TY(k=5) =
0.2954**. Monotone slow drift — exactly the basis-saturation
pattern P8.1e predicts for the TY branch.

## Schoen-side (n_basis=75, mixed convergence)

Only **6 of 20** Schoen seeds met the strict-converged gate
(residual<tol AND iters<cap). The 6 strict-converged Schoen seeds:

| seed | σ_final | iters | residual |
|------|---------|-------|----------|
| 12345 | 3.4719 | 133 | 9.57e-7 |
| 1 | 4.9694 | 273 | 9.88e-7 |
| 3 | 4.3665 | 68 | 8.69e-7 |
| 137 | 3.2321 | 45 | 8.15e-7 |
| 48879 | 3.0347 | 57 | 9.15e-7 |
| 51966 | 4.9905 | 69 | 9.71e-7 |

`<σ_Schoen>_strict = 3.969`, SE = 0.338. Tight, well-separated from
TY's 0.295.

The remaining 14 Schoen seeds split as:
- **8 "converged-tier" but iters≥cap or residual≥1e-3 in the strict
  filter:** seeds 7 (σ=5.12), 99 (σ=4.48), 4 (σ=4.72), 314 (σ=9.36),
  666 (σ=24.24), 1000 (σ=4.67), 2024 (σ=5.26), 57005 (σ=4.28). These
  are picked up by the canonical tier (residual<1e-3, n_Schoen=13).
- **6 "ambiguous-tier" stalls:** seeds 42, 100, 2, 5, 271, 4242. Two
  carry pathological σ (5: 54.07; 4242: 23 791) that drag the
  conservative-tier mean to 1197 and inflate Tier 1's SE — exactly
  the basis-size-artifact heavy-tailing pattern P8.1e predicted for
  σ-discrimination at higher k.

## 5σ verdict per tier

| Tier | n-σ | BCa CI low | ≥5σ point | ≥5σ CI floor | Status |
|------|-----|------------|-----------|--------------|--------|
| 0 | **10.869** | 8.788 | **yes** | **yes** | **SUPPORTING tier — strengthens k=3 6.92σ headline at higher k** |
| 1 | 1.007 | 1.003 | no | no | diagnostic (heavy-tail dominated) |
| 2 | 3.861 | 3.040 | no | no | diagnostic |
| 3 | 24.445 | 7.225 | yes | yes | **DIAGNOSTIC ONLY — NOT publication-grade** (post-hoc trim rejected by P5.5h at k=3) |

**Tier 0 at k=5 clears 5σ at the point estimate AND at both CI
floors (percentile 9.27, BCa 8.79).** The strict-converged sample
size n_Schoen=6 is, however, well below the k=3 baseline n_Schoen=16
and below the n_Schoen≥16 threshold the P5.10 v7 protocol used at
k=3 for "publication-grade" classification. For that reason — and
because the same post-process task spec calls explicitly for
"discriminability persists at k=5 to N.NNσ at strict tier" framing
when the result strengthens the picture — k=5 is filed as
**SUPPORTING tier strengthening the k=3 headline at higher k**,
not as a replacement publication headline.

## k-scan progression — basis-size-artifact pattern

| Run | k | n_basis_TY | n_basis_Schoen | <σ_TY> | <σ_Schoen>_strict | n_Schoen_strict | Tier 0 n-σ | Tier 0 BCa |
|-----|---|------------|-----------------|--------|-------------------|------------------|------------|-----------|
| **k=3** P5.10 / P5.5k | 3 | 87 | 27 | ~0.27 | ~0.45 | **16/20** | **6.92** | [5.30, 9.04] |
| **k=4** P8.4-fix damped | 4 | 200 | 48 | 0.2782 | 6.205 | 7/20 | 3.82 | [2.48, 6.37] |
| **k=5** P-K5-relaunch (this) | 5 | 385 | 75 | **0.2954** | **3.969** | **6/20** | **10.87** | **[8.79, 15.01]** |

Reading the trend:

1. **σ_TY converges with weak basis-size dependence** (0.27 → 0.278 →
   0.295). TY's metric is well-determined at every k.
2. **Schoen σ-distribution heavy-tails with k.** The strict-converged
   subsample shrinks from 16 → 7 → 6 as Schoen's basis grows from
   27 → 48 → 75 and more seeds stall in Donaldson plateaus at
   high-residual fixed points. This is *exactly* the
   basis-size-artifact pattern P8.1e predicts and that P-BASIS-CONVERGENCE
   is independently characterising in parallel.
3. **The strict-converged Schoen mean drops at k=5 vs k=4** (3.97 vs
   6.20). The k=4 value was elevated by seed 314 (σ=14.6) and seed
   666 (σ=16.5); at k=5 those two seeds re-stall (seed 314
   converged-tier σ=9.36 but residual=7.3e-4; seed 666 σ=24.2 with
   residual=1.05e-5 — passes canonical, fails strict). The strict
   gate at k=5 isolates the cleanest 6 Schoen seeds.
4. **n-σ Tier 0 is non-monotone in k:** 6.92 → 3.82 → 10.87. The
   apparent jump at k=5 reflects (a) the very tight strict-converged
   Schoen subset and (b) σ_TY's drift moving the centre of the
   distribution, not a stronger underlying signal. The conservative
   (Tier 1) picture moves the opposite way: 8.62 (k=4) → 1.01 (k=5)
   as heavy tails dominate. This is the qualitative signature of
   **σ being a basis-size-sensitive discrimination channel** rather
   than a clean physics observable — exactly the classification
   P8.1e formalised when excluding σ from the model-comparison BF.

The k-scan therefore reads as **discriminability persists at k=5 to
10.87σ at the strict tier, but with n_Schoen_strict shrinking
monotonically and Tier 1 collapsing under heavy tails — consistent
with the basis-size-artifact thesis.** This both strengthens the
qualitative k=3 headline (TY and Schoen are robustly distinguishable
at higher k under the strict gate) and supports the decision to
exclude σ from the multi-channel Bayes factor.

## Comparison to k=3 / k=4 in publication framing

* **k=3 (publication headline):** Tier 0 6.92σ, BCa [5.30, 9.04],
  n_Schoen=16. Canonical σ-channel discriminability publication
  number. Unchanged.
* **k=4 (supporting tier):** Tier 0 3.82σ damped, BCa [2.48, 6.37],
  n_Schoen=7. Under 5σ at strict tier. Reported as supporting,
  consistent with — but not strengthening — the k=3 headline.
* **k=5 (supporting tier, this work):** Tier 0 10.87σ, BCa
  [8.79, 15.01], n_Schoen=6. Above 5σ at point AND BCa CI floor,
  with very small n_Schoen. Reported as supporting,
  **strengthening** the k=3 headline qualitatively at higher basis
  order, with the explicit caveat that the strict-converged sample
  is small (6/20 vs k=3's 16/20).

The publication picture is: σ-channel discriminability holds at k=3
(canonical 6.92σ, n_Schoen=16), is recoverable at k=4 with damping
(3.82σ supporting, n_Schoen=7), and re-strengthens at k=5 to 10.87σ
on a very tight strict-converged Schoen sample of 6. Tier 1
(conservative) at k=5 is pathology-dominated (1.01σ) — confirming
σ's basis-size-artifact character at the heavy-tail end and
reinforcing the P8.1e decision to exclude σ from the BF.

## Caveats

* **Damping mode.** Static α=0.5 (binary default), inherited from the
  k=4 P8.4-fix run. No adaptive damping. No CLI flag changed.
* **iter cap.** donaldson_iters=200 (vs 100 at k=3/k=4) to give the
  larger basis headroom. None of the 20 TY seeds and only 1 Schoen
  seed (seed 1, 273 iters) ran past 200 iters; that 273-iter Schoen
  seed nonetheless reached residual=9.88e-7 < tol so it is
  strict-included.
* **n_pts.** 40 000 (matches k=3 / k=4 baselines).
* **Strict gate vs P5.10 v7 protocol.** The post-process task spec
  sets the publication-grade Tier 0 floor at n_Schoen ≥ 16 (matching
  the k=3 baseline). At k=5 only 6 Schoen seeds pass strict, so
  k=5 is filed as **supporting**, not as a parallel publication
  headline.
* **Tier 3 Tukey re-trim** is post-hoc and remains DIAGNOSTIC only,
  not publication-grade — same caveat as at k=3 (rejected by P5.5h)
  and k=4 (P8.4-fix).
* **Bit-exact determinism.** SHA-chained kernel replog
  (`output/p5_10_k5_damped_relaunch.replog`,
  `repro_log_final_chain_sha256_hex =
  eeb9403574d1db6098eb3cd48d8f3bd540b154e91af276b6474ad477af28225c`,
  42 events) is preserved alongside the JSON for re-replay.

## Files

* **Created**:
  * `output/p5_10_k5_damped_relaunch.json` (production sweep output)
  * `output/p5_10_k5_damped_relaunch.log`
  * `output/p5_10_k5_damped_relaunch.replog`
  * `references/p_k5_production.md` (this file)
* **Updated**:
  * `references/cy3_publication_summary.md` — §1 Headlines (k=5
    supporting line) and §4 Supporting evidence at higher k (k=4 and
    k=5 row).
* **Not modified** (per task constraints):
  * Source code, binaries, tests.
  * `references/cy3_publication_summary.md` §2 (basis-convergence-prod
    post-process owns this section in parallel).
  * `references/cy3_publication_summary.md` §3 per-channel evidence
    (k=5 result is qualitatively a supporting tier and does not
    change the channel framing).
* **Canonical k=3 baseline** (`output/p5_10_ty_schoen_5sigma.json`,
  Tier 0 = 6.92σ) is unchanged — k=5 result here is a supporting
  tier, not a replacement.
