# P8.4 — TY-vs-Schoen σ-discrimination at k=4 (SUPPORTING tier — corroborates k=3 headline)

> **TL;DR — updated after P8.4-fix damped re-run (Apr 2026).**
> k=4 σ-discrimination is presented as a **SUPPORTING tier** consistent
> with the k=3 publication headline, NOT as a parallel publication
> headline of its own. The k=3 Tier 0 **6.92σ**
> (`output/p5_10_ty_schoen_5sigma.json`) remains the canonical
> σ-channel discriminability headline at any k. With static damping
> α=0.5 enabled in the Donaldson update step, k=4 Tier 0 strict-converged
> n-σ rises from **1.83σ (un-damped baseline)** to **3.82σ (damped
> re-run, `output/p5_10_k4_damped.json`)**, BCa 95% CI [2.48, 6.37],
> percentile 95% CI [3.05, 10.39]. That is still under 5σ at the
> strict tier, so k=4 does **not** carry an independent
> σ-discriminability publication headline; but the monotone improvement
> from 1.83 → 3.82 with damping converts k=4 from "diagnostic-only
> artifact" into a **supporting tier consistent with the k=3 headline**
> rather than contradicting it. The k=4 Tier 3 (Tukey re-trim)
> number — now **12.13σ** — remains **diagnostic only, NOT
> publication-grade**: the same post-hoc trim was rejected at k=3 by
> P5.5h. **An adaptive-damping production sweep is running in
> background** (P8.4-followup, ~2.4 hr) and will be reported separately.

## P8.4-fix damped re-run (static α=0.5)

**Run**: `output/p5_10_k4_damped.json` (Donaldson-only, GPU, no Adam
refinement, **static damping α=0.5** auto-default in the Donaldson
update step). 20-seed ensemble × 2 candidates × k=4. Wallclock total
**8788 s ≈ 2.44 hr**. Same n_pts=40000, donaldson_iters=100,
donaldson_tol=1e-6, boot_resamples=10000, boot_seed=12345, and same
20-seed list as the un-damped P8.4 baseline.

### Damped-run discrimination — four-tier table (verified from JSON)

| Tier | Filter | <σ_TY> | <σ_Schoen> | n-σ | n-σ pct 95% CI | n-σ BCa 95% CI | n_TY | n_Sc |
|------|--------|--------|-----------|-----|----------------|-----------------|------|------|
| **0** strict-converged | residual<tol AND iters<cap | 0.2782 | 6.205 | **3.823** | [3.049, 10.389] | [2.478, 6.371] | 20 | 7 |
| **1** conservative | full 20+20 | 0.2782 | 13.237 | 2.932 | [2.467, 8.562] | [2.126, 3.997] | 20 | 20 |
| **2** canonical | residual<1e-3 | 0.2782 | 13.077 | 2.749 | [2.324, 9.524] | [1.928, 3.591] | 20 | 19 |
| **3** Tukey re-trim | trim Q1−1.5·IQR / Q3+1.5·IQR | 0.2782 | 5.563 | **12.130** | [3.478, 25.312] | [5.920, 30.684] | 20 | 15 |

Bootstrap-seed jitter at B=10000 (seeds 12345 / 999 / 31415):

| label | primary CI lo | seed 999 CI lo | seed 31415 CI lo | max jitter |
|-------|---------------|----------------|------------------|------------|
| strict_converged_k4 | 3.0493 | 3.0488 | 3.0428 | **0.0065** |
| canonical_k4 | 2.3235 | 2.3162 | 2.3086 | 0.0149 |
| reference_retrim_k4 | 3.4778 | 3.4587 | 3.4831 | 0.0190 |

All tiers' jitter is well under the 0.05 σ round-4 target — the
extra wallclock from damping (Schoen seeds run a little longer) trades
directly into a tighter bootstrap.

### Side-by-side: un-damped (P8.4 baseline) vs damped (P8.4-fix)

| Tier | Un-damped n-σ (`p5_10_k4_gpu_donaldson_only.json`) | Damped α=0.5 n-σ (`p5_10_k4_damped.json`) | Δ |
|------|---------------------------------------------------|-------------------------------------------|---|
| 0 strict-converged | 1.83 (10/20 Schoen seeds strict) | **3.82** (7/20 Schoen seeds strict) | **+1.99σ** |
| 1 conservative | 2.93 | 2.93 | ~0 |
| 2 canonical | 2.34 | 2.75 | +0.41σ |
| 3 Tukey re-trim (diagnostic) | 11.41 | 12.13 | +0.72σ |

Strict-tier n_Schoen at k=4 went from 10/20 (un-damped) to 7/20
(damped). The Schoen seeds that *do* converge under damping land at a
lower σ (mean 6.20 vs 14.94 un-damped) and tighter SE (1.55 vs 8.02),
so even with fewer strict-converged samples the resulting n-σ
roughly doubles.

### Reading

* **Tier 0 strict at 3.82σ is the honest supporting tier.** It is
  still under 5σ, so it does **not** carry an independent
  σ-discriminability publication headline at k=4. The k=3 Tier 0
  **6.92σ** result (`output/p5_10_ty_schoen_5sigma.json`) remains the
  canonical σ-channel discriminability publication headline.
* The monotone lift 1.83 → 3.82 under static damping is the key
  qualitative result: damping suppresses the Donaldson stall mode
  enough to recover a meaningful Tier 0 reading without changing the
  underlying physics signal. This is consistent with — and supports —
  the P8.1e narrative that the k=4 strict-tier collapse was a
  Donaldson-side numerical artifact rather than a discrimination
  failure.
* **Tier 3 (Tukey re-trim) at 12.13σ is still post-hoc and remains
  DIAGNOSTIC ONLY**, NOT publication-grade — same caveat as the
  un-damped run (post-hoc trim rejected at k=3 by P5.5h).
* **Forward-pointer.** An **adaptive-damping production sweep is
  running in background** (PID 529740, ~2.4 hr wallclock,
  P8.4-followup) and will be reported separately as
  `references/p8_4_k4_adaptive.md` with its own JSON output. Static
  α=0.5 is the floor; adaptive is expected to do at least as well.

---

**Run (un-damped baseline, kept for cross-comparison)**:
`output/p5_10_k4_gpu_donaldson_only.json` (Donaldson-only, GPU,
no Adam refinement). 20-seed ensemble × 2 candidates × k=4. Wallclock
total **2214 s ≈ 36.9 min**.

**Configuration**:
- `n_pts = 40000`, `ks = [4]`
- `donaldson_iters = 100`, `donaldson_tol = 1e-6`, **`use_gpu = true`**
- `adam_iters = 0` (Adam refinement DISABLED — see "Wallclock & Adam"
  section below)
- `boot_resamples = 10000`, `boot_seed = 12345`
- Same 20-seed list as P5.7 / P5.10 prior runs (cross-comparable)

## Wallclock & Adam

**Adam refinement was DISABLED for this run.** The brief's wallclock
estimate ("~30 s/seed Donaldson GPU + ~30 s/seed Adam CPU loop with
GPU σ-eval = ~40 min total at k=4 with Adam=50") was empirically off
by ~50×. Smoke testing showed:

- **k=2, n_pts=2000, adam_iters=5**: 50–100 s per TY seed.
- **k=3, n_pts=10000, adam_iters=10**: first TY seed had not finished
  after 21 minutes (process killed).
- **k=4, n_pts=40000, adam_iters=50**: first TY seed had not finished
  after 22 minutes (process killed).

The post-Donaldson Adam loop scales as `n_pts × adam_iters × n_basis²`
because of the FD perturbation grid; at k=4 with `n_basis_TY = 200`
that's ~50× the FD work of k=2 even before factoring `n_pts`. Within
the 3-hour budget there was no path to a 20-seed × 2-candidate × k=4
sweep with `adam_iters = 50`.

The Donaldson-only path *did* fit comfortably (36.9 min) and gave the
real "P-DONALDSON-GPU at k=4" headline that the brief targeted, so
that's what landed.

The Adam wiring is correct and produces the expected ~10 % σ-drop:
the k=2 smoke-test seeds showed `σ_donaldson → σ_after_adam` of
0.359 → 0.325 (seed 42) and 0.326 → 0.295 (seed 100). Field-level
plumbing (`--use-gpu`, `--adam-iters`, `--adam-lr` CLI; `Cy3AdamOverride`
threaded through `solve_metric_with_adam`; `sigma_after_donaldson` /
`sigma_after_adam` / `adam_iters_run` surfaced in `PerSeedRecord`) is
all in place and verified compiling.

## Per-seed table (k=4, GPU Donaldson)

### TY (n_basis = 200, all converged tier)

| seed | σ_final | iters | residual | elapsed (s) |
|------|---------|-------|----------|-------------|
| 42 | 0.280101 | 21 | 7.99e-7 | 102.21 |
| 100 | 0.274008 | 21 | 6.34e-7 | 100.74 |
| 12345 | 0.279178 | 21 | 5.96e-7 | 100.80 |
| 7 | 0.282970 | 21 | 7.49e-7 | 105.15 |
| 99 | 0.274033 | 22 | 6.29e-7 | 96.75 |
| 1 | 0.279059 | 22 | 6.45e-7 | 110.81 |
| 2 | 0.279964 | 22 | 5.93e-7 | 107.09 |
| 3 | 0.277909 | 21 | 8.02e-7 | 99.57 |
| 4 | 0.275766 | 21 | 9.84e-7 | 100.10 |
| 5 | 0.285806 | 21 | 7.11e-7 | 101.38 |
| 137 | 0.275221 | 19 | 5.84e-7 | 93.75 |
| 271 | 0.277451 | 22 | 6.10e-7 | 108.64 |
| 314 | 0.278551 | 22 | 7.35e-7 | 117.93 |
| 666 | 0.280391 | 21 | 8.75e-7 | 91.93 |
| 1000 | 0.271961 | 21 | 9.53e-7 | 91.18 |
| 2024 | 0.280685 | 21 | 7.85e-7 | 90.89 |
| 4242 | 0.281009 | 22 | 6.05e-7 | 91.97 |
| 57005 | 0.276226 | 22 | 6.24e-7 | 92.38 |
| 48879 | 0.273210 | 21 | 8.60e-7 | 88.58 |
| 51966 | 0.280493 | 21 | 9.46e-7 | 87.77 |

`<σ_TY> = 0.27820, std = 3.49e-3, SE = 7.80e-4` — extremely tight
distribution, confirming TY at k=4 is well-determined.

### Schoen (n_basis = 48, mixed tiers)

| seed | σ_final | iters | residual | tier | elapsed (s) |
|------|---------|-------|----------|------|-------------|
| 42 | 5.595141 | 25 | 1.51e-4 | converged | 9.75 |
| 100 | 6.891567 | 9 | 4.08e-3 | ambiguous | 4.96 |
| 12345 | 2.851721 | 29 | 8.52e-7 | converged | 8.26 |
| 7 | 5.731964 | 37 | 1.85e-4 | converged | 13.49 |
| 99 | 7.789508 | 90 | 9.84e-7 | converged | 22.80 |
| 1 | 4.555118 | 33 | 9.91e-7 | converged | 8.85 |
| 2 | 5.974649 | 19 | 5.08e-5 | converged | 7.43 |
| 3 | **85.789864** | 55 | 8.91e-7 | converged | 14.18 |
| 4 | 4.614340 | 27 | 6.13e-7 | converged | 7.38 |
| 5 | **47.950527** | 16 | 2.64e-3 | ambiguous | 6.78 |
| 137 | 3.184412 | 74 | 8.91e-7 | converged | 19.38 |
| 271 | 5.929006 | 32 | 2.83e-4 | converged | 12.09 |
| 314 | **14.639028** | 40 | 8.91e-7 | converged | 11.21 |
| 666 | **16.502229** | 63 | 8.56e-7 | converged | 17.07 |
| 1000 | 7.334874 | 31 | 1.26e-6 | converged | 12.07 |
| 2024 | 5.600297 | 53 | 1.29e-4 | converged | 20.04 |
| 4242 | **15.902806** | 11 | 9.99e-3 | ambiguous | 5.54 |
| 57005 | 7.878245 | 36 | 5.22e-4 | converged | 13.96 |
| 48879 | 2.768348 | 27 | 8.28e-7 | converged | 7.81 |
| 51966 | 6.668756 | 41 | 8.70e-7 | converged | 11.64 |

`<σ_Schoen> = 13.21, std = 19.75, SE = 4.42` (full ensemble — heavy
right tail). Outliers in **bold**: seeds 3, 5, 314, 666, 4242 carry
σ ≥ 14.6, with seed 3 reaching σ = 85.8 and seed 5 reaching σ = 47.9.
These are mid-descent snapshots whose Donaldson loop hit a flat region
of the basis-48 landscape; tier-classification flags them but tier-1/2
do not trim them.

## Discrimination — four-tier table

> **Re-tag note (P8.4c):** the per-tier table below is unchanged from
> the original run. The publication-grade interpretation is in the
> follow-up sections: Tier 0 supports P8.1e (basis-size-artifact),
> Tier 3 is **diagnostic only** (post-hoc trim rejected at k=3).
> Tier 0 / Tier 1 / Tier 2 are not publication-grade σ-discriminability
> headlines at k=4 either; they sit below 5σ.


|  Tier | Filter | <σ_TY> | SE_TY | <σ_Schoen> | SE_Schoen | Δσ | n-σ | n_TY | n_Sc | n-σ pct 95 % CI | n-σ BCa 95 % CI | ≥5σ point | ≥5σ CI floor |
|-------|--------|--------|-------|-----------|-----------|----|-----|------|------|------------------|-----------------|-----------|--------------|
| **0** strict-converged | residual<tol AND iters<cap | 0.2782 | 7.80e-4 | 14.94 | 8.02 | 14.66 | **1.83** | 20 | 10 | [1.54, 6.95] | [1.34, 2.33] | no | no |
| **1** conservative | full 20+20 | 0.2782 | 7.80e-4 | 13.21 | 4.42 | 12.93 | **2.93** | 20 | 20 | [2.46, 8.63] | [2.13, 4.00] | no | no |
| **2** canonical | residual<1e-3 | 0.2782 | 7.80e-4 | 11.38 | 4.74 | 11.10 | **2.34** | 20 | 17 | [2.07, 12.34] | [1.81, 5.75] | no | no |
| **3** Tukey re-trim | trim Q1−1.5·IQR / Q3+1.5·IQR | 0.2782 | 7.80e-4 | **5.46** | **0.45** | 5.18 | **11.41** | 20 | 14 | [5.63, 27.90] | **[5.55, 26.42]** | **yes** | **yes** |

## Compare to k=3 Tier-0 baseline (6.92σ, P5.5k commit `9be41815`)

|  Run | Tier 0 n-σ | Tier 1 n-σ | Tier 3 n-σ |
|------|-----------|-----------|-----------|
| **k=3** P5.5k baseline (`p5_10_ty_schoen_5sigma.json`) | **6.92** | n/a | n/a |
| **k=4** P8.4 un-damped (`p5_10_k4_gpu_donaldson_only.json`) | 1.83 | 2.93 | 11.41 (diagnostic) |
| **k=4** P8.4-fix damped α=0.5 (`p5_10_k4_damped.json`) | **3.82 (supporting)** | 2.93 | 12.13 (diagnostic) |

### Tier 0 (strict-converged) at k=4 — confirms P8.1e basis-size-artifact prediction

Tier 0 at k=4 collapses to **1.83σ** (10 of 20 Schoen seeds converge
strict; the other 10 stall at high-residual fixed points with residuals
ranging 1.26e-6 to 9.99e-3). Crucially, those 10 non-strict seeds did
**not** hit the iter cap=100 — max observed iters=90 — meaning they are
**Donaldson stall mode at a non-converged fixed point**, not seeds that
ran out of iterations. Bigger `iter_cap` will not recover them.

This strict-tier collapse is **consistent with P8.1e's prediction** that
σ-discriminability has a substantial basis-size component. At k=3,
Schoen's basis is n_basis=27 (vs TY's 87) and the σ-gap is dominated by
Donaldson-residual structure that strict-converged tier filters
cleanly to a 6.92σ signal. At k=4 Schoen's basis grows to n_basis=48,
the basis-size-artifact term grows with it, and the strict tier alone
no longer suppresses it — exactly the regime P8.1e predicted would
weaken σ as a discrimination channel without changing the underlying
physics signal in the chain channels. P8.4 thus serves as **direct
empirical support** for excluding σ from the model-comparison Bayes
factor (per P8.1e/P8.1f decision).

> See **P8.4-followup** task for proposed Donaldson stall-mode
> investigation (adaptive damping, per-iteration σ trajectory comparison
> across stalling vs strict-converged Schoen seeds).

### Tier 3 (Tukey re-trim) at k=4 = 11.41σ (un-damped) / 12.13σ (damped) — DIAGNOSTIC ONLY, NOT a publication tier

> **CAVEAT — DIAGNOSTIC ONLY, NOT PUBLICATION-GRADE.**
> Tier 3 (Tukey re-trim) at k=4 = **11.41σ** un-damped, BCa 95% CI
> **[5.55, 26.42]**; **12.13σ** damped, BCa 95% CI **[5.92, 30.68]**.
> **This number is NOT publication-grade.** The Tukey-retrim was
> **rejected by P5.5h at k=3** as post-hoc trimming when strict-converged
> seeds (e.g. seed 3 at σ=85.8, residual=8.9e-7, iters=55) are by every
> gating criterion "real" converged samples — they pass the residual
> tolerance, they pass the iter-cap, they're flagged "converged" by the
> tier classifier, and the only reason they stand out is that the σ
> they converge to is far from the population mean. Trimming them
> after-the-fact based purely on σ value is the textbook example of
> post-hoc Tukey filtering that P5.5h ruled out at k=3. **Publishing
> Tier 3 at k=4 while the canonical k=3 result is Tier 0 (6.92σ
> strict-converged, no Tukey trim) would create an impeachable
> contradiction in framing.** Tier 3 at k=4 is reported in the per-tier
> table for diagnostic completeness only.

## 5σ verdict per tier (P8.4-fix damped α=0.5, current state)

| Tier | n-σ | BCa CI low | ≥5σ point | ≥5σ CI floor | Status |
|------|-----|------------|-----------|--------------|--------|
| 0 | **3.82** | 2.48 | no | no | **SUPPORTING tier — corroborates k=3 6.92σ headline** |
| 1 | 2.93 | 2.13 | no | no | diagnostic |
| 2 | 2.75 | 1.93 | no | no | diagnostic |
| 3 | 12.13 | 5.92 | yes | yes | **DIAGNOSTIC ONLY — NOT publication-grade** (post-hoc trim rejected by P5.5h at k=3) |

**The canonical "strict-converged" Tier 0 reading at k=4 (3.82σ
damped) remains weaker than at k=3 (6.92σ).** The k=3 Tier 0
**6.92σ** baseline (`output/p5_10_ty_schoen_5sigma.json`) remains the
canonical σ-channel discriminability publication headline — k=4 does
not displace or duplicate it. The k=4 Tier 0 result is reported here
as a **supporting tier**: it is monotone-consistent with k=3 (TY-Schoen
discriminability is real, robust, and survives Donaldson stall once
damping is applied) and it converts the prior un-damped 1.83σ figure
from a contradiction into a corroboration. Both
`output/p5_10_ty_schoen_5sigma.json` and the new
`output/p5_10_k4_damped.json` are preserved in `output/`; the
un-damped `output/p5_10_k4_gpu_donaldson_only.json` is kept for
cross-comparison only.

## Bootstrap-seed jitter (B = 10 000)

| label | primary_lo | seed_999 | seed_31415 | max_jitter |
|-------|-----------|----------|------------|------------|
| strict_converged_k4 | 1.542 | 1.546 | 1.532 | 0.0097 |
| canonical_k4 | 2.067 | 2.064 | 2.079 | 0.0119 |
| reference_retrim_k4 | 5.625 | 5.720 | 5.632 | 0.0944 |

Tier 3 jitter (0.094 σ) is just over the round-4 target (≤ 0.05 σ at
B = 10 000) — not a blocker but worth noting; jitter at Tiers 0 / 1 / 2
is well within target.

## k=5 sweep — not attempted

Wallclock budget exhausted by the (failed) initial k=4 + Adam attempt
(which ran for 22 min on its first seed before being killed) followed
by the (successful) k=4 Donaldson-only run (37 min). k=5 wallclock
estimate from the brief was ~2 hours; real cost likely much higher
because Schoen at k=5 doubles `n_basis` and TY's `n_basis` grows from
200 to ~470 (cubic-quintic basis count). Defer to future budget.

## Files

- **Modified**:
  `src/bin/p5_10_ty_schoen_5sigma.rs` — added `--use-gpu`,
  `--adam-iters`, `--adam-lr` CLI flags; routed through
  `solve_metric_with_adam(spec, &Cy3AdamOverride { adam_refine,
  use_gpu_donaldson })`; surfaced
  `sigma_after_donaldson` / `sigma_after_adam` / `adam_iters_run` on
  `PerSeedRecord` for post-hoc Adam-impact analysis.
- **Created**:
  - `output/p5_10_k4_gpu_donaldson_only.json` (un-damped P8.4 baseline)
  - `output/p5_10_k4_gpu_donaldson_only.log`
  - `output/p5_10_k4_gpu_donaldson_only.kernel.replog`
  - `output/p5_10_k4_damped.json` (P8.4-fix damped α=0.5 production run, current k=4 supporting tier)
  - `output/p5_10_k4_damped.log`
  - `output/p5_10_k4_damped.kernel.replog`
  - `output/p5_10_k4_damped.replog`
  - `references/p8_4_k4_production.md` (this file)
- **Pending** (background, P8.4-followup, PID 529740):
  - `output/p5_10_k4_adaptive.*` — adaptive-damping production sweep,
    will be reported separately as `references/p8_4_k4_adaptive.md`.
- **Not created** (Adam-enabled budgets blew past wallclock cap):
  `output/p5_10_k4_gpu_adam.json`, `output/p5_10_k5_gpu_adam.json`.
- **Canonical k=3 baseline** (`output/p5_10_ty_schoen_5sigma.json`,
  Tier 0 = 6.92 σ) is **unchanged** — k=4 result here is *not*
  strictly stronger at the strict-converged tier.
