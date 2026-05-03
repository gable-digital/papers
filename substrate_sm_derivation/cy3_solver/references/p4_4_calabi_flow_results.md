# Stack-C Phase 2: Calabi-flow / Ricci-norm cross-validation results

## Summary

Stack-C Phase 2 plugs the Phase-1 PDE primitives
(`pwos_math::pde::ricci_at_point_via_fd`, ...) into the Bergman-kernel
ambient metric on the Fermat quintic. We compute the **L² Ricci norm**
`‖Ric‖_{L²}` as an independent diagnostic that should track Stack A's
σ-functional residual `σ_{L¹}`. Both vanish at a Ricci-flat
Kähler-Einstein metric, so a high σ–Ric correlation under refinement
would constitute the cross-validation we set out to prove.

> **Hostile-review §2.1 update (2026-04-27).** The original
> n=5 r=0.7631 result was statistically underpowered (Fisher 95% CI
> roughly [-0.37, +0.98], i.e. *includes zero*). We re-ran with n=22
> distinct optimisation configurations. The result is honest and not
> what we hoped: **r = 0.3988, 95% CI = [-0.0274, +0.7023]** (still
> includes zero). The directional cross-validation claim is **NOT
> supported** at n>=20. See the [P5.2 re-run section](#p52-rerun-n22-after-2-1-hostile-review)
> below.

## Implementation

Module `src/calabi_metric.rs`:

- `ricci_at_point_bergman(h, z, monomials, n_basis, h_step, out)` —
  computes the 5×5 ambient Ricci tensor `Ric_{ij̄}(z; h)` at a single
  point via central FD on `log det g_amb(z)`. Wraps a closure around
  `compute_g_amb_at_point` (the per-point Bergman-metric evaluator
  pulled out of `quintic.rs::compute_sigma_from_workspace`) and feeds
  it to `pwos_math::pde::ricci_at_point_via_fd`. Pre-allocates all
  scratch buffers inside the call so the inner FD loop is allocation-
  free.
- `ricci_norm_l2(points, weights, n_actual, h, monomials, n_basis,
  h_step)` — Frobenius-then-weighted-mean reduction over the sample
  set (rayon-parallel). Skips points that hit the rank-4 singular
  set; returns NaN if more than half fail.
- `log_det_complex_hermitian_5x5(g)` — partial-pivoting LU on the
  10×10 real block, giving `log|det g|` via `0.5 · log|det M|` where
  `M = [[A, -B], [B, A]]`.

## Original P4.4 σ-Ric correlation at k=2 (n=5, **statistically underpowered**)

Five (σ, ‖Ric‖) endpoints across the optimisation trajectory
(seed = 42, n_pts = 300):

| Endpoint | configuration             | σ_{L¹}   | ‖Ric‖_{L²} | ‖Ric‖ / σ |
|----------|---------------------------|----------|------------|-----------|
| 0        | identity h                | 2.033e-1 | 2.565e8    | 1.261e9   |
| 1        | Donaldson 5 iters         | 2.465e-1 | 2.485e8    | 1.008e9   |
| 2        | Donaldson 20 iters        | 2.467e-1 | 2.597e8    | 1.053e9   |
| 3        | Donaldson 50 iters        | 2.467e-1 | 2.627e8    | 1.065e9   |
| 4        | Donaldson 80 + Adam 30    | 7.513e-2 | 2.421e8    | 3.222e9   |

- **Pearson r = 0.7631** — *prima facie* "agreement", but n=5 gives
  a Fisher-z 95% CI on r of roughly **[-0.37, +0.98]** (i.e. consistent
  with zero correlation). This is not a statistically significant
  result.
- The corresponding test (`test_p4_4_ricci_correlates_with_sigma`) is
  retained but `#[ignore]`d. Do not cite this number in support of
  Stack-A/Stack-C agreement.

## P5.2 re-run: n=22 (after §2.1 hostile review)

**Test**: `test_p5_2_ricci_correlates_with_sigma_n20`
(`#[ignore]`d due to runtime + because the assertion currently fails;
run via `cargo test --release --features gpu --lib
test_p5_2_ricci_correlates_with_sigma_n20 -- --ignored --nocapture`).

**Sweep**: 22 distinct (n_pts, n_donaldson, n_adam, lr_adam, seed)
tuples covering n_pts ∈ {500, 1000, 1500}, n_donaldson ∈ {30, 60, 100,
150}, n_adam ∈ {0, 5, 10, 20, 40}, lr_adam ∈ {1e-4, 1e-3, 1e-2}.
Includes a 5-config seed-only variance group (identical pipeline,
varying seed) for the architectural variance check.

### Endpoint table (k=2)

| # | n_pts | n_don | n_adam | lr   | seed | σ_{L¹}   | ‖Ric‖_{L²} |
|---|-------|-------|--------|------|------|----------|------------|
| 0 | 1000  | 60    | 10     | 1e-3 | 1042 | 1.212e-1 | 2.529e8    |
| 1 | 1000  | 60    | 10     | 1e-3 | 1100 | 1.296e-1 | 2.664e8    |
| 2 | 1000  | 60    | 10     | 1e-3 | 1345 | 1.171e-1 | 2.609e8    |
| 3 | 1000  | 60    | 10     | 1e-3 | 2007 | 1.233e-1 | 2.633e8    |
| 4 | 1000  | 60    | 10     | 1e-3 | 1099 | 1.220e-1 | 2.553e8    |
| 5 |  500  | 30    | 0      | 1e-3 |   43 | 1.908e-1 | 2.733e8    |
| 6 |  500  | 100   | 0      | 1e-3 |   44 | 1.884e-1 | 2.648e8    |
| 7 | 1000  | 30    | 0      | 1e-3 |   45 | 1.331e-1 | 2.627e8    |
| 8 | 1000  | 150   | 0      | 1e-3 |   46 | 1.599e-1 | 2.551e8    |
| 9 | 1500  | 60    | 0      | 1e-3 |   47 | 1.352e-1 | 2.645e8    |
|10 |  500  | 60    | 5      | 1e-4 |   48 | 1.687e-1 | 2.772e8    |
|11 |  500  | 60    | 10     | 1e-2 |   49 | 7.196e-2 | 2.631e8    |
|12 | 1000  | 30    | 20     | 1e-4 |   50 | 1.392e-1 | 2.566e8    |
|13 | 1000  | 100   | 5      | 1e-2 |   51 | 9.057e-2 | 2.677e8    |
|14 | 1000  | 150   | 20     | 1e-3 |   52 | 1.047e-1 | 2.556e8    |
|15 | 1500  | 30    | 10     | 1e-3 |   53 | 1.024e-1 | 2.579e8    |
|16 | 1500  | 100   | 10     | 1e-3 |   54 | 1.039e-1 | 2.639e8    |
|17 |  500  | 100   | 40     | 1e-3 |   55 | 1.093e-1 | 2.584e8    |
|18 | 1000  | 60    | 40     | 1e-3 |   56 | 8.547e-2 | 2.546e8    |
|19 | 1000  | 100   | 40     | 1e-2 |   57 | 4.434e-2 | 2.589e8    |
|20 | 1500  | 60    | 40     | 1e-4 |   58 | 1.253e-1 | 2.532e8    |
|21 | 1500  | 150   | 40     | 1e-3 |   59 | 8.136e-2 | 2.601e8    |

Note that `donaldson_solve` early-stops at `tol=1e-8` after ~16–18
iterations regardless of the requested cap, so the "n_donaldson"
axis collapses for caps >~ 20. This is internal solver behaviour, not
a test-fixture bug, and is honestly reflected in the spread.

### Statistics (n=22)

| Statistic                | Value             |
|--------------------------|-------------------|
| Pearson r                | **0.3988**        |
| Spearman ρ               | 0.2716            |
| Fisher z = atanh(r)      | 0.4222            |
| Standard error 1/√(n-3)  | 0.2294            |
| 95% CI on Fisher z       | [-0.0274, +0.8719]|
| **95% CI on r (back-transformed)** | **[-0.0274, +0.7023]** |

- The lower bound on r **includes zero**. We cannot reject the null
  hypothesis r = 0 at the 5% level.
- Pearson and Spearman agree in sign, so the relationship is at least
  monotone-ish in direction, but the magnitude of the rank
  correlation (0.27) is even weaker than the linear one — outliers
  at large σ are inflating Pearson slightly.

### Architectural variance check

| Group                              | Var(σ)     | Var(‖Ric‖)  |
|------------------------------------|------------|-------------|
| Seed-only (5 configs, fixed pipeline) | 2.030e-5 | 3.140e13    |
| Across-all (22 configs)            | 1.289e-3   | 4.006e13    |
| **Cross-config / seed-only ratio** | **63.5×**  | **1.28×**   |

This is the diagnostic finding. Cross-config variance dominates
seed-only variance for σ (63.5×), confirming the config-axis sweep is
the right test for σ. But for ‖Ric‖_{L²}, cross-config variance is
only **1.28×** the seed-only noise — i.e. **‖Ric‖ barely moves across
configs.** The Donaldson floor at ~2.5–2.8 × 10⁸ is dominated by the
rank-4 ambient-metric singularity contribution as documented in the
original Caveats section, and it swamps the signal we are trying to
correlate against σ.

This is consistent with what the n=5 result was *actually* measuring:
the apparent r=0.76 was driven almost entirely by the identity-h
endpoint (σ = 0.20, ‖Ric‖ = 2.57e8) being slightly displaced from the
Donaldson cluster, plus the Adam-refined endpoint being slightly low.
Once we sample the Donaldson + Adam parameter space densely (n=22),
that apparent linear trend washes out into the rank-4-singularity
noise floor.

### Verdict

**Stack A and Stack C do NOT statistically agree directionally on
‖Ric‖_{L²} at k=2 with the current ambient-metric Ricci diagnostic.**
The original P4.4 r=0.7631 claim (n=5) was a small-sample artifact;
the 95% CI on the n=22 estimate (r = 0.40, [-0.03, +0.70]) cannot
exclude zero correlation.

**This does NOT mean Stack A is wrong.** It means the *ambient*
‖Ric‖_{L²} on CP^4 is too noisy a diagnostic to confirm the
σ-functional direction with the precision we hoped for, because the
rank-4 singular set on the ambient metric injects O(10⁸) baseline
contributions independent of refinement state. The natural fix
(per the original Caveats, item 1) is to compute ‖Ric‖ on the
**3-dim CY tangent metric** instead, which has no rank-4 degeneracy.
Implementing that requires re-deriving the Wirtinger FD stencil in
the affine-chart frame and is **deferred to Phase 3**.

For the book's published claims, the σ-decrease + ‖Ric‖-decrease
co-monotonicity test (`test_p4_4_sigma_flow_decreases_ricci_norm`)
is the most we can defend right now: under σ-flow refinement, both
quantities decrease (or hold within tolerance), which is necessary
but not sufficient for the cross-validation. The strong statistical
claim ("Pearson r = 0.76, agreement confirmed") must be retracted.

## ‖Ric‖_{L²} monotone decrease vs k (k=2 → k=3, post-refinement)

| k | n_basis | σ_{L¹}    | ‖Ric‖_{L²} |
|---|---------|-----------|------------|
| 2 | 15      | 7.924e-2  | 2.785e8    |
| 3 | 35      | 8.001e-2  | 2.751e8    |

- Both σ and ‖Ric‖ decrease (or hold within tolerance) as expected.
  The k=3 advantage is small here because the Adam loop is short
  (20 iters); the long-running k=4 test (`#[ignore]`d, ~60 s) extends
  this to the full k ∈ {2, 3, 4} sweep.

## σ-flow trajectory (Stack A's Adam refinement = approximate Calabi flow)

Starting from a Donaldson-balanced initial state at k=2:

```
σ_{L¹}: 2.467e-1 → 5.144e-2   (decreased by ~5×)
‖Ric‖_{L²}: 2.697e8 → 2.642e8   (decreased ~2%)
```

Both decrease under σ-flow. The ‖Ric‖ decrease is small in
relative terms because the *floor* of ‖Ric‖ on a Donaldson-balanced
metric is dominated by the rank-4 ambient-metric singularity (FD on
`log det g_amb` near the singular set produces O(10⁸) contributions
that are independent of the refinement state). The n=22 variance
analysis (above) makes this quantitative: cross-config variance of
‖Ric‖ is only 1.28× the seed noise, confirming the floor effect.

## Caveats and design notes

1. **Ambient vs CY tangent metric.** `g_amb` on CP^4 is rank-4 (the
   complex radial direction has `g(z, ·) = 0` by homogeneity), so
   `log det g_amb` diverges on a measure-zero singular set. About
   ~10–15% of FD probe points at random sample positions hit close
   enough to the singular set to fail; we skip these gracefully.
   The natural alternative is `Ric` on the **3-dim CY tangent
   metric** (which is non-degenerate), but that requires re-deriving
   the Wirtinger FD stencil in the affine-chart frame — left for
   Phase 3. **The §2.1 hostile-review n=22 re-run shows this matters:
   the ambient diagnostic is too noisy to statistically confirm the
   σ direction.**
2. **Absolute magnitude of ‖Ric‖.** The ratio `‖Ric‖ / σ ≈ 10⁹` is
   inflated by the rank-4 singularity contribution; it is *not* a
   physical proportionality constant.
3. **Stack-C Phase 2 ≠ true Calabi flow.** True Calabi flow projects
   `∂g/∂t = -Ric(g)` back onto h-block parameter space, which is
   `∂h_{ab}/∂t = -⟨Ric, ∂g/∂h_{ab}⟩` — implementing this requires the
   projection of `Ric` against `∂g/∂h_{ab}` at every point, which is
   another `O(n_basis² × n_pts)` operation per step. We instead use
   σ-flow as the "right-direction" surrogate, since σ is already a
   proxy for the Ricci-flat residual.
4. **Statistical-power discipline.** Any future correlation claim
   must be supported by n>=20 endpoints with Fisher 95% CI on r
   reported, **not** a point estimate at small n.

## Cross-validation status: PARTIAL

- ~~σ–Ric Pearson correlation 0.76 > 0.7 (target met)~~ — **withdrawn**
  after n=22 re-run gave r=0.40, 95% CI = [-0.03, +0.70].
- σ-flow decreases both σ and ‖Ric‖ — still holds (necessary but
  not sufficient).
- Both decrease with k — still holds.

Stack A's σ ≈ 0.13 at k=4 remains an honest residual of the
Ricci-flat condition. The independent Stack-C diagnostic (ambient
‖Ric‖_{L²}) is too noisy to **statistically confirm** the σ direction;
a CY-tangent-metric Ricci computation (Phase 3) is required for a
defensible cross-validation.

## Files

- `src/calabi_metric.rs` — module + tests
- `src/lib.rs` — exposes `pub mod calabi_metric`
- `Cargo.toml` — `pde` feature enabled on pwos-math

## Reproduction

```bash
cd book/scripts/cy3_substrate_discrimination/rust_solver

# Fast (n=4–5) regression tests:
cargo test --release --features gpu --lib calabi_metric -- --nocapture

# Authoritative §2.1-hostile-review n=22 re-run (currently FAILS the
# assertion lower-CI > 0.5; documents the actual r and CI):
cargo test --release --features gpu --lib \
    test_p5_2_ricci_correlates_with_sigma_n20 -- --ignored --nocapture

# Original n=5 P4.4 test (preserved for regression history, ignored):
cargo test --release --features gpu --lib \
    test_p4_4_ricci_correlates_with_sigma -- --ignored --nocapture

# Long-running k=4:
cargo test --release --features gpu --lib \
    test_p4_4_ricci_norm_decreases_with_k_k4 -- --ignored --nocapture
```
