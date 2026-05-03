# Stack-C Phase 3: tangent-frame Ricci diagnostic

## Summary

Phase 2 of the Stack-C Calabi-flow cross-validation computed
`‖Ric‖_{L²}` from the **5×5 ambient Bergman metric** `g_amb` on CP⁴.
The P5.2 n=22 sweep (April 2026) showed that diagnostic was
noise-dominated: the cross-config / seed-only variance ratio was only
**1.28×** for `‖Ric_amb‖` (vs. **63.5×** for `σ_L¹`). The structural
diagnosis was that `log det g_amb` diverges on the radial-direction
singular set, so FD probes pick up O(10⁸) noise-floor contributions
independent of refinement state, washing out any signal.

Phase 3 implements the architecturally clean fix: compute Ricci on the
**3-D CY tangent metric** `g_tan = T† g_amb T̄` that Stack A's σ-eval
already uses correctly. `g_tan` is rank-3 and non-singular on the
variety, so `log det g_tan` is bounded and FD-Ricci is signal-dominated.

**Headline results (n=22, k=2):**

| Quantity                       | Phase 2 (`g_amb`)        | Phase 3 (`g_tan`)         |
|--------------------------------|--------------------------|---------------------------|
| Magnitude of `‖Ric‖_{L²}`      | ~2.5 × 10⁸ (noise-floor) | ~6.0–7.5 (O(1)–O(10))     |
| Cross-config / seed-only ratio | **1.28×**                | **652.82×**               |
| Pearson r                      | +0.3988                  | **-0.5536**               |
| 95% CI on r                    | [-0.0274, +0.7023]       | **[-0.7907, -0.1721]**    |
| `r_lo > 0.5` gate              | FAIL (CI includes 0)     | FAIL (sign is negative)   |

The variance-ratio architectural problem **is fully fixed by Phase 3**:
‖Ric_tan‖ is no longer dominated by singular-set noise, and the signal
moves more than 650× across the config sweep. **But** the refined
diagnostic uncovers a different problem: the σ–‖Ric_tan‖ correlation is
**strongly NEGATIVE**, not positive. As σ_L¹ decreases (toward Stack A's
"more Ricci-flat" claim), the local pointwise ‖Ric_tan‖ Frobenius norm
*increases*. This is statistically robust (Pearson r = -0.55, lower CI
−0.79; Spearman ρ = −0.34, same sign), and it does **not** match the
naive cross-validation hypothesis that both quantities should track
each other downward toward zero.

## Implementation

Module `src/calabi_metric.rs` (additive, Phase 2 functions untouched):

- `ricci_at_point_bergman_tangent(h, z, monomials, n_basis, h_step, out)`
  — computes the 3×3 Hermitian Ricci tensor `Ric_tan_{αβ̄}(z; h)` at a
  single point, in the **affine-chart frame** of `quintic_affine_chart_frame`
  (the same frame Stack A's σ-eval uses for chart-invariant
  `det(g_tan)/|Ω|²`).
  - At base point `z_0`, builds the fixed affine-chart frame
    `T = [T_α]` (3 columns of length 5 complex).
  - For each FD probe in the 6-D real Wirtinger stencil
    (`zp ∈ R^6`, encoding 3 complex perturbations `δw_α`):
    1. Pushforward to ambient: `z_pert = z_0 + Σ_α δw_α · T_α`.
    2. Newton-project back onto `f = 0` via `newton_project_to_quintic`
       (5 iters with `tol=1e-12`).
    3. Evaluate ambient `g_amb(z_proj; h)` via the Phase-2
       `compute_g_amb_at_point`.
    4. Project to 3×3 `g_tan` using the FIXED frame `T` (not
       recomputed at the probe point — that would mix in a
       holomorphic Jacobian whose ∂∂̄ log vanishes in principle but
       adds O(h) noise in practice).
    5. Return `log|det g_tan|`.
  - Hand off to `pwos_math::pde::ricci_at_point_via_fd` with `n=3`.
    Output is 18 f64 (3×3 complex Hermitian, interleaved (re, im)).
- `ricci_norm_l2_tangent(...)` — Frobenius-then-weighted-mean reduction
  over the sample set. Same shape as the Phase-2 `ricci_norm_l2`.

The Phase-2 ambient functions `ricci_at_point_bergman` and
`ricci_norm_l2` are preserved unchanged for regression history.

### Frame-fixedness rationale

The choice to use the **fixed** frame at `z_0` (rather than recomputing
the affine chart at each FD probe) is deliberate and important:

- A recomputed frame at `z_0 + δZ` differs from the frame at `z_0` by a
  smooth holomorphic transformation `T(z + δz) = T(z) · J(z, δz)` where
  `J` is a 3×3 GL(3,ℂ)-valued Jacobian. Under this transformation,
  `det g_tan(z + δz; T(z+δz)) = |det J(z,δz)|² · det g_tan(z + δz; T(z))`,
  so `log det g_tan` picks up `2 Re log det J(z, δz)`.
- The mixed Wirtinger second derivative `∂_α ∂_β̄ log|det J|²` of any
  holomorphic-times-antiholomorphic function vanishes identically. So
  in *exact arithmetic*, the choice of fixed vs. recomputed frame
  doesn't matter for the FD-Ricci output.
- In *floating-point arithmetic* the Jacobian terms produce O(h) noise
  in the FD stencil because the chart/elim choice is a discrete
  argmax that can flip near patch boundaries — exactly the kind of
  numerical instability we are trying to remove.
- Therefore: fix the frame at `z_0` and let only the metric `g_amb`
  vary with the perturbation. This produces a clean scalar function
  `log|det g_tan(z + δZ; T_{fixed})|` whose Wirtinger Hessian is the
  Ricci we want.

## Tests

Module `src/calabi_metric.rs` `mod tests`:

- `test_p4_5_ricci_tangent_finite_at_donaldson_balanced` (default,
  ~0.04 s): at k=2 and k=3 with 50-iter Donaldson balance, evaluates
  `ricci_at_point_bergman_tangent` at 20 random sample points. Asserts:
  - >=75% probes succeed (we observed 100% at k=2 and k=3).
  - `max |Ric_tan_ij|` is in `(1e-3, 1e4)` — the **O(1)–O(10) regime**,
    NOT the O(10⁸) noise-floor of Phase 2.
- `test_p4_5_ricci_tangent_correlates_with_sigma_n20` (`#[ignore]`d,
  ~5 min): same 22-config sweep as P5.2, but with the tangent-frame
  Ricci. Asserts the lower 95% CI on Pearson r exceeds 0.5 — currently
  **fails** (lower CI = -0.79).

The Phase-2 `test_p5_2_ricci_correlates_with_sigma_n20` is preserved
unchanged as a `#[ignore]`d regression to document the original
ambient-metric failure mode.

## Phase-3 σ–Ric_tan endpoint table (n=22, k=2)

| #  | n_pts | n_don | n_adam | lr   | seed | σ_{L¹}    | ‖Ric_tan‖_{L²} |
|----|-------|-------|--------|------|------|-----------|----------------|
| 0  | 1000  | 60    | 10     | 1e-3 | 1042 | 1.212e-1  | 6.132          |
| 1  | 1000  | 60    | 10     | 1e-3 | 1100 | 1.296e-1  | 6.103          |
| 2  | 1000  | 60    | 10     | 1e-3 | 1345 | 1.171e-1  | 6.132          |
| 3  | 1000  | 60    | 10     | 1e-3 | 2007 | 1.233e-1  | 6.102          |
| 4  | 1000  | 60    | 10     | 1e-3 | 1099 | 1.220e-1  | 6.106          |
| 5  |  500  | 30    | 0      | 1e-3 |   43 | 1.908e-1  | 6.404          |
| 6  |  500  | 100   | 0      | 1e-3 |   44 | 1.884e-1  | 6.405          |
| 7  | 1000  | 30    | 0      | 1e-3 |   45 | 1.331e-1  | 6.188          |
| 8  | 1000  | 150   | 0      | 1e-3 |   46 | 1.599e-1  | 6.163          |
| 9  | 1500  | 60    | 0      | 1e-3 |   47 | 1.352e-1  | 6.035          |
| 10 |  500  | 60    | 5      | 1e-4 |   48 | 1.687e-1  | 6.222          |
| 11 |  500  | 60    | 10     | 1e-2 |   49 | 7.196e-2  | 7.363          |
| 12 | 1000  | 30    | 20     | 1e-4 |   50 | 1.392e-1  | 6.160          |
| 13 | 1000  | 100   | 5      | 1e-2 |   51 | 9.057e-2  | 6.699          |
| 14 | 1000  | 150   | 20     | 1e-3 |   52 | 1.047e-1  | 6.292          |
| 15 | 1500  | 30    | 10     | 1e-3 |   53 | 1.024e-1  | 6.046          |
| 16 | 1500  | 100   | 10     | 1e-3 |   54 | 1.039e-1  | 6.118          |
| 17 |  500  | 100   | 40     | 1e-3 |   55 | 1.093e-1  | 6.600          |
| 18 | 1000  | 60    | 40     | 1e-3 |   56 | 8.547e-2  | 6.348          |
| 19 | 1000  | 100   | 40     | 1e-2 |   57 | 4.434e-2  | 7.504          |
| 20 | 1500  | 60    | 40     | 1e-4 |   58 | 1.253e-1  | 6.108          |
| 21 | 1500  | 150   | 40     | 1e-3 |   59 | 8.136e-2  | 6.463          |

### Statistics (n=22)

| Statistic                          | Value                  |
|------------------------------------|------------------------|
| Pearson r                          | **-0.5536**            |
| Spearman ρ                         | -0.3371                |
| Fisher z = atanh(r)                | -0.6235                |
| Standard error 1/√(n-3)            | 0.2294                 |
| 95% CI on Fisher z                 | [-1.0731, -0.1739]     |
| **95% CI on r (back-transformed)** | **[-0.7907, -0.1721]** |

- The lower bound on r is **-0.79** (negative). The upper bound is also
  **negative** (-0.17), so the 95% CI **does not contain zero** —
  Phase 3 does, in fact, give a statistically significant correlation.
  The sign is just **wrong** for the cross-validation hypothesis.
- Pearson and Spearman agree in sign (both negative), so the
  relationship is at least monotone-ish in direction; outliers are
  not driving the sign.

### Architectural variance check

| Group                                       | Var(σ)    | Var(‖Ric_tan‖) |
|---------------------------------------------|-----------|----------------|
| Seed-only (5 configs, fixed pipeline)       | 2.030e-5  | 2.376e-4       |
| Across-all (22 configs)                     | 1.289e-3  | 1.551e-1       |
| **Cross-config / seed-only ratio**          | **63.51×**| **652.82×**    |

The variance-ratio gate (target ≫ 5×) is **passed with margin**:
‖Ric_tan‖ moves 652× more across configs than across seeds. The
Phase-2 noise-floor problem is gone. The signal is real; it just
points the opposite direction from the hypothesis.

## Verdict

**Phase 3 fixes the structural problem (variance-ratio gate
passes 652×), but reveals a different problem: σ-flow refinement does
NOT decrease the local pointwise tangent Ricci Frobenius norm — in
fact it slightly *increases* it.** The 95% CI on Pearson r is
[-0.79, -0.17], which excludes zero with the *wrong* sign for the
naive cross-validation claim.

This is a substantive scientific finding rather than a bug. Possible
interpretations:

1. **σ_L¹ is not a faithful proxy for pointwise Ricci-flatness.** The
   σ functional is the L¹ deviation of `det(g_tan)/|Ω|²` from its
   weighted mean κ — i.e. it measures the *uniformity* of the
   Monge-Ampère ratio, not its *value*. A metric can have low σ
   (Monge-Ampère ratio nearly constant across the variety) without
   having low local Ricci (the constant value can be large in
   absolute terms).
2. **The pointwise Ricci Frobenius norm is not the right object.** A
   physically meaningful "Ricci-flatness" diagnostic on a
   Kähler-Einstein candidate should arguably be the deviation
   `Ric - λ g` for some scalar λ, or the trace-free part `Ric -
   (tr Ric / n) g`. The bare Frobenius norm `‖Ric‖_{L²}` includes a
   uniform-curvature contribution that is NOT what σ is suppressing.
3. **σ-flow concentrates curvature.** As Adam refinement drives σ down
   (by smoothing the Monge-Ampère ratio), it may be redistributing
   curvature so that the *spread* of the ratio shrinks but local
   peaks of `‖Ric_tan‖` grow. Endpoints 11 (low σ, high ‖Ric‖) and
   19 (lowest σ, highest ‖Ric‖) support this picture.

### Cross-validation status: still **partial / unconfirmed**

- σ–‖Ric_tan‖ Pearson r = -0.55, 95% CI = [-0.79, -0.17]. The
  CI **excludes zero** but with negative sign. The "Stack C confirms
  Stack A direction" claim is **NOT supported** in the form originally
  hypothesised.
- σ-flow decrease + ‖Ric_amb‖ decrease (Phase 2) is still observed
  but is necessary, not sufficient.
- Both σ_L¹ and ‖Ric_amb‖ decrease with k (Phase 2). Tangent-Ricci
  k-scan is left as a Phase-3 follow-up.

### What Phase 3 unlocks

Phase 3 **does** unblock the diagnostic instrument: the tangent-frame
Ricci is signal-dominated (variance ratio 652× vs. Phase-2's 1.28×),
the magnitudes are physically meaningful (O(1)–O(10) instead of
O(10⁸) noise floor), and we can now measure the pointwise local
curvature on the variety with confidence.

It does **not** unblock the originally hoped-for σ ≈ ‖Ric‖ co-monotone
cross-validation. To do that we would need either:
- A pointwise-Ricci-deviation metric (e.g. `‖Ric - (κ_tr) g‖_{L²}`)
  whose definition tracks the σ functional's "uniformity" semantics
  rather than "magnitude"; or
- A direct ‖Ric‖ → 0 endpoint that doesn't go through σ-flow (e.g.
  Calabi-flow proper, projecting `∂g/∂t = -Ric` onto the h-block
  parameter space).

Both are out of scope for P4.5 / hostile-review §2.1 follow-up; they
are bookmarked as **Phase 4 candidates**.

## Files

- `src/calabi_metric.rs` — added `ricci_at_point_bergman_tangent`,
  `ricci_norm_l2_tangent`, and two new tests
  (`test_p4_5_ricci_tangent_finite_at_donaldson_balanced`,
  `test_p4_5_ricci_tangent_correlates_with_sigma_n20`).
- `references/p4_4_calabi_flow_results.md` — Phase 2 history (untouched).
- `references/p4_5_calabi_flow_phase3.md` — this file.

## Reproduction

```bash
cd book/scripts/cy3_substrate_discrimination/rust_solver

# Phase-3 finite/magnitude regression (default test, ~1 s):
cargo test --release --features gpu --lib \
    test_p4_5_ricci_tangent_finite_at_donaldson_balanced -- --nocapture

# Phase-3 n=22 correlation re-run (~1 min release; currently FAILS the
# r_lo > 0.5 gate due to negative correlation):
cargo test --release --features gpu --lib \
    test_p4_5_ricci_tangent_correlates_with_sigma_n20 -- --ignored --nocapture

# Phase-2 ambient regression (preserved for comparison):
cargo test --release --features gpu --lib \
    test_p5_2_ricci_correlates_with_sigma_n20 -- --ignored --nocapture
```
