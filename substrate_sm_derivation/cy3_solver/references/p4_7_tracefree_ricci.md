# P4.7 — Trace-free Ricci diagnostic on the post-Donaldson 22-endpoint sweep

## Problem

P4.5 Phase 3 found that on the n=22 sweep at k=2, σ_L¹ (Monge-Ampère
residual) and `‖Ric_tan‖_{L²}` (full pointwise tangent-frame Ricci
Frobenius norm) are NEGATIVELY correlated: Pearson r = −0.5536,
Fisher 95% CI [−0.7907, −0.1721]. The CI excludes zero with the wrong
sign for the cross-validation hypothesis (both should track each other
toward zero at the Calabi-Yau metric).

The working hypothesis driving P4.7 was that the right diagnostic might
not be full Ricci but the **trace-free** Ricci
`Ric₀ = Ric − (s/n) g`  (n = 3 complex dim, s = scalar curvature trace).
The Calabi-Yau condition is `Ric = 0`; the **Einstein** condition is
`Ric₀ = 0`. If σ-flow drove the Bergman candidate toward Einstein
(rather than Ricci-flat) at finite k, σ would correlate POSITIVELY with
`‖Ric₀‖`, and the P4.5 negative sign would resolve as σ-flow ≠ Calabi
flow but σ-flow = Einstein flow.

## Method

The binary `src/bin/p4_7_tracefree_ricci.rs` reproduces the P4.5 Phase 3
protocol exactly (same 22 endpoint sweep, k=2, varied n_pts /
Donaldson / Adam parameters from
`test_p4_5_ricci_tangent_correlates_with_sigma_n20` in
`src/calabi_metric.rs`) and at each endpoint computes:

* `sigma_L1`            — Stack A's Monge-Ampère L¹ residual.
* `ricci_L2_full`       — `‖Ric_tan‖_{L²}` (re-uses the P4.5 helper
  `ricci_norm_l2_tangent` so the numbers line up with P4.5 Phase 3 to
  4+ decimal places — verified, see "Sanity checks" below).
* `ricci_L2_tracefree`  — `‖Ric₀‖_{L²}` where
  `Ric₀_{αβ̄} = Ric_{αβ̄} − (s/3) g_tan_{αβ̄}` and
  `s = (g_tan)^{βᾱ} Ric_{αβ̄}` (real scalar trace).
* `scalar_L2`           — `‖s‖_{L²}` (independent diagnostic).

All three norms use the same plain-entry-Frobenius convention as the
P4.5 helper:  `Σ_{αβ} |T_{αβ̄}|²`, weighted-mean reduction over the
sample set.

The ambient metric `g_amb` and the affine-chart frame `T` are computed
at the un-perturbed base point z_0; `g_tan = T† g_amb T̄` (the proper
Kähler-form projection used by the σ-functional, see
`project_to_quintic_tangent` in `src/quintic.rs`); the inverse `g_tan⁻¹`
is computed via the closed-form 3×3 cofactor formula and Hermitised to
suppress floating-point drift.

The Ricci tensor itself is the existing P4.5 Phase 3 evaluator
`ricci_at_point_bergman_tangent` — same fixed-frame Wirtinger FD on
`log|det g_tan|` that produced the P4.5 number, unchanged.

The new helper that bundles `(‖Ric‖, ‖Ric₀‖, ‖s‖)` per-point and reduces
in parallel was added as `ricci_tracefree_and_scalar_norms_l2_tangent`
in `src/calabi_metric.rs` (additive — Phase 3 functions and tests
preserved).

For each of the three pairs `(σ, ‖Ric_full‖)`, `(σ, ‖Ric₀‖)`,
`(σ, ‖s‖)` the binary computes:

1. Pearson r and Spearman ρ.
2. Fisher-z 95% CI on Pearson r (closed-form; cross-checks the
   bootstrap and reproduces P4.5 Phase 3's interval to 4 dp).
3. Paired-bootstrap 95% CIs (n_resamples = 2000, seed = 12345):
   percentile method (Efron 1979) and BCa (Efron 1987, jackknife
   acceleration).

## Results

### Per-endpoint table (n = 22, all valid)

See `output/p4_7_tracefree_ricci.json` for the full record. Spot-checks
against P4.5 Phase 3 (`references/p4_5_calabi_flow_phase3.md`):

| #  | seed | σ_L¹      | ‖Ric‖   | ‖Ric₀‖  | ‖s‖     | P4.5 ‖Ric‖ | match |
|----|------|-----------|---------|---------|---------|------------|-------|
|  0 | 1042 | 1.212e−1  | 6.1318  | 4.1139  | 2.6051  | 6.132      |  ✓    |
|  5 |   43 | 1.908e−1  | 6.4041  | 4.1184  | 2.6397  | 6.404      |  ✓    |
| 11 |   49 | 7.196e−2  | 7.3625  | 4.8469  | 3.2215  | 7.363      |  ✓    |
| 19 |   57 | 4.434e−2  | 7.5039  | 5.8103  | 3.2786  | 7.504      |  ✓    |
| 21 |   59 | 8.136e−2  | 6.4626  | 4.3606  | 2.7217  | 6.463      |  ✓    |

`σ` and `‖Ric‖` reproduce P4.5 Phase 3 to 4+ decimal places, confirming
protocol identity.

### Three correlations table

| Pair                              | Pearson r | Spearman ρ | Fisher 95% CI       | Bootstrap percentile 95% CI | Bootstrap BCa 95% CI    |
|-----------------------------------|-----------|------------|---------------------|-----------------------------|-------------------------|
| `(σ_L¹, ‖Ric_full‖)`              | −0.5536   | −0.3371    | [−0.7907, −0.1721]  | [−0.8267, +0.0631]          | [−0.8618, −0.0544]      |
| `(σ_L¹, ‖Ric₀‖)`  *(NEW, P4.7)*   | −0.6993   | −0.7504    | [−0.8657, −0.3939]  | [−0.8655, −0.5172]          | [−0.8366, −0.4255]      |
| `(σ_L¹, ‖s‖)`                     | −0.6704   | −0.5788    | [−0.8514, −0.3468]  | [−0.8591, −0.3477]          | [−0.8760, −0.4115]      |

The first row reproduces P4.5 Phase 3's Fisher CI exactly
(`[−0.7907, −0.1721]`); the bootstrap CIs are slightly wider on the
upper end (BCa upper bound −0.054, percentile upper bound +0.063,
which crosses zero), reflecting that with n=22 the upper end of the
correlation distribution is unstable and the published Fisher-CI is on
the optimistic side.

### Decision-tree branch landed on

> **Branch 3: r(σ, ‖Ric₀‖) is also significantly NEGATIVE.**

The trace-free Ricci correlates *more strongly* (negatively) with σ
than the full Ricci does, with all three CIs (Fisher, percentile, BCa)
clearly excluding zero on the negative side. The scalar curvature
itself also correlates negatively (r = −0.67), with all three CIs
excluding zero.

The hypothesis that σ-flow drives the metric toward Einstein at finite
k is **not** supported. σ-minimisation simultaneously increases the
trace-free Ricci AND the scalar trace — both pieces of curvature grow
as σ shrinks. There is no meaningful sense in which the P4.5 negative
sign was a "wrong-channel" artefact.

This refines (rather than overturns) the P4.5 narrative: the σ
functional is measuring the *uniformity* of the Monge-Ampère ratio,
not its *value*, and σ-flow concentrates curvature rather than
suppressing it. Endpoints 19 (lowest σ = 4.4e−2) and 11 (σ = 7.2e−2)
are the most σ-refined and have the LARGEST `‖Ric‖`, `‖Ric₀‖`, AND
`‖s‖` — the "concentrate-curvature" picture from P4.5 Phase 3 is
confirmed across all three diagnostic channels.

A useful follow-up note: the Spearman ρ for the trace-free pair
(−0.7504) is much more negative than for the full-Ricci pair
(−0.3371). This says the **rank order** between σ and ‖Ric₀‖ is
near-monotone, while the rank order between σ and ‖Ric‖ has more
non-monotone scatter. The trace-free channel is therefore a *cleaner*
diagnostic — just one that points in the same (negative) direction.

## Sanity checks

* `‖Ric_full‖` from the P4.7 helper agrees with the P4.5 helper to
  better than 1e−3 relative on every endpoint (the binary asserts this
  and would emit a warning otherwise; no warnings were printed during
  the run).
* Per-endpoint σ_L¹ values reproduce P4.5 Phase 3's stated values to
  4+ decimal places.
* Pearson r for the `(σ, ‖Ric_full‖)` pair is −0.5536 versus P4.5's
  reported −0.5536; Fisher CI matches to 4 dp.
* Spearman ρ for the `(σ, ‖Ric_full‖)` pair is −0.3371, matching the
  P4.5 Phase-3 reported value (−0.3371).

## Wallclock

Total: **1.18 s** on a 16-core release build with rayon parallel
reduction across the 500–1500 sample points. The advertised P4.5
Phase 3 timing of "~1 min release" was a conservative upper bound;
the actual per-endpoint cost on this machine is dominated by the
~25 FD probes per sample point at k=2 (n_basis = 35) and is
sub-100 ms per endpoint, ×22 endpoints ≈ 1 s.

The protocol matches P4.5 Phase 3 verbatim — k=2, n_pts ranging
500–1500 across the 22 endpoints, no parameter changes. No protocol
deviation.

## Files

* `src/calabi_metric.rs` — added (additive) two helpers:
  * `invert_3x3_complex_hermitian_local` — module-private 3×3 complex
    Hermitian inverse via cofactor formula.
  * `ricci_tracefree_and_scalar_norms_l2_tangent` — public; per-point
    computes `(‖Ric‖², ‖Ric₀‖², ‖s‖²)` weighted-mean reduction, returns
    the three L² norms.
  Pre-existing functions (`ricci_at_point_bergman_tangent`,
  `ricci_norm_l2_tangent`, all P4.4 / P4.5 tests) are untouched.
* `src/bin/p4_7_tracefree_ricci.rs` — the binary (this report's
  reproduction harness).
* `output/p4_7_tracefree_ricci.json` — full per-seed records and
  correlation tables.
* `references/p4_7_tracefree_ricci.md` — this writeup.

## Reproduction

```bash
cd book/scripts/cy3_substrate_discrimination/rust_solver
cargo run --release --bin p4_7_tracefree_ricci -- \
    --output output/p4_7_tracefree_ricci.json \
    --boot-resamples 2000 --boot-seed 12345
```
