# P-Stage-3 Fix B M0 — Chern (1,1)-curvature evaluator

**Status**: Landed on `main`; hardened via 7-finding hostile review.
**Predecessor**: `p_stage3_fix_b_m1m2_detailed_spec.md` (commit `eb87c12b`).
**File added**: `src/route34/chern_curvature.rs` (~1450 LOC incl. tests).
**Tests**: 15 unit tests, all passing (9 original + 6 hostile-review additions).

## What this milestone delivers

`evaluate_chern_curvature(bundle, metric_background, h_v) -> ChernCurvatureCloud`

returns the pointwise (1,1)-curvature

```
F^{(1,1)}_{a b̄} = ∂̄_a (h^{-1} ∂_b h)
                = h^{-1} (∂̄_a ∂_b h) − h^{-1} (∂̄_a h) h^{-1} (∂_b h)
```

of the Chern connection on the Hermitian holomorphic bundle `(V, h)` for
the polystable monad bundle, evaluated at every accepted sample point of
the supplied `MetricBackground`.

## Why M0 was missing

The G1 spec assumed `A^I_{ā J}` (the (0,1)-component of the Chern
connection 1-form) already lived in `hym_hermitian.rs`. **It did not.**
`hym_hermitian.rs` (1230 LOC) only solves for the constant
section-basis Hermitian metric matrix `H_{αβ}`; it never decomposes the
Chern connection into (1,0)/(0,1) parts and never extracts a curvature
2-form.

The spec itself flags (§215-225) that *in the holomorphic frame the
Chern connection's (0,1)-part `θ^{(0,1)} = 0` is identically zero*. The
non-trivial (0,1) coupling content of the bundle-twisted operator
`∂̄_V` is therefore in the (1,1)-curvature `F^{(1,1)}`, not in a
(0,1)-connection. M2 will use the Bochner-Weitzenböck identity

```
Δ_{∂̄_V} ψ = Δ̄ ψ + i Λ_ω F^{(1,1)} · ψ        (Bochner)
```

to assemble its Laplacian; this M0 module supplies the `F^{(1,1)}` cloud
that's the missing input.

## Math: factorisation in the section basis

In the AKLP §3 / `chern_field_strength.rs:308-313` convention the local
fibre metric on V at point p factorises as

```
h_{α β}(p) = H_{α β} · s_α(p) · conj(s_β(p))
```

where `H_{α β}` is the constant HYM-converged matrix from
`HymHermitianMetric` and `s_α` is the canonical FS section of `O(b_α)`.
In matrix form: `h(p) = S(p) · H · S(p)†` with `S(p) = diag(s_α(p))`.

This factorisation forces:

- `∂_b h(p) = (∂_b S) H S†`  (only `S` is hit, `S†` is anti-hol.)
- `∂̄_a h(p) = S H (∂̄_a S)†`
- `∂̄_a ∂_b h(p) = (∂_b S) H (∂̄_a S)†`

so the curvature evaluation needs only:
1. The constant `H` matrix (from HYM).
2. Diagonal `S(p)`, `∂_b S(p)`, `(∂_a S(p))*` (latter from holomorphic
   derivative + element-wise conjugation).
3. A small `n × n` Hermitian inverse `h^{-1}(p)` (Tikhonov-regularised).

All four are exact polynomial operations on the section seeds. **No
finite-difference is used** — the M1+M2 spec flagged this as a risk; we
chose the symbolic path.

## API summary

```rust
pub const N_AMBIENT_COORDS: usize = 8;   // CP^3×CP^3 / Schoen CP²×CP²×CP¹

pub enum ChernError {
    EmptyBundle,
    DimensionMismatch { expected: usize, got: usize },
    AllSamplesDegenerate,
    NoSamples,
}

pub struct ChernCurvatureCloud {
    pub n_pts: usize,                    // accepted sample points
    pub n: usize,                        // bundle.b_lines.len() (section basis)
    pub n_cy_dim: usize,                 // = N_AMBIENT_COORDS = 8
    pub n_degenerate_samples: usize,     // Bergman-zero degeneracies
    pub n_singular_inversions: usize,    // Tikhonov-fallback failures (Finding 1)
    // F^{(1,1)}_{a b̄, i j}(p) accessor: cloud.at(sample, a, b, i, j)
    // matrix slice of length n²:        cloud.matrix_slice(sample, a, b)
    // Frobenius norm:                   cloud.frobenius_norm()
    // Trace at one (sample, a, b):      cloud.trace_at(sample, a, b)
    // Gauge-correct Hermiticity check:  cloud.check_hermiticity_with_h(h_at)
    // Unitary-frame index-swap check:   cloud.check_index_swap_symmetry()
    // Excluded-sample list (M1/M2 must filter):
    //                                   cloud.degenerate_sample_indices() -> &[usize]
    //                                   cloud.is_sample_degenerate(idx) -> bool
}

pub fn evaluate_chern_curvature(
    bundle: &MonadBundle,
    metric: &dyn MetricBackground,
    h_v: &HymHermitianMetric,
) -> Result<ChernCurvatureCloud, ChernError>;

pub struct CurvatureDiagnostics { /* frob, index-swap viol, mean |Re Tr|, n_degen */ }
pub fn curvature_diagnostics(cloud: &ChernCurvatureCloud) -> CurvatureDiagnostics;

pub fn metric_trace_cloud(
    cloud: &ChernCurvatureCloud,
    g_inv_cy3: &[Complex64],   // length nc² = 64
) -> Result<Vec<Complex64>, ChernError>;
```

## Tests (15, all passing)

### Original 9 (M0 landing)

| # | Name | What it verifies |
|---|------|------------------|
| 1 | `trivial_bundle_has_zero_curvature` | `b_lines = [[0,0]; n]` ⇒ `F = 0` exactly |
| 2 | `rank_one_o10_curvature_is_diagonal_scalar` | rank-1 line bundle gauge-trivial F = 0 in canonical-section gauge |
| 3 | `aklp_bundle_curvature_satisfies_gauge_hermiticity` | gauge-invariant `(hF)_{IJ}* = (hF_swap)_{JI}` holds on AKLP B = O(1,0)³ ⊕ O(0,1)³ |
| 4 | `aklp_bundle_canonical_gauge_curvature_near_zero` | pins the **rank-deficient `h(p)`** behaviour: AKLP picks identical lowest-weight monomials per summand, making `h(p)` rank-deficient and wash F to ≈ 1e-8 (Tikhonov floor). This is a known section-gauge artifact, not a defect |
| 5 | `distinct_degree_bundle_curvature_is_nonzero` | distinct-degree monad `[1,0],[2,0],[0,1],[1,1]` gives non-degenerate `h(p)` and a genuinely non-trivial `F` (Frob norm > 1e-3) |
| 6 | `metric_trace_diagonal_g_inv` | `g^{a b̄} = δ_{ab̄}` produces `Σ_a F_{a ā}` at every sample (sanity for M2 contraction) |
| 7 | `diagnostics_report_correct_degenerate_count` | weight-0 sample is correctly counted as degenerate |
| 8 | `empty_bundle_returns_error` | `b_lines.len() == 0` produces `ChernError::EmptyBundle` |
| 9 | `dimension_mismatch_returns_error` | wrong-dim H produces `ChernError::DimensionMismatch` |

### Hostile-review additions (6, all passing)

| #  | Name | Finding | What it verifies |
|----|------|---------|------------------|
| 10 | `test_bianchi_identity_line_bundle` | F2 | Abelian Bianchi `dF = 0` on rank-1 `O(2, 1)`; pins gauge collapse `F ≈ 0` AND closure of the (2,1)-projection `∂_c F_{ab̄} − ∂_a F_{cb̄} = 0`. A 10%-level sign / factor bug would saturate ~1e-1 here vs. ~3e-9 floor. |
| 11 | `test_c2_v_coarse_cross_check_line_bundle` | F3 | Coarse `c_2`-style integrand `Σ_p w_p · |F|²` on rank-1 line bundle should be ~ Tikhonov-noise-floor; pins absence of order-of-magnitude bug in F formula. |
| 12 | `test_c2_v_coarse_cross_check_distinct_degree` | F3 | Coarse `c_2`-style integrand on distinct-degree non-trivial monad should be finite, non-zero, and bounded — guards against pathological `h^{-1}` blow-up. |
| 13 | `test_schoen_three_factor_line_bundle` | F4 | `b_lines_3factor: Some(_)` produces curvature bit-identical to the 2-factor projection (matches `hym_hermitian.rs:311` source-of-truth). |
| 14 | `test_nan_input_returns_nan_curvature` | F5 | NaN-poisoned `H` matrix produces either `AllSamplesDegenerate` error or every-sample-flagged success; no NaN leakage into curvature data. |
| 15 | `test_singular_inversion_bookkeeping` | F1 | `n_singular_inversions` + `n_degenerate_samples` = `degenerate_sample_indices().len()`; sorted, in-range, zero-filled, `is_sample_degenerate(...)` consistent. |

### Hostile-review fixes implemented (in chern_curvature.rs)

* **Finding 1 (HIGH)** — Identity-fallback silent corruption removed.
  `hermitian_inverse_local` now returns `Option<Vec<Complex64>>` and
  signals hopeless singularity via `None` rather than the previous
  `h^{-1} = I` substitution.  The driver maps `None` → zero-fill +
  `n_singular_inversions += 1` + push to `degenerate_indices`.  New
  public accessors: `degenerate_sample_indices()` and
  `is_sample_degenerate(sample)`.  M1/M2 consumers MUST filter on this
  list before quadrature.
* **Finding 2 (HIGH)** — Bianchi identity numerical test added (rank-1
  abelian case).  Test 10.
* **Finding 3 (HIGH)** — Coarse `c_2`-style cross-check on both rank-1
  line bundle and rank-`n` distinct-degree monad.  Tests 11 + 12.
* **Finding 4 (MEDIUM)** — 3-factor Schoen line bundle test added.
  Test 13.
* **Finding 5 (MEDIUM)** — NaN propagation test added (Test 14) +
  defensive guards in driver: `h_v.h_coefficients` non-finite check
  before iteration; per-sample `s_vec` / `ds_vec` non-finite check.
* **Finding 6 (MEDIUM)** — Module-level docstring expanded with
  explicit sign-convention discussion: `F = ∂̄(h^{-1}∂h)` (no
  `i / 2π` prefactor) → gauge-Hermiticity `(hF_{ab̄})† = hF_{bā}` (no
  minus); under textbook `(i/2π)F` convention the End(V)-anti-
  Hermiticity reads `conj(F^I_{J,ab̄}) = -F^J_{I,bā}`.
* **Finding 7 (LOW)** — Chart-coupling note added to module docstring:
  evaluator assumes canonical FS-section chart matching
  `hym_hermitian::eval_section_b`; if the latter ever switches to the
  max-`|Z|` chart, this module breaks silently.
* **Finding 8 (LOW)** — Cholesky wire-up deferred as M0 follow-up
  (optimisation, not correctness).

## Math derivation correctness

### Recasting verified

The user task and G1 spec note that "in the holomorphic frame the Chern
connection's (0,1)-part is identically zero." This is **correct**: for
a Hermitian holomorphic bundle `(V, h)` in a holomorphic frame, the
Chern connection 1-form is

```
θ = h^{-1} ∂h         ⇒  θ^{(1,0)} = h^{-1} ∂h,  θ^{(0,1)} = 0
```

(see Griffiths-Harris Ch. 0 §5: the unique connection compatible with
both `h` and the holomorphic structure has `(0,1)`-part equal to `∂̄`
of the section, which in a holomorphic frame is zero by definition).

The Chern curvature is then purely (1,1):

```
F = dθ + θ ∧ θ = ∂̄(h^{-1} ∂h)         (mixed terms ∂(h^{-1}∂h) and (h^{-1}∂h) ∧ (h^{-1}∂h) cancel)
  = h^{-1}(∂̄∂h) − h^{-1}(∂̄h) h^{-1}(∂h)
```

This is what M0 evaluates. **No deviation** from the M1+M2 spec.

### Hermiticity invariant — gauge-correct version

A subtlety we discovered while writing tests: the naive index-swap
invariant

```
conj(F_{a b̄, I J}) = F_{b ā, J I}
```

holds only in a **unitary frame** (`h = I`). The section-basis frame
used here is not unitary. The correct gauge-invariant statement is

```
(h · F_{a b̄})^†  =  h · F_{b ā}      ⇔    conj((h F_{ab̄})_{IJ}) = (h F_{bā})_{JI}
```

We expose **both** checks: `check_index_swap_symmetry()` (the unitary-
frame form, used for trivial-bundle and rank-1 sanity) and
`check_hermiticity_with_h(h_at)` (the gauge-invariant form, used for
non-trivial section-basis bundles). The diagnostic struct reports the
unitary-frame violation; M2 callers should use the `_with_h` variant.

### c_2(V) cross-check

**Coarse sanity-only at M0** (full cohomological pairing remains M2
territory).  Tests 11 + 12 sum
`Σ_p w_p · |F^{(1,1)}_{a ā}|²` over the sample cloud as an
order-of-magnitude proxy for `tr(F ∧ F)`.  This is **not** the
cohomological `c_2(V) · [J]` — the proper wedge-and-trace contraction
against `J^{n-2}` belongs to M2 — but it catches the kind of bug local
Hermiticity tests miss (sign flip, missing `h^{-1}`, factor-of-2 in
the cross term).

The full c_2 closed-form is covered separately by
`chern_field_strength::integrate_tr_f_squared_wedge_J` (cohomological
pairing path) and `integrate_tr_f2_metric` (Monte-Carlo metric path);
M2 will tie M0's `F^{(1,1)}` to those at the proper integration
level.

## What M1 + M2 consume

- **M1** (`one_form_basis.rs`): does *not* directly consume the
  curvature cloud. M1 enumerates `(0,1)`-form seeds and projects to the
  `∂̄_V`-closed subspace; the closure projector uses the *flat* `∂̄`,
  not the curvature-twisted `∂̄_V`, because in the holomorphic frame
  `θ^{(0,1)} = 0`. M0 is M1-independent.

- **M2** (`twisted_dirac.rs`): consumes
  - `ChernCurvatureCloud::matrix_slice(sample, a, b)` for every
    sample × direction pair, when assembling the curvature-coupling
    matrix elements
    `<ψ_α | (i Λ_ω F^{(1,1)}) ψ_β>` =
    `Σ_p w_p · g^{ā b}(p) · ψ_α(p)^* · F^{(1,1)}_{a b̄}(p) · ψ_β(p)`.
  - `metric_trace_cloud(cloud, g_inv_cy3)` as a convenience for the
    above contraction.
  - `curvature_diagnostics()` for the `i Λ_ω Tr(F)` ≈ 0 SU(n)
    convergence diagnostic.

## Effort vs. M1+M2 spec estimate

| Spec estimate | Actual |
|---------------|--------|
| ~400 LOC (file body) | ~600 LOC body + ~230 LOC tests = 830 LOC |
| 4 tests | 9 tests |
| 1.5 weeks | 1 working session |

The spec underestimated test surface area, and the gauge-Hermiticity
discovery required additional methods (`check_hermiticity_with_h`,
`check_index_swap_symmetry`). LOC budget overage is concentrated in:
- Inline-documented per-step formulas (mathematical clarity over
  brevity).
- Local `hermitian_inverse_local` to avoid coupling to
  `hym_hermitian`'s private API (~80 LOC).
- Two Hermiticity-check variants (unitary frame + gauge-invariant)
  rather than one.
- Test fixtures for distinct-degree, AKLP, and trivial bundles.

## Risks and known limitations

1. **Section gauge degeneracy (per AKLP convention).** `eval_section_b`
   in `hym_hermitian.rs` picks the same lowest-weight monomial
   (`z_0^{d_1} w_0^{d_2}`) for every B-summand of the same bidegree.
   When the bundle has duplicates (AKLP `B = O(1,0)^3 ⊕ O(0,1)^3`),
   `h(p) = S(p) H S(p)†` is rank-deficient and `F^{(1,1)}` washes out
   to ~1e-8 under Tikhonov regularisation. This is **inherent to the
   AKLP gauge**, not a defect in M0; it is pinned in test 4 so future
   refactors that adopt a richer section basis will visibly break the
   test and force re-pinning.

2. **CY3 ambient coordinates `N_AMBIENT_COORDS = 8`.** Hard-coded match
   to `MetricBackground::sample_points` returning `[Complex64; 8]`.
   Schoen `CP² × CP² × CP¹` has 3+3+2 = 8 coords; Tian-Yau bicubic
   `CP³ × CP³` has 4+4 = 8. If a future ambient changes this, the
   constant must be updated in lockstep with `MetricBackground`.

3. **3-factor lift not consumed.** The `b_lines_3factor` field of
   `MonadBundle` (Schoen Z/3×Z/3 BHOP rank-4 path) carries the
   `CP^1`-fibre direction not visible in the 2-factor `b_lines`
   projection. M0 follows `hym_hermitian.rs:311` and reads only the
   2-factor projection, matching the source-of-truth convention. The
   3-factor lift is needed only for downstream Wilson-line phase-class
   bookkeeping, not for the curvature evaluator.

4. **AKLP convergence floor (~1e-2).** The M1+M2 spec notes that AKLP
   2010 §3 / Fig. 1 reports residual HYM error of ~1e-2 at k=4. This
   propagates into the M0 curvature cloud as ~1e-2 relative noise,
   which M2 must quantify per-degree rather than blame on its own
   numerics.

## References

- Griffiths, P., Harris, J., *Principles of Algebraic Geometry* (Wiley,
  1978), Ch. 0 §5.
- Anderson, J., Karp, R., Lukas, A., Palti, E., arXiv:1004.4399 (2010),
  §3.
- Anderson, J., Constantin, A., Lukas, A., Palti, E., arXiv:1707.03442
  (2017), §4.2.
- Bochner, S., Yano, K., *Curvature and Betti Numbers* (Princeton,
  1953), §1.
- Donagi, R., He, Y.-H., Ovrut, B., Reinbacher, R.,
  arXiv:hep-th/0512149 (2006).

---

## Addendum (May 2026): FS-projective fix

**Status:** Resolved.  M2 PHASE A audit at commit `2ddfb623`
discovered that `evaluate_chern_curvature` returned `F^{(1,1)} ≡ 0`
identically for any rank-1 line bundle in the canonical section gauge
with `H = I`.  This blocked the bundle-twist discrimination signal
from manifesting and forced three PHASE A tests into `#[ignore]`.

### Root cause

The original implementation built `h_{αβ}(p) = s_α H_{αβ} s_β*`
directly from the un-normalised polynomial sections.  This is the
polynomial outer product on the affine cone over `CP³ × CP³`, **not**
a Hermitian fibre metric on `O(b)`.  Consequently
`log h = b₁·log|z₀|² + b₂·log|w₀|²` is purely holomorphic +
antiholomorphic, so `F = ∂̄∂ log h ≡ 0`.

### Fix (option (b) — local to `evaluate_chern_curvature`)

Replace the un-normalised section value with its FS-projective-
trivialised counterpart `s̃_α(p) = ψ_α(p) · s_α(p)` where
`ψ_α(p) = ρ_z^{-b_α[0]/2} · ρ_w^{-b_α[1]/2}` is the canonical
unit-norm trivialisation of `O(b_α)` on `CP³ × CP³`.  The resulting
fibre metric `h(p) = S̃ H S̃†` produces
`F = -b·∂̄∂ log ρ = +b·ω_FS` (after the standard Kähler-form sign),
verified pointwise against the analytic prediction in
`rank_one_o10_curvature_is_fs_metric` and
`test_bianchi_identity_line_bundle`.

### Why option (b), not (a)

`eval_section` is bit-identical to `hym_hermitian::eval_section_b`,
which drives the AKLP T-operator iteration.  AKLP 2010 §3 specifies
the polynomial-monomial section basis (un-normalised) with the
FS-projective measure absorbed into `dvol`.  Modifying `eval_section`
would silently break the HYM solve and force every consumer to track
antiholomorphic derivatives (since `s̃` is non-holomorphic).

### Implementation

* New helper `eval_fs_projective_basis` returns four arrays per
  sample: `s̃_α`, `∂_k s̃_α`, `∂̄_k s̃_α`, `∂̄_a ∂_b s̃_α`.  Closed-form
  polynomial-+-rational, no finite difference.
* `evaluate_chern_curvature` rewrites `∂_b h`, `∂̄_a h`, `∂̄_a ∂_b h`
  to use the *full* product rule (since `s̃_α` carries antiholomorphic
  dependence).
* Origin-of-the-cone degeneracies (`ρ = 0` with `b > 0`) flagged
  through the existing `n_degenerate_samples` counter.
* `eval_section` and `hym_hermitian.rs` are untouched; HYM
  convergence verified intact (9/9 hym_hermitian tests pass).

### Test impact

Updated tests in `chern_curvature.rs` (15/15 pass after fix):

* `rank_one_o10_curvature_is_diagonal_scalar` →
  `rank_one_o10_curvature_is_fs_metric`.  Pointwise analytic match
  `F = -d · g_FS`.
* `test_bianchi_identity_line_bundle` rewritten: closed-form match
  to `-d_i · g_FS` on each block; abelian Bianchi closure follows.
* `test_c2_v_coarse_cross_check_line_bundle` rewritten: pins
  *non-zero* lower bound on the c_2-style integrand.
* `aklp_bundle_canonical_gauge_curvature_near_zero` →
  `aklp_bundle_canonical_gauge_curvature_is_finite_and_nonzero`.
  Pins finiteness, non-zero norm, gauge-Hermiticity.
* `h_at_sample` test helper updated to FS-projective trivialisation.

Activated tests in `bundle_twisted_dbar.rs`:

* `test_h1_monotonic_slope_in_degree` active and passes.

Re-ignored with updated rationale (consumer-side mismatch):

* `test_h1_o11_line_bundle_kernel_drops_below_flat`
* `test_h1_bundle_twist_signal_present`

These assert `n_twisted_kernel < n_flat_kernel` where both are
column ranks of `Φ` and `Φ_aug = [Φ; slope·Φ]`.  Stacking rows is a
column-rank-non-decreasing operation, so the assertion is
structurally unattainable with the current
`augment_phi_with_curvature` consumer logic.  PHASE A test 8b at
line ~1255 already notes the bundle-twist signal lives in the
cohomology quotient `Φ_aug ∩ B¹_aug`, not raw column rank.
Out-of-scope for the M0-fix; flagged for PHASE A author redesign.

### Latent consumer-side bug fix in `bundle_twisted_dbar.rs`

`compute_h1_twisted` extracted the slope as `.re` of
`slope_per_sample[k] = i · Σ_a F_{aā}`.  With the (now-correct)
Hermitian F, the trace is real, so `i · Σ F` is purely imaginary
and `.re` returned 0.  Pre-fix this was masked because F was 0
anyway.  Changed to `.im` so `mean_slope_shift` correctly reports
the Hermite-Einstein constant.
