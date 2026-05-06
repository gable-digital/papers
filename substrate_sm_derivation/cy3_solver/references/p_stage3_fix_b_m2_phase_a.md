# Stage 3 Fix B M2 PHASE A — bundle-twisted ∂̄_V on a line bundle

**Status:** PHASE A landed (algorithmic infrastructure). PHASE A's
bundle-twist target tests are gated on a separately-flagged M0 finding;
PHASE B and PHASE C remain explicitly deferred.

**Module:** [`src/route34/bundle_twisted_dbar.rs`].

**Tests (in-module unit):**
* `test_h1_trivial_line_bundle_matches_flat_dim` — passes.
* `test_h1_rank_3_bundle_rejected` — passes.
* `test_h1_hym_dim_mismatch_rejected` — passes.
* `test_h1_empty_bundle_and_no_samples` — passes.
* `test_h1_representatives_well_formed` — passes.
* `test_augmentation_nonzero_slope_increases_rank` — passes.
* `test_augmentation_preserves_rank_for_independent_cols` — passes.
* `test_h1_o11_line_bundle_kernel_drops_below_flat` — `#[ignore]` (M0 gauge finding).
* `test_h1_monotonic_slope_in_degree` — `#[ignore]` (M0 gauge finding).
* `test_h1_bundle_twist_signal_present` — `#[ignore]` (M0 gauge finding).

**Cross-references:**
* `references/p_stage3_fix_b_m1m2_detailed_spec.md` — full M1+M2 spec.
* `references/p_stage3_fix_b_m0_chern_curvature.md` — M0 reference.
* `references/p_stage3_fix_b_m1_zero_one_form_basis.md` — M1a reference
  (the module M2 PHASE A consumes).

## Scope

PHASE A delivers the **algorithmic infrastructure** for the
bundle-twisted Dolbeault operator `∂̄_V` on a rank-1 line bundle. The
operator consumes M0's [`ChernCurvatureCloud`] through the Bochner-
Weitzenböck identity:

```
Δ_{∂̄_V} ψ  =  Δ_{∂̄} ψ  +  i Λ_ω F^{(1,1)} · ψ
```

The `H^1(X, V)` kernel is computed by augmenting the M1a
sample-evaluation column-space decomposition with a curvature-weighted
row block — a simultaneous-kernel formulation that is equivalent to
diagonalising the augmented operator and identifying modes annihilated
by both the flat ∂̄ and the curvature multiplication.

## What PHASE A does NOT deliver

* **PHASE B (deferred):** Schoen `Z/3 × Z/3` quotient on sample points
  (CICY constraint enforcement + orbit identification).
* **PHASE C (deferred):** BHOP §6 extension-bundle adapter; proper
  `(W_1, W_2)` Cartan-vector Wilson-line character action.

These are tracked in the M1+M2 detailed spec and are independent
follow-up work.

## Finding: M0 rank-1 line-bundle gauge triviality

While wiring PHASE A we discovered that M0's
[`ChernCurvatureCloud::evaluate_chern_curvature`], in the canonical
section gauge with `H = I` and a single-summand rank-1 line bundle
`O(d_1, d_2)`, returns `F^{(1,1)} ≡ 0` identically. The reason:

```
h(p)              =  |s(p)|²              (s = z_0^{d_1} w_0^{d_2}, scalar)
h^{-1} ∂_b h(p)   =  ∂_b s / s            (a holomorphic function)
∂̄_a (∂_b s / s)   =  0                    (no z̄ dependence)
⇒ F^{(1,1)}       =  0                    identically.
```

This is the **section-gauge-trivial** statement: a rank-1 line bundle's
Chern connection in the holomorphic-section gauge is `θ = ∂ log h`; its
curvature is `∂̄ θ = ∂̄ ∂ log h`, which on the canonical FS section
`s = z_0^d` gives `log|s|² = d · log|z_0|²` (no anti-holomorphic
component outside `z_0 = 0`). The genuine FS line-bundle curvature
`F = d · ω_FS` lives in the projective Fubini-Study Kähler form, which
requires the projective denominator `|z|² = Σ_i |z_i|²` — a
normalisation `eval_section` does not apply.

M0's pre-existing test `rank_one_o10_curvature_is_diagonal_scalar`
([`chern_curvature.rs:1140-1194`]) actually pins this gauge triviality
(asserts `F^{(1,1)} = 0` for `O(1, 0)`).

### Implication for PHASE A

With the current M0, a rank-1 line bundle's curvature cloud is
identically zero, the Bochner shift `i Λ_ω F` is zero, and the
bundle-twisted kernel equals the flat-∂̄ kernel — NO bundle-twist
signal is observable on rank-1 line bundles.

The fix is in M0, not PHASE A. Per task constraints ("If you find a
real bug in M0 or M1a while wiring M2, FLAG (don't restructure those
modules)"), this finding is documented but not addressed at PHASE A.
The fix would be one of:

  (a) Upgrade M0 to compute `F` with the FS-projective normalisation
      factor `|z|^{-2(d_1 + d_2)}` (matching the standard FS line-bundle
      metric `h_FS(p) = |s|² / |z|^{2 d}`), OR
  (b) Test the bundle-twist signal at rank ≥ 2 with a section basis
      whose `H` matrix is non-trivial (no rank-1 short-circuit). PHASE
      C's BHOP adapter naturally provides this.

PHASE A's algorithm is correct as-written; the bundle-twist tests
will light up automatically when one of (a) or (b) is implemented.

## What PHASE A does deliver

1. **Bundle-twisted Dolbeault driver** [`compute_h1_twisted`] —
   accepts a rank-1 [`MonadBundle`], a `MetricBackground`, an
   `HymHermitianMetric`, and a [`ChernCurvatureCloud`]; returns an
   [`H1TwistedBasis`] with H¹ representatives in the raw
   (0,1)-form-basis-coefficient form, plus diagnostic ranks for the
   flat-∂̄ kernel (M1a's value), the bundle-twisted kernel (PHASE A's
   value), and the mean slope shift `i Λ_ω F`.

2. **Curvature-row augmentation** `augment_phi_with_curvature` —
   builds the augmented (Φ ; iΛF · Φ) matrix that implements the
   simultaneous-kernel formulation of Bochner-Weitzenböck.

3. **Trivial-bundle short-circuit verification** — pinned by
   `test_h1_trivial_line_bundle_matches_flat_dim`. For
   `O(0, 0)`, `F ≡ 0`, augmentation block is zero, and
   `n_twisted_kernel = n_flat_kernel` exactly.

4. **Algorithmic-internal correctness** — pinned by
   `test_augmentation_*` tests that exercise the augmentation logic
   on synthetic non-zero slopes (independent of M0).

5. **Error path coverage** — empty-bundle, no-samples, HYM-dim
   mismatch, non-rank-1 (current scope guard). All return
   `TwistedDbarError::*` variants.

6. **Forward-compatible algebra** — when M0 is upgraded to compute the
   FS-projective curvature, the existing PHASE A code requires no
   changes. The `#[ignore]`'d tests will automatically light up.

## Public API

```rust
pub fn compute_h1_twisted(
    bundle: &MonadBundle,
    metric: &dyn MetricBackground,
    h_v: &HymHermitianMetric,
    curvature: &ChernCurvatureCloud,
    polynomial_degree: u32,
    rank_tolerance: f64,
) -> Result<H1TwistedBasis, TwistedDbarError>;

pub struct H1TwistedBasis {
    pub representatives: Vec<Vec<Complex64>>,
    pub n_raw_basis: usize,
    pub n_flat_kernel: usize,
    pub n_twisted_kernel: usize,
    pub mean_slope_shift: f64,
    pub rank_tolerance: f64,
}

pub enum TwistedDbarError {
    EmptyBundle,
    DimensionMismatch { what, expected, got },
    BundleHymMismatch { bundle_n, hym_n },
    NonRankOneNotSupported { rank },
    LinAlgFailure,
    NoSamples,
}
```

## Algorithm summary

1. Build raw (0,1)-form basis Φ at polynomial degree `d`.
2. Build raw (0,0)-form coboundary basis B¹ at polynomial degree `d`.
3. Compute per-sample slope `i Λ_ω F` from the curvature cloud:
   `slope[k] = Σ_a i · F_{aa, 00}(p_k)` (rank-1 single matrix entry).
4. Augment Φ: rows [0..n_amb) = flat Φ; rows [n_amb..2 n_amb) =
   `slope[k] · Φ` (curvature-weighted copy).
5. Augment B¹: rows [0..n_amb) = flat B¹; rows [n_amb..2 n_amb) = 0
   (the (0,0)-form coboundary in the holomorphic frame has zero
   curvature coupling at the (0,1)-form level).
6. Compute orthonormal column-space bases of Φ_aug and B¹_aug via
   thin Gram-Schmidt (small problem sizes for PHASE A; M1a's Lanczos
   path is the production solver for larger problems).
7. Project each Φ_aug column-basis vector onto B¹_aug; orthogonalise
   against accepted H¹ representatives; accept if residual norm
   exceeds `rank_tolerance`.
8. Convert accepted H¹ reps from sample-value space to raw-(0,1)-basis
   coefficient form via Tikhonov-regularised normal-equations
   spectral solve.

## Honest delivery statement

PHASE A is an **infrastructure milestone**, not a science result. The
correctness-pinned tests verify the algorithm is plumbed correctly
(trivial-bundle short-circuit, augmentation logic, error paths,
output well-formedness). The science-pinned tests (Bott-formula
match, monotonic slope-in-degree, bundle-twist signal-correlates) are
`#[ignore]`'d with explicit cross-reference to the M0 gauge-triviality
finding that prevents them from passing at this scope.

The bundle-twisted Dolbeault operator is now in place. Its scientific
output for non-trivial cases is gated on M0's curvature evaluation
correctly capturing the FS line-bundle curvature — an M0 follow-up,
not a PHASE A re-do.

## What lights up after PHASE B

PHASE B applies the Schoen `Z/3 × Z/3` quotient to the sample cloud.
The resulting `MetricBackground` is on the quotient `X/Γ`, so M0's
curvature evaluation runs on the quotient sample points. The
bundle-twist signal still requires the M0 FS-projective fix to
manifest; PHASE B alone does not unlock the science tests.

## What lights up after PHASE C

PHASE C provides the BHOP §6 extension-bundle adapter. The bundle is
no longer a single line bundle but an SU(4) extension whose section
basis is naturally rank ≥ 2 with non-trivial `H`. The M0 gauge
triviality does NOT apply at rank ≥ 2 with non-identity `H` (see
[`chern_curvature.rs::aklp_bundle_curvature_satisfies_gauge_hermiticity`]
which exercises non-trivial `F` for the AKLP rank-6 bundle), so
PHASE C automatically unlocks the bundle-twist signal even without
M0's FS-projective fix.

## References

* Anderson, J., Karp, R., Lukas, A., Palti, E., "Numerical
  Hermitian-Yang-Mills connections" (AKLP), arXiv:1004.4399 (2010).
* Anderson, J., Constantin, A., Lukas, A., Palti, E., "Yukawa
  couplings in heterotic Calabi-Yau models" (ACLP),
  arXiv:1707.03442 (2017), §4.
* Griffiths, P., Harris, J., *Principles of Algebraic Geometry*
  (Wiley, 1978), Ch. 0 §5: Chern connection on Hermitian holomorphic
  vector bundle.
* Bott, R., "Homogeneous vector bundles", *Ann. Math.* **66** (1957),
  203–248: Bott formula for `H^q(CP^n, O(d))`.
* M0 reference: `references/p_stage3_fix_b_m0_chern_curvature.md`.
* M1a reference: `references/p_stage3_fix_b_m1_zero_one_form_basis.md`.
* M1+M2 detailed spec: `references/p_stage3_fix_b_m1m2_detailed_spec.md`.
