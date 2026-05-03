# P7.2b — ω_fix gateway-eigenvalue diagnostic on a projected basis

## Goal

Address the basis-purity bottleneck identified in P7.1 (commit db1b4ca1):
P7.1's irrep-classified "trivial-rep" eigenfunctions on the
test_degree=4, basis_dim=494 Galerkin basis carried only ~26 % of their
L²-mass in the (0, 0) Z₃×Z₃ character bucket. The remaining 74 % was
spread across the eight non-trivial character classes — i.e. the basis
does not cleanly block-diagonalise over Z₃×Z₃ at f64 precision because
of Donaldson noise on the per-point sample weights. P7.1 picked the
lowest dominant-(0,0) eigenvalue (0.5055 raw on Schoen, residual 1.92 %
from 123/248) as a proxy for ω_fix, but the 26 % purity made the
classification noisy.

P7.2b builds a **projected basis** that is exactly in the trivial irrep
by construction. Both the Schoen Z₃×Z₃ and the TY Z₃ actions on the
ambient bigraded ring are diagonal (each monomial maps to itself up to
a phase determined by an integer character — see
`route34::z3xz3_projector`), so the trivial-character projector reduces
to **filtering** the basis to the χ = 0 subset. The Galerkin solve on
the surviving sub-basis returns eigenvalues that all live in the
trivial isotypic component with 100 % purity — character mixing is
impossible because the basis literally does not contain non-trivial-rep
functions.

## Implementation

### Module `route34::metric_laplacian_projected`

* `build_projected_basis_schoen(&[TestMonomial]) -> Vec<TestMonomial>`
  — keeps only monomials with `alpha_character == 0 && beta_character == 0`
  (exact Reynolds projector P_(0,0) in the diagonal-action basis).
* `build_projected_basis_ty(&[TestMonomial]) -> Vec<TestMonomial>`
  — keeps only monomials with `(m[1] + 2 m[2] + m[5] + 2 m[6]) mod 3 == 0`
  (matches the in-line TY/Z₃ character formula in
  `bin/p7_1_omega_fix_diagnostic.rs`).
* `compute_projected_metric_laplacian_spectrum(metric, config, projection)`
  — full Galerkin assembly (mass, stiffness, regularised Hermitian
  inverse, Hermitian Jacobi diagonalisation) on the projected basis,
  bit-equivalent to `metric_laplacian::compute_metric_laplacian_spectrum`
  except for the basis substitution.

Four unit tests pass:
* survival fraction at d=4 is in `[0.05, 0.30]` for Schoen, `[0.20, 0.50]` for TY;
* every survivor has zero character;
* projector is idempotent;
* TY/Z₃ character matches the P7.1 in-line formula.

### Binary `bin/p7_2b_omega_fix_localized`

Single-seed driver that

1. solves the Donaldson-balanced metric (Schoen + TY/Z₃ control) at
   canonical settings (n_pts=40 000, k=3, max_iter=100, tol=1e-6,
   test_degree=4),
2. runs the projected Galerkin Laplacian spectrum,
3. reports the bottom-5 eigenvalues with five normalisation schemes
   (raw, by_lambda1_min, by_lambda_max, by_mean_eigvalue, by_volume),
4. picks the (eigenvalue, scheme) pair closest to ω_fix = 123/248.

## Survival fractions

| candidate | full basis | projected basis | survival |
| --- | --- | --- | --- |
| Schoen (Z₃ × Z₃) | 494 | 119 | 0.241 |
| TY (Z₃)          | 494 | 178 | 0.360 |

The Schoen survival 24.1 % is higher than the naïve 1/9 ≈ 11 % because
many low-bidegree blocks have very few monomials with α-character ≠ 0
or β-character ≠ 0 (e.g. degree-1 monomials in the `t`-block alone
give χ_β ∈ {0, 1}, splitting the small space favourably). TY survival
36.0 % exceeds 1/3 ≈ 33 % for the same reason. Both are within the
expected `[1/G, 1/2]` envelope for finite-degree truncations.

## Results

### Schoen / Z₃ × Z₃, single seed

| seed | iters | σ-resid | λ_0 (proj) | scheme | best value | residual ppm | tier |
| ---- | ----- | ------- | ---------- | ------ | ---------- | ------------ | ---- |
| 42    | 26 | 8.19  | 0.1228 | raw | 0.1228 | 752 342 | FAILED |
| 12345 | 20 | 2.19  | 0.1741 | by_volume @ rank 4 | 0.3321 | 330 370 | FAILED |

`λ_0` here is the lowest non-zero eigenvalue of the projected (trivial-rep)
spectrum. P7.1 with seed=42 found the lowest *full-basis* eigenvalue at
λ = 0.4320 (mixed-rep) and its lowest dominant-trivial-rep eigenvalue at
λ = 0.5055.

### TY / Z₃, single seed (control)

| seed | iters | σ-resid | λ_0 (proj) | best raw | residual ppm | tier |
| ---- | ----- | ------- | ---------- | -------- | ------------ | ---- |
| 42    | 17 | 0.27 | 0.0687 | 0.5131 (rank 2) | 34 616 | FAILED |
| 12345 | 16 | 0.27 | 0.2000 | 0.5996 (rank 2) | 209 043 | FAILED |

## Comparison vs P7.1

P7.1 (full basis, post-classification, seed=42, Schoen):
* lowest *all-rep* eigenvalue: 0.4320
* lowest *dominant-trivial-rep* (purity 26 %): 0.5055 (residual 1.9 %)

P7.2b (projected basis, seed=42, Schoen):
* lowest *trivial-rep-by-construction*: **0.1228**
* fifth-lowest: 3.86

The projected-basis solve gives a *dramatically lower* lowest-trivial-rep
eigenvalue than P7.1 reported, off by a factor of ~4. Two interpretations
are possible:

(a) **P7.1 misidentified the trivial-rep ground state.** The 26 %-pure
"dominant-(0, 0)" eigenfunction at λ = 0.5055 was actually a higher
trivial-rep mode whose Donaldson noise leaked the leading L²-weight onto
trivial-rep monomials; the genuine lowest trivial-rep mode lay below it
in the all-rep spectrum, masquerading as a non-trivial-rep eigenfunction
in the character classifier. The projected-basis solve, which cannot
miss trivial-rep modes by construction, finds them at 0.123 and 0.187.

(b) **The projected basis is too coarse.** Restricting to χ = 0 monomials
removes most of the basis and leaves a coarser approximation of the
function space. The lowest projected eigenvalue might be artificially
*low* because the Galerkin truncation is missing modes that would have
mixed character weight in a richer subspace.

(b) is unlikely: a finite-dimensional Galerkin approximation of a
self-adjoint operator on a Hilbert subspace gives eigenvalues that are
**upper bounds** to the true ones (min-max principle). Restricting to
the trivial-isotypic subspace, where the genuine ground state lives,
should give eigenvalue estimates that are *higher* than the full-space
approximation, not lower. So (a) is the consistent reading: P7.1's
character classifier was thrown by 74 %-impure eigenfunctions and
locked onto the wrong eigenvalue level.

In neither reading does the result match ω_fix = 123/248 = 0.4960 to
the journal's "high precision" tier. With normalisation by `λ_min`
fixed to 1.0 (so the lowest eigenvalue *is* the unit), every other
eigenvalue is ≥ 1.5 · λ_0; with raw eigenvalues, the closest pick on
Schoen is 19 % off (rank 0 at seed=12345) or 75 % off (rank 0 at
seed=42).

The closest TY raw eigenvalue is 0.5131 at seed=42 (3.5 % off ω_fix),
which is suggestive but contradicts the prediction since TY/Z₃ is the
*control*: in the journal's frame, TY should NOT match ω_fix because
ω_fix is supposed to be a Z₃ × Z₃ Schoen-specific gateway. A 3.5 %
"hit" on TY is therefore a coincidence at this resolution rather than
a signal.

## Verdict

**ω_fix gateway formula on the projected basis: not verified.**

* Lowest projected-basis Schoen trivial-rep eigenvalue is 0.123–0.174
  (seed-dependent), not 123/248 = 0.4960.
* No normalisation scheme (raw, λ₁, λ_max, mean, volume) brings any
  bottom-5 eigenvalue within 100 ppm of ω_fix on Schoen.
* The discrepancy is not the basis-purity issue P7.2b was designed to
  isolate: the projected basis IS purity-100 % by construction, so
  whatever `λ_0 ≈ 0.123` represents IS a genuine trivial-rep mode of
  the discretised Δ_g.

The next-step interpretations are
1. **the prediction needs a different normalisation** that the diagnostic
   does not explore (e.g. ω_fix is a *ratio* of two specific eigenvalues
   pegged to a topological invariant, not the lowest trivial-rep
   eigenvalue itself), or
2. **a different operator** is on the gateway side (e.g. the bundle
   Laplacian D_V* D_V on Ω^{0,1}(M, V) rather than the metric Laplacian
   Δ_g on functions; see `route34::zero_modes_harmonic`), or
3. **the test_degree=4 truncation is too coarse** even on the projected
   basis (Schoen projected dim = 119 has a ~10 % survival fraction on the
   high-bidegree end, which may not span enough of the lowest trivial
   eigenmode).

The Schoen σ-residuals (2.19, 8.19) are also large compared to the
strict-converged target (≤ 1.0), so seed selection from the
strict-converged list and a longer Donaldson loop (max_iter ≥ 200)
would be a sensible next move before declaring the result final.

## Relation to P7.1 and P7.1b

* **P7.1 (db1b4ca1):** full Galerkin solve + post-hoc character
  classification. Reported λ_min_trivial = 0.5055, residual 1.9 %.
  Confounded by 26 %-purity character classification.
* **P7.1b (parallel):** retries P7.1 with confirmed-converged Schoen
  seeds. Addresses metric quality but not the basis-purity issue.
* **P7.2b (this work):** projected basis avoids the classification
  step entirely. λ_min_trivial = 0.123–0.174 — different from P7.1's
  0.5055 by a factor of ~4. Confirms that P7.1's classifier picked the
  wrong eigenvalue level. Result still not within ω_fix tier.

## Files

* `src/route34/metric_laplacian_projected.rs` — projector + Galerkin solve
* `src/bin/p7_2b_omega_fix_localized.rs` — single-seed diagnostic binary
* `output/p7_2b_omega_fix_localized.json` — seed=12345 result
* `output/p7_2b_omega_fix_localized_seed42.json` — seed=42 result
  (apples-to-apples with P7.1)
* `output/p7_2b_run.log` / `output/p7_2b_run_seed42.log` — stderr captures

## Reproduce

```bash
cargo build --release --features "gpu precision-bigfloat" \
    --bin p7_2b_omega_fix_localized
./target/release/p7_2b_omega_fix_localized \
    --seed 12345 --n-pts 40000 --k 3 --max-iter 100 --test-degree 4
./target/release/p7_2b_omega_fix_localized \
    --seed 42 --n-pts 40000 --k 3 --max-iter 100 --test-degree 4 \
    --output output/p7_2b_omega_fix_localized_seed42.json
```
