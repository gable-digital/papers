# P8.3-followup-A — Upstream Zero-Mode Diagnostic

**Date:** 2026-04-30
**Binary:** `src/bin/p8_3_followup_a_zero_mode_diag.rs`
**Build:** clean (release, 13.98s incremental). Production binary `p5_10_ty_schoen_5sigma.exe` PID 2270068 still alive.

## Configuration

- TY metric, k=3, n_pts=200, seed=42, max_iter=50, donaldson_tol=1e-9.
- `MonadBundle::anderson_lukas_palti_example()` (B = O(1,0)³ ⊕ O(0,1)³, C = O(1,1)³).
- `AmbientCY3::tian_yau_upstairs()`.
- HYM: `HymConfig { max_iter: 8, damping: 0.5, .. }`.
- Harmonic solver invoked with `HarmonicConfig::default()`; observed empty kernel; retried with `kernel_dim_target = Some(9)` — same fallback `predict_fermion_masses_with_overrides` applies in production.

## Quantitative findings

| Metric | Value |
|---|---|
| TY σ residual at iter 50 | 7.50e-1 (UNCONVERGED — see hypothesis F below) |
| `seed_basis_dim` | 24 |
| `cohomology_dim_predicted` (BBW) | 9 |
| `cohomology_dim_observed` (default cfg) | **0** |
| `cohomology_dim_observed` (fixed-dim 9) | 9 |
| `orthonormality_residual` | 5.11e-10 |
| Coefficient-vector max off-diagonal `|cα·c̄β|` (a≠b) | 6.99e-16 |
| Average normalised raw correlation `|⟨ψα|ψβ⟩|/√(‖ψα‖²‖ψβ‖²)`, a≠b | **8.35e-2** |
| **Rank(Ψ_pts × modes) at floor σ/σmax > 1e-10** | **9 / 9 (FULL RANK)** |

### L = D*D eigenvalue spectrum (n_seeds=24)

```
λ[ 0..2] = 1.149e0, 1.152e0, 1.152e0
λ[ 3..5] = 1.246e0, 1.248e0, 1.248e0
λ[ 6..8] = 1.308e0, 1.313e0, 1.313e0
λ[ 9..11]= 1.406e0, 1.406e0, 1.411e0
λ[12..14]= 1.628e0, 1.628e0, 1.636e0
λ[15..17]= 1.916e0, 1.916e0, 1.917e0
λ[18..20]= 2.064e0, 2.064e0, 2.064e0
λ[21..23]= 2.456e0, 2.456e0, 2.460e0
```

No plateau-then-gap. Smallest λ = 1.149 ≫ 0.

### Singular values of Ψ (point-value matrix)

```
σ/σmax: 1.000, 0.960, 0.843, 0.790, 0.759, 0.729, 0.709, 0.616, 0.560
```

All within factor < 2 of each other. **No collapsed singular value.**

### Per-mode B-line support

```
mode 0: λ=1.149  b0:.33 b1:.33 b2:.33 b3:.01 b4:.01 b5:.01   (B(1,0)-only)
mode 1: λ=1.152  b0:.24 b1:.10 b2:.66 b3:.00 b4:.00 b5:.00   (B(1,0)-only)
mode 2: λ=1.152  b0:.43 b1:.56 b2:.01 b3:.00 b4:.00 b5:.00   (B(1,0)-only)
mode 3: λ=1.246  b0:.03 b1:.03 b2:.03 b3:.30 b4:.30 b5:.30   (B(0,1)-only)
mode 4: λ=1.248  b0:.00 b1:.00 b2:.00 b3:.17 b4:.67 b5:.16   (B(0,1)-only)
mode 5: λ=1.248  b0:.00 b1:.00 b2:.00 b3:.50 b4:.00 b5:.50   (B(0,1)-only)
mode 6: λ=1.308  b0:.29 b1:.29 b2:.29 b3:.04 b4:.04 b5:.04   (B(1,0)-only)
mode 7: λ=1.313  b0:.13 b1:.20 b2:.66 b3:.00 b4:.00 b5:.00   (B(1,0)-only)
mode 8: λ=1.313  b0:.53 b1:.46 b2:.00 b3:.00 b4:.00 b5:.00   (B(1,0)-only)
```

## Hypothesis triage

| Hyp | Survives? |
|---|---|
| (1) Galerkin Gram ill-conditioned | **No** — coefficient orthogonality is 7e-16 (machine precision); Gram is well-conditioned. |
| (2) Basis spans 1-D subspace | **No** — `seed_basis_dim = 24`, far larger than required 9. |
| (3) Eigenvector orthogonalisation broken | **No** — Jacobi orthogonality 7e-16; raw L² correlation between modes averages 8.4%, not >> 0; rank(Ψ) = 9/9. |
| (4) Z/3 quotient projector collapse | **No** (no Z/3 projector applied here — solver works upstream of quotient). |
| (5) Bundle structure has 1 zero mode | **No** — bundle BBW count = 9, observed kernel rank = 9. |

**ALL FIVE rank-1-collapse hypotheses from the hostile review are FALSIFIED.** The upstream zero-mode solver is producing 9 linearly independent modes.

## Surviving root causes (NEW)

### (F) The L=D*D spectrum is uniformly lifted off zero (no kernel)

`HarmonicConfig::default()` returns 0 modes because the smallest eigenvalue is 1.149 — three orders of magnitude above the `kernel_eigenvalue_ratio = 1e-3` cutoff times the largest eigenvalue (2.46), meaning every direction is "non-kernel". The pipeline only recovers 9 modes by overriding via `kernel_dim_target = Some(9)` (i.e. "take the lowest 9 eigenvalues regardless of magnitude"). At k=3, with σ_residual = 0.75 (metric NOT converged — k=3 should drive σ → 1e-2), the lowest 9 "kernel" modes are not actually harmonic representatives of H¹(M, V); they are the 9 lowest-eigenvalue non-zero modes of an ill-conditioned operator. These are NOT physical zero modes.

### (G) Wilson-line phase classification collapses to 2 classes, not 3

All 9 modes have B-summand support entirely within either `[1,0]` lines (modes 0,1,2,6,7,8 — 6 modes) or `[0,1]` lines (modes 3,4,5 — 3 modes). With `cartan_phases[6]*b[0] + cartan_phases[7]*b[1]`, only TWO distinct phase classes exist for these b-vectors. With Z/3 quotient, this means at most 2 of {class 0, class 1, class 2} are populated. `assign_sectors_dynamic` line 183 detects this and triggers the round-robin fallback, splitting modes by `i % 3`.

### (H) The 8-of-9 mass collapse signal traces to extract_3x3_from_tensor + h_0 contraction, not rank-1 ψ

`extract_3x3_from_tensor` after P8.3-followup-C uses ONLY `higgs[0]` (the lowest-eigenvalue Higgs candidate). With round-robin fallback, `higgs = [0, 3, 6]`; `higgs[0] = 0`. Then Y_ij = T_{i, j, 0}. For T to produce 8/9 zero singular values of Y, at least 8 of the 9 entries `T_{li, rj, h0=0}` must be ≈ 0 — i.e. mode 0 (the lowest-λ b(1,0) mode) decouples from 8 of the 9 (left_index, right_index) pairs. This is consistent with the up/down/lepton sector indices being drawn from MODES that are sparsely supported on the SAME b-lines as h_0, but the actual T tensor entries at single-h fixing must be checked downstream. **The contraction layer is now the load-bearing source of the rank-1 Y_ij.**

## Recommended fix

The hostile-review hypothesis (rank-1 ψ from upstream solver) is FALSIFIED. The mechanism is two-fold:

1. **Metric-level (root cause).** k=3 is not converged at max_iter=50 (σ=0.75). Real harmonic modes require σ ≪ 1. The `kernel_dim_target = Some(9)` fallback masks the underlying problem by FORCING 9 modes regardless of whether the L spectrum has a true kernel. Until the TY metric reaches σ ~ 1e-2 the "harmonic modes" are L=D*D-low-eigenvalue-modes, not Hodge-harmonic representatives. **This is the same Donaldson-stall issue P8.4-fix-c is addressing on a parallel branch.** No new fix needed here — wait for P8.4-fix-c.

2. **Contraction-level (signal source).** Even with full-rank linearly independent ψ_α, the single-h_0 contraction `Y_ij = T_{i, j, h0}` projects a rank-≤9 T_{ijk} tensor onto a single Higgs slice. If T's coupling structure has the entire 27-zero-mode triple-overlap concentrated on diagonal `(i, j, h)` triples that don't include `(i, j, h0=mode_0)`, then 8/9 of Y's entries vanish even with full-rank ψ. **The contraction layer needs additional diagnostics on T entries — write a P8.3-followup-A2 that prints the 9×9×9 = 729 T tensor entries in scaled form before extract_3x3 collapses them.** That isolates whether the issue is geometric (T sparsity) or assignment (sectors drawing from disjoint mode pools).

3. **No production code change required by this diagnostic.** The follow-up tasks are:
   - **A2** (new): T-tensor sparsity diagnostic (709 entries → identify the 8/9 zeros).
   - **B** (existing): Real Schoen 3-factor monad bundle.
   - **P8.4-fix-c** (in flight): Donaldson convergence at k=3 / k=4.

## Files touched

- ADDED `src/bin/p8_3_followup_a_zero_mode_diag.rs` (370 lines, throwaway diagnostic).
- ADDED bin entry in `Cargo.toml`.
- No production code changed.
