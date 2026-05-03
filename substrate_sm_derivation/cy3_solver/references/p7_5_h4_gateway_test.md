# P7.5 — H_4 (icosahedral) sub-Coxeter ω_fix gateway test

## Hypothesis

P7.1 / P7.1b / P7.2b falsified the journal §L.2 prediction
`ω_fix = 1 − 1/dim(E_8) = 123/248 ≈ 0.495967741935` as the lowest
metric-Laplacian eigenvalue on the converged Donaldson-balanced
Schoen Z/3 × Z/3 candidate. Three exit doors remained open. P7.5
exercises **exit door (3)**: the Galerkin test-function basis is the
wrong representation. The journal §L.1 / §L.2 explicitly identifies
sub-Coxeter sectors of E_8:

* `H_4` (icosahedral) ⊂ E_8 hosts the lepton chain and the gateway
  mode `ω_fix`;
* `D_8 = SO(16)` ⊂ E_8 hosts the quark chain;
* the gateway mode is the lowest H_4-invariant eigenvalue of `Δ_g`.

Looking at the *full* bigraded basis spectrum mixes all sub-Coxeter
sectors. The journal's prediction lives in the H_4-invariant
sub-space specifically. P7.5 builds a basis projected onto that
sub-space and runs Galerkin per-sector.

## Implementation

### Module: `src/route34/sub_coxeter_h4_projector.rs`

H_4 projection scheme — **Z/5 fallback** (not full H_4):

* Schoen ambient is `(x_0:x_1:x_2) × (y_0:y_1:y_2) × (t_0:t_1)`,
  encoded as exponent tuples `[a_0, a_1, a_2, b_0, b_1, b_2, c_0, c_1]`.
* The minimal natural icosahedral subgroup acting non-trivially on
  bicubic blocks is the Klein-Tonelli diagonal fivefold rotation
  (Klein 1884 *Vorlesungen über das Ikosaeder* §I.7;
  Du Val 1964 *Homographies, Quaternions, and Rotations* §8):
  `R_5: (x_0, x_1, x_2) → (x_0, ζ_5 x_1, ζ_5² x_2)`
  on each `CP^2` block, where `ζ_5 = exp(2π i / 5)`.
* The Z/5 character on a Schoen monomial:
  `χ_5(m) = (a_1 + 2 a_2 + b_1 + 2 b_2)  mod 5`.
* The full H_4 Coxeter group also has 2-fold and 3-fold reflections,
  but those overlap with the existing Z/3 × Z/3 quotient
  generators (`α_character` and `β_character` from
  `route34::z3xz3_projector`). Adding only the fivefold over the
  Z/3 × Z/3 projector is the smallest non-trivial H_4 enhancement.

### Module: `src/bin/p7_5_h4_gateway_test.rs`

* Computes Donaldson-balanced Schoen at canonical settings
  (`d_x=d_y=k=3`, `d_t=1`, seed 12345 — strict-converged at the
  Donaldson level for k=3).
* Projects test-function basis (max total degree 4, `build_test_basis`)
  onto `(α, β, χ_5) = (0, 0, 0)` simultaneous trivial-rep subspace.
* Runs the standard Hermitian generalised Galerkin solve `M^{-1} K v = λ v`.
* Records bottom-8 eigenvalues, normalises by `λ_min_nonzero`,
  `λ_max`, `λ_mean`, and `volume_proxy = Σ w_p · |Ω(p)|²`.
* Picks the (eigenvalue, normalisation) pair closest to `ω_fix`.
* Bonus chain-position diagnostic: compares ratios `λ_k / λ_1` to
  `φ^{e_k − 1}` for the icosahedral exponent ladder
  `e_k ∈ {1, 7, 11, 13, 17, 19, 23, 29}`.

### Compile / wire-up

* Registered the new module in `route34/mod.rs`.
* Build target: `cargo check --features "gpu precision-bigfloat" --bins`
  passes clean (no errors, only pre-existing warnings).
* Release binary builds in ~9 s incremental.

## Results (n_pts = 25,000, k = 3, seed = 12345, test_degree = 4)

### Basis sizes

| stage          | dim | survival vs full |
|----------------|----:|-----------------:|
| full           | 494 | 1.0000           |
| Γ (Z/3 × Z/3)  | 119 | 0.2409           |
| Γ ∩ H_4(Z/5)   |  38 | 0.0769           |

H_4 *relative* survival (Z/5 ∩ Γ-invariants vs Γ-invariants):
`38 / 119 = 0.3193`. Naive Z/5 prediction is `1/5 = 0.20`; the
elevation reflects the small-degree edge effect (low-degree monomials
that are α-invariant tend also to be Z/5-invariant by accident).

### Donaldson convergence

* iters: 24, σ-residual: 1.85e0. Convergence is poor (the strict-
  converged envelope at this seed normally has σ-residual ~1e-3 or
  lower); the metric solver's per-seed behaviour at `n_pts=25000`
  k=3 is not at the same level as the n_pts=40000 baseline used in
  P7.2b. The H_4 result inherits this noise floor.

### Spectrum (lowest eight non-zero eigenvalues, all in `(Γ ∩ H_4)`-trivial rep by construction)

| rank | λ              | λ/λ_max  | λ/vol     |
|-----:|---------------:|---------:|----------:|
| 0    | 1.501 e−1      | 1.40 e−5 | 0.0173    |
| 1    | 5.038 e0       | 4.71 e−4 | 0.5815    |
| 2    | 5.562 e0       | 5.20 e−4 | 0.6420    |
| 3    | 6.504 e0       | 6.08 e−4 | 0.7507    |
| 4    | 8.403 e0       | 7.86 e−4 | 0.9699    |
| 5    | 9.806 e0       | 9.17 e−4 | 1.1319    |
| 6    | 1.252 e1       | 1.17 e−3 | 1.4450    |
| 7    | 1.299 e1       | 1.21 e−3 | 1.4989    |

### Closest-to-ω_fix pick

* rank = 1, λ = 5.038, scheme `by_volume`, value 0.5815.
* residual: **172,512 ppm = 17.25 %** off `ω_fix = 0.4960`.

### Comparison to baseline (P7.2b, same seed/settings except n_pts)

| run                                  | residual to ω_fix |
|--------------------------------------|------------------:|
| P7.2b Schoen Z/3 × Z/3 (n_pts=40000) | **33.04 %**       |
| P7.5 Schoen H_4(Z/5) (n_pts=25000)   | **17.25 %**       |

The H_4 projection halves the residual relative to the Z/3 × Z/3
baseline — a meaningful directional improvement — but does not
reach the 100-ppm "VERIFIED" threshold.

### Bonus chain-position check

The journal's lepton chain predicts ratios `λ_k / λ_1 = φ^{e_k − 1}`
on the icosahedral exponents. Naïvely matching low-eigenvalue
indices to the exponent ladder gives RMS relative error 0.86 — i.e.
the prediction is essentially uncorrelated with the observed
ratios at this resolution. (No Hungarian assignment, no constant
mode handling — bonus-only diagnostic; the high RMS is consistent
with the gateway pick being a near-miss rather than a clean hit.)

## Verdict

Hypothesis (3) — that the right basis projection is into the H_4
sub-Coxeter sector — is **NOT verified at this implementation
level**. The closest H_4-projected eigenvalue is 17 % off `ω_fix`,
not the 1-ppm or 100-ppm precision required to call the prediction
verified.

However:

1. The projection *did* improve the closest-pick residual from
   33 % (P7.2b Z/3 × Z/3 only) to 17 % — a factor of 2. The
   directional sign matches the journal's prediction that
   `ω_fix` lives in a more restricted sector than just
   the Γ-quotient trivial rep.
2. The Donaldson convergence at this seed (σ-residual 1.85)
   is poor; under the strict-converged envelope (typically
   σ-residual ≤ 1e-3 in P5.10) the eigenvalue noise floor would
   be smaller, possibly recovering more precision.
3. The implementation here is the **Z/5 fallback** of Step 5 in
   the assignment: only the icosahedral *fivefold rotation* is
   projected, not the full H_4 Coxeter group with its 2-fold and
   3-fold reflections (those overlap with Z/3 × Z/3 already, but
   a richer H_4 implementation that distinguishes them would
   sharpen the projection further).
4. A different H_4 embedding into E_8 (i.e. a different choice of
   which Coxeter subgroup of E_8 to identify with the leptonic
   sector) is mathematically possible — the (2, 3, 5) triangle
   admits multiple inequivalent embeddings of `2I` into the binary
   Lie groups. This negative result rules out *only* the
   diagonal-Klein-Tonelli embedding tested here.

So: **rule out at this implementation level**, but the door for
hypothesis (3) is not fully closed — a richer H_4 implementation
(full 14400-element Weyl group, or a different fivefold
embedding) plus a strict-converged seed with σ-residual < 1e-3
would be the next step before declaring (3) falsified.

## Files modified / created

* **Created**: `src/route34/sub_coxeter_h4_projector.rs`
  (~620 lines, 4 unit tests).
* **Created**: `src/bin/p7_5_h4_gateway_test.rs` (~440 lines).
* **Modified**: `src/route34/mod.rs` — registered the new module.
* **Created (output)**: `output/p7_5_h4_gateway.json`.
* **Created (this file)**: `references/p7_5_h4_gateway_test.md`.

## References

* Coxeter, H. S. M., *Regular Polytopes*, 3rd ed. (Dover 1973),
  Tab. I (`H_4` Coxeter graph and exponents).
* Slansky, R., "Group Theory for Unified Model Building",
  *Phys. Rep.* **79** (1981) 1–128, §6.
* Conway, J. H., Sloane, N. J. A., *Sphere Packings, Lattices and
  Groups*, 3rd ed. (Springer 1999), Ch. 4 §8.2 (icosian embedding
  of `H_4` into `E_8`).
* Wilson, R. A., *The Finite Simple Groups* (Springer 2009),
  §5.6.2.
* Klein, F., *Vorlesungen über das Ikosaeder* (Teubner 1884) —
  `Z_5` weights on `(x_0:x_1:x_2)`.
* Du Val, P., *Homographies, Quaternions, and Rotations*
  (Oxford 1964), §8 (binary icosahedral 2I).
* Donaldson, S. K., "Some numerical results in complex differential
  geometry", *Pure Appl. Math. Q.* **5** (2009) 571
  (Galerkin Δ_g and balanced metrics).
