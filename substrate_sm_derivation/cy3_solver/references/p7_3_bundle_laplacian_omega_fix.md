# P7.3 — ω_fix gateway-eigenvalue diagnostic on the bundle Laplacian Δ_∂̄^V

## Goal

Test exit-door (1) for the falsified journal §L.2 prediction

```
ω_fix = 1/2 - 1/dim(E_8) = 123/248 = 0.495967741935...
```

P7.1 / P7.1b / P7.2b falsified ω_fix as an eigenvalue of the
**scalar metric Laplacian** Δ_g (residual ~1.9 % on Schoen with
character classification, similar on the projected basis). The
journal language ("gateway mode", "fermion zero mode") is
heterotic-string speak: the relevant operator is not Δ_g but the
**bundle Dolbeault Laplacian**

```
Δ_∂̄^V = ∂̄_V ∂̄_V^* + ∂̄_V^* ∂̄_V    on    Ω^{0,1}(M, V)
```

with `V = E_8 → E_6 × SU(3)` Wilson-line bundle. P7.3 evaluates
this operator on Donaldson-balanced TY and Schoen metrics and
asks whether its lowest non-zero eigenvalue (suitably
normalised) is 123/248.

## Method

Reuses `route34::zero_modes_harmonic::solve_harmonic_zero_modes`,
which builds the bundle-twisted Galerkin Laplacian

```
L_{αβ} := ⟨ D_V ψ_α , D_V ψ_β ⟩
```

on the AKLP polynomial-seed ansatz space, with `D_V = ∂̄_V + ∂̄_V*`
and bundle metric `h_V` from the HYM solver. The full ascending
spectrum is exposed in `eigenvalues_full` (length =
`seed_basis_dim`). Kernel modes (BBW = 9 for AKLP on TY/Z_3) are
the lowest eigenvalues; the lowest non-zero eigenvalue is the
gateway-mode mass scale.

Run parameters: `n_pts=40000`, `k=3`, `max_iter=100`,
`donaldson_tol=1e-6`, `seed=12345` (one of P7.2b's strict-converged
Schoen seeds). Bundle = `MonadBundle::anderson_lukas_palti_example`
(canonical AKLP / `E_8 → E_6 × SU(3)`); ambient =
`AmbientCY3::tian_yau_upstairs` for TY, `AmbientCY3::schoen_z3xz3_upstairs`
for Schoen.

Kernel classification: any eigenvalue with `|λ| < 1e-3 · λ_max`
counts as kernel. Donaldson-balanced metrics typically have NO such
eigenvalues (the BBW kernel sits at the Bergman-kernel numerical
residual ~ 0.7 to 1.2 on this run, NOT at zero); the binary falls
back to skipping the lowest `cohomology_dim_predicted` eigenvalues
when the threshold rule yields zero kernel modes.

Normalisation schemes tried for each non-zero eigenvalue: raw,
`/ λ_min_nonzero`, `/ λ_max`, `/ λ_mean`, `/ λ_trace`,
`/ vol_proxy`, `λ / (λ + λ_max)` (sigmoid), `/ λ_kernel_max`.

## Results

Seed-basis dim = 24 in both cases (`Σ_α C(b_α[0]+3,3) · C(b_α[1]+3,3)`
for the AKLP B-summands).

### Schoen / AKLP

* Donaldson: 20 iters, σ-residual = 2.19 (the σ-residual
  reported here is from `cy3_metric_unified::summary()`, which
  measures the un-normalised residual; it does NOT reflect the
  strict-converged P7.2b figure of 6.5e-7 because P7.2b uses a
  different residual norm. The metric is the same balanced
  metric in both runs.)
* Bundle Laplacian solve: 0.09 s on 24×24.
* `cohomology_dim_predicted (BBW) = 0` for AKLP-on-Schoen (the
  AKLP bundle is constructed for TY; the BBW chase on Schoen is
  not the canonical Schoen Z_3×Z_3 setting — this is a known
  limitation, see Notes below).
* Full spectrum (lowest 6): 0.722, 0.758, 0.758, 0.796, 0.796,
  0.839 — clear three-fold degeneracy structure but no kernel
  plateau-and-gap.
* Lowest eigenvalue: λ = 0.722. Best match to 123/248 across all
  schemes/ranks: rank 6 (λ = 2.379) via `by_mean_eigvalue` →
  0.5079, residual **24,040 ppm = 2.40 % off ω_fix** —
  **FAILED**.

### TY / AKLP (control)

* Donaldson: 16 iters, σ-residual = 2.67e-1.
* Bundle Laplacian solve: 0.09 s on 24×24.
* `cohomology_dim_predicted (BBW) = 9` (canonical AKLP-on-TY).
* Threshold rule yields 0 kernel modes (all λ_i ≥ 1.21, with
  λ_max = 1.76 — ratio 0.69, well above 1e-3). BBW fallback
  treats the lowest 9 as kernel; lowest non-zero is rank 0
  λ = 1.462.
* Best match to 123/248 across all schemes/ranks: rank 9
  (λ = 1.713) via `by_sigmoid` (= λ/(λ+λ_max)) → 0.4936, residual
  **4,734 ppm = 0.473 % off ω_fix** — **MARGINAL** (sub-1 %, above
  100 ppm).

### Best closest-pick across both candidates

| Geometry    | Rank | Raw λ     | Best scheme       | Value   | Residual ppm | Residual % |
| ----------- | ---- | --------- | ----------------- | ------- | ------------ | ---------- |
| Schoen/AKLP | 6    | 2.379e0   | by_mean_eigvalue  | 0.5079  | 24,040       | 2.40 %     |
| TY/AKLP     | 9    | 1.713e0   | by_sigmoid        | 0.4936  | 4,734        | 0.47 %     |

## Verdict

**Bundle Laplacian variant does NOT verify ω_fix at any
publication-quality precision.** The closest hit is TY rank-9
λ via the sigmoid-style normalisation, at 0.47 % off — well above
the 100-ppm "VERIFIED" threshold and even above 1,000 ppm.

This rules out hypothesis (1) — that the journal's `ω_fix = 123/248`
is genuinely the lowest non-zero eigenvalue of the bundle
Dolbeault Laplacian on the AKLP / `E_8 → E_6 × SU(3)` bundle on
either TY/Z_3 or Schoen — at the precision attainable with our
current Donaldson-balanced-metric + AKLP-polynomial-seed pipeline.

The result is **the same negative verdict** as P7.1b / P7.2b on
the metric Laplacian, but on a strictly different operator. We
have therefore now ruled out:

* Δ_g (scalar metric Laplacian) — P7.1, P7.1b, P7.2b, residual ≥ 1.9 %.
* Δ_∂̄^V (bundle Laplacian) on AKLP / `E_6 × SU(3)` — P7.3,
  residual ≥ 0.47 %.

## Limitations

1. **Bundle choice is NOT the journal's prescription.** The journal
   §F.1.5 / §F.1.6 prescribes a `Z_3 × Z_3` Wilson-line breaking
   that goes `E_8 → SU(5) × Z_3 × Z_3 × …`, not the canonical
   `E_8 → E_6 × SU(3)` realised by `MonadBundle::anderson_lukas_palti_example`
   and `WilsonLineE8::canonical_e8_to_e6_su3`. The `wilson_line_e8.rs`
   module only carries the `E_6 × SU(3)` decomposition. P7.3 therefore
   tests **the wrong bundle**, in the strict sense of "the bundle the
   journal prescribes". It does, however, test **the heterotic
   bundle Laplacian** — the operator the journal language refers to
   — on the standard production AKLP bundle, which is the only
   SU(3)-bundle currently realised in the pipeline.
2. **AKLP-on-Schoen is not the canonical Schoen pairing.** AKLP is
   constructed for the bicubic TY; on Schoen we use it as a
   placeholder (this matches `bayes_discriminate.rs`'s S5 TODO).
   `cohomology_dim_predicted` for AKLP-on-Schoen is 0 — the BBW
   chase doesn't predict any harmonic zero modes for that
   bundle/ambient pair. The Schoen entry in this report should
   therefore be read as a **smoke test**, not a quantitative
   prediction.
3. **Donaldson-balanced regime has no zero-eigenvalue plateau.**
   The well-known "Bergman-kernel numerical residual" issue (see the
   policy comment in `zero_modes_harmonic::solve_harmonic_zero_modes`)
   means the kernel modes sit at λ ≈ 0.72 (Schoen) and λ ≈ 1.21 (TY),
   not at zero. This raises the absolute-eigenvalue scale uniformly
   and shifts the "lowest non-zero" candidate above the natural
   ω_fix ≈ 0.5 target. A pipeline upgrade to FS-Gram-style
   identity-residual extraction (or a Hodge-decomposition projector)
   would be needed to access the genuine bundle-Laplacian zero modes.

## Relation to P7.1b / P7.2b

P7.1 / P7.1b / P7.2b ruled out ω_fix on the **scalar metric
Laplacian** under multiple normalisation schemes and projection
strategies. P7.3 closes the most natural exit-door — that the
journal's "gateway mode" was always meant to be the bundle
Laplacian's lowest non-zero eigenvalue — at the precision
attainable with the current pipeline. Combined, P7.1b, P7.2b, and
P7.3 cover the two most plausible interpretations of "ω_fix as an
eigenvalue".

Remaining exit-doors not closed by P7.3:

* (2) ω_fix = a quantity built from spinor-bundle Dirac eigenvalues
  on `Γ(M, V ⊗ S)` (the genuine spinor-coupled Dirac operator,
  not the holomorphic-tangent restriction we evaluate via
  `D_V` on `Ω^{0,1}`).
* (3) ω_fix = a derived **ratio** of bundle-cohomology numbers
  rather than a Laplacian eigenvalue (e.g. χ(V)/χ(O), or a Chern
  number ratio).
* (4) ω_fix lifts to the precise journal `Z_3 × Z_3`-Wilson-line
  bundle (not realised in the pipeline; would require new bundle
  module).

Closing (2)–(4) would require pipeline upgrades beyond the
`p7_3_bundle_laplacian_omega_fix` scope.

## Files

* Binary: `src/bin/p7_3_bundle_laplacian_omega_fix.rs`
* Output JSON: `output/p7_3_bundle_laplacian_omega_fix.json`
* Run log: `output/p7_3_run.log`
* This summary: `references/p7_3_bundle_laplacian_omega_fix.md`
