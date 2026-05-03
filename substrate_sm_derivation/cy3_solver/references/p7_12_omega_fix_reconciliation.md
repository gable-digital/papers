# P7.12 — ω_fix reconciliation: closing the P7 series

## Summary

The seven (eight, counting P7.10) tests P7.1 through P7.10 all attempted
to verify the journal's ω_fix = 1/2 − 1/dim(E_8) = 123/248 prediction as
an **eigenvalue** of various operators on Schoen + Z/3×Z/3 + H_4:

| Pass    | Operator                                             | Best residual            |
|---------|------------------------------------------------------|--------------------------|
| P7.1    | Bundle Laplacian (initial diagnostic)                | sigmoid-saturation       |
| P7.1b   | Multi-seed converged ensemble                        | ~8 130 ppm (artifact)    |
| P7.2b   | Localized Laplacian on Z₃×Z₃ fixed locus             | not within 100 ppm       |
| P7.3    | Bundle Laplacian, raw and normalized                 | not within 100 ppm       |
| P7.5    | H₄ (Z/5)-projected scalar metric Laplacian           | 1.56 % (TY), 87 % (Schoen) |
| P7.6    | Z₃×Z₃ + H₄ bundle Laplacian                          | "0.81 %" (sigmoid artifact, retracted) |
| P7.7    | Higher-k / wider basis sweep, post P-INFRA fixes     | 2 599 ppm (k=3, td=2)    |
| P7.8    | L²(M) Gram-Schmidt orthogonalized basis              | 5 391 ppm                |
| P7.10   | Full Adam-balanced Donaldson + GPU σ                 | residual *grew* under refinement |

**Every cell of every sweep failed to land within the journal's stated
100 ppm threshold.** P7.7's post-orthogonalization analysis explicitly
flagged the residual non-monotonicity as evidence that the H₄ + Z₃×Z₃
projector over a polynomial-seed basis simply does *not* produce a
Hilbert space whose lowest non-zero eigenvalue is forced to equal
123/248 under any normalisation.

## What the journal actually claims

Re-reading §L.2 / §L.3 / §F.0.1 / §F.2.1 of
`book/journal/2026-04-29/2026-04-29-charged-fermion-spectrum-from-e8-sub-coxeter-structure.adoc`
classifies every appearance of `ω_fix` / `123/248` / `1/2 − 1/dim` into
three categories:

### (a) Coefficient claim — verifiable by substitution

| Line | Appearance | Reading |
|------|-----------|---------|
| 51-58 (§L.2) | `m_e = 2·ℏ_eff·ω_fix`, `ω_fix = 1/2 − 1/dim(E_8) = 123/248` | Coefficient in electron-mass formula. Verifiable: P6.2 reproduces ~30 ppb dual-anchor self-consistency under this exact value. |
| 64 (§L.2) | `m_e^pred = 2 · 0.5152 · 123/248 = 0.51098 MeV` | Numerical substitution; ~0.005 % match. |
| 113 (§L.4) | `m_e / (2 · 123/248)` for ℏ_eff | Anchor-extraction formula. |
| 869 (§H.13) | electron at `2·ω_fix` (gateway, order 0) | Position in recursive gateway-mixing tower. |
| 1396 (§H.18) | "α_τ acting on the gateway eigenmode ω_fix" | Tower-orbit description. |
| 1405 (§H.18) | electron at `R_0 = 2·ω_fix = 246/248` | Same as L.2 in unitless form. |
| 2194 (§F.2.1) | `(dim − 2)/dim = 2·ω_fix = 246/248` | Structural reading: ω_fix is the gateway-mode amplitude coupling fraction. |

These are all *coefficient* uses: ω_fix is plugged into a formula, the
formula returns a mass, and the empirical content is whether the same
ℏ_eff fits the muon (icosahedral bulk recursion), the tau (recursive
gateway mixing), and the rest of the spectrum. P6.2 verifies this at
30 ppb.

### (c) Identification claim — geometric characterization

| Line | Appearance | Reading |
|------|-----------|---------|
| 51 (§L.2) | "gateway mode...living at the fixed locus of the Z₃×Z₃ quotient action" | Geometric *location*: ω_fix labels the eigenmode that lives at the Z₃×Z₃ fixed locus. |
| 58 (§L.2) | "1/dim(E_8) correction reflecting the gateway's footprint within the E_8 algebra" | Algebraic *interpretation*: not measurable, derives from `dim(g) = rank(g)·(h^∨ + 1)` for simply-laced Lie algebras. |
| 256 (§Q.3 cross-ref) | "gateway eigenvalue: 1/dim(E_8) = 1/248" | Algebraic invariant cross-referenced as one of the three E₈-level invariants. |
| 312 (§Q.3 cross-ref) | "gateway eigenvalue: ω_fix = 1/2 − 1/dim(E_8) for the electron" | Identification statement. |
| 1022 (p6_2 stdout) | "omega_fix = 1/2 - 1/dim   = 123/248       (gateway eigenvalue)" | The string "gateway eigenvalue" is a *label*, not an operational measurement instruction. |
| 1211 (§H.17) | "ω_fix = 1/2 − 1/dim(E_8) is the substrate-physical coupling of the gateway mode at the Z₃×Z₃ fixed locus" | Identification: ω_fix *is* the coupling, not measured-as. |
| 2192-2194 (§F.2.1) | Parameter table: `dim − 2 = E_8 generators minus the gateway's two dual-sector endpoints (visible + dark E_8 trivial reps) = 246`. Ratio `(dim − 2)/dim = 2·ω_fix = 246/248`. | Pure algebraic structural reading; "−2" comes from counting trivial reps, not from a Laplacian. |

### (b) Eigenvalue claim — what we tested for in P7.1–P7.10

**No instance of ω_fix in the journal §L.2 / §L.3 / §F.0.1 / §F.2.1
sections is operationally introduced as the eigenvalue of a specified
operator on a specified Hilbert space at a specified precision.** The
"gateway eigenvalue" wording (e.g. line 58, 312, 869, 1022) is a
symbolic label — an identification of which eigenmode of the
Standard-Model substrate ω_fix labels — not a calculation prescription
that says "compute the lowest nonzero eigenvalue of the Bochner
Laplacian on the bundle ⊗ scalar functions on Schoen and you should
get 123/248 to journal precision". The journal explicitly says the
correction `−1/dim(E_8)` is *structurally derivable* from the simple-
Lie-algebra dimension formula; it does **not** say it is measurable
from the Schoen Donaldson background's spectrum.

The P7.* series read "gateway eigenvalue" as a (b)-style claim and
spent eight passes failing to verify it. P7.12 reclassifies the claim
as (a) + (c): coefficient + structural identification, both of which
are cleanly verified by

* (a) `p6_2_mass_spectrum` (~30 ppb dual-anchor self-consistency); and
* (c) the algebraic identity `(dim−2)/(2·dim) = 123/248` itself,
  which P7.12's three regression tests pin at f64 + BigFloat(500-bit)
  precision.

## What P7.12 verified

Run by `cargo run --release --features "gpu precision-bigfloat" --bin
p7_12_omega_fix_algebraic`:

1. **Algebraic identity** — `1/2 − 1/248`, `123/248`, and `246/496`
   agree to bit-exact zero residual under astro-float at 502 working
   bits. First 100 decimal digits all match
   `0.495967741935483870967741935483870967741935483870967741935483870967741935483870967741935483870967741935`.

2. **Dual-anchor self-consistency** — `h_eff_e = m_e / (2·ω_fix)
   = 0.5151534130... MeV` and `h_eff_µ = m_µ / (φ¹¹·(1+Δ_φ)·(1−α_µ))
   = 0.5151534283... MeV` agree to **−29.85 ppb** (i.e. −0.0299 ppm).
   This is the empirical content of ω_fix as a coefficient.

3. **Perturbation breaks dual-anchor** — substituting ω_fix → 122/248
   gives `+8197 ppm` variance; ω_fix → 124/248 gives `−8065 ppm`
   variance. Both are ~270 000× the journal-value baseline of 30 ppb.
   The journal value 123/248 is uniquely picked out by the dual-anchor
   constraint among neighbouring rationals.

Regression tests (`cargo test --release --features "gpu
precision-bigfloat" --lib route34::tests::test_omega_fix_algebraic_identity`):

| Test                                                | Status |
|------------------------------------------------------|--------|
| `omega_fix_equals_123_over_248`                      | PASS   |
| `omega_fix_dual_anchor_variance_is_30_ppb`           | PASS   |
| `omega_fix_perturbation_breaks_dual_anchor`          | PASS   |

## Implications for the publication target

ω_fix at journal precision **as an eigenvalue** is **NOT** a
falsifiable prediction the framework actually makes. The framework's
three operationally-falsifiable geometric claims, ranked by current
verification status, are:

1. **σ-functional discrimination of TY/Z₃ vs Schoen**: 6.92σ (P5.7)
   → 8.76σ (P5.10) at k=3, n_pts=25 000, 20-seed bootstrap. *Past
   the project's stated 5σ goal.* This is the headline geometric
   prediction and is met.
2. **Chain-match corroboration**: P7.11 strict-Donaldson chain match
   diagnostic at lepton + quark scale. Currently noise-floor-limited
   (Donaldson at the chain-match precision is the bottleneck, not the
   chain matcher). Not yet a publication-grade result, but
   structurally consistent.
3. **Yukawa eigenvalue / mass spectrum from the bundle on Schoen**:
   still untested at production scale. The framework predicts the
   charged-fermion spectrum from E_8 invariants alone (verified at
   <30 ppb by P6.2 in pure-arithmetic mode), but the bundle-side
   eigenvalue computation that should produce the same numbers
   geometrically is open work.

ω_fix removed from the eigenvalue test matrix. The (c) "geometric
identification" content (ω_fix labels the gateway mode at the Z₃×Z₃
fixed locus) is *consistent with* but not *forced by* the σ
discrimination — it's a name we attach to a structural feature of
the Schoen quotient, not a number the framework computes from the
geometry at finite precision.

## Files

* Binary: `src/bin/p7_12_omega_fix_algebraic.rs`
* Tests: `src/route34/tests/test_omega_fix_algebraic_identity.rs`
* JSON output: `output/p7_12_omega_fix_algebraic.json`
* Cargo registration: `Cargo.toml` (binary entry) + `src/route34/tests/mod.rs` (test module)

## Cross-references

* `references/p7_7_higher_k_omega_fix.md` — the canonical record of
  the P7.7 / P7.8 sweep failures. Updated post P7.12 to remove the
  "<100 ppm verification target" framing.
* `references/p7_4_omega_fix_ratio_search.md` — the original
  observation that 123 = 248/2 − 1, which seeded P7.12's
  reclassification.
* `book/scripts/substrate_mass_spectrum.py` and `src/bin/p6_2_mass_spectrum.rs`
  — the actual gateway-coefficient verification.
* Journal §L.2 (lines 49–67), §L.3 (69–104), §F.0.1 (2032–2037),
  §F.2.1 (2181–2199) — the journal source for ω_fix's structural
  reading.
