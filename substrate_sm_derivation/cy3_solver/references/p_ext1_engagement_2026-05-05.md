# EXT1-ENGAGEMENT — `Ext¹(V_2, V_1)` shadow on the BHOP-2005 §6 SU(4) extension

**Date:** 2026-05-05
**Stream:** EXT1-ENGAGEMENT (single pass)
**Binary:** `src/bin/p_ext1_compute.rs` → `target-ext1/release/p_ext1_compute`
**Output:** `output/p_ext1_compute.json`, `output/p_ext1_compute.replog`
**Build ID:** `540d82e4_ext1_bbw`
**Replog SHA-256:** `60ccf305f4a4580b5e2a562123feb1048975fe55ebc8c66072d240fe0c5c3b82`

---

## Mission

Determine whether the BHOP-2005 §6 rank-4 SU(4) extension bundle on the Schoen
`Z/3 × Z/3` cover carries a non-trivial off-diagonal extension class
`Ext¹(V_2, V_1)`. By Serre duality on a Calabi-Yau:

```
Ext¹(V_2, V_1) ≅ H¹(X̃, V_1 ⊗ V_2*).
```

A non-trivial class would be the missing harmonic representative needed to
populate `BhopMonadAdapter::a01_su4_extension(κ)` with `κ ≠ 0`, mixing the
V_1 ↔ V_2 blocks of the SU(4) connection and letting the rank-4 H¹ pipeline
redistribute the `(10, +1)` SU(5) rep across {Q, u^c, e^c}. A vanishing class
would mean the BHOP framework's bundle is **split** at the line-bundle
shadow level, `κ = 0` IS the published physical setting, and the
HIGHER-DEGREE-H¹ structural finding (`e^c, ν^c` absent at every polynomial
degree) is a permanent feature.

---

## V_1 ⊗ V_2* construction (shadow)

BHOP §6.1 Eq. 86 publishes:

```
V_1 = O ⊕ O ⊕ O(-τ_1+τ_2) ⊕ O(-τ_1+τ_2)        (rank 4 in line-bundle expansion)
V_2 = O(τ_1-τ_2) ⊗ π_2*(W)                      (rank 2; W rank 2 on dP9)
```

The `BhopMonadAdapter` shadow truncates to four leading line-bundle summands
`b_lines = [O, O, O(-1,+1), O(+1,-1)]` with V_1 frames `{0, 1, 2}`, V_2 frame
`{3}` (per `route34/bhop_monad_adapter.rs:172-215`). The truncation drops one
`O(-1,+1)` from V_1 and the `π_2*(W)` factor from V_2.

Therefore at the shadow level:

```
V_1 (shadow) = O ⊕ O ⊕ O(-1,+1)         (rank 3)
V_2 (shadow) = O(+1,-1)                 (rank 1)
V_2* (shadow) = O(-1,+1)
V_1 ⊗ V_2* = O(-1,+1) ⊕ O(-1,+1) ⊕ O(-2,+2)     (rank 3, c_1 = (-4,+4))
```

This is a sum of three line bundles on the Schoen cover (in the 3-factor
`CP² × CP² × CP¹` ambient: `O(-1,+1,0)`, `O(-1,+1,0)`, `O(-2,+2,0)`).

---

## Method choice — BBW + Koszul (closed form), not the numerical Bergman pipeline

Initial attempt: feed each rank-1 summand into
`compute_h1_twisted_with_method` + `H1Method::CohomologyQuotientWithSchoenQuotient`
(the same path used by `p_lagrangian_eigenmodes`). **Result: every summand
fails with `ChernError::AllSamplesDegenerate`.** Investigation:

`route34/chern_curvature.rs::eval_section` returns `Complex64::new(0, 0)`
when **any** entry of `b_line` is negative (line 474):

```rust
if b_line[0] < 0 || b_line[1] < 0 {
    return Complex64::new(0.0, 0.0);
}
```

The Bergman-kernel approach grounds the fibre Hermitian on canonical FS
sections; bundles with no global sections (i.e. negative-bidegree line
bundles) give `H = 0` at every sample, so every sample is flagged
Bergman-degenerate, and the curvature evaluator returns the
`AllSamplesDegenerate` error after `n_degenerate + n_singular == n_pts`
(line 1310-1311). This is structural — the section-basis Bergman approach
fundamentally cannot handle line bundles whose `H⁰ = 0`, which is exactly
the regime where Bott-Borel-Weil predicts non-zero `H¹`.

**Pivot to BBW.** The codebase already has `route34::bbw_cohomology` —
closed-form Bott-Borel-Weil + Koszul-Schoen `Z/3 × Z/3`-cover line-bundle
cohomology, validated against AGLP-2011 line-bundle tables (14 unit tests
passing). It returns the full `[h⁰, h¹, h², h³]` cohomology vector in
microseconds with no numerical sampling, no curvature dependence, no
H⁰ ≠ 0 prerequisite. This is the correct tool.

`p_ext1_compute` calls `bbw_cohomology::h_star_X_line` directly via a
`SchoenFacade` mirror of the existing inline `h0_line_bundle_schoen`
facade.

---

## Numerical result

```
[BBW] line-bundle cohomology on Schoen X̃ (CP^2 × CP^2 × CP^1, cut by
       canonical (3,0,1) + (0,3,1) hypersurfaces):

  summand 0  O(-1, +1, 0)  : h^* = [0, 0, 0, 0]   V_1 frame 0 ⊗ V_2 frame 3*
  summand 1  O(-1, +1, 0)  : h^* = [0, 0, 0, 0]   V_1 frame 1 ⊗ V_2 frame 3*
  summand 2  O(-2, +2, 0)  : h^* = [0, 0, 0, 0]   V_1 frame 2 ⊗ V_2 frame 3*

  total h^0 = 0   (informational; expected 0 — none of the three line bundles
                   have global holomorphic sections on the Schoen cover)
  total h^1 = 0   (Ext¹ upstairs shadow — THE KEY NUMBER)
  total h^2 = 0
  total h^3 = 0   (Serre dual of h^0; consistent with h^0 = 0)
```

[BBW] elapsed = 89 microseconds.

---

## Sanity check (Bott-Borel-Weil first principles)

For `O(-1, +1, 0)` on `CP² × CP² × CP¹`:
- `h^p(CP², O(-1)) = 0` for all p (Bott: -1 lies in the gap `[-2, -1]`,
  i.e. `-n-1 < d < 0` so neither `h⁰` nor `h^n=h²` fires).
- Künneth: `h^p(CP² × CP² × CP¹, O(-1, +1, 0))` involves sums where the
  first-factor contribution is uniformly zero, so the entire ambient
  cohomology vanishes.
- Koszul chase from `[0, 0, 0, 0, 0]` on every Koszul piece (and on the
  shifted pieces `O(-1-3, +1, 0-1) = O(-4, +1, -1)` etc., which also vanish
  by ambient Bott) yields `[0, 0, 0, 0]` on the Schoen X̃.

Same argument for `O(-2, +2, 0)`: `h^p(CP², O(-2)) = 0` for all p
(`-2 = -n-1+1` is exactly on the upper boundary of the BBW vanishing band on
`CP²`, so `h² = C(-(-2)-1, 2) = C(1, 2) = 0`, and `h⁰`, `h¹` are zero by
definition for negative degree on `CP²`). All Künneth contributions
vanish; Koszul chase preserves zero. Consistent.

---

## Interpretation

### Definite negative finding: shadow Ext¹(V_2, V_1) = 0

The **line-bundle shadow** of `Ext¹(V_2, V_1)` in the BHOP-2005 §6 SU(4)
extension is exactly zero on the Schoen `Z/3 × Z/3` cover (and therefore
also on the quotient `X̃ / Γ`, since the Γ-invariant subspace of `0` is `0`).

This means:

1. **At the line-bundle shadow level, V is split**: `V_shadow = V_1_shadow ⊕
   V_2_shadow`. There is no non-trivial extension class to engage.

2. **`κ = 0` is the published BHOP physical setting** at the shadow level,
   not a deferred wiring task. The `BhopMonadAdapter::a01_su4_extension(κ)`
   default of `κ = 0` is correct.

3. **`e^c, ν^c` absence on the rank-4 shadow is STRUCTURAL.** The
   HIGHER-DEGREE-H¹ stream's empirical finding ("polynomial-degree extension
   cannot recover `e^c, ν^c` because the rank-4 frame enumeration hardcodes
   `frame_idx → SmIrrep`") was correct that polynomial degree was the wrong
   dial; we now also know the only other proposed dial (κ-engagement of
   Ext¹) is exactly zero at the shadow level. The `(10, +1)` SU(5) rep at
   frame 0 stays as `Q` and **does not redistribute to `{Q, u^c, e^c}`**
   under any rank-4-shadow construction.

### Caveat: the published BHOP V_2 carries an additional rank-2 dP9 bundle factor

The shadow truncation drops the `π_2*(W)` factor of `V_2 = O(τ_1-τ_2) ⊗
π_2*(W)`. The full BHOP `V_1 ⊗ V_2*` is:

```
V_1 ⊗ V_2* = (O^{⊕2} ⊕ O(-τ_1+τ_2)^{⊕2}) ⊗ O(-τ_1+τ_2) ⊗ π_2*(W*)
           = (O(-τ_1+τ_2)^{⊕2} ⊕ O(-2τ_1+2τ_2)^{⊕2}) ⊗ π_2*(W*).
```

The `π_2*(W*)` factor is a rank-2 bundle pulled back from dP9. Its
contribution to `H¹` would be:

```
H¹(X̃, line ⊗ π_2*(W*)) = H¹(X̃, line) ⊗ π_2*(W*)-untwisted-piece
                          ⊕ H⁰(X̃, line) ⊗ H¹(dP9, W*)-pieces
                          ⊕ ...   (Leray spectral sequence on π_2)
```

We have just shown each `H¹(X̃, line)` and `H⁰(X̃, line)` are zero for the
relevant bidegrees, so the Leray spectral sequence collapses term-by-term
**provided the line cohomology being multiplied is zero**. The Leray E₂
term `H^p(B_2, R^q π_2_* line) ⇒ H^{p+q}(X̃, line)`: with our line
cohomology vanishing on X̃, `H^{p+q}(X̃, line) = 0` for all p+q. By
non-degenerate Leray (E₂ = E_∞ since the spectral sequence converges to 0),
we cannot conclude `H¹(X̃, line ⊗ π_2*(W*)) = 0` directly because tensoring
W* in changes the sheaf computation. **A definitive vanishing on the full
BHOP V_1 ⊗ V_2* requires dP9-side cohomology of `W` (BHOP §6.1 Eq. 85), 
which is outside the BBW closed-form framework.**

That said: BHOP §6.5 itself **references but does not numerically publish**
the Ext¹ harmonic representative (per the `bhop_monad_adapter.rs` module
docstring lines 156-164: "BHOP §6.5 references but does not numerically
publish the extension class harmonic representative; non-zero κ is a
mathematical placeholder, not a literal BHOP value"). BHOP's polystability
argument shows Ext¹ is non-zero **somewhere** in the bundle (otherwise
Hoppe's criterion would fail), but the explicit numerical class is absent
from the paper. So the line-bundle shadow vanishing is consistent with the
non-zero piece living entirely in the `π_2*(W*)` direction the shadow
elides.

---

## Implications for downstream streams

### YUKAWA-COMPLETE-WIRE

The downstream YUKAWA-COMPLETE-WIRE stream cannot consume a κ-engaged
catalogue from this binary because **there is no κ to engage at the shadow
level**. Two paths forward (both out of scope for EXT1-ENGAGEMENT):

1. **Accept the shadow split as the production setting.** Run YUKAWA-OVERLAP-WIRE on
   the existing `p_lagrangian_eigenmodes` `_deg2_n144.json` (3.5× per-frame
   resolution at degree 2) and document `e^c, ν^c` as structurally absent
   from the BHOP rank-4 shadow predictions. This makes the published
   `Y_u`, `Y_d` Yukawa matrices (whatever they come out to numerically) the
   framework's actual content, and the 4-7-orders-of-magnitude PDG mismatch
   becomes a real prediction failure or a normalization convention to be
   reconciled separately.

2. **Lift the shadow truncation.** Build the `π_2*(W)` factor of V_2 as a
   genuine rank-2 dP9 bundle via Bondal-Orlov / Beilinson resolution on
   dP9 (BHOP §6.1 Eq. 85), tensor into V_1 ⊗ V_2*, and compute H¹ via a
   spectral-sequence chase that consumes both X̃-side line cohomology
   (BBW, this binary) and dP9-side bundle cohomology (a new BBW-on-dP9
   module). This is a multi-day project well beyond a single-pass stream.

### Documentation update

The `route34/bhop_monad_adapter.rs` module docstring (line 156-164) and the
`p_lagrangian_eigenmodes` "Structural finding" block (lines 62-122)
described the κ ≠ 0 path as a deferred wiring task. With the BBW shadow
result in hand, those docstrings should be updated to record that the
**shadow Ext¹ has been computed and is exactly zero**, so the wiring task
is not just deferred — it is, at the shadow level, void.

---

## Real bugs surfaced

1. **`evaluate_chern_curvature` cannot ground bundles with no global
   sections** (line 474 + 1310-1311 of `chern_curvature.rs`). The
   `eval_section` helper unconditionally returns 0 for any negative
   bidegree component, which is mathematically correct (no holomorphic
   sections) but causes the entire pipeline to abort on negative line
   bundles. The fix is the architecture of `bbw_cohomology` — closed-form
   BBW handles this regime correctly. No code change to
   `chern_curvature.rs` is needed; the lesson is "use BBW for negative
   line bundles, period." `p_lagrangian_eigenmodes`-style rank-r bundles
   work because their leading frame is `O = O(0,0)` (positive degree),
   and the curvature evaluator successfully grounds the rank-r Hermitian
   on that frame's section. Negative-only bundles cannot enter that
   pipeline at all. Documented in this report.

2. **No bug in BBW** — the 14 BBW unit tests pass; the closed-form
   `[0, 0, 0, 0]` answer agrees with hand-derived Bott-Borel-Weil for
   `O(-1, +1, 0)` and `O(-2, +2, 0)` on `CP² × CP² × CP¹`-cut-Schoen.

---

## Files

- `src/bin/p_ext1_compute.rs` — binary (407 lines)
- `Cargo.toml` — `[[bin]] p_ext1_compute` registered
- `output/p_ext1_compute.json` — full BBW catalogue + interpretation banner
- `output/p_ext1_compute.replog` — chained-SHA reproducibility sidecar
- `references/p_ext1_engagement_2026-05-05.md` — this document

## Replog
- final_chain_sha256 = `60ccf305f4a4580b5e2a562123feb1048975fe55ebc8c66072d240fe0c5c3b82`
- 5 events: 1 RunStart + 3 PerSeed (one per summand) + 1 RunEnd
