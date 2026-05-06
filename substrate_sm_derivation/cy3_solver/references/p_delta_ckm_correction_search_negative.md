# P-rule audit — δ_CKM closure-ladder correction search (negative result)

**Status:** NO STRUCTURALLY-DEFENSIBLE CORRECTION FOUND.
**Date:** 2026-05-04.
**Companion script:** `python_research/delta_ckm_correction_search.py`.
**Context:** Asked whether a closure-ladder correction term can bring
the framework's leading prediction `δ_CKM = 11π/30 = 66.0°` into tighter
agreement with PDG-2024 γ (CKM unitarity-triangle CP-violating phase)
γ = 65.4° ± 1.5° = 1.142 ± 0.026 rad.

## Executive verdict

The leading prediction `δ_CKM = (e_3/h_E8) · π = 11π/30 = 66.0°` is
**already within 0.40 σ of PDG γ central** (γ = 65.4° ± 1.5°), with the
+0.85 % point-residual fully covered by the experimental ±1.5° (≈ ±26 000 ppm)
uncertainty band.  **No correction is statistically required.**

The exhaustive enumeration in
`delta_ckm_correction_search.py` confirms that several closure-ladder
factors `F` exist that bring `(11/30)·F·π` to within ≤ 100 ppm of PDG
γ-central, but **none of them satisfies BOTH** of the user's mandatory
acceptance criteria:

  - **(A) Pattern-match:** the rational form `F` matches one of the
    pre-existing closure-ladder forms in the substrate paper
    (`(1 ± δ)`, `1/(1 ± δ)`, `(1 - 2 α²)` quadratic self-coupling, etc.);
  - **(B) Substrate-physical interpretation:** the atom dictionary used
    in the numerator and denominator of `δ` has an established
    framework reading.

The closest contenders are documented below.  None is acceptable.

## Search methodology

We enumerated closure-ladder factors `F` of the form

  - `F = (1 - δ)`              (delta_W-style additive shift)
  - `F = 1 / (1 + δ)`          (α_τ_full-style geometric series)
  - `F = (1 - δ²)`             (R_μ-style quadratic self-coupling)
  - `F = (1 + δ)`, `F = 1/(1 - δ)`, `F = (1 + δ²)`  (sign-mirrors)

over the canonical E_8 / Z_3×Z_3 atom dictionary
  `{1, 2, 3, 5, dim=248, rank=8, h=30, m_max=29, |Γ|=9, h_D=14,
    h_SU3=3, e_j ∈ {1,7,11,13,17,19,23,29}, φ, φ²}`
where `δ = P/Q` is a rational over monomials `P, Q` from the
dictionary.  Each closure form is evaluated for total atomic weight
`w(P) + w(Q) ≤ 5`.  Total candidates enumerated:

| weight | unique forms | F-deduped within ±5 % |
| ------ | ------------ | --------------------- |
| 3      | 396 mono     | 2 007                 |
| 4      | 1 715 mono   | 10 459                |
| 5      | 6 432 mono   | 45 228                |

For each candidate we computed `δ_CKM_pred = (e_3/h_E8) · F · π`,
the ppm-residual to PDG γ central, the analog `δ_PMNS_pred = (e_3/h_E8) · 3 · F · π = (11/10) · F · π`,
and three quality flags
  - `STRUCTURALLY_NAMED` — atoms are products of ≤ 2 dictionary tokens;
  - `MATCHES_PATTERN` — form matches an established closure-ladder kind;
  - `INTRA_LADDER_SCORE` ∈ {0,1,2,3} — composite of the above plus
    "mirror-rule" check (numerator/denominator atom set overlaps with
    an existing closure rule's atom set, e.g. `(h+m_max)·e_j`,
    `e_3` muon-slot, `φ²` icosahedral).

## Top candidates

### Sub-100-ppm (vs PDG γ central) candidates with the cleanest structural reading

| ppm  | F                       | form                                | pattern              | int_only |
| ---- | ----------------------- | ----------------------------------- | -------------------- | -------- |
| 0.0  | 109/110                 | `(1 - 1/(2 · 5 · e_3))`             | shift_minus          | yes      |
| 10.0 | 0.99091900              | `1 / (1 + 5²/(dim · e_3))`          | geom (α_τ_full-style)| yes      |
| 16.7 | 0.99092559              | `(1 - 5/(m_max · e_6))`             | shift_minus          | yes      |
| 20.9 | 0.99088838              | `1 / (1 + rank/(h · m_max))`        | geom_minus           | yes      |
| 105.4| 0.99080460              | `(1 - rank/(h · m_max))`            | shift_minus          | yes      |

### Failed user-hypothesis candidate

| ppm  | F                       | form                                | pattern              |
| ---- | ----------------------- | ----------------------------------- | -------------------- |
| 834.0| 120/121                 | `(1 - 1/e_3²)`                      | self_quad_minus      |

## Why each top candidate fails the substrate-physical bar

### 1. `(1 - 1/(2 · 5 · e_3)) = 109/110` (0 ppm)

Equivalent to `δ_CKM = 109 π / 300 = 109 π / (h · 10)`.  The prefactor
`2 · 5 = 10` reads as "binary × McKay-icosahedral", and `e_3` is the
muon-slot exponent, so the **denominator** `2 · 5 · e_3 = 110` is
defensible as the McKay-icosahedral ladder seated at the muon slot
with binary doubling (visible × dark).

**The numerator `109` is prime and not in the dictionary.**  It can
only be read as `109 = (2·5·e_3) − 1`, which writes the form back as
`(1 − 1/110)` — a bare "1" subtractive shift on top of an
arithmetically constructed denominator.  This is dictated by the
ppm-fit, not by structure.  **Verdict: ppm-overfit, not structurally
derivable.**

The form **does** match `delta_W = 1 − 1/((h+m_max)·e_6) = 1 − 1/1121`
in the structural-grammar sense, with `(h+m_max)·e_6 = 1121` replaced
by `2 · 5 · e_3 = 110`.  Yet the numerator/denominator pair
`(2·5, e_3)` has no antecedent in the existing closure ladders;
all `delta_W`-family corrections in the paper use `(h+m_max)` as the
ladder denominator, not `2·5`.  Adopting `(2·5)` here would be a new,
ad-hoc atom-dictionary entry chosen specifically to land 109/110.

### 2. `1 / (1 + 5²/(dim · e_3))` (10 ppm)

Form-match: this is the geometric-series form `1/(1 + δ)` with
`δ = 5²/(dim · e_3) = 25/2728`.

**Substrate reading attempted:** numerator `5² = 25` is McKay-
icosahedral squared (visible × dark icosahedral coupling); denominator
`dim · e_3 = 248 · 11 = 2728` is "E_8 dimension at the muon-slot
exponent".  This is **plausible-sounding** but has no precedent in
the framework: nowhere else does the closure ladder admit a direct
product `dim · e_j` in the denominator (the convention is
`(h+m_max)·e_j` or `m_max·e_j`).  Building a `5²` numerator out of
McKay × McKay would introduce a new closure rule unique to δ_CKM,
which the user's acceptance criterion (B) explicitly forbids.

### 3. `(1 - 5/(m_max · e_6))` (17 ppm)

Form-match: identical grammar to `delta_W = 1 − 1/((h+m_max)·e_6)`
with `(h+m_max)` replaced by the simpler `m_max`, and the numerator
`1` replaced by `5`.  The denominator `m_max · e_6 = 29 · 19 = 551`
is structurally clean (top exponent times palindrome partner of the
muon slot).  But the numerator `5` (McKay-icosahedral) appears
**unmotivated** at the CKM CP-phase: McKay-icosahedral 5 enters the
`gateway` formula and `R_v` in the visible–dark gateway sector, not
in the CP-phase sector.  Importing it here would require justifying
why the CKM CP-phase couples to the icosahedral McKay integer at
all; no such justification exists in the framework as written.

### 4. `1/(1 + rank/(h · m_max))` (21 ppm)

Form-match: geometric-series `1/(1 + δ)` with `δ = 8/870 = rank/(h·m_max)`.
Numerator `rank = 8` is the E_8 rank, denominator `h · m_max = 870`
is the maximal Coxeter–top-exponent product.  Both atoms are
canonical.

**Substrate reading attempted:** "the E_8 rank coupling to the
maximal Coxeter–top-exponent product."  This is **very generic** —
neither `rank` nor `h · m_max` appears in any existing closure rule
in this combination.  The form would have to be uniquely justified
for δ_CKM, again violating criterion (B).

### 5. User hypothesis `(1 − 1/e_3²) = 120/121` (834 ppm)

Form-match: `(1 − δ²)` with `δ = 1/e_3`.  This is the **R_μ-style
quadratic self-coupling** at the muon-slot exponent.  Substrate
reading: "CP violation receives a self-coupling correction at the
muon-slot Coxeter exponent at quadratic order, analogous to the
muon's own R_μ correction at α_τ²."

**This is the most structurally defensible form.**  It matches an
established closure-ladder pattern (R_μ = φ¹¹·(1+Δ_φ)·(1−2α_τ²))
and uses only one named atom (e_3 — the muon-slot exponent), so
it satisfies criterion (A) and arguably (B).

**But it fails ppm-acceptance.**  At 834 ppm, the prediction
`(11/30)·(120/121)·π = 1.142 397 rad = 65.45°` lies inside the PDG
1-σ window (`[63.9°, 66.9°]`), but at 32× the user's stated <100 ppm
bar.  Moreover, applied to δ_PMNS, the same correction predicts
`δ_PMNS = (11/10)·(120/121)·π = 12π/11 = 196.4°`, which is at
`(196.4 − 195)/π_uncertainty = 1.4°/29° = 0.05 σ` from NuFit-5.2 NO
best fit — fully consistent.

The user's framing required a **<100 ppm** correction.  The
quadratic self-coupling closure form, while defensible, does not
deliver that.

### Why the R_μ analog with stronger coefficients does not close the gap

Borrowing the explicit `(1 − 2α_τ²)` form from R_μ with
`α_τ = 1/(dim+h) = 1/278` produces only +0.0026 ppm of correction
(`F − 1 ≈ −2.6 × 10⁻⁵`), which is **3 000× too small**.  Inflating
the coefficient artificially to match the residual (e.g.
`(1 − 660 α_τ²)`) lands within 556 ppm but the prefactor 660 has no
structural reading.  The R_μ-style closure mechanism is the **wrong
sub-harmonic order** for the +9 mrad CKM residual; it operates at
the Cabibbo-suppression scale `α_τ² ≈ 1.3 × 10⁻⁵`, not at the
≈ 9 × 10⁻³ scale required here.

## Statistical context

PDG-2024 γ uncertainty `±1.5° = ±26 000 ppm` is **2.83×** the
framework's leading +9 174 ppm point-residual.  The framework's
prediction is therefore at **0.40 σ** above PDG γ central — well
within the experimental 1-σ window `[63.9°, 66.9°]` — and any
"correction" that lands the prediction inside ±100 ppm is
overfitting the central value at 260× higher precision than the
measurement supports.

The substrate paper's existing position
(§ "P6.5 verdict summary") classifies the closure rule
`δ_CKM/π = 11/30` as **typical or worse** under bounded-weight
rational enumeration (20.00 %ile at w ≤ 4, rising to 28.57 %ile at
w = 5), with the explicit footnote that "its structural commitment
is algebraic, not ppm-rare."  This negative-result audit is fully
consistent with that prior assessment: there is no shorter
substrate-physical pathway from the leading rational to a
ppm-tighter rational than the one the framework already takes.

## Recommendation

**Keep `δ_CKM = 11π/30` as the leading framework value, with no
closure correction.**  The +0.85 % point-residual to PDG γ central
is fully covered by the experimental ±1.5° (≈ 0.85 %) uncertainty
band.  The framework's prediction is at 0.40 σ — comfortably inside
PDG 1-σ — and no closure-ladder correction at weight ≤ 5 in the
canonical E_8 / Z_3×Z_3 atom dictionary simultaneously satisfies
the substrate-paper grammar and the < 100 ppm bar.

If the goal is purely to record a **structural-identification
overlay** that is consistent with both established closure forms
and the muon-slot identification of e_3 = 11, the closest defensible
candidate is

  ```
  δ_CKM ≈ (e_3/h_E8) · (1 − 1/e_3²) · π = 12π/(30·11/120) ... = 1320π/3630 = 4π/11
  ```

at 834 ppm.  This is the user's original hypothesis: the form
matches R_μ-style quadratic self-coupling at the muon-slot
exponent, has a clean substrate reading
("CP violation receives an order-e_3⁻² self-coupling correction at
the muon-slot Coxeter exponent"), and predicts
`δ_PMNS = 12π/11 = 196.4°` consistent with NuFit-5.2.  However:

  - it is **not below 100 ppm** (834 ppm);
  - it is **not statistically distinguishable** from the leading
    `11π/30` rule at present PDG precision;
  - it would **introduce a novel closure rule** (R_μ-quadratic at
    e_3 instead of α_τ) without an established precedent in the CP
    sector.

**Therefore: NO STRUCTURALLY-DERIVABLE CORRECTION FOUND at the
searched weight.**  The framework's leading 11π/30 stands without a
closure correction; the structural-identification overlay carries
this rule.

## Search reproducibility

```
cd book/scripts/cy3_substrate_discrimination/python_research
python delta_ckm_correction_search.py
# writes output/delta_ckm_correction_search.json
```

Total wall time on i9-12900: ~3 minutes for w ≤ 5 (45 228 candidate
forms).  w = 6 was attempted and aborted at 21 435² × 6 ≈ 2.7 B
candidate-evaluations (~37 min); the w ≤ 5 sweep is sufficient to
establish that no structurally-defensible correction exists at the
searched grammar, and structural extra weight only multiplies
candidate count without improving the underlying atom dictionary.
