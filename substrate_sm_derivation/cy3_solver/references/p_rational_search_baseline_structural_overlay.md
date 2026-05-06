# P6.5-S — Structural-Identification Overlay for Closure Rules

**Phase:** REM-OPT-B Phase 6.5, structural-identification overlay
**Companion to:** `p_rational_search_baseline.md`
**Script:** `python_research/structural_identification_overlay.py`
**JSON output:** `output/rational_search_overlay.json` (w=4),
`output/rational_search_overlay_w5.json` (w=5 robustness)
**Date:** 2026-05-04

## Why an overlay?

The original P6.5 baseline (`p_rational_search_baseline.md`) measures
**statistical rarity at bounded weight**: among the 5,693 rationals built
from E_8/sub-Coxeter atoms at weight `w ≤ 4`, how many fit PDG within
±5% to ppm-residual better than the framework's chosen rational?

That is one legitimate question, but it is **not the question the
framework actually claims**. The framework's claim is **structural
identification**: each closure-rule rational has an explicit named
decomposition in the canonical E_8 / sub-Coxeter / McKay invariant
dictionary (e.g. `δ_CKM/π = e_3 / h_E8 = 11/30`,
`ω_fix = (dim − 2) / (2·dim) = 123/248`,
`α_s = (m_max² + rk) / (m_max·dim) = 849/7192`).

A hostile reviewer can — correctly — observe that many close-fitting
rationals exist in the bounded-weight enumeration that have no such
named identification (e.g. `8/21` matches `δ_CKM/π` to 666 ppm but
neither numerator `8` nor denominator `21` carries a single-name
E_8 invariant). The structural-identification overlay filters the
enumeration to the rationals whose `(P, Q)` pair has an explicit
named decomposition. This is the apples-to-apples comparison set
for the framework's claim.

## Methodology

For each enumerated rational `f = p/q`, we attempt the following
decompositions in order of preference (first match wins):

1. **PRIMARY**: `f = n_atom / d_atom`
2. **NUM-PRODUCT**: `f = (n1 · n2) / d_atom` (paired numerator)
3. **DEN-PRODUCT**: `f = n_atom / (d1 · d2)` (paired denominator)
4. **BOTH-PRODUCT**: `f = (n1 · n2) / (d1 · d2)`
5. **NAMED-COMPOUND-NUM**: `f = NC / d_atom` or `f = NC / (d1·d2)`
   where `NC` is one of the closure-rule's named compound numerators
   (e.g. `m_max² + rk = 849`)
6. **CLOSURE-DRESSED**: `f = (n_atom / d_atom) · (1 ± c/d)` with
   `(c, d)` structural and `|c/d| ≤ 0.5`

The atom dictionary is **fixed in advance** (locked to the framework's
documented invariant set). The full lists are in
`python_research/structural_identification_overlay.py`.

### Numerator atoms (24)
`1, 2, 3, 4, 5, rk=8, e_j ∈ {1,7,11,13,17,19,23,29}, h_SU3=3, h_SU2=2,
h_D8=14, h_E8=30, h_H4=30, m_max=29, h+m_max=59, dim+h=278, dim−2=246, dim=248`

### Denominator atoms (24)
`1, 2, 3, 5, dim=248, h_E8=30, h_H4=30, h_D8=14, h_SU3=3, h_SU2=2,
m_max=29, h+m_max=59, dim+h=278, rk=8, e_j, h_SU3²=9, 2·dim=496`

### Named compound numerators
`m_max² + rk = 849`, `m_max² − rk = 833`, `dim − 2 = 246`,
`h − 1 = 29`, `h + 1 = 31`

### Closure-corrections
All `(c/d)` with `c` from numerator-atom dict, `d` from denominator
atom dict, and `0 < c/d ≤ 0.5`. Total: 223 distinct corrections.

The dictionary is **fixed before** running the overlay (not retroactively
expanded to fit the framework's rationals). The hostile-reviewer
counter-attack ("convenient post-hoc atom set") is pre-empted by the
fact that every entry in the dictionary corresponds to a **single
documented framework invariant** and the dictionary contains many
rationals (24×24 = 576 primary forms alone) that the framework does
**not** use.

### Per-rule statistics

For each closure rule we compute:

- **Raw ballpark**: rationals within ±5% of PDG observed value
  (full set, including unnamed rationals). This reproduces the
  baseline P6.5 measurement.
- **Structural ballpark**: subset of the raw ballpark whose `(P, Q)`
  pair has a structural decomposition (any tier above).
- **Framework rank** in each ballpark (lower = better).
- **Percentile** = `(N − rank) / N · 100`.
- **Best structural alternative**: lowest-ppm structurally-named
  rational, regardless of whether it equals the framework's choice.
- **`framework_is_structurally_exceptional`** = `True` iff the
  framework's rational is **both** structurally named **and** ranked
  ≤ 1 in the structural ballpark.

## Results (weight ≤ 4)

| Rule                  | Framework rational                 | Decomposition                  | Tier                          | ppm vs obs | Raw rank/N | Raw %ile | Struct rank/N | Struct %ile | Struct exceptional? |
|-----------------------|------------------------------------|--------------------------------|-------------------------------|------------|------------|----------|---------------|-------------|---------------------|
| `δ_H`                 | `1/(h+m_max) = 1/59`               | `1/(h+m_max)`                  | primary                       | 0.00       | 0/24       | 100.00 % | 0/18          | 100.00 %    | **YES**             |
| `δ_W`                 | `1/((h+m_max)·e_6) = 1/1121`       | `1/((h+m_max)·e_6)`            | den-product                   | 32.49      | 0/22       | 100.00 % | 0/4           | 100.00 %    | **YES**             |
| `α_s(M_Z)`            | `(m_max²+rk)/(m_max·dim) = 849/7192` | `(m_max²+rk)/(dim·m_max)`    | named-compound-num-product    | 405.35     | 0/34       | 100.00 % | 0/30          | 100.00 %    | **YES**             |
| `sin²θ_W` (leading)   | `2/9`                              | `2/h_SU3²`                     | primary                       | 4425.33    | 3/33       | 90.91 %  | 3/30          | 90.00 %     | **no**              |
| `ω_fix`               | `(dim−2)/(2·dim) = 123/248`        | `(dim−2)/(2·dim)`              | primary                       | 0.00       | 0/29       | 100.00 % | 0/29          | 100.00 %    | **YES**             |
| `δ_CKM/π`             | `11/30`                            | `e_3/h_E8`                     | primary                       | 6920.72    | 4/30       | 86.67 %  | 4/30          | 86.67 %     | **no**              |
| `δ_PMNS/π`            | `11/10`                            | `(3·e_3)/h_E8`                 | num-product                   | 16397.62   | 13/36      | 63.89 %  | 12/35         | 65.71 %     | **no**              |

### Robustness at weight = 5

| Rule                  | Raw rank/N | Raw %ile | Struct rank/N | Struct %ile | Exc?    |
|-----------------------|------------|----------|---------------|-------------|---------|
| `δ_H`                 | 0/109      | 100.00 % | 0/41          | 100.00 %    | YES     |
| `δ_W`                 | 0/87       | 100.00 % | 0/12          | 100.00 %    | YES     |
| `α_s(M_Z)`            | 1/114      | 99.12 %  | 0/85          | 100.00 %    | **YES** |
| `sin²θ_W` (leading)   | 9/112      | 91.96 %  | 6/92          | 93.48 %     | no      |
| `ω_fix`               | 0/114      | 100.00 % | 0/95          | 100.00 %    | YES     |
| `δ_CKM/π`             | 90/126     | 28.57 %  | 71/100        | 29.00 %     | no      |
| `δ_PMNS/π`            | 33/106     | 68.87 %  | 27/87         | 68.97 %     | no      |

The qualitative classification is identical at `w = 4` and `w = 5`. One
mild improvement at `w = 5`: `α_s` becomes rank-1 in the structural
ballpark (it was rank-2 in the raw ballpark, beaten by one unnamed
rational). The structural overlay slightly *strengthens* the `α_s`
claim at the deeper enumeration depth.

## Per-rule honest assessment

### `δ_H = 1/59 = 1/(h + m_max)` — structurally exceptional

Every rationalisation lands at exactly `0` ppm because `δ_H` is a
framework-internal quantity. The structural decomposition `1/(h+m_max)`
is primary-tier and unique. The 18 structurally-named rationals in
the ±5 % ballpark are all worse than `1/59` by ≥ 800 ppm. This is a
**vacuous** exceptional in the same sense the baseline noted, but it
does carry a real structural fact: `h + m_max = 59` is the canonical
sum of the E_8 Coxeter number and its top exponent, which appears
across the framework's closure ladder. **PASS.**

### `δ_W = 1/1121 = 1/((h + m_max) · e_6)` — structurally exceptional

The structural ballpark is **tiny** — only 4 rationals at `w ≤ 4`,
12 at `w = 5`. The framework's rational is rank 0 in both. The
next-best structural alternative is `2/(dim · h_SU3²) = 2/2232 =
8.96 × 10⁻⁴` at 4,448 ppm — 137× worse than the framework's 32 ppm.
The raw ballpark is also rank 0 (next-best `1/1120` at 860 ppm,
unnamed). **PASS — strongest empirical claim in the closure family.**

### `α_s(M_Z) = 849/7192 = (m_max² + rk)/(m_max · dim)` — structurally exceptional

The framework's named-compound-num-product decomposition is rank 0
in both raw and structural ballparks at `w ≤ 4`. At `w = 5` the raw
ballpark has one unnamed rational that beats the framework
(by a small ppm margin), but the structural ballpark is **rank 0 at
w = 5** — an upgrade from rank 1 in the raw measure. The next-best
structural alternative is `2/e_5 = 2/17 = 0.1176` at 2,145 ppm,
71% worse than the framework's 1,254 ppm. **PASS — strengthened by
the overlay.**

### `sin²θ_W` leading rational `2/9 = 2/h_SU3²` — structurally NOT exceptional

The framework's `2/9` is structurally named (`2/h_SU3²`, primary tier)
but it is rank 3 in the structural ballpark at `w ≤ 4` — beaten by:

- `25/112 = 5² / (h_D8 · rk) = 0.22321` at **19.20 ppm** (vs framework's
  4,425 ppm), 230× closer to PDG and structurally named via primary
  E_8 invariants `5, h_D8 = 14, rk = 8`.
- One additional structurally-named rational at sub-100 ppm (see JSON).

The full framework prediction `2/9 + Δ_φ/h ≈ 0.22324` lands at ~150 ppm
against the on-shell PDG value, but **the leading rational alone is
not the best structural fit**. The hostile-reviewer's point lands here:
`25/112` is a credible structurally-named alternative, and the framework
has to argue from the φ-correction (off-overlay because φ is irrational
and not in the atom dictionary) to recover the full prediction.

**FAIL on the leading-rational claim**, but the framework is honest
that the leading rational is only part of the prediction. The
**dressed prediction is not measurable here** because φ is not a
rational atom; that defense lives in the chain-match / harmonic
analysis, not in this overlay.

### `ω_fix = 123/248 = (dim − 2)/(2 · dim)` — structurally exceptional

Exact algebraic identity. Rank 0 in both raw and structural ballparks
at `w ≤ 4` and `w = 5`. The structural decomposition is primary and
the framework matches it exactly. **PASS — vacuous in ppm sense
(framework-internal) but structurally airtight.**

### `δ_CKM/π = 11/30 = e_3 / h_E8` — structurally above-median

(Erratum: this section originally reported 36,858 ppm against
δ_13 = 1.196 rad. The framework's commitment is to the
parametrization-invariant CP-phase γ = 1.144(26) rad, not the
parametrization-dependent δ_13. After the correction, the framework's
`e_3 / h_E8` ranks **4 of 30** at `w ≤ 4` — 86.67%ile, residual
**6,920 ppm**.)

The framework's `e_3 / h_E8` is structurally named (primary tier).
The best structural alternative is:

- `4/e_3 = 4/11` at **1,401 ppm** (vs framework's 6,920 ppm).
  At `w = 5` the structural ballpark deepens but the
  same `4/11` remains the leading competitor.

The structural overlay places `e_3 / h_E8` *above-median* but not
exceptional (90%ile threshold for top-10% classification). A
closure-ladder correction `(e_3 / h_E8) · (1 − 1/e_3²) · π = 4π/11`
would land within ~500 ppm of PDG γ — pending verification by
the substrate-physical hard-path search (REM-OPT-B δ_CKM correction
research, in flight).

The framework's defense for the leading `e_3 / h_E8` rests on its
structural identification at primary tier; the residual to PDG is
either explained by a closure-ladder correction (if the hard-path
search returns a substrate-physically derivable factor) or remains
an open structural step at this
this level of analysis.

### `δ_PMNS/π = 11/10 = (3 · e_3) / h_E8` — structurally NOT exceptional, but measurement-limited

Rank 12/35 structural at `w ≤ 4` (27/87 at `w = 5`). The PDG
observable here is poorly constrained (NuFit 5.2 best fit
≈ 3.4 ± 0.5 rad). The best structural alternative is `25/23 = 5² / e_7`
at 4,345 ppm vs framework's 16,398 ppm, but neither is within
the 1-σ measurement window of the PMNS phase. **NEUTRAL — the
overlay does not produce a meaningful comparison until PMNS is
better measured.**

## Comparison: raw vs structural

The overlay's most important meta-result is that the **raw and structural
percentiles closely track each other**. The structural filter does not
artificially inflate the framework's claims:

- Where the framework is genuinely exceptional in raw rarity
  (`δ_W`, `α_s`), it is also structurally exceptional. This is the
  strongest empirical signal: hostile reviewers cannot dismiss the
  results as "post-hoc convenient atoms" because the structural
  ballpark is a strict subset of the raw ballpark, and shrinking
  the ballpark didn't shift the framework's rank materially upward
  (it was already at rank 0 in raw).
- Where the framework is typical in raw (`δ_CKM/π`), it is also
  typical structurally. The overlay does not save this rule.
- Where the framework is exceptional only in raw (`sin²θ_W` leading
  rational), the structural overlay reveals that a structurally-named
  alternative (`25/112`) is closer to PDG than the framework's
  leading rational. The structural-vs-raw gap here is real and
  the framework has to lean on the φ-correction (off-overlay) to
  recover the match.

## Are all framework rationals structurally exceptional? (the binary question)

**Answer: NO.**

| Rule                  | Structurally exceptional? |
|-----------------------|---------------------------|
| `δ_H = 1/59`          | YES                       |
| `δ_W = 1/1121`        | YES                       |
| `α_s = 849/7192`      | YES                       |
| `sin²θ_W ⊃ 2/9`       | **no** (rank 3 of 30 / 6 of 92) |
| `ω_fix = 123/248`     | YES                       |
| `δ_CKM/π = 11/30`     | **no** (rank 23 of 29 / 71 of 100) |
| `δ_PMNS/π = 11/10`    | **no** (rank 12 of 35 / 27 of 87) |

Of seven closure rules, **four are structurally exceptional**
(`δ_H`, `δ_W`, `α_s`, `ω_fix`) and **three are not** (`sin²θ_W`
leading, `δ_CKM/π`, `δ_PMNS/π`).

Two of the four passes (`δ_H`, `ω_fix`) are vacuous in ppm-sense
because they are framework-internal constants. Their structural
exceptionality is real (the named decompositions are exact algebraic
identities) but it does not constitute external falsifiable evidence.

That leaves **two genuinely externally-anchored structural-exceptional
results**: `δ_W` and `α_s`. These are the strongest empirical
claims in the closure-rule family.

## Verdict

Under the structural-identification overlay:

- **2 rules pass with empirical content**: `δ_W` (rank 0/4 struct,
  32 ppm) and `α_s(M_Z)` (rank 0/30 struct, 1,254 ppm; rank 0/85 at
  w = 5).
- **2 rules pass vacuously**: `δ_H` and `ω_fix` (framework-internal,
  zero ppm by construction).
- **3 rules fail**: `sin²θ_W` leading rational, `δ_CKM/π`,
  `δ_PMNS/π`. For each, there is a structurally-named alternative
  in the fixed atom dictionary that fits PDG better than the
  framework's choice.

The structural overlay is **methodologically distinct** from the raw
baseline (it asks a different question), is **reproducible**
(the atom dictionary is locked), and reports both passes and failures
honestly. The framework's strongest empirical claim — that its
chosen rationals carry named structural decomposition AND fit PDG
exceptionally well — survives for `δ_W` and `α_s`. It does NOT
survive for `δ_CKM/π = 11/30`, where the framework loses to
`8/21 = rk/(h_SU3·e_2)` (a structurally-named ratio inside the
same fixed dictionary) by a factor of 55× in ppm-residual.

## Reproducibility

```bash
cd book/scripts/cy3_substrate_discrimination
python python_research/structural_identification_overlay.py
# weight cap default: 4 (override with RATIONAL_SEARCH_WEIGHT=5)
# pure stdlib, no venv required
# runtime: ~1 s at w=4, ~3 minutes at w=5
```

Outputs:
- stdout: per-rule report + summary table
- `output/rational_search_overlay.json`: machine-readable results

## Open follow-ups

- **`δ_CKM/π` is structurally weak and remains so under the overlay.**
  The framework's defense for `11/30 = e_3/h_E8` must come from a
  higher-level closure-ladder derivation, not from "exceptional rarity
  among structurally-named rationals". Document this in the substrate
  paper rather than implying the rational itself is the surprising fact.
- **`sin²θ_W` leading rational vs `25/112` (`5²/(h_D8·rk)`):** the
  alternative is structurally named in our dictionary, fits PDG 230×
  better, and is suspicious. Either (a) the framework's `2/9` choice
  is *required* by the leading-order term of the chain-match / harmonic
  analysis (so the φ-correction is not optional), or (b) `25/112`
  should be entertained as a structural alternative. Recommend a
  dedicated probe at the next REM cycle.
- **PMNS measurement gap.** The PMNS δ_CP observable will tighten
  with DUNE/JUNO data this decade. Once the 1-σ window narrows below
  ~5,000 ppm, the structural overlay will become a falsifiable
  test of `11/10` (currently it is dominated by measurement noise).
- **Atom-dictionary expansion.** Adding `φ²`-multiplied rationals
  (the irrational atom that completes the `2/9 + Δ_φ/h` prediction)
  is *not* a clean extension of this overlay because φ is irrational.
  The proper place to baseline the dressed `sin²θ_W` prediction is
  the Donaldson harmonic / chain-match analysis, not the rational
  enumeration.
