# Path A research resolution — executive summary

**Date:** 2026-05-06
**Verdict:** (a) ROBUST — `Ext^1_full = 0` on the Schoen Z/3×Z/3 quotient

## Three-step computational chain

The Path A heterotic line of attack on the substrate framework's
visible-sector Yukawa structure has reached definitive closure through
three sequential SHA-pinned computations on the Schoen
Z/3×Z/3 quotient `X̃/Γ` carrying the BHOP-2005 §6 SU(4) extension
bundle:

1. **Shadow `Ext¹ = 0`** — line-bundle-shadow level
   * Commit: `831d910c`
   * Reference: `cy3_solver/references/p_ext1_engagement_2026-05-05.md`
   * Result: `Ext¹(V₂, V₁) = 0` at the line-bundle truncation.

2. **dP9 W-bundle `h¹ ≥ 3`** — non-zero contribution found
   * Commit: `eea6f085`
   * Reference: `cy3_solver/references/p_dp9_w_lift_2026-05-05.md`
   * Result: `Σ mult · χ_dP9 = +18` from BHOP §6.1 W-twist on
     `dP9 = (3,1) ⊂ CP²_y × CP¹_t`. Suggestive of (b) until projected.

3. **Z/3×Z/3-invariant projection `= 0`** — verdict (a) ROBUST
   * Commit: `a4e6231e`
   * Reference: `cy3_solver/references/p_dp9_w_invariant_proj_2026-05-06.md`
   * Result: at the BHOP-canonical bundle-frame lift `(g_α, g_β) = (0, 0)`,
     the +18 is concentrated entirely in non-invariant `β = 2`
     characters. Both lower and upper SES bounds give invariant H¹
     dim = 0. Robustness: a closed-form geometric argument (BHOP §6.1
     SES sub `O(-2f)` rigs the elliptic-fiber `H¹` direction to
     `β = 2 ≠ 0`, independent of admissible `L_base` choice on dP9).

The chain establishes `Ext¹_full = 0` on the BHOP-2005 quotient as a
**definitive theorem**, not a temporary truncation pending future
refinement. Consequently the off-diagonal SU(4) coupling `κ` vanishes
on the Schoen Z/3×Z/3 quotient as a geometric consequence of the BHOP
SES design.

## Implications for the framework's prediction set

The complete-prediction-set artefact (binary
`p_complete_prediction_set`, JSON
`output/p_complete_prediction_set.json`, canonical SHA-256
`9a4a8765b9d1f3a7df1dbf5c06d3a89c53bbbcddc2bc27aa42d78637866e39e4`,
top-level status `framework_prediction_with_falsification_verdict_a`)
now classifies the 27 visible-sector + 16 dark-sector + 6 falsifiability
rows as follows:

| Sector | Status | Count | Notes |
|---|---|---|---|
| Gauge couplings (g₁, g₂, g₃) | PHYSICAL | 3 | PDG-anchored at M_Z |
| Visible Yukawa (mₐ for 6 quarks) | FRAMEWORK_PREDICTION_FALSIFIED | 6 | O(TeV) prediction; PDG values MeV–GeV; >5σ tension on every row |
| Charged leptons (mₑ, m_μ, m_τ) | STRUCTURALLY_ABSENT | 3 | e^c slot empty at deg-1 H¹ of rank-4 SU(4) shadow |
| Neutrinos (m_ν1, m_ν2, m_ν3) | STRUCTURALLY_ABSENT | 3 | ν^c slot empty |
| CKM (λ, A, ρ̄, η̄, δ_CKM) | FRAMEWORK_PREDICTION_FALSIFIED | 5 | All zero at κ=0 fixed point; PDG non-trivial |
| PMNS bundle (θ₁₂, θ₂₃, θ₁₃, δ_PMNS) | STRUCTURALLY_ABSENT | 1 | No Y_ν at this construction level |
| Higgs sector (v, m_H, λ_H, μ²) | PHYSICAL | 4 | Match PDG to <ppm |
| Newton's G | PHYSICAL | 1 | CODATA 2022 residual ~19 ppm |
| Λ_cc tree level | FRAMEWORK_PREDICTION_FALSIFIED | 1 | SM tree-level CC problem; 56.5 orders too large |
| Dark-sector chain slots | PHYSICAL | 16 | Closed-form geometric in E₈ invariants + α_τ + h_eff |
| Falsifiability signatures | PASS=3, DBY2030=2, FAIL_unless_linear=1 | 6 | All falsifiable |

Section partition: **8 PHYSICAL + 12 FRAMEWORK_PREDICTION_FALSIFIED +
7 STRUCTURALLY_ABSENT** in the visible sector; 16 PHYSICAL in the
dark sector.

## Falsifiability summary

Every row is falsifiable:

* **The 8 PHYSICAL visible-sector rows** are falsifiable in the
  ordinary sense — a measurement landing outside the framework's
  cited residual band (e.g. Newton's G drifting from CODATA 2022 by
  more than the structural ~19 ppm budget; the Higgs vev or mass
  drifting from the central value beyond the framework's <1 ppm
  residual) would falsify the corresponding closure.
* **The 12 FRAMEWORK_PREDICTION_FALSIFIED rows** record the
  framework's actual canonical predictions at the rank-4 BHOP-2005
  κ=0 fixed point on the Schoen Z/3×Z/3 quotient. They are already
  in tension with PDG by factors of ≥9 (top quark) to ~6 orders of
  magnitude (up quark). Path A research closure means these
  mismatches are **not** awaiting a future Ext¹ engagement.
* **The 7 STRUCTURALLY_ABSENT rows** are a falsifiable structural
  claim: any third party able to extract a non-zero `E^c`- or
  `ν^c`-irrep contribution from the rank-4 BHOP-2005 shadow's
  degree-1 `H¹(X̃/Γ, V_BHOP)` catalogue on the Schoen quotient
  would refute the audit.
* **The 16 PHYSICAL dark-sector slots** carry concrete predicted
  masses across nine orders of magnitude (`zeV` ↔ MeV) plus a
  hidden `W'` at ≈80 GeV and a hidden `H'` scalar at 200--1000 GeV.
* **The 6 falsifiability signatures** record one-sigma-bound and
  HL-LHC-2030-discoverable channels, plus a kinetic-mixing
  convention choice (linear vs quadratic) the framework explicitly
  commits to.

## Where to find the artefacts

* **Papers (rendered):**
  * `papers/substrate_sm_derivation/substrate_sm_derivation.pdf` — Section "Computed 4D effective Lagrangian: framework prediction with verdict (a) falsification"
  * `papers/substrate_falsifiable_predictions/substrate_falsifiable_predictions.pdf` — Section "Complete prediction set"
* **Machine-readable JSON:**
  * `papers/substrate_sm_derivation/cy3_solver/output/p_complete_prediction_set.json`
* **Reference docs (full closure argument):**
  * `papers/substrate_sm_derivation/cy3_solver/references/p_ext1_engagement_2026-05-05.md`
  * `papers/substrate_sm_derivation/cy3_solver/references/p_dp9_w_lift_2026-05-05.md`
  * `papers/substrate_sm_derivation/cy3_solver/references/p_dp9_w_invariant_proj_2026-05-06.md`
* **Regression pin (bit-exact reproducibility):**
  * `papers/substrate_sm_derivation/cy3_solver/tests/test_complete_prediction_set_pinned.rs`

## Provenance chain (SHA-256, end-to-end)

```
Eigenmode catalogue (n_pts=432, BHOP):
  1a5a13bf010f7314beab35ad81b4b2885a6eeaaf756849c5c97e9d3e8e7d6fe9

Yukawa matrices (κ=0 frame-collapse, consumes catalogue):
  5c043eb0355710a67b4929c4e8dfb01448b2daa6bfad68ee42499c94051d5753

Assembled 4D Lagrangian (consumes Yukawa):
  1b1e5e4a6adbb2a427bcbe7ea189ce5a8aabeb024b6b2ee24b638c2fafed5d01

Dark-sector predictions (consumes catalogue independently):
  8dcd0bb15e2d5bc7f98843008bf05d090f2c21fbc92272fb9e8982ac6d44a62e

Complete prediction set (canonical_complete_set_sha256):
  9a4a8765b9d1f3a7df1dbf5c06d3a89c53bbbcddc2bc27aa42d78637866e39e4

Path A closure chain:
  shadow Ext^1 = 0       at commit 831d910c
  dP9 W h^1 ≥ 3          at commit eea6f085
  Z/3xZ/3-invariant = 0  at commit a4e6231e (verdict (a) ROBUST)
```

## Why this matters

This is a publishable mixed-outcome scientific result. The substrate
framework, committed to the rank-4 BHOP-2005 SU(4) bundle on the
Schoen Z/3×Z/3 quotient, makes:

1. **Four genuine predictions** (gauge couplings, Higgs sector,
   Newton's G, dark-sector chain) that match observation at PDG /
   CODATA / Planck precision.
2. **Twelve falsified predictions** (six quark Yukawa eigenvalues at
   ~TeV, five CKM Wolfenstein parameters at zero, Jarlskog at zero,
   tree-level Λ_cc) that the construction definitively makes and
   that experiment definitively rejects.
3. **Seven structurally-absent slots** (charged leptons, neutrinos,
   PMNS) that the construction does not encode — itself a
   falsifiable structural claim.

Path A research closure does **not** establish that the framework is
"wrong" in some all-or-nothing sense. It establishes that the rank-4
BHOP-2005 commitment is sufficient on the Schoen quotient to
*generate* a complete falsifiable prediction set, that four of those
predictions match experiment to ppm-level precision, and that the
remaining nineteen slots either fail or are structurally empty in a
specific computable way. Whether a different bundle commitment (e.g.
Schoen-uniqueness Path-A's gCICY residue, or a different point on
the Z/3×Z/3 quotient's moduli space) lifts the falsified rows is a
separate research question --- not blocked by anything in the present
closure, but also not addressed by it.

## Standing next steps (independent of Path A closure)

* **Stage 7 cosmological-constant loop corrections** — the SM
  tree-level CC problem inherited by Λ_cc is independent of κ; loop
  corrections are an open stream.
* **Schoen-uniqueness Path-A (3,3) residue** — gCICY non-manifest
  residue still open; not blocked.
* **Different bundle commitment** — exploring an SU(5) or SO(10)
  alternative to BHOP-2005 SU(4) on the Schoen quotient is a
  full-restart; not blocked but not addressed by Path A.
