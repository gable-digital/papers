# Stage 6 — PPN parameters from substrate structural corrections (back-of-envelope)

**HEADLINE (revised post-reviewer-critique).** The framework's 4D EFT,
after BHOP-2005 §6 dimensional reduction of the Schoen `Z_3 × Z_3` /
rank-4 SU(4) bundle setup, is a standard heterotic-orbifold Standard
Model: SU(3)_C × SU(2)_L × U(1)_Y gauge theory + three chiral
generations + a Higgs doublet, all minimally coupled to gravity at
tree level. **It therefore trivially passes every Cassini, LLR,
Mercury-MESSENGER, and binary-pulsar PPN test** — for the same reason
GR, Brans-Dicke at large ω, and the Standard Model coupled to gravity
all pass them. **This result is NECESSARY (any working heterotic-SM
compactification has to clear PPN) but NOT DISCRIMINATING (it does not
test the framework's structural content). PPN does not validate the
σ-discrimination, the chain-position formulas, the closure-rational
identifications, or the bundle eigenmode catalogue.** It only confirms
that the 4D EFT we land on is Lorentz-invariant and minimally coupled
— a property the framework shares with essentially any
Lorentz-invariant scalar-tensor or gauge theory.

The discriminating tests for the framework live elsewhere; see §7.

**Reviewer's critique (the source of this reframe).** A peer reviewer
flagged three precise issues with the prior "PASS by 31 orders of
magnitude" framing, and this revision incorporates all three:

1. *Trivially zero from operator absence is not discriminating.* PPN
   parameters are zero when the corresponding operators aren't in the
   action; that is a property the framework shares with any
   Lorentz-invariant scalar-tensor theory minimally coupled to gravity,
   and it does not validate the framework's structural content.
2. *31-orders-of-magnitude margin needs explicit small-parameter
   identification.* When a theory predicts effects 31 orders below the
   bound, either there is an identifiable small parameter, the calculation
   has implicitly set the relevant coupling to zero, or it is not
   computing what is claimed. We now identify the small parameters
   explicitly per reading per PPN parameter (§4.1, §4.2).
3. *The "framework's φ" is a single scalar or many?* The reviewer asked
   that we write down, in one paragraph, the 4D field content. We do
   that in §3 (Field-content audit) and make explicit that φ is the
   collection of bundle-Laplacian eigenmodes from BHOP §6, not a single
   scalar.

The numerical results, the PPN formulas, and the operator-content
reading definitions (Readings A/B/C) are unchanged from the prior
version of this document; what changes is the framing.

---

## 1. Framework structural corrections summary

From `book/paper/substrate_particle_equations.tex`, the four structural
corrections appearing in the Higgs-vev ladder
`v_C = (dim − 2) · 1/(1 − δ_W) · (1 + 5α_τ³/8) · (1 + ζ_DM)`
(Eq. v-ladder, line ~2181) and in the dark-matter-tower kinetic-mixing
register (Eq. m-DM-tower / epsilon-DM-tower, lines 2212–2216) are:

| Symbol      | Closed form                            | Magnitude          | Substrate physical reading |
|-------------|----------------------------------------|--------------------|----------------------------|
| `α_τ`       | `1/(dim + h_{E_8}) = 1/278`            | `3.597 × 10⁻³`     | Inverse "tower depth", master expansion parameter |
| `δ_H`       | `1/(h_{E_8}+m_max) = 1/59`             | `1.695 × 10⁻²`     | Higgs-scalar dark-sector kinetic mixing (visible Higgs ↔ hidden-E_8' Higgs') |
| `δ_W`       | `δ_H / e_6 = 1/1121`                   | `8.921 × 10⁻⁴`     | W-boson dark-sector kinetic mixing (visible W ↔ hidden W'); palindromic pair (e_3=11, e_6=19) |
| `5α_τ³/8`   | `5·(1/278)³/8`                         | `2.908 × 10⁻⁸`     | Third-order self-coupling closure (icosahedral McKay 5 / E_8 rank 8) |
| `ζ_DM`      | `Σ_{n=5}^{11} α_τ^n`                   | `6.04 × 10⁻¹³`     | Dark-matter sub-gateway tower kinetic-mixing summed bleed-through |

The substrate paper itself flags Step 2.5's reading of `δ_W` as
**"(I, palindromic-pair reading)"** — interpretive, not derived from
heterotic principles. The numerical magnitudes are fixed by
Coxeter-exponent arithmetic and are not tunable.

**Crucial physical claim for PPN.** Both `δ_W` and `δ_H` are described
in the paper as **kinetic-mixing leakages between the visible and
hidden E_8 sectors**. The dark-matter tower's per-mode kinetic-mixing
weight `ε_n = α_τ^n` is identified as "visible-sector kinetic mixing"
(Eq. epsilon-DM-tower, line 2214) — Holdom (1986) style off-diagonal
`(ε/2) F^{μν} F'_{μν}`, **not** a non-minimal scalar-curvature
coupling `ξ φ² R`.

Whether such mixings actually appear in the 4D EFT depends on Stage 4
(explicit dimensional reduction), which is not yet derived. The
operator-content readings (§4) span three options for what δ_W and
δ_H *might* mean as 4D operators.

---

## 2. PPN parameter bounds (citations)

Standard reference: **Will, "The Confrontation between General
Relativity and Experiment", Living Rev. Relativ. 17, 4 (2014)**,
Table 4 ("Current limits on the PPN parameters"). Updated post-2014
where applicable.

| Parameter | Meaning                              | Bound             | Source |
|-----------|--------------------------------------|-------------------|--------|
| γ − 1     | Light-bending / Shapiro time-delay   | (2.1 ± 2.3)·10⁻⁵  | Bertotti, Iess, Tortora 2003 (Cassini), Nature 425, 374 |
| β − 1     | Perihelion advance / Nordtvedt       | < 8·10⁻⁵          | Park et al. 2017 (MESSENGER/Mercury) — Will 2018 update; LLR + Cassini combined |
| α_1       | Preferred-frame (orbital)            | < 4·10⁻⁵          | Müller, Williams, Turyshev 2008 (LLR); Will 2014 §4.1.2 |
| α_2       | Preferred-frame (spin)               | < 2·10⁻⁹          | Shao & Wex 2012, 2013 (PSR J1738+0333, J1012+5307) |
| α_3       | Preferred-frame energy non-conservation | < 4·10⁻²⁰      | Bell & Damour 1996, pulsar timing |
| ξ         | Whitehead / preferred-location       | < 4·10⁻⁹          | Shao, Sanders, Schmidt-Wellenburg 2018 |
| ζ_1       | Conservation of momentum             | < 2·10⁻²          | Will 2014 §4.4 |
| ζ_2       | Conservation of momentum             | < 4·10⁻⁵          | Will 2014, binary pulsars |
| ζ_3       | Conservation of angular momentum     | < 1·10⁻⁸          | Lunar laser ranging |
| ζ_4       | Conservation                         | not directly bounded |  Will 2014 §4.4 |

For GR, all PPN parameters except γ = β = 1 are zero. Any deviation
must fit inside these bands.

---

## 3. Field-content audit (Reviewer Issue 3: "what is φ?")

The reviewer asked us to write down, in one paragraph, what the
framework's φ actually is — one scalar, or many — and where the QCD
gauge structure, chiral fermions, and SU(2)_L Higgs doublet live in
the action.

**The framework's φ is NOT one scalar.** It is the **collection of
bundle-Laplacian eigenmodes** of the box operator `□_V` acting on
sections of the BHOP rank-4 SU(4) holomorphic bundle `V` (BHOP-2005,
"Stable Bundles and Yukawa Couplings on a Calabi-Yau Threefold")
restricted to the Schoen `Z_3 × Z_3` quotient `X/Γ`. The 4D field
content, as specified in BHOP-2005 §6 + Stage 1–4 of the framework's
program plan, is:

- **Gauge fields.** SU(3)_C × SU(2)_L × U(1)_Y. These descend from
  the unbroken subgroup of E_8 after `Z_3 × Z_3` Wilson-line breaking
  via the embedding `SU(4) ⊂ E_8 → SU(3)_C × SU(2)_L × U(1)_Y ×
  (E_8' hidden)`, exactly as in standard heterotic orbifold model
  building (see BHOP §3 for the Wilson-line embedding, BHOP §4 for
  the unbroken-subgroup analysis).
- **Three chiral generations.** Three copies of the SO(10) **16**
  (with right-handed neutrinos automatic), realised as the
  cohomology classes `H¹(X/Γ, V_BHOP)` decomposing **3:3:3** under
  the Z_3 × Z_3 Wilson-line characters. The "three generations" claim
  is the cohomology-counting result of BHOP §6 + Stage-2 verifier.
- **Higgs doublet.** The SU(2)_L doublet H descends from a **10** of
  SO(10) cohomology class (in BHOP's setup, from `H¹(X/Γ, ad V)` or
  `H¹(X/Γ, V ⊗ V*)` depending on convention).
- **Yukawa couplings.** Triple-overlap integrals
  `∫_{X/Γ} ψ_i ⋆ ψ_j ⋆ φ_H` between three eigenmodes of `□_V` (two
  fermion modes, one Higgs mode), evaluated on Schoen with `V_BHOP`.
  These are **Stage 3 deliverables** (currently blocked on the
  bundle-twisted Dirac kernel implementation in `pwos-torch`).
- **Mass formulas `m_X = h_eff · R_X`.** Structural predictions about
  what the eigenmodes' physical masses turn out to be once the
  Yukawa overlap integrals + EW symmetry breaking are evaluated. The
  closure-rational `R_X` quantities are **eigenvalue ratios of the
  bundle Laplacian**, predicted by the geometric setup; they are
  **NOT** a mechanism by which one scalar produces all SM masses.

So when this document refers to "φ" or "the Higgs scalar" in PPN
formulas, it specifically means **the SM Higgs doublet H** — the 4D
mode that descends from the Higgs cohomology class above. The
"substrate" δ_W, δ_H, ε_n parameters are **not** other scalars sitting
in the 4D action; they are either (i) substrate-level mode-pattern
overlap rates that happen to determine v_C numerically without
appearing as 4D Lagrangian operators (Reading A) or (ii) Holdom-style
gauge kinetic mixings between the visible and hidden E_8' sectors
(Reading B). They are not 4D scalar fields.

This is essentially the standard heterotic-SM field content. The
substrate framework's discriminating content is in (1) which CY3 is
chosen (Schoen vs. TY — settled by σ-discrimination at 6.92σ; see §7),
(2) which bundle is chosen (BHOP rank-4 SU(4)), (3) which Wilson-line
breaking pattern is used (Z_3 × Z_3), and (4) the closure-rational
predictions for the eigenmode mass spectrum (R_X). It is **not** in
the gravitational sector.

---

## 4. Per-PPN-parameter computation at leading order

### 4.0 Three operator-content readings

The structural arithmetic fixes `v_C` numerically but does not, by
itself, fix the 4D operator structure of the dark-sector mixings.
There are three plausible readings, all computed in the companion
Python script (`ppn_gr_tests.py`):

- **Reading A — substrate-overlap (canonical).** δ_W, δ_H are
  substrate-level mode-pattern overlap rates. They never appear as
  4D Lagrangian operators; they only set the coefficient relating
  bare to physical electroweak VEV. All PPN parameters are
  identically zero at tree level. **(necessary, not discriminating;
  see §4.1.)**
- **Reading B — Holdom kinetic mixing (stress test).** δ_W is a
  gauge kinetic-mixing coefficient `(ε/2) B^{μν} B'_{μν}` between
  visible U(1) and hidden U(1)'. Two convention variants: `ε = δ_W`
  (linear) vs `ε² = δ_W` (quadratic); the linear convention is
  more conservative and is the default. **(stress test; see §4.2.)**
- **Reading C — Higgs non-minimal coupling (stress test).** The
  heterotic UV completion generates `ξ H†H R` at one loop or via
  warping, with ξ at most `O(α_τ²) ~ 10⁻⁵` (or a generic loop-level
  estimate `1/(16π²) ~ 6·10⁻³`). **(stress test; see §4.2.)**

### 4.1 Reading A — necessary, not discriminating

For Reading A, all PPN parameters are zero **for the same reason
they vanish in any Lorentz-invariant scalar-tensor theory minimally
coupled to gravity**: the action has no preferred-frame vector,
no preferred-location tensor, no `ξ φ² R` non-minimal coupling, and
no `(ε/2) F F'` gauge mixing in the 4D EFT. Specifically:

- γ − 1 = 0 because there is no `F(φ) R` non-minimal coupling
  (Will 2014, Eq. 3.46 with F' = 0).
- β − 1 = 0 by the same operator absence (Will 2014, Eq. 3.48).
- α_1 = α_2 = α_3 = 0 because the 4D EFT is Lorentz-invariant
  (BHOP §6 Wilson-line breaking on a static internal manifold
  preserves 4D Lorentz covariance) and there is no preferred-frame
  vector field in the action.
- ξ = 0 because there is no preferred-location tensor (the CY3 has
  no 4D-translation-breaking remnant in the EFT).
- ζ_i = 0 because the 4D EFT is energy-momentum and angular-momentum
  conserving.

**This is exactly what GR predicts. It is what Brans-Dicke at large ω
predicts. It is what the Standard Model coupled minimally to gravity
predicts.** Reading A's PPN result is therefore a property the
framework shares with essentially any Lorentz-invariant scalar-tensor
theory — a **NECESSARY** consistency check (any working heterotic-SM
compactification must reproduce it) but **NOT DISCRIMINATING** (it
does not test what makes the framework distinct from GR or any other
Lorentz-invariant theory).

The conclusion to draw from Reading A is: *the framework's
gravitational sector is consistent with the GR-PPN regime*. It is
**not** a free win on a non-trivial GR test, and it does not validate
the framework's structural content (CY3 choice, bundle, Wilson-line,
eigenmode catalogue).

### 4.2 Readings B and C — explicit small-parameter identification

For Readings B and C the answers are not zero but are still microscopic.
Per Reviewer Issue 2 we now identify the **explicit small parameters**
producing the suppression, so a referee can audit the suppression
chain.

#### Reading B: small parameters

The leading PPN entry under Reading B is

`α_1 ~ 4 ε² · (m_W / M_Pl)² · O(1)`

(Will 2014 §3.4, Damour-Esposito-Farèse 1992; Wagoner-style
dimensional counting for one-graviton-exchange between visible W and
hidden W' through their kinetic mixing). The **two physically
meaningful small parameters** are:

1. **`ε² = δ_W ≈ 8 × 10⁻⁴`** under the *linear* convention `ε = δ_W`
   (the framework's δ_W is the kinetic-mixing rate squared in this
   convention). Under the *quadratic* convention `ε² = δ_W`, instead
   `ε² ≈ 9 × 10⁻⁴` — i.e. four orders larger. The convention switch
   matters: **linear `ε = δ_W` gives 4 orders more suppression than
   quadratic `ε² = δ_W`.** We default to linear (more conservative)
   in the Python script and flag the alternative.
2. **`(m_W / M_Pl_red)² ≈ (80.4 GeV / 2.435·10¹⁸ GeV)² ≈ 1.1·10⁻³³`**
   — the **graviton-mediated dark-photon-to-photon two-graviton-exchange
   suppression**, i.e. the standard Wagoner-style dimensional counting
   for a hidden-sector vector at one-graviton-exchange in the PPN
   reduction. This is the universal `(m_V / M_Pl)²` suppression that
   any visible-sector gauge-mixing operator inherits when promoted
   to a PPN preferred-frame parameter, since the PPN expansion lives
   at order `(v/c)² ~ G M / r c² ~ (m / M_Pl)²` for a system of mass
   m at separation `r ~ G M`.

Putting these together:

`α_1 ~ 4 · 8·10⁻⁴ · 10⁻³³ ~ 10⁻³⁵ ... 10⁻³⁶ (linear convention)`
`α_1 ~ 4 · 10⁻³ · 10⁻³³ ~ 10⁻³⁵ (quadratic convention; ~3.5× larger)`

Bound: `|α_1| < 4·10⁻⁵`. **Margin ~ 10³¹** under either convention.

The two physical statements above are auditable: ε² is the kinetic-
mixing rate of a Holdom-style operator (look at the 4D Lagrangian
coefficient), and `(m_W/M_Pl)²` is the dimensional counting of how
that coefficient enters the gravitational sector through one-
graviton-exchange. Both are **small for physical reasons**, not by
implicit gauge choice.

#### Reading C: small parameters

The leading PPN entry under Reading C is

`γ − 1 ~ −F'² / F  with  F = M_Pl² + ξ v² ,  F' ~ 2 ξ v`

so `|γ − 1| ~ (2 ξ v / M_Pl)² = 4 ξ² (v / M_Pl_red)²`. The **two
physically meaningful small parameters** are:

1. **`ξ² ≈ 10⁻⁴`** — the natural-size loop-level non-minimal
   coupling under the conservative tower-mode estimate `ξ ~ α_τ²
   ~ 10⁻⁵`, giving `ξ² ~ 10⁻¹⁰`; or under a generic one-loop
   estimate `ξ ~ 1/(16π²) ~ 6·10⁻³`, giving `ξ² ~ 4·10⁻⁵`. Either
   way, `ξ²` is small for the standard EFT-loop-suppression reason.
2. **`(v / M_Pl_red)² ≈ (246 GeV / 2.435·10¹⁸ GeV)² ≈ 10⁻³²`** —
   the **EW-to-Planck hierarchy squared**. This is the same
   small parameter that suppresses every standard-scalar-tensor-PPN
   estimate involving the SM Higgs, and it is the textbook reason
   that scalar-tensor theories with EW-scale scalars trivially clear
   Cassini.

Putting these together:

`|γ − 1| ~ 4 · 10⁻⁴ · 10⁻³² ~ 4·10⁻³⁶ (one-loop ξ estimate)`
`|γ − 1| ~ 4 · 10⁻¹⁰ · 10⁻³² ~ 4·10⁻⁴² (tower α_τ² ξ estimate)`

Bound: `|γ − 1| < 2.3·10⁻⁵` (Cassini). **Margin ~ 10³¹ to 10³⁷.**

The two physical statements above are auditable: ξ is bounded by EFT
loop counting (any heterotic-SM compactification with no fine-tuning
hits this bound) and the EW-to-Planck hierarchy is the single most
firmly established small parameter in particle physics.

In summary: **the 31-orders-of-magnitude PPN margin under Readings
B/C is the product of explicit small parameters [ε², (m_W/M_Pl)², ξ²,
(v/M_Pl)²] and is sensitive to convention choice (linear vs quadratic
ε convention shifts the result 4 orders).** None of the suppressions
is "implicit zero coupling"; each is identifiable EFT power counting.

### 4.3 γ — Cassini light-bending (detailed)

In a Brans-Dicke / scalar-tensor theory with non-minimal coupling
`F(φ) R`,

`γ − 1 = −F'(φ)² / (F + 2 F'²)`        (Will 2014, Eq. 3.46, evaluated at φ = v).

For the SM Higgs, F = M_Pl²/(16π) is constant (no `ξ H†H R` in the
SM at tree level), so F' = 0 and **γ = 1 exactly** at tree level. The
framework reduces to this in Reading A (no `ξ` operator generated by
BHOP §6 dimensional reduction). In Reading C, γ − 1 ranges from
~10⁻³⁶ to ~10⁻⁴² depending on which one-loop ξ estimate is used (see
§4.2). All cases sit within Cassini's `2.3·10⁻⁵` band.

### 4.4 β — Mercury / LLR perihelion advance (detailed)

`β − 1 = (1/2) F F'' / (F + 3 F'/2)²`        (Will 2014, Eq. 3.48).

Same logic as γ: zero in Reading A; ~10⁻³⁶..10⁻⁴² in Reading C.
Within Mercury/MESSENGER's `8·10⁻⁵` band.

### 4.5 α_1, α_2 — preferred-frame channels (detailed)

In Reading A: zero (Lorentz invariance + minimal coupling).

In Reading B: `α_1, α_2 ~ ε² · (m_W/M_Pl)² ~ 10⁻³⁵..10⁻³⁶`,
small-parameter chain detailed in §4.2. Bounds:
`|α_1| < 4·10⁻⁵` (LLR), `|α_2| < 2·10⁻⁹` (PSR J1738+0333).

### 4.6 α_3 — preferred-frame energy non-conservation (detailed)

In all three readings: zero at the operator level. The 10D heterotic
action is Lorentz-covariant; KK reduction on a static internal
manifold preserves 4D Lorentz invariance. Kinetic-mixing operators
`(ε/2) F F'` are themselves Lorentz-covariant and do not source α_3.

Subleading corrections from cosmological time-dependence of moduli
(α_τ, ζ_DM running) are bounded by `1/H_0 ~ 10⁴² s` and give at most
α_3 ~ 10⁻⁴², deep below the `4·10⁻²⁰` pulsar bound.

### 4.7 ξ_PPN, ζ_1...ζ_4 — preferred-location and conservation laws

All zero at operator level under all three readings (4D Lorentz +
translation invariance, energy-momentum conservation in the visible
sector). Bounds cleared by ∞ margin.

---

## 5. Verdict table (revised)

| PARAMETER | READING A (canonical) | READING B (Holdom) | READING C (ξ HHR) | EXPERIMENTAL BOUND | VERDICT |
|-----------|----------------------|--------------------|--------------------|--------------------|---------|
| γ − 1     | 0 (Lorentz-inv. + min. coupling) | 0 | ~4·10⁻³⁶..10⁻⁴² (ξ², (v/M_Pl)²) | < 2.3·10⁻⁵ | **PASS** (necessary, not discriminating) |
| β − 1     | 0                                  | 0 | ~10⁻³⁶..10⁻⁴²                     | < 8·10⁻⁵   | **PASS** (necessary, not discriminating) |
| α_1       | 0                                  | ~10⁻³⁵..10⁻³⁶ (ε², (m_W/M_Pl)²) | 0       | < 4·10⁻⁵   | **PASS** (necessary, not discriminating) |
| α_2       | 0                                  | ~10⁻³⁵..10⁻³⁶                    | 0       | < 2·10⁻⁹   | **PASS** (necessary, not discriminating) |
| α_3       | 0                                  | 0 | 0                                                  | < 4·10⁻²⁰  | **PASS** |
| ξ_PPN     | 0                                  | 0 | 0                                                  | < 4·10⁻⁹   | **PASS** |
| ζ_1...ζ_4 | 0                                  | 0 | 0                                                  | various    | **PASS** |

**HEADLINE (revised).** The framework's 4D EFT, after BHOP §6
dimensional reduction, is a Lorentz-invariant heterotic-SM minimally
coupled to gravity at tree level. It trivially passes all PPN tests.
**The result is necessary (any working heterotic-SM compactification
must pass) but not discriminating (it does not test the framework's
structural content).** The 31-order-of-magnitude numerical margin
under Readings B/C is the product of explicit, auditable small
parameters: `ε²` or `ξ²` (gauge mixing or non-minimal coupling, both
small for EFT loop-counting reasons) times `(m_W/M_Pl)²` or
`(v/M_Pl)²` (the universal EW-to-Planck hierarchy). The result is
sensitive to convention choice (`ε = δ_W` vs `ε² = δ_W` shifts the
result 4 orders) but in either convention sits microscopically deep
inside the experimental bands.

---

## 6. UV completion questions that remain open

Only one bit of the calculation is **inconclusive** without explicit
Stage-4 input:

**Q1. Does the heterotic KK reduction generate a non-minimal
`ξ H†H R` operator at 4D, and if so, what is the natural size of ξ?**

The two natural answers (Reading A vs Reading C) give γ − 1 = 0 or
γ − 1 ~ 10⁻³⁶..10⁻⁴², respectively. Both are deep within Cassini.
The framework cannot fail γ unless the UV completion generates `ξ`
at an *enhanced* scale `ξ ~ O(1)` — which would require strong
coupling in the heterotic Higgs sector and is not motivated by any
of the structural commitments. We mark this scenario implausible
but not strictly excluded.

**Q2 (minor).** Does the dark-matter tower's per-mode kinetic mixing
`ε_n = α_τ^n` for n = 5..11 contribute coherently or incoherently to
α_1, α_2 under Reading B? Sum-vs-quadrature changes the estimate by
a factor ~7 at most; either way the result sits at ~10⁻³⁶ versus a
bound ~10⁻⁵, so the question is decoration on a 31-order margin.

**Q3 (also minor; new since reviewer revision).** Linear vs quadratic
ε convention under Reading B shifts the result 4 orders (10⁻³⁶ vs
10⁻³⁵). We default to linear (`ε = δ_W`, more conservative). The
Stage-4 explicit Lagrangian will fix this.

---

## 7. What PPN does NOT test, and where the discriminating tests live

Per Reviewer Issue 1, this section is now explicit.

**PPN does NOT test:**

- The σ-discrimination Schoen-vs-TY at 6.92σ (Tier 0; geometry test,
  bundle-independent). See `project_cy3_5sigma_discrimination_achieved.md`.
- Path-A Schoen uniqueness (catalogue test against the 473M-polytope
  KS catalogue). See `project_cy3_schoen_uniqueness.md`.
- Closure-rational structural identifications that are E_8-algebraic
  invariants (e.g. ω_fix = 123/248 as an exact algebraic identity).
  See `project_cy3_omega_fix_gateway_ladder.md`.
- The triple-anchor h_eff at 30 ppb (one-anchor predictivity).
- The chain-position formulas as bundle-Laplacian eigenvalues
  (Stage 3 work, currently blocked on bundle-twisted Dirac kernel
  implementation in `pwos-torch`).
- Precision m_W and m_H measurements vs the (1 − δ_W) and (1 − δ_H)
  structural predictions. Current PDG precision on m_W is ~10⁻⁴
  (12 MeV/80385 MeV); the framework's prediction precision under the
  closure-rational (1 − δ_W) factor is ~10⁻³, so even today's PDG is
  starting to bite into the framework's prediction band.

**PPN IS consistent with:**

- GR (γ = β = 1, all other PPN parameters = 0, Lorentz-invariant
  4D EFT).
- Brans-Dicke at large ω (γ − 1 → 0 as ω → ∞).
- The Standard Model minimally coupled to gravity at tree level.
- Any heterotic-SM compactification with no enhanced UV `ξ H†H R`
  operator.
- The substrate framework under Readings A, B, and C.

**PPN therefore confirms** that the framework's 4D EFT is what we
expect from a heterotic-SM compactification (Lorentz-invariant,
minimally coupled, no preferred-frame vector). It does **not**
validate the framework's structural content, which lives in the
internal manifold geometry, the bundle choice, the Wilson-line
breaking pattern, and the eigenmode catalogue.

**The discriminating tests for the framework are:**

| Discriminating test                                           | Status                                          |
|---------------------------------------------------------------|-------------------------------------------------|
| σ-discrimination Schoen-vs-TY at 6.92σ (Tier 0)               | **DONE**, n_pts=40k strict, BCa CI floor ≥5.30σ |
| Path-A Schoen uniqueness (cycles 1/2/2.5)                     | **DONE**, 0 non-Schoen survivors                |
| Triple-anchor h_eff at 30 ppb                                 | **DONE**, dual-anchor verified algebraically    |
| ω_fix = 123/248 as exact E_8 algebraic invariant              | **DONE**, P7.12 classification                  |
| Chain-position formulas as bundle-Laplacian eigenvalues       | **BLOCKED** on Stage 3 Dirac-kernel impl.       |
| Precision m_W (1 − δ_W), m_H (1 − δ_H) vs PDG                 | **OPEN**, framework precision ~10⁻³ vs PDG ~10⁻⁴ |
| Yukawa overlap integrals on Schoen + V_BHOP                   | **BLOCKED** on Stage 3                          |

The σ-discrimination + Schoen-uniqueness + triple-anchor h_eff +
ω_fix algebraic identification are the **publication-target
falsifiable claims** for the framework. PPN is a sanity check that
the gravitational sector does not blow up. PPN is not the
discriminating test.

---

## 8. Recommended action

### 8.1 Paper-prose updates

1. **Add a Stage-6 sub-section to `book/paper/substrate_particle_equations.tex`**
   under "Empirical predictions" or "Stage program", citing this
   analysis. **Headline claim (revised):**
   > The framework's 4D EFT, after BHOP-2005 §6 dimensional reduction
   > of the Schoen `Z_3 × Z_3` / rank-4 SU(4) bundle setup, is a
   > standard heterotic-SM minimally coupled to gravity at tree
   > level. It therefore trivially passes every Cassini, LLR, MESSENGER,
   > and binary-pulsar PPN test — for the same reason GR, Brans-Dicke
   > at large ω, and the SM-coupled-to-gravity all do. **This result
   > is NECESSARY (any working heterotic-SM compactification must
   > pass) but NOT DISCRIMINATING (it does not test the framework's
   > structural content).** The framework's discriminating tests are
   > the σ-discrimination at 6.92σ (Schoen vs TY), Path-A Schoen
   > uniqueness, triple-anchor h_eff at 30 ppb, the ω_fix = 123/248
   > algebraic identification, the bundle-Laplacian eigenvalue chain
   > formulas (Stage 3), and precision m_W, m_H measurements vs the
   > closure-rational `(1 − δ_W)`, `(1 − δ_H)` corrections. PPN is a
   > consistency check, not a validation of the framework's content.

2. **Cite Will (2014)** as the standard PPN reference and the
   reviewer's critique as the source of the "necessary not
   discriminating" framing.

3. **Do not over-claim.** Avoid "PASS by 31 orders of magnitude"
   language unless paired with explicit small-parameter
   identification (§4.2). The 31-order margin is the product of
   `ε²` or `ξ²` × `(m/M_Pl)²` and is auditable.

### 8.2 Additional research

1. **Stage 3 / Stage 4 still primary.** The bundle-Laplacian
   eigenmode calculation and the explicit 4D Lagrangian remain the
   highest-priority deliverables; PPN is an existence check, not a
   substitute.

2. **When Stage 4 lands**, redo this calculation with the explicit ξ
   value (or proof ξ = 0 at tree level). Converts Reading A vs
   Reading C disjunction into a single sharp prediction.

3. **Optional: dark-photon direct-detection cross-check** (Reading B
   side-effect). `ε² ~ δ_W ~ 9·10⁻⁴` is at or above existing
   dark-photon bounds for `m_{W'} ~ m_W`; testable at LDMX, FASER,
   SeaQuest if `m_{W'}` is sub-100 GeV. Out of scope for this
   Stage-6 ticket, but worth flagging.

4. **Lorentz-invariance test.** α_3 = 0 depends on the visible-sector
   Lagrangian being exactly Lorentz-invariant after KK reduction.
   If the framework's `Z_3 × Z_3` Wilson breaking leaves any
   tensor-structured remnant, α_3 could pick up a non-zero
   contribution. The current reading (Step 4.1–4.4) does not introduce
   a preferred 4D frame, but Step 4.4 is interpretive ("(I)"); should
   be re-checked when Stage 4 lands.

### 8.3 Bottom line (revised)

**The framework's 4D EFT is standard heterotic-SM at tree level
(after BHOP §6 dimensional reduction); it trivially passes all
GR/PPN tests for the same reason GR, Brans-Dicke at large ω, and
the SM-coupled-to-gravity all do.** This is a NECESSARY consistency
check (any working heterotic-SM compactification must pass) but
NOT a DISCRIMINATING test of the framework's structural content.
The 31-order numerical margin under stress-test Readings B and C is
the product of explicit small parameters [`ε²` or `ξ²`, `(m_W/M_Pl)²`
or `(v/M_Pl)²`] and is convention-sensitive (linear `ε = δ_W` vs
quadratic `ε² = δ_W` shifts the result 4 orders).

The framework's **discriminating** tests live in:

- σ-discrimination at 6.92σ (already done).
- Path-A Schoen uniqueness (already done).
- Triple-anchor h_eff at 30 ppb (already done).
- ω_fix = 123/248 as E_8 algebraic invariant (already done).
- Bundle-Laplacian eigenvalue chain formulas (Stage 3, blocked).
- Precision m_W, m_H vs `(1 − δ_W)`, `(1 − δ_H)` (open; PDG ~10⁻⁴
  starts to bite into framework prediction precision ~10⁻³).

PPN is not on this list, and presenting it as "a free win on a
non-trivial GR test" would be misleading. The honest framing is:
"the framework's gravitational sector is consistent with GR; the
falsifiable structural content lives elsewhere".

---

## References

- C. M. Will, *The Confrontation between General Relativity and
  Experiment*, **Living Rev. Relativ. 17, 4 (2014)**.
  arXiv:1403.7377. Standard PPN reference.
- B. Bertotti, L. Iess, P. Tortora, *A test of general relativity
  using radio links with the Cassini spacecraft*,
  **Nature 425, 374 (2003)**. Cassini γ bound.
- R. S. Park et al., *Precession of Mercury's Perihelion from Ranging
  to the MESSENGER Spacecraft*, **AJ 153, 121 (2017)**. β bound.
- J. Müller, J. G. Williams, S. G. Turyshev, *Lunar laser ranging
  contributions to relativity and geodesy*,
  **Astrophys. Space Sci. Proc. (2008)**. α_1 LLR bound.
- L. Shao, N. Wex, *New tests of local Lorentz invariance of gravity
  with small-eccentricity binary pulsars*, **Class. Quantum Grav. 29,
  215018 (2012); 30, 165020 (2013)**. α_2 pulsar bound.
- B. Holdom, *Two U(1)'s and ε charge shifts*,
  **Phys. Lett. B 166, 196 (1986)**. Foundational kinetic-mixing paper.
- M. Pospelov, *Secluded U(1) below the weak scale*,
  **Phys. Rev. D 80, 095002 (2009)**. Dark-photon phenomenology.
- R. Essig et al., *Dark Sectors and New, Light, Weakly-Coupled
  Particles*, **Snowmass 2013 report**, arXiv:1311.0029.
- T. Damour, G. Esposito-Farèse, *Tensor-multi-scalar theories of
  gravitation*, **Class. Quantum Grav. 9, 2093 (1992)**. Scalar-tensor
  PPN derivation.
- V. Braun, Y.-H. He, B. A. Ovrut, T. Pantev (BHOP), *Stable Bundles
  and Yukawa Couplings on a Calabi-Yau Threefold*, **JHEP (2005)**.
  Field-content audit (§3 above) source.
- R. M. Wagoner, *Scalar-Tensor Theory and Gravitational Waves*,
  **Phys. Rev. D 1, 3209 (1970)**. Wagoner-style dimensional counting
  for hidden-sector PPN at one-graviton-exchange.
- Substrate framework paper:
  `book/paper/substrate_particle_equations.tex`, especially:
  - Eq. (v-ladder) line 2181: four-factor Higgs vev ladder
  - Eq. (zetaDM) line 2188: dark-tower kinetic-mixing sum
  - Eq. (epsilon-DM-tower) line 2214: per-mode mixing weights
  - Step 2.5 lines 801–810: `δ_W` palindromic-pair reading (I-marked)
  - Lines 496–500: `δ_H` and `δ_W` definitions and physical readings
- Companion documents:
  - `project_cy3_5sigma_discrimination_achieved.md` — σ-discrimination
    at 6.92σ, the framework's discriminating geometry test.
  - `project_cy3_schoen_uniqueness.md` — Path-A Schoen uniqueness in
    the 473M-polytope KS catalogue.
  - `project_cy3_omega_fix_gateway_ladder.md` — ω_fix = 123/248 as
    E_8 algebraic invariant.

---

**Revision history.**

- *Original (Apr 2026, pre-reviewer-critique).* "PASS by ≥27 orders of
  magnitude" framing, single Reading; treated as a free win.
- *First reviewer wave.* Added Readings A/B/C, parametric-estimate
  caveat under Reading B, prefactor convention note (-8 vs -32) under
  Reading C.
- *This revision (May 2026, post-reviewer-critique #2).* Reframed as
  "necessary not discriminating" per Reviewer Issue 1. Added explicit
  small-parameter identification per Reviewer Issue 2 (§4.2). Added
  Field-content audit per Reviewer Issue 3 (§3). Added §7 listing what
  PPN does NOT test, what it IS consistent with, and where the
  framework's discriminating tests actually live. Numerical results,
  PPN formulas, and operator-content reading definitions unchanged.
