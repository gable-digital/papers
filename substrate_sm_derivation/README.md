# Substrate Standard Model Derivation

**Status: Working draft — not yet submitted for peer review.**

This folder holds `substrate_sm_derivation.pdf`, a working draft titled
*"One-anchor structural construction of Standard Model observables, a dark-matter
tower, and Newton's constant from E_8 × E_8 Coxeter geometry on a Schoen
Calabi–Yau threefold"* by Eliot Gable (Gable Digital Solutions, Inc.).

The PDF in this folder is a snapshot. It will be updated as the open questions
listed below are resolved. Treat any specific numbers as current best values,
not as published results.

## Background

The paper presents a structural construction of the Standard Model parameter
spectrum from a single dimensional anchor (the measured electron rest mass)
plus the discrete geometry of E_8 × E_8 heterotic compactification on a
Schoen Calabi–Yau threefold quotient X / (Z_3 × Z_3), combined with the McKay
correspondence between E_8 and the binary icosahedral group.

Three threads converge:

1. **Substrate-physical foundations.** The speed of light is interpreted as the
   universal substrate coupling-rate ratio c = ℓ_c / τ_c, Planck's constant as
   the universal substrate threshold magnitude, and general relativity, quantum
   mechanics, Newton's laws, and Maxwell's equations are recovered as
   macroscopic limits of one elastic-medium structure
   c = sqrt(Θ_restoring / I_inertial).

2. **E_8 spectrum and McKay duality.** Coldea et al. (2010) confirmed the
   Zamolodchikov E_8 mass spectrum in the quantum-critical point of
   CoNb_2O_6, with golden-ratio mass ratios. The McKay correspondence
   identifies the binary icosahedral group as the SU(2) dual of E_8, fixing
   icosahedral order-5 structure as the harmonic seed of the φ-chain.

3. **Heterotic compactification on the Schoen quotient.** A
   substrate-tension interpretation of the Higgs mechanism on the Schoen
   fiber-product quotient X / (Z_3 × Z_3) (Schoen 1988; Braun–He–Ovrut–Pantev
   2005; Donagi–Ovrut 2005) closes the geometric-to-spectral loop.

The longer pedagogical exposition of the substrate ontology, the discrete
isotropic update rule, and the elastic-medium identification lives in the
companion book *Exist: The Universal Ontology*, Book III. Readers wanting
the derivational story rather than the compressed paper form should start
there.

## Current state of the work

What the paper currently constructs from a single anchor (electron mass):

- All 26 Standard Model parameters in the mass-spectrum construction (gauge
  couplings, Higgs sector, charged-fermion Yukawas, three neutrino masses,
  CKM and PMNS, θ_QCD).
- Newton's gravitational constant G to ~2.5 ppm of CODATA 2022 (below the
  current inter-experiment scatter in direct laboratory determinations).
- Planck mass.
- A discrete seven-mode dark-matter mass tower.
- A misalignment-style Ω_DM h^2 abundance estimate of 0.119982, consistent
  with Planck 2018 (0.1200 ± 0.001) at 0.015%, with no fitted cosmological
  parameter.
- Two-anchor self-consistency (electron and muon) verified at 0.0011 ppm
  (~1.1 ppb) across the prediction set.
- Predictions for both CP phases as rational multiples of π
  (δ_CP = π + π/10, δ_CKM = 11π/30), the weak mixing angle as
  sin^2 θ_W = 2/9 + Δ_φ / h, and α_s(M_Z) = 849/7192. Normal neutrino
  mass ordering is selected.
- A conditional geometric route to vanishing strong CP via the Z_3 × Z_3
  quotient projection, without invoking a Peccei–Quinn axion.

No fitted continuous parameters are introduced in the mass-spectrum
construction; every structural integer is fixed by E_8 algebra, the Schoen
quotient geometry, the McKay correspondence, or framework assignments
built from those structures.

**Epistemic status.** The relationship of the substrate-physical inputs to
the specific structural configuration used here is one of *strong selection*
rather than *forced derivation*: the substrate, heterotic, McKay, and
Lie-algebraic inputs narrow the space of admissible configurations sharply
enough that this configuration is the natural landing — but a single
deductive link is missing (see open questions).

## Open questions and efforts underway

The list below is what the draft does *not* yet close. Active work is
tagged.

### 1. Substrate → heterotic moduli projection

The single missing deductive link required to upgrade the construction from
*strong selection* to *forced derivation* is an explicit substrate-physical
projection onto the heterotic moduli of X / (Z_3 × Z_3). Until this is
written down, the choice of Schoen quotient is selected by the framework
rather than forced by it.

*Effort underway:* characterizing the substrate-physical observables that
would constrain the heterotic moduli space, and identifying which moduli
are fixed by substrate inputs vs. left as discrete framework choices.

### 2. Schoen vs. competing CY3 backgrounds — discrimination experiment

A separate σ-discrimination program in the monorepo (`book/scripts/
cy3_substrate_discrimination/`) tests whether the framework's predictions
are tight enough to *distinguish* the Schoen Z/3 × Z/3 background from
plausible alternatives (TY/Z3 line bundles, Fermat quintic) via numerical
Donaldson-metric statistics. Current state: TY-vs-Schoen at 6.92σ Tier 0,
publication-quality after seven rounds of hostile review (n_pts=40k, strict
residual+iter convergence, BCa CI alongside percentile). Schoen wins both
σ-discrimination and the BBW-correct Yukawa chain-match.

### 3. E_8 ω_fix invariant — algebraic vs. eigenvalue interpretation

The journal-classification question of whether ω_fix = 123/248 is an
*algebraic E_8 invariant* (used as a coefficient) or a *measurable
eigenvalue* (testable as a spectrum entry) was resolved on the algebraic
side: ω_fix = (dim − 2) / (2 · dim) is an exact E_8 invariant used as a
coefficient in the m_e formula, not a measurable eigenvalue. The empirical
content (0.0011 ppm dual-anchor self-consistency) verifies algebraically. The
P7.1–P7.10 series tested a hypothesis the framework does not actually
make and has been removed from the test matrix.

### 4. Standalone bundle-provenance audit

The line-bundle Standard Model provenance was re-audited (arXiv:1106.4804
and hep-th/0512149 do *not* contain the TY/Z3 or Schoen Z/3 × Z/3
line-bundle SMs cited in earlier drafts; the correct papers are 0911.1569,
0910.5464, 1202.1757). The `*_canonical_*` builders that propagated the
false provenance have been retracted. Any future paper revision must cite
only the corrected sources.

### 5. Yukawa kernel-selection robustness

For Yukawa overlaps on real Donaldson metrics, the harmonic kernel must be
selected by the BBW dimension count, not by the FS-Gram-identity
heuristic. The σ → Yukawa decoupling has been re-verified bit-identical
under the corrected (BBW-correct, 9-mode) kernel.

### 6. Independent reproducibility of the numerical pipeline

The Schoen sampler (DHOR-2006 §3 fiber-product on CP² × CP² × CP¹) and the
Donaldson-metric numerics live alongside the paper in the monorepo. The
remaining work is packaging these so an external reviewer can rerun the
pipeline end-to-end without the in-house tooling, and producing a
reduced reproducibility artifact alongside the next paper revision.

## Reading order for new readers

1. The paper PDF in this folder — compressed, formal, REVTeX.
2. *Exist: The Universal Ontology*, Book III — pedagogical, narrative,
   with the substrate-ontology motivation that the paper assumes.
3. The `cy3_substrate_discrimination` numerical-experiment write-ups for
   the Schoen-vs-alternatives discrimination evidence.

## Contact

Eliot Gable — 10015996+egable@users.noreply.github.com — Gable Digital Solutions, Inc.
