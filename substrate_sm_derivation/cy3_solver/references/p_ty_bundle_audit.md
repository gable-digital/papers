# P-TY-Bundle-Audit — No citable 3-generation line-bundle SM on Tian-Yau Z/3

**Date:** 2026-04-30
**Scope:** The Yukawa channel on the TY/Z3 candidate.
**Verdict:** **STOP.** No published heterotic line-bundle (or monad)
Standard-Model construction on the Tian-Yau Z/3 quotient exists in the
post-2008 catalogue literature. Fabricating bundle bidegrees to "match"
the 3:3:3 Wilson partition would be exactly the kind of unprovenanced
data this codebase has already retracted (see
`src/route34/hidden_bundle.rs:352-371`, the deprecated
`ty_aglp_2011_standard` alias, and the project memory note
`project_cy3_bundle_provenance_audit.md`). The Yukawa channel as a third
Bayes-factor entry is therefore **deferred** until a genuine reference
is wired. The standing 5.43σ chain-channel publication headline (Schoen
beats TY in chain-match by 5.43σ) is unaffected.

---

## 1. Why the current AKLP-aliased bundle on TY/Z3 is structurally wrong

`MonadBundle::anderson_lukas_palti_example` is the only bundle attached
to the TY/Z3 ambient `CP³ × CP³` (`AmbientCY3::tian_yau_upstairs`), via
`predict_fermion_masses_with_overrides` (`yukawa_pipeline.rs:547`) and
the Yukawa-production binary
(`bin/p8_3_yukawa_production.rs:398`). Its bidegrees are
`B = O(1,0)³ ⊕ O(0,1)³`, `C = O(1,1)³` — a 2-factor split-symmetric
monad.

Two structural problems:

1. **Wilson Z/3 phase classes are degenerate.** The new fix in
   `assign_sectors_dynamic` (`yukawa_sectors_real.rs`, post P-Wilson-Fix
   from 2026-04-30) projects each B-summand onto the SU(3)-Cartan
   diagonal via the splitting principle. On AKLP's 2-factor bidegrees
   the index→class map gives `up=[2]`, `down=[1,3,4,7]`,
   `lepton=[0,5,6,8]` — a 1:4:4 split, NOT the 3:3:3 partition required
   for a balanced 3-generation Yukawa channel on a Z/3-Wilson breaking
   (`references/p_wilson_fix_diagnostic.md`).

2. **Citation provenance was retracted.** The earlier
   `ty_aglp_2011_standard` builder claimed AGLP-2011 provenance. AGLP-
   2011 (`arXiv:1106.4804` / JHEP 06 (2012) 113) §5.3 explicitly states
   "No phenomenologically viable model were found for h^{1,1}(X) = 2,
   3." TY/Z3 has h^{1,1}(X) = 2, so it lies **outside** the AGLP-2011
   scanned range. The bundle is correct as a canonical SU(3) (or SU(5))
   monad shape but does **not** come from any published TY/Z3 SM
   catalogue row (`hidden_bundle.rs:352-371`).

The bundle is therefore being used as a Yukawa input it was never built
for, on a CY3 the cited paper actively excluded.

## 2. Constraints a real TY/Z3 line-bundle SM must satisfy

A citable TY/Z3 3-generation line-bundle SM bundle V → X (X = TY/Z3)
must satisfy:

* **Ambient & quotient.** Cover X̃ is the Tian-Yau bicubic
  `(3,0)+(0,3)` in `CP³ × CP³`; π₁(X) = Z/3 (single Z/3, not Z/3 × Z/3
  as on Schoen). Hodge structure (h¹¹, h²¹) = (1, 4) downstairs,
  (14, 23) upstairs.
* **Bidegrees are 2-tuples.** `[a, b]` for line bundles `O_X(a, b)` —
  not 3-tuples like Schoen on `CP² × CP² × CP¹`.
* **SU(n) condition.** c₁(V) = 0 ∈ H²(X, ℤ).
* **Generation count.** Index theorem
  `∫_X c₃(V) / |Z/3| = ±3` net generations downstairs (i.e. ±9 net
  upstairs).
* **Wilson Z/3 spans 3 phase classes.** Under the splitting-principle
  projection used by `assign_sectors_dynamic`, the b-line bidegrees
  must hit all three Z/3 classes in a balanced 3:3:3 partition. AKLP's
  `[1,0]³ ⊕ [0,1]³` collapses to 1:4:4 once the Cartan-diagonal
  projection is applied (see §1).
* **Polystability.** V poly-stable w.r.t. some Kähler class
  ω = t₁ J₁ + t₂ J₂ in the (1+1)-dim Kähler cone.
* **Anomaly cancellation.** c₂(V) ≤ c₂(TX) componentwise (or
  pointwise after Whitney decomposition), so a 5-brane class
  W = c₂(TX) − c₂(V) is effective.

A balanced 2-factor candidate that *would* span 3 Z/3 classes under the
`(a − b) mod 3` projection (analogous to the Schoen
`schoen_z3xz3_canonical` 3-factor lift) — for example
`B = O(1,0)² ⊕ O(0,1)² ⊕ O(2,1)²`, `C = O(1,1)² ⊕ O(2,0) ⊕ O(0,2)`
or any of dozens of other algebraically valid choices — exists in the
abstract. **None of them is published.** Selecting one would be a
fabrication.

## 3. Literature candidates that were NOT confirmed in-tree or
   accessible at audit time

The canonical sources to investigate if real TY/Z3 work is later
authorised (per `project_cy3_bundle_provenance_audit.md` memory):

| arXiv ID | Authors | What it claims |
|----------|---------|---------------|
| `0911.1569` | Anderson-Gray-He-Lukas | Positive monad bundles on TY (parent CY3), pre-quotient |
| `hep-th/0501070` | Braun-He-Ovrut-Pantev | Heterotic SM on TY/Z3 via Wilson lines (spectral-cover construction, not line-bundle sums) |
| `1404.2767` | Buchbinder-Constantin-Lukas | Line-bundle moduli space on smooth CY3s (not TY-specific) |
| `0805.1996` | Bouchard-Donagi 2008 | Schoen with Z/2 bundles (not TY) |

**None of these were confirmed in the audit pass to literally tabulate
a 3-generation line-bundle (or monad) Standard-Model bundle on the
Tian-Yau Z/3 *quotient*** with the ingredients §2 lists — many work on
the parent (unquotiented) TY, others use spectral-cover / extension
constructions (not summands of line bundles), and the AGLP scan
(`1106.4804`, `1202.1757`) explicitly excluded the h^{1,1} = 2 TY/Z3.

## 4. Why this is a hard "no", not a "look harder"

A fair reading of the post-2008 heterotic-SM literature is that the
genuine catalogue work happened on:

* the elliptically-fibred 3-folds with h^{1,1} ≥ 4 in the AGLP scans
  (`1106.4804` / `1202.1757`),
* Schoen with Z/3 × Z/3 (Braun-Candelas-Davies `0910.5464`,
  AGLP `1202.1757`),
* spectral-cover constructions on TY-type / Schoen-type quotients
  (BHOP `hep-th/0505041` / DHOR `hep-th/0512149`, both rank-≥2 vector
  bundles, not direct sums of line bundles).

The Tian-Yau Z/3 quotient with its h^{1,1} = 2 Picard rank simply does
not have enough Kähler-class freedom for a generic line-bundle SM scan
to find polystable rank-3 (or rank-4 / rank-5) bundles with c₁ = 0 and
the right c₃, which is the published reason it was excluded from
AGLP-2012 §5.3.

## 5. Disposition

* The TY Yukawa channel is **deferred** until a citable bundle is
  found.
* `MonadBundle::anderson_lukas_palti_example` will continue to be used
  on TY/Z3 by `predict_fermion_masses_with_overrides` and the
  production Yukawa binary, but solely as a **structurally
  inappropriate placeholder**, NOT as a publication input. The
  expected bucket-hits on this bundle (under the post P-Wilson-Fix
  sector assignment) are 1:4:4-skewed and `Y_u/Y_d/Y_e` will not all
  reach 9/9 — this is a feature of the diagnostic, not a regression.
* The 5.43σ Schoen-vs-TY chain-match channel discrimination
  (`p7_11_chain_match`) is geometric and bundle-independent for TY:
  it does not consume the Yukawa pipeline, so it stands.
* The Schoen Z/3 × Z/3 Yukawa channel — which uses
  `MonadBundle::schoen_z3xz3_canonical` (3-factor lift, P8.3-followup-B)
  — is a real constructed line-bundle bundle in this code base and is
  the only Yukawa input that is wired to a properly-balanced
  3-generation bundle. Promoting it to a single-candidate Yukawa
  channel for the Bayes factor is admissible (it does not pretend to
  cite TY/Z3 data); promoting Schoen + TY together as a discriminating
  pair is **not** admissible until §3 is resolved.
* The honest publication line in the journal is: **chain-match
  discriminates at 5.43σ; Yukawa-channel discrimination is gated on
  finding a citable TY/Z3 bundle and is not currently part of the
  combined ln BF.**

## 6. What would re-open this audit

Any of:

1. A peer-reviewed paper post-2008 that tabulates a line-bundle (or
   monad) SM bundle on the Tian-Yau Z/3 quotient (i.e. h^{1,1} = 2,
   π₁ = Z/3) with a 3-generation index, c₁ = 0, polystability,
   and anomaly cancellation. AGLP-2012 §5.3 would have to be
   superseded by a follow-up paper that found a viable model.
2. A first-bundle construction on TY/Z3 by the project authors,
   accompanied by an algebraic-geometry proof of c₁ = 0,
   polystability, and the index-theorem 3-generation count. That is a
   research paper, not an in-tree implementation task.
3. A broader project decision to switch to a different quotient
   (e.g. TY/Z3 × Z3 via a freely-acting Z3 × Z3 if one exists, or one
   of the AGLP-2012 h^{1,1} ≥ 4 manifolds) and rerun the chain-match
   channel against the new candidate pair.

---

**End of audit.** Per `CLAUDE.md` "no backdoors / no fabrication" and
the project's published-bundle integrity policy, the Yukawa channel on
TY/Z3 is intentionally not wired into `bayes_factor_multichannel.rs`
as a third entry, and no fabricated `tian_yau_z3_canonical_v2()`
constructor has been added to `MonadBundle`.
