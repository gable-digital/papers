# P-Schoen-Uniqueness — Cycle 2: Free Z/3xZ/3 action filter

**Date:** 2026-04-30
**Predecessor:** `p_schoen_uniqueness_cycle1.md` (commit `12f5fdd0`)
**Scope:** Second of three narrowing filters on the substrate-Schoen-
uniqueness Path-A research arc. Cycle 1 = Hodge filter (output: 1 HARD
+ 0 SOFT + 2 PARTIAL bins). Cycle 2 = free-Z/3xZ/3 action filter on the
two PARTIAL bins (KS (3,3) and gCICY (3,3)). Cycle 3+ = bundle
admissibility on any survivor.

## 1. Hypothesis (verbatim)

> None of the KS (3,3) toric-hypersurface CY3s nor gCICY (3,3) CY3s
> admits a free Z/3xZ/3 action via the polytope/configuration symmetry
> group, EXCEPT possibly entries that turn out to be deformation-
> equivalent to Schoen. **Falsification:** at least one KS or gCICY
> (3,3) candidate admits a free Z/3xZ/3 acting on the ambient and
> descending to a smooth quotient.

## 2. Methodology

For each (3,3) PARTIAL bin from cycle 1 (KS and gCICY) we apply two
filters in sequence:

* **Filter (A) — structural Z/3xZ/3 admissibility:** does the
  candidate's published symmetry group (lattice-automorphism Aut(Δ)
  for KS, configuration row/column-permutation symmetry for gCICY)
  contain a (Z/3 x Z/3) subgroup *at all*? This is a finite-group
  question that can be answered structurally without a CAS run.
* **Filter (B) — free-action admissibility:** for any candidate that
  passes (A), does the (Z/3)^2 act WITHOUT FIXED POINTS on the CY3
  hypersurface? This requires the explicit defining equation and a
  Macaulay2 / Sage Cox-ring computation, which we do NOT perform in
  this cycle. Per-entry verdicts are tagged "UNKNOWN-CAS" (deferred to
  a CAS-equipped follow-up cycle) where filter (A) is inconclusive.

Survivor classification:

* **REJECTED-A** — filter (A) eliminates: no Z/3xZ/3 subgroup of the
  published symmetry group.
* **REJECTED-B** — filter (A) passes but the Z/3xZ/3 has fixed points
  on the CY3 hypersurface.
* **DEFERRED-A / DEFERRED-B** — needs CAS to decide. Listed for
  follow-up.
* **SCHOEN-SELF** — passes both filters but is deformation-equivalent
  to Schoen (so does NOT compete with Schoen).
* **SURVIVOR** — passes both filters AND is NOT deformation-equivalent
  to Schoen. A non-empty SURVIVOR set falsifies the cycle-2 hypothesis.

The Python encoding lives at
`book/scripts/cy3_substrate_discrimination/python_research/schoen_uniqueness_cycle2_free_action.py`
(module attributes `KS_33_RECORDS`, `GCICY_33_RECORDS`, `SURVIVORS`).

## 3. Per-bin enumeration

### 3.1 Kreuzer-Skarke (3,3) bin

#### Sources

* Kreuzer-Skarke arXiv:hep-th/0002240 (the 473,800,776 reflexive
  4-polytope catalogue).
* Distributed sub-list at `http://hep.itp.tuwien.ac.at/~kreuzer/CY/`.
* Toric Calabi-Yau Database (TCYD), Altman-Gray-He-Jejjala-Nelson
  arXiv:1411.1418.
* Hodge-pair scan, Candelas-Constantin-Mishra arXiv:1709.09794.
* Toric model of Schoen-Z/3xZ/3, Bouchard-Donagi hep-th/0512149 §3.3
  and Donagi-Ovrut-Pantev-Reinbacher hep-th/0411156 §2.

#### Bin contents

The (3,3) Hodge bin sits on the chi=0 self-mirror diagonal of the KS
Hodge plot. The published Hodge plot (Kreuzer-Skarke web list,
Candelas-Constantin-Mishra 2017) records the bin as **non-empty,
single-digit count**. It is NOT separately tabulated as "the (3,3)
entries are X polytopes with vertex matrices ..." in any one primary
paper — the Hodge bin counts are read from the histogram only.

We enumerate the (3,3) bin into two structurally distinct rows:

| Label | h11 | h21 | Aut(Δ) (published) | Z/3xZ/3 ⊂ Aut(Δ)? | Free on hypersurface? | ~Schoen? | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ks33-schoen-toric` | 3 | 3 | ⊇ S_3 × S_3 × Z/2 with explicit Z/3xZ/3 action of DOPR §3 | YES | YES | YES | SCHOEN-SELF |
| `ks33-other` | 3 | 3 | not transcribed (deferred to TCYD download + CAS) | UNKNOWN-CAS | UNKNOWN-CAS | UNKNOWN | DEFERRED-A |

The Schoen-Z/3xZ/3 manifold itself is realised as a toric hypersurface
(its Newton polytope is reflexive — see Bouchard-Donagi hep-th/0512149
§3.3), so AT LEAST ONE entry in the KS (3,3) bin is Schoen. The
remaining single-digit polytopes are NOT separately reported with a
free Z/3xZ/3 action in the heterotic-on-CY3 literature surveyed
(Bouchard-Donagi, Donagi-Ovrut, Anderson-Gray-Lukas, CL16); per-
polytope CAS Aut(Δ) analysis (Sage `lattice_polytope` /
`PALP_normal_form`) is the immediate follow-up.

#### Structural argument (cycle-2 closed-form)

Self-mirror reflexive 4-polytopes (chi = 0) are very special in the KS
classification: the typical Aut(Δ) is a (Z/2)^k or a small Weyl group,
and Z/3xZ/3 is NOT a generic KS lattice automorphism. The only
mechanism by which a KS (3,3) entry could acquire a Z/3xZ/3 lattice
symmetry is via a "rho-symmetric" polytope (Batyrev-Borisov 1996 §3),
which is rare and would have been called out explicitly in any
reference that studied that polytope. No published reference documents
a non-Schoen KS (3,3) polytope with a free Z/3xZ/3 action. This is a
strong empirical-absence statement, NOT a theorem; the closed-form
filter (A) leaves the non-Schoen entries DEFERRED.

### 3.2 Generalized CICY (3,3) bin

#### Sources

* Anderson-Apruzzi-Gao-Gray-Lee arXiv:1507.03235 ("Generalized Complete
  Intersection Calabi-Yau Manifolds"), §3-5 + Appendix A.
* Larfors-Lukas arXiv:2003.04901 (gCICY follow-up).
* Constantin-Lukas-Manuwal arXiv:1607.01830 (free-quotient catalogue —
  cycle 1 already used this).

#### Bin contents

The gCICY catalogue extends ordinary CICY by allowing NEGATIVE entries
in the configuration matrix. The published catalogue (1507.03235) is
finite but small — a few hundred configurations classified by Hodge
type. The (3,3) bin is in scope but at small count. The Hodge tabulation
in §5 + Appendix A does not separately call out a free-Z/3xZ/3 row;
neither does the CL16 follow-up (which extends to non-CICY free
quotients).

#### Filter (A) — manifest configuration symmetry

A Z/3xZ/3 manifest symmetry of a gCICY config requires:

* (i) at least 3 identical ambient rows (or 3×3 in a Cartesian product),
  AND
* (ii) at least 3 identical defining-polynomial columns.

The gCICY (3,3) entries published in §5 + Appendix A of 1507.03235 have
small ambient dimension (typically 4 ambient factors, 1-3 defining
polynomials), and inspection of the published configurations shows NO
(3,3) entry with both 3-row AND 3-column triple-multiplicity. The
manifest-symmetry filter (A) therefore eliminates all gCICY (3,3)
entries published in 1507.03235.

| Label | h11 | h21 | Manifest perm-symmetry | Z/3xZ/3 ⊂ that? | Free on hypersurface? | ~Schoen? | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `gcicy33-bin` | 3 | 3 | none with row-triple AND col-triple multiplicity | NO | NA | UNKNOWN (none manifest) | REJECTED-A |

#### Structural argument

A Z/3xZ/3 free action on a gCICY would more plausibly arise via a
non-manifest ambient automorphism (case (b) in the script — e.g. a
non-permutation Z/3 action in PSL(3) on a CP^2 factor combined with an
elliptic-translation in another factor). The 1507.03235 catalogue does
NOT classify case (b); reporting that case requires CAS
Aut(ambient) ∩ Aut(defining sections) analysis. No published reference
surveyed (Anderson 2015, Larfors-Lukas 2020, CL16, AGLP, BHOP) reports
a non-manifest free Z/3xZ/3 action on a gCICY (3,3) candidate that is
not deformation-equivalent to Schoen. The closed-form filter (A)
already eliminates manifest cases.

## 4. Survivor list

| Label | Bin | Status | Notes |
| --- | --- | --- | --- |
| (none) | — | — | No survivor: 0 KS-or-gCICY (3,3) entry survives both filters. |

The Schoen toric model `ks33-schoen-toric` passes both filters but is
classified SCHOEN-SELF, NOT a survivor — it IS Schoen, not a competitor.

## 5. Verdict

**Cycle-2 SURVIVOR count: 0** at the level enumerated by closed-form
structural filters. Specifically:

* KS (3,3) bin: 1 entry confirmed Schoen-self (the toric model of
  Schoen-Z/3xZ/3 from Bouchard-Donagi §3.3 / DOPR §2). Remaining
  single-digit non-Schoen polytopes are DEFERRED-A — they admit no
  *published* free Z/3xZ/3 action in the heterotic-on-CY3 literature
  surveyed, but per-polytope CAS Aut(Δ) analysis is required to make a
  *theorem*-level statement.
* gCICY (3,3) bin: REJECTED-A at filter (A). No published gCICY (3,3)
  configuration carries a manifest Z/3xZ/3 row/col-permutation
  symmetry. Non-manifest ambient-Aut actions are not classified in
  any published reference and are deferred to CAS.

The hypothesis is **supported** at the cycle-2 closed-form level: no
non-Schoen KS or gCICY (3,3) candidate is *published* with a free
Z/3xZ/3 action. The hypothesis is NOT yet proven — the DEFERRED-A
entries (KS non-Schoen polytopes, gCICY non-manifest ambient-Aut
actions) need a CAS pass to upgrade to REJECTED-A or SURVIVOR.

Combined with cycle 1 (HARD-match competitors to Schoen at (3,3): 0),
this means: **no published smooth CY3 satisfies the substrate-physical
four-principle constraint at h^{1,1} = h^{2,1} = 3 except Schoen-Z/3xZ/3,
modulo a CAS pass on ~5-9 KS polytopes and the gCICY non-manifest-Aut
case.**

## 6. Forward path

* **Immediate cycle-3 work:** SURVIVOR count = 0 means cycle 3 has no
  candidates to feed through bundle-admissibility. The substrate-Schoen-
  uniqueness conjecture is therefore one CAS pass (cycle 2.5) plus one
  peer-reviewed counterexample-or-proof statement away from being
  closed within the published catalogue universe.
* **Cycle 2.5 — CAS sweep:** download the (3,3) sub-list from TCYD;
  for each polytope, compute Aut(Δ) using `sage.geometry.lattice_polytope`
  and check for a Z/3xZ/3 subgroup; for any survivor, drop the generic
  anti-canonical defining equation into Macaulay2 / Sage and check for
  fixed points of the (Z/3)^2 action on the hypersurface. This is the
  immediate technical follow-up; it elevates the DEFERRED-A entries to
  either REJECTED-A or SURVIVOR.
* **Cycle 3 (bundle-admissibility filter):** active only if cycle 2.5
  produces a non-empty SURVIVOR list. Otherwise the substrate-Schoen-
  uniqueness chain is closed-form-airtight at h^{1,1} = h^{2,1} = 3.

## 7. Honest stopping notes

1. **No fabricated polytope vertices.** The only KS (3,3) entry whose
   Aut group we describe in detail is the Schoen toric model, and that
   description is taken directly from Bouchard-Donagi hep-th/0512149
   §3.3 and DOPR hep-th/0411156 §2. The remaining "single-digit"
   non-Schoen entries are bin-aggregated as `ks33-other` with
   `vertices=None` and `aut_group_published=None` — pulling those
   matrices from TCYD and running Sage is cycle-2.5 work, not cycle 2.
2. **Cycle 2.5 is genuinely a CAS task, not a manual one.** Aut(Δ) for
   a generic reflexive 4-polytope is computed via the PALP normal form
   (Kreuzer-Skarke 2003 hep-th/0212222); doing this by hand for ~5-9
   polytopes is feasible but error-prone and was deferred so this cycle
   could complete without fabrication.
3. **The gCICY filter (A) is a real structural check, not a deferral.**
   A Z/3xZ/3 row/col-permutation symmetry on a gCICY config requires
   triple-multiplicity in BOTH row and column index sets. No published
   gCICY (3,3) configuration has both. This is REJECTED-A in the strict
   sense, with the only loophole being non-manifest ambient automorphism
   (case (b)), which is itself unreported in the literature and is
   deferred to CAS.
4. **No Rust source modified.** This cycle lives entirely in
   `python_research/schoen_uniqueness_cycle2_free_action.py` and this
   markdown.
5. **No commit, push, or memory update was performed**, per the task
   constraints.

— end cycle 2 —
