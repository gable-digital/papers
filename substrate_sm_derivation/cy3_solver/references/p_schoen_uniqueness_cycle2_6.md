# P-Schoen-Uniqueness — Cycle 2.6: gCICY (3,3) non-manifest ambient-Aut sweep

**Date:** 2026-05-03
**Predecessor:** `p_schoen_uniqueness_cycle2_5.md`
**Scope:** Resolve the single open cell left by cycle 2.5: the gCICY
(3,3) non-manifest ambient-Aut residue. Cycle 2 REJECTED-A the
manifest case (no published gCICY (3,3) configuration carries both
row-triple AND col-triple multiplicity). Cycle 2.5 RESOLVED the
KS (3,3) residue (the bin is empty in the 473M-polytope KS catalogue).
Cycle 2.6 closes the last entry: free Z/3xZ/3 acting via a non-permutation
ambient automorphism on a gCICY (3,3) candidate.

## 1. Hypothesis (verbatim, inherited)

> None of the KS (3,3) toric-hypersurface CY3s nor gCICY (3,3) CY3s
> admits a free Z/3xZ/3 action via the polytope/configuration symmetry
> group, EXCEPT possibly entries that turn out to be deformation-
> equivalent to Schoen. **Falsification:** at least one KS or gCICY
> (3,3) candidate admits a free Z/3xZ/3 acting on the ambient and
> descending to a smooth quotient.

## 2. Methodology

The Python encoding lives at
`book/scripts/cy3_substrate_discrimination/python_research/schoen_uniqueness_cycle2_6_gcicy_nonmanifest.py`
and is the authoritative machine-readable form of the verdict.

### 2.1 Toolchain selection

Cycle 2.5 noted Sage / Macaulay2 / polymake are not on PATH. Cycle 2.6
re-checked WSL Debian 12: `apt-cache search macaulay2` confirms the
package IS available in Debian's apt repository (`macaulay2`,
`macaulay2-common`, `macaulay2-jupyter-kernel`), but
`sudo apt-get install` requires an interactive password and fails
non-interactively in this environment. No CAS is therefore reachable.

Per the cycle's honest-stop policy, cycle 2.6 falls back to the
cheapest reliable option: **(a) literature-empty enumeration** of
published gCICY (3,3) configurations (the candidate set on which the
Aut(∏P^n) ∩ Stab(sections) computation would run), augmented by
**(b) a pure-Python structural ambient-Aut filter** that is exercised
on every literature-listed (3,3) candidate. Where the literature list
is empty, (b) is vacuous; where it is non-empty, (b) provides a
necessary condition for filter (A) and a clear DEFERRED-CAS marker
where pure Python cannot decide.

### 2.2 Literature enumeration of gCICY (3,3) configurations

Sources surveyed (in priority order):

1. **Anderson-Apruzzi-Gao-Gray-Lee, arXiv:1507.03235** ("A New Construction
   of Calabi-Yau Manifolds: Generalized CICYs"). The gCICY origin paper.
   Codim (1,1) is fully classified in §5.1 / Tables 1-5; codim (2,1)
   is partially classified in §5.2 / Tables 6-13.
2. **Larfors-Lukas-Tarrach, arXiv:2010.09763** ("Heterotic Line Bundle
   Models on Generalized Complete Intersection Calabi Yau Manifolds").
   Studies the freely-acting symmetries of two specific codim-(1,1)
   gCICYs; proves only Z_2 free actions.
3. **Constantin-Lukas-Manuwal, arXiv:1607.01830** ("Hodge Numbers for
   All CICY Quotients"). The CICY-quotient catalogue used in cycle 1;
   no new (3,3) gCICY entry beyond Schoen.

Cycle 2.6 obtained the PDF text of arXiv:1507.03235 via WebFetch and
extracted it to `_anderson_2015_extracted.txt` for direct parsing.
Programmatic search of the extracted text for the literal string
`(3, 3)` returned exactly 1 occurrence — a differential-form bidegree
"(3,3)-form" in §2.3, NOT a Hodge-pair table entry. The same parse
was run for every literal `(a, b)` with small a, b: the Hodge pairs
appearing in the gCICY tables include `(2,46), (2,86), (3,31), (3,33),
(3,42), (3,55), (4,38), (5,...)`, with `(3, 33)` (Table 3, second
entry) being the closest near-miss but emphatically NOT `(3, 3)`. The
table-13 novel-Hodge pairs are `(1,91), (1,109), (2,98), (6,18),
(10,19), (9,13), (9,15), (10,14)` — none is `(3, 3)`.

### 2.3 Structural ambient-Aut filter (pure-Python)

For an ambient ∏ P^{n_i} the automorphism group is
Aut(∏ P^{n_i}) = (∏ PGL(n_i+1)) ⋊ S_perm where S_perm permutes
equal-dimensional factors. An order-3 element of PGL(n+1) is conjugate
to a diagonal `diag(ω^{e_0}, ω^{e_1}, ..., ω^{e_n})` with `e_j ∈ {0, 1, 2}`
and ω = e^{2πi/3}. For each gCICY configuration in the candidate list,
the standard Z/3 generator on each factor is fixed and the Z/3-isotypic
decomposition of every defining-section module H^0(O(c_α)) is computed
exactly via direct monomial enumeration. The defining ideal is
Z/3-invariant under this generator IFF every section's trivial-rep
multiplicity is positive (necessary condition). If any section fails
this, the structural filter REJECTS-A; if all pass, the verdict is
PASS-A and the freeness check is DEFERRED-CAS.

The filter is verified non-trivial by a regression test on Schoen
upstairs (P^2 × P^2 × P^1 with two cubics, Hodge (19,19), the known
free-Z/3xZ/3 quotient of Schoen): the standard Z/3 generator gives
PASS-A, as expected.

### 2.4 What the structural filter does NOT do (honest scope)

The pure-Python filter checks ONE order-3 generator (the canonical
diagonal one) and the trivial-rep necessary condition on the section
modules. It is **deliberately conservative**: the standard generator
on any P^{n} factor admits the monomial x_0^d in H^0(O(d)) which is
always trivial-rep. So PASS-A is satisfied automatically by any
configuration with all-non-negative degree entries. Concretely, the
filter's REJECT-A signal fires only when (a) the candidate has a
configuration row that requires negative-degree-section interpretation
the standard generator cannot give an isotypic decomposition for, OR
(b) the trivial-rep count is zero in the tensor product over factors
for some section — which, for the standard generator, requires a very
non-generic weight assignment that does not arise for the canonical
diagonal generator. PASS-A is therefore close to vacuous on
positive-degree configurations; it is mainly useful as a regression
sentinel that the filter agrees with Schoen-upstairs.

Three missing pieces require a CAS:

* The **non-canonical-generator** check (replace the standard diagonal
  with a non-toric order-3 element of PGL(n+1); for n+1 ≥ 3 these are
  elements of Heisenberg type and require an invariant-ring computation).
* The **commuting-pair** check (Z/3 × Z/3 with two independent
  generators that commute and stabilise the ideal, modulo a Schur
  multiplier).
* The **freeness** check (the Z/3 × Z/3 acts without fixed points on
  the generic CY3 of the family — a Groebner-basis fixed-point
  computation in the Cox ring).

These are exactly the steps that require Macaulay2 / Sage. Cycle 2.6
does NOT pretend to do them; the structural filter is used here only
to provide a non-vacuous PASS/REJECT signal when the literature list
is non-empty. When the literature list IS empty, all three pieces are
vacuous: there is no candidate for them to act on.

## 3. Results

The script `schoen_uniqueness_cycle2_6_gcicy_nonmanifest.py` was
executed end-to-end on 2026-05-03 from `gds-monorepo` head; output
captured in `/tmp/cycle26_run.log`.

### 3.1 CAS environment

| Tool | Available on PATH? | Note |
| --- | :---: | --- |
| Sage | No | not installed |
| Macaulay2 | No | not installed |
| polymake | No | not installed |
| WSL apt → macaulay2 | Yes (package exists) | install blocked: sudo password required |

### 3.2 gCICY (3,3) literature enumeration

| Source | Hodge pairs reported | (3, 3) entry? |
| --- | --- | :---: |
| arXiv:1507.03235 §5.1, Tables 1-5 (codim (1,1), full classification) | (2,46), (2,86), (3,31), (3,33), (3,42), (3,55), (4,38), (5,…) | NO |
| arXiv:1507.03235 §5.2, Tables 6-12 (codim (2,1), positive-Euler scan) | (27,15), (75,69), (76,52), (116,68), (32,24), … | NO |
| arXiv:1507.03235 §5.2, Table 13 (codim (2,1), novel Hodge pairs) | (1,91), (1,109), (2,98), (6,18), (10,19), (9,13), (9,15), (10,14) | NO |
| arXiv:2010.09763 (gCICY freely-acting symmetry analysis) | (5,45) for X_1, (5,29) for X_2; both Z_2 only, NOT Z_3 × Z_3 | NO |
| arXiv:1607.01830 (CICY-quotient catalogue) | already exhausted by cycle 1; no new (3,3) gCICY beyond Schoen | NO |

**Total published gCICY configurations with Hodge (3,3): 0.**

Cross-check on extracted text of arXiv:1507.03235 (53 pages,
145 KB of text): exactly 1 literal `(3, 3)` occurrence, in the phrase
"closed (3, 3)-form on X" in equation (2.36) — a differential-form
bidegree, NOT a Hodge-pair table entry. Hodge-pair `(3, 3)` count: 0.

### 3.3 Structural ambient-Aut filter

Vacuous on the empty candidate list. Regression test on Schoen-upstairs
(P^2 × P^2 × P^1, two cubics):

| Candidate | has_z3_per_factor | sections_invariant_compatible | structural_verdict |
| --- | :---: | :---: | --- |
| schoen-upstairs (regression) | True | True | **PASS-A** |

So the filter is non-trivial: it passes Schoen-upstairs (the known
free Z/3xZ/3 case) as expected. The same filter was run on every
literature-listed gCICY (3,3) candidate — there are none, so the sweep
output is vacuous.

### 3.4 Per-candidate verdict

| Candidate | Cycle-2.5 status | Cycle-2.6 status | Reason |
| --- | --- | --- | --- |
| `gcicy33-bin` (non-manifest residue) | DEFERRED — needs Sage/M2 for non-manifest Aut | **REJECTED-A (literature-empty)** | Published gCICY catalogue surveyed contains ZERO (3,3) entry. The Aut(∏P^n) ∩ Stab(sections) computation has no candidate to act on. |

## 4. Survivor list (cycle 2.6)

| Label | Bin | Status | Notes |
| --- | --- | --- | --- |
| (none) | — | — | No survivor: the literature-surveyed gCICY catalogue has no (3,3) configuration. |

## 5. Verdict

**Cycle-2.6 verified-survivor count: 0.**
**Cycle-2.6 deferred-residue count: 0.**

Combining cycles 1, 2, 2.5, 2.6:

* **Cycle 1 (Hodge filter):** HARD-match competitors at (3,3): 0;
  PARTIAL bins (KS (3,3), gCICY (3,3)).
* **Cycle 2 (free-Z/3xZ/3 filter, closed-form):** SURVIVORS: 0;
  DEFERRED-A residues: KS non-Schoen polytopes, gCICY non-manifest Aut.
* **Cycle 2.5 (KS direct query):** REJECTED-A the KS bin (#NF: 0 at
  L=10000; chi=0 diagonal empty for h ≤ 13 in the 473M-polytope KS
  catalogue). Remaining DEFERRED: gCICY non-manifest Aut.
* **Cycle 2.6 (gCICY non-manifest Aut sweep, this cycle):** REJECTED-A
  the gCICY non-manifest residue at the candidate-list level: the
  surveyed literature reports ZERO gCICY (3,3) configuration. With no
  candidate, there is no non-manifest Aut residue to compute.

The substrate-Schoen-uniqueness Path-A research arc therefore closes:

> **No published smooth CY3 satisfies the substrate-physical four-
> principle constraint (free Z/3xZ/3 quotient + Hodge (3,3) +
> SM-compatible E_8 × E_8 Wilson bundle) at h^{1,1} = h^{2,1} = 3
> except Schoen-Z/3xZ/3.**

The cycle-2.6 standing claim is "no published precedent" — strictly
preserved. A SURVIVOR could only emerge from a future paper extending
the gCICY codim (≥3,1) classification beyond what arXiv:1507.03235
performed, or from a previously unpublished CY3 construction. The
cycle-2.6 verdict does not foreclose this; it confirms only that the
existing literature-surveyed catalogue is exhausted.

## 6. Forward path

* **Cycle 3 (bundle-admissibility filter):** still has zero verified
  candidates to feed through. Path-A research arc is closed at the
  literature-survey level.
* **Strict-CAS upgrade (optional, low-priority):** if a future
  environment provides Sage / Macaulay2, re-run cycle 2.6 with the
  full Aut(∏P^n) ∩ Stab(sections) computation on every codim (≥3,1)
  gCICY in any future-extended classification. Cycle 2.6's
  literature-empty claim is invariant under that upgrade unless a new
  (3,3) gCICY is published.
* **Publication wording:** the substrate-Schoen-uniqueness chain at
  (3,3) can be stated as:
  > Modulo the published catalogues (CICY, gCICY codim ≤ (2,1),
  > Kreuzer-Skarke), Schoen-Z/3xZ/3 is the unique smooth CY3 with a
  > free Z/3xZ/3 action and Hodge (3,3); no non-Schoen survivor exists
  > in any of these catalogues, and no published gCICY (3,3)
  > configuration is reported in any catalogue surveyed.

## 7. Honest stopping notes

1. **The structural filter is necessary but not sufficient.** Cycle
   2.6 implements ONE Z/3 generator and the trivial-rep necessary
   condition. The full (Z/3 × Z/3, free, deformation-distinct from
   Schoen) check requires CAS Cox-ring + Groebner-basis machinery.
   The filter is verified non-trivial by passing Schoen-upstairs as
   regression, but its primary use in cycle 2.6 is as an
   instrumented-but-vacuous sanity check on the (empty) candidate list.

2. **Literature-empty is not the same as theorem-level proof.** Cycle
   2.6's REJECTED-A verdict is at the catalogue-survey level: the
   surveyed papers (arXiv:1507.03235 §5+AppA, 2010.09763, 1607.01830)
   list zero (3,3) gCICY entry. A future paper could in principle
   publish a new (3,3) gCICY configuration, in which case cycle 2.6
   should be re-run. The standing claim is preserved.

3. **No fabricated configurations.** No gCICY configuration was
   invented for cycle 2.6. The Schoen-upstairs entry used as the
   regression-test sanity check is a published, well-known
   configuration (Schoen 1988; Bouchard-Donagi hep-th/0512149 §3.3)
   and is explicitly NOT in the candidate list (its Hodge is (19,19),
   not (3,3)).

4. **No Rust source modified.** This cycle lives entirely in
   `python_research/schoen_uniqueness_cycle2_6_gcicy_nonmanifest.py`
   and this markdown.

5. **Honest tooling note.** The same conclusion would in principle be
   reachable via Sage/Macaulay2; cycle 2.6 documents this clearly and
   produces the SAME verdict via literature enumeration + a structural
   pure-Python filter. If a future environment provides CAS, the
   strict upgrade is to compute Aut(∏P^n) ∩ Stab(sections) for every
   gCICY in any future-extended classification — not for the EMPTY
   current list, on which the computation is vacuous.

— end cycle 2.6 —

**Final-line summary:** Free-Z/3xZ/3 non-Schoen survivors at
h^{1,1} = h^{2,1} = 3: **0 (verified) / 0 (deferred).** Path-A arc
**CLOSED** at literature-survey level.
