# P-Schoen-Uniqueness — Cycle 2.5: CAS sweep (KS direct-query resolution)

**Date:** 2026-05-03
**Predecessor:** `p_schoen_uniqueness_cycle2.md`
**Scope:** Resolve the two DEFERRED-A residues left by cycle 2:
(i) the "single-digit" non-Schoen Kreuzer-Skarke (3,3) polytopes, and
(ii) the gCICY non-manifest ambient-Aut residue at (3,3). Cycle 2.5
upgrades these from DEFERRED to either REJECTED or SURVIVOR by
attempting a CAS / direct-database resolution.

## 1. Hypothesis (verbatim, inherited from cycle 2)

> None of the KS (3,3) toric-hypersurface CY3s nor gCICY (3,3) CY3s
> admits a free Z/3xZ/3 action via the polytope/configuration symmetry
> group, EXCEPT possibly entries that turn out to be deformation-
> equivalent to Schoen. **Falsification:** at least one KS or gCICY
> (3,3) candidate admits a free Z/3xZ/3 acting on the ambient and
> descending to a smooth quotient.

## 2. Methodology

The Python encoding lives at
`book/scripts/cy3_substrate_discrimination/python_research/schoen_uniqueness_cycle2_5_cas_sweep.py`
and is the authoritative machine-readable form of the verdict table.

### 2.1 KS residue (i) — direct CGI query

The Kreuzer-Skarke project distributes a CGI front-end at
`http://quark.itp.tuwien.ac.at/cgi-bin/cy/cydata.cgi` that runs PALP's
`class.x` against the full 473,800,776-polytope catalogue. The query
parameters are `h11`, `h12`, and `L` (output limit). Three response
patterns:

* `#NF: 0   done` — the bin is genuinely empty in KS.
* `Exceeded limit of L` — the bin has > L entries; only the first L
  were returned.
* one or more polytope normal forms with `H:h11,h12 [chi]` headers and
  a `dim x n_vertices` integer matrix of M-lattice vertices.

The script issues queries with retry-on-disconnect (the CGI is
intermittently unreachable; we backoff 5 s and retry up to 8 times)
and parses the response into a structured `KSResponse` object.

### 2.2 KS residue (i) cross-check — chi=0 diagonal sweep

To rule out a parser bug or a one-off CGI hiccup at the (3,3) bin, the
script also sweeps the chi=0 self-mirror diagonal h11=h21 from h=1
through h=14 with `L=2`. Three regimes are expected if the bin-empty
result is real:

* h=1..k: all `#NF: 0` (the chi=0 diagonal is genuinely empty at
  small h);
* h=k+1..: populated, returning either polytopes or `Exceeded limit of L`.

This is a sanity check, NOT a separate hypothesis test.

### 2.3 gCICY residue (ii) — Sage / Macaulay2 fallback

Cycle 2's filter (A) already REJECTED the *manifest-symmetry* case for
all published gCICY (3,3) configurations (no entry has both row-triple
AND column-triple multiplicity required for a Z/3xZ/3 row/col-permutation
symmetry). The remaining residue is the *non-manifest* case: a Z/3xZ/3
acting through an ambient automorphism that does not coincide with a
config row/col permutation. Resolving this requires:

1. Computing the full ambient automorphism group Aut(prod P^{n_i}) and
   its intersection with the stabiliser of the configuration's defining
   polynomial sections, AND
2. Checking which subgroups of Z/3xZ/3 type act freely on a generic CY3
   in the family.

Both steps need Sage's `multivariate_polynomials.PolynomialRing` /
Macaulay2's `RationalMaps` machinery. The script therefore detects
whether `sage`, `M2`/`macaulay2`, or `polymake` are reachable on PATH;
if none are, residue (ii) is DEFERRED honestly.

## 3. Results

The script `schoen_uniqueness_cycle2_5_cas_sweep.py` was executed end-
to-end on 2026-05-03 from `gds-monorepo` head. Output captured in
`/tmp/cycle25_run.log`. Summary:

### 3.1 CAS environment

| Tool | Available? |
| --- | --- |
| Sage | No |
| Macaulay2 | No |
| polymake | No |

(The KS CGI does NOT require any of these — it runs PALP server-side.)

### 3.2 KS (3,3) direct query

```
Search command:  class.x -di x -He EH3:3MVNFL10000
Result:          #NF: 0
                   done (0s)
```

The Kreuzer-Skarke catalogue contains **0** reflexive 4-polytopes whose
generic anti-canonical CY3 hypersurface has Hodge numbers (3, 3). This
result was cross-verified at `L=100`, `L=1000`, and `L=10000` — all
return `#NF: 0`.

### 3.3 chi=0 self-mirror diagonal sweep

| h11=h21 | bin empty? | exceeded L=2? | note |
| ---: | :---: | :---: | --- |
| 1 | YES | — | #NF: 0 |
| 2 | YES | — | #NF: 0 |
| 3 | YES | — | #NF: 0 |
| 4 | YES | — | #NF: 0 |
| 5 | YES | — | #NF: 0 |
| 6 | YES | — | #NF: 0 |
| 7 | YES | — | #NF: 0 |
| 8 | YES | — | #NF: 0 |
| 9 | YES | — | #NF: 0 |
| 10 | YES | — | #NF: 0 |
| 11 | YES | — | #NF: 0 |
| 12 | YES | — | #NF: 0 |
| 13 | YES | — | #NF: 0 |
| 14 | NO | YES (L=2 hit) | bin populated |

The chi=0 diagonal is *empty* in the Kreuzer-Skarke catalogue for
h ≤ 13 and *populated* from h=14 onward. The (3,3) bin-emptiness is a
genuine low-h phenomenon, not a parser artifact or CGI failure.

### 3.4 gCICY (3,3) — non-manifest residue

Sage / Macaulay2 are not installed in the current environment. The non-
manifest Aut residue therefore remains DEFERRED in the strict cycle-2.5
sense. The literature surveyed (Anderson-Apruzzi-Gao-Gray-Lee 2015,
Larfors-Lukas 2020, Constantin-Lukas-Manuwal 2016, AGLP 2011, BHOP
2005) reports no non-manifest free Z/3xZ/3 action on a gCICY (3,3)
candidate that is not deformation-equivalent to Schoen.

## 4. Per-candidate verdict table

| Candidate | Cycle-2 status | Cycle-2.5 status | Reason |
| --- | --- | --- | --- |
| `ks33-schoen-toric` | SCHOEN-SELF | SCHOEN-SELF (unchanged; not a competitor) | Schoen itself, the target |
| `ks33-other` | DEFERRED-A | **REJECTED-A (KS bin empty)** | KS CGI `#NF: 0` for (3,3) at L=10000; chi=0 diagonal empty for h ≤ 13 |
| `gcicy33-bin` (manifest) | REJECTED-A | REJECTED-A (unchanged) | No (3,3) gCICY config has both row-triple AND column-triple multiplicity (cycle 2) |
| `gcicy33-bin` (non-manifest) | DEFERRED | DEFERRED (CAS unavailable) | Requires Sage / Macaulay2; not installed |

### 4.1 Important nuance on `ks33-schoen-toric`

The cycle-2 row `ks33-schoen-toric` was annotated as the "toric model
of Schoen-Z/3xZ/3" with citations to Bouchard-Donagi hep-th/0512149 §3.3
and DOPR hep-th/0411156 §2. The cycle-2.5 KS direct query result that
the (3,3) bin is empty implies that **the Schoen-Z/3xZ/3 quotient is
NOT itself a generic anti-canonical CY3 hypersurface in any 4-d
reflexive polytope** — it is a free Z/3xZ/3 quotient of the (19,19)
upstairs fiber product, and the quotient does not realise as a single
toric hypersurface. The Bouchard-Donagi / DOPR construction realises
the *upstairs* X̃ (at (19,19)) as a CICY in CP² × CP² × CP¹ with two
defining cubic equations, and the *downstairs* X = X̃ / Γ at (3,3) is
defined as a quotient, not as a toric hypersurface.

This sharpens cycle-2's bookkeeping: the `ks33-schoen-toric` row was
strictly speaking mislabelled — there is no KS (3,3) toric hypersurface
representing Schoen, only a CICY upstairs whose free quotient lands at
(3,3). It does not change the verdict (Schoen-self is still Schoen-
self; it was never a competitor); it merely clarifies that the entire
KS (3,3) row is a non-issue and there are no candidates whatsoever to
push through filter (A) or filter (B).

## 5. Survivor list (cycle-2.5)

| Label | Bin | Status | Notes |
| --- | --- | --- | --- |
| (none) | — | — | No survivor in any of (KS-direct (3,3), gCICY-manifest (3,3)). |

The remaining DEFERRED entry (gCICY non-manifest Aut at (3,3)) is **not**
a survivor — it is a residue that requires a CAS-equipped follow-up
cycle to either REJECT or upgrade to SURVIVOR. It does NOT yet
falsify the hypothesis.

## 6. Verdict

**Cycle-2.5 verified-survivor count: 0.**
**Cycle-2.5 deferred-residue count: 1** (gCICY non-manifest Aut at (3,3)).

Specifically:

* **KS (3,3) bin: REJECTED-A by direct database query.** The Kreuzer-
  Skarke 473M-polytope catalogue contains zero (3,3) entries. There
  are no candidates to feed through filter (B) and no candidates to
  pass to cycle 3.
* **gCICY (3,3), manifest case: REJECTED-A** (cycle 2 result, unchanged).
* **gCICY (3,3), non-manifest case: DEFERRED** to a Sage/Macaulay2-
  equipped follow-up cycle.

Combined with cycle 1 (HARD-match competitors at (3,3): 0) and cycle 2
(structural-filter survivors at (3,3): 0), the conclusion is:

> **No published smooth CY3 satisfies the substrate-physical four-
> principle constraint at h^{1,1} = h^{2,1} = 3 except Schoen-Z/3xZ/3,
> modulo a Sage/Macaulay2 pass on the gCICY non-manifest-Aut case at
> (3,3) which the literature surveyed reports no candidate for.**

The substrate-Schoen-uniqueness hypothesis is therefore **supported**
at the cycle-2.5 level (0 verified survivors, 1 strictly-bounded
deferred residue with no published precedent).

## 7. Forward path

* **Cycle 3 (bundle-admissibility filter):** still has zero verified
  candidates to feed through. The substrate-Schoen-uniqueness chain at
  (3,3) is closed-form-airtight modulo the deferred gCICY non-manifest-
  Aut residue.
* **Cycle 2.6 (gCICY non-manifest Aut sweep):** install Sage or
  Macaulay2 in a CAS-equipped environment and run an ambient-Aut
  computation on each (3,3) gCICY configuration in
  arXiv:1507.03235 §5 + Appendix A. Expected outcome (per the
  empirical-absence reading of the literature): every entry REJECTED,
  upgrading the cycle-2.5 deferred count from 1 to 0. This is the
  cleanest finishing move for the Path-A research arc.

## 8. Honest stopping notes

1. **No fabricated polytope vertices were added.** The script issues
   queries to a public CGI and parses the response. The (3,3) result
   is not derived from any private dataset or local computation.
2. **The KS CGI's `#NF: 0` response is unambiguous.** The same query
   returns 1 polytope at the (1,101) quintic bin and many polytopes at
   (15,15), confirming the parser handles populated bins correctly.
   The (3,3) emptiness is therefore a real property of the KS catalogue,
   not a query-syntax error.
3. **Sage / Macaulay2 / polymake are not installed.** The script
   detects this and emits a `DEFERRED — needs Sage/M2` verdict for the
   gCICY non-manifest residue rather than pretending to compute it.
4. **The cycle-2 row `ks33-schoen-toric` was mislabelled.** Cycle 2
   listed Schoen as a (3,3) "toric hypersurface" (citing Bouchard-Donagi
   §3.3); cycle-2.5's direct KS query shows the (3,3) bin is empty in
   KS, so Schoen at (3,3) is the *quotient* of the upstairs (19,19)
   CICY, not a toric hypersurface itself. The verdict is unchanged
   (Schoen is not a competitor to itself); the row is now annotated.
5. **No Rust source modified.** This cycle lives entirely in
   `python_research/schoen_uniqueness_cycle2_5_cas_sweep.py` and this
   markdown.

— end cycle 2.5 —

**Final-line summary:** Free-Z/3xZ/3 non-Schoen survivors at
h^{1,1}=h^{2,1}=3: **0 (verified) / 1 (deferred — gCICY non-manifest
Aut, needs Sage/M2)**.
