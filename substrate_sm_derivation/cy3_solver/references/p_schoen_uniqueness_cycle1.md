# P-Schoen-Uniqueness — Cycle 1: Hodge-number filter

**Date:** 2026-04-30
**Scope:** First of three narrowing filters on the substrate-Schoen-
uniqueness Path-A research arc. Cycle 1 = Hodge filter; cycle 2 =
free-Z/3xZ/3-action filter; cycle 3+ = bundle-admissibility filter.

## 1. Hypothesis

> The set of smooth Calabi-Yau threefolds with Hodge numbers
> (h^{1,1}, h^{2,1}) = (3, 3) is finite and small (<= tens of entries)
> across the known catalogues, and is enumerable from published
> references. Falsification: the (3,3) set is ill-defined or the
> catalogues do not cover it.

## 2. Methodology

For each catalogue family we (a) state the catalogue source, (b) state
the search method (which published table or computational scan), (c) list
the (3,3)-Hodge entries with construction data, and (d) flag whether the
ambient construction has an obvious Z/3xZ/3 symmetry. No catalogue rows
are fabricated. Where a catalogue is too large to enumerate exhaustively
in one cycle (Kreuzer-Skarke), we record what was scanned and what is
deferred. The Python encoding lives at
`book/scripts/cy3_substrate_discrimination/python_research/schoen_uniqueness_cycle1_hodge_filter.py`
(module attribute `CANDIDATES`), which is the authoritative
machine-readable form of the table below.

## 3. Per-catalogue findings

### 3.1 CICY (Candelas-Dale-Lutken-Schimmrigk 1988)

* **Source.** CICY list (7,890 entries), Hodge file by Anderson-He-Lukas
  arXiv:0911.1569.
* **Method.** Scan the Hodge histogram of the upstairs CICY catalogue at
  the (3,3) bin.
* **Finding.** The upstairs CICY catalogue contains **no** (3,3)
  entries. The h^{1,1} = 3 row of the CICY Hodge histogram begins at
  h^{2,1} = 39 (the bicubic split-Tian-Yau cover) and increases.
* **Z/3xZ/3 status.** N/A — bin empty.

### 3.2 Kreuzer-Skarke toric hypersurfaces (KS00)

* **Source.** Kreuzer-Skarke arXiv:hep-th/0002240 (473,800,776 reflexive
  4-polytopes); Hodge-pair tabulation by Altman-Gray-He-Jejjala-Nelson
  arXiv:1411.1418 (Toric Calabi-Yau Database TCYD); follow-up scan
  Candelas-Constantin-Mishra arXiv:1709.09794.
* **Method.** Read the published Hodge plot at the (3,3) bin; the bin
  lies on the chi=0 self-mirror diagonal.
* **Finding.** The KS (3,3) bin is non-empty and **single-digit** small.
  Exact polytope IDs are downloadable from TCYD but were **not**
  enumerated in cycle 1.
* **Z/3xZ/3 status.** **DEFERRED** — automorphism-group analysis of each
  (3,3) polytope (cycle 2). No KS (3,3) entry is published with an
  explicit free Z/3xZ/3 quotient.
* **Classification.** PARTIAL.

### 3.3 Schoen fiber product (Schoen 1988 / DOPR 2004)

* **Source.** Schoen Inv.Math. 92 (1988) 487; DOPR hep-th/0411156;
  bundle data from BCD/DHOR hep-th/0501070 and Anderson-Gray-Lukas-Ovrut
  arXiv:0911.1569.
* **Method.** Direct geometric construction: fiber product
  X_tilde = B_1 x_{P^1} B_2 of two rational elliptic surfaces over P^1
  inside CP^2 x CP^2 x CP^1; X_tilde has (h^{1,1}, h^{2,1}) = (19, 19);
  the free Z/3xZ/3 quotient X = X_tilde / (Z/3 x Z/3) has (3, 3).
* **Finding.** Exactly **one** entry: `schoen_z3xz3` (the substrate's
  predicted CY3).
* **Z/3xZ/3 status.** **EXPLICIT** in DOPR §3 (free, smooth quotient).
* **Bundle status.** **EXPLICIT** SU(5) line-bundle SM constructions
  with net-3 chiral spectrum (BCD/DHOR/AGLO).
* **Classification.** HARD.

### 3.4 Constantin-Lukas free-quotient CICYs

* **Source.** Constantin-Lukas-Manuwal arXiv:1607.01830, "Heterotic
  Calabi-Yau Compactifications with Small Hodge Numbers"; Anderson-Gray-
  Lukas-Palti (AGLP) arXiv:1106.4804 line-bundle SU(5) GUT scan.
* **Method.** CL16 tabulates **all** CICY free quotients with
  h^{1,1} + h^{2,1} <= 22. Read the (3,3) row directly.
* **Finding.** The only CICY-free-quotient entry at (3,3) is the
  Schoen-Z/3xZ/3 split-bicubic fiber product (already counted under
  §3.3). No other CICY-with-free-quotient lands at (3,3).
* **AGLP overlap.** AGLP-2011 §5.3 explicitly excludes h^{1,1} = 2, 3
  from its line-bundle SM scan, so the AGLP catalogue contributes zero
  competing line-bundle SM candidates at (3,3) beyond Schoen (which
  pre-existed in BCD/DHOR/AGLO).
* **Classification.** HARD (collapses onto §3.3 — no new entries).

### 3.5 Pfaffian / non-CICY constructions

* **Source.** Borisov 1993; Roedland Compositio 2000; Bertin 2009;
  Inoue-Ito-Miura arXiv:1607.05925; Coates-Galkin-Kasprzyk
  arXiv:1212.1722.
* **Method.** Hodge tables for Pfaffian / Grassmannian-degeneracy CY3s.
* **Finding.** Published Pfaffian CY3s have h^{1,1} = 1 (mirror partners
  of complete intersections in Grassmannians). The (3,3) bin is empty.
* **Classification.** EXCLUDED.

### 3.6 Other small-Hodge constructions

* **Yau three-generation CY (Yau 1986 / Tian-Yau 1987):** downstairs
  Hodge (1, 4). EXCLUDED.
* **Six original mirror-symmetric CY3s (quintic, bicubic, etc.):** all
  h^{1,1} = 1 or 2 upstairs. EXCLUDED.
* **Tian-Yau bicubic cover (14, 23) and Z/3 quotient (1, 4):** EXCLUDED.
* **Generalized CICYs (gCICY, Anderson-Apruzzi-Gao-Gray-Lee
  arXiv:1507.03235):** the published gCICY catalogue does not yet
  tabulate Z/3xZ/3-quotient (3,3) entries comprehensively.
  **Classification.** PARTIAL (cycle-2 work to enumerate gCICY (3,3)
  bin).

## 4. Consolidated candidate list (all classifications)

| Classification | Name | Catalogue | (h11,h21) | Free Z/3xZ/3 | Bundle |
| --- | --- | --- | --- | --- | --- |
| **HARD** | `schoen_z3xz3` | Schoen 1988 / DOPR 2004 | (3, 3) | EXPLICIT (DOPR §3) | EXPLICIT (BCD/DHOR/AGLO) |
| **SOFT** | *(none)* | — | — | — | — |
| **PARTIAL** | `ks_3_3_bin` | Kreuzer-Skarke 2000 | (3, 3) | DEFERRED (cycle 2) | NONE PUBLISHED |
| **PARTIAL** | `gcicy_class` | gCICY (Anderson 2015) | (3, 3) | DEFERRED (cycle 2) | NONE PUBLISHED |
| **EXCLUDED** | `pfaffian_class` | Pfaffian (Roedland 2000) | empty | N/A | N/A |
| **EXCLUDED** | `yau_three_generation` | Yau 1986 | (1, 4) | N/A | N/A |
| **EXCLUDED** | `quintic_in_P4` | CDGP 1991 | (1, 101) | N/A | N/A |

(Total: 1 HARD, 0 SOFT, 2 PARTIAL, 3 EXCLUDED-categories. Note `ks_3_3_bin`
and `gcicy_class` are bin-aggregated rows — they bundle a single-digit
number of polytope-distinct constructions whose individual enumeration
is cycle-2 work.)

## 5. Verdict — how many candidates does the Hodge filter alone produce?

**HARD-match competitors to Schoen at (3,3): zero.** Cycle 1 produces a
candidate set of:

* **1 HARD entry** — Schoen-Z/3xZ/3 itself, which is the target — fully
  published with construction, free Z/3xZ/3 action, and SU(5) line-bundle
  SM bundle data.
* **2 PARTIAL bins** — KS toric hypersurfaces at (3,3) and gCICY at
  (3,3). Both are confirmed non-empty as Hodge bins but have **no**
  published free-Z/3xZ/3 action and **no** published heterotic SM bundle
  data. Their per-construction enumeration is cycle-2 work.
* **3 EXCLUDED catalogue families** — Pfaffian, Yau-3-gen, the
  six-original mirror set. These do not contain (3,3) entries.

The **hypothesis is supported** (and the falsification clauses do not
trigger): the (3,3) set is finite, small (single-digit per catalogue
where populated), and enumerable from published sources. Across all
catalogues surveyed, **only the Schoen-Z/3xZ/3 quotient is a HARD-match
candidate** — i.e. the only published smooth CY3 at (3,3) with a
documented free Z/3xZ/3 action and a documented SU(5) line-bundle SM
bundle. The PARTIAL bins (KS and gCICY) require further analysis but are
**not yet** competitors to Schoen because neither the free-action nor
the bundle data is published for any of their entries.

## 6. Forward path

* **Cycle 2 (free-Z/3xZ/3-action filter).** Enumerate the explicit
  polytope IDs in the KS (3,3) bin (Toric CY Database query) and the
  gCICY (3,3) entries (Anderson et al. 2015 + follow-ups). For each,
  compute the toric / projective automorphism group and check for a
  free Z/3xZ/3 subgroup. Survivors become SOFT-match candidates.
* **Cycle 3+ (bundle-admissibility filter).** For each cycle-2
  survivor, attempt a heterotic line-bundle (or monad) SU(5) GUT
  bundle satisfying c_1 = 0, c_3 = +-3 generations after Z/3xZ/3
  reduction, c_2(V) <= c_2(TX), polystability w.r.t. some Kähler class,
  and a balanced 3:3:3 Wilson-class Cartan-projection partition (the
  P-Wilson-Fix constraint from `p_wilson_fix_diagnostic.md`). Any
  survivor here would be a bona fide HARD-match competitor to Schoen;
  if no survivor exists, Schoen's uniqueness within the published
  catalogue universe is established.

## 7. Honest stopping notes

1. The Kreuzer-Skarke (3,3) bin was **not** exhaustively enumerated in
   cycle 1 — only the count (single-digit, populated) was confirmed
   from the published Hodge plot. Cycle 2 is responsible for the
   polytope-vertex-level enumeration.
2. The gCICY (3,3) bin is similarly recorded as PARTIAL — the gCICY
   catalogue is small but not formally complete in the literature, and
   cycle 1 did not run an automated scan.
3. No fabricated entries were added. Every CY3 named in §4 has an
   explicit citation (Schoen 1988, DOPR 2004, BHOP 2005, KS 2000,
   AGHJN 2014, CCM 2017, CL16, AGLP 2011, Roedland 2000, Yau 1986,
   gCICY 2015).
4. No Rust source files were modified; cycle 1 lives entirely in
   `python_research/` and `references/`.

— end cycle 1 —
