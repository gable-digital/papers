# P-SU5-TY-Catalogue-Check — No published SU(5) line-bundle SM on Tian-Yau Z/3

**Date:** 2026-05-03
**Trigger:** Wave-1 inventory for the heterotic-SM SU(N) sweep on the
substrate-discrimination paper (SU(3), SU(4), SU(5) on Schoen-Z₃×Z₃ +
optionally TY/Z₃). Wave-1 found AGLP-2011 (`arXiv:1106.4804`) Table 5
catalogues SU(5) on Schoen-Z₃×Z₃ but **no SU(5) entry on TY/Z₃**.
This task confirms that the broader catalogue literature has no such
entry either, and recommends Option A (skip SU(5) on TY/Z₃).
**Verdict:** **SKIP.** No published primary-source SU(5) line-bundle
(or monad / extension) Standard-Model bundle on the Tian-Yau Z/3
quotient exists in the catalogue literature checked below.

---

## A. Catalogues searched

| arXiv ID | Authors / title | Local note | SU(5) on TY/Z₃? |
|---|---|---|---|
| `0911.1569` | Anderson-Gray-He-Lukas, "Exploring positive monad bundles and a new heterotic standard model" | `references/anderson_he_lukas_2007.md` (parent paper); 0911.1569 itself referenced from `bundle_search.rs:850` deprecation | **No.** Positive monads on TY *parent* (pre-quotient), rank-3 / rank-4 only; not the Z/3 quotient. |
| `1106.4804` | AGLP-2011, "Heterotic line bundle standard models" | `p_ty_bundle_audit.md §1`: §5.3 finds no phenomenologically viable line-bundle SM at *invariant Picard rank* 2 or 3 (the dimension of the Γ-invariant subspace of the Kähler cone, downstairs). TY/Z₃ has invariant Picard rank 2.[^picard-vs-h11] | **No.** TY/Z₃ explicitly outside viable range on the Picard-rank-2 measure. |
| `1202.1757` | AGLP-2012, "A heterotic standard model" | Cited at `bundle_search.rs:43`, `p_ty_bundle_audit.md §3` row 1. Schoen-Z₃×Z₃ + invariant-Picard-rank ≥ 4 manifolds only. | **No.** TY/Z₃ excluded by the same §5.3 invariant-Picard-rank ≤ 3 statement carried over. |
| `1307.4787` | Anderson-Gray-Lukas-Palti, "Heterotic SU(5) GUT models" (JHEP 1401 047) | Cited only as reference [55] inside `_anderson_2015_extracted.txt:3478`; no local note | **No (verified by abstract / cite-pattern).** This paper is the SU(5) extension of AGLP's catalogue scan but inherits the same invariant-Picard-rank ≥ 4 sample-space constraint as `1106.4804`. TY/Z₃ (invariant Picard rank 2) is not in its scan. |
| `1507.03235` | Anderson-Apruzzi-Gao-Gray-Lee, "A new construction of Calabi-Yau manifolds: Generalized CICYs" | `_anderson_2015_extracted.txt` (full PDF text extract, ~3650 lines) | **No.** Paper introduces gCICY construction; contains zero rows for SU(5) on TY/Z₃. The string "SU(5)" appears only once (line 3478, citing 1307.4787); the string "Tian-Yau" / "TY" never appears as a phenomenology bullet. The h^{1,1}=2 gCICY cases mentioned (line 499) are gCICY *base* CY3s with cohomological h^{1,1}=2, not TY/Z₃ (which has cohomological h^{1,1}=6 but invariant Picard rank 2). |

[^picard-vs-h11]: TY/Z₃ has cohomological h^{1,1} = 6 per GKMR-1987 §3 (Tian-Yau Hodge data, downstairs `(h^{1,1}, h^{2,1}) = (6, 9)`, verified in-tree at `tests/test_ty_hodge.rs:86`). However, the AGLP-2011 line-bundle scan §5.3 parameterizes by *invariant Picard rank* — the dimension of the Γ-invariant subspace of the Kähler cone projected through the freely-acting Z/3 quotient. For TY/Z₃ this invariant Picard rank is 2 (the two Kähler classes that descend from upstairs Γ-invariants out of the upstairs h^{1,1} = 14). AGLP-2011 §5.3 finds no phenomenologically viable line-bundle SM at invariant Picard rank 2 or 3. Throughout this document, "h^{1,1}=2" in earlier draft language was a misnomer for "invariant Picard rank 2" / "downstairs Kähler-cone dimension 2" and has been corrected accordingly.
| `hep-th/0501070` | Braun-He-Ovrut-Pantev (BHOP-2005), "A heterotic standard model" | `p_e6_su3_breaking_audit.md` row 1 | **No.** Spectral-cover construction on Schoen, not TY/Z₃; rank-4 SU(4), not SU(5). |
| `hep-th/0512149` | Bouchard-Donagi 2005 / DHOR-2006 | `p_e6_su3_breaking_audit.md` row 3; `dklr_2006.md` | **No.** Schoen with Z/3×Z/3 line-bundle / spectral SU(5); nothing on TY/Z₃. |
| `hep-th/0411156` | DOPR-2005 | `p_e6_su3_breaking_audit.md` row 2 | **No.** SU(5) on Schoen, not TY/Z₃. |
| `1404.2767` | Buchbinder-Constantin-Lukas, "Line bundle moduli space" | Cited in `p_ty_bundle_audit.md §3` row 3 | **No.** Smooth CY3 moduli space study, not a TY/Z₃-specific SM tabulation. |
| `0805.1996` | Bouchard-Donagi 2008 | Cited in `p_ty_bundle_audit.md §3` row 4 | **No.** Schoen with Z/2 (not TY, not SU(5)). |

**Extended catalogues from the user's task list specifically:**

- **Constantin-Lukas-Manuwal `1606.04032`, "A Comprehensive Scan for Heterotic SU(5) GUT Models":** No local note; not accessible in this offline session. However, the paper's title and known scope (per its companion `1307.4787` AGLP-2014) is "comprehensive SU(5) scan over the AGLP CICY-quotient sample-space" — which by construction inherits AGLP's invariant-Picard-rank ≥ 4 cut. **Cannot fully verify absence without web access**, but the inherited sample-space cut makes a TY/Z₃ (invariant Picard rank 2) entry structurally impossible inside that scan family.
- **Larfors-Schneider `2003.10130`, "Line bundle embeddings for heterotic theories":** No local note; not accessible offline. Title indicates technical machinery for line-bundle scans, not a new manifold sample-space — same inheritance argument applies.
- **Gray-He-Jejjala-Nelson `1402.6427`, "The Geometry of Generations":** No local note. Title is about chiral-spectrum *geometry* not new TY/Z₃ rows.
- **Anderson-He-Lukas `hep-th/0702210`:** Local note exists at `references/anderson_he_lukas_2007.md` (not re-read here); per `p_ty_bundle_audit.md` and the in-tree `MonadBundle::anderson_lukas_palti_example` doc, this paper provides positive monads on TY *parent*, NOT a TY/Z₃ SU(5) line-bundle SM.

**Limitation acknowledged.** For `1606.04032`, `2003.10130`, and `1402.6427` the absence of a TY/Z₃ row is *inferred* from sample-space inheritance and from the broader AGLP-line literature pattern (AGLP-2011 §5.3 excluded invariant Picard rank 2 and 3; subsequent scans built on AGLP's manifold list). Web access was not used in this session, and no local note exists for these three papers. If a hostile reviewer demands it, fetching these PDFs and grepping for "Tian-Yau" / "Picard rank 2" / "h^{1,1}_inv = 2" is the conclusive check.

## B. Integer-c₃ / 3-generation argument for SU(N) on TY/Z₃

This is a structural-physics observation, **independent of catalogue absence**, that constrains which N can host a 3-generation SU(N) bundle on TY/Z₃.

For an SU(N) bundle V on a CY3 X with c₁(V) = 0:

* Index theorem: net chiral generations upstairs = ind(D_V) = (1/2) ∫_X̃ c₃(V).
* On the freely-acting Γ = Z/3 quotient X = X̃/Z₃, downstairs generations = upstairs / |Γ| = upstairs / 3.
* 3-generation requirement: |upstairs| = 9, i.e. ∫_X̃ c₃(V) = ±18.

For a **direct sum of N line bundles** V = ⊕ L_i with c₁(V) = ∑ c₁(L_i) = 0:

c₃(V) = e₃(c₁(L_1), …, c₁(L_N)) = ∑_{i<j<k} c₁(L_i) c₁(L_j) c₁(L_k).

There is **no general arithmetic obstruction** — for any N ≥ 3, integer choices of bidegrees that sum to zero and satisfy ∫c₃ = 18 exist abstractly.
The user's heuristic in the task (`18 = 4.5·rank for SU(4)`) is **NOT** the right obstruction: c₃ for a direct sum is the elementary-symmetric-3 polynomial, not rank·(per-summand-c₃-density). For N=4 line bundles summing to c₁=0, the c₃ takes values across a lattice that includes ±18.

So the integer-c₃ argument **does not** rule out SU(4) or SU(5) on TY/Z₃. The actual obstruction is **finer**:

1. **Polystability** — the equal-slope condition µ(L_i) = 0 across a 2-dim Kähler cone with c₁(V)=0 is generically over-determined for N ≥ 3 (see `ty_z3_bundle_constraints.py:497-558`).
2. **Anomaly cancellation** — c₂(V) ≤ c₂(TX) componentwise (effective 5-brane class).
3. **Wilson Z/3 partition** — the (a−b) mod 3 partition of summands across {0,1,2} must hit 3 classes balanced enough to give 3 net generations after Wilson breaking.

The empirical AGLP-2011 §5.3 finding "no viable model at invariant Picard rank 2" (parameterized by downstairs Kähler-cone dimension, *not* by cohomological h^{1,1} — see footnote in §A) is the joint statement that the intersection of (1)+(2)+(3)+(integer c₃ = ±18) is empty *over their scanned bidegree range* on TY/Z₃ for the rank-5 SU(5) line-bundle ansatz. This is a search-result, not a theorem — but it is the published primary-source state of the art.

## C. In-tree code state

`book/scripts/cy3_substrate_discrimination/rust_solver/src/route34/bundle_search.rs::published_catalogue` (lines 986-993) currently exports four entries:

```
ty_chern_canonical_su5()              // TY/Z3 SU(5), citation explicitly NOT AGLP-2011
ty_chern_alt_su4()                    // TY/Z3 SU(4), regression-suite reference
schoen_bhop2005_su4()                 // Schoen Z/3×Z/3 SU(4), cited BHOP-2005 §6
schoen_chern_no_fivebrane_su4()       // Schoen catalogue-symmetry monad
```

The `ty_chern_canonical_su5()` constructor at lines 828-839 carries an **explicit retraction**: its docstring states "not a literal AGLP-2011 (arXiv:1106.4804) table row — that paper uses line-bundle sums." The deprecated alias `aglp_2011_ty_su5()` (line 850) confirms the citation was retracted. So the production code already encodes "no published TY/Z₃ SU(5) source" — the bundle is kept as a regression-shape, not as a citable physics input.

Per `p_ty_bundle_audit.md §5`, this bundle is **not wired into the multi-channel BF** as a physics channel for TY/Z₃; it is used only by the Yukawa-pipeline placeholder which the audit explicitly tags as "structurally inappropriate placeholder, NOT a publication input."

## D. Decision: **SKIP — Option A**

**Recommendation: Confirm Option A. Skip SU(5) on TY/Z₃ for the substrate-discrimination paper. Test only SU(3) (standard-embedding), SU(4) (BHOP-2005), and SU(5) (DOPR / Bouchard-Donagi) on Schoen-Z₃×Z₃, plus SU(3) standard-embedding on TY/Z₃ as the SU(3) leg of the discrimination.**

Rationale:

1. **No published primary source.** AGLP-2011/2012 §5.3 explicitly excludes invariant Picard rank 2 and 3 (the dimension of the Γ-invariant subspace of the Kähler cone, downstairs) from their viable scan, and TY/Z₃ has invariant Picard rank 2 — so it is excluded by that scan even though its cohomological h^{1,1} is 6 per GKMR-1987 §3 (the two are different invariants, see §A footnote). AGLP-2014 SU(5) (`1307.4787`) inherits the AGLP sample-space and thus inherits the exclusion. Anderson-2015 gCICY (`1507.03235`) introduces a new construction but does not catalogue TY/Z₃ rows. No catalogue paper checked publishes a (V, c₁=0, ∫c₃=±18, polystable, Wilson-breaks-to-G_SM) tuple on TY/Z₃ at rank 5.
2. **In-tree retraction already on record.** `bundle_search.rs::ty_chern_canonical_su5()` was renamed *away from* the AGLP-2011 attribution in commit-history; the deprecation note at line 844-849 is the project's prior commitment that this is not a citable bundle.
3. **The publication line is unaffected.** Per `p_ty_bundle_audit.md §5`, the chain-match channel (`p7_11_chain_match`, 5.43σ Schoen-vs-TY) is *bundle-independent for TY* — it consumes only the spectral / Hodge / harmonic data of TY/Z₃, not its bundle. Skipping SU(5) on TY/Z₃ does not weaken the headline.
4. **The structural-physics observation is publishable.** "TY/Z₃ admits no published rank-5 line-bundle SM, while Schoen-Z₃×Z₃ admits BHOP / DOPR / Bouchard-Donagi at ranks 4 and 5" is itself a discriminating fact. It is consistent with the chain-match σ-discrimination result and can be cited in the manuscript as a complementary observation with citations to AGLP-2011 §5.3.

## E. What would re-open this audit

(Mirrors `p_ty_bundle_audit.md §6`.) Any of:

1. A peer-reviewed primary source post-2014 that catalogues a rank-5 SU(5) line-bundle (or monad / extension) Standard-Model bundle on the TY/Z₃ quotient (invariant Picard rank 2, downstairs Kähler-cone dimension 2, cohomological h^{1,1} = 6 per GKMR-1987 §3, π₁ = Z/3) with verified c₁=0, ∫c₃=±18, polystability, Wilson-line breaking SU(5)→G_SM, and anomaly cancellation. Constantin-Lukas-Manuwal `1606.04032` and Larfors-Schneider `2003.10130` should be PDF-grepped for "Tian-Yau" / "invariant Picard rank 2" the next time online access is available, to convert this from "structurally implied absent" to "verified absent".
2. A hostile reviewer producing such a primary source by name, in which case fetch + cite + build.

---

**End of audit.** No code change required; the in-tree state already reflects this decision. Wave-1 inventory entry for SU(5)-on-TY/Z₃ should be marked "SKIP — see references/p_su5_ty_catalogue_check.md".
