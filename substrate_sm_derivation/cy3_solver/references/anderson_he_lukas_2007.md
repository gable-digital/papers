# Heterotic Compactification, An Algorithmic Approach (Anderson-He-Lukas 2007)

**Authors:** Lara B. Anderson, Yang-Hui He, Andre Lukas
**arXiv:** [hep-th/0702210](https://arxiv.org/abs/hep-th/0702210)
**Journal:** JHEP 07 (2007) 049
**Status:** abstract only via arXiv; ar5iv HTML extraction failed (encoding error in our retrieval pipeline); body NOT retrieved

---

## ⚠️ Symmetry-reduced invariant counts NOT confirmed

Our task brief said: confirm against this paper's Table 2 the values **6, 9, 12, 18, 27, 63** (degrees k=2..7 ?) for the **Z_5⁴ ⋊ S_5** invariant subspace dimension on the Fermat quintic, and **k=5 → 2, k=10 → 4** for the SU(5)-restricted (multiples-of-5) Z_5⁴ subspace.

We were **unable to retrieve** the body of this paper. The arXiv abstract page does not contain the table; the ar5iv HTML rendering returns an encoding error; the Springer page returns 404 for our automated path.

What we *did* find in adjacent literature (search hits, not direct retrieval):

- The **Sylvester invariants of binary quintics** are at degrees 4, 8, 12, 18 — relevant numerology but a different problem (binary quintics, not the Fermat quintic threefold).
- The general statement "S_5 acts on 5 variables, so primary invariants have degrees (2, 3, 4, 5, 6)" — these are the elementary symmetric polynomials, not the same as the Z_5⁴ ⋊ S_5 invariants on the *quotient* H⁰(O(k))^Γ.

## Adjacent paper that may have the table directly

[Braun-Brodie-Lukas-Ruehle, "Learning Group Invariant Calabi-Yau Metrics by Fundamental Domain Projections", arXiv:2407.06914] explicitly reduces the Donaldson spectral basis using Γ = Z_5⁴ ⋊ S_5 invariance on the Fermat quintic and quotes σ ~ 3.1/k² − 4.2/k³ (smaller pre-factor than full-basis AKLP because of the symmetry reduction). The body of *that* paper likely has the dim H⁰(O(k))^Γ table that's actually wanted here, but we didn't retrieve it either.

## Citation for tests

```rust
// Anderson-He-Lukas 2007 arXiv:hep-th/0702210, JHEP 07 (2007) 049.
//
// Body NOT retrieved. Our placeholder Z_5^4 ⋊ S_5 invariant counts
// for the Fermat quintic at k=2..7  (6, 9, 12, 18, 27, 63)  could NOT
// be confirmed against this paper. Same for the SU(5)-restricted
// (multiples-of-5) counts (k=5 → 2, k=10 → 4).
//
// Action: source from
//   Braun-Brodie-Lukas-Ruehle 2024 (arXiv:2407.06914), or
//   Bull-He-Hirst-Lukas symmetric-quintic papers, or
//   direct LiE / Macaulay2 computation of the Hilbert series of
//     ℂ[x_0..x_4] / (x_0^5+...+x_4^5)  graded by total degree,
//     under the Γ = Z_5^4 ⋊ S_5 action,
// before pinning these as test constants.
const ANDERSON_HE_LUKAS_2007_INVARIANT_COUNTS_VERIFIED: bool = false;
```

## Direct-quote citations

> Abstract: "We approach string phenomenology from the perspective of computational algebraic geometry, by providing new and efficient techniques for proving stability and calculating particle spectra in heterotic compactifications. Stability of monad bundles is decided… particle spectra including singlets are computed."

(Body content not retrieved.)

## Action item

Either:
1. Get a copy of the JHEP 0707:049 PDF and paste the relevant table, or
2. Compute the dimensions directly using a CAS (LiE, Macaulay2, sage) and use *that* as the citation for the test, with this paper as cross-validation only.
