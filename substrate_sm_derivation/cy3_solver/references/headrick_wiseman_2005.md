# Numerical Ricci-flat metrics on K3 (Headrick-Wiseman 2005)

**Authors:** Matthew Headrick, Toby Wiseman
**arXiv:** [hep-th/0506129](https://arxiv.org/abs/hep-th/0506129)
**Journal:** Class. Quant. Grav. 22 (2005) 4931 — DOI: 10.1088/0264-9381/22/23/008
**Status:** retrieved (ar5iv HTML), but **paper is K3, not the quintic**

---

## ⚠️ MISMATCH WITH OUR USE

The original task brief named this paper as a source for "σ values at k=2..6 for the Fermat quintic". **That is wrong.** This paper is titled **"Numerical Ricci-flat metrics on K3"**. K3 is a 4-real-dimensional Calabi-Yau, not the Fermat quintic. The methodology is **lattice-based discretization on coordinate patches**, not the Donaldson algebraic-metric / balanced-metric approach used by DKLR and the rest of our pipeline.

Any "σ_HW(k=2,4,6)" value cited in our test suite for the *quintic* is mis-attributed. Likely the original placeholder confused this paper with DKLR (hep-th/0612075). The HW2005 σ does not exist for our problem.

## What the HW2005 paper actually does

- **Manifold:** K3 (4-real-dim CY two-fold)
- **Method:** finite-difference / lattice solution of the Einstein equation on coordinate patches
- **Resolution:** up to ≈ 2 × 10⁷ ≈ 80⁴ lattice points; effective resolution ≈ 400⁴ after using discrete symmetries
- **Parameter scanned:** α = 4πa²/b² ∈ (0,1), nine values: α = 0.03, 0.13, 0.28, 0.50, 0.61, 0.72, 0.79, 0.85, 0.92
- **No "σ at degree k"** — there's no algebraic-metric expansion in this paper, so no analogue of DKLR/Donaldson σ_k.

## σ-functional convention

Not applicable. The σ symbol that appears (σ = |z¹|² + |z²|², Eq. 34) is a **geometric coordinate combination** for parametrising K3, not an error functional.

## Sampling convention

Lattice / finite-difference, not point sampling.

## Citation for tests

```rust
// HW2005 hep-th/0506129 is K3, NOT the quintic, and uses lattice-FD,
// NOT algebraic-metric expansion. There is no HW2005 "σ at degree k"
// applicable to our Fermat-quintic Donaldson pipeline.
//
// If a test currently cites HW2005 for σ values on the quintic,
// it is mis-attributed. Re-source from DKLR 2006 (dklr_2006.md) or
// Anderson et al. 2010 (aklp_2010.md).
```

## Direct-quote citations

> Title: "Numerical Ricci-flat metrics on K3"
>
> "α = 4πa²/b²" — definition of the modulus parameter, varied over (0, 1).
>
> "α = 0.03, 0.13, 0.28, 0.50, 0.61, 0.72, 0.79, 0.85, 0.92" — list of computed values.
>
> "≈ 80⁴ points… effective resolution around 400⁴" — discretisation level.

## Action item for our test suite

Any test using `HW2005_SIGMA_K*` constants should be **re-sourced** to:
- Anderson et al. 2010 (1004.4399) for fitted `σ_k = 3.51/k² − 5.19/k³` on the Fermat quintic, OR
- DKLR 2006 (hep-th/0612075) Fig. 1 with explicit caveat that DKLR did not tabulate.
