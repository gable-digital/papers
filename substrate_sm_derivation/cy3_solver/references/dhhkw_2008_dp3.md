# Numerical Kähler-Einstein Metric on the Third del Pezzo (DHHKW 2008)

**Authors:** Charles Doran, Matthew Headrick, Christopher Herzog, Joshua Kantor, Toby Wiseman
**arXiv:** [0803.4555](https://arxiv.org/abs/0803.4555)
**Journal:** Commun. Math. Phys. 282 (2008) 357-393 — DOI: 10.1007/s00220-008-0558-6
**Status:** abstract only via arXiv; ar5iv HTML rendering failed (PDF too large for our extractor); full body content NOT retrieved

---

## ⚠️ dP3 IS NOT A CALABI-YAU

The third del Pezzo surface is a **Fano** Kähler-Einstein manifold with **positive** scalar curvature (Ric = +g, after normalisation). It is **not Ricci-flat**. The "KE error" measured here is therefore **not directly comparable** to a CY-3 σ:

- CY case (DKLR/AKLP): we want Ric = 0, error = ∫ |η_k − 1|, η_k = det(ω) / (Ω ∧ Ω̄).
- dP3 KE case (DHHKW): we want Ric = +g (no holomorphic top form trivialisation); error is measured against the volume form proportional to ω², not Ω ∧ Ω̄.

If our test suite cites DHHKW for a Calabi-Yau σ comparison, that's a category error.

## What we *can* extract from the abstract / search-result summaries

- **Method:** three different algorithms — Ricci flow in complex coordinates, Ricci flow in symplectic coordinates, and a functional minimisation on a space of "symplectic algebraic" metrics.
- **Manifold structure used:** dP3's toric symmetry reduces the Einstein equation to a single Monge-Ampère equation in 2 real dimensions.
- **Numerical Ricci-tensor norms, sample point counts, σ-vs-k tables** are in the body of the paper which we did not retrieve.

## Sampling / σ-functional convention

Not extracted (paper body unavailable through our channels).

## Tabulated values

Not extracted. **Status: abstract-only.**

## Citation for tests

```rust
// Doran-Headrick-Herzog-Kantor-Wiseman 2008, arXiv:0803.4555
// Commun. Math. Phys. 282 (2008) 357.
//
// dP3 is a Fano KE manifold (Ric = +g), NOT a Calabi-Yau (Ric = 0).
// DHHKW tabulated values are NOT directly comparable to our CY3 σ_k.
//
// Numerical body of paper not retrieved; do NOT pin test values to this
// reference until the body is sourced (paywalled at Springer; PDF is
// large and not parseable by our WebFetch path).
//
// Action: if a CY3 test cites DHHKW, the citation is wrong. Re-source.
```

## Direct-quote citations (abstract only)

> "The third del Pezzo surface admits a unique Kähler-Einstein metric, which is not known in closed form. The manifold's toric structure reduces the Einstein equation to a single Monge-Ampère equation in two real dimensions."
>
> "We numerically solve this nonlinear PDE using three different algorithms… simulation of Ricci flow, in complex and symplectic coordinates respectively. The third algorithm involves turning the PDE into an optimization problem on a certain space of metrics, which are symplectic analogues of the 'algebraic' metrics used in numerical work on Calabi-Yau manifolds."

## Action item

Either
1. Obtain the published Springer PDF (institutional access) and paste the relevant table directly, or
2. Drop DHHKW citations from CY3-specific σ tests and use it only as a methodology reference for the symplectic-algebraic-metrics technique.
