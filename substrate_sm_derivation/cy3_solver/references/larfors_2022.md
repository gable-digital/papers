# Numerical Metrics for Complete Intersection and Kreuzer-Skarke Calabi-Yau Manifolds (Larfors et al. 2022)

**Authors:** Magdalena Larfors, Andre Lukas, Fabian Ruehle, Robin Schneider
**arXiv:** [2205.13408](https://arxiv.org/abs/2205.13408)
**Journal:** Mach. Learn.: Sci. Tech. 3 (2022) 035014
**Status:** retrieved (ar5iv HTML)

> **Naming note:** Task brief called this "Larfors-Schneider-Strominger". Strominger is not on this paper; the authors are Larfors-Lukas-Ruehle-Schneider. The brief's arXiv ID 2103.07472 is a different paper (Ashmore-Ruehle, on KK towers / swampland distance, also on the quintic but a different topic). Use 2205.13408 for numerical CY metrics with line bundles.

---

## σ-functional convention

DKLR-style L¹, written in the form (Eq. 4.8 in the paper):

```
σ  =  (1 / Vol_Ω)  ·  ∫_X  | 1  −  κ · (Ω ∧ Ω̄) / (J_pr)^n |
```

(here J_pr is the predicted Kähler form, κ is a normalisation enforcing Vol_J = Vol_Ω.)

The training loss is **not σ** directly; it is a Monge-Ampère regression loss (Eq. 4.3):

```
L_MA  =  ‖ 1  −  (1/κ) · det(g_pr) / (Ω ∧ Ω̄) ‖_n
```

Plus an explicit **Kähler-class loss** that pins the slopes of fixed line bundles to topological values — that's what makes this paper relevant to "CY metrics with line bundles".

## Sampling convention

- Standard DKLR-style sampling on quintic / bicubic / Kreuzer-Skarke manifolds.
- φ-model architecture (a neural-network approach), trained for ≈ 100 epochs.

## Tabulated σ values

| Manifold        | Best σ (φ-model)      | Source       | Notes                                                       |
|-----------------|-----------------------|--------------|-------------------------------------------------------------|
| Fermat quintic  | **σ = 0.0086**        | Fig. 1e      | Best validation; ≈ 1 hour single CPU; ~ k=20 Donaldson      |
| Bicubic CY3     | (in §5.2)             | Fig.s of §5.2 | Specific numerical value not extracted in our retrieval     |
| KS Picard-2 CY3 | (in §5.3)             | Fig.s of §5.3 | Specific numerical value not extracted in our retrieval     |

The paper notes the loss reduces by ≈ 2 orders of magnitude across 100 epochs, plateauing.

## Citation for tests

```rust
// Larfors-Lukas-Ruehle-Schneider 2022, arXiv:2205.13408
// Mach. Learn.: Sci. Tech. 3 (2022) 035014.
//
// φ-model neural network achieves σ = 0.0086 on the Fermat quintic
// (best validation, Fig. 1e), comparable to k≈20 Donaldson balanced.
const LARFORS2022_QUINTIC_SIGMA_BEST: f64 = 0.0086;
const LARFORS2022_QUINTIC_SIGMA_BEST_TOL_REL: f64 = 0.10;  // ±10%

// σ definition: L¹ Monge-Ampère error, Eq. 4.8.
//   σ = (1/Vol_Ω) ∫_X |1 − κ (Ω∧Ω̄)/(J_pr)^n|
//
// Training loss is L_MA (Eq. 4.3), NOT σ; σ is held-out validation.
```

## Direct-quote citations

> "σ = (1/Vol_Ω) ∫_X |1 − κ (Ω∧Ω̄)/(J_pr)^n|" — Eq. (4.8).
>
> "ℒ_MA = ‖1 − (1/κ) det(g_pr) / (Ω ∧ Ω̄)‖_n" — Eq. (4.3), training loss.
>
> "the φ-models with lowest validation loss reach… a mean accuracy of σ = 0.0086" — body, around Fig. 1e.
>
> "matches k=20 Donaldson algorithm performance" — body comparison.
>
> "stable values after the 100 epoch experiments" — training duration.

## Convention notes

- The paper's σ uses the **same DKLR L¹ form** (good — directly comparable).
- The reciprocal in the integrand is `(Ω∧Ω̄) / J^n` rather than DKLR's `J^n / (Ω∧Ω̄)` — but |1 − x| and |1 − 1/x| differ only at higher order around x = 1, so for converged metrics they agree to leading order.
