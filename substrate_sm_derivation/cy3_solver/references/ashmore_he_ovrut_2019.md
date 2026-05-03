# Machine Learning Calabi-Yau Metrics (Ashmore-He-Ovrut 2019)

**Authors:** Anthony Ashmore, Yang-Hui He, Burt A. Ovrut
**arXiv:** [1910.08605](https://arxiv.org/abs/1910.08605)
**Journal:** Fortschr. Phys. 68 (2020) 2000068 — DOI: 10.1002/prop.202000068
**Status:** retrieved (ar5iv HTML)

> **Naming note:** Our task brief called this "Ashmore-Lukas 2020". The actual authors are **Ashmore, He, Ovrut**. Lukas is not on this paper. There is a related Ashmore-Ruehle paper at [2103.07472](https://arxiv.org/abs/2103.07472) and a Larfors-et-al at [2205.13408](https://arxiv.org/abs/2205.13408) — see those refs.

---

## σ-functional convention

Same DKLR-style L¹ Monge-Ampère error, written explicitly as:

```
σ_k  ≡  (1 / Vol_CY) · ∫ dVol_CY · | 1  −  ((ω_k³ / Vol_K) / (Ω ∧ Ω̄ / Vol_CY)) |
```

This is the **integrated normalised** form of DKLR's σ — divided by Vol_CY, with η_k re-expressed using volume ratios. Numerically equivalent to DKLR's σ when both volumes are correctly normalised.

The training loss for the ML model is **not σ directly** — it is a regression loss on the metric or its determinant, with σ used as a held-out validation metric.

## Sampling convention

- Standard DKLR point cloud on the Fermat quintic.
- ML method: gradient-boosted decision trees (NOT neural networks, in this paper). Authors note "we have tried some other ML structures, such as forward-feeding multi-layer perceptron neural networks" but found decision trees performed comparably or better.

## Tabulated σ values

| k  | σ_k                | Source                         | Notes                                                    |
|----|--------------------|--------------------------------|----------------------------------------------------------|
| 1  | **0.375301**       | direct numerical quote, body   | Below TYZL asymptotic regime                             |
| 2  | **0.273948**       | direct numerical quote, body   | First non-trivial k                                      |
| 12 | **≈ 0.05**         | Fig. 1 plot read               | Estimated from figure to ±10%                            |

Authors note: **"σ_k approaches zero at least as fast as k⁻²"** (consistent with TYZL).

There is **no σ ≈ 10⁻⁴ value** in this paper. The "best σ ≈ 10⁻⁴" placeholder our task brief mentions is most likely confusion with the **Larfors et al. 2022** ([2205.13408](https://arxiv.org/abs/2205.13408)) paper, which reports σ = 0.0086 ≈ 10⁻² (not 10⁻⁴) for the φ-model best validation. Or further confusion with much-later neural-network papers (Douglas-Lakshminarasimhan-Qi 2020, Larfors-Ruehle-Schneider 2021/22) that get into 10⁻³ regime.

## Other quantities

- **Validation performance** plateau: improvement plateaus around k=7 when predicting the metric from g^(1) alone (Fig. 5).
- **Speed-up** vs. Donaldson alone: ≈ 50× for the metric, ≈ 75× for the determinant.

## Citation for tests

```rust
// Ashmore-He-Ovrut 2019 arXiv:1910.08605, Fortschr. Phys. 2020.
// σ_k for the Fermat quintic, gradient-boosted-decision-tree ML model:
const AHO2019_QUINTIC_SIGMA_K1: f64 = 0.375301;   // body, directly quoted
const AHO2019_QUINTIC_SIGMA_K2: f64 = 0.273948;   // body, directly quoted
const AHO2019_QUINTIC_SIGMA_K12_FIG1: f64 = 0.05; // estimated from Fig. 1, ±10%

// Asymptotic: σ_k → 0 at least as fast as k^{-2} (consistent with TYZL).
//
// NOTE: Our placeholder "σ ≈ 10^{-4} (Ashmore-Lukas)" is mis-attributed.
// AHO2019 reports σ ≈ 0.05 at k=12. For sub-10^{-3} σ on the quintic,
// re-source from Larfors et al. 2022 (larfors_2022.md, σ ≈ 0.0086) or
// later neural-network papers, NOT this one.
const AHO2019_SIGMA_K12_TOL_REL: f64 = 0.10;  // ±10% absolute on figure read
```

## Direct-quote citations

> "σ_1 = 0.375301" — body, first row.
>
> "σ_2 = 0.273948" — body, second row.
>
> "as k increases, σ_k approaches zero at least as fast as k⁻²" — body discussion.
>
> "σ_k ≡ (1/Vol_CY) ∫ dVol_CY |1 − (ω_k³/Vol_K)/(Ω ∧ Ω̄/Vol_CY)|" — definition, Eq. (around 3.X).
>
> "the improvement plateaus around k=7" — Fig. 5 caption.
>
> "factor of 50 speed-up over using Donaldson's algorithm alone… 75 speed-up for the determinant" — body.
