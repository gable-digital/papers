# Some numerical results in complex differential geometry (Donaldson 2009)

**Author:** S. K. Donaldson
**arXiv:** [math.DG/0512625](https://arxiv.org/abs/math/0512625)
**Journal:** Pure Appl. Math. Q. 5 (2009) no. 2, 571-618
**Status:** retrieved (ar5iv HTML)

---

## T-operator definition (the foundational equation for our pipeline)

Donaldson's iteration on the space of positive-definite Hermitian forms G = (G_{αβ}) acting on H⁰(X, L^k):

```
Start with positive-definite G_{αβ}; let G^{αβ} be its inverse.
For z ∈ ℂ^{N+1} set
        D(z) = Σ_{α,β} G^{αβ} z_α z̄_β.

Then  ( T(G) )_{γδ}  =  R · ∫_X f_{γδ} dμ,
        where  f_{γδ} = z_γ z̄_δ / D(z),  R is a fixed normalisation.
```

Fixed points of T are the **balanced metrics** (Donaldson 2001).

## Bergman-kernel asymptotics (the σ ~ 1/k² statement)

The Tian-Yau-Zelditch-Lu expansion is the source of σ ~ 1/k²:

```
ρ_k(x) = Σ_α |s_α(x)|²  ~  k^n + a_1(x) k^(n-1) + a_2(x) k^(n-2) + ...
```

where the a_i are local invariants and a_1 is proportional to the scalar curvature. **Donaldson does not state σ_k ~ 1/k² as a free-standing theorem in this paper** — that formulation is the standard "informed" rephrasing in the numerical CY literature, derived from the TYZL expansion plus the T-iteration's fixed-point structure.

## Convergence rate — Proposition 1

> "If automorphism group is discrete and balanced metric G exists, then for any G_0 ∈ M, sequence Tʳ(G_0) converges to G as r → ∞."

The convergence is **geometric**:

```
ε_i(r)  ~  c_i · σ^r,    σ ∈ (0, 1)
```

Here r is the iteration index (NOT the polynomial degree k). Don't confuse this iteration-σ with the k-degree σ from DKLR.

## Worked numerical example

The paper presents a **CP¹ toy** at k=6:
- The eigenvector sequence converges to (1, 6, 15, 20) after **40 T-iterations**.
- The spectral gap σ ≈ 0.8 governs the geometric decay rate.
- No Fermat-quintic numerical example in this paper.
- No dP3 numerical example either (dP3 is later, Doran-Headrick-Herzog-Kantor-Wiseman 2008).

## Citation for tests

```rust
// Donaldson 2009 math/0512625, Proposition 1: T-iteration converges
// geometrically with rate σ_iter ∈ (0,1) when Aut is discrete and balanced
// metric exists. CP^1 toy at k=6 reaches eigenvector (1, 6, 15, 20) in
// 40 iterations with σ_iter ≈ 0.8 spectral gap.
//
// The Bergman kernel TYZL expansion is the *source* of the k^{-2}
// scaling later observed numerically by DKLR/AKLP, but Donaldson 2009
// does NOT prove σ_k ~ 1/k² for the Monge-Ampère error directly.
const DONALDSON_CP1_K6_NITERS_TO_CONVERGE: usize = 40;
const DONALDSON_CP1_K6_SPECTRAL_GAP: f64 = 0.8;
```

## Direct-quote citations

> "ρ_k = Σ |s_α|² ~ k^n + a_1 k^{n-1} + a_2 k^{n-2} + …" — Bergman-kernel expansion.
>
> "Proposition 1: If automorphism group is discrete and balanced metric G exists, then for any G_0 ∈ M, sequence Tʳ(G_0) converges to G as r → ∞."
>
> "(T(G))_{γδ} = R ∫_X f_{γδ} dμ" — T-operator definition.

## Convention note for our use

The "Donaldson balanced metric" at finite k is *not* the Calabi-Yau metric — it differs by O(1/k²). That residual is what DKLR/AKLP measure as σ_k. Donaldson's 2009 paper is the existence/convergence theorem; quantitative σ_k values must come from the numerical follow-ups (DKLR, AKLP).
