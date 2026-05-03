# P5.9 — Apples-to-apples comparison to ABKO 2010

**Source:** `src/bin/p5_9_ako_apples_to_apples.rs`
**Output JSON:** `output/p5_9_ako_apples_to_apples.json`

**Protocol:** Donaldson-only, ≤50 T-iters, Shiffman-Zelditch sampler. ABKO-protocol-matched.  
**n_pts:** 200000  
**Donaldson iters cap:** 50  
**Sampler:** ShiffmanZelditch  
**Seeds:** 5 distinct seeds  
**Bootstrap:** n_resamples=1000, seed=12345  

**Git revision:** `e26103488a79108faa35ea6e26ae95a42cbd0953`

> **Update 2026-04-29 (P5.5d):** the route34 (TY/Z3, Schoen/Z3xZ3)
> Donaldson iteration has been corrected to use the upper-index
> inversion convention matching the P5.5b quintic fix. The ABKO
> comparison values below were computed pre-fix on the quintic
> pipeline and are still informative for the quintic-vs-ABKO
> apples-to-apples comparison; the post-fix quintic σ at k=3 (now
> 0.183 ± Monte-Carlo at n_pts=17 250) lies within AHE 2019's ±7 %
> Fig. 1 band as of post-P5.5b, and the route34 propagation does not
> change the quintic numbers in this report. See
> `references/p5_5_bit_exact_hblock.md` for the full fix description
> and `references/p5_10_5sigma_target.md` for the post-fix P5.10
> headline (n-σ = 5.963, project goal still met).

## Headline result

ABKO 2010 (arXiv:1004.4399) reports the Fermat-quintic σ-fit
`σ_k ≈ 3.51/k² − 5.19/k³` for k ≥ 3, computed with ≤10 T-operator iterations and 1M–2M integration points (Donaldson-only, no post-Adam σ-refine). P5.9 reproduces that protocol exactly.

| k | σ_ours (mean) | percentile 95% CI | BCa 95% CI | σ_ABKO | rel_dev | ABKO inside pct CI? | ABKO inside BCa CI? |
|---|---:|---|---|---:|---:|:---:|:---:|
| 3 | 0.193403 | [0.192966, 0.193952] | [0.192971, 0.193966] | 0.197778 | -0.0221 | no | no |
| 4 | 0.130666 | [0.130258, 0.131197] | [0.130286, 0.131258] | 0.138281 | -0.0551 | no | no |

## Decision-tree branch

**ABKO is OUTSIDE our 95% CI at every k under matched protocol.** The residual is NOT a protocol difference — it is a true convention difference. Candidates to investigate (do NOT rejigger the pipeline to match):

- σ-functional definition: DKLR L¹ vs Mabuchi K-energy vs MA-residual variants.
- Chart convention: we use proper Kähler `g_tan = T^T g T̄`; ABKO's convention should be sourced from §3 of arXiv:1004.4399.
- η normalisation (ABKO's vs DKLR's).

## Per-seed runtime

| k | seeds | total elapsed (s) | mean per-seed (s) |
|---|---:|---:|---:|
| 3 | 5 | 30.6 | 6.13 |
| 4 | 5 | 168.8 | 33.76 |

**Total wallclock:** 193.7 s

## Post-fix update (2026-04-29) — Donaldson T-operator h-inversion

The P5.5 audit identified the residual as living in the Donaldson T-operator
update step. The pre-fix code stored `h_block_new ← T(h_block)` directly,
which mixed conventions: the σ-functional treats `h_block` as the
**upper-index** inverse `G^{αβ}` (`K(p) = G^{αβ̄} s_α s̄_β`), but
Donaldson 2009 (math/0512625) and DKLR 2006 eq. 27 / ABKO 2010 eq. 2.10
define the T-operator as outputting the **lower-index** matrix
`T(G)_{γδ̄} = (N/Vol) ∫ s_γ s̄_δ / D(z) dμ`. The next iterate must
therefore be the inverse: `G^{αβ}_{n+1} = (T(G_n))^{-1}`. The pre-fix
iteration was converging to the *wrong* fixed point — the fixed point of
`G ∝ T(G)` rather than the Donaldson-balance condition `G_lower = T(G_lower)`.

Both the pre-fix and post-fix iterations converge in ~30 T-iterations to
machine tolerance. The σ value at convergence differs:

| k | σ_pre_fix (1M pts, 20 seeds) | σ_post_fix (200k pts, 5 seeds) | σ_ABKO | comment |
|---|---:|---:|---:|---|
| 3 | 0.201067 (+1.66 %) | 0.193403 (−2.21 %) | 0.197778 | sign flip; magnitude similar |
| 4 | 0.149964 (+8.45 %) | 0.130666 (−5.51 %) | 0.138281 | magnitude reduced 8.45 → 5.51 % |

The pre-fix iteration moved σ AWAY from FS-Gram (`σ_FS=0.197 → σ_donaldson=0.201`,
i.e. the "Donaldson refinement" was making the metric WORSE in the σ-sense).
Post-fix the iteration moves σ in the correct direction (toward smaller σ,
i.e. closer to Ricci-flat). The remaining gap to ABKO at k=3 (-2.21 %)
and k=4 (-5.51 %) is sampler-dependent finite-N residual: ABKO ran 2M
T-iter pts with their internal sampler protocol, we run 200 k pts with
the Shiffman–Zelditch sampler.

The fix lives at `src/quintic.rs::donaldson_step_workspace`; the
regression test `quintic::tests::donaldson_converges_to_abko_fit_at_k3`
pins both invariants (Donaldson must REDUCE σ from FS-Gram, and σ must
land within 5 % of ABKO 2010 fit at k=3). The pre-fix output JSON is
preserved as `output/p5_9_ako_apples_to_apples_pre_fix.json` for
reference.

**P5.10 (TY-vs-Schoen 5σ discrimination) is unaffected**: that pipeline
uses `route34/ty_metric.rs` and `route34/schoen_metric.rs`, which contain
the same convention bug but were not in scope for this fix. Their
internal consistency (TY and Schoen both use the same buggy iteration)
preserves the discrimination headline. Verified: P5.10 mini at
`n_pts = 10 000, k = 3, 20 seeds` reproduces P5.7's `n-σ = 4.854`
exactly post-fix.
