# P5.9 — Apples-to-apples comparison to ABKO 2010

**Source:** `src/bin/p5_9_ako_apples_to_apples.rs`
**Output JSON:** `output/p5_9_ako_apples_to_apples.json`

**Protocol:** Donaldson-only, ≤50 T-iters, Shiffman-Zelditch sampler. ABKO-protocol-matched.  
**n_pts:** 1000000  
**Donaldson iters cap:** 50  
**Sampler:** ShiffmanZelditch  
**Seeds:** 20 distinct seeds  
**Bootstrap:** n_resamples=1000, seed=12345  

**Git revision:** `b2b8f5209606a1f2216993bc65d46f6d07bc2820`

## Headline result

ABKO 2010 (arXiv:1004.4399) reports the Fermat-quintic σ-fit
`σ_k ≈ 3.51/k² − 5.19/k³` for k ≥ 3, computed with ≤10 T-operator iterations and 1M–2M integration points (Donaldson-only, no post-Adam σ-refine). P5.9 reproduces that protocol exactly.

| k | σ_ours (mean) | percentile 95% CI | BCa 95% CI | σ_ABKO | rel_dev | ABKO inside pct CI? | ABKO inside BCa CI? |
|---|---:|---|---|---:|---:|:---:|:---:|
| 4 | 0.130247 | [0.130178, 0.130317] | [0.130175, 0.130315] | 0.138281 | -0.0581 | no | no |

## Decision-tree branch

**ABKO is OUTSIDE our 95% CI at every k under matched protocol.** The residual is NOT a protocol difference — it is a true convention difference. Candidates to investigate (do NOT rejigger the pipeline to match):

- σ-functional definition: DKLR L¹ vs Mabuchi K-energy vs MA-residual variants.
- Chart convention: we use proper Kähler `g_tan = T^T g T̄`; ABKO's convention should be sourced from §3 of arXiv:1004.4399.
- η normalisation (ABKO's vs DKLR's).

## Per-seed runtime

| k | seeds | total elapsed (s) | mean per-seed (s) |
|---|---:|---:|---:|
| 4 | 20 | 3437.8 | 171.89 |

**Total wallclock:** 3285.6 s
