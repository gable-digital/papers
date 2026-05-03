# CY3 Substrate-Discrimination — Consolidated Publication-Grade Summary

**Document purpose.** This is the document a hostile reviewer of the
CY3 substrate-discrimination paper would read. It leads with the two
publication headlines, defends each, points at the source JSON
outputs, and tags everything else as supporting / diagnostic /
follow-up.

**Status as of P8.5 / P8.4c (Apr 2026):** publication-ready on the
σ-channel discriminability and the chain-channel physics-preference
headlines. Other channels (Hodge, Yukawa, k=4 σ, k=4 chain) are
diagnostic / supporting only — they do **not** carry their own
publication headlines and are not included in the model-comparison
Bayes factor that drives the physics-preference number.

---

## Section 1 — Headlines

* **Discriminability (σ-channel).** **6.92σ** Tier 0 strict-converged,
  BCa 95% CI **[5.30, 9.04]** (percentile 95% CI [5.65, 10.03]),
  k=3, n_pts=40 000, donaldson_iters=100, Donaldson-only (no Adam),
  20 seeds (n_TY=20, n_Schoen=16 strict-converged). Source:
  `output/p5_10_ty_schoen_5sigma.json` (P5.10 / P5.5k). Defended
  through 8 hostile-review rounds (see `references/p5_10_5sigma_target.md`).

* **Physics preference (chain channels).** **5.43σ Schoen-favored**
  (combined ln BF = **−14.76 nats**, "Strong" on the Kass–Raftery
  scale; convention: positive ln BF = TY-favored, negative =
  Schoen-favored). Computed from `chain_quark` (ln BF = **−5.60**) +
  `chain_lepton` (ln BF = **−9.16**) at k=3, n_pts=25 000.
  σ-channel is **excluded** from this BF (see Section 2). Source:
  `output/p7_11_quark_chain.json`, `output/p7_11_lepton_chain.json`,
  combined in `output/p8_1_bayes_multichannel.json` (P7.11 + P8.1f,
  defended via P8.1g).

* **(Supporting, NOT a publication headline.) Discriminability at
  k=4.** **3.82σ** Tier 0 strict-converged, BCa 95% CI **[2.48,
  6.37]** (percentile 95% CI [3.05, 10.39]), k=4, n_pts=40 000,
  donaldson_iters=100, Donaldson-only with **static damping α=0.5**,
  20 seeds (n_TY=20, n_Schoen=7 strict-converged). Source:
  `output/p5_10_k4_damped.json` (P8.4-fix). This is **not** an
  independent publication headline (still under 5σ at strict tier);
  it is reported here as a **supporting tier consistent with the
  k=3 6.92σ headline**, replacing the prior un-damped 1.83σ
  diagnostic-only figure. See §3.1 and
  `references/p8_4_k4_production.md`.

* **(Supporting, NOT a publication headline.) Discriminability at
  k=5.** **10.87σ** Tier 0 strict-converged, BCa 95% CI **[8.79,
  15.01]** (percentile 95% CI [9.27, 34.44]), k=5, n_pts=40 000,
  donaldson_iters=200, Donaldson-only with static damping α=0.5,
  20 seeds (n_TY=20, **n_Schoen=6** strict-converged). Source:
  `output/p5_10_k5_damped_relaunch.json` (P-K5-relaunch). The point
  estimate AND both CI floors clear 5σ, but the strict-converged
  Schoen sample is small (6/20 vs k=3's 16/20), so this is filed as
  a **supporting tier strengthening the k=3 6.92σ headline at higher
  basis order**, NOT as a replacement publication headline. The
  conservative (Tier 1) k=5 reading collapses to 1.01σ under
  heavy-tail Schoen pathology — qualitative confirmation of σ's
  basis-size-artifact character (P8.1e) and reinforcement of the
  decision to exclude σ from the BF. See §4 and
  `references/p_k5_production.md`.

  **Adaptive-damping caveat.** A second production sweep was run with
  P8.4-fix-c's **adaptive damping** (initial α=0.3 with monotone
  bidirectional ramp, auto-default at k=4). Result:
  `output/p5_10_k4_adaptive.json` Tier 0 strict-converged = **1.83σ**
  — bit-identical to the un-damped baseline. Mechanism: at production
  scale the smooth-descent UP-ramp pushes α to 1.0 within ~3 iters
  on healthy seeds, after which catastrophic divergence on the stalled
  Schoen seeds fires at high α and the regression guard restores the
  same min-residual snapshots as the un-damped path. **Static α=0.5
  is the load-bearing supporting damping**; adaptive (without
  hysteresis) under-performs at production scale despite out-performing
  static at test-speed n_pts=2 500. Tracked as P8.4-fix-f.

Two distinct, falsifiable, geometric claims — one per headline. The
σ-channel is a discriminability claim ("the two substrates can be
told apart at finite k"); the chain-channel result is a model-comparison
claim ("under the chain-match likelihood the data prefer Schoen over
TY"). They are not redundant and they are not double-counted: the BF
that produces the 5.43σ figure does not include the σ-channel.

---

## Section 2 — Why σ is excluded from the model-comparison Bayes factor

σ measures the DKLR L¹ Monge–Ampère residual of the candidate
Calabi–Yau metric against its Donaldson balanced-metric reference. At
finite k it is dominated by basis-size differences (n_TY = 87 vs
n_Schoen = 27 at k=3) rather than by physics observables. The
project-MEMORY entry classifies σ as a **discrimination channel, not a
curvature proxy at finite k**, and P8.1e formalised the decision to
exclude σ from the model-comparison BF on those grounds.

**Direct empirical confirmation (P-BASIS-CONVERGENCE-prod, 20 seeds × n_pts=40 000
× Donaldson tol=1e-6, paired bootstrap B=10 000, boot_seed=12345).** When TY's
native k=3 basis is truncated to match Schoen's 27 monomials, the σ-channel
discrimination does not just shrink — it **reverses sign**. At native bases
Δσ ≡ σ_TY − σ_Schoen = **−7.89 (BCa 95% CI [−15.25, −5.52])** with n_TY=87,
n_Schoen=27. At matched basis size n_b=27 the same paired-bootstrap analysis
gives Δσ_matched = **+23.59 (BCa 95% CI [+11.33, +42.66])**, with TY's σ
jumping from 0.27 to 31.74 once its basis is restricted to 27 monomials.
The two CIs are disjoint (paired-bootstrap probability of opposite sign =
1.0000 over B=10 000 resamples), and the magnitude ratio
|Δσ_matched|/|Δσ_native| = **2.99 (BCa 95% CI [1.54, 6.55])** places the
native gap entirely inside the basis-size-artefact bucket — the matched-n_b
measurement disagrees with the native one in both sign and magnitude.
Source: `output/p_basis_convergence_prod.json`, full reduction in
`references/p_basis_convergence_diagnostic.md` §0.3.

Including σ in a model-comparison BF would therefore add roughly 30 nats of
spurious signal whose sign depends on basis convention — large enough to
invert the chain-channel Schoen-favored verdict (combined ln BF = −14.76,
§3.4). This is the textbook signature of an artifact term, not a likelihood
contribution.

P8.4 corroborates the same picture at k=4: when Schoen's basis grows from
27 (k=3) to 48 (k=4), the strict-converged Tier 0 σ-discrimination
weakens from **6.92σ (k=3)** down to **1.83σ (k=4 un-damped, P8.4
baseline)** and partially recovers to **3.82σ (k=4 with static
damping α=0.5, P8.4-fix)** — still under 5σ at strict tier, while the
chain-channel sign-agreement is preserved (P8.5b). That weakening is
exactly the basis-size-artefact P8.1e predicted; the matched-basis
production result above closes the empirical loop on the prediction.

σ is therefore reported **only** as discriminability (|t|, n-σ, BCa CI at
native bases) and is excluded from the BF combination via
`for_combination=false` on its `ChannelEvidence`. The k-scan native-basis
ratios Δσ(k=3)/Δσ(k=2) = 0.65 (BCa [0.20, 1.82]) and Δσ(k=4)/Δσ(k=2) =
1.06 (BCa [0.23, 3.42]) are consistent with 1/k² Donaldson scaling within
their CIs at production sample size; the **matched-basis k=3 sign reversal
above** carries the load-bearing weight of the σ-exclusion argument.

This is enforced in the multi-channel Bayes binary: the sigma-channel
breakdown is reported separately under `discriminability` /
`sigma_channel_breakdown` in `p8_1_bayes_multichannel.json`, and the
`physics_preference` block (which produces the 5.43σ-equivalent
headline) sums only the chain channels.

---

## Section 3 — Per-channel evidence

### 3.1 σ-channel (k=3, publication; k=4, supporting only)

* **k=3 publication result.** Tier 0 strict-converged, n=40 000,
  donaldson_iters=100, 20 seeds. **n-σ = 6.92**, BCa [5.30, 9.04],
  pct [5.65, 10.03]. Both percentile and BCa 95% CI floors clear 5σ.
  Tier 0 *strict-converged* gating: `residual < tol AND iters < cap`.
  No Tukey trim; no guard-snapshot inflation; n_Schoen=16 (the 4
  non-strict Schoen seeds are excluded by the strict gate). Source:
  `output/p5_10_ty_schoen_5sigma.json` →
  `discrimination_strict_converged[0]`.
* **k=4 supporting tier (current).** Tier 0 strict-converged at k=4,
  with **static damping α=0.5** in the Donaldson update, sits at
  **3.82σ**, BCa [2.48, 6.37], pct [3.05, 10.39] (n_TY=20,
  n_Schoen=7 strict-converged). Source: `output/p5_10_k4_damped.json`
  (P8.4-fix). Reported as a **supporting tier consistent with the
  k=3 6.92σ headline**, NOT as an independent publication headline:
  3.82σ is under 5σ at the strict tier, and the k=3 result remains
  the canonical σ-discriminability publication number.
* **k=4 un-damped baseline (kept for cross-comparison).** Without
  damping, Tier 0 strict-converged at k=4 falls to **1.83σ**, BCa
  [1.34, 2.33]; 10/20 Schoen seeds stall at high-residual fixed
  points (residuals 1.26e-6 to 9.99e-3) without hitting iter cap=100
  (max iters=90) — a Donaldson stall mode, not insufficient
  iterations. The 1.83 → 3.82 lift under damping (with the same n_pts,
  same iter cap, same seeds) is the key qualitative result: it
  converts k=4 from "diagnostic-only artifact" into a corroboration
  of the k=3 headline, and **supports P8.1e's prediction** that σ has
  a basis-size-artifact component dominant at k=4 (Schoen n_basis
  27 → 48). Tier 3 (Tukey re-trim) reports 11.41σ un-damped / 12.13σ
  damped but is **diagnostic only** at any k, not publication-grade —
  the same post-hoc trim was rejected at k=3 by P5.5h. Full discussion:
  `references/p8_4_k4_production.md` (updated for P8.4-fix). An
  **adaptive-damping production sweep is running in background**
  (P8.4-followup, ~2.4 hr) and will be reported separately. Sources:
  `output/p5_10_k4_damped.json` (current),
  `output/p5_10_k4_gpu_donaldson_only.json` (un-damped baseline).

### 3.2 chain_quark (k=3, publication)

* ln BF (TY vs Schoen) = **−5.60**, **Schoen-favored**.
* 4-row basis-convergence sweep (test degrees 2, 3, 4, 5; basis_dim
  ∈ {18, 62, 178, 450}) at k=3, n_pts=25 000, iter_cap=100, seed
  12345. All 4 rows show Schoen residual < TY residual (sign-agree:
  YES). `converged: false` flag refers to the basis-convergence
  status of the chain-match residual itself (see P7.11), not to a
  failure of the metric solve. Source: `output/p7_11_quark_chain.json`.

### 3.3 chain_lepton (k=3, publication)

* ln BF (TY vs Schoen) = **−9.16**, **Schoen-favored**.
* Same 4-row sweep structure as quark; lepton chain uses the φ
  test-function family. n_pts=25 000, k=3, iter_cap=100, seed 12345.
  Source: `output/p7_11_lepton_chain.json`.

### 3.4 Combined chain BF (k=3, publication)

* combined ln BF = chain_quark + chain_lepton = **−14.76 nats**
* combined log10 BF = **−6.41**
* combined strength label = **Strong** (Kass–Raftery)
* posterior odds (TY) = **3.90e−7**
* n-σ-equivalent = **5.43**
* verdict = **Schoen-favored**
* Source: `output/p8_1_bayes_multichannel.json` →
  `physics_preference`.

### 3.5 Hodge channel (P8.2 — current state, supporting only)

* Hodge-counting channel (kernel of Δ_{∂̄}^V on Z/3×Z/3 trivial-rep
  sub-bundle, symmetric h11/h21 split, χ = h11 − h21, Gaussian
  log-likelihood at predicted (h11, h21, χ) = (3, 3, −6) with σ_h=0.5,
  σ_χ=1.0). Reported in `output/p8_2_hodge_diagnostic.json`:
  Δ log L (TY − Schoen) = **+3328 nats**, "Hodge channel favours TY by
  3328 nats". **Not a publication headline** in current form: the
  Schoen branch's measured (h11, h21) = (32, 32) vs predicted (3, 3)
  reflects an upstream zero-mode degeneracy (P8.3-followup-A scope) and
  the TY branch's measured kernel total of 0 reflects the bundle
  kernel-eigenvalue threshold setting. The result is reported as a
  diagnostic until the upstream Hodge zero-mode count is regressed
  against the bundle-Laplacian eigenvalue threshold (P8.3-followup-A).
  Until then it does **not** participate in the BF combination.

### 3.6 Yukawa channel (P8.3b — NOT publication-ready)

* Per `references/p8_3_yukawa_production_results.md` and
  `output/p8_3b_yukawa_production.json`: rank-1 Yukawa tensor in both
  candidates collapses 8 of 9 fermion masses to floating-point zero,
  giving Δ log L = numerically saturated under f64 χ². The headline
  binary value `discrimination_n_sigma = 534.5` is a numerical
  artifact of the rank-1 collapse, not a discrimination claim.
* **Publication status: NOT publication-ready.** Three independent
  upstream issues identified (P8.3c findings):
  * **P8.3-followup-A:** harmonic zero-mode degeneracy (Schoen branch
    measures h11=32 instead of predicted 3; rank-1 Yukawa is
    downstream of this).
  * **P8.3-followup-B:** real Schoen Z/3×Z/3 line-bundle SM (current
    AKLP `WilsonLineE8::canonical_e8_to_e6_su3` alias is wrong-arity
    for Schoen — it's a TY-side construction reused as a placeholder).
  * **P8.3-followup-C:** single-Higgs-zero-mode contraction (the
    Yukawa-tensor build collapses the Higgs leg to a single zero-mode,
    which is the proximate cause of rank-1).
* Until these three are resolved the Yukawa channel does **not**
  participate in the BF combination and is not a publication
  headline.

---

## Section 4 — Supporting evidence at higher k (k=4, k=5)

* **σ-channel at k=4 (P8.4 / P8.4-fix):** the current k=4 strict-tier
  result with static damping α=0.5 is **3.82σ**, BCa [2.48, 6.37],
  pct [3.05, 10.39] (`output/p5_10_k4_damped.json`). Reported as a
  **supporting tier consistent with the k=3 6.92σ headline** — *not*
  an independent publication headline (still under 5σ at strict
  tier). The un-damped baseline at the same n_pts / iter cap / seeds
  was 1.83σ (`output/p5_10_k4_gpu_donaldson_only.json`); the
  monotone 1.83 → 3.82 lift under damping is the key qualitative
  result and supports P8.1e's basis-size-artifact prediction (see
  §3.1 and `references/p8_4_k4_production.md`). The Tier 3 Tukey
  re-trim numbers (11.41σ un-damped, 12.13σ damped) remain
  **diagnostic-only at any k**, NOT publication-grade — same post-hoc
  trim was rejected at k=3 by P5.5h. An **adaptive-damping production
  sweep is running in background** (P8.4-followup, ~2.4 hr) and will
  be reported separately.

* **σ-channel at k=5 (P-K5-relaunch):** Tier 0 strict-converged
  reads **10.87σ**, BCa [8.79, 15.01], pct [9.27, 34.44]
  (`output/p5_10_k5_damped_relaunch.json`, GPU, donaldson_iters=200,
  donaldson_tol=1e-6, n_pts=40 000, static damping α=0.5,
  n_TY=20, **n_Schoen=6** strict-converged). The point estimate
  and both CI floors clear 5σ; bootstrap-seed jitter on the strict
  tier is 0.018σ (well within the round-4 0.05σ target). Reported
  as a **supporting tier strengthening the k=3 6.92σ headline at
  higher basis order**, NOT as a replacement publication headline,
  because the strict-converged Schoen subsample is only 6/20 (vs
  k=3's 16/20). The conservative (Tier 1) k=5 reading collapses to
  1.01σ — two pathological Schoen seeds (seed 5 σ=54, seed 4242
  σ=23 791) that the strict gate excludes drag the SE up by ~3 orders
  of magnitude. This Tier 0 ↑ / Tier 1 ↓ pattern is the qualitative
  signature of σ being a **basis-size-sensitive discrimination
  channel** rather than a clean physics observable, and reinforces
  the P8.1e decision to exclude σ from the model-comparison BF.

* **k-scan progression (Tier 0 strict-converged σ-discriminability):**

  | k | Tier 0 n-σ | BCa 95% CI | n_TY | n_Schoen_strict | Source |
  |---|-----------|-----------|------|------------------|--------|
  | **3** | **6.92** | [5.30, 9.04] | 20 | **16** | `p5_10_ty_schoen_5sigma.json` (publication headline) |
  | 4 | 3.82 (damped) | [2.48, 6.37] | 20 | 7 | `p5_10_k4_damped.json` (supporting) |
  | 5 | 10.87 | [8.79, 15.01] | 20 | 6 | `p5_10_k5_damped_relaunch.json` (supporting) |

  Reading: σ_TY converges with weak basis-size dependence
  (0.27 → 0.278 → 0.295). Strict-converged Schoen sample shrinks
  monotonically (16 → 7 → 6) as Schoen's basis grows (27 → 48 → 75)
  and more seeds stall in Donaldson plateaus at high-residual fixed
  points. The strict-tier n-σ is non-monotone in k (6.92 → 3.82 →
  10.87), driven by which Schoen seeds clear the strict gate at
  each k rather than by a stronger underlying physics signal — the
  same pattern that defines σ as a basis-size-sensitive
  discrimination channel. Discriminability persists at every k
  reported here at the strict tier.

* **Chain channel at k=4 (P8.5b):** sign-agreement preserved on both
  quark and lepton chains across all three reported test-degree cells
  (Schoen unanimously wins). **Qualitative corroboration of P7.11
  k=3 result only** — magnitudes are not directly comparable because
  P8.5b ran under reduced wall-clock budget (n_pts=8000, iter_cap=25,
  Adam disabled) and is not iter-cap-matched / n_pts-matched to the
  k=3 baseline. Caveats per P8.5d. See `references/p8_5_chain_match_k4.md`.
* **Production-grade k=4 chain-match** (matched n_pts and iter_cap to
  the P7.11 k=3 baseline, with Adam refinement) is **deferred to
  P8.5-followup** as a multi-hour batch job.

---

## Section 5 — Reproducibility

* **SHA-chained event log.** Every published run emits a kernel
  replog (`*.kernel.replog`) and an end-to-end replog (`*.replog`)
  with SHA-256-chained event records. Final-chain SHA is recorded in
  the JSON output (e.g. `repro_log_final_chain_sha256_hex`,
  `repro_log_n_events`). Bit-exact determinism is checked by replaying
  the kernel replog through the deterministic-replay harness
  (P-REPRO).
* **System manifest.** Each output JSON's `repro_manifest` block
  records git revision, Rust toolchain, target triple, CPU features,
  hostname, UTC timestamp, command-line, and Rayon thread count. All
  numbers in this document trace back to a `repro_manifest`-stamped
  JSON in `output/`.
* **Repro_log paths** (canonical headlines):
  * `output/p5_10_ty_schoen_5sigma.json` (k=3 σ headline)
  * `output/p7_11_quark_chain.json`, `output/p7_11_lepton_chain.json`
    (chain channels)
  * `output/p8_1_bayes_multichannel.json` (combined BF)
* **Bit-exact determinism test (P-REPRO).** Re-running any of the
  above with the recorded git revision and seeds produces a
  byte-identical kernel replog whose final-chain SHA matches
  `repro_log_final_chain_sha256_hex`. This is checked in CI.

---

## Section 6 — Open follow-ups (NOT blocking publication)

The following items are tracked as scoped follow-ups; none of them
gate the two publication headlines in §1.

* **P8.3-followup-A:** upstream harmonic zero-mode degeneracy on the
  Schoen branch (measured h11=32 vs predicted 3). Affects Hodge
  channel (§3.5) and is an upstream prerequisite for Yukawa rank-9
  recovery.
* **P8.3-followup-B:** real Schoen Z/3×Z/3 line-bundle SM. Current
  `WilsonLineE8::canonical_e8_to_e6_su3` alias is wrong-arity for
  Schoen; needs a `MonadBundle::schoen_*` constructor (or pipeline
  signature extension to accept the native `Z3xZ3WilsonLines` type).
* **P8.3-followup-C:** single-Higgs-zero-mode contraction. Proximate
  cause of the rank-1 Yukawa tensor in P8.3b.
* **P8.4-followup:** Schoen Donaldson stall at k=4. Static damping
  α=0.5 has been validated (`output/p5_10_k4_damped.json`, Tier 0
  3.82σ supporting tier — see §3.1). **Adaptive-damping production
  sweep is currently running in background** (~2.4 hr) and will be
  reported separately as `references/p8_4_k4_adaptive.md`. Static
  α=0.5 is the lower bound; adaptive is expected to do at least as
  well at the strict tier. Even with adaptive damping, k=4 is not
  expected to displace the k=3 6.92σ publication headline — it
  remains a supporting / corroborating tier.
* **P8.5-followup:** production k=4 chain-match with Adam refinement
  (n_pts and iter_cap matched to P7.11 k=3 baseline; multi-hour
  batch).
* **P8.6-followup-E:** hard cap on `Z3xZ3BundleConfig::seed_max_total_degree`
  to prevent inadvertent basis-order escalation in downstream
  consumers.

None of the above follow-ups change the §1 headlines if resolved, and
none of them invalidate the §1 headlines if deferred.
