# §5.7 — first TY-vs-Schoen σ-discrimination result

**Date**: 2026-04-27
**Crate**: `book/scripts/cy3_substrate_discrimination/rust_solver`
**Pipelines**:
- `route34::ty_metric` (driven via `route34::cy3_metric_unified::TianYauSolver`)
- `route34::schoen_metric` (driven via `route34::cy3_metric_unified::SchoenSolver`)

**Tests**: `src/route34/tests/test_p5_7_ty_schoen_discrimination.rs`
**Binary**: `src/bin/p5_7_ty_schoen_ensemble.rs`
**Data**:   `output/p5_7_ty_schoen_ensemble.json`

## Why this report exists

P5.4 produced σ_TY/Z3(k=2, n=10000, seed=42) ≈ **0.355** — the first σ
on a physics candidate (closing §4.5 / §7.1 of the hostile review). That
single-seed number, however, is statistically meaningless for
discrimination: §2.2 of the same review (and the P5.3 multi-seed
harness) flagged that "canonical reference σ" pinned at single seeds is
not a science value without a distribution.

P5.7 closes that gap for TY/Z3 vs Schoen/Z3×Z3:

1. Establishes the **Schoen baseline σ** at the equivalent setting
   (`d_x=2, d_y=2, d_t=1, n=10000, seed=42`).
2. Runs a **20-seed multi-candidate ensemble** at k ∈ {2, 3} for both
   varieties.
3. Computes the first **n-σ discrimination statistic** between the two
   physics candidates.

## Pipeline parameters

| Parameter | Value |
|---|---|
| Candidates | TY/Z3, Schoen/Z3×Z3 |
| k values | 2, 3 |
| Seeds | `[42, 100, 12345, 7, 99, 1, 2, 3, 4, 5, 137, 271, 314, 666, 1000, 2024, 4242, 0xDEAD, 0xBEEF, 0xCAFE]` (n=20, identical to P5.3 `SEEDS_20`) |
| n_pts | 10 000 |
| Donaldson max_iter | 25 |
| Donaldson tol | 1.0e-3 |
| Schoen tuple mapping | k=2 → (d_x=2, d_y=2, d_t=1); k=3 → (d_x=3, d_y=3, d_t=1) (matches `schoen_solver_dispatches_correctly` and `schoen_publication_default`) |
| Adam refine | **none** — fair across candidates (Adam wiring exists for `QuinticSolver` only; CY3 candidates would each need a bespoke wiring) |
| Bootstrap | n_resamples=1000, seed=12345, ci_level=0.95 |
| Z/3 quotient (TY) | enabled |
| Z/3 × Z/3 quotient (Schoen) | enabled |
| Total wall clock | 45.4 s on one workstation |

## Schoen baseline σ (Step 1)

Mirroring the P5.4 TY baseline test:

| Setting | n_basis | iters | σ_final | wall (s) |
|---|---:|---:|---:|---:|
| Schoen (d_x=2, d_y=2, d_t=1), n=10000, seed=42 | 12 | 8 | **1.6424** | 0.092 |

Source test: `test_p5_7_schoen_sigma_at_k2_is_finite`. The Donaldson
balance converges in 8 iterations (residual 9.34 × 10⁻⁴, below the 1.0
× 10⁻³ tolerance). σ ≈ 1.64 vs the TY baseline σ ≈ 0.355 — Schoen at
k=2 is roughly **4.6× larger** than TY at the same setting. This is
the expected hierarchy: the Schoen Z/3×Z/3 invariant section basis at
(2,2,1) has only n_basis = 12 (vs 28 for TY at k=2), so Donaldson
balancing has very little freedom.

We deliberately do **not** assert σ < 1 in the Schoen baseline test
(the TY P5.4 test does): with n_basis = 12 and n=10000, σ > 1 is
consistent with under-resolution of the Ricci-flat metric, not a
pipeline bug. The discrimination harness handles this by working with
ensemble means and standard errors rather than the bare value.

## Multi-seed ensemble (Step 2)

Per (candidate, k), Donaldson-only balance, n=10000, 20 seeds. σ stats:

| Candidate | k | n_basis | n_ok | mean σ | stderr | std | pct95 CI | BCa95 CI |
|---|---:|---:|---:|---:|---:|---:|---|---|
| TY     | 2 | 28 | 20 | **0.343987** | 1.818e-3 | 8.13e-3 | [0.3404, 0.3473] | [0.3402, 0.3472] |
| TY     | 3 | 87 | 20 | **1.017993** | 6.971e-3 | 3.12e-2 | [1.0047, 1.0311] | [1.0023, 1.0295] |
| Schoen | 2 | 12 | 20 | **1.863836** | 5.166e-1 | 2.31e0  | [1.2156, 3.0289] | [1.2871, 3.5454] |
| Schoen | 3 | 27 | 20 | **3.339399** | 4.782e-1 | 2.14e0  | [2.5551, 4.3333] | [2.6931, 4.5911] |

Observations:
- **TY σ is tightly distributed** (relative std ≤ 3% at both k). The 28
  / 87 invariant + Gröbner-reduced section basis gives Donaldson enough
  freedom to converge consistently across seeds.
- **Schoen σ is heavy-tailed**: most seeds land in σ ∈ (1.0, 1.6) at
  k=2 and σ ∈ (2.0, 3.0) at k=3, but a handful (seed=4 → 11.4 at k=2;
  seeds 666, 4242 → 9 at k=3) blow out, dragging the mean and SE
  upward. This is consistent with the very small Schoen invariant basis
  at this setting (n_basis = 12 / 27), which leaves Donaldson with too
  little freedom to recover from a poor sampler initialisation. The
  fix is **not** to truncate the Schoen distribution — that would bias
  the comparison — but to bump the basis at k=4 (n_basis = 67 with the
  publication default (4,4,2)) and `n_pts ≥ 50000`.
- Schoen at k=3 takes more Donaldson iterations to converge (median
  ~13 vs ~8 at k=2), and several seeds hit the `max_iter = 25` cap
  without reaching the tolerance (donaldson residual still > 1e-3).

## n-σ discrimination (Step 3)

Per k, **n-σ = |⟨σ_TY⟩ − ⟨σ_Schoen⟩| / √(SE_TY² + SE_Schoen²)**:

| k | ⟨σ_TY⟩ | SE_TY | ⟨σ_Schoen⟩ | SE_Schoen | Δσ | SE_comb | **n-σ** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.343987 | 1.818e-3 | 1.863836 | 5.166e-1 | −1.5198 | 5.166e-1 | **2.942** |
| 3 | 1.017993 | 6.971e-3 | 3.339399 | 4.782e-1 | −2.3214 | 4.783e-1 | **4.854** |

**max n-σ = 4.854 at k=3.**

## Verdict

**PARTIAL DISCRIMINATION — σ alone, at this budget, is just shy of the
5σ project goal.**

Specifically:
- **k=2 → 2.94σ**: σ alone would be a > 99.5% one-tailed signal but
  not a 5σ discovery. The mean separation Δσ ≈ −1.52 is large; the
  bottleneck is Schoen's heavy tails at n_basis = 12.
- **k=3 → 4.85σ**: within rounding of the 5σ threshold. A modest budget
  bump (n_pts → 25 000 or n_seeds → 40) would push this past 5σ purely
  by tightening SE_Schoen, **without** changing the σ-eval logic.

Sign of Δσ: **σ_TY < σ_Schoen** at every k. This is the expected
hierarchy if TY's larger Gröbner-reduced invariant basis (n_basis = 28
/ 87 vs Schoen's 12 / 27) gives Donaldson more degrees of freedom to
balance the Bergman metric to Ricci-flat. It does *not* mean TY is
"more correct" as a physics candidate — that's what the joint
likelihood (Bayes-factor harness, P5.x next) decides.

## Channels available beyond σ

If the project goal is hard 5σ on σ alone, the cheapest paths are:

1. **Bump n_pts to 25 000–50 000** at k=3 — already costs < 4 min in
   this binary's framing. Tightens both SEs.
2. **Push to k=4** with the publication-default (4, 4, 2) tuple for
   Schoen and matching `max_iter ≥ 50`. n_basis grows to 67 (Schoen)
   and ~140 (TY); Donaldson has enough freedom for both candidates to
   converge tightly, which closes Schoen's heavy tail. Wall clock per
   seed at k=4 will be ~10 s (Schoen) / ~30 s (TY) — full ensemble
   ~15 min.
3. **Trim the per-seed σ outliers** by adding Adam post-refine to both
   candidates. The P5.3 protocol (sigma_refine_analytic + running min)
   delivered ~50% σ reductions on the Fermat quintic; analogous wiring
   for TY and Schoen should land σ_Schoen tightly under 1 at k=3.

If σ alone cannot cross 5σ at any practical budget (extremely unlikely
given the k=3 result), the project's η-integral, hidden-bundle, and
Yukawa-overlap channels (already implemented in `route34::eta_evaluator`,
`hidden_bundle`, `yukawa_overlap_real`) provide independent
discriminators. Combining them via `route34::bayes_factor`
(nested-sampling joint likelihood) is the canonical 5σ path.

## Files added this wave

- `src/route34/tests/test_p5_7_ty_schoen_discrimination.rs` — Schoen
  baseline test (mirrors P5.4) + multi-seed ensemble test (`#[ignore]`d
  long-run; n=4000 to keep test wall-clock under a minute).
- `src/route34/tests/mod.rs` — wires the new module.
- `src/bin/p5_7_ty_schoen_ensemble.rs` — full 20-seed × 2-candidate ×
  2-k ensemble binary; mirrors `p5_3_multi_seed_ensemble.rs`.
- `Cargo.toml` — registers the new binary.
- `output/p5_7_ty_schoen_ensemble.json` — full ensemble dump including
  per-seed σ records, bootstrap CIs, and discrimination rows.
- `references/p5_7_ty_schoen_discrimination.md` — this report.

## Reproducibility

```bash
cd book/scripts/cy3_substrate_discrimination/rust_solver

# Step 1 — Schoen baseline σ at k=2 (mirrors P5.4 TY baseline)
cargo test --release --features gpu --lib test_p5_7_schoen_sigma_at_k2_is_finite \
    -- --nocapture

# Step 2+3 — full 20-seed ensemble + n-σ discrimination
cargo run --release --features gpu --bin p5_7_ty_schoen_ensemble

# Step 4 — ignored ensemble regression test (n_pts=4000, ~20 s)
cargo test --release --features gpu --lib test_p5_7_ty_schoen_discrimination_n20 \
    -- --ignored --nocapture
```

All seeds and the bootstrap seed (12345) are deterministic on
x86_64 Linux/Windows; numbers above are reproducible to floating-point
rounding.

## Headline numbers

- σ_Schoen(k=2, n=10000, seed=42) = **1.6424**
- σ_TY(k=2, n=10000, seed=42)     = **0.3554** (P5.4)
- 20-seed ensemble: max n-σ = **4.854 at k=3** (just shy of 5σ)
- Verdict: **PARTIAL DISCRIMINATION — need n_pts/k bump or other
  channels for hard 5σ; the architecture and pipeline both work
  end-to-end on the actual physics candidates**.

## Update — 2026-04-27 — P5.10 (Path A) cleared 5σ

P5.10 ran the same harness with `n_pts = 25 000` at k=3 (same 20 seeds,
same Donaldson params, same bootstrap seed) and reached:

| k | ⟨σ_TY⟩ ± SE | ⟨σ_Schoen⟩ ± SE | **n-σ** |
|---:|---:|---:|---:|
| 3 | 1.014688 ± 5.225e-3 | 3.036530 ± 2.307e-1 | **8.761** |

**5σ DISCRIMINATION ACHIEVED ON σ ALONE — project goal met.**
Wall clock 114 s. Full report at
`references/p5_10_5sigma_target.md`; binary at
`src/bin/p5_10_ty_schoen_5sigma.rs`; data at
`output/p5_10_ty_schoen_5sigma.json`. The σ-eval logic was *not* changed
between P5.7 and P5.10 — only the sample budget. The P5.7 forward-look
prediction ("n_pts → 25 000 should push past 5σ") is therefore confirmed
empirically at n-σ = 8.761 (vs the analytic estimate of ~7.7).
