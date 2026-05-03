# §6.3 — Substrate-mass-chain matcher (P8.5 update — k=4 sweep qualitatively corroborates P7.11)

## P8.5 — k=4 GPU+orthogonalize update (2026-04-30)

Re-ran the P7.11 chain-match sweep at the next Donaldson basis order
**k=4** (P7.11 was at k=3) using the same projected/orthogonalized
machinery. Schoen unanimously wins both chains across all three
test_degree cells available; the **sign of the discrimination from
P7.11 is preserved at k=4** under a reduced-budget Donaldson-only
baseline.

| chain × deg | k=3 (P7.11) Δ | k=4 (P8.5) Δ | sign |
|:---|---:|---:|:---|
| quark × 2  |  +8.93 |  +15.33 | same (Schoen wins) |
| quark × 3  |  +6.99 |  +14.46 | same (Schoen wins) |
| quark × 4  |  +9.32 |  +10.39 | same (Schoen wins) |
| lepton × 2 | +12.78 |  +38.01 | same (Schoen wins) |
| lepton × 3 | +11.33 |  +35.68 | same (Schoen wins) |
| lepton × 4 | +15.33 |  +30.86 | same (Schoen wins) |

(Δ convention: positive = Schoen wins.)

Settings: `--k 4 --n-pts 8000 --max-iter 25 --donaldson-tol 1e-6
--adam-iters 0 --use-gpu --orthogonalize --seed 12345
--test-degree-list "2,3,4"` (Donaldson-only; Adam refinement
disabled because adam_iters>0 runs stalled in the TY metric solve
within agent wall-clock). Total wall-clock 42.6 s on GPU.

**Magnitudes are NOT directly comparable across k.** The P7.11 (k=3)
baseline ran at iter_cap=100 / n_pts=25000; the P8.5 (k=4) baseline
ran at iter_cap=25 / n_pts=8000 because larger budgets stalled in
the agent session. Both candidates hit the iter_cap rather than
reaching `donaldson_tol=1e-6`, and TY's σ_Donaldson is *larger* at
k=4 (0.285) than at k=3 (0.272) — i.e. **the metric at k=4 in this
sweep is *less* converged**, not more refined. Schoen's σ rises
asymmetrically (1.852 → 2.431, +31%) under the smaller iter_cap.
Schoen's chain-match residual is also **not converged in
test_degree** at k=4 (rel-Δ ≈ 0.42–0.47, *increasing* with td;
test_degree=5 was dropped because basis_dim ≈ 285 sits at the f64
edge). The raw-Δ values are residual_log distances, not log Bayes
factors, so re-stating them as a "stronger discrimination" claim
without running `bayes_factor_multichannel` would be a category
error.

**Verdict (qualitative)**: the k=4 sweep corroborates P7.11's sign
result — Schoen wins both chains on every td cell available — but
the *magnitude* of the discrimination at k=4 is not comparable to
k=3 under the current parameter mismatch. A production-grade re-run
(multi-hour batch job, iter_cap matched to or larger than P7.11's
100, n_pts ≥ 25000, Adam re-enabled, all five test_degrees) is
required before any cross-k strength claim. See
[`p8_5_chain_match_k4.md`](p8_5_chain_match_k4.md) for full
per-cell residuals, σ values, basis dimensions, the seven-item
caveat list, and the production-rerun parameter spec.

Chain-match remains corroborating evidence; the σ-channel
(`p5_10_ty_schoen_5sigma`) remains the headline 5σ-grade
discriminator.

---

## P7.11 — TL;DR — sign-flip pathology gone, retraction PARTIALLY LIFTED

The post-P5.5k + P-INFRA + P7.8/P7.11 re-run uses the trivial-irrep
projected basis (TY → Z/3, Schoen → Z/3 × Z/3) under
[`metric_laplacian_projected`] with **modified Gram-Schmidt
orthogonalization under the L²(M) inner product followed by a
standard Hermitian EVP on the orthonormal basis** — the same
technique P7.8 applied to the bundle Laplacian. Run at
`n_pts=25000`, `k=3`, `max_iter=100`, `donaldson_tol=1e-6`,
`seed=12345`, `test_degree ∈ {2, 3, 4, 5}`, Hungarian assignment.

| chain  | deg=2 | deg=3 | deg=4 | deg=5 | sign-agree |
|:---|---:|---:|---:|---:|:---|
| quark  | Δ=−8.93 (Schoen wins) | Δ=−6.99 (Schoen wins) | Δ=−9.32 (Schoen wins) | Δ=−11.20 (Schoen wins) | YES |
| lepton | Δ=−12.78 (Schoen wins) | Δ=−11.33 (Schoen wins) | Δ=−15.33 (Schoen wins) | Δ=−18.31 (Schoen wins) | YES |

**Sign-flip pathology of P6.3 / P7.2 is GONE.** Δ keeps the same
sign on every (chain, td) cell and on both chains. TY's residual is
stable across td=3..5 (rel-Δ ≤ 3%). Schoen's residual still drifts
(rel-Δ 12–19%, trending DOWN with td) — driven by the Schoen
Donaldson noise floor (`σ ≈ 1.85` vs TY's `σ = 0.27`), not by basis
non-convergence. No negative eigenvalues at any td (orthogonal-basis
Hermitian EVP cannot produce them).

**Verdict**: chain match is now a usable discrimination channel —
both chains unanimously favour **Schoen** across all four basis
sizes, by ≈9 nats on quark and ≈13–18 nats on lepton. The original
P6.3 headline (TY +29.0 over Schoen on quark) is RETRACTED in the
opposite sense: post-fixes the answer is Schoen wins by ~9 nats,
not TY by +29 nats. Strict 10% rel-Δ convergence on Schoen is not
yet met — the σ-channel
(`p5_10_ty_schoen_5sigma`) remains the headline 5σ-grade
discriminator; chain match is corroborating evidence.

### What changed (P7.11)

1. New `MetricLaplacianConfig::orthogonalize_first` and
   `MetricLaplacianConfig::orthogonalize_tol` flags. Default `false`
   (legacy behaviour preserved). When `true`, the projected-basis
   solver in `metric_laplacian_projected` performs modified
   Gram-Schmidt with deflation under the L²(M) inner product
   (re-using the projected mass matrix), drops the numerical null
   space, and runs a standard Hermitian EVP on the orthonormal
   basis. Mirrors P7.8's `run_orthogonalized` in
   `zero_modes_harmonic_z3xz3.rs`.
2. New struct field
   `MetricLaplacianSpectrum::orthogonalized_basis_dim` reports the
   post-orthogonalization rank (= projected-basis dim minus the
   number of L²(M) null-space directions deflated).
3. New `--orthogonalize` and `--donaldson-tol` CLI flags on
   `p6_3_chain_match_diagnostic`. When `--orthogonalize` is set,
   the binary routes through the projected basis with Z/3 (TY) or
   Z/3 × Z/3 (Schoen) filters automatically.
4. Two new regression tests
   (`p7_11_orthogonalize_drops_null_space`,
   `p7_11_orthogonalize_agrees_on_well_conditioned_diag`) that
   exercise the new code path on synthetic Hermitian-PSD pencils.
5. `--test-degree-list` accepted as an alias for `--sweep-degrees`
   to keep the CLI naming consistent with sibling P7.* binaries.
6. Sweep output path now respects `--output` (otherwise falls back
   to the legacy `output/p6_3b_basis_convergence_sweep.json`).

### Projected-basis dimensions (post-orthogonalization)

| td | TY/Z3 (Z/3-trivial) | Schoen/Z3xZ3 (Z/3×Z/3-trivial) |
|---:|---:|---:|
| 2 |  18 |  13 |
| 3 |  62 |  44 |
| 4 | 178 | 119 |
| 5 | 450 | 285 |

These are projected-basis dimensions (not full bigraded). Schoen's
Z/3 × Z/3 filter is stricter (~1/9 survival) than TY's Z/3 (~1/3),
hence the smaller Schoen dim at each td.

### Files modified / created (P7.11)

- `src/route34/metric_laplacian.rs` — added
  `MetricLaplacianConfig::{orthogonalize_first, orthogonalize_tol}`
  and `MetricLaplacianSpectrum::orthogonalized_basis_dim`. Legacy
  path now sets `orthogonalized_basis_dim = 0` (orthogonalization
  is honoured only by the projected variant; the unprojected path
  retains its M^{-1}K Galerkin EVP for back-compat).
- `src/route34/metric_laplacian_projected.rs` — added
  `run_orthogonalized_metric_laplacian` (modified Gram-Schmidt under
  L²(M), then standard Hermitian EVP on the orthonormal basis).
  `compute_projected_metric_laplacian_spectrum` branches on
  `config.orthogonalize_first`. Two new regression tests.
- `src/route34/sub_coxeter_h4_projector.rs` — populate the new
  `orthogonalized_basis_dim` field on the empty + populated
  spectrum constructions.
- `src/bin/p6_3_chain_match_diagnostic.rs` — new flags
  `--orthogonalize`, `--donaldson-tol`, `--test-degree-list`;
  `run_candidate` and the P7.2 sweep loop both honour the
  orthogonalization flag and route through the projected solver
  when set; sweep output respects `--output`.
- `output/p7_11_quark_chain.json`, `output/p7_11_lepton_chain.json`
  — full per-(td, candidate) sweep records.
- `output/p7_11_chain_match_post_fixes.json` — consolidated
  cross-chain summary used in this section.
- `output/p7_11_quark_run.log`, `output/p7_11_lepton_run.log` —
  stderr logs from each sweep.

---

## Pre-P7.11 history (PROVENANCE — kept verbatim below)

# §6.3 — Substrate-mass-chain matcher (RETRACTED + post-fix re-runs)

**Date**: 2026-04-29 (original P6.3); 2026-04-29 (P6.3b post-fix sweep);
2026-04-29 (P7.2 post-P5.5k re-run with Hungarian assignment).
**Crate**: `book/scripts/cy3_substrate_discrimination/rust_solver`
**Pipelines**:
- `route34::metric_laplacian` (Galerkin Δ_g spectrum on bigraded test
  monomials)
- `route34::chain_matcher` (greedy NN + P7.2 Hungarian optimal assignment
  to Coxeter / D_8 chain positions, plus consecutive-ratio pattern
  analysis)

**Binary**: `src/bin/p6_3_chain_match_diagnostic.rs`
**Data**:
- `output/p6_3_chain_match_diagnostic.json` (PRE-FIX, retracted)
- `output/p6_3b_basis_convergence_sweep.json` (P6.3b sweep)
- `output/p7_2_chain_sweep.json` (P7.2 post-P5.5k + Hungarian sweep)

## TL;DR — retraction RETAINED post-P5.5k + Hungarian

The published P6.3 headline (TY +29.0 over Schoen on the quark chain)
remains **RETRACTED**. The P7.2 re-run with the corrected P5.5k
Donaldson metric, Hungarian optimal assignment in log-eigenvalue
space, and a basis-size sweep at canonical settings (n_pts=40k, k=3,
iter_cap=100, seed=42) reproduces the basis-size pathology
diagnosed by P6.3b:

| chain  | deg=3 | deg=4 | deg=5 | sign-agree | stable |
|:---|---:|---:|---:|:---|:---|
| quark  | Δ=−7.26 (Schoen wins) | Δ=+9.70 (TY wins) | Δ=+8.64 (TY wins) | NO | NO |
| lepton | Δ=−12.58 (Schoen wins) | Δ=+17.58 (TY wins) | Δ=+17.54 (TY wins) | NO | NO |

- **Sign of Δ flips between deg=3 and deg=4** for both chains.
- TY's residual moves 17.5 → 5.6 → 2.2 (no consecutive pair within
  10%). Schoen oscillates 10.2 → 15.3 → 10.9 (quark) and 20.7 →
  30.3 → 19.4 (lepton).
- **Galerkin condition log10 reaches 15 (TY) and 17 (Schoen) at
  deg=5** — Schoen's M^{−1}K is past f64 machine precision
  (1e−16 ≈ epsilon). The deg=5 numbers are corrupted by
  ill-conditioning, not refined estimates.
- Hungarian (P7.2) and greedy (P6.3) match exactly on the chain-anchor
  path because the cost is monotone-1D once λ_min is pinned to
  slot 0; the P7.2 fix removes a class of suboptimality but not the
  underlying basis-size non-convergence.

**There is no converged chain-match discrimination on either chain.**
The substrate-mass-chain channel cannot currently support either
direction of TY-vs-Schoen claim from this codebase as configured.

## P7.2 — what changed

1. **Hungarian (optimal) assignment**. Replaces greedy nearest-
   neighbour with O(n³) Hungarian / Jonker-Volgenant in log-
   eigenvalue space. Cost matrix `C[j][i] = (log(λ_i) − k_j · log(h))²`
   over predicted chain positions × eigenvalues. Returns the absolute-
   log residual at the optimal assignment (squared optimum and |·|
   optimum coincide modulo ties). Implementation:
   `chain_matcher::hungarian_assign` (square-padded standard
   Hungarian; ≤ 1 ms at our sizes) and `chain_matcher::match_chain_hungarian`.
2. **Regression tests**:
   - `hungarian_beats_greedy_on_adversarial_costs`: 3×3 matrix where
     greedy-by-row picks total 103 and Hungarian gives total 5.
     Demonstrates the algorithmic improvement directly.
   - `hungarian_assign_3x3`: explicit minimum verified on a hand-
     worked 3×3.
   - `hungarian_never_regresses_vs_greedy_on_chain`: Hungarian
     residual ≤ greedy residual on a perfect chain plus decoys
     (parity check; both should be ~0).
   - `hungarian_matches_greedy_on_perfect_chain`: parity on the
     non-pathological case.
   - `ratio_pattern_clean_sqrt2_chain`: ratio analyser returns the
     expected (sqrt2, Δk=1) match on a clean chain.
   All 8 chain_matcher tests pass after the change. Pre-fix: the
   adversarial-cost test would not exist; the hungarian symbol would
   not resolve.
3. **Eigenvalue-ratio pattern analyser**. New
   `chain_matcher::ratio_pattern(eigvals, n_floor, n_top)` returns
   the top-N largest consecutive eigenvalue ratios in the lowest
   `n_floor` positive eigenvalues, each tagged with its closest
   φ^Δk or (√2)^Δk (Δk in {1,2,…,7} for φ; ditto plus halves for
   √2) and the relative residual.
4. **Shared metric/spectrum across chains**. The P7.2 sweep solves
   the metric ONCE per candidate and the Galerkin spectrum ONCE per
   (candidate, test_degree), then runs both quark and lepton chain
   matches against the shared spectrum. Eliminates 4× redundant
   metric solves and 2× redundant spectrum solves vs the P6.3b
   structure; total wallclock ≈ 19 min for both candidates ×
   {3,4,5} × both chains at canonical settings.
5. **`--p7-2` and `--assign hungarian|greedy` flags** on the
   diagnostic binary. The legacy single-chain sweep
   (`p6_3b_basis_convergence_sweep.json`) is preserved.

## P7.2 sweep results (n_pts=40k, k=3, iter_cap=100, seed=42)

### Per-(candidate, chain, test_degree) residuals and condition

| candidate | chain  | deg | basis_dim | residual | smallest λ | largest λ | cond log10 |
|:---|:---|---:|---:|---:|---:|---:|---:|
| TY/Z3        | quark  | 3 |  164 | 17.49 | 2.39e−1  | 6.20e9 | 10.4 |
| TY/Z3        | quark  | 4 |  494 |  5.57 | 2.75e−3  | 5.65e9 | 12.3 |
| TY/Z3        | quark  | 5 | 1286 |  2.21 | 5.35e−6  | 6.44e9 | 15.1 |
| TY/Z3        | lepton | 3 |  164 | 33.24 | (same)   | (same) | (same) |
| TY/Z3        | lepton | 4 |  494 | 12.68 | (same)   | (same) | (same) |
| TY/Z3        | lepton | 5 | 1286 |  1.87 | (same)   | (same) | (same) |
| Schoen/Z3xZ3 | quark  | 3 |  164 | 10.23 | 3.17e−1  | 1.91e9 |  9.8 |
| Schoen/Z3xZ3 | quark  | 4 |  494 | 15.28 | 4.32e−1  | 4.03e9 | 10.0 |
| Schoen/Z3xZ3 | quark  | 5 | 1286 | 10.85 | 3.94e−8  | 6.83e9 | **17.2** |
| Schoen/Z3xZ3 | lepton | 3 |  164 | 20.66 | (same)   | (same) | (same) |
| Schoen/Z3xZ3 | lepton | 4 |  494 | 30.26 | (same)   | (same) | (same) |
| Schoen/Z3xZ3 | lepton | 5 | 1286 | 19.40 | (same)   | (same) | (same) |

Schoen's smallest positive eigenvalue at d=5 (3.94e−8) sits below
f64 machine epsilon × largest (1e−16 × 7e9 ≈ 7e−7); the Galerkin
pencil is fully ill-conditioned. TY at d=5 is borderline (5.4e−6 vs
6.4e−7 noise floor — about a factor 8 above noise, but condition
log10=15.1 is itself at the edge of f64 precision).

### Stability between consecutive degrees

| chain  | step  | TY rel-Δ | Schoen rel-Δ | Δ rel-Δ | sign-kept |
|:---|:---|---:|---:|---:|:---|
| quark  | 3 → 4 | 0.681 | 0.493 | 2.34 | **NO** |
| quark  | 4 → 5 | 0.603 | 0.290 | 0.110 | yes |
| lepton | 3 → 4 | 0.619 | 0.465 | 2.40 | **NO** |
| lepton | 4 → 5 | 0.853 | 0.359 | 0.002 | yes |

Sign flips between deg=3 and deg=4 on both chains, and no consecutive
pair has every metric within 10%. **No degree pair satisfies the
convergence criterion on either chain.**

### degree=6 not attempted

`test_degree=5` exceeded the 5-min/point time cap (TY 482s, Schoen
554s), so the binary correctly skipped degree=6. At basis_dim ≈ 3003,
deg=6 would have ~13× the deg=5 cost (~6500s/candidate) AND the
Galerkin condition would push further past f64 precision —
running it would not yield a meaningful number. This matches the
P6.3b assessment: the Galerkin-on-bigraded-monomials scheme as
written cannot converge at any reasonable cost.

## P7.2 §4 — eigenvalue-ratio pattern at deg=4

For each (candidate, chain), the top-5 consecutive eigenvalue ratios
in the lowest 30 positive eigenvalues at test_degree=4, each tagged
with closest φ^Δk or (√2)^Δk:

### TY/Z3 (same ratios for both chains; ratios are spectrum properties)

| idx | λ_lo | λ_hi | ratio | closest | rel_res |
|---:|---:|---:|---:|:---|---:|
| 1 | 1.45e−2 | 1.97e−1 | 13.5930 | (√2)^7 = 11.3137 | +20.1% |
| 0 | 2.75e−3 | 1.42e−2 |  5.1552 | (√2)^4.5 = 4.7568 | +8.4% |
| 4 | 4.21e−1 | 6.94e−1 |  1.6477 | φ^1 = 1.6180 | +1.8% |
| 3 | 4.43e−1 | 6.76e−1 |  1.5255 | φ^1 = 1.6180 | −5.7% |
| 6 | 8.66e−1 | 1.17e0  |  1.3482 | (√2)^1 = 1.4142 | −4.7% |

The two best matches are at φ^1 (1.8% and 5.7% off φ ≈ 1.618). The
biggest gap (13.6 between λ_2 and λ_3) is 20% off (√2)^7 — not a
clean match. Three of the five ratios match within 6% of either φ^1
or (√2)^{1, 4.5}.

### Schoen/Z3xZ3

| idx | λ_lo | λ_hi | ratio | closest | rel_res |
|---:|---:|---:|---:|:---|---:|
| 1 | 1.61e−1 | 3.66e−1 | 1.8797 | (√2)^2 = 2.0000 | −6.0% |
| 2 | 4.13e−1 | 7.19e−1 | 1.6881 | (√2)^1.5 = 1.6818 | +0.4% |
| 0 | 1.47e−1 | 1.61e−1 | 1.1703 | (√2)^0.5 = 1.1892 | −1.6% |
| 8 | 1.18e0  | 1.52e0  | 1.1427 | (√2)^0.5 = 1.1892 | −3.9% |
| 6 | 9.28e−1 | 1.14e0  | 1.1382 | (√2)^0.5 = 1.1892 | −4.3% |

**Schoen's ratios cluster MUCH more tightly on the (√2)-half-integer
scale than TY's**: best match (√2)^1.5 within 0.4%, three matches
within 4% of (√2)^0.5. None match φ powers in the top-5 — Schoen's
gap structure is dominantly (√2)-typed.

This is the opposite of what the chain-match residual reports (TY
"wins" the chain-match at deg=4). The chain-match residual is
dominated by which eigenvalues are routed to the chain extrema (k=22.5,
26, 36.5 for the quark chain — log distances of 6.4, 7.6, 11.3 ·
log(√2)), where TY's spectrum happens to span a similar log range
to the chain. The local gap structure tells a different story: Schoen
exhibits cleaner (√2)^Δk gap clustering than TY at deg=4.

This corroborates round-1's hostile review: chain-match residual is
the wrong comparator for chain-position cluster structure. The gap-
based comparator is closer to the right test, but neither the
residual nor the gap pattern is converged in basis size.

## Verdict

- **Original P6.3 headline** ("TY 7.26 vs Schoen 36.29; +29.0") —
  RETAINED RETRACTION.
- **Post-P5.5k Donaldson + Hungarian re-run**: NOT CONVERGED.
  Sign of discrimination Δ flips between deg=3 and deg=4 on BOTH
  the quark and lepton chains; no consecutive degree pair satisfies
  the 10% rule.
- **deg=5 corrupted by Galerkin ill-conditioning**: cond log10=15.1
  on TY (edge of f64 precision) and 17.2 on Schoen (well past
  precision). The deg=5 number is a numerical artifact, not a
  converged value.
- **Schoen's per-degree gap structure clusters (√2)^Δk** more
  tightly than TY's at deg=4 (best match within 0.4%), but the
  chain-match residual on the chain-extrema-driven comparator says
  the opposite. The two comparators disagree, which is itself a
  sign that neither is converged.

**The chain-match channel is not a usable discriminator at canonical
post-P5.5k settings.** Sign instability between basis sizes blocks
either direction of TY-vs-Schoen claim. Discrimination claims should
be sourced from `p5_10_ty_schoen_5sigma` (σ-channel) for now.

## Files modified / created (P7.2)

- `src/route34/chain_matcher.rs` — added `hungarian_assign`,
  `match_chain_hungarian`, `ratio_pattern`, `RatioMatch`; new
  regression tests `hungarian_beats_greedy_on_adversarial_costs`,
  `hungarian_assign_3x3`, `hungarian_never_regresses_vs_greedy_on_chain`,
  `hungarian_matches_greedy_on_perfect_chain`, `ratio_pattern_clean_sqrt2_chain`.
- `src/bin/p6_3_chain_match_diagnostic.rs` — added `--p7-2`,
  `--assign`, `--ratio-floor`, `--ratio-top` flags; shared-metric/
  shared-spectrum loop; condition-log10 + smallest/largest eigvals
  in JSON; per-(candidate, chain, deg) row with embedded ratio
  pattern at deg=4.
- `references/p6_3_chain_match.md` — this file (P7.2 update).
- `output/p7_2_chain_sweep.json` — P7.2 sweep data.
- `output/p7_2_run.log` — P7.2 stderr log.

## Files unchanged from P6.3b

- `src/route34/metric_laplacian.rs` — constant-mode exclusion fix
  carried forward; Galerkin pencil construction unchanged. Spectrum
  result struct gained `basis_exponents` and `eigenvectors_full`
  optional fields between P6.3b and P7.2 (unrelated change), which
  the P7.2 binary populates as `None`.
- Bug-3 Schoen Donaldson noise floor: at iter_cap=100, Schoen's
  σ landed at 8.20 — still well above TY's 0.27. The Donaldson noise
  on Schoen remains the underlying issue limiting the chain-match
  precision regardless of basis choice.
