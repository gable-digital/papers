# P8.6 — Forward-Model Channel Honesty Audit (α_em / M_W / Λ_QCD)

**Status:** STOPPED at audit. Channels are NOT wired into `bayes_factor_multichannel`.
**Date:** 2026-04-30
**Verdict:** All three forward-model channels are **informational / calibration-only** at the current implementation level. They cannot supply genuine TY-vs-Schoen discrimination because their inputs are not derived from per-candidate Donaldson-converged metric or per-candidate bundle data.

## Functions audited

All three live in `src/refine.rs`:

| Function | Signature | Source lines |
|---|---|---|
| `predict_alpha_em_from_metric` | `(em_sector_norm: f64, h_spectral_max: f64) -> f64` | refine.rs:842 |
| `predict_m_w_gev_from_metric` | `(weak_sector_norm: f64, h_spectral_gap: f64) -> f64` | refine.rs:892 |
| `predict_lambda_qcd_gev_from_metric` | `(qcd_sector_norm: f64, h_spectral_max: f64) -> f64` | refine.rs:917 |

Each function takes **two f64 scalars**. None takes a `Cy3MetricResult`, an h-matrix array, a CICY geometry, a bundle object, a Wilson-loop integral, or any structure that varies meaningfully between TY/Z3 and Schoen/Z3xZ3 once seeded.

### What each prediction actually depends on

1. **`predict_alpha_em_from_metric`** — `g_sq * z_factor / (4π)` where `g_sq = em_sector_norm²` and `z_factor = h_spectral_max`. This is a 2-parameter scalar map. It has no awareness of which CY3 candidate produced the inputs; the same `(em_sector_norm, h_spectral_max)` pair from TY or Schoen returns identical α_em.

2. **`predict_m_w_gev_from_metric`** — `v_eff * weak_sector_norm * (1 + 0.05·(h_spectral_gap − 1))` with `v_eff = 246·0.6535/2 ≈ 80.4` (Higgs VEV·sin θ_W). The Higgs VEV is a calibration constant identical across candidates; only `weak_sector_norm` and `h_spectral_gap` depend on the input. Same scalar-map issue.

3. **`predict_lambda_qcd_gev_from_metric`** — `M_pl · exp(−log_ratio · modulation)` with `log_ratio = 8π² / (b₀ · qcd_sector_norm²)`, `b₀ = 7`. M_pl is a universal constant; only the two scalars vary.

The functions are honestly named "from_metric" but in practice receive **two scalar summaries**, not the metric.

### Where the input scalars come from (the load-bearing fact)

In the existing pipeline (`src/bench_pipeline.rs:376-378` and `src/bench_pipeline.rs:536-538`):

```rust
let em_sector_norm   = bundle_norm_slice(c, 5,  5);
let weak_sector_norm = bundle_norm_slice(c, 10, 5);
let qcd_sector_norm  = bundle_norm_slice(c, 15, 5);
```

`bundle_norm_slice(c, start, count)` (`bench_pipeline.rs:614-622`) is just an L₂ norm over five entries of `c.bundle_moduli`. And `c.bundle_moduli` itself is documented in `src/pipeline.rs:70-82` as:

> ```
> // LEGACY-SUPERSEDED-BY-ROUTE34: `bundle_moduli: Vec<f64>` is an
> // unstructured raw-moduli vector consumed by the legacy heterotic.rs
> // monad bundle constructor. The publication-grade replacement is:
> //   * route34::bundle_search::CandidateBundle  (structured
> //     parameterisation: line-bundle degrees, monad map data, derived
> //     Chern classes via the splitting principle, Wilson-line element
> //     ...)
> ```

So `bundle_moduli` is **a per-candidate seeded Vec<f64> placeholder**, not a quantity computed from solving the Hermitian-Yang-Mills / Donaldson PDE on the actual TY or Schoen manifold. Different seeds yield different scalars, so the *raw* prediction values would superficially differ between any two candidates — but that "difference" reflects only RNG seed entropy, not geometric content. It is the *generator* speaking, not the *manifold*.

Similarly, `h_spectral_max` and `h_spectral_gap` come from the legacy refine loop's h-matrix in `bench_pipeline.rs`, not from `Cy3MetricSolver::solve_metric` on the canonical TY-vs-Schoen Donaldson runs used in P5.10 / P7.11. The two pipelines never touch.

### Why this fails the genuineness test

A channel is genuine discrimination iff the prediction depends on per-candidate metric/bundle data **with geometric semantics**. Concretely, a bona-fide α_em forward-model would need at minimum one of:

- a Wilson-loop integral around the U(1)_Y cycle on the *actual* CY3 (TY vs Schoen),
- a Bergman-kernel-based gauge-kinetic-function integral on the converged Donaldson metric,
- a Chern-class integral against the *real* bundle (route34::bundle_search::CandidateBundle, not legacy bundle_moduli).

None of these is present. The current code is a **calibration-equation-fitter**: given two free scalars per channel, it can hit any target value. That makes the prediction informational at best (it says "if you had the bundle, this is the formula"), and definitely not a likelihood worth multiplying into a BF.

### Numerical sanity check (without re-running)

The unit tests in `refine.rs:2050+` confirm this directly: at the calibration point `em_sector_norm = sqrt(4π·α(m_Z))`, `h_max = 1.0` the predictor recovers α(m_Z) exactly. Doubling `em_sector_norm` sends predicted α to 4× measured. Both behaviors are deterministic functions of the two scalars only — no manifold ever enters.

## What it would take to make these channels genuine

To promote any of these to a real BF channel:

1. **For α_em:** integrate the gauge kinetic function Re τ = ∫_{Σ_em} ω ∧ Tr(F ∧ F) over the U(1)_Y supporting cycle Σ_em on the *converged Donaldson metric* of each candidate. That requires (a) a candidate-specific U(1)_Y cycle defined on the CICY geometry (currently absent for both TY and Schoen in `route34::bundle_search`), (b) a Bergman-kernel evaluator that consumes `Cy3MetricResult.h_matrix` plus the cycle's defining ideal. Mirror the Yukawa-spectrum harmonic-kernel pipeline (`src/route34/yukawa_*.rs`).

2. **For M_W:** compute the W zero-mode wavefunction overlap `∫ |ψ_W|² ω³` against the Higgs zero-mode `ψ_H` on each candidate's Donaldson metric. Requires a SU(2) divisor representative on each manifold.

3. **For Λ_QCD:** integrate the QCD-bundle Chern class `c₂(V_QCD)` against the candidate's actual cohomology, then exponentiate at one loop. Requires `CandidateBundle::qcd_chern_2` implemented for both Tian-Yau Z/3 and Schoen Z/3×Z/3 line-bundle SMs.

All three require infrastructure that does not exist yet. P8.6's task spec correctly anticipated the stopping condition; this audit is the honest answer.

## Action taken

- **No** binary `p8_6_em_w_qcd_channels.rs` was created.
- **No** adapters added to `bayes_factor_multichannel.rs`.
- **No** wiring in `p8_1_bayes_multichannel.rs`.
- **No** modification to `predict_*_from_metric` (read-only).
- The 5.43σ chain-channel headline (P7.11) stands as the publication-grade result; α_em / M_W / Λ_QCD are deferred until the harmonic-kernel + bundle-cycle integrators are in place.

## Cross-references

- `src/refine.rs:824-940` — forward-model definitions (audited).
- `src/bench_pipeline.rs:376-395`, `:536-550`, `:614-622` — the two existing call-sites; both pass `bundle_norm_slice(c, ...)` summaries, not metric data.
- `src/pipeline.rs:70-103` — `Candidate.bundle_moduli` documented as legacy/superseded.
- `src/route34/bundle_search.rs` — the structured replacement (`CandidateBundle`) where future genuine forward-models would source data.
- `references/cy3_publication_summary.md` — current headline numbers; unchanged by this audit.
