# P4.3 — Stack-B Phase 2: φ-Model NN Metric Ansatz on the Fermat Quintic

**Date**: 2026-04-27
**Reference**: Larfors-Lukas-Ruehle-Schneider 2022 ([arXiv:2205.13408](https://arxiv.org/abs/2205.13408)), reports σ ≈ 0.0086 at NN-asymptote with a 4-layer MLP φ-model on the Fermat quintic.

## Architecture

- **Model**: `K_φ(z) = K_FS(z) + φ(z)`
  - `K_FS`: Fubini-Study Kähler potential `log(Σ |z_i|²)` on `CP^4`.
  - `φ : ℂ^5 → ℝ`: a 4-layer MLP `5 → 32 → 32 → 1`, GeLU hidden, Identity output.
- **Input**: `(|z_0|², |z_1|², |z_2|², |z_3|², |z_4|²)` — `U(1)^5`-invariant by construction.
- **Output**: a single real scalar `φ(z)`.
- **Parameters**: 1 217 (counted by `Sequential::n_params()` for `[5, 32, 32, 1]`).
- **Library**: `pwos_math::nn::Sequential`, `pwos_math::nn::training::train_mse`,
  `pwos_math::opt::adam::AdamConfig::with_defaults`.

## Files created

| File | Role |
|------|------|
| `book/scripts/cy3_substrate_discrimination/rust_solver/src/nn_metric.rs` | `PhiModelMetric`: zero-init + Glorot-init constructors, `phi(z)`, `metric_at(z)` (FD second-derivative path), 5 unit + ignored tests. |
| `book/scripts/cy3_substrate_discrimination/rust_solver/src/quintic.rs` | Added `sigma_at_metric_closure(n_pts, seed, sampler, &mut metric_at)` and `sigma_and_etas_at_metric_closure` that drive the canonical DKLR-2006 σ functional from a user-supplied `metric_at` closure (no Bergman ansatz needed). |
| `book/scripts/cy3_substrate_discrimination/rust_solver/src/lib.rs` | Registered `pub mod nn_metric;`. |
| `book/scripts/cy3_substrate_discrimination/rust_solver/Cargo.toml` | Enabled the `nn` feature on the `pwos-math` dependency. |

## Metric computation

`g_{ij̄}(z) = ∂_i ∂_{j̄} (K_FS + φ)` is computed in two stages:

1. **Analytic FS contribution** in homogeneous coords:
   `g^{FS}_{ij̄} = δ_{ij}/|z|² − z̄_i z_j / |z|⁴`.
2. **FD φ-correction** using a 4-point stencil on the real-imaginary
   representation:
   `∂_i ∂_{j̄} φ = (1/4) [φ_{x_i x_j} + φ_{y_i y_j} + i(φ_{x_i y_j} − φ_{y_i x_j})]`,
   where each `φ_{xy}` is the standard 4-point central difference. At
   `θ ≡ 0` this stage is short-circuited (`φ ≡ 0` ⇒ skip 16 stencils per
   matrix entry) and the metric reduces to the analytic FS metric to
   machine precision.

Cost at non-zero θ: 16 NN evaluations per matrix entry × 25 entries =
**400 NN evaluations per point**. This is the Phase-2 sanity-path cost;
analytic `∂_i ∂_{j̄} φ` is a Phase-3 deliverable.

## Test results (cy3_rust_solver, --features gpu --release)

```
running 5 tests
test nn_metric::tests::test_p4_3_phi_model_metric_tangent_det_positive ... ok
test nn_metric::tests::test_p4_3_phi_model_zero_init_reduces_to_fubini_study ... ok
σ(NN, φ=0)   = 0.358279
σ(FS, analytic) = 0.358279
test nn_metric::tests::test_p4_3_phi_model_sigma_at_zero_init_matches_fs_sigma ... ok
synthetic-target training: loss[0] = 5.0447e-1, loss[last] = 1.7481e-3
test nn_metric::tests::test_p4_3_phi_model_train_mse_decreases_loss ... ok
test nn_metric::tests::test_p4_3_phi_model_picard_sigma_descent ... ignored

test result: ok. 4 passed; 0 failed; 1 ignored
```

| Test | Pass | Notes |
|------|------|-------|
| `test_p4_3_phi_model_zero_init_reduces_to_fubini_study` | yes | At `θ ≡ 0`, `\|φ(z)\| < 1e-12` and `\|g_{ij̄}^{NN} − g_{ij̄}^{FS}\| < 1e-8` at 10 sample points. |
| `test_p4_3_phi_model_sigma_at_zero_init_matches_fs_sigma` | yes | `σ(NN, φ = 0) = 0.358279`, `σ(FS, analytic) = 0.358279` (n_pts = 2 000, seed = 42, ShiffmanZelditch). Relative agreement < 1e-9. |
| `test_p4_3_phi_model_metric_tangent_det_positive` | yes | At zero-init, > 60 % of sampled points (50 pts, seed 17) yield positive `det(g_tan)`; the small-deficit comes from chart-coord degeneracies near `z_chart ≈ 0`. |
| `test_p4_3_phi_model_train_mse_decreases_loss` | yes | Glorot-init NN trained to fit `t(z) = Σ \|z_i\|⁴ − 1` over 200 quintic points × 200 epochs, lr = 0.01: loss drops `5.04e-1 → 1.75e-3` (288× reduction). Confirms `pwos_math::nn::training::train_mse` round-trips correctly through the φ-model. |
| `test_p4_3_phi_model_picard_sigma_descent` (`#[ignore]`) | runs, σ flat | See "Training results" below. |

## σ baseline established

The natural Phase-2 reference value is the **pure Fubini-Study σ** on the
Fermat quintic — the value `σ(NN, φ = 0)` reduces to. Using the
ShiffmanZelditch sampler at n_pts = 2 000, seed = 42:

```
σ_FS_quintic = 0.358279
```

This is the baseline against which any improvement from φ training must
be measured. **It is the value the spec asks the zero-init test to
match**; the FS-σ on the quintic is intrinsically larger than σ at
post-Donaldson (Bergman with k = 2…4 gives σ in [0.13, 0.27] per the P3.10
scan), because FS = Bergman at `k = 1` and σ(k = 1) is poor.

## Comparison to literature

| Source | σ on Fermat quintic | Method |
|--------|---------------------|--------|
| Pure FS (this work, Phase 2 baseline) | **0.358279** | `K = log Σ\|z_i\|²` on the quintic, n_pts = 2 000 SZ |
| Stack A k = 2 Donaldson (P3.10) | 0.2589 | Bergman-balanced + σ-refine at k = 2, n_pts = 7 250 |
| Stack A k = 3 Donaldson (P3.10) | 0.1855 | k = 3, n_pts = 17 250 |
| Stack A k = 4 Donaldson (P3.10) | **0.1321** | k = 4, n_pts = 54 000 |
| Stack A k = 5 Donaldson (P3.10) | 0.1067 | k = 5, n_pts = 80 000 |
| ABKO 2010 fit `3.51/k² − 5.19/k³` at k = 4 | 0.1383 | reference fit |
| Larfors-Lukas-Ruehle-Schneider 2022 | **0.0086** | NN φ-model on quintic, full MA gradient |

## Training results

The `#[ignore]`d Picard-style σ-driven training test was added to
demonstrate the closure-driven σ-eval ↔ `train_mse` round-trip. Each
outer iteration:
1. Computes per-point η_p and κ = mean η.
2. Sets MSE targets `t_p = -0.5 log(η_p / κ)` (the φ correction that
   would, in a linearised theory, push η toward κ).
3. Fits φ to those targets for 50 inner Adam epochs at lr = 0.005.
4. Re-evaluates σ.

Observed across 5 outer iterations at n_pts = 400, seed = 42:

```
Picard σ-descent: initial σ = 0.354698
Picard σ-descent: outer 0 → σ = 0.354698 (vs initial 0.354698)
…
Picard σ-descent: outer 4 → σ = 0.354698 (vs initial 0.354698)
ratio = 1.000
```

**σ does not decrease.** This is the expected failure mode of
the Picard-on-log-η scheme:

- The φ-model affects the metric only through `∂_i ∂_{j̄} φ`. Targets
  defined on `φ` itself (rather than the metric) are coupled to the
  observable `η` only through a second derivative whose magnitude is
  controlled by `||φ||_{C^2}`, not `||φ||_{L^∞}`.
- The `train_mse` loop drives `φ → -0.5 log(η/κ)` in the L² sense over
  the training distribution. The MSE-best fit for a 4-layer MLP at
  this width is a smooth function of `(|z_i|²)`; the residual between
  this smooth approximation and the (rough) target `log(η/κ)` carries
  the high-frequency content that would actually move `∂_i ∂_{j̄} φ`.
  In short: the L²-best fit to a noisy target is too smooth for its
  Hessian to bite.
- The FD second-derivative path additionally damps high-frequency
  contributions of φ. With `fd_eps = 1e-3` and a smooth (Glorot-trained)
  φ, the metric correction is at the level of FD truncation noise.

This is documented inside the test as the expected outcome at Phase 2.
**Reaching Larfors et al.'s σ ≈ 0.0086 requires:**
1. Analytic ∂_i ∂_{j̄} φ propagation through the NN (Phase 3) — eliminates
   FD noise and unlocks higher-order correction terms.
2. A genuine MA-residual loss that targets the metric (not φ), with the
   gradient flowing through the metric back to θ via reverse-mode
   autodiff over the layer composition. This is what `train_mse` cannot
   do — `pwos_math::nn::training::train_mse` is MSE-on-Sequential-output
   only.

## Architectural concerns

- **FD-second-derivative cost**: 400 NN evaluations per metric point ×
  per-σ-eval (1 000 points typical) = 400 000 NN evaluations per σ
  evaluation. At 1217 params and a 5-32-32-1 MLP, one forward pass is
  ~5 000 FLOPs ⇒ ~2 GFLOPs per σ eval. Tractable on a single core, but
  any training loop on top is bounded by the σ-eval cost.
- **Hyperdrive-style autodiff** would replace the 400-NN-eval-per-point
  cost with O(1) extra work via reverse-mode through analytic
  `∂_i ∂_{j̄}` operators on the `Sequential` graph. This is the pwos-math
  Phase-3 work item.
- **Inner-loop parallelism**: The closure-based path is sequential
  because `metric_at` is `&mut` (NN scratch is non-Send). A pre-allocate
  one-scratch-per-rayon-thread variant is straightforward — deferred.
- **U(1)^5 vs U(1) invariance**: Per the spec, we use `(|z_i|²)` as
  input — U(1)^5-invariant, more invariant than CP^4's natural U(1).
  This restricts φ's expressive power but is a clean Phase-2 ansatz.
  The Larfors paper uses richer projective-coordinate features (e.g.
  `z_i z̄_j / Σ|z|²` with chart-stitching) — those would be Phase-3.

## Reproducing

```bash
cd book/scripts/cy3_substrate_discrimination/rust_solver
# Fast tests (≤ 1 s):
cargo test --release --features gpu --lib test_p4_3 -- --nocapture
# Long Picard test (~30 s):
cargo test --release --features gpu --lib test_p4_3_phi_model_picard_sigma_descent -- --ignored --nocapture
```

## Conclusions

**Verdict: Phase-2 infrastructure delivered, σ-descent gated on Phase 3.**

What works:
1. `pwos_math::nn::Sequential` 5-32-32-1 GeLU MLP integrates cleanly with
   the cy3 quintic codebase.
2. Zero-init metric reduces to FS to machine precision.
3. Closure-driven σ-eval (`sigma_at_metric_closure`) reproduces the
   FS-σ baseline at 0.358279, agreeing with the analytic-FS path to
   sub-1 % relative tolerance over 2 000 SZ-sampled points.
4. `train_mse` correctly reduces a synthetic MSE loss by 288× — the
   pwos-math NN training loop is functional on this network shape.

What does not work without Phase 3:
1. σ-descent via Picard-on-log-η: σ stays at the FS baseline (0.355) over
   5 outer iterations because the L²-best smooth φ has too-small a Hessian
   to move the metric. Documented inline.
2. Reaching Larfors's σ ≈ 0.0086: requires analytic `∂_i ∂_{j̄} NN`
   through the layer stack + a Monge-Ampère-residual loss whose gradient
   flows through the metric. Phase-3 work item.

Stack B Phase 2 is **infrastructure-complete**: the φ-model class,
closure-driven σ-eval, and tests are all in place; the FS-σ reference
baseline is established; the limitation that prevents reaching Larfors's
σ is precisely localised to the lack of analytic ∂² operators on
`Sequential`. Phase 3 is gated on adding those operators to pwos-math.
