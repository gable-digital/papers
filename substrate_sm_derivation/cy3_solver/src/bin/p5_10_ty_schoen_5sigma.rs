//! P5.10 — push the TY-vs-Schoen σ-discrimination past 5σ.
//!
//! P5.7 reached **n-σ = 4.854 at k=3** with `n_pts = 10_000`, Donaldson-only.
//! That's ~3% short of the 5σ project goal. The σ-eval logic is correct;
//! the bottleneck is purely statistical — Schoen's σ distribution at low
//! k has heavy tails (small invariant basis: n_basis=12 at k=2, n_basis=27
//! at k=3), so SE_Schoen ≈ 0.48 dominates the combined SE.
//!
//! Three paths to 5σ were considered (see task brief):
//!   * **Path A** — bump n_pts at k=3 from 10 000 → 25 000 (this binary's
//!     default). SE scales roughly as 1/√n_pts, so SE_Schoen should drop
//!     ≈ √2.5 ≈ 1.58×, pushing n-σ from 4.85 → ~7.7σ. Cheapest path.
//!   * Path B — k=4 with default n_pts. Bigger basis (Schoen at k=4 has
//!     n_basis ~67; TY ~189). Tightens Schoen's tail by giving Donaldson
//!     more freedom. Per-run cost 5–10× larger than k=3.
//!   * Path C — Adam refinement post-Donaldson. Cheapest if wired, but the
//!     CY3 pipelines don't have Adam wiring on the candidates yet (only the
//!     Fermat quintic does). Skipped here for fairness.
//!
//! This binary runs **Path A** by default. Same 20-seed list as P5.7
//! (matches `SEEDS_20` in `p5_3_multi_seed_ensemble.rs`) for direct
//! per-seed cross-comparability. All knobs (n_pts, ks, donaldson params,
//! bootstrap config) are CLI-overridable.
//!
//! σ-eval logic is **not** modified. We only change the sampling budget.
//!
//! Output:
//!   * Stdout: per-(candidate, k) tables + n-σ summary + 5σ verdict.
//!   * JSON file at `--output`
//!     (default `output/p5_10_ty_schoen_5sigma.json`).
//!
//! Usage:
//! ```text
//!   cargo run --release --features gpu --bin p5_10_ty_schoen_5sigma
//!   cargo run --release --features gpu --bin p5_10_ty_schoen_5sigma -- \
//!       --n-pts 40000 --ks 3 --output output/p5_10_aggressive.json
//! ```

use clap::Parser;
use cy3_rust_solver::route34::cy3_metric_unified::{
    Cy3AdamOverride, Cy3MetricResultKind, Cy3MetricSolver, Cy3MetricSpec, SchoenSolver,
    TianYauSolver,
};
use cy3_rust_solver::route34::schoen_metric::AdamRefineConfig;
use cy3_rust_solver::route34::repro::{
    PerSeedEvent, ReplogEvent, ReplogWriter, ReproManifest,
};
use pwos_math::stats::bootstrap::{Bootstrap, BootstrapConfig};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// 20-seed ensemble — same list as P5.3's `SEEDS_20` and P5.7's
/// `SEEDS_20`, kept identical so per-seed σ values are directly
/// cross-comparable across the project.
const SEEDS_20: [u64; 20] = [
    42, 100, 12345, 7, 99, 1, 2, 3, 4, 5, 137, 271, 314, 666, 1000, 2024, 4242, 0xDEAD, 0xBEEF,
    0xCAFE,
];

#[derive(Parser, Debug)]
#[command(about = "P5.10 TY-vs-Schoen 5σ-target ensemble (Path A: n_pts boost)")]
struct Cli {
    /// Number of CY3 sample points per seed. Default 25 000 → ~1.58×
    /// SE shrink vs P5.7's 10 000.
    #[arg(long, default_value_t = 25_000)]
    n_pts: usize,

    /// Comma-separated list of k values to scan (default: 3 — the regime
    /// where P5.7 already cleared 4.85σ; bumping n_pts here is the
    /// fastest known route past 5σ).
    #[arg(long, default_value = "3")]
    ks: String,

    /// Donaldson max iterations. Raised from P5.7's 25 to 50 in P5.5f
    /// after the regression / singular-T(G) guards were added — the
    /// guards make the tighter tolerance below safe. Round-6 (P5.5k):
    /// raised from 50 to 100. At canonical n_pts=40k, 5 Schoen seeds
    /// (7, 314, 1000, 2024, 57005) hit the 50-iter cap with residuals
    /// ~3e-5..3e-4 — above tol=1e-6 so they're excluded from the strict-
    /// converged tier even though no-guard trajectories show they
    /// converge cleanly when given more iters. cap=100 lets these seeds
    /// reach tol; the wallclock impact is bounded because the vast
    /// majority of seeds finish well below 50 iters.
    #[arg(long, default_value_t = 100)]
    donaldson_iters: usize,

    /// Donaldson tolerance. Tightened from P5.7's 1e-3 to 1e-6 in
    /// P5.5f. At the prior 1e-3 default the iteration stopped before
    /// numerical instability fired on Schoen seed 271; at 1e-6 the
    /// regression-guard catches it explicitly (see P5.5d).
    #[arg(long, default_value_t = 1.0e-6)]
    donaldson_tol: f64,

    /// Bootstrap n_resamples. Round-4 hostile review (Defect C) showed
    /// that at B=1000 the CI lower bound jitters ±0.14σ across boot_seed
    /// choices; bumping to B=10000 cuts that to ±0.033σ. The bootstrap
    /// loop is fast (~seconds) so the wallclock cost is negligible vs
    /// the per-seed Donaldson eval.
    #[arg(long, default_value_t = 10000)]
    boot_resamples: usize,

    /// Bootstrap seed (matches P5.7 for cross-comparability).
    #[arg(long, default_value_t = 12345)]
    boot_seed: u64,

    /// If true, also report the CI lower bound for boot_seed ∈
    /// {boot_seed, 999, 31415} as a stability check (round-4 Defect C).
    /// Default true because the cost is trivial (3× a few-seconds
    /// bootstrap loop).
    #[arg(long, default_value_t = true)]
    report_boot_seed_jitter: bool,

    /// Output JSON path.
    #[arg(long, default_value = "output/p5_10_ty_schoen_5sigma.json")]
    output: PathBuf,

    /// P-DONALDSON-GPU: route the Donaldson T-operator through the
    /// CUDA kernel (commit 67f2b8c5). Requires the `gpu` feature; falls
    /// back to CPU if GPU init fails. ~140-300× speedup at parity 1e-12
    /// vs the CPU loop, which is what makes k=4 feasible inside a
    /// ~40-min production wallclock budget.
    #[arg(long, default_value_t = false)]
    use_gpu: bool,

    /// P7.9 — Adam σ-functional refinement iterations to run AFTER
    /// Donaldson balancing. 0 disables Adam (fair-comparison default,
    /// matches P5.7). P7.10 found 50 Adam iters reduces σ by ~22% on
    /// Schoen at k=3.
    #[arg(long, default_value_t = 0)]
    adam_iters: usize,

    /// P7.10 — Adam learning rate. 1e-3 is the standard.
    #[arg(long, default_value_t = 1.0e-3)]
    adam_lr: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct PerSeedRecord {
    seed: u64,
    candidate: String,
    k: u32,
    elapsed_s: f64,
    n_basis: usize,
    iterations_run: usize,
    sigma_final: f64,
    /// σ at the FS-Gram identity (`h = I`) before any Donaldson
    /// iteration — sourced from `SchoenMetricResult::sigma_fs_identity`
    /// (or `TyMetricResult::sigma_fs_identity`, both added in P5.5c).
    /// Used by the round-2 hostile-review re-analysis (Concern B) to
    /// identify seeds that early-bailed via the P5.5f regression guard
    /// and returned a near-FS-identity snapshot rather than a balanced
    /// metric. Seeds with `iterations_run < 10` and
    /// `|sigma_final - sigma_fs_identity| / sigma_fs_identity` within a
    /// few percent are flagged as "balanced-only" exclusions.
    #[serde(default)]
    sigma_fs_identity: f64,
    /// Donaldson balance-residual at the snapshot returned by the
    /// solver. For seeds that ran to `donaldson_tol`, this is < tol.
    /// For seeds that bailed via the P5.5f regression guard, this is
    /// the residual at the iter-min snapshot — and round-3 hostile
    /// review found this residual is mid-descent (typically 1e-2 to 1).
    /// We use it (NOT `iterations_run`) as the primary classifier:
    /// only seeds with `final_donaldson_residual < CONVERGED_RESIDUAL_THRESHOLD`
    /// are treated as Donaldson-balanced for the canonical n-σ.
    #[serde(default)]
    final_donaldson_residual: f64,
    /// Convergence tier:
    ///  * "converged"      — donaldson residual < 1e-3 (1000× tol).
    ///  * "ambiguous"      — residual in [1e-3, 1e-2].
    ///  * "non-converged"  — residual ≥ 1e-2 OR non-finite.
    /// Set by `classify_tier`.
    #[serde(default)]
    convergence_tier: String,
    /// P8.4 — σ AFTER Donaldson balancing but BEFORE Adam refinement.
    /// Equals `sigma_final` when `adam_iters == 0`.
    #[serde(default)]
    sigma_after_donaldson: f64,
    /// P8.4 — σ AFTER the optional P7.9 Adam σ-functional refinement.
    /// NaN when no Adam was run.
    #[serde(default)]
    sigma_after_adam: f64,
    /// P8.4 — number of Adam iterations actually executed.
    #[serde(default)]
    adam_iters_run: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct CheckpointStats {
    name: String,
    n: usize,
    mean: f64,
    std: f64,
    stderr: f64,
    percentile_ci_low: f64,
    percentile_ci_high: f64,
    bca_ci_low: Option<f64>,
    bca_ci_high: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct CandidateKEnsemble {
    candidate: String,
    k: u32,
    n_pts: usize,
    seeds: Vec<u64>,
    per_seed: Vec<PerSeedRecord>,
    sigma_stats: CheckpointStats,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct DiscriminationRow {
    k: u32,
    mean_ty: f64,
    se_ty: f64,
    n_ty: usize,
    mean_schoen: f64,
    se_schoen: f64,
    n_schoen: usize,
    delta_sigma: f64,
    se_combined: f64,
    n_sigma: f64,
    /// Bootstrap 95% CI on n-σ (paired resample of TY and Schoen
    /// σ-vectors, recomputing n-σ on each replicate). `None` when the
    /// subset is degenerate (n < 2 on either side). Percentile method
    /// (2.5%/97.5% quantiles of resample n-σ vector).
    #[serde(default)]
    n_sigma_ci_low: Option<f64>,
    #[serde(default)]
    n_sigma_ci_high: Option<f64>,
    /// Round-6 (P5.5k): BCa (bias-corrected accelerated) 95% CI on n-σ
    /// using Efron 1987 with paired-statistic jackknife. Tighter on the
    /// lower bound than percentile when the resample distribution is
    /// skewed. Reported alongside percentile for referee anticipation.
    /// `None` when bias-correction is undefined (no resamples below the
    /// observed value, or resamples are all equal) or the jackknife
    /// degenerates (acceleration denominator vanishes).
    #[serde(default)]
    n_sigma_bca_ci_low: Option<f64>,
    #[serde(default)]
    n_sigma_bca_ci_high: Option<f64>,
    /// Number of bootstrap resamples used (0 if CI not computed).
    #[serde(default)]
    n_sigma_boot_resamples: usize,
    /// Subset label: "conservative" (full ensemble), "canonical"
    /// (residual < 1e-3), "reference" (Tukey-trimmed converged).
    #[serde(default)]
    subset_label: String,
}

/// Per-seed bail / balance classification used by Concern-B re-analysis.
/// `early_bail = iterations_run < BALANCED_MIN_ITERS`; the round-2
/// hostile-review threshold (`BALANCED_MIN_ITERS=10`) classifies any
/// seed that exited via the P5.5f regression guard before iter 10 as a
/// near-FS-identity snapshot rather than a converged balanced metric.
#[derive(Debug, Serialize, Deserialize, Clone)]
struct EarlyBailFlag {
    seed: u64,
    candidate: String,
    iterations_run: usize,
    sigma_final: f64,
    sigma_fs_identity: f64,
    /// `(sigma_final - sigma_fs_identity) / sigma_fs_identity` —
    /// negative if the guard restored a near-FS state and the iteration
    /// barely budged.
    rel_diff_to_fs: f64,
}

const BALANCED_MIN_ITERS: usize = 10;

// ---------------------------------------------------------------------------
// Round-3 hostile-review (Apr 2026): residual-based classification.
//
// The previous `BALANCED_MIN_ITERS` rule (iters >= 10) is iter-based, but
// the round-3 review showed that 6 Schoen seeds bail at iters 2-9 via the
// regression guard with snapshot residuals 0.0045 .. 0.485 — none of them
// below 1e-3. Including them in a comparison against TY (which IS
// Donaldson-balanced to residual < tol) is methodologically wrong.
//
// CONVERGED_RESIDUAL_THRESHOLD is set to 1000× the Donaldson tolerance
// (1e-6 → 1e-3). A seed is "converged" if its final donaldson residual
// is below this threshold; "ambiguous" in [1e-3, 1e-2]; otherwise
// "non-converged" / early-bail-suspect.
// ---------------------------------------------------------------------------
const CONVERGED_RESIDUAL_THRESHOLD: f64 = 1.0e-3;
const AMBIGUOUS_RESIDUAL_THRESHOLD: f64 = 1.0e-2;

fn classify_tier(residual: f64) -> &'static str {
    if !residual.is_finite() {
        return "non-converged";
    }
    if residual < CONVERGED_RESIDUAL_THRESHOLD {
        "converged"
    } else if residual < AMBIGUOUS_RESIDUAL_THRESHOLD {
        "ambiguous"
    } else {
        "non-converged"
    }
}

/// Tukey 1.5×IQR outlier filter. Returns the indices kept after dropping
/// any point outside [Q1 - 1.5·IQR, Q3 + 1.5·IQR]. Empty inputs and
/// inputs of length < 4 return all indices unchanged (IQR is not
/// well-defined on tiny samples; we'd rather keep all data than throw it
/// away).
fn tukey_keep_mask(values: &[f64]) -> Vec<bool> {
    let n = values.len();
    if n < 4 {
        return vec![true; n];
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q1_idx = (n as f64 * 0.25) as usize;
    let q3_idx = (n as f64 * 0.75) as usize;
    let q1 = sorted[q1_idx.min(n - 1)];
    let q3 = sorted[q3_idx.min(n - 1)];
    let iqr = q3 - q1;
    let lo = q1 - 1.5 * iqr;
    let hi = q3 + 1.5 * iqr;
    values.iter().map(|&v| v >= lo && v <= hi).collect()
}

/// In-place Tukey filter — applies `tukey_keep_mask` to each input and
/// returns the kept values as a new pair of vectors. Empty / very-small
/// inputs are passed through unchanged.
fn tukey_trim_pair(ty: &[f64], sc: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let ty_keep = tukey_keep_mask(ty);
    let sc_keep = tukey_keep_mask(sc);
    let ty_kept: Vec<f64> = ty
        .iter()
        .zip(&ty_keep)
        .filter_map(|(v, k)| if *k { Some(*v) } else { None })
        .collect();
    let sc_kept: Vec<f64> = sc
        .iter()
        .zip(&sc_keep)
        .filter_map(|(v, k)| if *k { Some(*v) } else { None })
        .collect();
    (ty_kept, sc_kept)
}

/// Compute n-σ point statistics from raw samples (no trim).
fn compute_n_sigma(ty: &[f64], sc: &[f64]) -> Option<f64> {
    if ty.len() < 2 || sc.len() < 2 {
        return None;
    }
    let n_ty = ty.len() as f64;
    let n_sc = sc.len() as f64;
    let sum_ty: f64 = ty.iter().sum();
    let sum2_ty: f64 = ty.iter().map(|v| v * v).sum();
    let mean_ty = sum_ty / n_ty;
    let var_ty = (sum2_ty - sum_ty * mean_ty) / (n_ty - 1.0);
    let se_ty = (var_ty.max(0.0) / n_ty).sqrt();
    let sum_sc: f64 = sc.iter().sum();
    let sum2_sc: f64 = sc.iter().map(|v| v * v).sum();
    let mean_sc = sum_sc / n_sc;
    let var_sc = (sum2_sc - sum_sc * mean_sc) / (n_sc - 1.0);
    let se_sc = (var_sc.max(0.0) / n_sc).sqrt();
    let delta = mean_ty - mean_sc;
    let se_comb = (se_ty * se_ty + se_sc * se_sc).sqrt();
    if se_comb > 0.0 {
        Some(delta.abs() / se_comb)
    } else {
        None
    }
}

/// Standard-normal CDF Φ via complementary error function.
/// Φ(z) = 0.5 × erfc(-z / √2).
#[inline]
fn phi(z: f64) -> f64 {
    const INV_SQRT2: f64 = 0.707_106_781_186_547_5;
    0.5 * libm::erfc(-z * INV_SQRT2)
}

/// Inverse standard-normal CDF Φ⁻¹. Reuses the Acklam-2003 implementation
/// in `pwos_math::stats::chi2::p_value_to_n_sigma` (which computes the
/// upper-tail equivalent: p_value_to_n_sigma(p_upper) = Φ⁻¹(1 - p_upper)).
/// So Φ⁻¹(p) = p_value_to_n_sigma(1 - p).
#[inline]
fn inv_phi(p: f64) -> f64 {
    pwos_math::stats::chi2::p_value_to_n_sigma(1.0 - p)
}

/// Bootstrap CIs on the paired n-σ statistic.
///
/// Returns `(percentile_lo, percentile_hi, bca_lo, bca_hi)` where the
/// BCa pair is `Some` only when the bias-correction (z₀) and acceleration
/// (a) are both well-defined.
///
/// Resamples TY and Schoen independently (with replacement) `n_resamples`
/// times, recomputes Δσ / SE_combined on each replicate. Uses an inline
/// xorshift64* RNG seeded from `boot_seed` for determinism.
///
/// If `per_resample_trim = true`, the Tukey 1.5×IQR filter is re-applied
/// to each resampled (TY, Schoen) pair before computing n-σ. This is
/// methodologically required for the Tier 3 (Tukey-trimmed) row, where
/// the trim is part of the estimator: a baked-once trim followed by a
/// raw bootstrap (the previous behaviour) gives an artifactually narrow
/// CI because the resampled data inherits the post-trim variance, not
/// the data-generating-process variance. Round-4 hostile-review Defect B.
///
/// BCa (round-6, P5.5k): standard Efron 1987 formulation on the paired
/// statistic θ = |⟨σ_TY⟩ − ⟨σ_Schoen⟩| / SE_combined.
///   * Bias correction z₀ = Φ⁻¹(#{θ_b < θ̂} / B).
///   * Acceleration a via paired jackknife: drop one TY observation OR
///     one Schoen observation, recompute θ; combine across the union.
///   * Adjusted percentiles α₁ = Φ(z₀ + (z₀ + Φ⁻¹(α/2)) / (1 - a(z₀+Φ⁻¹(α/2))));
///     α₂ analogous with Φ⁻¹(1-α/2). BCa CI = empirical quantiles at
///     (α₁, α₂) of the sorted resample n-σ vector.
fn bootstrap_n_sigma_ci(
    ty: &[f64],
    sc: &[f64],
    n_resamples: usize,
    boot_seed: u64,
    per_resample_trim: bool,
) -> Option<(f64, f64, Option<f64>, Option<f64>)> {
    if ty.len() < 2 || sc.len() < 2 || n_resamples == 0 {
        return None;
    }
    // xorshift64* (Vigna 2014 §3.1) — small, deterministic, independent
    // of the global Bootstrap workspace so paired resampling is clean.
    let mut state: u64 = if boot_seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { boot_seed };
    let mut next_u64 = || -> u64 {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let n_ty = ty.len();
    let n_sc = sc.len();
    let mut nsigmas: Vec<f64> = Vec::with_capacity(n_resamples);
    let mut ty_resamp: Vec<f64> = Vec::with_capacity(n_ty);
    let mut sc_resamp: Vec<f64> = Vec::with_capacity(n_sc);
    for _ in 0..n_resamples {
        // Resample TY (with replacement).
        ty_resamp.clear();
        for _ in 0..n_ty {
            let idx = (next_u64() as usize) % n_ty;
            ty_resamp.push(ty[idx]);
        }
        // Resample Schoen (with replacement).
        sc_resamp.clear();
        for _ in 0..n_sc {
            let idx = (next_u64() as usize) % n_sc;
            sc_resamp.push(sc[idx]);
        }
        // If the estimator includes a Tukey trim, re-apply it to the
        // resample (round-4 Defect B). Otherwise use the raw resample.
        let nsig_opt = if per_resample_trim {
            let (ty_t, sc_t) = tukey_trim_pair(&ty_resamp, &sc_resamp);
            compute_n_sigma(&ty_t, &sc_t)
        } else {
            compute_n_sigma(&ty_resamp, &sc_resamp)
        };
        if let Some(nsig) = nsig_opt {
            if nsig.is_finite() {
                nsigmas.push(nsig);
            }
        }
    }
    if nsigmas.len() < 2 {
        return None;
    }
    nsigmas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo_idx = ((nsigmas.len() as f64) * 0.025) as usize;
    let hi_idx =
        (((nsigmas.len() as f64) * 0.975) as usize).min(nsigmas.len() - 1);
    let pct_lo = nsigmas[lo_idx];
    let pct_hi = nsigmas[hi_idx];

    // BCa: bias correction z₀ + acceleration a via paired jackknife.
    // For BCa we need the observed point estimate computed on the same
    // estimator (with Tukey trim applied if `per_resample_trim`).
    let (ty_obs, sc_obs) = if per_resample_trim {
        tukey_trim_pair(ty, sc)
    } else {
        (ty.to_vec(), sc.to_vec())
    };
    let theta_hat = match compute_n_sigma(&ty_obs, &sc_obs) {
        Some(v) if v.is_finite() => v,
        _ => return Some((pct_lo, pct_hi, None, None)),
    };

    // z₀ = Φ⁻¹( #{θ_b < θ̂} / B )
    let count_below: usize = nsigmas.iter().filter(|&&v| v < theta_hat).count();
    let p0 = (count_below as f64) / (nsigmas.len() as f64);
    if !(p0 > 0.0 && p0 < 1.0) {
        return Some((pct_lo, pct_hi, None, None));
    }
    let z0 = inv_phi(p0);
    if !z0.is_finite() {
        return Some((pct_lo, pct_hi, None, None));
    }

    // Paired jackknife on (TY, Schoen). Drop one observation at a time
    // from the union, recompute θ, accumulate Σ(θ̄ - θ_i)^k for k=2,3.
    // Optionally re-apply Tukey trim per jackknife replicate so the
    // estimator definition matches the bootstrap.
    let mut jack_thetas: Vec<f64> = Vec::with_capacity(n_ty + n_sc);
    let recompute_with_trim = |ty_in: &[f64], sc_in: &[f64]| -> Option<f64> {
        if per_resample_trim {
            let (t, s) = tukey_trim_pair(ty_in, sc_in);
            compute_n_sigma(&t, &s)
        } else {
            compute_n_sigma(ty_in, sc_in)
        }
    };
    for drop_i in 0..n_ty {
        let ty_jack: Vec<f64> = ty
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if i != drop_i { Some(v) } else { None })
            .collect();
        if let Some(t) = recompute_with_trim(&ty_jack, sc) {
            if t.is_finite() {
                jack_thetas.push(t);
            }
        }
    }
    for drop_i in 0..n_sc {
        let sc_jack: Vec<f64> = sc
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if i != drop_i { Some(v) } else { None })
            .collect();
        if let Some(t) = recompute_with_trim(ty, &sc_jack) {
            if t.is_finite() {
                jack_thetas.push(t);
            }
        }
    }
    if jack_thetas.len() < 2 {
        return Some((pct_lo, pct_hi, None, None));
    }
    let jbar: f64 = jack_thetas.iter().sum::<f64>() / (jack_thetas.len() as f64);
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for &j in &jack_thetas {
        let d = jbar - j;
        num += d * d * d;
        den += d * d;
    }
    if !(den > 0.0) {
        return Some((pct_lo, pct_hi, None, None));
    }
    let a = num / (6.0 * den.powf(1.5));
    if !a.is_finite() {
        return Some((pct_lo, pct_hi, None, None));
    }

    let z_lo = inv_phi(0.025);
    let z_hi = inv_phi(0.975);
    if !z_lo.is_finite() || !z_hi.is_finite() {
        return Some((pct_lo, pct_hi, None, None));
    }
    let alpha1 = phi(z0 + (z0 + z_lo) / (1.0 - a * (z0 + z_lo)));
    let alpha2 = phi(z0 + (z0 + z_hi) / (1.0 - a * (z0 + z_hi)));
    if !(alpha1.is_finite() && alpha2.is_finite()) {
        return Some((pct_lo, pct_hi, None, None));
    }
    if alpha1 <= 0.0 || alpha1 >= 1.0 || alpha2 <= 0.0 || alpha2 >= 1.0 {
        return Some((pct_lo, pct_hi, None, None));
    }
    // Empirical quantiles on the sorted resample vector. Linear interp.
    let q = |alpha: f64| -> f64 {
        let n = nsigmas.len();
        if n == 0 {
            return f64::NAN;
        }
        if n == 1 {
            return nsigmas[0];
        }
        let pos = alpha * ((n - 1) as f64);
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        if lo == hi {
            return nsigmas[lo];
        }
        let frac = pos - (lo as f64);
        nsigmas[lo] * (1.0 - frac) + nsigmas[hi] * frac
    };
    let bca_lo = q(alpha1);
    let bca_hi = q(alpha2);
    Some((pct_lo, pct_hi, Some(bca_lo), Some(bca_hi)))
}

/// Build a discrimination row from raw σ vectors.
///
/// `per_resample_trim` controls the bootstrap behaviour:
///   * `false` (default): standard bootstrap on the input vectors.
///     Used for Tier 0 / 1 / 2 (no trim is part of the estimator).
///   * `true`: Tukey-trimmed estimator. The CALLER must pass the *raw*
///     (untrimmed) canonical vectors AND the point-estimate row will be
///     computed from the post-trim values. The bootstrap re-applies the
///     Tukey filter to each resample so the CI represents the
///     distribution of "trimmed-n-σ if we'd sampled different data".
///     Round-4 hostile-review Defect B.
fn build_discrimination_row(
    label: &str,
    k: u32,
    ty: &[f64],
    sc: &[f64],
    boot_resamples: usize,
    boot_seed: u64,
    per_resample_trim: bool,
) -> Option<DiscriminationRow> {
    if ty.is_empty() || sc.is_empty() {
        return None;
    }
    // Compute the point estimate from the (possibly trimmed) vectors.
    // For per_resample_trim=true the input is the RAW canonical vectors
    // and we trim once for the point estimate; the bootstrap re-trims
    // each resample.
    let (ty_pt, sc_pt) = if per_resample_trim {
        tukey_trim_pair(ty, sc)
    } else {
        (ty.to_vec(), sc.to_vec())
    };
    if ty_pt.is_empty() || sc_pt.is_empty() {
        return None;
    }
    let n_ty = ty_pt.len() as f64;
    let mu_ty = ty_pt.iter().sum::<f64>() / n_ty;
    let var_ty = if ty_pt.len() > 1 {
        ty_pt.iter().map(|&v| (v - mu_ty).powi(2)).sum::<f64>() / (n_ty - 1.0)
    } else {
        0.0
    };
    let se_ty = (var_ty / n_ty).sqrt();
    let n_sc = sc_pt.len() as f64;
    let mu_sc = sc_pt.iter().sum::<f64>() / n_sc;
    let var_sc = if sc_pt.len() > 1 {
        sc_pt.iter().map(|&v| (v - mu_sc).powi(2)).sum::<f64>() / (n_sc - 1.0)
    } else {
        0.0
    };
    let se_sc = (var_sc / n_sc).sqrt();
    let delta = mu_ty - mu_sc;
    let se_comb = (se_ty * se_ty + se_sc * se_sc).sqrt();
    let n_sigma = if se_comb > 0.0 {
        delta.abs() / se_comb
    } else {
        f64::INFINITY
    };
    // For per-resample-trim runs, pass the *raw* (untrimmed) vectors
    // into the bootstrap so it can resample → re-trim. For non-trim
    // runs, the input is already the final estimator-input.
    let ci = bootstrap_n_sigma_ci(
        ty,
        sc,
        boot_resamples,
        boot_seed,
        per_resample_trim,
    );
    let (pct_lo, pct_hi, bca_lo, bca_hi) = match ci {
        Some((p_lo, p_hi, b_lo, b_hi)) => (Some(p_lo), Some(p_hi), b_lo, b_hi),
        None => (None, None, None, None),
    };
    Some(DiscriminationRow {
        k,
        mean_ty: mu_ty,
        se_ty,
        n_ty: ty_pt.len(),
        mean_schoen: mu_sc,
        se_schoen: se_sc,
        n_schoen: sc_pt.len(),
        delta_sigma: delta,
        se_combined: se_comb,
        n_sigma,
        n_sigma_ci_low: pct_lo,
        n_sigma_ci_high: pct_hi,
        n_sigma_bca_ci_low: bca_lo,
        n_sigma_bca_ci_high: bca_hi,
        n_sigma_boot_resamples: if ci.is_some() { boot_resamples } else { 0 },
        subset_label: label.to_string(),
    })
}

/// Bootstrap-seed sensitivity check (round-4 Defect C). Returns the CI
/// lower bounds across {primary, 999, 31415}; jitter at B=10 000 should
/// be ≤ 0.05σ. This is a stability assertion, not part of the headline.
#[derive(Debug, Serialize, Deserialize, Clone)]
struct BootSeedJitter {
    label: String,
    primary_seed: u64,
    primary_ci_low: f64,
    seed_999_ci_low: Option<f64>,
    seed_31415_ci_low: Option<f64>,
    /// max(|primary - other|) across the alternate seeds.
    max_jitter_sigma: f64,
}

fn bootstrap_seed_jitter(
    label: &str,
    ty: &[f64],
    sc: &[f64],
    boot_resamples: usize,
    primary_seed: u64,
    per_resample_trim: bool,
) -> Option<BootSeedJitter> {
    let primary = bootstrap_n_sigma_ci(
        ty,
        sc,
        boot_resamples,
        primary_seed,
        per_resample_trim,
    )?;
    let primary_lo = primary.0;
    let alt_seeds: [u64; 2] = [999, 31415];
    let mut alt_lows: Vec<Option<f64>> = Vec::new();
    for &s in &alt_seeds {
        let lo = bootstrap_n_sigma_ci(ty, sc, boot_resamples, s, per_resample_trim)
            .map(|c| c.0);
        alt_lows.push(lo);
    }
    let mut max_jitter = 0.0_f64;
    for lo in alt_lows.iter().flatten() {
        let d = (primary_lo - lo).abs();
        if d > max_jitter {
            max_jitter = d;
        }
    }
    Some(BootSeedJitter {
        label: label.to_string(),
        primary_seed,
        primary_ci_low: primary_lo,
        seed_999_ci_low: alt_lows[0],
        seed_31415_ci_low: alt_lows[1],
        max_jitter_sigma: max_jitter,
    })
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct EnsembleReport {
    label: String,
    path_label: String,
    n_pts: usize,
    seeds: Vec<u64>,
    donaldson_iters: usize,
    donaldson_tol: f64,
    /// P-REPRO-2 — compute path used for the Donaldson tree reduction.
    /// CPU and GPU paths are deterministic-but-bit-distinct (different
    /// reduction order), so persisting this field lets audit-time
    /// detect cross-path drift in otherwise-identical seeds. Optional
    /// for backward compatibility with legacy JSON outputs.
    #[serde(default)]
    compute_path: Option<String>,
    boot_resamples: usize,
    boot_seed: u64,
    candidates: Vec<CandidateKEnsemble>,
    /// Tier 1 — conservative (full 20+20 ensemble). Includes guard-
    /// restored mid-descent snapshots; this is a worst-case upper bound
    /// on σ_Schoen.
    discrimination: Vec<DiscriminationRow>,
    /// Round-2 (legacy) balanced-only subset — kept for back-compat.
    /// Iter-based filter (`iterations_run >= 10`).
    discrimination_balanced_only: Vec<DiscriminationRow>,
    /// Tier 0 — strict-converged (residual < tol AND iters < cap).
    /// Round-4 hostile review (Defect E): the canonical Tier 2 filter
    /// (residual < 1e-3) admits seeds that hit the iteration cap with
    /// residuals just above tol (e.g. seed 99 at residual 2.4e-6,
    /// iters=cap). The strict tier requires both the actual tol target
    /// AND non-cap completion — i.e. the seed's Donaldson iteration
    /// genuinely converged.
    #[serde(default)]
    discrimination_strict_converged: Vec<DiscriminationRow>,
    /// Tier 2 — canonical (residual < 1e-3). Round-3 residual-based
    /// classifier; defensible but admits cap-hitting seeds at residuals
    /// just above tol.
    #[serde(default)]
    discrimination_canonical: Vec<DiscriminationRow>,
    /// Tier 3 — Tukey-trimmed canonical subset. Drops Q1−1.5·IQR /
    /// Q3+1.5·IQR outliers from the converged-only σ vectors.
    /// Round-4 fix: the bootstrap CI now re-applies the Tukey trim to
    /// each resample (Defect B), giving a wider but methodologically
    /// correct CI.
    #[serde(default)]
    discrimination_reference: Vec<DiscriminationRow>,
    /// Round-4: bootstrap-seed jitter sensitivity (Defect C). Reports
    /// CI lower bound across {primary_seed, 999, 31415}; max_jitter
    /// at B=10 000 should be ≤ 0.05σ.
    #[serde(default)]
    boot_seed_jitter: Vec<BootSeedJitter>,
    /// Seeds flagged as early-bail (returned a near-FS-identity snapshot
    /// via the P5.5f regression guard rather than a balanced metric).
    early_bail_seeds: Vec<EarlyBailFlag>,
    balanced_min_iters: usize,
    /// Round-3: residual threshold below which a seed is treated as
    /// Donaldson-converged.
    #[serde(default)]
    converged_residual_threshold: f64,
    /// Round-3: residual threshold above which a seed is treated as
    /// non-converged (early-bail-suspect).
    #[serde(default)]
    ambiguous_residual_threshold: f64,
    max_n_sigma: f64,
    max_n_sigma_k: u32,
    five_sigma_achieved: bool,
    total_elapsed_s: f64,
    git_revision: Option<String>,
    /// Reproducibility manifest collected at run start: rust toolchain,
    /// target triple, CPU SIMD features, hostname, RFC 3339 UTC timestamp,
    /// command line, rayon thread count. Captured for publication-grade
    /// auditability — see `route34::repro::ReproManifest`.
    #[serde(default)]
    repro_manifest: Option<ReproManifest>,
    /// Sidecar path to the chained-SHA event replog. Each line is a
    /// `ReplogRecord` (run_start, per_seed × N, run_end), with the per-event
    /// chain hash computed via SHA-256(prev_chain_hash || event_json_bytes).
    /// The pwos-math `ReproLog` byte payload lives at the same stem with
    /// extension `.kernel.replog`.
    #[serde(default)]
    repro_log_path: Option<String>,
    /// Final SHA-256 chain hash (lowercase hex) of the replog event stream.
    /// Lets a downstream verifier replay the binary and compare hashes
    /// without reading the .replog file at all.
    #[serde(default)]
    repro_log_final_chain_sha256_hex: Option<String>,
    /// Number of events written to the replog (run_start + per_seed + run_end).
    #[serde(default)]
    repro_log_n_events: Option<usize>,
}

/// Schoen (d_x, d_y, d_t) tuple for a given "k" label, matching the
/// existing crate convention (`schoen_solver_dispatches_correctly` uses
/// (3,3,1) for k=3, `schoen_publication_default` uses (4,4,2) for k=4).
///
/// k=5 extension (P-K5, Apr 2026): (5,5,2) — preserves d_t=⌊k/2⌋
/// scaling from k=4 → k=5 and keeps total bidegree d_x+d_y+d_t=12 in
/// the k=4-equivalent stall regime trigger band (≥10), so the auto-
/// damping / Tikhonov rules in `auto_schoen_*` still fire as intended.
fn schoen_tuple_for_k(k: u32) -> (u32, u32, u32) {
    match k {
        2 => (2, 2, 1),
        3 => (3, 3, 1),
        4 => (4, 4, 2),
        5 => (5, 5, 2),
        other => panic!("unsupported k for Schoen mapping: {}", other),
    }
}

fn build_adam_override(
    use_gpu: bool,
    adam_iters: usize,
    adam_lr: f64,
) -> Cy3AdamOverride {
    let adam_refine = if adam_iters > 0 {
        Some(AdamRefineConfig {
            max_iters: adam_iters,
            learning_rate: adam_lr,
            fd_step: Some(1.0e-3),
            tol: 1.0e-6,
            use_gpu,
        })
    } else {
        None
    };
    Cy3AdamOverride {
        adam_refine,
        use_gpu_donaldson: use_gpu,
    }
}

fn run_ty_one(
    k: u32,
    n_pts: usize,
    seed: u64,
    donaldson_iters: usize,
    donaldson_tol: f64,
    use_gpu: bool,
    adam_iters: usize,
    adam_lr: f64,
) -> Result<PerSeedRecord, String> {
    let t0 = Instant::now();
    let solver = TianYauSolver;
    let spec = Cy3MetricSpec::TianYau {
        k,
        n_sample: n_pts,
        max_iter: donaldson_iters,
        donaldson_tol,
        seed,
    };
    let adam_override = build_adam_override(use_gpu, adam_iters, adam_lr);
    let r = solver
        .solve_metric_with_adam(&spec, &adam_override)
        .map_err(|e| format!("TY (k={k}, seed={seed}): {e}"))?;
    let s = r.summary();
    let sigma_fs_identity = r.sigma_fs_identity();
    let final_donaldson_residual = s.final_donaldson_residual;
    let convergence_tier = classify_tier(final_donaldson_residual).to_string();
    let (sigma_after_donaldson, sigma_after_adam, adam_iters_run) = match &r {
        Cy3MetricResultKind::TianYau(inner) => (
            inner.sigma_after_donaldson,
            inner.sigma_after_adam,
            inner.adam_iters_run,
        ),
        Cy3MetricResultKind::Schoen(_) => (f64::NAN, f64::NAN, 0),
    };
    let elapsed_s = t0.elapsed().as_secs_f64();
    Ok(PerSeedRecord {
        seed,
        candidate: "TY".to_string(),
        k,
        elapsed_s,
        n_basis: s.n_basis,
        iterations_run: s.iterations_run,
        sigma_final: s.final_sigma_residual,
        sigma_fs_identity,
        final_donaldson_residual,
        convergence_tier,
        sigma_after_donaldson,
        sigma_after_adam,
        adam_iters_run,
    })
}

fn run_schoen_one(
    k: u32,
    n_pts: usize,
    seed: u64,
    donaldson_iters: usize,
    donaldson_tol: f64,
    use_gpu: bool,
    adam_iters: usize,
    adam_lr: f64,
) -> Result<PerSeedRecord, String> {
    let t0 = Instant::now();
    let (d_x, d_y, d_t) = schoen_tuple_for_k(k);
    let solver = SchoenSolver;
    let spec = Cy3MetricSpec::Schoen {
        d_x,
        d_y,
        d_t,
        n_sample: n_pts,
        max_iter: donaldson_iters,
        donaldson_tol,
        seed,
    };
    let adam_override = build_adam_override(use_gpu, adam_iters, adam_lr);
    let r = solver
        .solve_metric_with_adam(&spec, &adam_override)
        .map_err(|e| format!("Schoen (k={k} → ({d_x},{d_y},{d_t}), seed={seed}): {e}"))?;
    let s = r.summary();
    let sigma_fs_identity = r.sigma_fs_identity();
    let final_donaldson_residual = s.final_donaldson_residual;
    let convergence_tier = classify_tier(final_donaldson_residual).to_string();
    let (sigma_after_donaldson, sigma_after_adam, adam_iters_run) = match &r {
        Cy3MetricResultKind::Schoen(inner) => (
            inner.sigma_after_donaldson,
            inner.sigma_after_adam,
            inner.adam_iters_run,
        ),
        Cy3MetricResultKind::TianYau(_) => (f64::NAN, f64::NAN, 0),
    };
    let elapsed_s = t0.elapsed().as_secs_f64();
    Ok(PerSeedRecord {
        seed,
        candidate: "Schoen".to_string(),
        k,
        elapsed_s,
        n_basis: s.n_basis,
        iterations_run: s.iterations_run,
        sigma_final: s.final_sigma_residual,
        sigma_fs_identity,
        final_donaldson_residual,
        convergence_tier,
        sigma_after_donaldson,
        sigma_after_adam,
        adam_iters_run,
    })
}

fn compute_stats(
    name: &str,
    samples: &[f64],
    boot_resamples: usize,
    boot_seed: u64,
    ci_level: f64,
) -> CheckpointStats {
    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / n as f64;
    let var = if n > 1 {
        samples.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)
    } else {
        0.0
    };
    let std = var.sqrt();
    let stderr = std / (n as f64).sqrt();
    let cfg = BootstrapConfig {
        n_resamples: boot_resamples,
        seed: boot_seed,
        ci_level,
    };
    let mut boot = Bootstrap::new(cfg, n).expect("bootstrap workspace");
    let result = boot
        .run(samples, |sample: &[f64]| -> f64 {
            sample.iter().sum::<f64>() / sample.len() as f64
        })
        .expect("bootstrap run");
    CheckpointStats {
        name: name.to_string(),
        n,
        mean,
        std,
        stderr,
        percentile_ci_low: result.percentile_ci.0,
        percentile_ci_high: result.percentile_ci.1,
        bca_ci_low: result.bca_ci.map(|c| c.0),
        bca_ci_high: result.bca_ci.map(|c| c.1),
    }
}

fn print_checkpoint(stats: &CheckpointStats) {
    eprintln!(
        "    {:>26}: mean={:.6} std={:.4e} stderr={:.4e}  pct95=[{:.6}, {:.6}]  bca95=[{}, {}]",
        stats.name,
        stats.mean,
        stats.std,
        stats.stderr,
        stats.percentile_ci_low,
        stats.percentile_ci_high,
        stats
            .bca_ci_low
            .map(|v| format!("{:.6}", v))
            .unwrap_or_else(|| "n/a".to_string()),
        stats
            .bca_ci_high
            .map(|v| format!("{:.6}", v))
            .unwrap_or_else(|| "n/a".to_string()),
    );
}

fn git_revision() -> Option<String> {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
            } else {
                None
            }
        })
}

fn main() {
    let cli = Cli::parse();
    let ks: Vec<u32> = cli
        .ks
        .split(',')
        .map(|s| s.trim().parse::<u32>().expect("invalid k in --ks"))
        .collect();
    let seeds = SEEDS_20.as_slice();

    // Reproducibility manifest + chained-SHA event log. Capacity is sized
    // for the worst case (run_start + 2 candidates × |ks| × |seeds| + run_end).
    let repro_manifest = ReproManifest::collect();
    let mut replog = ReplogWriter::new(2 + 2 * ks.len() * seeds.len());
    replog.push(ReplogEvent::RunStart {
        binary: "p5_10_ty_schoen_5sigma".to_string(),
        manifest: repro_manifest.clone(),
        config_json: serde_json::json!({
            "n_pts": cli.n_pts,
            "ks": ks,
            "donaldson_iters": cli.donaldson_iters,
            "donaldson_tol": cli.donaldson_tol,
            "boot_resamples": cli.boot_resamples,
            "boot_seed": cli.boot_seed,
            "report_boot_seed_jitter": cli.report_boot_seed_jitter,
            "output": cli.output.display().to_string(),
            "seeds": seeds,
        }),
    });

    eprintln!("=== P5.10 TY-vs-Schoen 5σ-target ensemble (Path A: n_pts boost) ===");
    eprintln!(
        "n_pts={}, ks={:?}, seeds(n={})={:?}",
        cli.n_pts,
        ks,
        seeds.len(),
        seeds
    );
    let adam_tag = if cli.adam_iters > 0 {
        format!(
            ", Adam(iters={}, lr={:.1e})",
            cli.adam_iters, cli.adam_lr
        )
    } else {
        " (no Adam refine)".to_string()
    };
    let gpu_tag = if cli.use_gpu { "GPU" } else { "CPU" };
    eprintln!(
        "Donaldson({}, {:.0e}, {gpu_tag}){adam_tag}",
        cli.donaldson_iters, cli.donaldson_tol
    );
    eprintln!(
        "Bootstrap: n_resamples={}, seed={}",
        cli.boot_resamples, cli.boot_seed
    );
    eprintln!();

    let t_total = Instant::now();
    let mut candidates: Vec<CandidateKEnsemble> = Vec::new();

    for &candidate in &["TY", "Schoen"] {
        for &k in &ks {
            eprintln!("--- {candidate} k={k} (n_pts={}) ---", cli.n_pts);
            let mut per_seed: Vec<PerSeedRecord> = Vec::with_capacity(seeds.len());
            for (i, &seed) in seeds.iter().enumerate() {
                let res = match candidate {
                    "TY" => run_ty_one(
                        k,
                        cli.n_pts,
                        seed,
                        cli.donaldson_iters,
                        cli.donaldson_tol,
                        cli.use_gpu,
                        cli.adam_iters,
                        cli.adam_lr,
                    ),
                    "Schoen" => run_schoen_one(
                        k,
                        cli.n_pts,
                        seed,
                        cli.donaldson_iters,
                        cli.donaldson_tol,
                        cli.use_gpu,
                        cli.adam_iters,
                        cli.adam_lr,
                    ),
                    _ => unreachable!(),
                };
                match res {
                    Ok(rec) => {
                        let adam_tag = if rec.adam_iters_run > 0
                            && rec.sigma_after_adam.is_finite()
                        {
                            format!(
                                "  σ_donaldson={:.6} σ_after_adam={:.6} adam_iters={}",
                                rec.sigma_after_donaldson,
                                rec.sigma_after_adam,
                                rec.adam_iters_run,
                            )
                        } else {
                            String::new()
                        };
                        eprintln!(
                            "  [{:2}/{}] seed={:>6}: σ={:.6}  n_basis={:>4}  iters={:>2}  res_donaldson={:.3e}  tier={:<13}  ({:.2}s){}",
                            i + 1,
                            seeds.len(),
                            rec.seed,
                            rec.sigma_final,
                            rec.n_basis,
                            rec.iterations_run,
                            rec.final_donaldson_residual,
                            rec.convergence_tier,
                            rec.elapsed_s,
                            adam_tag,
                        );
                        // Append a chained-SHA event for this Donaldson
                        // solve. Replay-verifier replays this stream,
                        // recomputes chain[i] = SHA256(chain[i-1] || event_i),
                        // and compares the final hash to the JSON output's
                        // `repro_log_final_chain_sha256_hex` field.
                        replog.push(ReplogEvent::PerSeed(PerSeedEvent {
                            seed: rec.seed,
                            candidate: rec.candidate.clone(),
                            k: rec.k,
                            iters_run: rec.iterations_run,
                            final_residual: rec.final_donaldson_residual,
                            sigma_fs_identity: rec.sigma_fs_identity,
                            sigma_final: rec.sigma_final,
                            n_basis: rec.n_basis,
                            elapsed_ms: rec.elapsed_s * 1000.0,
                        }));
                        per_seed.push(rec);
                    }
                    Err(e) => {
                        eprintln!(
                            "  [{:2}/{}] seed={:>6}: SKIPPED — {e}",
                            i + 1,
                            seeds.len(),
                            seed
                        );
                    }
                }
            }
            if per_seed.is_empty() {
                eprintln!("  {candidate} k={k}: no seeds succeeded; skipping stats");
                continue;
            }
            let sigmas: Vec<f64> = per_seed
                .iter()
                .map(|r| r.sigma_final)
                .filter(|s| s.is_finite() && *s > 0.0)
                .collect();
            if sigmas.is_empty() {
                eprintln!("  {candidate} k={k}: no finite σ values; skipping stats");
                continue;
            }
            let label = format!("sigma_{}_k{}", candidate.to_lowercase(), k);
            let stats = compute_stats(
                &label,
                &sigmas,
                cli.boot_resamples,
                cli.boot_seed,
                0.95,
            );
            eprintln!("  Stats:");
            print_checkpoint(&stats);
            eprintln!();
            candidates.push(CandidateKEnsemble {
                candidate: candidate.to_string(),
                k,
                n_pts: cli.n_pts,
                seeds: seeds.to_vec(),
                per_seed,
                sigma_stats: stats,
            });
        }
    }

    // ---------------------------------------------------------------
    // Four-tier discrimination (round-4 hostile-review).
    //
    // Tier 0 — strict-converged: residual < tol AND iters < cap. The
    //          most defensible "fully converged" subset — both the
    //          residual threshold AND the iteration-cap criterion are
    //          satisfied. Round-4 Defect E: Tier 2's residual<1e-3
    //          filter admits seeds that hit the iter cap (e.g. seed 99
    //          residual=2.4e-6 at cap iters=50, ABOVE the actual
    //          tol=1e-6 target).
    // Tier 1 — conservative: full ensemble (worst case if guard
    //          snapshots are trusted).
    // Tier 2 — canonical:    residual-based filter, residual < 1e-3
    //          (1000× tol). Round-3 classifier; less strict than Tier 0.
    // Tier 3 — Tukey-trimmed canonical with RE-TRIM bootstrap (round-4
    //          Defect B): the trim is part of the estimator and is
    //          re-applied to each bootstrap resample.
    // ---------------------------------------------------------------
    let mut discrimination: Vec<DiscriminationRow> = Vec::new();
    let mut discrimination_strict_converged: Vec<DiscriminationRow> = Vec::new();
    let mut discrimination_canonical: Vec<DiscriminationRow> = Vec::new();
    let mut discrimination_reference: Vec<DiscriminationRow> = Vec::new();
    let mut boot_seed_jitter: Vec<BootSeedJitter> = Vec::new();
    let mut max_n_sigma: f64 = 0.0;
    let mut max_n_sigma_k: u32 = 0;
    for &k in &ks {
        let ty = candidates
            .iter()
            .find(|c| c.candidate == "TY" && c.k == k);
        let sc = candidates
            .iter()
            .find(|c| c.candidate == "Schoen" && c.k == k);
        let (ty, sc) = match (ty, sc) {
            (Some(a), Some(b)) => (a, b),
            _ => {
                eprintln!("  [n-σ k={k}] missing data — skipping");
                continue;
            }
        };
        // Conservative: full ensemble σ vectors (after finite-positive filter).
        let ty_full: Vec<f64> = ty
            .per_seed
            .iter()
            .filter(|r| r.sigma_final.is_finite() && r.sigma_final > 0.0)
            .map(|r| r.sigma_final)
            .collect();
        let sc_full: Vec<f64> = sc
            .per_seed
            .iter()
            .filter(|r| r.sigma_final.is_finite() && r.sigma_final > 0.0)
            .map(|r| r.sigma_final)
            .collect();
        if let Some(row) = build_discrimination_row(
            "conservative",
            k,
            &ty_full,
            &sc_full,
            cli.boot_resamples,
            cli.boot_seed,
            false,
        ) {
            if row.n_sigma > max_n_sigma {
                max_n_sigma = row.n_sigma;
                max_n_sigma_k = k;
            }
            discrimination.push(row);
        }
        // Tier 0 — strict-converged: residual < tol AND iters < cap.
        let strict_filter = |r: &PerSeedRecord| -> bool {
            r.sigma_final.is_finite()
                && r.sigma_final > 0.0
                && r.final_donaldson_residual.is_finite()
                && r.final_donaldson_residual < cli.donaldson_tol
                && r.iterations_run < cli.donaldson_iters
        };
        let ty_strict: Vec<f64> = ty
            .per_seed
            .iter()
            .filter(|r| strict_filter(r))
            .map(|r| r.sigma_final)
            .collect();
        let sc_strict: Vec<f64> = sc
            .per_seed
            .iter()
            .filter(|r| strict_filter(r))
            .map(|r| r.sigma_final)
            .collect();
        if let Some(row) = build_discrimination_row(
            "strict_converged",
            k,
            &ty_strict,
            &sc_strict,
            cli.boot_resamples,
            cli.boot_seed,
            false,
        ) {
            discrimination_strict_converged.push(row);
        } else {
            eprintln!(
                "  [strict k={k}] empty subset (TY n={}, Schoen n={}) — skipping",
                ty_strict.len(),
                sc_strict.len()
            );
        }
        // Canonical (Tier 2): residual < 1e-3.
        let ty_canon: Vec<f64> = ty
            .per_seed
            .iter()
            .filter(|r| {
                r.convergence_tier == "converged"
                    && r.sigma_final.is_finite()
                    && r.sigma_final > 0.0
            })
            .map(|r| r.sigma_final)
            .collect();
        let sc_canon: Vec<f64> = sc
            .per_seed
            .iter()
            .filter(|r| {
                r.convergence_tier == "converged"
                    && r.sigma_final.is_finite()
                    && r.sigma_final > 0.0
            })
            .map(|r| r.sigma_final)
            .collect();
        if let Some(row) = build_discrimination_row(
            "canonical",
            k,
            &ty_canon,
            &sc_canon,
            cli.boot_resamples,
            cli.boot_seed,
            false,
        ) {
            discrimination_canonical.push(row);
        } else {
            eprintln!(
                "  [canonical k={k}] empty subset (TY n={}, Schoen n={}) — skipping",
                ty_canon.len(),
                sc_canon.len()
            );
        }
        // Tier 3 — Tukey-trimmed canonical with RE-TRIM bootstrap.
        // Pass the RAW canonical vectors; build_discrimination_row with
        // per_resample_trim=true will (a) compute the point estimate on
        // the trimmed vectors and (b) re-trim each bootstrap resample.
        if let Some(row) = build_discrimination_row(
            "reference_retrim",
            k,
            &ty_canon,
            &sc_canon,
            cli.boot_resamples,
            cli.boot_seed,
            true,
        ) {
            discrimination_reference.push(row);
        }
        // Round-4 Defect C: bootstrap-seed jitter sensitivity check.
        if cli.report_boot_seed_jitter {
            // Tier 0 (no trim).
            if let Some(j) = bootstrap_seed_jitter(
                &format!("strict_converged_k{}", k),
                &ty_strict,
                &sc_strict,
                cli.boot_resamples,
                cli.boot_seed,
                false,
            ) {
                boot_seed_jitter.push(j);
            }
            // Tier 2 (no trim).
            if let Some(j) = bootstrap_seed_jitter(
                &format!("canonical_k{}", k),
                &ty_canon,
                &sc_canon,
                cli.boot_resamples,
                cli.boot_seed,
                false,
            ) {
                boot_seed_jitter.push(j);
            }
            // Tier 3 (re-trim).
            if let Some(j) = bootstrap_seed_jitter(
                &format!("reference_retrim_k{}", k),
                &ty_canon,
                &sc_canon,
                cli.boot_resamples,
                cli.boot_seed,
                true,
            ) {
                boot_seed_jitter.push(j);
            }
        }
    }

    fn print_discrimination_table(label: &str, rows: &[DiscriminationRow]) {
        eprintln!("=== n-σ discrimination ({label}) ===");
        eprintln!(
            "| {:>2} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8} | {:>5} | {:>5} | {:>16} | {:>16} |",
            "k", "<σ_TY>", "SE_TY", "<σ_Schoen>", "SE_Schoen", "Δσ", "SE_comb", "n-σ",
            "n_TY", "n_Sc", "n-σ pct 95% CI", "n-σ BCa 95% CI"
        );
        eprintln!(
            "|----|------------|------------|------------|------------|------------|------------|----------|-------|-------|------------------|------------------|"
        );
        for row in rows {
            let pct_str = match (row.n_sigma_ci_low, row.n_sigma_ci_high) {
                (Some(lo), Some(hi)) => format!("[{:.3}, {:.3}]", lo, hi),
                _ => "n/a".to_string(),
            };
            let bca_str = match (row.n_sigma_bca_ci_low, row.n_sigma_bca_ci_high) {
                (Some(lo), Some(hi)) => format!("[{:.3}, {:.3}]", lo, hi),
                _ => "n/a".to_string(),
            };
            eprintln!(
                "| {:>2} | {:>10.6} | {:>10.4e} | {:>10.6} | {:>10.4e} | {:>+10.6} | {:>10.4e} | {:>8.3} | {:>5} | {:>5} | {:>16} | {:>16} |",
                row.k,
                row.mean_ty,
                row.se_ty,
                row.mean_schoen,
                row.se_schoen,
                row.delta_sigma,
                row.se_combined,
                row.n_sigma,
                row.n_ty,
                row.n_schoen,
                pct_str,
                bca_str,
            );
        }
    }
    print_discrimination_table(
        "Tier 0: STRICT-CONVERGED (residual < tol AND iters < cap)",
        &discrimination_strict_converged,
    );
    eprintln!();
    print_discrimination_table("Tier 1: conservative (full 20+20)", &discrimination);
    eprintln!();
    print_discrimination_table(
        "Tier 2: canonical (residual < 1e-3)",
        &discrimination_canonical,
    );
    eprintln!();
    print_discrimination_table(
        "Tier 3: Tukey-trimmed canonical with RE-TRIM bootstrap (round-4 Defect B fix)",
        &discrimination_reference,
    );

    // ---------------------------------------------------------------
    // Round-4 hostile-review verdict logic. The most-defensible headline
    // is the highest-tier row that clears 5σ at BOTH the point estimate
    // and the 95% CI lower bound. We check Tier 0 first (strictest), then
    // Tier 3 (Tukey-trimmed with re-trim), Tier 2 (canonical), Tier 1
    // (conservative). If none clear 5σ at the CI floor, we report the
    // strongest tier that clears at the point estimate AND honestly
    // state that the CI floor falls short.
    // ---------------------------------------------------------------
    fn check_tier(rows: &[DiscriminationRow]) -> (Option<f64>, Option<f64>, Option<f64>, Option<u32>) {
        let r = rows.first();
        (
            r.map(|x| x.n_sigma),
            r.and_then(|x| x.n_sigma_ci_low),
            r.and_then(|x| x.n_sigma_ci_high),
            r.map(|x| x.k),
        )
    }
    let (t0_pt, t0_lo, t0_hi, t0_k) = check_tier(&discrimination_strict_converged);
    let (t1_pt, t1_lo, t1_hi, t1_k) = check_tier(&discrimination);
    let (t2_pt, t2_lo, t2_hi, t2_k) = check_tier(&discrimination_canonical);
    let (t3_pt, t3_lo, t3_hi, t3_k) = check_tier(&discrimination_reference);

    fn clears(pt: Option<f64>, lo: Option<f64>) -> (bool, bool) {
        let p = pt.map(|v| v.is_finite() && v >= 5.0).unwrap_or(false);
        let l = lo.map(|v| v.is_finite() && v >= 5.0).unwrap_or(false);
        (p, l)
    }
    let (t0_pt_ok, t0_lo_ok) = clears(t0_pt, t0_lo);
    let (t1_pt_ok, t1_lo_ok) = clears(t1_pt, t1_lo);
    let (t2_pt_ok, t2_lo_ok) = clears(t2_pt, t2_lo);
    let (t3_pt_ok, t3_lo_ok) = clears(t3_pt, t3_lo);

    // Pick the headline tier: strictest tier that clears point AND CI floor.
    // If none clear CI floor, pick the strictest tier that clears point.
    // If none clear point either, report "5σ NOT MET".
    let (headline_label, headline_pt, headline_lo, headline_hi, headline_k, ci_clears) =
        if t0_pt_ok && t0_lo_ok {
            ("Tier 0 (strict-converged)", t0_pt, t0_lo, t0_hi, t0_k, true)
        } else if t3_pt_ok && t3_lo_ok {
            ("Tier 3 (Tukey-trimmed with re-trim bootstrap)", t3_pt, t3_lo, t3_hi, t3_k, true)
        } else if t2_pt_ok && t2_lo_ok {
            ("Tier 2 (canonical residual<1e-3)", t2_pt, t2_lo, t2_hi, t2_k, true)
        } else if t1_pt_ok && t1_lo_ok {
            ("Tier 1 (conservative full ensemble)", t1_pt, t1_lo, t1_hi, t1_k, true)
        } else if t0_pt_ok {
            ("Tier 0 (strict-converged) — POINT ONLY, CI<5σ", t0_pt, t0_lo, t0_hi, t0_k, false)
        } else if t3_pt_ok {
            ("Tier 3 (Tukey-trimmed with re-trim) — POINT ONLY, CI<5σ", t3_pt, t3_lo, t3_hi, t3_k, false)
        } else if t2_pt_ok {
            ("Tier 2 (canonical) — POINT ONLY, CI<5σ", t2_pt, t2_lo, t2_hi, t2_k, false)
        } else if t1_pt_ok {
            ("Tier 1 (conservative) — POINT ONLY, CI<5σ", t1_pt, t1_lo, t1_hi, t1_k, false)
        } else {
            ("NONE — 5σ NOT MET AT ANY TIER POINT ESTIMATE", t2_pt, t2_lo, t2_hi, t2_k, false)
        };

    let any_pt_clears = t0_pt_ok || t1_pt_ok || t2_pt_ok || t3_pt_ok;
    let _any_lo_clears = t0_lo_ok || t1_lo_ok || t2_lo_ok || t3_lo_ok;
    let five_sigma_achieved = any_pt_clears;

    let verdict = if ci_clears {
        "5σ DISCRIMINATION ACHIEVED — point estimate AND 95% CI lower bound both >= 5σ at headline tier"
    } else if any_pt_clears {
        "5σ AT POINT ESTIMATE ONLY — 95% CI lower bound < 5σ at every tier; reported honestly, NOT papered over"
    } else {
        "5σ NOT MET — point estimate < 5σ at every tier; need more budget"
    };

    fn fmt_or_na(v: Option<f64>) -> String {
        v.map(|x| format!("{:.3}", x)).unwrap_or_else(|| "n/a".to_string())
    }

    eprintln!();
    eprintln!("=== Round-4 5σ verdict — four-tier ===");
    eprintln!(
        "  Tier 0 (strict-converged): n-σ = {} at k={}  CI=[{}, {}]   point>=5={}  CI_lo>=5={}",
        fmt_or_na(t0_pt), t0_k.map(|k| k.to_string()).unwrap_or_else(|| "?".to_string()),
        fmt_or_na(t0_lo), fmt_or_na(t0_hi), t0_pt_ok, t0_lo_ok,
    );
    eprintln!(
        "  Tier 1 (conservative):     n-σ = {} at k={}  CI=[{}, {}]   point>=5={}  CI_lo>=5={}",
        fmt_or_na(t1_pt), t1_k.map(|k| k.to_string()).unwrap_or_else(|| "?".to_string()),
        fmt_or_na(t1_lo), fmt_or_na(t1_hi), t1_pt_ok, t1_lo_ok,
    );
    eprintln!(
        "  Tier 2 (canonical):        n-σ = {} at k={}  CI=[{}, {}]   point>=5={}  CI_lo>=5={}",
        fmt_or_na(t2_pt), t2_k.map(|k| k.to_string()).unwrap_or_else(|| "?".to_string()),
        fmt_or_na(t2_lo), fmt_or_na(t2_hi), t2_pt_ok, t2_lo_ok,
    );
    eprintln!(
        "  Tier 3 (Tukey re-trim):    n-σ = {} at k={}  CI=[{}, {}]   point>=5={}  CI_lo>=5={}",
        fmt_or_na(t3_pt), t3_k.map(|k| k.to_string()).unwrap_or_else(|| "?".to_string()),
        fmt_or_na(t3_lo), fmt_or_na(t3_hi), t3_pt_ok, t3_lo_ok,
    );
    eprintln!();
    eprintln!("  HEADLINE TIER: {}", headline_label);
    eprintln!(
        "  HEADLINE n-σ = {} at k={}  (95% CI = [{}, {}])",
        fmt_or_na(headline_pt),
        headline_k.map(|k| k.to_string()).unwrap_or_else(|| "?".to_string()),
        fmt_or_na(headline_lo),
        fmt_or_na(headline_hi),
    );
    eprintln!("  VERDICT      = {}", verdict);
    eprintln!(
        "  conservative max n-σ (full 20+20, INCLUDES guard-restored snapshots) = {:.3} at k={}",
        max_n_sigma, max_n_sigma_k
    );

    // Bootstrap-seed jitter (round-4 Defect C).
    if !boot_seed_jitter.is_empty() {
        eprintln!();
        eprintln!("=== Bootstrap-seed jitter sensitivity (B={}) ===", cli.boot_resamples);
        eprintln!(
            "  | {:>32} | {:>10} | {:>10} | {:>10} | {:>14} |",
            "label", "primary_lo", "seed_999", "seed_31415", "max_jitter (σ)"
        );
        for j in &boot_seed_jitter {
            eprintln!(
                "  | {:>32} | {:>10.3} | {:>10} | {:>10} | {:>14.4} |",
                j.label,
                j.primary_ci_low,
                j.seed_999_ci_low.map(|v| format!("{:.3}", v)).unwrap_or_else(|| "n/a".to_string()),
                j.seed_31415_ci_low.map(|v| format!("{:.3}", v)).unwrap_or_else(|| "n/a".to_string()),
                j.max_jitter_sigma,
            );
        }
        let max_jitter = boot_seed_jitter.iter().map(|j| j.max_jitter_sigma).fold(0.0_f64, f64::max);
        eprintln!(
            "  overall max_jitter = {:.4}σ (target: ≤0.05σ at B={})",
            max_jitter, cli.boot_resamples
        );
    }

    eprintln!(
        "P5.7 baseline (n_pts=10000, k=3): n-σ = 4.854.  P5.10 conservative (n_pts={}, k=3): n-σ = {:.3}.",
        cli.n_pts, max_n_sigma
    );

    // ---------------------------------------------------------------
    // Round-2 hostile-review re-analysis (Concern B).
    //
    // The P5.5f regression guard restores the iter-min snapshot when a
    // seed's Donaldson residual blows up. On Schoen, several seeds bail
    // before iter ~10 and return σ within a few % of `sigma_fs_identity`
    // — i.e. the guard's restored state is essentially the FS identity,
    // not a balanced metric. This biases the canonical Schoen ensemble
    // mean toward the FS-identity floor.
    //
    // Compute a balanced-only subset (seeds with `iterations_run >=
    // BALANCED_MIN_ITERS=10`) for both candidates and re-run the n-σ
    // calc as a sharper "what TY-vs-Schoen looks like when both are
    // converged" comparator. The canonical 6.48σ above stands as the
    // production-pipeline headline; the balanced-only number lives
    // alongside it for transparency.
    // ---------------------------------------------------------------
    let mut early_bail_seeds: Vec<EarlyBailFlag> = Vec::new();
    for cand in &candidates {
        for rec in &cand.per_seed {
            if rec.iterations_run < BALANCED_MIN_ITERS {
                let rel = if rec.sigma_fs_identity.is_finite()
                    && rec.sigma_fs_identity.abs() > 0.0
                {
                    (rec.sigma_final - rec.sigma_fs_identity)
                        / rec.sigma_fs_identity.abs()
                } else {
                    f64::NAN
                };
                early_bail_seeds.push(EarlyBailFlag {
                    seed: rec.seed,
                    candidate: rec.candidate.clone(),
                    iterations_run: rec.iterations_run,
                    sigma_final: rec.sigma_final,
                    sigma_fs_identity: rec.sigma_fs_identity,
                    rel_diff_to_fs: rel,
                });
            }
        }
    }

    let mut discrimination_balanced_only: Vec<DiscriminationRow> = Vec::new();
    for &k in &ks {
        let ty = candidates
            .iter()
            .find(|c| c.candidate == "TY" && c.k == k);
        let sc = candidates
            .iter()
            .find(|c| c.candidate == "Schoen" && c.k == k);
        let (ty, sc) = match (ty, sc) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };
        let ty_balanced: Vec<f64> = ty
            .per_seed
            .iter()
            .filter(|r| {
                r.iterations_run >= BALANCED_MIN_ITERS
                    && r.sigma_final.is_finite()
                    && r.sigma_final > 0.0
            })
            .map(|r| r.sigma_final)
            .collect();
        let sc_balanced: Vec<f64> = sc
            .per_seed
            .iter()
            .filter(|r| {
                r.iterations_run >= BALANCED_MIN_ITERS
                    && r.sigma_final.is_finite()
                    && r.sigma_final > 0.0
            })
            .map(|r| r.sigma_final)
            .collect();
        if ty_balanced.is_empty() || sc_balanced.is_empty() {
            eprintln!(
                "  [balanced-only k={k}] empty subset (TY n={}, Schoen n={}) — skipping",
                ty_balanced.len(),
                sc_balanced.len()
            );
            continue;
        }
        if let Some(row) = build_discrimination_row(
            "balanced_only_iters_ge_10",
            k,
            &ty_balanced,
            &sc_balanced,
            cli.boot_resamples,
            cli.boot_seed,
            false,
        ) {
            discrimination_balanced_only.push(row);
        }
    }

    eprintln!();
    print_discrimination_table(
        &format!(
            "Legacy balanced-only (iters >= {}) — round-2 (Concern B), kept for back-compat",
            BALANCED_MIN_ITERS
        ),
        &discrimination_balanced_only,
    );
    if !early_bail_seeds.is_empty() {
        eprintln!();
        eprintln!(
            "=== early-bail seeds (iters < {}) — final σ vs FS-identity σ ===",
            BALANCED_MIN_ITERS
        );
        eprintln!(
            "| {:>8} | {:>6} | {:>5} | {:>10} | {:>14} | {:>10} |",
            "candidate", "seed", "iters", "σ_final", "σ_fs_identity", "rel_diff"
        );
        eprintln!(
            "|----------|--------|-------|------------|----------------|------------|"
        );
        for f in &early_bail_seeds {
            eprintln!(
                "| {:>8} | {:>6} | {:>5} | {:>10.6} | {:>14.6} | {:>+10.4} |",
                f.candidate,
                f.seed,
                f.iterations_run,
                f.sigma_final,
                f.sigma_fs_identity,
                f.rel_diff_to_fs,
            );
        }
    }

    let total_elapsed_s = t_total.elapsed().as_secs_f64();
    eprintln!("=== total elapsed: {:.1}s ===", total_elapsed_s);

    // Final replog event: per-tier discrimination summary.
    replog.push(ReplogEvent::RunEnd {
        summary: serde_json::json!({
            "tier0_strict_converged": discrimination_strict_converged.first().map(|r| serde_json::json!({
                "n_sigma": r.n_sigma,
                "n_sigma_pct_ci": [r.n_sigma_ci_low, r.n_sigma_ci_high],
                "n_sigma_bca_ci": [r.n_sigma_bca_ci_low, r.n_sigma_bca_ci_high],
                "n_ty": r.n_ty,
                "n_schoen": r.n_schoen,
            })),
            "tier1_conservative": discrimination.first().map(|r| serde_json::json!({
                "n_sigma": r.n_sigma,
                "n_sigma_pct_ci": [r.n_sigma_ci_low, r.n_sigma_ci_high],
                "n_sigma_bca_ci": [r.n_sigma_bca_ci_low, r.n_sigma_bca_ci_high],
            })),
            "tier2_canonical": discrimination_canonical.first().map(|r| serde_json::json!({
                "n_sigma": r.n_sigma,
                "n_sigma_pct_ci": [r.n_sigma_ci_low, r.n_sigma_ci_high],
                "n_sigma_bca_ci": [r.n_sigma_bca_ci_low, r.n_sigma_bca_ci_high],
            })),
            "tier3_reference_retrim": discrimination_reference.first().map(|r| serde_json::json!({
                "n_sigma": r.n_sigma,
                "n_sigma_pct_ci": [r.n_sigma_ci_low, r.n_sigma_ci_high],
                "n_sigma_bca_ci": [r.n_sigma_bca_ci_low, r.n_sigma_bca_ci_high],
            })),
            "max_n_sigma": max_n_sigma,
            "max_n_sigma_k": max_n_sigma_k,
            "five_sigma_achieved": five_sigma_achieved,
        }),
        total_elapsed_s,
    });

    // Persist the .replog sidecar (and its .kernel.replog companion).
    let replog_path = cli.output.with_extension("replog");
    let replog_n_events = replog.events.len();
    let replog_final_chain = replog.final_chain_hex();
    if let Err(e) = replog.write_to_path(&replog_path) {
        eprintln!("WARNING: failed to write {}: {}", replog_path.display(), e);
    } else {
        eprintln!(
            "Wrote {} ({} events, final chain SHA-256 = {})",
            replog_path.display(),
            replog_n_events,
            replog_final_chain
        );
    }

    let report = EnsembleReport {
        label: "p5_10_ty_schoen_5sigma_n20".to_string(),
        path_label: "Path A: n_pts boost".to_string(),
        n_pts: cli.n_pts,
        seeds: seeds.to_vec(),
        donaldson_iters: cli.donaldson_iters,
        donaldson_tol: cli.donaldson_tol,
        compute_path: Some(if cli.use_gpu { "gpu".to_string() } else { "cpu".to_string() }),
        boot_resamples: cli.boot_resamples,
        boot_seed: cli.boot_seed,
        candidates,
        discrimination,
        discrimination_balanced_only,
        discrimination_strict_converged,
        discrimination_canonical,
        discrimination_reference,
        boot_seed_jitter,
        early_bail_seeds,
        balanced_min_iters: BALANCED_MIN_ITERS,
        converged_residual_threshold: CONVERGED_RESIDUAL_THRESHOLD,
        ambiguous_residual_threshold: AMBIGUOUS_RESIDUAL_THRESHOLD,
        max_n_sigma,
        max_n_sigma_k,
        five_sigma_achieved,
        total_elapsed_s,
        git_revision: git_revision(),
        repro_manifest: Some(repro_manifest),
        repro_log_path: Some(replog_path.display().to_string()),
        repro_log_final_chain_sha256_hex: Some(replog_final_chain),
        repro_log_n_events: Some(replog_n_events),
    };

    if let Some(parent) = cli.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).expect("create output dir");
        }
    }
    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    fs::write(&cli.output, json).expect("write JSON output");
    eprintln!("Wrote {}", cli.output.display());
}
