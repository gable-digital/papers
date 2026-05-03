//! P4.7 — Trace-free Ricci diagnostic on the post-Donaldson 22-config
//! ensemble.
//!
//! P4.5 Phase 3 found that on the n=22 sweep at k=2 the σ_L¹ functional
//! and `‖Ric_tan‖_{L²}` (full pointwise tangent-frame Ricci Frobenius
//! norm) are NEGATIVELY correlated: Pearson r = -0.55, 95% CI
//! [-0.79, -0.17]. This was unexpected — at the Calabi-Yau metric both
//! σ → 0 and Ric → 0, so at finite k they should track each other
//! downward.
//!
//! Working hypothesis: the Calabi-Yau condition is `Ric = 0`, but the
//! Einstein condition is `Ric₀ = Ric - (s/n) g = 0`, where s is the
//! scalar curvature trace. σ-functional minimisation may drive the
//! metric toward Einstein (`Ric₀ → 0`) at finite k while leaving a
//! non-zero scalar trace s, so the *full* `‖Ric‖_{L²}` does not have
//! to track σ but the trace-free `‖Ric₀‖_{L²}` should.
//!
//! This binary reproduces the P4.5 Phase 3 protocol exactly (same 22
//! configs, k=2, varied n_pts/Donaldson/Adam parameters) and at each
//! endpoint computes:
//!
//! * `sigma_L1` — Monge-Ampère L¹ residual (Stack A's σ functional).
//! * `ricci_L2_full` — `‖Ric_tan‖_{L²}` (re-uses the P4.5
//!   `ricci_norm_l2_tangent`).
//! * `ricci_L2_tracefree` — `‖Ric₀_tan‖_{L²}` where
//!   `Ric₀ = Ric_tan - (s/3) g_tan`, n=3.
//! * `scalar_L2` — `‖s‖_{L²}` (independent diagnostic).
//!
//! For each of the three pairs `(σ, full)`, `(σ, tracefree)`,
//! `(σ, scalar)` it then computes Pearson r with:
//!   * Fisher-z 95% CI (closed-form, cross-checks the bootstrap).
//!   * Bootstrap percentile + BCa 95% CIs from a paired bootstrap of n
//!     pairs.
//!
//! Output: `output/p4_7_tracefree_ricci.json` (per-seed records + the
//! three correlation tables).
//!
//! Wallclock budget: ~5–10 min on a 16-core release build. The 22
//! endpoints run sequentially because each `ricci_norm_l2_tangent` call
//! is already rayon-parallel internally.

use clap::Parser;
use cy3_rust_solver::calabi_metric::{
    ricci_norm_l2_tangent, ricci_tracefree_and_scalar_norms_l2_tangent,
};
use cy3_rust_solver::quintic::{compute_sigma_from_workspace, QuinticSolver};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Same 22-endpoint sweep as `test_p4_5_ricci_tangent_correlates_with_sigma_n20`
/// in `src/calabi_metric.rs`. Tuple is
/// `(n_pts, n_donaldson, n_adam, lr_adam, seed_offset)`. The actual
/// sampler seed is `42 + seed_offset`.
const CONFIGS_22: [(usize, usize, usize, f64, u64); 22] = [
    (1000, 60, 10, 1e-3, 1000),
    (1000, 60, 10, 1e-3, 1058),
    (1000, 60, 10, 1e-3, 1303),
    (1000, 60, 10, 1e-3, 1965),
    (1000, 60, 10, 1e-3, 1057),
    (500, 30, 0, 1e-3, 1),
    (500, 100, 0, 1e-3, 2),
    (1000, 30, 0, 1e-3, 3),
    (1000, 150, 0, 1e-3, 4),
    (1500, 60, 0, 1e-3, 5),
    (500, 60, 5, 1e-4, 6),
    (500, 60, 10, 1e-2, 7),
    (1000, 30, 20, 1e-4, 8),
    (1000, 100, 5, 1e-2, 9),
    (1000, 150, 20, 1e-3, 10),
    (1500, 30, 10, 1e-3, 11),
    (1500, 100, 10, 1e-3, 12),
    (500, 100, 40, 1e-3, 13),
    (1000, 60, 40, 1e-3, 14),
    (1000, 100, 40, 1e-2, 15),
    (1500, 60, 40, 1e-4, 16),
    (1500, 150, 40, 1e-3, 17),
];

/// Donaldson tolerance — matches P4.5 Phase 3 exactly.
const DONALDSON_TOL: f64 = 1.0e-8;
/// FD step in tangent Wirtinger coords — matches P4.5 Phase 3 exactly.
const RICCI_H_STEP: f64 = 1.0e-4;
/// k value — matches P4.5 Phase 3 exactly. (Task description's "k=4"
/// did not match the actual P4.5 protocol; k=2 is what produced the
/// r ≈ -0.55 reference number we are trying to refine.)
const K_VALUE: u32 = 2;
/// Newton tol for the sampler — matches the P4.5 test.
const NEWTON_TOL: f64 = 1.0e-8;

#[derive(Parser, Debug)]
#[command(about = "P4.7 trace-free Ricci diagnostic on the P4.5 22-endpoint sweep")]
struct Cli {
    /// Output JSON path.
    #[arg(long, default_value = "output/p4_7_tracefree_ricci.json")]
    output: PathBuf,

    /// Bootstrap n_resamples.
    #[arg(long, default_value_t = 2000)]
    boot_resamples: usize,

    /// Bootstrap seed (reproducibility).
    #[arg(long, default_value_t = 12345)]
    boot_seed: u64,

    /// Skip the heavy ‖Ric_full,tracefree‖ pass — emit only σ for
    /// debugging the protocol.
    #[arg(long, default_value_t = false)]
    sigma_only: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct SeedRecord {
    /// Index into the 22-config sweep (matches P4.5 Phase 3 ordering).
    endpoint_idx: usize,
    /// Sampler seed actually used (= 42 + seed_offset).
    seed: u64,
    n_pts: usize,
    n_donaldson: usize,
    n_donaldson_done: i64,
    n_adam: usize,
    lr_adam: f64,
    elapsed_s: f64,
    sigma_l1: f64,
    ricci_l2_full: f64,
    ricci_l2_tracefree: f64,
    scalar_l2: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct CorrelationStats {
    /// Pair label, e.g. `"sigma_L1__ricci_L2_full"`.
    pair: String,
    /// Number of valid (finite, finite) pairs used.
    n: usize,
    /// Pearson product-moment correlation.
    pearson_r: f64,
    /// Spearman rank correlation (with average-rank tie handling).
    spearman_rho: f64,
    /// Fisher-z 95% CI on Pearson r (closed-form).
    fisher_ci_low: f64,
    fisher_ci_high: f64,
    /// Paired-bootstrap percentile 95% CI.
    boot_percentile_low: f64,
    boot_percentile_high: f64,
    /// Paired-bootstrap BCa 95% CI (Efron 1987). `None` when degenerate.
    boot_bca_low: Option<f64>,
    boot_bca_high: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Report {
    label: String,
    k: u32,
    donaldson_tol: f64,
    ricci_h_step: f64,
    newton_tol: f64,
    boot_resamples: usize,
    boot_seed: u64,
    seeds: Vec<SeedRecord>,
    correlations: Vec<CorrelationStats>,
    total_elapsed_s: f64,
    git_revision: Option<String>,
}

/// SplitMix64 next-state, matching the inline RNG used by
/// `pwos_math::stats::Bootstrap` so this binary's bootstrap stream is
/// reproducibly seeded.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn rand_index(state: &mut u64, n: usize) -> usize {
    // Lemire's nearly-divisionless bounded random integer.
    let r = splitmix64(state);
    ((r as u128 * n as u128) >> 64) as usize
}

/// Pearson product-moment correlation on the supplied paired samples.
/// Returns `f64::NAN` when either coordinate has zero variance.
fn pearson_r(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    debug_assert_eq!(ys.len(), n);
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mean_x = xs.iter().sum::<f64>() / nf;
    let mean_y = ys.iter().sum::<f64>() / nf;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for i in 0..n {
        let dx = xs[i] - mean_x;
        let dy = ys[i] - mean_y;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    let denom = (sxx * syy).sqrt();
    if !denom.is_finite() || denom < 1e-30 {
        return f64::NAN;
    }
    sxy / denom
}

/// Average-rank tie-handling, identical convention to the P4.5 helper
/// `rank_with_average_ties` in `calabi_metric.rs`.
fn rank_with_average_ties(xs: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| xs[a].partial_cmp(&xs[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && xs[idx[j]] == xs[idx[i]] {
            j += 1;
        }
        // Tied group is [i..j); average rank is (i + j - 1) / 2 + 1
        // (1-based ranks).
        let avg = ((i + j - 1) as f64) * 0.5 + 1.0;
        for k in i..j {
            ranks[idx[k]] = avg;
        }
        i = j;
    }
    ranks
}

fn spearman_rho(xs: &[f64], ys: &[f64]) -> f64 {
    let rx = rank_with_average_ties(xs);
    let ry = rank_with_average_ties(ys);
    pearson_r(&rx, &ry)
}

/// Fisher-z 95% CI on Pearson r at sample size n.
fn fisher_ci_95(r: f64, n: usize) -> (f64, f64) {
    if n <= 3 {
        return (f64::NAN, f64::NAN);
    }
    let r_clamped = r.clamp(-0.999_999, 0.999_999);
    let z = 0.5 * ((1.0 + r_clamped) / (1.0 - r_clamped)).ln();
    let se = 1.0 / ((n as f64) - 3.0).sqrt();
    let z_lo = z - 1.96 * se;
    let z_hi = z + 1.96 * se;
    let inv = |z: f64| -> f64 {
        let e = (2.0 * z).exp();
        (e - 1.0) / (e + 1.0)
    };
    (inv(z_lo), inv(z_hi))
}

/// Inverse of the standard-normal CDF (Acklam 2003 rational
/// approximation). Used by the BCa CI.
fn inv_norm_cdf(p: f64) -> f64 {
    // Beasley-Springer-Moro / Acklam approximation. ~1e-9 relative error.
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    let a = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    let b = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    let c = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    let d = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

/// Standard-normal CDF via erf (numerical Abramowitz & Stegun 7.1.26).
fn norm_cdf(x: f64) -> f64 {
    // Φ(x) = 0.5 (1 + erf(x / √2)).
    let t = x / std::f64::consts::SQRT_2;
    let sign = if t < 0.0 { -1.0 } else { 1.0 };
    let a = t.abs();
    let p = 0.327_591_1;
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let tt = 1.0 / (1.0 + p * a);
    let y = 1.0
        - (((((a5 * tt + a4) * tt) + a3) * tt + a2) * tt + a1) * tt * (-(a * a)).exp();
    0.5 * (1.0 + sign * y)
}

/// Paired bootstrap of Pearson r. Returns
/// `(percentile_low, percentile_high, bca_low_or_None, bca_high_or_None)`.
fn paired_bootstrap_pearson(
    xs: &[f64],
    ys: &[f64],
    n_resamples: usize,
    seed: u64,
    ci_level: f64,
) -> (f64, f64, Option<f64>, Option<f64>) {
    let n = xs.len();
    debug_assert_eq!(ys.len(), n);
    let r_observed = pearson_r(xs, ys);

    let mut state = seed ^ 0xA5A5_5A5A_DEAD_BEEF;
    let mut x_buf = vec![0.0f64; n];
    let mut y_buf = vec![0.0f64; n];
    let mut stats: Vec<f64> = Vec::with_capacity(n_resamples);
    for _ in 0..n_resamples {
        for k in 0..n {
            let idx = rand_index(&mut state, n);
            x_buf[k] = xs[idx];
            y_buf[k] = ys[idx];
        }
        let r = pearson_r(&x_buf, &y_buf);
        if r.is_finite() {
            stats.push(r);
        }
    }
    if stats.len() < 10 {
        return (f64::NAN, f64::NAN, None, None);
    }
    stats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - ci_level;
    let lo_q = alpha * 0.5;
    let hi_q = 1.0 - lo_q;
    let percentile = |q: f64| -> f64 {
        let m = stats.len();
        let idx = ((m - 1) as f64 * q).round() as usize;
        stats[idx.min(m - 1)]
    };
    let pct_lo = percentile(lo_q);
    let pct_hi = percentile(hi_q);

    // BCa: bias-correction z0 and acceleration a from jackknife.
    let p_below = stats.iter().filter(|&&v| v < r_observed).count() as f64;
    let p_eq = stats.iter().filter(|&&v| v == r_observed).count() as f64;
    let prop = (p_below + 0.5 * p_eq) / stats.len() as f64;
    let bca = if !(0.0..1.0).contains(&prop) || !r_observed.is_finite() {
        None
    } else {
        let z0 = inv_norm_cdf(prop);

        // Jackknife pseudovalues: leave-one-out Pearson r.
        let mut jk: Vec<f64> = Vec::with_capacity(n);
        for skip in 0..n {
            let mut xs_jk: Vec<f64> = Vec::with_capacity(n - 1);
            let mut ys_jk: Vec<f64> = Vec::with_capacity(n - 1);
            for i in 0..n {
                if i != skip {
                    xs_jk.push(xs[i]);
                    ys_jk.push(ys[i]);
                }
            }
            let rj = pearson_r(&xs_jk, &ys_jk);
            if rj.is_finite() {
                jk.push(rj);
            }
        }
        if jk.len() < 3 {
            None
        } else {
            let mean_jk = jk.iter().sum::<f64>() / jk.len() as f64;
            let mut num = 0.0;
            let mut den = 0.0;
            for &v in &jk {
                let d = mean_jk - v;
                num += d * d * d;
                den += d * d;
            }
            let den_pow = den.powf(1.5);
            if !num.is_finite() || !den_pow.is_finite() || den_pow < 1e-30 {
                None
            } else {
                let acc = num / (6.0 * den_pow);
                let z_lo = inv_norm_cdf(lo_q);
                let z_hi = inv_norm_cdf(hi_q);
                let adj = |z_q: f64| -> f64 {
                    let term = z0 + (z0 + z_q) / (1.0 - acc * (z0 + z_q));
                    norm_cdf(term)
                };
                let q_lo_adj = adj(z_lo);
                let q_hi_adj = adj(z_hi);
                if q_lo_adj.is_finite()
                    && q_hi_adj.is_finite()
                    && q_lo_adj >= 0.0
                    && q_hi_adj <= 1.0
                    && q_lo_adj < q_hi_adj
                {
                    Some((percentile(q_lo_adj), percentile(q_hi_adj)))
                } else {
                    None
                }
            }
        }
    };
    let (bca_lo, bca_hi) = match bca {
        Some((lo, hi)) => (Some(lo), Some(hi)),
        None => (None, None),
    };
    (pct_lo, pct_hi, bca_lo, bca_hi)
}

fn correlation_stats(
    pair_label: &str,
    xs: &[f64],
    ys: &[f64],
    n_resamples: usize,
    seed: u64,
    ci_level: f64,
) -> CorrelationStats {
    let n = xs.len();
    let r = pearson_r(xs, ys);
    let rho = spearman_rho(xs, ys);
    let (fci_lo, fci_hi) = fisher_ci_95(r, n);
    let (boot_lo, boot_hi, bca_lo, bca_hi) =
        paired_bootstrap_pearson(xs, ys, n_resamples, seed, ci_level);
    CorrelationStats {
        pair: pair_label.to_string(),
        n,
        pearson_r: r,
        spearman_rho: rho,
        fisher_ci_low: fci_lo,
        fisher_ci_high: fci_hi,
        boot_percentile_low: boot_lo,
        boot_percentile_high: boot_hi,
        boot_bca_low: bca_lo,
        boot_bca_high: bca_hi,
    }
}

fn run_one_endpoint(
    endpoint_idx: usize,
    n_pts: usize,
    n_donaldson: usize,
    n_adam: usize,
    lr_adam: f64,
    seed: u64,
    sigma_only: bool,
) -> Option<SeedRecord> {
    let t0 = Instant::now();
    let mut solver = QuinticSolver::new(K_VALUE, n_pts, seed, NEWTON_TOL)?;
    let n_done = solver.donaldson_solve(n_donaldson, DONALDSON_TOL);
    if n_adam > 0 {
        let _ = solver.sigma_refine_analytic(n_adam, lr_adam);
    }
    let sigma_l1 = compute_sigma_from_workspace(&mut solver);

    let (ricci_full, ricci_tf, scalar) = if sigma_only {
        (f64::NAN, f64::NAN, f64::NAN)
    } else {
        // P4.7 trace-free reduction (returns all three).
        let (full, tf, sc) = ricci_tracefree_and_scalar_norms_l2_tangent(
            &solver.points,
            &solver.weights,
            solver.n_points,
            &solver.h_block,
            &solver.monomials,
            solver.n_basis,
            RICCI_H_STEP,
        );
        // Sanity cross-check: the existing P4.5 ‖Ric_full‖ helper should
        // give the same answer up to FP. We re-call it for the audit JSON
        // and keep its value as the canonical `ricci_l2_full` (so the
        // numbers in this output line up with P4.5 Phase 3).
        let full_p4_5 = ricci_norm_l2_tangent(
            &solver.points,
            &solver.weights,
            solver.n_points,
            &solver.h_block,
            &solver.monomials,
            solver.n_basis,
            RICCI_H_STEP,
        );
        if full.is_finite() && full_p4_5.is_finite() {
            let rel = ((full - full_p4_5).abs() / full_p4_5.max(1e-30)).max(0.0);
            if rel > 1e-3 {
                eprintln!(
                    "  endpoint {endpoint_idx}: WARNING ‖Ric_full‖ mismatch \
                     between P4.5 helper ({full_p4_5:.6e}) and P4.7 helper \
                     ({full:.6e}); rel diff = {rel:.2e}",
                );
            }
        }
        (full_p4_5, tf, sc)
    };
    let elapsed_s = t0.elapsed().as_secs_f64();
    Some(SeedRecord {
        endpoint_idx,
        seed,
        n_pts,
        n_donaldson,
        n_donaldson_done: n_done as i64,
        n_adam,
        lr_adam,
        elapsed_s,
        sigma_l1,
        ricci_l2_full: ricci_full,
        ricci_l2_tracefree: ricci_tf,
        scalar_l2: scalar,
    })
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

    eprintln!("=== P4.7 trace-free Ricci diagnostic ===");
    eprintln!(
        "k={K_VALUE}, donaldson_tol={DONALDSON_TOL:.0e}, ricci_h_step={RICCI_H_STEP:.0e}, \
         newton_tol={NEWTON_TOL:.0e}",
    );
    eprintln!(
        "Bootstrap: n_resamples={}, seed={}",
        cli.boot_resamples, cli.boot_seed,
    );
    eprintln!("");

    let t_total = Instant::now();
    let mut seeds: Vec<SeedRecord> = Vec::with_capacity(CONFIGS_22.len());
    for (i, &(n_pts, n_donaldson, n_adam, lr_adam, seed_off)) in
        CONFIGS_22.iter().enumerate()
    {
        let seed = 42u64.wrapping_add(seed_off);
        eprintln!(
            "[{:>2}/{}] n_pts={n_pts:>4} n_don={n_donaldson:>3} n_adam={n_adam:>2} \
             lr={lr_adam:.0e} seed={seed:>5} ...",
            i + 1,
            CONFIGS_22.len(),
        );
        match run_one_endpoint(i, n_pts, n_donaldson, n_adam, lr_adam, seed, cli.sigma_only)
        {
            Some(rec) => {
                eprintln!(
                    "    σ={:.4e}  ‖Ric‖={:.4e}  ‖Ric₀‖={:.4e}  ‖s‖={:.4e}  ({:.1}s)",
                    rec.sigma_l1,
                    rec.ricci_l2_full,
                    rec.ricci_l2_tracefree,
                    rec.scalar_l2,
                    rec.elapsed_s,
                );
                seeds.push(rec);
            }
            None => {
                eprintln!("    SKIPPED (solver init failed)");
            }
        }
    }

    // Build the three correlation tables on (σ, X) where X ∈ {full, tf, scalar}.
    let mut xs_sigma: Vec<f64> = Vec::new();
    let mut xs_full: Vec<f64> = Vec::new();
    let mut xs_tf: Vec<f64> = Vec::new();
    let mut xs_scalar: Vec<f64> = Vec::new();
    for r in &seeds {
        if r.sigma_l1.is_finite()
            && r.ricci_l2_full.is_finite()
            && r.ricci_l2_tracefree.is_finite()
            && r.scalar_l2.is_finite()
        {
            xs_sigma.push(r.sigma_l1);
            xs_full.push(r.ricci_l2_full);
            xs_tf.push(r.ricci_l2_tracefree);
            xs_scalar.push(r.scalar_l2);
        }
    }
    eprintln!(
        "\n=== Correlation analysis on n={} valid endpoints ===",
        xs_sigma.len(),
    );
    let mut correlations = Vec::new();
    for (label, ys) in [
        ("sigma_L1__ricci_L2_full", &xs_full),
        ("sigma_L1__ricci_L2_tracefree", &xs_tf),
        ("sigma_L1__scalar_L2", &xs_scalar),
    ] {
        let stats = correlation_stats(
            label,
            &xs_sigma,
            ys,
            cli.boot_resamples,
            cli.boot_seed,
            0.95,
        );
        eprintln!(
            "  {label}: r = {:.4}  (Fisher 95% CI [{:.4}, {:.4}])",
            stats.pearson_r, stats.fisher_ci_low, stats.fisher_ci_high,
        );
        eprintln!(
            "    Spearman ρ = {:.4}",
            stats.spearman_rho,
        );
        eprintln!(
            "    bootstrap percentile 95% CI = [{:.4}, {:.4}]",
            stats.boot_percentile_low, stats.boot_percentile_high,
        );
        match (stats.boot_bca_low, stats.boot_bca_high) {
            (Some(lo), Some(hi)) => eprintln!(
                "    bootstrap BCa 95% CI        = [{:.4}, {:.4}]",
                lo, hi,
            ),
            _ => eprintln!("    bootstrap BCa 95% CI        = degenerate (N/A)"),
        }
        correlations.push(stats);
    }

    let total_elapsed_s = t_total.elapsed().as_secs_f64();
    eprintln!("\n=== total elapsed: {:.1}s ===", total_elapsed_s);

    let report = Report {
        label: "p4_7_tracefree_ricci_22endpoint_k2".to_string(),
        k: K_VALUE,
        donaldson_tol: DONALDSON_TOL,
        ricci_h_step: RICCI_H_STEP,
        newton_tol: NEWTON_TOL,
        boot_resamples: cli.boot_resamples,
        boot_seed: cli.boot_seed,
        seeds,
        correlations,
        total_elapsed_s,
        git_revision: git_revision(),
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
