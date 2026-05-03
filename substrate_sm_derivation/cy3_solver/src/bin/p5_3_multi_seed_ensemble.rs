//! P5.3 — Multi-seed σ ensemble with bootstrap CIs.
//!
//! Addresses §2.2 of the hostile review of P3.13: every "canonical
//! reference σ" pinned at single seeds is statistically meaningless
//! without a distribution. Spread across 3 seeds was ~0.7%; that is
//! NOT what we should report as a science value.
//!
//! For each (k, n_pts) ∈ {(2, 10000), (3, 10000), (4, 10000)}:
//! * Run 20 distinct seeds.
//! * For each seed: build solver → orthonormalise FS-Gram (h ← I) →
//!   record σ_FS_identity → Donaldson(50, 1e-10) → record σ_donaldson →
//!   sigma_refine_analytic(20, 1e-3) → record σ_post_refine (min over
//!   history, matching P3.10 protocol).
//!
//! Statistics per (k, checkpoint, n_pts):
//! * mean σ, sample std, sample standard error = std / √n
//! * bootstrap percentile 95% CI (n_resamples = 1000, seed = 12345)
//! * bootstrap BCa 95% CI
//!
//! AKO 2010 comparison (post-refine only):
//! * σ_AKO(k) = 3.51/k² − 5.19/k³
//! * relative deviation from mean
//! * AKO inside 95% CI (yes/no)
//!
//! Output:
//! * Stdout: human-readable table per checkpoint.
//! * JSON file at `--output` (default `output/p5_3_multi_seed_ensemble.json`).
//!
//! Usage:
//! ```text
//!   cargo run --release --features gpu --bin p5_3_multi_seed_ensemble -- \
//!       --n-pts 10000 --output output/p5_3_multi_seed_ensemble.json
//!   cargo run --release --features gpu --bin p5_3_multi_seed_ensemble -- \
//!       --quick   # 5 seeds × k=2 only, smoke run
//! ```

use clap::Parser;
use cy3_rust_solver::quintic::{QuinticSolver, SamplerKind};
use pwos_math::stats::bootstrap::{Bootstrap, BootstrapConfig};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// 20-seed ensemble used for the science value. Hand-picked spread of
/// magic numbers + arbitrary integers + popular constants so each seed
/// drives a distinguishable RNG trajectory. SplitMix64 / ChaCha state is
/// uncorrelated across these inputs.
const SEEDS_20: [u64; 20] = [
    42, 100, 12345, 7, 99, 1, 2, 3, 4, 5,
    137, 271, 314, 666, 1000, 2024, 4242,
    0xDEAD, 0xBEEF, 0xCAFE,
];

/// Quick-mode subset: first 5 seeds. Used for the daily-CI variant
/// of the regression test (runtime < 2 min at k=2, n_pts=10000).
const SEEDS_5: [u64; 5] = [42, 100, 12345, 7, 99];

#[derive(Parser, Debug)]
#[command(about = "P5.3 multi-seed σ ensemble with bootstrap CIs")]
struct Cli {
    /// Number of CY sample points per seed.
    #[arg(long, default_value_t = 10_000)]
    n_pts: usize,

    /// Comma-separated list of k values to scan (default: 2,3,4).
    #[arg(long, default_value = "2,3,4")]
    ks: String,

    /// Run quick mode (5 seeds, k=2 only).
    #[arg(long, default_value_t = false)]
    quick: bool,

    /// Donaldson max iterations.
    #[arg(long, default_value_t = 50)]
    donaldson_iters: usize,

    /// Donaldson tolerance.
    #[arg(long, default_value_t = 1.0e-10)]
    donaldson_tol: f64,

    /// Adam refine iterations.
    #[arg(long, default_value_t = 20)]
    refine_iters: usize,

    /// Adam refine learning rate.
    #[arg(long, default_value_t = 1.0e-3)]
    refine_lr: f64,

    /// Newton tolerance for sampler.
    #[arg(long, default_value_t = 1.0e-12)]
    newton_tol: f64,

    /// Bootstrap n_resamples.
    #[arg(long, default_value_t = 1000)]
    boot_resamples: usize,

    /// Bootstrap seed (reproducibility of the CI itself).
    #[arg(long, default_value_t = 12345)]
    boot_seed: u64,

    /// Output JSON path.
    #[arg(long, default_value = "output/p5_3_multi_seed_ensemble.json")]
    output: PathBuf,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct PerSeedRecord {
    seed: u64,
    elapsed_s: f64,
    sigma_fs_identity: f64,
    sigma_donaldson: f64,
    sigma_post_refine: f64,
    refine_min_idx: usize,
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
struct AkoComparison {
    sigma_ako: f64,
    rel_dev_from_mean: f64,
    inside_percentile_ci: bool,
    inside_bca_ci: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct KEnsemble {
    k: u32,
    n_pts: usize,
    n_basis: usize,
    seeds: Vec<u64>,
    per_seed: Vec<PerSeedRecord>,
    fs_identity: CheckpointStats,
    donaldson: CheckpointStats,
    post_refine: CheckpointStats,
    ako_comparison: AkoComparison,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct EnsembleReport {
    label: String,
    n_pts: usize,
    seeds: Vec<u64>,
    donaldson_iters: usize,
    donaldson_tol: f64,
    refine_iters: usize,
    refine_lr: f64,
    newton_tol: f64,
    boot_resamples: usize,
    boot_seed: u64,
    sampler: String,
    ks: Vec<KEnsemble>,
    total_elapsed_s: f64,
    git_revision: Option<String>,
}

fn ako_2010(k: u32) -> f64 {
    let kf = k as f64;
    3.51 / (kf * kf) - 5.19 / (kf * kf * kf)
}

fn run_one_seed(
    k: u32,
    n_pts: usize,
    seed: u64,
    newton_tol: f64,
    donaldson_iters: usize,
    donaldson_tol: f64,
    refine_iters: usize,
    refine_lr: f64,
) -> Result<PerSeedRecord, String> {
    let t0 = Instant::now();
    let mut solver = QuinticSolver::new_with_sampler(
        k,
        n_pts,
        seed,
        newton_tol,
        SamplerKind::ShiffmanZelditch,
    )
    .ok_or_else(|| format!("workspace construction failed (k={k}, seed={seed})"))?;
    solver
        .orthonormalise_basis_fs_gram()
        .map_err(|e| format!("Cholesky failed (k={k}, seed={seed}): {e}"))?;
    // After orthonormalisation, h is identity in the new basis.
    let sigma_fs_identity = solver.sigma();
    solver.donaldson_solve(donaldson_iters, donaldson_tol);
    let sigma_donaldson = solver.sigma();
    let history = solver.sigma_refine_analytic(refine_iters, refine_lr);
    // Per P3.10 protocol: take the running minimum over (donaldson, history).
    let mut refine_min_idx = 0_usize;
    let mut sigma_post_refine = sigma_donaldson;
    for (i, &v) in history.iter().enumerate() {
        if v < sigma_post_refine {
            sigma_post_refine = v;
            refine_min_idx = i + 1; // 0 = post-Donaldson, 1.. = history index+1
        }
    }
    let elapsed_s = t0.elapsed().as_secs_f64();
    Ok(PerSeedRecord {
        seed,
        elapsed_s,
        sigma_fs_identity,
        sigma_donaldson,
        sigma_post_refine,
        refine_min_idx,
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
        "    {:>22}: mean={:.6} std={:.4e} stderr={:.4e}  pct95=[{:.6}, {:.6}]  bca95=[{}, {}]",
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

fn ako_status(stats: &CheckpointStats, sigma_ako: f64) -> AkoComparison {
    let rel_dev_from_mean = (stats.mean - sigma_ako) / sigma_ako;
    let inside_percentile_ci =
        sigma_ako >= stats.percentile_ci_low && sigma_ako <= stats.percentile_ci_high;
    let inside_bca_ci = match (stats.bca_ci_low, stats.bca_ci_high) {
        (Some(lo), Some(hi)) => Some(sigma_ako >= lo && sigma_ako <= hi),
        _ => None,
    };
    AkoComparison {
        sigma_ako,
        rel_dev_from_mean,
        inside_percentile_ci,
        inside_bca_ci,
    }
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
    let (label, seeds): (&'static str, &[u64]) = if cli.quick {
        ("p5_3_quick_5seeds", SEEDS_5.as_slice())
    } else {
        ("p5_3_full_20seeds", SEEDS_20.as_slice())
    };
    let ks: Vec<u32> = if cli.quick {
        vec![2]
    } else {
        cli.ks
            .split(',')
            .map(|s| s.trim().parse::<u32>().expect("invalid k"))
            .collect()
    };

    eprintln!("=== P5.3 multi-seed σ ensemble ===");
    eprintln!(
        "label={}, n_pts={}, ks={:?}, seeds(n={})={:?}",
        label,
        cli.n_pts,
        ks,
        seeds.len(),
        seeds
    );
    eprintln!(
        "Donaldson({}, {:.0e}) -> sigma_refine_analytic({}, {:.0e})",
        cli.donaldson_iters, cli.donaldson_tol, cli.refine_iters, cli.refine_lr
    );
    eprintln!("Bootstrap: n_resamples={}, seed={}", cli.boot_resamples, cli.boot_seed);
    eprintln!("");

    let t_total = Instant::now();
    let mut k_ensembles: Vec<KEnsemble> = Vec::new();

    for &k in &ks {
        let n_basis =
            cy3_rust_solver::quintic::build_degree_k_quintic_monomials(k).len();
        eprintln!("--- k={k} (n_basis={n_basis}, n_pts={}) ---", cli.n_pts);
        let mut per_seed: Vec<PerSeedRecord> = Vec::with_capacity(seeds.len());
        for (i, &seed) in seeds.iter().enumerate() {
            match run_one_seed(
                k,
                cli.n_pts,
                seed,
                cli.newton_tol,
                cli.donaldson_iters,
                cli.donaldson_tol,
                cli.refine_iters,
                cli.refine_lr,
            ) {
                Ok(rec) => {
                    eprintln!(
                        "  [{:2}/{}] seed={:>6}: σ_FS={:.6}  σ_D={:.6}  σ_R={:.6}  ({:.2}s)",
                        i + 1,
                        seeds.len(),
                        rec.seed,
                        rec.sigma_fs_identity,
                        rec.sigma_donaldson,
                        rec.sigma_post_refine,
                        rec.elapsed_s,
                    );
                    per_seed.push(rec);
                }
                Err(e) => {
                    eprintln!("  [{:2}/{}] seed={:>6}: SKIPPED — {e}", i + 1, seeds.len(), seed);
                }
            }
        }
        if per_seed.is_empty() {
            eprintln!("  k={k}: no seeds succeeded; skipping stats");
            continue;
        }
        let fs_vals: Vec<f64> = per_seed.iter().map(|r| r.sigma_fs_identity).collect();
        let don_vals: Vec<f64> = per_seed.iter().map(|r| r.sigma_donaldson).collect();
        let refine_vals: Vec<f64> = per_seed.iter().map(|r| r.sigma_post_refine).collect();
        let fs_stats = compute_stats(
            "sigma_fs_identity",
            &fs_vals,
            cli.boot_resamples,
            cli.boot_seed,
            0.95,
        );
        let don_stats = compute_stats(
            "sigma_donaldson",
            &don_vals,
            cli.boot_resamples,
            cli.boot_seed,
            0.95,
        );
        let refine_stats = compute_stats(
            "sigma_post_refine",
            &refine_vals,
            cli.boot_resamples,
            cli.boot_seed,
            0.95,
        );
        eprintln!("  Stats:");
        print_checkpoint(&fs_stats);
        print_checkpoint(&don_stats);
        print_checkpoint(&refine_stats);
        let sigma_ako = ako_2010(k);
        let ako_cmp = ako_status(&refine_stats, sigma_ako);
        eprintln!(
            "  AKO 2010: σ_AKO({}) = {:.6}, rel_dev_from_mean = {:+.4}",
            k, ako_cmp.sigma_ako, ako_cmp.rel_dev_from_mean
        );
        eprintln!(
            "    inside percentile 95% CI: {}, inside BCa 95% CI: {}",
            ako_cmp.inside_percentile_ci,
            match ako_cmp.inside_bca_ci {
                Some(true) => "true",
                Some(false) => "false",
                None => "n/a",
            },
        );
        eprintln!("");
        k_ensembles.push(KEnsemble {
            k,
            n_pts: cli.n_pts,
            n_basis,
            seeds: seeds.to_vec(),
            per_seed,
            fs_identity: fs_stats,
            donaldson: don_stats,
            post_refine: refine_stats,
            ako_comparison: ako_cmp,
        });
    }

    let total_elapsed_s = t_total.elapsed().as_secs_f64();
    eprintln!("=== total elapsed: {:.1}s ===", total_elapsed_s);

    let report = EnsembleReport {
        label: label.to_string(),
        n_pts: cli.n_pts,
        seeds: seeds.to_vec(),
        donaldson_iters: cli.donaldson_iters,
        donaldson_tol: cli.donaldson_tol,
        refine_iters: cli.refine_iters,
        refine_lr: cli.refine_lr,
        newton_tol: cli.newton_tol,
        boot_resamples: cli.boot_resamples,
        boot_seed: cli.boot_seed,
        sampler: "ShiffmanZelditch".to_string(),
        ks: k_ensembles,
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
