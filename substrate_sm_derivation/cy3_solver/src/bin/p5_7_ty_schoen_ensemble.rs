//! P5.7 — first TY-vs-Schoen σ-discrimination ensemble.
//!
//! For each (candidate, k) ∈ {(TY, 2), (TY, 3), (Schoen, 2), (Schoen, 3)}:
//!   * Run 20 distinct seeds (matching P5.3's `SEEDS_20`).
//!   * Pipeline: candidate-specific Donaldson balance only — no Adam
//!     post-refine. Adam wiring exists for the Fermat-quintic
//!     (`QuinticSolver::sigma_refine_analytic`) but not yet for the
//!     CY3 candidates; running Donaldson-only across both candidates
//!     keeps the comparison fair and isolates "raw" candidate σ.
//!   * Record σ_final per seed.
//!
//! Statistics per (candidate, k):
//!   * mean σ ± standard error = std / √n
//!   * bootstrap percentile 95% CI (n_resamples=1000, seed=12345)
//!   * bootstrap BCa 95% CI
//!
//! n-σ discrimination per k:
//!   * Δσ            = ⟨σ_TY⟩ − ⟨σ_Schoen⟩
//!   * SE_combined   = √(SE_TY² + SE_Schoen²)
//!   * **n-σ = |Δσ| / SE_combined**
//!
//! Verdict:
//!   * n-σ > 5 → 5σ discrimination achieved on σ alone (project goal).
//!   * 1 < n-σ < 5 → partial discrimination; need other channels (η,
//!     hidden bundle, Yukawa overlap, …).
//!   * n-σ < 1 → σ alone insufficient; documented as a finding.
//!
//! Output:
//!   * Stdout: human-readable per-(candidate, k) tables + n-σ summary.
//!   * JSON file at `--output` (default
//!     `output/p5_7_ty_schoen_ensemble.json`).
//!
//! Usage:
//! ```text
//!   cargo run --release --features gpu --bin p5_7_ty_schoen_ensemble
//!   cargo run --release --features gpu --bin p5_7_ty_schoen_ensemble -- \
//!       --n-pts 4000 --output output/p5_7_quick.json
//! ```

use clap::Parser;
use cy3_rust_solver::route34::cy3_metric_unified::{
    Cy3MetricSolver, Cy3MetricSpec, SchoenSolver, TianYauSolver,
};
use pwos_math::stats::bootstrap::{Bootstrap, BootstrapConfig};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// 20-seed ensemble — same list as P5.3's `SEEDS_20`, kept identical so
/// per-seed σ values are directly cross-comparable across the project.
const SEEDS_20: [u64; 20] = [
    42, 100, 12345, 7, 99, 1, 2, 3, 4, 5, 137, 271, 314, 666, 1000, 2024, 4242, 0xDEAD, 0xBEEF,
    0xCAFE,
];

#[derive(Parser, Debug)]
#[command(about = "P5.7 TY-vs-Schoen multi-seed σ ensemble + n-σ discrimination")]
struct Cli {
    /// Number of CY3 sample points per seed.
    #[arg(long, default_value_t = 10_000)]
    n_pts: usize,

    /// Comma-separated list of k values to scan (default: 2,3).
    #[arg(long, default_value = "2,3")]
    ks: String,

    /// Donaldson max iterations.
    #[arg(long, default_value_t = 25)]
    donaldson_iters: usize,

    /// Donaldson tolerance.
    #[arg(long, default_value_t = 1.0e-3)]
    donaldson_tol: f64,

    /// Bootstrap n_resamples (matches P5.3).
    #[arg(long, default_value_t = 1000)]
    boot_resamples: usize,

    /// Bootstrap seed (matches P5.3 for cross-comparability).
    #[arg(long, default_value_t = 12345)]
    boot_seed: u64,

    /// Output JSON path.
    #[arg(long, default_value = "output/p5_7_ty_schoen_ensemble.json")]
    output: PathBuf,
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
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct EnsembleReport {
    label: String,
    n_pts: usize,
    seeds: Vec<u64>,
    donaldson_iters: usize,
    donaldson_tol: f64,
    boot_resamples: usize,
    boot_seed: u64,
    candidates: Vec<CandidateKEnsemble>,
    discrimination: Vec<DiscriminationRow>,
    max_n_sigma: f64,
    max_n_sigma_k: u32,
    five_sigma_achieved: bool,
    total_elapsed_s: f64,
    git_revision: Option<String>,
}

/// Schoen (d_x, d_y, d_t) tuple for a given "k" label, matching the
/// existing crate convention (`schoen_solver_dispatches_correctly` uses
/// (3,3,1) for k=3, `schoen_publication_default` uses (4,4,2) for k=4).
fn schoen_tuple_for_k(k: u32) -> (u32, u32, u32) {
    match k {
        2 => (2, 2, 1),
        3 => (3, 3, 1),
        4 => (4, 4, 2),
        other => panic!("unsupported k for Schoen mapping: {}", other),
    }
}

fn run_ty_one(
    k: u32,
    n_pts: usize,
    seed: u64,
    donaldson_iters: usize,
    donaldson_tol: f64,
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
    let r = solver
        .solve_metric(&spec)
        .map_err(|e| format!("TY (k={k}, seed={seed}): {e}"))?;
    let s = r.summary();
    let elapsed_s = t0.elapsed().as_secs_f64();
    Ok(PerSeedRecord {
        seed,
        candidate: "TY".to_string(),
        k,
        elapsed_s,
        n_basis: s.n_basis,
        iterations_run: s.iterations_run,
        sigma_final: s.final_sigma_residual,
    })
}

fn run_schoen_one(
    k: u32,
    n_pts: usize,
    seed: u64,
    donaldson_iters: usize,
    donaldson_tol: f64,
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
    let r = solver
        .solve_metric(&spec)
        .map_err(|e| format!("Schoen (k={k} → ({d_x},{d_y},{d_t}), seed={seed}): {e}"))?;
    let s = r.summary();
    let elapsed_s = t0.elapsed().as_secs_f64();
    Ok(PerSeedRecord {
        seed,
        candidate: "Schoen".to_string(),
        k,
        elapsed_s,
        n_basis: s.n_basis,
        iterations_run: s.iterations_run,
        sigma_final: s.final_sigma_residual,
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

    eprintln!("=== P5.7 TY-vs-Schoen multi-seed σ ensemble ===");
    eprintln!(
        "n_pts={}, ks={:?}, seeds(n={})={:?}",
        cli.n_pts,
        ks,
        seeds.len(),
        seeds
    );
    eprintln!(
        "Donaldson({}, {:.0e})  (no Adam refine — fair across candidates)",
        cli.donaldson_iters, cli.donaldson_tol
    );
    eprintln!(
        "Bootstrap: n_resamples={}, seed={}",
        cli.boot_resamples, cli.boot_seed
    );
    eprintln!();

    let t_total = Instant::now();
    let mut candidates: Vec<CandidateKEnsemble> = Vec::new();

    // Order: (TY, 2), (TY, 3), (Schoen, 2), (Schoen, 3) — matches the
    // task description.
    for &candidate in &["TY", "Schoen"] {
        for &k in &ks {
            eprintln!("--- {candidate} k={k} (n_pts={}) ---", cli.n_pts);
            let mut per_seed: Vec<PerSeedRecord> = Vec::with_capacity(seeds.len());
            for (i, &seed) in seeds.iter().enumerate() {
                let res = match candidate {
                    "TY" => run_ty_one(k, cli.n_pts, seed, cli.donaldson_iters, cli.donaldson_tol),
                    "Schoen" => run_schoen_one(
                        k,
                        cli.n_pts,
                        seed,
                        cli.donaldson_iters,
                        cli.donaldson_tol,
                    ),
                    _ => unreachable!(),
                };
                match res {
                    Ok(rec) => {
                        eprintln!(
                            "  [{:2}/{}] seed={:>6}: σ={:.6}  n_basis={:>4}  iters={:>2}  ({:.2}s)",
                            i + 1,
                            seeds.len(),
                            rec.seed,
                            rec.sigma_final,
                            rec.n_basis,
                            rec.iterations_run,
                            rec.elapsed_s,
                        );
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
            // Filter non-finite / non-positive σ before stats so a single
            // pathological run can't poison the ensemble.
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

    // -- n-σ discrimination per k --------------------------------------
    let mut discrimination: Vec<DiscriminationRow> = Vec::new();
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
        let mu_ty = ty.sigma_stats.mean;
        let se_ty = ty.sigma_stats.stderr;
        let mu_sc = sc.sigma_stats.mean;
        let se_sc = sc.sigma_stats.stderr;
        let delta_sigma = mu_ty - mu_sc;
        let se_combined = (se_ty * se_ty + se_sc * se_sc).sqrt();
        let n_sigma = if se_combined > 0.0 {
            delta_sigma.abs() / se_combined
        } else {
            f64::INFINITY
        };
        if n_sigma > max_n_sigma {
            max_n_sigma = n_sigma;
            max_n_sigma_k = k;
        }
        discrimination.push(DiscriminationRow {
            k,
            mean_ty: mu_ty,
            se_ty,
            n_ty: ty.sigma_stats.n,
            mean_schoen: mu_sc,
            se_schoen: se_sc,
            n_schoen: sc.sigma_stats.n,
            delta_sigma,
            se_combined,
            n_sigma,
        });
    }

    // Print discrimination table.
    eprintln!("=== n-σ discrimination summary ===");
    eprintln!(
        "| {:>2} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8} |",
        "k", "<σ_TY>", "SE_TY", "<σ_Schoen>", "SE_Schoen", "Δσ", "SE_comb", "n-σ"
    );
    eprintln!(
        "|----|------------|------------|------------|------------|------------|------------|----------|"
    );
    for row in &discrimination {
        eprintln!(
            "| {:>2} | {:>10.6} | {:>10.4e} | {:>10.6} | {:>10.4e} | {:>+10.6} | {:>10.4e} | {:>8.3} |",
            row.k,
            row.mean_ty,
            row.se_ty,
            row.mean_schoen,
            row.se_schoen,
            row.delta_sigma,
            row.se_combined,
            row.n_sigma,
        );
    }
    let five_sigma_achieved = max_n_sigma >= 5.0;
    let verdict = if five_sigma_achieved {
        "5σ DISCRIMINATION ACHIEVED ON σ ALONE"
    } else if max_n_sigma >= 1.0 {
        "PARTIAL DISCRIMINATION — need additional channels for 5σ"
    } else {
        "σ ALONE INSUFFICIENT — candidates indistinguishable at this budget"
    };
    eprintln!();
    eprintln!(
        "max n-σ = {:.3} at k={}  →  {}",
        max_n_sigma, max_n_sigma_k, verdict
    );

    let total_elapsed_s = t_total.elapsed().as_secs_f64();
    eprintln!("=== total elapsed: {:.1}s ===", total_elapsed_s);

    let report = EnsembleReport {
        label: "p5_7_ty_schoen_n20".to_string(),
        n_pts: cli.n_pts,
        seeds: seeds.to_vec(),
        donaldson_iters: cli.donaldson_iters,
        donaldson_tol: cli.donaldson_tol,
        boot_resamples: cli.boot_resamples,
        boot_seed: cli.boot_seed,
        candidates,
        discrimination,
        max_n_sigma,
        max_n_sigma_k,
        five_sigma_achieved,
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
