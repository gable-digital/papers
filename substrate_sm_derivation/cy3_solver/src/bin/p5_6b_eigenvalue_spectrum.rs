//! P5.6b — Δ-eigenvalue spectrum diagnostic for the harmonic-zero-mode
//! solver on the Tian-Yau Z/3 CY3 / AKLP bundle pair.
//!
//! Goal: determine, on the same family of σ-endpoints used by
//! `p5_6_yukawa_propagation`, the actual eigenvalue distribution of
//! the genuine ∂̄ Laplacian. From this we can decide whether the
//! "empty kernel" failure of the legacy
//! `HarmonicConfig::kernel_eigenvalue_ratio = 1e-3` cutoff is
//!
//!   (a) plateau-then-gap that simply needs a better threshold,
//!   (b) a smooth spectrum that needs a fixed-N strategy where N is
//!       the BBW cohomology dimension (= 9 for AKLP), or
//!   (c) a structured spectrum lifted off zero by Bergman-kernel
//!       residual where the proper fix is full Hodge decomposition.
//!
//! ## Method
//!
//! For each endpoint (σ-endpoint label, Donaldson `max_iter`):
//!   1. Solve the TY metric.
//!   2. Build the wrapper background.
//!   3. Solve HYM (same config as p5_6).
//!   4. Call `solve_harmonic_zero_modes(... HarmonicConfig::default())`.
//!   5. Print the **full Δ-eigenvalue spectrum** plus the BBW count
//!      `cohomology_dim_predicted`.
//!
//! ## Reduced sweep
//!
//! Runs only 3 endpoints (FS-Gram-style baseline at small budget,
//! mid-budget, full Donaldson) to keep wall clock under the task
//! budget. The full 7-endpoint sweep can be re-enabled via
//! `--full-sweep`.
//!
//! Output:
//!   * Stdout: per-endpoint header + ascending-sorted spectrum.
//!   * JSON file at `--output` (default
//!     `output/p5_6b_eigenvalue_spectrum.json`).

use clap::Parser;
use cy3_rust_solver::route34::cy3_metric_unified::{
    Cy3MetricResultKind, Cy3MetricSolver, Cy3MetricSpec, TianYauSolver,
};
use cy3_rust_solver::route34::hym_hermitian::{solve_hym_metric, HymConfig};
use cy3_rust_solver::route34::yukawa_pipeline::Cy3MetricResultBackground;
use cy3_rust_solver::route34::zero_modes_harmonic::{
    solve_harmonic_zero_modes, HarmonicConfig,
};
use cy3_rust_solver::zero_modes::{AmbientCY3, MonadBundle};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// (label, max_iter, donaldson_tol)
const ENDPOINTS_REDUCED: &[(&str, usize, f64)] = &[
    ("budget_1",   1,  1.0e-9),
    ("budget_10", 10,  1.0e-9),
    ("budget_50", 50,  1.0e-9),
];

const ENDPOINTS_FULL: &[(&str, usize, f64)] = &[
    ("budget_1",   1,  1.0e-9),
    ("budget_2",   2,  1.0e-9),
    ("budget_3",   3,  1.0e-9),
    ("budget_5",   5,  1.0e-9),
    ("budget_10", 10,  1.0e-9),
    ("budget_25", 25,  1.0e-9),
    ("budget_50", 50,  1.0e-9),
];

#[derive(Parser, Debug)]
#[command(about = "P5.6b Δ-eigenvalue spectrum diagnostic (Tian-Yau k=2 / AKLP)")]
struct Cli {
    #[arg(long, default_value_t = 200)]
    n_pts: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = 2)]
    k: u32,

    /// Run all 7 endpoints instead of the 3-endpoint reduced sweep.
    #[arg(long, default_value_t = false)]
    full_sweep: bool,

    #[arg(long, default_value = "output/p5_6b_eigenvalue_spectrum.json")]
    output: PathBuf,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct EndpointSpectrum {
    label: String,
    max_iter: usize,
    donaldson_tol: f64,
    sigma: f64,
    iterations_run: usize,
    seed_basis_dim: usize,
    cohomology_dim_predicted: usize,
    cohomology_dim_observed_default: usize,
    eigenvalues_full: Vec<f64>,
    eigenvalue_min: f64,
    eigenvalue_max: f64,
    /// Largest ratio λ_{i+1} / λ_i for i in [0, n/2). A clear plateau-
    /// then-gap shows up here as a value ≫ 1 at i = (kernel_dim - 1).
    largest_lower_half_gap: f64,
    largest_lower_half_gap_index: usize,
    elapsed_s: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct SpectrumReport {
    label: &'static str,
    geometry: &'static str,
    bundle: &'static str,
    n_pts: usize,
    seed: u64,
    k: u32,
    endpoints: Vec<EndpointSpectrum>,
    total_elapsed_s: f64,
}

fn run_endpoint(
    label: &str,
    n_pts: usize,
    k: u32,
    seed: u64,
    max_iter: usize,
    donaldson_tol: f64,
    bundle: &MonadBundle,
    ambient: &AmbientCY3,
) -> Result<EndpointSpectrum, String> {
    let t0 = Instant::now();
    let solver = TianYauSolver;
    let spec = Cy3MetricSpec::TianYau {
        k,
        n_sample: n_pts,
        max_iter,
        donaldson_tol,
        seed,
    };

    let r = solver
        .solve_metric(&spec)
        .map_err(|e| format!("{label}: TY solve failed: {e}"))?;
    let summary = r.summary();
    let sigma = summary.final_sigma_residual;

    let bg = match &r {
        Cy3MetricResultKind::TianYau(t) => Cy3MetricResultBackground::from_ty(t.as_ref()),
        Cy3MetricResultKind::Schoen(_) => {
            return Err(format!("{label}: TY solver returned non-TY result"))
        }
    };

    let hym_cfg = HymConfig {
        max_iter: 8,
        damping: 0.5,
        ..HymConfig::default()
    };
    let h_v = solve_hym_metric(bundle, &bg, &hym_cfg);

    // Default config: the legacy `kernel_eigenvalue_ratio = 1e-3` path.
    let cfg = HarmonicConfig::default();
    let res = solve_harmonic_zero_modes(bundle, ambient, &bg, &h_v, &cfg);

    let n = res.eigenvalues_full.len();
    let (eigenvalue_min, eigenvalue_max) = if n == 0 {
        (f64::NAN, f64::NAN)
    } else {
        (res.eigenvalues_full[0], res.eigenvalues_full[n - 1])
    };

    // Largest gap in the lower half of the spectrum (search region for
    // a plateau-then-gap structure).
    let half = (n / 2).max(1);
    let mut largest_gap = 1.0_f64;
    let mut largest_gap_idx = 0_usize;
    for i in 0..half.saturating_sub(1) {
        let lo = res.eigenvalues_full[i].abs().max(1.0e-300);
        let hi = res.eigenvalues_full[i + 1].abs().max(1.0e-300);
        let ratio = hi / lo;
        if ratio.is_finite() && ratio > largest_gap {
            largest_gap = ratio;
            largest_gap_idx = i;
        }
    }

    Ok(EndpointSpectrum {
        label: label.to_string(),
        max_iter,
        donaldson_tol,
        sigma,
        iterations_run: summary.iterations_run,
        seed_basis_dim: res.seed_basis_dim,
        cohomology_dim_predicted: res.cohomology_dim_predicted,
        cohomology_dim_observed_default: res.cohomology_dim_observed,
        eigenvalues_full: res.eigenvalues_full,
        eigenvalue_min,
        eigenvalue_max,
        largest_lower_half_gap: largest_gap,
        largest_lower_half_gap_index: largest_gap_idx,
        elapsed_s: t0.elapsed().as_secs_f64(),
    })
}

fn main() {
    let cli = Cli::parse();
    let endpoints: &[(&str, usize, f64)] = if cli.full_sweep {
        ENDPOINTS_FULL
    } else {
        ENDPOINTS_REDUCED
    };

    eprintln!(
        "=== P5.6b eigenvalue-spectrum diagnostic (Tian-Yau k={}, n_pts={}, seed={}) ===",
        cli.k, cli.n_pts, cli.seed
    );
    eprintln!(
        "Sweep: {} ({} endpoints)",
        if cli.full_sweep { "full" } else { "reduced" },
        endpoints.len()
    );
    eprintln!();

    let bundle = MonadBundle::anderson_lukas_palti_example();
    let ambient = AmbientCY3::tian_yau_upstairs();

    let t_total = Instant::now();
    let mut records: Vec<EndpointSpectrum> = Vec::new();
    for (label, max_iter, tol) in endpoints {
        eprintln!("--- {label}: max_iter={max_iter}, tol={tol:.0e} ---");
        match run_endpoint(label, cli.n_pts, cli.k, cli.seed, *max_iter, *tol, &bundle, &ambient) {
            Ok(rec) => {
                eprintln!(
                    "  σ={:.6e}  iters={}  basis_dim={}  BBW_dim={}  observed_default={}",
                    rec.sigma,
                    rec.iterations_run,
                    rec.seed_basis_dim,
                    rec.cohomology_dim_predicted,
                    rec.cohomology_dim_observed_default
                );
                eprintln!(
                    "  λ_min={:.3e}  λ_max={:.3e}  ratio_max/min={:.3e}",
                    rec.eigenvalue_min,
                    rec.eigenvalue_max,
                    rec.eigenvalue_max / rec.eigenvalue_min.abs().max(1.0e-300)
                );
                eprintln!(
                    "  largest lower-half gap λ_{{i+1}}/λ_i = {:.3e} at i={}",
                    rec.largest_lower_half_gap, rec.largest_lower_half_gap_index
                );
                let n_print = rec.eigenvalues_full.len().min(20);
                eprint!("  spectrum[0..{}]: ", n_print);
                for v in &rec.eigenvalues_full[..n_print] {
                    eprint!("{:.3e} ", v);
                }
                eprintln!();
                records.push(rec);
            }
            Err(e) => {
                eprintln!("  SKIPPED: {e}");
            }
        }
    }

    let total_elapsed_s = t_total.elapsed().as_secs_f64();
    let report = SpectrumReport {
        label: "p5_6b_eigenvalue_spectrum",
        geometry: "tian_yau_z3_bicubic",
        bundle: "anderson_lukas_palti_example",
        n_pts: cli.n_pts,
        seed: cli.seed,
        k: cli.k,
        endpoints: records,
        total_elapsed_s,
    };

    if let Some(parent) = cli.output.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let json = serde_json::to_string_pretty(&report).expect("serde_json::to_string_pretty");
    fs::write(&cli.output, json).expect("write output JSON");
    eprintln!();
    eprintln!("=== Wrote {} ===", cli.output.display());
    eprintln!("Total elapsed: {:.1}s", total_elapsed_s);
}
