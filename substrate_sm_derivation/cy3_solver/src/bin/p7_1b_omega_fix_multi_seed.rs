// P7.1b — ω_fix gateway-electron multi-seed sweep on confirmed-converged
// Schoen Donaldson metrics.
//
// Background. P7.1 (commit db1b4ca1) ran the diagnostic on Schoen seed=42 and
// reported a 1.92% residual from ω_fix = 123/248. The follow-up question:
// is that 1.92% a feature of seed=42's particular Donaldson noise, or is it
// a robust property of the Schoen Z₃×Z₃ Donaldson metric across the
// strict-converged seed set?
//
// Protocol audit (P7.1 binary, default args).
//   n_pts=40_000, k=3, max_iter=100, donaldson_tol=1e-6, test_degree=4,
//   seed=42 (single).
//
// Per `output/p5_10_ty_schoen_5sigma.json::candidates[Schoen]::per_seed`,
// seed=42 has Donaldson residual 9.35e-7 < tol=1e-6 in 26 iterations —
// strictly converged. The σ_residual=8.20 reported by P7.1 is the
// Monge-Ampère sigma spread (natural scale of the η-statistic on Schoen,
// 2-13 across the converged set), NOT a Donaldson non-convergence
// indicator. So seed=42 was a fine pick — the 1.92% is not a stale-cache
// artifact.
//
// What this binary does. Re-run the P7.1 diagnostic across the five cleanest
// strict-converged Schoen seeds (lowest residual + lowest iters):
//
//   seed=48879  iters=20  residual=5.04e-7
//   seed=    2  iters=20  residual=5.39e-7
//   seed=12345  iters=20  residual=5.50e-7
//   seed=51966  iters=18  residual=5.69e-7
//   seed=    4  iters=22  residual=5.82e-7
//
// Tabulate the lowest dominant-trivial-rep eigenvalue across seeds and
// compute mean ± SE. Compare to ω_fix = 123/248.
//
// We retain the deprecated `compute_metric_laplacian_spectrum` API for
// the Galerkin solve (the deprecation is over the chain-match consumer,
// not the eigenvalue solver itself).
#![allow(deprecated)]

use clap::Parser;
use cy3_rust_solver::route34::cy3_metric_unified::{
    Cy3MetricResultKind, Cy3MetricSolver, Cy3MetricSpec, SchoenSolver,
};
use cy3_rust_solver::route34::metric_laplacian::{
    compute_metric_laplacian_spectrum, MetricLaplacianConfig,
};
use cy3_rust_solver::route34::yukawa_pipeline::Cy3MetricResultBackground;
use cy3_rust_solver::route34::z3xz3_projector::{alpha_character, beta_character};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Predicted gateway-electron eigenvalue: `1/2 − 1/dim(E_8) = 123/248`.
const E8_DIM: u32 = 248;
fn predicted_omega_fix() -> f64 {
    0.5 - 1.0 / (E8_DIM as f64)
}

#[derive(Parser, Debug)]
#[command(about = "P7.1b ω_fix multi-seed sweep on converged Schoen seeds")]
struct Cli {
    /// Sample-cloud size (canonical: 40_000).
    #[arg(long, default_value_t = 40_000)]
    n_pts: usize,

    /// Bigraded section-basis degree (canonical: 3).
    #[arg(long, default_value_t = 3)]
    k: u32,

    /// Donaldson iteration cap (canonical: 100).
    #[arg(long, default_value_t = 100)]
    max_iter: usize,

    /// Donaldson convergence tolerance (canonical: 1e-6).
    #[arg(long, default_value_t = 1.0e-6)]
    donaldson_tol: f64,

    /// Comma-separated PRNG seed list. Default: five cleanest
    /// strict-converged Schoen seeds at n_pts=40k.
    #[arg(long, default_value = "48879,2,12345,51966,4")]
    seeds: String,

    /// Maximum total degree of the test-function basis.
    #[arg(long, default_value_t = 4)]
    test_degree: u32,

    /// Output JSON path.
    #[arg(long, default_value = "output/p7_1b_omega_fix_converged_seeds.json")]
    output: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerSeed {
    seed: u64,
    metric_iters: usize,
    metric_sigma_residual: f64,
    metric_seconds: f64,
    spectrum_seconds: f64,
    basis_dim: usize,
    n_points: usize,
    /// Lowest dominant-trivial-rep raw eigenvalue (filtered: dom-χ=(0,0), λ>1e-10).
    lowest_trivial_eigenvalue: f64,
    /// L²-fraction of |c_i|² supported on (α=0,β=0) monomials of that
    /// eigenvector.
    trivial_rep_purity: f64,
    /// Index in |·|-sorted full spectrum.
    full_index: usize,
    /// |λ − 123/248| / (123/248) in ppm.
    residual_ppm_from_omega_fix: f64,
    /// Donaldson final residual reported by the metric solver
    /// (sanity: should be < tol).
    final_donaldson_residual: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Aggregate {
    n_seeds: usize,
    mean_lowest_trivial_eigenvalue: f64,
    se_lowest_trivial_eigenvalue: f64,
    std_lowest_trivial_eigenvalue: f64,
    spread_pct: f64,
    mean_residual_ppm_from_omega_fix: f64,
    /// Residual of the *mean* eigenvalue (not the mean of per-seed residuals).
    residual_of_mean_ppm: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Report {
    label: &'static str,
    e8_dim: u32,
    predicted_omega_fix: f64,
    n_pts: usize,
    k: u32,
    max_iter: usize,
    donaldson_tol: f64,
    test_degree: u32,
    seeds: Vec<u64>,
    per_seed: Vec<PerSeed>,
    aggregate: Aggregate,
    verdict: String,
    total_elapsed_s: f64,
}

fn classify_eigenvector(
    evec_real_imag: &[(f64, f64)],
    n_b: usize,
    col: usize,
    chi: &[(u32, u32)],
) -> ([f64; 9], (u32, u32), f64) {
    let mut weights = [0.0_f64; 9];
    let mut total = 0.0_f64;
    for i in 0..n_b {
        let (re, im) = evec_real_imag[i * n_b + col];
        let mag2 = re * re + im * im;
        if !mag2.is_finite() || mag2 == 0.0 {
            continue;
        }
        let (a, b) = chi[i];
        let bucket = (a * 3 + b) as usize;
        weights[bucket] += mag2;
        total += mag2;
    }
    if total > 0.0 {
        for w in weights.iter_mut() {
            *w /= total;
        }
    }
    let mut best_b = 0usize;
    let mut best_w = weights[0];
    for k in 1..9 {
        if weights[k] > best_w {
            best_w = weights[k];
            best_b = k;
        }
    }
    let dom = ((best_b / 3) as u32, (best_b % 3) as u32);
    (weights, dom, weights[0])
}

fn run_seed(seed: u64, cli: &Cli, omega_fix: f64) -> Result<PerSeed, String> {
    let spec = Cy3MetricSpec::Schoen {
        d_x: cli.k,
        d_y: cli.k,
        d_t: 1,
        n_sample: cli.n_pts,
        max_iter: cli.max_iter,
        donaldson_tol: cli.donaldson_tol,
        seed,
    };
    eprintln!("\n--- Schoen seed={} ---", seed);

    let t_metric = Instant::now();
    let r = SchoenSolver
        .solve_metric(&spec)
        .map_err(|e| format!("seed={seed}: metric solve failed: {e}"))?;
    let summary = r.summary();
    let metric_seconds = t_metric.elapsed().as_secs_f64();
    let final_residual = summary.final_donaldson_residual;
    eprintln!(
        "  metric: iters={} σ-residual={:.3e} donaldson-residual={:.3e} t={:.1}s",
        summary.iterations_run, summary.final_sigma_residual, final_residual, metric_seconds
    );

    let bg = match &r {
        Cy3MetricResultKind::Schoen(t) => Cy3MetricResultBackground::from_schoen(t.as_ref()),
        _ => return Err(format!("seed={seed}: expected Schoen result")),
    };

    let cfg = MetricLaplacianConfig {
        max_total_degree: cli.test_degree,
        n_low_eigenvalues: 50,
        return_eigenvectors: true,
        ..MetricLaplacianConfig::default()
    };
    let t_spec = Instant::now();
    let spectrum = compute_metric_laplacian_spectrum(&bg, &cfg);
    let spectrum_seconds = t_spec.elapsed().as_secs_f64();

    let basis = spectrum
        .basis_exponents
        .as_ref()
        .ok_or_else(|| format!("seed={seed}: basis_exponents missing"))?;
    let evecs = spectrum
        .eigenvectors_full
        .as_ref()
        .ok_or_else(|| format!("seed={seed}: eigenvectors_full missing"))?;
    let n_b = spectrum.basis_dim;
    if n_b == 0 {
        return Err(format!("seed={seed}: empty basis"));
    }
    if spectrum.eigenvalues_full.len() != n_b || evecs.len() != n_b * n_b {
        return Err(format!(
            "seed={seed}: spectrum/eigenvector size mismatch ({} eigs, {} evec entries, n_b={})",
            spectrum.eigenvalues_full.len(),
            evecs.len(),
            n_b
        ));
    }

    let chi: Vec<(u32, u32)> = basis
        .iter()
        .map(|m| (alpha_character(m), beta_character(m)))
        .collect();

    let evals = &spectrum.eigenvalues_full;
    let mut classified: Vec<(usize, f64, (u32, u32), f64)> = (0..n_b)
        .map(|j| {
            let (_w, dom, triv) = classify_eigenvector(evecs, n_b, j, &chi);
            (j, evals[j], dom, triv)
        })
        .collect();
    classified.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let pick = classified
        .iter()
        .find(|(_, ev, dom, _)| *dom == (0, 0) && *ev > 1.0e-10)
        .ok_or_else(|| format!("seed={seed}: no trivial-rep eigenvalue above 1e-10 floor"))?;

    let (full_index, eigenvalue, _dom, trivial_purity) = (pick.0, pick.1, pick.2, pick.3);
    let residual_ppm = 1.0e6 * (eigenvalue - omega_fix).abs() / omega_fix.abs().max(1.0e-300);

    eprintln!(
        "  trivial-rep λ_min = {:.10}  ({:.4} ppm from ω_fix={:.10})  purity={:.4}",
        eigenvalue, residual_ppm, omega_fix, trivial_purity
    );

    Ok(PerSeed {
        seed,
        metric_iters: summary.iterations_run,
        metric_sigma_residual: summary.final_sigma_residual,
        metric_seconds,
        spectrum_seconds,
        basis_dim: n_b,
        n_points: spectrum.n_points,
        lowest_trivial_eigenvalue: eigenvalue,
        trivial_rep_purity: trivial_purity,
        full_index,
        residual_ppm_from_omega_fix: residual_ppm,
        final_donaldson_residual: final_residual,
    })
}

fn aggregate(rows: &[PerSeed], omega_fix: f64) -> Aggregate {
    let n = rows.len();
    if n == 0 {
        return Aggregate {
            n_seeds: 0,
            mean_lowest_trivial_eigenvalue: f64::NAN,
            se_lowest_trivial_eigenvalue: f64::NAN,
            std_lowest_trivial_eigenvalue: f64::NAN,
            spread_pct: f64::NAN,
            mean_residual_ppm_from_omega_fix: f64::NAN,
            residual_of_mean_ppm: f64::NAN,
        };
    }
    let mean = rows.iter().map(|r| r.lowest_trivial_eigenvalue).sum::<f64>() / (n as f64);
    let var = if n > 1 {
        rows.iter()
            .map(|r| {
                let d = r.lowest_trivial_eigenvalue - mean;
                d * d
            })
            .sum::<f64>()
            / ((n - 1) as f64)
    } else {
        0.0
    };
    let std = var.sqrt();
    let se = std / (n as f64).sqrt();
    let lo = rows
        .iter()
        .map(|r| r.lowest_trivial_eigenvalue)
        .fold(f64::INFINITY, f64::min);
    let hi = rows
        .iter()
        .map(|r| r.lowest_trivial_eigenvalue)
        .fold(f64::NEG_INFINITY, f64::max);
    let spread_pct = if mean.abs() > 0.0 {
        100.0 * (hi - lo) / mean.abs()
    } else {
        f64::NAN
    };
    let mean_resid_ppm =
        rows.iter().map(|r| r.residual_ppm_from_omega_fix).sum::<f64>() / (n as f64);
    let resid_of_mean_ppm = 1.0e6 * (mean - omega_fix).abs() / omega_fix.abs().max(1.0e-300);
    Aggregate {
        n_seeds: n,
        mean_lowest_trivial_eigenvalue: mean,
        se_lowest_trivial_eigenvalue: se,
        std_lowest_trivial_eigenvalue: std,
        spread_pct,
        mean_residual_ppm_from_omega_fix: mean_resid_ppm,
        residual_of_mean_ppm: resid_of_mean_ppm,
    }
}

fn build_verdict(agg: &Aggregate) -> String {
    let r = agg.residual_of_mean_ppm;
    let tier = if r <= 100.0 {
        "VERIFIED — mean within 100 ppm of ω_fix (4-digit match)"
    } else if r <= 5000.0 {
        "PARTIAL — mean within 0.5% but >100 ppm of ω_fix (basis-truncation or normalisation residual)"
    } else if r <= 1.0e4 {
        "MARGINAL — mean within 1% of ω_fix (~P7.1 level)"
    } else {
        "FAILED — mean off ω_fix by >1% (framework gateway formula has a residual at this basis size)"
    };
    format!(
        "{} | mean={:.10} ± {:.3e}  spread={:.3}%  mean-residual={:.3} ppm  residual-of-mean={:.3} ppm",
        tier,
        agg.mean_lowest_trivial_eigenvalue,
        agg.se_lowest_trivial_eigenvalue,
        agg.spread_pct,
        agg.mean_residual_ppm_from_omega_fix,
        agg.residual_of_mean_ppm,
    )
}

fn main() {
    let cli = Cli::parse();
    let omega_fix = predicted_omega_fix();
    eprintln!("================================================");
    eprintln!("P7.1b — ω_fix multi-seed sweep on Schoen converged seeds");
    eprintln!("  predicted ω_fix = 123/248 = {:.18}", omega_fix);
    eprintln!(
        "  n_pts={} k={} max_iter={} tol={:.0e} test_degree={}",
        cli.n_pts, cli.k, cli.max_iter, cli.donaldson_tol, cli.test_degree
    );
    let seeds: Vec<u64> = cli
        .seeds
        .split(',')
        .map(|s| s.trim().parse::<u64>().expect("invalid seed in --seeds"))
        .collect();
    eprintln!("  seeds = {:?}", seeds);
    eprintln!("================================================");

    let t_total = Instant::now();
    let mut rows: Vec<PerSeed> = Vec::with_capacity(seeds.len());
    for &seed in &seeds {
        match run_seed(seed, &cli, omega_fix) {
            Ok(r) => rows.push(r),
            Err(e) => eprintln!("  SKIPPED seed={}: {}", seed, e),
        }
    }
    let agg = aggregate(&rows, omega_fix);
    let verdict = build_verdict(&agg);
    let total_elapsed_s = t_total.elapsed().as_secs_f64();

    eprintln!("\n================================================");
    eprintln!("Per-seed summary:");
    eprintln!(
        "  {:>5} {:>6} {:>14} {:>10} {:>14} {:>14}",
        "seed", "iters", "λ_lowest_triv", "purity", "ppm-from-ωfix", "donald-res"
    );
    for r in &rows {
        eprintln!(
            "  {:>5} {:>6} {:>14.10} {:>10.4} {:>14.3} {:>14.3e}",
            r.seed,
            r.metric_iters,
            r.lowest_trivial_eigenvalue,
            r.trivial_rep_purity,
            r.residual_ppm_from_omega_fix,
            r.final_donaldson_residual
        );
    }
    eprintln!("\nVERDICT: {}", verdict);
    eprintln!("================================================");

    let report = Report {
        label: "p7_1b_omega_fix_converged_seeds",
        e8_dim: E8_DIM,
        predicted_omega_fix: omega_fix,
        n_pts: cli.n_pts,
        k: cli.k,
        max_iter: cli.max_iter,
        donaldson_tol: cli.donaldson_tol,
        test_degree: cli.test_degree,
        seeds: seeds.clone(),
        per_seed: rows,
        aggregate: agg,
        verdict,
        total_elapsed_s,
    };

    if let Some(parent) = cli.output.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let json = serde_json::to_string_pretty(&report).expect("serialise P7.1b report");
    fs::write(&cli.output, json).expect("write P7.1b JSON");
    eprintln!(
        "\nWrote {}\nTotal elapsed: {:.1}s",
        cli.output.display(),
        total_elapsed_s
    );
}
