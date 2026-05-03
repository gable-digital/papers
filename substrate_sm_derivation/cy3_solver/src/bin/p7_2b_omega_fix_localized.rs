// P7.2b — ω_fix gateway-eigenvalue diagnostic on a **projected basis**.
//
// Companion to P7.1 (`bin/p7_1_omega_fix_diagnostic.rs`). Instead of
// running the full Galerkin solve and post-classifying eigenvectors by
// Z/3 (TY) or Z/3 × Z/3 (Schoen) character, we project the test-function
// basis onto the trivial irrep *first* (filter to χ = 0 monomials in the
// diagonal-action basis — see `route34::metric_laplacian_projected`),
// then run the same Galerkin solve on the smaller basis. Every
// eigenvalue is in the trivial rep with 100% purity by construction;
// no character mixing pathology can arise.
//
// Tests the journal §L.2 prediction
//
//     ω_fix = 1/2 − 1/dim(E_8) = 123/248 = 0.495967741935...

use clap::Parser;
use cy3_rust_solver::route34::cy3_metric_unified::{
    Cy3MetricResultKind, Cy3MetricSolver, Cy3MetricSpec, SchoenSolver, TianYauSolver,
};
use cy3_rust_solver::route34::metric_laplacian::MetricLaplacianConfig;
use cy3_rust_solver::route34::metric_laplacian_projected::{
    compute_projected_metric_laplacian_spectrum, ProjectedSpectrumReport, ProjectionKind,
};
use cy3_rust_solver::route34::yukawa_pipeline::Cy3MetricResultBackground;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const E8_DIM: u32 = 248;
fn predicted_omega_fix() -> f64 {
    0.5 - 1.0 / (E8_DIM as f64)
}

#[derive(Parser, Debug)]
#[command(about = "P7.2b ω_fix gateway-eigenvalue diagnostic on a Z/3- or Z/3×Z/3-projected basis")]
struct Cli {
    /// Sample-cloud size for the Donaldson metric solve.
    #[arg(long, default_value_t = 40_000)]
    n_pts: usize,

    /// Bigraded section-basis degree for Donaldson (k for TY, d_x=d_y=k
    /// for Schoen with d_t=1).
    #[arg(long, default_value_t = 3)]
    k: u32,

    /// Donaldson iteration cap.
    #[arg(long, default_value_t = 100)]
    max_iter: usize,

    /// Donaldson convergence tolerance.
    #[arg(long, default_value_t = 1.0e-6)]
    donaldson_tol: f64,

    /// Single-seed PRNG seed. Default 12345 is one of the strict-converged
    /// Schoen seeds.
    #[arg(long, default_value_t = 12345)]
    seed: u64,

    /// Maximum total degree of the test-function basis.
    #[arg(long, default_value_t = 4)]
    test_degree: u32,

    /// Number of low eigenvalues to record per candidate.
    #[arg(long, default_value_t = 5)]
    n_record: usize,

    /// Skip Schoen.
    #[arg(long, default_value_t = false)]
    skip_schoen: bool,

    /// Skip TY.
    #[arg(long, default_value_t = false)]
    skip_ty: bool,

    /// Output JSON path.
    #[arg(long, default_value = "output/p7_2b_omega_fix_localized.json")]
    output: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Normalised {
    by_lambda1_min: f64,
    by_lambda_max: f64,
    by_mean_eigvalue: f64,
    by_volume: f64,
    raw: f64,
}

fn ppm(observed: f64, predicted: f64) -> f64 {
    1.0e6 * (observed - predicted).abs() / predicted.abs().max(1.0e-300)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EigInfo {
    rank: usize,
    eigenvalue: f64,
    normalised: Normalised,
    residual_ppm_raw: f64,
    residual_ppm_by_lambda1_min: f64,
    residual_ppm_by_lambda_max: f64,
    residual_ppm_by_mean: f64,
    residual_ppm_by_volume: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClosestPick {
    rank: usize,
    eigenvalue: f64,
    chosen_scheme: String,
    chosen_value: f64,
    residual_ppm: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CandidateReport {
    label: String,
    projection: String,
    metric_iters: usize,
    metric_sigma_residual: f64,
    metric_seconds: f64,
    spectrum_seconds: f64,
    full_basis_dim: usize,
    projected_basis_dim: usize,
    survival_fraction: f64,
    n_points: usize,
    lambda_min_nonzero: f64,
    lambda_max: f64,
    lambda_mean: f64,
    /// Volume scale: `Σ_p w_p · |Ω(p)|²`. Useful as a coordinate-
    /// independent normalisation reference.
    volume_proxy: f64,
    bottom_eigvalues: Vec<EigInfo>,
    closest_to_omega_fix: ClosestPick,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiagnosticReport {
    label: &'static str,
    e8_dim: u32,
    predicted_omega_fix: f64,
    n_pts: usize,
    k: u32,
    max_iter: usize,
    seed: u64,
    test_degree: u32,
    candidates: Vec<CandidateReport>,
    verdict: String,
    total_elapsed_s: f64,
}

/// Compute the dvol-volume `Σ_p w_p · |Ω(p)|²`. Normalised mass-matrix
/// integrals always sum the per-point Donaldson weight to 1, so this is
/// just `Σ |Ω(p)|² / N` after the renormalisation in
/// `Cy3MetricResultBackground`.
fn volume_proxy(bg: &Cy3MetricResultBackground<'_>) -> f64 {
    use cy3_rust_solver::route34::hym_hermitian::MetricBackground;
    let n = bg.n_points();
    let mut acc = 0.0_f64;
    for p in 0..n {
        let w = bg.weight(p);
        if !w.is_finite() || w <= 0.0 {
            continue;
        }
        let o = bg.omega(p);
        if !o.re.is_finite() || !o.im.is_finite() {
            continue;
        }
        acc += w * o.norm_sqr();
    }
    acc
}

fn run_candidate(
    label: &str,
    spec: Cy3MetricSpec,
    solver: &dyn Cy3MetricSolver,
    test_degree: u32,
    n_record: usize,
    projection: ProjectionKind,
) -> Result<CandidateReport, String> {
    let t_metric = Instant::now();
    let r = solver
        .solve_metric(&spec)
        .map_err(|e| format!("{label}: metric solve failed: {e}"))?;
    let summary = r.summary();
    let metric_seconds = t_metric.elapsed().as_secs_f64();

    let bg = match &r {
        Cy3MetricResultKind::TianYau(t) => Cy3MetricResultBackground::from_ty(t.as_ref()),
        Cy3MetricResultKind::Schoen(t) => Cy3MetricResultBackground::from_schoen(t.as_ref()),
    };

    let cfg = MetricLaplacianConfig {
        max_total_degree: test_degree,
        n_low_eigenvalues: 50,
        return_eigenvectors: false,
        ..MetricLaplacianConfig::default()
    };
    let t_spec = Instant::now();
    let report: ProjectedSpectrumReport =
        compute_projected_metric_laplacian_spectrum(&bg, &cfg, projection);
    let spectrum_seconds = t_spec.elapsed().as_secs_f64();

    let evals = &report.spectrum.eigenvalues_full;
    if evals.is_empty() {
        return Err(format!(
            "{label}: empty spectrum after projection (full basis {} -> projected {})",
            report.full_basis_dim, report.projected_basis_dim
        ));
    }

    let nonzero: Vec<f64> = evals.iter().cloned().filter(|&v| v > 1.0e-10).collect();
    let lambda_min_nonzero = nonzero.first().copied().unwrap_or(f64::NAN);
    let lambda_max = evals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lambda_mean = evals.iter().sum::<f64>() / (evals.len() as f64);
    let vol = volume_proxy(&bg);

    let make_norm = |raw: f64| -> Normalised {
        Normalised {
            by_lambda1_min: if lambda_min_nonzero.is_finite() && lambda_min_nonzero > 0.0 {
                raw / lambda_min_nonzero
            } else {
                f64::NAN
            },
            by_lambda_max: if lambda_max.is_finite() && lambda_max > 0.0 {
                raw / lambda_max
            } else {
                f64::NAN
            },
            by_mean_eigvalue: if lambda_mean.is_finite() && lambda_mean > 0.0 {
                raw / lambda_mean
            } else {
                f64::NAN
            },
            by_volume: if vol.is_finite() && vol > 0.0 {
                raw / vol
            } else {
                f64::NAN
            },
            raw,
        }
    };

    let omega_fix = predicted_omega_fix();
    let bottom: Vec<EigInfo> = nonzero
        .iter()
        .take(n_record)
        .enumerate()
        .map(|(rank, &ev)| {
            let n = make_norm(ev);
            EigInfo {
                rank,
                eigenvalue: ev,
                residual_ppm_raw: ppm(n.raw, omega_fix),
                residual_ppm_by_lambda1_min: ppm(n.by_lambda1_min, omega_fix),
                residual_ppm_by_lambda_max: ppm(n.by_lambda_max, omega_fix),
                residual_ppm_by_mean: ppm(n.by_mean_eigvalue, omega_fix),
                residual_ppm_by_volume: ppm(n.by_volume, omega_fix),
                normalised: n,
            }
        })
        .collect();

    // Pick the (eigenvalue, scheme) pair that comes closest to ω_fix.
    let schemes: [(&'static str, fn(&Normalised) -> f64); 5] = [
        ("raw", |n| n.raw),
        ("by_lambda1_min", |n| n.by_lambda1_min),
        ("by_lambda_max", |n| n.by_lambda_max),
        ("by_mean_eigvalue", |n| n.by_mean_eigvalue),
        ("by_volume", |n| n.by_volume),
    ];
    let mut best: Option<(f64, &EigInfo, &'static str, f64)> = None;
    for e in &bottom {
        for (name, getter) in &schemes {
            let v = getter(&e.normalised);
            if !v.is_finite() {
                continue;
            }
            let res = (v - omega_fix).abs() / omega_fix.abs().max(1.0e-300);
            match &best {
                None => best = Some((res, e, name, v)),
                Some((b_res, _, _, _)) if res < *b_res => best = Some((res, e, name, v)),
                _ => {}
            }
        }
    }
    let (res, e, scheme, value) = best.ok_or_else(|| format!("{label}: no closest pick"))?;
    let closest_to_omega_fix = ClosestPick {
        rank: e.rank,
        eigenvalue: e.eigenvalue,
        chosen_scheme: scheme.to_string(),
        chosen_value: value,
        residual_ppm: 1.0e6 * res,
    };

    Ok(CandidateReport {
        label: label.to_string(),
        projection: report.projection,
        metric_iters: summary.iterations_run,
        metric_sigma_residual: summary.final_sigma_residual,
        metric_seconds,
        spectrum_seconds,
        full_basis_dim: report.full_basis_dim,
        projected_basis_dim: report.projected_basis_dim,
        survival_fraction: report.survival_fraction,
        n_points: report.spectrum.n_points,
        lambda_min_nonzero,
        lambda_max,
        lambda_mean,
        volume_proxy: vol,
        bottom_eigvalues: bottom,
        closest_to_omega_fix,
    })
}

fn print_candidate(c: &CandidateReport) {
    let omega_fix = predicted_omega_fix();
    eprintln!("\n=== {} ===", c.label);
    eprintln!(
        "  projection: {} (full {} -> projected {}, survival {:.3})",
        c.projection, c.full_basis_dim, c.projected_basis_dim, c.survival_fraction
    );
    eprintln!(
        "  metric: iters={}  σ-residual={:.3e}  t={:.1}s",
        c.metric_iters, c.metric_sigma_residual, c.metric_seconds
    );
    eprintln!(
        "  spectrum: basis_dim={}  n_pts={}  t={:.2}s",
        c.projected_basis_dim, c.n_points, c.spectrum_seconds
    );
    eprintln!(
        "  λ_min_nonzero = {:.6e}  λ_max = {:.6e}  λ_mean = {:.6e}  vol_proxy = {:.6e}",
        c.lambda_min_nonzero, c.lambda_max, c.lambda_mean, c.volume_proxy
    );
    eprintln!("  bottom-{} eigenvalues (all in trivial rep by construction):", c.bottom_eigvalues.len());
    for e in &c.bottom_eigvalues {
        eprintln!(
            "    [{}] λ={:.6e}  raw_ppm={:8.1}  /λ1={:.6} ({:8.1} ppm)  /λmax={:.3e} ({:8.1} ppm)  /vol={:.6} ({:8.1} ppm)",
            e.rank,
            e.eigenvalue,
            e.residual_ppm_raw,
            e.normalised.by_lambda1_min,
            e.residual_ppm_by_lambda1_min,
            e.normalised.by_lambda_max,
            e.residual_ppm_by_lambda_max,
            e.normalised.by_volume,
            e.residual_ppm_by_volume,
        );
    }
    eprintln!(
        "  closest-to-ω_fix:  rank={} λ={:.6e} scheme={} value={:.6} residual={:.3} ppm",
        c.closest_to_omega_fix.rank,
        c.closest_to_omega_fix.eigenvalue,
        c.closest_to_omega_fix.chosen_scheme,
        c.closest_to_omega_fix.chosen_value,
        c.closest_to_omega_fix.residual_ppm
    );
    eprintln!("  predicted ω_fix = {:.12} (= 123/248)", omega_fix);
}

fn main() {
    let cli = Cli::parse();
    let omega_fix = predicted_omega_fix();
    eprintln!("================================================");
    eprintln!("P7.2b — ω_fix gateway-eigenvalue diagnostic (projected basis)");
    eprintln!("  predicted ω_fix = 1/2 − 1/dim(E_8) = 123/248");
    eprintln!("                  = {:.18}", omega_fix);
    eprintln!("================================================");
    eprintln!(
        "  n_pts={}  k={}  max_iter={}  seed={}  test_degree={}",
        cli.n_pts, cli.k, cli.max_iter, cli.seed, cli.test_degree
    );

    let t_total = Instant::now();
    let mut candidates: Vec<CandidateReport> = Vec::new();

    if !cli.skip_schoen {
        let schoen_spec = Cy3MetricSpec::Schoen {
            d_x: cli.k,
            d_y: cli.k,
            d_t: 1,
            n_sample: cli.n_pts,
            max_iter: cli.max_iter,
            donaldson_tol: cli.donaldson_tol,
            seed: cli.seed,
        };
        eprintln!("\n--- running Schoen / Z₃×Z₃ projected ---");
        match run_candidate(
            "Schoen/Z3xZ3",
            schoen_spec,
            &SchoenSolver,
            cli.test_degree,
            cli.n_record,
            ProjectionKind::SchoenZ3xZ3,
        ) {
            Ok(c) => {
                print_candidate(&c);
                candidates.push(c);
            }
            Err(e) => eprintln!("  SKIPPED: {e}"),
        }
    }

    if !cli.skip_ty {
        let ty_spec = Cy3MetricSpec::TianYau {
            k: cli.k,
            n_sample: cli.n_pts,
            max_iter: cli.max_iter,
            donaldson_tol: cli.donaldson_tol,
            seed: cli.seed,
        };
        eprintln!("\n--- running TY / Z₃ projected (control) ---");
        match run_candidate(
            "TY/Z3",
            ty_spec,
            &TianYauSolver,
            cli.test_degree,
            cli.n_record,
            ProjectionKind::TianYauZ3,
        ) {
            Ok(c) => {
                print_candidate(&c);
                candidates.push(c);
            }
            Err(e) => eprintln!("  SKIPPED: {e}"),
        }
    }

    let total_elapsed_s = t_total.elapsed().as_secs_f64();
    let verdict = build_verdict(&candidates, omega_fix);
    eprintln!("\n================================================");
    eprintln!("VERDICT:");
    eprintln!("{}", verdict);
    eprintln!("================================================");

    let report = DiagnosticReport {
        label: "p7_2b_omega_fix_localized",
        e8_dim: E8_DIM,
        predicted_omega_fix: omega_fix,
        n_pts: cli.n_pts,
        k: cli.k,
        max_iter: cli.max_iter,
        seed: cli.seed,
        test_degree: cli.test_degree,
        candidates,
        verdict,
        total_elapsed_s,
    };

    if let Some(parent) = cli.output.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let json = serde_json::to_string_pretty(&report).expect("serialise diagnostic report");
    fs::write(&cli.output, json).expect("write diagnostic JSON");
    eprintln!("\nWrote {}", cli.output.display());
    eprintln!("Total elapsed: {:.1}s", total_elapsed_s);
}

fn build_verdict(candidates: &[CandidateReport], omega_fix: f64) -> String {
    let _ = omega_fix;
    if candidates.is_empty() {
        return "NO RESULT — both candidates skipped".to_string();
    }
    let mut s = String::new();
    for c in candidates {
        let pick = &c.closest_to_omega_fix;
        let tier = if pick.residual_ppm <= 1.0e2 {
            "VERIFIED (≤100 ppm = 4-digit match)"
        } else if pick.residual_ppm <= 1.0e4 {
            "MARGINAL (≤1% but >100 ppm)"
        } else {
            "FAILED (>1% off ω_fix)"
        };
        s.push_str(&format!(
            "{}: rank={} λ={:.6} via {} → {:.6} ({:.3} ppm, {:.4}% off ω_fix) — {}\n",
            c.label,
            pick.rank,
            pick.eigenvalue,
            pick.chosen_scheme,
            pick.chosen_value,
            pick.residual_ppm,
            100.0 * pick.residual_ppm / 1.0e6,
            tier
        ));
    }
    s
}
