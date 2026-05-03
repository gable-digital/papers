// P7.5 — H_4 (icosahedral) sub-Coxeter gateway-eigenvalue test on
// the Schoen Z/3 × Z/3 Calabi-Yau three-fold.
//
// Companion to P7.1 / P7.1b / P7.2b. Whereas P7.2b projected the
// bigraded test-function basis onto the Z/3 × Z/3 trivial irrep
// before the Galerkin solve and found no eigenvalue near
//
//     ω_fix = 1/2 − 1/dim(E_8) = 123/248 = 0.495967741935…,
//
// this test additionally projects onto the **H_4 (icosahedral)
// sub-Coxeter sector** identified by the journal §L.1 / §L.2 as
// hosting the lepton spectrum and the gateway mode. The H_4
// projection is implemented (in [`crate::route34::sub_coxeter_h4_projector`])
// as the diagonal fivefold rotation `R_5: x_k → ζ_5^k x_k` (Klein
// 1884; Du Val 1964 §8) on each `CP^2` block of the bicubic Schoen
// ambient, which is the minimal Z/5 sub-group of the icosahedral
// `2I ⊂ SU(2)`.
//
// Hypothesis (3) of the falsification triangle — that the
// previously-tested basis is too coarse, mixing all sub-Coxeter
// sectors of E_8 — predicts that ω_fix lives in the H_4-invariant
// subspace specifically, and a per-sector Galerkin solve should
// recover it.

use clap::Parser;
use cy3_rust_solver::route34::cy3_metric_unified::{
    Cy3MetricResultKind, Cy3MetricSolver, Cy3MetricSpec, SchoenSolver,
};
use cy3_rust_solver::route34::metric_laplacian::MetricLaplacianConfig;
use cy3_rust_solver::route34::metric_laplacian_projected::ProjectionKind;
use cy3_rust_solver::route34::sub_coxeter_h4_projector::{
    compute_h4_projected_metric_laplacian_spectrum, H4ProjectedSpectrumReport,
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

/// Golden ratio `φ = (1 + √5)/2`. Used for the optional Step 4
/// chain-position bonus check (icosahedral / E_8 exponent ladder).
fn phi() -> f64 {
    0.5 * (1.0 + 5.0_f64.sqrt())
}

/// Journal §L.1 lepton chain exponents — the residues of `1, …, 29`
/// coprime to `30` (= the Coxeter number of E_8 / H_4). The chain
/// matcher will compare ratios `λ_k / λ_min` to `φ^{e_k − e_1}`
/// where `e_1 = 1`.
const ICOSAHEDRAL_EXPONENTS: [u32; 8] = [1, 7, 11, 13, 17, 19, 23, 29];

#[derive(Parser, Debug)]
#[command(about = "P7.5 H_4-projected ω_fix gateway-eigenvalue test (Schoen)")]
struct Cli {
    /// Sample-cloud size for the Donaldson metric solve.
    #[arg(long, default_value_t = 25_000)]
    n_pts: usize,

    /// Bigraded section-basis degree for Donaldson.
    #[arg(long, default_value_t = 3)]
    k: u32,

    /// Donaldson iteration cap.
    #[arg(long, default_value_t = 100)]
    max_iter: usize,

    /// Donaldson convergence tolerance.
    #[arg(long, default_value_t = 1.0e-6)]
    donaldson_tol: f64,

    /// Single-seed PRNG seed. Default 12345 is one of the strict-
    /// converged Schoen seeds.
    #[arg(long, default_value_t = 12345)]
    seed: u64,

    /// Maximum total degree of the test-function basis.
    #[arg(long, default_value_t = 4)]
    test_degree: u32,

    /// Number of low eigenvalues to record.
    #[arg(long, default_value_t = 8)]
    n_record: usize,

    /// Output JSON path.
    #[arg(long, default_value = "output/p7_5_h4_gateway.json")]
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
    residual_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChainPositionEntry {
    /// Index `k` in the icosahedral-exponent list `e_k ∈
    /// {1, 7, 11, 13, 17, 19, 23, 29}`.
    k: usize,
    exponent: u32,
    /// Predicted ratio `φ^{e_k − e_1}` (with `e_1 = 1`).
    predicted_ratio: f64,
    /// Observed ratio `λ_k / λ_1` from the H_4-projected spectrum
    /// (after dropping near-zero modes).
    observed_ratio: f64,
    /// `|observed − predicted| / predicted`.
    relative_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChainPositionReport {
    note: String,
    n_eigvals_used: usize,
    entries: Vec<ChainPositionEntry>,
    rms_relative_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GatewayCandidateReport {
    label: String,
    projection: String,
    metric_iters: usize,
    metric_sigma_residual: f64,
    metric_seconds: f64,
    spectrum_seconds: f64,
    full_basis_dim: usize,
    gamma_projected_basis_dim: usize,
    h4_projected_basis_dim: usize,
    gamma_survival_fraction: f64,
    h4_survival_fraction: f64,
    h4_relative_survival_fraction: f64,
    n_points: usize,
    lambda_min_nonzero: f64,
    lambda_max: f64,
    lambda_mean: f64,
    volume_proxy: f64,
    bottom_eigvalues: Vec<EigInfo>,
    closest_to_omega_fix: ClosestPick,
    chain_position: Option<ChainPositionReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiagnosticReport {
    label: &'static str,
    h4_projection_scheme: &'static str,
    e8_dim: u32,
    predicted_omega_fix: f64,
    n_pts: usize,
    k: u32,
    max_iter: usize,
    seed: u64,
    test_degree: u32,
    candidate: GatewayCandidateReport,
    verdict: String,
    total_elapsed_s: f64,
}

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

fn compute_chain_position(report: &H4ProjectedSpectrumReport) -> Option<ChainPositionReport> {
    let evs: Vec<f64> = report
        .spectrum
        .eigenvalues_full
        .iter()
        .copied()
        .filter(|&v| v.is_finite() && v > 1.0e-10)
        .collect();
    if evs.len() < 2 {
        return None;
    }
    let lambda1 = evs[0];
    let phi_v = phi();
    let mut entries = Vec::new();
    let n_use = evs.len().min(ICOSAHEDRAL_EXPONENTS.len());
    let mut sq_sum = 0.0_f64;
    for k in 0..n_use {
        let exponent = ICOSAHEDRAL_EXPONENTS[k];
        let predicted_ratio = phi_v.powi((exponent as i32) - (ICOSAHEDRAL_EXPONENTS[0] as i32));
        let observed_ratio = evs[k] / lambda1;
        let rel_err = (observed_ratio - predicted_ratio).abs() / predicted_ratio.abs().max(1.0e-300);
        sq_sum += rel_err * rel_err;
        entries.push(ChainPositionEntry {
            k,
            exponent,
            predicted_ratio,
            observed_ratio,
            relative_error: rel_err,
        });
    }
    let rms = (sq_sum / (n_use as f64)).sqrt();
    Some(ChainPositionReport {
        note: "Naive index-ordered match against icosahedral exponents \
               {1,7,11,13,17,19,23,29} with predicted ratio φ^{e_k - 1}; \
               no Hungarian assignment, no constant-mode handling — bonus \
               diagnostic only."
            .to_string(),
        n_eigvals_used: n_use,
        entries,
        rms_relative_error: rms,
    })
}

fn run_schoen_h4(cli: &Cli) -> Result<GatewayCandidateReport, String> {
    let spec = Cy3MetricSpec::Schoen {
        d_x: cli.k,
        d_y: cli.k,
        d_t: 1,
        n_sample: cli.n_pts,
        max_iter: cli.max_iter,
        donaldson_tol: cli.donaldson_tol,
        seed: cli.seed,
    };
    let solver = SchoenSolver;

    let t_metric = Instant::now();
    let r = solver
        .solve_metric(&spec)
        .map_err(|e| format!("Schoen H_4: metric solve failed: {e}"))?;
    let summary = r.summary();
    let metric_seconds = t_metric.elapsed().as_secs_f64();

    let bg = match &r {
        Cy3MetricResultKind::Schoen(t) => Cy3MetricResultBackground::from_schoen(t.as_ref()),
        Cy3MetricResultKind::TianYau(_) => {
            return Err("Schoen H_4: solver returned TY background unexpectedly".to_string());
        }
    };

    let cfg = MetricLaplacianConfig {
        max_total_degree: cli.test_degree,
        n_low_eigenvalues: 50,
        return_eigenvectors: false,
        ..MetricLaplacianConfig::default()
    };

    let t_spec = Instant::now();
    let report =
        compute_h4_projected_metric_laplacian_spectrum(&bg, &cfg, ProjectionKind::SchoenZ3xZ3);
    let spectrum_seconds = t_spec.elapsed().as_secs_f64();

    let evs = &report.spectrum.eigenvalues_full;
    if evs.is_empty() {
        return Err(format!(
            "Schoen H_4: empty spectrum after H_4 projection (full {} → Γ {} → H_4 {})",
            report.full_basis_dim, report.gamma_projected_basis_dim, report.h4_projected_basis_dim
        ));
    }

    let nonzero: Vec<f64> = evs.iter().cloned().filter(|&v| v > 1.0e-10).collect();
    let lambda_min_nonzero = nonzero.first().copied().unwrap_or(f64::NAN);
    let lambda_max = evs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lambda_mean = evs.iter().sum::<f64>() / (evs.len() as f64);
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
        .take(cli.n_record)
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
    let (res, e, scheme, value) = best.ok_or_else(|| "Schoen H_4: no closest pick".to_string())?;
    let closest_to_omega_fix = ClosestPick {
        rank: e.rank,
        eigenvalue: e.eigenvalue,
        chosen_scheme: scheme.to_string(),
        chosen_value: value,
        residual_ppm: 1.0e6 * res,
        residual_pct: 100.0 * res,
    };

    let chain_position = compute_chain_position(&report);

    Ok(GatewayCandidateReport {
        label: "Schoen/Z3xZ3 ∩ H_4(icosa-Z5)".to_string(),
        projection: report.projection,
        metric_iters: summary.iterations_run,
        metric_sigma_residual: summary.final_sigma_residual,
        metric_seconds,
        spectrum_seconds,
        full_basis_dim: report.full_basis_dim,
        gamma_projected_basis_dim: report.gamma_projected_basis_dim,
        h4_projected_basis_dim: report.h4_projected_basis_dim,
        gamma_survival_fraction: report.gamma_survival_fraction,
        h4_survival_fraction: report.h4_survival_fraction,
        h4_relative_survival_fraction: report.h4_relative_survival_fraction,
        n_points: report.spectrum.n_points,
        lambda_min_nonzero,
        lambda_max,
        lambda_mean,
        volume_proxy: vol,
        bottom_eigvalues: bottom,
        closest_to_omega_fix,
        chain_position,
    })
}

fn print_candidate(c: &GatewayCandidateReport) {
    let omega_fix = predicted_omega_fix();
    eprintln!("\n=== {} ===", c.label);
    eprintln!(
        "  projection: {} (full {} → Γ {} → H_4 {}; survival Γ {:.4}, H_4 {:.4}, rel {:.4})",
        c.projection,
        c.full_basis_dim,
        c.gamma_projected_basis_dim,
        c.h4_projected_basis_dim,
        c.gamma_survival_fraction,
        c.h4_survival_fraction,
        c.h4_relative_survival_fraction
    );
    eprintln!(
        "  metric: iters={} σ-residual={:.3e} t={:.1}s",
        c.metric_iters, c.metric_sigma_residual, c.metric_seconds
    );
    eprintln!(
        "  spectrum: basis_dim={} n_pts={} t={:.2}s",
        c.h4_projected_basis_dim, c.n_points, c.spectrum_seconds
    );
    eprintln!(
        "  λ_min_nonzero = {:.6e}  λ_max = {:.6e}  λ_mean = {:.6e}  vol_proxy = {:.6e}",
        c.lambda_min_nonzero, c.lambda_max, c.lambda_mean, c.volume_proxy
    );
    eprintln!(
        "  bottom-{} eigenvalues (all in (Γ ∩ H_4)-trivial rep by construction):",
        c.bottom_eigvalues.len()
    );
    for e in &c.bottom_eigvalues {
        eprintln!(
            "    [{}] λ={:.6e}  raw_ppm={:9.1}  /λ1={:.6} ({:9.1} ppm)  /λmax={:.3e} ({:9.1} ppm)  /vol={:.6} ({:9.1} ppm)",
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
        "  closest-to-ω_fix:  rank={} λ={:.6e} scheme={} value={:.6} residual={:.3} ppm ({:.4} %)",
        c.closest_to_omega_fix.rank,
        c.closest_to_omega_fix.eigenvalue,
        c.closest_to_omega_fix.chosen_scheme,
        c.closest_to_omega_fix.chosen_value,
        c.closest_to_omega_fix.residual_ppm,
        c.closest_to_omega_fix.residual_pct,
    );
    eprintln!("  predicted ω_fix = {:.12} (= 123/248)", omega_fix);

    if let Some(chain) = &c.chain_position {
        eprintln!(
            "\n  --- bonus chain-position check ({} eigvals) ---",
            chain.n_eigvals_used
        );
        for ent in &chain.entries {
            eprintln!(
                "    k={:2} e_k={:2}  predicted φ^{{e_k-1}}={:.6}  observed={:.6}  rel_err={:.4}",
                ent.k, ent.exponent, ent.predicted_ratio, ent.observed_ratio, ent.relative_error,
            );
        }
        eprintln!("    RMS relative error = {:.4}", chain.rms_relative_error);
    }
}

fn build_verdict(c: &GatewayCandidateReport) -> String {
    let pick = &c.closest_to_omega_fix;
    let tier = if pick.residual_ppm <= 1.0e0 {
        "VERIFIED at 1 ppm precision (≤1 ppm)"
    } else if pick.residual_ppm <= 1.0e2 {
        "VERIFIED at 100 ppm precision (≤100 ppm = 4-digit match)"
    } else if pick.residual_ppm <= 1.0e4 {
        "MARGINAL (≤1 % but >100 ppm)"
    } else {
        "FAILED (>1 % off ω_fix)"
    };
    format!(
        "{}: rank={} λ={:.6} via {} → {:.6} ({:.3} ppm, {:.4} % off ω_fix) — {}\n\
         H_4 projection: {} (basis dim {} after projection)",
        c.label,
        pick.rank,
        pick.eigenvalue,
        pick.chosen_scheme,
        pick.chosen_value,
        pick.residual_ppm,
        pick.residual_pct,
        tier,
        c.projection,
        c.h4_projected_basis_dim,
    )
}

fn main() {
    let cli = Cli::parse();
    let omega_fix = predicted_omega_fix();
    eprintln!("================================================");
    eprintln!("P7.5 — H_4 (icosahedral) sub-Coxeter ω_fix gateway test");
    eprintln!("  predicted ω_fix = 1/2 − 1/dim(E_8) = 123/248");
    eprintln!("                  = {:.18}", omega_fix);
    eprintln!("  H_4 projection:  Z/3 × Z/3 (Schoen Γ-quotient)");
    eprintln!("                 ∩ icosahedral fivefold (Z/5 ⊂ 2I ⊂ SU(2))");
    eprintln!("                   on bicubic blocks (Klein 1884 §I.7).");
    eprintln!("================================================");
    eprintln!(
        "  n_pts={}  k={}  max_iter={}  seed={}  test_degree={}",
        cli.n_pts, cli.k, cli.max_iter, cli.seed, cli.test_degree
    );

    let t_total = Instant::now();
    eprintln!("\n--- running Schoen / H_4-projected ---");
    let candidate = match run_schoen_h4(&cli) {
        Ok(c) => {
            print_candidate(&c);
            c
        }
        Err(e) => {
            eprintln!("\nFATAL: {e}");
            std::process::exit(1);
        }
    };

    let total_elapsed_s = t_total.elapsed().as_secs_f64();
    let verdict = build_verdict(&candidate);
    eprintln!("\n================================================");
    eprintln!("VERDICT:");
    eprintln!("{}", verdict);
    eprintln!("================================================");

    let report = DiagnosticReport {
        label: "p7_5_h4_gateway_test",
        h4_projection_scheme:
            "fallback-Z5: icosahedral fivefold rotation only (R_5 acting as ζ_5^k on x_k, y_k); \
             no 2-fold or 3-fold reflections beyond the existing Z/3 × Z/3 Γ-quotient",
        e8_dim: E8_DIM,
        predicted_omega_fix: omega_fix,
        n_pts: cli.n_pts,
        k: cli.k,
        max_iter: cli.max_iter,
        seed: cli.seed,
        test_degree: cli.test_degree,
        candidate,
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
