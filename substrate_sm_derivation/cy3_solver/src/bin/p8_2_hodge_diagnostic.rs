//! P8.2 — Hodge-number consistency channel diagnostic.
//!
//! Runs the bundle Laplacian `Δ_∂̄^V` on the Z/3 × Z/3 (Schoen) /
//! Z/3 (TY) trivial-rep sub-bundle for both candidates at the
//! canonical P8.2 settings (n_pts = 25 000, k = 3, seed = 12345),
//! counts kernel modes, splits via Hodge symmetry, and compares to
//! the journal prediction `(h^{1,1}, h^{2,1}, χ) = (3, 3, -6)` via
//! a Gaussian log-likelihood.
//!
//! Reports the per-candidate measured Hodge triple, per-candidate
//! log-likelihood vs prediction, and the discrimination contribution
//! `Δ ln L = ln L_TY - ln L_Schoen` (in nats) — the Hodge channel's
//! contribution to the multi-channel TY-vs-Schoen Bayes factor.
//!
//! Output JSON: `output/p8_2_hodge_diagnostic.json`.

use clap::Parser;
use cy3_rust_solver::route34::cy3_metric_unified::{
    Cy3MetricResultKind, Cy3MetricSolver, Cy3MetricSpec, SchoenSolver, TianYauSolver,
};
use cy3_rust_solver::route34::hodge_channel::{
    compute_hodge_channel, HodgeCandidateSpec, HodgeChannelConfig, HodgeChannelResult,
};
use cy3_rust_solver::route34::hym_hermitian::{solve_hym_metric, HymConfig};
use cy3_rust_solver::route34::wilson_line_e8_z3xz3::Z3xZ3WilsonLines;
use cy3_rust_solver::route34::yukawa_pipeline::Cy3MetricResultBackground;
use cy3_rust_solver::route34::zero_modes_harmonic_z3xz3::{Z3xZ3BundleConfig, Z3xZ3Geometry};
use cy3_rust_solver::zero_modes::MonadBundle;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "P8.2 Hodge-number consistency channel diagnostic (TY vs Schoen)")]
struct Cli {
    /// Sample-cloud size. P8.2 canonical: 25 000.
    #[arg(long, default_value_t = 25_000)]
    n_pts: usize,

    /// Bigraded section-basis degree.
    #[arg(long, default_value_t = 3)]
    k: u32,

    /// Donaldson iteration cap.
    #[arg(long, default_value_t = 100)]
    max_iter: usize,

    /// Donaldson tolerance.
    #[arg(long, default_value_t = 1.0e-6)]
    donaldson_tol: f64,

    /// PRNG seed. Default 12345 (canonical P5.10 / P7.6 seed).
    #[arg(long, default_value_t = 12345)]
    seed: u64,

    /// Numerical-zero threshold for kernel classification
    /// (`|λ| ≤ τ · λ_max`).
    #[arg(long, default_value_t = 1.0e-3)]
    kernel_zero_thresh: f64,

    /// Gaussian width on `h^{p,q}` counts.
    #[arg(long, default_value_t = 0.5)]
    sigma_h: f64,

    /// Gaussian width on `χ`.
    #[arg(long, default_value_t = 1.0)]
    sigma_chi: f64,

    /// Disable the H_4 (icosa Z/5) filter — keep Z/3 × Z/3 only.
    #[arg(long, default_value_t = false)]
    no_h4: bool,

    #[arg(long, default_value_t = false)]
    skip_schoen: bool,

    #[arg(long, default_value_t = false)]
    skip_ty: bool,

    /// Reduce `n_pts` and `max_iter` to fast settings (for smoke tests).
    #[arg(long, default_value_t = false)]
    fast: bool,

    /// Maximum total polynomial degree (per CP² block) of the AKLP
    /// b_lines seed-basis expansion. Higher values yield a larger
    /// projected basis with finer kernel resolution. Default 3
    /// matches the P7.8 production sweep.
    #[arg(long, default_value_t = 3)]
    seed_max_total_degree: usize,

    /// P7.8 basis-orthogonalization flag. Set to true at
    /// `seed_max_total_degree ≥ 3` for numerical stability of the
    /// Galerkin solve. Default true to match P7.8 production.
    #[arg(long, default_value_t = true)]
    orthogonalize_first: bool,

    #[arg(long, default_value = "output/p8_2_hodge_diagnostic.json")]
    output: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiagnosticReport {
    label: &'static str,
    n_pts: usize,
    k: u32,
    max_iter: usize,
    seed: u64,
    kernel_zero_thresh: f64,
    sigma_h: f64,
    sigma_chi: f64,
    h4_applied: bool,

    schoen: Option<HodgeChannelResult>,
    ty: Option<HodgeChannelResult>,

    /// `Δ ln L = ln L_TY - ln L_Schoen`. Positive ⇒ TY favoured by
    /// the Hodge channel; negative ⇒ Schoen favoured.
    delta_log_likelihood_ty_minus_schoen: Option<f64>,

    /// Same value rendered as a magnitude in nats with a sign label.
    /// Convenience for human-readable summaries.
    discrimination_summary: String,

    total_elapsed_s: f64,
    notes: Vec<String>,
}

fn run_one(
    label: &str,
    candidate: HodgeCandidateSpec,
    spec: Cy3MetricSpec,
    solver: &dyn Cy3MetricSolver,
    bundle: &MonadBundle,
    wilson: &Z3xZ3WilsonLines,
    laplacian_config: Z3xZ3BundleConfig,
    channel_config: &HodgeChannelConfig,
) -> Result<HodgeChannelResult, String> {
    let t = Instant::now();
    let r = solver
        .solve_metric(&spec)
        .map_err(|e| format!("{label}: metric solve failed: {e}"))?;
    let summary = r.summary();
    eprintln!(
        "  [{}] metric: iters={} σ-residual={:.3e}",
        label, summary.iterations_run, summary.final_sigma_residual
    );

    let bg = match &r {
        Cy3MetricResultKind::TianYau(t) => Cy3MetricResultBackground::from_ty(t.as_ref()),
        Cy3MetricResultKind::Schoen(t) => Cy3MetricResultBackground::from_schoen(t.as_ref()),
    };

    let hym_cfg = HymConfig {
        max_iter: 8,
        damping: 0.5,
        ..HymConfig::default()
    };
    let h_v = solve_hym_metric(bundle, &bg, &hym_cfg);

    let result = compute_hodge_channel(
        &candidate,
        bundle,
        &bg,
        &h_v,
        wilson,
        &laplacian_config,
        channel_config,
    )
    .map_err(|e| format!("{label}: hodge channel failed: {e}"))?;

    let dt = t.elapsed().as_secs_f64();
    eprintln!("  [{}] hodge channel: {:.1}s", label, dt);
    Ok(result)
}

fn print_candidate(c: &HodgeChannelResult) {
    eprintln!("\n=== {} ===", c.candidate);
    eprintln!(
        "  predicted (h11, h21, χ) = ({}, {}, {})",
        c.predicted_h11, c.predicted_h21, c.predicted_chi
    );
    eprintln!(
        "  measured  (h11, h21, χ) = ({}, {}, {})  [kernel total K = {}]",
        c.measured_h11, c.measured_h21, c.measured_chi, c.measured_kernel_total
    );
    eprintln!(
        "  log-likelihood = {:.6} nats",
        c.log_likelihood_match
    );
    eprintln!(
        "  λ_max = {:.6e}    kernel cutoff = {:.6e}    λ_lowest_nonzero = {}",
        c.lambda_max,
        c.kernel_eigenvalue_threshold,
        match c.lambda_lowest_nonzero {
            Some(v) => format!("{:.6e}", v),
            None => "<none>".to_string(),
        }
    );
    eprintln!(
        "  basis dim (post-projection) = {}    n_points = {}",
        c.final_basis_dim, c.n_points
    );
    if c.kernel_count_truncated {
        eprintln!("  WARNING: kernel count was truncated to {}", c.measured_kernel_total);
    }
}

fn main() {
    let mut cli = Cli::parse();
    if cli.fast {
        cli.n_pts = 1000;
        cli.max_iter = 16;
    }

    eprintln!("================================================");
    eprintln!("P8.2 — Hodge-number consistency discrimination channel");
    eprintln!("  predicted (h11, h21, χ) = (3, 3, -6) downstairs");
    eprintln!("================================================");
    eprintln!(
        "  n_pts={}  k={}  max_iter={}  seed={}  τ_kernel={:.0e}  σ_h={}  σ_χ={}  h4={}",
        cli.n_pts,
        cli.k,
        cli.max_iter,
        cli.seed,
        cli.kernel_zero_thresh,
        cli.sigma_h,
        cli.sigma_chi,
        !cli.no_h4
    );

    let bundle = MonadBundle::anderson_lukas_palti_example();
    let wilson = Z3xZ3WilsonLines::canonical_aklp_schoen();

    // Sanity-check Wilson lines.
    let (q1, q2) = wilson.quantization_residuals();
    if q1 > 1.0e-7 || q2 > 1.0e-7 {
        eprintln!(
            "FATAL: Z/3 × Z/3 Wilson-line pair not properly Z/3-quantized: q1={:.3e} q2={:.3e}",
            q1, q2
        );
        std::process::exit(1);
    }

    let chan_cfg = HodgeChannelConfig {
        kernel_zero_thresh: cli.kernel_zero_thresh,
        sigma_h: cli.sigma_h,
        sigma_chi: cli.sigma_chi,
        ..HodgeChannelConfig::default()
    };

    let t_total = Instant::now();
    let mut schoen_result: Option<HodgeChannelResult> = None;
    let mut ty_result: Option<HodgeChannelResult> = None;

    if !cli.skip_schoen {
        eprintln!("\n--- Schoen / Z3×Z3 Wilson-line bundle ---");
        let spec = Cy3MetricSpec::Schoen {
            d_x: cli.k,
            d_y: cli.k,
            d_t: 1,
            n_sample: cli.n_pts,
            max_iter: cli.max_iter,
            donaldson_tol: cli.donaldson_tol,
            seed: cli.seed,
        };
        let lap_cfg = Z3xZ3BundleConfig {
            geometry: Z3xZ3Geometry::Schoen,
            apply_h4: !cli.no_h4,
            seed_max_total_degree: cli.seed_max_total_degree,
            orthogonalize_first: cli.orthogonalize_first,
            ..Z3xZ3BundleConfig::default()
        };
        match run_one(
            "Schoen/Z3xZ3",
            HodgeCandidateSpec::schoen_z3xz3(),
            spec,
            &SchoenSolver,
            &bundle,
            &wilson,
            lap_cfg,
            &chan_cfg,
        ) {
            Ok(c) => {
                print_candidate(&c);
                schoen_result = Some(c);
            }
            Err(e) => eprintln!("  SKIPPED: {e}"),
        }
    }

    if !cli.skip_ty {
        eprintln!("\n--- TY / Z3 Wilson-line bundle ---");
        let spec = Cy3MetricSpec::TianYau {
            k: cli.k,
            n_sample: cli.n_pts,
            max_iter: cli.max_iter,
            donaldson_tol: cli.donaldson_tol,
            seed: cli.seed,
        };
        let lap_cfg = Z3xZ3BundleConfig {
            geometry: Z3xZ3Geometry::TianYau,
            apply_h4: !cli.no_h4,
            seed_max_total_degree: cli.seed_max_total_degree,
            orthogonalize_first: cli.orthogonalize_first,
            ..Z3xZ3BundleConfig::default()
        };
        match run_one(
            "TY/Z3",
            HodgeCandidateSpec::ty_z3(),
            spec,
            &TianYauSolver,
            &bundle,
            &wilson,
            lap_cfg,
            &chan_cfg,
        ) {
            Ok(c) => {
                print_candidate(&c);
                ty_result = Some(c);
            }
            Err(e) => eprintln!("  SKIPPED: {e}"),
        }
    }

    let total_elapsed_s = t_total.elapsed().as_secs_f64();

    let (delta, summary) = match (&ty_result, &schoen_result) {
        (Some(ty), Some(s)) => {
            let d = HodgeChannelResult::log_bayes_factor_ty_vs_schoen(ty, s);
            let s = if d > 0.0 {
                format!("Hodge channel favours TY by {:.3} nats", d)
            } else if d < 0.0 {
                format!("Hodge channel favours Schoen by {:.3} nats", -d)
            } else {
                "Hodge channel: TY and Schoen tied".to_string()
            };
            (Some(d), s)
        }
        _ => (None, "Discrimination unavailable (one or both candidates skipped/failed)".to_string()),
    };

    eprintln!("\n================================================");
    eprintln!("DISCRIMINATION (Hodge channel only):");
    eprintln!("  Δ ln L = ln L_TY - ln L_Schoen = {}", match delta {
        Some(d) => format!("{:.6} nats", d),
        None => "<unavailable>".to_string(),
    });
    eprintln!("  {}", summary);
    eprintln!("================================================");

    let mut notes = Vec::new();
    notes.push(
        "Channel: kernel count K of bundle Laplacian Δ_∂̄^V on Z/3×Z/3 (or Z/3) trivial-rep sub-bundle"
            .to_string(),
    );
    notes.push("Symmetric split: h11 = K/2, h21 = K - h11; χ = h11 - h21".to_string());
    notes.push("Gaussian log-likelihood at predicted (3, 3, -6); σ_h = 0.5, σ_χ = 1.0".to_string());
    notes.push(
        "Hodge ≃ harmonic forms: H^{p,q}(M, V) ≃ ker Δ_∂̄^V|_{(p,q)-forms} (Griffiths-Harris 1978 §0.7)"
            .to_string(),
    );

    let report = DiagnosticReport {
        label: "p8_2_hodge_diagnostic",
        n_pts: cli.n_pts,
        k: cli.k,
        max_iter: cli.max_iter,
        seed: cli.seed,
        kernel_zero_thresh: cli.kernel_zero_thresh,
        sigma_h: cli.sigma_h,
        sigma_chi: cli.sigma_chi,
        h4_applied: !cli.no_h4,
        schoen: schoen_result,
        ty: ty_result,
        delta_log_likelihood_ty_minus_schoen: delta,
        discrimination_summary: summary,
        total_elapsed_s,
        notes,
    };

    if let Some(parent) = cli.output.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let json = serde_json::to_string_pretty(&report).expect("serialise report");
    fs::write(&cli.output, json).expect("write diagnostic JSON");
    eprintln!("\nWrote {}", cli.output.display());
    eprintln!("Total elapsed: {:.1}s", total_elapsed_s);
}
