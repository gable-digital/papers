//! P8.3 — Yukawa-eigenvalue / mass-spectrum production sweep
//! comparing TY/Z3 vs Schoen/Z3xZ3.
//!
//! ## Goal
//!
//! P5.6/P5.6b proved `predict_fermion_masses` runs end-to-end on TY/Z3
//! once the BBW-correct kernel-selection knob (`auto_use_predicted_dim`)
//! is engaged. P8.3 lifts that result to a **production-scale**
//! discrimination channel:
//!
//!   * Solve Donaldson-balanced metrics for BOTH TY/Z3 and
//!     Schoen/Z3xZ3 at canonical settings (n_pts=25 000, k=3,
//!     iter_cap=100, tol=1e-6, seed=12345).
//!   * Run the publication-grade Yukawa pipeline on each.
//!   * Score predicted fermion masses against PDG 2024 via χ².
//!   * Report Δ(log-likelihood) and the n-σ-equivalent
//!     discrimination from the Yukawa channel alone.
//!
//! ## Geometry / bundle wiring
//!
//! TY/Z3 — single Z/3 quotient, the AKLP example bundle on the
//!   bicubic CP^3×CP^3, single-Z/3 Wilson line
//!   `WilsonLineE8::canonical_e8_to_e6_su3(3)`. This is the same
//!   wiring P5.6b used.
//!
//! Schoen/Z3xZ3 — Z/3×Z/3 quotient on CP^2×CP^2×CP^1. The pipeline
//!   signature takes a single `WilsonLineE8`, not the
//!   `Z3xZ3WilsonLines` from `wilson_line_e8_z3xz3.rs`. We follow
//!   the convention already used by `bayes_discriminate.rs`
//!   `production_candidates`: pass `WilsonLineE8::canonical_e8_to_e6_su3(9)`
//!   (quotient order 9 = 3·3, encoding the joint Z/3×Z/3
//!   action's torsion order). The Z/3×Z/3 commutator structure is
//!   not exposed through the existing `predict_fermion_masses` API
//!   surface; it would require a Z3xZ3-aware overload of the
//!   Wilson-line argument, which is owned by the `route34` layer
//!   (P7.6) and explicitly out of scope for this P8.3 binary.
//!
//! ## Output
//!
//! Writes `output/p8_3_yukawa_production.json` and prints a
//! per-particle table of predicted-vs-PDG residuals to stderr.
//!
//! Run with `cargo run --release --features "gpu precision-bigfloat"
//! --bin p8_3_yukawa_production -- --output output/p8_3_yukawa_production.json`.

use clap::Parser;
use cy3_rust_solver::pdg::{chi_squared_test, ChiSquaredResult, Pdg2024};
use cy3_rust_solver::route34::cy3_metric_unified::{
    Cy3MetricResultKind, Cy3MetricSolver, Cy3MetricSpec, SchoenSolver, TianYauSolver,
};
use cy3_rust_solver::route34::hym_hermitian::HymConfig;
use cy3_rust_solver::route34::wilson_line_e8::WilsonLineE8;
use cy3_rust_solver::route34::yukawa_overlap_real::YukawaConfig as YukawaPipelineConfig;
use cy3_rust_solver::route34::yukawa_pipeline::{
    log_chi2_masses, predict_fermion_masses, predict_fermion_masses_with_overrides,
    Cy3MetricResultBackground, FermionMassPrediction, PipelineConfig,
};
use cy3_rust_solver::route34::zero_modes_harmonic::HarmonicConfig;
use cy3_rust_solver::zero_modes::{AmbientCY3, MonadBundle};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "P8.3 Yukawa production sweep (TY/Z3 vs Schoen/Z3xZ3)")]
struct Cli {
    /// Number of CY3 sample points per candidate (canonical: 25 000).
    #[arg(long, default_value_t = 25_000)]
    n_pts: usize,

    /// Bigraded section degree (canonical: k=3).
    #[arg(long, default_value_t = 3)]
    k: u32,

    /// Donaldson iteration cap (canonical: 100).
    #[arg(long, default_value_t = 100)]
    iter_cap: usize,

    /// Donaldson Frobenius tolerance (canonical: 1e-6).
    #[arg(long, default_value_t = 1.0e-6)]
    donaldson_tol: f64,

    /// PRNG seed (canonical: 12345).
    #[arg(long, default_value_t = 12345)]
    seed: u64,

    /// Output JSON path.
    #[arg(long, default_value = "output/p8_3b_yukawa_production.json")]
    output: PathBuf,
}

// ---------------------------------------------------------------------
// PDG 2024 reference values (MeV) — used for residuals_ppm reporting.
// The pipeline's internal chi-squared uses Pdg2024::default() with
// asymmetric error bars.
// ---------------------------------------------------------------------

const PARTICLE_NAMES: [&str; 9] = [
    "m_e", "m_mu", "m_tau",
    "m_u", "m_d", "m_s",
    "m_c", "m_b", "m_t",
];

/// PDG 2024 central values in MeV.
/// (m_u, m_d, m_s at 2 GeV; m_c at m_c; m_b at m_b; m_t pole.)
const PDG_CENTRAL_MEV: [f64; 9] = [
    0.51099895069, // m_e
    105.6583755,   // m_mu
    1776.86,       // m_tau
    2.16,          // m_u (2 GeV)
    4.67,          // m_d (2 GeV)
    93.4,          // m_s (2 GeV)
    1273.0,        // m_c (m_c)
    4183.0,        // m_b (m_b)
    172570.0,      // m_t (pole)
];

/// PDG 2024 (symmetrized) sigma values in MeV. Used for the
/// log-space chi^2 computation (Blocker 3 fix). Asymmetric
/// uncertainties are symmetrized to the larger half-width.
/// Source: same as the in-pipeline `Pdg2024::default()` table.
const PDG_SIGMA_MEV: [f64; 9] = [
    1.6e-13, // m_e
    2.3e-6,  // m_mu
    0.12,    // m_tau
    0.49,    // m_u (2 GeV) symmetrized
    0.48,    // m_d (2 GeV) symmetrized
    8.6,     // m_s (2 GeV) symmetrized
    8.0,     // m_c (m_c)
    7.0,     // m_b (m_b)
    700.0,   // m_t (pole)
];

#[derive(Debug, Serialize, Deserialize, Clone)]
struct PdgEntry {
    name: &'static str,
    central_mev: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct PerCandidateRecord {
    label: String,
    geometry: String,
    bundle: String,
    /// Status: "OK" or error message.
    status: String,
    /// Donaldson-balanced metric solve residual (final).
    donaldson_residual: Option<f64>,
    /// σ residual at the FS-Gram identity (`h = I`).
    sigma_fs_identity: Option<f64>,
    /// Final Donaldson-balanced σ residual.
    sigma_final: Option<f64>,
    /// Number of section-basis monomials.
    n_basis: Option<usize>,
    /// Donaldson iterations executed.
    iterations_run: Option<usize>,
    /// Number of accepted sample points after weight filtering.
    n_accepted_points: Option<usize>,
    /// Predicted masses in MeV (m_e, m_mu, m_tau, m_u, m_d, m_s, m_c, m_b, m_t).
    predicted_masses_mev: Option<[f64; 9]>,
    /// Per-particle residuals in ppm: (predicted - observed) / observed * 1e6.
    residuals_ppm: Option<[f64; 9]>,
    /// Per-particle PDG-σ-weighted χ² contributions (from `chi_squared_test`).
    chi2_per_particle: Option<[f64; 9]>,
    /// χ² across the 9 mass entries (m_e ... m_t).
    chi2_masses: Option<f64>,
    /// χ² across all 13 PDG entries (masses + CKM + Jarlskog).
    chi2_total: Option<f64>,
    /// **Blocker 3 fix.** Per-particle log-space chi^2 contributions.
    /// Avoids the m_e PDG-σ-floor saturation that erased the
    /// discrimination signal in P8.3. See
    /// [`cy3_rust_solver::route34::yukawa_pipeline::log_chi2_per_particle`].
    log_chi2_per_particle: Option<[f64; 9]>,
    /// Log-space χ² across the 9 mass entries.
    log_chi2_masses: Option<f64>,
    /// Log-likelihood = -0.5 * chi2_total (Gaussian-likelihood proxy).
    log_likelihood: Option<f64>,
    /// Log-likelihood derived from log-space chi^2 (Blocker 3 channel).
    log_likelihood_from_log_chi2: Option<f64>,
    /// **Blocker 2 diagnostic.** Whether the Schoen pipeline had to
    /// fall back to `kernel_dim_target = Some(9)` after the BBW
    /// cohomology prediction returned 0 on the 3-factor Schoen
    /// ambient.
    used_kernel_dim_fallback: Option<bool>,
    /// HYM final residual.
    hym_residual: Option<f64>,
    /// Cohomology dim used by harmonic kernel (BBW count).
    cohomology_dim_predicted: Option<usize>,
    /// Cohomology dim observed (kernel rank actually selected).
    cohomology_dim_observed: Option<usize>,
    /// Yukawa-tensor max bootstrap relative uncertainty.
    yukawa_uncertainty_relative: Option<f64>,
    /// Quadrature uniformity (1.0 = perfect).
    quadrature_uniformity_score: Option<f64>,
    /// Convergence ratio for triple overlap.
    convergence_ratio: Option<f64>,
    /// Harmonic basis orthonormality residual.
    harmonic_orthonormality_residual: Option<f64>,
    /// Wallclock for metric solve (s).
    elapsed_s_metric: f64,
    /// Wallclock for Yukawa pipeline (s).
    elapsed_s_yukawa: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ProductionReport {
    label: &'static str,
    settings: Settings,
    pdg_reference: Vec<PdgEntry>,
    candidates: Vec<&'static str>,
    per_candidate: Vec<PerCandidateRecord>,
    /// Δ(log-likelihood) = log_lik(TY) - log_lik(Schoen).
    /// Positive → TY fits PDG better (lower χ²).
    delta_log_likelihood_ty_minus_schoen: Option<f64>,
    /// n-σ-equivalent discrimination from Yukawa-mass channel alone:
    /// sqrt(2 * |Δ log L|) (Wilks-equivalent for nested Gaussian
    /// likelihoods, used elsewhere in this codebase as a fast
    /// discrimination proxy).
    discrimination_n_sigma: Option<f64>,
    verdict: String,
    total_elapsed_s: f64,
    git_revision: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Settings {
    n_pts: usize,
    k: u32,
    iter_cap: usize,
    donaldson_tol: f64,
    seed: u64,
    /// Asymptotic rule: harmonic kernel selection uses
    /// `auto_use_predicted_dim = true` (BBW-correct, P5.6b fix).
    harmonic_auto_use_predicted_dim: bool,
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

/// Schoen (d_x, d_y, d_t) for a given "k" label, matching the
/// convention in `p5_10_ty_schoen_5sigma.rs::schoen_tuple_for_k`.
fn schoen_tuple_for_k(k: u32) -> (u32, u32, u32) {
    match k {
        2 => (2, 2, 1),
        3 => (3, 3, 1),
        4 => (4, 4, 2),
        other => panic!("unsupported k for Schoen mapping: {}", other),
    }
}

fn pipeline_config() -> PipelineConfig {
    // BBW-correct kernel selection — the P5.6b fix. The legacy
    // `kernel_eigenvalue_ratio = 1e-3` returns an empty kernel on
    // Donaldson-balanced metrics where the lowest Δ-eigenvalues sit
    // at the Bergman-kernel residual (~1.2), not at zero.
    PipelineConfig {
        hym: HymConfig {
            max_iter: 8,
            damping: 0.5,
            ..HymConfig::default()
        },
        harmonic: HarmonicConfig {
            auto_use_predicted_dim: true,
            ..HarmonicConfig::default()
        },
        yukawa: YukawaPipelineConfig {
            n_bootstrap: 8,
            ..YukawaPipelineConfig::default()
        },
        ..PipelineConfig::default()
    }
}

/// Convert a `FermionMassPrediction` (masses in GeV at M_Z) into the
/// canonical 9-vector ordering used here, in MeV.
fn masses_mev_from_prediction(p: &FermionMassPrediction) -> [f64; 9] {
    [
        p.lepton_masses_mz[0]    * 1.0e3, // m_e (GeV → MeV)
        p.lepton_masses_mz[1]    * 1.0e3, // m_mu
        p.lepton_masses_mz[2]    * 1.0e3, // m_tau
        p.up_quark_masses_mz[0]  * 1.0e3, // m_u
        p.down_quark_masses_mz[0] * 1.0e3, // m_d
        p.down_quark_masses_mz[1] * 1.0e3, // m_s
        p.up_quark_masses_mz[1]  * 1.0e3, // m_c
        p.down_quark_masses_mz[2] * 1.0e3, // m_b
        p.up_quark_masses_mz[2]  * 1.0e3, // m_t
    ]
}

fn residuals_ppm(predicted_mev: &[f64; 9]) -> [f64; 9] {
    let mut out = [0.0; 9];
    for i in 0..9 {
        let obs = PDG_CENTRAL_MEV[i];
        if obs.abs() > 1.0e-30 && predicted_mev[i].is_finite() {
            out[i] = (predicted_mev[i] - obs) / obs * 1.0e6;
        } else {
            out[i] = f64::NAN;
        }
    }
    out
}

/// Extract per-particle χ² contributions from `pipeline_chi_squared`.
/// Indexes the `terms` vector by name, in our ordering.
fn chi2_per_particle_from_result(r: &ChiSquaredResult) -> ([f64; 9], f64) {
    // The pdg term names are: m_e, m_mu, m_tau, m_u(2GeV), m_d(2GeV),
    // m_s(2GeV), m_c(m_c), m_b(m_b), m_t(pole), |V_us|, |V_cb|,
    // |V_ub|, J. We map our 9 mass slots to these by name.
    let lookup = |name: &str| -> f64 {
        r.terms
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.chi2_contribution)
            .unwrap_or(f64::NAN)
    };
    let arr = [
        lookup("m_e"),
        lookup("m_mu"),
        lookup("m_tau"),
        lookup("m_u(2GeV)"),
        lookup("m_d(2GeV)"),
        lookup("m_s(2GeV)"),
        lookup("m_c(m_c)"),
        lookup("m_b(m_b)"),
        lookup("m_t(pole)"),
    ];
    let chi2_masses: f64 = arr.iter().filter(|v| v.is_finite()).sum();
    (arr, chi2_masses)
}

/// Run Yukawa pipeline on TY/Z3 and produce a record.
fn run_ty(cli: &Cli, pcfg: &PipelineConfig) -> PerCandidateRecord {
    let t_metric = Instant::now();
    let solver = TianYauSolver;
    let spec = Cy3MetricSpec::TianYau {
        k: cli.k,
        n_sample: cli.n_pts,
        max_iter: cli.iter_cap,
        donaldson_tol: cli.donaldson_tol,
        seed: cli.seed,
    };
    let r = match solver.solve_metric(&spec) {
        Ok(r) => r,
        Err(e) => {
            return PerCandidateRecord {
                label: "TY/Z3".to_string(),
                geometry: "TianYau bicubic CP^3 x CP^3, Z/3 quotient".to_string(),
                bundle: "AKLP (Anderson-Lukas-Palti) SU(5) monad bundle, single-Z/3 Wilson line".to_string(),
                status: format!("metric_solve_failed: {e}"),
                donaldson_residual: None,
                sigma_fs_identity: None,
                sigma_final: None,
                n_basis: None,
                iterations_run: None,
                n_accepted_points: None,
                predicted_masses_mev: None,
                residuals_ppm: None,
                chi2_per_particle: None,
                chi2_masses: None,
                chi2_total: None,
                log_chi2_per_particle: None,
                log_chi2_masses: None,
                log_likelihood: None,
                log_likelihood_from_log_chi2: None,
                used_kernel_dim_fallback: None,
                hym_residual: None,
                cohomology_dim_predicted: None,
                cohomology_dim_observed: None,
                yukawa_uncertainty_relative: None,
                quadrature_uniformity_score: None,
                convergence_ratio: None,
                harmonic_orthonormality_residual: None,
                elapsed_s_metric: t_metric.elapsed().as_secs_f64(),
                elapsed_s_yukawa: 0.0,
            };
        }
    };
    let elapsed_s_metric = t_metric.elapsed().as_secs_f64();
    let summary = r.summary();
    let bg = match &r {
        Cy3MetricResultKind::TianYau(t) => Cy3MetricResultBackground::from_ty(t.as_ref()),
        Cy3MetricResultKind::Schoen(_) => unreachable!("TY solver returned non-TY result"),
    };
    let n_accepted = bg.n_accepted();

    let bundle = MonadBundle::anderson_lukas_palti_example();
    let ambient = AmbientCY3::tian_yau_upstairs();
    let wilson = WilsonLineE8::canonical_e8_to_e6_su3(3);

    let t_yuk = Instant::now();
    let pred_result = predict_fermion_masses(&bundle, &ambient, &bg, &wilson, pcfg);
    let elapsed_s_yukawa = t_yuk.elapsed().as_secs_f64();

    match pred_result {
        Ok(pred) => {
            let masses_mev = masses_mev_from_prediction(&pred);
            let residuals = residuals_ppm(&masses_mev);
            let pdg = Pdg2024::default();
            let obs = pred.to_pdg_observables();
            let chi2_res = chi_squared_test(&obs, &pdg);
            let (chi2_pp, chi2_m) = chi2_per_particle_from_result(&chi2_res);
            let log_lik = -0.5 * chi2_res.chi2_total;
            // Blocker 3: log-space chi^2 (avoids m_e PDG-σ-floor saturation).
            let (log_chi2_pp, log_chi2_m) =
                log_chi2_masses(&masses_mev, &PDG_CENTRAL_MEV, &PDG_SIGMA_MEV);
            let log_lik_log = -0.5 * log_chi2_m;
            PerCandidateRecord {
                label: "TY/Z3".to_string(),
                geometry: "TianYau bicubic CP^3 x CP^3, Z/3 quotient".to_string(),
                bundle: "AKLP (Anderson-Lukas-Palti) SU(5) monad bundle, single-Z/3 Wilson line".to_string(),
                status: "OK".to_string(),
                donaldson_residual: Some(summary.final_donaldson_residual),
                sigma_fs_identity: Some(summary.sigma_fs_identity),
                sigma_final: Some(summary.final_sigma_residual),
                n_basis: Some(summary.n_basis),
                iterations_run: Some(summary.iterations_run),
                n_accepted_points: Some(n_accepted),
                predicted_masses_mev: Some(masses_mev),
                residuals_ppm: Some(residuals),
                chi2_per_particle: Some(chi2_pp),
                chi2_masses: Some(chi2_m),
                chi2_total: Some(chi2_res.chi2_total),
                log_chi2_per_particle: Some(log_chi2_pp),
                log_chi2_masses: Some(log_chi2_m),
                log_likelihood: Some(log_lik),
                log_likelihood_from_log_chi2: Some(log_lik_log),
                used_kernel_dim_fallback: Some(false),
                hym_residual: Some(pred.hym_residual),
                cohomology_dim_predicted: Some(pred.cohomology_dim_predicted),
                cohomology_dim_observed: Some(pred.cohomology_dim_observed),
                yukawa_uncertainty_relative: Some(pred.yukawa_uncertainty_relative),
                quadrature_uniformity_score: Some(pred.quadrature_uniformity_score),
                convergence_ratio: Some(pred.convergence_ratio),
                harmonic_orthonormality_residual: Some(pred.harmonic_orthonormality_residual),
                elapsed_s_metric,
                elapsed_s_yukawa,
            }
        }
        Err(e) => PerCandidateRecord {
            label: "TY/Z3".to_string(),
            geometry: "TianYau bicubic CP^3 x CP^3, Z/3 quotient".to_string(),
            bundle: "AKLP (Anderson-Lukas-Palti) SU(5) monad bundle, single-Z/3 Wilson line".to_string(),
            status: format!("yukawa_pipeline_failed: {e}"),
            donaldson_residual: Some(summary.final_donaldson_residual),
            sigma_fs_identity: Some(summary.sigma_fs_identity),
            sigma_final: Some(summary.final_sigma_residual),
            n_basis: Some(summary.n_basis),
            iterations_run: Some(summary.iterations_run),
            n_accepted_points: Some(n_accepted),
            predicted_masses_mev: None,
            residuals_ppm: None,
            chi2_per_particle: None,
            chi2_masses: None,
            chi2_total: None,
            log_chi2_per_particle: None,
            log_chi2_masses: None,
            log_likelihood: None,
            log_likelihood_from_log_chi2: None,
            used_kernel_dim_fallback: None,
            hym_residual: None,
            cohomology_dim_predicted: None,
            cohomology_dim_observed: None,
            yukawa_uncertainty_relative: None,
            quadrature_uniformity_score: None,
            convergence_ratio: None,
            harmonic_orthonormality_residual: None,
            elapsed_s_metric,
            elapsed_s_yukawa,
        },
    }
}

/// Run Yukawa pipeline on Schoen/Z3xZ3 and produce a record.
fn run_schoen(cli: &Cli, pcfg: &PipelineConfig) -> PerCandidateRecord {
    let t_metric = Instant::now();
    let (d_x, d_y, d_t) = schoen_tuple_for_k(cli.k);
    let solver = SchoenSolver;
    let spec = Cy3MetricSpec::Schoen {
        d_x,
        d_y,
        d_t,
        n_sample: cli.n_pts,
        max_iter: cli.iter_cap,
        donaldson_tol: cli.donaldson_tol,
        seed: cli.seed,
    };
    let r = match solver.solve_metric(&spec) {
        Ok(r) => r,
        Err(e) => {
            return PerCandidateRecord {
                label: "Schoen/Z3xZ3".to_string(),
                geometry: format!("Schoen CP^2 x CP^2 x CP^1, (d_x,d_y,d_t)=({d_x},{d_y},{d_t}), Z/3xZ/3 quotient"),
                bundle: "AKLP placeholder bundle, WilsonLineE8 with quotient_order=9 (3*3 torsion)".to_string(),
                status: format!("metric_solve_failed: {e}"),
                donaldson_residual: None,
                sigma_fs_identity: None,
                sigma_final: None,
                n_basis: None,
                iterations_run: None,
                n_accepted_points: None,
                predicted_masses_mev: None,
                residuals_ppm: None,
                chi2_per_particle: None,
                chi2_masses: None,
                chi2_total: None,
                log_chi2_per_particle: None,
                log_chi2_masses: None,
                log_likelihood: None,
                log_likelihood_from_log_chi2: None,
                used_kernel_dim_fallback: None,
                hym_residual: None,
                cohomology_dim_predicted: None,
                cohomology_dim_observed: None,
                yukawa_uncertainty_relative: None,
                quadrature_uniformity_score: None,
                convergence_ratio: None,
                harmonic_orthonormality_residual: None,
                elapsed_s_metric: t_metric.elapsed().as_secs_f64(),
                elapsed_s_yukawa: 0.0,
            };
        }
    };
    let elapsed_s_metric = t_metric.elapsed().as_secs_f64();
    let summary = r.summary();
    let bg = match &r {
        Cy3MetricResultKind::Schoen(s) => Cy3MetricResultBackground::from_schoen(s.as_ref()),
        Cy3MetricResultKind::TianYau(_) => unreachable!("Schoen solver returned non-Schoen result"),
    };
    let n_accepted = bg.n_accepted();

    // **Blocker 2 fix.** Schoen-canonical monad bundle wired through
    // the new `predict_fermion_masses_with_overrides` driver, which
    // auto-applies the `kernel_dim_target = Some(9)` fallback if the
    // BBW prediction returns 0 (the standard outcome on the 3-factor
    // Schoen ambient because [`MonadBundle::chern_classes`] is
    // hard-coded to nf == 2). The new constructor is documented as
    // the canonical Schoen-side bundle (Z/3 × Z/3 Wilson-line
    // structure documented in `wilson_line_e8_z3xz3.rs`).
    let bundle = MonadBundle::schoen_z3xz3_canonical();
    let ambient = AmbientCY3::schoen_z3xz3_upstairs();
    let wilson = WilsonLineE8::canonical_e8_to_e6_su3(9);

    let t_yuk = Instant::now();
    let pred_result = predict_fermion_masses_with_overrides(
        &bundle, &ambient, &bg, &wilson, pcfg, 9,
    );
    let elapsed_s_yukawa = t_yuk.elapsed().as_secs_f64();

    match pred_result {
        Ok((pred, used_fallback)) => {
            if used_fallback {
                eprintln!(
                    "  Schoen pipeline auto-applied kernel_dim_target=Some(9) \
                     fallback (BBW prediction was 0 on 3-factor ambient)."
                );
            }
            let masses_mev = masses_mev_from_prediction(&pred);
            let residuals = residuals_ppm(&masses_mev);
            let pdg = Pdg2024::default();
            let obs = pred.to_pdg_observables();
            let chi2_res = chi_squared_test(&obs, &pdg);
            let (chi2_pp, chi2_m) = chi2_per_particle_from_result(&chi2_res);
            let log_lik = -0.5 * chi2_res.chi2_total;
            // Blocker 3: log-space chi^2 (avoids m_e PDG-σ-floor saturation).
            let (log_chi2_pp, log_chi2_m) =
                log_chi2_masses(&masses_mev, &PDG_CENTRAL_MEV, &PDG_SIGMA_MEV);
            let log_lik_log = -0.5 * log_chi2_m;
            PerCandidateRecord {
                label: "Schoen/Z3xZ3".to_string(),
                geometry: format!("Schoen CP^2 x CP^2 x CP^1, (d_x,d_y,d_t)=({d_x},{d_y},{d_t}), Z/3xZ/3 quotient"),
                bundle: "Schoen Z/3×Z/3 canonical monad bundle, WilsonLineE8 with quotient_order=9".to_string(),
                status: "OK".to_string(),
                donaldson_residual: Some(summary.final_donaldson_residual),
                sigma_fs_identity: Some(summary.sigma_fs_identity),
                sigma_final: Some(summary.final_sigma_residual),
                n_basis: Some(summary.n_basis),
                iterations_run: Some(summary.iterations_run),
                n_accepted_points: Some(n_accepted),
                predicted_masses_mev: Some(masses_mev),
                residuals_ppm: Some(residuals),
                chi2_per_particle: Some(chi2_pp),
                chi2_masses: Some(chi2_m),
                chi2_total: Some(chi2_res.chi2_total),
                log_chi2_per_particle: Some(log_chi2_pp),
                log_chi2_masses: Some(log_chi2_m),
                log_likelihood: Some(log_lik),
                log_likelihood_from_log_chi2: Some(log_lik_log),
                used_kernel_dim_fallback: Some(used_fallback),
                hym_residual: Some(pred.hym_residual),
                cohomology_dim_predicted: Some(pred.cohomology_dim_predicted),
                cohomology_dim_observed: Some(pred.cohomology_dim_observed),
                yukawa_uncertainty_relative: Some(pred.yukawa_uncertainty_relative),
                quadrature_uniformity_score: Some(pred.quadrature_uniformity_score),
                convergence_ratio: Some(pred.convergence_ratio),
                harmonic_orthonormality_residual: Some(pred.harmonic_orthonormality_residual),
                elapsed_s_metric,
                elapsed_s_yukawa,
            }
        }
        Err(e) => PerCandidateRecord {
            label: "Schoen/Z3xZ3".to_string(),
            geometry: format!("Schoen CP^2 x CP^2 x CP^1, (d_x,d_y,d_t)=({d_x},{d_y},{d_t}), Z/3xZ/3 quotient"),
            bundle: "AKLP placeholder bundle, WilsonLineE8 with quotient_order=9 (3*3 torsion)".to_string(),
            status: format!("yukawa_pipeline_failed: {e}"),
            donaldson_residual: Some(summary.final_donaldson_residual),
            sigma_fs_identity: Some(summary.sigma_fs_identity),
            sigma_final: Some(summary.final_sigma_residual),
            n_basis: Some(summary.n_basis),
            iterations_run: Some(summary.iterations_run),
            n_accepted_points: Some(n_accepted),
            predicted_masses_mev: None,
            residuals_ppm: None,
            chi2_per_particle: None,
            chi2_masses: None,
            chi2_total: None,
            log_chi2_per_particle: None,
            log_chi2_masses: None,
            log_likelihood: None,
            log_likelihood_from_log_chi2: None,
            used_kernel_dim_fallback: None,
            hym_residual: None,
            cohomology_dim_predicted: None,
            cohomology_dim_observed: None,
            yukawa_uncertainty_relative: None,
            quadrature_uniformity_score: None,
            convergence_ratio: None,
            harmonic_orthonormality_residual: None,
            elapsed_s_metric,
            elapsed_s_yukawa,
        },
    }
}

fn print_table(rec: &PerCandidateRecord) {
    eprintln!();
    eprintln!("=== {} ===", rec.label);
    eprintln!("  status: {}", rec.status);
    if let (Some(_), Some(masses), Some(res), Some(chi2_pp)) = (
        &rec.predicted_masses_mev,
        &rec.predicted_masses_mev,
        &rec.residuals_ppm,
        &rec.chi2_per_particle,
    ) {
        eprintln!(
            "  | {:>6} | {:>14} | {:>14} | {:>14} | {:>10} |",
            "name", "predicted(MeV)", "PDG(MeV)", "residual(ppm)", "χ²"
        );
        for i in 0..9 {
            eprintln!(
                "  | {:>6} | {:>14.6e} | {:>14.6e} | {:>14.3e} | {:>10.3e} |",
                PARTICLE_NAMES[i],
                masses[i],
                PDG_CENTRAL_MEV[i],
                res[i],
                chi2_pp[i],
            );
        }
        if let (Some(chi2_m), Some(chi2_t), Some(ll)) =
            (rec.chi2_masses, rec.chi2_total, rec.log_likelihood)
        {
            eprintln!(
                "  χ²(masses)={:.3e}  χ²(total,13dof)={:.3e}  log_L = -χ²/2 = {:.3e}",
                chi2_m, chi2_t, ll
            );
        }
        if let (Some(log_chi2_m), Some(log_lik_log)) =
            (rec.log_chi2_masses, rec.log_likelihood_from_log_chi2)
        {
            eprintln!(
                "  log-χ²(masses)={:.3e}  log_L_from_log_chi2 = {:.3e}",
                log_chi2_m, log_lik_log
            );
            if let Some(log_chi2_pp) = &rec.log_chi2_per_particle {
                let mut s = String::from("  log-χ² per particle:");
                for i in 0..9 {
                    s.push_str(&format!(" {}={:.3e}", PARTICLE_NAMES[i], log_chi2_pp[i]));
                }
                eprintln!("{}", s);
            }
        }
        if let (Some(hd), Some(o), Some(p)) = (
            rec.hym_residual,
            rec.cohomology_dim_observed,
            rec.cohomology_dim_predicted,
        ) {
            eprintln!(
                "  hym_res={:.3e}  coh_dim observed/predicted={}/{}",
                hd, o, p
            );
        }
    }
    eprintln!(
        "  metric_solve = {:.1}s, yukawa = {:.1}s",
        rec.elapsed_s_metric, rec.elapsed_s_yukawa
    );
}

fn main() {
    let cli = Cli::parse();
    let t_total = Instant::now();
    eprintln!("=== P8.3 Yukawa production sweep ===");
    eprintln!(
        "n_pts={}, k={}, iter_cap={}, donaldson_tol={:.0e}, seed={}",
        cli.n_pts, cli.k, cli.iter_cap, cli.donaldson_tol, cli.seed
    );
    eprintln!("PDG 2024 reference (MeV):");
    for i in 0..9 {
        eprintln!("  {:>6} = {:.6e}", PARTICLE_NAMES[i], PDG_CENTRAL_MEV[i]);
    }

    let pcfg = pipeline_config();

    eprintln!();
    eprintln!("--- TY/Z3 ---");
    let rec_ty = run_ty(&cli, &pcfg);
    print_table(&rec_ty);

    eprintln!();
    eprintln!("--- Schoen/Z3xZ3 ---");
    let rec_schoen = run_schoen(&cli, &pcfg);
    print_table(&rec_schoen);

    // **Blocker 3 fix.** Discrimination signal computed from the
    // log-space chi^2 (`log_likelihood_from_log_chi2`) when both
    // candidates have it, falling back to the linear chi^2
    // log-likelihood otherwise. The log-space channel is the
    // discrimination signal that survives PDG-σ-floor saturation
    // (m_e σ = 1.6e-13 MeV makes the linear chi^2 saturate at
    // ~1e25 on any predicted m_e ≠ observed).
    let (delta_ll, n_sigma, verdict) = match (
        rec_ty.log_likelihood_from_log_chi2,
        rec_schoen.log_likelihood_from_log_chi2,
    ) {
        (Some(ll_ty), Some(ll_schoen)) => {
            let d = ll_ty - ll_schoen;
            let n_sigma = (2.0 * d.abs()).sqrt();
            let v = if !d.is_finite() {
                "INCONCLUSIVE: non-finite log-likelihoods (log-χ² channel)".to_string()
            } else if d > 0.0 {
                format!(
                    "TY/Z3 fits PDG better than Schoen/Z3xZ3 (log-χ² channel): \
                     Δ log L = +{:.3e}, n-σ ≈ {:.3}",
                    d, n_sigma
                )
            } else if d < 0.0 {
                format!(
                    "Schoen/Z3xZ3 fits PDG better than TY/Z3 (log-χ² channel): \
                     Δ log L = {:.3e}, n-σ ≈ {:.3}",
                    d, n_sigma
                )
            } else {
                "Tie: Δ log L = 0 (log-χ² channel)".to_string()
            };
            (Some(d), Some(n_sigma), v)
        }
        (None, None) => (
            None,
            None,
            "INCONCLUSIVE: both Yukawa pipelines failed (no log-likelihoods)".to_string(),
        ),
        (None, Some(_)) => (
            None,
            None,
            "ASYMMETRIC: TY pipeline failed; only Schoen produced a fit".to_string(),
        ),
        (Some(_), None) => (
            None,
            None,
            "ASYMMETRIC: Schoen pipeline failed; only TY produced a fit".to_string(),
        ),
    };

    eprintln!();
    eprintln!("=== Discrimination ===");
    eprintln!("  {}", verdict);

    let total_elapsed_s = t_total.elapsed().as_secs_f64();
    eprintln!();
    eprintln!("=== Total elapsed: {:.1}s ===", total_elapsed_s);

    // Persist JSON.
    let report = ProductionReport {
        label: "p8_3_yukawa_production",
        settings: Settings {
            n_pts: cli.n_pts,
            k: cli.k,
            iter_cap: cli.iter_cap,
            donaldson_tol: cli.donaldson_tol,
            seed: cli.seed,
            harmonic_auto_use_predicted_dim: true,
        },
        pdg_reference: PARTICLE_NAMES
            .iter()
            .zip(PDG_CENTRAL_MEV.iter())
            .map(|(name, &c)| PdgEntry {
                name,
                central_mev: c,
            })
            .collect(),
        candidates: vec!["TY/Z3", "Schoen/Z3xZ3"],
        per_candidate: vec![rec_ty, rec_schoen],
        delta_log_likelihood_ty_minus_schoen: delta_ll,
        discrimination_n_sigma: n_sigma,
        verdict,
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
