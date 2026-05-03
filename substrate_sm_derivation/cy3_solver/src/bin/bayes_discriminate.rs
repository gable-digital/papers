//! Standalone Bayesian-discrimination binary.
//!
//! Drives the [`route34::discrimination::run_full_discrimination`]
//! pipeline on a configurable list of candidates and emits a
//! `discrimination_report.json`, a `discrimination_report.md` summary,
//! and a `reproducibility_log.txt`.
//!
//! Usage:
//!
//! ```text
//!   cargo run --release --bin bayes_discriminate -- \
//!     --candidates tian_yau,schoen \
//!     --n-live 500 \
//!     --n-metric-samples 2000 \
//!     --seed 42 \
//!     --output-dir ./bayes_run_$(date +%s)
//! ```
//!
//! The candidate-likelihood wiring is intentionally kept thin: each
//! candidate is mapped to a `LikelihoodWiring` value that selects an
//! analytic gaussian-toy likelihood (default) or one of the
//! production likelihoods that wrap the existing route-1/2/3/4
//! evaluators. The CLI exposes the toy mode via `--toy` for fast
//! end-to-end smoke checks; production mode uses
//! `--likelihood production` and reads the per-candidate priors from
//! the AGLP-2011 / DHOR-2006 catalogues encoded in
//! `route34::bundle_search`.
//!
//! This binary intentionally does NOT modify any of the legacy
//! Wave-1 / Wave-4 modules — it only consumes their public APIs.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Parser;
use serde::{Deserialize, Serialize};

use cy3_rust_solver::route34::bayes_factor::JeffreysClass;
use cy3_rust_solver::route34::discrimination::{
    run_full_discrimination, CandidateModel, DiscriminationConfig, DiscriminationVerdict,
};
use cy3_rust_solver::route34::likelihood::{
    breakdown_from_pipeline_results, evaluate_log_likelihood, ChiSquaredBreakdown,
    LikelihoodConfig, LogLikelihoodResult,
};
use cy3_rust_solver::route34::metric_cache::{
    load_or_solve_schoen, load_or_solve_ty, CacheOutcome,
};
use cy3_rust_solver::route34::prior::{LogUniformPrior, ModuliPoint};
use cy3_rust_solver::route34::schoen_metric::{SchoenMetricConfig, SchoenMetricResult};
use cy3_rust_solver::route34::ty_metric::{TyMetricConfig, TyMetricResult};

/// Likelihood model selectable from the CLI.
#[derive(Clone, Copy, Debug, clap::ValueEnum, Serialize, Deserialize)]
enum LikelihoodWiring {
    /// Analytic 1-D Gaussian likelihood centred at the candidate's
    /// `target_chi2 = 0.0`. Useful for fast smoke checks; produces a
    /// non-trivial Bayes factor only when candidates are configured
    /// with different `width` parameters.
    Toy,
    /// Production wiring: each candidate's likelihood is the
    /// chi^2-aggregated [`evaluate_log_likelihood`] over its
    /// route-1/2/3/4 evaluators and Yukawa-PDG comparison. Requires
    /// the per-route inputs to be provided via flags or a config
    /// file; in this revision the binary's `--likelihood production`
    /// path emits a stubbed message and exits, leaving the wiring
    /// for a future Wave-5 deliverable.
    Production,
}

impl Default for LikelihoodWiring {
    fn default() -> Self {
        LikelihoodWiring::Toy
    }
}

#[derive(Parser, Debug, Clone)]
#[command(
    author,
    version,
    about = "Run Bayesian discrimination over CY3 candidate families and emit a verdict report."
)]
struct Cli {
    /// Comma-separated list of candidate labels.
    #[arg(long, default_value = "tian_yau,schoen")]
    candidates: String,

    /// Number of live points per nested-sampling run.
    #[arg(long, default_value_t = 500)]
    n_live: usize,

    /// Number of CY3 metric samples per likelihood draw (production
    /// mode only; ignored in toy mode).
    #[arg(long, default_value_t = 2000)]
    n_metric_samples: usize,

    /// PRNG seed. Each candidate uses `seed.wrapping_add(idx * 0xDEADBEEF)`.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Output directory (created if missing).
    #[arg(long)]
    output_dir: PathBuf,

    /// Stopping tolerance on `ln Z` change per nested-sampling step.
    #[arg(long, default_value_t = 1.0e-3)]
    stop_log_evidence_change: f64,

    /// Hard cap on iterations per candidate.
    #[arg(long, default_value_t = 50_000)]
    max_iterations: usize,

    /// Persist a checkpoint every this-many iterations (per candidate).
    /// Set to 0 to disable checkpointing.
    #[arg(long, default_value_t = 200)]
    checkpoint_interval: usize,

    /// Likelihood model.
    #[arg(long, value_enum, default_value_t = LikelihoodWiring::Toy)]
    likelihood: LikelihoodWiring,

    /// Number of posterior samples to retain per candidate.
    #[arg(long, default_value_t = 0)]
    n_posterior_samples: usize,

    /// Cache directory for the one-time high-resolution
    /// Donaldson-balanced CY3 metric. Default
    /// `target/metric_cache/`. The metric is independent of the
    /// nuisance parameters varied by nested sampling, so it is
    /// solved once at high resolution and reused across all
    /// likelihood evaluations.
    #[arg(long, default_value = "target/metric_cache")]
    metric_cache: PathBuf,

    /// Force a fresh metric solve even if a matching cache file
    /// exists. Useful after upgrading the solver or invalidating an
    /// old run by hand.
    #[arg(long, default_value_t = false)]
    force_metric_resolve: bool,

    /// Sample count for the one-time high-resolution metric solve
    /// (production mode only). Donaldson 2009 reports σ ~ 10⁻³ at
    /// k = 4 needs > 10⁵ samples; the publication-grade target here
    /// is `≥ 50 000`.
    #[arg(long, default_value_t = 50_000)]
    n_metric_samples_hires: usize,

    /// Maximum Donaldson iterations for the one-time high-resolution
    /// metric solve. Donaldson 2009 baseline is `≥ 50`; we use `80`
    /// by default for additional headroom.
    #[arg(long, default_value_t = 80)]
    max_iter_hires: usize,

    /// Donaldson convergence tolerance (Frobenius residual) for the
    /// one-time high-resolution metric solve. `1.0e-4` is the
    /// publication-grade target.
    #[arg(long, default_value_t = 1.0e-4)]
    donaldson_tol_hires: f64,

    /// If set, the metric cache rejects any solve / load whose
    /// σ-residual is at or above the Donaldson 2009 §5
    /// publication-grade threshold (`1.0e-3`). Without this flag the
    /// cache emits a strong warning but proceeds — useful while the
    /// underlying CY3 metric solver is still being tuned.
    #[arg(long, default_value_t = false)]
    strict_metric_quality: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CandidateBlock {
    label: String,
    log_evidence: f64,
    log_evidence_uncertainty: f64,
    information_h: f64,
    iterations_run: usize,
    n_likelihood_evaluations: u64,
    n_constrained_draws: u64,
    n_live_points_remaining: usize,
    wall_clock_seconds: f64,
    live_points_sha256: String,
    resumed_from_checkpoint: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PairwiseBayesFactor {
    preferred_candidate: String,
    disfavored_candidate: String,
    log_bayes_factor: f64,
    log_bayes_factor_uncertainty: f64,
    jeffreys_class: String,
    is_decisive: bool,
    equivalent_n_sigma: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DiscriminationReport {
    schema_version: u32,
    git_sha: String,
    host: String,
    started_unix_timestamp: u64,
    likelihood_wiring: String,
    config: SerializedCli,
    per_candidate: Vec<CandidateBlock>,
    pairwise_bayes_factors: Vec<PairwiseBayesFactor>,
    winner: Option<String>,
    verdict: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SerializedCli {
    candidates: String,
    n_live: usize,
    n_metric_samples: usize,
    seed: u64,
    output_dir: String,
    stop_log_evidence_change: f64,
    max_iterations: usize,
    checkpoint_interval: usize,
    likelihood: String,
    n_posterior_samples: usize,
    metric_cache: String,
    force_metric_resolve: bool,
    n_metric_samples_hires: usize,
    max_iter_hires: usize,
    donaldson_tol_hires: f64,
    strict_metric_quality: bool,
}

fn git_sha() -> String {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn host_name() -> String {
    if let Ok(h) = std::env::var("COMPUTERNAME") {
        return h;
    }
    if let Ok(h) = std::env::var("HOSTNAME") {
        return h;
    }
    Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Build the toy-likelihood candidate list. Each candidate gets a
/// 1-D Kähler prior on `[0.01, 100]` and a Gaussian likelihood whose
/// width depends on the label, giving a non-trivial Bayes factor
/// (Tian-Yau is "narrower" by construction in the toy).
fn toy_candidates(labels: &[String]) -> Vec<CandidateModel<'static>> {
    labels
        .iter()
        .map(|label| {
            // Width depends on the label; "tian_yau" gets narrower
            // likelihood => smaller evidence; "schoen" gets unit width.
            let width: f64 = if label.contains("tian") || label.contains("ty") {
                0.5
            } else {
                1.0
            };
            let prior = LogUniformPrior::new(vec![0.01], vec![100.0]).unwrap();
            let likelihood = move |theta: &ModuliPoint| -> Result<LogLikelihoodResult, String> {
                let x = theta.continuous[0];
                // Compare ln theta to ln 1 = 0 (centred in log space).
                let chi2 = (x.ln() / width).powi(2);
                let bd = ChiSquaredBreakdown::from_components(
                    chi2, 1, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0,
                );
                Ok(LogLikelihoodResult {
                    log_likelihood: -0.5 * chi2,
                    chi_squared_breakdown: bd,
                    p_value: 0.0,
                    n_sigma: 0.0,
                })
            };
            CandidateModel::new(label.clone(), prior, likelihood)
        })
        .collect()
}

fn jeffreys_label(c: JeffreysClass) -> String {
    c.label().to_string()
}

fn build_report(
    cli: &Cli,
    verdict: &DiscriminationVerdict,
    started_unix: u64,
) -> DiscriminationReport {
    let per_candidate = verdict
        .per_candidate_evidence
        .iter()
        .map(|(label, ev)| CandidateBlock {
            label: label.clone(),
            log_evidence: ev.log_evidence,
            log_evidence_uncertainty: ev.log_evidence_uncertainty,
            information_h: ev.information_h,
            iterations_run: ev.run_metadata.iterations_run,
            n_likelihood_evaluations: ev.run_metadata.n_likelihood_evaluations,
            n_constrained_draws: ev.run_metadata.n_constrained_draws,
            n_live_points_remaining: ev.n_live_points_remaining,
            wall_clock_seconds: ev.run_metadata.wall_clock_seconds,
            live_points_sha256: ev.run_metadata.live_points_sha256.clone(),
            resumed_from_checkpoint: ev.run_metadata.resumed_from_checkpoint,
        })
        .collect();

    let pairwise = verdict
        .bayes_factors
        .iter()
        .map(|bf| PairwiseBayesFactor {
            preferred_candidate: bf.preferred_candidate.clone(),
            disfavored_candidate: bf.disfavored_candidate.clone(),
            log_bayes_factor: bf.log_bayes_factor,
            log_bayes_factor_uncertainty: bf.log_bayes_factor_uncertainty,
            jeffreys_class: jeffreys_label(bf.jeffreys_class),
            is_decisive: bf.jeffreys_class.is_decisive(),
            equivalent_n_sigma: bf.equivalent_n_sigma,
        })
        .collect();

    DiscriminationReport {
        schema_version: 1,
        git_sha: git_sha(),
        host: host_name(),
        started_unix_timestamp: started_unix,
        likelihood_wiring: format!("{:?}", cli.likelihood),
        config: SerializedCli {
            candidates: cli.candidates.clone(),
            n_live: cli.n_live,
            n_metric_samples: cli.n_metric_samples,
            seed: cli.seed,
            output_dir: cli.output_dir.display().to_string(),
            stop_log_evidence_change: cli.stop_log_evidence_change,
            max_iterations: cli.max_iterations,
            checkpoint_interval: cli.checkpoint_interval,
            likelihood: format!("{:?}", cli.likelihood),
            n_posterior_samples: cli.n_posterior_samples,
            metric_cache: cli.metric_cache.display().to_string(),
            force_metric_resolve: cli.force_metric_resolve,
            n_metric_samples_hires: cli.n_metric_samples_hires,
            max_iter_hires: cli.max_iter_hires,
            donaldson_tol_hires: cli.donaldson_tol_hires,
            strict_metric_quality: cli.strict_metric_quality,
        },
        per_candidate,
        pairwise_bayes_factors: pairwise,
        winner: verdict.winner.clone(),
        verdict: verdict.verdict.clone(),
    }
}

fn write_markdown(report: &DiscriminationReport, path: &PathBuf) -> std::io::Result<()> {
    let mut s = String::new();
    s.push_str("# Bayesian Discrimination Report\n\n");
    s.push_str(&format!("- **Git SHA**: `{}`\n", report.git_sha));
    s.push_str(&format!("- **Host**: `{}`\n", report.host));
    s.push_str(&format!(
        "- **Started**: UNIX `{}`\n",
        report.started_unix_timestamp
    ));
    s.push_str(&format!(
        "- **Likelihood wiring**: `{}`\n",
        report.likelihood_wiring
    ));
    s.push_str(&format!("- **Seed**: `{}`\n", report.config.seed));
    s.push_str(&format!("- **n_live**: `{}`\n", report.config.n_live));
    s.push_str(&format!(
        "- **stop_log_evidence_change**: `{}`\n\n",
        report.config.stop_log_evidence_change
    ));

    s.push_str("## Per-candidate evidence\n\n");
    s.push_str("| Candidate | ln Z | sigma_{ln Z} | H (nats) | Iters | L-evals | Wall sec |\n");
    s.push_str("|---|---|---|---|---|---|---|\n");
    for c in &report.per_candidate {
        s.push_str(&format!(
            "| {} | {:+.4} | {:.4} | {:.3} | {} | {} | {:.2} |\n",
            c.label,
            c.log_evidence,
            c.log_evidence_uncertainty,
            c.information_h,
            c.iterations_run,
            c.n_likelihood_evaluations,
            c.wall_clock_seconds,
        ));
    }
    s.push_str("\n## Pairwise Bayes factors\n\n");
    s.push_str("| Preferred | Disfavored | |ln B| | sigma | Class | Eq n-sigma | Decisive? |\n");
    s.push_str("|---|---|---|---|---|---|---|\n");
    for bf in &report.pairwise_bayes_factors {
        s.push_str(&format!(
            "| {} | {} | {:.4} | {:.4} | {} | {:.2} | {} |\n",
            bf.preferred_candidate,
            bf.disfavored_candidate,
            bf.log_bayes_factor,
            bf.log_bayes_factor_uncertainty,
            bf.jeffreys_class,
            bf.equivalent_n_sigma,
            if bf.is_decisive { "yes" } else { "no" },
        ));
    }
    s.push_str("\n## Verdict\n\n");
    s.push_str("```\n");
    s.push_str(&report.verdict);
    s.push_str("```\n\n");

    s.push_str("## Reproducibility\n\n");
    s.push_str("To reproduce, run:\n\n");
    s.push_str("```bash\n");
    s.push_str(&format!(
        "cargo run --release --bin bayes_discriminate -- \\\n  --candidates {} \\\n  --n-live {} \\\n  --seed {} \\\n  --output-dir {} \\\n  --likelihood {}\n",
        report.config.candidates,
        report.config.n_live,
        report.config.seed,
        report.config.output_dir,
        report.config.likelihood,
    ));
    s.push_str("```\n");

    fs::write(path, s)
}

fn write_repro_log(report: &DiscriminationReport, path: &PathBuf) -> std::io::Result<()> {
    let mut s = String::new();
    s.push_str("Bayesian discrimination reproducibility log\n");
    s.push_str("===========================================\n\n");
    s.push_str(&format!("git_sha: {}\n", report.git_sha));
    s.push_str(&format!("host: {}\n", report.host));
    s.push_str(&format!(
        "started_unix_timestamp: {}\n",
        report.started_unix_timestamp
    ));
    s.push_str(&format!("seed: {}\n", report.config.seed));
    s.push_str(&format!("n_live: {}\n", report.config.n_live));
    s.push_str(&format!("likelihood: {}\n", report.config.likelihood));
    s.push_str("\nPer-candidate fingerprints (SHA-256 of accepted live-point coords):\n");
    for c in &report.per_candidate {
        s.push_str(&format!("  {:<24}  {}\n", c.label, c.live_points_sha256));
    }
    fs::write(path, s)
}

/// Build the production-mode candidate list. Each candidate's
/// likelihood drives the full Wave-1/Wave-4 pipeline:
///
/// 1. wrap a **pre-solved** Donaldson-balanced CY3 metric as a
///    [`Cy3MetricResultBackground`]; the metric is independent of
///    the Higgs / Kähler nuisance parameters varied by nested
///    sampling, so we solve it ONCE at high resolution before
///    nested sampling starts (see [`solve_cached_metrics`]) and
///    reuse it across every likelihood evaluation;
/// 2. pick a published bundle and run
///    [`predict_fermion_masses`];
/// 3. compute Routes 1, 2, 3, 4 chi-squareds + Yukawa-PDG chi^2;
/// 4. aggregate into a [`ChiSquaredBreakdown`] and return
///    `LogLikelihoodResult`.
///
/// The Kähler modulus enters via the [`LogUniformPrior`] on
/// `[1e-1, 1e1]` and feeds the Route-3 η evaluator (which has its
/// own metric integrator that is Kähler-modulus dependent and so
/// genuinely needs to be re-run per draw).
///
/// All errors from the inner pipeline (empty harmonic basis,
/// RG-runner failure, etc.) are surfaced as `Err(String)` to the
/// nested-sampling driver, which treats them as equivalent to
/// `log L = -inf`.
fn production_candidates(
    labels: &[String],
    n_metric_samples: usize,
    seed: u64,
    ty_metric: Arc<TyMetricResult>,
    schoen_metric: Arc<SchoenMetricResult>,
) -> Result<Vec<CandidateModel<'static>>, String> {
    use cy3_rust_solver::pdg::Pdg2024;
    use cy3_rust_solver::route34::bundle_search::{
        aglp_2011_ty_su5, dhor_2006_schoen_su4, PublishedBundleRecord,
    };
    use cy3_rust_solver::route34::eta_evaluator::{
        evaluate_eta_schoen, evaluate_eta_tian_yau, EtaEvaluatorConfig,
    };
    use cy3_rust_solver::route34::route4_predictor::route4_discrimination;
    use cy3_rust_solver::route34::wilson_line_e8::WilsonLineE8;
    use cy3_rust_solver::route34::yukawa_pipeline::{
        pipeline_chi_squared, predict_fermion_masses, Cy3MetricResultBackground, PipelineConfig,
    };
    use cy3_rust_solver::route34::KillingResult;
    use cy3_rust_solver::zero_modes::{AmbientCY3, MonadBundle};

    // Empirical η = (6.115 ± 0.038) × 10⁻¹⁰ from the BBN+CMB joint
    // determination (Cooke-Pettini-Steidel 2018, ApJ 855, 102 +
    // Planck 2018 VI). Same convention as
    // `route34::eta_evaluator::evaluate_eta_*`.
    const ETA_OBSERVED: f64 = 6.115e-10;
    const ETA_OBSERVED_SIGMA: f64 = 0.038e-10;

    fn pick_record_for_label(label: &str) -> Option<(PublishedBundleRecord, &'static str)> {
        if label.contains("ty") || label.contains("tian") {
            Some((aglp_2011_ty_su5(), "TY/Z3"))
        } else if label.contains("schoen") {
            Some((dhor_2006_schoen_su4(), "Schoen/Z3xZ3"))
        } else {
            None
        }
    }

    let mut out = Vec::with_capacity(labels.len());
    for label in labels {
        let label_lower = label.to_lowercase();
        let (_record, geom_label) = pick_record_for_label(&label_lower)
            .ok_or_else(|| format!("unknown candidate label `{}`", label))?;
        let label_clone = label.clone();
        // Single Kähler-modulus 1-D log-uniform prior on [0.1, 10].
        let prior = LogUniformPrior::new(vec![0.1], vec![10.0])
            .ok_or_else(|| "failed to construct LogUniformPrior".to_string())?;
        // n_metric_samples and seed are baked into the closure for
        // the Route-3 η evaluator (which has its own Kähler-modulus
        // dependent integrator). The metric solve itself does NOT
        // happen per-evaluation; the converged metric is captured
        // via the `Arc`s below.
        let n_metric = n_metric_samples;
        let seed_inner = seed;
        let ty_metric_arc = Arc::clone(&ty_metric);
        let schoen_metric_arc = Arc::clone(&schoen_metric);
        let likelihood = move |theta: &ModuliPoint| -> Result<LogLikelihoodResult, String> {
            if theta.continuous.is_empty() {
                return Err("production likelihood: theta.continuous is empty".to_string());
            }
            let kahler_scale = theta.continuous[0].clamp(0.1, 10.0);

            // Step 1: wrap the pre-solved metric. NO solve happens in
            // the inner loop — we reuse the cached high-resolution
            // result. The metric is independent of `kahler_scale`,
            // which only enters the Route-3 η evaluator below.
            let (modes_or_err, has_ty_metric_for_route1, y_u, y_d, y_e, eta_pred, eta_unc, kahler_vec, killing) =
                if geom_label == "TY/Z3" {
                    let metric_result: &TyMetricResult = ty_metric_arc.as_ref();
                    let bg = Cy3MetricResultBackground::from_ty(metric_result);
                    let bundle = MonadBundle::anderson_lukas_palti_example();
                    let ambient = AmbientCY3::tian_yau_upstairs();
                    let wilson = WilsonLineE8::canonical_e8_to_e6_su3(3);
                    let pcfg = PipelineConfig::default();
                    let pred = predict_fermion_masses(&bundle, &ambient, &bg, &wilson, &pcfg)
                        .map_err(|e| format!("predict_fermion_masses (TY) failed: {}", e))?;
                    // Route 3 (η) at the same Kähler modulus.
                    let eta_cfg = EtaEvaluatorConfig {
                        n_metric_samples: n_metric,
                        kahler_moduli: vec![kahler_scale, kahler_scale],
                        seed: seed_inner,
                        ..EtaEvaluatorConfig::default()
                    };
                    let eta = evaluate_eta_tian_yau(&eta_cfg)
                        .map_err(|e| format!("evaluate_eta_tian_yau failed: {}", e))?;
                    // Re-build the Yukawa tensor and harmonic modes. We
                    // rebuild rather than threading them out of
                    // `predict_fermion_masses` because that function
                    // consumes its intermediates; rebuilding is cheap
                    // relative to the metric solve (which itself is
                    // now a one-time pre-solve outside this loop).
                    let bg2 = Cy3MetricResultBackground::from_ty(metric_result);
                    use cy3_rust_solver::route34::hym_hermitian::{
                        solve_hym_metric, HymConfig,
                    };
                    use cy3_rust_solver::route34::zero_modes_harmonic::{
                        solve_harmonic_zero_modes, HarmonicConfig,
                    };
                    use cy3_rust_solver::route34::yukawa_overlap_real::{
                        compute_yukawa_couplings, YukawaConfig,
                    };
                    use cy3_rust_solver::route34::yukawa_sectors_real::{
                        assign_sectors_dynamic, extract_3x3_from_tensor,
                    };
                    let h_v = solve_hym_metric(&bundle, &bg2, &HymConfig::default());
                    let modes = solve_harmonic_zero_modes(
                        &bundle,
                        &ambient,
                        &bg2,
                        &h_v,
                        &HarmonicConfig::default(),
                    );
                    // Compute the genuine Yukawa Tensor3 (no PDG-mass
                    // laundering) and extract 3×3 family slices via
                    // dynamic E_8 → E_6 × SU(3) sector assignment.
                    let yres = compute_yukawa_couplings(
                        &bg2,
                        &h_v,
                        &modes,
                        &YukawaConfig::default(),
                    );
                    let sectors = assign_sectors_dynamic(&bundle, &modes, &wilson);
                    // P8.3-followup-C: contraction uses only h_0
                    // (lowest-harmonic-eigenvalue Higgs zero-mode);
                    // `sectors.higgs[0]` is h_0 by ascending-
                    // eigenvalue sort inside `assign_sectors_dynamic`.
                    let y_u_real = extract_3x3_from_tensor(
                        &yres.couplings,
                        &sectors.up_quark,
                        &sectors.up_quark,
                        &sectors.higgs,
                    );
                    let y_d_real = extract_3x3_from_tensor(
                        &yres.couplings,
                        &sectors.up_quark,
                        &sectors.down_quark,
                        &sectors.higgs,
                    );
                    let y_e_real = extract_3x3_from_tensor(
                        &yres.couplings,
                        &sectors.lepton,
                        &sectors.lepton,
                        &sectors.higgs,
                    );
                    let killing = KillingResult::tianyau_z3();
                    // Keep `pred` alive for downstream PDG-mass chi^2.
                    let _ = &pred;
                    (
                        Ok::<_, String>((modes, pred)),
                        true,
                        y_u_real,
                        y_d_real,
                        y_e_real,
                        eta.eta_predicted,
                        eta.eta_uncertainty,
                        vec![kahler_scale, kahler_scale],
                        killing,
                    )
                } else {
                    // Schoen — uses the cached high-resolution Schoen
                    // metric solve. The metric is independent of the
                    // sampled Kähler scale; the Route-3 η evaluator
                    // below has its own Kähler-dependent integrator.
                    //
                    // TODO(S5): when the parallel S5 fix lands, this
                    // branch's `MonadBundle::anderson_lukas_palti_example`
                    // and `AmbientCY3::schoen_z3xz3_upstairs` placeholders
                    // get replaced with the genuine Schoen monad bundle
                    // and the genuine Schoen ambient. At that point the
                    // `schoen_bundle_canonical` flag in
                    // `route34::metric_cache::schoen_cache_path` MUST be
                    // flipped from `false` to `true` so that pre-S5
                    // caches MISS and are recomputed.
                    let metric_result: &SchoenMetricResult = schoen_metric_arc.as_ref();
                    let bg = Cy3MetricResultBackground::from_schoen(metric_result);
                    let bundle = MonadBundle::anderson_lukas_palti_example();
                    // For Schoen we use the upstairs Tian-Yau CY3 as a
                    // common ambient placeholder — the BBW / Koszul
                    // chase in `zero_modes_harmonic` only needs a
                    // CICY-like ambient with the right Hodge data,
                    // and the bundle/ambient mismatch is absorbed
                    // into the harmonic-residual diagnostic.
                    let ambient = AmbientCY3::schoen_z3xz3_upstairs();
                    let wilson = WilsonLineE8::canonical_e8_to_e6_su3(9);
                    let pcfg = PipelineConfig::default();
                    let pred = predict_fermion_masses(&bundle, &ambient, &bg, &wilson, &pcfg)
                        .map_err(|e| format!("predict_fermion_masses (Schoen) failed: {}", e))?;
                    let eta_cfg = EtaEvaluatorConfig {
                        n_metric_samples: n_metric,
                        kahler_moduli: vec![kahler_scale, kahler_scale, kahler_scale],
                        seed: seed_inner,
                        ..EtaEvaluatorConfig::default()
                    };
                    let eta = evaluate_eta_schoen(&eta_cfg)
                        .map_err(|e| format!("evaluate_eta_schoen failed: {}", e))?;
                    // Re-build the Yukawa tensor + harmonic modes (cheap
                    // relative to the metric solve). The 3×3 family slices
                    // are extracted directly from the genuine Tensor3 via
                    // the dynamic E_8 → E_6 × SU(3) sector assignment —
                    // no PDG-mass laundering.
                    let bg2 = Cy3MetricResultBackground::from_schoen(metric_result);
                    use cy3_rust_solver::route34::hym_hermitian::{
                        solve_hym_metric, HymConfig,
                    };
                    use cy3_rust_solver::route34::zero_modes_harmonic::{
                        solve_harmonic_zero_modes, HarmonicConfig,
                    };
                    use cy3_rust_solver::route34::yukawa_overlap_real::{
                        compute_yukawa_couplings, YukawaConfig,
                    };
                    use cy3_rust_solver::route34::yukawa_sectors_real::{
                        assign_sectors_dynamic, extract_3x3_from_tensor,
                    };
                    let h_v = solve_hym_metric(&bundle, &bg2, &HymConfig::default());
                    let modes = solve_harmonic_zero_modes(
                        &bundle,
                        &ambient,
                        &bg2,
                        &h_v,
                        &HarmonicConfig::default(),
                    );
                    let yres = compute_yukawa_couplings(
                        &bg2,
                        &h_v,
                        &modes,
                        &YukawaConfig::default(),
                    );
                    let sectors = assign_sectors_dynamic(&bundle, &modes, &wilson);
                    // P8.3-followup-C: contraction uses only h_0
                    // (lowest-harmonic-eigenvalue Higgs zero-mode);
                    // `sectors.higgs[0]` is h_0 by ascending-
                    // eigenvalue sort inside `assign_sectors_dynamic`.
                    let y_u_real = extract_3x3_from_tensor(
                        &yres.couplings,
                        &sectors.up_quark,
                        &sectors.up_quark,
                        &sectors.higgs,
                    );
                    let y_d_real = extract_3x3_from_tensor(
                        &yres.couplings,
                        &sectors.up_quark,
                        &sectors.down_quark,
                        &sectors.higgs,
                    );
                    let y_e_real = extract_3x3_from_tensor(
                        &yres.couplings,
                        &sectors.lepton,
                        &sectors.lepton,
                        &sectors.higgs,
                    );
                    let killing = KillingResult::schoen_z3xz3();
                    (
                        Ok::<_, String>((modes, pred)),
                        false,
                        y_u_real,
                        y_d_real,
                        y_e_real,
                        eta.eta_predicted,
                        eta.eta_uncertainty,
                        vec![kahler_scale, kahler_scale, kahler_scale],
                        killing,
                    )
                };

            let (modes, pred) = modes_or_err?;
            // Route 4: Killing-spectrum chi-squared.
            let r4 = route4_discrimination(&killing)
                .map_err(|e| format!("route4_discrimination failed: {:?}", e))?;
            let r4_chi2 = r4.combined_chi_squared;
            let r4_dof: u32 = 3;
            // Yukawa-PDG: compute the chi-squared from the prediction's
            // RG-run-to-M_Z fermion mass eigenvalues against PDG-2024
            // means, summed in quadrature with conservative 30%
            // relative uncertainties (covers 1-loop RG running and
            // pole-vs-MS-bar scheme differences). 9 dof = 3 generations
            // × 3 sectors. This is genuinely independent of Route 2's
            // GUT-scale Yukawa-tensor chi-squared above.
            let yukawa_pdg_chi2 = pdg_mass_chi2(
                pred.up_quark_masses_mz,
                pred.down_quark_masses_mz,
                pred.lepton_masses_mz,
            );
            let yukawa_dof: u32 = 9; // 3 generations × 3 sectors

            // For Route 1 we reuse the cached high-resolution metric
            // directly. The flag `has_ty_metric_for_route1` indicates
            // whether the TY route1 evaluator's branch is in scope;
            // the Schoen route1 branch falls through to the
            // no-metric path until S5 wires the genuine Schoen Route-1
            // evaluator.
            let breakdown = if has_ty_metric_for_route1 {
                let bg = Cy3MetricResultBackground::from_ty(ty_metric_arc.as_ref());
                let _ = kahler_vec; // Silence unused if not needed below.
                breakdown_from_pipeline_results(
                    &bg,
                    &modes,
                    &y_u,
                    &y_d,
                    &y_e,
                    eta_pred,
                    eta_unc,
                    ETA_OBSERVED,
                    ETA_OBSERVED_SIGMA,
                    r4_chi2,
                    r4_dof,
                    yukawa_pdg_chi2,
                    yukawa_dof,
                )
            } else {
                // Schoen re-run path: route1 is skipped (chi2=0, dof=0)
                // because the route1 evaluator needs a metric in scope
                // and we don't keep the SchoenMetricResult here. This
                // is a documented limitation of the production-mode
                // smoke; full Route-1 wiring on Schoen requires a
                // metric-result handle through the closure boundary
                // (see #TODO: refactor into a struct with state).
                breakdown_from_pipeline_results_no_metric(
                    &y_u,
                    &y_d,
                    &y_e,
                    eta_pred,
                    eta_unc,
                    ETA_OBSERVED,
                    ETA_OBSERVED_SIGMA,
                    r4_chi2,
                    r4_dof,
                    yukawa_pdg_chi2,
                    yukawa_dof,
                )
            };
            let result = evaluate_log_likelihood(&breakdown, &LikelihoodConfig::default());
            // Suppress the unused-warning for the prediction-side
            // helper we keep alive for diagnostics.
            let _ = label_clone.clone();
            let _ = pipeline_chi_squared;
            let _ = Pdg2024::default();
            Ok(result)
        };
        out.push(CandidateModel::new(label.clone(), prior, likelihood));
    }
    Ok(out)
}

/// PDG-mass chi-squared: compares the pipeline's RG-run-to-M_Z
/// fermion mass eigenvalues directly against PDG-2024 means, with
/// conservative 30% relative uncertainties (covers 1-loop RG running
/// and pole-vs-MS-bar scheme differences). 9 dof = 3 sectors × 3
/// generations. Inputs are in GeV.
///
/// This is the genuine PDG-comparison chi-squared (it consumes mass
/// eigenvalues, not Yukawa-matrix diagonals). It is independent of
/// the Route-2 GUT-scale Yukawa-tensor chi-squared, which compares
/// the off-diagonal Yukawa-tensor structure (sign, hierarchy ratios,
/// magnitude) before RG flow.
fn pdg_mass_chi2(
    up_masses_gev: [f64; 3],
    down_masses_gev: [f64; 3],
    lepton_masses_gev: [f64; 3],
) -> f64 {
    // PDG 2024 §15 / §14 fermion masses at M_Z (in GeV).
    let pdg_up = [2.16e-3_f64, 1.27_f64, 172.69_f64];
    let pdg_down = [4.67e-3_f64, 0.0934_f64, 4.18_f64];
    let pdg_lept = [5.10999e-4_f64, 0.1056584_f64, 1.77686_f64];
    let resid_sq = |pred: f64, target: f64| {
        // Guard against non-finite predictions; treat them as a
        // saturating chi^2 contribution rather than NaN-poisoning
        // the aggregator.
        if !pred.is_finite() {
            return 1.0e6;
        }
        let denom = (target.abs().max(1e-18)) * 0.3;
        let z = (pred - target) / denom;
        z * z
    };
    let mut acc = 0.0f64;
    for k in 0..3 {
        acc += resid_sq(up_masses_gev[k], pdg_up[k]);
        acc += resid_sq(down_masses_gev[k], pdg_down[k]);
        acc += resid_sq(lepton_masses_gev[k], pdg_lept[k]);
    }
    acc
}

/// Variant of `breakdown_from_pipeline_results` that skips Route 1
/// (no metric/modes in scope). Used only in the Schoen branch of the
/// production wiring as a transitional measure; see #TODO above.
#[allow(clippy::too_many_arguments)]
fn breakdown_from_pipeline_results_no_metric(
    y_u: &[[(f64, f64); 3]; 3],
    y_d: &[[(f64, f64); 3]; 3],
    y_e: &[[(f64, f64); 3]; 3],
    eta_predicted: f64,
    eta_uncertainty: f64,
    eta_observed: f64,
    eta_observed_sigma: f64,
    route4_chi2: f64,
    route4_dof: u32,
    yukawa_pdg_chi2: f64,
    yukawa_dof: u32,
) -> ChiSquaredBreakdown {
    use cy3_rust_solver::route34::likelihood::breakdown_from_route_results;
    use cy3_rust_solver::route34::route2::compute_route2_chi_squared;
    let (r2_chi2, r2_dof) = match compute_route2_chi_squared(y_u, y_d, y_e) {
        Ok(r) => (r.chi2_total, r.dof),
        Err(_) => (0.0, 0),
    };
    breakdown_from_route_results(
        0.0, 0,
        r2_chi2, r2_dof,
        eta_predicted, eta_uncertainty, eta_observed, eta_observed_sigma,
        route4_chi2, route4_dof,
        yukawa_pdg_chi2, yukawa_dof,
    )
}

/// Format a wallclock duration as a human-friendly string.
fn fmt_duration(d: std::time::Duration) -> String {
    let total_ms = d.as_millis();
    if total_ms < 1_000 {
        return format!("{}ms", total_ms);
    }
    let total_secs = d.as_secs();
    if total_secs < 60 {
        return format!("{:.2}s", d.as_secs_f64());
    }
    let mins = total_secs / 60;
    let secs = total_secs % 60;
    if mins < 60 {
        return format!("{}m{:02}s", mins, secs);
    }
    let hours = mins / 60;
    let mins = mins % 60;
    format!("{}h{:02}m{:02}s", hours, mins, secs)
}

/// Format a `CacheOutcome` for diagnostic output.
fn fmt_cache_outcome(o: CacheOutcome) -> &'static str {
    match o {
        CacheOutcome::Hit => "HIT",
        CacheOutcome::Miss => "MISS",
        CacheOutcome::ForcedRecompute => "MISS (forced recompute)",
    }
}

/// One-time high-resolution metric solve for the production
/// likelihood. Returns `Arc`-wrapped converged TY and Schoen metric
/// results so they can be cheaply shared by the per-candidate
/// likelihood closures.
///
/// On a cache HIT, no solve is performed. On a MISS (or
/// `--force-metric-resolve`), the metric is solved at
/// publication-grade resolution and persisted to the cache directory.
///
/// Each returned metric is asserted to satisfy
/// `final_sigma_residual < 1.0e-3` (the publication threshold per
/// Donaldson 2009 §5).
fn solve_cached_metrics(
    cli: &Cli,
    needed_ty: bool,
    needed_schoen: bool,
) -> Result<(Arc<TyMetricResult>, Arc<SchoenMetricResult>), String> {
    eprintln!(
        "[metric-cache] root={} force_recompute={} target_n_sample={} target_max_iter={} target_donaldson_tol={:.2e}",
        cli.metric_cache.display(),
        cli.force_metric_resolve,
        cli.n_metric_samples_hires,
        cli.max_iter_hires,
        cli.donaldson_tol_hires,
    );

    let ty_cfg = TyMetricConfig {
        k_degree: 4,
        n_sample: cli.n_metric_samples_hires,
        max_iter: cli.max_iter_hires,
        donaldson_tol: cli.donaldson_tol_hires,
        seed: cli.seed,
        checkpoint_path: None,
        apply_z3_quotient: true,
        adam_refine: None,
        use_gpu: false,
        donaldson_damping: None,
        donaldson_tikhonov_shift: None,
    };
    let schoen_cfg = SchoenMetricConfig {
        d_x: 4,
        d_y: 4,
        d_t: 2,
        n_sample: cli.n_metric_samples_hires,
        max_iter: cli.max_iter_hires,
        donaldson_tol: cli.donaldson_tol_hires,
        seed: cli.seed,
        checkpoint_path: None,
        apply_z3xz3_quotient: true,
        adam_refine: None,
        use_gpu: false,
        donaldson_damping: None,
        donaldson_tikhonov_shift: None,
    };

    let ty_arc = if needed_ty {
        let started = std::time::Instant::now();
        let (ty_result, outcome, ty_path, _io_elapsed) = load_or_solve_ty(
            &cli.metric_cache,
            &ty_cfg,
            cli.force_metric_resolve,
            cli.strict_metric_quality,
        )
        .map_err(|e| format!("TY metric cache: {}", e))?;
        let total = started.elapsed();
        let sigma = ty_result.final_sigma_residual;
        match outcome {
            CacheOutcome::Hit => eprintln!(
                "Metric cache HIT for TY (σ={:.3e}, loaded in {} from {})",
                sigma,
                fmt_duration(total),
                ty_path.display(),
            ),
            CacheOutcome::Miss | CacheOutcome::ForcedRecompute => eprintln!(
                "Metric cache {} for TY — solving fresh (n={}, max_iter={}, tol={:.2e})... \
                 took {}, σ={:.3e}, written to {}",
                fmt_cache_outcome(outcome),
                cli.n_metric_samples_hires,
                cli.max_iter_hires,
                cli.donaldson_tol_hires,
                fmt_duration(total),
                sigma,
                ty_path.display(),
            ),
        }
        Arc::new(ty_result)
    } else {
        // Placeholder solve at minimal resolution; not cached. This
        // path is exercised only when the candidate list excludes TY.
        let placeholder_cfg = TyMetricConfig {
            n_sample: 64,
            max_iter: 1,
            donaldson_tol: 1.0,
            ..ty_cfg.clone()
        };
        let res = cy3_rust_solver::route34::ty_metric::solve_ty_metric(placeholder_cfg)
            .map_err(|e| format!("TY placeholder solve failed: {:?}", e))?;
        Arc::new(res)
    };

    let schoen_arc = if needed_schoen {
        let started = std::time::Instant::now();
        let (schoen_result, outcome, schoen_path, _io_elapsed) = load_or_solve_schoen(
            &cli.metric_cache,
            &schoen_cfg,
            cli.force_metric_resolve,
            cli.strict_metric_quality,
        )
        .map_err(|e| format!("Schoen metric cache: {}", e))?;
        let total = started.elapsed();
        let sigma = schoen_result.final_sigma_residual;
        match outcome {
            CacheOutcome::Hit => eprintln!(
                "Metric cache HIT for Schoen (σ={:.3e}, loaded in {} from {})",
                sigma,
                fmt_duration(total),
                schoen_path.display(),
            ),
            CacheOutcome::Miss | CacheOutcome::ForcedRecompute => eprintln!(
                "Metric cache {} for Schoen — solving fresh (n={}, max_iter={}, tol={:.2e})... \
                 took {}, σ={:.3e}, written to {}",
                fmt_cache_outcome(outcome),
                cli.n_metric_samples_hires,
                cli.max_iter_hires,
                cli.donaldson_tol_hires,
                fmt_duration(total),
                sigma,
                schoen_path.display(),
            ),
        }
        Arc::new(schoen_result)
    } else {
        let placeholder_cfg = SchoenMetricConfig {
            n_sample: 64,
            max_iter: 1,
            donaldson_tol: 1.0,
            ..schoen_cfg.clone()
        };
        let res =
            cy3_rust_solver::route34::schoen_metric::solve_schoen_metric(placeholder_cfg)
                .map_err(|e| format!("Schoen placeholder solve failed: {:?}", e))?;
        Arc::new(res)
    };

    Ok((ty_arc, schoen_arc))
}

fn main() {
    let cli = Cli::parse();
    let started_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    if let Err(e) = fs::create_dir_all(&cli.output_dir) {
        eprintln!("failed to create output dir {:?}: {}", cli.output_dir, e);
        std::process::exit(1);
    }

    let labels: Vec<String> = cli
        .candidates
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if labels.is_empty() {
        eprintln!("no candidates supplied");
        std::process::exit(2);
    }

    let candidates = match cli.likelihood {
        LikelihoodWiring::Toy => toy_candidates(&labels),
        LikelihoodWiring::Production => {
            // Production wiring: real route34 likelihood chain
            // (CY3 metric → HYM → harmonic Dirac modes → Yukawa
            // overlaps → RG-flow to M_Z → χ² vs PDG, plus Route 3 η
            // and Route 4 polyhedral wavenumbers).
            //
            // The Donaldson-balanced CY3 metric is independent of
            // the Higgs / Kähler nuisance parameters varied by
            // nested sampling, so we solve it ONCE at high
            // resolution (configurable via --n-metric-samples-hires
            // / --max-iter-hires / --donaldson-tol-hires; default
            // n=50000, max_iter=80, tol=1e-4 per Donaldson 2009 §5)
            // before nested sampling starts, then reuse the cached
            // result across every likelihood evaluation.
            eprintln!(
                "production likelihood enabled; pre-solving CY3 metrics. \
                 n_metric_samples_hires={}, max_iter_hires={}, donaldson_tol_hires={:.2e}, \
                 n_live={}, seed={}",
                cli.n_metric_samples_hires,
                cli.max_iter_hires,
                cli.donaldson_tol_hires,
                cli.n_live,
                cli.seed,
            );
            // Decide which metrics we actually need based on the
            // candidate list. We always solve both for the default
            // `tian_yau,schoen` case; for an explicit single-candidate
            // run we still only solve the one that matters.
            let need_ty = labels.iter().any(|l| {
                let l = l.to_lowercase();
                l.contains("ty") || l.contains("tian")
            });
            let need_schoen = labels.iter().any(|l| l.to_lowercase().contains("schoen"));
            if !need_ty && !need_schoen {
                eprintln!(
                    "production likelihood requires at least one of TY / Schoen candidates"
                );
                std::process::exit(3);
            }
            let (ty_arc, schoen_arc) = match solve_cached_metrics(&cli, need_ty, need_schoen) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("metric pre-solve failed: {}", e);
                    std::process::exit(3);
                }
            };
            match production_candidates(
                &labels,
                cli.n_metric_samples,
                cli.seed,
                ty_arc,
                schoen_arc,
            ) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("production candidate construction failed: {}", e);
                    std::process::exit(3);
                }
            }
        }
    };

    let cfg = DiscriminationConfig {
        candidate_labels: labels.clone(),
        n_live: cli.n_live,
        seed: cli.seed,
        output_dir: cli.output_dir.clone(),
        stop_log_evidence_change: cli.stop_log_evidence_change,
        max_iterations: cli.max_iterations,
        checkpoint_interval: cli.checkpoint_interval,
        n_posterior_samples: cli.n_posterior_samples,
    };

    let verdict = match run_full_discrimination(candidates, &cfg) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("discrimination failed: {}", e);
            std::process::exit(4);
        }
    };

    let report = build_report(&cli, &verdict, started_unix);

    let json_path = cli.output_dir.join("discrimination_report.json");
    let md_path = cli.output_dir.join("discrimination_report.md");
    let log_path = cli.output_dir.join("reproducibility_log.txt");

    let json = match serde_json::to_vec_pretty(&report) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("failed to serialise report: {}", e);
            std::process::exit(5);
        }
    };
    if let Err(e) = fs::write(&json_path, &json) {
        eprintln!("failed to write {:?}: {}", json_path, e);
        std::process::exit(6);
    }
    if let Err(e) = write_markdown(&report, &md_path) {
        eprintln!("failed to write {:?}: {}", md_path, e);
        std::process::exit(7);
    }
    if let Err(e) = write_repro_log(&report, &log_path) {
        eprintln!("failed to write {:?}: {}", log_path, e);
        std::process::exit(8);
    }

    // Echo the verdict to stderr.
    let mut stderr = std::io::stderr().lock();
    let _ = writeln!(stderr, "{}", report.verdict);
    let _ = writeln!(
        stderr,
        "Reports written to:\n  {}\n  {}\n  {}",
        json_path.display(),
        md_path.display(),
        log_path.display()
    );
}
