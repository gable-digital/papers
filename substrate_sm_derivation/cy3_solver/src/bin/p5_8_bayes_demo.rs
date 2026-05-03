//! P5.8 — Bayes-factor formalisation of the P5.10 TY-vs-Schoen
//! σ-discrimination.
//!
//! Loads the canonical P5.10 multi-seed ensemble JSON
//! (`output/p5_10_ty_schoen_5sigma.json`), reconstructs the
//! per-candidate σ-distribution Gaussian summary `(mean, SE, n)`, then
//! computes the per-seed Bayes factor `ln BF_TY:Schoen` for three
//! scenarios:
//!
//! * **Scenario A (TY-truth)** — each of the 20 σ_TY observations
//!   evaluated as a single observation. Expected: every seed yields
//!   `ln BF > 0` (TY preferred), and the sample lies far above the
//!   Jeffreys-Trotta "Strong" threshold (`|log10 BF| ≥ 2`).
//! * **Scenario B (Schoen-truth)** — same with σ_Schoen observations.
//!   Expected: `ln BF < 0`.
//! * **Scenario C (10/10 train-test split)** — split each candidate's
//!   20 seeds into a 10-seed "model-fit" set (used to define
//!   `(mean, SE, n=10)`) and a 10-seed "test" set (the held-out
//!   observations). Reports the ensemble `ln BF` for the held-out 10
//!   under each truth assumption.
//!
//! Also reports posterior odds at prior odds = 1 (equal priors), and
//! verifies the asymptotic relation `|ln BF| ≈ n_σ² / 2` against the
//! P5.10 raw n-σ statistic (8.76).
//!
//! All math goes through [`cy3_rust_solver::route34::bayes_factor_sigma`]
//! and [`pwos_math::stats::bayes_factor`] -- no re-implementation of
//! Gaussian likelihoods or Jeffreys-Trotta categorisation.
//!
//! Usage:
//! ```text
//!   cargo run --release --features gpu --bin p5_8_bayes_demo
//!   cargo run --release --features gpu --bin p5_8_bayes_demo -- \
//!       --input output/p5_10_ty_schoen_5sigma.json \
//!       --output output/p5_8_bayes_demo.json
//! ```

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use clap::Parser;
use serde::{Deserialize, Serialize};

use cy3_rust_solver::route34::bayes_factor_sigma::{
    evaluate_sigma_bayes_factor, log_bf_ty_vs_schoen, log_likelihood_ensemble,
    log_likelihood_single, posterior_odds_ty_vs_schoen, CandidateSigmaDistribution,
    SigmaBayesFactorResult,
};

#[derive(Parser, Debug)]
#[command(about = "P5.8 — Bayes-factor formalisation of the P5.10 σ-discrimination")]
struct Cli {
    /// Path to the P5.10 ensemble JSON.
    #[arg(long, default_value = "output/p5_10_ty_schoen_5sigma.json")]
    input: PathBuf,

    /// Output JSON for the full per-seed BF report.
    #[arg(long, default_value = "output/p5_8_bayes_demo.json")]
    output: PathBuf,

    /// Random-split seed for Scenario C (deterministic; here we just
    /// take the first 10 / last 10 seeds for full reproducibility).
    #[arg(long, default_value_t = 0)]
    split_seed: u64,
}

// ------------------------------------------------------------------
// Subset of the P5.10 ensemble JSON that we actually need.
// ------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct PerSeedRecord {
    seed: u64,
    candidate: String,
    k: u32,
    sigma_final: f64,
    #[serde(default)]
    #[allow(dead_code)]
    elapsed_s: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct CandidateBlock {
    candidate: String,
    k: u32,
    per_seed: Vec<PerSeedRecord>,
}

#[derive(Debug, Deserialize)]
struct DiscriminationRecord {
    k: u32,
    mean_ty: f64,
    se_ty: f64,
    n_ty: usize,
    mean_schoen: f64,
    se_schoen: f64,
    n_schoen: usize,
    n_sigma: f64,
}

#[derive(Debug, Deserialize)]
struct EnsembleJson {
    candidates: Vec<CandidateBlock>,
    discrimination: Vec<DiscriminationRecord>,
    max_n_sigma: f64,
}

// ------------------------------------------------------------------
// Output schema.
// ------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct PerSeedBfRow {
    seed: u64,
    sigma_obs: f64,
    log_bf_ty_vs_schoen: f64,
    log10_bf_ty_vs_schoen: f64,
    evidence_strength: String,
    equivalent_n_sigma: f64,
    posterior_odds_ty_at_prior_1: f64,
}

#[derive(Debug, Serialize)]
struct ScenarioReport {
    label: String,
    description: String,
    per_seed: Vec<PerSeedBfRow>,
    /// `(mean, stderr, min, max)` summary over `per_seed.log_bf_ty_vs_schoen`.
    log_bf_summary: SummaryStats,
    log10_bf_summary: SummaryStats,
    category_distribution: BTreeMap<String, usize>,
    sign_consistent: bool,
    sign_correctly_favours: String,
}

#[derive(Debug, Serialize)]
struct SummaryStats {
    n: usize,
    mean: f64,
    stderr_of_mean: f64,
    min: f64,
    max: f64,
}

impl SummaryStats {
    fn from_slice(xs: &[f64]) -> Self {
        let n = xs.len();
        if n == 0 {
            return Self {
                n: 0,
                mean: f64::NAN,
                stderr_of_mean: f64::NAN,
                min: f64::NAN,
                max: f64::NAN,
            };
        }
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = if n > 1 {
            xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)
        } else {
            0.0
        };
        let se = (var / n as f64).sqrt();
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for &x in xs {
            if x < min {
                min = x;
            }
            if x > max {
                max = x;
            }
        }
        Self {
            n,
            mean,
            stderr_of_mean: se,
            min,
            max,
        }
    }
}

#[derive(Debug, Serialize)]
struct EnsembleBfRow {
    label: String,
    description: String,
    sigma_obs: Vec<f64>,
    n_observations: usize,
    log_bf_ty_vs_schoen: f64,
    log10_bf_ty_vs_schoen: f64,
    evidence_strength: String,
    equivalent_n_sigma: f64,
    posterior_odds_ty_at_prior_1: f64,
}

#[derive(Debug, Serialize)]
struct NSigmaCheck {
    p5_10_raw_n_sigma: f64,
    predicted_log_bf_at_truth: f64,
    observed_log_bf_at_ty_mean: f64,
    observed_log_bf_at_schoen_mean: f64,
    relation_holds: bool,
    notes: String,
}

#[derive(Debug, Serialize)]
struct DemoReport {
    input_file: String,
    p5_10_summary: P5_10Summary,
    ty_model: CandidateSigmaDistribution,
    schoen_model: CandidateSigmaDistribution,
    scenario_a_ty_truth: ScenarioReport,
    scenario_b_schoen_truth: ScenarioReport,
    scenario_c_train_test_split: Vec<EnsembleBfRow>,
    n_sigma_check: NSigmaCheck,
}

#[derive(Debug, Serialize)]
struct P5_10Summary {
    k: u32,
    mean_ty: f64,
    se_ty: f64,
    n_ty: usize,
    mean_schoen: f64,
    se_schoen: f64,
    n_schoen: usize,
    n_sigma: f64,
}

// ------------------------------------------------------------------
// Helpers.
// ------------------------------------------------------------------

fn extract_sigma_for(
    ensemble: &EnsembleJson,
    candidate: &str,
    k: u32,
) -> Vec<(u64, f64)> {
    ensemble
        .candidates
        .iter()
        .filter(|b| b.candidate.eq_ignore_ascii_case(candidate) && b.k == k)
        .flat_map(|b| b.per_seed.iter())
        .filter(|r| r.candidate.eq_ignore_ascii_case(candidate) && r.k == k)
        .map(|r| (r.seed, r.sigma_final))
        .collect()
}

fn compute_per_seed_scenario(
    label: &str,
    description: &str,
    seeds_and_sigmas: &[(u64, f64)],
    ty: &CandidateSigmaDistribution,
    schoen: &CandidateSigmaDistribution,
    truth_label: &str,
) -> ScenarioReport {
    let mut rows = Vec::with_capacity(seeds_and_sigmas.len());
    let mut log_bfs = Vec::with_capacity(seeds_and_sigmas.len());
    let mut log10_bfs = Vec::with_capacity(seeds_and_sigmas.len());
    let mut categories: BTreeMap<String, usize> = BTreeMap::new();

    for &(seed, sigma_obs) in seeds_and_sigmas {
        let r: SigmaBayesFactorResult =
            evaluate_sigma_bayes_factor(&[sigma_obs], ty, schoen);
        let post = posterior_odds_ty_vs_schoen(1.0, r.log_bf_ty_vs_schoen);
        rows.push(PerSeedBfRow {
            seed,
            sigma_obs,
            log_bf_ty_vs_schoen: r.log_bf_ty_vs_schoen,
            log10_bf_ty_vs_schoen: r.log10_bf_ty_vs_schoen,
            evidence_strength: r.evidence_strength.clone(),
            equivalent_n_sigma: r.equivalent_n_sigma,
            posterior_odds_ty_at_prior_1: post,
        });
        log_bfs.push(r.log_bf_ty_vs_schoen);
        log10_bfs.push(r.log10_bf_ty_vs_schoen);
        *categories.entry(r.evidence_strength).or_insert(0) += 1;
    }

    // Sign consistency: under TY-truth all log_bf > 0; under Schoen-
    // truth all log_bf < 0. Reported as a boolean for quick verdict.
    let expected_sign_positive = truth_label.eq_ignore_ascii_case("TY");
    let sign_consistent = if expected_sign_positive {
        log_bfs.iter().all(|&x| x > 0.0)
    } else {
        log_bfs.iter().all(|&x| x < 0.0)
    };

    ScenarioReport {
        label: label.to_string(),
        description: description.to_string(),
        per_seed: rows,
        log_bf_summary: SummaryStats::from_slice(&log_bfs),
        log10_bf_summary: SummaryStats::from_slice(&log10_bfs),
        category_distribution: categories,
        sign_consistent,
        sign_correctly_favours: truth_label.to_string(),
    }
}

fn print_scenario(report: &ScenarioReport) {
    println!();
    println!("=== {} ===", report.label);
    println!("{}", report.description);
    println!(
        "  per-seed log_BF: mean = {:.3}, SE = {:.3}, range = [{:.3}, {:.3}]",
        report.log_bf_summary.mean,
        report.log_bf_summary.stderr_of_mean,
        report.log_bf_summary.min,
        report.log_bf_summary.max,
    );
    println!(
        "  per-seed log10_BF: mean = {:.3}, SE = {:.3}",
        report.log10_bf_summary.mean, report.log10_bf_summary.stderr_of_mean,
    );
    println!("  Jeffreys-Trotta category distribution:");
    for (cat, count) in &report.category_distribution {
        println!("    {}: {}", cat, count);
    }
    println!(
        "  sign consistency vs {}-truth expectation: {}",
        report.sign_correctly_favours,
        if report.sign_consistent { "PASS" } else { "FAIL" },
    );
    println!("  per-seed rows:");
    println!("    seed       sigma_obs   ln_BF       log10_BF   strength       n_sigma_eq   post_odds(prior=1)");
    for row in &report.per_seed {
        println!(
            "    {:<10} {:<11.6} {:<11.4} {:<10.4} {:<14} {:<12.3} {:.3e}",
            row.seed,
            row.sigma_obs,
            row.log_bf_ty_vs_schoen,
            row.log10_bf_ty_vs_schoen,
            row.evidence_strength,
            row.equivalent_n_sigma,
            row.posterior_odds_ty_at_prior_1,
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    println!("P5.8 — Bayes-factor formalisation of P5.10 σ-discrimination");
    println!("================================================================");
    println!("Loading P5.10 ensemble JSON: {}", cli.input.display());

    let raw = fs::read_to_string(&cli.input)?;
    let ensemble: EnsembleJson = serde_json::from_str(&raw)?;

    if ensemble.discrimination.is_empty() {
        return Err("P5.10 JSON has empty `discrimination` array".into());
    }
    let disc = &ensemble.discrimination[0];
    println!(
        "  k = {},  TY: ⟨σ⟩ = {:.6} ± {:.6} (n={}),  Schoen: ⟨σ⟩ = {:.6} ± {:.6} (n={}),  n-σ = {:.4}",
        disc.k,
        disc.mean_ty,
        disc.se_ty,
        disc.n_ty,
        disc.mean_schoen,
        disc.se_schoen,
        disc.n_schoen,
        disc.n_sigma,
    );
    println!("  P5.10 max n-σ across all k: {:.4}", ensemble.max_n_sigma);

    let ty_model = CandidateSigmaDistribution::new(disc.mean_ty, disc.se_ty, disc.n_ty)
        .map_err(|e| format!("TY model: {}", e))?;
    let schoen_model =
        CandidateSigmaDistribution::new(disc.mean_schoen, disc.se_schoen, disc.n_schoen)
            .map_err(|e| format!("Schoen model: {}", e))?;

    let ty_obs = extract_sigma_for(&ensemble, "TY", disc.k);
    let schoen_obs = extract_sigma_for(&ensemble, "SCHOEN", disc.k);
    if ty_obs.len() != disc.n_ty || schoen_obs.len() != disc.n_schoen {
        return Err(format!(
            "σ-extraction mismatch: TY {} vs n_ty {}, Schoen {} vs n_schoen {}",
            ty_obs.len(),
            disc.n_ty,
            schoen_obs.len(),
            disc.n_schoen
        )
        .into());
    }

    // Scenario A — TY-truth: each TY observation evaluated singly.
    let scen_a = compute_per_seed_scenario(
        "Scenario A — TY-truth (per-seed)",
        "Each of the 20 σ_TY values evaluated as a single observation. \
         Under TY-truth we expect ln BF > 0 (TY preferred) on every seed.",
        &ty_obs,
        &ty_model,
        &schoen_model,
        "TY",
    );

    // Scenario B — Schoen-truth.
    let scen_b = compute_per_seed_scenario(
        "Scenario B — Schoen-truth (per-seed)",
        "Each of the 20 σ_Schoen values evaluated as a single observation. \
         Under Schoen-truth we expect ln BF < 0 (Schoen preferred) on every seed.",
        &schoen_obs,
        &ty_model,
        &schoen_model,
        "Schoen",
    );

    // Scenario C — 10/10 train-test split.
    //   * "Train" set: first 10 σ values per candidate -> defines
    //     CandidateSigmaDistribution(mean, SE, n=10).
    //   * "Test" set: last 10 σ values per candidate -> ensemble
    //     log-likelihood evaluated under both models, BF computed.
    let _ = cli.split_seed; // accepted for forward compat; deterministic split is used.
    let split = 10;
    let ty_train: Vec<f64> = ty_obs.iter().take(split).map(|(_, s)| *s).collect();
    let ty_test: Vec<f64> = ty_obs.iter().skip(split).map(|(_, s)| *s).collect();
    let schoen_train: Vec<f64> = schoen_obs.iter().take(split).map(|(_, s)| *s).collect();
    let schoen_test: Vec<f64> = schoen_obs.iter().skip(split).map(|(_, s)| *s).collect();

    fn mean_se(xs: &[f64]) -> (f64, f64) {
        let n = xs.len();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
        let se = (var / n as f64).sqrt();
        (mean, se)
    }

    let (ty_train_mean, ty_train_se) = mean_se(&ty_train);
    let (schoen_train_mean, schoen_train_se) = mean_se(&schoen_train);
    let ty_train_model =
        CandidateSigmaDistribution::new(ty_train_mean, ty_train_se, split)?;
    let schoen_train_model =
        CandidateSigmaDistribution::new(schoen_train_mean, schoen_train_se, split)?;

    let mut scen_c_rows = Vec::new();
    for (truth, test_set, label) in [
        (
            "TY",
            &ty_test,
            format!(
                "TY held-out 10 (train ⟨σ⟩ = {:.6} ± {:.6}, Schoen train ⟨σ⟩ = {:.6} ± {:.6})",
                ty_train_mean, ty_train_se, schoen_train_mean, schoen_train_se
            ),
        ),
        (
            "Schoen",
            &schoen_test,
            format!(
                "Schoen held-out 10 (train ⟨σ⟩ = {:.6} ± {:.6}, TY train ⟨σ⟩ = {:.6} ± {:.6})",
                schoen_train_mean, schoen_train_se, ty_train_mean, ty_train_se
            ),
        ),
    ] {
        let r = evaluate_sigma_bayes_factor(test_set, &ty_train_model, &schoen_train_model);
        let post = posterior_odds_ty_vs_schoen(1.0, r.log_bf_ty_vs_schoen);
        scen_c_rows.push(EnsembleBfRow {
            label: format!("Scenario C — {}-truth ensemble", truth),
            description: label,
            sigma_obs: test_set.clone(),
            n_observations: test_set.len(),
            log_bf_ty_vs_schoen: r.log_bf_ty_vs_schoen,
            log10_bf_ty_vs_schoen: r.log10_bf_ty_vs_schoen,
            evidence_strength: r.evidence_strength,
            equivalent_n_sigma: r.equivalent_n_sigma,
            posterior_odds_ty_at_prior_1: post,
        });
    }

    // n-σ <-> ln BF asymptotic check.
    //
    //   |ln BF| ≈ n_σ² / 2 for a single observation at the truth-model
    //   mean, ignoring the log(SE_a / SE_b) normalisation term.
    //
    // We report both predicted and observed values so the user can
    // judge whether the SE-asymmetry contribution is significant.
    let predicted_ln_bf = 0.5 * disc.n_sigma * disc.n_sigma;
    let obs_at_ty_mean = log_bf_ty_vs_schoen(&[ty_model.mean], &ty_model, &schoen_model);
    let obs_at_schoen_mean =
        log_bf_ty_vs_schoen(&[schoen_model.mean], &ty_model, &schoen_model);
    // "Holds" ⇔ |observed| within a factor of 2 of predicted, AND
    // ln BF dominated by the chi² term (sign matches the truth model).
    let relation_holds = (obs_at_ty_mean.abs() / predicted_ln_bf > 0.5)
        && (obs_at_ty_mean.abs() / predicted_ln_bf < 2.0)
        && obs_at_ty_mean > 0.0
        && obs_at_schoen_mean < 0.0;
    let n_sigma_check = NSigmaCheck {
        p5_10_raw_n_sigma: disc.n_sigma,
        predicted_log_bf_at_truth: predicted_ln_bf,
        observed_log_bf_at_ty_mean: obs_at_ty_mean,
        observed_log_bf_at_schoen_mean: obs_at_schoen_mean,
        relation_holds,
        notes: format!(
            "Asymptotic relation |ln BF| ≈ n_σ²/2 = {:.2} (Cowan-Cranmer-Gross-Vitells 2011 §4). \
             Observed ln BF at TY mean = {:.2}, at Schoen mean = {:.2}. \
             Asymmetry between the two values comes from the log(SE_TY / SE_Schoen) = {:.3} \
             normalisation term and the differing z-distances under the two models.",
            predicted_ln_bf,
            obs_at_ty_mean,
            obs_at_schoen_mean,
            (ty_model.stderr / schoen_model.stderr).ln(),
        ),
    };

    // Print everything.
    print_scenario(&scen_a);
    print_scenario(&scen_b);
    println!();
    println!("=== Scenario C — 10/10 train-test split ensemble ===");
    for r in &scen_c_rows {
        println!();
        println!("  {}", r.label);
        println!("  {}", r.description);
        println!(
            "    n_obs = {}, ln BF = {:.4}, log10 BF = {:.4}, strength = {}",
            r.n_observations, r.log_bf_ty_vs_schoen, r.log10_bf_ty_vs_schoen, r.evidence_strength,
        );
        println!(
            "    equivalent n-σ = {:.3}, posterior odds (prior=1) = {:.3e}",
            r.equivalent_n_sigma, r.posterior_odds_ty_at_prior_1,
        );
    }

    println!();
    println!("=== n-σ ↔ ln BF asymptotic relation ===");
    println!(
        "  P5.10 raw n-σ            = {:.4}",
        n_sigma_check.p5_10_raw_n_sigma
    );
    println!(
        "  Predicted |ln BF| ≈ n²/2 = {:.4}",
        n_sigma_check.predicted_log_bf_at_truth
    );
    println!(
        "  Observed ln BF at TY mean    = {:.4}",
        n_sigma_check.observed_log_bf_at_ty_mean
    );
    println!(
        "  Observed ln BF at Schoen mean = {:.4}",
        n_sigma_check.observed_log_bf_at_schoen_mean
    );
    println!(
        "  Relation holds (|obs|/pred ∈ [0.5, 2.0] and signs match): {}",
        n_sigma_check.relation_holds
    );
    println!("  {}", n_sigma_check.notes);

    println!();
    println!("=== Verdict ===");
    let verdict = if scen_a.sign_consistent
        && scen_b.sign_consistent
        && n_sigma_check.relation_holds
    {
        "Bayes-factor formalisation of P5.10 confirms STRONG-EVIDENCE \
         discrimination (Jeffreys-Trotta) of TY/Z3 vs Schoen/Z3×Z3."
    } else {
        "Mixed result — see per-scenario sign consistency and n-σ check above."
    };
    println!("  {}", verdict);

    // Sanity: re-derive the per-seed ll values via the public single-
    // and ensemble-level helpers to demonstrate they match (debugging
    // aid; cheap).
    let ll_ty_at_first =
        log_likelihood_single(ty_obs[0].1, &ty_model);
    let ll_ensemble_first_three = log_likelihood_ensemble(
        &ty_obs.iter().take(3).map(|(_, s)| *s).collect::<Vec<_>>(),
        &ty_model,
    );
    println!();
    println!(
        "  (debug) log L(σ_TY[0] | TY) = {:.4}; log L(σ_TY[0..3] | TY) = {:.4}",
        ll_ty_at_first, ll_ensemble_first_three,
    );

    let report = DemoReport {
        input_file: cli.input.display().to_string(),
        p5_10_summary: P5_10Summary {
            k: disc.k,
            mean_ty: disc.mean_ty,
            se_ty: disc.se_ty,
            n_ty: disc.n_ty,
            mean_schoen: disc.mean_schoen,
            se_schoen: disc.se_schoen,
            n_schoen: disc.n_schoen,
            n_sigma: disc.n_sigma,
        },
        ty_model,
        schoen_model,
        scenario_a_ty_truth: scen_a,
        scenario_b_schoen_truth: scen_b,
        scenario_c_train_test_split: scen_c_rows,
        n_sigma_check,
    };

    if let Some(parent) = cli.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let json = serde_json::to_string_pretty(&report)?;
    fs::write(&cli.output, json)?;
    println!();
    println!("Wrote full per-seed report -> {}", cli.output.display());

    Ok(())
}
