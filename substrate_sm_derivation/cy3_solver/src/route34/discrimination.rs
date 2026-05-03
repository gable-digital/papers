//! # End-to-end discrimination orchestrator
//!
//! Top-level entry point that runs nested sampling per candidate
//! family, computes pairwise Bayes factors, classifies them against the
//! Jeffreys 1961 thresholds, and emits a [`DiscriminationVerdict`].
//!
//! This module is the family-discrimination analog of
//! [`crate::route34::eta_evaluator`]: it consumes a description of each
//! candidate family (label + likelihood closure + prior) and produces a
//! self-contained verdict suitable for serialisation into the
//! `discrimination_report.json` produced by the
//! `bin/bayes_discriminate.rs` CLI.
//!
//! ## Why "closure" rather than "candidate id"
//!
//! The discrimination layer is intentionally decoupled from the actual
//! physics evaluators. Two reasons:
//!
//! 1. The likelihood closures are large (HYM solve + harmonic zero
//!    modes + Yukawa quadrature + RG run + chi^2-vs-PDG). Stuffing them
//!    behind a stringly-typed dispatch table would couple the
//!    discrimination crate to every Wave-1/Wave-4 piece.
//! 2. The same orchestrator must drive the analytic gaussian-toy test
//!    used for cross-checking the Bayes-factor result against the
//!    closed-form expression.
//!
//! See `bin/bayes_discriminate.rs` for the production wiring that
//! plugs in `predict_fermion_masses` + `evaluate_eta_*` +
//! `route4_discrimination` + the `pdg::chi_squared_test` chi^2.
//!
//! ## References
//!
//! As in [`crate::route34::nested_sampling`] and
//! [`crate::route34::bayes_factor`].

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::route34::bayes_factor::{compute_bayes_factor, BayesFactorResult};
use crate::route34::likelihood::LogLikelihoodResult;
use crate::route34::nested_sampling::{
    compute_evidence, EvidenceResult, NestedSamplingConfig, NestedSamplingError,
};
use crate::route34::prior::{ModuliPoint, Prior};

// ----------------------------------------------------------------------
// Configuration / verdict types.
// ----------------------------------------------------------------------

/// Top-level configuration for [`run_full_discrimination`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiscriminationConfig {
    /// Descriptive labels for each candidate family. Used in the
    /// verdict string and the per-pair Bayes-factor outputs.
    pub candidate_labels: Vec<String>,
    /// Number of live points per nested-sampling run.
    pub n_live: usize,
    /// PRNG seed (each candidate uses a derived seed via
    /// `seed.wrapping_add(candidate_index * 0xDEADBEEF)`).
    pub seed: u64,
    /// Output directory (caller-managed; this module does not write).
    pub output_dir: PathBuf,
    /// Stopping tolerance on `ln Z` change per nested-sampling step.
    pub stop_log_evidence_change: f64,
    /// Hard cap on iterations.
    pub max_iterations: usize,
    /// Persist a checkpoint every this-many iterations (per candidate).
    pub checkpoint_interval: usize,
    /// Number of posterior samples to retain per candidate.
    pub n_posterior_samples: usize,
}

impl Default for DiscriminationConfig {
    fn default() -> Self {
        Self {
            candidate_labels: vec!["TY/Z3".to_string(), "Schoen/Z3xZ3".to_string()],
            n_live: 500,
            seed: 42,
            output_dir: PathBuf::from("."),
            stop_log_evidence_change: 1e-3,
            max_iterations: 100_000,
            checkpoint_interval: 200,
            n_posterior_samples: 0,
        }
    }
}

/// Top-level discrimination verdict.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiscriminationVerdict {
    /// Per-candidate `(label, evidence)` pairs in input order.
    pub per_candidate_evidence: Vec<(String, EvidenceResult)>,
    /// Pairwise Bayes-factor results. For two candidates there is one
    /// entry; for n candidates there are `n*(n-1)/2` entries in the
    /// canonical (i, j) order with i < j.
    pub bayes_factors: Vec<BayesFactorResult>,
    /// Winning candidate, if any pairwise Bayes factor is in the
    /// "decisive" class (>= 5-sigma equivalent). `None` if no pair
    /// reaches the threshold.
    pub winner: Option<String>,
    /// Human-readable verdict string.
    pub verdict: String,
}

// ----------------------------------------------------------------------
// CandidateModel trait — the production-side wiring point.
// ----------------------------------------------------------------------

/// One candidate family — a triple `(label, prior, likelihood)`.
///
/// The likelihood is a closure boxed behind `Send + Sync` so the
/// orchestrator can drive multi-candidate runs in parallel via rayon
/// in a future extension. Today we run them sequentially to keep
/// memory pressure low (each likelihood draw allocates large CY3
/// metric workspaces).
pub struct CandidateModel<'a> {
    pub label: String,
    pub prior: Box<dyn Prior + 'a>,
    pub likelihood:
        Box<dyn Fn(&ModuliPoint) -> Result<LogLikelihoodResult, String> + Send + Sync + 'a>,
}

impl<'a> CandidateModel<'a> {
    pub fn new<P, F>(label: impl Into<String>, prior: P, likelihood: F) -> Self
    where
        P: Prior + 'a,
        F: Fn(&ModuliPoint) -> Result<LogLikelihoodResult, String> + Send + Sync + 'a,
    {
        Self {
            label: label.into(),
            prior: Box::new(prior),
            likelihood: Box::new(likelihood),
        }
    }
}

// ----------------------------------------------------------------------
// Top-level driver.
// ----------------------------------------------------------------------

/// Discrimination errors.
#[derive(Debug)]
pub enum DiscriminationError {
    NoCandidates,
    NestedSampling { candidate: String, source: NestedSamplingError },
    LabelMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for DiscriminationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiscriminationError::NoCandidates => {
                write!(f, "no candidates supplied to discrimination driver")
            }
            DiscriminationError::NestedSampling { candidate, source } => {
                write!(f, "nested sampling failed for candidate `{}`: {}", candidate, source)
            }
            DiscriminationError::LabelMismatch { expected, got } => write!(
                f,
                "candidate-label count {} does not match candidate-model count {}",
                got, expected
            ),
        }
    }
}

impl std::error::Error for DiscriminationError {}

/// Run the full discrimination pipeline.
pub fn run_full_discrimination(
    candidates: Vec<CandidateModel<'_>>,
    config: &DiscriminationConfig,
) -> Result<DiscriminationVerdict, DiscriminationError> {
    if candidates.is_empty() {
        return Err(DiscriminationError::NoCandidates);
    }

    let mut per_candidate_evidence: Vec<(String, EvidenceResult)> =
        Vec::with_capacity(candidates.len());

    for (idx, cand) in candidates.into_iter().enumerate() {
        let cand_seed = config.seed.wrapping_add((idx as u64).wrapping_mul(0xDEAD_BEEF));
        let ckpt_path = if config.checkpoint_interval > 0 {
            Some(config.output_dir.join(format!("checkpoint_{}.json", sanitize(&cand.label))))
        } else {
            None
        };
        let ns_cfg = NestedSamplingConfig {
            n_live: config.n_live,
            stop_log_evidence_change: config.stop_log_evidence_change,
            max_iterations: config.max_iterations,
            seed: cand_seed,
            checkpoint_path: ckpt_path,
            checkpoint_interval: config.checkpoint_interval,
            ellipsoid_enlargement: 1.5,
            max_constrained_draw_attempts: 10_000,
            n_posterior_samples: config.n_posterior_samples,
        };
        let result = compute_evidence(
            |theta| (cand.likelihood)(theta),
            cand.prior.as_ref(),
            &ns_cfg,
        )
        .map_err(|e| DiscriminationError::NestedSampling {
            candidate: cand.label.clone(),
            source: e,
        })?;
        per_candidate_evidence.push((cand.label.clone(), result));
    }

    // Pairwise Bayes factors (i < j).
    let mut bfs = Vec::new();
    for i in 0..per_candidate_evidence.len() {
        for j in (i + 1)..per_candidate_evidence.len() {
            let (la, ea) = &per_candidate_evidence[i];
            let (lb, eb) = &per_candidate_evidence[j];
            let bf = compute_bayes_factor(la, ea, lb, eb);
            bfs.push(bf);
        }
    }

    // Winner: the consistently-preferred candidate across every
    // decisive Bayes factor. If only one is decisive, the winner is
    // its preferred. If two contradict, no winner.
    let winner = compute_winner(&bfs);

    let verdict = format_verdict(&per_candidate_evidence, &bfs, &winner);

    Ok(DiscriminationVerdict {
        per_candidate_evidence,
        bayes_factors: bfs,
        winner,
        verdict,
    })
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect()
}

/// Identify the consistently-preferred decisive candidate (if any).
fn compute_winner(bfs: &[BayesFactorResult]) -> Option<String> {
    let decisives: Vec<&BayesFactorResult> = bfs
        .iter()
        .filter(|b| b.jeffreys_class.is_decisive())
        .collect();
    if decisives.is_empty() {
        return None;
    }
    // All decisive verdicts must agree on the same preferred label.
    let first = &decisives[0].preferred_candidate;
    if decisives.iter().all(|b| b.preferred_candidate == *first) {
        Some(first.clone())
    } else {
        None
    }
}

fn format_verdict(
    per_candidate_evidence: &[(String, EvidenceResult)],
    bfs: &[BayesFactorResult],
    winner: &Option<String>,
) -> String {
    let mut s = String::new();
    s.push_str("Bayesian discrimination verdict\n");
    s.push_str("===============================\n\n");

    s.push_str("Per-candidate evidence:\n");
    for (label, ev) in per_candidate_evidence {
        s.push_str(&format!(
            "  {:<24}  ln Z = {:+.4}  +/-  {:.4}  (info H = {:.3} nats; iters = {})\n",
            label,
            ev.log_evidence,
            ev.log_evidence_uncertainty,
            ev.information_h,
            ev.run_metadata.iterations_run,
        ));
    }
    s.push('\n');

    s.push_str("Pairwise Bayes factors:\n");
    for bf in bfs {
        let dec = if bf.jeffreys_class.is_decisive() {
            " (>= 5-sigma equivalent)"
        } else {
            ""
        };
        s.push_str(&format!(
            "  {} vs {}: |ln B| = {:.4} +/- {:.4}; class = {}; eq. n-sigma = {:.2}{}\n",
            bf.preferred_candidate,
            bf.disfavored_candidate,
            bf.log_bayes_factor,
            bf.log_bayes_factor_uncertainty,
            bf.jeffreys_class.label(),
            bf.equivalent_n_sigma,
            dec,
        ));
    }
    s.push('\n');

    match winner {
        Some(w) => s.push_str(&format!(
            "Winner: {} (decisive at >= 5-sigma equivalent across all pairwise comparisons).\n",
            w
        )),
        None => {
            s.push_str("Winner: NONE — no pair reaches the Jeffreys 'decisive' (5-sigma) threshold.\n");
        }
    }

    s
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route34::likelihood::ChiSquaredBreakdown;
    use crate::route34::prior::UniformPrior;
    use std::f64::consts::PI;

    fn unit_gaussian_lik(theta: &ModuliPoint) -> Result<LogLikelihoodResult, String> {
        let mut chi2 = 0.0;
        for &x in &theta.continuous {
            chi2 += x * x;
        }
        let d = theta.continuous.len() as f64;
        let log_norm = -0.5 * d * (2.0 * PI).ln();
        let bd = ChiSquaredBreakdown::from_components(
            chi2,
            theta.continuous.len() as u32,
            0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0,
        );
        Ok(LogLikelihoodResult {
            log_likelihood: -0.5 * chi2 + log_norm,
            chi_squared_breakdown: bd,
            p_value: 0.0,
            n_sigma: 0.0,
        })
    }

    fn shifted_gaussian_lik(theta: &ModuliPoint) -> Result<LogLikelihoodResult, String> {
        // Very far-from-data: peak at +5, much smaller evidence.
        let x = theta.continuous[0];
        let chi2 = (x - 5.0).powi(2);
        let log_norm = -0.5 * (2.0 * PI).ln();
        let bd =
            ChiSquaredBreakdown::from_components(chi2, 1, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0);
        Ok(LogLikelihoodResult {
            log_likelihood: -0.5 * chi2 + log_norm,
            chi_squared_breakdown: bd,
            p_value: 0.0,
            n_sigma: 0.0,
        })
    }

    #[test]
    fn test_run_full_discrimination_two_candidate_toy() {
        let prior_a = UniformPrior::new(vec![-3.0], vec![3.0]).unwrap();
        let prior_b = UniformPrior::new(vec![-3.0], vec![3.0]).unwrap();

        let cand_a = CandidateModel::new("good_fit", prior_a, |t| unit_gaussian_lik(t));
        let cand_b = CandidateModel::new("bad_fit", prior_b, |t| shifted_gaussian_lik(t));

        let cfg = DiscriminationConfig {
            candidate_labels: vec!["good_fit".to_string(), "bad_fit".to_string()],
            n_live: 200,
            seed: 12345,
            stop_log_evidence_change: 1e-2,
            max_iterations: 5_000,
            checkpoint_interval: 0,
            n_posterior_samples: 0,
            output_dir: std::env::temp_dir(),
        };

        let verdict = run_full_discrimination(vec![cand_a, cand_b], &cfg).unwrap();
        assert_eq!(verdict.per_candidate_evidence.len(), 2);
        assert_eq!(verdict.bayes_factors.len(), 1);
        // The good fit must have higher evidence.
        assert_eq!(verdict.bayes_factors[0].preferred_candidate, "good_fit");
        // And the verdict must mention both candidates.
        assert!(verdict.verdict.contains("good_fit"));
        assert!(verdict.verdict.contains("bad_fit"));
    }

    #[test]
    fn test_winner_only_set_when_decisive_and_consistent() {
        // Build a fake verdict where one BF is decisive on A and one
        // is decisive on B -> no winner.
        use crate::route34::bayes_factor::{BayesFactorResult, JeffreysClass};
        use crate::route34::nested_sampling::RunMetadata;
        let meta = RunMetadata {
            seed: 1,
            n_live: 10,
            iterations_run: 1,
            n_likelihood_evaluations: 1,
            n_constrained_draws: 1,
            wall_clock_seconds: 0.0,
            live_points_sha256: "x".to_string(),
            resumed_from_checkpoint: false,
        };
        let bf_a = BayesFactorResult {
            log_bayes_factor: 12.0,
            log_bayes_factor_uncertainty: 0.1,
            preferred_candidate: "A".to_string(),
            disfavored_candidate: "B".to_string(),
            jeffreys_class: JeffreysClass::Decisive,
            equivalent_n_sigma: (24.0_f64).sqrt(),
            run_metadata_a: meta.clone(),
            run_metadata_b: meta.clone(),
        };
        let bf_b = BayesFactorResult {
            log_bayes_factor: 12.0,
            log_bayes_factor_uncertainty: 0.1,
            preferred_candidate: "C".to_string(),
            disfavored_candidate: "A".to_string(),
            jeffreys_class: JeffreysClass::Decisive,
            equivalent_n_sigma: (24.0_f64).sqrt(),
            run_metadata_a: meta.clone(),
            run_metadata_b: meta.clone(),
        };
        let winner = compute_winner(&[bf_a, bf_b]);
        assert_eq!(winner, None);

        // Now both decisive on A.
        let bf_a2 = BayesFactorResult {
            log_bayes_factor: 12.0,
            log_bayes_factor_uncertainty: 0.1,
            preferred_candidate: "A".to_string(),
            disfavored_candidate: "B".to_string(),
            jeffreys_class: JeffreysClass::Decisive,
            equivalent_n_sigma: (24.0_f64).sqrt(),
            run_metadata_a: meta.clone(),
            run_metadata_b: meta.clone(),
        };
        let bf_a3 = BayesFactorResult {
            log_bayes_factor: 12.0,
            log_bayes_factor_uncertainty: 0.1,
            preferred_candidate: "A".to_string(),
            disfavored_candidate: "C".to_string(),
            jeffreys_class: JeffreysClass::Decisive,
            equivalent_n_sigma: (24.0_f64).sqrt(),
            run_metadata_a: meta.clone(),
            run_metadata_b: meta.clone(),
        };
        let winner_a = compute_winner(&[bf_a2, bf_a3]);
        assert_eq!(winner_a, Some("A".to_string()));
    }

    #[test]
    fn test_no_candidates_errors() {
        let cfg = DiscriminationConfig::default();
        let r = run_full_discrimination(Vec::new(), &cfg);
        assert!(matches!(r, Err(DiscriminationError::NoCandidates)));
    }
}
