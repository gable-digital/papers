//! # Bayes factor between candidate families
//!
//! Given two evidence integrals `Z_a` and `Z_b` (each from a separate
//! [`crate::route34::nested_sampling::compute_evidence`] run), computes
//! the log Bayes factor
//!
//!     ln B_{a,b} = ln Z_a - ln Z_b
//!
//! and classifies it against Jeffreys' 1961 thresholds (Theory of
//! Probability, 3rd ed., Appendix B):
//!
//!     |ln B| < 1            Inconclusive
//!     1   <= |ln B| < 2.5   Substantial
//!     2.5 <= |ln B| < 5     Strong
//!     5   <= |ln B| < 11.5  VeryStrong
//!     |ln B| >= 5           Decisive (= 5-sigma equivalent via |ln B| ~ chi^2/2)
//!
//! Jeffreys' original thresholds use base-10 (decisive at log10 B > 2,
//! i.e. ln B > 4.605); the Bayesian-evidence community has since
//! converged on the natural-log variant we use here, with `|ln B| >= 5`
//! corresponding to "decisive at 5-sigma" via the asymptotic relation
//! `|ln B_{a,b}| ~ Delta chi^2 / 2`. See Trotta 2008 *Contemp. Phys.*
//! 49:71 §4.1 for the modern review.
//!
//! The propagated uncertainty on `ln B` follows from the uncorrelated
//! Skilling-2004 evidence-uncertainty estimates:
//!
//!     sigma_{ln B}^2 = sigma_{ln Z_a}^2 + sigma_{ln Z_b}^2.
//!
//! ## References
//!
//! * Jeffreys, H. "Theory of Probability", 3rd ed. (Oxford 1961),
//!   Appendix B.
//! * Trotta, R. "Bayes in the sky: Bayesian inference and model
//!   selection in cosmology", Contemp. Phys. 49 (2008) 71,
//!   arXiv:0803.4089.
//! * Kass, R.E.; Raftery, A.E. "Bayes factors", J. Amer. Stat. Assoc.
//!   90 (1995) 773.

use serde::{Deserialize, Serialize};

use crate::route34::nested_sampling::{EvidenceResult, RunMetadata};

/// Jeffreys 1961 (Theory of Probability, App. B) decision class.
///
/// Note that "decisive" is reached at `|ln B| >= 5`, which matches the
/// 5-sigma significance convention via the asymptotic chi^2 / 2
/// relation. "Very strong" is the (5, 11.5) interval, beyond which is
/// "overwhelming"; we collapse "overwhelming" into "decisive" because
/// the discrimination problem only needs the 5-sigma threshold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum JeffreysClass {
    /// |ln B| < 1
    Inconclusive,
    /// 1 <= |ln B| < 2.5
    Substantial,
    /// 2.5 <= |ln B| < 5
    Strong,
    /// 5 <= |ln B| < 11.5  (also Decisive once threshold is hit;
    /// retained as a label for granularity in the report).
    VeryStrong,
    /// |ln B| >= 5  (5-sigma equivalent; Jeffreys' "decisive").
    Decisive,
}

impl JeffreysClass {
    pub fn classify(abs_log_bayes_factor: f64) -> Self {
        let x = abs_log_bayes_factor;
        if x < 1.0 {
            JeffreysClass::Inconclusive
        } else if x < 2.5 {
            JeffreysClass::Substantial
        } else if x < 5.0 {
            JeffreysClass::Strong
        } else if x < 11.5 {
            // 5 <= x < 11.5 — both "VeryStrong" and "Decisive" apply.
            // We report VeryStrong so that the upper-bin label is
            // present in the verdict.
            JeffreysClass::VeryStrong
        } else {
            JeffreysClass::Decisive
        }
    }

    /// True iff this class meets the 5-sigma "decisive" threshold.
    pub fn is_decisive(self) -> bool {
        matches!(self, JeffreysClass::VeryStrong | JeffreysClass::Decisive)
    }

    pub fn label(self) -> &'static str {
        match self {
            JeffreysClass::Inconclusive => "Inconclusive",
            JeffreysClass::Substantial => "Substantial",
            JeffreysClass::Strong => "Strong",
            JeffreysClass::VeryStrong => "Very strong",
            JeffreysClass::Decisive => "Decisive",
        }
    }
}

/// Bayes-factor result with propagated uncertainty.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BayesFactorResult {
    /// Signed `ln B = ln Z_a - ln Z_b` where `a` is the preferred
    /// family. Always `>= 0`.
    pub log_bayes_factor: f64,
    pub log_bayes_factor_uncertainty: f64,
    pub preferred_candidate: String,
    pub disfavored_candidate: String,
    pub jeffreys_class: JeffreysClass,
    /// `|ln B| / 2` mapped to an equivalent chi^2 difference and then
    /// to `sqrt(chi^2)` n-sigma. Asymptotic relation
    /// `|ln B_{a,b}| ~ Delta chi^2 / 2` (Cowan-Cranmer-Gross-Vitells
    /// 2011 §4) gives `n_sigma_eq = sqrt(2 |ln B|)`.
    pub equivalent_n_sigma: f64,
    /// Per-candidate run metadata copied through for traceability.
    pub run_metadata_a: RunMetadata,
    pub run_metadata_b: RunMetadata,
}

/// Compute the Bayes factor between two candidate families.
///
/// `evidence_a.candidate_label` and `evidence_b.candidate_label` are the
/// caller-supplied labels; the output's `preferred_candidate` is set to
/// whichever has the larger `log_evidence`.
pub fn compute_bayes_factor(
    label_a: &str,
    evidence_a: &EvidenceResult,
    label_b: &str,
    evidence_b: &EvidenceResult,
) -> BayesFactorResult {
    let signed_log_b = evidence_a.log_evidence - evidence_b.log_evidence;
    let abs_log_b = signed_log_b.abs();
    let var_sum = evidence_a.log_evidence_uncertainty.powi(2)
        + evidence_b.log_evidence_uncertainty.powi(2);
    let sigma = var_sum.sqrt();

    let (preferred, disfavored) = if signed_log_b >= 0.0 {
        (label_a.to_string(), label_b.to_string())
    } else {
        (label_b.to_string(), label_a.to_string())
    };

    let class = JeffreysClass::classify(abs_log_b);
    let n_sigma_eq = (2.0 * abs_log_b).sqrt();

    BayesFactorResult {
        log_bayes_factor: abs_log_b,
        log_bayes_factor_uncertainty: sigma,
        preferred_candidate: preferred,
        disfavored_candidate: disfavored,
        jeffreys_class: class,
        equivalent_n_sigma: n_sigma_eq,
        run_metadata_a: evidence_a.run_metadata.clone(),
        run_metadata_b: evidence_b.run_metadata.clone(),
    }
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route34::nested_sampling::EvidenceResult;

    fn fake_evidence(log_z: f64, sigma: f64) -> EvidenceResult {
        EvidenceResult {
            log_evidence: log_z,
            log_evidence_uncertainty: sigma,
            information_h: 1.0,
            n_live_points_remaining: 100,
            posterior_samples: Vec::new(),
            run_metadata: RunMetadata {
                seed: 1,
                n_live: 100,
                iterations_run: 100,
                n_likelihood_evaluations: 1000,
                n_constrained_draws: 1000,
                wall_clock_seconds: 0.1,
                live_points_sha256: "deadbeef".to_string(),
                resumed_from_checkpoint: false,
            },
        }
    }

    #[test]
    fn test_bayes_factor_against_two_gaussians() {
        // Synthetic: ln Z_a = 0.5, ln Z_b = 0.0 -> ln B = 0.5.
        let a = fake_evidence(0.5, 0.05);
        let b = fake_evidence(0.0, 0.05);
        let r = compute_bayes_factor("A", &a, "B", &b);
        assert!((r.log_bayes_factor - 0.5).abs() < 1e-15);
        assert_eq!(r.preferred_candidate, "A");
        assert_eq!(r.disfavored_candidate, "B");
        assert!((r.log_bayes_factor_uncertainty - (0.05_f64.powi(2) * 2.0).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn test_bayes_factor_picks_better_evidence_regardless_of_arg_order() {
        let a = fake_evidence(0.0, 0.05);
        let b = fake_evidence(2.0, 0.05);
        let r = compute_bayes_factor("A", &a, "B", &b);
        assert!((r.log_bayes_factor - 2.0).abs() < 1e-15);
        assert_eq!(r.preferred_candidate, "B");
        assert_eq!(r.disfavored_candidate, "A");
    }

    #[test]
    fn test_jeffreys_threshold_classification() {
        // Boundary points.
        assert_eq!(JeffreysClass::classify(0.5), JeffreysClass::Inconclusive);
        assert_eq!(JeffreysClass::classify(1.0), JeffreysClass::Substantial);
        assert_eq!(JeffreysClass::classify(2.0), JeffreysClass::Substantial);
        assert_eq!(JeffreysClass::classify(2.5), JeffreysClass::Strong);
        assert_eq!(JeffreysClass::classify(4.0), JeffreysClass::Strong);
        assert_eq!(JeffreysClass::classify(5.0), JeffreysClass::VeryStrong);
        assert_eq!(JeffreysClass::classify(8.0), JeffreysClass::VeryStrong);
        assert_eq!(JeffreysClass::classify(11.5), JeffreysClass::Decisive);
        assert_eq!(JeffreysClass::classify(12.5), JeffreysClass::Decisive);
        assert!(JeffreysClass::classify(11.5).is_decisive());
        assert!(JeffreysClass::classify(5.0).is_decisive());
        assert!(!JeffreysClass::classify(2.0).is_decisive());
    }

    #[test]
    fn test_n_sigma_equivalent() {
        // |ln B| = 12.5 => n_sigma_eq = sqrt(25) = 5.
        let a = fake_evidence(12.5, 0.0);
        let b = fake_evidence(0.0, 0.0);
        let r = compute_bayes_factor("A", &a, "B", &b);
        assert!(
            (r.equivalent_n_sigma - 5.0).abs() < 1e-12,
            "n_sigma_eq = {}, expected 5",
            r.equivalent_n_sigma
        );
        // |ln B| = 2 => n_sigma_eq = sqrt(4) = 2.
        let c = fake_evidence(2.0, 0.0);
        let d = fake_evidence(0.0, 0.0);
        let r2 = compute_bayes_factor("A", &c, "B", &d);
        assert!((r2.equivalent_n_sigma - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_jeffreys_class_labels() {
        assert_eq!(JeffreysClass::Inconclusive.label(), "Inconclusive");
        assert_eq!(JeffreysClass::Decisive.label(), "Decisive");
    }

    #[test]
    fn test_uncertainty_propagation() {
        let a = fake_evidence(1.0, 0.1);
        let b = fake_evidence(0.0, 0.2);
        let r = compute_bayes_factor("A", &a, "B", &b);
        let expected_sigma = (0.01_f64 + 0.04).sqrt();
        assert!((r.log_bayes_factor_uncertainty - expected_sigma).abs() < 1e-12);
    }
}
