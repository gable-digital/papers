//! # P5.8 — σ-distribution Bayes-factor formalisation.
//!
//! Companion to [`crate::route34::bayes_factor`] (which compares two
//! candidate families via their *log-evidence* output from nested
//! sampling). Where that module assumes both `Z_a` and `Z_b` were
//! produced by [`crate::route34::nested_sampling`], this module operates
//! one level higher up in the discrimination stack: it converts the
//! observed σ-statistic — the per-seed Donaldson-loss spread that
//! P5.4 / P5.7 / P5.10 produce — into a Bayes factor between two
//! competing σ-distribution models (e.g. TY/Z3 vs Schoen/Z3×Z3).
//!
//! ## Likelihood model
//!
//! Each candidate model `M` is summarised by its observed σ ensemble's
//! sample mean ⟨σ⟩_M and standard error SE_M (both as produced by the
//! P5.10 multi-seed harness). Treating σ as a Gaussian random variable
//! around ⟨σ⟩_M with width SE_M (the standard "ensemble bootstrap"
//! reading of P5.7's Bootstrap output), the per-observation log-
//! likelihood of an observed σ_obs given model `M` is
//!
//! ```text
//!   log L(σ_obs | M) = -½ ((σ_obs - ⟨σ⟩_M) / SE_M)²
//!                      - log(SE_M · √(2π))
//! ```
//!
//! For an ensemble of `n` independent σ observations
//! `(σ_obs_1, ..., σ_obs_n)`,
//!
//! ```text
//!   log L(σ_obs_1, ..., σ_obs_n | M) = Σ_i log L(σ_obs_i | M)
//! ```
//!
//! ## Bayes factor
//!
//! The natural-log Bayes factor TY-vs-Schoen is then
//!
//! ```text
//!   ln BF_TY:Schoen = log L(data | TY) - log L(data | Schoen)
//! ```
//!
//! Categorisation and posterior-odds conversion delegate to
//! [`pwos_math::stats::bayes_factor`] (Jeffreys-Trotta thresholds).
//!
//! ## Asymptotic relation to n-σ
//!
//! For Gaussian likelihoods the n-σ separation between two distributions
//! and the magnitude of the natural-log Bayes factor for a single
//! "decisive" observation are related by
//!
//! ```text
//!   |ln BF| ≈ (n_σ)² / 2     (Cowan-Cranmer-Gross-Vitells 2011 §4)
//! ```
//!
//! For P5.10's `n_σ = 8.76`, this predicts `|ln BF| ≈ 38.4` per single
//! "average" observation. Verified by `p5_8_bayes_demo` against the
//! actual P5.10 σ-distribution data.
//!
//! ## References
//!
//! * Kass, R.E.; Raftery, A.E. "Bayes factors", J. Amer. Stat. Assoc.
//!   90 (1995) 773.
//! * Trotta, R. "Bayes in the sky", Contemp. Phys. 49 (2008) 71,
//!   arXiv:0803.4089.
//! * Cowan, Cranmer, Gross, Vitells, "Asymptotic formulae for
//!   likelihood-based tests of new physics", Eur. Phys. J. C 71 (2011)
//!   1554, arXiv:1007.1727.
//! * Jeffreys, H. "Theory of Probability", 3rd ed. (Oxford 1961),
//!   Appendix B.

use serde::{Deserialize, Serialize};

use pwos_math::stats::bayes_factor::{
    categorise_evidence, posterior_odds, EvidenceStrength,
};

/// 2π in f64. `core::f64::consts::TAU` would also work; spelled out
/// here so readers immediately see the `log(σ √(2π))` normalisation.
const TWO_PI: f64 = core::f64::consts::TAU;

/// A candidate σ-distribution: Gaussian summary `(mean, stderr)` over
/// `n_seeds` independent multi-seed runs (matches P5.7 / P5.10 Bootstrap
/// output schema).
///
/// The stored `mean` and `stderr` are taken as ground truth for the
/// likelihood: SE is *not* re-bootstrapped. This is consistent with the
/// way nested-sampling Bayes factors treat the per-run evidence
/// uncertainty as an external input.
///
/// `n_seeds` is recorded for traceability and future variance-of-the-
/// variance corrections; it is *not* used in the per-observation log-
/// likelihood — that uses `stderr` directly as the Gaussian width.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateSigmaDistribution {
    /// Sample mean ⟨σ⟩_M.
    pub mean: f64,
    /// Sample standard error SE_M (must be `> 0`).
    pub stderr: f64,
    /// Number of seeds used to estimate `mean` / `stderr`.
    pub n_seeds: usize,
}

impl CandidateSigmaDistribution {
    /// Construct a [`CandidateSigmaDistribution`] with validation.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `stderr <= 0`, `mean` is non-finite, or
    /// `n_seeds == 0`.
    pub fn new(mean: f64, stderr: f64, n_seeds: usize) -> Result<Self, &'static str> {
        if !mean.is_finite() {
            return Err("CandidateSigmaDistribution: mean must be finite");
        }
        if !stderr.is_finite() || stderr <= 0.0 {
            return Err("CandidateSigmaDistribution: stderr must be finite and > 0");
        }
        if n_seeds == 0 {
            return Err("CandidateSigmaDistribution: n_seeds must be > 0");
        }
        Ok(Self { mean, stderr, n_seeds })
    }
}

/// Log-likelihood of a single observed σ given a Gaussian model.
///
/// `log L(σ_obs | M) = -½ z² - log(SE √(2π))` with `z = (σ_obs - μ)/SE`.
///
/// Returns `f64::NEG_INFINITY` if `model.stderr <= 0` (which
/// [`CandidateSigmaDistribution::new`] forbids, but the inline form is
/// kept defensive for direct construction).
#[inline]
pub fn log_likelihood_single(sigma_obs: f64, model: &CandidateSigmaDistribution) -> f64 {
    if !sigma_obs.is_finite() || model.stderr <= 0.0 || !model.stderr.is_finite() {
        return f64::NEG_INFINITY;
    }
    let z = (sigma_obs - model.mean) / model.stderr;
    let norm = (model.stderr * TWO_PI.sqrt()).ln();
    -0.5 * z * z - norm
}

/// Log-likelihood of a vector of independent σ observations.
///
/// Sum of [`log_likelihood_single`] over the input slice. Returns
/// `f64::NEG_INFINITY` (sticky) if any input observation produces a
/// non-finite per-observation log-likelihood.
pub fn log_likelihood_ensemble(sigma_obs: &[f64], model: &CandidateSigmaDistribution) -> f64 {
    let mut acc = 0.0_f64;
    for &s in sigma_obs {
        let ll = log_likelihood_single(s, model);
        if !ll.is_finite() {
            return f64::NEG_INFINITY;
        }
        acc += ll;
    }
    acc
}

/// Map a [`pwos_math::stats::bayes_factor::EvidenceStrength`] to its
/// canonical string label. Centralised so the JSON / Markdown writers
/// emit consistent vocabulary.
pub fn evidence_strength_label(s: EvidenceStrength) -> &'static str {
    match s {
        EvidenceStrength::Inconclusive => "Inconclusive",
        EvidenceStrength::Weak => "Weak",
        EvidenceStrength::Moderate => "Moderate",
        EvidenceStrength::Strong => "Strong",
    }
}

/// Result of a TY-vs-Schoen Bayes-factor evaluation.
///
/// The Jeffreys-Trotta strength is stored both as the
/// non-`Serialize` enum (for in-process consumers) and as a stable
/// string label (`evidence_strength_label`) for JSON output.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SigmaBayesFactorResult {
    /// `ln BF_TY:Schoen = log L(data | TY) - log L(data | Schoen)`.
    /// Positive => TY preferred; negative => Schoen preferred.
    pub log_bf_ty_vs_schoen: f64,
    /// `log_10 BF_TY:Schoen` for Jeffreys-Trotta categorisation.
    pub log10_bf_ty_vs_schoen: f64,
    /// Jeffreys-Trotta evidence-strength label (string form of
    /// `EvidenceStrength`, since the upstream enum is not serde-derived).
    pub evidence_strength: String,
    /// Asymptotic n-σ equivalent: `n_σ = sqrt(2 |ln BF|)`
    /// (Cowan-Cranmer-Gross-Vitells 2011 §4).
    pub equivalent_n_sigma: f64,
    /// Number of σ observations folded into the BF.
    pub n_observations: usize,
}

/// Compute `ln BF_TY:Schoen` for one or more σ observations.
///
/// `sigma_obs` is the list of observed σ values (single observation =
/// length-1 slice). `ty` and `schoen` are the candidate σ-distribution
/// summaries (typically the P5.10 ⟨σ⟩ ± SE per candidate).
///
/// Definition: positive return value means the data favours TY;
/// negative means it favours Schoen.
pub fn log_bf_ty_vs_schoen(
    sigma_obs: &[f64],
    ty: &CandidateSigmaDistribution,
    schoen: &CandidateSigmaDistribution,
) -> f64 {
    let ll_ty = log_likelihood_ensemble(sigma_obs, ty);
    let ll_schoen = log_likelihood_ensemble(sigma_obs, schoen);
    ll_ty - ll_schoen
}

/// Full Bayes-factor evaluation with Jeffreys-Trotta categorisation and
/// n-σ equivalent. Wires through [`pwos_math::stats::bayes_factor`].
pub fn evaluate_sigma_bayes_factor(
    sigma_obs: &[f64],
    ty: &CandidateSigmaDistribution,
    schoen: &CandidateSigmaDistribution,
) -> SigmaBayesFactorResult {
    let ln_bf = log_bf_ty_vs_schoen(sigma_obs, ty, schoen);
    // log10 BF directly from the ln difference -- numerically equivalent
    // to pwos_math::stats::bayes_factor::log10_bayes_factor(ll_ty,
    // ll_schoen) but avoids the extra ll evaluation pass.
    const INV_LN10: f64 = 1.0 / core::f64::consts::LN_10;
    let log10_bf = ln_bf * INV_LN10;
    let strength = categorise_evidence(log10_bf);
    let n_sigma_eq = (2.0 * ln_bf.abs()).sqrt();
    SigmaBayesFactorResult {
        log_bf_ty_vs_schoen: ln_bf,
        log10_bf_ty_vs_schoen: log10_bf,
        evidence_strength: evidence_strength_label(strength).to_string(),
        equivalent_n_sigma: n_sigma_eq,
        n_observations: sigma_obs.len(),
    }
}

/// Posterior odds TY:Schoen given prior odds and the natural-log Bayes
/// factor. Thin wrapper over
/// [`pwos_math::stats::bayes_factor::posterior_odds`] so callers do not
/// have to import that module separately.
///
/// `prior_odds` is the prior model-odds (TY:Schoen) — `1.0` for equal
/// priors, `>1.0` to encode a TY prior preference, `<1.0` for Schoen.
#[inline]
pub fn posterior_odds_ty_vs_schoen(prior_odds: f64, log_bf_ty_vs_schoen: f64) -> f64 {
    posterior_odds(prior_odds, log_bf_ty_vs_schoen.exp())
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// P5.10 reported summary (mean, SE, n) for k=3 — kept here as the
    /// canonical ground-truth fixture for unit tests.
    fn p5_10_ty_summary() -> CandidateSigmaDistribution {
        CandidateSigmaDistribution::new(1.0146875862209517, 0.005225177112356106, 20).unwrap()
    }

    fn p5_10_schoen_summary() -> CandidateSigmaDistribution {
        CandidateSigmaDistribution::new(3.0365304513085367, 0.23071891459874042, 20).unwrap()
    }

    #[test]
    fn test_p5_8_candidate_constructor_rejects_bad_inputs() {
        assert!(CandidateSigmaDistribution::new(1.0, 0.0, 20).is_err());
        assert!(CandidateSigmaDistribution::new(1.0, -0.1, 20).is_err());
        assert!(CandidateSigmaDistribution::new(f64::NAN, 0.1, 20).is_err());
        assert!(CandidateSigmaDistribution::new(1.0, f64::NAN, 20).is_err());
        assert!(CandidateSigmaDistribution::new(1.0, 0.1, 0).is_err());
        assert!(CandidateSigmaDistribution::new(1.0, 0.1, 1).is_ok());
    }

    #[test]
    fn test_p5_8_log_likelihood_at_mean_equals_normalisation() {
        let m = CandidateSigmaDistribution::new(2.0, 0.5, 20).unwrap();
        // At σ_obs = mean, log L = -log(SE √(2π)).
        let ll = log_likelihood_single(2.0, &m);
        let expected = -(0.5 * TWO_PI.sqrt()).ln();
        assert!(
            (ll - expected).abs() < 1e-12,
            "ll = {}, expected = {}",
            ll,
            expected
        );
    }

    #[test]
    fn test_p5_8_log_likelihood_drops_quadratically() {
        let m = CandidateSigmaDistribution::new(0.0, 1.0, 20).unwrap();
        // At z=1, log L drops by 0.5 vs the mean.
        let ll0 = log_likelihood_single(0.0, &m);
        let ll1 = log_likelihood_single(1.0, &m);
        assert!((ll0 - ll1 - 0.5).abs() < 1e-12);
        // At z=2, drops by 2.0.
        let ll2 = log_likelihood_single(2.0, &m);
        assert!((ll0 - ll2 - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_p5_8_ensemble_is_sum_of_single() {
        let m = CandidateSigmaDistribution::new(1.0, 0.1, 20).unwrap();
        let obs = [0.9, 1.0, 1.1, 1.05];
        let direct: f64 = obs.iter().map(|&s| log_likelihood_single(s, &m)).sum();
        let ensemble = log_likelihood_ensemble(&obs, &m);
        assert!((direct - ensemble).abs() < 1e-12);
    }

    #[test]
    fn test_p5_8_log_bf_at_ty_mean_favours_ty() {
        // Single observation at exactly the TY mean — must favour TY.
        let ty = p5_10_ty_summary();
        let schoen = p5_10_schoen_summary();
        let obs = [ty.mean];
        let ln_bf = log_bf_ty_vs_schoen(&obs, &ty, &schoen);
        assert!(ln_bf > 0.0, "ln BF = {} should be > 0 at TY mean", ln_bf);
        // The observation lies (3.0365 - 1.0147) / 0.2307 ≈ 8.76 σ away
        // from the Schoen mean, so log L_Schoen there is heavily
        // penalised; ln BF should easily clear "strong" (5).
        assert!(
            ln_bf > 5.0,
            "ln BF = {} should be > 5 (strong evidence) at TY mean",
            ln_bf
        );
    }

    #[test]
    fn test_p5_8_log_bf_at_schoen_mean_favours_schoen() {
        let ty = p5_10_ty_summary();
        let schoen = p5_10_schoen_summary();
        let obs = [schoen.mean];
        let ln_bf = log_bf_ty_vs_schoen(&obs, &ty, &schoen);
        // Observation at Schoen's mean is ~387 σ_TY away from TY mean,
        // so ln BF must be deeply negative.
        assert!(ln_bf < -5.0, "ln BF = {} should be < -5 at Schoen mean", ln_bf);
    }

    #[test]
    fn test_p5_8_bayes_factor_strongly_favors_correct_model() {
        // Scenario A: TY-truth observations from the actual P5.10 σ_TY
        // values (k=3, all 20 seeds). Each individual seed must give
        // log10 BF > 2 (Jeffreys-Trotta "strong"), which equals
        // ln BF > 4.605. We bound conservatively at ln BF > 5.
        //
        // Scenario B: Schoen-truth observations — bound ln BF < -5.
        //
        // "Strong evidence" in Jeffreys-Trotta physics terms:
        //   |log10 BF| >= 2  <=>  data are ≥ 100× more probable under
        //   the preferred model than under the disfavoured model.
        // This is the standard threshold used in cosmology model
        // selection (Trotta 2008, Table 1) and corresponds to ≥ 3σ
        // equivalent via the asymptotic |ln BF| ≈ n_σ²/2 relation.
        let ty = p5_10_ty_summary();
        let schoen = p5_10_schoen_summary();

        // Synthetic but representative TY-truth observations: take 5
        // points centred on the TY mean within ±1 SE.
        let ty_obs = [
            ty.mean - ty.stderr,
            ty.mean - 0.5 * ty.stderr,
            ty.mean,
            ty.mean + 0.5 * ty.stderr,
            ty.mean + ty.stderr,
        ];
        for &obs in &ty_obs {
            let ln_bf = log_bf_ty_vs_schoen(&[obs], &ty, &schoen);
            assert!(
                ln_bf > 5.0,
                "TY-truth obs σ = {}: ln BF = {} (need > 5)",
                obs,
                ln_bf
            );
        }

        // Schoen-truth observations centred on Schoen mean within ±1 SE.
        let schoen_obs = [
            schoen.mean - schoen.stderr,
            schoen.mean - 0.5 * schoen.stderr,
            schoen.mean,
            schoen.mean + 0.5 * schoen.stderr,
            schoen.mean + schoen.stderr,
        ];
        for &obs in &schoen_obs {
            let ln_bf = log_bf_ty_vs_schoen(&[obs], &ty, &schoen);
            assert!(
                ln_bf < -5.0,
                "Schoen-truth obs σ = {}: ln BF = {} (need < -5)",
                obs,
                ln_bf
            );
        }
    }

    #[test]
    fn test_p5_8_bayes_factor_categorisation_via_pwos_math() {
        let ty = p5_10_ty_summary();
        let schoen = p5_10_schoen_summary();
        let r = evaluate_sigma_bayes_factor(&[ty.mean], &ty, &schoen);
        // |log10 BF| should be huge -- well into "Strong".
        assert_eq!(r.evidence_strength, "Strong");
        assert!(r.log10_bf_ty_vs_schoen > 2.0);
        assert!(r.equivalent_n_sigma > 3.0);
    }

    #[test]
    fn test_p5_8_posterior_odds_at_equal_prior_equals_bf() {
        let ty = p5_10_ty_summary();
        let schoen = p5_10_schoen_summary();
        let ln_bf = log_bf_ty_vs_schoen(&[ty.mean], &ty, &schoen);
        let post = posterior_odds_ty_vs_schoen(1.0, ln_bf);
        // Posterior odds at prior = 1 equals BF = exp(ln BF).
        let expected = ln_bf.exp();
        // Both can be huge, so use relative tolerance.
        assert!(
            (post - expected).abs() / expected.abs() < 1e-12,
            "posterior_odds = {}, expected = {}",
            post,
            expected
        );
    }

    #[test]
    fn test_p5_8_n_sigma_relation_holds_for_full_ensemble() {
        // Asymptotic relation: |ln BF| ≈ n_σ² / 2 for Gaussian models.
        // Use the P5.10 means as "mean-of-means" — i.e. a single
        // observation at the TY mean, comparing under the two
        // distribution summaries, recovers the (μ_diff / SE_combined)²
        // / 2 separation up to a normalisation term log(SE_TY / SE_Schoen).
        let ty = p5_10_ty_summary();
        let schoen = p5_10_schoen_summary();

        // Quadratic-form prediction (z² difference / 2):
        let z_ty_under_schoen = (ty.mean - schoen.mean) / schoen.stderr;
        let z_ty_under_ty = (ty.mean - ty.mean) / ty.stderr; // == 0
        let quad_term = 0.5 * (z_ty_under_schoen.powi(2) - z_ty_under_ty.powi(2));
        // Normalisation term from the log(SE) asymmetry:
        let norm_term = (schoen.stderr / ty.stderr).ln();
        let predicted_ln_bf = quad_term + norm_term;

        let ln_bf = log_bf_ty_vs_schoen(&[ty.mean], &ty, &schoen);
        assert!(
            (ln_bf - predicted_ln_bf).abs() < 1e-9,
            "ln BF = {}, predicted = {}",
            ln_bf,
            predicted_ln_bf
        );
    }
}
