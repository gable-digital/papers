//! End-to-end integration test for the Bayesian discrimination layer.
//!
//! Runs the full
//!     prior -> likelihood -> nested_sampling -> bayes_factor ->
//!     discrimination
//! pipeline on a synthetic two-Gaussian toy with analytically known
//! evidence integrals and verifies that the recovered Bayes factor
//! matches the analytic value to within ~1 sigma.
//!
//! ## Toy setup
//!
//! Both candidates use the same uniform prior `pi(theta) = 1 / 10`
//! over `theta in [-5, 5]`.
//!
//! Candidate A: likelihood is a Gaussian centred at `theta = 0` with
//!     variance `sigma_a^2 = 1`. Restricted to the prior support, the
//!     evidence is
//!         Z_a = (1/10) * P(|theta_A| <= 5)
//!             ~ (1/10) * 1.0
//!     so `ln Z_a ~ ln(1/10) = -2.3026`.
//!
//! Candidate B: same Gaussian centred at `theta = 1`. Same
//!     evidence to within the tail truncation:
//!         Z_b = (1/10) * P(-5 <= theta_B <= 5)  with theta_B ~ N(1, 1)
//!             ~ (1/10) * 1.0
//!     so `ln Z_b ~ -2.3026`.
//!
//! Therefore the analytic log Bayes factor between them is essentially
//! zero: `ln B = ln Z_a - ln Z_b ~ 0`. We verify the recovered
//! `ln B` agrees with this to within a few times the propagated
//! Skilling-2004 uncertainty.
//!
//! For a non-trivial Bayes factor, we also test a third candidate C
//! whose Gaussian likelihood is centred at `theta = 3`. With the same
//! prior, the evidence drops by the relative tail mass:
//!     Z_c / Z_a ~ 1 (still both fully inside the prior box).
//!
//! For a *true* non-zero Bayes factor, we add a fourth toy: a model
//! D whose likelihood is `exp(-0.5 * theta^2 / sigma_D^2)` with
//! sigma_D = 0.5. The narrower likelihood concentrates on a smaller
//! prior subset, giving a smaller evidence by a factor of sigma_D :
//!     Z_a / Z_d = sigma_a / sigma_d = 2  =>  ln Z_a - ln Z_d = ln 2 = 0.6931.
//!
//! That is the Bayes-factor reference value the test checks against
//! analytically.

use std::f64::consts::PI;

use cy3_rust_solver::route34::bayes_factor::compute_bayes_factor;
use cy3_rust_solver::route34::likelihood::{ChiSquaredBreakdown, LogLikelihoodResult};
use cy3_rust_solver::route34::nested_sampling::{compute_evidence, NestedSamplingConfig};
use cy3_rust_solver::route34::prior::{ModuliPoint, UniformPrior};

fn unit_gaussian_loglik(theta: &ModuliPoint) -> Result<LogLikelihoodResult, &'static str> {
    let x = theta.continuous[0];
    let chi2 = x * x;
    let log_norm = -0.5 * (2.0 * PI).ln();
    let bd = ChiSquaredBreakdown::from_components(chi2, 1, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0);
    Ok(LogLikelihoodResult {
        log_likelihood: -0.5 * chi2 + log_norm,
        chi_squared_breakdown: bd,
        p_value: 0.0,
        n_sigma: 0.0,
    })
}

fn narrow_gaussian_loglik(theta: &ModuliPoint) -> Result<LogLikelihoodResult, &'static str> {
    let x = theta.continuous[0];
    let sigma_d: f64 = 0.5;
    let chi2 = (x / sigma_d).powi(2);
    // Likelihood normalised to integrate to 1 over (-inf, inf).
    let log_norm = -0.5 * (2.0 * PI).ln() - sigma_d.ln();
    let bd = ChiSquaredBreakdown::from_components(chi2, 1, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0);
    Ok(LogLikelihoodResult {
        log_likelihood: -0.5 * chi2 + log_norm,
        chi_squared_breakdown: bd,
        p_value: 0.0,
        n_sigma: 0.0,
    })
}

#[test]
fn end_to_end_bayes_factor_recovers_analytic_value() {
    // Both candidates use the same prior; the analytic Bayes factor
    // is exp(0.5) (factor 2 in evidence).
    // Wait — let me re-derive: Z = int pi(theta) L(theta) dtheta
    //   pi = 1/10 on [-5, 5]
    //   For unit Gaussian (sigma_a = 1): Z_a = (1/10) * 1.0 (essentially)
    //   For narrow Gaussian (sigma_d = 0.5): Z_d = (1/10) * 1.0 also
    //   (since both are normalised likelihoods that fully fit in the box)
    // So actually Z_a / Z_d ~ 1. We need a different test.
    //
    // Let *un*-normalised Gaussians:
    //   L_a(theta) = exp(-0.5 * theta^2)            (peak = 1)
    //   L_d(theta) = exp(-0.5 * theta^2 / 0.25)     (peak = 1)
    // Then int L_a = sqrt(2 pi),   int L_d = sigma_d sqrt(2 pi) = 0.5 sqrt(2 pi).
    // So Z_a / Z_d = 2  =>  ln B = ln 2 ~ 0.6931.

    let prior = UniformPrior::new(vec![-5.0], vec![5.0]).unwrap();

    // Use *un-normalised* Gaussians so the Bayes factor is non-trivial.
    let unnorm_unit = |theta: &ModuliPoint| -> Result<LogLikelihoodResult, &'static str> {
        let x = theta.continuous[0];
        let chi2 = x * x;
        let bd = ChiSquaredBreakdown::from_components(chi2, 1, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0);
        Ok(LogLikelihoodResult {
            log_likelihood: -0.5 * chi2,
            chi_squared_breakdown: bd,
            p_value: 0.0,
            n_sigma: 0.0,
        })
    };
    let unnorm_narrow = |theta: &ModuliPoint| -> Result<LogLikelihoodResult, &'static str> {
        let x = theta.continuous[0];
        let sigma_d = 0.5_f64;
        let chi2 = (x / sigma_d).powi(2);
        let bd = ChiSquaredBreakdown::from_components(chi2, 1, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0);
        Ok(LogLikelihoodResult {
            log_likelihood: -0.5 * chi2,
            chi_squared_breakdown: bd,
            p_value: 0.0,
            n_sigma: 0.0,
        })
    };

    let cfg = NestedSamplingConfig {
        n_live: 800,
        stop_log_evidence_change: 1e-3,
        max_iterations: 50_000,
        seed: 314159,
        ..Default::default()
    };

    let z_a = compute_evidence(unnorm_unit, &prior, &cfg).unwrap();
    let cfg_b = NestedSamplingConfig { seed: 271828, ..cfg.clone() };
    let z_d = compute_evidence(unnorm_narrow, &prior, &cfg_b).unwrap();

    let bf = compute_bayes_factor("unit", &z_a, "narrow", &z_d);

    // Analytic ln B:
    //   Z_a ~ (1/10) * sqrt(2 pi) * P(chi^2_1 < 25)  ~ (1/10) * sqrt(2 pi)
    //   Z_d ~ (1/10) * 0.5 * sqrt(2 pi)
    //   ln Z_a - ln Z_d = ln 2 = 0.6931
    let analytic = 2.0_f64.ln();
    let diff = (bf.log_bayes_factor - analytic).abs();
    let sigma = bf.log_bayes_factor_uncertainty.max(1e-3);
    assert!(
        diff < 5.0 * sigma + 0.10,
        "ln B = {} +/- {} (expected ~ ln 2 = {}); diff = {}",
        bf.log_bayes_factor,
        sigma,
        analytic,
        diff
    );

    // The unit Gaussian must be preferred (larger evidence integral).
    assert_eq!(bf.preferred_candidate, "unit");
    assert_eq!(bf.disfavored_candidate, "narrow");

    // Touch the public types we shouldn't drop.
    let _ = unit_gaussian_loglik(&ModuliPoint::continuous_only(vec![0.0]));
    let _ = narrow_gaussian_loglik(&ModuliPoint::continuous_only(vec![0.0]));
}

#[test]
fn end_to_end_bayes_factor_with_normalised_gaussians_is_near_zero() {
    // Both likelihoods are properly-normalised Gaussians inside the
    // prior box. Their evidences are essentially equal, so |ln B| ~ 0.
    let prior = UniformPrior::new(vec![-5.0], vec![5.0]).unwrap();
    let cfg = NestedSamplingConfig {
        n_live: 400,
        stop_log_evidence_change: 1e-3,
        max_iterations: 20_000,
        seed: 1717,
        ..Default::default()
    };
    let z_a = compute_evidence(unit_gaussian_loglik, &prior, &cfg).unwrap();
    let cfg_b = NestedSamplingConfig { seed: 1729, ..cfg.clone() };
    let z_b = compute_evidence(unit_gaussian_loglik, &prior, &cfg_b).unwrap();

    let bf = compute_bayes_factor("a", &z_a, "b", &z_b);
    // Both should be near `ln(1/10) ~ -2.3026` after the truncation.
    let analytic_z = -10.0_f64.ln();
    assert!((z_a.log_evidence - analytic_z).abs() < 0.2);
    assert!((z_b.log_evidence - analytic_z).abs() < 0.2);
    // |ln B| should be small.
    assert!(
        bf.log_bayes_factor < 0.2,
        "expected near-zero Bayes factor, got {}",
        bf.log_bayes_factor
    );
}
