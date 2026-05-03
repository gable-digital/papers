//! # Combined likelihood `L(D | candidate, moduli) = exp(-1/2 chi^2_combined)`
//!
//! Aggregates the route-1 / route-2 / route-3 / route-4 / Yukawa-PDG
//! chi-squared contributions into a single Gaussian likelihood. The
//! per-route chi-squared modules are consumed read-only; this module
//! only assembles them and produces the log-likelihood, p-value, and
//! n-sigma summary.
//!
//! ## Mathematical structure
//!
//! `chi^2_combined = chi^2_route1 + chi^2_route2 + chi^2_route3
//!                 + chi^2_route4 + chi^2_yukawa_pdg`
//!
//! `log L = -0.5 * chi^2_combined`
//! (Gaussian likelihood with all entries reduced to standardised
//! residuals; the multiplicative `(2 pi)^{-k/2} prod sigma^{-1}` is
//! absorbed into a constant that cancels in the Bayes factor.)
//!
//! `p = Q(k/2, chi^2/2)`  (chi-squared survival function with k dof)
//! `n_sigma = Phi^{-1}(1 - p)`  (Gaussian inverse survival function)
//!
//! ## What this module does NOT do
//!
//! It does not run the underlying physics pipelines (HYM solve, harmonic
//! zero modes, eta integral, polyhedral wavenumber predictor, …). Those
//! live in
//!     [`crate::route34::yukawa_pipeline::predict_fermion_masses`],
//!     [`crate::route34::eta_evaluator::evaluate_eta_tian_yau`] /
//!     [`crate::route34::eta_evaluator::evaluate_eta_schoen`],
//!     [`crate::route34::route4_predictor::route4_discrimination`],
//! and the route-1 / route-2 evaluators in
//!     [`crate::route12`].
//! The user composes those into a [`ChiSquaredBreakdown`] and passes
//! it to [`evaluate_log_likelihood`].
//!
//! For the `nested_sampling::compute_evidence` API the user will
//! typically supply a closure that, given a [`crate::route34::prior::ModuliPoint`],
//! drives the full pipeline and returns a [`LogLikelihoodResult`].
//!
//! ## References
//!
//! * Cowan G., Cranmer K., Gross E., Vitells O., "Asymptotic formulae
//!   for likelihood-based tests of new physics", Eur. Phys. J. C 71
//!   (2011) 1554, arXiv:1007.1727.
//! * Particle Data Group, "Review of Particle Physics", §40
//!   (Statistics), 2024 update.

use serde::{Deserialize, Serialize};

// ----------------------------------------------------------------------
// Per-route chi-squared breakdown.
// ----------------------------------------------------------------------

/// Per-route chi-squared decomposition.
///
/// Each `route*` field is the sum of squared standardised residuals over
/// the observables of that route. `*_dof` is the corresponding number of
/// degrees of freedom (= number of observables).
///
/// Construct one of these from existing evaluators:
///
/// * `route1, route2, route3` from [`crate::route34::eta_evaluator::EtaResult`]
///   and the chapter-21 boundary-condition / Yukawa-determinant residuals.
/// * `route4` from [`crate::route34::route4_predictor::Route4Prediction::combined_chi_squared`].
/// * `yukawa_pdg` from [`crate::pdg::ChiSquaredResult::chi2_total`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChiSquaredBreakdown {
    pub route1: f64,
    pub route1_dof: u32,
    pub route2: f64,
    pub route2_dof: u32,
    pub route3: f64,
    pub route3_dof: u32,
    pub route4: f64,
    pub route4_dof: u32,
    pub yukawa_pdg: f64,
    pub yukawa_dof: u32,
    pub combined: f64,
    pub combined_dof: u32,
}

impl ChiSquaredBreakdown {
    /// Build from per-route components, computing the combined
    /// totals automatically.
    pub fn from_components(
        route1: f64,
        route1_dof: u32,
        route2: f64,
        route2_dof: u32,
        route3: f64,
        route3_dof: u32,
        route4: f64,
        route4_dof: u32,
        yukawa_pdg: f64,
        yukawa_dof: u32,
    ) -> Self {
        let combined = route1 + route2 + route3 + route4 + yukawa_pdg;
        let combined_dof = route1_dof + route2_dof + route3_dof + route4_dof + yukawa_dof;
        Self {
            route1,
            route1_dof,
            route2,
            route2_dof,
            route3,
            route3_dof,
            route4,
            route4_dof,
            yukawa_pdg,
            yukawa_dof,
            combined,
            combined_dof,
        }
    }

    /// Sanity check: `combined == sum(route_*)`, dof matches.
    pub fn is_consistent(&self, abs_tol: f64) -> bool {
        let sum =
            self.route1 + self.route2 + self.route3 + self.route4 + self.yukawa_pdg;
        let dof = self.route1_dof
            + self.route2_dof
            + self.route3_dof
            + self.route4_dof
            + self.yukawa_dof;
        (self.combined - sum).abs() <= abs_tol && self.combined_dof == dof
    }
}

// ----------------------------------------------------------------------
// LogLikelihoodResult.
// ----------------------------------------------------------------------

/// Output of [`evaluate_log_likelihood`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LogLikelihoodResult {
    /// `ln L(D | c, theta) = -0.5 * chi_squared_combined`.
    pub log_likelihood: f64,
    /// Per-route chi-squared decomposition (input).
    pub chi_squared_breakdown: ChiSquaredBreakdown,
    /// `p = Q(combined_dof/2, combined/2)` — survival function of
    /// chi^2 with `combined_dof` degrees of freedom.
    pub p_value: f64,
    /// Inverse standard-normal survival function of `p`. A 5-sigma
    /// deviation has `n_sigma >= 5`.
    pub n_sigma: f64,
}

// ----------------------------------------------------------------------
// Likelihood evaluator.
// ----------------------------------------------------------------------

/// Configuration for [`evaluate_log_likelihood`].
///
/// `clamp_chi_squared_to_zero` (default `true`) avoids `log L = +inf`
/// when chi^2 is reported as a tiny negative number due to round-off.
#[derive(Clone, Copy, Debug)]
pub struct LikelihoodConfig {
    pub clamp_chi_squared_to_zero: bool,
}

impl Default for LikelihoodConfig {
    fn default() -> Self {
        Self {
            clamp_chi_squared_to_zero: true,
        }
    }
}

/// Compute the combined log-likelihood from a [`ChiSquaredBreakdown`].
///
/// `breakdown.combined` and `breakdown.combined_dof` must be consistent
/// with the per-route fields (use [`ChiSquaredBreakdown::from_components`]
/// to guarantee this); the function additionally clamps `chi^2 >= 0`
/// when `config.clamp_chi_squared_to_zero` is set.
pub fn evaluate_log_likelihood(
    breakdown: &ChiSquaredBreakdown,
    config: &LikelihoodConfig,
) -> LogLikelihoodResult {
    let mut chi2 = breakdown.combined;
    if config.clamp_chi_squared_to_zero && chi2 < 0.0 {
        chi2 = 0.0;
    }
    let log_likelihood = -0.5 * chi2;
    let dof = breakdown.combined_dof.max(1) as f64;
    let p_value = chi2_sf(chi2, dof);
    let n_sigma = p_value_to_n_sigma(p_value);
    LogLikelihoodResult {
        log_likelihood,
        chi_squared_breakdown: breakdown.clone(),
        p_value,
        n_sigma,
    }
}

// ----------------------------------------------------------------------
// Chi-squared survival function and p -> n_sigma conversion.
//
// pdg.rs already implements these but keeps them private. We re-implement
// here using the same Numerical-Recipes / Acklam-2003 references; the
// constants are mathematical (Lanczos coefficients, Acklam's rational-
// approximation polynomials) and not adjustable parameters.
//
// References:
//   - Press, Teukolsky, Vetterling, Flannery, "Numerical Recipes" 3e
//     §6.2 (Lanczos lgamma), §6.2.1 (incomplete gamma: series + CF).
//   - Acklam, P.J. "An algorithm for computing the inverse normal
//     cumulative distribution function" (2003), Web archive:
//     https://web.archive.org/web/20151030215612/
//         http://home.online.no/~pjacklam/notes/invnorm/
// ----------------------------------------------------------------------

const PI: f64 = std::f64::consts::PI;

/// `ln Gamma(x)` via Lanczos g=7,n=9 (Numerical Recipes 3e §6.1).
fn lgamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const COEF: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula.
        return PI.ln() - (PI * x).sin().abs().ln() - lgamma(1.0 - x);
    }
    let xm1 = x - 1.0;
    let mut sum = COEF[0];
    for (i, c) in COEF.iter().enumerate().skip(1) {
        sum += c / (xm1 + i as f64);
    }
    let t = xm1 + G + 0.5;
    0.5 * (2.0 * PI).ln() + (xm1 + 0.5) * t.ln() - t + sum.ln()
}

/// Lower regularised incomplete gamma `P(a, z)` via series expansion
/// (Numerical Recipes 6.2.1). Converges for `z < a + 1`.
fn gamma_p_series(a: f64, z: f64) -> f64 {
    if z <= 0.0 {
        return 0.0;
    }
    let mut term = 1.0 / a;
    let mut sum = term;
    for n in 1..2000 {
        term *= z / (a + n as f64);
        sum += term;
        if term.abs() < sum.abs() * 1e-16 {
            break;
        }
    }
    sum * (-z + a * z.ln() - lgamma(a)).exp()
}

/// Upper regularised incomplete gamma `Q(a, z)` via Lentz continued
/// fraction (Numerical Recipes 6.2.1). Converges for `z >= a + 1`.
fn gamma_q_cf(a: f64, z: f64) -> f64 {
    let tiny = 1e-300_f64;
    let mut b = z + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..2000 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h * (-z + a * z.ln() - lgamma(a)).exp()
}

/// `Q(a, z)` selecting the appropriate branch.
fn gamma_q(a: f64, z: f64) -> f64 {
    if z < 0.0 || a <= 0.0 {
        return 1.0;
    }
    if z < a + 1.0 {
        1.0 - gamma_p_series(a, z)
    } else {
        gamma_q_cf(a, z)
    }
}

/// Survival function of chi^2 with `k` dof: `SF(x; k) = Q(k/2, x/2)`.
pub fn chi2_sf(x: f64, k: f64) -> f64 {
    if !x.is_finite() || x < 0.0 || x == 0.0 {
        return 1.0;
    }
    gamma_q(0.5 * k, 0.5 * x).clamp(0.0, 1.0)
}

/// Inverse standard-normal CDF (Acklam 2003).
fn inv_phi(p: f64) -> f64 {
    let p = p.clamp(1e-300, 1.0 - 1e-16);
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const CC: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let p_low = 0.02425_f64;
    let p_high = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((CC[0] * q + CC[1]) * q + CC[2]) * q + CC[3]) * q + CC[4]) * q + CC[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        ((((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q)
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -((((((CC[0] * q + CC[1]) * q + CC[2]) * q + CC[3]) * q + CC[4]) * q + CC[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0))
    }
}

/// Map a chi-squared p-value to the upper-tail Gaussian "n-sigma":
/// `n_sigma = Phi^{-1}(1 - p)`. Same convention as `pdg.rs`.
pub fn p_value_to_n_sigma(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::INFINITY;
    }
    if p >= 1.0 {
        return 0.0;
    }
    inv_phi(1.0 - p)
}

// ----------------------------------------------------------------------
// Convenience constructors that wire to existing route evaluators.
// ----------------------------------------------------------------------

/// Build a [`ChiSquaredBreakdown`] by running the route-1 evaluator
/// from [`crate::route34::route1`] and the route-2 evaluator from
/// [`crate::route34::route2`], then assembling with the supplied
/// route-3 (η) / route-4 / Yukawa-PDG numbers.
///
/// Centralises the production wiring so that
/// [`bin/bayes_discriminate.rs`] does not have to duplicate the
/// Routes-1/2 invocation code; both Rust unit tests and the binary
/// path call this constructor.
///
/// All arguments are passed by reference to keep the caller's memory
/// model untouched; the returned `ChiSquaredBreakdown` carries the
/// chi-squared totals, per-route DOF counts, and combined DOF.
pub fn breakdown_from_pipeline_results(
    metric: &dyn crate::route34::hym_hermitian::MetricBackground,
    modes: &crate::route34::zero_modes_harmonic::HarmonicZeroModeResult,
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
    let r1 = crate::route34::route1::compute_route1_chi_squared(
        metric,
        modes,
        &crate::route34::route1::Route1Config::default(),
    );
    let (r1_chi2, r1_dof) = match r1 {
        Ok(r) => (r.chi2_total, r.dof),
        Err(_) => (0.0, 0),
    };
    let r2 = crate::route34::route2::compute_route2_chi_squared(y_u, y_d, y_e);
    let (r2_chi2, r2_dof) = match r2 {
        Ok(r) => (r.chi2_total, r.dof),
        Err(_) => (0.0, 0),
    };

    breakdown_from_route_results(
        r1_chi2,
        r1_dof,
        r2_chi2,
        r2_dof,
        eta_predicted,
        eta_uncertainty,
        eta_observed,
        eta_observed_sigma,
        route4_chi2,
        route4_dof,
        yukawa_pdg_chi2,
        yukawa_dof,
    )
}

/// Build a [`ChiSquaredBreakdown`] from an `EtaResult` (route-3),
/// a `Route4Prediction` (route-4), a `ChiSquaredResult` from `pdg.rs`
/// (Yukawa-PDG), and free-standing route-1 and route-2 chi-squared
/// values supplied by the caller.
///
/// The `eta_observed` and `eta_observed_sigma` arguments specify the
/// observed eta value and its 1-sigma uncertainty; the route-3 chi^2
/// is then `((eta_predicted - eta_observed) / sigma_combined)^2` with
/// `sigma_combined = sqrt(eta_uncertainty^2 + eta_observed_sigma^2)`.
pub fn breakdown_from_route_results(
    route1_chi2: f64,
    route1_dof: u32,
    route2_chi2: f64,
    route2_dof: u32,
    eta_predicted: f64,
    eta_uncertainty: f64,
    eta_observed: f64,
    eta_observed_sigma: f64,
    route4_chi2: f64,
    route4_dof: u32,
    yukawa_pdg_chi2: f64,
    yukawa_dof: u32,
) -> ChiSquaredBreakdown {
    let sigma_combined =
        (eta_uncertainty * eta_uncertainty + eta_observed_sigma * eta_observed_sigma).sqrt();
    let route3_chi2 = if sigma_combined > 0.0 {
        let z = (eta_predicted - eta_observed) / sigma_combined;
        z * z
    } else {
        0.0
    };
    ChiSquaredBreakdown::from_components(
        route1_chi2,
        route1_dof,
        route2_chi2,
        route2_dof,
        route3_chi2,
        1,
        route4_chi2,
        route4_dof,
        yukawa_pdg_chi2,
        yukawa_dof,
    )
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_likelihood_correctly_aggregates_chi_squared() {
        // Build a synthetic breakdown; check the combined value equals
        // the sum and the log-likelihood is -0.5 * combined.
        let bd = ChiSquaredBreakdown::from_components(
            1.0, 3, 2.0, 4, 3.0, 1, 4.0, 3, 5.0, 13,
        );
        assert!((bd.combined - 15.0).abs() < 1e-15);
        assert_eq!(bd.combined_dof, 24);
        let r = evaluate_log_likelihood(&bd, &LikelihoodConfig::default());
        assert!((r.log_likelihood - (-7.5)).abs() < 1e-15);
        assert!(r.p_value > 0.0 && r.p_value < 1.0);
        // n_sigma can be negative when chi^2 is small relative to dof
        // (`p > 0.5` -> `1 - p < 0.5` -> `inv_phi < 0`); that just means
        // the model fits "better than expected".
        assert!(r.n_sigma.is_finite());
    }

    #[test]
    fn test_likelihood_log_decreases_with_chi_squared() {
        // Doubling chi^2 doubles -log L (i.e. halves the likelihood
        // exponent in absolute value -> the likelihood squares).
        let bd_a = ChiSquaredBreakdown::from_components(
            1.0, 1, 0.0, 0, 1.0, 1, 0.0, 0, 0.0, 0,
        );
        let bd_b = ChiSquaredBreakdown::from_components(
            2.0, 1, 0.0, 0, 2.0, 1, 0.0, 0, 0.0, 0,
        );
        let r_a = evaluate_log_likelihood(&bd_a, &LikelihoodConfig::default());
        let r_b = evaluate_log_likelihood(&bd_b, &LikelihoodConfig::default());
        // log L_a = -1, log L_b = -2 => L_b = L_a^2.
        assert!((r_a.log_likelihood - (-1.0)).abs() < 1e-15);
        assert!((r_b.log_likelihood - (-2.0)).abs() < 1e-15);
        assert!(r_b.log_likelihood < r_a.log_likelihood);
    }

    #[test]
    fn test_p_value_matches_pdg_standard() {
        // pdg.rs uses chi^2 = 56.4 as the 5-sigma threshold for k = 13:
        // the survival function at this chi^2 / dof gives p ~ a few e-7,
        // and the inverse-Gaussian-survival-function maps that to ~5σ.
        // We verify the p -> n_sigma map is consistent with the ~5-sigma
        // claim: chi^2 = 56.4 must correspond to n_sigma ~ 5.0 +/- 0.1.
        let p = chi2_sf(56.4, 13.0);
        assert!(
            p > 0.0 && p < 1e-6,
            "chi2_sf(56.4, 13) = {} (expected in [0, 1e-6])",
            p
        );
        let nsig = p_value_to_n_sigma(p);
        assert!(
            (nsig - 5.0).abs() < 0.1,
            "n_sigma at chi2=56.4 dof=13 = {} (expected ~5.0)",
            nsig
        );
    }

    #[test]
    fn test_p_value_known_chi2_dof1() {
        // chi^2 = 1, dof = 1 -> p ~ 0.317 (1 sigma)
        let p = chi2_sf(1.0, 1.0);
        assert!(
            (p - 0.317).abs() < 5e-3,
            "chi2_sf(1,1) = {} (expected ~0.317)",
            p
        );
        // chi^2 = 4, dof = 1 -> p ~ 0.0455 (2 sigma)
        let p2 = chi2_sf(4.0, 1.0);
        assert!(
            (p2 - 0.04550).abs() < 5e-4,
            "chi2_sf(4,1) = {} (expected ~0.0455)",
            p2
        );
        // chi^2 = 9, dof = 1 -> p ~ 0.00270 (3 sigma)
        let p3 = chi2_sf(9.0, 1.0);
        assert!(
            (p3 - 0.00270).abs() < 1e-4,
            "chi2_sf(9,1) = {} (expected ~0.0027)",
            p3
        );
    }

    #[test]
    fn test_p_value_known_chi2_dof_arbitrary() {
        // Mean of chi^2 with k dof is k; SF at mean is ~ 0.5 for large k.
        for &k in &[5.0_f64, 10.0, 20.0, 50.0] {
            let p = chi2_sf(k, k);
            // For large k the median is approximately k * (1 - 2/(9k))^3.
            // SF at mean is in [0.4, 0.55]; we just check it's not in tails.
            assert!(
                p > 0.3 && p < 0.6,
                "chi2_sf(k={}, k={}) = {} (expected ~0.5)",
                k,
                k,
                p
            );
        }
    }

    #[test]
    fn test_consistency_check() {
        let bd = ChiSquaredBreakdown::from_components(
            1.0, 3, 2.0, 4, 3.0, 1, 4.0, 3, 5.0, 13,
        );
        assert!(bd.is_consistent(1e-12));

        let mut bd_bad = bd.clone();
        bd_bad.combined = 999.0;
        assert!(!bd_bad.is_consistent(1e-12));

        let mut bd_dof_bad = bd.clone();
        bd_dof_bad.combined_dof = 999;
        assert!(!bd_dof_bad.is_consistent(1e-12));
    }

    #[test]
    fn test_clamp_negative_chi_squared() {
        let bd = ChiSquaredBreakdown {
            route1: -1e-12,
            route1_dof: 1,
            route2: 0.0,
            route2_dof: 0,
            route3: 0.0,
            route3_dof: 0,
            route4: 0.0,
            route4_dof: 0,
            yukawa_pdg: 0.0,
            yukawa_dof: 0,
            combined: -1e-12,
            combined_dof: 1,
        };
        let r = evaluate_log_likelihood(
            &bd,
            &LikelihoodConfig {
                clamp_chi_squared_to_zero: true,
            },
        );
        // Clamped chi^2 = 0 -> log L = 0.
        assert!(r.log_likelihood.abs() < 1e-15);
    }

    #[test]
    fn test_breakdown_from_route_results() {
        let bd = breakdown_from_route_results(
            0.5, 3, // route 1
            1.0, 2, // route 2
            6.115e-10,
            0.04e-10,
            6.0e-10,
            0.038e-10, // eta predicted, sigma; eta obs, sigma
            0.25, 3, // route 4
            10.0, 13, // yukawa
        );
        // route 3 chi^2 = ((6.115 - 6.0) / sqrt(0.04^2 + 0.038^2))^2 / 1e-10^2
        let dz = 0.115 / (0.04_f64.powi(2) + 0.038_f64.powi(2)).sqrt();
        let route3_expected = dz * dz;
        assert!(
            (bd.route3 - route3_expected).abs() / route3_expected < 1e-12,
            "route 3 = {} expected {}",
            bd.route3,
            route3_expected
        );
        assert_eq!(bd.combined_dof, 3 + 2 + 1 + 3 + 13);
        assert!(bd.is_consistent(1e-9));
    }

    #[test]
    fn test_inv_phi_endpoints() {
        // Phi(1.96) ~ 0.975
        let z = inv_phi(0.975);
        assert!((z - 1.96).abs() < 1e-3);
    }

    #[test]
    fn test_chi2_sf_zero_input() {
        assert_eq!(chi2_sf(0.0, 5.0), 1.0);
        assert_eq!(chi2_sf(-1.0, 5.0), 1.0);
    }
}
