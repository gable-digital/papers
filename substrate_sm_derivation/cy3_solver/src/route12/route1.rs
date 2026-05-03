//! Route 1: empirical observables as boundary-condition penalties on
//! the CY3 metric. See [`crate::route12`] for context.
//!
//! The penalty functions in this module take a candidate metric (or
//! summary statistics derived from one) plus a discrete observable
//! (e.g. observed Coulomb-falloff exponent, observed weak-interaction
//! range, observed polyhedral-resonance wavenumbers) and return a
//! `≥ 0` violation score. The metric solver folds these into its
//! Adam objective as additional terms.
//!
//! Each penalty has the same shape: zero when the candidate metric
//! reproduces the observable to within the empirical uncertainty,
//! quadratically growing as the predicted value moves away. The
//! `_residual_*` variants return the signed dimensionless residual
//! (predicted − observed) / σ for use in χ²-aggregation upstream.
//!
//! ## Status
//!
//! These are leading-order penalty stubs that take the *summary
//! statistic* (e.g. an exponent, a length scale, a wavenumber set)
//! rather than the full metric. The full implementation —
//! computing the predicted summary statistic from the candidate CY3
//! metric directly — depends on the bundle Hermite-Einstein metric
//! work (P3 in the lib-level preamble) and the full Bott-representative
//! basis enumeration (task #50). The current penalties exist so that
//! when those upstream items land, the boundary-constraint piece is
//! already shaped and ready to receive the predicted statistics.

/// Coulomb falloff exponent: physically `≡ 2.0` (1/r² → integrates to
/// 1/r long-range potential). The substrate-physics reading is that
/// the photon-mediator-mode zero-mode wavefunction's strain-tail
/// exponent on the CY3 must reproduce this. Returns the squared
/// dimensionless residual `((predicted − 2.0) / σ)²` with `σ`
/// defaulting to the LEP+LHC bound on photon-anomaly contributions
/// (≈ `0.001`); a larger user-supplied `sigma` widens the tolerance.
pub fn coulomb_falloff_penalty(predicted_exponent: f64, sigma: f64) -> f64 {
    let s = sigma.max(1.0e-6);
    let r = (predicted_exponent - 2.0) / s;
    r * r
}

/// Coulomb falloff residual (signed, dimensionless), for χ² aggregation.
pub fn coulomb_falloff_residual(predicted_exponent: f64, sigma: f64) -> f64 {
    let s = sigma.max(1.0e-6);
    (predicted_exponent - 2.0) / s
}

/// Weak-interaction range. The Yukawa-potential range of the W/Z
/// mediators is `λ_W = ℏ / (M_W c) ≈ 2.5 × 10⁻¹⁸ m`. The candidate
/// CY3 metric must produce a W/Z mediator-mode whose effective range
/// matches this to within the empirical bound.
///
/// `predicted_range_m` is the predicted range in metres;
/// `observed_range_m` defaults to PDG's `M_W = 80.377 GeV` →
/// `λ ≈ 2.452 × 10⁻¹⁸ m`. `relative_sigma` is the fractional
/// uncertainty on the observed range (PDG bound is `~ 10⁻⁴` on
/// `M_W`, so `relative_sigma = 1e-4` is appropriate).
pub fn weak_range_penalty(
    predicted_range_m: f64,
    observed_range_m: f64,
    relative_sigma: f64,
) -> f64 {
    let s = relative_sigma.max(1.0e-6) * observed_range_m.abs();
    if s <= 0.0 {
        return 0.0;
    }
    let r = (predicted_range_m - observed_range_m) / s;
    r * r
}

/// Strong-force confinement scale. The `Λ_QCD ≈ 200 MeV` confinement
/// scale corresponds to a long-distance suppression of the
/// gluon-class cross-term content; the candidate CY3 metric must
/// produce a gluon-mode strain tail that is exponentially suppressed
/// at distances `≳ 1 fm`.
///
/// Returns a quadratic penalty on the dimensionless residual between
/// predicted and observed `Λ_QCD` in MeV.
pub fn strong_confinement_penalty(
    predicted_lambda_qcd_mev: f64,
    observed_lambda_qcd_mev: f64,
    sigma_mev: f64,
) -> f64 {
    let s = sigma_mev.max(0.5);
    let r = (predicted_lambda_qcd_mev - observed_lambda_qcd_mev) / s;
    r * r
}

/// Polyhedral-resonance wavenumber penalty. The substrate framework's
/// polyhedral-resonance hypothesis predicts that stable polyhedral
/// patterns at rotating-body polar regions take ADE-classified
/// wavenumbers. The observed set on solar-system bodies is
/// `{6, 8, 5}` (Saturn polar hexagon; Jupiter north polar octagon;
/// Jupiter south polar pentagon — see chapter 8 §"Pinning Down
/// Route 4").
///
/// `predicted_wavenumbers` is the set of stable polyhedral
/// wavenumbers the candidate CY3's Killing-vector spectrum
/// projected through Arnold's catastrophe-theory ADE classification
/// at the polar critical boundary admits. Returns the symmetric
/// difference cardinality (number of observed wavenumbers missing
/// + number of predicted-but-not-observed extras) — `0` for an
/// exact match.
///
/// This is a discrete penalty (integer-valued cast to `f64`); the
/// substrate framework's prediction is structural rather than
/// continuous, so a wavenumber being predicted-but-not-observed is
/// a definite falsifier rather than a soft tension.
pub fn polyhedral_resonance_penalty(
    predicted_wavenumbers: &[u32],
    observed_wavenumbers: &[u32],
) -> f64 {
    use std::collections::HashSet;
    let p: HashSet<u32> = predicted_wavenumbers.iter().copied().collect();
    let o: HashSet<u32> = observed_wavenumbers.iter().copied().collect();
    let missing = o.difference(&p).count();
    let extra = p.difference(&o).count();
    (missing + extra) as f64
}

/// Convenience: total Route 1 χ² for a candidate, given a bundle of
/// predicted summary statistics and the empirical observed values
/// + uncertainties. Sums the four individual penalties.
#[allow(clippy::too_many_arguments)]
pub fn route1_total_chi2(
    coulomb_predicted_exponent: f64,
    coulomb_sigma: f64,
    weak_predicted_range_m: f64,
    weak_observed_range_m: f64,
    weak_relative_sigma: f64,
    strong_predicted_lambda_qcd_mev: f64,
    strong_observed_lambda_qcd_mev: f64,
    strong_sigma_mev: f64,
    polyhedral_predicted_wavenumbers: &[u32],
    polyhedral_observed_wavenumbers: &[u32],
) -> f64 {
    coulomb_falloff_penalty(coulomb_predicted_exponent, coulomb_sigma)
        + weak_range_penalty(
            weak_predicted_range_m,
            weak_observed_range_m,
            weak_relative_sigma,
        )
        + strong_confinement_penalty(
            strong_predicted_lambda_qcd_mev,
            strong_observed_lambda_qcd_mev,
            strong_sigma_mev,
        )
        + polyhedral_resonance_penalty(
            polyhedral_predicted_wavenumbers,
            polyhedral_observed_wavenumbers,
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coulomb_zero_at_observed() {
        assert_eq!(coulomb_falloff_penalty(2.0, 0.001), 0.0);
        assert!(coulomb_falloff_penalty(2.001, 0.001).abs() - 1.0 < 1e-9);
    }

    #[test]
    fn coulomb_residual_sign() {
        assert!(coulomb_falloff_residual(2.5, 0.001) > 0.0);
        assert!(coulomb_falloff_residual(1.5, 0.001) < 0.0);
    }

    #[test]
    fn weak_zero_at_observed() {
        let observed = 2.452e-18;
        assert!(weak_range_penalty(observed, observed, 1e-4).abs() < 1e-12);
    }

    #[test]
    fn weak_quadratic_growth() {
        let observed = 2.452e-18;
        let p1 = weak_range_penalty(observed * 1.001, observed, 1e-4);
        let p2 = weak_range_penalty(observed * 1.002, observed, 1e-4);
        assert!((p2 / p1 - 4.0).abs() < 0.01); // doubling residual → quadrupling penalty
    }

    #[test]
    fn strong_zero_at_observed() {
        assert_eq!(strong_confinement_penalty(200.0, 200.0, 5.0), 0.0);
    }

    #[test]
    fn polyhedral_exact_match_zero() {
        assert_eq!(polyhedral_resonance_penalty(&[5, 6, 8], &[6, 8, 5]), 0.0);
    }

    #[test]
    fn polyhedral_missing_predicted_costs_one_per_wavenumber() {
        // Predicted {6, 8} but observed {5, 6, 8} → 1 missing.
        assert_eq!(polyhedral_resonance_penalty(&[6, 8], &[5, 6, 8]), 1.0);
        // Predicted {5, 6, 7, 8} but observed {5, 6, 8} → 1 extra.
        assert_eq!(polyhedral_resonance_penalty(&[5, 6, 7, 8], &[5, 6, 8]), 1.0);
    }

    #[test]
    fn polyhedral_two_off() {
        // Predicted {5, 7, 8}, observed {5, 6, 8} → 1 missing (6) + 1
        // extra (7) = 2.
        assert_eq!(polyhedral_resonance_penalty(&[5, 7, 8], &[5, 6, 8]), 2.0);
    }

    #[test]
    fn route1_total_chi2_zero_at_perfect_match() {
        let observed_wn = [5, 6, 8];
        let chi2 = route1_total_chi2(
            2.0,           // coulomb predicted == observed
            0.001,
            2.452e-18,     // weak predicted == observed
            2.452e-18,
            1e-4,
            200.0,         // Λ_QCD predicted == observed
            200.0,
            5.0,
            &observed_wn,
            &observed_wn,
        );
        assert_eq!(chi2, 0.0);
    }

    #[test]
    fn route1_total_chi2_nonzero_when_one_off() {
        // Tweak Coulomb exponent only — total χ² should be exactly
        // the Coulomb penalty.
        let observed_wn = [5, 6, 8];
        let total = route1_total_chi2(
            2.001, 0.001,
            2.452e-18, 2.452e-18, 1e-4,
            200.0, 200.0, 5.0,
            &observed_wn, &observed_wn,
        );
        let coulomb = coulomb_falloff_penalty(2.001, 0.001);
        assert!((total - coulomb).abs() < 1e-12);
    }
}
