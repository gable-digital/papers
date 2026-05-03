//! Route 5 chi^2: scalar spectral index `n_s` from `E_8 × E_8`
//! Coxeter geometry, with merger-class correction per chapter 8.
//!
//! See [`crate::route5`] for the three-step derivation chain.

/// Coxeter number of `E_8`. Chapter 8 §"The Three-Step Derivation
/// Chain" Step A: the parent-horizon angular-mode cutoff is the
/// surviving `E_8` Coxeter content per heterotic sector. Standard
/// representation-theory value, no free parameter.
pub const COXETER_E8: u32 = 30;

/// Total e-fold count `N = 2 h_E8 = 60` (Step B). Both heterotic
/// `E_8` sectors contribute Coxeter-number `30` each at the
/// inversion-boundary level (pre-Wilson-line breaking).
pub const N_EFOLD_LEADING: u32 = 2 * COXETER_E8;

/// Leading-order substrate prediction for the scalar spectral
/// index: `n_s = 1 − 2/N = 1 − 2/60 = 58/60 ≈ 0.9667`. No free
/// parameters; candidate-CY3-independent at this order.
pub fn n_s_leading() -> f64 {
    1.0 - 2.0 / N_EFOLD_LEADING as f64
}

/// Planck 2018 measured central value of the scalar spectral
/// index, `n_s = 0.9649 ± 0.0042` (Aghanim et al. 2020,
/// arXiv:1807.06209 §3, "Planck 2018 results VI: cosmological
/// parameters", Table 2 base-ΛCDM TT,TE,EE+lowE+lensing column).
pub const N_S_PLANCK_2018_CENTRAL: f64 = 0.9649;

/// Planck 2018 1σ uncertainty on `n_s`.
pub const N_S_PLANCK_2018_SIGMA: f64 = 0.0042;

/// CMB-S4 forecast precision on `n_s` (chapter-8 figure).
/// Approximately 0.001, two orders of magnitude tighter than
/// Planck. Quoted here so callers / report formatters can flag
/// sub-leading discrimination signals as "in CMB-S4 reach" vs
/// "below the next-decade observable threshold".
pub const N_S_CMB_S4_FORECAST_SIGMA: f64 = 0.001;

/// Per-candidate merger-class correction to the e-fold count `N`,
/// returned as a signed `Δn_s = -2 ΔN / N²` shift on the leading-
/// order prediction.
///
/// Per chapter 8 §"Discrimination Program Implications" the shift
/// is "of order `Δn_s ~ 0.001`" and depends on the candidate-CY3's
/// Killing-vector projection structure at the inversion boundary.
/// **The framework does not yet supply a closed-form formula** for
/// the per-candidate Δ; this function returns the chapter's
/// declared structure (different magnitude per candidate, sub-
/// leading) using a placeholder dependence on
/// [`crate::route34::KillingResult::cyclic_factors`].
///
/// The placeholder is anchored at the chapter-8 figure: TY/Z3 and
/// Schoen/Z3×Z3 differ by ~ 1 unit of `ΔN`, so their `Δn_s` differ
/// by ~ `2 / N² ≈ 5.6 × 10⁻⁴`. Concretely:
///
///   * TY/Z3 (one Z/3 factor)        : `ΔN = +0.5` → `Δn_s = −2.8e-4`
///   * Schoen/Z3×Z3 (two Z/3 factors): `ΔN = −0.5` → `Δn_s = +2.8e-4`
///
/// Symmetric around zero so the `Δn_s` differential between
/// candidates is the chapter's full ~0.001-class signal. **When
/// the framework's actual closed-form arrives** (gated on the
/// route34 Killing-vector projection at the inversion boundary
/// being implemented), replace this with the real expression.
/// Until then this function delivers a chapter-respecting
/// discrimination signal that is honestly labelled placeholder.
pub fn merger_class_dn_s(fundamental_group: &str) -> f64 {
    let two_over_n_sq = 2.0 / (N_EFOLD_LEADING as f64).powi(2);
    let delta_n: f64 = match fundamental_group {
        "Z3" => 0.5,           // TY/Z3
        "Z3xZ3" => -0.5,       // Schoen/Z3 × Z/3
        _ => 0.0,              // unknown / future candidates
    };
    -delta_n * two_over_n_sq
}

/// Candidate-specific predicted spectral index: leading order +
/// merger-class correction. The leading order is the same for all
/// candidates (chapter 8 §"Discrimination Program Implications");
/// only the small merger-class shift is candidate-specific.
pub fn n_s_predicted(fundamental_group: &str) -> f64 {
    n_s_leading() + merger_class_dn_s(fundamental_group)
}

/// Per-candidate Route 5 χ² against Planck 2018 + CMB-S4-forecast
/// (whichever is tighter at the precision the candidate's prediction
/// has been computed). Returns a tuple
///
///   `(chi2, n_s_pred, sigma_used)`
///
/// where `chi2 = ((n_s_pred − n_s_obs) / σ)²` with σ the **larger**
/// of `N_S_PLANCK_2018_SIGMA` and a candidate-side error budget the
/// caller may pass (left implicit at zero for now since the
/// framework's leading-order prediction is exact in `1/N`).
pub fn route5_chi2_against_planck(
    fundamental_group: &str,
) -> (f64, f64, f64) {
    let pred = n_s_predicted(fundamental_group);
    let chi2 = ((pred - N_S_PLANCK_2018_CENTRAL) / N_S_PLANCK_2018_SIGMA).powi(2);
    (chi2, pred, N_S_PLANCK_2018_SIGMA)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coxeter_e8_is_thirty() {
        // E_8 Coxeter number is 30 (Bourbaki 1968; Humphreys 1990
        // Reflection groups & Coxeter groups Table 1.1 p. 32).
        assert_eq!(COXETER_E8, 30);
    }

    #[test]
    fn n_efold_is_sixty() {
        // N = 2 h_E8 = 60. Step B of the derivation chain.
        assert_eq!(N_EFOLD_LEADING, 60);
    }

    #[test]
    fn n_s_leading_is_58_over_60() {
        let pred = n_s_leading();
        // 58 / 60 = 0.96666...
        let target = 58.0_f64 / 60.0_f64;
        assert!(
            (pred - target).abs() < 1.0e-15,
            "n_s_leading should be exactly 58/60 = {target}; got {pred}"
        );
        // Sanity: ≈ 0.9667
        assert!((pred - 0.9666666666_f64).abs() < 1.0e-9);
    }

    #[test]
    fn merger_class_zero_for_unknown_group() {
        // No correction for unrecognised fundamental groups.
        assert_eq!(merger_class_dn_s("trivial"), 0.0);
        assert_eq!(merger_class_dn_s(""), 0.0);
        assert_eq!(merger_class_dn_s("Z2"), 0.0);
    }

    #[test]
    fn merger_class_is_subleading_size() {
        // Per chapter 8: Δn_s ~ 0.001 between candidates.
        let ty = merger_class_dn_s("Z3");
        let sch = merger_class_dn_s("Z3xZ3");
        // Each is ≤ 1e-3 in magnitude.
        assert!(ty.abs() < 1.0e-3);
        assert!(sch.abs() < 1.0e-3);
        // Their differential is ~ 5.6e-4 (chapter's "~ 0.001-class").
        let diff = (ty - sch).abs();
        assert!(
            diff > 5.0e-4 && diff < 1.0e-3,
            "TY-vs-Schoen merger-class differential should be ~5e-4..1e-3; got {diff}"
        );
    }

    #[test]
    fn predicted_n_s_is_leading_plus_correction() {
        let leading = n_s_leading();
        let ty = n_s_predicted("Z3");
        let sch = n_s_predicted("Z3xZ3");
        assert!((ty - leading - merger_class_dn_s("Z3")).abs() < 1.0e-15);
        assert!((sch - leading - merger_class_dn_s("Z3xZ3")).abs() < 1.0e-15);
        // Both predictions live within ±0.001 of 58/60.
        assert!((ty - leading).abs() < 1.0e-3);
        assert!((sch - leading).abs() < 1.0e-3);
    }

    #[test]
    fn route5_chi2_against_planck_is_finite_for_both_candidates() {
        for group in ["Z3", "Z3xZ3"] {
            let (chi2, pred, sigma) = route5_chi2_against_planck(group);
            assert!(chi2.is_finite() && chi2 >= 0.0, "chi2 not finite/positive: {chi2}");
            assert!(pred > 0.9 && pred < 1.0, "n_s_pred out of plausible range: {pred}");
            assert_eq!(sigma, N_S_PLANCK_2018_SIGMA);
        }
    }

    #[test]
    fn route5_chi2_close_to_0_4_sigma_squared() {
        // Per chapter 8: leading-order prediction sits ~ 0.4σ above
        // Planck central. So chi^2 from leading-only should be
        // ~ (0.4)^2 = 0.16. Candidate corrections only nudge.
        let (chi2_ty, _, _) = route5_chi2_against_planck("Z3");
        let (chi2_sch, _, _) = route5_chi2_against_planck("Z3xZ3");
        // Both lie in [0.0, ~0.5] under the chapter's claim.
        assert!(chi2_ty >= 0.0 && chi2_ty < 0.5,
            "TY chi^2 outside expected band: {chi2_ty}");
        assert!(chi2_sch >= 0.0 && chi2_sch < 0.5,
            "Schoen chi^2 outside expected band: {chi2_sch}");
    }
}
