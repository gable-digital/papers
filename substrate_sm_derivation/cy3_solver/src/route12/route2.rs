//! Route 2: Yukawa magnitudes from empirical force constants via the
//! cross-term-as-coupling identity. See [`crate::route12`] for context.
//!
//! ## The substrate-physics reading
//!
//! Chapter 8 §"Four Substrate-Specific Computational Routes" §Route 2
//! commits that the Yukawa coupling between two participating modes
//! IS the substrate's cross-term at their mode-overlap, with sign
//! determined by relative substrate-amplitude phase-alignment. For
//! the **gauge-coupling Yukawas** — i.e. the photon-fermion-fermion,
//! W/Z-fermion-fermion, gluon-quark-quark interactions — the cross-
//! term magnitude is empirically measured (Coulomb constant `e ≈
//! 0.303`, weak coupling `g_W ≈ 0.65`, strong coupling `g_s ≈ 1.22`
//! at `M_Z`). The substrate framework's identity gives the
//! gauge-Yukawa magnitude **directly** as that empirical force
//! magnitude, bypassing the triple-overlap-integral computation.
//!
//! Sign comes from relative substrate-amplitude phase-alignment at
//! the cross-term (see [`predict_gauge_yukawa_sign`]). Mainstream
//! string-theory phenomenology computes this sign via discrete
//! data (bundle's structure-group representation, Wilson-line
//! charge); the substrate framework reads it as a purely-geometric
//! phase-alignment property of the candidate CY3.
//!
//! ## Matter Yukawas (Higgs-fermion-fermion)
//!
//! The matter Yukawas (electron mass / Higgs VEV ≈ `2.94 × 10⁻⁶`,
//! up-quark `≈ 1.27 × 10⁻⁵`, …, top `≈ 1.0`) follow from the same
//! phase-alignment structure once the gauge ones are fixed, but the
//! reduction requires bundle-Hermite-Einstein data (P3 in the
//! lib-level preamble) that is itself a deferred research item. The
//! matter-Yukawa side of Route 2 is therefore a placeholder that
//! surfaces the *form* of the prediction without claiming the
//! bundle-dependent prefactor; see [`MatterYukawaPrediction`].
//!
//! ## Discrimination chi^2
//!
//! [`route2_chi2_against_pdg`] returns a chi-squared sum over the
//! gauge-coupling Yukawas comparing predicted (Route 2) to observed
//! (PDG) values. For the gauge sector the prediction is ≡ observation
//! by construction (Route 2's identity); the chi^2 is therefore zero
//! up to RG-running corrections between the input scale and `M_Z`.
//! The matter-sector chi^2 is left at zero with a documentation
//! note pending the deferred research items.

use crate::pdg::Pdg2024;

/// Predicted gauge-Yukawa magnitude from a fine-structure-style
/// coupling `α`. Returns the dimensionless coupling `g = √(4 π α)`
/// — the textbook conversion that identifies the gauge coupling
/// (in QFT normalisation) with the substrate cross-term magnitude.
///
/// Examples:
/// * α_em (M_Z) ≈ 1/127.952 → g_em ≈ 0.3134 (electroweak photon coupling)
/// * α_s  (M_Z) ≈ 0.1181    → g_s  ≈ 1.218
pub fn predict_gauge_yukawa_from_alpha(alpha: f64) -> f64 {
    if alpha <= 0.0 || !alpha.is_finite() {
        return 0.0;
    }
    (4.0 * std::f64::consts::PI * alpha).sqrt()
}

/// Predicted Yukawa sign for a cross-term at the given relative
/// substrate-amplitude phase-alignment angle (radians). Per the
/// substrate framework's `hyp_substrate_force_unification_via_cross-
/// term_sign`, the sign is `+1` when the participating modes are
/// in phase-alignment (`cos(phase) > 0`), `-1` when antialigned,
/// and `0` at exact orthogonality (`cos(phase) = 0`).
///
/// `phase_radians` is the relative phase-alignment angle between the
/// participating modes' substrate-amplitude patterns at the
/// cross-term overlap. Mainstream string-theory phenomenology
/// computes this sign from discrete Wilson-line / structure-group
/// data; the substrate framework reads it as a purely-geometric
/// property of the candidate CY3.
pub fn predict_gauge_yukawa_sign(phase_radians: f64) -> f64 {
    let c = phase_radians.cos();
    if c > 0.0 { 1.0 } else if c < 0.0 { -1.0 } else { 0.0 }
}

/// Per-sector predicted gauge-Yukawa magnitudes at `M_Z`, derived
/// from the PDG-2024 fine-structure measurements via the Route 2
/// cross-term-as-coupling identity. Sign is left as `+1` (assumes
/// in-phase substrate-amplitude alignment); a real CY3 candidate
/// would supply the per-vertex phase via [`predict_gauge_yukawa_sign`].
///
/// Returned values:
///   * `electromagnetic`: `√(4π α_em(M_Z))` ≈ `0.3134`
///   * `weak`            : `√(4π α_2(M_Z))`  ≈ `0.6517`
///   * `strong`          : `√(4π α_s(M_Z))`  ≈ `1.218`
///
/// Uses PDG-2024 values: `α_em⁻¹(M_Z) = 127.952`, `α_s(M_Z) = 0.1181`,
/// `sin²θ_W = 0.23121` → `α_2(M_Z) = α_em(M_Z) / sin²θ_W ≈ 0.0338`.
pub struct GaugeYukawaPrediction {
    pub electromagnetic: f64,
    pub weak: f64,
    pub strong: f64,
}

impl GaugeYukawaPrediction {
    pub fn from_pdg(_pdg: &Pdg2024) -> Self {
        // PDG-2024 EW review (July 2024).
        const ALPHA_EM_MZ: f64 = 1.0 / 127.952;
        const ALPHA_S_MZ: f64 = 0.1181;
        const SIN2_THETA_W: f64 = 0.23121;
        let alpha_2 = ALPHA_EM_MZ / SIN2_THETA_W;
        Self {
            electromagnetic: predict_gauge_yukawa_from_alpha(ALPHA_EM_MZ),
            weak: predict_gauge_yukawa_from_alpha(alpha_2),
            strong: predict_gauge_yukawa_from_alpha(ALPHA_S_MZ),
        }
    }
}

/// Placeholder for the matter-Yukawa prediction side of Route 2.
/// Currently returns zeros for every entry: the precise
/// bundle-dependent prefactor that maps phase-alignment structure
/// to per-fermion Yukawa magnitudes is the open computation that
/// task #48 (Hermite-Einstein H) and #49 (full ε-tensor cup
/// product) target.
///
/// Once those upstream items land, this struct can carry the
/// predicted per-fermion Yukawas (m_e/v, m_μ/v, …, m_t/v) and
/// [`route2_chi2_against_pdg`] can fold them into the χ² sum.
pub struct MatterYukawaPrediction {
    pub up_quark: f64,
    pub down_quark: f64,
    pub strange_quark: f64,
    pub charm_quark: f64,
    pub bottom_quark: f64,
    pub top_quark: f64,
    pub electron: f64,
    pub muon: f64,
    pub tau: f64,
}

impl Default for MatterYukawaPrediction {
    fn default() -> Self {
        // All zeros: the bundle-dependent prefactor is the deferred
        // research item. A non-zero return would imply Route 2
        // had a numerical prediction for the matter Yukawas, which
        // it currently does not.
        Self {
            up_quark: 0.0, down_quark: 0.0, strange_quark: 0.0,
            charm_quark: 0.0, bottom_quark: 0.0, top_quark: 0.0,
            electron: 0.0, muon: 0.0, tau: 0.0,
        }
    }
}

/// Total Route-2 χ² for a candidate CY3 versus PDG observation.
///
/// At present this returns a small *sanity* χ² for the gauge sector
/// (essentially zero up to floating-point noise — the prediction is
/// the PDG value by construction) plus zero for the matter sector
/// (the matter-Yukawa prefactor is deferred).
///
/// The `_predicted_*` arguments exist so a future caller can pass
/// in CY3-derived predictions (e.g. from the Bott-representative
/// basis enumeration once #50 lands) and have them χ²-aggregated
/// against PDG. Until then, callers should pass
/// [`GaugeYukawaPrediction::from_pdg`] for the gauge side and
/// [`MatterYukawaPrediction::default()`] for the matter side; the
/// χ² will be ≈ 0 for the gauge sector and exactly 0 for the
/// matter sector — i.e. Route 2 currently adds no discrimination
/// signal beyond the structural identification, which is honest.
pub fn route2_chi2_against_pdg(
    predicted_gauge: &GaugeYukawaPrediction,
    predicted_matter: &MatterYukawaPrediction,
    pdg: &Pdg2024,
) -> f64 {
    let observed = GaugeYukawaPrediction::from_pdg(pdg);
    let observed_matter = MatterYukawaPrediction::default();

    // Gauge sector: predicted vs observed, both via the same
    // cross-term-as-coupling identity. χ² ≈ 0 by construction.
    // Use a tight 1‰ tolerance (well below the per-coupling PDG
    // uncertainties) so any future drift in the predicted-side
    // prefactor immediately surfaces.
    let rel_residual = |pred: f64, obs: f64| -> f64 {
        if obs.abs() < 1.0e-12 {
            return 0.0;
        }
        let r = (pred - obs) / (obs * 1.0e-3);
        r * r
    };
    let chi2_gauge = rel_residual(predicted_gauge.electromagnetic, observed.electromagnetic)
        + rel_residual(predicted_gauge.weak, observed.weak)
        + rel_residual(predicted_gauge.strong, observed.strong);

    // Matter sector: until Route 2's bundle-prefactor side is
    // implemented, we compare predicted (zeros by default) against
    // a likewise-zero reference, giving χ² = 0. Once the upstream
    // research items land this comparison switches to PDG fermion
    // masses divided by the Higgs VEV.
    let mat_diff = |pred: f64, obs: f64| -> f64 {
        let r = pred - obs;
        r * r
    };
    let chi2_matter = mat_diff(predicted_matter.up_quark, observed_matter.up_quark)
        + mat_diff(predicted_matter.down_quark, observed_matter.down_quark)
        + mat_diff(predicted_matter.strange_quark, observed_matter.strange_quark)
        + mat_diff(predicted_matter.charm_quark, observed_matter.charm_quark)
        + mat_diff(predicted_matter.bottom_quark, observed_matter.bottom_quark)
        + mat_diff(predicted_matter.top_quark, observed_matter.top_quark)
        + mat_diff(predicted_matter.electron, observed_matter.electron)
        + mat_diff(predicted_matter.muon, observed_matter.muon)
        + mat_diff(predicted_matter.tau, observed_matter.tau);

    chi2_gauge + chi2_matter
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_em_gives_expected_em_coupling() {
        // α_em(M_Z) ≈ 1/127.952  → g_em ≈ √(4π · 1/127.952) ≈ 0.3134
        let g = predict_gauge_yukawa_from_alpha(1.0 / 127.952);
        assert!((g - 0.3134).abs() < 1.0e-3, "got g_em = {g}");
    }

    #[test]
    fn alpha_s_gives_expected_strong_coupling() {
        // α_s(M_Z) ≈ 0.1181  → g_s ≈ √(4π · 0.1181) ≈ 1.218
        let g = predict_gauge_yukawa_from_alpha(0.1181);
        assert!((g - 1.218).abs() < 0.005, "got g_s = {g}");
    }

    #[test]
    fn negative_alpha_returns_zero() {
        assert_eq!(predict_gauge_yukawa_from_alpha(-1.0), 0.0);
        assert_eq!(predict_gauge_yukawa_from_alpha(0.0), 0.0);
        assert_eq!(predict_gauge_yukawa_from_alpha(f64::NAN), 0.0);
    }

    #[test]
    fn yukawa_sign_matches_phase() {
        // 0 rad: cos = 1 > 0 → +1
        assert_eq!(predict_gauge_yukawa_sign(0.0), 1.0);
        // π rad: cos = -1 → -1
        assert_eq!(predict_gauge_yukawa_sign(std::f64::consts::PI), -1.0);
        // 3π/2 (well past π): cos = 0 nominally but f64 representable
        // is ≈ -1.84e-16 < 0 → -1.
        assert_eq!(
            predict_gauge_yukawa_sign(3.0 * std::f64::consts::FRAC_PI_2),
            -1.0
        );
        // 2π: cos = 1 (full turn) → +1
        assert_eq!(predict_gauge_yukawa_sign(2.0 * std::f64::consts::PI), 1.0);
    }

    #[test]
    fn gauge_prediction_from_pdg_self_consistent() {
        let pdg = Pdg2024::new();
        let pred = GaugeYukawaPrediction::from_pdg(&pdg);
        // EM at M_Z is ~ 0.31; weak ~ 0.65; strong ~ 1.22.
        assert!(pred.electromagnetic > 0.30 && pred.electromagnetic < 0.32);
        assert!(pred.weak > 0.63 && pred.weak < 0.67);
        assert!(pred.strong > 1.20 && pred.strong < 1.23);
    }

    #[test]
    fn route2_chi2_zero_when_predicted_equals_pdg() {
        let pdg = Pdg2024::new();
        let pred_gauge = GaugeYukawaPrediction::from_pdg(&pdg);
        let pred_matter = MatterYukawaPrediction::default();
        let chi2 = route2_chi2_against_pdg(&pred_gauge, &pred_matter, &pdg);
        // Route 2 identifies prediction with PDG-derived value, so
        // chi^2 vanishes (up to FP noise — well below the unit
        // tolerance the residual division uses).
        assert!(chi2.abs() < 1.0e-10, "chi^2 not zero: {chi2}");
    }

    #[test]
    fn route2_chi2_grows_when_gauge_perturbed() {
        let pdg = Pdg2024::new();
        let mut pred_gauge = GaugeYukawaPrediction::from_pdg(&pdg);
        pred_gauge.electromagnetic *= 1.01; // +1% off
        let pred_matter = MatterYukawaPrediction::default();
        let chi2 = route2_chi2_against_pdg(&pred_gauge, &pred_matter, &pdg);
        // 1% off with 0.1% tolerance → 10σ → chi^2 = 100.
        assert!(
            (chi2 - 100.0).abs() < 5.0,
            "chi^2 should be ~100 for 1% perturbation under 0.1% tolerance; got {chi2}"
        );
    }
}
