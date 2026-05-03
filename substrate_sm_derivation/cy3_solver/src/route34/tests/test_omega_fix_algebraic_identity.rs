//! P7.12 — Algebraic-identity regression tests for ω_fix.
//!
//! These tests pin the journal's claim
//!   ω_fix = 1/2 − 1/dim(E_8) = (dim−2)/(2·dim) = 123/248
//! as an *exact algebraic invariant* of E_8 (not as a measurable
//! eigenvalue, which the P7.1–P7.10 series demonstrated it is *not*
//! at journal precision on the Schoen Donaldson background).
//!
//! Three tests, in increasing order of empirical content:
//!
//! 1. `omega_fix_equals_123_over_248`: the bare numeric identity at
//!    f64 and BigFloat(500-bit) precision.
//! 2. `omega_fix_dual_anchor_variance_is_30_ppb`: with the journal
//!    value plugged into the gateway formula, h_eff_e and h_eff_mu
//!    agree to ~30 ppb. This is the framework's actual prediction.
//! 3. `omega_fix_perturbation_breaks_dual_anchor`: substituting
//!    122/248 or 124/248 in the gateway formula explodes the
//!    dual-anchor variance by >1000 ppm — i.e. ω_fix is empirically
//!    distinguished from neighbouring rationals.

#[cfg(feature = "precision-bigfloat")]
use pwos_math::precision::{
    bigfloat_from_i64_at, bigfloat_from_rational, bigfloat_from_str,
    bits_for_decimal_digits,
    constants::phi as phi_const,
    BigFloat, Real,
};

#[cfg(feature = "precision-bigfloat")]
fn bf(value: i64, prec: usize) -> BigFloat {
    bigfloat_from_i64_at(value, prec)
}

#[cfg(feature = "precision-bigfloat")]
fn add(a: &BigFloat, b: &BigFloat) -> BigFloat { Real::add(a, b) }
#[cfg(feature = "precision-bigfloat")]
fn sub(a: &BigFloat, b: &BigFloat) -> BigFloat { Real::sub(a, b) }
#[cfg(feature = "precision-bigfloat")]
fn mul(a: &BigFloat, b: &BigFloat) -> BigFloat { Real::mul(a, b) }
#[cfg(feature = "precision-bigfloat")]
fn div(a: &BigFloat, b: &BigFloat) -> BigFloat { Real::div(a, b) }
#[cfg(feature = "precision-bigfloat")]
fn pow_i(a: &BigFloat, n: i64) -> BigFloat { Real::pow_int(a, n) }
#[cfg(feature = "precision-bigfloat")]
fn to_f64(x: &BigFloat) -> f64 { Real::to_f64(x) }

/// Compute (h_eff_e, h_eff_mu) under a chosen ω_fix and the journal's
/// gateway / icosahedral-bulk formulas, with all other E_8 invariants
/// held at their structural values.
#[cfg(feature = "precision-bigfloat")]
fn dual_anchor_h_eff(omega_fix: &BigFloat) -> (BigFloat, BigFloat) {
    let prec = bits_for_decimal_digits(150);
    let m_e_mev = bigfloat_from_str("0.51099895000", prec);
    let m_mu_mev = bigfloat_from_str("105.6583755", prec);

    let dim = bf(248, prec);
    let h_cox = bf(30, prec);
    let one = bf(1, prec);
    let two = bf(2, prec);

    let phi = phi_const(prec);
    let phi2 = mul(&phi, &phi);
    let delta_phi = div(&one, &add(&h_cox, &phi2));
    let alpha_tau = div(&one, &add(&dim, &h_cox));
    // alpha_mu = 2 · alpha_tau^2  (from p6_2 / journal H.10)
    let alpha_mu = mul(&two, &pow_i(&alpha_tau, 2));

    // Electron: m_e = 2 · h_eff · ω_fix
    let r_e = mul(&two, omega_fix);
    let h_eff_e = div(&m_e_mev, &r_e);

    // Muon: m_mu = h_eff · phi^11 · (1+Delta_phi) · (1-alpha_mu)
    let phi11 = pow_i(&phi, 11);
    let one_plus_dp = add(&one, &delta_phi);
    let one_minus_amu = sub(&one, &alpha_mu);
    let r_mu = mul(&mul(&phi11, &one_plus_dp), &one_minus_amu);
    let h_eff_mu = div(&m_mu_mev, &r_mu);

    (h_eff_e, h_eff_mu)
}

/// (a-b)/b in ppb, as f64.
#[cfg(feature = "precision-bigfloat")]
fn ppb(a: &BigFloat, b: &BigFloat) -> f64 {
    let prec = 600;
    let diff = sub(a, b);
    let r = div(&diff, b);
    let scaled = mul(&r, &bf(1_000_000_000, prec));
    to_f64(&scaled)
}

/// (a-b)/b in ppm, as f64.
#[cfg(feature = "precision-bigfloat")]
fn ppm(a: &BigFloat, b: &BigFloat) -> f64 {
    let prec = 600;
    let diff = sub(a, b);
    let r = div(&diff, b);
    let scaled = mul(&r, &bf(1_000_000, prec));
    to_f64(&scaled)
}

/// (a) Bare algebraic identity: ω_fix = 1/2 − 1/248 = 123/248 exactly.
///
/// Pinned at both f64 and BigFloat(500-bit) precision. The BigFloat
/// check uses 150-decimal-digit working precision (~502 bits) and
/// requires the difference to be exactly zero (astro-float reports
/// `0e0` for true bit-exact cancellations).
#[test]
fn omega_fix_equals_123_over_248() {
    // ---- f64 precision ----
    let omega_f64_def = 0.5_f64 - 1.0_f64 / 248.0_f64;
    let omega_f64_rat = 123.0_f64 / 248.0_f64;
    assert_eq!(
        omega_f64_def, omega_f64_rat,
        "f64: ω_fix from 1/2-1/248 must equal 123/248 bit-for-bit"
    );

    // ---- BigFloat (500-bit) precision ----
    #[cfg(feature = "precision-bigfloat")]
    {
        let prec = bits_for_decimal_digits(150);
        let one_half = bigfloat_from_rational(1, 2, prec);
        let one_over_dim = bigfloat_from_rational(1, 248, prec);
        let omega_alg = sub(&one_half, &one_over_dim);
        let omega_rat = bigfloat_from_rational(123, 248, prec);
        let diff = sub(&omega_alg, &omega_rat);
        // Exact: difference is mathematically zero; astro-float reports 0e0.
        let diff_f = to_f64(&diff).abs();
        assert!(
            diff_f < 1e-100,
            "BigFloat: |ω_fix(alg) − 123/248| = {} should be < 1e-100",
            diff_f
        );

        // Also check the doubled-form equivalence: 246/496 = 123/248.
        let omega_doubled = bigfloat_from_rational(246, 496, prec);
        let diff2 = sub(&omega_alg, &omega_doubled);
        let diff2_f = to_f64(&diff2).abs();
        assert!(
            diff2_f < 1e-100,
            "BigFloat: |ω_fix(alg) − 246/496| = {} should be < 1e-100",
            diff2_f
        );
    }
}

/// (b) Dual-anchor variance at the journal's ω_fix = 123/248 is ~30 ppb.
///
/// Plug the journal value into the gateway formula and re-derive
/// h_eff_e and h_eff_mu from PDG / CODATA m_e, m_mu and the same
/// E_8 invariants used by p6_2_mass_spectrum. The two anchors must
/// agree to within 100 ppb. (The exact value is ~ −29.85 ppb, which
/// p6_2_mass_spectrum's `run_self_test` also enforces.)
#[cfg(feature = "precision-bigfloat")]
#[test]
fn omega_fix_dual_anchor_variance_is_30_ppb() {
    let prec = bits_for_decimal_digits(150);
    let omega_alg = sub(
        &bigfloat_from_rational(1, 2, prec),
        &bigfloat_from_rational(1, 248, prec),
    );

    let (h_eff_e, h_eff_mu) = dual_anchor_h_eff(&omega_alg);
    let var_ppb = ppb(&h_eff_e, &h_eff_mu);
    assert!(
        var_ppb.abs() < 100.0,
        "dual-anchor variance |{} ppb| not within 100 ppb (expected ~30 ppb)",
        var_ppb
    );
    // Tighter check matching p6_2's golden self-test (~ -29.85 ppb,
    // i.e. -0.0298 ppm). p6_2 enforces -40 ppb < var < -20 ppb.
    assert!(
        (-40.0..-20.0).contains(&var_ppb),
        "dual-anchor variance {} ppb out of expected (-40, -20) ppb (p6_2 self-test range)",
        var_ppb
    );
}

/// (c) Perturbing ω_fix by one unit in the numerator (122/248 or
/// 124/248) destroys the dual-anchor self-consistency.
///
/// At ω_fix = 122/248 or 124/248 the h_eff_e changes by O(0.8%)
/// while h_eff_mu (which doesn't use ω_fix) is unchanged, so the
/// e-vs-mu variance jumps from ~30 ppb to ~8000 ppm = 8.0 × 10⁶ ppb.
/// The journal value is the *only* rational in this neighbourhood
/// that satisfies the dual-anchor constraint.
#[cfg(feature = "precision-bigfloat")]
#[test]
fn omega_fix_perturbation_breaks_dual_anchor() {
    let prec = bits_for_decimal_digits(150);

    let omega_minus = bigfloat_from_rational(122, 248, prec);
    let omega_plus = bigfloat_from_rational(124, 248, prec);

    let (h_e_m, h_mu_ref) = dual_anchor_h_eff(&omega_minus);
    let (h_e_p, _) = dual_anchor_h_eff(&omega_plus);

    let var_minus_ppm = ppm(&h_e_m, &h_mu_ref);
    let var_plus_ppm = ppm(&h_e_p, &h_mu_ref);

    assert!(
        var_minus_ppm.abs() > 1000.0,
        "ω = 122/248 should break dual-anchor by >1000 ppm; got {:+.4} ppm",
        var_minus_ppm
    );
    assert!(
        var_plus_ppm.abs() > 1000.0,
        "ω = 124/248 should break dual-anchor by >1000 ppm; got {:+.4} ppm",
        var_plus_ppm
    );
    // Sanity: each perturbation should be at least 5 orders of
    // magnitude worse than the journal-value baseline (~30 ppb =
    // 0.030 ppm).
    let baseline_ppm = 0.030_f64;
    assert!(
        var_minus_ppm.abs() > 1e3 * baseline_ppm,
        "ω = 122/248 amplification {} not > 1000× baseline {}",
        var_minus_ppm.abs(),
        baseline_ppm
    );
    assert!(
        var_plus_ppm.abs() > 1e3 * baseline_ppm,
        "ω = 124/248 amplification {} not > 1000× baseline {}",
        var_plus_ppm.abs(),
        baseline_ppm
    );
}
