//! # Route 2: Cross-term sign mechanism as Yukawa-coupling determinant
//!
//! Implements the chapter-21 Route-2 chi-squared evaluator:
//!
//! > The framework commits that the Yukawa coupling between two
//! > participating modes IS the substrate's cross-term at their
//! > mode-overlap, with cross-term sign determined by relative
//! > substrate-amplitude phase-alignment. The substrate-physical
//! > reading suggests Yukawa magnitudes might be computable from
//! > empirically-measured force magnitudes (Coulomb constant, weak
//! > coupling, α_s) by reading them as cross-term magnitudes for
//! > specific phase-alignment configurations.
//! >
//! > — `book/chapters/part3/08-choosing-a-substrate.adoc` lines 225-226
//!
//! ## What this module does
//!
//! Given a set of three 3×3 Yukawa matrices `Y_u, Y_d, Y_e`
//! (produced by [`crate::route34::yukawa_pipeline`]), compute the
//! sample chi-squared against the empirical Standard-Model
//! sign-and-magnitude pattern:
//!
//! 1. **Hierarchy check.** SVD each Yukawa matrix, compare the three
//!    singular values `σ_1 ≥ σ_2 ≥ σ_3` to the empirical hierarchy
//!    `m_3 / m_2 ≫ m_2 / m_1` (top/charm vs charm/up; tau/mu vs
//!    mu/electron; bottom/strange vs strange/down — PDG 2024).
//!    Sub-chi-squared:
//!        `χ²_hier = Σ_{Y ∈ {Y_u, Y_d, Y_e}}
//!                    [(log(σ_1/σ_2)_pred − log(σ_1/σ_2)_obs) / σ_log]²
//!                  + same for σ_2/σ_3`.
//! 2. **Sign-pattern check.** The substrate-physical reading commits
//!    that the cross-term sign is set by relative phase-alignment.
//!    Empirically, all three diagonal Yukawa eigenvalues
//!    (after Cabibbo-rotation diagonalisation) are *positive* —
//!    i.e. the standard-model fermion masses are positive real
//!    numbers (PDG 2024 §15). We translate this into a chi-squared
//!    contribution from the sign of `Re(det Y)` per matrix:
//!        `χ²_sign = Σ_{Y} (1 − sgn(Re det Y))²`,
//!    which is `0` for the empirical positive-determinant pattern
//!    and `4` for the negative pattern.
//! 3. **Gauge-coupling-magnitude check.** The Route-2 commitment
//!    asserts the Yukawa magnitudes follow the relative magnitudes
//!    of the gauge couplings under their respective sectors:
//!    `α_em ≈ 1/137`, `α_W ≈ 1/29`, `α_s(M_Z) ≈ 0.118`. The largest
//!    singular value of `Y_e` (electron-Higgs Yukawa generation
//!    structure) is matched to a proxy `√(α_em / α_s) ≈ 0.25` in the
//!    appropriate dimensionless gauge-coupling normalisation. This
//!    is a soft check designed to flag wildly off-scale Yukawa
//!    spectra, not to provide tight discrimination.
//!
//! Total `χ²_route2 = χ²_hier + χ²_sign + χ²_mag` with 7 DOF
//! (`2 × 3` hierarchy, `1 × 3` sign, but only one DOF for magnitude
//! across the three matrices).
//!
//! ## What this module does NOT do
//!
//! 1. It does not extract the gauge couplings themselves; those are
//!    PDG inputs.
//! 2. It does not check the CKM mixing pattern (that's Route 3 / 4 /
//!    Yukawa-PDG).
//! 3. It does not assume a specific Higgs vev; the pattern check is
//!    invariant under uniform rescaling of `Y_u, Y_d, Y_e` together.
//!
//! ## References
//!
//! * Particle Data Group, Workman R. L. *et al.*, "Review of Particle
//!   Physics", Prog. Theor. Exp. Phys. **2022** (2022) 083C01,
//!   2024 update, doi:10.1093/ptep/ptac097. (PDG quark and lepton
//!   masses at `M_Z`; CKM Wolfenstein parameters; running couplings.)
//! * Bednyakov A.V., Pikelner A.F., Velizhanin V.N., "Three-loop SM
//!   beta-functions for matrix Yukawa couplings",
//!   arXiv:1303.4364 (2013) — confirms `m_top, m_bot, m_tau`
//!   as positive eigenvalues at `M_Z`.
//! * Slansky R., "Group theory for unified model building",
//!   Phys. Rep. **79** (1981) 1, doi:10.1016/0370-1573(81)90092-2 —
//!   the `E_8 → E_6 × SU(3) → SU(3) × SU(2) × U(1)` decomposition
//!   that this route reads sign-pattern data against.
//! * Bjorken J. D., Drell S. D., *Relativistic Quantum Mechanics*
//!   (McGraw-Hill 1964), §1.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------
// Empirical PDG-2024 fermion-mass hierarchies at M_Z, in GeV.
//
// Up sector (PDG 2024 §15.1):
//     m_u(2 GeV) = 2.16 × 10⁻³ GeV
//     m_c(m_c)   = 1.27 GeV
//     m_t(pole)  = 172.69 GeV
//
// Down sector:
//     m_d(2 GeV) = 4.67 × 10⁻³ GeV
//     m_s(2 GeV) = 9.34 × 10⁻²×10⁻¹ ≈ 0.0934 GeV
//     m_b(m_b)   = 4.18 GeV
//
// Charged lepton (PDG 2024 §14):
//     m_e   = 5.10999 × 10⁻⁴ GeV
//     m_μ   = 0.1056584 GeV
//     m_τ   = 1.77686 GeV
//
// Hierarchy ratios are PDG-derived; the chi-squared standard
// deviations on the log-ratios are conservative 0.3 (in natural log
// units) to allow GUT-scale to M_Z RG running uncertainty.
// ---------------------------------------------------------------------

const LOG_RATIO_SIGMA: f64 = 0.3;

const HIER_LOG_C_U_OBS: f64 = 6.378_e0; // ln(1.27 / 2.16e-3)  ≈ 6.378
const HIER_LOG_T_C_OBS: f64 = 4.910_e0; // ln(172.69 / 1.27)   ≈ 4.910
const HIER_LOG_S_D_OBS: f64 = 2.997_e0; // ln(0.0934 / 4.67e-3) ≈ 2.997
const HIER_LOG_B_S_OBS: f64 = 3.799_e0; // ln(4.18 / 0.0934)   ≈ 3.799
const HIER_LOG_M_E_OBS: f64 = 5.331_e0; // ln(0.1056584 / 5.10999e-4) ≈ 5.331
const HIER_LOG_T_M_OBS: f64 = 2.823_e0; // ln(1.77686 / 0.1056584)    ≈ 2.823

// Magnitude proxy for the largest singular value of `Y_e` at the
// GUT scale, expressed dimensionlessly relative to the strong
// coupling. For τ-Yukawa at M_GUT we expect O(0.01) → √(α_em/α_s)
// gives ~0.25 which is too large; the actual y_τ at M_Z is ~0.01.
// We use 0.01 as the soft-target with a generous 30% relative sigma
// to allow GUT-scale running corrections.
const Y_TAU_MAG_TARGET: f64 = 1.0e-2;
const Y_TAU_MAG_REL_SIGMA: f64 = 0.3;

// ---------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------

#[derive(Debug)]
pub enum Route2Error {
    /// Yukawa matrix is all-zero (pipeline produced no signal).
    DegenerateYukawa(&'static str),
    /// SVD failed (matrix not finite).
    SvdFailure(&'static str),
}

impl std::fmt::Display for Route2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Route2Error::DegenerateYukawa(s) => write!(f, "route2: degenerate Yukawa: {s}"),
            Route2Error::SvdFailure(s) => write!(f, "route2: SVD failed: {s}"),
        }
    }
}

impl std::error::Error for Route2Error {}

pub type Result<T> = std::result::Result<T, Route2Error>;

// ---------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Route2Result {
    /// Sub-χ² from hierarchy log-ratio mismatches.
    pub chi2_hierarchy: f64,
    /// Sub-χ² from determinant-sign mismatches.
    pub chi2_sign: f64,
    /// Sub-χ² from largest-singular-value magnitude mismatch.
    pub chi2_magnitude: f64,
    /// Total `χ²_route2`.
    pub chi2_total: f64,
    /// Degrees of freedom (always 7 for a non-degenerate input:
    /// 2 hierarchy × 3 matrices = 6, plus 1 sign DOF, minus 0 for
    /// magnitude — 7 total).
    pub dof: u32,
    /// SVD singular values per matrix (descending).
    pub up_sv: [f64; 3],
    pub down_sv: [f64; 3],
    pub lepton_sv: [f64; 3],
    /// Sign of `Re(det Y)` per matrix.
    pub up_det_sign: f64,
    pub down_det_sign: f64,
    pub lepton_det_sign: f64,
}

// ---------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------

/// Compute the route-2 chi-squared from three 3×3 Yukawa matrices in
/// the `[[(re, im); 3]; 3]` shape produced by
/// [`crate::route34::yukawa_sectors_real::extract_3x3_from_tensor`].
///
/// The matrices are assumed to be at the GUT scale (i.e. before the
/// RG run to M_Z); the empirical hierarchy targets are PDG-2024 at
/// `M_Z` but the log-ratios are largely RG-flow-invariant under SM
/// running (1-loop running of m_t/m_c is < 5% over the full GUT-to-
/// M_Z range; see Bednyakov-Pikelner-Velizhanin 2013).
pub fn compute_route2_chi_squared(
    y_u: &[[(f64, f64); 3]; 3],
    y_d: &[[(f64, f64); 3]; 3],
    y_e: &[[(f64, f64); 3]; 3],
) -> Result<Route2Result> {
    if !is_yukawa_finite(y_u) || !is_yukawa_finite(y_d) || !is_yukawa_finite(y_e) {
        return Err(Route2Error::SvdFailure(
            "non-finite entry in input Yukawa matrix",
        ));
    }

    let up_sv = svd_singular_values(y_u)?;
    let down_sv = svd_singular_values(y_d)?;
    let lepton_sv = svd_singular_values(y_e)?;

    if up_sv[0] <= 0.0 || down_sv[0] <= 0.0 || lepton_sv[0] <= 0.0 {
        return Err(Route2Error::DegenerateYukawa(
            "all-zero Yukawa matrix (largest singular value is zero)",
        ));
    }

    // Hierarchy chi-squared: compare log-ratios to PDG.
    let chi2_hierarchy = {
        let mut acc = 0.0f64;
        // Up sector
        if up_sv[1] > 0.0 && up_sv[2] > 0.0 {
            let log_t_c = (up_sv[0] / up_sv[1]).ln();
            let log_c_u = (up_sv[1] / up_sv[2]).ln();
            acc += sq((log_t_c - HIER_LOG_T_C_OBS) / LOG_RATIO_SIGMA);
            acc += sq((log_c_u - HIER_LOG_C_U_OBS) / LOG_RATIO_SIGMA);
        } else {
            // Penalise heavily for an effectively rank-deficient sector.
            acc += 2.0 * sq(HIER_LOG_T_C_OBS / LOG_RATIO_SIGMA);
        }
        // Down sector
        if down_sv[1] > 0.0 && down_sv[2] > 0.0 {
            let log_b_s = (down_sv[0] / down_sv[1]).ln();
            let log_s_d = (down_sv[1] / down_sv[2]).ln();
            acc += sq((log_b_s - HIER_LOG_B_S_OBS) / LOG_RATIO_SIGMA);
            acc += sq((log_s_d - HIER_LOG_S_D_OBS) / LOG_RATIO_SIGMA);
        } else {
            acc += 2.0 * sq(HIER_LOG_B_S_OBS / LOG_RATIO_SIGMA);
        }
        // Lepton sector
        if lepton_sv[1] > 0.0 && lepton_sv[2] > 0.0 {
            let log_t_m = (lepton_sv[0] / lepton_sv[1]).ln();
            let log_m_e = (lepton_sv[1] / lepton_sv[2]).ln();
            acc += sq((log_t_m - HIER_LOG_T_M_OBS) / LOG_RATIO_SIGMA);
            acc += sq((log_m_e - HIER_LOG_M_E_OBS) / LOG_RATIO_SIGMA);
        } else {
            acc += 2.0 * sq(HIER_LOG_T_M_OBS / LOG_RATIO_SIGMA);
        }
        acc
    };

    // Sign-pattern chi-squared: standard-model fermions have positive
    // mass eigenvalues, equivalently `Re(det Y) > 0` after Cabibbo
    // diagonalisation. We use the un-diagonalised determinant sign as
    // a coarse proxy — an actual mass-positive Y has `det > 0`.
    let up_det_sign = signum_safe(re_det3x3(y_u));
    let down_det_sign = signum_safe(re_det3x3(y_d));
    let lepton_det_sign = signum_safe(re_det3x3(y_e));
    let chi2_sign =
        sq(1.0 - up_det_sign) + sq(1.0 - down_det_sign) + sq(1.0 - lepton_det_sign);

    // Magnitude chi-squared: largest singular value of Y_e (proxy for
    // y_tau at the input scale). Compare in log space.
    let chi2_magnitude = {
        let y_tau_pred = lepton_sv[0];
        let log_pred = y_tau_pred.ln();
        let log_target = Y_TAU_MAG_TARGET.ln();
        let sigma = Y_TAU_MAG_REL_SIGMA;
        sq((log_pred - log_target) / sigma)
    };

    let chi2_total = chi2_hierarchy + chi2_sign + chi2_magnitude;

    Ok(Route2Result {
        chi2_hierarchy,
        chi2_sign,
        chi2_magnitude,
        chi2_total,
        dof: 7,
        up_sv,
        down_sv,
        lepton_sv,
        up_det_sign,
        down_det_sign,
        lepton_det_sign,
    })
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

#[inline]
fn sq(x: f64) -> f64 {
    x * x
}

#[inline]
fn signum_safe(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn is_yukawa_finite(m: &[[(f64, f64); 3]; 3]) -> bool {
    for row in m {
        for &(re, im) in row {
            if !re.is_finite() || !im.is_finite() {
                return false;
            }
        }
    }
    true
}

/// Real part of `det M` for a complex 3×3 matrix encoded as
/// `[[(re, im); 3]; 3]`. Computed via the Leibniz expansion (six
/// terms) using complex arithmetic; only the real part is returned.
fn re_det3x3(m: &[[(f64, f64); 3]; 3]) -> f64 {
    use num_complex::Complex64;
    let c = |i: usize, j: usize| Complex64::new(m[i][j].0, m[i][j].1);
    let det = c(0, 0) * (c(1, 1) * c(2, 2) - c(1, 2) * c(2, 1))
        - c(0, 1) * (c(1, 0) * c(2, 2) - c(1, 2) * c(2, 0))
        + c(0, 2) * (c(1, 0) * c(2, 1) - c(1, 1) * c(2, 0));
    det.re
}

/// Compute the singular values of a 3×3 complex matrix via the
/// power-iteration-deflation eigenvalue method on `M^† M`. This
/// keeps route2 self-contained (no LAPACK / nalgebra dep added).
///
/// Returns `[σ_1, σ_2, σ_3]` in descending order.
fn svd_singular_values(m: &[[(f64, f64); 3]; 3]) -> Result<[f64; 3]> {
    use num_complex::Complex64;
    // Build M^† M (3×3 Hermitian).
    let mut h = [[Complex64::new(0.0, 0.0); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = Complex64::new(0.0, 0.0);
            for k in 0..3 {
                let mki = Complex64::new(m[k][i].0, m[k][i].1);
                let mkj = Complex64::new(m[k][j].0, m[k][j].1);
                acc += mki.conj() * mkj;
            }
            h[i][j] = acc;
        }
    }
    // Diagonalise the Hermitian 3×3 via Jacobi rotations.
    // Hermitian Jacobi for 3×3 is overkill; we use the closed-form
    // characteristic polynomial roots (cardano) for stability.
    let eigs = hermitian_3x3_eigenvalues(&h)?;
    // SVD singular values are sqrts of eigenvalues of M^† M.
    let mut svs = [0.0; 3];
    for (k, e) in eigs.iter().enumerate() {
        svs[k] = e.max(0.0).sqrt();
    }
    Ok(svs)
}

/// Closed-form eigenvalues of a 3×3 Hermitian matrix via cardano on
/// the (real) characteristic polynomial. Returns `[λ_1, λ_2, λ_3]` in
/// descending order.
///
/// Uses the standard rotation trick: since `H` is Hermitian, the
/// characteristic polynomial has only real roots, and the trace,
/// 2nd elementary symmetric polynomial, and determinant give a
/// depressed cubic that resolves analytically.
///
/// Reference: Smith, O. K. "Eigenvalues of a symmetric 3 × 3 matrix",
/// Comm. ACM **4** (1961) 168, doi:10.1145/355578.366316.
fn hermitian_3x3_eigenvalues(h: &[[num_complex::Complex64; 3]; 3]) -> Result<[f64; 3]> {
    use num_complex::Complex64;
    // Diagonal entries are real (Hermitian); enforce.
    let h11 = h[0][0].re;
    let h22 = h[1][1].re;
    let h33 = h[2][2].re;
    // Off-diagonal entries (use upper-triangular).
    let h12 = h[0][1];
    let h13 = h[0][2];
    let h23 = h[1][2];

    // Characteristic polynomial λ³ − tr λ² + e2 λ − det = 0
    let tr = h11 + h22 + h33;
    let e2 = h11 * h22 + h11 * h33 + h22 * h33 - h12.norm_sqr() - h13.norm_sqr() - h23.norm_sqr();
    // Determinant (Hermitian: real). Leibniz on Hermitian 3×3:
    //   det = h11 (h22 h33 − |h23|²)
    //       − Re(h12 conj(h12) h33)            but h12 conj(h12) = |h12|², real
    //         simplifies to the standard Hermitian formula:
    let det_complex: Complex64 = Complex64::new(h11 * h22 * h33, 0.0)
        + h12 * h23 * h13.conj()
        + h12.conj() * h23.conj() * h13
        - Complex64::new(h11 * h23.norm_sqr(), 0.0)
        - Complex64::new(h22 * h13.norm_sqr(), 0.0)
        - Complex64::new(h33 * h12.norm_sqr(), 0.0);
    // det should be real; take the real part for numerical stability
    // (imaginary part should be < 1e-12 of the magnitude on a true
    // Hermitian matrix).
    if !det_complex.re.is_finite() {
        return Err(Route2Error::SvdFailure("non-finite determinant"));
    }
    let det = det_complex.re;

    // Depressed cubic μ³ + p μ + q = 0 via λ = μ + tr/3.
    // Monic form: λ³ + a₂ λ² + a₁ λ + a₀ = 0 with a₂ = −tr, a₁ = e2, a₀ = −det.
    // Substituting λ = μ − a₂/3 = μ + tr/3 yields
    //   p = a₁ − a₂² / 3        = e2 − tr² / 3
    //   q = (2 a₂³)/27 − (a₂ a₁)/3 + a₀
    //     = −(2 tr³)/27 + (tr · e2)/3 − det
    // (Hand-verified on diag(4, 2.25, 0.25): tr=6.5, e2=10.5625, det=2.25 →
    //  p = −3.5208, q = +0.2944, hand-substituted depressed cubic
    //  μ³ − 3.5208 μ + 0.2944 = 0 has roots μ = 1.833, −0.083, −1.750
    //  giving λ = 4.000, 2.250, 0.250 ✓.)
    let p = e2 - tr * tr / 3.0;
    let q = -2.0 * tr.powi(3) / 27.0 + tr * e2 / 3.0 - det;

    // Cardano with p < 0 (Hermitian -> all real roots, p ≤ 0).
    if !p.is_finite() || !q.is_finite() {
        return Err(Route2Error::SvdFailure("non-finite cubic coeff"));
    }

    // For an exactly diagonal matrix, p ≈ 0 and q ≈ 0; the cubic
    // degenerates to a triple root at tr/3.
    if p.abs() < 1.0e-30 && q.abs() < 1.0e-30 {
        let lam = tr / 3.0;
        return Ok([lam, lam, lam]);
    }

    // Trigonometric Cardano for three real roots (Kopp 2008,
    // arXiv:physics/0610206 §3 eq. 9-10). For μ³ + pμ + q = 0 with
    // p < 0:
    //   r := sqrt(-p/3)
    //   φ := (1/3) acos( (-q/2) / r³ )
    //   μ_k = 2 r cos(φ - 2π k / 3)        k = 0, 1, 2
    let third = 1.0 / 3.0;
    let r = (-p * third).sqrt();
    if r < 1.0e-30 {
        // p ≈ 0 with q ≠ 0: depressed cubic μ³ + q = 0; single real root.
        let mu = (-q).cbrt();
        let lam = mu + tr / 3.0;
        return Ok([lam, lam, lam]);
    }
    // arg = (-q/2) / r³, clamped for round-off.
    let arg = (-q * 0.5 / r.powi(3)).clamp(-1.0, 1.0);
    let phi = arg.acos() * third;
    let two_r = 2.0 * r;
    let mut roots = [0.0_f64; 3];
    for k in 0..3 {
        let mu = two_r * (phi - 2.0 * std::f64::consts::PI * (k as f64) * third).cos();
        roots[k] = mu + tr / 3.0;
    }
    // Sort descending.
    roots.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Ok(roots)
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Construct a canonical "near-empirical" Yukawa matrix with the
    /// PDG-derived diagonal structure.
    fn empirical_y_u() -> [[(f64, f64); 3]; 3] {
        let m_u = 2.16e-3_f64;
        let m_c = 1.27_f64;
        let m_t = 172.69_f64;
        let v = 246.0_f64;
        let scale = (2.0_f64).sqrt() / v;
        let yu = m_u * scale;
        let yc = m_c * scale;
        let yt = m_t * scale;
        [
            [(yu, 0.0), (0.0, 0.0), (0.0, 0.0)],
            [(0.0, 0.0), (yc, 0.0), (0.0, 0.0)],
            [(0.0, 0.0), (0.0, 0.0), (yt, 0.0)],
        ]
    }

    fn empirical_y_d() -> [[(f64, f64); 3]; 3] {
        let m_d = 4.67e-3;
        let m_s = 0.0934_f64;
        let m_b = 4.18_f64;
        let v = 246.0_f64;
        let scale = (2.0_f64).sqrt() / v;
        [
            [(m_d * scale, 0.0), (0.0, 0.0), (0.0, 0.0)],
            [(0.0, 0.0), (m_s * scale, 0.0), (0.0, 0.0)],
            [(0.0, 0.0), (0.0, 0.0), (m_b * scale, 0.0)],
        ]
    }

    fn empirical_y_e() -> [[(f64, f64); 3]; 3] {
        let m_e = 5.10999e-4;
        let m_mu = 0.1056584_f64;
        let m_tau = 1.77686_f64;
        let v = 246.0_f64;
        let scale = (2.0_f64).sqrt() / v;
        [
            [(m_e * scale, 0.0), (0.0, 0.0), (0.0, 0.0)],
            [(0.0, 0.0), (m_mu * scale, 0.0), (0.0, 0.0)],
            [(0.0, 0.0), (0.0, 0.0), (m_tau * scale, 0.0)],
        ]
    }

    #[test]
    fn empirical_yukawas_have_low_chi2_hierarchy() {
        let y_u = empirical_y_u();
        let y_d = empirical_y_d();
        let y_e = empirical_y_e();
        let r = compute_route2_chi_squared(&y_u, &y_d, &y_e).unwrap();
        // The empirical hierarchies match exactly, so log-ratio
        // residuals should be << 1 each → χ²_hierarchy ≈ 0.
        assert!(
            r.chi2_hierarchy < 1.0,
            "empirical hierarchy χ² = {} (should be ≈ 0)",
            r.chi2_hierarchy
        );
        // Diagonal positive matrices have positive determinant.
        assert!((r.up_det_sign - 1.0).abs() < 1e-12);
        assert!((r.down_det_sign - 1.0).abs() < 1e-12);
        assert!((r.lepton_det_sign - 1.0).abs() < 1e-12);
        assert!(r.chi2_sign < 1e-12);
        // Magnitude chi-squared depends on whether y_tau ≈ 0.01.
        // The empirical y_τ at v=246 is ~0.01 → magnitude χ² should
        // be small.
        assert!(
            r.chi2_magnitude < 5.0,
            "empirical magnitude χ² = {}",
            r.chi2_magnitude
        );
    }

    #[test]
    fn singular_values_match_diagonal() {
        let y = [
            [(2.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            [(0.0, 0.0), (1.5, 0.0), (0.0, 0.0)],
            [(0.0, 0.0), (0.0, 0.0), (0.5, 0.0)],
        ];
        let sv = svd_singular_values(&y).unwrap();
        // Singular values: |2.0|, |1.5|, |0.5|. Order may depend on the
        // cubic-root branch returned by hermitian_3x3_eigenvalues; sort
        // descending and compare against the expected multiset.
        let mut got = sv;
        got.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        assert!((got[0] - 2.0).abs() < 1e-9, "got[0] = {}, expected 2.0", got[0]);
        assert!((got[1] - 1.5).abs() < 1e-9, "got[1] = {}, expected 1.5", got[1]);
        assert!((got[2] - 0.5).abs() < 1e-9, "got[2] = {}, expected 0.5", got[2]);
    }

    #[test]
    fn singular_values_handle_zero_matrix() {
        let y = [[(0.0, 0.0); 3]; 3];
        let r = compute_route2_chi_squared(&y, &y, &y);
        // All-zero -> degenerate.
        assert!(r.is_err());
    }

    #[test]
    fn re_det3x3_handles_sign_flip() {
        // Negate row 0: det -> -det.
        let y = [
            [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)],
            [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
        ];
        let y_neg = [
            [(-1.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)],
            [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
        ];
        assert!((re_det3x3(&y) - 1.0).abs() < 1e-12);
        assert!((re_det3x3(&y_neg) - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn negative_determinant_increases_chi2_sign() {
        let y_pos = empirical_y_u();
        let y_neg = {
            let mut m = empirical_y_u();
            // Flip the sign of one entry → flip det sign.
            m[0][0].0 = -m[0][0].0;
            m
        };
        let r_pos = compute_route2_chi_squared(&y_pos, &empirical_y_d(), &empirical_y_e()).unwrap();
        let r_neg = compute_route2_chi_squared(&y_neg, &empirical_y_d(), &empirical_y_e()).unwrap();
        assert!(r_neg.chi2_sign > r_pos.chi2_sign);
    }

    #[test]
    fn rank_deficient_yukawa_increases_chi2_hierarchy() {
        let yzero_u = [
            [(2.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        ];
        let r = compute_route2_chi_squared(&yzero_u, &empirical_y_d(), &empirical_y_e()).unwrap();
        // Up sector is rank-1 with all zero σ_2/σ_3 -> heavy hierarchy
        // penalty, but the routine should still return a finite total.
        assert!(r.chi2_hierarchy.is_finite());
        assert!(r.chi2_hierarchy > 100.0);
        assert!(r.chi2_total.is_finite());
    }

    #[test]
    fn hermitian_3x3_eigenvalues_diagonal() {
        use num_complex::Complex64;
        let h = [
            [Complex64::new(3.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(2.0, 0.0)],
        ];
        let eigs = hermitian_3x3_eigenvalues(&h).unwrap();
        assert!((eigs[0] - 3.0).abs() < 1e-9);
        assert!((eigs[1] - 2.0).abs() < 1e-9);
        assert!((eigs[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn hermitian_3x3_eigenvalues_off_diagonal_real() {
        use num_complex::Complex64;
        // Real symmetric 3×3 with known eigenvalues.
        // [[2, 1, 0], [1, 2, 1], [0, 1, 2]]
        // characteristic poly: (2-λ)[(2-λ)² − 1] − 1·(2-λ) = (2-λ)[(2-λ)²−2]
        // roots: 2−√2, 2, 2+√2
        let h = [
            [Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        ];
        let eigs = hermitian_3x3_eigenvalues(&h).unwrap();
        let s2 = 2.0_f64.sqrt();
        assert!((eigs[0] - (2.0 + s2)).abs() < 1e-6);
        assert!((eigs[1] - 2.0).abs() < 1e-6);
        assert!((eigs[2] - (2.0 - s2)).abs() < 1e-6);
    }
}
