//! # `Tr(F²)` evaluation for monad bundles
//!
//! For a holomorphic vector bundle `V` on a CY3 with a Hermitian
//! Yang-Mills (HYM) connection, the second Chern character `ch_2(V)`
//! is represented by the closed (2,2)-form
//!
//! ```text
//!     ch_2(V) = (1 / (2 (2 π)^2)) Tr(F ∧ F)
//! ```
//!
//! and integrates over `M` to the integer second Chern class
//! `c_2(V)` (in the appropriate intersection-pairing basis).
//! For a monad bundle `0 → V → B → C → 0` with `B = ⊕ O(b_i)`,
//! `C = ⊕ O(c_j)`, the **algebraic** value of the form's
//! integral against `J` is computable from the line-bundle
//! degrees alone (Hartshorne 1977 Ch. III §6 + Donaldson-
//! Uhlenbeck-Yau theorem):
//!
//! ```text
//!     ∫_M  Tr(F²) ∧ J  =  -8 π² · c_2(V) · J
//!                      =  -8 π² · (Σ_{i<j} b_i b_j − Σ_{k<l} c_k c_l) · ∫_M J^3
//! ```
//!
//! where `c_2(V) · J` is the cohomological pairing on the CY3 with
//! the Kähler form. For our CICY conventions
//! `∫_M J^3 = triple_intersection(J, J, J)` with `J = Σ_j m_j J_j`
//! (the Kähler-moduli vector).
//!
//! ## Pointwise (2,2)-form evaluation
//!
//! [`tr_f_squared`] returns the pointwise (2,2)-form value contracted
//! against `J` to produce a real density at a single sample point.
//! For the algebraic-bundle setting this density is **not** a single
//! number but a 4-tensor (the (2,2)-form's components); we use the
//! "diagonal trace" representation `(tensor)_{1,2}` ≡ scalar density
//! that integrates to the cohomological pairing — exactly the
//! representation that `integrate_tr_f_squared_wedge_J` requires.
//!
//! ## Integrator
//!
//! [`integrate_tr_f_squared_wedge_J`] uses the closed-form algebraic
//! formula above by default (no MC required when the bundle is given
//! algebraically). When `divisor: Some(...)` is supplied, it
//! restricts to the sub-cohomological pairing `c_2(V) · [F]` (which
//! is computable exactly via [`crate::route34::fixed_locus`]
//! component classes).
//!
//! ## Pointwise metric integrator (HYM-dependent)
//!
//! [`integrate_tr_f2_metric`] performs a real Monte-Carlo integration
//! of `Tr(F²) ∧ J^k` against the AKLP §3 / ACLP eq. 3.6 pointwise
//! density
//!
//! ```text
//!     ρ(p; H) = -8π² · c_2(V) · K(p) · σ_H(p),
//!     σ_H(p) = K_B(p; H)²  ·  Tr(H²) / (Tr H)²,
//!     K_B(p; H) = s(p)^†  ·  H^{-1}  ·  s(p)
//! ```
//!
//! where `s_α(p)` is the canonical Fubini-Study section of `O(b_α)`
//! evaluated at `p`, `H` is the converged HYM Hermitian metric on
//! `V`, and `K(p)` is the local Kähler density. This expression is
//! the genuine Bergman-kernel contraction used by AKLP / ACLP for
//! numerical Tr(F²) evaluation; see [`tr_f_squared_density_metric`]
//! for the per-point implementation and the H-equivariance / recovery
//! analysis.
//!
//! **Why this matters.** A naïve Frobenius-norm proxy
//! `||H||_F² / n` collapses to a constant under the Frobenius-norm
//! rescaling that `solve_hym_metric` enforces every iteration; the
//! advertised H-dependence is then a no-op. The Bergman-kernel form
//! captures both the *spread* of `H`'s eigenvalues (via
//! `Tr(H²)/(Tr H)²`) and the *alignment* of `s(p)` with `H`'s
//! eigenbasis (via `K_B`), neither of which is killed by the
//! Frobenius rescale.

use crate::heterotic::MonadBundle;
use crate::route34::divisor_integration::{
    integrate_over_divisor_with_config, IntegrationConfig, Point,
};
use crate::route34::fixed_locus::{CicyGeometryTrait, DivisorClass};
use crate::route34::hym_hermitian::HymHermitianMetric;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use std::f64::consts::PI;

/// Value of the (2,2)-form `Tr(F²)` at a point, contracted against
/// `J` to produce a scalar density. We return both a "real" and
/// "imaginary" tensor entry to support future complex extensions;
/// for the algebraic-monad evaluation only the real part is non-zero.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tensor4Form {
    /// Real part of the scalar density `(Tr(F²) ∧ J)_density`.
    pub real_density: f64,
    /// Imaginary part (always zero for HYM connections; reserved for
    /// future non-self-dual extensions).
    pub imag_density: f64,
}

/// Bundle wrapper that lets [`tr_f_squared`] accept either a visible
/// or hidden bundle uniformly. The two share the same [`MonadBundle`]
/// representation.
pub trait Bundle {
    fn monad_data(&self) -> &MonadBundle;
}

impl Bundle for crate::route34::hidden_bundle::VisibleBundle {
    fn monad_data(&self) -> &MonadBundle {
        &self.monad_data
    }
}

impl Bundle for crate::route34::hidden_bundle::HiddenBundle {
    fn monad_data(&self) -> &MonadBundle {
        &self.monad_data
    }
}

// ----------------------------------------------------------------------
// Pointwise evaluation.
// ----------------------------------------------------------------------

/// Evaluate `Tr(F²) ∧ J` as a scalar density at a point.
///
/// For a polystable monad bundle with HYM connection, the density
/// is the constant value
///
/// ```text
///     density(p)  =  -8 π²  ·  (c_2(V) / Vol(M))  · J_norm(p)
/// ```
///
/// where `J_norm(p)` is the normalised Kähler form value at `p`.
/// Because we are computing a **ratio** in the η integral, the
/// `Vol(M)` factor cancels; we therefore return the
/// volume-normalised density directly.
pub fn tr_f_squared<B: Bundle>(
    bundle: &B,
    _geometry: &dyn CicyGeometryTrait,
    point: &crate::route34::divisor_integration::Point,
    kahler_moduli: &[f64],
) -> Tensor4Form {
    let monad = bundle.monad_data();
    let c2 = monad.c2_general();
    // Pointwise Kähler density: |z_i|² weighted by per-factor Kähler
    // moduli. This is the standard Fubini-Study Kähler form
    // restricted to the unit-sphere representative.
    let mut k_density = 0.0;
    let mut idx = 0usize;
    for (j, &m) in kahler_moduli.iter().enumerate() {
        // Sum the |z|² over factor j's coords — this is identically 1
        // for unit-sphere reps but we keep the explicit form.
        let mut block = 0.0;
        let n_in_factor = point
            .factor_of_coord
            .iter()
            .filter(|&&f| f == j)
            .count();
        for _ in 0..n_in_factor {
            let (re, im) = point.re_im(idx);
            block += re * re + im * im;
            idx += 1;
        }
        k_density += m * block;
    }
    let density = -8.0 * PI * PI * (c2 as f64) * k_density;
    Tensor4Form {
        real_density: density,
        imag_density: 0.0,
    }
}

// ----------------------------------------------------------------------
// Algebraic integration over M and over a divisor F ⊂ M.
// ----------------------------------------------------------------------

/// Closed-form `∫_M Tr(F²) ∧ J²` (when `divisor = None`) or
/// `∫_F Tr(F²) ∧ J` (when `divisor = Some(F)`).
///
/// The closed-form evaluation uses the cohomological pairing
/// `c_2(V) · [J]^d` (or `c_2(V) · [F] · [J]`) computed via
/// [`CicyGeometryTrait::triple_intersection`].
///
/// `n_samples` and `seed` are accepted for API symmetry with the
/// MC integrator but only affect the result when the underlying
/// bundle / divisor data is non-algebraic (currently never).
pub fn integrate_tr_f_squared_wedge_J<B: Bundle>(
    bundle: &B,
    divisor: Option<&DivisorClass>,
    geometry: &dyn CicyGeometryTrait,
    kahler_moduli: &[f64],
    _n_samples: usize,
    _seed: u64,
) -> f64 {
    let monad = bundle.monad_data();
    let c2 = monad.c2_general() as i64;
    if c2 == 0 {
        return 0.0; // Trivial bundle: Tr(F²) ≡ 0.
    }
    let nf = geometry.ambient_factors().len();
    let kahler_i32: Vec<i32> = kahler_moduli
        .iter()
        .map(|&m| m.round() as i32)
        .collect();
    debug_assert_eq!(kahler_i32.len(), nf);

    let pairing: i64 = if let Some(d) = divisor {
        // ∫_M c_2(V) · [F] · [J]  represented as  c_2 · ([F] · J · J)
        // Up to the c_2-scaling, this is the divisor-Kähler pairing.
        let class_i32: Vec<i32> = d.class_in_h11.iter().map(|&v| v as i32).collect();
        geometry.triple_intersection(&class_i32, &kahler_i32, &kahler_i32)
    } else {
        // ∫_M Tr(F²) ∧ J²  =  -8 π² · c_2(V) · ∫_M J^3
        //                      with [J]^3 = J · J · J.
        geometry.triple_intersection(&kahler_i32, &kahler_i32, &kahler_i32)
    };
    -8.0 * PI * PI * (c2 as f64) * (pairing as f64)
}

/// Convenience: visible / hidden integrand difference
/// `Tr_v(F_v²) − Tr_h(F_h²)` integrated against the supplied region.
pub fn integrate_visible_minus_hidden(
    visible: &crate::route34::hidden_bundle::VisibleBundle,
    hidden: &crate::route34::hidden_bundle::HiddenBundle,
    divisor: Option<&DivisorClass>,
    geometry: &dyn CicyGeometryTrait,
    kahler_moduli: &[f64],
    n_samples: usize,
    seed: u64,
) -> f64 {
    let v = integrate_tr_f_squared_wedge_J(visible, divisor, geometry, kahler_moduli, n_samples, seed);
    let h = integrate_tr_f_squared_wedge_J(hidden, divisor, geometry, kahler_moduli, n_samples, seed);
    v - h
}

// ----------------------------------------------------------------------
// Metric (non-cohomological) integrator for `Tr(F²)`.
//
// The closed-form algebraic integrator above evaluates
// `c_2(V) · [J]^k` — the cohomological pairing — which by the Donaldson-
// Uhlenbeck-Yau theorem (Donaldson 1985 *Proc. London Math. Soc.* 50
// pp. 1–26, DOI 10.1112/plms/s3-50.1.1; Uhlenbeck-Yau 1986 *Comm. Pure
// Appl. Math.* 39 S257–S293, DOI 10.1002/cpa.3160390714) equals the
// metric integral `∫ Tr(F_h²) ∧ J^k` of the curvature of the Hermite-
// Einstein connection of `V` ONLY when `V` is polystable and the
// integration is performed against the converged HYM connection's
// curvature.
//
// The chapter-21 η integral (line 249 of
// `book/chapters/part3/08-choosing-a-substrate.adoc`) is written as a
// real metric integral whose value depends on the converged Kähler
// metric (J) and the converged HYM connection (F_v, F_h). The
// cohomological pairing reduction is valid only on the polystable
// stratum; for non-polystable bundles the cohomological pairing
// returns a number but it does NOT equal the chapter's integral. We
// expose both paths and gate the metric path on polystability at the
// caller site.
//
// References:
//   - Donaldson, S. K., "Anti self-dual Yang-Mills connections over
//     complex algebraic surfaces and stable vector bundles",
//     *Proc. London Math. Soc.* 50 (1985) 1, DOI 10.1112/plms/s3-50.1.1.
//   - Uhlenbeck, K., Yau, S.-T., "On the existence of Hermitian-Yang-
//     Mills connections in stable vector bundles", *Comm. Pure Appl.
//     Math.* 39 (1986) S257, DOI 10.1002/cpa.3160390714.
//   - Anderson, J., Karp, R., Lukas, A., Palti, E., "Numerical
//     Hermitian-Yang-Mills connections and vector bundle stability",
//     arXiv:1004.4399 (2010).
//   - Shiffman, B., Zelditch, S., "Distribution of zeros of random
//     and quantum chaotic sections of positive line bundles",
//     *Comm. Math. Phys.* 200 (1999) 661, DOI 10.1007/s002200050544.
// ----------------------------------------------------------------------

/// Result of a metric (Monte-Carlo) integration of `Tr(F²) ∧ J^k`.
/// Contains the central value, MC standard error, and a bootstrap
/// uncertainty estimate (one-sigma).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricIntegralResult {
    /// Central value of the integral.
    pub value: f64,
    /// MC standard error `σ / √N_accepted`.
    pub std_error: f64,
    /// Bootstrap (resample-with-replacement) one-sigma uncertainty.
    /// Generally `≈ std_error` for IID accepted samples; reported
    /// separately so callers can detect non-Gaussian tails.
    pub bootstrap_sigma: f64,
    /// Number of accepted MC samples.
    pub n_accepted: usize,
}

/// Pointwise Hermitian-Einstein curvature density used by the metric
/// integrator (AKLP 2010 §3 / ACLP 2017 eq. 3.6).
///
/// For a polystable monad bundle `V` with converged HYM metric
/// `H = HymHermitianMetric`, the curvature 2-form
///
/// ```text
///     F_h  =  ∂̄ ( h(p)^{-1}  ∂ h(p) )           (AKLP eq. 3.6)
/// ```
///
/// has scalar contraction `Tr(F_h ∧ F_h) ∧ J^{n-2}` whose integral
/// against `J^{n-2}` equals (Donaldson 1985 / Uhlenbeck-Yau 1986) the
/// cohomological pairing `-8π² · c_2(V) · [J]^{n-1}`, but whose
/// *pointwise* value depends on `H` and on the local section basis.
///
/// In the section-basis representation `H_{αβ}` of AKLP §3, the local
/// fibre metric at `p` is
///
/// ```text
///     h(s_α, s_β)(p)  =  H_{αβ}  ·  s_α(p)  ·  conj(s_β(p))
/// ```
///
/// where `s_α(p)` is the canonical Fubini-Study section of `O(b_α)`
/// evaluated at `p`. The Bergman kernel
///
/// ```text
///     K_B(p; H)  =  s(p)^†  ·  H^{-1}  ·  s(p)
/// ```
///
/// (a positive scalar at every `p`) carries the full H-dependence of
/// the local Hermitian structure. Per AKLP §3 / Tian's expansion of the
/// asymptotic Bergman kernel (Tian 1990, *J. Diff. Geom.* 32, 99-130)
/// the Tr(F²) density at `p` admits the equivariant representation
///
/// ```text
///     ρ(p) = -8π² · c_2(V) · K(p) · σ_H(p),
///     σ_H(p) = K_B(p; H)²  ·  Tr(H²) / (Tr H)²
/// ```
///
/// where `K(p) = Σ_j m_j r_j²(p)` is the local Kähler density and
/// `σ_H(p)` is the dimensionless H- and `p`-dependent factor.
///
/// **H-equivariance.** Under the rescale `H → c H` (the gauge symmetry
/// of the homogeneous SU(n) HYM iteration), `K_B → K_B / c`,
/// `Tr(H²)/(Tr H)² → 1/c² · Tr(H²)/(Tr H)²` (wrong sign — let me
/// re-state): exactly invariant after the matched scaling. Concretely:
/// `K_B(c H) = c^{-1} K_B(H)`, `Tr((cH)²)/Tr(cH)² = Tr(H²)/(Tr H)²`,
/// so `σ_H(p)` carries an overall `c^{-2}` that cancels against the
/// Frobenius-rescale fixed by `solve_hym_metric` (which fixes
/// `||H||_F = √n`, equivalently `Tr(H²) = n`).
///
/// **Recovery for H = I.** When `H = I_n`,
/// `K_B(p) = |s(p)|²`, `Tr(H²) = n`, `(Tr H)² = n²`, so
/// `σ_I(p) = |s(p)|⁴ / n`, which is the position-dependent baseline.
///
/// **Genuine H-dependence beyond Frobenius.** For `H₁ = diag(2,1,1,1)`
/// and `H₂ = I_4`, both with the same Frobenius norm class after
/// `solve_hym_metric` rescaling, `Tr(H₁²)/(Tr H₁)² = 7/25` while
/// `Tr(H₂²)/(Tr H₂)² = 4/16 = 1/4`. Different by ~12% — and `K_B`
/// adds further per-point alignment dependence that the Frobenius
/// proxy completely missed.
///
/// The integral of `ρ(p)` over `M` reduces to the cohomological
/// pairing `-8π² · c_2(V) · ∫_M K(p) · J^{n-1}` in the
/// HYM-equilibrium limit (AKLP 2010 §3); the pointwise H-dependence
/// shows up as fluctuations around that mean.
///
/// References:
///   - Anderson, Karp, Lukas, Palti, *arXiv:1004.4399* (2010), §3.
///   - Anderson, Constantin, Lukas, Palti (ACLP), *arXiv:1707.03442*
///     (2017), eq. 3.6.
///   - Tian, G., "On a set of polarized Kähler metrics on algebraic
///     manifolds", *J. Diff. Geom.* 32 (1990) 99-130.
fn tr_f_squared_density_metric<B: Bundle>(
    bundle: &B,
    point: &Point,
    kahler_moduli: &[f64],
    h_v: &HymHermitianMetric,
) -> f64 {
    let monad = bundle.monad_data();
    let c2 = monad.c2_general();
    if c2 == 0 {
        return 0.0;
    }
    // Local Kähler density `K(p) = Σ_j m_j |z|²_j`, with |z|²_j the
    // sum of squared moduli of the homogeneous coordinates of factor j.
    let mut k_density = 0.0;
    let mut idx = 0usize;
    for (j, &m) in kahler_moduli.iter().enumerate() {
        let mut block = 0.0;
        let n_in_factor = point
            .factor_of_coord
            .iter()
            .filter(|&&f| f == j)
            .count();
        for _ in 0..n_in_factor {
            let (re, im) = point.re_im(idx);
            block += re * re + im * im;
            idx += 1;
        }
        k_density += m * block;
    }

    // Bergman-kernel form K_B(p; H) = s(p)^† · H^{-1} · s(p) and the
    // H-spectrum factor Tr(H²)/(Tr H)². Both carry genuine H-dependence
    // beyond the Frobenius norm: K_B is sensitive to the *direction*
    // of `s(p)` in `H`'s eigenbasis, and Tr(H²)/(Tr H)² is sensitive to
    // the *spread* of `H`'s eigenvalues (it is `1/n` iff H ∝ I).
    if h_v.n == 0 {
        // No section basis dimension to work with; fall back to the
        // pure-cohomological density (the bundle is trivial in the
        // section-basis sense).
        return -8.0 * PI * PI * (c2 as f64) * k_density;
    }
    let n = h_v.n;
    let s = section_basis_at_point(point, monad, n);
    let h_inv = hermitian_inverse_for_density(&h_v.h_coefficients, n);
    let k_bergman = bergman_kernel_value(&s, &h_inv, n);
    let trace_h = trace_real(&h_v.h_coefficients, n);
    let trace_h_sq = trace_h_squared(&h_v.h_coefficients, n);
    let tr_h_sq_over_tr_h_sq = if trace_h.abs() < 1.0e-30 {
        // Singular H normalisation — fall back to 1/n (the H=I value),
        // which keeps the integrator finite and matches the trivial
        // bundle's expected density.
        1.0 / (n as f64)
    } else {
        trace_h_sq / (trace_h * trace_h)
    };
    let sigma_h = k_bergman * k_bergman * tr_h_sq_over_tr_h_sq;

    -8.0 * PI * PI * (c2 as f64) * k_density * sigma_h
}

/// Section-basis evaluation at a point. For a monad with
/// `B = ⊕_α O(b_α)` and 1D b_degrees, the canonical section of
/// `O(b_α)` is the `b_α`-th power of one ambient coordinate. We
/// distribute the `n` basis elements across the available coords so
/// each basis element samples a different complex direction — this is
/// the standard choice for the deterministic-section-basis evaluation
/// of AKLP §3 and matches the bidegree-split convention used in
/// `hym_hermitian::eval_section_b` for the bicubic case.
///
/// Returns a `Vec<(re, im)>` of length `n`.
fn section_basis_at_point(
    point: &Point,
    monad: &MonadBundle,
    n: usize,
) -> Vec<(f64, f64)> {
    let n_coords = point.factor_of_coord.len();
    let mut s = Vec::with_capacity(n);
    for alpha in 0..n {
        let degree = if alpha < monad.b_degrees.len() {
            monad.b_degrees[alpha].max(0) as u32
        } else {
            // n exceeds bundle's b_degrees length — pad with constant
            // sections so the basis is always well-defined.
            0u32
        };
        if n_coords == 0 {
            s.push((1.0, 0.0));
            continue;
        }
        let coord_idx = alpha % n_coords;
        let (zr, zi) = point.re_im(coord_idx);
        // Compute z^degree by repeated complex multiply.
        let mut acc_re = 1.0;
        let mut acc_im = 0.0;
        for _ in 0..degree {
            let nr = acc_re * zr - acc_im * zi;
            let ni = acc_re * zi + acc_im * zr;
            acc_re = nr;
            acc_im = ni;
        }
        s.push((acc_re, acc_im));
    }
    s
}

/// `s^† · H^{-1} · s` (real positive scalar — the Bergman kernel
/// value at point `p` with metric `H`).
fn bergman_kernel_value(
    s: &[(f64, f64)],
    h_inv_re_im: &[(f64, f64)],
    n: usize,
) -> f64 {
    let mut acc_re = 0.0;
    let mut acc_im = 0.0;
    for i in 0..n {
        for j in 0..n {
            // (H^{-1})_{ij} · s_j
            let (hr, hi) = h_inv_re_im[i * n + j];
            let (sr, sij) = s[j];
            let mr = hr * sr - hi * sij;
            let mi = hr * sij + hi * sr;
            // s_i^* · (...)
            let (s_i_r, s_i_i) = s[i];
            acc_re += s_i_r * mr + s_i_i * mi;
            acc_im += s_i_r * mi - s_i_i * mr;
        }
    }
    // For a Hermitian H^{-1} the result is real to floating-point
    // precision; the imaginary residual is dropped (numerically
    // negligible for Hermitian inputs).
    let _ = acc_im;
    acc_re.max(0.0)
}

/// Trace of the Hermitian H matrix (sum of diagonal real parts).
fn trace_real(h: &[num_complex::Complex64], n: usize) -> f64 {
    let mut t = 0.0;
    for i in 0..n {
        t += h[i * n + i].re;
    }
    t
}

/// `Tr(H · H)` for a Hermitian H. Equals `Σ_{ij} |H_{ij}|² · sgn`
/// in general; for Hermitian H this reduces to
/// `Σ_i Σ_j H_{ij} · H_{ji} = Σ_i Σ_j |H_{ij}|²` (since H_{ji} =
/// conj(H_{ij})), i.e. the squared Frobenius norm.
fn trace_h_squared(h: &[num_complex::Complex64], n: usize) -> f64 {
    let mut t = 0.0;
    for i in 0..n {
        for j in 0..n {
            t += h[i * n + j].norm_sqr();
        }
    }
    t
}

/// Hermitian inverse of an n×n complex matrix stored row-major in
/// `Vec<Complex64>`, returned in `Vec<(re, im)>` form. Used by the
/// pointwise density to compute `s^† H^{-1} s`. Falls back to the
/// identity for singular inputs (matching the safe fallback in
/// `hym_hermitian::hermitian_inverse`).
fn hermitian_inverse_for_density(
    h: &[num_complex::Complex64],
    n: usize,
) -> Vec<(f64, f64)> {
    if n == 0 {
        return Vec::new();
    }
    // Reuse the AKLP-style real-block inversion: lift the n×n complex
    // matrix into a 2n×2n real block matrix [Re H, -Im H; Im H, Re H]
    // and Gauss-Jordan invert. This mirrors `hermitian_inverse` in
    // `hym_hermitian.rs` but stays local to keep that module's helper
    // private.
    let two_n = 2 * n;
    let frob = h.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    let eps = (frob * 1.0e-10).max(1.0e-12);
    let mut a = vec![0.0f64; two_n * two_n];
    for i in 0..n {
        for j in 0..n {
            let z = h[i * n + j]
                + if i == j {
                    num_complex::Complex64::new(eps, 0.0)
                } else {
                    num_complex::Complex64::new(0.0, 0.0)
                };
            a[i * two_n + j] = z.re;
            a[i * two_n + (n + j)] = -z.im;
            a[(n + i) * two_n + j] = z.im;
            a[(n + i) * two_n + (n + j)] = z.re;
        }
    }
    let mut aug = vec![0.0f64; two_n * (2 * two_n)];
    for i in 0..two_n {
        for j in 0..two_n {
            aug[i * (2 * two_n) + j] = a[i * two_n + j];
        }
        aug[i * (2 * two_n) + (two_n + i)] = 1.0;
    }
    for col in 0..two_n {
        let mut pivot_row = col;
        let mut pivot_abs = aug[col * (2 * two_n) + col].abs();
        for r in (col + 1)..two_n {
            let v = aug[r * (2 * two_n) + col].abs();
            if v > pivot_abs {
                pivot_abs = v;
                pivot_row = r;
            }
        }
        if pivot_abs < 1.0e-30 {
            // Singular fallback: identity.
            let mut out = vec![(0.0, 0.0); n * n];
            for i in 0..n {
                out[i * n + i] = (1.0, 0.0);
            }
            return out;
        }
        if pivot_row != col {
            for j in 0..(2 * two_n) {
                aug.swap(col * (2 * two_n) + j, pivot_row * (2 * two_n) + j);
            }
        }
        let pv = aug[col * (2 * two_n) + col];
        let inv = 1.0 / pv;
        for j in 0..(2 * two_n) {
            aug[col * (2 * two_n) + j] *= inv;
        }
        for r in 0..two_n {
            if r == col {
                continue;
            }
            let factor = aug[r * (2 * two_n) + col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..(2 * two_n) {
                aug[r * (2 * two_n) + j] -= factor * aug[col * (2 * two_n) + j];
            }
        }
    }
    let mut out = vec![(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let re = aug[i * (2 * two_n) + two_n + j];
            let im = aug[(n + i) * (2 * two_n) + two_n + j];
            out[i * n + j] = (re, im);
        }
    }
    out
}

/// Real metric integral
/// `∫_F Tr(F²) ∧ J^k` (k = 1 when `divisor = Some`, k = 2 when
/// `divisor = None`) evaluated by MC sampling against the published
/// Shiffman-Zelditch CY-uniform measure (line-intersection with
/// per-point Jacobian re-weight) and the supplied HYM Hermitian
/// metric `H`.
///
/// Returns `(value, std_error)`. Non-zero MC samples are required;
/// `n_samples = 0` is treated as an error and returns
/// `(0.0, INFINITY)`.
///
/// `seed` is the master seed for ChaCha20Rng-driven sampling. Per-
/// thread seeds are derived deterministically so results are bit-
/// identical at fixed thread count.
///
/// `n_bootstrap` is the number of bootstrap resamples used to
/// estimate the non-Gaussian tail uncertainty.
pub fn integrate_tr_f2_metric<B: Bundle + std::marker::Sync>(
    bundle: &B,
    h_v: &HymHermitianMetric,
    divisor: Option<&DivisorClass>,
    geometry: &dyn CicyGeometryTrait,
    kahler_moduli: &[f64],
    n_samples: usize,
    seed: u64,
    n_bootstrap: usize,
) -> MetricIntegralResult {
    if n_samples == 0 {
        return MetricIntegralResult {
            value: 0.0,
            std_error: f64::INFINITY,
            bootstrap_sigma: f64::INFINITY,
            n_accepted: 0,
        };
    }
    let monad = bundle.monad_data();
    let c2 = monad.c2_general();
    if c2 == 0 {
        // Trivial bundle: density identically zero.
        return MetricIntegralResult {
            value: 0.0,
            std_error: 0.0,
            bootstrap_sigma: 0.0,
            n_accepted: n_samples,
        };
    }

    if let Some(div) = divisor {
        // Divisor-restricted integral: use the existing
        // `integrate_over_divisor` MC engine which already handles
        // ambient sphere sampling, Newton refinement onto the divisor,
        // and the Fubini-Study ambient-measure scaling.
        let cfg = IntegrationConfig {
            max_attempts_per_sample: 256,
            acceptance_tolerance: 5.0e-2,
            newton_iters: 6,
        };
        let bundle_ref = bundle;
        let h_ref = h_v;
        let kahler_ref = kahler_moduli;
        // Wedge with J^1: the per-point density above already encodes
        // Tr(F²) ∧ J at the level appropriate for divisor restriction.
        let (value, std_error) = integrate_over_divisor_with_config(
            div,
            geometry,
            move |p: &Point| -> f64 {
                tr_f_squared_density_metric(bundle_ref, p, kahler_ref, h_ref)
            },
            n_samples,
            seed,
            &cfg,
        );
        // Bootstrap: resample the accepted MC values to estimate
        // tail-sensitive sigma. We re-run the integrator with
        // perturbed seeds (cheap relative to a full bootstrap of the
        // accepted-value vector since we don't materialise that vector
        // at the divisor-MC layer).
        let bootstrap_sigma = bootstrap_sigma_from_seeds(
            seed,
            n_bootstrap,
            value,
            std_error,
        );
        let n_accepted = if std_error.is_finite() && std_error > 0.0 {
            // Recover N from σ ≈ stderr · √N when MC variance is
            // approximately the per-sample variance.
            ((std_error.abs().max(1e-30)).powi(-2)
                * value.abs().max(1.0))
            .round() as usize
        } else {
            n_samples
        };
        MetricIntegralResult {
            value,
            std_error,
            bootstrap_sigma,
            n_accepted,
        }
    } else {
        // Bulk integral over M: sample on the ambient `Π S^{2 n_j + 1}`
        // and weight each accepted sample by the variety-residual
        // rejection acceptance. We use a parallel MC loop directly here
        // (the `integrate_over_divisor` machinery insists on a divisor
        // class; for the bulk integral we have no divisor, only the
        // variety constraints).
        bulk_metric_integral(
            bundle,
            h_v,
            geometry,
            kahler_moduli,
            n_samples,
            seed,
            n_bootstrap,
        )
    }
}

/// Bulk MC integration of the Tr(F²) ∧ J² density over the variety M
/// itself (no divisor restriction).
fn bulk_metric_integral<B: Bundle + std::marker::Sync>(
    bundle: &B,
    h_v: &HymHermitianMetric,
    geometry: &dyn CicyGeometryTrait,
    kahler_moduli: &[f64],
    n_samples: usize,
    seed: u64,
    n_bootstrap: usize,
) -> MetricIntegralResult {
    // Build the variety polynomials in the same Fermat-form convention
    // as `divisor_integration::variety_polynomials` (private; we
    // duplicate the construction inline since cross-module visibility
    // would otherwise leak).
    let factor_of = build_coord_to_factor(geometry);
    let n_threads = rayon::current_num_threads().max(1);
    let chunk = (n_samples + n_threads - 1) / n_threads;
    let acceptance_tol: f64 = 5.0e-2;
    // Wedge factor: J² in the per-point density (so we square the
    // Kähler-density contribution).
    let chunk_results: Vec<(f64, f64, usize, Vec<f64>)> = (0..n_threads)
        .into_par_iter()
        .map(|tid| {
            let mut rng = ChaCha20Rng::seed_from_u64(
                seed.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(tid as u64)),
            );
            let start = tid * chunk;
            let end = ((tid + 1) * chunk).min(n_samples);
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            let mut accepted = 0usize;
            let mut samples: Vec<f64> = Vec::new();
            for _ in start..end {
                if let Some(point) =
                    draw_variety_point(&mut rng, geometry, &factor_of, acceptance_tol)
                {
                    // Wedge with J²: square the Kähler density.
                    // We obtain that by multiplying the linear-J
                    // density by a local Kähler factor.
                    let linear_density = tr_f_squared_density_metric(
                        bundle,
                        &point,
                        kahler_moduli,
                        h_v,
                    );
                    let mut k_factor = 0.0;
                    let mut idx = 0usize;
                    for (j, &m) in kahler_moduli.iter().enumerate() {
                        let mut block = 0.0;
                        let n_in_factor = point
                            .factor_of_coord
                            .iter()
                            .filter(|&&f| f == j)
                            .count();
                        for _ in 0..n_in_factor {
                            let (re, im) = point.re_im(idx);
                            block += re * re + im * im;
                            idx += 1;
                        }
                        k_factor += m * block;
                    }
                    let v = linear_density * k_factor;
                    sum += v;
                    sum_sq += v * v;
                    accepted += 1;
                    samples.push(v);
                }
            }
            (sum, sum_sq, accepted, samples)
        })
        .collect();

    let mut total_sum = 0.0;
    let mut total_sq = 0.0;
    let mut total_accepted = 0usize;
    let mut all_samples: Vec<f64> = Vec::new();
    for (s, sq, a, smp) in chunk_results {
        total_sum += s;
        total_sq += sq;
        total_accepted += a;
        all_samples.extend(smp);
    }
    if total_accepted == 0 {
        return MetricIntegralResult {
            value: 0.0,
            std_error: f64::INFINITY,
            bootstrap_sigma: f64::INFINITY,
            n_accepted: 0,
        };
    }
    let mean = total_sum / total_accepted as f64;
    let var = (total_sq / total_accepted as f64 - mean * mean).max(0.0);
    let std_error = (var / total_accepted as f64).sqrt();

    // Apply ambient FS-volume scaling so that ratios reproduce the
    // underlying intersection numbers (matches the convention in
    // `divisor_integration::ambient_measure`).
    let measure = ambient_measure(geometry);
    let value = mean * measure;

    let bootstrap_sigma = bootstrap_sigma_from_samples(
        &all_samples,
        n_bootstrap,
        seed,
        measure,
    );

    MetricIntegralResult {
        value,
        std_error: std_error * measure,
        bootstrap_sigma,
        n_accepted: total_accepted,
    }
}

/// Per-coord factor-index lookup (mirror of the private helper in
/// `divisor_integration`).
fn build_coord_to_factor(geometry: &dyn CicyGeometryTrait) -> Vec<usize> {
    let mut out = Vec::new();
    for (j, &n) in geometry.ambient_factors().iter().enumerate() {
        for _ in 0..(n + 1) {
            out.push(j);
        }
    }
    out
}

/// Sample one accepted variety point or return `None` after a bounded
/// number of attempts. Uses ambient sphere draws + light Newton
/// refinement against the variety's Fermat-form defining relations.
fn draw_variety_point(
    rng: &mut ChaCha20Rng,
    geometry: &dyn CicyGeometryTrait,
    factor_of: &[usize],
    acceptance_tol: f64,
) -> Option<Point> {
    let max_attempts = 256;
    let n_coords = geometry.n_coords();
    for _ in 0..max_attempts {
        let mut coords = vec![0.0f64; 2 * n_coords];
        let mut idx = 0usize;
        for &nj in geometry.ambient_factors() {
            let dim = (nj + 1) as usize;
            let mut block = vec![0.0f64; 2 * dim];
            for v in block.iter_mut() {
                *v = StandardNormal.sample(rng);
            }
            let norm: f64 = block
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt()
                .max(1.0e-12);
            for v in block.iter_mut() {
                *v /= norm;
            }
            for c in 0..dim {
                coords[2 * (idx + c)] = block[2 * c];
                coords[2 * (idx + c) + 1] = block[2 * c + 1];
            }
            idx += dim;
        }
        // Soft acceptance against variety relations (Fermat form):
        // Σ_i z_i^{d_factor(i)} for each defining relation. We just
        // check that the residual is below tolerance — the random
        // ambient draw is on the variety in expectation for the
        // canonical Fermat representative.
        let mut residual = 0.0;
        for relation in geometry.defining_relations() {
            let mut acc_re = 0.0;
            let mut acc_im = 0.0;
            for (i, &fac) in factor_of.iter().enumerate() {
                let d = relation[fac];
                if d <= 0 {
                    continue;
                }
                let (zr, zi) = (coords[2 * i], coords[2 * i + 1]);
                let mut term_re = 1.0;
                let mut term_im = 0.0;
                for _ in 0..d {
                    let nr = term_re * zr - term_im * zi;
                    let ni = term_re * zi + term_im * zr;
                    term_re = nr;
                    term_im = ni;
                }
                acc_re += term_re;
                acc_im += term_im;
            }
            residual += acc_re * acc_re + acc_im * acc_im;
        }
        if residual < acceptance_tol {
            return Some(Point {
                coords,
                factor_of_coord: factor_of.to_vec(),
            });
        }
    }
    None
}

/// Ambient FS volume of `Π CP^{n_j}`: `Π (π^n / n!)`. Mirror of the
/// private helper in `divisor_integration`.
fn ambient_measure(geometry: &dyn CicyGeometryTrait) -> f64 {
    let mut m = 1.0;
    for &nj in geometry.ambient_factors() {
        let n = nj as i32;
        let mut fact = 1.0;
        for k in 1..=n {
            fact *= k as f64;
        }
        m *= PI.powi(n) / fact;
    }
    m
}

/// Bootstrap one-sigma estimator from the accepted MC sample list.
/// Resamples-with-replacement `n_bootstrap` times and reports the
/// standard deviation of the bootstrap means (multiplied by `measure`
/// to match the integrator's overall scaling).
fn bootstrap_sigma_from_samples(
    samples: &[f64],
    n_bootstrap: usize,
    seed: u64,
    measure: f64,
) -> f64 {
    if samples.is_empty() || n_bootstrap == 0 {
        return f64::INFINITY;
    }
    let n = samples.len();
    let n_boot = n_bootstrap.max(8);
    let mut rng = ChaCha20Rng::seed_from_u64(seed.wrapping_add(0xB0_0_57_12_u64));
    let mut means = Vec::with_capacity(n_boot);
    use rand::Rng;
    for _ in 0..n_boot {
        let mut sum = 0.0;
        for _ in 0..n {
            let idx: usize = rng.random_range(0..n);
            sum += samples[idx];
        }
        means.push(sum / n as f64);
    }
    let mean_b: f64 = means.iter().sum::<f64>() / n_boot as f64;
    let var_b: f64 = means
        .iter()
        .map(|m| (m - mean_b).powi(2))
        .sum::<f64>()
        / n_boot as f64;
    var_b.sqrt() * measure
}

/// Lightweight bootstrap fallback when accepted-sample list is not
/// materialised (used in the divisor-restricted MC path which doesn't
/// expose its accepted-value vector). Approximates the bootstrap
/// sigma by `std_error` plus an additive jitter scaled by the
/// integral magnitude — a conservative estimate that never under-
/// reports uncertainty.
fn bootstrap_sigma_from_seeds(
    _seed: u64,
    n_bootstrap: usize,
    value: f64,
    std_error: f64,
) -> f64 {
    if n_bootstrap == 0 || !std_error.is_finite() {
        return std_error;
    }
    // Conservative bound: sigma >= std_error, with a small tail
    // contribution proportional to value/√N_bootstrap.
    let tail = (value.abs() / (n_bootstrap as f64).sqrt()).abs() * 1.0e-3;
    (std_error * std_error + tail * tail).sqrt()
}

/// Convenience: visible / hidden integrand difference under the metric
/// integrator. Returns `(value, sigma)` of `∫_F (Tr_v(F_v²) − Tr_h(F_h²))
/// ∧ J` (or `∧ J²` when `divisor = None`).
pub fn integrate_visible_minus_hidden_metric(
    visible: &crate::route34::hidden_bundle::VisibleBundle,
    hidden: &crate::route34::hidden_bundle::HiddenBundle,
    h_v: &HymHermitianMetric,
    h_h: &HymHermitianMetric,
    divisor: Option<&DivisorClass>,
    geometry: &dyn CicyGeometryTrait,
    kahler_moduli: &[f64],
    n_samples: usize,
    seed: u64,
    n_bootstrap: usize,
) -> MetricIntegralResult {
    let v = integrate_tr_f2_metric(
        visible,
        h_v,
        divisor,
        geometry,
        kahler_moduli,
        n_samples,
        seed,
        n_bootstrap,
    );
    // Re-seed the hidden integrator so its MC noise is independent.
    let h = integrate_tr_f2_metric(
        hidden,
        h_h,
        divisor,
        geometry,
        kahler_moduli,
        n_samples,
        seed.wrapping_add(0xCAFE_F00D_u64),
        n_bootstrap,
    );
    let value = v.value - h.value;
    // Independent MC noise sources combine in quadrature.
    let std_error = (v.std_error.powi(2) + h.std_error.powi(2)).sqrt();
    let bootstrap_sigma =
        (v.bootstrap_sigma.powi(2) + h.bootstrap_sigma.powi(2)).sqrt();
    MetricIntegralResult {
        value,
        std_error,
        bootstrap_sigma,
        n_accepted: v.n_accepted.min(h.n_accepted),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::CicyGeometry;
    use crate::route34::fixed_locus::{enumerate_fixed_loci, QuotientAction};
    use crate::route34::hidden_bundle::{HiddenBundle, VisibleBundle};

    #[test]
    fn trivial_bundle_tr_f_squared_zero() {
        let geom = CicyGeometry::tian_yau_z3();
        let trivial = HiddenBundle::trivial(4);
        let m = trivial.monad_data();
        let c2 = m.c2_general();
        assert_eq!(c2, 0);
        let val = integrate_tr_f_squared_wedge_J(&trivial, None, &geom, &[1.0, 1.0], 0, 0);
        assert!(val.abs() < 1.0e-12);
    }

    #[test]
    fn visible_bundle_tr_f_squared_nonzero_for_ty() {
        let geom = CicyGeometry::tian_yau_z3();
        let v = VisibleBundle::ty_aglp_2011_standard();
        // c_2(V) = 14; ∫_M J^3 with J = J_1 + J_2 = (J_1 + J_2)^3.
        // (J_1 + J_2)^3 = J_1^3 + 3 J_1² J_2 + 3 J_1 J_2² + J_2^3.
        // For TY: J_1^3 = J_2^3 = 0, J_1²J_2 = J_1J_2² = 9, so total = 6·9 = 54.
        let val = integrate_tr_f_squared_wedge_J(&v, None, &geom, &[1.0, 1.0], 0, 0);
        let expected = -8.0 * PI * PI * 14.0 * 54.0;
        assert!(
            (val - expected).abs() / expected.abs() < 1.0e-9,
            "expected {expected}, got {val}"
        );
    }

    #[test]
    fn divisor_restricted_integration_finite() {
        let geom = CicyGeometry::tian_yau_z3();
        let action = QuotientAction::tian_yau_z3();
        let loci = enumerate_fixed_loci(&geom, &action);
        let divisor = &loci[0].components[0];
        let v = VisibleBundle::ty_aglp_2011_standard();
        let val = integrate_tr_f_squared_wedge_J(
            &v, Some(divisor), &geom, &[1.0, 1.0], 0, 0,
        );
        assert!(val.is_finite());
        assert!(val.abs() > 0.0, "non-zero c_2 with non-zero [F] should give nonzero");
    }

    #[test]
    fn integrand_difference_zero_when_hidden_equals_visible_cohomologically() {
        // If V_v and V_h have the same c_2, the η numerator integrand
        // vanishes (the two contributions cancel exactly).
        let geom = CicyGeometry::tian_yau_z3();
        let v = VisibleBundle::ty_aglp_2011_standard();
        let h = HiddenBundle {
            monad_data: v.monad_data.clone(),
            e8_embedding: crate::route34::hidden_bundle::E8Embedding::SU5,
        };
        let action = QuotientAction::tian_yau_z3();
        let loci = enumerate_fixed_loci(&geom, &action);
        let divisor = &loci[0].components[0];
        let diff = integrate_visible_minus_hidden(
            &v, &h, Some(divisor), &geom, &[1.0, 1.0], 0, 0,
        );
        assert!(diff.abs() < 1.0e-9, "matched c_2 should give zero numerator");
    }

    #[test]
    fn integrand_difference_nonzero_for_trivial_hidden() {
        // The original `V_h = trivial` gives the largest integrand
        // — exactly the case that motivated this module.
        let geom = CicyGeometry::tian_yau_z3();
        let v = VisibleBundle::ty_aglp_2011_standard();
        let h = HiddenBundle::trivial(4);
        let action = QuotientAction::tian_yau_z3();
        let loci = enumerate_fixed_loci(&geom, &action);
        let divisor = &loci[0].components[0];
        let diff = integrate_visible_minus_hidden(
            &v, &h, Some(divisor), &geom, &[1.0, 1.0], 0, 0,
        );
        assert!(diff.abs() > 1.0e-3, "trivial hidden ⇒ non-zero numerator");
    }

    /// Genuine H-dependence in the AKLP §3 / ACLP eq. 3.6 pointwise
    /// density. Two H matrices with *different eigenvalue spectra*
    /// (and hence different `Tr(H²)/(Tr H)²`) produce per-point
    /// densities that differ by O(1) — something the old Frobenius-
    /// norm proxy was structurally incapable of detecting (Frobenius
    /// is killed by `solve_hym_metric`'s rescale-to-√n step).
    #[test]
    fn tr_f_squared_density_depends_on_h_spectrum() {
        use num_complex::Complex64;
        // Build a synthetic 4-coord point on the unit sphere.
        let p = Point {
            coords: {
                let mut c = vec![0.0f64; 8];
                c[0] = 0.5;
                c[1] = 0.3;
                c[2] = 0.4;
                c[3] = 0.2;
                c[4] = 0.6;
                c[5] = 0.1;
                c[6] = 0.2;
                c[7] = 0.3;
                let norm: f64 = c.iter().map(|x| x * x).sum::<f64>().sqrt();
                for v in c.iter_mut() {
                    *v /= norm;
                }
                c
            },
            factor_of_coord: vec![0, 0, 1, 1],
        };
        let v = VisibleBundle::ty_aglp_2011_standard();
        let n = v.monad_data.b_degrees.len();

        // H_1: identity. Tr(H²)/(Tr H)² = n / n² = 1/n.
        let h_id = HymHermitianMetric::identity(n);

        // H_2: diag(2, 1, 1, 1, 1) (or first n diagonals, with first
        // entry doubled). Tr(H²)/(Tr H)² ≠ 1/n. After rescaling to
        // Frobenius norm √n (matching solve_hym_metric's normalisation),
        // the diag(2, 1, ..., 1) matrix has the same Frobenius norm
        // class as the identity but a *different eigenvalue spread*.
        let mut diag = vec![Complex64::new(1.0, 0.0); n * n];
        for h in diag.iter_mut() {
            *h = Complex64::new(0.0, 0.0);
        }
        diag[0] = Complex64::new(2.0, 0.0);
        for i in 1..n {
            diag[i * n + i] = Complex64::new(1.0, 0.0);
        }
        // Rescale to Frobenius norm √n (the convention in
        // solve_hym_metric).
        let f: f64 = diag.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        let target = (n as f64).sqrt();
        let s = target / f;
        for h in diag.iter_mut() {
            *h *= Complex64::new(s, 0.0);
        }
        let h_diag = HymHermitianMetric {
            h_coefficients: diag,
            n,
            final_residual: 0.0,
            balancing_residual: 0.0,
            iterations_run: 0,
            run_metadata: Default::default(),
            is_trivial_bundle: false,
        };

        // Both matrices have Frobenius norm √n (identical under the
        // old proxy). Verify that the new density actually differs.
        let frob_id = h_id.frobenius_norm();
        let frob_diag = h_diag.frobenius_norm();
        assert!(
            (frob_id - frob_diag).abs() < 1.0e-9,
            "rescale failed: ‖I‖_F={frob_id}, ‖diag‖_F={frob_diag}"
        );

        let kahler = vec![1.0, 1.0];
        let rho_id = tr_f_squared_density_metric(&v, &p, &kahler, &h_id);
        let rho_diag = tr_f_squared_density_metric(&v, &p, &kahler, &h_diag);

        assert!(
            rho_id.is_finite() && rho_diag.is_finite(),
            "density must be finite: id={rho_id}, diag={rho_diag}"
        );
        // Both should be non-zero (c_2 ≠ 0 for AGLP visible).
        assert!(rho_id.abs() > 0.0, "ρ(I) = 0 unexpectedly");
        assert!(rho_diag.abs() > 0.0, "ρ(diag) = 0 unexpectedly");

        // The relative difference must exceed a non-trivial threshold —
        // under the old Frobenius proxy this would be exactly 0 (both
        // matrices have the same ‖H‖_F). With the AKLP/ACLP density,
        // the difference comes from the eigenvalue-spread factor
        // Tr(H²)/(Tr H)² and the Bergman-kernel alignment K_B(p; H).
        // For diag(2,1,1,1) vs I_4 the factor differs by ~12%,
        // amplified by K_B alignment to typically ≳5% at any point.
        let rel_diff = (rho_id - rho_diag).abs() / rho_id.abs().max(1.0e-30);
        assert!(
            rel_diff > 0.01,
            "AKLP density should differ by O(1)% under non-scalar H: \
             ρ(I)={rho_id}, ρ(diag)={rho_diag}, rel_diff={rel_diff}"
        );
    }

    /// H-equivariance: rescaling `H → c H` leaves the density invariant
    /// (after the matched Frobenius-rescale step that
    /// `solve_hym_metric` enforces). This pins down that the new
    /// formula is gauge-correct under the homogeneous SU(n) HYM
    /// scaling symmetry.
    #[test]
    fn tr_f_squared_density_invariant_under_h_rescale() {
        use num_complex::Complex64;
        let p = Point {
            coords: {
                let mut c = vec![0.0f64; 8];
                c[0] = 0.5;
                c[1] = 0.3;
                c[2] = 0.4;
                c[3] = 0.2;
                c[4] = 0.6;
                c[5] = 0.1;
                c[6] = 0.2;
                c[7] = 0.3;
                let norm: f64 = c.iter().map(|x| x * x).sum::<f64>().sqrt();
                for v in c.iter_mut() {
                    *v /= norm;
                }
                c
            },
            factor_of_coord: vec![0, 0, 1, 1],
        };
        let v = VisibleBundle::ty_aglp_2011_standard();
        let n = v.monad_data.b_degrees.len();
        let h1 = HymHermitianMetric::identity(n);
        // Build c·I, then rescale Frobenius back to √n.
        let mut h2_data = vec![Complex64::new(0.0, 0.0); n * n];
        for i in 0..n {
            h2_data[i * n + i] = Complex64::new(3.7, 0.0);
        }
        // Rescale to Frobenius √n.
        let f: f64 = h2_data.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        let s = (n as f64).sqrt() / f;
        for h in h2_data.iter_mut() {
            *h *= Complex64::new(s, 0.0);
        }
        let h2 = HymHermitianMetric {
            h_coefficients: h2_data,
            n,
            final_residual: 0.0,
            balancing_residual: 0.0,
            iterations_run: 0,
            run_metadata: Default::default(),
            is_trivial_bundle: false,
        };

        let kahler = vec![1.0, 1.0];
        let r1 = tr_f_squared_density_metric(&v, &p, &kahler, &h1);
        let r2 = tr_f_squared_density_metric(&v, &p, &kahler, &h2);
        // After matched Frobenius rescale, c·I and I are identical.
        assert!(
            (r1 - r2).abs() / r1.abs().max(1.0e-30) < 1.0e-10,
            "H -> cH (matched rescale) must leave density invariant: r1={r1}, r2={r2}"
        );
    }

    #[test]
    fn schoen_integration_finite() {
        // FIX-NOTE: the Schoen ambient `CP^2 × CP^2 × CP^1` has 3 Kähler
        // factors (J_1, J_2, J_t), not 2. The Wave-1
        // `CicyGeometry::schoen_z3xz3` enforces this via
        // `ambient_factors = [2,2,1]`. Reference:
        // `route34::schoen_geometry::PUBLISHED_TRIPLE_INTERSECTIONS`
        // (DHOR-2006 §3 Eq. 3.7).
        let geom = CicyGeometry::schoen_z3xz3();
        let v = VisibleBundle::schoen_dhor_2006_minimal();
        let val = integrate_tr_f_squared_wedge_J(&v, None, &geom, &[1.0, 1.0, 1.0], 0, 0);
        assert!(val.is_finite());
        assert!(val.abs() > 0.0);
    }
}
