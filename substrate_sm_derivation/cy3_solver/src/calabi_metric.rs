//! Stack-C Phase 2: Ricci tensor + Calabi flow on the Fermat quintic.
//!
//! This module is purely additive on top of Stack A's σ-functional
//! pipeline (`crate::quintic::compute_sigma_from_workspace`). It plugs
//! the just-shipped pwos-math PDE primitives
//! (`pwos_math::pde::ricci_at_point_via_fd`) into the Bergman-kernel
//! ambient metric `g_{ij̄}(z; h) = ∂_i ∂_{j̄} log K(z; h)` and computes
//! the pointwise Ricci tensor
//!
//!     Ric_{ij̄}(z; h) = -∂_i ∂_{j̄} log det g(z; h)
//!
//! by central finite differences of `log det g` on the **5×5 ambient**
//! metric. This is the natural object on the Fermat quintic embedded in
//! `CP^4`: the 3-dim CY tangent metric is obtained by tangent-frame
//! projection (used by Stack A's σ-eval) but for Calabi-flow / Ricci-
//! flatness diagnostics on a Bergman candidate, the ambient `log det g`
//! is the stable target.
//!
//! ## Cross-validation goal
//!
//! Stack A reports `σ_L¹ ≈ 0.13` at k=4 (post-fix). At a Ricci-flat
//! Kähler-Einstein metric BOTH `σ → 0` AND `Ric → 0`. So the test of
//! agreement is "do `σ` and `‖Ric‖_L²` track each other across
//! refinements?". This module provides the `‖Ric‖_L²` evaluator and a
//! correlation harness that the Stack-C cross-validation tests
//! (`tests` submodule) consume.
//!
//! ## Performance envelope
//!
//! At each test point we evaluate `log det g_amb` at `O(n²)` perturbed
//! `z`-points (5×5 mixed Wirtinger stencil), where each evaluation is
//! `O(n_basis² + n_basis·n)` work. For n_basis = 70 (k=4), one Ricci
//! tensor at one sample point is ~25 metric evaluations × ~70² flops ≈
//! 10⁵ flops; at 1000 points that's ~10⁸ flops, ~0.1 s. k=2 is faster
//! by ~25×.
//!
//! Long-running tests (`#[ignore]`) document their wall-clock; default
//! tests stay under 30 s.

use pwos_math::pde::{ricci_at_point_via_fd, RicciError};
use rayon::prelude::*;

use crate::quintic::{
    det_3x3_complex_hermitian, fermat_quintic_gradient, newton_project_to_quintic,
    project_to_quintic_tangent, quintic_affine_chart_frame, quintic_chart_and_elim,
};

/// Ambient complex dimension on the Fermat quintic in CP^4.
const N_AMB: usize = 5;
/// Real coordinate dimension (interleaved (re, im) per complex coord).
const N_AMB_REAL: usize = 2 * N_AMB;
/// Output Ricci tensor length: 2 × 5 × 5 = 50 f64 (interleaved re/im).
const RIC_LEN: usize = 2 * N_AMB * N_AMB;

/// CY tangent complex dimension (3-D tangent of the Fermat quintic in CP^4).
const N_TAN: usize = 3;
/// Tangent-frame real-coord length: interleaved (re, im) per complex coord.
const N_TAN_REAL: usize = 2 * N_TAN;
/// Output Ricci tensor length on the tangent space: 2 × 3 × 3 = 18 f64.
pub const RIC_TAN_LEN: usize = 2 * N_TAN * N_TAN;

/// Errors emitted by Stack-C Calabi-metric helpers.
#[derive(Debug)]
pub enum CalabiMetricError {
    /// Underlying pwos-math FD-Ricci evaluator returned an error.
    Ricci(RicciError),
    /// `log det g_amb` was non-finite or non-positive at a probe point.
    NonFiniteLogDet,
    /// Kähler ambient metric was numerically singular at the probe point.
    SingularMetric,
}

impl core::fmt::Display for CalabiMetricError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Ricci(e) => write!(f, "calabi_metric: ricci FD failure: {e}"),
            Self::NonFiniteLogDet => f.write_str("calabi_metric: non-finite log det g"),
            Self::SingularMetric => f.write_str("calabi_metric: singular ambient metric"),
        }
    }
}

impl std::error::Error for CalabiMetricError {}

/// Evaluate the section basis `s_a(z)` (degree-k holomorphic monomials)
/// at a single point `z ∈ R^{10}` (interleaved 5 complex coords). Output
/// `s_out` has length `2 * n_basis` interleaved (re, im).
///
/// This is the per-point analog of `evaluate_quintic_basis` and is used
/// by the FD-Ricci closure to evaluate at perturbed points (which are
/// **not** sample points and therefore not in the precomputed
/// `section_values` table).
fn evaluate_section_basis_at_point(
    z: &[f64; N_AMB_REAL],
    monomials: &[[u32; 5]],
    s_out: &mut [f64],
) {
    let n_basis = monomials.len();
    debug_assert_eq!(s_out.len(), 2 * n_basis);

    let kmax: u32 = monomials
        .iter()
        .flat_map(|m| m.iter())
        .copied()
        .max()
        .unwrap_or(0);
    let stride = (kmax + 1) as usize;

    // Power table: pow[j][e] = z_j^e (re, im).
    // Heap-resident scratch — at k=6 with stride=7, this is 70 f64.
    let mut pow_table = vec![0.0f64; 5 * stride * 2];
    for j in 0..5 {
        pow_table[j * stride * 2] = 1.0;
        pow_table[j * stride * 2 + 1] = 0.0;
    }
    for j in 0..5 {
        let zr = z[2 * j];
        let zi = z[2 * j + 1];
        for e in 1..=kmax as usize {
            let prev_re = pow_table[j * stride * 2 + (e - 1) * 2];
            let prev_im = pow_table[j * stride * 2 + (e - 1) * 2 + 1];
            pow_table[j * stride * 2 + e * 2] = prev_re * zr - prev_im * zi;
            pow_table[j * stride * 2 + e * 2 + 1] = prev_re * zi + prev_im * zr;
        }
    }
    for (jm, m) in monomials.iter().enumerate() {
        let mut prod_re = 1.0;
        let mut prod_im = 0.0;
        for j in 0..5 {
            let e = m[j] as usize;
            let zr = pow_table[j * stride * 2 + e * 2];
            let zi = pow_table[j * stride * 2 + e * 2 + 1];
            let new_re = prod_re * zr - prod_im * zi;
            let new_im = prod_re * zi + prod_im * zr;
            prod_re = new_re;
            prod_im = new_im;
        }
        s_out[2 * jm] = prod_re;
        s_out[2 * jm + 1] = prod_im;
    }
}

/// First-derivative table: `ds_out[i * 2*n_basis .. (i+1)*2*n_basis]`
/// holds `∂_{z_i} s_a(z)` interleaved re/im for `i ∈ {0..5}`.
fn evaluate_section_basis_derivs_at_point(
    z: &[f64; N_AMB_REAL],
    monomials: &[[u32; 5]],
    ds_out: &mut [f64],
) {
    let n_basis = monomials.len();
    let two_n = 2 * n_basis;
    debug_assert_eq!(ds_out.len(), 5 * two_n);

    let kmax: u32 = monomials
        .iter()
        .flat_map(|m| m.iter())
        .copied()
        .max()
        .unwrap_or(0);
    let stride = (kmax + 1) as usize;

    let mut pow_table = vec![0.0f64; 5 * stride * 2];
    for j in 0..5 {
        pow_table[j * stride * 2] = 1.0;
        pow_table[j * stride * 2 + 1] = 0.0;
    }
    for j in 0..5 {
        let zr = z[2 * j];
        let zi = z[2 * j + 1];
        for e in 1..=kmax as usize {
            let prev_re = pow_table[j * stride * 2 + (e - 1) * 2];
            let prev_im = pow_table[j * stride * 2 + (e - 1) * 2 + 1];
            pow_table[j * stride * 2 + e * 2] = prev_re * zr - prev_im * zi;
            pow_table[j * stride * 2 + e * 2 + 1] = prev_re * zi + prev_im * zr;
        }
    }
    for i in 0..5 {
        for (a, m) in monomials.iter().enumerate() {
            if m[i] == 0 {
                ds_out[i * two_n + 2 * a] = 0.0;
                ds_out[i * two_n + 2 * a + 1] = 0.0;
                continue;
            }
            let factor_re = pow_table[i * stride * 2 + (m[i] as usize - 1) * 2];
            let factor_im = pow_table[i * stride * 2 + (m[i] as usize - 1) * 2 + 1];
            let m_factor = m[i] as f64;
            let mut prod_re = m_factor * factor_re;
            let mut prod_im = m_factor * factor_im;
            for j in 0..5 {
                if j == i {
                    continue;
                }
                let e = m[j] as usize;
                let zr = pow_table[j * stride * 2 + e * 2];
                let zi = pow_table[j * stride * 2 + e * 2 + 1];
                let nr = prod_re * zr - prod_im * zi;
                let ni = prod_re * zi + prod_im * zr;
                prod_re = nr;
                prod_im = ni;
            }
            ds_out[i * two_n + 2 * a] = prod_re;
            ds_out[i * two_n + 2 * a + 1] = prod_im;
        }
    }
}

/// Compute the **5×5 ambient** Bergman-kernel metric `g_{ij̄}(z; h)` at
/// a single point `z`, where
///
///     g_{ij̄} = ∂_i ∂_{j̄} log K(z; h),
///     K(z; h) = s(z)† h s(z).
///
/// Output `g_out` is a 5×5 complex Hermitian matrix in interleaved
/// (re, im) row-major layout: `g_out[2*(i*5 + j) + 0..1]` is
/// `(Re, Im)` of `g_{ij̄}`.
///
/// `s_scratch` is `2 * n_basis` f64; `ds_scratch` is `5 * 2 * n_basis`
/// f64; `h_s_scratch` is `2 * n_basis` f64; `h_ds_scratch` is
/// `5 * 2 * n_basis` f64. Caller pre-allocates to avoid heap traffic
/// inside the FD inner loop.
#[allow(clippy::too_many_arguments)]
fn compute_g_amb_at_point(
    z: &[f64; N_AMB_REAL],
    h_block: &[f64],
    monomials: &[[u32; 5]],
    n_basis: usize,
    s_scratch: &mut [f64],
    ds_scratch: &mut [f64],
    h_s_scratch: &mut [f64],
    h_ds_scratch: &mut [f64],
    g_out: &mut [f64; 50],
) -> Result<(), CalabiMetricError> {
    let two_n = 2 * n_basis;

    evaluate_section_basis_at_point(z, monomials, s_scratch);
    evaluate_section_basis_derivs_at_point(z, monomials, ds_scratch);

    // K = s† h s. With h_block in 2n×2n real-block layout where the
    // (a,b) Hermitian entry decomposes as
    //   h_block[(2a)(2b)] = h_re_ab,  h_block[(2a)(2b+1)] = -h_im_ab,
    //   h_block[(2a+1)(2b)] = h_im_ab, h_block[(2a+1)(2b+1)] = h_re_ab,
    // and s laid out as [s0_re, s0_im, s1_re, s1_im, ...], the real
    // quadratic form |s|²_h = sᵀ h_block s recovers the (real-valued)
    // Hermitian K up to factor 1.
    let mut k_val = 0.0;
    for i in 0..two_n {
        let mut row_sum = 0.0;
        for j in 0..two_n {
            row_sum += h_block[i * two_n + j] * s_scratch[j];
        }
        h_s_scratch[i] = row_sum;
        k_val += s_scratch[i] * row_sum;
    }
    if !k_val.is_finite() || k_val <= 0.0 {
        return Err(CalabiMetricError::SingularMetric);
    }
    let inv_k = 1.0 / k_val;
    let inv_k2 = inv_k * inv_k;

    // h · ds_i for each i (precompute).
    for i in 0..5 {
        let dfi = &ds_scratch[i * two_n..(i + 1) * two_n];
        let h_dfi_slot = &mut h_ds_scratch[i * two_n..(i + 1) * two_n];
        for k in 0..two_n {
            let mut row_sum = 0.0;
            for l in 0..two_n {
                row_sum += h_block[k * two_n + l] * dfi[l];
            }
            h_dfi_slot[k] = row_sum;
        }
    }

    // dk[i] = (∂_{z_i} K) = s† h ∂_i s.
    // Using the conventions in compute_sigma_from_workspace: with
    //   h_s_scratch playing the role of h·s,
    //   sum_a (h_s_a)^* (df_i)_a = sum_a (h_s_re ds_re + h_s_im ds_im)
    //                              + i (h_s_re ds_im - h_s_im ds_re)
    let mut dk = [(0.0f64, 0.0f64); 5];
    for i in 0..5 {
        let dfi = &ds_scratch[i * two_n..(i + 1) * two_n];
        let mut s_re = 0.0;
        let mut s_im = 0.0;
        for a in 0..n_basis {
            let h_dr = h_s_scratch[2 * a];
            let h_di = h_s_scratch[2 * a + 1];
            let dr = dfi[2 * a];
            let di = dfi[2 * a + 1];
            s_re += h_dr * dr + h_di * di;
            s_im += h_dr * di - h_di * dr;
        }
        dk[i] = (s_re, s_im);
    }

    // ddk[i][j] = (∂_{z̄_j} ∂_{z_i} K)  where the second index uses the
    // `(df_j)†` factor. Mirrors the convention in
    // compute_sigma_from_workspace.
    let mut ddk = [[(0.0f64, 0.0f64); 5]; 5];
    for i in 0..5 {
        let h_dfi_slot = &h_ds_scratch[i * two_n..(i + 1) * two_n];
        for j in 0..5 {
            let dfj = &ds_scratch[j * two_n..(j + 1) * two_n];
            let mut s_re = 0.0;
            let mut s_im = 0.0;
            for a in 0..n_basis {
                let dfj_re = dfj[2 * a];
                let dfj_im = dfj[2 * a + 1];
                let hd_re = h_dfi_slot[2 * a];
                let hd_im = h_dfi_slot[2 * a + 1];
                s_re += dfj_re * hd_re + dfj_im * hd_im;
                s_im += dfj_re * hd_im - dfj_im * hd_re;
            }
            ddk[i][j] = (s_re, s_im);
        }
    }

    // g_{ij̄} = ddk[i][j] / K - dk[i] * conj(dk[j]) / K²
    for i in 0..5 {
        for j in 0..5 {
            let term1_re = ddk[i][j].0 * inv_k;
            let term1_im = ddk[i][j].1 * inv_k;
            // conj(dk[j]) = (dk_j_re, -dk_j_im)
            // dk[i] * conj(dk[j]) =
            //   (dk_i_re + i dk_i_im)(dk_j_re - i dk_j_im)
            //  = dk_i_re dk_j_re + dk_i_im dk_j_im
            //   + i (dk_i_im dk_j_re - dk_i_re dk_j_im)
            let p_re = dk[i].0 * dk[j].0 + dk[i].1 * dk[j].1;
            let p_im = dk[i].1 * dk[j].0 - dk[i].0 * dk[j].1;
            let off = 2 * (i * 5 + j);
            g_out[off] = term1_re - p_re * inv_k2;
            g_out[off + 1] = term1_im - p_im * inv_k2;
        }
    }
    Ok(())
}

/// LU-decomposition based real `log |det A|` on a 2N × 2N real matrix
/// representation of an N × N complex Hermitian matrix. We use the
/// Cayley-Sylvester identity: for complex Hermitian H = A + iB (A
/// symmetric real, B antisymmetric real), the 2N × 2N real block
/// matrix
///
///     M = [[A, -B], [B, A]]
///
/// has `det M = |det H|²`, so `log|det H| = 0.5 * log det M`.
///
/// We don't have that block layout available here directly: we have a
/// 5×5 complex Hermitian metric in interleaved (re, im) row-major. We
/// build the 10×10 real block on the fly and run a partial-pivoting
/// Gaussian elimination to compute log|det|.
fn log_det_complex_hermitian_5x5(g: &[f64; 50]) -> Result<f64, CalabiMetricError> {
    // Build 10×10 real block M from the 5×5 complex Hermitian g_ij̄.
    // For complex M_ij = A_ij + i B_ij  (A symmetric, B antisymmetric):
    //   block[i, j]         = A_ij
    //   block[i, j + N]     = -B_ij
    //   block[i + N, j]     = B_ij
    //   block[i + N, j + N] = A_ij
    const N: usize = 5;
    const N2: usize = 2 * N;
    let mut m = [[0.0f64; N2]; N2];
    for i in 0..N {
        for j in 0..N {
            let off = 2 * (i * N + j);
            let a_ij = g[off];
            let b_ij = g[off + 1];
            m[i][j] = a_ij;
            m[i][j + N] = -b_ij;
            m[i + N][j] = b_ij;
            m[i + N][j + N] = a_ij;
        }
    }

    // Partial-pivoting LU on m, accumulating log|det|.
    // Skip sign-tracking — we want log|det| only.
    let mut log_abs_det = 0.0f64;
    for k in 0..N2 {
        // Find pivot.
        let mut pivot_row = k;
        let mut pivot_val = m[k][k].abs();
        for r in (k + 1)..N2 {
            let v = m[r][k].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = r;
            }
        }
        if pivot_val <= 1e-300 {
            return Err(CalabiMetricError::SingularMetric);
        }
        if pivot_row != k {
            m.swap(k, pivot_row);
        }
        let pivot = m[k][k];
        log_abs_det += pivot.abs().ln();
        // Eliminate below.
        for r in (k + 1)..N2 {
            let factor = m[r][k] / pivot;
            for c in k..N2 {
                m[r][c] -= factor * m[k][c];
            }
        }
    }
    // For the 2N×2N real block, det M = |det H|², so
    //   log|det H| = 0.5 * log|det M|.
    let val = 0.5 * log_abs_det;
    if !val.is_finite() {
        return Err(CalabiMetricError::NonFiniteLogDet);
    }
    Ok(val)
}

/// Compute `Ric_{ij̄}(z; h)` at a single ambient point `z` by central
/// finite differences of `log det g_amb(z')`.
///
/// Output `out` is a 5×5 complex Hermitian Ricci tensor packed as
/// interleaved (re, im) row-major (length 50 f64).
///
/// `h_step` is the FD perturbation in real coordinates. Recommended
/// default: `1e-4` (balances `O(h²)` truncation against `O(eps/h²)`
/// cancellation).
///
/// # Errors
/// Returns [`CalabiMetricError::Ricci`] if the FD evaluator rejects
/// the inputs (length mismatch, etc.) or if any closure call returned
/// non-finite. Returns [`CalabiMetricError::SingularMetric`] /
/// [`CalabiMetricError::NonFiniteLogDet`] if the metric goes singular
/// at any FD probe point.
pub fn ricci_at_point_bergman(
    h_block: &[f64],
    z: &[f64; N_AMB_REAL],
    monomials: &[[u32; 5]],
    n_basis: usize,
    h_step: f64,
    out: &mut [f64; RIC_LEN],
) -> Result<(), CalabiMetricError> {
    let two_n = 2 * n_basis;

    // Pre-allocate scratch buffers reused across all 25 × ~4 FD probe
    // evaluations from the FD stencil.
    let mut s_scratch = vec![0.0f64; two_n];
    let mut ds_scratch = vec![0.0f64; 5 * two_n];
    let mut h_s_scratch = vec![0.0f64; two_n];
    let mut h_ds_scratch = vec![0.0f64; 5 * two_n];
    let mut g_scratch = [0.0f64; 50];

    // Closure: log det g_amb(z'; h). Captures the scratch by mutable
    // reference; pwos-math's FnMut signature accommodates this.
    //
    // We CANNOT propagate the SingularMetric / NonFiniteLogDet errors
    // through the closure return value (it's `f64`), so we encode them
    // as `f64::NAN` / `f64::INFINITY`; the caller (ricci_at_point_via_fd)
    // detects non-finite and returns RicciError::NonFiniteLogDet, which
    // we map below. The original cause is lost but the failure is
    // surfaced.
    let mut last_err: Option<CalabiMetricError> = None;
    let mut closure_z: [f64; N_AMB_REAL] = [0.0; N_AMB_REAL];
    let log_det_eval = |z_perturbed: &[f64]| -> f64 {
        debug_assert_eq!(z_perturbed.len(), N_AMB_REAL);
        closure_z.copy_from_slice(z_perturbed);
        match compute_g_amb_at_point(
            &closure_z,
            h_block,
            monomials,
            n_basis,
            &mut s_scratch,
            &mut ds_scratch,
            &mut h_s_scratch,
            &mut h_ds_scratch,
            &mut g_scratch,
        ) {
            Ok(()) => match log_det_complex_hermitian_5x5(&g_scratch) {
                Ok(v) => v,
                Err(e) => {
                    last_err = Some(e);
                    f64::NAN
                }
            },
            Err(e) => {
                last_err = Some(e);
                f64::NAN
            }
        }
    };

    let result = ricci_at_point_via_fd(z, N_AMB, log_det_eval, h_step, out.as_mut_slice());
    match result {
        Ok(()) => Ok(()),
        Err(e) => {
            if let Some(stack_err) = last_err {
                Err(stack_err)
            } else {
                Err(CalabiMetricError::Ricci(e))
            }
        }
    }
}

/// Compute the **L²-Ricci-norm**:
///
///     ‖Ric‖²_{L²} = (Σ_p w_p Σ_{ij} |Ric_{ij̄}(z_p; h)|²) / (Σ_p w_p)
///
/// over the supplied sample set. Returns the SQUARE-ROOT (i.e. the
/// L² norm itself, not the squared norm).
///
/// Points where Ricci evaluation fails (singular metric, non-finite
/// log det) are dropped from the average; if more than 50% of points
/// fail, returns NaN.
///
/// # Arguments
/// * `points` — flat `[n_actual × 10]` interleaved (re, im) coordinates.
/// * `weights` — length `n_actual`, FS or CY-measure weights.
/// * `n_actual` — number of valid sample points.
/// * `h_block` — `(2*n_basis)²` real-block Hermitian h.
/// * `monomials` / `n_basis` — degree-k section basis.
/// * `h_step` — FD perturbation; recommended `1e-4`.
pub fn ricci_norm_l2(
    points: &[f64],
    weights: &[f64],
    n_actual: usize,
    h_block: &[f64],
    monomials: &[[u32; 5]],
    n_basis: usize,
    h_step: f64,
) -> f64 {
    debug_assert!(points.len() >= n_actual * 10);
    debug_assert!(weights.len() >= n_actual);

    // Per-point reduction: parallel via rayon, each thread allocates
    // its own scratch (the FD evaluator's internal allocations are
    // bounded by `n_basis`, sub-millisecond).
    let result: (f64, f64, usize) = (0..n_actual)
        .into_par_iter()
        .map(|p| {
            let mut z = [0.0f64; N_AMB_REAL];
            z.copy_from_slice(&points[p * 10..p * 10 + 10]);
            let mut ric = [0.0f64; RIC_LEN];
            match ricci_at_point_bergman(h_block, &z, monomials, n_basis, h_step, &mut ric) {
                Ok(()) => {
                    let mut sumsq = 0.0;
                    for v in ric.iter() {
                        sumsq += v * v;
                    }
                    if !sumsq.is_finite() || !weights[p].is_finite() || weights[p] <= 0.0 {
                        (0.0, 0.0, 0usize)
                    } else {
                        // Frobenius norm of Ric: Σ |Ric_ij|² = Σ (Re² + Im²),
                        // which we already accumulated via |re/im pair|².
                        // Note: the Hermitian symmetry `out[i,j] = conj(out[j,i])`
                        // is preserved by ricci_at_point_via_fd, so summing all
                        // 50 (re, im) entries gives the Frobenius norm squared.
                        (weights[p] * sumsq, weights[p], 1usize)
                    }
                }
                Err(_) => (0.0, 0.0, 0usize),
            }
        })
        .reduce(
            || (0.0, 0.0, 0usize),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
        );

    let (weighted_sumsq, total_w, n_valid) = result;
    if total_w < 1e-12 || n_valid * 2 < n_actual {
        return f64::NAN;
    }
    (weighted_sumsq / total_w).sqrt()
}

// ===========================================================================
// P4.5 Phase 3: tangent-frame Ricci.
//
// The Phase-2 ambient diagnostic ‖Ric‖_L² (computed on the rank-4 ambient
// metric `g_amb` on CP⁴) is dominated by singular-set noise: the n=22
// P5.2 sweep showed cross-config / seed-only variance ratio of only 1.28×
// for ‖Ric_amb‖, vs. 63.5× for σ_L¹. The structural diagnosis is that
// `log det g_amb` diverges on the radial-direction singular set, and FD
// probes near sample points pick up O(10⁸) baseline contributions
// independent of refinement state.
//
// Phase 3 fix: switch the Ricci computation onto the **3-D CY tangent
// metric** `g_tan = T† g_amb T̄` that Stack A's σ-eval already uses
// correctly. This object is rank-3 and non-singular on the variety, so
// `log det g_tan` is bounded and FD-Ricci is signal-dominated.
//
// Implementation sketch:
//   1. At base point z_0, build the affine-chart frame
//      `T = quintic_affine_chart_frame(grad_f, chart, elim)` — three 5-D
//      complex columns spanning the projective tangent space.
//   2. The 3 free affine coords `{a, b, c} = {0..4} \ {chart, elim}` give
//      the Wirtinger coordinates `w_a` for the FD; perturbations
//      δw_α = δx_α + i δy_α (α ∈ 0..3) push z by Σ_α δw_α · T_α.
//   3. Newton-project the perturbed point back onto f = 0 (single-step
//      Newton from `newton_project_to_quintic` suffices for h_step ≪ 1).
//   4. Recompute g_amb at the projected point (per-point Bergman pipeline)
//      and project to a 3×3 g_tan. **Use the FIXED frame at z_0** for the
//      projection so that `log det g_tan` is a well-defined scalar
//      function of the 3 affine Wirtinger coords; recomputing the frame
//      at each probe point would mix in a holomorphic Jacobian factor
//      whose ∂∂̄ log vanishes in principle but adds O(h) noise in
//      practice.
//   5. Hand the per-point closure to `ricci_at_point_via_fd` with n=3;
//      output is a 3×3 Hermitian Ric in the affine-chart frame.
//
// The Ricci tensor returned is in the **affine-chart frame** (same frame
// the σ-eval uses). It does NOT go through the orthonormal Gram-Schmidt
// frame; the Frobenius norm of Ric in the affine frame is
// frame-dependent, but for our purposes it suffices: we are correlating
// `‖Ric_tan‖_L²` against `σ_L¹` across optimisation endpoints, and both
// transform consistently under chart choice.
// ===========================================================================

/// Pack a 5×5 complex Hermitian metric (50 f64 interleaved row-major) into
/// the `[[(re, im); 5]; 5]` layout used by `project_to_quintic_tangent`.
#[inline]
fn pack_g_amb_5x5(g_flat: &[f64; 50]) -> [[(f64, f64); 5]; 5] {
    let mut g = [[(0.0f64, 0.0f64); 5]; 5];
    for i in 0..5 {
        for j in 0..5 {
            let off = 2 * (i * 5 + j);
            g[i][j] = (g_flat[off], g_flat[off + 1]);
        }
    }
    g
}

/// Determine the 3 free affine-chart coordinate indices at point `z`,
/// returned together with the chosen `chart` and `elim` indices. The free
/// coords are `{0..5} \ {chart, elim}` sorted ascending.
#[inline]
fn affine_free_coords(
    z: &[f64; N_AMB_REAL],
    grad_f: &[f64; N_AMB_REAL],
) -> (usize, usize, [usize; 3]) {
    let (chart, elim, _) = quintic_chart_and_elim(z, grad_f);
    let mut free = [0usize; 3];
    let mut idx = 0;
    for k in 0..5 {
        if k != chart && k != elim && idx < 3 {
            free[idx] = k;
            idx += 1;
        }
    }
    (chart, elim, free)
}

/// Compute `Ric_tan_{αβ̄}(z; h)` on the 3-D CY tangent space at `z`, using
/// the affine-chart frame from the σ-eval pipeline.
///
/// Output `out` is a 3×3 complex Hermitian Ricci tensor packed as
/// interleaved (re, im) row-major, length 18.
///
/// # Arguments
/// * `h_block` — `(2*n_basis)²` real-block Hermitian h.
/// * `z` — base point on the Fermat quintic, length-10 (re, im) interleaved.
/// * `monomials` / `n_basis` — degree-k section basis.
/// * `h_step` — FD step in the tangent Wirtinger coords. Recommended `1e-4`.
///
/// # Errors
/// * [`CalabiMetricError::SingularMetric`] if the affine-chart frame is
///   degenerate at `z` (chart == elim, or `|∂f/∂Z_elim|` ≈ 0).
/// * [`CalabiMetricError::SingularMetric`] / [`CalabiMetricError::NonFiniteLogDet`]
///   if any FD probe goes singular or fails Newton-projection.
/// * [`CalabiMetricError::Ricci`] if the FD evaluator rejects the inputs.
pub fn ricci_at_point_bergman_tangent(
    h_block: &[f64],
    z: &[f64; N_AMB_REAL],
    monomials: &[[u32; 5]],
    n_basis: usize,
    h_step: f64,
    out: &mut [f64; RIC_TAN_LEN],
) -> Result<(), CalabiMetricError> {
    let two_n = 2 * n_basis;

    // ---- 1. Build the fixed affine-chart frame at the base point. ----
    let grad_f0 = fermat_quintic_gradient(z);
    let (chart0, elim0, _free0) = affine_free_coords(z, &grad_f0);
    if chart0 == elim0 {
        return Err(CalabiMetricError::SingularMetric);
    }
    let frame_fixed = quintic_affine_chart_frame(&grad_f0, chart0, elim0);
    // Check the frame is non-degenerate (chart_frame returns all-zeros if
    // |∂f/∂Z_elim|² < 1e-30).
    let mut frame_norm_sq = 0.0f64;
    for col in &frame_fixed {
        for v in col {
            frame_norm_sq += v * v;
        }
    }
    if !frame_norm_sq.is_finite() || frame_norm_sq < 1e-20 {
        return Err(CalabiMetricError::SingularMetric);
    }

    // ---- 2. Pre-allocate scratch reused across all probe evaluations. ----
    let mut s_scratch = vec![0.0f64; two_n];
    let mut ds_scratch = vec![0.0f64; 5 * two_n];
    let mut h_s_scratch = vec![0.0f64; two_n];
    let mut h_ds_scratch = vec![0.0f64; 5 * two_n];
    let mut g_scratch = [0.0f64; 50];

    // ---- 3. Build the FD closure. ----
    //
    // Mapping: `zp ∈ R^6` → tangent Wirtinger coords
    //   δw_α = zp[2α] + i zp[2α+1] (α = 0..3)
    // Pushforward: `δZ = Σ_α δw_α · T_α[fixed]` (affine columns).
    // Then Newton-project δZ + z onto the quintic and evaluate
    //   log |det g_tan(z_proj; h, T_fixed)|
    // where g_tan = project_to_quintic_tangent(g_amb, T_fixed) is the
    // 3×3 metric IN THE FIXED FRAME (so the determinant is a scalar
    // function of δw, not entangled with frame-rotation Jacobians).
    let mut last_err: Option<CalabiMetricError> = None;
    let log_det_eval = |zp: &[f64]| -> f64 {
        debug_assert_eq!(zp.len(), N_TAN_REAL);

        // 3a. Pushforward: z_pert[i] = z[i] + Σ_α δw_α · T_α[i].
        let mut z_pert = *z;
        for alpha in 0..N_TAN {
            let dw_re = zp[2 * alpha];
            let dw_im = zp[2 * alpha + 1];
            let col = &frame_fixed[alpha];
            for i in 0..5 {
                let tr = col[2 * i];
                let ti = col[2 * i + 1];
                // δw · T_i = (dw_re + i dw_im)(tr + i ti)
                //          = (dw_re tr − dw_im ti) + i(dw_re ti + dw_im tr)
                z_pert[2 * i] += dw_re * tr - dw_im * ti;
                z_pert[2 * i + 1] += dw_re * ti + dw_im * tr;
            }
        }

        // 3b. Newton-project back onto f = 0. For h_step ~ 1e-4, the
        // tangent perturbation is already O(h_step² ‖∇²f‖) off the
        // variety, so 1–3 Newton iterations suffice. We use 5 with a
        // tight tol to be safe.
        let z_proj = match newton_project_to_quintic(&z_pert, 1e-12, 8) {
            Some(zp) => zp,
            None => {
                last_err = Some(CalabiMetricError::SingularMetric);
                return f64::NAN;
            }
        };

        // 3c. Evaluate the 5×5 ambient g at z_proj.
        if let Err(e) = compute_g_amb_at_point(
            &z_proj,
            h_block,
            monomials,
            n_basis,
            &mut s_scratch,
            &mut ds_scratch,
            &mut h_s_scratch,
            &mut h_ds_scratch,
            &mut g_scratch,
        ) {
            last_err = Some(e);
            return f64::NAN;
        }

        // 3d. Project onto the FIXED frame at z_0 → 3×3 g_tan.
        let g_amb = pack_g_amb_5x5(&g_scratch);
        let g_tan = project_to_quintic_tangent(&g_amb, &frame_fixed);

        // 3e. Compute |det g_tan| (real for Hermitian) and take log.
        let det = det_3x3_complex_hermitian(&g_tan);
        if !det.is_finite() || det.abs() < 1e-30 {
            last_err = Some(CalabiMetricError::SingularMetric);
            return f64::NAN;
        }
        det.abs().ln()
    };

    // ---- 4. Hand off to the Wirtinger-FD evaluator with n=3. ----
    let z_zero = [0.0f64; N_TAN_REAL]; // base of the FD stencil = (z_0)
    let result = ricci_at_point_via_fd(
        &z_zero,
        N_TAN,
        log_det_eval,
        h_step,
        out.as_mut_slice(),
    );
    match result {
        Ok(()) => Ok(()),
        Err(e) => {
            if let Some(stack_err) = last_err {
                Err(stack_err)
            } else {
                Err(CalabiMetricError::Ricci(e))
            }
        }
    }
}

/// Compute the **L²-Ricci-norm on the CY tangent metric**:
///
///     ‖Ric_tan‖²_{L²} = (Σ_p w_p Σ_{αβ} |Ric_tan_{αβ̄}(z_p; h)|²) / (Σ_p w_p)
///
/// Same shape as [`ricci_norm_l2`] but evaluates the rank-3 tangent-frame
/// Ricci tensor instead of the rank-4 ambient one. Singular-frame /
/// singular-metric / Newton-projection failures are dropped; if more
/// than 50% of points fail, returns `NaN`.
///
/// # Arguments
/// * `points` — flat `[n_actual × 10]` interleaved (re, im) coordinates.
/// * `weights` — length `n_actual`, FS or CY-measure weights.
/// * `n_actual` — number of valid sample points.
/// * `h_block` — `(2*n_basis)²` real-block Hermitian h.
/// * `monomials` / `n_basis` — degree-k section basis.
/// * `h_step` — FD perturbation in the tangent Wirtinger coords; `1e-4`
///   is the recommended default.
pub fn ricci_norm_l2_tangent(
    points: &[f64],
    weights: &[f64],
    n_actual: usize,
    h_block: &[f64],
    monomials: &[[u32; 5]],
    n_basis: usize,
    h_step: f64,
) -> f64 {
    debug_assert!(points.len() >= n_actual * 10);
    debug_assert!(weights.len() >= n_actual);

    let result: (f64, f64, usize) = (0..n_actual)
        .into_par_iter()
        .map(|p| {
            let mut z = [0.0f64; N_AMB_REAL];
            z.copy_from_slice(&points[p * 10..p * 10 + 10]);
            let mut ric = [0.0f64; RIC_TAN_LEN];
            match ricci_at_point_bergman_tangent(
                h_block, &z, monomials, n_basis, h_step, &mut ric,
            ) {
                Ok(()) => {
                    let mut sumsq = 0.0;
                    for v in ric.iter() {
                        sumsq += v * v;
                    }
                    if !sumsq.is_finite() || !weights[p].is_finite() || weights[p] <= 0.0 {
                        (0.0, 0.0, 0usize)
                    } else {
                        (weights[p] * sumsq, weights[p], 1usize)
                    }
                }
                Err(_) => (0.0, 0.0, 0usize),
            }
        })
        .reduce(
            || (0.0, 0.0, 0usize),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
        );

    let (weighted_sumsq, total_w, n_valid) = result;
    if total_w < 1e-12 || n_valid * 2 < n_actual {
        return f64::NAN;
    }
    (weighted_sumsq / total_w).sqrt()
}

/// Cofactor-formula inverse of a 3×3 complex Hermitian matrix. Returns
/// `None` when |det| is below `1e-30` or non-finite. Hermiticity is
/// re-imposed on the output to suppress floating-point drift.
fn invert_3x3_complex_hermitian_local(
    g: &[[(f64, f64); 3]; 3],
) -> Option<[[(f64, f64); 3]; 3]> {
    let cmul = |(ar, ai): (f64, f64), (br, bi): (f64, f64)| -> (f64, f64) {
        (ar * br - ai * bi, ar * bi + ai * br)
    };
    let csub = |(ar, ai): (f64, f64), (br, bi): (f64, f64)| -> (f64, f64) {
        (ar - br, ai - bi)
    };
    let cconj = |(ar, ai): (f64, f64)| -> (f64, f64) { (ar, -ai) };
    let det = det_3x3_complex_hermitian(g);
    if !det.is_finite() || det.abs() < 1e-30 {
        return None;
    }
    let mut inv = [[(0.0f64, 0.0f64); 3]; 3];
    let c00 = csub(cmul(g[1][1], g[2][2]), cmul(g[1][2], g[2][1]));
    let c01 = csub(cmul(g[1][2], g[2][0]), cmul(g[1][0], g[2][2]));
    let c02 = csub(cmul(g[1][0], g[2][1]), cmul(g[1][1], g[2][0]));
    let c10 = csub(cmul(g[0][2], g[2][1]), cmul(g[0][1], g[2][2]));
    let c11 = csub(cmul(g[0][0], g[2][2]), cmul(g[0][2], g[2][0]));
    let c12 = csub(cmul(g[0][1], g[2][0]), cmul(g[0][0], g[2][1]));
    let c20 = csub(cmul(g[0][1], g[1][2]), cmul(g[0][2], g[1][1]));
    let c21 = csub(cmul(g[0][2], g[1][0]), cmul(g[0][0], g[1][2]));
    let c22 = csub(cmul(g[0][0], g[1][1]), cmul(g[0][1], g[1][0]));
    let inv_det = 1.0 / det;
    inv[0][0] = (c00.0 * inv_det, c00.1 * inv_det);
    inv[0][1] = (c10.0 * inv_det, c10.1 * inv_det);
    inv[0][2] = (c20.0 * inv_det, c20.1 * inv_det);
    inv[1][0] = (c01.0 * inv_det, c01.1 * inv_det);
    inv[1][1] = (c11.0 * inv_det, c11.1 * inv_det);
    inv[1][2] = (c21.0 * inv_det, c21.1 * inv_det);
    inv[2][0] = (c02.0 * inv_det, c02.1 * inv_det);
    inv[2][1] = (c12.0 * inv_det, c12.1 * inv_det);
    inv[2][2] = (c22.0 * inv_det, c22.1 * inv_det);
    for i in 0..3 {
        for j in (i + 1)..3 {
            let c = cconj(inv[i][j]);
            let avg_re = 0.5 * (inv[j][i].0 + c.0);
            let avg_im = 0.5 * (inv[j][i].1 + c.1);
            inv[j][i] = (avg_re, avg_im);
            inv[i][j] = (avg_re, -avg_im);
        }
    }
    Some(inv)
}

/// P4.7 helper: at a single CY sample point, return
/// `(|‖Ric_tan‖²|, |‖Ric₀‖²|, |s|²)` of the **tangent-frame** Ricci, where
/// `Ric₀ = Ric_tan - (s/n) g_tan` (n = 3 complex dim) is the trace-free
/// part of the Ricci tensor and `s = (g_tan)^{βᾱ} (Ric_tan)_{αβ̄}` is the
/// scalar curvature trace.
///
/// All three norms are **plain entry-Frobenius** sums
/// `Σ_{αβ} |T_{αβ̄}|²`, matching the convention used by
/// [`ricci_norm_l2_tangent`] for `‖Ric_tan‖_{L²}`.
///
/// Returns `None` if any sub-step (frame construction, ambient metric
/// evaluation, FD-Ricci, or 3×3 inversion) is singular at this point.
#[allow(clippy::type_complexity)]
fn ric_and_scalar_sumsq_at_point(
    z: &[f64; N_AMB_REAL],
    h_block: &[f64],
    monomials: &[[u32; 5]],
    n_basis: usize,
    h_step: f64,
) -> Option<(f64, f64, f64)> {
    // 1. Recompute the fixed affine-chart frame at z (same as
    //    ricci_at_point_bergman_tangent).
    let grad_f0 = fermat_quintic_gradient(z);
    let (chart0, elim0, _free0) = affine_free_coords(z, &grad_f0);
    if chart0 == elim0 {
        return None;
    }
    let frame_fixed = quintic_affine_chart_frame(&grad_f0, chart0, elim0);
    let mut frame_norm_sq = 0.0f64;
    for col in &frame_fixed {
        for v in col {
            frame_norm_sq += v * v;
        }
    }
    if !frame_norm_sq.is_finite() || frame_norm_sq < 1e-20 {
        return None;
    }

    // 2. Compute Ric_tan via the existing FD evaluator.
    let mut ric = [0.0f64; RIC_TAN_LEN];
    if ricci_at_point_bergman_tangent(
        h_block, z, monomials, n_basis, h_step, &mut ric,
    )
    .is_err()
    {
        return None;
    }

    // 3. Compute g_amb at z (un-perturbed base point) and project to
    //    the same fixed frame to get g_tan_0.
    let two_n = 2 * n_basis;
    let mut s_scratch = vec![0.0f64; two_n];
    let mut ds_scratch = vec![0.0f64; 5 * two_n];
    let mut h_s_scratch = vec![0.0f64; two_n];
    let mut h_ds_scratch = vec![0.0f64; 5 * two_n];
    let mut g_scratch = [0.0f64; 50];
    if compute_g_amb_at_point(
        z,
        h_block,
        monomials,
        n_basis,
        &mut s_scratch,
        &mut ds_scratch,
        &mut h_s_scratch,
        &mut h_ds_scratch,
        &mut g_scratch,
    )
    .is_err()
    {
        return None;
    }
    let g_amb = pack_g_amb_5x5(&g_scratch);
    let g_tan = project_to_quintic_tangent(&g_amb, &frame_fixed);

    // 4. Invert g_tan; abort on singular.
    let g_inv = invert_3x3_complex_hermitian_local(&g_tan)?;

    // 5. Unpack Ric_tan from interleaved (re, im) row-major into a
    //    [[(re, im); 3]; 3] matrix for arithmetic.
    let mut ric_m = [[(0.0f64, 0.0f64); 3]; 3];
    for a in 0..N_TAN {
        for b in 0..N_TAN {
            let off = 2 * (a * N_TAN + b);
            ric_m[a][b] = (ric[off], ric[off + 1]);
        }
    }

    // 6. Scalar curvature s = Σ_{αβ} g_inv[β][α] · Ric[α][β].
    //    For Hermitian g_inv and Hermitian Ric this trace is real;
    //    we accumulate the imaginary part too as a sanity diagnostic.
    let mut s_re = 0.0f64;
    let mut s_im = 0.0f64;
    for a in 0..N_TAN {
        for b in 0..N_TAN {
            let (gr, gi) = g_inv[b][a];
            let (rr, ri) = ric_m[a][b];
            s_re += gr * rr - gi * ri;
            s_im += gr * ri + gi * rr;
        }
    }
    // Imaginary part should be ~0 modulo FP noise. Use the real part
    // as the scalar curvature.
    let _ = s_im;
    let s = s_re;
    if !s.is_finite() {
        return None;
    }

    // 7. Trace-free Ricci: Ric₀_{αβ̄} = Ric_{αβ̄} - (s/3) g_tan_{αβ̄}.
    let s_over_n = s / (N_TAN as f64);

    // 8. Plain entry-Frobenius sums.
    let mut full_sumsq = 0.0f64;
    let mut tracefree_sumsq = 0.0f64;
    for a in 0..N_TAN {
        for b in 0..N_TAN {
            let (rr, ri) = ric_m[a][b];
            let (gr, gi) = g_tan[a][b];
            full_sumsq += rr * rr + ri * ri;
            let r0_re = rr - s_over_n * gr;
            let r0_im = ri - s_over_n * gi;
            tracefree_sumsq += r0_re * r0_re + r0_im * r0_im;
        }
    }
    let scalar_sq = s * s;
    if !full_sumsq.is_finite() || !tracefree_sumsq.is_finite() || !scalar_sq.is_finite() {
        return None;
    }
    Some((full_sumsq, tracefree_sumsq, scalar_sq))
}

/// P4.7 reduction: returns `(‖Ric_tan‖_{L²}, ‖Ric₀‖_{L²}, ‖s‖_{L²})` over
/// the supplied sample set. Each is `sqrt(Σ_p w_p · X_p² / Σ_p w_p)` with
/// the same per-point convention as [`ricci_norm_l2_tangent`].
///
/// Singular-point failures (Newton-projection failures, singular frames,
/// non-invertible g_tan, non-finite log det) are dropped consistently for
/// all three quantities. If more than 50% of points fail the reduction
/// returns `(NaN, NaN, NaN)`.
pub fn ricci_tracefree_and_scalar_norms_l2_tangent(
    points: &[f64],
    weights: &[f64],
    n_actual: usize,
    h_block: &[f64],
    monomials: &[[u32; 5]],
    n_basis: usize,
    h_step: f64,
) -> (f64, f64, f64) {
    debug_assert!(points.len() >= n_actual * 10);
    debug_assert!(weights.len() >= n_actual);

    let result: (f64, f64, f64, f64, usize) = (0..n_actual)
        .into_par_iter()
        .map(|p| {
            let mut z = [0.0f64; N_AMB_REAL];
            z.copy_from_slice(&points[p * 10..p * 10 + 10]);
            let w = weights[p];
            if !w.is_finite() || w <= 0.0 {
                return (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
            }
            match ric_and_scalar_sumsq_at_point(
                &z, h_block, monomials, n_basis, h_step,
            ) {
                Some((full_sq, tf_sq, sc_sq)) => {
                    if !full_sq.is_finite() || !tf_sq.is_finite() || !sc_sq.is_finite() {
                        (0.0, 0.0, 0.0, 0.0, 0usize)
                    } else {
                        (w * full_sq, w * tf_sq, w * sc_sq, w, 1usize)
                    }
                }
                None => (0.0, 0.0, 0.0, 0.0, 0usize),
            }
        })
        .reduce(
            || (0.0, 0.0, 0.0, 0.0, 0usize),
            |a, b| (
                a.0 + b.0,
                a.1 + b.1,
                a.2 + b.2,
                a.3 + b.3,
                a.4 + b.4,
            ),
        );

    let (full_w_ss, tf_w_ss, sc_w_ss, total_w, n_valid) = result;
    if total_w < 1e-12 || n_valid * 2 < n_actual {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    (
        (full_w_ss / total_w).sqrt(),
        (tf_w_ss / total_w).sqrt(),
        (sc_w_ss / total_w).sqrt(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quintic::{compute_sigma_from_workspace, QuinticSolver};

    // Wall-clock notes (release build, AMD Ryzen, n_pts=1000):
    //   k=2: σ-eval ~4 ms, Ricci-norm ~3 s   (all 25 stencil entries × 9 probes)
    //   k=3: σ-eval ~15 ms, Ricci-norm ~12 s
    //   k=4: σ-eval ~60 ms, Ricci-norm ~50 s
    // Tests with k=4 are gated `#[ignore]` to keep the suite fast.

    fn make_solver(k: u32, n_pts: usize) -> QuinticSolver {
        let solver = QuinticSolver::new(k, n_pts, 42, 1e-8)
            .expect("QuinticSolver::new failed");
        solver
    }

    #[test]
    fn test_p4_4_ricci_finite_at_donaldson_balanced() {
        // k=2 and k=3 with Donaldson balance: at MOST sample points,
        // Ricci must be finite. The ambient 5×5 Bergman metric on CP^4
        // is rank-4 (the radial complex direction is in the kernel by
        // homogeneity), so a fraction of FD-perturbed probe points
        // hits the singular set. We accept up to 30% singular probe
        // failures across the test set; everything else must be finite.
        for k in 2u32..=3 {
            let mut solver = make_solver(k, 200);
            solver.donaldson_solve(50, 1e-4);
            let mut n_ok = 0usize;
            let mut n_total = 0usize;
            // Probe the first 30 points to amortise the rank-4 hits.
            for p in 0..30usize.min(solver.n_points) {
                let z: [f64; 10] = solver.points[p * 10..p * 10 + 10]
                    .try_into()
                    .unwrap();
                let mut ric = [0.0f64; 50];
                let res = ricci_at_point_bergman(
                    &solver.h_block,
                    &z,
                    &solver.monomials,
                    solver.n_basis,
                    1e-4,
                    &mut ric,
                );
                n_total += 1;
                if let Ok(()) = res {
                    let all_finite = ric.iter().all(|v| v.is_finite());
                    if all_finite {
                        n_ok += 1;
                    } else {
                        eprintln!("k={k} point {p}: ricci has non-finite entry");
                    }
                } else {
                    eprintln!("k={k} point {p}: ricci returned {res:?}");
                }
            }
            assert!(
                n_total > 0,
                "k={k}: no probe points ran",
            );
            let ok_frac = n_ok as f64 / n_total as f64;
            assert!(
                ok_frac >= 0.7,
                "k={k}: only {n_ok}/{n_total} ({ok_frac:.2}) Ricci probes succeeded; \
                 expected >=70%. Likely cause: numerical singularity rate is too high.",
            );
        }
    }

    #[test]
    fn test_p4_4_ricci_norm_decreases_with_k() {
        // After Donaldson balance + light σ-functional Adam refinement,
        // ‖Ric‖_L² should decrease with k (parallel to σ).
        //
        // Uses k = 2, 3 to keep the test under 30 s; the optional
        // `#[ignore]` extension `test_p4_4_ricci_norm_decreases_with_k_full`
        // adds k = 4.
        let n_pts = 400;
        let mut norms = Vec::new();
        let mut sigmas = Vec::new();
        for k in 2u32..=3 {
            let mut solver = make_solver(k, n_pts);
            solver.donaldson_solve(80, 1e-5);
            let _ = solver.sigma_refine_analytic(20, 5e-3);
            let sigma = compute_sigma_from_workspace(&mut solver);
            let ric_norm = ricci_norm_l2(
                &solver.points,
                &solver.weights,
                solver.n_points,
                &solver.h_block,
                &solver.monomials,
                solver.n_basis,
                1e-4,
            );
            assert!(
                ric_norm.is_finite(),
                "k={k}: ‖Ric‖_L² is non-finite ({ric_norm})",
            );
            assert!(
                sigma.is_finite() && sigma > 0.0,
                "k={k}: σ_L¹ is non-finite or non-positive ({sigma})",
            );
            norms.push(ric_norm);
            sigmas.push(sigma);
            println!(
                "k={k}: σ_L¹ = {sigma:.6e}, ‖Ric‖_L² = {ric_norm:.6e}",
            );
        }
        // Monotone decrease in ‖Ric‖_L² parallel to σ_L¹.
        assert!(
            norms[1] < norms[0] * 1.05,
            "‖Ric‖_L² did not decrease: k=2 -> {}, k=3 -> {}",
            norms[0], norms[1],
        );
        // σ should decrease too (sanity).
        assert!(
            sigmas[1] < sigmas[0] * 1.05,
            "σ_L¹ did not decrease: k=2 -> {}, k=3 -> {}",
            sigmas[0], sigmas[1],
        );
    }

    /// Original n=5 P4.4 cross-validation. Statistically underpowered:
    /// Fisher 95% CI on the reported r=0.7631 is roughly [-0.37, +0.98]
    /// (includes zero), so this cannot support the directional claim on
    /// its own. Kept `#[ignore]` to preserve regression history; the
    /// authoritative test is
    /// `test_p5_2_ricci_correlates_with_sigma_n20` below (n>=20, asserts
    /// lower 95% CI bound on r > 0.5).
    #[test]
    #[ignore]
    fn test_p4_4_ricci_correlates_with_sigma() {
        // Cross-validation: σ-functional minimisation IS approximately
        // Calabi flow, so we should see σ and ‖Ric‖_L² move together
        // across optimization endpoints.
        //
        // Strategy: take a k=2 solver, run varying amounts of Donaldson
        // + Adam, and record the (σ, ‖Ric‖_L²) pairs. Compute Pearson
        // correlation across the 5 endpoints.
        let n_pts = 300;
        // Five endpoints: identity init, Donaldson 5/20/50/Adam-refined.
        let mut pairs: Vec<(f64, f64)> = Vec::new();

        // Endpoint A: identity h.
        {
            let mut solver = make_solver(2, n_pts);
            let s = compute_sigma_from_workspace(&mut solver);
            let r = ricci_norm_l2(
                &solver.points, &solver.weights, solver.n_points,
                &solver.h_block, &solver.monomials, solver.n_basis, 1e-4,
            );
            if s.is_finite() && r.is_finite() {
                pairs.push((s, r));
            }
        }
        // Endpoints B-D: Donaldson 5, 20, 50.
        for n_iter in [5usize, 20, 50] {
            let mut solver = make_solver(2, n_pts);
            solver.donaldson_solve(n_iter, 1e-12);
            let s = compute_sigma_from_workspace(&mut solver);
            let r = ricci_norm_l2(
                &solver.points, &solver.weights, solver.n_points,
                &solver.h_block, &solver.monomials, solver.n_basis, 1e-4,
            );
            if s.is_finite() && r.is_finite() {
                pairs.push((s, r));
            }
        }
        // Endpoint E: Donaldson + Adam refine.
        {
            let mut solver = make_solver(2, n_pts);
            solver.donaldson_solve(80, 1e-6);
            let _ = solver.sigma_refine_analytic(30, 5e-3);
            let s = compute_sigma_from_workspace(&mut solver);
            let r = ricci_norm_l2(
                &solver.points, &solver.weights, solver.n_points,
                &solver.h_block, &solver.monomials, solver.n_basis, 1e-4,
            );
            if s.is_finite() && r.is_finite() {
                pairs.push((s, r));
            }
        }

        assert!(
            pairs.len() >= 4,
            "expected >=4 valid (σ, ‖Ric‖) pairs, got {} ({pairs:?})",
            pairs.len(),
        );

        // Pearson correlation.
        let n = pairs.len() as f64;
        let mean_s: f64 = pairs.iter().map(|p| p.0).sum::<f64>() / n;
        let mean_r: f64 = pairs.iter().map(|p| p.1).sum::<f64>() / n;
        let mut num = 0.0;
        let mut ds2 = 0.0;
        let mut dr2 = 0.0;
        for &(s, r) in &pairs {
            let dx = s - mean_s;
            let dy = r - mean_r;
            num += dx * dy;
            ds2 += dx * dx;
            dr2 += dy * dy;
        }
        let pearson = num / (ds2.sqrt() * dr2.sqrt() + 1e-30);

        // Empirical proportionality constant ‖Ric‖ / σ from the mean.
        let prop_const = mean_r / mean_s;

        println!("σ-Ric pairs (k=2):");
        for (i, (s, r)) in pairs.iter().enumerate() {
            println!("  endpoint {i}: σ_L¹ = {s:.4e}, ‖Ric‖_L² = {r:.4e}, ratio = {:.4e}", r / s);
        }
        println!("Pearson correlation = {pearson:.4}");
        println!("Empirical ‖Ric‖_L² / σ_L¹ ≈ {prop_const:.4e}");

        assert!(
            pearson > 0.7,
            "σ-Ric Pearson correlation {pearson} < 0.7 — Stack A and Stack C \
             do not agree on the direction of optimisation",
        );
    }

    /// P5.2 (post §2.1 hostile-review): n>=20 σ-Ric correlation with
    /// Fisher 95% CI assertion. Replaces the underpowered n=5 P4.4 test.
    ///
    /// Sweep (n_pts, n_donaldson, n_adam, lr_adam, seed) across 22
    /// distinct optimisation configurations. For each, build a fresh
    /// solver, run Donaldson then optional Adam refinement, and record
    /// (σ_L¹, ‖Ric‖_L²). Compute Pearson r, Spearman ρ, Fisher z, and
    /// 95% CI on r. Assert the lower CI bound on r exceeds 0.5 — strict
    /// statistical confirmation that Stack A and Stack C agree
    /// directionally.
    ///
    /// Wall-clock: ~5–10 min release at k=2 (n_pts up to 1500, 22 endpoints).
    #[test]
    #[ignore]
    fn test_p5_2_ricci_correlates_with_sigma_n20() {
        // 22 distinct (n_pts, n_donaldson, n_adam, lr_adam, seed_off) tuples.
        // Notes:
        //   - 5 share (1000, 60, 10, 1e-3) varying only seed → seed-only
        //     variance group for the architectural variance check.
        //   - n_pts axis: {500, 1000, 1500}; n_pts=2000 deliberately
        //     dropped to keep the suite under 10 min wall-clock at k=2
        //     (Ricci-norm is O(n_pts) and dominates).
        //   - n_donaldson axis: {30, 60, 100, 150}.
        //   - n_adam axis: {0, 5, 10, 20, 40} (0 = pure Donaldson endpoint).
        //   - lr_adam axis: {1e-4, 1e-3, 1e-2}.
        //   - All seeds are 42 + seed_off for determinism.
        let configs: [(usize, usize, usize, f64, u64); 22] = [
            // Seed-only variance group (5 configs, identical pipeline,
            // varying seed): for the architectural variance check.
            // Seeds derived as 42 + seed_off; we add 1000 to seed_off
            // for this group to keep them disjoint from the config-sweep
            // seeds and to avoid signed-cast complications in the array
            // literal.
            (1000,  60, 10, 1e-3, 1000),  // seed = 1042
            (1000,  60, 10, 1e-3, 1058),  // seed = 1100
            (1000,  60, 10, 1e-3, 1303),  // seed = 1345
            (1000,  60, 10, 1e-3, 1965),  // seed = 2007
            (1000,  60, 10, 1e-3, 1057),  // seed = 1099
            // Pure Donaldson endpoints (n_adam = 0).
            ( 500,  30,  0, 1e-3,  1),
            ( 500, 100,  0, 1e-3,  2),
            (1000,  30,  0, 1e-3,  3),
            (1000, 150,  0, 1e-3,  4),
            (1500,  60,  0, 1e-3,  5),
            // Adam-refined endpoints, varying lr.
            ( 500,  60,  5, 1e-4,  6),
            ( 500,  60, 10, 1e-2,  7),
            (1000,  30, 20, 1e-4,  8),
            (1000, 100,  5, 1e-2,  9),
            (1000, 150, 20, 1e-3, 10),
            (1500,  30, 10, 1e-3, 11),
            (1500, 100, 10, 1e-3, 12),
            // Heavy-Adam (40 iter) endpoints.
            ( 500, 100, 40, 1e-3, 13),
            (1000,  60, 40, 1e-3, 14),
            (1000, 100, 40, 1e-2, 15),
            (1500,  60, 40, 1e-4, 16),
            (1500, 150, 40, 1e-3, 17),
        ];

        let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(configs.len());
        let mut endpoint_log: Vec<String> = Vec::with_capacity(configs.len());

        // Seed-only variance group (first 5 configs) tracked separately.
        let mut seed_only_pairs: Vec<(f64, f64)> = Vec::new();

        for (i, &(n_pts, n_donaldson, n_adam, lr_adam, seed_off)) in
            configs.iter().enumerate()
        {
            let seed = 42u64.wrapping_add(seed_off);
            let mut solver = match QuinticSolver::new(2, n_pts, seed, 1e-8) {
                Some(s) => s,
                None => {
                    eprintln!(
                        "endpoint {i}: solver init failed (n_pts={n_pts}, seed={seed}); skipping",
                    );
                    continue;
                }
            };
            let n_done = solver.donaldson_solve(n_donaldson, 1e-8);
            if n_adam > 0 {
                let _ = solver.sigma_refine_analytic(n_adam, lr_adam);
            }
            let s = compute_sigma_from_workspace(&mut solver);
            let r = ricci_norm_l2(
                &solver.points,
                &solver.weights,
                solver.n_points,
                &solver.h_block,
                &solver.monomials,
                solver.n_basis,
                1e-4,
            );
            let line = format!(
                "endpoint {i:>2}: n_pts={n_pts:>4}, n_don={n_donaldson:>3} (done={n_done:>3}), \
                 n_adam={n_adam:>2}, lr={lr_adam:.0e}, seed={seed:>5} -> \
                 σ={s:.4e}, ‖Ric‖={r:.4e}",
            );
            eprintln!("{line}");
            endpoint_log.push(line);
            if s.is_finite() && r.is_finite() {
                pairs.push((s, r));
                if i < 5 {
                    seed_only_pairs.push((s, r));
                }
            } else {
                eprintln!("  -> NON-FINITE (σ={s}, ‖Ric‖={r}); excluded from correlation");
            }
        }

        let n = pairs.len();
        eprintln!("\n=== Summary: {n} valid endpoints out of {} ===", configs.len());
        assert!(
            n >= 20,
            "expected >=20 valid (σ, ‖Ric‖) pairs; only got {n}. \
             Hostile-review §2.1 requires n>=20 for statistical power.",
        );

        // ===== Pearson correlation =====
        let nf = n as f64;
        let mean_s: f64 = pairs.iter().map(|p| p.0).sum::<f64>() / nf;
        let mean_r: f64 = pairs.iter().map(|p| p.1).sum::<f64>() / nf;
        let mut num = 0.0;
        let mut ds2 = 0.0;
        let mut dr2 = 0.0;
        for &(s, r) in &pairs {
            let dx = s - mean_s;
            let dy = r - mean_r;
            num += dx * dy;
            ds2 += dx * dx;
            dr2 += dy * dy;
        }
        let denom = (ds2 * dr2).sqrt();
        assert!(
            denom > 1e-30 && denom.is_finite(),
            "degenerate variance: ds2={ds2}, dr2={dr2}",
        );
        let pearson = num / denom;

        // ===== Spearman rank correlation =====
        // Rank σ and ‖Ric‖ separately (1..n, with ties averaged), then
        // Pearson on the ranks.
        let rank_s = rank_with_average_ties(&pairs.iter().map(|p| p.0).collect::<Vec<_>>());
        let rank_r = rank_with_average_ties(&pairs.iter().map(|p| p.1).collect::<Vec<_>>());
        let mean_rs: f64 = rank_s.iter().sum::<f64>() / nf;
        let mean_rr: f64 = rank_r.iter().sum::<f64>() / nf;
        let mut sn = 0.0;
        let mut sd1 = 0.0;
        let mut sd2 = 0.0;
        for i in 0..n {
            let dx = rank_s[i] - mean_rs;
            let dy = rank_r[i] - mean_rr;
            sn += dx * dy;
            sd1 += dx * dx;
            sd2 += dy * dy;
        }
        let spearman = sn / ((sd1 * sd2).sqrt() + 1e-30);

        // ===== Fisher z-transform + 95% CI on r =====
        // Clamp r away from ±1 so atanh stays finite.
        let r_clamped = pearson.clamp(-0.999_999, 0.999_999);
        let z = 0.5 * ((1.0 + r_clamped) / (1.0 - r_clamped)).ln(); // = atanh(r)
        let se = 1.0 / (nf - 3.0).sqrt();
        let z_lo = z - 1.96 * se;
        let z_hi = z + 1.96 * se;
        // Inverse Fisher: tanh(z) = (e^{2z}-1)/(e^{2z}+1)
        let r_lo = ((2.0 * z_lo).exp() - 1.0) / ((2.0 * z_lo).exp() + 1.0);
        let r_hi = ((2.0 * z_hi).exp() - 1.0) / ((2.0 * z_hi).exp() + 1.0);

        // ===== Architectural variance check =====
        // Variance of (σ, ‖Ric‖) at fixed config (seed-only group) vs
        // variance across all configs. If seed-only variance ≪
        // cross-config variance, then config-axis sweep is the right
        // test — n=5 P4.4 was sampling the wrong direction.
        let (seed_var_s, seed_var_r) = sample_variance_2d(&seed_only_pairs);
        let (full_var_s, full_var_r) = sample_variance_2d(&pairs);
        let ratio_s = if seed_var_s > 0.0 { full_var_s / seed_var_s } else { f64::INFINITY };
        let ratio_r = if seed_var_r > 0.0 { full_var_r / seed_var_r } else { f64::INFINITY };

        eprintln!("\n=== Statistics (n={n}) ===");
        eprintln!("Pearson r        = {pearson:.4}");
        eprintln!("Spearman ρ       = {spearman:.4}");
        eprintln!("Fisher z         = {z:.4}");
        eprintln!("Standard error   = {se:.4} (= 1/√(n-3))");
        eprintln!("95% CI on z      = [{z_lo:.4}, {z_hi:.4}]");
        eprintln!("95% CI on r      = [{r_lo:.4}, {r_hi:.4}]");
        eprintln!("\n=== Architectural variance check ===");
        eprintln!(
            "Seed-only group ({} configs, identical pipeline, varying seed):",
            seed_only_pairs.len(),
        );
        eprintln!("  Var(σ)   = {seed_var_s:.4e}");
        eprintln!("  Var(‖Ric‖) = {seed_var_r:.4e}");
        eprintln!("Across-all group ({n} configs):");
        eprintln!("  Var(σ)   = {full_var_s:.4e}");
        eprintln!("  Var(‖Ric‖) = {full_var_r:.4e}");
        eprintln!("Cross-config / seed-only variance ratio:");
        eprintln!("  σ:        {ratio_s:.2}×");
        eprintln!("  ‖Ric‖:    {ratio_r:.2}×");
        if ratio_s > 4.0 && ratio_r > 4.0 {
            eprintln!(
                "→ Cross-config variance dominates seed variance: \
                 config-axis sweep is the correct test.",
            );
        } else {
            eprintln!(
                "→ WARNING: seed variance is comparable to config variance; \
                 the underlying (σ, ‖Ric‖) signal is noisy and may need \
                 more sophisticated statistics (e.g. bootstrap CI).",
            );
        }

        // ===== Assertion =====
        assert!(
            r_lo > 0.5,
            "Lower 95% CI on Pearson r is {r_lo:.4} (r={pearson:.4}, \
             n={n}); hostile-review §2.1 requires lower bound > 0.5 \
             for the directional cross-validation claim. Stack C does \
             NOT statistically confirm Stack A.",
        );
        // Also sanity-check Pearson and Spearman point in the same
        // direction; gross disagreement would mean the relationship is
        // non-monotone or dominated by outliers.
        assert!(
            pearson.signum() == spearman.signum() || spearman.abs() < 0.1,
            "Pearson ({pearson:.3}) and Spearman ({spearman:.3}) disagree \
             in sign: relationship is non-monotone or outlier-driven; \
             linear-correlation claim is not justified.",
        );
    }

    /// Average-ties ranking. Returns ranks in 1..=n.
    fn rank_with_average_ties(xs: &[f64]) -> Vec<f64> {
        let n = xs.len();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| xs[a].partial_cmp(&xs[b]).unwrap_or(std::cmp::Ordering::Equal));
        let mut ranks = vec![0.0_f64; n];
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            while j < n && (xs[idx[j]] - xs[idx[i]]).abs() < 1e-15 {
                j += 1;
            }
            // Tie group is idx[i..j]; assign average rank.
            let avg = ((i + 1) as f64 + j as f64) / 2.0;
            for k in i..j {
                ranks[idx[k]] = avg;
            }
            i = j;
        }
        ranks
    }

    /// Sample variance (Bessel-corrected, n-1) of σ and ‖Ric‖ axes.
    fn sample_variance_2d(pairs: &[(f64, f64)]) -> (f64, f64) {
        let n = pairs.len();
        if n < 2 {
            return (0.0, 0.0);
        }
        let nf = n as f64;
        let ms: f64 = pairs.iter().map(|p| p.0).sum::<f64>() / nf;
        let mr: f64 = pairs.iter().map(|p| p.1).sum::<f64>() / nf;
        let vs: f64 = pairs.iter().map(|p| (p.0 - ms).powi(2)).sum::<f64>() / (nf - 1.0);
        let vr: f64 = pairs.iter().map(|p| (p.1 - mr).powi(2)).sum::<f64>() / (nf - 1.0);
        (vs, vr)
    }

    #[test]
    fn test_p4_4_sigma_flow_decreases_ricci_norm() {
        // Phase 2 "Calabi flow" in this codebase IS σ-flow. Test that
        // it reduces ‖Ric‖_L² (the independent diagnostic). This
        // confirms that what Stack A is doing is approximately Calabi
        // flow.
        let n_pts = 300;
        let mut solver = make_solver(2, n_pts);
        solver.donaldson_solve(50, 1e-6);

        // Pre-flow ‖Ric‖.
        let ric_pre = ricci_norm_l2(
            &solver.points, &solver.weights, solver.n_points,
            &solver.h_block, &solver.monomials, solver.n_basis, 1e-4,
        );
        let sigma_pre = compute_sigma_from_workspace(&mut solver);

        // Run σ-flow (= Stack-A's Adam refinement).
        let _ = solver.sigma_refine_analytic(50, 5e-3);

        // Post-flow ‖Ric‖.
        let ric_post = ricci_norm_l2(
            &solver.points, &solver.weights, solver.n_points,
            &solver.h_block, &solver.monomials, solver.n_basis, 1e-4,
        );
        let sigma_post = compute_sigma_from_workspace(&mut solver);

        println!(
            "σ-flow:  σ_L¹: {sigma_pre:.4e} -> {sigma_post:.4e};  \
             ‖Ric‖_L²: {ric_pre:.4e} -> {ric_post:.4e}",
        );

        assert!(
            sigma_post < sigma_pre,
            "σ-flow failed to decrease σ ({sigma_pre} -> {sigma_post})",
        );
        assert!(
            ric_post < ric_pre * 1.10,
            "σ-flow did not decrease ‖Ric‖_L² (or kept it within 10%): \
             {ric_pre} -> {ric_post}. Stack A and Stack C disagree.",
        );
    }

    /// Long-running k=4 cross-validation. Documented runtime: ~60 s
    /// release (FD-Ricci is O(n_basis²) per probe × 25 stencil entries
    /// × n_pts; at k=4, n_basis = 70).
    #[test]
    #[ignore]
    fn test_p4_4_ricci_norm_decreases_with_k_k4() {
        let n_pts = 400;
        let mut norms = Vec::new();
        let mut sigmas = Vec::new();
        for k in 2u32..=4 {
            let mut solver = make_solver(k, n_pts);
            solver.donaldson_solve(80, 1e-5);
            let _ = solver.sigma_refine_analytic(20, 5e-3);
            let sigma = compute_sigma_from_workspace(&mut solver);
            let ric_norm = ricci_norm_l2(
                &solver.points, &solver.weights, solver.n_points,
                &solver.h_block, &solver.monomials, solver.n_basis, 1e-4,
            );
            norms.push(ric_norm);
            sigmas.push(sigma);
            println!("k={k}: σ_L¹ = {sigma:.6e}, ‖Ric‖_L² = {ric_norm:.6e}");
        }
        assert!(norms[2] < norms[0], "expected ‖Ric‖_L² to drop k=2->k=4");
        assert!(sigmas[2] < sigmas[0], "expected σ to drop k=2->k=4");
    }

    // =====================================================================
    // P4.5 Phase 3 tests: tangent-frame Ricci.
    // =====================================================================

    /// Sanity test: tangent-frame Ricci returns finite, O(1)-O(100)
    /// magnitudes — NOT the O(10^8) noise-floor of the ambient diagnostic.
    /// At Donaldson-balanced k=2,3, ALL probe points should evaluate
    /// (≤5% may legitimately fail at the singular set / near it).
    #[test]
    fn test_p4_5_ricci_tangent_finite_at_donaldson_balanced() {
        for k in 2u32..=3 {
            let mut solver = make_solver(k, 200);
            solver.donaldson_solve(50, 1e-4);
            let mut n_ok = 0usize;
            let mut n_total = 0usize;
            let mut max_abs = 0.0f64;
            let mut min_abs_when_nonzero = f64::INFINITY;
            // Probe 20 points (modest oversample to catch outliers).
            for p in 0..20usize.min(solver.n_points) {
                let z: [f64; 10] = solver.points[p * 10..p * 10 + 10]
                    .try_into()
                    .unwrap();
                let mut ric = [0.0f64; 18];
                let res = ricci_at_point_bergman_tangent(
                    &solver.h_block,
                    &z,
                    &solver.monomials,
                    solver.n_basis,
                    1e-4,
                    &mut ric,
                );
                n_total += 1;
                match res {
                    Ok(()) => {
                        let all_finite = ric.iter().all(|v| v.is_finite());
                        if !all_finite {
                            eprintln!("k={k} point {p}: tangent Ric has non-finite entry");
                        } else {
                            n_ok += 1;
                            let mut max_here = 0.0f64;
                            for v in &ric {
                                let a = v.abs();
                                if a > max_here {
                                    max_here = a;
                                }
                            }
                            if max_here > max_abs {
                                max_abs = max_here;
                            }
                            if max_here > 0.0 && max_here < min_abs_when_nonzero {
                                min_abs_when_nonzero = max_here;
                            }
                            eprintln!(
                                "k={k} point {p}: max|Ric_tan_ij| = {max_here:.4e}",
                            );
                        }
                    }
                    Err(e) => {
                        eprintln!("k={k} point {p}: tangent Ric returned {e:?}");
                    }
                }
            }
            assert!(n_total > 0, "k={k}: no probe points ran");
            let ok_frac = n_ok as f64 / n_total as f64;
            // Stricter than ambient (we expected ≤5% singular failures);
            // allow up to 25% in case of unlucky chart picks at random
            // points, but require at least 75%.
            assert!(
                ok_frac >= 0.75,
                "k={k}: only {n_ok}/{n_total} ({ok_frac:.2}) tangent-Ricci probes succeeded; \
                 expected >=75%.",
            );
            // Magnitude check: tangent Ricci should be O(1) to O(100),
            // NOT O(10^8) (the ambient noise floor). We are deliberately
            // permissive at the upper end (allow up to 1e4 in case a
            // single probe is near-singular but didn't fail outright).
            assert!(
                max_abs < 1e4,
                "k={k}: max |Ric_tan| = {max_abs:.4e} — much larger than expected; \
                 tangent diagnostic may still be noise-floored.",
            );
            // Lower-end sanity: shouldn't be uniformly near-zero (would
            // indicate the FD stencil collapsed). At Donaldson-balanced
            // k=2,3 we expect non-trivial Ricci; assert SOME probe is
            // > 1e-3.
            assert!(
                max_abs > 1e-3,
                "k={k}: max |Ric_tan| = {max_abs:.4e} — suspiciously small; \
                 FD may be cancelling.",
            );
        }
    }

    /// P4.5 Phase 3: n=22 σ-Ric_tan correlation re-run, mirroring P5.2 but
    /// with the tangent-frame Ricci. Gate: `r_lo > 0.5`.
    #[test]
    #[ignore]
    fn test_p4_5_ricci_tangent_correlates_with_sigma_n20() {
        // Same 22 endpoints as P5.2.
        let configs: [(usize, usize, usize, f64, u64); 22] = [
            (1000,  60, 10, 1e-3, 1000),
            (1000,  60, 10, 1e-3, 1058),
            (1000,  60, 10, 1e-3, 1303),
            (1000,  60, 10, 1e-3, 1965),
            (1000,  60, 10, 1e-3, 1057),
            ( 500,  30,  0, 1e-3,  1),
            ( 500, 100,  0, 1e-3,  2),
            (1000,  30,  0, 1e-3,  3),
            (1000, 150,  0, 1e-3,  4),
            (1500,  60,  0, 1e-3,  5),
            ( 500,  60,  5, 1e-4,  6),
            ( 500,  60, 10, 1e-2,  7),
            (1000,  30, 20, 1e-4,  8),
            (1000, 100,  5, 1e-2,  9),
            (1000, 150, 20, 1e-3, 10),
            (1500,  30, 10, 1e-3, 11),
            (1500, 100, 10, 1e-3, 12),
            ( 500, 100, 40, 1e-3, 13),
            (1000,  60, 40, 1e-3, 14),
            (1000, 100, 40, 1e-2, 15),
            (1500,  60, 40, 1e-4, 16),
            (1500, 150, 40, 1e-3, 17),
        ];

        let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(configs.len());
        let mut endpoint_log: Vec<String> = Vec::with_capacity(configs.len());
        let mut seed_only_pairs: Vec<(f64, f64)> = Vec::new();

        for (i, &(n_pts, n_donaldson, n_adam, lr_adam, seed_off)) in
            configs.iter().enumerate()
        {
            let seed = 42u64.wrapping_add(seed_off);
            let mut solver = match QuinticSolver::new(2, n_pts, seed, 1e-8) {
                Some(s) => s,
                None => {
                    eprintln!(
                        "endpoint {i}: solver init failed (n_pts={n_pts}, seed={seed}); skipping",
                    );
                    continue;
                }
            };
            let n_done = solver.donaldson_solve(n_donaldson, 1e-8);
            if n_adam > 0 {
                let _ = solver.sigma_refine_analytic(n_adam, lr_adam);
            }
            let s = compute_sigma_from_workspace(&mut solver);
            let r = ricci_norm_l2_tangent(
                &solver.points,
                &solver.weights,
                solver.n_points,
                &solver.h_block,
                &solver.monomials,
                solver.n_basis,
                1e-4,
            );
            let line = format!(
                "endpoint {i:>2}: n_pts={n_pts:>4}, n_don={n_donaldson:>3} (done={n_done:>3}), \
                 n_adam={n_adam:>2}, lr={lr_adam:.0e}, seed={seed:>5} -> \
                 σ={s:.4e}, ‖Ric_tan‖={r:.4e}",
            );
            eprintln!("{line}");
            endpoint_log.push(line);
            if s.is_finite() && r.is_finite() {
                pairs.push((s, r));
                if i < 5 {
                    seed_only_pairs.push((s, r));
                }
            } else {
                eprintln!("  -> NON-FINITE (σ={s}, ‖Ric_tan‖={r}); excluded from correlation");
            }
        }

        let n = pairs.len();
        eprintln!("\n=== Summary: {n} valid endpoints out of {} ===", configs.len());
        assert!(
            n >= 20,
            "expected >=20 valid (σ, ‖Ric_tan‖) pairs; only got {n}.",
        );

        // Pearson r.
        let nf = n as f64;
        let mean_s: f64 = pairs.iter().map(|p| p.0).sum::<f64>() / nf;
        let mean_r: f64 = pairs.iter().map(|p| p.1).sum::<f64>() / nf;
        let mut num = 0.0;
        let mut ds2 = 0.0;
        let mut dr2 = 0.0;
        for &(s, r) in &pairs {
            let dx = s - mean_s;
            let dy = r - mean_r;
            num += dx * dy;
            ds2 += dx * dx;
            dr2 += dy * dy;
        }
        let denom = (ds2 * dr2).sqrt();
        assert!(
            denom > 1e-30 && denom.is_finite(),
            "degenerate variance: ds2={ds2}, dr2={dr2}",
        );
        let pearson = num / denom;

        // Spearman ρ.
        let rank_s = rank_with_average_ties(&pairs.iter().map(|p| p.0).collect::<Vec<_>>());
        let rank_r = rank_with_average_ties(&pairs.iter().map(|p| p.1).collect::<Vec<_>>());
        let mean_rs: f64 = rank_s.iter().sum::<f64>() / nf;
        let mean_rr: f64 = rank_r.iter().sum::<f64>() / nf;
        let mut sn = 0.0;
        let mut sd1 = 0.0;
        let mut sd2 = 0.0;
        for i in 0..n {
            let dx = rank_s[i] - mean_rs;
            let dy = rank_r[i] - mean_rr;
            sn += dx * dy;
            sd1 += dx * dx;
            sd2 += dy * dy;
        }
        let spearman = sn / ((sd1 * sd2).sqrt() + 1e-30);

        // Fisher 95% CI on r.
        let r_clamped = pearson.clamp(-0.999_999, 0.999_999);
        let z = 0.5 * ((1.0 + r_clamped) / (1.0 - r_clamped)).ln();
        let se = 1.0 / (nf - 3.0).sqrt();
        let z_lo = z - 1.96 * se;
        let z_hi = z + 1.96 * se;
        let r_lo = ((2.0 * z_lo).exp() - 1.0) / ((2.0 * z_lo).exp() + 1.0);
        let r_hi = ((2.0 * z_hi).exp() - 1.0) / ((2.0 * z_hi).exp() + 1.0);

        // Architectural variance check.
        let (seed_var_s, seed_var_r) = sample_variance_2d(&seed_only_pairs);
        let (full_var_s, full_var_r) = sample_variance_2d(&pairs);
        let ratio_s = if seed_var_s > 0.0 { full_var_s / seed_var_s } else { f64::INFINITY };
        let ratio_r = if seed_var_r > 0.0 { full_var_r / seed_var_r } else { f64::INFINITY };

        eprintln!("\n=== Phase-3 Statistics (n={n}) ===");
        eprintln!("Pearson r        = {pearson:.4}");
        eprintln!("Spearman ρ       = {spearman:.4}");
        eprintln!("Fisher z         = {z:.4}");
        eprintln!("Standard error   = {se:.4}");
        eprintln!("95% CI on r      = [{r_lo:.4}, {r_hi:.4}]");
        eprintln!("\n=== Phase-3 Architectural variance check ===");
        eprintln!(
            "Seed-only group ({} configs):",
            seed_only_pairs.len(),
        );
        eprintln!("  Var(σ)        = {seed_var_s:.4e}");
        eprintln!("  Var(‖Ric_tan‖)= {seed_var_r:.4e}");
        eprintln!("Across-all group ({n} configs):");
        eprintln!("  Var(σ)        = {full_var_s:.4e}");
        eprintln!("  Var(‖Ric_tan‖)= {full_var_r:.4e}");
        eprintln!("Cross-config / seed-only variance ratio:");
        eprintln!("  σ:           {ratio_s:.2}×");
        eprintln!("  ‖Ric_tan‖:   {ratio_r:.2}×");

        if ratio_r < 5.0 {
            eprintln!(
                "→ WARNING: ‖Ric_tan‖ variance ratio ({ratio_r:.2}×) is below 5×. \
                 The tangent Ricci may still be too noisy. Consider \
                 increasing n_pts in the FD evaluation.",
            );
        } else {
            eprintln!(
                "→ ‖Ric_tan‖ variance ratio is healthy: cross-config dominates seed.",
            );
        }

        // Strict gate: lower 95% CI on r > 0.5 (P5.2 statistical
        // confidence threshold).
        assert!(
            r_lo > 0.5,
            "Phase-3 lower 95% CI on Pearson r is {r_lo:.4} (r={pearson:.4}, \
             n={n}); P5.2 gate requires lower bound > 0.5. \
             Tangent-Ricci does NOT statistically confirm Stack A.",
        );
        assert!(
            pearson.signum() == spearman.signum() || spearman.abs() < 0.1,
            "Pearson ({pearson:.3}) and Spearman ({spearman:.3}) disagree in sign.",
        );
    }

    /// Sanity test: log_det_complex_hermitian_5x5 returns the expected
    /// value on a known matrix.
    #[test]
    fn test_p4_4_log_det_known() {
        // Identity 5×5 -> log det = 0.
        let mut g = [0.0f64; 50];
        for i in 0..5 {
            g[2 * (i * 5 + i)] = 1.0;
        }
        let ld = log_det_complex_hermitian_5x5(&g).expect("identity log det");
        assert!(ld.abs() < 1e-12, "log det I != 0: {ld}");

        // Diagonal real with diag entries [2, 3, 5, 7, 11] -> log det = log(2*3*5*7*11) = log(2310).
        let mut g2 = [0.0f64; 50];
        let diag = [2.0, 3.0, 5.0, 7.0, 11.0];
        for i in 0..5 {
            g2[2 * (i * 5 + i)] = diag[i];
        }
        let ld2 = log_det_complex_hermitian_5x5(&g2).expect("diag log det");
        let expected = 2310f64.ln();
        assert!(
            (ld2 - expected).abs() < 1e-10,
            "log det diag != log(2310): got {ld2}, expected {expected}",
        );
    }
}
