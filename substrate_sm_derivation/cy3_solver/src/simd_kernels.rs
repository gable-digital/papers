//! SIMD-vectorised helpers built on pwos-math's reduce/binary/unary
//! kernels for the cy3 pipeline's hot loops.
//!
//! Each function here is a drop-in replacement for an inline scalar
//! loop in the existing pipeline. Callers retain control of whether
//! to use the SIMD or scalar form (the scalar form remains in place
//! for testability and for cases where SIMD overhead exceeds the
//! payload size).
//!
//! # Backing kernels
//!
//! All vectorised primitives live under
//! `products/pwos/libs/pwos-math/src/kernels/`. The relevant submodules
//! used here are:
//!   - `pwos_math::kernels::reduce::{sum_f64, dot_f64}`
//!     (`.../kernels/reduce/{sum_f64.rs,dot_f64.rs}`)
//!   - `pwos_math::kernels::binary::{add_f64_inplace, mul_f64,
//!      mul_f64_inplace, mul_f64_scalar, sub_f64_inplace,
//!      div_f64, add_f64}`
//!     (`.../kernels/binary/{add,sub,mul,div}.rs`)
//!   - `pwos_math::kernels::unary::{exp_f64, exp_f64_inplace,
//!      sqrt_f64, sqrt_f64_inplace, neg_f64_inplace}`
//!     (`.../kernels/unary/{exp,sqrt,neg}.rs`)
//!
//! # Allocation contract
//!
//! None of the helpers exposed by this module allocate on the heap.
//! Every output buffer is supplied by the caller, and any temporaries
//! are stack-resident fixed-size arrays. This is intentional so the
//! helpers can sit inside Adam / Gauss / Gram-matrix hot loops without
//! pressuring the allocator.
//!
//! # NdArray boundary
//!
//! `pwos_math::ndarray::NdArray<f64>` is only used at function
//! boundaries elsewhere in the crate. Internally these helpers operate
//! on raw `&[f64]` / `&mut [f64]` slices, exactly matching the
//! pwos-math kernel signatures.

use pwos_math::kernels::binary::{
    add_f64_inplace, mul_f64, mul_f64_scalar,
};
use pwos_math::kernels::reduce::dot_f64;
use pwos_math::kernels::unary::{exp_f64, sqrt_f64};

// ---------------------------------------------------------------------------
// Adam optimiser helpers
// ---------------------------------------------------------------------------

/// Adam moment update:
///   `m = β1 * m + (1 - β1) * g`
///   `v = β2 * v + (1 - β2) * g²`
///
/// Performed as a fused pass over `n_dof = m.len()` entries. Uses
/// pwos-math `binary` kernels under the hood (`mul_f64_inplace`,
/// `mul_f64_scalar`, `add_f64_inplace`, plus an in-place square via
/// `mul_f64_inplace` on a stack-allocated chunk buffer).
///
/// All buffers must have the same length and must not alias.
///
/// # Panics
///
/// Panics if `m.len()`, `v.len()`, and `g.len()` are not all equal.
///
/// # Drop-in target
///
/// Replaces the inline scalar loop in
/// `src/quintic.rs::sigma_adam_step_workspace` (~line 245) and
/// `src/quintic.rs::sigma_refine_analytic` (~line 252).
pub fn adam_moment_update(
    m: &mut [f64],
    v: &mut [f64],
    g: &[f64],
    beta1: f64,
    beta2: f64,
) {
    let n = m.len();
    assert_eq!(n, v.len(), "adam_moment_update: m and v length mismatch");
    assert_eq!(n, g.len(), "adam_moment_update: m and g length mismatch");
    if n == 0 {
        return;
    }

    let one_minus_b1 = 1.0 - beta1;
    let one_minus_b2 = 1.0 - beta2;

    // Work in fixed-size chunks so the temporaries never touch the heap.
    // The chunk size is large enough to amortise dispatch overhead in the
    // pwos-math AVX2 paths (their threshold is 8 lanes for f64).
    const CHUNK: usize = 1024;
    let mut g2_buf = [0.0f64; CHUNK];
    let mut scaled_buf = [0.0f64; CHUNK];

    let mut offset = 0;
    while offset < n {
        let end = (offset + CHUNK).min(n);
        let len = end - offset;

        // ---- m update --------------------------------------------------
        // m_chunk *= beta1
        // scaled = (1 - beta1) * g_chunk
        // m_chunk += scaled
        let m_chunk = &mut m[offset..end];
        let g_chunk = &g[offset..end];
        let scaled = &mut scaled_buf[..len];

        // m *= beta1   (pwos-math binary::mul scalar in-place)
        // No `mul_f64_scalar_inplace` is exposed, so we route through
        // `mul_f64_scalar(m, beta1, scratch)` then copy back -- but to
        // avoid an extra copy we instead call `mul_f64_scalar` from
        // `m -> scaled` (overwriting scaled), then immediately do the
        // axpy below. We then compute m = scaled + (1-b1)*g via two
        // mul_scalar+add steps. To keep arithmetic simple and correct,
        // use the canonical pattern: scale m in place via a helper.
        scale_inplace(m_chunk, beta1);

        // scaled = one_minus_b1 * g
        mul_f64_scalar(g_chunk, one_minus_b1, scaled);

        // m += scaled        (pwos-math binary::add in-place)
        add_f64_inplace(m_chunk, scaled);

        // ---- v update --------------------------------------------------
        // g2 = g * g         (pwos-math binary::mul, NOT inplace -- we
        //                    need the squared value separately from g)
        let v_chunk = &mut v[offset..end];
        let g2 = &mut g2_buf[..len];
        mul_f64(g_chunk, g_chunk, g2);

        // v *= beta2
        scale_inplace(v_chunk, beta2);

        // g2 *= (1 - beta2)
        // Reuse g2 as scratch -- it now holds (1-b2) * g²
        scale_inplace(g2, one_minus_b2);

        // v += g2
        add_f64_inplace(v_chunk, g2);

        offset = end;
    }
}

/// Adam parameter update:
///   `theta -= scale * m / (sqrt(v) + eps)`
///
/// Fused pass over `theta.len()` entries. Uses pwos-math `unary::sqrt_f64`
/// for the SIMD square root and falls back to a scalar fused
/// multiply-subtract for the per-element division (no fused FMA-style
/// kernel for `(a + b) * c` is exposed by pwos-math, so the
/// addition/division pair is the scalar fallback flagged in the TODO at
/// the bottom of this file).
///
/// # Panics
///
/// Panics if any of `theta`, `m`, `v` differ in length.
pub fn adam_apply_update(
    theta: &mut [f64],
    m: &[f64],
    v: &[f64],
    scale: f64,
    eps: f64,
) {
    let n = theta.len();
    assert_eq!(n, m.len(), "adam_apply_update: theta and m length mismatch");
    assert_eq!(n, v.len(), "adam_apply_update: theta and v length mismatch");
    if n == 0 {
        return;
    }

    const CHUNK: usize = 1024;
    let mut sqrt_v_buf = [0.0f64; CHUNK];

    let mut offset = 0;
    while offset < n {
        let end = (offset + CHUNK).min(n);
        let len = end - offset;

        let theta_chunk = &mut theta[offset..end];
        let m_chunk = &m[offset..end];
        let v_chunk = &v[offset..end];
        let sqrt_v = &mut sqrt_v_buf[..len];

        // sqrt_v = sqrt(v)        (pwos-math unary::sqrt_f64)
        sqrt_f64(v_chunk, sqrt_v);

        // theta[i] -= scale * m[i] / (sqrt_v[i] + eps)
        //
        // TODO(pwos-math): no kernel currently exposes
        // `(a + scalar) -> reciprocal -> mul_acc` as a single SIMD pass,
        // so the per-lane division is scalar. When pwos-math grows a
        // `binary::div_with_offset` or a fused-axpy kernel, swap this
        // loop for that call.
        for i in 0..len {
            theta_chunk[i] -= scale * m_chunk[i] / (sqrt_v[i] + eps);
        }

        offset = end;
    }
}

// ---------------------------------------------------------------------------
// Gaussian basis evaluation
// ---------------------------------------------------------------------------

/// Element-wise Gaussian:
///   `out[i] = exp(-acc[i] / two_sigma_sq)`
///
/// where `acc[i]` is the precomputed squared distance for sample `i`.
/// Uses pwos-math `unary::exp_f64` over a chunked scratch buffer of
/// negated/scaled inputs.
///
/// # Panics
///
/// Panics if `out.len() != acc.len()`.
///
/// # Drop-in target
///
/// Replaces the scalar `exp(-||x - c||² / σ²)` loop in
/// `src/lib.rs` around lines 482-500.
pub fn gaussian_basis_eval(out: &mut [f64], acc: &[f64], two_sigma_sq: f64) {
    let n = out.len();
    assert_eq!(n, acc.len(), "gaussian_basis_eval: out and acc length mismatch");
    if n == 0 {
        return;
    }
    assert!(
        two_sigma_sq.is_finite() && two_sigma_sq > 0.0,
        "gaussian_basis_eval: two_sigma_sq must be positive and finite"
    );

    let inv_neg = -1.0 / two_sigma_sq;
    // Reject sub/sub-subnormal `two_sigma_sq` whose reciprocal overflows
    // — `is_finite() && > 0` accepts these, but `1/x` then becomes ±inf
    // and `exp(-inf) = 0` silently zeroes the output.
    assert!(
        inv_neg.is_finite(),
        "gaussian_basis_eval: two_sigma_sq = {two_sigma_sq:e} is too small; reciprocal overflows"
    );

    const CHUNK: usize = 1024;
    let mut tmp = [0.0f64; CHUNK];

    let mut offset = 0;
    while offset < n {
        let end = (offset + CHUNK).min(n);
        let len = end - offset;
        let acc_chunk = &acc[offset..end];
        let out_chunk = &mut out[offset..end];
        let scratch = &mut tmp[..len];

        // scratch = acc * (-1/two_sigma_sq)   (pwos-math binary::mul_scalar)
        mul_f64_scalar(acc_chunk, inv_neg, scratch);

        // out = exp(scratch)                  (pwos-math unary::exp_f64)
        exp_f64(scratch, out_chunk);

        offset = end;
    }
}

// ---------------------------------------------------------------------------
// Squared L2 distance for 8-D points
// ---------------------------------------------------------------------------

/// Squared L2 distance accumulator for a batch of 8-D points against a
/// fixed center:
///   `out[i] = Σ_{j=0}^{7} (xs[i*8 + j] - center[j])²`
///
/// Vectorised over the points axis. Each per-point inner sum is
/// dispatched to `pwos_math::kernels::reduce::dot_f64` (operating on
/// the per-point delta vector). The 8-element delta is built into a
/// stack array, so no allocation occurs per point.
///
/// # Panics
///
/// Panics if `xs.len() != out.len() * 8`.
///
/// # Drop-in target
///
/// Used inside the Gaussian basis pipeline before
/// [`gaussian_basis_eval`] (see `src/lib.rs:482-500`).
pub fn squared_distance_8d(out: &mut [f64], xs: &[f64], center: &[f64; 8]) {
    let n_points = out.len();
    assert_eq!(
        xs.len(),
        n_points * 8,
        "squared_distance_8d: xs.len() must equal 8 * out.len()"
    );

    // The pwos-math dot_f64 dispatch threshold is 8 lanes for f64,
    // which is exactly our per-point vector length. That means each
    // call lands on the AVX2 path (when available) for the full
    // 8-wide reduction.
    for i in 0..n_points {
        let base = i * 8;
        let mut delta = [0.0f64; 8];
        for j in 0..8 {
            delta[j] = xs[base + j] - center[j];
        }
        // out[i] = delta · delta   (pwos-math reduce::dot_f64)
        out[i] = dot_f64(&delta, &delta);
    }
}

// ---------------------------------------------------------------------------
// Weighted dot / axpy
// ---------------------------------------------------------------------------

/// Weighted dot product:
///   `Σ_i w[i] * a[i]`
///
/// using pwos-math `reduce::dot_f64`.
///
/// # Panics
///
/// Panics if `w.len() != a.len()`.
///
/// # Drop-in target
///
/// Inner reduction loop in
/// `src/quintic.rs::compute_fs_gram_matrix` (~line 1466) for the
/// per-pair contraction over basis indices (real or imaginary part).
pub fn weighted_dot(w: &[f64], a: &[f64]) -> f64 {
    assert_eq!(w.len(), a.len(), "weighted_dot: w and a length mismatch");
    if w.is_empty() {
        return 0.0;
    }
    dot_f64(w, a)
}

/// AXPY-style weighted vector add:
///   `out[i] += scalar * src[i]`
///
/// Uses pwos-math `binary::mul_f64_scalar` into a chunked scratch
/// buffer followed by `binary::add_f64_inplace`. No heap allocation.
///
/// # Panics
///
/// Panics if `out.len() != src.len()`.
pub fn axpy(out: &mut [f64], scalar: f64, src: &[f64]) {
    let n = out.len();
    assert_eq!(n, src.len(), "axpy: out and src length mismatch");
    if n == 0 {
        return;
    }

    const CHUNK: usize = 1024;
    let mut scratch = [0.0f64; CHUNK];

    let mut offset = 0;
    while offset < n {
        let end = (offset + CHUNK).min(n);
        let len = end - offset;
        let src_chunk = &src[offset..end];
        let out_chunk = &mut out[offset..end];
        let buf = &mut scratch[..len];

        // buf = scalar * src       (pwos-math binary::mul_f64_scalar)
        mul_f64_scalar(src_chunk, scalar, buf);
        // out += buf               (pwos-math binary::add_f64_inplace)
        add_f64_inplace(out_chunk, buf);

        offset = end;
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// In-place scalar multiply: `data[i] *= scalar`.
///
/// pwos-math exposes `mul_f64_scalar(src, scalar, dst)` and
/// `mul_f64_inplace(lhs, rhs)` (vector × vector), but no
/// `mul_f64_scalar_inplace`. The closest fit is a self-multiply by a
/// broadcast vector, but that would require an allocation. Instead we
/// In-place scale by a scalar. The previous implementation routed
/// through a stack scratch buffer + `copy_from_slice`, paying ~2× the
/// memory bandwidth (one read + write to scratch, one read + write back
/// to data). The autovectoriser produces an AVX2 mul-by-broadcast loop
/// that matches pwos-math's `mul_f64_scalar` performance without the
/// round-trip copy.
#[inline]
fn scale_inplace(data: &mut [f64], scalar: f64) {
    for v in data.iter_mut() {
        *v *= scalar;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_slice(a: &[f64], b: &[f64], rel: f64, abs: f64) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            let tol = abs + rel * y.abs().max(x.abs());
            assert!(
                diff <= tol,
                "mismatch at index {i}: simd={x} scalar={y} diff={diff} tol={tol}"
            );
        }
    }

    /// REGRESSION (audit): `gaussian_basis_eval` must reject `two_sigma_sq`
    /// values whose reciprocal overflows. Previously `is_finite() && > 0`
    /// accepted subnormals like 1e-310, after which `inv_neg = -inf` and
    /// `exp(-inf) = 0` silently zeroed the output.
    #[test]
    #[should_panic(expected = "two_sigma_sq")]
    fn gaussian_basis_eval_rejects_subnormal_sigma() {
        let mut out = vec![0.0_f64; 16];
        let acc = vec![1.0_f64; 16];
        // Subnormal that overflows reciprocal: 1/1e-310 = inf.
        gaussian_basis_eval(&mut out, &acc, 1.0e-310);
    }

    /// REGRESSION (audit): edge-case sizes (n=1, 7, 8, 9, 1024, 1025).
    /// Verify gaussian_basis_eval gives the scalar-equivalent result at
    /// chunk boundaries.
    #[test]
    fn simd_kernels_handle_edge_case_sizes() {
        for &n in &[1_usize, 7, 8, 9, 1024, 1025] {
            let mut out = vec![0.0_f64; n];
            let acc: Vec<f64> = (0..n).map(|i| 0.1 + 0.01 * i as f64).collect();
            gaussian_basis_eval(&mut out, &acc, 0.5);
            for (i, &a) in acc.iter().enumerate() {
                let expected = (-a / 0.5).exp();
                let diff = (out[i] - expected).abs();
                let tol = 1.0e-12 * expected.abs().max(1.0);
                assert!(
                    diff <= tol,
                    "n={n}, i={i}: out={} expected={} diff={diff}",
                    out[i],
                    expected
                );
            }
        }
    }

    /// REGRESSION (audit): `scale_inplace` must produce the same result
    /// as the scalar reference at all sizes. The previous round-trip-
    /// through-scratch implementation was correct but slow; the in-place
    /// loop must remain bit-identical to it.
    #[test]
    fn scale_inplace_matches_scalar_reference() {
        for &n in &[1_usize, 7, 8, 1024, 1025] {
            let scalar = 0.7f64;
            let input: Vec<f64> = (0..n).map(|i| (i as f64) * 0.013 - 0.5).collect();
            let mut data = input.clone();
            scale_inplace(&mut data, scalar);
            for i in 0..n {
                let expected = input[i] * scalar;
                assert!(
                    (data[i] - expected).abs() < 1e-15,
                    "n={n}, i={i}: scale_inplace gave {} but expected {}",
                    data[i],
                    expected
                );
            }
        }
    }

    #[test]
    fn adam_moment_update_matches_scalar() {
        let n = 1000;
        let mut m_simd = vec![0.0f64; n];
        let mut v_simd = vec![0.0f64; n];
        let mut m_ref = vec![0.0f64; n];
        let mut v_ref = vec![0.0f64; n];
        let g: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 0.013).sin() * 0.5 - 0.25)
            .collect();

        // Seed with non-trivial state to stress the in-place update.
        for i in 0..n {
            let s = (i as f64) * 0.001;
            m_simd[i] = s.cos() * 0.1;
            v_simd[i] = (s * 1.7).cos().abs() * 0.05 + 1e-3;
            m_ref[i] = m_simd[i];
            v_ref[i] = v_simd[i];
        }

        let beta1 = 0.9;
        let beta2 = 0.999;

        adam_moment_update(&mut m_simd, &mut v_simd, &g, beta1, beta2);

        for i in 0..n {
            m_ref[i] = beta1 * m_ref[i] + (1.0 - beta1) * g[i];
            v_ref[i] = beta2 * v_ref[i] + (1.0 - beta2) * g[i] * g[i];
        }

        approx_eq_slice(&m_simd, &m_ref, 1e-12, 1e-15);
        approx_eq_slice(&v_simd, &v_ref, 1e-12, 1e-15);
    }

    #[test]
    fn adam_apply_update_matches_scalar() {
        let n = 1000;
        let m: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 0.011).sin() * 0.05)
            .collect();
        let v: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 0.007).cos().abs() * 0.02 + 1e-4)
            .collect();
        let mut theta_simd: Vec<f64> = (0..n).map(|i| (i as f64) * 1e-3).collect();
        let mut theta_ref = theta_simd.clone();

        let scale = 1e-2;
        let eps = 1e-8;

        adam_apply_update(&mut theta_simd, &m, &v, scale, eps);

        for i in 0..n {
            theta_ref[i] -= scale * m[i] / (v[i].sqrt() + eps);
        }

        approx_eq_slice(&theta_simd, &theta_ref, 1e-12, 1e-15);
    }

    #[test]
    fn gaussian_basis_eval_matches_scalar() {
        let n = 100;
        let acc: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 0.05).sin().abs() * 4.0 + 0.1)
            .collect();
        let two_sigma_sq = 2.0;
        let mut out_simd = vec![0.0f64; n];
        let mut out_ref = vec![0.0f64; n];

        gaussian_basis_eval(&mut out_simd, &acc, two_sigma_sq);

        for i in 0..n {
            out_ref[i] = (-acc[i] / two_sigma_sq).exp();
        }

        // pwos-math exp_f64 routes through libm, so the result should
        // agree with the scalar (-acc/two_sigma_sq).exp() to within
        // a few ULPs after the mul_scalar pass.
        approx_eq_slice(&out_simd, &out_ref, 1e-13, 1e-15);
    }

    #[test]
    fn squared_distance_8d_matches_scalar() {
        let n_points = 100;
        let center = [0.1f64, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let mut xs = vec![0.0f64; n_points * 8];
        for i in 0..n_points {
            for j in 0..8 {
                xs[i * 8 + j] = ((i * 8 + j) as f64).sin() * 0.5
                    + ((i + j) as f64).cos() * 0.25;
            }
        }
        let mut out_simd = vec![0.0f64; n_points];
        let mut out_ref = vec![0.0f64; n_points];

        squared_distance_8d(&mut out_simd, &xs, &center);

        for i in 0..n_points {
            let mut acc = 0.0;
            for j in 0..8 {
                let d = xs[i * 8 + j] - center[j];
                acc += d * d;
            }
            out_ref[i] = acc;
        }

        approx_eq_slice(&out_simd, &out_ref, 1e-13, 1e-15);
    }

    #[test]
    fn weighted_dot_matches_scalar() {
        let n = 1000;
        let w: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 0.017).sin())
            .collect();
        let a: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 0.023).cos())
            .collect();

        let simd = weighted_dot(&w, &a);
        let mut scalar = 0.0;
        for i in 0..n {
            scalar += w[i] * a[i];
        }

        // dot_f64 uses independent accumulators, so the order of
        // additions differs from the naive left-fold. Tolerance is
        // chosen relative to the magnitude of the final reduction.
        let tol = 1e-12 * scalar.abs().max(1.0);
        assert!(
            (simd - scalar).abs() <= tol,
            "weighted_dot: simd={simd} scalar={scalar} diff={} tol={tol}",
            (simd - scalar).abs()
        );
    }

    #[test]
    fn axpy_matches_scalar() {
        let n = 1000;
        let src: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 0.019).sin() * 0.3)
            .collect();
        let mut out_simd: Vec<f64> = (0..n).map(|i| (i as f64) * 1e-2).collect();
        let mut out_ref = out_simd.clone();
        let scalar = 0.75;

        axpy(&mut out_simd, scalar, &src);

        for i in 0..n {
            out_ref[i] += scalar * src[i];
        }

        approx_eq_slice(&out_simd, &out_ref, 1e-13, 1e-15);
    }

    #[test]
    fn empty_inputs_are_no_ops() {
        let mut m: Vec<f64> = vec![];
        let mut v: Vec<f64> = vec![];
        let g: Vec<f64> = vec![];
        adam_moment_update(&mut m, &mut v, &g, 0.9, 0.999);

        let mut theta: Vec<f64> = vec![];
        adam_apply_update(&mut theta, &m, &v, 1e-2, 1e-8);

        let mut out: Vec<f64> = vec![];
        let acc: Vec<f64> = vec![];
        gaussian_basis_eval(&mut out, &acc, 2.0);

        let mut sd: Vec<f64> = vec![];
        let xs: Vec<f64> = vec![];
        let center = [0.0f64; 8];
        squared_distance_8d(&mut sd, &xs, &center);

        let dot = weighted_dot(&[], &[]);
        assert_eq!(dot, 0.0);

        let mut ax: Vec<f64> = vec![];
        let src: Vec<f64> = vec![];
        axpy(&mut ax, 0.5, &src);
    }
}

// ---------------------------------------------------------------------------
// TODO(pwos-math): outstanding kernel gaps
// ---------------------------------------------------------------------------
//
// 1. `mul_f64_scalar_inplace(data: &mut [f64], scalar: f64)`
//    -- currently emulated via `mul_f64_scalar` + `copy_from_slice`
//    in `scale_inplace`. Used by `adam_moment_update` (β1·m, β2·v,
//    (1-β2)·g²) and `adam_apply_update` (sqrt(v)+eps step is also
//    affected indirectly).
//
// 2. Fused `(a + scalar) -> reciprocal -> mul -> sub_inplace` for the
//    Adam parameter step `theta -= scale * m / (sqrt(v) + eps)`.
//    Currently the per-lane division is a scalar inner loop.
//
// 3. A vectorised batch dot-along-axis kernel (e.g.
//    `dot_along_inner_axis(xs: &[f64], stride: usize, center:
//    &[f64; D], out: &mut [f64])`) would let `squared_distance_8d`
//    avoid the per-point delta-buffer copy. Today we hit the AVX2 path
//    of `dot_f64` once per point, which is already fast for 8-wide,
//    but a true outer-loop SIMD pass would be better for D = 8.
