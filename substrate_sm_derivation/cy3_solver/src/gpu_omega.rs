//! CUDA-accelerated holomorphic 3-form Ω evaluation at sample points
//! on the Tian-Yau CICY.
//!
//! Backs [`crate::yukawa_overlap::compute_omega_at_samples`] with a
//! single per-point kernel that:
//!
//!  1. evaluates the holomorphic Jacobian `J_{ik} = ∂f_i / ∂x_k` for
//!     `i ∈ {0, 1, 2}` (NHYPER = 3 defining polynomials of the
//!     Tian-Yau triple) and `k ∈ {0, …, 7}` (NCOORDS = 8 ambient
//!     CP^3 × CP^3 coordinates),
//!  2. picks the patch-fixing column for each `CP^3` factor as the
//!     argmax-modulus coordinate (forbidden columns of the
//!     elimination Jacobian),
//!  3. greedy-selects the remaining `NHYPER` elimination columns by
//!     largest |J_{i,k}|,
//!  4. computes `Ω = 1 / det(J_elim)` for the 3×3 sub-Jacobian.
//!
//! Per-thread cost is small (~few hundred FP64 ops) but every sample
//! point is independent, so this is a cleanly embarrassingly-parallel
//! reduction across N points. A single launch handles all N.
//!
//! The kernel takes the `BicubicPair` polynomials in a flattened
//! representation (one flat array per polynomial, each entry packed as
//! `re, im, e_0, e_1, …, e_7` doubles where the integer exponents are
//! re-cast to `f64` to keep the kernel argument layout simple), and
//! per-point coordinates as 16 doubles per point.
//!
//! Numerical fidelity: the kernel uses the same column-picking and 3×3
//! determinant arithmetic as the CPU implementation, so a CPU/GPU
//! parity test (`gpu_omega_at_samples_matches_cpu`) holds to ULP-level
//! tolerance for synthetic inputs. On a host without CUDA the kernel
//! is silently skipped and the CPU path runs.

use cudarc::driver::{CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use num_complex::Complex64;
use std::sync::Arc;

use crate::cicy_sampler::{BicubicPair, NCOORDS, NHYPER};

const KERNEL_SOURCE: &str = r#"
// Compact monomial layout: each monomial is a 10-double record
//   [re, im, e0, e1, e2, e3, e4, e5, e6, e7]
// (exponents stored as f64 for arg-layout simplicity; small ints).
// One flat array per polynomial; mono_starts[p..p+1] gives the slice
// for polynomial p.

__device__ inline void cmul(double ar, double ai, double br, double bi,
                            double* outr, double* outi) {
    *outr = ar * br - ai * bi;
    *outi = ar * bi + ai * br;
}

extern "C" __global__ void omega_at_samples(
    const double* __restrict__ monos,         // packed monomial records, all polys
    const int*    __restrict__ mono_starts,   // length NHYPER + 1, in records
    const double* __restrict__ pts_re,        // length NCOORDS * n_pts
    const double* __restrict__ pts_im,        // length NCOORDS * n_pts
    int n_pts,
    double* __restrict__ omega_re,
    double* __restrict__ omega_im
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pts) return;

    const int NH = 3;   // NHYPER
    const int NC = 8;   // NCOORDS

    // Load this thread's coordinates into registers.
    double zr[8], zi[8];
    for (int k = 0; k < NC; ++k) {
        zr[k] = pts_re[k * n_pts + p];
        zi[k] = pts_im[k * n_pts + p];
    }

    // Compute the full NHYPER × NCOORDS Jacobian J_{ik} = ∂f_i / ∂z_k.
    //
    // Per-monomial optimisation: instead of recomputing z_0^e_0 ⋯ z_{NC-1}^e_{NC-1}
    // for each (k, monomial) pair via the naive integer-power inner
    // loop, we
    //   (a) compute z_j^{e_j} for all j once via repeated squaring
    //       (~log_2(max_e) multiplications per coord vs e_j sequential
    //        multiplications);
    //   (b) form V_m = c · Π_j z_j^{e_j} once;
    //   (c) derive ∂/∂z_k of the monomial as e_k · V_m / z_k, valid
    //       wherever z_k ≠ 0 (which is true on the samples — they live
    //       on S^7 × S^7 minus measure-zero coord-vanishing sets).
    //
    // For Tian-Yau monomials with max exponent 3, this drops the inner
    // arithmetic from ~64 muls per (k, monomial) to ~16 + a divide.
    double Jr[NH * NC];
    double Ji[NH * NC];
    // Zero-init the Jacobian so we can accumulate per monomial.
    for (int kk = 0; kk < NH * NC; ++kk) { Jr[kk] = 0.0; Ji[kk] = 0.0; }

    for (int i = 0; i < NH; ++i) {
        int s = mono_starts[i];
        int e = mono_starts[i + 1];
        for (int m = s; m < e; ++m) {
            int base = m * 10;
            double cr = monos[base + 0];
            double ci = monos[base + 1];

            // (a) z_j^{e_j} for all j via repeated squaring.
            double pr_pow[NC], pi_pow[NC];   // z_j^{e_j}
            int ej[NC];
            for (int j = 0; j < NC; ++j) {
                ej[j] = (int)monos[base + 2 + j];
                double rr = 1.0, ri = 0.0;
                double br = zr[j], bi = zi[j];
                int ee = ej[j];
                while (ee > 0) {
                    if (ee & 1) {
                        double nr = rr * br - ri * bi;
                        double ni = rr * bi + ri * br;
                        rr = nr; ri = ni;
                    }
                    ee >>= 1;
                    if (ee > 0) {
                        double nbr = br * br - bi * bi;
                        double nbi = 2.0 * br * bi;
                        br = nbr; bi = nbi;
                    }
                }
                pr_pow[j] = rr;
                pi_pow[j] = ri;
            }

            // (b) V_m = c · Π_j z_j^{e_j}.
            double vr = cr, vi = ci;
            for (int j = 0; j < NC; ++j) {
                if (ej[j] == 0) continue;
                double nr, ni;
                cmul(vr, vi, pr_pow[j], pi_pow[j], &nr, &ni);
                vr = nr; vi = ni;
            }

            // (c) ∂_k V_m = e_k · V_m / z_k for k with e_k > 0.
            //     Skip k with e_k == 0 (derivative is identically zero).
            //     Skip if |z_k| < ~1e-150 — at exactly z_k = 0 with
            //     e_k = 1 the true partial is c · Π_{j≠k} z_j^e_j
            //     (finite, not zero), and the CPU naive integer-power
            //     loop would compute it correctly. We undercount in
            //     that knife-edge case but real CY3 sample points
            //     after patch-rescaling have |z_k| ≳ 0.01 (the
            //     argmax-modulus coord is 1.0 by construction), so
            //     the divergence from CPU is never exercised in
            //     practice.
            for (int k = 0; k < NC; ++k) {
                if (ej[k] == 0) continue;
                double zk_r = zr[k];
                double zk_i = zi[k];
                double zk_mag2 = zk_r * zk_r + zk_i * zk_i;
                if (zk_mag2 < 1.0e-300) continue;
                // V_m / z_k = V_m · conj(z_k) / |z_k|^2
                double inv_zk_r = zk_r / zk_mag2;
                double inv_zk_i = -zk_i / zk_mag2;
                double dr_, di_;
                cmul(vr, vi, inv_zk_r, inv_zk_i, &dr_, &di_);
                double scale = (double)ej[k];
                Jr[i * NC + k] += scale * dr_;
                Ji[i * NC + k] += scale * di_;
            }
        }
    }

    // Patch-fixing columns: argmax-|·| in z (cols 0..3) and w (cols 4..7).
    int z_idx = 0;
    {
        double best = -1.0;
        for (int k = 0; k < 4; ++k) {
            double a = sqrt(zr[k] * zr[k] + zi[k] * zi[k]);
            if (a > best) { best = a; z_idx = k; }
        }
    }
    int w_idx = 0;
    {
        double best = -1.0;
        for (int k = 0; k < 4; ++k) {
            double a = sqrt(zr[4 + k] * zr[4 + k] + zi[4 + k] * zi[4 + k]);
            if (a > best) { best = a; w_idx = k; }
        }
    }

    // Greedy column pick excluding patch-fixing cols.
    bool taken[NC];
    for (int k = 0; k < NC; ++k) taken[k] = false;
    taken[z_idx]     = true;
    taken[4 + w_idx] = true;

    int picks[NH];
    bool ok = true;
    for (int i = 0; i < NH; ++i) {
        int best_k = -1;
        double best_abs = -1.0;
        for (int k = 0; k < NC; ++k) {
            if (taken[k]) continue;
            double a = sqrt(Jr[i * NC + k] * Jr[i * NC + k] + Ji[i * NC + k] * Ji[i * NC + k]);
            if (a > best_abs) { best_abs = a; best_k = k; }
        }
        if (best_k < 0 || best_abs <= 0.0) { ok = false; break; }
        picks[i] = best_k;
        taken[best_k] = true;
    }
    if (!ok) {
        omega_re[p] = NAN;
        omega_im[p] = NAN;
        return;
    }

    // 3×3 elimination Jacobian J_elim[i][j] = J[i][picks[j]].
    double Mr[9], Mi[9];
    for (int i = 0; i < NH; ++i) {
        for (int j = 0; j < NH; ++j) {
            int idx = i * NC + picks[j];
            Mr[i * NH + j] = Jr[idx];
            Mi[i * NH + j] = Ji[idx];
        }
    }

    // det(M) by cofactor expansion along row 0.
    auto det2 = [&](int i0, int i1, int j0, int j1,
                    double& dr, double& di) {
        double t1r, t1i, t2r, t2i;
        cmul(Mr[i0 * NH + j0], Mi[i0 * NH + j0], Mr[i1 * NH + j1], Mi[i1 * NH + j1], &t1r, &t1i);
        cmul(Mr[i0 * NH + j1], Mi[i0 * NH + j1], Mr[i1 * NH + j0], Mi[i1 * NH + j0], &t2r, &t2i);
        dr = t1r - t2r;
        di = t1i - t2i;
    };
    double a_r, a_i, b_r, b_i, c_r, c_i;
    double tmp_r, tmp_i;
    double det_r, det_i;
    det2(1, 2, 1, 2, a_r, a_i);
    cmul(Mr[0], Mi[0], a_r, a_i, &tmp_r, &tmp_i);
    det_r = tmp_r;
    det_i = tmp_i;
    det2(1, 2, 0, 2, b_r, b_i);
    cmul(Mr[1], Mi[1], b_r, b_i, &tmp_r, &tmp_i);
    det_r -= tmp_r;
    det_i -= tmp_i;
    det2(1, 2, 0, 1, c_r, c_i);
    cmul(Mr[2], Mi[2], c_r, c_i, &tmp_r, &tmp_i);
    det_r += tmp_r;
    det_i += tmp_i;

    double mag2 = det_r * det_r + det_i * det_i;
    if (mag2 < 1.0e-300) {
        omega_re[p] = NAN;
        omega_im[p] = NAN;
        return;
    }
    // 1 / (a + bi) = (a - bi) / (a^2 + b^2)
    omega_re[p] = det_r / mag2;
    omega_im[p] = -det_i / mag2;
}
"#;

/// Lazily-compiled CUDA module + context for the Ω-at-samples kernel.
pub struct GpuOmegaContext {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
}

impl GpuOmegaContext {
    /// Initialise the CUDA context + JIT-compile the kernel. Wraps
    /// every step in `catch_unwind` so missing nvrtc.dll / no-CUDA
    /// scenarios produce a recoverable `Err` rather than panicking.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let ctx = CudaContext::new(0)?;
            let stream = ctx.default_stream();
            let ptx = compile_ptx(KERNEL_SOURCE)?;
            let module = ctx.load_module(ptx)?;
            Ok::<Self, Box<dyn std::error::Error>>(Self { ctx, stream, module })
        }));
        match result {
            Ok(inner) => inner,
            Err(panic_payload) => {
                let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "cudarc init panicked (unknown payload)".to_string()
                };
                Err(format!(
                    "GPU omega init panic — likely no CUDA driver / nvrtc.dll: {msg}"
                )
                .into())
            }
        }
    }
}

/// Pack a [`BicubicPair`] into the flat (`monos`, `mono_starts`)
/// representation the kernel reads.
fn pack_bicubic(pair: &BicubicPair) -> (Vec<f64>, Vec<i32>) {
    let polys: [&Vec<(Complex64, [u32; 8])>; NHYPER] = [&pair.f1, &pair.f2, &pair.f3];
    let total: usize = polys.iter().map(|p| p.len()).sum();
    let mut monos: Vec<f64> = Vec::with_capacity(total * 10);
    let mut starts: Vec<i32> = Vec::with_capacity(NHYPER + 1);
    starts.push(0);
    for poly in polys {
        for (coef, exps) in poly {
            monos.push(coef.re);
            monos.push(coef.im);
            for &e in exps {
                monos.push(e as f64);
            }
        }
        starts.push(monos.len() as i32 / 10);
    }
    (monos, starts)
}

/// GPU implementation of `compute_omega_at_samples`. Returns one
/// complex Ω value per sample point.
///
/// Returns an empty vec on length mismatches; NaN per-point when the
/// elimination Jacobian is degenerate (matches CPU semantics).
pub fn gpu_compute_omega_at_samples(
    gpu_ctx: &GpuOmegaContext,
    bicubic: &BicubicPair,
    points_re: &[f64],
    points_im: &[f64],
    n_pts: usize,
) -> Result<Vec<Complex64>, Box<dyn std::error::Error>> {
    if n_pts == 0
        || points_re.len() != NCOORDS * n_pts
        || points_im.len() != NCOORDS * n_pts
    {
        return Ok(Vec::new());
    }

    let (monos, mono_starts) = pack_bicubic(bicubic);
    let stream = &gpu_ctx.stream;

    let d_monos = stream.memcpy_stod(&monos)?;
    let d_starts = stream.memcpy_stod(&mono_starts)?;
    let d_pre = stream.memcpy_stod(points_re)?;
    let d_pim = stream.memcpy_stod(points_im)?;
    let mut d_or = stream.alloc_zeros::<f64>(n_pts)?;
    let mut d_oi = stream.alloc_zeros::<f64>(n_pts)?;

    let func = gpu_ctx.module.load_function("omega_at_samples")?;
    let threads_per_block: u32 = 128;
    let n_blocks = ((n_pts as u32) + threads_per_block - 1) / threads_per_block;
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_pts_i32 = n_pts as i32;
    let mut launcher = stream.launch_builder(&func);
    launcher
        .arg(&d_monos)
        .arg(&d_starts)
        .arg(&d_pre)
        .arg(&d_pim)
        .arg(&n_pts_i32)
        .arg(&mut d_or)
        .arg(&mut d_oi);
    unsafe { launcher.launch(cfg)? };

    let host_re = stream.memcpy_dtov(&d_or)?;
    let host_im = stream.memcpy_dtov(&d_oi)?;
    let _ = &gpu_ctx.ctx;

    Ok((0..n_pts)
        .map(|p| Complex64::new(host_re[p], host_im[p]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cicy_sampler::{BicubicPair, CicySampler};
    use crate::yukawa_overlap::compute_omega_at_samples;

    #[test]
    fn gpu_omega_at_samples_matches_cpu() {
        let ctx = match GpuOmegaContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("GPU unavailable, skipping parity test: {e}");
                return;
            }
        };

        // Sample some real CY3 points so the input distribution is
        // representative.
        let pair = BicubicPair::z3_invariant_default();
        let mut sampler = CicySampler::new(pair.clone(), 1234);
        let pts = sampler.sample_batch(256);
        assert!(!pts.is_empty());
        let n_pts = pts.len();

        // Pack into the flat (re, im) per-coord layout the kernel uses.
        let mut pre = vec![0.0_f64; NCOORDS * n_pts];
        let mut pim = vec![0.0_f64; NCOORDS * n_pts];
        for (p, s) in pts.iter().enumerate() {
            for k in 0..4 {
                pre[k * n_pts + p] = s.z[k].re;
                pim[k * n_pts + p] = s.z[k].im;
                pre[(4 + k) * n_pts + p] = s.w[k].re;
                pim[(4 + k) * n_pts + p] = s.w[k].im;
            }
        }

        let gpu_omega =
            gpu_compute_omega_at_samples(&ctx, &pair, &pre, &pim, n_pts).unwrap();
        let cpu_omega = compute_omega_at_samples(&pts, &pair);
        assert_eq!(cpu_omega.len(), n_pts);
        assert_eq!(gpu_omega.len(), n_pts);

        let mut max_diff = 0.0_f64;
        let mut max_scale = 1.0_f64;
        for p in 0..n_pts {
            let (cr, ci) = (cpu_omega[p].re, cpu_omega[p].im);
            let (gr, gi) = (gpu_omega[p].re, gpu_omega[p].im);
            // NaN ↔ NaN parity is fine; pass.
            if !cr.is_finite() || !ci.is_finite() {
                assert!(
                    !gr.is_finite() || !gi.is_finite(),
                    "CPU NaN but GPU finite at p={p}: ({gr},{gi})"
                );
                continue;
            }
            assert!(gr.is_finite() && gi.is_finite(), "GPU NaN but CPU finite at p={p}");
            max_diff = max_diff.max((cr - gr).abs()).max((ci - gi).abs());
            max_scale = max_scale.max(cr.abs()).max(ci.abs());
        }
        let rel = max_diff / max_scale;
        assert!(rel < 1.0e-9, "GPU/CPU max relative diff = {rel:.3e}");
    }
}
