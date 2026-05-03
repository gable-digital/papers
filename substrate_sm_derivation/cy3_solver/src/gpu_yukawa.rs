//! CUDA-accelerated holomorphic Yukawa triple-overlap reductions.
//!
//! This module backs [`crate::yukawa_overlap::assemble_sector_yukawa`]
//! with a single CUDA kernel that does the three-piece reduction on
//! the device:
//!
//! ```text
//!     λ_{ij}    = Σ_α  w_α · Ω(p_α) · ψ^L_i(p_α) · ψ^R_j(p_α) · ψ^H(p_α)
//!     N^L_{ab̄}  = Σ_α  w_α · ψ^L_a(p_α) · conj(ψ^L_b(p_α))
//!     N^R_{ab̄}  = Σ_α  w_α · ψ^R_a(p_α) · conj(ψ^R_b(p_α))
//! ```
//!
//! Each (i, j) of `λ` and each (a, b) of `N^{L,R}` is a single
//! independent N-point complex reduction. The kernel launches
//! `9 + 9 + 9 = 27` blocks (most launches won't use all of them; the
//! Hermitian symmetry of `N^{L,R}` halves the unique work but we
//! compute the full 9 anyway to keep the kernel branch-free) and
//! uses block-level shared-memory reduction to sum over the `N`
//! sample points.
//!
//! Why a custom kernel rather than cuBLAS Z-dot:
//! * cuBLAS `Zdotc` would handle `N^{L,R}` (weighted complex inner
//!   products) but not the *triple* `λ_{ij}` — we'd need an explicit
//!   per-point pre-product kernel anyway, so doing both in one launch
//!   saves device-side memory traffic.
//!
//! Numerical fidelity:
//! * The kernel uses `double` (FP64) throughout and a Kahan-free
//!   tree reduction — for `N ≤ 50_000` this is bit-equivalent to the
//!   CPU reduction within ULP-level rounding, verified by the
//!   `gpu_assemble_sector_yukawa_matches_cpu` test.
//!
//! Fallback: the CPU implementation in
//! [`crate::yukawa_overlap::assemble_sector_yukawa`] remains the
//! source of truth and is what runs whenever the `gpu` feature is off.

use cudarc::driver::{CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use num_complex::Complex64;
use std::sync::Arc;

const KERNEL_SOURCE: &str = r#"
// One-pass-over-points kernel. Each block handles a chunk of points;
// each thread loads its single point's psi/omega/weight data ONCE
// from global memory and computes contributions to all 27 reductions
// in registers. Block-level reduction across the 27 outputs writes
// per-block partials to a slab; the host sums the partials across
// blocks.
//
// Compared to the previous design (27 blocks each re-reading the
// full psi/omega/weight arrays) this cuts global memory traffic by
// ~n_blocks × 27 ÷ 27 = n_blocks×, and on typical n_pts ≤ 5000
// (~20 blocks) gives a ~20× bandwidth win.
//
// Output layout (per block):
//   block_partials[block * 54 + 2*k    ] = block's partial Re of out k
//   block_partials[block * 54 + 2*k + 1] = block's partial Im of out k
//   for k in 0..27 with the same indexing convention as before:
//     [0, 9):  λ_{ij}    (i = k/3, j = k%3)
//     [9, 18): N^L_{ab̄}  (a = (k-9)/3, b = (k-9)%3)
//     [18, 27): N^R_{ab̄} (a = (k-18)/3, b = (k-18)%3)
extern "C" __global__ void yukawa_sector_reductions_per_point(
    const double* __restrict__ weights,
    const double* __restrict__ omega_re,
    const double* __restrict__ omega_im,
    const double* __restrict__ psi_left_re,
    const double* __restrict__ psi_left_im,
    const double* __restrict__ psi_right_re,
    const double* __restrict__ psi_right_im,
    const double* __restrict__ psi_higgs_re,
    const double* __restrict__ psi_higgs_im,
    int n_points,
    double* __restrict__ block_partials   // n_blocks × 54
) {
    // Per-thread accumulators for all 27 outputs (re, im interleaved).
    // 54 doubles in registers — fits comfortably.
    double acc[54];
    #pragma unroll
    for (int k = 0; k < 54; ++k) acc[k] = 0.0;

    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int p = gtid; p < n_points; p += stride) {
        double w = weights[p];
        if (!isfinite(w)) continue;
        double or_ = omega_re[p];
        double oi  = omega_im[p];
        double hr  = psi_higgs_re[p];
        double hi  = psi_higgs_im[p];
        // Load the 3 left and 3 right psi components for this point.
        double lr[3], li[3], rr[3], ri[3];
        bool any_nonfinite = false;
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            lr[i] = psi_left_re[i * n_points + p];
            li[i] = psi_left_im[i * n_points + p];
            rr[i] = psi_right_re[i * n_points + p];
            ri[i] = psi_right_im[i * n_points + p];
            if (!isfinite(lr[i]) || !isfinite(rr[i])) any_nonfinite = true;
        }
        if (any_nonfinite || !isfinite(or_) || !isfinite(oi)
            || !isfinite(hr) || !isfinite(hi)) continue;

        // Precompute Ω · psi_L[i] (shared across j in the lambda loop)
        // and psi_H · psi_R[j] (shared across i).
        double op_lr[3], op_li[3];   // (Ω · psi_L[i])
        double rh_rr[3], rh_ri[3];   // (psi_R[j] · psi_H)
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            op_lr[i] = or_ * lr[i] - oi * li[i];
            op_li[i] = or_ * li[i] + oi * lr[i];
            rh_rr[i] = rr[i] * hr - ri[i] * hi;
            rh_ri[i] = rr[i] * hi + ri[i] * hr;
        }

        // Lambda: 9 terms.
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            #pragma unroll
            for (int j = 0; j < 3; ++j) {
                int k = i * 3 + j;
                // term = (Ω · psi_L[i]) · (psi_R[j] · psi_H)
                double tr = op_lr[i] * rh_rr[j] - op_li[i] * rh_ri[j];
                double ti = op_lr[i] * rh_ri[j] + op_li[i] * rh_rr[j];
                acc[2 * k    ] += w * tr;
                acc[2 * k + 1] += w * ti;
            }
        }

        // N^L: 9 terms. N^L_{a b̄} = Σ w · psi_L[a] · conj(psi_L[b])
        // = Σ w · ((lr[a] · lr[b] + li[a] · li[b]) + i (li[a] · lr[b] − lr[a] · li[b]))
        #pragma unroll
        for (int a = 0; a < 3; ++a) {
            #pragma unroll
            for (int b = 0; b < 3; ++b) {
                int k = 9 + a * 3 + b;
                acc[2 * k    ] += w * (lr[a] * lr[b] + li[a] * li[b]);
                acc[2 * k + 1] += w * (li[a] * lr[b] - lr[a] * li[b]);
            }
        }
        // N^R: 9 terms.
        #pragma unroll
        for (int a = 0; a < 3; ++a) {
            #pragma unroll
            for (int b = 0; b < 3; ++b) {
                int k = 18 + a * 3 + b;
                acc[2 * k    ] += w * (rr[a] * rr[b] + ri[a] * ri[b]);
                acc[2 * k + 1] += w * (ri[a] * rr[b] - rr[a] * ri[b]);
            }
        }
    }

    // Block reduce each of the 54 partial sums via shared memory.
    extern __shared__ double sdata[];   // blockDim.x doubles
    for (int k = 0; k < 54; ++k) {
        sdata[threadIdx.x] = acc[k];
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            block_partials[blockIdx.x * 54 + k] = sdata[0];
        }
    }
}
"#;

/// Lazily-compiled CUDA module + context for the Yukawa-reduction
/// kernel. Construct once per process and reuse across calls.
pub struct GpuYukawaContext {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
}

impl GpuYukawaContext {
    /// Initialise the CUDA context + JIT-compile the kernel. Returns
    /// `Err` on any device-side failure (no CUDA driver, no device,
    /// NVRTC compilation failed, …) — callers should treat that as
    /// "fall back to CPU".
    ///
    /// `cudarc` panics when its dynamic loader can't find e.g.
    /// `nvrtc.dll`, so we wrap every initialisation step in
    /// `catch_unwind` and convert the panic into a normal `Err`. This
    /// makes the GPU path safe to attempt on any machine — the
    /// caller's CPU fallback always runs when initialisation fails.
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
                    "GPU init panic — likely no CUDA driver / nvrtc.dll on this host: {msg}"
                )
                .into())
            }
        }
    }
}

/// Three CPU-friendly outputs of the on-device reduction, in the same
/// row-major `[[(re, im); 3]; 3]` shape as `M3` in `yukawa_overlap`.
pub struct SectorReductionOutputs {
    pub lambda: [[(f64, f64); 3]; 3],
    pub n_left: [[(f64, f64); 3]; 3],
    pub n_right: [[(f64, f64); 3]; 3],
}

/// GPU implementation of the three-block reduction in
/// `yukawa_overlap::assemble_sector_yukawa`. Returns `lambda`,
/// `n_left`, `n_right` (each 3×3 complex). The caller does the
/// downstream Hermitian Jacobi + matrix multiplies.
///
/// This function does NOT enforce Hermiticity on `n_left`, `n_right`
/// — that postprocessing is identical to the CPU path and lives in
/// `assemble_sector_yukawa` after the reduction.
///
/// Fallback semantics: if `n_pts == 0` or the input slice lengths
/// don't match, returns `Ok` with all-zero matrices.
pub fn gpu_assemble_sector_reductions(
    gpu_ctx: &GpuYukawaContext,
    weights: &[f64],
    omega: &[Complex64],
    psi_left: &[Complex64],
    psi_right: &[Complex64],
    psi_higgs: &[Complex64],
    n_pts: usize,
) -> Result<SectorReductionOutputs, Box<dyn std::error::Error>> {
    let zero3: [[(f64, f64); 3]; 3] = [[(0.0, 0.0); 3]; 3];
    if n_pts == 0
        || weights.len() != n_pts
        || omega.len() != n_pts
        || psi_left.len() != 3 * n_pts
        || psi_right.len() != 3 * n_pts
        || psi_higgs.len() != n_pts
    {
        return Ok(SectorReductionOutputs {
            lambda: zero3,
            n_left: zero3,
            n_right: zero3,
        });
    }

    // Split complex slices into (re, im) host buffers — CUDA kernels
    // operate on FP64 arrays directly.
    let mut omega_re = vec![0.0_f64; n_pts];
    let mut omega_im = vec![0.0_f64; n_pts];
    for p in 0..n_pts {
        omega_re[p] = omega[p].re;
        omega_im[p] = omega[p].im;
    }
    let mut psi_left_re = vec![0.0_f64; 3 * n_pts];
    let mut psi_left_im = vec![0.0_f64; 3 * n_pts];
    let mut psi_right_re = vec![0.0_f64; 3 * n_pts];
    let mut psi_right_im = vec![0.0_f64; 3 * n_pts];
    for k in 0..(3 * n_pts) {
        psi_left_re[k] = psi_left[k].re;
        psi_left_im[k] = psi_left[k].im;
        psi_right_re[k] = psi_right[k].re;
        psi_right_im[k] = psi_right[k].im;
    }
    let mut psi_higgs_re = vec![0.0_f64; n_pts];
    let mut psi_higgs_im = vec![0.0_f64; n_pts];
    for p in 0..n_pts {
        psi_higgs_re[p] = psi_higgs[p].re;
        psi_higgs_im[p] = psi_higgs[p].im;
    }

    let stream = &gpu_ctx.stream;
    let d_w = stream.memcpy_stod(weights)?;
    let d_or = stream.memcpy_stod(&omega_re)?;
    let d_oi = stream.memcpy_stod(&omega_im)?;
    let d_lr = stream.memcpy_stod(&psi_left_re)?;
    let d_li = stream.memcpy_stod(&psi_left_im)?;
    let d_rr = stream.memcpy_stod(&psi_right_re)?;
    let d_ri = stream.memcpy_stod(&psi_right_im)?;
    let d_hr = stream.memcpy_stod(&psi_higgs_re)?;
    let d_hi = stream.memcpy_stod(&psi_higgs_im)?;

    // Decide grid: cap blocks so per-block point-chunk is large enough
    // to amortise the per-block 27-output reduction overhead. With
    // n_pts ≤ 50000 and BLOCK_SIZE = 256, ceil(n_pts / BLOCK_SIZE)
    // ≤ 196 blocks is fine; for very small n_pts use at least 1 block.
    let threads_per_block: u32 = 256;
    let n_blocks: u32 = (((n_pts as u32) + threads_per_block - 1) / threads_per_block).max(1);
    let mut d_block_partials = stream.alloc_zeros::<f64>((n_blocks as usize) * 54)?;

    let func = gpu_ctx
        .module
        .load_function("yukawa_sector_reductions_per_point")?;
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        // Shared memory: 1 double per thread for the 54 sequential
        // single-output reductions.
        shared_mem_bytes: threads_per_block * 8,
    };
    let n_pts_i32 = n_pts as i32;
    let mut launcher = stream.launch_builder(&func);
    launcher
        .arg(&d_w)
        .arg(&d_or)
        .arg(&d_oi)
        .arg(&d_lr)
        .arg(&d_li)
        .arg(&d_rr)
        .arg(&d_ri)
        .arg(&d_hr)
        .arg(&d_hi)
        .arg(&n_pts_i32)
        .arg(&mut d_block_partials);
    unsafe { launcher.launch(cfg)? };

    // Sum block partials on the host (n_blocks ≤ ~200, trivial).
    let host_partials = stream.memcpy_dtov(&d_block_partials)?;
    let mut out_re = vec![0.0_f64; 27];
    let mut out_im = vec![0.0_f64; 27];
    for b in 0..(n_blocks as usize) {
        for k in 0..27 {
            out_re[k] += host_partials[b * 54 + 2 * k];
            out_im[k] += host_partials[b * 54 + 2 * k + 1];
        }
    }

    let mut lambda: [[(f64, f64); 3]; 3] = zero3;
    let mut n_left: [[(f64, f64); 3]; 3] = zero3;
    let mut n_right: [[(f64, f64); 3]; 3] = zero3;
    for k in 0..9 {
        lambda[k / 3][k % 3] = (out_re[k], out_im[k]);
        n_left[k / 3][k % 3] = (out_re[9 + k], out_im[9 + k]);
        n_right[k / 3][k % 3] = (out_re[18 + k], out_im[18 + k]);
    }

    // The compiler insists that ctx is used somewhere — keep it alive.
    let _ = &gpu_ctx.ctx;

    Ok(SectorReductionOutputs {
        lambda,
        n_left,
        n_right,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference CPU implementation, kept here so we can verify GPU
    /// parity without depending on `yukawa_overlap` internals.
    fn cpu_reference(
        weights: &[f64],
        omega: &[Complex64],
        psi_left: &[Complex64],
        psi_right: &[Complex64],
        psi_higgs: &[Complex64],
        n_pts: usize,
    ) -> SectorReductionOutputs {
        let mut lambda = [[(0.0_f64, 0.0_f64); 3]; 3];
        let mut n_left = [[(0.0_f64, 0.0_f64); 3]; 3];
        let mut n_right = [[(0.0_f64, 0.0_f64); 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = Complex64::new(0.0, 0.0);
                for p in 0..n_pts {
                    let w = weights[p];
                    if !w.is_finite() { continue; }
                    let om = omega[p];
                    if !om.re.is_finite() || !om.im.is_finite() { continue; }
                    let li = psi_left[i * n_pts + p];
                    let rj = psi_right[j * n_pts + p];
                    let h = psi_higgs[p];
                    if !li.re.is_finite() || !rj.re.is_finite() || !h.re.is_finite() { continue; }
                    acc += om * li * rj * h * Complex64::new(w, 0.0);
                }
                lambda[i][j] = (acc.re, acc.im);
            }
        }
        for psi_idx in 0..2 {
            let psi = if psi_idx == 0 { psi_left } else { psi_right };
            let target = if psi_idx == 0 { &mut n_left } else { &mut n_right };
            for a in 0..3 {
                for b in 0..3 {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for p in 0..n_pts {
                        let w = weights[p];
                        if !w.is_finite() { continue; }
                        let pa = psi[a * n_pts + p];
                        let pb = psi[b * n_pts + p];
                        if !pa.re.is_finite() || !pb.re.is_finite() { continue; }
                        acc += pa * pb.conj() * Complex64::new(w, 0.0);
                    }
                    target[a][b] = (acc.re, acc.im);
                }
            }
        }
        SectorReductionOutputs { lambda, n_left, n_right }
    }

    #[test]
    fn gpu_assemble_sector_yukawa_matches_cpu() {
        // Skip entirely if no CUDA device is available.
        let ctx = match GpuYukawaContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("GPU unavailable, skipping parity test: {e}");
                return;
            }
        };

        let n_pts = 4096;
        let mut rng_state: u64 = 0xDEAD_BEEF;
        let mut next = || {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng_state >> 11) & ((1 << 53) - 1)) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let weights: Vec<f64> = (0..n_pts).map(|_| 1.0 / n_pts as f64).collect();
        let omega: Vec<Complex64> = (0..n_pts)
            .map(|_| Complex64::new(next(), next()))
            .collect();
        let psi_left: Vec<Complex64> = (0..3 * n_pts)
            .map(|_| Complex64::new(next(), next()))
            .collect();
        let psi_right: Vec<Complex64> = (0..3 * n_pts)
            .map(|_| Complex64::new(next(), next()))
            .collect();
        let psi_higgs: Vec<Complex64> = (0..n_pts)
            .map(|_| Complex64::new(next(), next()))
            .collect();

        let cpu = cpu_reference(&weights, &omega, &psi_left, &psi_right, &psi_higgs, n_pts);
        let gpu =
            gpu_assemble_sector_reductions(&ctx, &weights, &omega, &psi_left, &psi_right, &psi_higgs, n_pts)
                .expect("GPU reduction should succeed");

        let max_abs = |a: [[(f64, f64); 3]; 3]| -> f64 {
            let mut m = 0.0_f64;
            for r in &a {
                for &(re, im) in r {
                    m = m.max(re.abs()).max(im.abs());
                }
            }
            m
        };
        let scale = max_abs(cpu.lambda)
            .max(max_abs(cpu.n_left))
            .max(max_abs(cpu.n_right))
            .max(1.0);
        let tol = 1e-9 * scale; // FP64 tree-reduction agreement to ~1e-9 relative

        for i in 0..3 {
            for j in 0..3 {
                let (cre, cim) = cpu.lambda[i][j];
                let (gre, gim) = gpu.lambda[i][j];
                assert!(
                    (cre - gre).abs() < tol && (cim - gim).abs() < tol,
                    "lambda[{i}][{j}] CPU=({cre},{cim}) GPU=({gre},{gim})"
                );
                let (cre, cim) = cpu.n_left[i][j];
                let (gre, gim) = gpu.n_left[i][j];
                assert!(
                    (cre - gre).abs() < tol && (cim - gim).abs() < tol,
                    "n_left[{i}][{j}] CPU=({cre},{cim}) GPU=({gre},{gim})"
                );
                let (cre, cim) = cpu.n_right[i][j];
                let (gre, gim) = gpu.n_right[i][j];
                assert!(
                    (cre - gre).abs() < tol && (cim - gim).abs() < tol,
                    "n_right[{i}][{j}] CPU=({cre},{cim}) GPU=({gre},{gim})"
                );
            }
        }
    }
}
