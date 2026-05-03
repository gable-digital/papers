//! CUDA-accelerated polynomial-seed evaluation for monad-bundle zero
//! modes.
//!
//! Backs [`crate::zero_modes::evaluate_polynomial_seeds`] with a single
//! kernel that, per `(mode, point)` pair, picks a `B`-generator
//! `b_lines[a % n_b]`, chooses a `(z_idx, w_idx)` rotation per mode-
//! group, and evaluates `z[z_idx]^d1 · w[w_idx]^d2` on-device.
//!
//! The work per thread is a small repeated-squaring complex power; the
//! parallelism is the natural `n_modes × n_pts` grid. For the typical
//! 5σ pipeline (`n_modes ≤ 18`, `n_pts ≤ ~10⁴`) the launch is a single
//! short-lived kernel with negligible host↔device transfer overhead
//! once the bundle data is uploaded.
//!
//! Same dispatch shape as [`crate::gpu_yukawa`] /
//! [`crate::gpu_omega`]: the public `gpu_evaluate_polynomial_seeds`
//! entry point is `Result<Vec<Complex64>>`-returning, and
//! [`GpuPolySeedsContext::new`] wraps `cudarc` initialisation in
//! `catch_unwind` so missing-CUDA hosts return an error rather than
//! aborting the process.

use cudarc::driver::{CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use num_complex::Complex64;
use std::sync::Arc;

const KERNEL_SOURCE: &str = r#"
// ψ_a(p) = z_{z_idx}^{d1} · w_{w_idx}^{d2}   for each (mode a, point p).
// b_lines is uploaded as a packed (d1, d2) pair per generator.
// Negative degrees produce a zero ψ_a (matches CPU).
extern "C" __global__ void polynomial_seeds(
    const int*    __restrict__ b_lines,    // 2 * n_b ints, (d1, d2) per gen
    int n_b,
    const double* __restrict__ pts_re,     // 8 * n_pts
    const double* __restrict__ pts_im,
    int n_pts,
    int n_modes,
    double* __restrict__ psi_re,           // n_modes * n_pts
    double* __restrict__ psi_im
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int a = blockIdx.y;
    if (p >= n_pts || a >= n_modes) return;

    int i   = a % n_b;
    int rot = a / n_b;
    int z_idx = rot % 4;
    int w_idx = (rot / 4) % 4;

    int d1_signed = b_lines[2 * i + 0];
    int d2_signed = b_lines[2 * i + 1];
    if (d1_signed < 0 || d2_signed < 0) {
        psi_re[a * n_pts + p] = 0.0;
        psi_im[a * n_pts + p] = 0.0;
        return;
    }
    int d1 = d1_signed;
    int d2 = d2_signed;

    double zr = pts_re[z_idx * n_pts + p];
    double zi = pts_im[z_idx * n_pts + p];
    double wr = pts_re[(4 + w_idx) * n_pts + p];
    double wi = pts_im[(4 + w_idx) * n_pts + p];

    // Repeated squaring for z^d1.
    double rr = 1.0, ri = 0.0;
    {
        double br = zr, bi = zi;
        int e = d1;
        while (e > 0) {
            if (e & 1) {
                double nr = rr * br - ri * bi;
                double ni = rr * bi + ri * br;
                rr = nr; ri = ni;
            }
            e >>= 1;
            if (e > 0) {
                double nbr = br * br - bi * bi;
                double nbi = 2.0 * br * bi;
                br = nbr; bi = nbi;
            }
        }
    }
    // Repeated squaring for w^d2 then multiply.
    double sr = 1.0, si = 0.0;
    {
        double br = wr, bi = wi;
        int e = d2;
        while (e > 0) {
            if (e & 1) {
                double nr = sr * br - si * bi;
                double ni = sr * bi + si * br;
                sr = nr; si = ni;
            }
            e >>= 1;
            if (e > 0) {
                double nbr = br * br - bi * bi;
                double nbi = 2.0 * br * bi;
                br = nbr; bi = nbi;
            }
        }
    }
    double or_ = rr * sr - ri * si;
    double oi  = rr * si + ri * sr;
    psi_re[a * n_pts + p] = or_;
    psi_im[a * n_pts + p] = oi;
}
"#;

pub struct GpuPolySeedsContext {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
}

impl GpuPolySeedsContext {
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
                    "GPU poly-seed init panic — likely no CUDA driver / nvrtc.dll: {msg}"
                )
                .into())
            }
        }
    }
}

/// GPU implementation of `evaluate_polynomial_seeds`.
///
/// `b_lines` is the bundle's flat `Vec<[i32; 2]>` of B-generator
/// degrees; `points_re/im` are the per-coordinate-per-point flat
/// arrays (`points_re[k * n_pts + p]` is the real part of the
/// `k`-th coordinate of point `p`); returns `n_modes * n_pts`
/// complex values laid out as `[mode][point]` (mode slowest), the
/// same layout the CPU function returns.
pub fn gpu_evaluate_polynomial_seeds(
    gpu_ctx: &GpuPolySeedsContext,
    b_lines: &[[i32; 2]],
    points_re: &[f64],
    points_im: &[f64],
    n_pts: usize,
    n_modes: u32,
) -> Result<Vec<Complex64>, Box<dyn std::error::Error>> {
    if n_pts == 0 || n_modes == 0 || b_lines.is_empty() {
        return Ok(Vec::new());
    }
    if points_re.len() != 8 * n_pts || points_im.len() != 8 * n_pts {
        return Ok(Vec::new());
    }

    let n_modes_us = n_modes as usize;
    let mut b_lines_flat: Vec<i32> = Vec::with_capacity(2 * b_lines.len());
    for d in b_lines {
        b_lines_flat.push(d[0]);
        b_lines_flat.push(d[1]);
    }
    let n_b_i32 = b_lines.len() as i32;
    let n_pts_i32 = n_pts as i32;
    let n_modes_i32 = n_modes as i32;

    let stream = &gpu_ctx.stream;
    let d_b = stream.memcpy_stod(&b_lines_flat)?;
    let d_pre = stream.memcpy_stod(points_re)?;
    let d_pim = stream.memcpy_stod(points_im)?;
    let mut d_psi_re = stream.alloc_zeros::<f64>(n_modes_us * n_pts)?;
    let mut d_psi_im = stream.alloc_zeros::<f64>(n_modes_us * n_pts)?;

    let func = gpu_ctx.module.load_function("polynomial_seeds")?;
    let threads_per_block: u32 = 128;
    let nblocks_x = ((n_pts as u32) + threads_per_block - 1) / threads_per_block;
    let cfg = LaunchConfig {
        grid_dim: (nblocks_x, n_modes, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launcher = stream.launch_builder(&func);
    launcher
        .arg(&d_b)
        .arg(&n_b_i32)
        .arg(&d_pre)
        .arg(&d_pim)
        .arg(&n_pts_i32)
        .arg(&n_modes_i32)
        .arg(&mut d_psi_re)
        .arg(&mut d_psi_im);
    unsafe { launcher.launch(cfg)? };

    let host_re = stream.memcpy_dtov(&d_psi_re)?;
    let host_im = stream.memcpy_dtov(&d_psi_im)?;
    let _ = &gpu_ctx.ctx;

    Ok((0..n_modes_us * n_pts)
        .map(|i| Complex64::new(host_re[i], host_im[i]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zero_modes::{evaluate_polynomial_seeds, MonadBundle};

    #[test]
    fn gpu_polynomial_seeds_matches_cpu() {
        let ctx = match GpuPolySeedsContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("GPU unavailable, skipping parity test: {e}");
                return;
            }
        };

        let bundle = MonadBundle::anderson_lukas_palti_example();
        let n_pts = 256;
        let n_modes: u32 = 12; // exercise mode rotation (n_modes > n_b)

        // Synthetic points on S^7 × S^7 with deterministic non-trivial
        // complex values.
        let mut state: u64 = 0xC0FFEE;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) & ((1 << 53) - 1)) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let mut pts_complex: Vec<[Complex64; 8]> = Vec::with_capacity(n_pts);
        let mut pts_re = vec![0.0_f64; 8 * n_pts];
        let mut pts_im = vec![0.0_f64; 8 * n_pts];
        for p in 0..n_pts {
            let mut pt = [Complex64::new(0.0, 0.0); 8];
            for k in 0..8 {
                pt[k] = Complex64::new(next(), next());
                pts_re[k * n_pts + p] = pt[k].re;
                pts_im[k * n_pts + p] = pt[k].im;
            }
            pts_complex.push(pt);
        }

        let cpu = evaluate_polynomial_seeds(&bundle, &pts_complex, n_modes);
        let gpu = gpu_evaluate_polynomial_seeds(
            &ctx,
            &bundle.b_lines,
            &pts_re,
            &pts_im,
            n_pts,
            n_modes,
        )
        .unwrap();

        assert_eq!(cpu.len(), gpu.len());
        let mut max_diff = 0.0_f64;
        let mut max_scale = 1.0_f64;
        for (c, g) in cpu.iter().zip(gpu.iter()) {
            max_diff = max_diff.max((c.re - g.re).abs()).max((c.im - g.im).abs());
            max_scale = max_scale.max(c.re.abs()).max(c.im.abs());
        }
        let rel = max_diff / max_scale;
        assert!(rel < 1.0e-12, "GPU/CPU max relative diff = {rel:.3e}");
    }
}
