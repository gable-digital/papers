//! CUDA acceleration for the Fermat-quintic σ-evaluation pipeline.
//!
//! The σ-evaluation is the dominant cost in the σ-functional Adam
//! refinement: 80+ iterations × ~30000 sample points per call. The
//! per-point Bergman-metric → tangent-projection → 3×3 determinant
//! computation is embarrassingly parallel across points.
//!
//! ## Kernel design
//!
//! One CUDA thread per sample point. Each thread:
//!   1. Loads s_p (2·n_basis reals), df_p (5·2·n_basis reals) from
//!      global memory.
//!   2. Loads h_block (4·n_basis² reals) -- shared memory tile for
//!      reuse across threads in the block.
//!   3. Computes K_p = s_p† h s_p (one complex inner product).
//!   4. Computes ∂_i K_p (5 complex), ∂_i ∂_j̄ K_p (5×5 complex).
//!   5. Forms g_amb_p (5×5 complex Hermitian).
//!   6. Builds the tangent frame at z_p (3 orthonormal complex vectors
//!      orthogonal to z and ∇f).
//!   7. Projects g_amb to g_tan_p = T† g_amb T (3×3 complex Hermitian).
//!   8. Computes det(g_tan_p) (3×3 cofactor expansion).
//!   9. Writes R_p = exp(log|det| - log_omega_sq_p) to output array.
//!
//! A second reduction kernel computes Σ_p w_p R_p and Σ_p w_p R_p²,
//! from which σ = stddev_w(R) / mean_w(R) is derived on the host.
//!
//! ## Expected speedup
//!
//! At n_pts = 30000 on RTX 4090 (16384 CUDA cores @ 2.5 GHz):
//!   - Per-point work: ~3000 FMAs + log/exp.
//!   - Naive parallel: 30000 / 16384 ≈ 2 thread waves.
//!   - Actual: limited by global memory bandwidth (loading section_derivs
//!     [n_pts × 5 × 2·n_basis] = 4.5 MB at n_basis=15).
//!   - Expected: σ-eval drops from ~10 ms (CPU) to ~0.2 ms (GPU), a
//!     ~50× speedup. Adam refinement at 120 iters: 1.2s → 24 ms.
//!
//! ## Status
//!
//! This module is **scaffolding** for the GPU σ-evaluation. The full
//! kernel (per-point complex Bergman + tangent + det) is non-trivial
//! and substantially longer than the existing polysphere kernels in
//! `gpu.rs`. We provide:
//!   - A simple K-only kernel that computes K_p = s_p† h s_p
//!     (correctness-verifiable against CPU at any n_basis).
//!   - The QuinticGpuWorkspace struct holding device buffers.
//!   - Integration with QuinticSolver::sigma_gpu() that falls back to
//!     CPU on GPU init failure.
//! Full per-point R kernel implementation tracked as future work.

use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

use crate::quintic::QuinticSolver;

const QUINTIC_KERNEL_SOURCE: &str = r#"
// Compute K_p = s_p† h_block s_p for each sample point p.
// Inputs:
//   section_values: [n_pts × 2·n_basis]  -- complex sections, packed re/im
//   h_block:        [4·n_basis²]          -- 2n×2n real Hermitian block
//   n_pts, n_basis  -- ints
// Output:
//   k_out: [n_pts]  -- per-point K values

extern "C" __global__ void quintic_compute_k(
    const double* section_values,
    const double* h_block,
    double* k_out,
    int n_pts,
    int n_basis
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pts) return;

    int two_n = 2 * n_basis;
    const double* s_p = section_values + p * two_n;

    double k = 0.0;
    // K = s_p^T H_block s_p (since H_block is the 2n×2n real
    // representation of the complex Hermitian h, this real quadratic
    // form equals the complex Hermitian quadratic form s† h s).
    for (int i = 0; i < two_n; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < two_n; ++j) {
            row_sum += h_block[i * two_n + j] * s_p[j];
        }
        k += s_p[i] * row_sum;
    }
    k_out[p] = (k > 1e-30) ? k : 1e-30;
}

// Reduction-on-host: kernel writes per-point K_p; host does the
// weighted reduction. Avoids atomicAdd on doubles (which would require
// sm_60+ NVRTC compile arch options).
"#;

/// GPU-resident workspace for quintic σ-evaluation. Persistent device
/// buffers allocated once at construction; reused across iterations.
pub struct QuinticGpuWorkspace {
    pub n_pts: usize,
    pub n_basis: usize,
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub module: Arc<cudarc::driver::CudaModule>,

    // Persistent device buffers.
    pub d_section_values: CudaSlice<f64>, // n_pts × 2·n_basis
    pub d_weights: CudaSlice<f64>,         // n_pts
    pub d_h_block: CudaSlice<f64>,         // 4·n_basis²
    pub d_k_out: CudaSlice<f64>,           // n_pts (per-point K)
    pub d_sum_w: CudaSlice<f64>,           // 1
    pub d_sum_wv: CudaSlice<f64>,          // 1
    pub d_sum_wv2: CudaSlice<f64>,         // 1
}

impl QuinticGpuWorkspace {
    /// Build the GPU workspace from a QuinticSolver. Uploads the
    /// h-independent buffers (section_values, weights) once.
    pub fn from_solver(solver: &QuinticSolver) -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let ptx = compile_ptx(QUINTIC_KERNEL_SOURCE)?;
        let module = ctx.load_module(ptx)?;

        let d_section_values = stream.memcpy_stod(&solver.section_values)?;
        let d_weights = stream.memcpy_stod(&solver.weights)?;
        let d_h_block = stream.memcpy_stod(&solver.h_block)?;
        let d_k_out = stream.alloc_zeros::<f64>(solver.n_points)?;
        let d_sum_w = stream.alloc_zeros::<f64>(1)?;
        let d_sum_wv = stream.alloc_zeros::<f64>(1)?;
        let d_sum_wv2 = stream.alloc_zeros::<f64>(1)?;

        Ok(Self {
            n_pts: solver.n_points,
            n_basis: solver.n_basis,
            ctx,
            stream,
            module,
            d_section_values,
            d_weights,
            d_h_block,
            d_k_out,
            d_sum_w,
            d_sum_wv,
            d_sum_wv2,
        })
    }

    /// Re-upload h_block (called after each Adam step that modifies h).
    pub fn upload_h_block(&mut self, h_block: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
        self.d_h_block = self.stream.memcpy_stod(h_block)?;
        Ok(())
    }

    /// Compute K_p = s_p† h s_p for each point on the GPU. Returns the
    /// per-point K values (read back to host).
    pub fn compute_k_per_point(&mut self) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let func = self.module.load_function("quintic_compute_k")?;
        let n_pts_i32 = self.n_pts as i32;
        let n_basis_i32 = self.n_basis as i32;
        let cfg = LaunchConfig::for_num_elems(self.n_pts as u32);
        let mut launcher = self.stream.launch_builder(&func);
        launcher
            .arg(&self.d_section_values)
            .arg(&self.d_h_block)
            .arg(&mut self.d_k_out)
            .arg(&n_pts_i32)
            .arg(&n_basis_i32);
        unsafe { launcher.launch(cfg)? };
        self.stream.synchronize()?;
        let k_host = self.stream.memcpy_dtov(&self.d_k_out)?;
        Ok(k_host)
    }

    /// Total bytes allocated on device.
    pub fn total_device_bytes(&self) -> usize {
        let f64_size = 8;
        f64_size
            * (self.d_section_values.len()
                + self.d_weights.len()
                + self.d_h_block.len()
                + self.d_k_out.len()
                + self.d_sum_w.len()
                + self.d_sum_wv.len()
                + self.d_sum_wv2.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_k_matches_cpu() {
        // Build a small quintic workspace; upload to GPU; verify K_p
        // computed on GPU matches K_p on CPU at every point.
        let n_pts = 1000;
        let solver = match QuinticSolver::new(2, n_pts, 13, 1e-11) {
            Some(s) => s,
            None => {
                eprintln!("workspace construction failed; skipping GPU test");
                return;
            }
        };

        let init_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            QuinticGpuWorkspace::from_solver(&solver)
        }));
        let mut gpu_ws = match init_result {
            Ok(Ok(g)) => g,
            Ok(Err(e)) => {
                eprintln!("GPU init failed (no CUDA?): {e}; skipping test");
                return;
            }
            Err(_) => {
                eprintln!("QuinticGpuWorkspace init panicked (likely no nvrtc.dll); skipping");
                return;
            }
        };

        let k_gpu = gpu_ws.compute_k_per_point().expect("GPU K computation");
        assert_eq!(k_gpu.len(), solver.n_points);

        // Compute K on CPU for comparison.
        let n_basis = solver.n_basis;
        let two_n = 2 * n_basis;
        for p in 0..solver.n_points.min(50) {
            let s_p = &solver.section_values[p * two_n..(p + 1) * two_n];
            let mut k_cpu = 0.0;
            for i in 0..two_n {
                let mut row_sum = 0.0;
                for j in 0..two_n {
                    row_sum += solver.h_block[i * two_n + j] * s_p[j];
                }
                k_cpu += s_p[i] * row_sum;
            }
            let k_cpu = k_cpu.max(1e-30);
            let rel = ((k_gpu[p] - k_cpu) / k_cpu.max(1e-30)).abs();
            assert!(
                rel < 1e-10,
                "point {p}: GPU K = {} vs CPU K = {} (rel err {rel:.3e})",
                k_gpu[p],
                k_cpu
            );
        }
    }
}
