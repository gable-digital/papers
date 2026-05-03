//! P7.11 — GPU Donaldson T-operator for Schoen and TY (NCOORDS=8).
//!
//! Mirrors the per-iteration T(G) accumulation in
//! [`schoen_metric::donaldson_iteration`] and
//! [`ty_metric::donaldson_iteration`]. Both geometries share an
//! identical T-operator structure on `H^0(O(k,k,…))`:
//!
//! 1. Per-point Bergman kernel  `K(p) = s_p^† H s_p`           (real)
//! 2. Outer product accumulator `h_pair[a,b] = Σ_p (w_p/K_p) · s_a(p) · conj(s_b(p))`
//!
//! Step 2 is the dominant bottleneck (n_basis² × n_pts work). At
//! production scale (k=4, n_basis≈70, n_pts=25000), this is
//! ~1.2×10⁸ flops per iteration. With 100 iterations and 20 seeds
//! the CPU bill is hours.
//!
//! ## GPU strategy
//!
//! Same upload-once / launch-many pattern as `cy3_sigma_gpu.rs`:
//!
//! * `section_values` (n_pts × 2n) and `weights` (n_pts) are uploaded
//!   ONCE at solver entry; they're moduli-independent.
//! * Per Donaldson iteration we upload H (small: n_basis² complex)
//!   and read back `h_pair` (n_basis² complex). The 2N×2N real-block
//!   inversion + symmetrisation + trace renorm stays on CPU (cheap).
//!
//! Two kernels:
//!   * `compute_k`           — 1 thread per point computes K(p) = s_p† H s_p
//!   * `accumulate_h_pair`   — 1 block per (a,b) pair; threads reduce over p
//!
//! ## Numerics
//!
//! Double-precision (FP64). Parity test in
//! `tests/test_cy3_donaldson_gpu_parity.rs` verifies CPU↔GPU full
//! Schoen + TY donaldson iterations agree to ≤ 1e-10 relative on
//! `final_sigma_residual`.

#[cfg(feature = "gpu")]
use cudarc::driver::{
    CudaContext as CudarcContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
#[cfg(feature = "gpu")]
use cudarc::nvrtc::compile_ptx;
#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
pub type CudaContext = Arc<CudarcContext>;
pub type CudaError = Box<dyn std::error::Error + Send + Sync>;

#[cfg(feature = "gpu")]
const DONALDSON_KERNEL_SOURCE: &str = r#"
// Per-point K(p) = s_p^† H s_p.
//
// Layout:
//   section_values[p * (2n) + 2*a + 0] = Re s_a(p)
//   section_values[p * (2n) + 2*a + 1] = Im s_a(p)
//   h_re[a*n + b], h_im[a*n + b]       = entries of Hermitian H (n×n)
//
//   K = Σ_{a,b} s_a^* H_{ab} s_b
//     = Σ_{a,b} [ H_re ( s_a_re s_b_re + s_a_im s_b_im )
//              + H_im ( s_a_re s_b_im - s_a_im s_b_re ) ]
extern "C" __global__ void cy3_donaldson_compute_k(
    const double* __restrict__ section_values,
    const double* __restrict__ h_re,
    const double* __restrict__ h_im,
    int n_points,
    int n_basis,
    double* __restrict__ k_out
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_points) return;
    int two_n = 2 * n_basis;
    const double* s = section_values + (long long)p * two_n;
    double k = 0.0;
    for (int a = 0; a < n_basis; ++a) {
        double sar = s[2*a];
        double sai = s[2*a + 1];
        for (int b = 0; b < n_basis; ++b) {
            double sbr = s[2*b];
            double sbi = s[2*b + 1];
            double hr = h_re[a * n_basis + b];
            double hi = h_im[a * n_basis + b];
            k += hr * (sar * sbr + sai * sbi)
               + hi * (sar * sbi - sai * sbr);
        }
    }
    if (!(k > 1.0e-30)) k = 1.0e-30;
    k_out[p] = k;
}

// Outer-product accumulator:
//   h_pair_re[a,b] = Σ_p (w_p / K_p) (s_a_re s_b_re + s_a_im s_b_im)
//   h_pair_im[a,b] = Σ_p (w_p / K_p) (s_a_re s_b_im - s_a_im s_b_re)
//
// One thread block per (a,b). Threads in the block grid-stride
// over points; per-thread partial then warp/block reduction in
// shared memory. Output is written ONCE per block by thread 0.
//
// Block size (blockDim.x) must be a power of two ≤ 1024.
extern "C" __global__ void cy3_donaldson_accum_h_pair(
    const double* __restrict__ section_values,
    const double* __restrict__ k_values,
    const double* __restrict__ weights,
    int n_points,
    int n_basis,
    double* __restrict__ h_pair_re,   // n_basis * n_basis
    double* __restrict__ h_pair_im    // n_basis * n_basis
) {
    int ab = blockIdx.x;
    int n2 = n_basis * n_basis;
    if (ab >= n2) return;
    int a = ab / n_basis;
    int b = ab - a * n_basis;
    int two_n = 2 * n_basis;

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    extern __shared__ double sdata[];
    double* sre = sdata;                  // blockSize doubles
    double* sim = sdata + blockSize;      // blockSize doubles

    double acc_re = 0.0;
    double acc_im = 0.0;
    for (int p = tid; p < n_points; p += blockSize) {
        const double* s = section_values + (long long)p * two_n;
        double sar = s[2*a];
        double sai = s[2*a + 1];
        double sbr = s[2*b];
        double sbi = s[2*b + 1];
        double w = weights[p];
        double kp = k_values[p];
        // Skip non-finite / non-positive entries.
        if (!isfinite(w) || w <= 0.0) continue;
        if (!isfinite(kp) || kp <= 0.0) continue;
        double factor = w / kp;
        acc_re += factor * (sar * sbr + sai * sbi);
        acc_im += factor * (sar * sbi - sai * sbr);
    }
    sre[tid] = acc_re;
    sim[tid] = acc_im;
    __syncthreads();
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sre[tid] += sre[tid + s];
            sim[tid] += sim[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        h_pair_re[ab] = sre[0];
        h_pair_im[ab] = sim[0];
    }
}
"#;

/// GPU Donaldson T-operator workspace for NCOORDS=8 CY3 geometries.
///
/// Owns device buffers for `section_values`, `weights`, and per-iter
/// scratch (`K`, H, output `h_pair`). The CY3-specific 2N×2N
/// real-block inversion + symmetrisation step stays on CPU because
/// it's small (O(n_basis³)) and not worth a GPU round-trip.
#[cfg(feature = "gpu")]
pub struct Cy3DonaldsonGpu {
    pub n_points: usize,
    pub n_basis: usize,
    pub context: CudaContext,
    pub stream: Arc<CudaStream>,
    pub module: Arc<cudarc::driver::CudaModule>,

    d_section_values: CudaSlice<f64>,
    d_weights: CudaSlice<f64>,
    d_k_values: CudaSlice<f64>,
    d_h_re: CudaSlice<f64>,
    d_h_im: CudaSlice<f64>,
    d_h_pair_re: CudaSlice<f64>,
    d_h_pair_im: CudaSlice<f64>,
}

#[cfg(feature = "gpu")]
impl Cy3DonaldsonGpu {
    pub fn new(n_points: usize, n_basis: usize) -> Result<Self, CudaError> {
        if n_points == 0 || n_basis == 0 {
            return Err("Cy3DonaldsonGpu::new: n_points and n_basis must be > 0".into());
        }
        let context = CudarcContext::new(0)?;
        let stream = context.default_stream();
        let ptx = compile_ptx(DONALDSON_KERNEL_SOURCE)?;
        let module = context.load_module(ptx)?;
        let two_n = 2 * n_basis;
        let n_basis_sq = n_basis * n_basis;
        Ok(Self {
            n_points,
            n_basis,
            context,
            stream: stream.clone(),
            module,
            d_section_values: stream.alloc_zeros::<f64>(n_points * two_n)?,
            d_weights: stream.alloc_zeros::<f64>(n_points)?,
            d_k_values: stream.alloc_zeros::<f64>(n_points)?,
            d_h_re: stream.alloc_zeros::<f64>(n_basis_sq)?,
            d_h_im: stream.alloc_zeros::<f64>(n_basis_sq)?,
            d_h_pair_re: stream.alloc_zeros::<f64>(n_basis_sq)?,
            d_h_pair_im: stream.alloc_zeros::<f64>(n_basis_sq)?,
        })
    }

    /// Upload the moduli-independent per-point data. Call ONCE per
    /// solver invocation (after sample → section evaluation, before
    /// the Donaldson loop).
    pub fn upload_static(
        &mut self,
        section_values: &[f64],
        weights: &[f64],
    ) -> Result<(), CudaError> {
        let two_n = 2 * self.n_basis;
        if section_values.len() != self.n_points * two_n {
            return Err(format!(
                "section_values length {}; expected {}",
                section_values.len(),
                self.n_points * two_n
            )
            .into());
        }
        if weights.len() != self.n_points {
            return Err(format!(
                "weights length {}; expected {}",
                weights.len(),
                self.n_points
            )
            .into());
        }
        self.stream
            .memcpy_htod(section_values, &mut self.d_section_values)?;
        self.stream.memcpy_htod(weights, &mut self.d_weights)?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// One T-operator evaluation:
    ///   1. upload (h_re, h_im),
    ///   2. launch K kernel (1 thread per point),
    ///   3. launch accumulator kernel (1 block per (a,b)),
    ///   4. read back `h_pair_re`, `h_pair_im` (n_basis × n_basis each).
    ///
    /// Caller is responsible for the post-T(G) trace renormalisation,
    /// 2N×2N real-block inversion, and Hermitian symmetrisation. See
    /// `Cy3DonaldsonGpu::donaldson_step` for the full one-shot API.
    pub fn t_operator_raw(
        &mut self,
        h_re: &[f64],
        h_im: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), CudaError> {
        let n_basis_sq = self.n_basis * self.n_basis;
        if h_re.len() != n_basis_sq || h_im.len() != n_basis_sq {
            return Err(format!(
                "h_re/h_im length {}/{}; expected {}",
                h_re.len(),
                h_im.len(),
                n_basis_sq
            )
            .into());
        }
        self.stream.memcpy_htod(h_re, &mut self.d_h_re)?;
        self.stream.memcpy_htod(h_im, &mut self.d_h_im)?;

        // Pass 1: K(p)
        let func_k = self.module.load_function("cy3_donaldson_compute_k")?;
        let block_k: u32 = 64;
        let grid_k: u32 = ((self.n_points as u32) + block_k - 1) / block_k;
        let cfg_k = LaunchConfig {
            grid_dim: (grid_k, 1, 1),
            block_dim: (block_k, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_pts_i32 = self.n_points as i32;
        let n_basis_i32 = self.n_basis as i32;
        {
            let mut launcher = self.stream.launch_builder(&func_k);
            launcher
                .arg(&self.d_section_values)
                .arg(&self.d_h_re)
                .arg(&self.d_h_im)
                .arg(&n_pts_i32)
                .arg(&n_basis_i32)
                .arg(&mut self.d_k_values);
            unsafe { launcher.launch(cfg_k)? };
        }

        // Pass 2: outer product accumulator. One block per (a,b).
        let func_acc = self.module.load_function("cy3_donaldson_accum_h_pair")?;
        let block_acc: u32 = 256;
        let grid_acc: u32 = n_basis_sq as u32;
        let smem_acc: u32 = (block_acc as u32) * 8 * 2; // re + im
        let cfg_acc = LaunchConfig {
            grid_dim: (grid_acc, 1, 1),
            block_dim: (block_acc, 1, 1),
            shared_mem_bytes: smem_acc,
        };
        {
            let mut launcher = self.stream.launch_builder(&func_acc);
            launcher
                .arg(&self.d_section_values)
                .arg(&self.d_k_values)
                .arg(&self.d_weights)
                .arg(&n_pts_i32)
                .arg(&n_basis_i32)
                .arg(&mut self.d_h_pair_re)
                .arg(&mut self.d_h_pair_im);
            unsafe { launcher.launch(cfg_acc)? };
        }
        self.stream.synchronize()?;
        let h_pair_re = self.stream.memcpy_dtov(&self.d_h_pair_re)?;
        let h_pair_im = self.stream.memcpy_dtov(&self.d_h_pair_im)?;
        Ok((h_pair_re, h_pair_im))
    }

    pub fn total_device_bytes(&self) -> usize {
        8 * (self.d_section_values.len()
            + self.d_weights.len()
            + self.d_k_values.len()
            + self.d_h_re.len()
            + self.d_h_im.len()
            + self.d_h_pair_re.len()
            + self.d_h_pair_im.len())
    }
}

// ---------------------------------------------------------------------------
// CPU reference for parity testing.
// ---------------------------------------------------------------------------

/// CPU reference implementation of the T-operator outer-product step.
/// Mirrors `donaldson_iteration` in `schoen_metric.rs` /
/// `ty_metric.rs` lines that compute `K(p)` and `h_pair[a,b]`.
/// Used by the parity test; production CPU path is the original
/// rayon-parallel inline version in those files.
pub fn cpu_t_operator_reference(
    section_values: &[f64],
    weights: &[f64],
    h_re: &[f64],
    h_im: &[f64],
    n_points: usize,
    n_basis: usize,
) -> (Vec<f64>, Vec<f64>) {
    let two_n = 2 * n_basis;
    let mut k_values = vec![0.0_f64; n_points];
    for p in 0..n_points {
        let s = &section_values[p * two_n..(p + 1) * two_n];
        let mut k = 0.0_f64;
        for a in 0..n_basis {
            let sar = s[2 * a];
            let sai = s[2 * a + 1];
            for b in 0..n_basis {
                let sbr = s[2 * b];
                let sbi = s[2 * b + 1];
                let hr = h_re[a * n_basis + b];
                let hi = h_im[a * n_basis + b];
                k += hr * (sar * sbr + sai * sbi) + hi * (sar * sbi - sai * sbr);
            }
        }
        k_values[p] = k.max(1.0e-30);
    }
    let n_basis_sq = n_basis * n_basis;
    let mut h_pair_re = vec![0.0_f64; n_basis_sq];
    let mut h_pair_im = vec![0.0_f64; n_basis_sq];
    for a in 0..n_basis {
        for b in 0..n_basis {
            let mut acc_re = 0.0_f64;
            let mut acc_im = 0.0_f64;
            for p in 0..n_points {
                let s = &section_values[p * two_n..(p + 1) * two_n];
                let sar = s[2 * a];
                let sai = s[2 * a + 1];
                let sbr = s[2 * b];
                let sbi = s[2 * b + 1];
                let factor = weights[p] / k_values[p];
                acc_re += factor * (sar * sbr + sai * sbi);
                acc_im += factor * (sar * sbi - sai * sbr);
            }
            h_pair_re[a * n_basis + b] = acc_re;
            h_pair_im[a * n_basis + b] = acc_im;
        }
    }
    (h_pair_re, h_pair_im)
}
