//! P7.10 — GPU σ-evaluator for Schoen and TY (NCOORDS=8) Calabi-Yau threefolds.
//!
//! Mirrors the per-point η computation in
//! [`schoen_metric::compute_eta_at_point`] and
//! [`ty_metric::compute_eta_at_point`]. Both geometries share an
//! NCOORDS=8 ambient space and an NFOLD=3 tangent fold; only the
//! per-point chart frame `T_a[i]` differs (Schoen is bidegree (3,3,1)
//! on CP² × CP² × CP¹; TY is bidegree (3,3) on CP² × CP² × CP¹ with
//! Wilson line).
//!
//! Key insight for the σ-FD-Adam outer loop: the per-point chart
//! frame depends only on the sample point (and the defining polynomial
//! Jacobians at that point), NOT on the Hermitian moduli `H`. So we
//! precompute the 3×8 complex frame on CPU once and upload it as a
//! static device buffer. Each subsequent σ-evaluation only uploads
//! the (small) `H` matrix and reads back a single scalar σ.
//!
//! ## Speedup target
//!
//! At k=4, n_basis=70, n_pts=25000 the FD-Adam outer loop performs
//! `1 + n_basis² + n_basis(n_basis-1)/2 ≈ 7350` σ-evaluations per
//! Adam iteration, each requiring n_pts × O(n_basis²) work. That's
//! ~6×10¹¹ FLOPs per Adam iter — ~60 s/iter on CPU rayon. GPU target
//! is ~1 s/iter (60× speedup).
//!
//! ## Numerics
//!
//! All math is double-precision (FP64). The kernel is a line-for-line
//! port of `compute_eta_at_point` in both `schoen_metric.rs` and
//! `ty_metric.rs`. Parity test (in
//! `tests/test_cy3_sigma_gpu_parity.rs`) verifies CPU↔GPU agreement
//! to ≤ 1e-10 relative on σ for both geometries.

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

/// Number of ambient C-coordinates for both Schoen and TY (CP² × CP² × CP¹ → 8 homogeneous coords).
pub const NCOORDS: usize = 8;
/// Tangent fold (complex dim of CY3).
pub const NFOLD: usize = 3;

#[cfg(feature = "gpu")]
const SIGMA_KERNEL_SOURCE: &str = r#"
// Per-point η kernel for NCOORDS=8 CY3 (Schoen / TY).
//
// Inputs:
//   section_values [n_pts * 2n]            — basis section s_a(p), interleaved (re,im)
//   section_derivs [n_pts * NCOORDS * 2n]  — ∂_i s_a(p), per-point block 8 × (2n)
//   frames         [n_pts * 2 * NFOLD * NCOORDS]
//                                          — precomputed 3×8 complex frame T_a[i] per point
//   omega_sq       [n_pts]                 — |Ω(p)|²
//   weights        [n_pts]                 — quadrature weight
//   h_re, h_im     [n_basis * n_basis]     — Hermitian moduli H
// Output:
//   eta_out        [n_pts]                 — per-point η = det(g_tan)/|Ω|²; NaN on singular
//
// Math (matches CPU compute_eta_at_point, line-for-line):
//   1. K   = s† H s                    (real, K > 0 for SPD H)
//   2. dk  = s† H ∂_i s                (NCOORDS complex)
//   3. M   = (∂_j s)† H (∂_i s)        (NCOORDS × NCOORDS Hermitian)
//   4. g_amb_{ij} = M_{ij}/K - dk_i conj(dk_j)/K²    (NCOORDS × NCOORDS Hermitian)
//   5. g_tan_{ab} = Σ_{i,j} conj(T_a[i]) g_amb_{ij} T_b[j]    (3×3 Hermitian)
//   6. η = det(g_tan).real / |Ω|²

// Stack-size limits keyed to known max n_basis. n_basis at k=4 reaches
// ~70 in the orthogonal Z₃×Z₃ basis. Header guard with N_BASIS_MAX.
#define NCOORDS 8
#define NFOLD 3
#define N_BASIS_MAX 256

// Hermitian quadratic form (u† H v) for complex n-vectors u, v with
// real/imag interleaved; H stored as separate h_re/h_im row-major n×n.
__device__ __forceinline__ void hqf(
    const double* __restrict__ h_re,
    const double* __restrict__ h_im,
    int n,
    const double* __restrict__ u,    // 2n
    const double* __restrict__ v,    // 2n
    double* outr, double* outi
) {
    double s_re = 0.0, s_im = 0.0;
    for (int a = 0; a < n; ++a) {
        double hv_re = 0.0, hv_im = 0.0;
        for (int b = 0; b < n; ++b) {
            double hr = h_re[a * n + b];
            double hi = h_im[a * n + b];
            double vr = v[2*b], vi = v[2*b+1];
            hv_re += hr * vr - hi * vi;
            hv_im += hr * vi + hi * vr;
        }
        double ur = u[2*a], ui = u[2*a+1];
        // (u_a)* . (Hv)_a = (u_re - i u_im)(hv_re + i hv_im)
        s_re += ur * hv_re + ui * hv_im;
        s_im += ur * hv_im - ui * hv_re;
    }
    *outr = s_re;
    *outi = s_im;
}

// 3×3 complex Hermitian determinant (returns real part; result is real for Hermitian).
// g stored as 3*3 complex flat: g[2*(3*i + j)] = re, +1 = im.
__device__ __forceinline__ double det3_complex_re(const double* g) {
    #define GR(i,j) g[2*(3*(i)+(j))]
    #define GI(i,j) g[2*(3*(i)+(j))+1]
    // m1 = g11*g22 - g12*g21
    double m1r = GR(1,1)*GR(2,2) - GI(1,1)*GI(2,2) - (GR(1,2)*GR(2,1) - GI(1,2)*GI(2,1));
    double m1i = GR(1,1)*GI(2,2) + GI(1,1)*GR(2,2) - (GR(1,2)*GI(2,1) + GI(1,2)*GR(2,1));
    // m2 = g10*g22 - g12*g20
    double m2r = GR(1,0)*GR(2,2) - GI(1,0)*GI(2,2) - (GR(1,2)*GR(2,0) - GI(1,2)*GI(2,0));
    double m2i = GR(1,0)*GI(2,2) + GI(1,0)*GR(2,2) - (GR(1,2)*GI(2,0) + GI(1,2)*GR(2,0));
    // m3 = g10*g21 - g11*g20
    double m3r = GR(1,0)*GR(2,1) - GI(1,0)*GI(2,1) - (GR(1,1)*GR(2,0) - GI(1,1)*GI(2,0));
    double m3i = GR(1,0)*GI(2,1) + GI(1,0)*GR(2,1) - (GR(1,1)*GI(2,0) + GI(1,1)*GR(2,0));
    // d = g00*m1 - g01*m2 + g02*m3 ; return d.re
    double dr = GR(0,0)*m1r - GI(0,0)*m1i
              - (GR(0,1)*m2r - GI(0,1)*m2i)
              + (GR(0,2)*m3r - GI(0,2)*m3i);
    return dr;
    #undef GR
    #undef GI
}

extern "C" __global__ void cy3_sigma_eta(
    const double* __restrict__ section_values,
    const double* __restrict__ section_derivs,
    const double* __restrict__ frames,        // 2 * NFOLD * NCOORDS = 48 doubles per point
    const double* __restrict__ omega_sq,
    const double* __restrict__ weights,
    const double* __restrict__ h_re,
    const double* __restrict__ h_im,
    double* __restrict__ eta_out,
    int n_points,
    int n_basis
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_points) return;

    int two_n = 2 * n_basis;
    double w_p = weights[p];
    double om = omega_sq[p];
    if (!isfinite(w_p) || w_p <= 0.0 || !isfinite(om) || om <= 0.0) {
        eta_out[p] = nan("");
        return;
    }

    const double* s  = section_values + (long long)p * two_n;
    const double* d  = section_derivs + (long long)p * NCOORDS * two_n;
    const double* fp = frames         + (long long)p * 2 * NFOLD * NCOORDS;

    // 1. K = s† H s  (real, ≥ 0 for SPD H)
    double k_re, k_im;
    hqf(h_re, h_im, n_basis, s, s, &k_re, &k_im);
    double k_safe = (k_re > 1.0e-30) ? k_re : 1.0e-30;
    double inv_k = 1.0 / k_safe;
    double inv_k2 = inv_k * inv_k;

    // 2. dk[i] = s† H ∂_i s  (NCOORDS complex). Stored interleaved.
    double dk[2 * NCOORDS];
    for (int i = 0; i < NCOORDS; ++i) {
        double rr, ii;
        hqf(h_re, h_im, n_basis, s, d + i * two_n, &rr, &ii);
        dk[2*i]   = rr;
        dk[2*i+1] = ii;
    }

    // 3-4. g_amb[i][j] = M_{ij}/K - dk[i] conj(dk[j])/K²
    //      where M_{ij} = (∂_j s)† H (∂_i s).
    //      Note CPU code computes hqf(dsj, dsi) — i.e. u=∂_j s, v=∂_i s.
    double g_amb_re[NCOORDS * NCOORDS];
    double g_amb_im[NCOORDS * NCOORDS];
    for (int i = 0; i < NCOORDS; ++i) {
        for (int j = 0; j < NCOORDS; ++j) {
            double mr, mi;
            hqf(h_re, h_im, n_basis,
                d + j * two_n, d + i * two_n, &mr, &mi);
            // dk[i] * conj(dk[j])
            double dr = dk[2*i] * dk[2*j] + dk[2*i+1] * dk[2*j+1];
            double di = dk[2*i+1] * dk[2*j] - dk[2*i] * dk[2*j+1];
            g_amb_re[i * NCOORDS + j] = mr * inv_k - dr * inv_k2;
            g_amb_im[i * NCOORDS + j] = mi * inv_k - di * inv_k2;
        }
    }

    // 5. g_tan[a][b] = Σ_{i,j} conj(T_a[i]) g_amb[i][j] T_b[j]
    // Frame layout: fp[2*(a*NCOORDS + i)] = re, +1 = im for T_a[i].
    double g_tan[2 * NFOLD * NFOLD];
    for (int a = 0; a < NFOLD; ++a) {
        for (int b = 0; b < NFOLD; ++b) {
            double sre = 0.0, sim = 0.0;
            for (int i = 0; i < NCOORDS; ++i) {
                double tair =  fp[2*(a*NCOORDS + i)];
                double taii = -fp[2*(a*NCOORDS + i) + 1];   // conj on a
                double row_re = 0.0, row_im = 0.0;
                for (int j = 0; j < NCOORDS; ++j) {
                    double gr = g_amb_re[i * NCOORDS + j];
                    double gi = g_amb_im[i * NCOORDS + j];
                    double tbjr = fp[2*(b*NCOORDS + j)];
                    double tbji = fp[2*(b*NCOORDS + j) + 1];
                    row_re += gr * tbjr - gi * tbji;
                    row_im += gr * tbji + gi * tbjr;
                }
                sre += tair * row_re - taii * row_im;
                sim += tair * row_im + taii * row_re;
            }
            g_tan[2*(3*a + b)]     = sre;
            g_tan[2*(3*a + b) + 1] = sim;
        }
    }

    // 6. η = det(g_tan).real / |Ω|²
    double det = det3_complex_re(g_tan);
    if (!isfinite(det) || fabs(det) < 1.0e-30) {
        eta_out[p] = nan("");
        return;
    }
    eta_out[p] = fabs(det) / om;
}

// Reduction kernel: given per-point η and weights, compute σ.
//   κ = Σ w_p η_p / Σ w_p   (over points where η is finite, > 0, w finite, > 0)
//   σ = Σ w_p (η_p/κ - 1)² / Σ w_p
//
// Two-pass design: pass1 computes per-block partials of (Σw, Σwη);
// host reduces blocks → κ; pass2 computes per-block partials of
// Σw(η/κ-1)²; host reduces blocks. We avoid atomicAdd on f64 because
// it requires SM ≥ 6.0; the host-side block reduction is O(grid_dim_x)
// = O(64) work per σ-eval, which is negligible.
//
// Output layout:
//   pass1_out[2 * b + 0] = block-b partial of total_w
//   pass1_out[2 * b + 1] = block-b partial of sum_w_eta
//   pass2_out[b]         = block-b partial of sum_w_dev_sq
extern "C" __global__ void cy3_sigma_reduce_pass1(
    const double* __restrict__ eta,
    const double* __restrict__ weights,
    int n_points,
    double* __restrict__ pass1_out
) {
    extern __shared__ double sdata[];
    double* sw  = sdata;
    double* swe = sdata + blockDim.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    double acc_w = 0.0, acc_we = 0.0;
    for (int p = blockIdx.x * blockSize + tid; p < n_points; p += gridDim.x * blockSize) {
        double e = eta[p];
        double w = weights[p];
        if (!isfinite(e) || e <= 0.0 || !isfinite(w) || w <= 0.0) continue;
        acc_w  += w;
        acc_we += w * e;
    }
    sw[tid]  = acc_w;
    swe[tid] = acc_we;
    __syncthreads();
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sw[tid]  += sw[tid + s];
            swe[tid] += swe[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        pass1_out[2 * blockIdx.x + 0] = sw[0];
        pass1_out[2 * blockIdx.x + 1] = swe[0];
    }
}

extern "C" __global__ void cy3_sigma_reduce_pass2(
    const double* __restrict__ eta,
    const double* __restrict__ weights,
    double kappa,
    int n_points,
    double* __restrict__ pass2_out
) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    double inv_kappa = 1.0 / kappa;
    double acc = 0.0;
    for (int p = blockIdx.x * blockSize + tid; p < n_points; p += gridDim.x * blockSize) {
        double e = eta[p];
        double w = weights[p];
        if (!isfinite(e) || e <= 0.0 || !isfinite(w) || w <= 0.0) continue;
        double r = e * inv_kappa - 1.0;
        acc += w * r * r;
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) pass2_out[blockIdx.x] = sdata[0];
}
"#;

/// GPU σ-evaluator workspace for NCOORDS=8 CY3 geometries (Schoen / TY).
///
/// All sample-point-dependent state (section values, section derivs,
/// frames, omega_sq, weights) is uploaded **once** in `upload_static`.
/// Each subsequent `compute_sigma(h_re, h_im)` only uploads the small
/// H matrix and reads back a scalar σ.
#[cfg(feature = "gpu")]
pub struct Cy3SigmaGpu {
    pub n_points: usize,
    pub n_basis: usize,
    pub context: CudaContext,
    pub stream: Arc<CudaStream>,
    pub module: Arc<cudarc::driver::CudaModule>,

    d_section_values: CudaSlice<f64>, // n_pts * 2n
    d_section_derivs: CudaSlice<f64>, // n_pts * NCOORDS * 2n
    d_frames: CudaSlice<f64>,         // n_pts * 2 * NFOLD * NCOORDS
    d_omega_sq: CudaSlice<f64>,       // n_pts
    d_weights: CudaSlice<f64>,        // n_pts
    d_h_re: CudaSlice<f64>,           // n_basis^2
    d_h_im: CudaSlice<f64>,           // n_basis^2
    d_eta: CudaSlice<f64>,            // n_pts
    d_pass1_out: CudaSlice<f64>,      // [2 * grid_dim_r]
    d_pass2_out: CudaSlice<f64>,      // [grid_dim_r]
    grid_dim_r: u32,                   // # blocks for reduction passes
}

#[cfg(feature = "gpu")]
impl Cy3SigmaGpu {
    pub fn new(n_points: usize, n_basis: usize) -> Result<Self, CudaError> {
        if n_points == 0 || n_basis == 0 {
            return Err("Cy3SigmaGpu::new: n_points and n_basis must be > 0".into());
        }
        if n_basis > 256 {
            return Err(format!(
                "Cy3SigmaGpu: n_basis={} exceeds N_BASIS_MAX=256",
                n_basis
            )
            .into());
        }
        let context = CudarcContext::new(0)?;
        let stream = context.default_stream();
        let ptx = compile_ptx(SIGMA_KERNEL_SOURCE)?;
        let module = context.load_module(ptx)?;

        let two_n = 2 * n_basis;
        let n_basis_sq = n_basis * n_basis;
        let frame_size = 2 * NFOLD * NCOORDS;
        let grid_dim_r: u32 = 64; // grid-strided reduction

        Ok(Self {
            n_points,
            n_basis,
            context,
            stream: stream.clone(),
            module,
            d_section_values: stream.alloc_zeros::<f64>(n_points * two_n)?,
            d_section_derivs: stream.alloc_zeros::<f64>(n_points * NCOORDS * two_n)?,
            d_frames: stream.alloc_zeros::<f64>(n_points * frame_size)?,
            d_omega_sq: stream.alloc_zeros::<f64>(n_points)?,
            d_weights: stream.alloc_zeros::<f64>(n_points)?,
            d_h_re: stream.alloc_zeros::<f64>(n_basis_sq)?,
            d_h_im: stream.alloc_zeros::<f64>(n_basis_sq)?,
            d_eta: stream.alloc_zeros::<f64>(n_points)?,
            d_pass1_out: stream.alloc_zeros::<f64>(2 * grid_dim_r as usize)?,
            d_pass2_out: stream.alloc_zeros::<f64>(grid_dim_r as usize)?,
            grid_dim_r,
        })
    }

    /// Upload all sample-point-dependent state. Call this ONCE per
    /// metric-solver invocation (after Donaldson finishes, before
    /// the σ-FD-Adam loop starts).
    ///
    /// `frames` layout: `frames[p * 2 * NFOLD * NCOORDS + 2*(a*NCOORDS+i)]`
    /// = `Re T_a[i]` at point p; `+1` = `Im T_a[i]`.
    pub fn upload_static(
        &mut self,
        section_values: &[f64],
        section_derivs: &[f64],
        frames: &[f64],
        omega_sq: &[f64],
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
        if section_derivs.len() != self.n_points * NCOORDS * two_n {
            return Err(format!(
                "section_derivs length {}; expected {}",
                section_derivs.len(),
                self.n_points * NCOORDS * two_n
            )
            .into());
        }
        let frame_size = 2 * NFOLD * NCOORDS;
        if frames.len() != self.n_points * frame_size {
            return Err(format!(
                "frames length {}; expected {}",
                frames.len(),
                self.n_points * frame_size
            )
            .into());
        }
        if omega_sq.len() != self.n_points {
            return Err(format!(
                "omega_sq length {}; expected {}",
                omega_sq.len(),
                self.n_points
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
        self.stream.memcpy_htod(section_values, &mut self.d_section_values)?;
        self.stream.memcpy_htod(section_derivs, &mut self.d_section_derivs)?;
        self.stream.memcpy_htod(frames, &mut self.d_frames)?;
        self.stream.memcpy_htod(omega_sq, &mut self.d_omega_sq)?;
        self.stream.memcpy_htod(weights, &mut self.d_weights)?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// One σ-evaluation: upload (h_re, h_im), launch η kernel, two
    /// reduction passes, return scalar σ. Returns NaN on any singular
    /// path (mirrors CPU).
    pub fn compute_sigma(&mut self, h_re: &[f64], h_im: &[f64]) -> Result<f64, CudaError> {
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

        // η kernel launch (1 thread per point).
        let func_eta = self.module.load_function("cy3_sigma_eta")?;
        let block_dim_e: u32 = 64;
        let grid_dim_e: u32 = ((self.n_points as u32) + block_dim_e - 1) / block_dim_e;
        let cfg_e = LaunchConfig {
            grid_dim: (grid_dim_e, 1, 1),
            block_dim: (block_dim_e, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_pts_i32 = self.n_points as i32;
        let n_basis_i32 = self.n_basis as i32;
        {
            let mut launcher = self.stream.launch_builder(&func_eta);
            launcher
                .arg(&self.d_section_values)
                .arg(&self.d_section_derivs)
                .arg(&self.d_frames)
                .arg(&self.d_omega_sq)
                .arg(&self.d_weights)
                .arg(&self.d_h_re)
                .arg(&self.d_h_im)
                .arg(&mut self.d_eta)
                .arg(&n_pts_i32)
                .arg(&n_basis_i32);
            unsafe { launcher.launch(cfg_e)? };
        }

        // Pass 1 reduction: per-block partials of (Σw, Σwη).
        let func_p1 = self.module.load_function("cy3_sigma_reduce_pass1")?;
        let block_dim_r: u32 = 256;
        let grid_dim_r: u32 = self.grid_dim_r;
        let smem_p1 = (block_dim_r as u32) * 8 * 2;
        let cfg_p1 = LaunchConfig {
            grid_dim: (grid_dim_r, 1, 1),
            block_dim: (block_dim_r, 1, 1),
            shared_mem_bytes: smem_p1,
        };
        {
            let mut launcher = self.stream.launch_builder(&func_p1);
            launcher
                .arg(&self.d_eta)
                .arg(&self.d_weights)
                .arg(&n_pts_i32)
                .arg(&mut self.d_pass1_out);
            unsafe { launcher.launch(cfg_p1)? };
        }
        self.stream.synchronize()?;
        let p1 = self.stream.memcpy_dtov(&self.d_pass1_out)?;
        // Sum the per-block partials on host (grid_dim_r ≤ 64 entries — trivial).
        let mut total_w = 0.0_f64;
        let mut sum_w_eta = 0.0_f64;
        for b in 0..(grid_dim_r as usize) {
            total_w   += p1[2 * b + 0];
            sum_w_eta += p1[2 * b + 1];
        }
        if !total_w.is_finite() || total_w < 1.0e-30 {
            return Ok(f64::NAN);
        }
        let kappa = sum_w_eta / total_w;
        if !kappa.is_finite() || kappa.abs() < 1.0e-30 {
            return Ok(f64::NAN);
        }

        // Pass 2 reduction: per-block partials of Σ w (η/κ - 1)².
        let func_p2 = self.module.load_function("cy3_sigma_reduce_pass2")?;
        let smem_p2 = (block_dim_r as u32) * 8;
        let cfg_p2 = LaunchConfig {
            grid_dim: (grid_dim_r, 1, 1),
            block_dim: (block_dim_r, 1, 1),
            shared_mem_bytes: smem_p2,
        };
        {
            let mut launcher = self.stream.launch_builder(&func_p2);
            launcher
                .arg(&self.d_eta)
                .arg(&self.d_weights)
                .arg(&kappa)
                .arg(&n_pts_i32)
                .arg(&mut self.d_pass2_out);
            unsafe { launcher.launch(cfg_p2)? };
        }
        self.stream.synchronize()?;
        let p2 = self.stream.memcpy_dtov(&self.d_pass2_out)?;
        let mut sum_w_dev_sq = 0.0_f64;
        for b in 0..(grid_dim_r as usize) {
            sum_w_dev_sq += p2[b];
        }
        Ok(sum_w_dev_sq / total_w)
    }

    pub fn total_device_bytes(&self) -> usize {
        8 * (self.d_section_values.len()
            + self.d_section_derivs.len()
            + self.d_frames.len()
            + self.d_omega_sq.len()
            + self.d_weights.len()
            + self.d_h_re.len()
            + self.d_h_im.len()
            + self.d_eta.len()
            + self.d_pass1_out.len()
            + self.d_pass2_out.len())
    }
}

// ---------------------------------------------------------------------------
// CPU-side helpers for assembling the precomputed-frame buffer.
// ---------------------------------------------------------------------------

/// Pack a Schoen / TY chart frame `[[Complex64; 8]; 3]` into the flat
/// `[2 * NFOLD * NCOORDS] = [48]` buffer expected by the GPU kernel.
///
/// Layout: out[2*(a*NCOORDS+i)] = Re T_a[i]; out[2*(a*NCOORDS+i)+1] = Im T_a[i].
pub fn pack_frame(frame: &[[num_complex::Complex64; NCOORDS]; NFOLD], out: &mut [f64]) {
    assert_eq!(out.len(), 2 * NFOLD * NCOORDS);
    for a in 0..NFOLD {
        for i in 0..NCOORDS {
            out[2 * (a * NCOORDS + i)] = frame[a][i].re;
            out[2 * (a * NCOORDS + i) + 1] = frame[a][i].im;
        }
    }
}

/// CPU reference σ-evaluation. Used by the parity test; mirrors
/// `compute_eta_at_point` + reduction in `schoen_metric.rs` /
/// `ty_metric.rs`. Frames are passed precomputed (so the test exactly
/// matches what the GPU sees).
pub fn cpu_sigma_reference(
    section_values: &[f64],
    section_derivs: &[f64],
    frames_packed: &[f64],
    omega_sq: &[f64],
    weights: &[f64],
    h_re: &[f64],
    h_im: &[f64],
    n_points: usize,
    n_basis: usize,
) -> f64 {
    use num_complex::Complex64;
    let two_n = 2 * n_basis;
    let stride_sd = NCOORDS * two_n;
    let frame_size = 2 * NFOLD * NCOORDS;

    let mut total_w = 0.0_f64;
    let mut sum_w_eta = 0.0_f64;
    let mut etas: Vec<f64> = Vec::with_capacity(n_points);
    for p in 0..n_points {
        let w = weights[p];
        let om = omega_sq[p];
        if !w.is_finite() || w <= 0.0 || !om.is_finite() || om <= 0.0 {
            etas.push(f64::NAN);
            continue;
        }
        let s = &section_values[p * two_n..(p + 1) * two_n];
        let d = &section_derivs[p * stride_sd..(p + 1) * stride_sd];
        let fp = &frames_packed[p * frame_size..(p + 1) * frame_size];

        let (k_re, _) = hqf_cpu(h_re, h_im, n_basis, s, s);
        let k_safe = k_re.max(1.0e-30);

        let mut dk = [Complex64::new(0.0, 0.0); NCOORDS];
        for i in 0..NCOORDS {
            let dsi = &d[i * two_n..(i + 1) * two_n];
            let (rr, ii) = hqf_cpu(h_re, h_im, n_basis, s, dsi);
            dk[i] = Complex64::new(rr, ii);
        }

        let mut g_amb = [[Complex64::new(0.0, 0.0); NCOORDS]; NCOORDS];
        for i in 0..NCOORDS {
            let dsi = &d[i * two_n..(i + 1) * two_n];
            for j in 0..NCOORDS {
                let dsj = &d[j * two_n..(j + 1) * two_n];
                let (rr, ii) = hqf_cpu(h_re, h_im, n_basis, dsj, dsi);
                let m = Complex64::new(rr, ii) / k_safe;
                let dki = dk[i];
                let dkj = dk[j];
                let pre = (dki * dkj.conj()) / (k_safe * k_safe);
                g_amb[i][j] = m - pre;
            }
        }

        // Frame: fp[2*(a*NCOORDS+i)] = re, +1 = im.
        let mut frame = [[Complex64::new(0.0, 0.0); NCOORDS]; NFOLD];
        for a in 0..NFOLD {
            for i in 0..NCOORDS {
                frame[a][i] = Complex64::new(
                    fp[2 * (a * NCOORDS + i)],
                    fp[2 * (a * NCOORDS + i) + 1],
                );
            }
        }

        // g_tan[a][b] = Σ T_a*[i] g_amb[i][j] T_b[j]
        let mut g_tan = [[Complex64::new(0.0, 0.0); NFOLD]; NFOLD];
        for a in 0..NFOLD {
            for b in 0..NFOLD {
                let mut s_acc = Complex64::new(0.0, 0.0);
                for i in 0..NCOORDS {
                    let tai = frame[a][i].conj();
                    let mut row = Complex64::new(0.0, 0.0);
                    for j in 0..NCOORDS {
                        row += g_amb[i][j] * frame[b][j];
                    }
                    s_acc += tai * row;
                }
                g_tan[a][b] = s_acc;
            }
        }
        // det 3x3 (real part)
        let m1 = g_tan[1][1] * g_tan[2][2] - g_tan[1][2] * g_tan[2][1];
        let m2 = g_tan[1][0] * g_tan[2][2] - g_tan[1][2] * g_tan[2][0];
        let m3 = g_tan[1][0] * g_tan[2][1] - g_tan[1][1] * g_tan[2][0];
        let det = (g_tan[0][0] * m1 - g_tan[0][1] * m2 + g_tan[0][2] * m3).re;
        if !det.is_finite() || det.abs() < 1.0e-30 {
            etas.push(f64::NAN);
            continue;
        }
        let eta = det.abs() / om;
        etas.push(eta);
        total_w += w;
        sum_w_eta += w * eta;
    }
    if total_w < 1.0e-30 {
        return f64::NAN;
    }
    let kappa = sum_w_eta / total_w;
    if kappa.abs() < 1.0e-30 {
        return f64::NAN;
    }
    let mut sum_w_dev_sq = 0.0_f64;
    for p in 0..n_points {
        let e = etas[p];
        let w = weights[p];
        if !e.is_finite() || e <= 0.0 || !w.is_finite() || w <= 0.0 {
            continue;
        }
        let r = e / kappa - 1.0;
        sum_w_dev_sq += w * r * r;
    }
    sum_w_dev_sq / total_w
}

#[inline]
fn hqf_cpu(h_re: &[f64], h_im: &[f64], n: usize, u: &[f64], v: &[f64]) -> (f64, f64) {
    let mut s_re = 0.0_f64;
    let mut s_im = 0.0_f64;
    for a in 0..n {
        let mut hv_re = 0.0_f64;
        let mut hv_im = 0.0_f64;
        for b in 0..n {
            let hr = h_re[a * n + b];
            let hi = h_im[a * n + b];
            let vr = v[2 * b];
            let vi = v[2 * b + 1];
            hv_re += hr * vr - hi * vi;
            hv_im += hr * vi + hi * vr;
        }
        let ur = u[2 * a];
        let ui = u[2 * a + 1];
        s_re += ur * hv_re + ui * hv_im;
        s_im += ur * hv_im - ui * hv_re;
    }
    (s_re, s_im)
}
