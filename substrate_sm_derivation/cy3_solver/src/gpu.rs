//! CUDA-accelerated discrimination pipeline for the RTX 4090 (or any
//! CUDA-capable device). Uses cudarc for the device management and
//! cuBLAS for the heavy GEMM. Custom CUDA kernels (compiled at runtime
//! via NVRTC) handle the per-point work.
//!
//! Memory layout: every persistent buffer is allocated once on the
//! device (mirroring the CPU `DiscriminationWorkspace`) and reused
//! across iterations. The CPU only holds the final mass-eigenvalue
//! result.

use cudarc::cublas::{CudaBlas, GemmConfig, Gemm};
use cudarc::driver::sys::CUstream;
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

use crate::workspace::build_degree2_monomials;

const KERNEL_SOURCE: &str = r#"
extern "C" __global__ void sample_points_polysphere(
    double* points,
    unsigned long long seed,
    int n_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    // Per-thread linear-congruential RNG for reproducibility
    unsigned long long state = (seed + (unsigned long long)idx * 0x9e3779b97f4a7c15ULL)
                               * 6364136223846793005ULL + 1442695040888963407ULL;

    double z[8];
    for (int k = 0; k < 8; ++k) {
        // Box-Muller: two LCG draws
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        double u1 = (double)((state >> 11) & ((1ULL << 53) - 1)) / (double)(1ULL << 53);
        if (u1 < 1e-12) u1 = 1e-12;
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        double u2 = (double)((state >> 11) & ((1ULL << 53) - 1)) / (double)(1ULL << 53);
        double r = sqrt(-2.0 * log(u1));
        z[k] = r * cos(2.0 * 3.14159265358979323846 * u2);
    }

    double n1 = sqrt(z[0]*z[0] + z[1]*z[1] + z[2]*z[2] + z[3]*z[3]);
    double n2 = sqrt(z[4]*z[4] + z[5]*z[5] + z[6]*z[6] + z[7]*z[7]);
    if (n1 < 1e-10) n1 = 1e-10;
    if (n2 < 1e-10) n2 = 1e-10;
    for (int k = 0; k < 4; ++k) z[k] /= n1;
    for (int k = 4; k < 8; ++k) z[k] /= n2;

    for (int k = 0; k < 8; ++k) {
        points[idx * 8 + k] = z[k];
    }
}

extern "C" __global__ void evaluate_section_basis_d2(
    const double* points,
    const int* monomials,    // n_basis x 8 ints, flattened
    double* section_values,  // n_points x n_basis
    int n_points,
    int n_basis
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    const double* z = &points[idx * 8];
    double pow_table[24];
    for (int k = 0; k < 8; ++k) {
        pow_table[k * 3] = 1.0;
        pow_table[k * 3 + 1] = z[k];
        pow_table[k * 3 + 2] = z[k] * z[k];
    }

    for (int j = 0; j < n_basis; ++j) {
        const int* m = &monomials[j * 8];
        double v = pow_table[m[0]]
                 * pow_table[3 + m[1]]
                 * pow_table[6 + m[2]]
                 * pow_table[9 + m[3]]
                 * pow_table[12 + m[4]]
                 * pow_table[15 + m[5]]
                 * pow_table[18 + m[6]]
                 * pow_table[21 + m[7]];
        section_values[idx * n_basis + j] = v;
    }
}

extern "C" __global__ void per_row_dot_donaldson(
    const double* t,         // n_points x n_basis (S @ h_inv)
    const double* s,         // n_points x n_basis (section values)
    double* weights,         // n_points
    int n_points,
    int n_basis
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    const double* t_i = &t[idx * n_basis];
    const double* s_i = &s[idx * n_basis];
    double w = 0.0;
    for (int a = 0; a < n_basis; ++a) {
        w += t_i[a] * s_i[a];
    }
    weights[idx] = w > 1e-12 ? w : 1e-12;
}

extern "C" __global__ void compute_sw_buffer(
    const double* s,         // n_points x n_basis
    const double* weights,   // n_points
    double* sw,              // n_points x n_basis
    int n_points,
    int n_basis
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points * n_basis) return;
    int i = idx / n_basis;
    double inv_sqrt_w = rsqrt(weights[i]);
    sw[idx] = s[idx] * inv_sqrt_w;
}

extern "C" __global__ void normalize_h(
    double* h,
    double scale,
    int n_basis
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_basis * n_basis) return;
    h[idx] *= scale;
}

extern "C" __global__ void compute_residual_diff(
    const double* h,
    const double* h_new,
    double* diff_sq_per_block,
    int n_total
) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    double diff = 0.0;
    if (idx < n_total) {
        double d = h_new[idx] - h[idx];
        diff = d * d;
    }
    sdata[tid] = diff;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) diff_sq_per_block[blockIdx.x] = sdata[0];
}

extern "C" __global__ void yukawa_kernel(
    const double* points,
    const double* centers,
    double* y_out,           // n_modes^3
    int n_points,
    int n_modes,
    int dim
) {
    // One block computes the contribution from one chunk of points to
    // a single output element y[ijk]. We index (i, j, k) by blockIdx.x.
    // Each thread in the block handles a subset of points and reduces.
    int ijk = blockIdx.x;
    int n_modes_sq = n_modes * n_modes;
    int i = ijk / n_modes_sq;
    int rem = ijk % n_modes_sq;
    int j = rem / n_modes;
    int k = rem % n_modes;

    extern __shared__ double sbuf[];
    int tid = threadIdx.x;
    double local_sum = 0.0;

    // Each thread strides through points
    for (int p = tid; p < n_points; p += blockDim.x) {
        const double* pt = &points[p * 8];
        double phi_i = 0.0, phi_j = 0.0, phi_k = 0.0;

        // phi_m = exp(-0.5 * sum_d (pt[d] - center[m,d])^2)
        const double* ci = &centers[i * 8];
        const double* cj = &centers[j * 8];
        const double* ck = &centers[k * 8];
        double r2_i = 0.0, r2_j = 0.0, r2_k = 0.0;
        for (int d = 0; d < dim; ++d) {
            double di = pt[d] - ci[d];
            double dj = pt[d] - cj[d];
            double dk = pt[d] - ck[d];
            r2_i += di * di;
            r2_j += dj * dj;
            r2_k += dk * dk;
        }
        phi_i = exp(-0.5 * r2_i);
        phi_j = exp(-0.5 * r2_j);
        phi_k = exp(-0.5 * r2_k);
        local_sum += phi_i * phi_j * phi_k;
    }

    sbuf[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sbuf[tid] += sbuf[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        y_out[ijk] = sbuf[0] / (double)n_points;
    }
}
"#;

/// GPU workspace mirror of DiscriminationWorkspace. Holds CUDA device
/// buffers and cached cuBLAS handle.
pub struct GpuDiscriminationWorkspace {
    pub n_points: usize,
    pub n_basis: usize,
    pub n_modes: usize,
    pub max_iter: usize,

    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub blas: CudaBlas,

    // Persistent device buffers (allocated once)
    pub d_points: CudaSlice<f64>,           // n_points * 8
    pub d_monomials: CudaSlice<i32>,        // n_basis * 8
    pub d_section_values: CudaSlice<f64>,   // n_points * n_basis
    pub d_h: CudaSlice<f64>,                // n_basis * n_basis
    pub d_h_new: CudaSlice<f64>,            // n_basis * n_basis
    pub d_h_inv: CudaSlice<f64>,            // n_basis * n_basis
    pub d_t_matrix: CudaSlice<f64>,         // n_points * n_basis
    pub d_sw: CudaSlice<f64>,               // n_points * n_basis
    pub d_weights: CudaSlice<f64>,          // n_points
    pub d_yukawa_centers: CudaSlice<f64>,   // n_modes * 8
    pub d_yukawa_tensor: CudaSlice<f64>,    // n_modes^3
    pub d_residual_blocks: CudaSlice<f64>,  // for residual reduction

    // Compiled kernel functions
    pub module: Arc<cudarc::driver::CudaModule>,
}

impl GpuDiscriminationWorkspace {
    pub fn new(
        n_points: usize,
        n_basis: usize,
        n_modes: usize,
        max_iter: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();

        let ptx = compile_ptx(KERNEL_SOURCE)?;
        let module = ctx.load_module(ptx)?;

        let blas = CudaBlas::new(stream.clone())?;

        let monomials_cpu = build_degree2_monomials();
        let monomials_flat: Vec<i32> = monomials_cpu
            .iter()
            .flat_map(|m| m.iter().map(|x| *x as i32))
            .collect();

        let d_points = stream.alloc_zeros::<f64>(n_points * 8)?;
        let d_monomials = stream.memcpy_stod(&monomials_flat)?;
        let d_section_values = stream.alloc_zeros::<f64>(n_points * n_basis)?;
        let d_h = stream.alloc_zeros::<f64>(n_basis * n_basis)?;
        let d_h_new = stream.alloc_zeros::<f64>(n_basis * n_basis)?;
        let d_h_inv = stream.alloc_zeros::<f64>(n_basis * n_basis)?;
        let d_t_matrix = stream.alloc_zeros::<f64>(n_points * n_basis)?;
        let d_sw = stream.alloc_zeros::<f64>(n_points * n_basis)?;
        let d_weights = stream.alloc_zeros::<f64>(n_points)?;
        let d_yukawa_centers = stream.alloc_zeros::<f64>(n_modes * 8)?;
        let d_yukawa_tensor = stream.alloc_zeros::<f64>(n_modes * n_modes * n_modes)?;

        let n_blocks_residual = (n_basis * n_basis + 255) / 256;
        let d_residual_blocks = stream.alloc_zeros::<f64>(n_blocks_residual)?;

        Ok(Self {
            n_points,
            n_basis,
            n_modes,
            max_iter,
            ctx,
            stream,
            blas,
            d_points,
            d_monomials,
            d_section_values,
            d_h,
            d_h_new,
            d_h_inv,
            d_t_matrix,
            d_sw,
            d_weights,
            d_yukawa_centers,
            d_yukawa_tensor,
            d_residual_blocks,
            module,
        })
    }

    pub fn total_device_bytes(&self) -> usize {
        let n_p = self.n_points;
        let n_b = self.n_basis;
        let n_m = self.n_modes;
        8 * (n_p * 8
            + n_p * n_b
            + n_b * n_b * 3
            + n_p * n_b * 2
            + n_p
            + n_m * 8
            + n_m * n_m * n_m)
            + 4 * n_b * 8
    }
}

/// Sample points on the polysphere, in-place into d_points.
pub fn gpu_sample_points(
    ws: &mut GpuDiscriminationWorkspace,
    seed: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let func = ws.module.load_function("sample_points_polysphere")?;
    let cfg = LaunchConfig::for_num_elems(ws.n_points as u32);
    let n_points_i32 = ws.n_points as i32;
    let mut launcher = ws.stream.launch_builder(&func);
    launcher
        .arg(&mut ws.d_points)
        .arg(&seed)
        .arg(&n_points_i32);
    unsafe { launcher.launch(cfg)? };
    Ok(())
}

/// Evaluate section basis on GPU.
pub fn gpu_evaluate_section_basis(
    ws: &mut GpuDiscriminationWorkspace,
) -> Result<(), Box<dyn std::error::Error>> {
    let func = ws.module.load_function("evaluate_section_basis_d2")?;
    let cfg = LaunchConfig::for_num_elems(ws.n_points as u32);
    let n_points_i32 = ws.n_points as i32;
    let n_basis_i32 = ws.n_basis as i32;
    let mut launcher = ws.stream.launch_builder(&func);
    launcher
        .arg(&ws.d_points)
        .arg(&ws.d_monomials)
        .arg(&mut ws.d_section_values)
        .arg(&n_points_i32)
        .arg(&n_basis_i32);
    unsafe { launcher.launch(cfg)? };
    Ok(())
}

/// One Donaldson iteration on GPU. Uses cuBLAS GEMM and custom kernels.
pub fn gpu_donaldson_iter(
    ws: &mut GpuDiscriminationWorkspace,
) -> Result<(), Box<dyn std::error::Error>> {
    let n_p = ws.n_points;
    let n_b = ws.n_basis;

    // Step 1: invert h. cuBLAS doesn't have a direct inverse; we use
    // LU factorisation + back-substitution. For simplicity at the
    // n_basis = 100 size we copy h to host, invert there with our linalg
    // module, and copy h_inv back to device. This is a 80KB transfer +
    // a few-ms inversion — small relative to the GEMM work below.
    let h_host = ws.stream.memcpy_dtov(&ws.d_h)?;
    let mut h_lu = vec![0.0f64; n_b * n_b];
    let mut perm = vec![0usize; n_b];
    let mut h_inv_host = vec![0.0f64; n_b * n_b];
    let mut col_buf = vec![0.0f64; n_b];
    crate::linalg::invert(&h_host, n_b, &mut h_lu, &mut perm, &mut h_inv_host, &mut col_buf)?;
    ws.stream.memcpy_htod(&h_inv_host, &mut ws.d_h_inv)?;

    // Step 2: T = S @ h_inv via cuBLAS GEMM
    // cuBLAS assumes column-major. Our matrices are row-major. The
    // identity (A @ B)^T = B^T @ A^T means: row-major A @ B equals
    // column-major B^T @ A^T. We can compute T_rowmajor = S_row @ H_inv_row
    // by calling cuBLAS gemm with B = S, A = H_inv (treating them as
    // column-major where they appear transposed).
    let alpha: f64 = 1.0;
    let beta: f64 = 0.0;
    unsafe {
        ws.blas.gemm(
            GemmConfig {
                transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                m: n_b as i32,
                n: n_p as i32,
                k: n_b as i32,
                alpha,
                lda: n_b as i32,
                ldb: n_b as i32,
                beta,
                ldc: n_b as i32,
            },
            &ws.d_h_inv,
            &ws.d_section_values,
            &mut ws.d_t_matrix,
        )?;
    }

    // Step 3: weights[i] = T[i] · S[i] (custom kernel)
    {
        let func = ws.module.load_function("per_row_dot_donaldson")?;
        let cfg = LaunchConfig::for_num_elems(n_p as u32);
        let n_p_i32 = n_p as i32;
        let n_b_i32 = n_b as i32;
        let mut launcher = ws.stream.launch_builder(&func);
        launcher
            .arg(&ws.d_t_matrix)
            .arg(&ws.d_section_values)
            .arg(&mut ws.d_weights)
            .arg(&n_p_i32)
            .arg(&n_b_i32);
        unsafe { launcher.launch(cfg)? };
    }

    // Step 4: sw[i, a] = s[i, a] / sqrt(w_i) (custom kernel)
    {
        let func = ws.module.load_function("compute_sw_buffer")?;
        let cfg = LaunchConfig::for_num_elems((n_p * n_b) as u32);
        let n_p_i32 = n_p as i32;
        let n_b_i32 = n_b as i32;
        let mut launcher = ws.stream.launch_builder(&func);
        launcher
            .arg(&ws.d_section_values)
            .arg(&ws.d_weights)
            .arg(&mut ws.d_sw)
            .arg(&n_p_i32)
            .arg(&n_b_i32);
        unsafe { launcher.launch(cfg)? };
    }

    // Step 5: h_new = sw^T @ sw via cuBLAS GEMM.
    // For row-major sw @ sw^T = h_new (n_b x n_b), we want
    // h_new[a,b] = sum_i sw[i,a] sw[i,b]. cuBLAS column-major:
    // treat sw as col-major (n_b x n_p) -> sw^T @ sw is sw col-major
    // multiplied by its transpose. Use cuBLAS syrk for a more direct
    // formulation, but gemm with explicit transpose works too.
    unsafe {
        ws.blas.gemm(
            GemmConfig {
                transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
                m: n_b as i32,
                n: n_b as i32,
                k: n_p as i32,
                alpha,
                lda: n_b as i32,
                ldb: n_b as i32,
                beta,
                ldc: n_b as i32,
            },
            &ws.d_sw,
            &ws.d_sw,
            &mut ws.d_h_new,
        )?;
    }

    // Step 6: normalise h_new to fixed trace (host computes scale, kernel applies)
    let h_new_host = ws.stream.memcpy_dtov(&ws.d_h_new)?;
    let mut trace = 0.0;
    for a in 0..n_b {
        trace += h_new_host[a * n_b + a];
    }
    if trace > 1e-10 {
        let scale: f64 = (n_b as f64) / trace;
        let func = ws.module.load_function("normalize_h")?;
        let cfg = LaunchConfig::for_num_elems((n_b * n_b) as u32);
        let n_b_i32 = (n_b * n_b) as i32;
        let mut launcher = ws.stream.launch_builder(&func);
        launcher
            .arg(&mut ws.d_h_new)
            .arg(&scale)
            .arg(&n_b_i32);
        unsafe { launcher.launch(cfg)? };
    }

    Ok(())
}

/// Run Donaldson balancing solve on GPU. Initialises h to identity once.
pub fn gpu_donaldson_solve(
    ws: &mut GpuDiscriminationWorkspace,
    tol: f64,
) -> Result<usize, Box<dyn std::error::Error>> {
    // Initialise h = I on host then copy
    let n_b = ws.n_basis;
    let mut h_init = vec![0.0f64; n_b * n_b];
    for a in 0..n_b {
        h_init[a * n_b + a] = 1.0;
    }
    ws.stream.memcpy_htod(&h_init, &mut ws.d_h)?;

    let mut iters = 0;
    for _ in 0..ws.max_iter {
        gpu_donaldson_iter(ws)?;
        iters += 1;

        // Check residual: compare h_new to h, take Frobenius norm of diff
        let h_host = ws.stream.memcpy_dtov(&ws.d_h)?;
        let h_new_host = ws.stream.memcpy_dtov(&ws.d_h_new)?;
        let mut diff_sq = 0.0;
        for k in 0..(n_b * n_b) {
            let d = h_new_host[k] - h_host[k];
            diff_sq += d * d;
        }
        let residual = diff_sq.sqrt();

        // Swap h and h_new (clone d_h_new into d_h)
        let h_new_data = ws.stream.memcpy_dtov(&ws.d_h_new)?;
        ws.stream.memcpy_htod(&h_new_data, &mut ws.d_h)?;

        if residual < tol {
            break;
        }
    }
    Ok(iters)
}

/// Compute Yukawa overlap tensor on GPU.
pub fn gpu_yukawa_tensor(
    ws: &mut GpuDiscriminationWorkspace,
    centers_seed: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let n_p = ws.n_points;
    let n_m = ws.n_modes;

    // Initialise centers via host RNG, copy to device
    let mut rng = crate::LCG::new(centers_seed);
    let mut centers_host = vec![0.0f64; n_m * 8];
    for v in centers_host.iter_mut() {
        *v = rng.next_normal() * 0.5;
    }
    ws.stream.memcpy_htod(&centers_host, &mut ws.d_yukawa_centers)?;

    // Launch yukawa_kernel: one block per (i,j,k) tensor element
    let func = ws.module.load_function("yukawa_kernel")?;
    let n_blocks = (n_m * n_m * n_m) as u32;
    let threads_per_block = 128;
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: (threads_per_block as u32) * 8,
    };
    let n_p_i32 = n_p as i32;
    let n_m_i32 = n_m as i32;
    let dim_i32 = 8i32;
    let mut launcher = ws.stream.launch_builder(&func);
    launcher
        .arg(&ws.d_points)
        .arg(&ws.d_yukawa_centers)
        .arg(&mut ws.d_yukawa_tensor)
        .arg(&n_p_i32)
        .arg(&n_m_i32)
        .arg(&dim_i32);
    unsafe { launcher.launch(cfg)? };

    Ok(())
}

/// Read Yukawa tensor back from device, compute dominant eigenvalue on
/// host (small operation, no need for GPU).
pub fn gpu_dominant_eigenvalue(
    ws: &mut GpuDiscriminationWorkspace,
    n_iter: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let y_host = ws.stream.memcpy_dtov(&ws.d_yukawa_tensor)?;
    let n_m = ws.n_modes;

    let h_val = 1.0 / (n_m as f64).sqrt();
    let mut m = vec![0.0f64; n_m * n_m];
    for i in 0..n_m {
        for j in 0..n_m {
            let mut s = 0.0;
            for k in 0..n_m {
                s += y_host[i * n_m * n_m + j * n_m + k] * h_val;
            }
            m[i * n_m + j] = s;
        }
    }

    let mut v = vec![1.0 / (n_m as f64).sqrt(); n_m];
    let mut lambda = 0.0;
    for _ in 0..n_iter {
        let mut mv = vec![0.0f64; n_m];
        for i in 0..n_m {
            for j in 0..n_m {
                mv[i] += m[i * n_m + j] * v[j];
            }
        }
        let norm: f64 = mv.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            break;
        }
        lambda = 0.0;
        for i in 0..n_m {
            lambda += v[i] * mv[i];
        }
        for i in 0..n_m {
            v[i] = mv[i] / norm;
        }
    }
    Ok(lambda.abs())
}

/// Full GPU discrimination pass.
pub fn gpu_discriminate(
    ws: &mut GpuDiscriminationWorkspace,
    sample_seed: u64,
    centers_seed: u64,
    donaldson_tol: f64,
    eigenvalue_iters: usize,
) -> Result<(usize, f64), Box<dyn std::error::Error>> {
    gpu_sample_points(ws, sample_seed)?;
    gpu_evaluate_section_basis(ws)?;
    let iters = gpu_donaldson_solve(ws, donaldson_tol)?;
    gpu_yukawa_tensor(ws, centers_seed)?;
    let lambda = gpu_dominant_eigenvalue(ws, eigenvalue_iters)?;
    ws.stream.synchronize()?;
    Ok((iters, lambda))
}
