//! CUDA-accelerated harmonic-projection Adam loop for monad-bundle
//! zero modes.
//!
//! Backs [`crate::zero_modes::project_to_harmonic`]'s per-mode Adam
//! loop. Per iteration the CPU loop does:
//!
//!  1. Compute `ψ_θ(p) = ψ⁰(p) + Σ_k (θ_{2k} + i θ_{2k+1}) b_k(p)`
//!     for every sample point `p` (`n_pts × n_basis` complex muls).
//!  2. Compute `‖ψ_θ‖² = Σ_p w_p |ψ_θ(p)|²` (`n_pts` reductions).
//!  3. Compute the gradient `∂L/∂θ_k` requiring `n_pts × n_basis`
//!     more complex muls.
//!  4. Adam update of `θ` (`n_theta = 2 n_basis` scalar updates).
//!
//! Steps 1, 2, 3 are the heavy work; step 4 is trivial. The natural
//! GPU port is one kernel per Adam iteration that takes the current
//! `θ` (kept resident on device across iterations) and writes back
//! `ψ`, `‖ψ‖²`, and `∇L`. Adam state (`m`, `v`) and `θ` itself live
//! permanently on the device for the lifetime of the optimisation;
//! only `‖ψ‖²` is copied back to the host per iteration to test the
//! convergence tolerance.
//!
//! For `n_pts ≤ ~10⁵` and `n_basis ≤ ~100` (the typical 5σ
//! configuration) the kernel runs in microseconds per iteration; the
//! launch overhead dominates and stays below ~50 μs/iter.
//!
//! Same `catch_unwind`-safe init pattern as the other `gpu_*`
//! modules; falls back transparently to the CPU `project_to_harmonic`
//! when CUDA is unavailable.

use cudarc::driver::{CudaContext, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use num_complex::Complex64;
use std::sync::Arc;

const KERNEL_SOURCE: &str = r#"
// One iteration's worth of compute. One block, one mode.
//
//   ψ(p) = seed[p] + Σ_k (θ[2k] + i θ[2k+1]) b_k(p)
//   ‖ψ‖² = Σ_p w_p |ψ(p)|²
//   resid = ‖ψ‖² − 1
//   loss  = resid² + λ ‖θ‖²
//   ∂‖ψ‖²/∂θ_{2k}     =  2 Σ_p w_p Re(conj(ψ_p) b_k(p))
//   ∂‖ψ‖²/∂θ_{2k+1}   = -2 Σ_p w_p Im(conj(ψ_p) b_k(p))
//   ∂L/∂θ            = 2 resid · ∂‖ψ‖²/∂θ + 2 λ θ
//
// Adam:
//   m[k]  = β1 m[k] + (1-β1) g[k]
//   v[k]  = β2 v[k] + (1-β2) g[k]²
//   θ[k] -= lr · (m[k]/(1-β1^t)) / (sqrt(v[k]/(1-β2^t)) + ε)
//
// All intermediate state (m, v, θ, ψ) lives on the device; only
// ‖ψ‖² is published per-iteration to host for the convergence test.
//
// Single-block design: the full per-mode Adam loop runs inside ONE
// CUDA block (256 threads) so we can use __syncthreads() between
// the per-iteration phases without expensive global synchronisation.
// One block per mode → grid_dim.x = n_modes.
//
// Per-iteration work splits as:
//   phase 1  threads stride over points, compute ψ[p] (global mem write)
//            partial reductions to shared norm2, then block-reduce
//   phase 2  threads stride over (k, point) pairs, accumulate grad
//            re/im in shared mem (atomicAdd; n_basis ≤ 256)
//   phase 3  threads stride over θ indices, do Adam update
//
// The kernel returns the final ψ for the mode in `psi_*` global
// memory; ‖ψ‖² history is written to `loss_history` (per-iter).

#define MAX_BASIS 256
#define BLOCK_SIZE 256

extern "C" __global__ void harmonic_adam_loop_per_mode(
    int n_pts,
    int n_basis,
    int max_iter,
    double tol,
    double lr,
    double beta1,
    double beta2,
    double eps_adam,
    double lambda_ridge,
    // Per-mode device buffers (mode index = blockIdx.x)
    const double* __restrict__ seed_re,    // n_modes * n_pts
    const double* __restrict__ seed_im,
    const double* __restrict__ basis_re,   // n_pts * n_basis (shared across modes)
    const double* __restrict__ basis_im,
    const double* __restrict__ weights,    // n_pts
    double* __restrict__ theta,            // n_modes * (2*n_basis)
    double* __restrict__ m_state,          // same shape as theta
    double* __restrict__ v_state,
    double* __restrict__ psi_re,           // n_modes * n_pts
    double* __restrict__ psi_im,
    int* __restrict__ iters_done,          // n_modes (output)
    double* __restrict__ final_loss        // n_modes
) {
    int mode = blockIdx.x;
    int tid = threadIdx.x;
    int n_theta = 2 * n_basis;

    // Dynamic shared memory: laid out as
    //   [ s_norm2 (BLOCK_SIZE) | s_grad_re (n_basis) | s_grad_im (n_basis) ]
    // sized at launch by the host. s_norm2_total and s_done are scalar
    // statics — small, fixed size.
    extern __shared__ double smem[];
    double* s_norm2   = smem;
    double* s_grad_re = smem + BLOCK_SIZE;
    double* s_grad_im = smem + BLOCK_SIZE + n_basis;
    __shared__ double s_norm2_total;
    __shared__ int    s_done;

    double* my_seed_re = (double*)seed_re + mode * n_pts;
    double* my_seed_im = (double*)seed_im + mode * n_pts;
    double* my_psi_re  = psi_re + mode * n_pts;
    double* my_psi_im  = psi_im + mode * n_pts;
    double* my_theta   = theta + mode * n_theta;
    double* my_m       = m_state + mode * n_theta;
    double* my_v       = v_state + mode * n_theta;

    if (tid == 0) {
        s_done = 0;
    }
    __syncthreads();

    int it_for_loss = 0;
    double last_loss = 0.0;

    // Incremental Adam bias-correction products:
    //   bc1[t] = 1 - β1^t,  bc2[t] = 1 - β2^t.
    // Maintain pow1 = β1^t and pow2 = β2^t by multiplication each
    // iteration to avoid a transcendental pow() call per step.
    double pow1 = 1.0;
    double pow2 = 1.0;

    for (int it = 1; it <= max_iter; ++it) {
        if (s_done) break;
        pow1 *= beta1;
        pow2 *= beta2;

        // ----- phase 1: compute ψ and ‖ψ‖² partial reductions -----
        double local_norm2 = 0.0;
        for (int p = tid; p < n_pts; p += BLOCK_SIZE) {
            double pr = my_seed_re[p];
            double pi = my_seed_im[p];
            for (int k = 0; k < n_basis; ++k) {
                double cr = my_theta[2 * k];
                double ci = my_theta[2 * k + 1];
                double br = basis_re[p * n_basis + k];
                double bi = basis_im[p * n_basis + k];
                pr += cr * br - ci * bi;
                pi += cr * bi + ci * br;
            }
            my_psi_re[p] = pr;
            my_psi_im[p] = pi;
            double w = weights[p];
            if (isfinite(w)) {
                local_norm2 += w * (pr * pr + pi * pi);
            }
        }
        s_norm2[tid] = local_norm2;
        __syncthreads();
        for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) s_norm2[tid] += s_norm2[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            s_norm2_total = s_norm2[0];
        }
        __syncthreads();
        double norm2 = s_norm2_total;

        // ----- compute loss and check tolerance -----
        double ridge = 0.0;
        for (int k = tid; k < n_theta; k += BLOCK_SIZE) {
            ridge += my_theta[k] * my_theta[k];
        }
        s_norm2[tid] = ridge;
        __syncthreads();
        for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) s_norm2[tid] += s_norm2[tid + s];
            __syncthreads();
        }
        double total_ridge = s_norm2[0];
        double resid = norm2 - 1.0;
        double loss = resid * resid + lambda_ridge * total_ridge;
        last_loss = loss;
        it_for_loss = it;
        if (loss < tol) {
            if (tid == 0) s_done = 1;
            __syncthreads();
            break;
        }

        // ----- phase 2: gradient (warp/thread-parallel over basis k) -----
        // Earlier implementation used atomicAdd to shared memory inside
        // a (point × basis) double loop with 256 threads contending for
        // each s_grad_re[k] slot. That serialised through the FP64
        // atomic unit. Switch to a contention-free pattern: each
        // thread owns one basis index k = tid (idle if tid >= n_basis)
        // and iterates over all points sequentially in registers,
        // writing the final sum to shared memory once. No atomics.
        if (tid < n_basis) {
            int k = tid;
            double gr = 0.0;
            double gi = 0.0;
            for (int p = 0; p < n_pts; ++p) {
                double w = weights[p];
                if (!isfinite(w)) continue;
                double pr = my_psi_re[p];
                double pi = my_psi_im[p];
                double br = basis_re[p * n_basis + k];
                double bi = basis_im[p * n_basis + k];
                // z = conj(ψ) · b = (pr − i pi)(br + i bi)
                double zr =  pr * br + pi * bi;
                double zi =  pr * bi - pi * br;
                gr += w * 2.0 * zr;
                gi += w * (-2.0) * zi;
            }
            s_grad_re[k] = gr;
            s_grad_im[k] = gi;
        }
        // For n_basis > BLOCK_SIZE (cannot happen today; n_basis ≤ 256
        // = MAX_BASIS = BLOCK_SIZE) the strided form below would handle
        // overflow. Kept disabled but documented.
        // for (int k = tid + BLOCK_SIZE; k < n_basis; k += BLOCK_SIZE) {
        //     ... same body ...
        // }
        __syncthreads();

        // ----- phase 3: Adam update -----
        double bc1 = 1.0 - pow1;
        double bc2 = 1.0 - pow2;
        for (int k = tid; k < n_theta; k += BLOCK_SIZE) {
            int kb = k >> 1;       // basis index
            int re = (k & 1) == 0; // 1 if real component
            double base_grad = re ? s_grad_re[kb] : s_grad_im[kb];
            double g = 2.0 * resid * base_grad + 2.0 * lambda_ridge * my_theta[k];
            double mk = beta1 * my_m[k] + (1.0 - beta1) * g;
            double vk = beta2 * my_v[k] + (1.0 - beta2) * g * g;
            my_m[k] = mk;
            my_v[k] = vk;
            double m_hat = mk / bc1;
            double v_hat = vk / bc2;
            double step = lr * m_hat / (sqrt(v_hat) + eps_adam);
            if (isfinite(step)) {
                my_theta[k] -= step;
            }
        }
        __syncthreads();
    }

    // ----- final renormalise to exact unit L² norm -----
    // (Recompute ψ at converged θ; deterministic projection step.)
    double local_norm2 = 0.0;
    for (int p = tid; p < n_pts; p += BLOCK_SIZE) {
        double pr = my_seed_re[p];
        double pi = my_seed_im[p];
        for (int k = 0; k < n_basis; ++k) {
            double cr = my_theta[2 * k];
            double ci = my_theta[2 * k + 1];
            double br = basis_re[p * n_basis + k];
            double bi = basis_im[p * n_basis + k];
            pr += cr * br - ci * bi;
            pi += cr * bi + ci * br;
        }
        my_psi_re[p] = pr;
        my_psi_im[p] = pi;
        double w = weights[p];
        if (isfinite(w)) {
            local_norm2 += w * (pr * pr + pi * pi);
        }
    }
    s_norm2[tid] = local_norm2;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) s_norm2[tid] += s_norm2[tid + s];
        __syncthreads();
    }
    double norm2_final = s_norm2[0];
    double scale = (norm2_final > 0.0 && isfinite(norm2_final))
                     ? (1.0 / sqrt(norm2_final))
                     : 1.0;
    for (int p = tid; p < n_pts; p += BLOCK_SIZE) {
        my_psi_re[p] *= scale;
        my_psi_im[p] *= scale;
    }

    if (tid == 0) {
        iters_done[mode] = it_for_loss;
        final_loss[mode] = last_loss;
    }
}
"#;

pub struct GpuHarmonicContext {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
}

impl GpuHarmonicContext {
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
                    "GPU harmonic init panic — likely no CUDA driver / nvrtc.dll: {msg}"
                )
                .into())
            }
        }
    }
}

/// Per-mode output of the harmonic-projection Adam loop.
#[derive(Debug, Clone)]
pub struct HarmonicAdamOutputs {
    /// `n_modes × n_pts` complex psi values (mode-major).
    pub psi: Vec<Complex64>,
    /// Final loss value per mode (length `n_modes`).
    pub final_loss: Vec<f64>,
    /// Iterations actually executed per mode (length `n_modes`).
    pub iterations: Vec<usize>,
}

/// Run `max_iter` Adam steps on the device for `n_modes` independent
/// modes, returning the final `ψ` values and convergence info.
///
/// `seed`, `basis`, `weights`, `theta_init`, `m_init`, `v_init` must
/// all be in the layout the kernel expects (see comments above each
/// argument). For most callers the simpler dispatch through
/// [`crate::zero_modes::project_to_harmonic`] is preferred.
#[allow(clippy::too_many_arguments)]
pub fn gpu_harmonic_adam_loop(
    gpu_ctx: &GpuHarmonicContext,
    n_pts: usize,
    n_modes: usize,
    n_basis: usize,
    max_iter: usize,
    tol: f64,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps_adam: f64,
    lambda_ridge: f64,
    seed_re: &[f64],
    seed_im: &[f64],
    basis_re: &[f64],
    basis_im: &[f64],
    weights: &[f64],
) -> Result<HarmonicAdamOutputs, Box<dyn std::error::Error>> {
    if n_basis > 256 {
        return Err(format!(
            "n_basis = {n_basis} exceeds kernel MAX_BASIS = 256; recompile or reduce correction degree"
        )
        .into());
    }
    if n_pts == 0 || n_modes == 0 || n_basis == 0 {
        return Ok(HarmonicAdamOutputs {
            psi: Vec::new(),
            final_loss: Vec::new(),
            iterations: Vec::new(),
        });
    }
    let n_theta = 2 * n_basis;
    if seed_re.len() != n_modes * n_pts
        || seed_im.len() != n_modes * n_pts
        || basis_re.len() != n_pts * n_basis
        || basis_im.len() != n_pts * n_basis
        || weights.len() != n_pts
    {
        return Err("input slice length mismatch".into());
    }

    let stream = &gpu_ctx.stream;

    let d_seed_re = stream.memcpy_stod(seed_re)?;
    let d_seed_im = stream.memcpy_stod(seed_im)?;
    let d_basis_re = stream.memcpy_stod(basis_re)?;
    let d_basis_im = stream.memcpy_stod(basis_im)?;
    let d_weights = stream.memcpy_stod(weights)?;

    let theta_init = vec![0.0_f64; n_modes * n_theta];
    let m_init = vec![0.0_f64; n_modes * n_theta];
    let v_init = vec![0.0_f64; n_modes * n_theta];
    let mut d_theta: CudaSlice<f64> = stream.memcpy_stod(&theta_init)?;
    let mut d_m: CudaSlice<f64> = stream.memcpy_stod(&m_init)?;
    let mut d_v: CudaSlice<f64> = stream.memcpy_stod(&v_init)?;
    let mut d_psi_re: CudaSlice<f64> = stream.alloc_zeros::<f64>(n_modes * n_pts)?;
    let mut d_psi_im: CudaSlice<f64> = stream.alloc_zeros::<f64>(n_modes * n_pts)?;
    let mut d_iters: CudaSlice<i32> = stream.alloc_zeros::<i32>(n_modes)?;
    let mut d_loss: CudaSlice<f64> = stream.alloc_zeros::<f64>(n_modes)?;

    let func = gpu_ctx.module.load_function("harmonic_adam_loop_per_mode")?;
    // Dynamic shared memory: BLOCK_SIZE doubles for s_norm2 + 2 ×
    // n_basis doubles for s_grad_re/im. Sized exactly per
    // n_basis instead of the static MAX_BASIS upper bound.
    let block_size: u32 = 256;
    let dyn_smem_bytes: u32 =
        (block_size as u32 + 2 * n_basis as u32) * std::mem::size_of::<f64>() as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_modes as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: dyn_smem_bytes,
    };
    let n_pts_i32 = n_pts as i32;
    let n_basis_i32 = n_basis as i32;
    let max_iter_i32 = max_iter as i32;

    let mut launcher = stream.launch_builder(&func);
    launcher
        .arg(&n_pts_i32)
        .arg(&n_basis_i32)
        .arg(&max_iter_i32)
        .arg(&tol)
        .arg(&lr)
        .arg(&beta1)
        .arg(&beta2)
        .arg(&eps_adam)
        .arg(&lambda_ridge)
        .arg(&d_seed_re)
        .arg(&d_seed_im)
        .arg(&d_basis_re)
        .arg(&d_basis_im)
        .arg(&d_weights)
        .arg(&mut d_theta)
        .arg(&mut d_m)
        .arg(&mut d_v)
        .arg(&mut d_psi_re)
        .arg(&mut d_psi_im)
        .arg(&mut d_iters)
        .arg(&mut d_loss);
    unsafe { launcher.launch(cfg)? };

    let host_re = stream.memcpy_dtov(&d_psi_re)?;
    let host_im = stream.memcpy_dtov(&d_psi_im)?;
    let host_iters = stream.memcpy_dtov(&d_iters)?;
    let host_loss = stream.memcpy_dtov(&d_loss)?;
    let _ = &gpu_ctx.ctx;

    let psi: Vec<Complex64> = (0..n_modes * n_pts)
        .map(|i| Complex64::new(host_re[i], host_im[i]))
        .collect();
    let iterations: Vec<usize> = host_iters.into_iter().map(|i| i.max(0) as usize).collect();

    Ok(HarmonicAdamOutputs {
        psi,
        final_loss: host_loss,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity test: the GPU loop runs end-to-end and returns
    /// finite, unit-norm psi values.
    #[test]
    fn gpu_harmonic_adam_loop_runs_and_normalises() {
        let ctx = match GpuHarmonicContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("GPU unavailable, skipping: {e}");
                return;
            }
        };

        let n_pts = 256;
        let n_modes = 3;
        let n_basis = 16;
        let max_iter = 50;

        // Synthetic random inputs.
        let mut state: u64 = 0xCAFEBABE;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) & ((1 << 53) - 1)) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let seed_re: Vec<f64> = (0..n_modes * n_pts).map(|_| next()).collect();
        let seed_im: Vec<f64> = (0..n_modes * n_pts).map(|_| next()).collect();
        let basis_re: Vec<f64> = (0..n_pts * n_basis).map(|_| next()).collect();
        let basis_im: Vec<f64> = (0..n_pts * n_basis).map(|_| next()).collect();
        let weights: Vec<f64> = (0..n_pts).map(|_| 1.0 / n_pts as f64).collect();

        let out = gpu_harmonic_adam_loop(
            &ctx,
            n_pts,
            n_modes,
            n_basis,
            max_iter,
            1e-12,
            1e-2,
            0.9,
            0.999,
            1e-8,
            1e-6,
            &seed_re,
            &seed_im,
            &basis_re,
            &basis_im,
            &weights,
        )
        .unwrap();

        assert_eq!(out.psi.len(), n_modes * n_pts);
        assert_eq!(out.iterations.len(), n_modes);
        assert_eq!(out.final_loss.len(), n_modes);

        for a in 0..n_modes {
            let norm2: f64 = (0..n_pts)
                .map(|p| weights[p] * out.psi[a * n_pts + p].norm_sqr())
                .sum();
            assert!(
                (norm2 - 1.0).abs() < 1e-9,
                "mode {a} not unit-normalised: norm2 = {norm2}"
            );
            for p in 0..n_pts {
                let v = out.psi[a * n_pts + p];
                assert!(v.re.is_finite() && v.im.is_finite(), "non-finite psi at ({a}, {p})");
            }
        }
    }
}
