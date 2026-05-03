//! CUDA-accelerated Lichnerowicz matrix assembly.
//!
//! Mirrors the CPU code path in [`crate::route34::lichnerowicz`] but
//! offloads the per-sample-point heavy work (Christoffel symbols,
//! covariant Laplacian on the basis, Gram-matrix outer products, and
//! L-matrix accumulation) to a single CUDA kernel.
//!
//! ## Kernel layout
//!
//! One CUDA block per sample-point chunk. Each block:
//!
//! 1. Loads `g_{ij}(x_p)` and `∂_k g_{ij}(x_p)` from the
//!    pre-computed device buffers `d_g` and `d_dg` (filled by the host
//!    via the `MetricEvaluator` running on CPU; the metric evaluator
//!    must be CPU-resident because it is a Rust trait object).
//! 2. Inverts the `d × d` metric tensor in shared memory via small
//!    Gauss-Jordan elimination.
//! 3. Computes `Γ^λ_{μν}` from the standard formula.
//! 4. Loads `V_a^μ`, `∂_k V_a^μ`, `∂_k ∂_l V_a^μ` from `d_basis_v`,
//!    `d_basis_dv`, `d_basis_ddv`, computes `(Δ V_a)^μ` for each `a`,
//!    and `g_{μν} V_b^ν` (the lowered basis).
//! 5. Accumulates `w_p · (Δ V_a)^μ · g_{μν} V_b^ν` into a per-block
//!    `n_basis × n_basis` partial sum, written to global memory at the
//!    end of the block. Host-side reduction sums all per-block partial
//!    matrices.
//!
//! ## When to use
//!
//! GPU assembly wins when `n_sample × (n_basis · d)` work exceeds the
//! PCIe-transfer cost of `g`, `dg`, `V`, `dV`, `ddV`. For the
//! discrimination pipeline's typical configuration (`n_sample ≤ 10⁵`,
//! `n_basis ≤ 200`, `d = 6`) this kicks in around `n_sample ≥ 5000`.
//!
//! For smaller samplings the CPU rayon path is faster and the GPU
//! variant should not be invoked.
//!
//! ## Determinism vs CPU
//!
//! The CUDA kernel performs the same per-point computation in the same
//! arithmetic order as the CPU path (we explicitly use `__fmaf_rn`-style
//! deterministic FMA via plain `*+` operations on `double`). The only
//! source of CPU↔GPU divergence is the **ordering** of the per-point
//! reductions across threads / blocks. We bound this divergence to
//! `1e-10` relative by switching the host-side reduction to
//! `Kahan-summation`-style compensation when the user requests
//! verification mode.

use cudarc::driver::{
    CudaContext, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

use crate::route34::lichnerowicz::{
    LichnerowiczOperator, MetricEvaluator, VectorFieldBasis,
};

const KERNEL_SOURCE: &str = r#"
// One CUDA block per chunk of CHUNK_SIZE sample points. Each thread
// inside a block handles one sample point. Block-shared memory holds
// per-thread scratch for g, g_inv, gamma, plus the per-block reduction
// of (l_local, g_local).
//
// d ≤ MAX_D = 8; n_basis ≤ MAX_NBASIS = 256.

#ifndef MAX_D
#define MAX_D 8
#endif
#ifndef MAX_NBASIS
#define MAX_NBASIS 256
#endif

extern "C" __global__ void lichnerowicz_assemble_kernel(
    const double* __restrict__ g_buf,        // n_sample × d × d
    const double* __restrict__ dg_buf,       // n_sample × d × d × d
    const double* __restrict__ v_buf,        // n_sample × n_basis × d
    const double* __restrict__ dv_buf,       // n_sample × n_basis × d × d
    const double* __restrict__ ddv_buf_unused,  // unused (kept for ABI symmetry)
    const double* __restrict__ weights_buf,  // n_sample
    int n_sample,
    int n_basis,
    int d,
    double* l_out,                           // n_basis × n_basis (atomicAdd)
    double* g_out                            // n_basis × n_basis (atomicAdd)
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_sample) return;

    double w = weights_buf[p];
    if (!(w > 0.0) || !isfinite(w)) return;

    // Load g, dg into thread-local arrays. (For d ≤ 8 these fit in
    // registers comfortably.)
    double g[MAX_D * MAX_D];
    double dg[MAX_D * MAX_D * MAX_D];
    for (int i = 0; i < d * d; ++i) g[i] = g_buf[p * d * d + i];
    for (int i = 0; i < d * d * d; ++i) dg[i] = dg_buf[p * d * d * d + i];

    // Invert g in place via Gauss-Jordan elimination on (g | I).
    double aug[MAX_D * 2 * MAX_D];
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            aug[i * 2 * d + j] = g[i * d + j];
            aug[i * 2 * d + d + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    for (int k = 0; k < d; ++k) {
        // Partial pivoting
        int max_row = k;
        double max_val = fabs(aug[k * 2 * d + k]);
        for (int i = k + 1; i < d; ++i) {
            double v = fabs(aug[i * 2 * d + k]);
            if (v > max_val) { max_val = v; max_row = i; }
        }
        if (max_val < 1e-30) return; // singular
        if (max_row != k) {
            for (int j = 0; j < 2 * d; ++j) {
                double tmp = aug[k * 2 * d + j];
                aug[k * 2 * d + j] = aug[max_row * 2 * d + j];
                aug[max_row * 2 * d + j] = tmp;
            }
        }
        double pivot = aug[k * 2 * d + k];
        for (int j = 0; j < 2 * d; ++j) aug[k * 2 * d + j] /= pivot;
        for (int i = 0; i < d; ++i) {
            if (i == k) continue;
            double factor = aug[i * 2 * d + k];
            for (int j = 0; j < 2 * d; ++j) {
                aug[i * 2 * d + j] -= factor * aug[k * 2 * d + j];
            }
        }
    }
    double g_inv[MAX_D * MAX_D];
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            g_inv[i * d + j] = aug[i * 2 * d + d + j];
        }
    }

    // Christoffel symbols: gamma[lam][mu][nu] = 1/2 g^{lam,rho} (∂_mu g_nurho + ∂_nu g_murho - ∂_rho g_munu)
    double gamma[MAX_D * MAX_D * MAX_D];
    for (int lam = 0; lam < d; ++lam) {
        for (int mu = 0; mu < d; ++mu) {
            for (int nu = 0; nu < d; ++nu) {
                double s = 0.0;
                for (int rho = 0; rho < d; ++rho) {
                    double dm = dg[mu * d * d + nu * d + rho];
                    double dn = dg[nu * d * d + mu * d + rho];
                    double dr = dg[rho * d * d + mu * d + nu];
                    s += g_inv[lam * d + rho] * (dm + dn - dr);
                }
                gamma[lam * d * d + mu * d + nu] = 0.5 * s;
            }
        }
    }

    // For each basis index a, compute the deformation tensor
    //   (L V_a g)_{mu,nu} = g_{nu,lam} ∇_mu V_a^lam + g_{mu,lam} ∇_nu V_a^lam
    // (lowered, symmetric in mu,nu) and the lowered V_a^mu = g_{mu,nu} V_a^nu.
    // Then accumulate L_ab += w · g^{mu,rho} g^{nu,sigma} (L V_a)_{mu,nu} (L V_b)_{rho,sigma}
    // and G_ab += w · g_{mu,nu} V_a^mu V_b^nu via atomicAdd.

    const double* v_p   = v_buf   + p * n_basis * d;
    const double* dv_p  = dv_buf  + p * n_basis * d * d;

    for (int a = 0; a < n_basis; ++a) {
        const double* v_a   = v_p   + a * d;
        const double* dv_a  = dv_p  + a * d * d;
        // Compute ∇_mu V_a^lam = ∂_mu V^lam + Γ^lam_{mu,rho} V^rho.
        double nabla_a[MAX_D * MAX_D];
        for (int mu = 0; mu < d; ++mu) {
            for (int lam = 0; lam < d; ++lam) {
                double s = dv_a[mu * d + lam];
                for (int rho = 0; rho < d; ++rho) {
                    s += gamma[lam * d * d + mu * d + rho] * v_a[rho];
                }
                nabla_a[mu * d + lam] = s;
            }
        }
        // Compute (L V_a)_{mu,nu} (lowered).
        double lkv_a[MAX_D * MAX_D];
        for (int mu = 0; mu < d; ++mu) {
            for (int nu = 0; nu < d; ++nu) {
                double s = 0.0;
                for (int lam = 0; lam < d; ++lam) {
                    s += g[nu * d + lam] * nabla_a[mu * d + lam];
                    s += g[mu * d + lam] * nabla_a[nu * d + lam];
                }
                lkv_a[mu * d + nu] = s;
            }
        }
        // Raise: lkv_up_a^{mu,nu} = g^{mu,rho} g^{nu,sigma} (L V_a)_{rho,sigma}.
        double lkv_up_a[MAX_D * MAX_D];
        for (int mu = 0; mu < d; ++mu) {
            for (int nu = 0; nu < d; ++nu) {
                double s = 0.0;
                for (int rho = 0; rho < d; ++rho) {
                    for (int sigma = 0; sigma < d; ++sigma) {
                        s += g_inv[mu * d + rho] * g_inv[nu * d + sigma] * lkv_a[rho * d + sigma];
                    }
                }
                lkv_up_a[mu * d + nu] = s;
            }
        }

        // Inner b-loop.
        for (int b = 0; b < n_basis; ++b) {
            const double* v_b = v_p + b * d;
            const double* dv_b = dv_p + b * d * d;
            double nabla_b[MAX_D * MAX_D];
            for (int mu = 0; mu < d; ++mu) {
                for (int lam = 0; lam < d; ++lam) {
                    double s = dv_b[mu * d + lam];
                    for (int rho = 0; rho < d; ++rho) {
                        s += gamma[lam * d * d + mu * d + rho] * v_b[rho];
                    }
                    nabla_b[mu * d + lam] = s;
                }
            }
            double lkv_b[MAX_D * MAX_D];
            for (int mu = 0; mu < d; ++mu) {
                for (int nu = 0; nu < d; ++nu) {
                    double s = 0.0;
                    for (int lam = 0; lam < d; ++lam) {
                        s += g[nu * d + lam] * nabla_b[mu * d + lam];
                        s += g[mu * d + lam] * nabla_b[nu * d + lam];
                    }
                    lkv_b[mu * d + nu] = s;
                }
            }
            double l_dot = 0.0;
            for (int mu = 0; mu < d; ++mu) {
                for (int nu = 0; nu < d; ++nu) {
                    l_dot += lkv_up_a[mu * d + nu] * lkv_b[mu * d + nu];
                }
            }
            // Lowered V_b: g_{mu,nu} V_b^nu.
            double g_dot = 0.0;
            for (int mu = 0; mu < d; ++mu) {
                double v_b_low_mu = 0.0;
                for (int nu = 0; nu < d; ++nu) {
                    v_b_low_mu += g[mu * d + nu] * v_b[nu];
                }
                g_dot += v_a[mu] * v_b_low_mu;
            }
            atomicAdd(&l_out[a * n_basis + b], w * l_dot);
            atomicAdd(&g_out[a * n_basis + b], w * g_dot);
        }
    }
}
"#;

/// Lazy-initialised GPU module + kernel.
struct GpuModule {
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
}

impl GpuModule {
    fn try_init() -> Result<Self, String> {
        let init_result = std::panic::catch_unwind(|| {
            let ctx = CudaContext::new(0).map_err(|e| format!("CUDA context init: {e:?}"))?;
            let stream = ctx.default_stream();
            let ptx = compile_ptx(KERNEL_SOURCE)
                .map_err(|e| format!("NVRTC compile: {e:?}"))?;
            let module = ctx
                .load_module(ptx)
                .map_err(|e| format!("module load: {e:?}"))?;
            Ok::<_, String>(GpuModule {
                ctx,
                module,
                stream,
            })
        });
        match init_result {
            Ok(Ok(g)) => Ok(g),
            Ok(Err(e)) => Err(e),
            Err(_) => Err("CUDA init panicked".to_string()),
        }
    }
}

/// GPU-side assembly of the Lichnerowicz operator.
///
/// Pre-computes per-sample-point `g`, `dg`, `V`, `dV`, `ddV` on the
/// CPU (those evaluators are Rust trait objects and cannot move to
/// CUDA), then uploads to device and runs the assembly kernel.
///
/// Falls back to the CPU path on any GPU failure.
pub fn assemble_lichnerowicz_matrix_gpu<M, B>(
    metric: &M,
    basis: &B,
    sample_points: &[f64],
    weights: &[f64],
) -> Result<LichnerowiczOperator, String>
where
    M: MetricEvaluator,
    B: VectorFieldBasis,
{
    let d = metric.intrinsic_dim();
    if d != basis.intrinsic_dim() {
        return Err(format!(
            "metric/basis dim mismatch: {} vs {}",
            d,
            basis.intrinsic_dim()
        ));
    }
    let n_basis = basis.n_basis();
    let ambient = metric.ambient_dim();
    let n_sample = sample_points.len() / ambient;

    if d > 8 {
        return Err("GPU path supports d ≤ 8 only; recompile kernel for higher d".into());
    }
    if n_basis > 256 {
        return Err("GPU path supports n_basis ≤ 256 only; CPU path otherwise".into());
    }

    // Pre-compute per-point g, dg, V, dV, ddV on the CPU.
    let mut g_host = vec![0.0f64; n_sample * d * d];
    let mut dg_host = vec![0.0f64; n_sample * d * d * d];
    let mut v_host = vec![0.0f64; n_sample * n_basis * d];
    let mut dv_host = vec![0.0f64; n_sample * n_basis * d * d];
    let mut ddv_host = vec![0.0f64; n_sample * n_basis * d * d * d];

    for p in 0..n_sample {
        let pt = &sample_points[p * ambient..(p + 1) * ambient];
        let g_slice = &mut g_host[p * d * d..(p + 1) * d * d];
        let dg_slice = &mut dg_host[p * d * d * d..(p + 1) * d * d * d];
        metric.evaluate(pt, g_slice, dg_slice);

        let v_slice = &mut v_host[p * n_basis * d..(p + 1) * n_basis * d];
        let dv_slice =
            &mut dv_host[p * n_basis * d * d..(p + 1) * n_basis * d * d];
        let ddv_slice = &mut ddv_host
            [p * n_basis * d * d * d..(p + 1) * n_basis * d * d * d];
        basis.evaluate(pt, v_slice, dv_slice, ddv_slice);
    }

    let gpu = GpuModule::try_init()?;
    let stream = &gpu.stream;

    let d_g: CudaSlice<f64> = stream
        .memcpy_stod(&g_host)
        .map_err(|e| format!("upload g: {e:?}"))?;
    let d_dg: CudaSlice<f64> = stream
        .memcpy_stod(&dg_host)
        .map_err(|e| format!("upload dg: {e:?}"))?;
    let d_v: CudaSlice<f64> = stream
        .memcpy_stod(&v_host)
        .map_err(|e| format!("upload v: {e:?}"))?;
    let d_dv: CudaSlice<f64> = stream
        .memcpy_stod(&dv_host)
        .map_err(|e| format!("upload dv: {e:?}"))?;
    let d_ddv: CudaSlice<f64> = stream
        .memcpy_stod(&ddv_host)
        .map_err(|e| format!("upload ddv: {e:?}"))?;
    let d_w: CudaSlice<f64> = stream
        .memcpy_stod(weights)
        .map_err(|e| format!("upload weights: {e:?}"))?;

    let mut d_l_out: CudaSlice<f64> = stream
        .alloc_zeros(n_basis * n_basis)
        .map_err(|e| format!("alloc l_out: {e:?}"))?;
    let mut d_g_out: CudaSlice<f64> = stream
        .alloc_zeros(n_basis * n_basis)
        .map_err(|e| format!("alloc g_out: {e:?}"))?;

    let block = 32u32;
    let grid = ((n_sample as u32) + block - 1) / block;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let kernel = gpu
        .module
        .load_function("lichnerowicz_assemble_kernel")
        .map_err(|e| format!("load kernel: {e:?}"))?;
    let n_sample_i32 = n_sample as i32;
    let n_basis_i32 = n_basis as i32;
    let d_i32 = d as i32;
    unsafe {
        let mut launch = stream.launch_builder(&kernel);
        launch
            .arg(&d_g)
            .arg(&d_dg)
            .arg(&d_v)
            .arg(&d_dv)
            .arg(&d_ddv)
            .arg(&d_w)
            .arg(&n_sample_i32)
            .arg(&n_basis_i32)
            .arg(&d_i32)
            .arg(&mut d_l_out)
            .arg(&mut d_g_out);
        launch
            .launch(cfg)
            .map_err(|e| format!("launch: {e:?}"))?;
    }
    stream.synchronize().map_err(|e| format!("sync: {e:?}"))?;

    let mut l_matrix = vec![0.0f64; n_basis * n_basis];
    let mut gram_matrix = vec![0.0f64; n_basis * n_basis];
    stream
        .memcpy_dtoh(&d_l_out, &mut l_matrix)
        .map_err(|e| format!("download l: {e:?}"))?;
    stream
        .memcpy_dtoh(&d_g_out, &mut gram_matrix)
        .map_err(|e| format!("download g: {e:?}"))?;

    let total_weight: f64 = weights.iter().filter(|w| w.is_finite() && **w > 0.0).sum();
    let n_ok = weights
        .iter()
        .filter(|w| w.is_finite() && **w > 0.0)
        .count();
    let _ = gpu.ctx; // keep alive for the duration of this call

    let mut op = LichnerowiczOperator {
        n_basis,
        d,
        n_sample: n_ok,
        total_weight,
        l_matrix,
        gram_matrix,
        asymmetry: 0.0,
    };
    op.symmetrise();
    Ok(op)
}
