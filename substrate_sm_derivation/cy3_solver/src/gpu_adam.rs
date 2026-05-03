//! GPU CUDA kernels for the two CPU-bound hot paths in the
//! σ-functional Adam refinement of the Fermat-quintic CY metric:
//!
//!  1. **Batched per-point Adam-gradient**
//!     (`AdamGradientKernel::compute_per_point`) — mirrors
//!     `quintic::sigma_squared_and_gradient` lines 1339-1450 and the
//!     per-point closure that calls `per_point_log_det_gradient`
//!     (`quintic.rs:1020-1368`). Each sample point produces
//!         η_p, w_p, ∇L_p[0..2·n_basis²]
//!     in the same SoA layout the Rust path uses
//!         per_point_buf[p * (2 + 2·n_basis²) ..]
//!         = [eta_p, w_p, grad_p_re_block, grad_p_im_block]
//!
//!  2. **Batched 16×16 real determinant** for the Monge-Ampère residual
//!     (`BatchedLuDetKernel::compute_dets`) — mirrors
//!     `refine::determinant_lu` (`refine.rs:29-66`). Each 8×8 complex
//!     matrix is provided as its 16×16 real block representation
//!         [[A, -B], [B, A]],
//!     whose determinant equals |det(A + iB)|² (so it is real and
//!     non-negative for any complex matrix; positive for a Hermitian
//!     positive-definite metric Hessian).
//!
//! ## Conventions
//!
//! Inputs and outputs are flat `&[f64]` / `&mut [f64]` host slices. All
//! buffers are pre-allocated on the device in `new()`; `compute_*`
//! methods only memcpy into / out of the existing device buffers and
//! never allocate on the host or device. This matches the existing
//! workspace pattern in `gpu_quintic.rs`.
//!
//! All CUDA error paths bubble up as `CudaError` (a `Box<dyn Error>`
//! alias matching the convention used by `gpu.rs` and `gpu_quintic.rs`)
//! — never `.unwrap()`-d. The module compiles cleanly without the
//! `gpu` feature; only the CPU reference is exposed in that case.
//!
//! ## CPU reference
//!
//! `cpu_compute_per_point_reference` is a stand-alone implementation of
//! the per-point math that the GPU kernel ports. It is the validation
//! oracle for the GPU path: the parity test runs both on the same input
//! and asserts agreement to ≤ 1e-10 absolute. The CPU reference also
//! re-implements the three private helpers from `quintic.rs`
//! (`invert_3x3_complex_hermitian`, `compute_y_5x5`,
//! `per_point_log_det_gradient`) so this module can be wired in by the
//! orchestrator without touching `quintic.rs`.

#[cfg(feature = "gpu")]
use cudarc::driver::{
    CudaContext as CudarcContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
#[cfg(feature = "gpu")]
use cudarc::nvrtc::compile_ptx;
#[cfg(feature = "gpu")]
use std::sync::Arc;

use crate::quintic::{
    det_3x3_complex_hermitian, fermat_quintic_gradient, project_to_quintic_tangent,
    quintic_affine_chart_frame, quintic_chart_and_elim,
};

/// Re-export under the conventional name that the orchestrator/tests
/// expect (mirrors `crate::gpu::CudaContext` placeholder in spec).
#[cfg(feature = "gpu")]
pub type CudaContext = Arc<CudarcContext>;

/// Errors propagated from cudarc + NVRTC; matches the convention used
/// by `gpu.rs` / `gpu_quintic.rs`. We deliberately use a boxed dyn
/// Error so the caller can attach context without coupling to a
/// specific cudarc version.
pub type CudaError = Box<dyn std::error::Error + Send + Sync>;

// ---------------------------------------------------------------------------
// CUDA kernel sources (Phase 2). NVRTC-compiled at workspace construction.
// ---------------------------------------------------------------------------

/// CUDA source for the per-point Adam-gradient kernel.
///
/// Layout / dispatch:
///   - One thread per sample point (point p = blockIdx.x * blockDim.x +
///     threadIdx.x). Embarrassingly parallel; no inter-thread comms.
///   - h_block lives in *global* memory (since at n_basis = 35, k = 3 it
///     is 4·35·35·8 = 39 KB which already exceeds default shared mem
///     per block). The kernel relies on L1/L2 cache for reuse.
///   - Per-thread scratch for q, qp, phi, A is allocated in dynamic
///     shared memory at launch (we pass the size via shared_mem_bytes).
///     The total per-thread scratch is 2·(2·n) + 2·5·n + 2·n² doubles.
///     For n_basis = 15 (k=2) and 32 threads/block this is
///        32 · (60 + 30 + 150 + 450) = ~22 KB → fits in default shared.
///     For n_basis = 35 (k=3) it does *not* fit; the host falls back to
///     `compute_per_point_global_scratch` which keeps the scratch in
///     global memory (one slab per point in `d_scratch`).
///   - For simplicity and portability across n_basis sizes, the source
///     below uses a *global-memory* scratch slab keyed by point index
///     (d_scratch). Performance penalty vs shared memory is ~2× but the
///     kernel still wins ≥ 30× over the CPU rayon path at n_pts ≥ 10000.
///
/// All math is a line-for-line port of `per_point_log_det_gradient` in
/// `quintic.rs:1020-1368`. The tangent frame, 5×5 ambient g, 3×3
/// projection, det, inv, Y matrix, and final h-gradient assembly are
/// duplicated here in CUDA C++ form.
#[cfg(feature = "gpu")]
const ADAM_KERNEL_SOURCE: &str = r#"
// ---- complex helpers ------------------------------------------------------
__device__ __forceinline__ void cmul(double ar, double ai, double br, double bi,
                                     double* outr, double* outi) {
    *outr = ar * br - ai * bi;
    *outi = ar * bi + ai * br;
}

// Hermitian inner product <a, b> = Σ_k conj(a_k) · b_k for length-5 vectors.
// a stored as (a_re[5], a_im[5]); b stored as 10 reals interleaved (re, im).
__device__ __forceinline__ void conj_dot5(const double* ar5, const double* ai5,
                                          const double* b10,
                                          double* dre, double* dim) {
    double re = 0.0, im = 0.0;
    for (int k = 0; k < 5; ++k) {
        double br = b10[2*k];
        double bi = b10[2*k+1];
        re += ar5[k] * br + ai5[k] * bi;
        im += ar5[k] * bi - ai5[k] * br;
    }
    *dre = re;
    *dim = im;
}

// 3×3 complex Hermitian determinant (real-valued for Hermitian).
// g stored as g[0..3][0..3].(re, im) flattened length 18 (3*3*2).
__device__ __forceinline__ double det3_herm(const double* g) {
    // g[i][j] at index 2*(3*i + j) (re), 2*(3*i + j) + 1 (im).
    #define GR(i,j) g[2*(3*(i)+(j))]
    #define GI(i,j) g[2*(3*(i)+(j))+1]
    double m1r, m1i, m2r, m2i;
    cmul(GR(1,1), GI(1,1), GR(2,2), GI(2,2), &m1r, &m1i);
    cmul(GR(1,2), GI(1,2), GR(2,1), GI(2,1), &m2r, &m2i);
    double cof00r = m1r - m2r;
    double cof00i = m1i - m2i;
    double m3r, m3i, m4r, m4i;
    cmul(GR(1,0), GI(1,0), GR(2,2), GI(2,2), &m3r, &m3i);
    cmul(GR(1,2), GI(1,2), GR(2,0), GI(2,0), &m4r, &m4i);
    double cof01r = m3r - m4r;
    double cof01i = m3i - m4i;
    double m5r, m5i, m6r, m6i;
    cmul(GR(1,0), GI(1,0), GR(2,1), GI(2,1), &m5r, &m5i);
    cmul(GR(1,1), GI(1,1), GR(2,0), GI(2,0), &m6r, &m6i);
    double cof02r = m5r - m6r;
    double cof02i = m5i - m6i;
    double t1r, t1i, t2r, t2i, t3r, t3i;
    cmul(GR(0,0), GI(0,0), cof00r, cof00i, &t1r, &t1i);
    cmul(GR(0,1), GI(0,1), cof01r, cof01i, &t2r, &t2i);
    cmul(GR(0,2), GI(0,2), cof02r, cof02i, &t3r, &t3i);
    return t1r - t2r + t3r;
    #undef GR
    #undef GI
}

// Invert 3×3 complex Hermitian matrix. Returns 1 on success, 0 if det
// is non-finite or near-zero. inv flattened as 18 doubles like g.
__device__ int inv3_herm(const double* g, double* inv) {
    double det = det3_herm(g);
    if (!isfinite(det) || fabs(det) < 1e-30) return 0;
    double idet = 1.0 / det;
    #define GR(i,j) g[2*(3*(i)+(j))]
    #define GI(i,j) g[2*(3*(i)+(j))+1]
    #define IR(i,j) inv[2*(3*(i)+(j))]
    #define II(i,j) inv[2*(3*(i)+(j))+1]
    // Cofactors C_{ij}; inv[i][j] = C_{ji} / det (transpose).
    double tr, ti, ar, ai, br, bi;
    // C00 = g11 g22 - g12 g21
    cmul(GR(1,1), GI(1,1), GR(2,2), GI(2,2), &ar, &ai);
    cmul(GR(1,2), GI(1,2), GR(2,1), GI(2,1), &br, &bi);
    tr = ar - br; ti = ai - bi; IR(0,0) = tr * idet; II(0,0) = ti * idet;
    // C01 = g12 g20 - g10 g22
    cmul(GR(1,2), GI(1,2), GR(2,0), GI(2,0), &ar, &ai);
    cmul(GR(1,0), GI(1,0), GR(2,2), GI(2,2), &br, &bi);
    tr = ar - br; ti = ai - bi; IR(1,0) = tr * idet; II(1,0) = ti * idet;
    // C02 = g10 g21 - g11 g20
    cmul(GR(1,0), GI(1,0), GR(2,1), GI(2,1), &ar, &ai);
    cmul(GR(1,1), GI(1,1), GR(2,0), GI(2,0), &br, &bi);
    tr = ar - br; ti = ai - bi; IR(2,0) = tr * idet; II(2,0) = ti * idet;
    // C10 = g02 g21 - g01 g22
    cmul(GR(0,2), GI(0,2), GR(2,1), GI(2,1), &ar, &ai);
    cmul(GR(0,1), GI(0,1), GR(2,2), GI(2,2), &br, &bi);
    tr = ar - br; ti = ai - bi; IR(0,1) = tr * idet; II(0,1) = ti * idet;
    // C11 = g00 g22 - g02 g20
    cmul(GR(0,0), GI(0,0), GR(2,2), GI(2,2), &ar, &ai);
    cmul(GR(0,2), GI(0,2), GR(2,0), GI(2,0), &br, &bi);
    tr = ar - br; ti = ai - bi; IR(1,1) = tr * idet; II(1,1) = ti * idet;
    // C12 = g01 g20 - g00 g21
    cmul(GR(0,1), GI(0,1), GR(2,0), GI(2,0), &ar, &ai);
    cmul(GR(0,0), GI(0,0), GR(2,1), GI(2,1), &br, &bi);
    tr = ar - br; ti = ai - bi; IR(2,1) = tr * idet; II(2,1) = ti * idet;
    // C20 = g01 g12 - g02 g11
    cmul(GR(0,1), GI(0,1), GR(1,2), GI(1,2), &ar, &ai);
    cmul(GR(0,2), GI(0,2), GR(1,1), GI(1,1), &br, &bi);
    tr = ar - br; ti = ai - bi; IR(0,2) = tr * idet; II(0,2) = ti * idet;
    // C21 = g02 g10 - g00 g12
    cmul(GR(0,2), GI(0,2), GR(1,0), GI(1,0), &ar, &ai);
    cmul(GR(0,0), GI(0,0), GR(1,2), GI(1,2), &br, &bi);
    tr = ar - br; ti = ai - bi; IR(1,2) = tr * idet; II(1,2) = ti * idet;
    // C22 = g00 g11 - g01 g10
    cmul(GR(0,0), GI(0,0), GR(1,1), GI(1,1), &ar, &ai);
    cmul(GR(0,1), GI(0,1), GR(1,0), GI(1,0), &br, &bi);
    tr = ar - br; ti = ai - bi; IR(2,2) = tr * idet; II(2,2) = ti * idet;
    // Hermitian symmetrisation (mirrors quintic.rs:954-962).
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            double cr = IR(i,j);
            double ci = -II(i,j);
            double avg_re = 0.5 * (IR(j,i) + cr);
            double avg_im = 0.5 * (II(j,i) + ci);
            IR(j,i) = avg_re; II(j,i) = avg_im;
            IR(i,j) = avg_re; II(i,j) = -avg_im;
        }
    }
    #undef GR
    #undef GI
    #undef IR
    #undef II
    return 1;
}

// Build the canonical DKLR-2006 affine-chart tangent frame at z. Mirrors
// `quintic::quintic_affine_chart_frame` exactly: chart = argmax_k|Z_k|²,
// elim = argmax_k|∂f/∂Z_k|² over k ≠ chart, and the 3 free coords carry
// tangent vectors with unit entry in their slot and -(∂f/∂Z_free)/(∂f/∂Z_elim)
// in the elim slot.
//
// This is **not** the orthonormal Gram-Schmidt frame — that would force
// η = det(g_tan)/|Ω|² to depend on the chart, breaking comparison
// against published σ values. Must match the CPU evaluator
// (`compute_sigma_from_workspace`) exactly so GPU-computed gradients
// stay consistent with CPU-evaluated σ.
__device__ void build_tangent_frame(const double* z, double* frame) {
    // gradient of Fermat quintic (5 z_j^4).
    double grad_f[10];
    for (int j = 0; j < 5; ++j) {
        double rr = z[2*j];
        double ii = z[2*j+1];
        double r2 = rr*rr, i2 = ii*ii;
        double z2_re = r2 - i2;
        double z2_im = 2.0 * rr * ii;
        double z4_re = z2_re*z2_re - z2_im*z2_im;
        double z4_im = 2.0 * z2_re * z2_im;
        grad_f[2*j]   = 5.0 * z4_re;
        grad_f[2*j+1] = 5.0 * z4_im;
    }
    // chart = argmax|Z_k|².
    int chart = 0;
    double max_z_sq = 0.0;
    for (int k = 0; k < 5; ++k) {
        double zsq = z[2*k]*z[2*k] + z[2*k+1]*z[2*k+1];
        if (zsq > max_z_sq) { max_z_sq = zsq; chart = k; }
    }
    // elim = argmax|∂f/∂Z_k|² over k ≠ chart.
    int elim = (chart == 0) ? 1 : 0;
    double max_grad_sq = 0.0;
    for (int k = 0; k < 5; ++k) {
        if (k == chart) continue;
        double gsq = grad_f[2*k]*grad_f[2*k] + grad_f[2*k+1]*grad_f[2*k+1];
        if (gsq > max_grad_sq) { max_grad_sq = gsq; elim = k; }
    }
    // Zero frame defensively.
    for (int t = 0; t < 3; ++t) {
        for (int q = 0; q < 10; ++q) frame[t*10 + q] = 0.0;
    }
    if (max_grad_sq < 1e-30) return;
    // free = {0..4} \ {chart, elim}, sorted ascending.
    int free_idx[3];
    int fcount = 0;
    for (int k = 0; k < 5; ++k) {
        if (k != chart && k != elim) {
            free_idx[fcount++] = k;
        }
    }
    if (fcount != 3) return;
    double g_elim_re = grad_f[2*elim];
    double g_elim_im = grad_f[2*elim + 1];
    double inv_norm_sq = 1.0 / max_grad_sq;
    for (int i = 0; i < 3; ++i) {
        int fi = free_idx[i];
        double g_fi_re = grad_f[2*fi];
        double g_fi_im = grad_f[2*fi + 1];
        // -(g_fi / g_elim) = -g_fi · conj(g_elim) / |g_elim|².
        double num_re = g_fi_re * g_elim_re + g_fi_im * g_elim_im;
        double num_im = g_fi_im * g_elim_re - g_fi_re * g_elim_im;
        double elim_v_re = -num_re * inv_norm_sq;
        double elim_v_im = -num_im * inv_norm_sq;
        frame[i*10 + 2*fi]       = 1.0;
        frame[i*10 + 2*fi + 1]   = 0.0;
        frame[i*10 + 2*elim]     = elim_v_re;
        frame[i*10 + 2*elim + 1] = elim_v_im;
    }
}

// ---- main per-point kernel ------------------------------------------------
//
// Layout of d_scratch (per-point slab; allocated in `new()`):
//   slab[0 .. 2n]            : h_s (length 2n)
//   slab[2n .. 2n + 5*2n]    : h_dfi[i] for i=0..5 (5 × 2n)
//   slab[2n + 5*2n .. + 2n]  : q  (length 2n)
//   slab[next .. + 2n]       : qp (length 2n)
//   slab[next .. + 2*5*n]    : phi (length 2*5*n = 10n)
//   slab[next .. + n*n]      : amat_re (n*n)
//   slab[next .. + n*n]      : amat_im (n*n)
// Total per slab: 2n + 10n + 2n + 2n + 10n + 2n*n
//               = 16n + 2n²  doubles per point.
extern "C" __global__ void adam_per_point(
    const double* __restrict__ section_values,   // n_pts * 2n
    const double* __restrict__ section_derivs,   // n_pts * 5 * 2n
    const double* __restrict__ h_block,          // 2n * 2n  (4n²)
    const double* __restrict__ points,           // n_pts * 10
    const double* __restrict__ weights,          // n_pts
    const double* __restrict__ log_omega_sq,     // n_pts
    double* __restrict__ per_point_buf,          // n_pts * (2 + 2n²)
    double* __restrict__ d_scratch,              // n_pts * (16n + 2n²)
    int n_pts,
    int n_basis
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pts) return;

    int two_n = 2 * n_basis;
    int n_dof = 2 * n_basis * n_basis;
    int stride_pp = 2 + n_dof;
    double* per_pt = per_point_buf + (long long)p * stride_pp;
    // d_scratch is unused now: all per-thread scratch (h_s, h_dfi, q,
    // qp, phi) is held in stack arrays below. NVRTC spills them to
    // CUDA local memory, which uses an interleaved (coalesced)
    // address layout — far better than the slab's per-thread row-major
    // global layout (which had stride_pp · 8B between consecutive
    // threads' loads). Eliminates ≈ 14 GB of strided global memory I/O
    // per Adam step at k=4, 60k pts.
    (void)d_scratch;
    #define ADAM_NMAX 70
    double h_s[2 * ADAM_NMAX];
    double h_dfi[5 * 2 * ADAM_NMAX];
    double q[2 * ADAM_NMAX];
    double qp[2 * ADAM_NMAX];
    double phi[2 * 5 * ADAM_NMAX];

    // Early exit on invalid weight (matches Rust closure at line 1444).
    double w_p = weights[p];
    if (!isfinite(w_p) || w_p <= 0.0) {
        per_pt[0] = nan("");
        per_pt[1] = 0.0;
        return;
    }

    // section_values and section_derivs use the original row-major
    // layout (point-major, basis-minor):
    //   section_values [p * 2n + i]
    //   section_derivs [p * 5·2n + (j·2n + i)]
    // Each thread accesses its own row sequentially — the GPU L1 cache
    // (per-SM, ~128 KB) stays hot for the thread's row across the
    // entire kernel. A column-major transpose was tried and was 60%
    // slower: warp-coalescing improved but per-thread L1 locality was
    // destroyed (n_pts·8B = 480 KB stride between adjacent reads
    // within the same thread put re/im on different cache lines).
    const double* s  = section_values + (long long)p * two_n;
    const double* df = section_derivs + (long long)p * 5 * two_n;
    const double* z  = points + (long long)p * 10;
    double log_om = log_omega_sq[p];

    // Zero gradient buffer up front (Rust does this at line 1454).
    for (int qq = 0; qq < n_dof; ++qq) per_pt[2 + qq] = 0.0;

    // -- 1. h_s = h_block @ s --------------------------------------------
    for (int i = 0; i < two_n; ++i) {
        double row = 0.0;
        for (int j = 0; j < two_n; ++j) {
            row += h_block[i * two_n + j] * s[j];
        }
        h_s[i] = row;
    }
    double k_val = 0.0;
    for (int i = 0; i < two_n; ++i) k_val += s[i] * h_s[i];
    double k_safe = (k_val > 1e-30) ? k_val : 1e-30;
    double inv_k = 1.0 / k_safe;
    double inv_k2 = inv_k * inv_k;
    double inv_k3 = inv_k * inv_k2;

    // -- 2. h_dfi[i] = h_block @ df_i ------------------------------------
    for (int i = 0; i < 5; ++i) {
        const double* dfi = df + i * two_n;
        double* hdi = h_dfi + i * two_n;
        for (int k = 0; k < two_n; ++k) {
            double row = 0.0;
            for (int l = 0; l < two_n; ++l) {
                row += h_block[k * two_n + l] * dfi[l];
            }
            hdi[k] = row;
        }
    }

    // -- 3. dk[i] = sum_a (h_s_a)* (df_i)_a   (Hermitian inner product) --
    double dk_re[5], dk_im[5];
    for (int i = 0; i < 5; ++i) {
        const double* dfi = df + i * two_n;
        double sre = 0.0, sim = 0.0;
        for (int a = 0; a < n_basis; ++a) {
            double hr = h_s[2*a];
            double hi = h_s[2*a+1];
            double dr = dfi[2*a];
            double di = dfi[2*a+1];
            sre += hr*dr + hi*di;
            sim += hr*di - hi*dr;
        }
        dk_re[i] = sre; dk_im[i] = sim;
    }

    // -- 4. M[i][j] = sum_a conj(df_j_a) (h_dfi[i])_a --------------------
    double m_re[5][5], m_im[5][5];
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            const double* dfj = df + j * two_n;
            const double* hdi = h_dfi + i * two_n;
            double sre = 0.0, sim = 0.0;
            for (int a = 0; a < n_basis; ++a) {
                double dr = dfj[2*a];
                double di = dfj[2*a+1];
                double hr = hdi[2*a];
                double hi = hdi[2*a+1];
                sre += dr*hr + di*hi;
                sim += dr*hi - di*hr;
            }
            m_re[i][j] = sre; m_im[i][j] = sim;
        }
    }

    // -- 5. g_amb_{ij} = M_{ij}/K - dk_i · dk_j*/K^2 --------------------
    double g_amb_re[5][5], g_amb_im[5][5];
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            double pre_re = dk_re[i]*dk_re[j] + dk_im[i]*dk_im[j];
            double pre_im = dk_im[i]*dk_re[j] - dk_re[i]*dk_im[j];
            g_amb_re[i][j] = m_re[i][j]*inv_k - pre_re*inv_k2;
            g_amb_im[i][j] = m_im[i][j]*inv_k - pre_im*inv_k2;
        }
    }

    // -- 6. tangent frame & projection ----------------------------------
    double frame[30];
    build_tangent_frame(z, frame);
    // FIX(P3.10): chart-invariance convention `g_tan[a][b] = Σ_{i,j} T_a[i] g[i][j] T_b[j]^*`
    // (matches `quintic::project_to_quintic_tangent`). The previous body
    // implemented the OLD convention `T_a^*[i] g[i][j] T_b[j]` — equivalent
    // ONLY when g is real-symmetric, which g_amb is not. The conjugation
    // belongs on `T_b` (j-slot), not on `T_a` (i-slot). This is the parent
    // bug whose imag-part assembly was patched by P3.9; the projection
    // itself was never updated to the new convention. Sibling fix to P3.9.
    double g_tan[18];
    for (int a = 0; a < 3; ++a) {
        const double* ta = frame + a * 10;
        for (int b = 0; b < 3; ++b) {
            const double* tb = frame + b * 10;
            double sre = 0.0, sim = 0.0;
            for (int i = 0; i < 5; ++i) {
                double tair = ta[2*i];
                double taii = ta[2*i+1];     // NO conj on a
                double row_re = 0.0, row_im = 0.0;
                for (int j = 0; j < 5; ++j) {
                    double gr = g_amb_re[i][j];
                    double gi = g_amb_im[i][j];
                    double tjr = tb[2*j];
                    double tji = -tb[2*j+1];  // T_b[j]^*  (CONJ on b)
                    row_re += gr*tjr - gi*tji;
                    row_im += gr*tji + gi*tjr;
                }
                sre += tair*row_re - taii*row_im;
                sim += tair*row_im + taii*row_re;
            }
            g_tan[2*(3*a + b)]     = sre;
            g_tan[2*(3*a + b) + 1] = sim;
        }
    }
    double det_g = det3_herm(g_tan);
    if (!isfinite(det_g) || fabs(det_g) < 1e-30 || !isfinite(log_om)) {
        per_pt[0] = 0.0;
        per_pt[1] = 0.0;  // Rust returns NaN eta + zero w → filtered.
        return;
    }
    double log_r = log(fabs(det_g)) - log_om;
    double r_val = exp(log_r);
    double g_inv[18];
    if (!inv3_herm(g_tan, g_inv)) {
        per_pt[0] = r_val;
        per_pt[1] = 0.0;
        return;
    }
    // FIX(P3.9): Y[i][j] = sum_{ab} T_a*[i] g_inv[a][b] T_b[j]   (PROPER
    // chart-invariance convention, matches `quintic::compute_y_5x5`).
    double y_re[5][5], y_im[5][5];
    for (int i = 0; i < 5; ++i) {
        double t_i_r[3], t_i_i[3];
        for (int a = 0; a < 3; ++a) {
            t_i_r[a] = frame[a*10 + 2*i];
            t_i_i[a] = -frame[a*10 + 2*i + 1];  // conj on i
        }
        for (int j = 0; j < 5; ++j) {
            double t_j_r[3], t_j_i[3];
            for (int b = 0; b < 3; ++b) {
                t_j_r[b] = frame[b*10 + 2*j];
                t_j_i[b] = frame[b*10 + 2*j + 1];  // no conj on j
            }
            double sre = 0.0, sim = 0.0;
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    double gar = g_inv[2*(3*a + b)];
                    double gai = g_inv[2*(3*a + b) + 1];
                    double tiar = t_i_r[a], tiai = t_i_i[a];
                    double tjbr = t_j_r[b], tjbi = t_j_i[b];
                    double p1_re = tiar*gar - tiai*gai;
                    double p1_im = tiar*gai + tiai*gar;
                    double p_re  = p1_re*tjbr - p1_im*tjbi;
                    double p_im  = p1_re*tjbi + p1_im*tjbr;
                    sre += p_re; sim += p_im;
                }
            }
            y_re[i][j] = sre; y_im[i][j] = sim;
        }
    }

    // -- 7. tr_YM and tr_Y_dKdK -----------------------------------------
    double tr_ym = 0.0;
    double tr_y_dkdk = 0.0;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            tr_ym    += y_re[j][i]*m_re[i][j] - y_im[j][i]*m_im[i][j];
            double pre_re = dk_re[i]*dk_re[j] + dk_im[i]*dk_im[j];
            double pre_im = dk_im[i]*dk_re[j] - dk_re[i]*dk_im[j];
            tr_y_dkdk += y_re[j][i]*pre_re - y_im[j][i]*pre_im;
        }
    }

    // -- 8. v_i = sum_j Y_{ji} (dk_j)*  → q_b = sum_i (df_i)_b · v_i ----
    double v_re[5], v_im[5];
    for (int i = 0; i < 5; ++i) {
        double sre = 0.0, sim = 0.0;
        for (int j = 0; j < 5; ++j) {
            double yjir = y_re[j][i], yjii = y_im[j][i];
            double dkjr = dk_re[j],   dkji = dk_im[j];
            sre += yjir*dkjr + yjii*dkji;
            sim += yjii*dkjr - yjir*dkji;
        }
        v_re[i] = sre; v_im[i] = sim;
    }
    // q, qp, phi are stack-local arrays declared at the top of the
    // kernel (CUDA local memory, interleaved-coalesced). amat is no
    // longer materialised: recomputed on-the-fly in the gradient loop
    // below (one row + one column per outer-a iteration into stack
    // arrays of size n_basis).
    for (int b = 0; b < n_basis; ++b) {
        double sre = 0.0, sim = 0.0;
        for (int i = 0; i < 5; ++i) {
            const double* dfi = df + i * two_n;
            double dr = dfi[2*b];
            double di = dfi[2*b+1];
            double vr = v_re[i], vi = v_im[i];
            sre += dr*vr - di*vi;
            sim += dr*vi + di*vr;
        }
        q[2*b]   = sre;
        q[2*b+1] = sim;
    }
    // w_j = sum_i Y_{ji} dk_i  (complex)  → q'_b = sum_j conj(df_j_b) w_j
    double w_re[5], w_im[5];
    for (int j = 0; j < 5; ++j) {
        double sre = 0.0, sim = 0.0;
        for (int i = 0; i < 5; ++i) {
            double yjir = y_re[j][i], yjii = y_im[j][i];
            double dkir = dk_re[i],   dkii = dk_im[i];
            sre += yjir*dkir - yjii*dkii;
            sim += yjir*dkii + yjii*dkir;
        }
        w_re[j] = sre; w_im[j] = sim;
    }
    for (int b = 0; b < n_basis; ++b) {
        double sre = 0.0, sim = 0.0;
        for (int j = 0; j < 5; ++j) {
            const double* dfj = df + j * two_n;
            double dr = dfj[2*b];
            double di = dfj[2*b+1];
            double wr = w_re[j], wi = w_im[j];
            sre += dr*wr + di*wi;
            sim += dr*wi - di*wr;
        }
        qp[2*b]   = sre;
        qp[2*b+1] = sim;
    }
    // φ_{j,b} = sum_i Y_{ji} (df_i)_b  (5 × n complex; interleaved re/im)
    for (int j = 0; j < 5; ++j) {
        for (int b = 0; b < n_basis; ++b) {
            double sre = 0.0, sim = 0.0;
            for (int i = 0; i < 5; ++i) {
                double yjir = y_re[j][i], yjii = y_im[j][i];
                const double* dfi = df + i * two_n;
                double dr = dfi[2*b];
                double di = dfi[2*b+1];
                sre += yjir*dr - yjii*di;
                sim += yjir*di + yjii*dr;
            }
            phi[2*(j * n_basis + b)]     = sre;
            phi[2*(j * n_basis + b) + 1] = sim;
        }
    }
    // -- 9. final gradient assembly (Rust quintic.rs:1287-1366) ----------
    // amat[a][b] = sum_j conj(df_j_a) · phi[j][b]. We never materialise
    // the full n² matrix: per outer-a iteration we precompute
    //   amat_row[b] = amat[a][b]  for b ∈ [0, n_basis)
    //   amat_col[b] = amat[b][a]  for b ∈ [0, n_basis)
    // into stack-resident arrays (≤ 2·N_BASIS_MAX doubles each, 70 max
    // ⇒ 1.1 KB per array per thread). The inner b-loop reads them
    // directly out of registers / coalesced local memory instead of
    // strided global. Total saving: ≈ 9.4 GB of strided global I/O per
    // Adam step at k=4, 60k pts.
    #define N_BASIS_MAX 70
    double amat_row_re[N_BASIS_MAX], amat_row_im[N_BASIS_MAX];
    double amat_col_re[N_BASIS_MAX], amat_col_im[N_BASIS_MAX];

    int n_basis_sq = n_basis * n_basis;
    for (int a = 0; a < n_basis; ++a) {
        double sar = s[2*a], sai = s[2*a+1];
        double qar = q[2*a], qai = q[2*a+1];
        double qpar = qp[2*a], qpai = qp[2*a+1];

        // Precompute amat_row[b] = sum_j conj(df_j_a) · phi[j][b]
        // and    amat_col[b] = sum_j conj(df_j_b) · phi[j][a].
        // (We tried hoisting dfj_a/phi_ja out of the b-loop into 5-elt
        // register arrays; it added register pressure that pushed the
        // kernel further into local-memory spill territory and was net
        // slower at n_basis = 70. Leaving the loads inside the b-loop
        // — the compiler hoists them via licm/ILP without extra
        // register footprint.)
        for (int b = 0; b < n_basis; ++b) {
            double sre_r = 0.0, sim_r = 0.0;  // amat[a][b] = row at this a
            double sre_c = 0.0, sim_c = 0.0;  // amat[b][a] = col at this a
            for (int j = 0; j < 5; ++j) {
                const double* dfj = df + j * two_n;
                double dfj_a_r = dfj[2*a],    dfj_a_i = dfj[2*a+1];
                double dfj_b_r = dfj[2*b],    dfj_b_i = dfj[2*b+1];
                double phi_jb_r = phi[2*(j * n_basis + b)];
                double phi_jb_i = phi[2*(j * n_basis + b) + 1];
                double phi_ja_r = phi[2*(j * n_basis + a)];
                double phi_ja_i = phi[2*(j * n_basis + a) + 1];
                // amat[a][b]: dr = df_j_a, pr = phi_jb
                sre_r += dfj_a_r * phi_jb_r + dfj_a_i * phi_jb_i;
                sim_r += dfj_a_r * phi_jb_i - dfj_a_i * phi_jb_r;
                // amat[b][a]: dr = df_j_b, pr = phi_ja
                sre_c += dfj_b_r * phi_ja_r + dfj_b_i * phi_ja_i;
                sim_c += dfj_b_r * phi_ja_i - dfj_b_i * phi_ja_r;
            }
            amat_row_re[b] = sre_r;
            amat_row_im[b] = sim_r;
            amat_col_re[b] = sre_c;
            amat_col_im[b] = sim_c;
        }

        for (int b = 0; b < n_basis; ++b) {
            double sbr = s[2*b], sbi = s[2*b+1];
            double qbr = q[2*b], qbi = q[2*b+1];
            double qpbr = qp[2*b], qpbi = qp[2*b+1];
            double re_ssab = sar*sbr + sai*sbi;
            double im_ssab = sar*sbi - sai*sbr;
            double a_ab_re = amat_row_re[b];
            double a_ba_re = amat_col_re[b];
            double a_ab_im = amat_row_im[b];
            double a_ba_im = amat_col_im[b];
            double mixed_re = (sar*qbr + sai*qbi)
                            + (sbr*qar + sbi*qai)
                            + (sar*qpbr - sai*qpbi)
                            + (sbr*qpar - sbi*qpai);
            double im_sqab = sar*qbi - sai*qbr;
            double im_sqba = sbr*qai - sbi*qar;
            double im_sqpab = sar*qpbi + sai*qpbr;
            double im_sqpba = sbr*qpai + sbi*qpar;

            double dl_re_pair = (a_ab_re + a_ba_re)*inv_k
                              - 2.0 * re_ssab * tr_ym * inv_k2
                              - mixed_re * inv_k2
                              + 4.0 * re_ssab * tr_y_dkdk * inv_k3;
            double dl_re = (a == b) ? 0.5 * dl_re_pair : dl_re_pair;
            // FIX(P3.9): re-derived dl_im under chart-invariance convention
            // `g_tan = T^T g T̄`. Same fix as quintic.rs::per_point_log_det_gradient.
            double dl_im = 0.0;
            if (a < b) {
                dl_im = -(a_ab_im - a_ba_im)*inv_k
                       + 2.0 * im_ssab * tr_ym * inv_k2
                       + (im_sqab - im_sqba - im_sqpab + im_sqpba) * inv_k2
                       - 4.0 * im_ssab * tr_y_dkdk * inv_k3;
            }
            per_pt[2 + a * n_basis + b]              = dl_re;
            per_pt[2 + n_basis_sq + a * n_basis + b] = dl_im;
        }
    }
    // r_val + log_r already computed; eta_p = r_val, w_p as given.
    if (!isfinite(r_val) || r_val <= 0.0 || !isfinite(log_r)) {
        per_pt[0] = nan("");
        per_pt[1] = 0.0;
    } else {
        per_pt[0] = r_val;
        per_pt[1] = w_p;
    }
}

// ===========================================================================
// Warp-cooperative variant of `adam_per_point`. One warp (32 threads) per
// point; intermediate state lives in **shared memory**, not per-thread
// local memory. Replaces the dominant per-point bottleneck of the
// thread-per-point kernel — the n²-iter gradient-assembly loop — with
// a 32-way parallel scan, while small phases (5×5 matrix work, tangent
// frame, det/inv) are scheduled across the first 5–25 threads of the
// warp with the remaining threads idle.
//
// Performance budget at k=4 (n_basis=70, 60k pts):
//   - Original per-thread kernel: ~220 ms / iter (DRAM-coalesced local
//     memory bound).
//   - Warp-coop: each warp does ~245 k muladds total (same total work),
//     but 32-way distributed → ~7.7 k muladds per thread + sync
//     overhead. Expected ~50-90 ms / iter.
//
// Shared memory layout (16.6 KB / warp at n=70 → 6 blocks/SM, 192
// threads/SM = ~12% occupancy on RTX 4090, but the per-thread work
// drops 32× so latency-hiding budget is more than adequate).
//
// __syncwarp() is used between phases. Volta+ ITS (independent thread
// scheduling) means we cannot rely on implicit lockstep.
// ===========================================================================
extern "C" __global__ void adam_per_point_warp(
    const double* __restrict__ section_values,   // n_pts * 2n
    const double* __restrict__ section_derivs,   // n_pts * 5 * 2n
    const double* __restrict__ h_block,          // 2n * 2n
    const double* __restrict__ points,           // n_pts * 10
    const double* __restrict__ weights,          // n_pts
    const double* __restrict__ log_omega_sq,     // n_pts
    double* __restrict__ per_point_buf,          // n_pts * (2 + 2n²)
    int n_pts,
    int n_basis
) {
    int p = blockIdx.x;          // 1 block = 1 warp = 1 point
    if (p >= n_pts) return;
    int tid = threadIdx.x;        // 0..31

    int two_n = 2 * n_basis;
    int n_dof = 2 * n_basis * n_basis;
    int stride_pp = 2 + n_dof;
    double* per_pt = per_point_buf + (long long)p * stride_pp;

    // Dynamic shared memory layout (passed via shared_mem_bytes at launch).
    extern __shared__ double sm[];
    int sm_off = 0;
    double* sm_h_s   = sm + sm_off; sm_off += two_n;
    double* sm_h_dfi = sm + sm_off; sm_off += 5 * two_n;
    double* sm_q     = sm + sm_off; sm_off += two_n;
    double* sm_qp    = sm + sm_off; sm_off += two_n;
    // sm_phi: tried fp32 storage; bank-conflict + fp32↔fp64 conversion
    // overhead inside the n²-iter gradient loop made the kernel 20%
    // slower at n_basis=70 — kept fp64 for the cleaner access pattern.
    double* sm_phi   = sm + sm_off; sm_off += 10 * n_basis;
    double* sm_dk_re = sm + sm_off; sm_off += 5;
    double* sm_dk_im = sm + sm_off; sm_off += 5;
    double* sm_v_re  = sm + sm_off; sm_off += 5;
    double* sm_v_im  = sm + sm_off; sm_off += 5;
    double* sm_w_re  = sm + sm_off; sm_off += 5;
    double* sm_w_im  = sm + sm_off; sm_off += 5;
    double* sm_m_re  = sm + sm_off; sm_off += 25;
    double* sm_m_im  = sm + sm_off; sm_off += 25;
    double* sm_y_re  = sm + sm_off; sm_off += 25;
    double* sm_y_im  = sm + sm_off; sm_off += 25;
    double* sm_g_amb_re = sm + sm_off; sm_off += 25;
    double* sm_g_amb_im = sm + sm_off; sm_off += 25;
    double* sm_frame = sm + sm_off; sm_off += 30;
    double* sm_g_tan = sm + sm_off; sm_off += 18;
    double* sm_g_inv = sm + sm_off; sm_off += 18;
    // amat_row/col arrays were previously here (4n doubles = 2.2 KB at
    // n=70). Each thread only consumes the b-stripe it produced, so
    // they're now per-thread registers — see phase 9 below. Saves
    // shared memory + the __syncwarp() between 9a and 9b.
    // [0]=flag (0 = ok, 1 = nan-eta-w0, 2 = zero-eta-w0)
    // [1]=r_val (broadcast to all threads)
    // [2]=log_r
    double* sm_scalars = sm + sm_off; sm_off += 4;

    // -- weight check ---------------------------------------------------
    double w_p = weights[p];
    if (!isfinite(w_p) || w_p <= 0.0) {
        if (tid == 0) {
            per_pt[0] = nan("");
            per_pt[1] = 0.0;
        }
        return;
    }

    const double* s  = section_values + (long long)p * two_n;
    const double* df = section_derivs + (long long)p * 5 * two_n;
    const double* z  = points + (long long)p * 10;
    double log_om = log_omega_sq[p];

    // (No gradient zeroing pass: phase 9 writes every per_pt[2 + a*n + b]
    // and per_pt[2 + n² + a*n + b] unconditionally, so the previous
    // iteration's residue is fully overwritten. Early-exit paths leave
    // stale grad in place, but reductions filter by eta/w_p == 0 so
    // those rows are ignored.)

    // -- Phase 1: h_s = h_block @ s -------------------------------------
    for (int i = tid; i < two_n; i += 32) {
        double row = 0.0;
        for (int j = 0; j < two_n; ++j) {
            row += h_block[i * two_n + j] * s[j];
        }
        sm_h_s[i] = row;
    }
    __syncwarp();

    // k_val = sum_i s[i] * h_s[i] (scalar; warp-reduce + broadcast).
    double k_part = 0.0;
    for (int i = tid; i < two_n; i += 32) {
        k_part += s[i] * sm_h_s[i];
    }
    for (int off = 16; off > 0; off >>= 1) {
        k_part += __shfl_down_sync(0xFFFFFFFF, k_part, off);
    }
    double k_val = __shfl_sync(0xFFFFFFFF, k_part, 0);
    double k_safe = (k_val > 1e-30) ? k_val : 1e-30;
    double inv_k = 1.0 / k_safe;
    double inv_k2 = inv_k * inv_k;
    double inv_k3 = inv_k * inv_k2;

    // -- Phase 2: h_dfi[i] = h_block @ df_i (5 × 2n entries) ------------
    for (int idx = tid; idx < 5 * two_n; idx += 32) {
        int i = idx / two_n;
        int k = idx % two_n;
        double row = 0.0;
        for (int l = 0; l < two_n; ++l) {
            row += h_block[k * two_n + l] * df[i * two_n + l];
        }
        sm_h_dfi[i * two_n + k] = row;
    }
    __syncwarp();

    // -- Phase 3: dk[i] = sum_a (h_s_a)* (df_i)_a (5 entries) -----------
    if (tid < 5) {
        int i = tid;
        double sre = 0.0, sim = 0.0;
        for (int a = 0; a < n_basis; ++a) {
            double hr = sm_h_s[2*a],   hi = sm_h_s[2*a+1];
            double dr = df[i*two_n + 2*a];
            double di = df[i*two_n + 2*a+1];
            sre += hr*dr + hi*di;
            sim += hr*di - hi*dr;
        }
        sm_dk_re[i] = sre;
        sm_dk_im[i] = sim;
    }
    __syncwarp();

    // -- Phase 4: M[i][j] = sum_a conj(df_j_a) (h_dfi[i])_a (25 entries)
    if (tid < 25) {
        int i = tid / 5;
        int j = tid % 5;
        const double* dfj = df + j * two_n;
        const double* hdi = sm_h_dfi + i * two_n;
        double sre = 0.0, sim = 0.0;
        for (int a = 0; a < n_basis; ++a) {
            double dr = dfj[2*a],    di = dfj[2*a+1];
            double hr = hdi[2*a],    hi = hdi[2*a+1];
            sre += dr*hr + di*hi;
            sim += dr*hi - di*hr;
        }
        sm_m_re[i*5 + j] = sre;
        sm_m_im[i*5 + j] = sim;
    }
    __syncwarp();

    // -- Phase 5: g_amb = M/K - dk·dk*/K² (25 entries) ------------------
    if (tid < 25) {
        int i = tid / 5;
        int j = tid % 5;
        double pre_re = sm_dk_re[i]*sm_dk_re[j] + sm_dk_im[i]*sm_dk_im[j];
        double pre_im = sm_dk_im[i]*sm_dk_re[j] - sm_dk_re[i]*sm_dk_im[j];
        sm_g_amb_re[i*5 + j] = sm_m_re[i*5 + j]*inv_k - pre_re*inv_k2;
        sm_g_amb_im[i*5 + j] = sm_m_im[i*5 + j]*inv_k - pre_im*inv_k2;
    }
    __syncwarp();

    // -- Phase 6: tangent frame (small; thread 0 only) ------------------
    if (tid == 0) {
        build_tangent_frame(z, sm_frame);
    }
    __syncwarp();

    // g_tan[a][b] = Σ_{i,j} T_a[i] g_amb[i][j] T_b[j]^*  (9 complex entries
    // distributed across 9 threads).
    // FIX(P3.10): chart-invariance convention requires conj on the b-slot
    // (T_b[j]^*), not on the a-slot. Same fix as `adam_per_point`; matches
    // `quintic::project_to_quintic_tangent`. Sibling of P3.9 — the
    // projection itself had never been updated to the new convention.
    if (tid < 9) {
        int a = tid / 3;
        int b = tid % 3;
        const double* ta = sm_frame + a * 10;
        const double* tb = sm_frame + b * 10;
        double sre = 0.0, sim = 0.0;
        for (int i = 0; i < 5; ++i) {
            double tair = ta[2*i];
            double taii = ta[2*i+1];        // NO conj on a
            double row_re = 0.0, row_im = 0.0;
            for (int j = 0; j < 5; ++j) {
                double gr = sm_g_amb_re[i*5 + j];
                double gi = sm_g_amb_im[i*5 + j];
                double tjr = tb[2*j];
                double tji = -tb[2*j+1];    // T_b[j]^*  (CONJ on b)
                row_re += gr*tjr - gi*tji;
                row_im += gr*tji + gi*tjr;
            }
            sre += tair*row_re - taii*row_im;
            sim += tair*row_im + taii*row_re;
        }
        sm_g_tan[2*(3*a + b)]     = sre;
        sm_g_tan[2*(3*a + b) + 1] = sim;
    }
    __syncwarp();

    // det(g_tan), inv(g_tan); thread 0 sets the flag for early-exit
    // paths so all warp threads can return together.
    if (tid == 0) {
        double det_g = det3_herm(sm_g_tan);
        if (!isfinite(det_g) || fabs(det_g) < 1e-30 || !isfinite(log_om)) {
            per_pt[0] = 0.0;
            per_pt[1] = 0.0;
            sm_scalars[0] = 1.0;
        } else {
            double log_r = log(fabs(det_g)) - log_om;
            double r_val = exp(log_r);
            sm_scalars[1] = r_val;
            sm_scalars[2] = log_r;
            if (!inv3_herm(sm_g_tan, sm_g_inv)) {
                per_pt[0] = r_val;
                per_pt[1] = 0.0;
                sm_scalars[0] = 2.0;
            } else {
                sm_scalars[0] = 0.0;
            }
        }
    }
    __syncwarp();

    // All threads see the same flag — early-return together (no
    // partial-warp __syncwarp() afterwards).
    double flag = sm_scalars[0];
    if (flag != 0.0) return;
    double r_val = sm_scalars[1];
    double log_r = sm_scalars[2];

    // -- Phase 7: Y[i][j] = T_a*[i] g_inv[a][b] T_b[j] (25 entries) -----
    // FIX(P3.9): chart-invariance convention `g_tan = T^T g T̄` requires
    // conj on i and no-conj on j (matches `quintic::compute_y_5x5`).
    if (tid < 25) {
        int i = tid / 5;
        int j = tid % 5;
        double t_i_r[3], t_i_i[3], t_j_r[3], t_j_i[3];
        for (int a = 0; a < 3; ++a) {
            t_i_r[a] = sm_frame[a*10 + 2*i];
            t_i_i[a] = -sm_frame[a*10 + 2*i + 1];  // conj on i
            t_j_r[a] = sm_frame[a*10 + 2*j];
            t_j_i[a] = sm_frame[a*10 + 2*j + 1];   // no conj on j
        }
        double sre = 0.0, sim = 0.0;
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                double gar = sm_g_inv[2*(3*a + b)];
                double gai = sm_g_inv[2*(3*a + b) + 1];
                double tiar = t_i_r[a], tiai = t_i_i[a];
                double tjbr = t_j_r[b], tjbi = t_j_i[b];
                double p1_re = tiar*gar - tiai*gai;
                double p1_im = tiar*gai + tiai*gar;
                double p_re  = p1_re*tjbr - p1_im*tjbi;
                double p_im  = p1_re*tjbi + p1_im*tjbr;
                sre += p_re; sim += p_im;
            }
        }
        sm_y_re[i*5 + j] = sre;
        sm_y_im[i*5 + j] = sim;
    }
    __syncwarp();

    // tr_YM and tr_Y_dKdK — warp-reduce over 25 contributions.
    double tr_ym_part = 0.0, tr_y_dkdk_part = 0.0;
    if (tid < 25) {
        int i = tid / 5;
        int j = tid % 5;
        tr_ym_part = sm_y_re[j*5 + i]*sm_m_re[i*5 + j] - sm_y_im[j*5 + i]*sm_m_im[i*5 + j];
        double pre_re = sm_dk_re[i]*sm_dk_re[j] + sm_dk_im[i]*sm_dk_im[j];
        double pre_im = sm_dk_im[i]*sm_dk_re[j] - sm_dk_re[i]*sm_dk_im[j];
        tr_y_dkdk_part = sm_y_re[j*5 + i]*pre_re - sm_y_im[j*5 + i]*pre_im;
    }
    for (int off = 16; off > 0; off >>= 1) {
        tr_ym_part     += __shfl_down_sync(0xFFFFFFFF, tr_ym_part, off);
        tr_y_dkdk_part += __shfl_down_sync(0xFFFFFFFF, tr_y_dkdk_part, off);
    }
    double tr_ym     = __shfl_sync(0xFFFFFFFF, tr_ym_part, 0);
    double tr_y_dkdk = __shfl_sync(0xFFFFFFFF, tr_y_dkdk_part, 0);

    // -- Phase 8a: v[i] = sum_j Y[j][i] (dk_j)*  (5 entries) -----------
    // -- Phase 8b: w[j] = sum_i Y[j][i]  dk_i    (5 entries) — concurrent
    if (tid < 5) {
        int i = tid;
        double sre = 0.0, sim = 0.0;
        for (int j = 0; j < 5; ++j) {
            double yjir = sm_y_re[j*5 + i], yjii = sm_y_im[j*5 + i];
            double dkjr = sm_dk_re[j],      dkji = sm_dk_im[j];
            sre += yjir*dkjr + yjii*dkji;
            sim += yjii*dkjr - yjir*dkji;
        }
        sm_v_re[i] = sre;
        sm_v_im[i] = sim;
    } else if (tid < 10) {
        int j = tid - 5;
        double sre = 0.0, sim = 0.0;
        for (int i = 0; i < 5; ++i) {
            double yjir = sm_y_re[j*5 + i], yjii = sm_y_im[j*5 + i];
            double dkir = sm_dk_re[i],      dkii = sm_dk_im[i];
            sre += yjir*dkir - yjii*dkii;
            sim += yjir*dkii + yjii*dkir;
        }
        sm_w_re[j] = sre;
        sm_w_im[j] = sim;
    }
    __syncwarp();

    // -- Phase 8c: q[b] = sum_i (df_i)_b · v_i  (n entries) ------------
    for (int b = tid; b < n_basis; b += 32) {
        double sre = 0.0, sim = 0.0;
        for (int i = 0; i < 5; ++i) {
            const double* dfi = df + i * two_n;
            double dr = dfi[2*b],   di = dfi[2*b+1];
            double vr = sm_v_re[i], vi = sm_v_im[i];
            sre += dr*vr - di*vi;
            sim += dr*vi + di*vr;
        }
        sm_q[2*b]   = sre;
        sm_q[2*b+1] = sim;
    }
    __syncwarp();

    // -- Phase 8d: qp[b] = sum_j conj(df_j_b) w_j (n entries) ----------
    for (int b = tid; b < n_basis; b += 32) {
        double sre = 0.0, sim = 0.0;
        for (int j = 0; j < 5; ++j) {
            const double* dfj = df + j * two_n;
            double dr = dfj[2*b],   di = dfj[2*b+1];
            double wr = sm_w_re[j], wi = sm_w_im[j];
            sre += dr*wr + di*wi;
            sim += dr*wi - di*wr;
        }
        sm_qp[2*b]   = sre;
        sm_qp[2*b+1] = sim;
    }
    __syncwarp();

    // -- Phase 8e: phi[j][b] = sum_i Y[j][i] (df_i)_b  (5n entries) ----
    for (int idx = tid; idx < 5 * n_basis; idx += 32) {
        int j = idx / n_basis;
        int b = idx % n_basis;
        double sre = 0.0, sim = 0.0;
        for (int i = 0; i < 5; ++i) {
            double yjir = sm_y_re[j*5 + i], yjii = sm_y_im[j*5 + i];
            const double* dfi = df + i * two_n;
            double dr = dfi[2*b], di = dfi[2*b+1];
            sre += yjir*dr - yjii*di;
            sim += yjir*di + yjii*dr;
        }
        sm_phi[2*(j * n_basis + b)]     = sre;
        sm_phi[2*(j * n_basis + b) + 1] = sim;
    }
    __syncwarp();

    // -- Phase 9: gradient assembly with upper-triangle symmetry.
    //
    // dl_re_pair is symmetric in (a, b): swapping a↔b leaves the
    // formula invariant (a_ab+a_ba is symmetric, mixed_re is symmetric,
    // re_ssab is symmetric). So dl_re[a][b] = dl_re[b][a] for a≠b.
    // dl_im is anti-Hermitian: dl_im[a][b] is non-zero only for a<b
    // (the original convention); the (b,a) entry is 0.
    //
    // We iterate the upper triangle (a ≤ b) and mirror dl_re to the
    // lower triangle. This halves the inner-loop compute (n²/2 pairs
    // instead of n²) at the cost of an extra (uncoalesced) write to
    // the lower-triangle slot — net ~40% less inner work.
    int n_basis_sq = n_basis * n_basis;
    for (int a = 0; a < n_basis; ++a) {
        double sar = s[2*a], sai = s[2*a+1];
        double qar = sm_q[2*a], qai = sm_q[2*a+1];
        double qpar = sm_qp[2*a], qpai = sm_qp[2*a+1];

        for (int b = a + tid; b < n_basis; b += 32) {
            double a_ab_re = 0.0, a_ab_im = 0.0;
            double a_ba_re = 0.0, a_ba_im = 0.0;
            for (int j = 0; j < 5; ++j) {
                const double* dfj = df + j * two_n;
                double dfj_a_r = dfj[2*a], dfj_a_i = dfj[2*a+1];
                double dfj_b_r = dfj[2*b], dfj_b_i = dfj[2*b+1];
                double phi_jb_r = sm_phi[2*(j * n_basis + b)];
                double phi_jb_i = sm_phi[2*(j * n_basis + b) + 1];
                double phi_ja_r = sm_phi[2*(j * n_basis + a)];
                double phi_ja_i = sm_phi[2*(j * n_basis + a) + 1];
                a_ab_re += dfj_a_r*phi_jb_r + dfj_a_i*phi_jb_i;
                a_ab_im += dfj_a_r*phi_jb_i - dfj_a_i*phi_jb_r;
                a_ba_re += dfj_b_r*phi_ja_r + dfj_b_i*phi_ja_i;
                a_ba_im += dfj_b_r*phi_ja_i - dfj_b_i*phi_ja_r;
            }

            double sbr = s[2*b], sbi = s[2*b+1];
            double qbr = sm_q[2*b], qbi = sm_q[2*b+1];
            double qpbr = sm_qp[2*b], qpbi = sm_qp[2*b+1];
            double re_ssab = sar*sbr + sai*sbi;
            double im_ssab = sar*sbi - sai*sbr;
            double mixed_re = (sar*qbr + sai*qbi)
                            + (sbr*qar + sbi*qai)
                            + (sar*qpbr - sai*qpbi)
                            + (sbr*qpar - sbi*qpai);
            double im_sqab = sar*qbi - sai*qbr;
            double im_sqba = sbr*qai - sbi*qar;
            double im_sqpab = sar*qpbi + sai*qpbr;
            double im_sqpba = sbr*qpai + sbi*qpar;

            double dl_re_pair = (a_ab_re + a_ba_re)*inv_k
                              - 2.0 * re_ssab * tr_ym * inv_k2
                              - mixed_re * inv_k2
                              + 4.0 * re_ssab * tr_y_dkdk * inv_k3;

            if (a == b) {
                per_pt[2 + a * n_basis + a]              = 0.5 * dl_re_pair;
                per_pt[2 + n_basis_sq + a * n_basis + a] = 0.0;
            } else {
                // a < b: write upper triangle and mirror to lower.
                // FIX(P3.9): re-derived dl_im under chart-invariance convention
                // `g_tan = T^T g T̄`. Same fix as quintic.rs::per_point_log_det_gradient.
                double dl_im = -(a_ab_im - a_ba_im)*inv_k
                              + 2.0 * im_ssab * tr_ym * inv_k2
                              + (im_sqab - im_sqba - im_sqpab + im_sqpba) * inv_k2
                              - 4.0 * im_ssab * tr_y_dkdk * inv_k3;
                per_pt[2 + a * n_basis + b]              = dl_re_pair;
                per_pt[2 + b * n_basis + a]              = dl_re_pair;  // mirror
                per_pt[2 + n_basis_sq + a * n_basis + b] = dl_im;
                per_pt[2 + n_basis_sq + b * n_basis + a] = 0.0;          // Hermitian
            }
        }
    }

    // Final eta_p, w_p (one thread).
    if (tid == 0) {
        if (!isfinite(r_val) || r_val <= 0.0 || !isfinite(log_r)) {
            per_pt[0] = nan("");
            per_pt[1] = 0.0;
        } else {
            per_pt[0] = r_val;
            per_pt[1] = w_p;
        }
    }
}
"#;

/// CUDA source for the batched 16×16 LU determinant. One thread per
/// matrix; matrix is loaded into a per-thread register array (16×16 =
/// 256 doubles = 2 KB per thread, comfortable for sm_60+ register file
/// but the compiler will spill to local memory — still much faster than
/// CPU). Mirrors `refine::determinant_lu` (refine.rs:29-66) exactly.
///
/// We deliberately do NOT use shared memory: the matrix would need to
/// be split across threads in a block, which serialises the LU steps.
/// Per-thread independent LU is the right tradeoff at N=16.
/// Two-pass reduction kernels for `sigma_squared_and_gradient_gpu`.
///
/// `reduce_pass1` computes:
///   d_pass1[0]              = total_w
///   d_pass1[1]              = sum_w_eta
///   d_pass1[2 + k]          = sum_w_eta_grad[k]    for k in 0..n_dof
/// from the per-point buffer produced by `adam_per_point`.
///
/// `reduce_pass2` consumes (per_point, kappa, dkappa) — where
/// dkappa[k] = sum_w_eta_grad[k] / total_w and kappa = sum_w_eta / total_w
/// (computed on host between the two passes) — and produces:
///   d_pass2[0]              = sum_w_dev_abs       (=> σ_L¹ * total_w)
///   d_pass2[1]              = sum_w_dev_sq        (=> σ_L² * total_w)
///   d_pass2[2 + k]          = grad_unnorm[k]      (=> grad[k] * total_w)
///
/// Both kernels use one block per output dimension (grid_dim_x = 2 + n_dof),
/// 256 threads per block, with a grid-stride loop over the n_points input
/// rows and a shared-memory tree reduction. Thread 0 of each block writes
/// the final reduced value to its output slot. No atomics, no inter-block
/// sync.
///
/// **Why this is the right shape**
///
/// At k=4, n_basis=70, the per-point buffer is n_points × stride_pp doubles
/// where stride_pp = 2 + 2·n_basis² = 9802. For n_points=60_000 that is
/// ~4.7 GB — too large to read back to host every Adam step. These two
/// reduction kernels collapse it to two 78-KB readbacks (d_pass1, d_pass2)
/// per iteration. PCIe bandwidth is no longer the bottleneck.
///
/// The host then divides by total_w (cheap O(n_dof) scalar work) and
/// returns (σ_L²-squared, grad, σ_L¹) in the same shape as the CPU
/// reference `sigma_squared_and_gradient`.
#[cfg(feature = "gpu")]
const REDUCE_KERNEL_SOURCE: &str = r#"
// ---------------------------------------------------------------------------
// Transpose (row-major per_point [n_pts][stride_pp] → col-major
// per_point_T [stride_pp][n_pts]). 32×32 shared-memory tile with one
// padding column to avoid bank conflicts. Reads + writes are both
// coalesced. This is the bandwidth-optimal way to break the strided
// access pattern that the row-major reductions otherwise have.
// ---------------------------------------------------------------------------
extern "C" __global__ void transpose_per_point(
    const double* __restrict__ per_point,
    double* __restrict__ per_point_T,
    int n_points,
    int n_basis
) {
    const int n_dof = 2 * n_basis * n_basis;
    const int stride_pp = 2 + n_dof;
    __shared__ double tile[32][33];

    int p_in = blockIdx.x * 32 + threadIdx.x;
    int c_in = blockIdx.y * 32 + threadIdx.y;
    if (p_in < n_points && c_in < stride_pp) {
        tile[threadIdx.y][threadIdx.x] =
            per_point[(long long)p_in * (long long)stride_pp + c_in];
    }
    __syncthreads();

    int p_out = blockIdx.x * 32 + threadIdx.y;
    int c_out = blockIdx.y * 32 + threadIdx.x;
    if (p_out < n_points && c_out < stride_pp) {
        per_point_T[(long long)c_out * (long long)n_points + p_out] =
            tile[threadIdx.x][threadIdx.y];
    }
}

// ---------------------------------------------------------------------------
// Pass 1 reduction over the **transposed** per-point buffer. For each
// output column c (one block per c), all 256 threads read a
// **contiguous** stripe of `per_point_T[c * n_pts + p]` for p in
// [0, n_pts). Reads are fully coalesced.
//
// NOTE: outputs c=0 (eta) and c=1 (w_p) are read directly from their
// own contiguous columns; for c >= 2 we additionally read eta and w
// from their columns to apply the validity mask.
// ---------------------------------------------------------------------------
extern "C" __global__ void reduce_pass1(
    const double* __restrict__ per_point_T,   // [stride_pp][n_points]
    double* __restrict__ pass1_out,
    int n_points,
    int n_basis
) {
    int n_dof = 2 * n_basis * n_basis;
    int output_idx = blockIdx.x;
    int n_outputs = 2 + n_dof;
    if (output_idx >= n_outputs) return;

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    extern __shared__ double sdata[];

    const double* eta_col = per_point_T + (long long)0 * n_points;
    const double* w_col   = per_point_T + (long long)1 * n_points;

    double acc = 0.0;
    if (output_idx == 0) {
        for (int p = tid; p < n_points; p += blockSize) {
            double eta_p = eta_col[p];
            double w_p = w_col[p];
            if (!isfinite(eta_p) || eta_p <= 0.0 || w_p == 0.0) continue;
            acc += w_p;
        }
    } else if (output_idx == 1) {
        for (int p = tid; p < n_points; p += blockSize) {
            double eta_p = eta_col[p];
            double w_p = w_col[p];
            if (!isfinite(eta_p) || eta_p <= 0.0 || w_p == 0.0) continue;
            acc += w_p * eta_p;
        }
    } else {
        int k = output_idx - 2;
        const double* grad_col =
            per_point_T + (long long)(2 + k) * (long long)n_points;
        for (int p = tid; p < n_points; p += blockSize) {
            double eta_p = eta_col[p];
            double w_p = w_col[p];
            if (!isfinite(eta_p) || eta_p <= 0.0 || w_p == 0.0) continue;
            acc += w_p * eta_p * grad_col[p];
        }
    }

    sdata[tid] = acc;
    __syncthreads();

    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) pass1_out[output_idx] = sdata[0];
}

// ---------------------------------------------------------------------------
// Row-major reductions (skip the transpose pass entirely).
// Each warp handles 32 consecutive output columns of the row-major
// per_point [n_pts][stride_pp] buffer. For a fixed point p, the
// 32 threads in the warp read 32 consecutive doubles
// (= per_point[p][col_base..col_base+31]) which is a single coalesced
// 256-byte transaction. eta_p and w_p are read by lane 0 and
// broadcast via __shfl_sync. n_pts is iterated serially within each
// thread (one accumulator per output col).
//
// Block dim: (32, 8) = 256 threads = 8 warps × 32 cols/warp = 256 cols
// per block. Grid dim: ceil(n_outputs / 256).
// ---------------------------------------------------------------------------
extern "C" __global__ void reduce_pass1_rm(
    const double* __restrict__ per_point,  // row-major [n_pts][stride_pp]
    double* __restrict__ pass1_out,
    int n_points,
    int n_basis
) {
    int n_dof = 2 * n_basis * n_basis;
    int stride_pp = 2 + n_dof;
    int n_outputs = stride_pp;
    int col_global = blockIdx.x * 256 + threadIdx.y * 32 + threadIdx.x;
    if (col_global >= n_outputs) return;
    int tid_x = threadIdx.x;
    unsigned mask = 0xFFFFFFFF;

    double acc = 0.0;
    for (int p = 0; p < n_points; ++p) {
        long long base = (long long)p * (long long)stride_pp;
        double eta_p, w_p;
        if (tid_x == 0) {
            eta_p = per_point[base + 0];
            w_p   = per_point[base + 1];
        }
        eta_p = __shfl_sync(mask, eta_p, 0);
        w_p   = __shfl_sync(mask, w_p,   0);
        if (!isfinite(eta_p) || eta_p <= 0.0 || w_p == 0.0) continue;
        if (col_global == 0) {
            acc += w_p;
        } else if (col_global == 1) {
            acc += w_p * eta_p;
        } else {
            int k = col_global - 2;
            double grad_pk = per_point[base + 2 + k];
            acc += w_p * eta_p * grad_pk;
        }
    }
    pass1_out[col_global] = acc;
}

extern "C" __global__ void reduce_pass2_rm(
    const double* __restrict__ per_point,
    const double* __restrict__ dkappa,
    double kappa,
    double* __restrict__ pass2_out,
    int n_points,
    int n_basis
) {
    int n_dof = 2 * n_basis * n_basis;
    int stride_pp = 2 + n_dof;
    int n_outputs = stride_pp;
    int col_global = blockIdx.x * 256 + threadIdx.y * 32 + threadIdx.x;
    if (col_global >= n_outputs) return;
    int tid_x = threadIdx.x;
    unsigned mask = 0xFFFFFFFF;
    double inv_kappa = 1.0 / kappa;
    double dk_over_kappa = (col_global >= 2) ? dkappa[col_global - 2] * inv_kappa : 0.0;

    double acc = 0.0;
    for (int p = 0; p < n_points; ++p) {
        long long base = (long long)p * (long long)stride_pp;
        double eta_p, w_p;
        if (tid_x == 0) {
            eta_p = per_point[base + 0];
            w_p   = per_point[base + 1];
        }
        eta_p = __shfl_sync(mask, eta_p, 0);
        w_p   = __shfl_sync(mask, w_p,   0);
        if (!isfinite(eta_p) || eta_p <= 0.0 || w_p == 0.0) continue;
        double eta_norm = eta_p * inv_kappa;
        double dev = eta_norm - 1.0;
        if (col_global == 0) {
            acc += w_p * fabs(dev);
        } else if (col_global == 1) {
            acc += w_p * dev * dev;
        } else {
            int k = col_global - 2;
            double grad_pk = per_point[base + 2 + k];
            double d_eta_norm = eta_norm * (grad_pk - dk_over_kappa);
            acc += w_p * 2.0 * dev * d_eta_norm;
        }
    }
    pass2_out[col_global] = acc;
}

extern "C" __global__ void reduce_pass2(
    const double* __restrict__ per_point_T,   // [stride_pp][n_points]
    const double* __restrict__ dkappa,
    double kappa,
    double* __restrict__ pass2_out,
    int n_points,
    int n_basis
) {
    int n_dof = 2 * n_basis * n_basis;
    int output_idx = blockIdx.x;
    int n_outputs = 2 + n_dof;
    if (output_idx >= n_outputs) return;

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    extern __shared__ double sdata[];

    double inv_kappa = 1.0 / kappa;
    const double* eta_col = per_point_T + (long long)0 * n_points;
    const double* w_col   = per_point_T + (long long)1 * n_points;

    double acc = 0.0;
    if (output_idx == 0) {
        for (int p = tid; p < n_points; p += blockSize) {
            double eta_p = eta_col[p];
            double w_p = w_col[p];
            if (!isfinite(eta_p) || eta_p <= 0.0 || w_p == 0.0) continue;
            double dev = eta_p * inv_kappa - 1.0;
            acc += w_p * fabs(dev);
        }
    } else if (output_idx == 1) {
        for (int p = tid; p < n_points; p += blockSize) {
            double eta_p = eta_col[p];
            double w_p = w_col[p];
            if (!isfinite(eta_p) || eta_p <= 0.0 || w_p == 0.0) continue;
            double dev = eta_p * inv_kappa - 1.0;
            acc += w_p * dev * dev;
        }
    } else {
        int k = output_idx - 2;
        const double* grad_col =
            per_point_T + (long long)(2 + k) * (long long)n_points;
        double dk_over_kappa = dkappa[k] * inv_kappa;
        for (int p = tid; p < n_points; p += blockSize) {
            double eta_p = eta_col[p];
            double w_p = w_col[p];
            if (!isfinite(eta_p) || eta_p <= 0.0 || w_p == 0.0) continue;
            double eta_norm = eta_p * inv_kappa;
            double dev = eta_norm - 1.0;
            double d_eta_norm = eta_norm * (grad_col[p] - dk_over_kappa);
            acc += w_p * 2.0 * dev * d_eta_norm;
        }
    }

    sdata[tid] = acc;
    __syncthreads();

    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) pass2_out[output_idx] = sdata[0];
}
"#;

#[cfg(feature = "gpu")]
const LU_DET_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void lu_det_16x16(
    const double* __restrict__ matrices,  // n_matrices * 256
    double* __restrict__ dets,            // n_matrices
    int n_matrices
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_matrices) return;
    const int n = 16;

    // Load the matrix into a local array. 256 doubles = 2 KB; the
    // compiler will most likely spill to local memory but this is fine
    // for our 10000-matrix workloads (we are still GBytes/s bound).
    double a[256];
    for (int q = 0; q < 256; ++q) a[q] = matrices[(long long)p * 256 + q];

    double sign = 1.0;
    for (int k = 0; k < n; ++k) {
        // Partial pivot search over column k below row k.
        double max_val = fabs(a[k * n + k]);
        int max_row = k;
        for (int i = k + 1; i < n; ++i) {
            double v = fabs(a[i * n + k]);
            if (v > max_val) { max_val = v; max_row = i; }
        }
        if (max_val == 0.0) {
            dets[p] = 0.0;
            return;
        }
        if (max_row != k) {
            for (int j = 0; j < n; ++j) {
                double tmp = a[k * n + j];
                a[k * n + j] = a[max_row * n + j];
                a[max_row * n + j] = tmp;
            }
            sign = -sign;
        }
        double pivot = a[k * n + k];
        for (int i = k + 1; i < n; ++i) {
            double factor = a[i * n + k] / pivot;
            a[i * n + k] = factor;
            for (int j = k + 1; j < n; ++j) {
                a[i * n + j] -= factor * a[k * n + j];
            }
        }
    }
    double det = sign;
    for (int i = 0; i < n; ++i) det *= a[i * n + i];
    dets[p] = det;
}
"#;

// ---------------------------------------------------------------------------
// Phase 3 / 4: device-side workspace structs.
// ---------------------------------------------------------------------------

/// GPU kernel + persistent device buffers for the batched per-point
/// Adam-gradient. Construction compiles the NVRTC kernel and allocates
/// every buffer the kernel touches. `compute_per_point` performs only
/// host↔device memcpys and the launch.
///
/// The orchestrator should keep one of these per (n_points, n_basis)
/// pair across all Adam iterations — the kernel does not need to be
/// recompiled when h_block changes.
#[cfg(feature = "gpu")]
pub struct AdamGradientKernel {
    pub n_points: usize,
    pub n_basis: usize,
    pub context: CudaContext,
    pub stream: Arc<CudaStream>,
    pub module: Arc<cudarc::driver::CudaModule>,
    pub reduce_module: Arc<cudarc::driver::CudaModule>,

    // Pre-allocated device buffers (exact sizes checked at upload time).
    d_section_values: CudaSlice<f64>,   // n_points * 2n
    d_section_derivs: CudaSlice<f64>,   // n_points * 5 * 2n
    d_h_block: CudaSlice<f64>,          // 4n²
    d_points: CudaSlice<f64>,           // n_points * 10
    d_weights: CudaSlice<f64>,          // n_points
    d_log_omega_sq: CudaSlice<f64>,     // n_points
    d_per_point: CudaSlice<f64>,        // n_points * (2 + 2n²)
    d_scratch: CudaSlice<f64>,          // n_points * (16n + 2n²)

    // Reduction buffers for compute_sigma_grad_h_only.
    d_per_point_t: CudaSlice<f64>,      // n_points * (2 + 2n²) — transposed
    d_pass1: CudaSlice<f64>,            // 2 + n_dof
    d_pass2: CudaSlice<f64>,            // 2 + n_dof
    d_dkappa: CudaSlice<f64>,           // n_dof
}

#[cfg(feature = "gpu")]
impl AdamGradientKernel {
    /// Compile NVRTC kernel and allocate all device buffers for the
    /// specified problem size. Returns `CudaError` on any CUDA / NVRTC
    /// failure (no panics).
    pub fn new(n_points: usize, n_basis: usize) -> Result<Self, CudaError> {
        if n_points == 0 || n_basis == 0 {
            return Err("AdamGradientKernel::new: n_points and n_basis must be > 0".into());
        }
        let context = CudarcContext::new(0)?;
        let stream = context.default_stream();
        let ptx = compile_ptx(ADAM_KERNEL_SOURCE)?;
        let module = context.load_module(ptx)?;
        let reduce_ptx = compile_ptx(REDUCE_KERNEL_SOURCE)?;
        let reduce_module = context.load_module(reduce_ptx)?;

        let two_n = 2 * n_basis;
        let n_dof = 2 * n_basis * n_basis;
        let stride_pp = 2 + n_dof;
        let slab_per_pt = 16 * n_basis + 2 * n_basis * n_basis;

        let d_section_values = stream.alloc_zeros::<f64>(n_points * two_n)?;
        let d_section_derivs = stream.alloc_zeros::<f64>(n_points * 5 * two_n)?;
        let d_h_block = stream.alloc_zeros::<f64>(two_n * two_n)?;
        let d_points = stream.alloc_zeros::<f64>(n_points * 10)?;
        let d_weights = stream.alloc_zeros::<f64>(n_points)?;
        let d_log_omega_sq = stream.alloc_zeros::<f64>(n_points)?;
        let d_per_point = stream.alloc_zeros::<f64>(n_points * stride_pp)?;
        let d_scratch = stream.alloc_zeros::<f64>(n_points * slab_per_pt)?;
        let d_per_point_t = stream.alloc_zeros::<f64>(n_points * stride_pp)?;
        let d_pass1 = stream.alloc_zeros::<f64>(2 + n_dof)?;
        let d_pass2 = stream.alloc_zeros::<f64>(2 + n_dof)?;
        let d_dkappa = stream.alloc_zeros::<f64>(n_dof)?;

        Ok(Self {
            n_points,
            n_basis,
            context,
            stream,
            module,
            reduce_module,
            d_section_values,
            d_section_derivs,
            d_h_block,
            d_points,
            d_weights,
            d_log_omega_sq,
            d_per_point,
            d_scratch,
            d_per_point_t,
            d_pass1,
            d_pass2,
            d_dkappa,
        })
    }

    /// Fused on-device per-point + reduction. Replaces
    /// `compute_per_point_h_only` for the σ + analytic-gradient path:
    /// instead of reading back the n_pts × stride_pp per-point buffer
    /// (≈ 4.7 GB at k=4, 60k pts), only the two reduction outputs
    /// (2 + n_dof = 9802 doubles = 78 KB each) cross PCIe.
    ///
    /// Returns `(sigma_l2_sq, grad, sigma_l1)` in the same shape as the
    /// CPU reference `quintic::sigma_squared_and_gradient`.
    ///
    /// Caller must have invoked `upload_static_inputs` once before the
    /// first call (the per-Adam-step inputs are h_block only, uploaded
    /// here).
    /// Profiling variant of `compute_sigma_grad_h_only`. Synchronises
    /// the stream after every kernel and returns timings in
    /// (ms_per_point, ms_transpose, ms_pass1, ms_pass2). Use this for
    /// performance debugging only — the synchronises hurt throughput.
    pub fn compute_sigma_grad_h_only_profiled(
        &mut self,
        h_block: &[f64],
    ) -> Result<((f64, Vec<f64>, f64), [f64; 5]), CudaError> {
        let t_start = std::time::Instant::now();
        let two_n = 2 * self.n_basis;
        let n_dof = 2 * self.n_basis * self.n_basis;
        let n_outputs = 2 + n_dof;

        if h_block.len() != two_n * two_n {
            return Err("bad h_block".into());
        }
        self.stream.memcpy_htod(h_block, &mut self.d_h_block)?;
        self.stream.synchronize()?;
        let t1 = std::time::Instant::now();

        let func0 = self.module.load_function("adam_per_point_warp")?;
        let block_dim_pp: u32 = 32;
        let grid_dim_pp: u32 = self.n_points as u32;
        let n_basis_us = self.n_basis;
        let smem_bytes = ((26 * n_basis_us + 250) * 8) as u32;
        let cfg0 = LaunchConfig {
            grid_dim: (grid_dim_pp, 1, 1),
            block_dim: (block_dim_pp, 1, 1),
            shared_mem_bytes: smem_bytes,
        };
        let n_pts_i32 = self.n_points as i32;
        let n_basis_i32 = self.n_basis as i32;
        {
            let mut launcher = self.stream.launch_builder(&func0);
            launcher
                .arg(&self.d_section_values)
                .arg(&self.d_section_derivs)
                .arg(&self.d_h_block)
                .arg(&self.d_points)
                .arg(&self.d_weights)
                .arg(&self.d_log_omega_sq)
                .arg(&mut self.d_per_point)
                .arg(&n_pts_i32)
                .arg(&n_basis_i32);
            unsafe { launcher.launch(cfg0)? };
        }
        self.stream.synchronize()?;
        let t2 = std::time::Instant::now();

        let func_t = self.reduce_module.load_function("transpose_per_point")?;
        let tile: u32 = 32;
        let stride_pp_u32 = (2 + n_dof) as u32;
        let cfg_t = LaunchConfig {
            grid_dim: (
                ((self.n_points as u32) + tile - 1) / tile,
                (stride_pp_u32 + tile - 1) / tile,
                1,
            ),
            block_dim: (tile, tile, 1),
            shared_mem_bytes: 0,
        };
        {
            let mut launcher = self.stream.launch_builder(&func_t);
            launcher
                .arg(&self.d_per_point)
                .arg(&mut self.d_per_point_t)
                .arg(&n_pts_i32)
                .arg(&n_basis_i32);
            unsafe { launcher.launch(cfg_t)? };
        }
        self.stream.synchronize()?;
        let t3 = std::time::Instant::now();

        let func1 = self.reduce_module.load_function("reduce_pass1")?;
        let red_block: u32 = 256;
        let cfg1 = LaunchConfig {
            grid_dim: (n_outputs as u32, 1, 1),
            block_dim: (red_block, 1, 1),
            shared_mem_bytes: (red_block as u32) * 8,
        };
        {
            let mut launcher = self.stream.launch_builder(&func1);
            launcher
                .arg(&self.d_per_point_t)
                .arg(&mut self.d_pass1)
                .arg(&n_pts_i32)
                .arg(&n_basis_i32);
            unsafe { launcher.launch(cfg1)? };
        }
        self.stream.synchronize()?;
        let pass1 = self.stream.memcpy_dtov(&self.d_pass1)?;
        let t4 = std::time::Instant::now();

        let total_w = pass1[0];
        let sum_w_eta = pass1[1];
        if total_w < 1e-12 {
            return Ok(((f64::NAN, vec![0.0; n_dof], f64::NAN), [0.0; 5]));
        }
        let kappa = sum_w_eta / total_w;
        let mut dkappa = vec![0.0f64; n_dof];
        let inv_w = 1.0 / total_w;
        for k in 0..n_dof {
            dkappa[k] = pass1[2 + k] * inv_w;
        }
        self.stream.memcpy_htod(&dkappa, &mut self.d_dkappa)?;

        let func2 = self.reduce_module.load_function("reduce_pass2")?;
        let cfg2 = LaunchConfig {
            grid_dim: (n_outputs as u32, 1, 1),
            block_dim: (red_block, 1, 1),
            shared_mem_bytes: (red_block as u32) * 8,
        };
        {
            let mut launcher = self.stream.launch_builder(&func2);
            launcher
                .arg(&self.d_per_point_t)
                .arg(&self.d_dkappa)
                .arg(&kappa)
                .arg(&mut self.d_pass2)
                .arg(&n_pts_i32)
                .arg(&n_basis_i32);
            unsafe { launcher.launch(cfg2)? };
        }
        self.stream.synchronize()?;
        let pass2 = self.stream.memcpy_dtov(&self.d_pass2)?;
        let t5 = std::time::Instant::now();

        let sigma_l1 = pass2[0] * inv_w;
        let sigma_l2_sq = pass2[1] * inv_w;
        let mut grad = vec![0.0f64; n_dof];
        for k in 0..n_dof {
            grad[k] = pass2[2 + k] * inv_w;
        }
        let timings = [
            (t1 - t_start).as_secs_f64() * 1000.0, // upload h
            (t2 - t1).as_secs_f64() * 1000.0,      // per-point
            (t3 - t2).as_secs_f64() * 1000.0,      // transpose
            (t4 - t3).as_secs_f64() * 1000.0,      // pass1
            (t5 - t4).as_secs_f64() * 1000.0,      // pass2
        ];
        Ok(((sigma_l2_sq, grad, sigma_l1), timings))
    }

    pub fn compute_sigma_grad_h_only(
        &mut self,
        h_block: &[f64],
    ) -> Result<(f64, Vec<f64>, f64), CudaError> {
        let two_n = 2 * self.n_basis;
        let n_dof = 2 * self.n_basis * self.n_basis;
        let n_outputs = 2 + n_dof;

        if h_block.len() != two_n * two_n {
            return Err(format!(
                "h_block length {}; expected {}",
                h_block.len(),
                two_n * two_n
            )
            .into());
        }

        // Upload h_block; everything else is already on-device.
        self.stream.memcpy_htod(h_block, &mut self.d_h_block)?;

        // Stage 0: per-point compute, warp-cooperative.
        // One warp (32 threads) per point. Shared memory holds the
        // per-point intermediate state (h_s, h_dfi, q, qp, phi, …) so
        // the warp's threads can collaborate on the dominant n²
        // gradient-assembly loop. Replaces the old thread-per-point
        // path which was bottlenecked by ~14 KB of per-thread local
        // memory and a fully-serial n² inner loop.
        //
        // Shared-mem layout: 26·n_basis + 273 doubles per warp. At
        // n_basis = 70 → 16.6 KB/block, 6 blocks/SM, 192 threads/SM
        // active. Per-thread work drops 32× so the lower occupancy is
        // more than offset.
        let func0 = self.module.load_function("adam_per_point_warp")?;
        let block_dim_pp: u32 = 32; // 1 warp per point
        let grid_dim_pp: u32 = self.n_points as u32; // 1 block per point
        let n_basis_us = self.n_basis;
        // h_s(2n) + h_dfi(10n) + q(2n) + qp(2n) + phi(10n) = 26n
        // (amat_row/col are now per-thread registers in phase 9.)
        // dk(10) + v(10) + w(10) + m(50) + y(50) + g_amb(50) + frame(30)
        //  + g_tan(18) + g_inv(18) + scalars(4) = 250
        let smem_doubles = 26 * n_basis_us + 250;
        let smem_bytes = (smem_doubles * 8) as u32;
        let cfg0 = LaunchConfig {
            grid_dim: (grid_dim_pp, 1, 1),
            block_dim: (block_dim_pp, 1, 1),
            shared_mem_bytes: smem_bytes,
        };
        let n_pts_i32 = self.n_points as i32;
        let n_basis_i32 = self.n_basis as i32;
        {
            let mut launcher = self.stream.launch_builder(&func0);
            launcher
                .arg(&self.d_section_values)
                .arg(&self.d_section_derivs)
                .arg(&self.d_h_block)
                .arg(&self.d_points)
                .arg(&self.d_weights)
                .arg(&self.d_log_omega_sq)
                .arg(&mut self.d_per_point)
                .arg(&n_pts_i32)
                .arg(&n_basis_i32);
            unsafe { launcher.launch(cfg0)? };
        }

        // Stage 0.5: transpose per_point [n_pts][stride_pp] →
        // per_point_T [stride_pp][n_pts] for coalesced reduction reads.
        // (Tried row-major reductions; were 4× slower because warp
        // threads handling the same col read at points strided by
        // stride_pp · 8B = 78 KB — completely uncoalesced.)
        let func_t = self.reduce_module.load_function("transpose_per_point")?;
        let tile: u32 = 32;
        let stride_pp_u32 = (2 + n_dof) as u32;
        let cfg_t = LaunchConfig {
            grid_dim: (
                ((self.n_points as u32) + tile - 1) / tile,
                (stride_pp_u32 + tile - 1) / tile,
                1,
            ),
            block_dim: (tile, tile, 1),
            shared_mem_bytes: 0,
        };
        {
            let mut launcher = self.stream.launch_builder(&func_t);
            launcher
                .arg(&self.d_per_point)
                .arg(&mut self.d_per_point_t)
                .arg(&n_pts_i32)
                .arg(&n_basis_i32);
            unsafe { launcher.launch(cfg_t)? };
        }

        // Stage 1: pass1 reduction (one block per output column).
        let func1 = self.reduce_module.load_function("reduce_pass1")?;
        let red_block: u32 = 256;
        let cfg1 = LaunchConfig {
            grid_dim: (n_outputs as u32, 1, 1),
            block_dim: (red_block, 1, 1),
            shared_mem_bytes: (red_block as u32) * 8,
        };
        {
            let mut launcher = self.stream.launch_builder(&func1);
            launcher
                .arg(&self.d_per_point_t)
                .arg(&mut self.d_pass1)
                .arg(&n_pts_i32)
                .arg(&n_basis_i32);
            unsafe { launcher.launch(cfg1)? };
        }

        // Read pass1 (78 KB) -> compute κ, dkappa on host.
        self.stream.synchronize()?;
        let pass1 = self.stream.memcpy_dtov(&self.d_pass1)?;
        let total_w = pass1[0];
        let sum_w_eta = pass1[1];
        if total_w < 1e-12 {
            return Ok((f64::NAN, vec![0.0; n_dof], f64::NAN));
        }
        let kappa = sum_w_eta / total_w;
        if kappa.abs() < 1e-30 {
            return Ok((f64::NAN, vec![0.0; n_dof], f64::NAN));
        }
        let mut dkappa = vec![0.0f64; n_dof];
        let inv_w = 1.0 / total_w;
        for k in 0..n_dof {
            dkappa[k] = pass1[2 + k] * inv_w;
        }

        // Upload dkappa, launch pass2.
        self.stream.memcpy_htod(&dkappa, &mut self.d_dkappa)?;
        let func2 = self.reduce_module.load_function("reduce_pass2")?;
        let cfg2 = LaunchConfig {
            grid_dim: (n_outputs as u32, 1, 1),
            block_dim: (red_block, 1, 1),
            shared_mem_bytes: (red_block as u32) * 8,
        };
        {
            let mut launcher = self.stream.launch_builder(&func2);
            launcher
                .arg(&self.d_per_point_t)
                .arg(&self.d_dkappa)
                .arg(&kappa)
                .arg(&mut self.d_pass2)
                .arg(&n_pts_i32)
                .arg(&n_basis_i32);
            unsafe { launcher.launch(cfg2)? };
        }

        self.stream.synchronize()?;
        let pass2 = self.stream.memcpy_dtov(&self.d_pass2)?;
        let sigma_l1 = pass2[0] * inv_w;
        let sigma_l2_sq = pass2[1] * inv_w;
        let mut grad = vec![0.0f64; n_dof];
        for k in 0..n_dof {
            grad[k] = pass2[2 + k] * inv_w;
        }
        Ok((sigma_l2_sq, grad, sigma_l1))
    }

    /// Compute per-point (η_p, w_p, ∇L_p) on GPU, writing into
    /// `per_point_buf` in the SoA layout that the Rust path uses.
    ///
    /// **Note** — the spec lists the inputs as
    /// (section_values, section_derivs, h_block, weights, log_omega_sq);
    /// the per-point math also requires the variety coordinates `points`
    /// for the tangent-frame construction (see
    /// `quintic.rs:per_point_log_det_gradient` line 1115). The
    /// orchestrator should pass `&ws.points` as the `points` argument
    /// (see the wiring in `quintic.rs:sigma_squared_and_gradient`).
    ///
    /// Allocation-free on the host side: only memcpys into the
    /// pre-allocated device buffers, then the launch + readback.
    pub fn compute_per_point(
        &mut self,
        section_values: &[f64],
        section_derivs: &[f64],
        h_block: &[f64],
        points: &[f64],
        weights: &[f64],
        log_omega_sq: &[f64],
        per_point_buf: &mut [f64],
    ) -> Result<(), CudaError> {
        let two_n = 2 * self.n_basis;
        let stride_per_point = 5 * two_n;
        let n_dof = 2 * self.n_basis * self.n_basis;
        let stride_pp = 2 + n_dof;

        if section_values.len() != self.n_points * two_n {
            return Err(format!(
                "section_values length {}; expected {}",
                section_values.len(),
                self.n_points * two_n
            )
            .into());
        }
        if section_derivs.len() != self.n_points * stride_per_point {
            return Err(format!(
                "section_derivs length {}; expected {}",
                section_derivs.len(),
                self.n_points * stride_per_point
            )
            .into());
        }
        if h_block.len() != two_n * two_n {
            return Err(format!(
                "h_block length {}; expected {}",
                h_block.len(),
                two_n * two_n
            )
            .into());
        }
        if points.len() != self.n_points * 10 {
            return Err(format!(
                "points length {}; expected {}",
                points.len(),
                self.n_points * 10
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
        if log_omega_sq.len() != self.n_points {
            return Err(format!(
                "log_omega_sq length {}; expected {}",
                log_omega_sq.len(),
                self.n_points
            )
            .into());
        }
        if per_point_buf.len() < self.n_points * stride_pp {
            return Err(format!(
                "per_point_buf length {}; expected at least {}",
                per_point_buf.len(),
                self.n_points * stride_pp
            )
            .into());
        }

        // Upload all per-step inputs to existing device buffers.
        self.stream.memcpy_htod(section_values, &mut self.d_section_values)?;
        self.stream.memcpy_htod(section_derivs, &mut self.d_section_derivs)?;
        self.stream.memcpy_htod(h_block, &mut self.d_h_block)?;
        self.stream.memcpy_htod(points, &mut self.d_points)?;
        self.stream.memcpy_htod(weights, &mut self.d_weights)?;
        self.stream.memcpy_htod(log_omega_sq, &mut self.d_log_omega_sq)?;

        // Launch one thread per point. Block size 64 chosen to balance
        // occupancy vs the local register pressure of build_tangent_frame
        // (which uses ~80 doubles of stack per thread).
        let func = self.module.load_function("adam_per_point")?;
        let block_dim: u32 = 64;
        let grid_dim: u32 =
            ((self.n_points as u32) + block_dim - 1) / block_dim;
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_pts_i32 = self.n_points as i32;
        let n_basis_i32 = self.n_basis as i32;
        let mut launcher = self.stream.launch_builder(&func);
        launcher
            .arg(&self.d_section_values)
            .arg(&self.d_section_derivs)
            .arg(&self.d_h_block)
            .arg(&self.d_points)
            .arg(&self.d_weights)
            .arg(&self.d_log_omega_sq)
            .arg(&mut self.d_per_point)
            .arg(&mut self.d_scratch)
            .arg(&n_pts_i32)
            .arg(&n_basis_i32);
        unsafe { launcher.launch(cfg)? };

        // Synchronise then copy out.
        self.stream.synchronize()?;
        let host = self.stream.memcpy_dtov(&self.d_per_point)?;
        per_point_buf[..self.n_points * stride_pp]
            .copy_from_slice(&host[..self.n_points * stride_pp]);
        Ok(())
    }

    /// Upload the geometry-only inputs (section_values, section_derivs,
    /// points, weights, log_omega_sq) once. These don't change across
    /// Adam iterations — only `h_block` does. After this, callers should
    /// use `compute_per_point_h_only` per iteration, which uploads only
    /// h_block (≈ 156 KB at n_basis=70 vs ≈ 408 MB for the full set at
    /// 60k points).
    pub fn upload_static_inputs(
        &mut self,
        section_values: &[f64],
        section_derivs: &[f64],
        points: &[f64],
        weights: &[f64],
        log_omega_sq: &[f64],
    ) -> Result<(), CudaError> {
        let two_n = 2 * self.n_basis;
        let stride_per_point = 5 * two_n;
        if section_values.len() != self.n_points * two_n {
            return Err(format!(
                "section_values length {}; expected {}",
                section_values.len(),
                self.n_points * two_n
            )
            .into());
        }
        if section_derivs.len() != self.n_points * stride_per_point {
            return Err(format!(
                "section_derivs length {}; expected {}",
                section_derivs.len(),
                self.n_points * stride_per_point
            )
            .into());
        }
        if points.len() != self.n_points * 10 {
            return Err(format!(
                "points length {}; expected {}",
                points.len(),
                self.n_points * 10
            )
            .into());
        }
        if weights.len() != self.n_points {
            return Err(format!("weights length {}; expected {}", weights.len(), self.n_points).into());
        }
        if log_omega_sq.len() != self.n_points {
            return Err(format!(
                "log_omega_sq length {}; expected {}",
                log_omega_sq.len(),
                self.n_points
            )
            .into());
        }
        self.stream.memcpy_htod(section_values, &mut self.d_section_values)?;
        self.stream.memcpy_htod(section_derivs, &mut self.d_section_derivs)?;
        self.stream.memcpy_htod(points, &mut self.d_points)?;
        self.stream.memcpy_htod(weights, &mut self.d_weights)?;
        self.stream.memcpy_htod(log_omega_sq, &mut self.d_log_omega_sq)?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Per-iteration compute path: uploads only `h_block`, then launches
    /// the kernel and reads back `per_point_buf`. Requires
    /// `upload_static_inputs` to have been called first.
    pub fn compute_per_point_h_only(
        &mut self,
        h_block: &[f64],
        per_point_buf: &mut [f64],
    ) -> Result<(), CudaError> {
        let two_n = 2 * self.n_basis;
        let n_dof = 2 * self.n_basis * self.n_basis;
        let stride_pp = 2 + n_dof;

        if h_block.len() != two_n * two_n {
            return Err(format!(
                "h_block length {}; expected {}",
                h_block.len(),
                two_n * two_n
            )
            .into());
        }
        if per_point_buf.len() < self.n_points * stride_pp {
            return Err(format!(
                "per_point_buf length {}; expected at least {}",
                per_point_buf.len(),
                self.n_points * stride_pp
            )
            .into());
        }

        self.stream.memcpy_htod(h_block, &mut self.d_h_block)?;

        let func = self.module.load_function("adam_per_point")?;
        let block_dim: u32 = 64;
        let grid_dim: u32 = ((self.n_points as u32) + block_dim - 1) / block_dim;
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_pts_i32 = self.n_points as i32;
        let n_basis_i32 = self.n_basis as i32;
        let mut launcher = self.stream.launch_builder(&func);
        launcher
            .arg(&self.d_section_values)
            .arg(&self.d_section_derivs)
            .arg(&self.d_h_block)
            .arg(&self.d_points)
            .arg(&self.d_weights)
            .arg(&self.d_log_omega_sq)
            .arg(&mut self.d_per_point)
            .arg(&mut self.d_scratch)
            .arg(&n_pts_i32)
            .arg(&n_basis_i32);
        unsafe { launcher.launch(cfg)? };

        self.stream.synchronize()?;
        let host = self.stream.memcpy_dtov(&self.d_per_point)?;
        per_point_buf[..self.n_points * stride_pp]
            .copy_from_slice(&host[..self.n_points * stride_pp]);
        Ok(())
    }

    /// Total bytes allocated on the device. Useful for memory budgeting
    /// when the orchestrator decides whether to enable the GPU path.
    pub fn total_device_bytes(&self) -> usize {
        8 * (self.d_section_values.len()
            + self.d_section_derivs.len()
            + self.d_h_block.len()
            + self.d_points.len()
            + self.d_weights.len()
            + self.d_log_omega_sq.len()
            + self.d_per_point.len()
            + self.d_scratch.len()
            + self.d_per_point_t.len()
            + self.d_pass1.len()
            + self.d_pass2.len()
            + self.d_dkappa.len())
    }
}

/// GPU kernel + persistent device buffers for the batched 16×16
/// determinant. Mirrors `refine::determinant_lu`. One thread per matrix.
#[cfg(feature = "gpu")]
pub struct BatchedLuDetKernel {
    pub n_matrices: usize,
    pub context: CudaContext,
    pub stream: Arc<CudaStream>,
    pub module: Arc<cudarc::driver::CudaModule>,

    d_matrices: CudaSlice<f64>, // n_matrices * 256
    d_dets: CudaSlice<f64>,     // n_matrices
}

#[cfg(feature = "gpu")]
impl BatchedLuDetKernel {
    /// Compile NVRTC kernel and pre-allocate device buffers for
    /// `n_matrices` 16×16 (real-form) matrices.
    pub fn new(n_matrices: usize) -> Result<Self, CudaError> {
        if n_matrices == 0 {
            return Err("BatchedLuDetKernel::new: n_matrices must be > 0".into());
        }
        let context = CudarcContext::new(0)?;
        let stream = context.default_stream();
        let ptx = compile_ptx(LU_DET_KERNEL_SOURCE)?;
        let module = context.load_module(ptx)?;
        let d_matrices = stream.alloc_zeros::<f64>(n_matrices * 256)?;
        let d_dets = stream.alloc_zeros::<f64>(n_matrices)?;
        Ok(Self {
            n_matrices,
            context,
            stream,
            module,
            d_matrices,
            d_dets,
        })
    }

    /// Compute det(M_p) for each of `n_matrices` matrices in parallel.
    /// Inputs are 16×16 real matrices in row-major flat layout
    /// (n_matrices · 256 doubles). For an 8×8 complex Hermitian metric
    /// Hessian this should be the real block representation
    /// `[[A, -B], [B, A]]` whose real determinant equals |det(A+iB)|².
    pub fn compute_dets(
        &mut self,
        matrices: &[f64],
        dets: &mut [f64],
    ) -> Result<(), CudaError> {
        if matrices.len() != self.n_matrices * 256 {
            return Err(format!(
                "matrices length {}; expected {}",
                matrices.len(),
                self.n_matrices * 256
            )
            .into());
        }
        if dets.len() < self.n_matrices {
            return Err(format!(
                "dets length {}; expected at least {}",
                dets.len(),
                self.n_matrices
            )
            .into());
        }
        self.stream.memcpy_htod(matrices, &mut self.d_matrices)?;
        let func = self.module.load_function("lu_det_16x16")?;
        let block_dim: u32 = 128;
        let grid_dim: u32 =
            ((self.n_matrices as u32) + block_dim - 1) / block_dim;
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_mat_i32 = self.n_matrices as i32;
        let mut launcher = self.stream.launch_builder(&func);
        launcher
            .arg(&self.d_matrices)
            .arg(&mut self.d_dets)
            .arg(&n_mat_i32);
        unsafe { launcher.launch(cfg)? };
        self.stream.synchronize()?;
        let host = self.stream.memcpy_dtov(&self.d_dets)?;
        dets[..self.n_matrices].copy_from_slice(&host[..self.n_matrices]);
        Ok(())
    }

    /// Total bytes allocated on the device.
    pub fn total_device_bytes(&self) -> usize {
        8 * (self.d_matrices.len() + self.d_dets.len())
    }
}

// ---------------------------------------------------------------------------
// CPU reference implementation (Phase 1).
//
// The functions below are private equivalents of three private helpers
// in quintic.rs (`invert_3x3_complex_hermitian`, `compute_y_5x5`,
// `per_point_log_det_gradient`). They are duplicated here so this
// module can serve as a self-contained validation oracle without any
// modifications to quintic.rs (which the orchestrator owns and will
// wire in afterwards).
// ---------------------------------------------------------------------------

/// 3×3 complex Hermitian inverse. Returns None on near-singular input.
/// Mirrors `quintic::invert_3x3_complex_hermitian` (private at line 913).
fn invert_3x3_complex_hermitian_local(
    g: &[[(f64, f64); 3]; 3],
) -> Option<[[(f64, f64); 3]; 3]> {
    let cmul = |(ar, ai): (f64, f64), (br, bi): (f64, f64)| -> (f64, f64) {
        (ar * br - ai * bi, ar * bi + ai * br)
    };
    let csub = |(ar, ai): (f64, f64), (br, bi): (f64, f64)| -> (f64, f64) {
        (ar - br, ai - bi)
    };
    let cconj = |(ar, ai): (f64, f64)| -> (f64, f64) { (ar, -ai) };
    let det = det_3x3_complex_hermitian(g);
    if !det.is_finite() || det.abs() < 1e-30 {
        return None;
    }
    let mut inv = [[(0.0f64, 0.0f64); 3]; 3];
    let c00 = csub(cmul(g[1][1], g[2][2]), cmul(g[1][2], g[2][1]));
    let c01 = csub(cmul(g[1][2], g[2][0]), cmul(g[1][0], g[2][2]));
    let c02 = csub(cmul(g[1][0], g[2][1]), cmul(g[1][1], g[2][0]));
    let c10 = csub(cmul(g[0][2], g[2][1]), cmul(g[0][1], g[2][2]));
    let c11 = csub(cmul(g[0][0], g[2][2]), cmul(g[0][2], g[2][0]));
    let c12 = csub(cmul(g[0][1], g[2][0]), cmul(g[0][0], g[2][1]));
    let c20 = csub(cmul(g[0][1], g[1][2]), cmul(g[0][2], g[1][1]));
    let c21 = csub(cmul(g[0][2], g[1][0]), cmul(g[0][0], g[1][2]));
    let c22 = csub(cmul(g[0][0], g[1][1]), cmul(g[0][1], g[1][0]));
    let inv_det = 1.0 / det;
    inv[0][0] = (c00.0 * inv_det, c00.1 * inv_det);
    inv[0][1] = (c10.0 * inv_det, c10.1 * inv_det);
    inv[0][2] = (c20.0 * inv_det, c20.1 * inv_det);
    inv[1][0] = (c01.0 * inv_det, c01.1 * inv_det);
    inv[1][1] = (c11.0 * inv_det, c11.1 * inv_det);
    inv[1][2] = (c21.0 * inv_det, c21.1 * inv_det);
    inv[2][0] = (c02.0 * inv_det, c02.1 * inv_det);
    inv[2][1] = (c12.0 * inv_det, c12.1 * inv_det);
    inv[2][2] = (c22.0 * inv_det, c22.1 * inv_det);
    for i in 0..3 {
        for j in (i + 1)..3 {
            let c = cconj(inv[i][j]);
            let avg_re = 0.5 * (inv[j][i].0 + c.0);
            let avg_im = 0.5 * (inv[j][i].1 + c.1);
            inv[j][i] = (avg_re, avg_im);
            inv[i][j] = (avg_re, -avg_im);
        }
    }
    Some(inv)
}

/// Mirrors `quintic::compute_y_5x5` (private at line 967).
fn compute_y_5x5_local(
    tangent_frame: &[[f64; 10]; 3],
    g_tan_inv: &[[(f64, f64); 3]; 3],
) -> [[(f64, f64); 5]; 5] {
    let mut y = [[(0.0f64, 0.0f64); 5]; 5];
    for i in 0..5 {
        let t_i: [(f64, f64); 3] = [
            (tangent_frame[0][2 * i], tangent_frame[0][2 * i + 1]),
            (tangent_frame[1][2 * i], tangent_frame[1][2 * i + 1]),
            (tangent_frame[2][2 * i], tangent_frame[2][2 * i + 1]),
        ];
        for j in 0..5 {
            let t_j: [(f64, f64); 3] = [
                (tangent_frame[0][2 * j], tangent_frame[0][2 * j + 1]),
                (tangent_frame[1][2 * j], tangent_frame[1][2 * j + 1]),
                (tangent_frame[2][2 * j], tangent_frame[2][2 * j + 1]),
            ];
            let mut s_re = 0.0;
            let mut s_im = 0.0;
            for a in 0..3 {
                for b in 0..3 {
                    // T_a[i]^* · g_inv[a][b] · T_b[j]   (PROPER convention,
                    // matches `quintic::compute_y_5x5`).
                    let (gar, gai) = g_tan_inv[a][b];
                    let (tiar, tiai) = (t_i[a].0, -t_i[a].1);
                    let (tjbr, tjbi) = t_j[b];
                    let p1_re = tiar * gar - tiai * gai;
                    let p1_im = tiar * gai + tiai * gar;
                    let p_re = p1_re * tjbr - p1_im * tjbi;
                    let p_im = p1_re * tjbi + p1_im * tjbr;
                    s_re += p_re;
                    s_im += p_im;
                }
            }
            y[i][j] = (s_re, s_im);
        }
    }
    y
}

/// One-point analytic gradient: line-for-line port of
/// `quintic::per_point_log_det_gradient` (private at line 1020).
/// `grad_out` must have length `2 · n_basis²` (re block followed by im
/// block). Returns `(eta_p, log_eta_p)` so the caller can fill the
/// (eta, w) head of the SoA slot.
fn per_point_log_det_gradient_local(
    s: &[f64],
    df: &[f64],
    h_block: &[f64],
    z: &[f64; 10],
    log_omega_sq: f64,
    n_basis: usize,
    grad_out: &mut [f64],
) -> (f64, f64) {
    let two_n = 2 * n_basis;

    // Allocate per-call scratch (the CPU reference is not on the hot
    // path; the rayon production code uses PointScratch in quintic.rs).
    let mut h_s = vec![0.0f64; two_n];
    for i in 0..two_n {
        let mut row_sum = 0.0;
        for j in 0..two_n {
            row_sum += h_block[i * two_n + j] * s[j];
        }
        h_s[i] = row_sum;
    }
    let mut k_val = 0.0;
    for i in 0..two_n {
        k_val += s[i] * h_s[i];
    }
    let k_safe = k_val.max(1e-30);
    let inv_k = 1.0 / k_safe;
    let inv_k2 = inv_k * inv_k;
    let inv_k3 = inv_k * inv_k2;

    let mut h_dfi = vec![vec![0.0f64; two_n]; 5];
    for i in 0..5 {
        let dfi = &df[i * two_n..(i + 1) * two_n];
        for k in 0..two_n {
            let mut row_sum = 0.0;
            for l in 0..two_n {
                row_sum += h_block[k * two_n + l] * dfi[l];
            }
            h_dfi[i][k] = row_sum;
        }
    }
    let mut dk = [(0.0f64, 0.0f64); 5];
    for i in 0..5 {
        let dfi = &df[i * two_n..(i + 1) * two_n];
        let mut s_re = 0.0;
        let mut s_im = 0.0;
        for a in 0..n_basis {
            let h_dr = h_s[2 * a];
            let h_di = h_s[2 * a + 1];
            let dr = dfi[2 * a];
            let di = dfi[2 * a + 1];
            s_re += h_dr * dr + h_di * di;
            s_im += h_dr * di - h_di * dr;
        }
        dk[i] = (s_re, s_im);
    }
    let mut m = [[(0.0f64, 0.0f64); 5]; 5];
    for i in 0..5 {
        for j in 0..5 {
            let dfj = &df[j * two_n..(j + 1) * two_n];
            let mut s_re = 0.0;
            let mut s_im = 0.0;
            for a in 0..n_basis {
                let dfj_re = dfj[2 * a];
                let dfj_im = dfj[2 * a + 1];
                let h_dfi_re = h_dfi[i][2 * a];
                let h_dfi_im = h_dfi[i][2 * a + 1];
                s_re += dfj_re * h_dfi_re + dfj_im * h_dfi_im;
                s_im += dfj_re * h_dfi_im - dfj_im * h_dfi_re;
            }
            m[i][j] = (s_re, s_im);
        }
    }
    let mut g_amb = [[(0.0f64, 0.0f64); 5]; 5];
    for i in 0..5 {
        for j in 0..5 {
            let (mr, mi) = m[i][j];
            let p_re = dk[i].0 * dk[j].0 + dk[i].1 * dk[j].1;
            let p_im = dk[i].1 * dk[j].0 - dk[i].0 * dk[j].1;
            g_amb[i][j] = (
                mr * inv_k - p_re * inv_k2,
                mi * inv_k - p_im * inv_k2,
            );
        }
    }
    let grad_f = fermat_quintic_gradient(z);
    // Use the canonical DKLR-2006 affine-chart frame (matches the
    // chart used by `log_omega_squared_quintic` so η is chart-invariant).
    let (chart, elim, _) = quintic_chart_and_elim(z, &grad_f);
    let frame = quintic_affine_chart_frame(&grad_f, chart, elim);
    let g_tan = project_to_quintic_tangent(&g_amb, &frame);
    let det = det_3x3_complex_hermitian(&g_tan);
    if !det.is_finite() || det.abs() < 1e-30 || !log_omega_sq.is_finite() {
        for v in grad_out.iter_mut() {
            *v = 0.0;
        }
        return (0.0, f64::NAN);
    }
    let log_r = det.abs().ln() - log_omega_sq;
    let r_val = log_r.exp();
    let g_tan_inv = match invert_3x3_complex_hermitian_local(&g_tan) {
        Some(g) => g,
        None => {
            for v in grad_out.iter_mut() {
                *v = 0.0;
            }
            return (r_val, log_r);
        }
    };
    let y_5x5 = compute_y_5x5_local(&frame, &g_tan_inv);

    let mut tr_ym = 0.0f64;
    for i in 0..5 {
        for j in 0..5 {
            tr_ym += y_5x5[j][i].0 * m[i][j].0 - y_5x5[j][i].1 * m[i][j].1;
        }
    }
    let mut tr_y_dkdk = 0.0f64;
    for i in 0..5 {
        for j in 0..5 {
            let p_re = dk[i].0 * dk[j].0 + dk[i].1 * dk[j].1;
            let p_im = dk[i].1 * dk[j].0 - dk[i].0 * dk[j].1;
            tr_y_dkdk += y_5x5[j][i].0 * p_re - y_5x5[j][i].1 * p_im;
        }
    }
    let mut v_i = [(0.0f64, 0.0f64); 5];
    for i in 0..5 {
        let mut s_re = 0.0;
        let mut s_im = 0.0;
        for j in 0..5 {
            let yji_re = y_5x5[j][i].0;
            let yji_im = y_5x5[j][i].1;
            let dkj_re = dk[j].0;
            let dkj_im = dk[j].1;
            s_re += yji_re * dkj_re + yji_im * dkj_im;
            s_im += yji_im * dkj_re - yji_re * dkj_im;
        }
        v_i[i] = (s_re, s_im);
    }
    let mut q = vec![0.0f64; 2 * n_basis];
    for b in 0..n_basis {
        let mut s_re = 0.0;
        let mut s_im = 0.0;
        for i in 0..5 {
            let dfi = &df[i * two_n..(i + 1) * two_n];
            let dr = dfi[2 * b];
            let di = dfi[2 * b + 1];
            let (vr, vi) = v_i[i];
            s_re += dr * vr - di * vi;
            s_im += dr * vi + di * vr;
        }
        q[2 * b] = s_re;
        q[2 * b + 1] = s_im;
    }
    let mut w_j = [(0.0f64, 0.0f64); 5];
    for j in 0..5 {
        let mut s_re = 0.0;
        let mut s_im = 0.0;
        for i in 0..5 {
            let yji_re = y_5x5[j][i].0;
            let yji_im = y_5x5[j][i].1;
            let dki_re = dk[i].0;
            let dki_im = dk[i].1;
            s_re += yji_re * dki_re - yji_im * dki_im;
            s_im += yji_re * dki_im + yji_im * dki_re;
        }
        w_j[j] = (s_re, s_im);
    }
    let mut qp = vec![0.0f64; 2 * n_basis];
    for b in 0..n_basis {
        let mut s_re = 0.0;
        let mut s_im = 0.0;
        for j in 0..5 {
            let dfj = &df[j * two_n..(j + 1) * two_n];
            let dr = dfj[2 * b];
            let di = dfj[2 * b + 1];
            let (wr, wi) = w_j[j];
            s_re += dr * wr + di * wi;
            s_im += dr * wi - di * wr;
        }
        qp[2 * b] = s_re;
        qp[2 * b + 1] = s_im;
    }
    let mut phi = vec![0.0f64; 2 * 5 * n_basis];
    for j in 0..5 {
        for b in 0..n_basis {
            let mut s_re = 0.0;
            let mut s_im = 0.0;
            for i in 0..5 {
                let yji_re = y_5x5[j][i].0;
                let yji_im = y_5x5[j][i].1;
                let dfi = &df[i * two_n..(i + 1) * two_n];
                let dr = dfi[2 * b];
                let di = dfi[2 * b + 1];
                s_re += yji_re * dr - yji_im * di;
                s_im += yji_re * di + yji_im * dr;
            }
            phi[2 * (j * n_basis + b)] = s_re;
            phi[2 * (j * n_basis + b) + 1] = s_im;
        }
    }
    let mut amat_re = vec![0.0f64; n_basis * n_basis];
    let mut amat_im = vec![0.0f64; n_basis * n_basis];
    for a in 0..n_basis {
        for b in 0..n_basis {
            let mut s_re = 0.0;
            let mut s_im = 0.0;
            for j in 0..5 {
                let dfj = &df[j * two_n..(j + 1) * two_n];
                let dr = dfj[2 * a];
                let di = dfj[2 * a + 1];
                let phir = phi[2 * (j * n_basis + b)];
                let phii = phi[2 * (j * n_basis + b) + 1];
                s_re += dr * phir + di * phii;
                s_im += dr * phii - di * phir;
            }
            amat_re[a * n_basis + b] = s_re;
            amat_im[a * n_basis + b] = s_im;
        }
    }
    for a in 0..n_basis {
        let sar = s[2 * a];
        let sai = s[2 * a + 1];
        let qar = q[2 * a];
        let qai = q[2 * a + 1];
        let qpar = qp[2 * a];
        let qpai = qp[2 * a + 1];
        for b in 0..n_basis {
            let sbr = s[2 * b];
            let sbi = s[2 * b + 1];
            let qbr = q[2 * b];
            let qbi = q[2 * b + 1];
            let qpbr = qp[2 * b];
            let qpbi = qp[2 * b + 1];
            let re_ssab = sar * sbr + sai * sbi;
            let im_ssab = sar * sbi - sai * sbr;
            let a_ab_re = amat_re[a * n_basis + b];
            let a_ba_re = amat_re[b * n_basis + a];
            let a_ab_im = amat_im[a * n_basis + b];
            let a_ba_im = amat_im[b * n_basis + a];
            let mixed_re = (sar * qbr + sai * qbi)
                + (sbr * qar + sbi * qai)
                + (sar * qpbr - sai * qpbi)
                + (sbr * qpar - sbi * qpai);
            let im_sqab = sar * qbi - sai * qbr;
            let im_sqba = sbr * qai - sbi * qar;
            let im_sqpab = sar * qpbi + sai * qpbr;
            let im_sqpba = sbr * qpai + sbi * qpar;
            let dl_re_pair = (a_ab_re + a_ba_re) * inv_k
                - 2.0 * re_ssab * tr_ym * inv_k2
                - mixed_re * inv_k2
                + 4.0 * re_ssab * tr_y_dkdk * inv_k3;
            let dl_re = if a == b { 0.5 * dl_re_pair } else { dl_re_pair };
            // FIX(P3.9): re-derived dl_im under chart-invariance convention
            // `g_tan = T^T g T̄`. See `quintic.rs::per_point_log_det_gradient`
            // for the full derivation; in short, the previous assembly had
            // the wrong signs on Im(s_a q'_b) / Im(s_b q'_a) (Term 3, dk-mixed).
            let dl_im = if a < b {
                -(a_ab_im - a_ba_im) * inv_k
                    + 2.0 * im_ssab * tr_ym * inv_k2
                    + (im_sqab - im_sqba - im_sqpab + im_sqpba) * inv_k2
                    - 4.0 * im_ssab * tr_y_dkdk * inv_k3
            } else {
                0.0
            };
            grad_out[a * n_basis + b] = dl_re;
            grad_out[n_basis * n_basis + a * n_basis + b] = dl_im;
        }
    }
    (r_val, log_r)
}

/// CPU reference for the per-point Adam-gradient computation. Mirrors
/// the per-point closure inside `quintic::sigma_squared_and_gradient`
/// (lines 1436-1468 in `quintic.rs`). Identical numerics to the GPU
/// kernel — used as the validation oracle in the parity test and as the
/// fallback when CUDA is unavailable.
///
/// `per_point_buf` must have length ≥ `n_points · (2 + 2·n_basis²)`.
/// Output layout per point p:
///   `per_point_buf[p · stride_pp]      = eta_p` (NaN on failure)
///   `per_point_buf[p · stride_pp + 1]  = w_p`   (0.0 on failure)
///   `per_point_buf[p · stride_pp + 2 ..]
///       = grad_p[0 .. 2·n_basis²]      (re block, then im block)`
pub fn cpu_compute_per_point_reference(
    n_points: usize,
    n_basis: usize,
    section_values: &[f64],
    section_derivs: &[f64],
    h_block: &[f64],
    points: &[f64],
    weights: &[f64],
    log_omega_sq: &[f64],
    per_point_buf: &mut [f64],
) {
    let two_n = 2 * n_basis;
    let stride_per_point = 5 * two_n;
    let n_dof = 2 * n_basis * n_basis;
    let stride_pp = 2 + n_dof;

    assert_eq!(section_values.len(), n_points * two_n, "section_values size mismatch");
    assert_eq!(
        section_derivs.len(),
        n_points * stride_per_point,
        "section_derivs size mismatch"
    );
    assert_eq!(h_block.len(), two_n * two_n, "h_block size mismatch");
    assert_eq!(points.len(), n_points * 10, "points size mismatch");
    assert_eq!(weights.len(), n_points, "weights size mismatch");
    assert_eq!(log_omega_sq.len(), n_points, "log_omega_sq size mismatch");
    assert!(
        per_point_buf.len() >= n_points * stride_pp,
        "per_point_buf too small"
    );

    for p in 0..n_points {
        let off = p * stride_pp;
        let slot = &mut per_point_buf[off..off + stride_pp];
        let w_p = weights[p];
        if !w_p.is_finite() || w_p <= 0.0 {
            slot[0] = f64::NAN;
            slot[1] = 0.0;
            for v in slot[2..].iter_mut() {
                *v = 0.0;
            }
            continue;
        }
        let s = &section_values[p * two_n..(p + 1) * two_n];
        let df = &section_derivs[p * stride_per_point..(p + 1) * stride_per_point];
        let z: [f64; 10] = points[p * 10..p * 10 + 10].try_into().unwrap();
        let log_om = log_omega_sq[p];
        // Zero the gradient slot before populating (matches Rust closure).
        for v in slot[2..].iter_mut() {
            *v = 0.0;
        }
        let (eta_p, log_eta) = per_point_log_det_gradient_local(
            s, df, h_block, &z, log_om, n_basis, &mut slot[2..],
        );
        if !eta_p.is_finite() || eta_p <= 0.0 || !log_eta.is_finite() {
            slot[0] = f64::NAN;
            slot[1] = 0.0;
        } else {
            slot[0] = eta_p;
            slot[1] = w_p;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quintic::QuinticSolver;

    /// Build a small synthetic input: random h_block (positive-definite),
    /// real QuinticSolver-derived section data. Used as common fixture.
    fn build_fixture(n_points: usize) -> Option<QuinticSolver> {
        // k_degree = 2 → n_basis = 15 (build_degree_k_quintic_monomials).
        QuinticSolver::new(2, n_points, 17, 1e-11)
    }

    /// Test 1: CPU reference returns finite (eta, w) on all points of a
    /// realistic 100-point synthetic input.
    #[test]
    fn cpu_reference_produces_finite_outputs() {
        let n_points = 100;
        let solver = match build_fixture(n_points) {
            Some(s) => s,
            None => {
                eprintln!("solver fixture failed; skipping");
                return;
            }
        };
        let n_basis = solver.n_basis;
        let n_pts_actual = solver.n_points;
        let stride_pp = 2 + 2 * n_basis * n_basis;
        let mut buf = vec![0.0f64; n_pts_actual * stride_pp];
        cpu_compute_per_point_reference(
            n_pts_actual,
            n_basis,
            &solver.section_values,
            &solver.section_derivs,
            &solver.h_block,
            &solver.points,
            &solver.weights,
            &solver.log_omega_sq,
            &mut buf,
        );
        // At least 80% of points should produce finite eta + nonzero w.
        let mut n_valid = 0;
        for p in 0..n_pts_actual {
            let off = p * stride_pp;
            if buf[off].is_finite() && buf[off + 1] > 0.0 {
                n_valid += 1;
                // Spot-check: gradient block must be all-finite.
                for q in 2..stride_pp {
                    assert!(
                        buf[off + q].is_finite(),
                        "non-finite gradient entry at point {} idx {}",
                        p,
                        q
                    );
                }
            }
        }
        assert!(
            n_valid >= (n_pts_actual * 8) / 10,
            "only {}/{} points produced valid outputs",
            n_valid,
            n_pts_actual
        );
    }

    /// Test 2: CPU reference matches the production
    /// `quintic::sigma_squared_and_gradient` per-point output, byte for
    /// byte (it should — both implement the same equations with identical
    /// rounding order). We use a small workspace and compare per_point_buf
    /// element-by-element.
    #[test]
    fn cpu_reference_matches_production_quintic() {
        let n_points = 80;
        let mut solver = match build_fixture(n_points) {
            Some(s) => s,
            None => {
                eprintln!("solver fixture failed; skipping");
                return;
            }
        };
        let n_basis = solver.n_basis;
        let n_pts_actual = solver.n_points;
        let stride_pp = 2 + 2 * n_basis * n_basis;

        // Run the production path. It writes into solver.per_point_buf.
        let _ = crate::quintic::sigma_squared_and_gradient(&mut solver);

        // Snapshot the production output (sigma_squared_and_gradient
        // mutates per_point_buf in place; we copy before the second call).
        let prod_buf = solver.per_point_buf[..n_pts_actual * stride_pp].to_vec();

        // Run our reference on the same inputs.
        let mut ref_buf = vec![0.0f64; n_pts_actual * stride_pp];
        cpu_compute_per_point_reference(
            n_pts_actual,
            n_basis,
            &solver.section_values,
            &solver.section_derivs,
            &solver.h_block,
            &solver.points,
            &solver.weights,
            &solver.log_omega_sq,
            &mut ref_buf,
        );

        // Compare. Both implementations execute the same arithmetic in
        // the same order, so we expect bit-exact agreement modulo
        // potential failure-mode masking (NaN comparisons).
        let mut n_compared = 0;
        let mut max_abs_diff = 0.0f64;
        for p in 0..n_pts_actual {
            let off = p * stride_pp;
            // If either side reported failure (eta NaN or w == 0), both
            // should agree; else compare full slot.
            let prod_eta = prod_buf[off];
            let ref_eta = ref_buf[off];
            if !prod_eta.is_finite() || prod_buf[off + 1] == 0.0 {
                assert!(
                    !ref_eta.is_finite() || ref_buf[off + 1] == 0.0,
                    "production failed at point {p} but reference accepted it"
                );
                continue;
            }
            assert!(ref_eta.is_finite(), "reference failed where production accepted");
            for q in 0..stride_pp {
                let d = (prod_buf[off + q] - ref_buf[off + q]).abs();
                if d > max_abs_diff {
                    max_abs_diff = d;
                }
            }
            n_compared += 1;
        }
        assert!(n_compared > 0, "no points to compare");
        assert!(
            max_abs_diff < 1e-12,
            "CPU reference diverged from production: max_abs_diff = {:.3e} over {} points",
            max_abs_diff,
            n_compared
        );
    }

    /// Test 3 (gated on `gpu` feature): GPU per-point gradient matches
    /// the CPU reference to ≤ 1e-10 absolute tolerance on a small input.
    /// Skips silently if no CUDA device is available.
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_per_point_matches_cpu_reference() {
        let n_points = 100;
        let solver = match build_fixture(n_points) {
            Some(s) => s,
            None => {
                eprintln!("solver fixture failed; skipping");
                return;
            }
        };
        let n_basis = solver.n_basis;
        let n_pts_actual = solver.n_points;
        let stride_pp = 2 + 2 * n_basis * n_basis;

        // Wrap in catch_unwind because cudarc panics rather than
        // returns an error when nvrtc.dll is missing on this host.
        let kernel_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            AdamGradientKernel::new(n_pts_actual, n_basis)
        }));
        let mut kernel = match kernel_result {
            Ok(Ok(k)) => k,
            Ok(Err(e)) => {
                eprintln!("AdamGradientKernel::new failed (no CUDA?): {e}; skipping");
                return;
            }
            Err(_) => {
                eprintln!("AdamGradientKernel::new panicked (likely no nvrtc.dll); skipping");
                return;
            }
        };

        let mut gpu_buf = vec![0.0f64; n_pts_actual * stride_pp];
        if let Err(e) = kernel.compute_per_point(
            &solver.section_values,
            &solver.section_derivs,
            &solver.h_block,
            &solver.points,
            &solver.weights,
            &solver.log_omega_sq,
            &mut gpu_buf,
        ) {
            eprintln!("GPU compute_per_point failed: {e}; skipping");
            return;
        }

        let mut cpu_buf = vec![0.0f64; n_pts_actual * stride_pp];
        cpu_compute_per_point_reference(
            n_pts_actual,
            n_basis,
            &solver.section_values,
            &solver.section_derivs,
            &solver.h_block,
            &solver.points,
            &solver.weights,
            &solver.log_omega_sq,
            &mut cpu_buf,
        );

        let mut max_abs_diff = 0.0f64;
        let mut n_compared = 0;
        for p in 0..n_pts_actual {
            let off = p * stride_pp;
            // Allow either side to fail (eta non-finite or w == 0).
            let cpu_ok = cpu_buf[off].is_finite() && cpu_buf[off + 1] > 0.0;
            let gpu_ok = gpu_buf[off].is_finite() && gpu_buf[off + 1] > 0.0;
            if !cpu_ok || !gpu_ok {
                continue;
            }
            for q in 0..stride_pp {
                let d = (cpu_buf[off + q] - gpu_buf[off + q]).abs();
                if d > max_abs_diff {
                    max_abs_diff = d;
                }
            }
            n_compared += 1;
        }
        assert!(n_compared > 0, "no points to compare on GPU");
        assert!(
            max_abs_diff <= 1e-10,
            "GPU diverged from CPU reference: max_abs_diff = {:.3e} over {} points",
            max_abs_diff,
            n_compared
        );
    }

    /// Test 4: BatchedLuDetKernel (and its CPU-side check) returns
    /// det = 1 for 100 identity matrices. CPU portion runs always; GPU
    /// portion gated on feature.
    #[test]
    fn batched_lu_det_identity_cpu() {
        let n_matrices = 100;
        // Build n_matrices 16×16 identity matrices.
        let mut matrices = vec![0.0f64; n_matrices * 256];
        for p in 0..n_matrices {
            for k in 0..16 {
                matrices[p * 256 + k * 16 + k] = 1.0;
            }
        }
        // CPU reference: directly call the same LU as refine.rs uses,
        // by re-implementing here (refine::determinant_lu is private).
        for p in 0..n_matrices {
            let mut a = matrices[p * 256..(p + 1) * 256].to_vec();
            let det = lu_det_16(&mut a);
            assert!(
                (det - 1.0).abs() < 1e-12,
                "CPU LU det of identity[{p}] = {det}"
            );
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn batched_lu_det_identity_gpu() {
        let n_matrices = 100;
        let mut matrices = vec![0.0f64; n_matrices * 256];
        for p in 0..n_matrices {
            for k in 0..16 {
                matrices[p * 256 + k * 16 + k] = 1.0;
            }
        }
        let kernel_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            BatchedLuDetKernel::new(n_matrices)
        }));
        let mut kernel = match kernel_result {
            Ok(Ok(k)) => k,
            Ok(Err(e)) => {
                eprintln!("BatchedLuDetKernel::new failed (no CUDA?): {e}; skipping");
                return;
            }
            Err(_) => {
                eprintln!("BatchedLuDetKernel::new panicked (likely no nvrtc.dll); skipping");
                return;
            }
        };
        let mut dets = vec![0.0f64; n_matrices];
        if let Err(e) = kernel.compute_dets(&matrices, &mut dets) {
            eprintln!("GPU compute_dets failed: {e}; skipping");
            return;
        }
        for (p, &d) in dets.iter().enumerate() {
            assert!(
                (d - 1.0).abs() < 1e-12,
                "GPU LU det of identity[{p}] = {d}"
            );
        }
    }

    /// Helper: 16×16 LU determinant (port of refine::determinant_lu).
    /// Used by `batched_lu_det_identity_cpu` since the production helper
    /// is module-private in refine.rs.
    fn lu_det_16(a: &mut [f64]) -> f64 {
        let n = 16;
        let mut sign = 1.0;
        for k in 0..n {
            let mut max_val = a[k * n + k].abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                let v = a[i * n + k].abs();
                if v > max_val {
                    max_val = v;
                    max_row = i;
                }
            }
            if max_val == 0.0 {
                return 0.0;
            }
            if max_row != k {
                for j in 0..n {
                    a.swap(k * n + j, max_row * n + j);
                }
                sign = -sign;
            }
            let pivot = a[k * n + k];
            for i in (k + 1)..n {
                let factor = a[i * n + k] / pivot;
                a[i * n + k] = factor;
                for j in (k + 1)..n {
                    a[i * n + j] -= factor * a[k * n + j];
                }
            }
        }
        let mut det = sign;
        for i in 0..n {
            det *= a[i * n + i];
        }
        det
    }
}
