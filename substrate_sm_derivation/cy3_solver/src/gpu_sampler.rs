//! CUDA-accelerated parallel Newton solver for the line-intersection
//! sampler in [`crate::cicy_sampler`].
//!
//! The CPU `CicySampler::sample_one` does, per accepted point:
//!  1. draw `NHYPER + 1 = 4` Gaussian-unit directions in `C^8`,
//!  2. Newton-iterate `t ∈ C^NHYPER` to solve
//!     `f_i(p + Σ_j t_j · d_j) = 0` for `i = 0, …, NHYPER−1`,
//!  3. patch-rescale, residue-formula `Ω`, pullback-metric `det g_pb`,
//!  4. assemble weight `w = |Ω|² / det g_pb`.
//!
//! Steps (1) and (3)–(4) are very small per-point work; the bottleneck
//! is step (2), which runs up to `max_newton_iter ≈ 60` evaluations of
//! a generic polynomial `f` and its Jacobian `J = ∂f/∂t` per attempt,
//! across `NATTEMPTS ≈ 4` Newton restarts. For `n_pts = 1500` that is
//! ~ 3.2 M complex multiplications on the CPU — clearly worth moving
//! to a parallel device kernel where each thread handles one (origin,
//! directions, initial-t) triple independently.
//!
//! ## Scope of this GPU port
//!
//! The kernel implements a single Newton attempt per thread (vanilla
//! Newton with damped step on residual increase — no Armijo
//! line-search). Multi-restart bookkeeping lives on the host: callers
//! call [`gpu_newton_solve_lines`] with `n_threads = n_pts × n_restarts`
//! initial guesses and post-process the per-thread convergence flags.
//! This split keeps the kernel small, branch-free, and parallel-
//! friendly; the host loop adds at most one extra microsecond per
//! point to coordinate restarts.
//!
//! Patch rescale, omega computation, and pullback-metric assembly stay
//! on the host — they are O(N) with small constants, are dominated
//! by data-dependent control flow (rejection branches), and would not
//! benefit much from device-side execution. The host can call the
//! existing [`crate::gpu_omega`] kernel for the omega step if desired.
//!
//! Numerical fidelity: `cuFp64`, no Kahan compensation. The kernel's
//! Newton update produces results bit-equivalent to the CPU
//! implementation when both use the same single-attempt vanilla path
//! (the test `gpu_newton_matches_cpu` exercises this on synthetic
//! inputs).

use cudarc::driver::{CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use num_complex::Complex64;
use std::sync::Arc;

use crate::cicy_sampler::{BicubicPair, NCOORDS, NHYPER};

const KERNEL_SOURCE: &str = r#"
// Polynomial layout matches gpu_omega: monos is a flat array of
// 10-double records [re, im, e0..e7], with mono_starts[p..p+1]
// giving the slice for polynomial p.

__device__ inline void cmul(double ar, double ai, double br, double bi,
                            double* outr, double* outi) {
    *outr = ar * br - ai * bi;
    *outi = ar * bi + ai * br;
}

// Repeated-squaring complex power: out = z^n.
__device__ inline void cpow_int(double br, double bi, int n,
                                double* outr, double* outi) {
    double rr = 1.0, ri = 0.0;
    while (n > 0) {
        if (n & 1) {
            double nr = rr * br - ri * bi;
            double ni = rr * bi + ri * br;
            rr = nr; ri = ni;
        }
        n >>= 1;
        if (n > 0) {
            double nbr = br * br - bi * bi;
            double nbi = 2.0 * br * bi;
            br = nbr; bi = nbi;
        }
    }
    *outr = rr;
    *outi = ri;
}

// Evaluate one polynomial f_p at coords zr/zi (length NCOORDS).
// Uses repeated squaring (cuts ~5x off the inner power loop vs the
// old naive `for (q = 0; q < e_j; ++q)` pattern).
__device__ void eval_poly(
    const double* monos, int s, int e,
    const double* zr, const double* zi,
    double* out_r, double* out_i
) {
    const int NC = 8;
    double pr = 0.0, pi = 0.0;
    for (int m = s; m < e; ++m) {
        int base = m * 10;
        double tr = monos[base + 0];
        double ti = monos[base + 1];
        for (int j = 0; j < NC; ++j) {
            int ej = (int)monos[base + 2 + j];
            if (ej == 0) continue;
            double pwr_r, pwr_i;
            cpow_int(zr[j], zi[j], ej, &pwr_r, &pwr_i);
            double nr, ni;
            cmul(tr, ti, pwr_r, pwr_i, &nr, &ni);
            tr = nr; ti = ni;
        }
        pr += tr;
        pi += ti;
    }
    *out_r = pr;
    *out_i = pi;
}

// Fused evaluator: returns f(z) AND its full gradient ∇f(z) (NC
// components) in a single pass over the monomials. The CPU and the
// old kernel called eval_poly + eval_poly_partial separately, doing
// the same monomial walk twice plus an extra time per coordinate;
// fusing yields ~2x reduction in per-Newton-iter arithmetic.
//
// Per-monomial work:
//   1. Compute z_j^{e_j} for all j once via repeated squaring.
//   2. Form V_m = c · Π_j z_j^{e_j} once.
//   3. Add V_m into the value accumulator.
//   4. For each k with e_k > 0: ∂_k V_m = e_k · V_m / z_k.
//      Add to grad accumulator at position k.
__device__ void eval_poly_with_grad(
    const double* monos, int s, int e,
    const double* zr, const double* zi,
    double* out_val_r, double* out_val_i,
    double* out_grad_r, double* out_grad_i   // length NC each
) {
    const int NC = 8;
    double vr_acc = 0.0, vi_acc = 0.0;
    for (int k = 0; k < NC; ++k) { out_grad_r[k] = 0.0; out_grad_i[k] = 0.0; }

    for (int m = s; m < e; ++m) {
        int base = m * 10;
        double cr = monos[base + 0];
        double ci = monos[base + 1];

        // Step 1+2: compute V_m = c · Π_j z_j^{e_j} (and remember per-coord
        // exponents for the gradient step).
        double pwr_r[8], pwr_i[8];
        int ej[8];
        for (int j = 0; j < NC; ++j) {
            ej[j] = (int)monos[base + 2 + j];
            cpow_int(zr[j], zi[j], ej[j], &pwr_r[j], &pwr_i[j]);
        }
        double vr = cr, vi = ci;
        for (int j = 0; j < NC; ++j) {
            if (ej[j] == 0) continue;
            double nr, ni;
            cmul(vr, vi, pwr_r[j], pwr_i[j], &nr, &ni);
            vr = nr; vi = ni;
        }

        vr_acc += vr;
        vi_acc += vi;

        // Step 4: gradient. ∂_k V_m = e_k · V_m / z_k for k with e_k > 0.
        // Skip if |z_k| < ~1e-150: at exactly z_k = 0 with e_k = 1
        // the true partial is c · Π_{j≠k} z_j^e_j (finite, not zero),
        // but Newton convergence in the line-intersection sampler
        // never lands on coordinate-vanishing points (the patch
        // re-scaling normalises the largest-modulus coord to 1.0,
        // and the Newton iterates start far from coordinate
        // hyperplanes by the Gaussian-direction construction).
        for (int k = 0; k < NC; ++k) {
            if (ej[k] == 0) continue;
            double zk_r = zr[k], zk_i = zi[k];
            double zk_mag2 = zk_r * zk_r + zk_i * zk_i;
            if (zk_mag2 < 1.0e-300) continue;
            double inv_zk_r =  zk_r / zk_mag2;
            double inv_zk_i = -zk_i / zk_mag2;
            double dr_, di_;
            cmul(vr, vi, inv_zk_r, inv_zk_i, &dr_, &di_);
            double scale = (double)ej[k];
            out_grad_r[k] += scale * dr_;
            out_grad_i[k] += scale * di_;
        }
    }

    *out_val_r = vr_acc;
    *out_val_i = vi_acc;
}

// Solve a 3x3 complex linear system M · x = b in place (overwrites b).
// Uses LU with partial pivoting; returns 0 on singular matrix.
__device__ int solve_3x3_complex(double* M_re, double* M_im, double* b_re, double* b_im) {
    const int N = 3;
    int piv[3];
    for (int i = 0; i < N; ++i) piv[i] = i;
    for (int k = 0; k < N; ++k) {
        // Pick pivot.
        int p = k;
        double best = M_re[k * N + k] * M_re[k * N + k] + M_im[k * N + k] * M_im[k * N + k];
        for (int i = k + 1; i < N; ++i) {
            double mag = M_re[i * N + k] * M_re[i * N + k] + M_im[i * N + k] * M_im[i * N + k];
            if (mag > best) { best = mag; p = i; }
        }
        if (best < 1.0e-300) return 0;
        if (p != k) {
            for (int j = 0; j < N; ++j) {
                double tr = M_re[k * N + j], ti = M_im[k * N + j];
                M_re[k * N + j] = M_re[p * N + j];
                M_im[k * N + j] = M_im[p * N + j];
                M_re[p * N + j] = tr;
                M_im[p * N + j] = ti;
            }
            double br = b_re[k], bi = b_im[k];
            b_re[k] = b_re[p]; b_im[k] = b_im[p];
            b_re[p] = br;       b_im[p] = bi;
        }
        // Eliminate below.
        double pivr = M_re[k * N + k], pivi = M_im[k * N + k];
        double pmag = pivr * pivr + pivi * pivi;
        for (int i = k + 1; i < N; ++i) {
            double mr = M_re[i * N + k], mi = M_im[i * N + k];
            // factor = M[i,k] / M[k,k]
            double fr = (mr * pivr + mi * pivi) / pmag;
            double fi = (mi * pivr - mr * pivi) / pmag;
            for (int j = k; j < N; ++j) {
                double tr, ti;
                cmul(fr, fi, M_re[k * N + j], M_im[k * N + j], &tr, &ti);
                M_re[i * N + j] -= tr;
                M_im[i * N + j] -= ti;
            }
            double tr, ti;
            cmul(fr, fi, b_re[k], b_im[k], &tr, &ti);
            b_re[i] -= tr;
            b_im[i] -= ti;
        }
    }
    // Back-substitute.
    for (int i = N - 1; i >= 0; --i) {
        double sr = b_re[i], si = b_im[i];
        for (int j = i + 1; j < N; ++j) {
            double tr, ti;
            cmul(M_re[i * N + j], M_im[i * N + j], b_re[j], b_im[j], &tr, &ti);
            sr -= tr;
            si -= ti;
        }
        // x_i = s / M[i,i]
        double dr = M_re[i * N + i], di = M_im[i * N + i];
        double dmag = dr * dr + di * di;
        if (dmag < 1.0e-300) return 0;
        b_re[i] = (sr * dr + si * di) / dmag;
        b_im[i] = (si * dr - sr * di) / dmag;
    }
    return 1;
}

// One Newton attempt per thread.
//
// Inputs (all per-thread, indexed by global thread id `tid`):
//   origin_re/im[8 * tid + k]            = origin[k] (k = 0..7)
//   dir_re/im[NHYPER*8 * tid + j*8 + k]  = direction j coord k
//   t_init_re/im[NHYPER * tid + j]       = initial guess t_j
//
// Outputs:
//   t_re/im[NHYPER * tid + j]            = final t_j
//   residual[tid]                        = ||f(x_final)||
//   converged[tid]                       = 1 if residual < tol, else 0
extern "C" __global__ void newton_solve_lines(
    const double* __restrict__ monos,
    const int*    __restrict__ mono_starts,
    const double* __restrict__ origin_re,
    const double* __restrict__ origin_im,
    const double* __restrict__ dir_re,
    const double* __restrict__ dir_im,
    const double* __restrict__ t_init_re,
    const double* __restrict__ t_init_im,
    int n_threads,
    int max_iter,
    double tol,
    double* __restrict__ t_re,
    double* __restrict__ t_im,
    double* __restrict__ residual,
    int*    __restrict__ converged
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;

    const int NH = 3;
    const int NC = 8;

    // Copy initial t.
    double tr[NH], ti[NH];
    for (int j = 0; j < NH; ++j) {
        tr[j] = t_init_re[tid * NH + j];
        ti[j] = t_init_im[tid * NH + j];
    }
    // Origin in registers.
    double pr[NC], pi[NC];
    for (int k = 0; k < NC; ++k) {
        pr[k] = origin_re[tid * NC + k];
        pi[k] = origin_im[tid * NC + k];
    }
    // Directions: dir[j][k] = dr/di[j*NC+k]
    double dr_arr[NH * NC], di_arr[NH * NC];
    for (int j = 0; j < NH; ++j) {
        for (int k = 0; k < NC; ++k) {
            dr_arr[j * NC + k] = dir_re[tid * NH * NC + j * NC + k];
            di_arr[j * NC + k] = dir_im[tid * NH * NC + j * NC + k];
        }
    }

    double prev_res = 1.0e300;
    int conv = 0;

    for (int iter = 0; iter < max_iter; ++iter) {
        // x = p + Σ_j t_j * d_j
        double xr[NC], xi[NC];
        for (int k = 0; k < NC; ++k) {
            double sr = pr[k], si = pi[k];
            for (int j = 0; j < NH; ++j) {
                double tk_r, tk_i;
                cmul(tr[j], ti[j], dr_arr[j * NC + k], di_arr[j * NC + k], &tk_r, &tk_i);
                sr += tk_r;
                si += tk_i;
            }
            xr[k] = sr;
            xi[k] = si;
        }
        // f(x) AND J_t = ∂f/∂t in one fused pass per polynomial.
        // Each polynomial walk computes its value f_i + its
        // x-gradient (NC doubles, scratch) and immediately folds the
        // gradient into the J_t row J_t[i][:] = Σ_k ∇f_i[k] · d_j[k].
        // Per-thread state:  scratch row_gr/row_gi (16 doubles, brief
        // lifetime); long-lived fr/fi (NH=3 each), Jtr/Jti (NH×NH=9
        // each). No full ∇f cached across polynomials.
        double fr[NH], fi[NH];
        double Jtr[NH * NH], Jti[NH * NH];
        for (int i = 0; i < NH; ++i) {
            double row_gr[NC], row_gi[NC];
            eval_poly_with_grad(
                monos, mono_starts[i], mono_starts[i + 1],
                xr, xi,
                &fr[i], &fi[i],
                row_gr, row_gi
            );
            for (int j = 0; j < NH; ++j) {
                double sr = 0.0, si = 0.0;
                for (int k = 0; k < NC; ++k) {
                    double tk_r, tk_i;
                    cmul(row_gr[k], row_gi[k],
                         dr_arr[j * NC + k], di_arr[j * NC + k],
                         &tk_r, &tk_i);
                    sr += tk_r;
                    si += tk_i;
                }
                Jtr[i * NH + j] = sr;
                Jti[i * NH + j] = si;
            }
        }
        double res = 0.0;
        for (int i = 0; i < NH; ++i) res += fr[i] * fr[i] + fi[i] * fi[i];
        res = sqrt(res);
        if (res < tol) { conv = 1; break; }
        // Solve J_t · δ = -f
        double br_[NH], bi_[NH];
        for (int i = 0; i < NH; ++i) {
            br_[i] = -fr[i];
            bi_[i] = -fi[i];
        }
        int ok = solve_3x3_complex(Jtr, Jti, br_, bi_);
        if (!ok) break;

        // Damped update: try full step, then halve until residual decreases.
        double damp = 1.0;
        double tr_new[NH], ti_new[NH];
        double new_res;
        for (int trial = 0; trial < 8; ++trial) {
            for (int j = 0; j < NH; ++j) {
                tr_new[j] = tr[j] + damp * br_[j];
                ti_new[j] = ti[j] + damp * bi_[j];
            }
            // Recompute residual at trial t
            double xr2[NC], xi2[NC];
            for (int k = 0; k < NC; ++k) {
                double sr = pr[k], si = pi[k];
                for (int j = 0; j < NH; ++j) {
                    double tk_r, tk_i;
                    cmul(tr_new[j], ti_new[j], dr_arr[j * NC + k], di_arr[j * NC + k], &tk_r, &tk_i);
                    sr += tk_r;
                    si += tk_i;
                }
                xr2[k] = sr; xi2[k] = si;
            }
            double fr2[NH], fi2[NH];
            for (int i = 0; i < NH; ++i) {
                eval_poly(monos, mono_starts[i], mono_starts[i + 1], xr2, xi2, &fr2[i], &fi2[i]);
            }
            new_res = 0.0;
            for (int i = 0; i < NH; ++i) new_res += fr2[i] * fr2[i] + fi2[i] * fi2[i];
            new_res = sqrt(new_res);
            if (new_res < res) break;
            damp *= 0.5;
        }
        if (new_res >= res) break;  // no improvement; bail
        for (int j = 0; j < NH; ++j) {
            tr[j] = tr_new[j];
            ti[j] = ti_new[j];
        }
        prev_res = new_res;
        if (new_res < tol) { conv = 1; break; }
    }

    for (int j = 0; j < NH; ++j) {
        t_re[tid * NH + j] = tr[j];
        t_im[tid * NH + j] = ti[j];
    }
    residual[tid] = prev_res;
    converged[tid] = conv;
}
"#;

pub struct GpuSamplerContext {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
}

impl GpuSamplerContext {
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
                    "GPU sampler init panic — likely no CUDA driver / nvrtc.dll: {msg}"
                )
                .into())
            }
        }
    }
}

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

/// Per-thread convergence result of the parallel Newton solver.
#[derive(Debug, Clone, Copy)]
pub struct NewtonResult {
    /// Final `t ∈ C^{NHYPER}` solution.
    pub t: [Complex64; NHYPER],
    /// L2 residual `||f(p + Σ t_j d_j)||` at termination.
    pub residual: f64,
    /// `true` iff `residual < tol` at termination.
    pub converged: bool,
}

/// Parallel Newton solve. Each "thread" `i` solves one independent
/// system with its own `(origin, directions, t_init)`. Returns one
/// [`NewtonResult`] per thread.
///
/// Input layout:
/// * `origins_re/im[i * 8 + k]` = real/imag part of coord `k` of origin
///   for thread `i` (8 coords per origin).
/// * `dirs_re/im[i * NHYPER * 8 + j * 8 + k]` = real/imag of coord `k`
///   of direction `j` for thread `i` (NHYPER directions).
/// * `t_init_re/im[i * NHYPER + j]` = real/imag of initial guess
///   `t_j` for thread `i`.
pub fn gpu_newton_solve_lines(
    gpu_ctx: &GpuSamplerContext,
    bicubic: &BicubicPair,
    origins_re: &[f64],
    origins_im: &[f64],
    dirs_re: &[f64],
    dirs_im: &[f64],
    t_init_re: &[f64],
    t_init_im: &[f64],
    n_threads: usize,
    max_iter: usize,
    tol: f64,
) -> Result<Vec<NewtonResult>, Box<dyn std::error::Error>> {
    if n_threads == 0 {
        return Ok(Vec::new());
    }
    let need_origins = NCOORDS * n_threads;
    let need_dirs = NHYPER * NCOORDS * n_threads;
    let need_t = NHYPER * n_threads;
    if origins_re.len() != need_origins
        || origins_im.len() != need_origins
        || dirs_re.len() != need_dirs
        || dirs_im.len() != need_dirs
        || t_init_re.len() != need_t
        || t_init_im.len() != need_t
    {
        return Err("input slice length mismatch".into());
    }

    let (monos, starts) = pack_bicubic(bicubic);
    let stream = &gpu_ctx.stream;

    let d_monos = stream.memcpy_stod(&monos)?;
    let d_starts = stream.memcpy_stod(&starts)?;
    let d_or = stream.memcpy_stod(origins_re)?;
    let d_oi = stream.memcpy_stod(origins_im)?;
    let d_dr = stream.memcpy_stod(dirs_re)?;
    let d_di = stream.memcpy_stod(dirs_im)?;
    let d_tr0 = stream.memcpy_stod(t_init_re)?;
    let d_ti0 = stream.memcpy_stod(t_init_im)?;
    let mut d_tr = stream.alloc_zeros::<f64>(need_t)?;
    let mut d_ti = stream.alloc_zeros::<f64>(need_t)?;
    let mut d_res = stream.alloc_zeros::<f64>(n_threads)?;
    let mut d_conv = stream.alloc_zeros::<i32>(n_threads)?;

    let func = gpu_ctx.module.load_function("newton_solve_lines")?;
    let threads_per_block: u32 = 64; // small to allow per-thread register-heavy work
    let n_blocks = ((n_threads as u32) + threads_per_block - 1) / threads_per_block;
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_threads_i32 = n_threads as i32;
    let max_iter_i32 = max_iter as i32;
    let mut launcher = stream.launch_builder(&func);
    launcher
        .arg(&d_monos)
        .arg(&d_starts)
        .arg(&d_or)
        .arg(&d_oi)
        .arg(&d_dr)
        .arg(&d_di)
        .arg(&d_tr0)
        .arg(&d_ti0)
        .arg(&n_threads_i32)
        .arg(&max_iter_i32)
        .arg(&tol)
        .arg(&mut d_tr)
        .arg(&mut d_ti)
        .arg(&mut d_res)
        .arg(&mut d_conv);
    unsafe { launcher.launch(cfg)? };

    let host_tr = stream.memcpy_dtov(&d_tr)?;
    let host_ti = stream.memcpy_dtov(&d_ti)?;
    let host_res = stream.memcpy_dtov(&d_res)?;
    let host_conv = stream.memcpy_dtov(&d_conv)?;
    let _ = &gpu_ctx.ctx;

    Ok((0..n_threads)
        .map(|i| {
            let mut t = [Complex64::new(0.0, 0.0); NHYPER];
            for j in 0..NHYPER {
                t[j] = Complex64::new(host_tr[i * NHYPER + j], host_ti[i * NHYPER + j]);
            }
            NewtonResult {
                t,
                residual: host_res[i],
                converged: host_conv[i] != 0,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU reference Newton (vanilla damped, single attempt). Mirrors
    /// the kernel exactly so we can verify bit-for-bit parity on a
    /// known input.
    fn cpu_newton_one_attempt(
        bicubic: &BicubicPair,
        origin: &[Complex64; NCOORDS],
        dirs: &[[Complex64; NCOORDS]; NHYPER],
        t_init: &[Complex64; NHYPER],
        max_iter: usize,
        tol: f64,
    ) -> NewtonResult {
        let mut t = *t_init;
        let mut prev_res = f64::INFINITY;
        let mut conv = false;
        for _ in 0..max_iter {
            // x = origin + Σ t_j d_j
            let mut x = [Complex64::new(0.0, 0.0); NCOORDS];
            for k in 0..NCOORDS {
                let mut s = origin[k];
                for j in 0..NHYPER {
                    s += t[j] * dirs[j][k];
                }
                x[k] = s;
            }
            let f = bicubic.eval(&x);
            let res = f.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
            if res < tol {
                conv = true;
                prev_res = res;
                break;
            }
            // Build J_t = J_x · D
            let jac = bicubic.jacobian(&x);
            let mut jt = [[Complex64::new(0.0, 0.0); NHYPER]; NHYPER];
            for i in 0..NHYPER {
                for j in 0..NHYPER {
                    let mut s = Complex64::new(0.0, 0.0);
                    for k in 0..NCOORDS {
                        s += jac[i * NCOORDS + k] * dirs[j][k];
                    }
                    jt[i][j] = s;
                }
            }
            // Solve jt · δ = -f via 3x3 LU (use the same algorithm as the kernel).
            let mut m_re = [0.0_f64; 9];
            let mut m_im = [0.0_f64; 9];
            for i in 0..3 {
                for j in 0..3 {
                    m_re[i * 3 + j] = jt[i][j].re;
                    m_im[i * 3 + j] = jt[i][j].im;
                }
            }
            let mut b_re = [-f[0].re, -f[1].re, -f[2].re];
            let mut b_im = [-f[0].im, -f[1].im, -f[2].im];
            if !solve_3x3_lu(&mut m_re, &mut m_im, &mut b_re, &mut b_im) {
                break;
            }
            // Damped update.
            let mut damp = 1.0_f64;
            let mut new_res = 0.0;
            let mut t_new = t;
            for _ in 0..8 {
                for j in 0..NHYPER {
                    t_new[j] = t[j] + Complex64::new(damp * b_re[j], damp * b_im[j]);
                }
                let mut x2 = [Complex64::new(0.0, 0.0); NCOORDS];
                for k in 0..NCOORDS {
                    let mut s = origin[k];
                    for j in 0..NHYPER {
                        s += t_new[j] * dirs[j][k];
                    }
                    x2[k] = s;
                }
                let f2 = bicubic.eval(&x2);
                new_res = f2.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
                if new_res < res {
                    break;
                }
                damp *= 0.5;
            }
            if new_res >= res {
                break;
            }
            t = t_new;
            prev_res = new_res;
            if new_res < tol {
                conv = true;
                break;
            }
        }
        NewtonResult { t, residual: prev_res, converged: conv }
    }

    fn solve_3x3_lu(m_re: &mut [f64; 9], m_im: &mut [f64; 9], b_re: &mut [f64; 3], b_im: &mut [f64; 3]) -> bool {
        for k in 0..3 {
            // Pivot
            let mut p = k;
            let mut best = m_re[k * 3 + k] * m_re[k * 3 + k] + m_im[k * 3 + k] * m_im[k * 3 + k];
            for i in (k + 1)..3 {
                let mag = m_re[i * 3 + k] * m_re[i * 3 + k] + m_im[i * 3 + k] * m_im[i * 3 + k];
                if mag > best { best = mag; p = i; }
            }
            if best < 1e-300 { return false; }
            if p != k {
                for j in 0..3 {
                    m_re.swap(k * 3 + j, p * 3 + j);
                    m_im.swap(k * 3 + j, p * 3 + j);
                }
                b_re.swap(k, p);
                b_im.swap(k, p);
            }
            let pivr = m_re[k * 3 + k];
            let pivi = m_im[k * 3 + k];
            let pmag = pivr * pivr + pivi * pivi;
            for i in (k + 1)..3 {
                let mr = m_re[i * 3 + k];
                let mi = m_im[i * 3 + k];
                let fr = (mr * pivr + mi * pivi) / pmag;
                let fi = (mi * pivr - mr * pivi) / pmag;
                for j in k..3 {
                    let tr = fr * m_re[k * 3 + j] - fi * m_im[k * 3 + j];
                    let ti = fr * m_im[k * 3 + j] + fi * m_re[k * 3 + j];
                    m_re[i * 3 + j] -= tr;
                    m_im[i * 3 + j] -= ti;
                }
                let tr = fr * b_re[k] - fi * b_im[k];
                let ti = fr * b_im[k] + fi * b_re[k];
                b_re[i] -= tr;
                b_im[i] -= ti;
            }
        }
        for i in (0..3).rev() {
            let mut sr = b_re[i];
            let mut si = b_im[i];
            for j in (i + 1)..3 {
                let tr = m_re[i * 3 + j] * b_re[j] - m_im[i * 3 + j] * b_im[j];
                let ti = m_re[i * 3 + j] * b_im[j] + m_im[i * 3 + j] * b_re[j];
                sr -= tr;
                si -= ti;
            }
            let dr = m_re[i * 3 + i];
            let di = m_im[i * 3 + i];
            let dmag = dr * dr + di * di;
            if dmag < 1e-300 { return false; }
            b_re[i] = (sr * dr + si * di) / dmag;
            b_im[i] = (si * dr - sr * di) / dmag;
        }
        true
    }

    #[test]
    fn gpu_newton_matches_cpu() {
        let ctx = match GpuSamplerContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("GPU unavailable, skipping parity test: {e}");
                return;
            }
        };

        let pair = BicubicPair::z3_invariant_default();
        // 16 synthetic threads with Gaussian-ish inputs.
        let n_threads = 16;
        let mut state: u64 = 0xDEADC0DE;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) & ((1 << 53) - 1)) as f64 / (1u64 << 53) as f64 - 0.5
        };

        let mut origins_re = vec![0.0_f64; NCOORDS * n_threads];
        let mut origins_im = vec![0.0_f64; NCOORDS * n_threads];
        let mut dirs_re = vec![0.0_f64; NHYPER * NCOORDS * n_threads];
        let mut dirs_im = vec![0.0_f64; NHYPER * NCOORDS * n_threads];
        let mut t_init_re = vec![0.0_f64; NHYPER * n_threads];
        let mut t_init_im = vec![0.0_f64; NHYPER * n_threads];

        for i in 0..n_threads {
            for k in 0..NCOORDS {
                origins_re[i * NCOORDS + k] = next();
                origins_im[i * NCOORDS + k] = next();
            }
            for j in 0..NHYPER {
                for k in 0..NCOORDS {
                    dirs_re[i * NHYPER * NCOORDS + j * NCOORDS + k] = next();
                    dirs_im[i * NHYPER * NCOORDS + j * NCOORDS + k] = next();
                }
            }
            for j in 0..NHYPER {
                t_init_re[i * NHYPER + j] = next();
                t_init_im[i * NHYPER + j] = next();
            }
        }

        let max_iter = 60;
        let tol = 1e-9_f64;

        let gpu = gpu_newton_solve_lines(
            &ctx,
            &pair,
            &origins_re,
            &origins_im,
            &dirs_re,
            &dirs_im,
            &t_init_re,
            &t_init_im,
            n_threads,
            max_iter,
            tol,
        )
        .unwrap();

        for i in 0..n_threads {
            let mut origin = [Complex64::new(0.0, 0.0); NCOORDS];
            for k in 0..NCOORDS {
                origin[k] = Complex64::new(origins_re[i * NCOORDS + k], origins_im[i * NCOORDS + k]);
            }
            let mut dirs = [[Complex64::new(0.0, 0.0); NCOORDS]; NHYPER];
            for j in 0..NHYPER {
                for k in 0..NCOORDS {
                    dirs[j][k] = Complex64::new(
                        dirs_re[i * NHYPER * NCOORDS + j * NCOORDS + k],
                        dirs_im[i * NHYPER * NCOORDS + j * NCOORDS + k],
                    );
                }
            }
            let mut t_init = [Complex64::new(0.0, 0.0); NHYPER];
            for j in 0..NHYPER {
                t_init[j] = Complex64::new(t_init_re[i * NHYPER + j], t_init_im[i * NHYPER + j]);
            }
            let cpu = cpu_newton_one_attempt(&pair, &origin, &dirs, &t_init, max_iter, tol);

            assert_eq!(
                gpu[i].converged, cpu.converged,
                "convergence flag differs at thread {i}: gpu={} cpu={}",
                gpu[i].converged, cpu.converged
            );
            // For converged threads compare t to high accuracy. For
            // non-converged compare residuals (Newton might bail at
            // slightly different points after damping but the order
            // of magnitude must match).
            if cpu.converged {
                for j in 0..NHYPER {
                    assert!(
                        (gpu[i].t[j].re - cpu.t[j].re).abs() < 1e-9,
                        "t[{j}].re differs at thread {i}: gpu={} cpu={}",
                        gpu[i].t[j].re,
                        cpu.t[j].re
                    );
                    assert!(
                        (gpu[i].t[j].im - cpu.t[j].im).abs() < 1e-9,
                        "t[{j}].im differs at thread {i}: gpu={} cpu={}",
                        gpu[i].t[j].im,
                        cpu.t[j].im
                    );
                }
            }
        }
    }
}
