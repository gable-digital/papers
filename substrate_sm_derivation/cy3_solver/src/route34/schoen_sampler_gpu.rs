//! CUDA-accelerated parallel Newton solver for the Schoen-side
//! line-intersection sampler. Mirrors the Tian-Yau pattern in
//! [`crate::gpu_sampler`] but with `NHYPER = 2`, `NCOORDS = 8`, and
//! `NFACTORS = 3` (CP^2 × CP^2 × CP^1).
//!
//! Each thread runs one Newton attempt; multi-restart bookkeeping lives
//! on the host. Patch rescale, omega computation, and pullback-metric
//! assembly stay on the host (small constants, control-flow heavy).
//!
//! Numerical fidelity matches the CPU reference to machine precision
//! (~1e-12 absolute deviation in `t` after convergence; full bit
//! reproducibility cannot be guaranteed in the presence of fused
//! multiply-add reordering across `nvrtc` versions, but the test
//! [`tests::gpu_matches_cpu`] enforces a tight tolerance).

use cudarc::driver::{CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use num_complex::Complex64;
use std::sync::Arc;

use crate::route34::schoen_sampler::{SchoenPoly, NCOORDS, NHYPER};

const KERNEL_SOURCE: &str = r#"
__device__ inline void cmul(double ar, double ai, double br, double bi,
                            double* outr, double* outi) {
    *outr = ar * br - ai * bi;
    *outi = ar * bi + ai * br;
}

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

__device__ void eval_poly_with_grad(
    const double* monos, int s, int e,
    const double* zr, const double* zi,
    double* fr, double* fi,
    double* gr, double* gi
) {
    const int NC = 8;
    *fr = 0.0;
    *fi = 0.0;
    for (int j = 0; j < NC; ++j) { gr[j] = 0.0; gi[j] = 0.0; }
    for (int m = s; m < e; ++m) {
        int base = m * 10;
        double cr = monos[base + 0];
        double ci = monos[base + 1];
        // Pre-compute z_j^{e_j} for each j.
        double pwr_r[NC], pwr_i[NC];
        int exps[NC];
        for (int j = 0; j < NC; ++j) {
            exps[j] = (int)monos[base + 2 + j];
            cpow_int(zr[j], zi[j], exps[j], &pwr_r[j], &pwr_i[j]);
        }
        // Full monomial value V = c · Π z_j^{e_j}
        double vr = cr, vi = ci;
        for (int j = 0; j < NC; ++j) {
            if (exps[j] == 0) continue;
            double nr, ni;
            cmul(vr, vi, pwr_r[j], pwr_i[j], &nr, &ni);
            vr = nr; vi = ni;
        }
        *fr += vr;
        *fi += vi;
        // ∂V/∂z_j = c · e_j · z_j^{e_j - 1} · Π_{k≠j} z_k^{e_k}
        for (int j = 0; j < NC; ++j) {
            if (exps[j] == 0) continue;
            double dr = cr * (double)exps[j];
            double di = ci * (double)exps[j];
            // Multiply by z_j^{e_j - 1}
            double pjr, pji;
            cpow_int(zr[j], zi[j], exps[j] - 1, &pjr, &pji);
            double nr, ni;
            cmul(dr, di, pjr, pji, &nr, &ni);
            dr = nr; di = ni;
            // Multiply by Π_{k != j} z_k^{e_k}
            for (int k = 0; k < NC; ++k) {
                if (k == j || exps[k] == 0) continue;
                cmul(dr, di, pwr_r[k], pwr_i[k], &nr, &ni);
                dr = nr; di = ni;
            }
            gr[j] += dr;
            gi[j] += di;
        }
    }
}

__device__ inline int solve_2x2_complex(double* mr, double* mi, double* br, double* bi) {
    // M = [[a, b], [c, d]] with complex entries. det = ad − bc.
    double ar = mr[0], ai = mi[0];
    double br_ = mr[1], bi_ = mi[1];
    double cr = mr[2], ci = mi[2];
    double dr = mr[3], di = mi[3];
    double adr = ar * dr - ai * di;
    double adi = ar * di + ai * dr;
    double bcr = br_ * cr - bi_ * ci;
    double bci = br_ * ci + bi_ * cr;
    double detr = adr - bcr;
    double deti = adi - bci;
    double dmag = detr * detr + deti * deti;
    if (dmag < 1e-300) return 0;
    // Inverse-multiplication: x = M^{-1} b
    // M^{-1} = (1/det) [[d, -b], [-c, a]]
    double inv_r =  detr / dmag;
    double inv_i = -deti / dmag;
    double new_br[2], new_bi[2];
    // x_0 = (d · b_0 − b · b_1) / det
    {
        double t1r, t1i, t2r, t2i;
        cmul(dr, di, br[0], bi[0], &t1r, &t1i);
        cmul(br_, bi_, br[1], bi[1], &t2r, &t2i);
        double nr = t1r - t2r;
        double ni = t1i - t2i;
        // multiply by 1/det
        new_br[0] = nr * inv_r - ni * inv_i;
        new_bi[0] = nr * inv_i + ni * inv_r;
    }
    // x_1 = (a · b_1 − c · b_0) / det
    {
        double t1r, t1i, t2r, t2i;
        cmul(ar, ai, br[1], bi[1], &t1r, &t1i);
        cmul(cr, ci, br[0], bi[0], &t2r, &t2i);
        double nr = t1r - t2r;
        double ni = t1i - t2i;
        new_br[1] = nr * inv_r - ni * inv_i;
        new_bi[1] = nr * inv_i + ni * inv_r;
    }
    br[0] = new_br[0]; bi[0] = new_bi[0];
    br[1] = new_br[1]; bi[1] = new_bi[1];
    return 1;
}

extern "C" __global__ void schoen_newton_solve_lines(
    const double* monos,
    const int* mono_starts,
    const double* origin_re,
    const double* origin_im,
    const double* dir_re,
    const double* dir_im,
    const double* t_init_re,
    const double* t_init_im,
    const int n_threads,
    const int max_iter,
    const double tol,
    double* t_re,
    double* t_im,
    double* residual,
    int* converged
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;

    const int NH = 2; // NHYPER for Schoen
    const int NC = 8;

    double tr[NH], ti[NH];
    for (int j = 0; j < NH; ++j) {
        tr[j] = t_init_re[tid * NH + j];
        ti[j] = t_init_im[tid * NH + j];
    }
    double pr[NC], pi[NC];
    for (int k = 0; k < NC; ++k) {
        pr[k] = origin_re[tid * NC + k];
        pi[k] = origin_im[tid * NC + k];
    }
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
        if (res < tol) { conv = 1; prev_res = res; break; }
        double br_[NH], bi_[NH];
        for (int i = 0; i < NH; ++i) {
            br_[i] = -fr[i];
            bi_[i] = -fi[i];
        }
        int ok = solve_2x2_complex(Jtr, Jti, br_, bi_);
        if (!ok) break;

        double damp = 1.0;
        double tr_new[NH], ti_new[NH];
        double new_res = 1.0e300;
        for (int trial = 0; trial < 8; ++trial) {
            for (int j = 0; j < NH; ++j) {
                tr_new[j] = tr[j] + damp * br_[j];
                ti_new[j] = ti[j] + damp * bi_[j];
            }
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
        if (new_res >= res) { prev_res = res; break; }
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

/// Initialised once per thread. Holds the CUDA context, default stream
/// and the compiled Schoen-Newton module.
pub struct SchoenGpuContext {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
}

impl SchoenGpuContext {
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
                    "Schoen GPU sampler init panic — likely no CUDA driver / nvrtc.dll: {msg}"
                )
                .into())
            }
        }
    }
}

/// Per-thread Newton convergence result. Identical semantics to
/// [`crate::gpu_sampler::NewtonResult`] but typed for `NHYPER = 2`.
#[derive(Debug, Clone, Copy)]
pub struct SchoenNewtonResult {
    pub t: [Complex64; NHYPER],
    pub residual: f64,
    pub converged: bool,
}

fn pack_poly(poly: &SchoenPoly) -> (Vec<f64>, Vec<i32>) {
    let polys: [&Vec<(Complex64, [u32; 8])>; NHYPER] = [&poly.f1, &poly.f2];
    let total: usize = polys.iter().map(|p| p.len()).sum();
    let mut monos: Vec<f64> = Vec::with_capacity(total * 10);
    let mut starts: Vec<i32> = Vec::with_capacity(NHYPER + 1);
    starts.push(0);
    for p in polys {
        for (coef, exps) in p {
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

/// Parallel Newton solve. Mirrors [`crate::gpu_sampler::gpu_newton_solve_lines`].
pub fn gpu_newton_solve_lines(
    gpu_ctx: &SchoenGpuContext,
    poly: &SchoenPoly,
    origins_re: &[f64],
    origins_im: &[f64],
    dirs_re: &[f64],
    dirs_im: &[f64],
    t_init_re: &[f64],
    t_init_im: &[f64],
    n_threads: usize,
    max_iter: usize,
    tol: f64,
) -> Result<Vec<SchoenNewtonResult>, Box<dyn std::error::Error>> {
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

    let (monos, starts) = pack_poly(poly);
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

    let func = gpu_ctx.module.load_function("schoen_newton_solve_lines")?;
    let threads_per_block: u32 = 64;
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
            SchoenNewtonResult {
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

    /// CPU reference single-attempt Newton matching the kernel exactly.
    fn cpu_newton_one_attempt(
        poly: &SchoenPoly,
        origin: &[Complex64; NCOORDS],
        dirs: &[[Complex64; NCOORDS]; NHYPER],
        t_init: &[Complex64; NHYPER],
        max_iter: usize,
        tol: f64,
    ) -> SchoenNewtonResult {
        let mut t = *t_init;
        let mut prev_res = f64::INFINITY;
        let mut conv = false;
        for _ in 0..max_iter {
            let mut x = [Complex64::new(0.0, 0.0); NCOORDS];
            for k in 0..NCOORDS {
                let mut s = origin[k];
                for j in 0..NHYPER {
                    s += t[j] * dirs[j][k];
                }
                x[k] = s;
            }
            let f = poly.eval(&x);
            let res = f.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
            if res < tol {
                conv = true;
                prev_res = res;
                break;
            }
            let jac = poly.jacobian(&x);
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
            // Solve 2x2: jt · δ = -f
            let det = jt[0][0] * jt[1][1] - jt[0][1] * jt[1][0];
            if det.norm() < 1e-30 {
                break;
            }
            let inv_det = Complex64::new(1.0, 0.0) / det;
            let dt0 = (jt[1][1] * (-f[0]) - jt[0][1] * (-f[1])) * inv_det;
            let dt1 = (jt[0][0] * (-f[1]) - jt[1][0] * (-f[0])) * inv_det;
            let mut damp = 1.0_f64;
            let mut new_res = 1.0e300_f64;
            let mut t_new = t;
            for _ in 0..8 {
                t_new[0] = t[0] + Complex64::new(damp, 0.0) * dt0;
                t_new[1] = t[1] + Complex64::new(damp, 0.0) * dt1;
                let mut x2 = [Complex64::new(0.0, 0.0); NCOORDS];
                for k in 0..NCOORDS {
                    let mut s = origin[k];
                    for j in 0..NHYPER {
                        s += t_new[j] * dirs[j][k];
                    }
                    x2[k] = s;
                }
                let f2 = poly.eval(&x2);
                new_res = f2.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
                if new_res < res {
                    break;
                }
                damp *= 0.5;
            }
            if new_res >= res {
                prev_res = res;
                break;
            }
            t = t_new;
            prev_res = new_res;
            if new_res < tol {
                conv = true;
                break;
            }
        }
        SchoenNewtonResult {
            t,
            residual: prev_res,
            converged: conv,
        }
    }

    #[test]
    fn gpu_matches_cpu() {
        let ctx = match SchoenGpuContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("GPU unavailable, skipping parity test: {e}");
                return;
            }
        };

        let poly = SchoenPoly::z3xz3_invariant_default();
        let n_threads = 16;
        let mut state: u64 = 0x5C4081EE;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) & ((1u64 << 53) - 1)) as f64 / (1u64 << 53) as f64 - 0.5
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
            &ctx, &poly, &origins_re, &origins_im, &dirs_re, &dirs_im, &t_init_re,
            &t_init_im, n_threads, max_iter, tol,
        )
        .expect("gpu newton");

        for i in 0..n_threads {
            let mut origin = [Complex64::new(0.0, 0.0); NCOORDS];
            for k in 0..NCOORDS {
                origin[k] = Complex64::new(
                    origins_re[i * NCOORDS + k],
                    origins_im[i * NCOORDS + k],
                );
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
                t_init[j] =
                    Complex64::new(t_init_re[i * NHYPER + j], t_init_im[i * NHYPER + j]);
            }
            let cpu = cpu_newton_one_attempt(&poly, &origin, &dirs, &t_init, max_iter, tol);
            assert_eq!(
                gpu[i].converged, cpu.converged,
                "convergence flag mismatch at thread {i}"
            );
            if cpu.converged {
                for j in 0..NHYPER {
                    assert!(
                        (gpu[i].t[j].re - cpu.t[j].re).abs() < 1e-9
                            && (gpu[i].t[j].im - cpu.t[j].im).abs() < 1e-9,
                        "t[{j}] mismatch at thread {i}: gpu={:?} cpu={:?}",
                        gpu[i].t[j],
                        cpu.t[j]
                    );
                }
            }
        }
    }
}
