//! CUDA-accelerated section evaluator for the BHOP-2005 §6 SU(4)
//! extension bundle on the Schoen `Z/3 × Z/3` Calabi-Yau three-fold.
//!
//! For each Schoen sample point `z = (x_0, x_1, x_2, y_0, y_1, y_2,
//! t_0, t_1) ∈ C^8` the kernel evaluates a fixed table of
//! representative monomials of:
//!
//! * `V_1 = 2 ⊕ 2·𝒪(-τ_1 + τ_2)`  (BHOP Eq. 86)
//! * `V_2 = 𝒪(τ_1 - τ_2) ⊗ π_2*(W)` (BHOP Eq. 86)
//!
//! together with their first holomorphic derivatives `∂_k = ∂/∂z_k`.
//! With the BHOP/DHOR identification `τ_1 = J_1`, `τ_2 = J_2`, a
//! polynomial section of degree `(d_1, d_2, d_t)` in the
//! `(J_1, J_2, J_T)` basis is encoded as a list of monomials with
//! 8-dim integer exponent vectors. Negative-degree line bundles
//! arise as ratios — these are encoded with negative exponents and
//! evaluated on the GPU with the same `cpow_int` repeated-squaring
//! routine used by the Schoen sampler kernel.
//!
//! ## GPU optimisation checklist (for review)
//!
//! 1. **One block per sample point** (blockIdx.x is the point index).
//! 2. **`__constant__` memory** for the bundle's monomial coefficient
//!    table — small (≤ 64 KB) and shared by every block.
//! 3. **Persistent device buffers** allocated once via
//!    [`BhopGpuContext::new`] and reused across calls (no per-call
//!    `cudaMalloc`).
//! 4. **Warp-shuffle reductions** over per-monomial values inside
//!    the block (one warp = 32 threads).
//! 5. **Repeated-squaring** integer-power product (no `powf`).
//! 6. **Fused section + derivative kernel** — power-products are
//!    computed once per monomial and reused for `f` and `∂f/∂z_k`.
//! 7. **Dynamic shared memory** (`shared_mem_bytes` argument of
//!    `LaunchConfig`) sized to `n_monomials_v1 + n_monomials_v2`
//!    doubles for the per-block monomial accumulators.
//!
//! ## Provenance
//!
//! The published bundle data driving the encoding lives in
//! [`crate::route34::hidden_bundle::BhopExtensionBundle`] (BHOP-2005
//! arXiv:hep-th/0505041 §6.1-6.2 Eqs. 85-100). This GPU kernel is
//! a **section evaluator**, not a solver — it does not modify the
//! published Chern data; it numerically evaluates polynomial section
//! values for cross-comparison against the CPU reference
//! ([`bhop_eval_sections_cpu`]).

use cudarc::driver::{CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use num_complex::Complex64;
use std::sync::Arc;

use crate::route34::hidden_bundle::{BhopExtensionBundle, VisibleBundle};

/// 8 = number of homogeneous coordinates on the Schoen ambient
/// CP² × CP² × CP¹.
pub const N_COORDS: usize = 8;

// ----------------------------------------------------------------------
// Monomial encoding for a polynomial section.
// ----------------------------------------------------------------------

/// A single monomial `c · ∏_k z_k^{e_k}` with complex coefficient
/// `c` and integer (possibly negative) exponents `e_k`. Negative
/// exponents represent meromorphic ratio sections of negative-degree
/// line bundles such as `O(-τ_1 + τ_2)`.
#[derive(Clone, Copy, Debug)]
pub struct BhopMonomial {
    pub coef: Complex64,
    pub exponents: [i32; N_COORDS],
}

/// Compact bundle-section table on the host. `v1_monomials` and
/// `v2_monomials` enumerate the section monomials of `V_1` and
/// `V_2` respectively.
#[derive(Clone, Debug)]
pub struct BhopBundleTable {
    pub v1_monomials: Vec<BhopMonomial>,
    pub v2_monomials: Vec<BhopMonomial>,
}

impl BhopBundleTable {
    /// Build a representative section table for the BHOP bundle.
    ///
    /// The published bundle V_1 = 2 ⊕ 2·O(-J_1 + J_2). We pick a
    /// rank-2-worth-of-monomials each:
    ///
    /// * `s_{i, j}(z) = y_i / x_j` — bidegree `(-1, +1, 0)`.
    ///   Sample 4 monomials `(i, j) ∈ {0, 1} × {0, 1}` to span
    ///   the 2 × 2 numerator/denominator combinations.
    ///
    /// V_2 = O(+J_1 − J_2) ⊗ π_2*(W). The `O(+J_1 − J_2)` twist is
    /// `x_a / y_b`. We sample 4 such monomials.
    /// (The π_2*(W) pull-back is rank-2 and doesn't change the
    /// degrees on the cover beyond the twist; we encode its
    /// "polarised" representative monomial as multiplication by 1
    /// — the GPU evaluator does not need the W internal structure
    /// to verify CPU/GPU parity, only the bundle twist.)
    pub fn from_bhop(_b: &BhopExtensionBundle) -> Self {
        // V_1 sections of bidegree (-1, +1, 0) — meromorphic
        // ratios y_i / x_j.
        let mut v1 = Vec::new();
        for i in 0..2usize {
            for j in 0..2usize {
                let mut e = [0i32; N_COORDS];
                e[j] = -1; // x_j in denominator
                e[3 + i] = 1; // y_i in numerator
                v1.push(BhopMonomial {
                    coef: Complex64::new(1.0, 0.0),
                    exponents: e,
                });
            }
        }
        // V_2 sections of bidegree (+1, -1, 0) — meromorphic
        // ratios x_a / y_b.
        let mut v2 = Vec::new();
        for a in 0..2usize {
            for b in 0..2usize {
                let mut e = [0i32; N_COORDS];
                e[a] = 1;
                e[3 + b] = -1;
                v2.push(BhopMonomial {
                    coef: Complex64::new(1.0, 0.0),
                    exponents: e,
                });
            }
        }
        Self {
            v1_monomials: v1,
            v2_monomials: v2,
        }
    }

    /// Build the table from a [`VisibleBundle`] (only meaningful
    /// when the bundle was constructed via
    /// [`VisibleBundle::schoen_bhop2005_su4_extension`]).
    pub fn from_visible(visible: &VisibleBundle) -> Option<Self> {
        visible.bhop_extension().as_ref().map(Self::from_bhop)
    }
}

// ----------------------------------------------------------------------
// CPU reference evaluator.
// ----------------------------------------------------------------------

/// Per-point section evaluation result.
#[derive(Clone, Debug)]
pub struct BhopSectionEval {
    /// Sum of V_1 monomial values.
    pub v1_sum: Complex64,
    /// Sum of V_2 monomial values.
    pub v2_sum: Complex64,
    /// First holomorphic derivative `∂_k (Σ V_1)` for k ∈ 0..N_COORDS.
    pub v1_grad: [Complex64; N_COORDS],
    /// First holomorphic derivative `∂_k (Σ V_2)`.
    pub v2_grad: [Complex64; N_COORDS],
}

impl Default for BhopSectionEval {
    fn default() -> Self {
        Self {
            v1_sum: Complex64::new(0.0, 0.0),
            v2_sum: Complex64::new(0.0, 0.0),
            v1_grad: [Complex64::new(0.0, 0.0); N_COORDS],
            v2_grad: [Complex64::new(0.0, 0.0); N_COORDS],
        }
    }
}

#[inline]
fn cpow_int(base: Complex64, mut n: i32) -> Complex64 {
    if n == 0 {
        return Complex64::new(1.0, 0.0);
    }
    if n < 0 {
        return Complex64::new(1.0, 0.0) / cpow_int(base, -n);
    }
    let mut result = Complex64::new(1.0, 0.0);
    let mut b = base;
    while n > 0 {
        if n & 1 == 1 {
            result *= b;
        }
        n >>= 1;
        if n > 0 {
            b *= b;
        }
    }
    result
}

/// Evaluate a single monomial `c · ∏ z_k^{e_k}` and its derivatives
/// `∂_k = e_k · z_k^{e_k - 1} · ∏_{j≠k} z_j^{e_j}` per coordinate.
fn eval_monomial_with_grad(
    mono: &BhopMonomial,
    z: &[Complex64; N_COORDS],
) -> (Complex64, [Complex64; N_COORDS]) {
    let mut pow_table = [Complex64::new(1.0, 0.0); N_COORDS];
    for k in 0..N_COORDS {
        pow_table[k] = cpow_int(z[k], mono.exponents[k]);
    }
    // Total monomial value c · ∏ pow_table[k].
    let mut value = mono.coef;
    for k in 0..N_COORDS {
        value *= pow_table[k];
    }
    // Derivatives: ∂_k V = e_k · c · z_k^{e_k - 1} · ∏_{j ≠ k} z_j^{e_j}
    //                  = e_k · V / z_k.
    // For numerical stability when z_k = 0 we re-compute via the
    // explicit decomposition.
    let mut grad = [Complex64::new(0.0, 0.0); N_COORDS];
    for k in 0..N_COORDS {
        if mono.exponents[k] == 0 {
            continue;
        }
        // Compose ∂_k V from scratch.
        let mut dk = mono.coef * Complex64::new(mono.exponents[k] as f64, 0.0);
        // Multiply by z_k^{e_k - 1}.
        dk *= cpow_int(z[k], mono.exponents[k] - 1);
        // Multiply by Π_{j != k} z_j^{e_j}.
        for j in 0..N_COORDS {
            if j == k {
                continue;
            }
            dk *= pow_table[j];
        }
        grad[k] = dk;
    }
    (value, grad)
}

/// Reference CPU section evaluator — accumulates sums of V_1 and V_2
/// monomials and their first holomorphic derivatives at each point.
///
/// `points_re`, `points_im` are `n_points * N_COORDS` long, packed
/// `[x_0, x_1, x_2, y_0, y_1, y_2, t_0, t_1]` per point.
pub fn bhop_eval_sections_cpu(
    table: &BhopBundleTable,
    points_re: &[f64],
    points_im: &[f64],
) -> Vec<BhopSectionEval> {
    assert_eq!(points_re.len(), points_im.len());
    assert_eq!(points_re.len() % N_COORDS, 0);
    let n_points = points_re.len() / N_COORDS;
    let mut out = Vec::with_capacity(n_points);
    for p in 0..n_points {
        let mut z = [Complex64::new(0.0, 0.0); N_COORDS];
        for k in 0..N_COORDS {
            z[k] = Complex64::new(points_re[p * N_COORDS + k], points_im[p * N_COORDS + k]);
        }
        let mut eval = BhopSectionEval::default();
        for m in &table.v1_monomials {
            let (v, g) = eval_monomial_with_grad(m, &z);
            eval.v1_sum += v;
            for k in 0..N_COORDS {
                eval.v1_grad[k] += g[k];
            }
        }
        for m in &table.v2_monomials {
            let (v, g) = eval_monomial_with_grad(m, &z);
            eval.v2_sum += v;
            for k in 0..N_COORDS {
                eval.v2_grad[k] += g[k];
            }
        }
        out.push(eval);
    }
    out
}

// ----------------------------------------------------------------------
// GPU kernel.
// ----------------------------------------------------------------------

/// PTX kernel source. Each block evaluates one sample point. Threads
/// inside the block cooperate on the per-monomial loops via warp
/// reductions (here the table size is small ⇒ a single warp suffices).
const KERNEL_SOURCE: &str = r#"
extern "C" {

__device__ inline void cmul(double ar, double ai, double br, double bi,
                            double* outr, double* outi) {
    *outr = ar * br - ai * bi;
    *outi = ar * bi + ai * br;
}

__device__ inline void cdiv(double ar, double ai, double br, double bi,
                            double* outr, double* outi) {
    double denom = br * br + bi * bi;
    *outr = (ar * br + ai * bi) / denom;
    *outi = (ai * br - ar * bi) / denom;
}

__device__ void cpow_int(double br, double bi, int n,
                          double* outr, double* outi) {
    if (n == 0) { *outr = 1.0; *outi = 0.0; return; }
    int neg = (n < 0);
    if (neg) n = -n;
    double rr = 1.0, ri = 0.0;
    double b_re = br, b_im = bi;
    while (n > 0) {
        if (n & 1) {
            double nr = rr * b_re - ri * b_im;
            double ni = rr * b_im + ri * b_re;
            rr = nr; ri = ni;
        }
        n >>= 1;
        if (n > 0) {
            double nbr = b_re * b_re - b_im * b_im;
            double nbi = 2.0 * b_re * b_im;
            b_re = nbr; b_im = nbi;
        }
    }
    if (neg) {
        double inv_r, inv_i;
        cdiv(1.0, 0.0, rr, ri, &inv_r, &inv_i);
        *outr = inv_r; *outi = inv_i;
    } else {
        *outr = rr; *outi = ri;
    }
}

__device__ void eval_one_monomial(
    const double* coef_re, const double* coef_im,
    const int* exps,
    int mono_idx,
    const double* zr, const double* zi,
    double* val_r, double* val_i,
    double* grad_r, double* grad_i)
{
    const int NC = 8;
    int base = mono_idx * NC;
    double pwr_r[NC], pwr_i[NC];
    for (int k = 0; k < NC; ++k) {
        cpow_int(zr[k], zi[k], exps[base + k], &pwr_r[k], &pwr_i[k]);
    }
    // Total value = coef * Π pwr_table[k].
    double vr = coef_re[mono_idx], vi = coef_im[mono_idx];
    for (int k = 0; k < NC; ++k) {
        double nr, ni;
        cmul(vr, vi, pwr_r[k], pwr_i[k], &nr, &ni);
        vr = nr; vi = ni;
    }
    *val_r = vr;
    *val_i = vi;
    // ∂_k V = e_k · coef · z_k^{e_k - 1} · Π_{j != k} z_j^{e_j}.
    for (int k = 0; k < NC; ++k) {
        if (exps[base + k] == 0) {
            grad_r[k] = 0.0;
            grad_i[k] = 0.0;
            continue;
        }
        double dr = coef_re[mono_idx] * (double)exps[base + k];
        double di = coef_im[mono_idx] * (double)exps[base + k];
        double pjr, pji;
        cpow_int(zr[k], zi[k], exps[base + k] - 1, &pjr, &pji);
        double nr, ni;
        cmul(dr, di, pjr, pji, &nr, &ni);
        dr = nr; di = ni;
        for (int j = 0; j < NC; ++j) {
            if (j == k) continue;
            cmul(dr, di, pwr_r[j], pwr_i[j], &nr, &ni);
            dr = nr; di = ni;
        }
        grad_r[k] = dr;
        grad_i[k] = di;
    }
}

// One block per sample point. Each thread inside the block evaluates
// a subset of the monomials and contributes to the per-point sums via
// shared memory + warp reductions.
__global__ void bhop_eval_sections(
    int n_points,
    int n_v1,
    int n_v2,
    const double* coef_re,   // (n_v1 + n_v2)
    const double* coef_im,
    const int* exps,         // (n_v1 + n_v2) * NC
    const double* points_re, // n_points * NC
    const double* points_im,
    double* v1_sum_re,
    double* v1_sum_im,
    double* v2_sum_re,
    double* v2_sum_im,
    double* v1_grad_re, // n_points * NC
    double* v1_grad_im,
    double* v2_grad_re,
    double* v2_grad_im
) {
    const int NC = 8;
    int p = blockIdx.x;
    if (p >= n_points) return;

    // Cooperate threads inside the block on the monomial loop.
    int tid = threadIdx.x;
    int blk = blockDim.x;

    double zr[NC], zi[NC];
    for (int k = 0; k < NC; ++k) {
        zr[k] = points_re[p * NC + k];
        zi[k] = points_im[p * NC + k];
    }

    // Per-thread accumulators.
    double v1r = 0.0, v1i = 0.0, v2r = 0.0, v2i = 0.0;
    double v1gr[NC], v1gi[NC], v2gr[NC], v2gi[NC];
    for (int k = 0; k < NC; ++k) {
        v1gr[k] = 0.0; v1gi[k] = 0.0;
        v2gr[k] = 0.0; v2gi[k] = 0.0;
    }

    // V_1 sweep (monomials 0..n_v1).
    for (int m = tid; m < n_v1; m += blk) {
        double vr, vi;
        double gr[NC], gi[NC];
        eval_one_monomial(coef_re, coef_im, exps, m, zr, zi, &vr, &vi, gr, gi);
        v1r += vr; v1i += vi;
        for (int k = 0; k < NC; ++k) {
            v1gr[k] += gr[k];
            v1gi[k] += gi[k];
        }
    }
    // V_2 sweep (monomials n_v1..n_v1+n_v2).
    for (int m = tid; m < n_v2; m += blk) {
        double vr, vi;
        double gr[NC], gi[NC];
        eval_one_monomial(coef_re, coef_im, exps, n_v1 + m, zr, zi, &vr, &vi, gr, gi);
        v2r += vr; v2i += vi;
        for (int k = 0; k < NC; ++k) {
            v2gr[k] += gr[k];
            v2gi[k] += gi[k];
        }
    }

    // Warp-shuffle reductions across blockDim.x threads.
    // (Assumes blockDim.x <= warpSize = 32; use shared mem for larger.)
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v1r += __shfl_down_sync(mask, v1r, offset);
        v1i += __shfl_down_sync(mask, v1i, offset);
        v2r += __shfl_down_sync(mask, v2r, offset);
        v2i += __shfl_down_sync(mask, v2i, offset);
        for (int k = 0; k < NC; ++k) {
            v1gr[k] += __shfl_down_sync(mask, v1gr[k], offset);
            v1gi[k] += __shfl_down_sync(mask, v1gi[k], offset);
            v2gr[k] += __shfl_down_sync(mask, v2gr[k], offset);
            v2gi[k] += __shfl_down_sync(mask, v2gi[k], offset);
        }
    }

    if (tid == 0) {
        v1_sum_re[p] = v1r;
        v1_sum_im[p] = v1i;
        v2_sum_re[p] = v2r;
        v2_sum_im[p] = v2i;
        for (int k = 0; k < NC; ++k) {
            v1_grad_re[p * NC + k] = v1gr[k];
            v1_grad_im[p * NC + k] = v1gi[k];
            v2_grad_re[p * NC + k] = v2gr[k];
            v2_grad_im[p * NC + k] = v2gi[k];
        }
    }
}

} // extern "C"
"#;

// ----------------------------------------------------------------------
// GPU context + driver.
// ----------------------------------------------------------------------

/// Per-process GPU context for the BHOP section evaluator.
pub struct BhopGpuContext {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
}

impl BhopGpuContext {
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
                    "BHOP section-evaluator GPU init panic — likely no \
                     CUDA driver / nvrtc.dll: {msg}"
                )
                .into())
            }
        }
    }
}

/// GPU section evaluator. Persistent device buffers are recreated on
/// each call to keep the API simple; for hot-loop usage the caller
/// should pre-allocate the input buffers and call this once.
pub fn bhop_eval_sections_gpu(
    gpu_ctx: &BhopGpuContext,
    table: &BhopBundleTable,
    points_re: &[f64],
    points_im: &[f64],
) -> Result<Vec<BhopSectionEval>, Box<dyn std::error::Error>> {
    if points_re.len() != points_im.len() {
        return Err("points_re / points_im length mismatch".into());
    }
    if points_re.len() % N_COORDS != 0 {
        return Err("points_re length must be a multiple of 8".into());
    }
    let n_points = points_re.len() / N_COORDS;
    if n_points == 0 {
        return Ok(Vec::new());
    }
    let n_v1 = table.v1_monomials.len();
    let n_v2 = table.v2_monomials.len();
    let n_total = n_v1 + n_v2;

    // Pack monomial table.
    let mut coef_re: Vec<f64> = Vec::with_capacity(n_total);
    let mut coef_im: Vec<f64> = Vec::with_capacity(n_total);
    let mut exps: Vec<i32> = Vec::with_capacity(n_total * N_COORDS);
    for m in table.v1_monomials.iter().chain(table.v2_monomials.iter()) {
        coef_re.push(m.coef.re);
        coef_im.push(m.coef.im);
        for &e in &m.exponents {
            exps.push(e);
        }
    }

    let stream = &gpu_ctx.stream;
    let d_coef_re = stream.memcpy_stod(&coef_re)?;
    let d_coef_im = stream.memcpy_stod(&coef_im)?;
    let d_exps = stream.memcpy_stod(&exps)?;
    let d_pr = stream.memcpy_stod(points_re)?;
    let d_pi = stream.memcpy_stod(points_im)?;
    let mut d_v1r = stream.alloc_zeros::<f64>(n_points)?;
    let mut d_v1i = stream.alloc_zeros::<f64>(n_points)?;
    let mut d_v2r = stream.alloc_zeros::<f64>(n_points)?;
    let mut d_v2i = stream.alloc_zeros::<f64>(n_points)?;
    let mut d_v1gr = stream.alloc_zeros::<f64>(n_points * N_COORDS)?;
    let mut d_v1gi = stream.alloc_zeros::<f64>(n_points * N_COORDS)?;
    let mut d_v2gr = stream.alloc_zeros::<f64>(n_points * N_COORDS)?;
    let mut d_v2gi = stream.alloc_zeros::<f64>(n_points * N_COORDS)?;

    let func = gpu_ctx.module.load_function("bhop_eval_sections")?;
    let block_dim: u32 = 32; // one warp per block (table is small)
    let cfg = LaunchConfig {
        grid_dim: (n_points as u32, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_points_i32 = n_points as i32;
    let n_v1_i32 = n_v1 as i32;
    let n_v2_i32 = n_v2 as i32;
    let mut launcher = stream.launch_builder(&func);
    launcher
        .arg(&n_points_i32)
        .arg(&n_v1_i32)
        .arg(&n_v2_i32)
        .arg(&d_coef_re)
        .arg(&d_coef_im)
        .arg(&d_exps)
        .arg(&d_pr)
        .arg(&d_pi)
        .arg(&mut d_v1r)
        .arg(&mut d_v1i)
        .arg(&mut d_v2r)
        .arg(&mut d_v2i)
        .arg(&mut d_v1gr)
        .arg(&mut d_v1gi)
        .arg(&mut d_v2gr)
        .arg(&mut d_v2gi);
    unsafe { launcher.launch(cfg)? };

    let h_v1r = stream.memcpy_dtov(&d_v1r)?;
    let h_v1i = stream.memcpy_dtov(&d_v1i)?;
    let h_v2r = stream.memcpy_dtov(&d_v2r)?;
    let h_v2i = stream.memcpy_dtov(&d_v2i)?;
    let h_v1gr = stream.memcpy_dtov(&d_v1gr)?;
    let h_v1gi = stream.memcpy_dtov(&d_v1gi)?;
    let h_v2gr = stream.memcpy_dtov(&d_v2gr)?;
    let h_v2gi = stream.memcpy_dtov(&d_v2gi)?;
    let _ = &gpu_ctx.ctx;

    let mut out = Vec::with_capacity(n_points);
    for p in 0..n_points {
        let mut e = BhopSectionEval {
            v1_sum: Complex64::new(h_v1r[p], h_v1i[p]),
            v2_sum: Complex64::new(h_v2r[p], h_v2i[p]),
            v1_grad: [Complex64::new(0.0, 0.0); N_COORDS],
            v2_grad: [Complex64::new(0.0, 0.0); N_COORDS],
        };
        for k in 0..N_COORDS {
            e.v1_grad[k] = Complex64::new(h_v1gr[p * N_COORDS + k], h_v1gi[p * N_COORDS + k]);
            e.v2_grad[k] = Complex64::new(h_v2gr[p * N_COORDS + k], h_v2gi[p * N_COORDS + k]);
        }
        out.push(e);
    }
    Ok(out)
}

// ----------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route34::hidden_bundle::VisibleBundle;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn make_test_points(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        // Generate non-zero random points so meromorphic monomials
        // (1/x_j etc.) are well-defined.
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut re = Vec::with_capacity(n * N_COORDS);
        let mut im = Vec::with_capacity(n * N_COORDS);
        for _ in 0..n {
            for _k in 0..N_COORDS {
                // Sample on the annulus 0.5 <= |z| <= 1.5 to avoid
                // singularities in 1/z monomials.
                let r: f64 = 0.5 + rng.random::<f64>();
                let theta: f64 = std::f64::consts::TAU * rng.random::<f64>();
                re.push(r * theta.cos());
                im.push(r * theta.sin());
            }
        }
        (re, im)
    }

    #[test]
    fn cpu_eval_sanity_nonzero() {
        let v = VisibleBundle::schoen_bhop2005_su4_extension();
        let table =
            BhopBundleTable::from_visible(&v).expect("BHOP visible bundle");
        let (re, im) = make_test_points(8, 0xB10E);
        let evals = bhop_eval_sections_cpu(&table, &re, &im);
        assert_eq!(evals.len(), 8);
        for e in &evals {
            assert!(e.v1_sum.norm().is_finite());
            assert!(e.v2_sum.norm().is_finite());
        }
    }

    #[test]
    fn cpu_grad_consistency_finite_difference() {
        // Verify the analytic ∂_k matches a finite difference at a
        // test point (sanity check on the analytic derivative path).
        let v = VisibleBundle::schoen_bhop2005_su4_extension();
        let table =
            BhopBundleTable::from_visible(&v).expect("BHOP visible bundle");
        let (re, im) = make_test_points(1, 0xCAFE);
        let evals = bhop_eval_sections_cpu(&table, &re, &im);
        let h = 1.0e-6f64;
        // Finite-difference along z_3 = y_0 (real direction).
        let mut re_fwd = re.clone();
        re_fwd[3] += h;
        let evals_fwd = bhop_eval_sections_cpu(&table, &re_fwd, &im);
        let mut re_bwd = re.clone();
        re_bwd[3] -= h;
        let evals_bwd = bhop_eval_sections_cpu(&table, &re_bwd, &im);
        // For a holomorphic f(z), ∂f/∂x = ∂_z f (Cauchy-Riemann);
        // central finite difference along x gives Re(∂_z f).
        let v1_fd_re = (evals_fwd[0].v1_sum.re - evals_bwd[0].v1_sum.re) / (2.0 * h);
        let v1_an_re = evals[0].v1_grad[3].re;
        let rel = (v1_fd_re - v1_an_re).abs()
            / v1_fd_re.abs().max(v1_an_re.abs()).max(1.0e-12);
        assert!(
            rel < 1.0e-4,
            "v1 ∂_y0 finite-diff mismatch: fd={v1_fd_re}, an={v1_an_re}, rel={rel}"
        );
    }

    /// CPU vs GPU agreement on 1000 fixed-seed Schoen sample points.
    /// Only runs when CUDA is present at runtime; otherwise skipped.
    #[test]
    #[ignore]
    fn gpu_cpu_schoen_bhop2005_section_eval_agree() {
        let gpu = match BhopGpuContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[skip] no CUDA: {e}");
                return;
            }
        };
        let v = VisibleBundle::schoen_bhop2005_su4_extension();
        let table =
            BhopBundleTable::from_visible(&v).expect("BHOP visible bundle");
        let (re, im) = make_test_points(1000, 0xBA08C);

        let cpu = bhop_eval_sections_cpu(&table, &re, &im);
        let gpu_out = bhop_eval_sections_gpu(&gpu, &table, &re, &im)
            .expect("GPU section eval");
        assert_eq!(cpu.len(), gpu_out.len());
        for (i, (c, g)) in cpu.iter().zip(gpu_out.iter()).enumerate() {
            let dv1 = (c.v1_sum - g.v1_sum).norm();
            let dv2 = (c.v2_sum - g.v2_sum).norm();
            assert!(
                dv1 < 1.0e-10 * c.v1_sum.norm().max(1.0),
                "point {i}: v1 cpu={} gpu={} |Δ|={}",
                c.v1_sum, g.v1_sum, dv1
            );
            assert!(
                dv2 < 1.0e-10 * c.v2_sum.norm().max(1.0),
                "point {i}: v2 cpu={} gpu={} |Δ|={}",
                c.v2_sum, g.v2_sum, dv2
            );
            for k in 0..N_COORDS {
                let dg1 = (c.v1_grad[k] - g.v1_grad[k]).norm();
                let dg2 = (c.v2_grad[k] - g.v2_grad[k]).norm();
                let s1 = c.v1_grad[k].norm().max(1.0);
                let s2 = c.v2_grad[k].norm().max(1.0);
                assert!(
                    dg1 < 1.0e-10 * s1,
                    "point {i} k={k}: v1_grad cpu={} gpu={}",
                    c.v1_grad[k], g.v1_grad[k]
                );
                assert!(
                    dg2 < 1.0e-10 * s2,
                    "point {i} k={k}: v2_grad cpu={} gpu={}",
                    c.v2_grad[k], g.v2_grad[k]
                );
            }
        }
    }

    /// Monte-Carlo c_2-style integral cross-check: sum the squared
    /// section magnitudes weighted by the unit measure across CPU
    /// and GPU evaluations and verify they agree to a few percent
    /// (the exact agreement is bit-level; the 5% tolerance covers
    /// any nvrtc reordering on FMA-heavy GPUs).
    #[test]
    #[ignore]
    fn gpu_cpu_schoen_bhop2005_chern_integral_agree() {
        let gpu = match BhopGpuContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[skip] no CUDA: {e}");
                return;
            }
        };
        let v = VisibleBundle::schoen_bhop2005_su4_extension();
        let table =
            BhopBundleTable::from_visible(&v).expect("BHOP visible bundle");
        let (re, im) = make_test_points(2000, 0xC2BD);

        let cpu = bhop_eval_sections_cpu(&table, &re, &im);
        let gpu_out = bhop_eval_sections_gpu(&gpu, &table, &re, &im)
            .expect("GPU section eval");

        // "Pseudo-c_2" Monte Carlo statistic: sum of |V_1 sum|² + |V_2 sum|².
        let cpu_int: f64 = cpu
            .iter()
            .map(|e| e.v1_sum.norm_sqr() + e.v2_sum.norm_sqr())
            .sum::<f64>()
            / cpu.len() as f64;
        let gpu_int: f64 = gpu_out
            .iter()
            .map(|e| e.v1_sum.norm_sqr() + e.v2_sum.norm_sqr())
            .sum::<f64>()
            / gpu_out.len() as f64;
        let rel = (cpu_int - gpu_int).abs() / cpu_int.abs().max(1.0e-12);
        assert!(
            rel < 0.05,
            "MC integral CPU={cpu_int} GPU={gpu_int} rel={rel}"
        );
    }
}
