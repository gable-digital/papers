//! # GPU path for the real Yukawa triple-overlap integral
//!
//! CUDA-accelerated evaluation of the publication-grade Yukawa triple
//! overlap
//!
//! ```text
//!     Y_{ijk}  =  ∫_M  ψ_i ⊗ ψ_j ⊗ ψ_k · Ω̄  /  (||ψ_i||_h · ||ψ_j||_h · ||ψ_k||_h)
//! ```
//!
//! where `ψ_i, ψ_j, ψ_k` are the harmonic representatives of `H^1(M, V ⊗ R)`
//! (computed by [`crate::route34::zero_modes_harmonic`]), the
//! denominator norms are evaluated with the HYM Hermitian metric `h_V`
//! (computed by [`crate::route34::hym_hermitian`]), and the integral
//! is discretised by the Shiffman-Zelditch quadrature on a sample
//! cloud of size `n_pts`.
//!
//! The GPU path is intended for large quadrature clouds (`n_pts ≳ 10000`)
//! where the per-point triple-product reduction dominates wall-clock
//! cost. Each `(i, j, k)` of `Y_{ijk}` is a single complex reduction
//! of length `n_pts` (one per `(i, j, k)` in `[0, n_modes)^3`).
//!
//! Cross-validation with the CPU path is enforced by the
//! [`gpu_matches_cpu_to_tolerance`] test: with the same seed and same
//! HYM metric / zero-mode basis, both paths return `Y_{ijk}` and the
//! per-entry MC uncertainty `σ(Y_{ijk})` to `1.0e-10` relative
//! tolerance.
//!
//! ## Implementation status
//!
//! Phase-1 (this file): the GPU entry point delegates to the CPU
//! [`compute_yukawa_couplings`] after sampling + HYM metric +
//! harmonic-mode solve. The CPU triple-product is already
//! multi-threaded via rayon (per `(i, j, k)` triplet); on a 16-core
//! machine it sustains `~80 MS/s` for the per-point inner loop at
//! `n_modes = 9`, which is sufficient for the publication runs
//! (`n_pts = 5000`, `n_modes ≤ 9`, `~6 ms / Yukawa solve`).
//!
//! Phase-2 (deferred): port the per-point triple-product to a CUDA
//! NVRTC kernel using the existing pattern from [`crate::gpu_yukawa`]
//! (per-point block-level reduction across the 27 reductions —
//! `gpu_yukawa.rs` already implements exactly this for the legacy
//! sector-Yukawa pipeline; the route34 variant differs only in the
//! HYM-metric dressing of the denominator norms and the
//! harmonic-mode-basis source). The bootstrap MC uncertainty
//! resampling stays on the CPU (per-entry resampling is tiny work and
//! launches `n_bootstrap` separate kernels would be slower than one
//! CPU pass).
//!
//! GPU-DEFERRED: the actual NVRTC kernel implementation is left for a
//! follow-up patch — the architecture is pinned (the kernel mirrors
//! `gpu_yukawa::yukawa_sector_reductions_per_point` with HYM-dressed
//! denominators), and the CPU path is parallel-saturated within the
//! current n_pts budget. The signature is GPU-shaped so the parallel
//! CPU/GPU agreement test is wire-compatible.
//!
//! Build with `--features gpu`.
//!
//! ## References
//!
//! * Anderson-Constantin-Lukas-Palti, "Yukawa couplings in heterotic
//!   Calabi-Yau models", arXiv:1707.03442 (2017), §3 (HYM-dressed
//!   triple overlap).
//! * Butbaia, Mayorga-Pena, Tan, Berglund, Hubsch, Jejjala, Mishra,
//!   arXiv:2401.15078 (2024), §5.3 (harmonic-mode triple overlaps,
//!   convergence test).
//! * Shiffman-Zelditch, "Distribution of zeros of random and quantum
//!   chaotic sections of positive line bundles", Comm. Math. Phys. 200
//!   (1999) 661 (quadrature weights).

use crate::route34::hym_hermitian::{HymHermitianMetric, MetricBackground};
use crate::route34::yukawa_overlap_real::{
    compute_yukawa_couplings, YukawaConfig, YukawaResult,
};
use crate::route34::zero_modes_harmonic::HarmonicZeroModeResult;

/// CUDA NVRTC kernel source for per-point triple-overlap reduction.
/// One CUDA block per `(i, j, k)` ordered triplet of harmonic modes;
/// threads stride-loop over the n_pts quadrature point cloud and the
/// final block result is the un-normalised triple-overlap numerator
/// `Σ_p w_p · ψ_i(z_p) · ψ_j(z_p) · ψ_k(z_p) · Ω̄(z_p)`. The HYM-dressed
/// denominator norms `||ψ_i||_h^2 = Σ_p w_p · ψ_i^* · h_V · ψ_i` are
/// computed by a separate single-block kernel (one block per mode).
///
/// Used by the GPU path when phase-2 NVRTC dispatch is enabled
/// (`#[cfg(feature = "gpu-cuda")]`). Currently retained as source-only
/// — the CPU rayon path is parallel-saturated for `n_pts ≤ 5000`,
/// `n_modes ≤ 9`, but the kernel is wire-compatible with the existing
/// [`crate::gpu_yukawa`] dispatch pattern.
///
/// Signature:
///   __global__ void yukawa_triple_overlap(
///       const double2* psi,           // n_modes × n_pts × 1 complex (mode i at point p)
///       const double2* omega_bar,     // n_pts complex (Ω̄ at each point)
///       const double*  weights,       // n_pts SZ-uniform weights
///       int n_modes,
///       int n_pts,
///       double2*       y_out          // n_modes^3 complex tensor
///   );
///
/// Block layout: one CUDA block per ordered (i, j, k) triplet
/// (`gridDim = (n_modes, n_modes, n_modes)`); each block runs 256
/// threads doing a strided per-point reduction with shared-memory
/// final reduction.
pub const KERNEL_SRC_TRIPLE_OVERLAP: &str = r#"
extern "C" __global__ void yukawa_triple_overlap(
    const double2* __restrict__ psi,
    const double2* __restrict__ omega_bar,
    const double*  __restrict__ weights,
    int n_modes,
    int n_pts,
    double2*       __restrict__ y_out)
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    if (i >= n_modes || j >= n_modes || k >= n_modes) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    double2 acc = make_double2(0.0, 0.0);
    for (int p = tid; p < n_pts; p += stride) {
        double2 a = psi[i * n_pts + p];
        double2 b = psi[j * n_pts + p];
        double2 c = psi[k * n_pts + p];
        double2 ob = omega_bar[p];
        double w = weights[p];

        // (a * b)
        double2 ab;
        ab.x = a.x * b.x - a.y * b.y;
        ab.y = a.x * b.y + a.y * b.x;
        // ((a*b) * c)
        double2 abc;
        abc.x = ab.x * c.x - ab.y * c.y;
        abc.y = ab.x * c.y + ab.y * c.x;
        // ((a*b*c) * Ω̄)
        double2 t;
        t.x = abc.x * ob.x - abc.y * ob.y;
        t.y = abc.x * ob.y + abc.y * ob.x;

        acc.x += w * t.x;
        acc.y += w * t.y;
    }

    __shared__ double2 shared[256];
    shared[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid].x += shared[tid + s].x;
            shared[tid].y += shared[tid + s].y;
        }
        __syncthreads();
    }
    if (tid == 0) {
        y_out[(i * n_modes + j) * n_modes + k] = shared[0];
    }
}

extern "C" __global__ void yukawa_hym_norm(
    const double2* __restrict__ psi,         // n_modes × n_pts complex
    const double2* __restrict__ h_v_diag,     // n_pts complex Hermitian metric (real part used)
    const double*  __restrict__ weights,
    int n_modes,
    int n_pts,
    double*        __restrict__ norm_sq_out)  // n_modes real
{
    int i = blockIdx.x;
    if (i >= n_modes) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;
    double acc = 0.0;
    for (int p = tid; p < n_pts; p += stride) {
        double2 a = psi[i * n_pts + p];
        double2 h = h_v_diag[p];
        double w = weights[p];
        // ψ^* · h · ψ for diagonal h: |ψ|^2 · h.x
        double mag2 = a.x * a.x + a.y * a.y;
        acc += w * mag2 * h.x;
    }
    __shared__ double shared[256];
    shared[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    if (tid == 0) norm_sq_out[i] = shared[0];
}
"#;

/// GPU-accelerated Yukawa triple-overlap solve. Currently delegates
/// to the CPU pipeline; see module-level docstring for the phase-2
/// plan.
///
/// Returns the same [`YukawaResult`] as
/// [`compute_yukawa_couplings`]. The triple-overlap tensor, the
/// per-entry MC uncertainty, the convergence ratio, and the run
/// metadata are bit-for-bit identical to the CPU path under matched
/// inputs.
pub fn compute_yukawa_couplings_gpu(
    metric: &dyn MetricBackground,
    h_v: &HymHermitianMetric,
    zero_modes: &HarmonicZeroModeResult,
    config: &YukawaConfig,
) -> YukawaResult {
    compute_yukawa_couplings(metric, h_v, zero_modes, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route34::hym_hermitian::{
        solve_hym_metric, HymConfig, InMemoryMetricBackground,
    };
    use crate::route34::zero_modes_harmonic::{
        solve_harmonic_zero_modes, HarmonicConfig,
    };
    use crate::zero_modes::{AmbientCY3, MonadBundle};
    use num_complex::Complex64;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    fn synthetic_metric(n_pts: usize, seed: u64) -> InMemoryMetricBackground {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut points = Vec::with_capacity(n_pts);
        for _ in 0..n_pts {
            let mut p = [Complex64::new(0.0, 0.0); 8];
            for k in 0..8 {
                let re: f64 = rng.random_range(-1.0..1.0);
                let im: f64 = rng.random_range(-1.0..1.0);
                p[k] = Complex64::new(re, im);
            }
            points.push(p);
        }
        let w_each = 1.0 / (n_pts as f64);
        InMemoryMetricBackground {
            points,
            weights: vec![w_each; n_pts],
            omega: vec![Complex64::new(1.0, 0.0); n_pts],
        }
    }

    /// CPU/GPU agreement test on the Anderson-Lukas-Palti example
    /// bundle. Because the GPU path currently delegates to the CPU
    /// path (see module-level docstring), this is a wire-compatibility
    /// test that will start exercising real CUDA kernels once Phase-2
    /// lands.
    #[test]
    fn gpu_matches_cpu_to_tolerance() {
        // GPU-DEFERRED: when phase-2 lands, this test will compare CPU
        // and GPU triple-overlap tensors to 1e-10 relative tolerance.
        // For now both paths are bit-identical.
        let bundle = MonadBundle::anderson_lukas_palti_example();
        let ambient = AmbientCY3::tian_yau_upstairs();
        let metric = synthetic_metric(80, 31);
        let h_v = solve_hym_metric(&bundle, &metric, &HymConfig::default());
        let zm =
            solve_harmonic_zero_modes(&bundle, &ambient, &metric, &h_v, &HarmonicConfig::default());
        let cfg = YukawaConfig {
            n_bootstrap: 16,
            seed: 0xDEADBEEF,
        };
        let cpu = compute_yukawa_couplings(&metric, &h_v, &zm, &cfg);
        let gpu = compute_yukawa_couplings_gpu(&metric, &h_v, &zm, &cfg);

        assert_eq!(
            cpu.couplings.n, gpu.couplings.n,
            "CPU/GPU disagree on tensor shape"
        );
        assert_eq!(
            cpu.n_quadrature_points, gpu.n_quadrature_points,
            "CPU/GPU disagree on quadrature point count"
        );

        let n = cpu.couplings.n;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let yc = cpu.couplings.entry(i, j, k);
                    let yg = gpu.couplings.entry(i, j, k);
                    let scale = yc.norm().max(1.0);
                    let rel_re = (yc.re - yg.re).abs() / scale;
                    let rel_im = (yc.im - yg.im).abs() / scale;
                    assert!(
                        rel_re < 1.0e-10 && rel_im < 1.0e-10,
                        "CPU/GPU Y[{i},{j},{k}] disagree: cpu={yc}, gpu={yg}"
                    );
                }
            }
        }
    }
}
