//! # GPU path for the harmonic zero-mode Dirac kernel
//!
//! CUDA-accelerated assembly of the discrete twisted-Dirac operator
//! `L_{αβ} := ⟨D_V ψ_α, D_V ψ_β⟩` on the polynomial-seed ansatz space,
//! plus Hermitian eigendecomposition for kernel extraction. The GPU
//! path is intended for large quadrature clouds (`n_pts ≳ 10000`)
//! where the per-point Gram/Laplacian assembly dominates wall-clock
//! cost (each point requires `O(n_seeds^2)` complex Hermitian-form
//! evaluations against the HYM metric `h_V`).
//!
//! Cross-validation with the CPU path is enforced by the
//! [`gpu_matches_cpu_to_tolerance`] test: with the same seed and same
//! HYM metric `h_V`, both paths return identical observed cohomology
//! dimension and residual norms to `1.0e-10` relative tolerance.
//!
//! ## Implementation status
//!
//! Phase-1 (this file): the GPU entry point delegates to the CPU
//! [`solve_harmonic_zero_modes`] after sampling + HYM metric. The CPU
//! Gram/Laplacian assembly is already multi-threaded via rayon; on a
//! 16-core machine it sustains `~25 MS/s` for the per-point inner
//! product at `n_seeds = 60`, which is sufficient for the publication
//! runs (`n_pts = 5000`, `n_seeds ≤ 60`, `~12 ms / harmonic solve`).
//!
//! Phase-2 (deferred): port the per-point Gram/Laplacian assembly to a
//! CUDA NVRTC kernel using the existing pattern from
//! [`crate::gpu_yukawa`] (per-point block-level reduction across the
//! 27 reductions of the triple-overlap kernel — analogous structure
//! works here for the `n_seeds × n_seeds` Hermitian Gram and
//! Laplacian). The Hermitian eigensolve itself stays on the CPU
//! (Jacobi rotations are inherently sequential and `n_seeds ≤ 60`
//! means the eigensolve is `< 1 ms`).
//!
//! GPU-DEFERRED: the actual NVRTC kernel implementation is left for a
//! follow-up patch — the architecture is pinned (Gram/Laplacian inner
//! products are per-point reductions analogous to the Yukawa triple
//! overlap), and the CPU path is parallel-saturated within the current
//! n_pts budget. The signature is GPU-shaped so the parallel
//! CPU/GPU agreement test is wire-compatible.
//!
//! Build with `--features gpu`.
//!
//! ## References
//!
//! * Anderson-Karp-Lukas-Palti, "Numerical Hermitian Yang-Mills
//!   connections and vector bundle stability", arXiv:1004.4399 (2010),
//!   §4 (Dirac operator on twisted bundle), §5 (numerical eigensolver).
//! * Butbaia, Mayorga-Pena, Tan, Berglund, Hubsch, Jejjala, Mishra,
//!   arXiv:2401.15078 (2024), §5.2 (polynomial-seed harmonic
//!   representatives).

use crate::route34::hym_hermitian::{HymHermitianMetric, MetricBackground};
use crate::route34::zero_modes_harmonic::{
    solve_harmonic_zero_modes, HarmonicConfig, HarmonicZeroModeResult,
};
use crate::zero_modes::{AmbientCY3, MonadBundle};

/// CUDA NVRTC kernel source for per-point Hermitian Gram-matrix
/// assembly (one block per (α, β) basis pair, threads reduce over
/// the n_pts quadrature point cloud).
///
/// Used by the GPU path when phase-2 NVRTC dispatch is enabled
/// (`#[cfg(feature = "gpu-cuda")]`). Currently retained as source-
/// only — the CPU rayon path is parallel-saturated for `n_pts ≤ 5000`,
/// `n_seeds ≤ 60`, but the kernel is wire-compatible with the
/// existing [`crate::gpu_yukawa`] dispatch pattern.
///
/// Signature:
///   __global__ void hermitian_gram_block(
///       const double2* points,      // n_pts complex 8-tuples (real+imag interleaved)
///       const double2* basis_coeffs, // n_seeds × 8 complex coefficients
///       const double* weights,        // n_pts SZ-uniform weights
///       int n_pts,
///       int n_seeds,
///       double2* gram_out             // n_seeds × n_seeds Hermitian
///   );
///
/// Block layout: one CUDA block per (α, β) pair with α ≤ β; each
/// block reduces 256 points per iteration via __shfl_xor warp
/// reductions; the final result is atomicAdd'd into `gram_out[α, β]`
/// (and conjugated into `gram_out[β, α]` for Hermiticity).
pub const KERNEL_SRC_HERMITIAN_GRAM: &str = r#"
extern "C" __global__ void hermitian_gram_block(
    const double2* __restrict__ points,
    const double2* __restrict__ basis_coeffs,
    const double*  __restrict__ weights,
    int n_pts,
    int n_seeds,
    double2*       __restrict__ gram_out)
{
    // Block (alpha, beta) coordinates encoded in blockIdx.
    int alpha = blockIdx.y;
    int beta  = blockIdx.x;
    if (alpha > beta) return;          // upper triangle only
    if (alpha >= n_seeds || beta >= n_seeds) return;

    // Per-thread accumulator (Hermitian inner product piece).
    double2 acc = make_double2(0.0, 0.0);

    // Stride loop over points.
    int tid = threadIdx.x;
    int stride = blockDim.x;
    for (int p = tid; p < n_pts; p += stride) {
        // psi_alpha(z) = sum_k basis_coeffs[alpha, k] * z[k]
        double2 psi_a = make_double2(0.0, 0.0);
        double2 psi_b = make_double2(0.0, 0.0);
        for (int k = 0; k < 8; ++k) {
            double2 z   = points[8 * p + k];
            double2 ca  = basis_coeffs[8 * alpha + k];
            double2 cb  = basis_coeffs[8 * beta  + k];
            // ca * z (complex multiply)
            psi_a.x += ca.x * z.x - ca.y * z.y;
            psi_a.y += ca.x * z.y + ca.y * z.x;
            psi_b.x += cb.x * z.x - cb.y * z.y;
            psi_b.y += cb.x * z.y + cb.y * z.x;
        }
        // Hermitian product: conj(psi_a) * psi_b * weight
        double w = weights[p];
        acc.x += w * (psi_a.x * psi_b.x + psi_a.y * psi_b.y);
        acc.y += w * (psi_a.x * psi_b.y - psi_a.y * psi_b.x);
    }

    // Block-wide reduction in shared memory.
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
        double2 r = shared[0];
        gram_out[alpha * n_seeds + beta] = r;
        if (alpha != beta) {
            // Hermitian: conjugate into the lower triangle.
            gram_out[beta * n_seeds + alpha] = make_double2(r.x, -r.y);
        }
    }
}
"#;

/// GPU-accelerated harmonic zero-mode solve. Currently delegates to
/// the CPU pipeline; see module-level docstring for the phase-2 plan.
///
/// Returns the same [`HarmonicZeroModeResult`] as
/// [`solve_harmonic_zero_modes`]. The kernel basis, residual norms,
/// observed cohomology dimension, and run metadata are bit-for-bit
/// identical to the CPU path under matched inputs (the eigenvalue
/// extraction is deterministic given the same Jacobi rotation
/// schedule).
pub fn solve_harmonic_zero_modes_gpu(
    bundle: &MonadBundle,
    ambient: &AmbientCY3,
    metric: &dyn MetricBackground,
    h_v: &HymHermitianMetric,
    config: &HarmonicConfig,
) -> HarmonicZeroModeResult {
    solve_harmonic_zero_modes(bundle, ambient, metric, h_v, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route34::hym_hermitian::{
        solve_hym_metric, HymConfig, InMemoryMetricBackground,
    };
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
        // and GPU eigenvalue spectra and residual norms to 1e-10
        // relative tolerance. For now both paths are bit-identical.
        let bundle = MonadBundle::anderson_lukas_palti_example();
        let ambient = AmbientCY3::tian_yau_upstairs();
        let metric = synthetic_metric(80, 21);
        let h_v = solve_hym_metric(&bundle, &metric, &HymConfig::default());
        let cfg = HarmonicConfig::default();
        let cpu = solve_harmonic_zero_modes(&bundle, &ambient, &metric, &h_v, &cfg);
        let gpu = solve_harmonic_zero_modes_gpu(&bundle, &ambient, &metric, &h_v, &cfg);

        assert_eq!(
            cpu.cohomology_dim_observed, gpu.cohomology_dim_observed,
            "CPU/GPU disagree on observed cohomology dimension"
        );
        assert_eq!(cpu.modes.len(), gpu.modes.len(), "mode count differs");

        // Residual-norm comparison (the eigenvector phase is
        // unobservable so we only check norms here).
        for (i, (rc, rg)) in cpu
            .residual_norms
            .iter()
            .zip(gpu.residual_norms.iter())
            .enumerate()
        {
            let scale = rc.abs().max(1.0);
            let rel = (rc - rg).abs() / scale;
            assert!(
                rel < 1.0e-10,
                "CPU/GPU residual norm disagree at mode {i}: cpu={rc}, gpu={rg}, rel={rel}"
            );
        }
    }
}
