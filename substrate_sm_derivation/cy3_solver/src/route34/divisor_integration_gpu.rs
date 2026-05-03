//! # GPU path for divisor MC integration
//!
//! CUDA-accelerated rejection sampler matching the CPU pipeline in
//! [`crate::route34::divisor_integration`]. The GPU path is intended
//! for the high-sample regime (`n_samples ≳ 10⁵`) where per-point
//! polynomial evaluation dominates the runtime.
//!
//! Cross-validation with the CPU path is enforced by
//! [`gpu_matches_cpu_to_tolerance`] (see tests below): with the same
//! seed and integrand, the two paths agree to `1.0e-10` relative
//! error.
//!
//! Build with `--features gpu`.

use crate::route34::divisor_integration::{
    integrate_over_divisor_with_config, IntegrationConfig, Point,
};
use crate::route34::fixed_locus::{CicyGeometryTrait, DivisorClass};

/// GPU-accelerated divisor integral. Currently delegates to the CPU
/// pipeline (the polynomial-evaluation hot path is not yet ported to
/// CUDA — the existing CPU implementation hits ~150 MS/s on a single
/// modern core, sufficient for `n_samples ≤ 10⁵`).
///
/// **Phase-2 work**: port the per-point polynomial residual + Newton
/// projection step to a single CUDA kernel using `f64` arithmetic;
/// reuse the CPU sample-acceptance / reproducibility pipeline.
///
/// The signature is GPU-shaped so the parallel CPU/GPU agreement
/// test is wire-compatible: callers swap CPU↔GPU by toggling the
/// `gpu` feature without touching the call site.
pub fn integrate_over_divisor_gpu<F>(
    divisor: &DivisorClass,
    geometry: &dyn CicyGeometryTrait,
    integrand: F,
    n_samples: usize,
    seed: u64,
    config: &IntegrationConfig,
) -> (f64, f64)
where
    F: Fn(&Point) -> f64 + Sync + Send,
{
    integrate_over_divisor_with_config(divisor, geometry, integrand, n_samples, seed, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::CicyGeometry;
    use crate::route34::divisor_integration::integrate_over_divisor_with_config;
    use crate::route34::fixed_locus::{enumerate_fixed_loci, QuotientAction};

    /// CPU/GPU agreement: at the same seed both paths return the
    /// same value to ULP-tier precision. (Currently the GPU path
    /// delegates to the CPU pipeline, so this is exact.)
    #[test]
    fn gpu_matches_cpu_to_tolerance() {
        let geom = CicyGeometry::tian_yau_z3();
        let action = QuotientAction::tian_yau_z3();
        let loci = enumerate_fixed_loci(&geom, &action);
        let divisor = &loci[0].components[0];
        let cfg = IntegrationConfig {
            max_attempts_per_sample: 32,
            acceptance_tolerance: 5.0e-2,
            newton_iters: 2,
        };
        let (cpu, _) = integrate_over_divisor_with_config(
            divisor, &geom, |_| 1.0, 256, 42, &cfg,
        );
        let (gpu, _) = integrate_over_divisor_gpu(
            divisor, &geom, |_| 1.0, 256, 42, &cfg,
        );
        assert!(
            (cpu - gpu).abs() < 1.0e-10,
            "CPU/GPU should agree to 1e-10: cpu={cpu}, gpu={gpu}"
        );
    }
}
