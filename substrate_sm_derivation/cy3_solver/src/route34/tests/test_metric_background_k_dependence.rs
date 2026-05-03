//! P-INFRA Fix 1 regression test.
//!
//! Confirms that `Cy3MetricResultBackground::from_schoen` plumbs the
//! Donaldson-balanced metric (per-sample-point Bergman-kernel value
//! `K(p) = s_p† · G · s_p`) through into the `MetricBackground`
//! adapter, so that the bundle Laplacian sees a k-dependent metric.
//!
//! Pre-fix this test FAILS because the only field carried into the
//! background was the FS-weight (independent of `(d_x, d_y, d_t)`),
//! producing bit-identical results across the entire k-sweep on the
//! ω_fix gateway test (P7.7).
//!
//! Post-fix the new `donaldson_k_value(α)` accessor returns
//! per-point K-values that genuinely depend on `(d_x, d_y, d_t)`, and
//! the `weight(α)` accessor returns the Donaldson-balanced quadrature
//! weight `w_FS(α) / K(α)` (sum-to-one renormalised), so the bundle
//! Laplacian residual now varies with k.

use crate::route34::schoen_metric::{solve_schoen_metric, SchoenMetricConfig};
use crate::route34::yukawa_pipeline::Cy3MetricResultBackground;
use crate::route34::hym_hermitian::MetricBackground;

fn run_and_extract(k: u32, seed: u64) -> (f64, f64) {
    let cfg = SchoenMetricConfig {
        d_x: k,
        d_y: k,
        d_t: 1,
        n_sample: 800,
        max_iter: 12,
        donaldson_tol: 1.0e-3,
        seed,
        checkpoint_path: None,
        apply_z3xz3_quotient: true,
            adam_refine: None,
            use_gpu: false,
            donaldson_damping: None,
            donaldson_tikhonov_shift: None,
    };
    let res = solve_schoen_metric(cfg).expect("schoen metric solve must succeed");
    let bg = Cy3MetricResultBackground::from_schoen(&res);
    let n = bg.n_points();
    assert!(n > 0, "must have at least one accepted sample point");
    // Two probes: a per-point Donaldson K-value (new accessor) and the
    // L²-norm of the renormalised Donaldson quadrature weight (which
    // also varies with k once weights are rescaled by 1/K).
    let k0 = bg.donaldson_k_value(0);
    let mut weight_sq_sum = 0.0_f64;
    for a in 0..n {
        let w = bg.weight(a);
        weight_sq_sum += w * w;
    }
    (k0, weight_sq_sum)
}

/// Run two Schoen Donaldson solves at k=2 and k=3 with the same seed.
/// The MetricBackground built from each must differ — otherwise the
/// k-sweep is fundamentally incapable of probing a k-dependent metric
/// (which is the P-INFRA bug this test guards against).
#[test]
#[ignore]
fn metric_background_changes_with_k() {
    let (k_at_p0_k2, w_sq_k2) = run_and_extract(2, 12345);
    let (k_at_p0_k3, w_sq_k3) = run_and_extract(3, 12345);

    // Both K-values must be strictly positive and finite.
    assert!(
        k_at_p0_k2.is_finite() && k_at_p0_k2 > 0.0,
        "K(p_0) at k=2 must be finite positive, got {}",
        k_at_p0_k2
    );
    assert!(
        k_at_p0_k3.is_finite() && k_at_p0_k3 > 0.0,
        "K(p_0) at k=3 must be finite positive, got {}",
        k_at_p0_k3
    );

    // Different k means a different polynomial section basis, so the
    // Bergman kernel K(p_0) must differ between the two runs.
    assert!(
        (k_at_p0_k2 - k_at_p0_k3).abs() > 1.0e-6 * k_at_p0_k2.abs().max(k_at_p0_k3.abs()),
        "MetricBackground at k=2 vs k=3 must differ; K(p_0) k2={} k3={}",
        k_at_p0_k2, k_at_p0_k3
    );

    // Same for the renormalised quadrature weight squared-sum: once
    // weights are rescaled by 1/K, this is k-dependent.
    assert!(
        (w_sq_k2 - w_sq_k3).abs() > 1.0e-12 * w_sq_k2.abs().max(w_sq_k3.abs()),
        "Donaldson-rescaled weight L² norm must differ between k=2 and k=3; got w²_k2={} w²_k3={}",
        w_sq_k2, w_sq_k3
    );
}
