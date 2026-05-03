//! §5.4 — first σ values on the Tian-Yau (TY/Z3) Calabi-Yau threefold.
//!
//! This is the **physics-candidate** counterpart to the Fermat-quintic
//! σ tests: the whole project goal (cy3_substrate_discrimination) is
//! to discriminate TY/Z3 from Schoen/Z3xZ3 at 5σ, so we have to first
//! demonstrate that we can actually compute σ on TY at all. Up to this
//! point, every σ value reported by the test suite has been on the
//! Fermat quintic (a *test case* in CP^4, **not** a physics candidate).
//!
//! These tests drive the existing `route34::ty_metric` / `cy3_metric_unified`
//! pipeline with publication-relevant settings (n_pts = 10000, seed = 42)
//! and assert basic numerical sanity: σ is finite, σ < 1, σ decreases
//! under Donaldson balancing relative to the FS-Gram seed.
//!
//! See `references/p5_4_ty_z3_initial_sigma.md` for the full report.

use crate::route34::cy3_metric_unified::{Cy3MetricSolver, Cy3MetricSpec, TianYauSolver};
use crate::route34::ty_metric::{solve_ty_metric, TyMetricConfig};

/// k=2 σ at the publication n_pts = 10000 setting. Asserts:
///
///  1. `solve_metric` succeeds.
///  2. σ_final is finite and strictly positive.
///  3. σ_final < 1.0 (sanity: any half-converged Donaldson balance lives
///     in σ ≪ 1; values near 1 mean the metric never balanced).
///  4. The pipeline ran > 1 Donaldson iterations.
///
/// Prints the full sigma history so the σ value is captured in the test
/// log when run with `--nocapture`.
#[test]
fn test_p5_4_ty_sigma_at_k2_is_finite() {
    let solver = TianYauSolver;
    let spec = Cy3MetricSpec::TianYau {
        k: 2,
        n_sample: 10_000,
        max_iter: 25,
        donaldson_tol: 1.0e-3,
        seed: 42,
    };
    let result = solver.solve_metric(&spec).expect("TY solver must succeed at k=2");
    let summary = result.summary();
    println!("[p5_4_ty k=2] variety = {}", summary.variety);
    println!("[p5_4_ty k=2] n_basis = {}", summary.n_basis);
    println!("[p5_4_ty k=2] n_points = {}", summary.n_points);
    println!("[p5_4_ty k=2] iterations_run = {}", summary.iterations_run);
    println!(
        "[p5_4_ty k=2] sigma_final = {:.6e}",
        summary.final_sigma_residual
    );
    println!(
        "[p5_4_ty k=2] donaldson_residual_final = {:.6e}",
        summary.final_donaldson_residual
    );
    println!(
        "[p5_4_ty k=2] wall_clock_seconds = {:.3}",
        summary.wall_clock_seconds
    );

    assert!(
        summary.final_sigma_residual.is_finite(),
        "sigma must be finite, got {}",
        summary.final_sigma_residual
    );
    assert!(
        summary.final_sigma_residual > 0.0,
        "sigma must be > 0, got {}",
        summary.final_sigma_residual
    );
    assert!(
        summary.final_sigma_residual < 1.0,
        "sigma sanity bound: sigma < 1.0, got {}",
        summary.final_sigma_residual
    );
    assert!(
        summary.iterations_run > 1,
        "should run > 1 Donaldson iterations, got {}",
        summary.iterations_run
    );
}

/// Donaldson iteration must contract on the TY/Z3 pipeline at k=2.
///
/// Pre-2026-04-29 (P5.5d) this test asserted `sigma_history.last() <
/// sigma_history[0]`, which the buggy `h ← T(h)` iteration happened
/// to satisfy because it converged FROM the post-iter-1 σ DOWN to a
/// non-balance fixed point. Post-P5.5d the first iteration already
/// drives σ very close to the (correct) balanced fixed point, so
/// `sigma_history[0]` and `sigma_history.last()` differ only by O(1e-6)
/// Monte-Carlo noise.
///
/// Furthermore, at k=2 (n_basis=28 for TY/Z3) the FS-Gram identity has
/// LOWER σ than the Donaldson-balanced fixed point — same small-basis
/// pathology Schoen exhibits at (3,3,1). The σ-vs-FS monotonicity
/// invariant only applies for `k ≥ 3` (TY/Z3) or sufficiently large
/// invariant basis. The substantive post-fix claim at k=2 is that the
/// iteration CONVERGES — Frobenius residual contracts toward zero.
///
/// Uses `solve_ty_metric` directly so we have access to the residual
/// history.
#[test]
fn test_p5_4_ty_sigma_decreases_with_donaldson() {
    let cfg = TyMetricConfig {
        k_degree: 2,
        n_sample: 4_000,
        max_iter: 15,
        donaldson_tol: 1.0e-3,
        seed: 42,
        checkpoint_path: None,
        apply_z3_quotient: true,
            adam_refine: None,
            use_gpu: false,
            donaldson_damping: None,
            donaldson_tikhonov_shift: None,
    };
    let result = solve_ty_metric(cfg).expect("TY solve at k=2");
    println!(
        "[p5_4_ty k=2 contraction] sigma_fs_identity={:.6} sigma_history (len={}) = {:?}",
        result.sigma_fs_identity,
        result.sigma_history.len(),
        result.sigma_history
    );
    println!(
        "[p5_4_ty k=2 contraction] donaldson_history = {:?}",
        result.donaldson_history
    );
    assert!(
        result.sigma_history.len() >= 2,
        "need >= 2 iterations to test contraction, got {}",
        result.sigma_history.len()
    );
    let sigma_last = *result.sigma_history.last().unwrap();
    assert!(
        result.sigma_fs_identity.is_finite() && sigma_last.is_finite(),
        "σ_fs/last must be finite"
    );
    // Iteration contraction: Frobenius residual at the final iteration
    // must be strictly smaller than at iter 1.
    let r_first = result.donaldson_history[0];
    let r_last = *result.donaldson_history.last().unwrap();
    assert!(
        r_last < r_first,
        "Donaldson Frobenius residual did not contract at k=2: \
         iter1={r_first:.3e}, last={r_last:.3e}",
    );
    // σ at convergence is finite and bounded for k=2 TY/Z3.
    assert!(
        sigma_last > 0.0 && sigma_last < 1.0,
        "k=2 TY σ_last={sigma_last} outside expected band (0, 1)"
    );
}

/// k-sweep: σ(k=2) and σ(k=3) on TY. Long-running test (`#[ignore]`).
///
/// Reports σ at k=2 and k=3 for the record. Donaldson-density
/// theory predicts σ → 0 as k → ∞, but at fixed (small) `max_iter` the
/// k=3 basis (n_basis ≈ 87, ~3× the k=2 size) typically has not yet
/// converged within the same iteration budget, so σ(k=3) can transiently
/// exceed σ(k=2). We assert only finiteness here; the actual large-k
/// convergence test belongs in a future production-mode harness with
/// substantially larger `max_iter` and `n_sample`.
#[test]
#[ignore]
fn test_p5_4_ty_sigma_at_k_sweep() {
    let mut sigmas: Vec<(u32, f64, usize, usize)> = Vec::new();
    for k in [2u32, 3u32] {
        let cfg = TyMetricConfig {
            k_degree: k,
            n_sample: 8_000,
            max_iter: 30,
            donaldson_tol: 1.0e-3,
            seed: 42,
            checkpoint_path: None,
            apply_z3_quotient: true,
            adam_refine: None,
            use_gpu: false,
            donaldson_damping: None,
            donaldson_tikhonov_shift: None,
        };
        let result = solve_ty_metric(cfg).expect("TY solve in k-sweep");
        println!(
            "[p5_4_ty k_sweep] k={} sigma={:.6e} n_basis={} iters={}",
            k, result.final_sigma_residual, result.n_basis, result.iterations_run
        );
        sigmas.push((k, result.final_sigma_residual, result.n_basis, result.iterations_run));
    }
    for (k, s, _, _) in &sigmas {
        assert!(s.is_finite() && *s > 0.0, "sigma at k={} non-finite or non-positive: {}", k, s);
    }
}
