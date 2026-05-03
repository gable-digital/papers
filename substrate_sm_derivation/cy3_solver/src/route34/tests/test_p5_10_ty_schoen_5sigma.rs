//! §5.10 — TY-vs-Schoen σ-discrimination at higher n_pts (5σ target).
//!
//! P5.7 reached n-σ = 4.854 at k=3 with n_pts=10 000. Just 3% short of
//! the project's 5σ goal. The σ-eval logic is correct; the bottleneck is
//! statistical (Schoen's heavy tails at low k). Path A (this module) bumps
//! n_pts at k=3 to push n-σ past 5.0.
//!
//! Test:
//!  * `test_p5_10_ty_schoen_5sigma_at_higher_npts` — `#[ignore]`d
//!    long-running 20-seed ensemble at k=3, n_pts=25 000. Asserts
//!    n-σ > 5.0; on failure documents what was actually achieved and
//!    what additional budget would be required.
//!
//! The full multi-seed harness lives in
//! `src/bin/p5_10_ty_schoen_5sigma.rs`. The test below is a lightweight
//! wrapper that runs the same Path A pipeline so regressions and budget
//! shortfalls are caught by `cargo test --release --features gpu --lib
//! test_p5_10 -- --ignored --nocapture`.
//!
//! See `references/p5_10_5sigma_target.md` for the full report.

use crate::route34::cy3_metric_unified::{
    Cy3MetricSolver, Cy3MetricSpec, SchoenSolver, TianYauSolver,
};

/// Schoen (d_x, d_y, d_t) tuple for a "k" label, matching
/// `schoen_solver_dispatches_correctly` and `schoen_publication_default`.
fn schoen_tuple_for_k(k: u32) -> (u32, u32, u32) {
    match k {
        2 => (2, 2, 1),
        3 => (3, 3, 1),
        4 => (4, 4, 2),
        _ => panic!("schoen_tuple_for_k: unsupported k={}", k),
    }
}

/// 20-seed list — same as P5.3 / P5.7 `SEEDS_20` for cross-comparability.
const SEEDS_20: [u64; 20] = [
    42, 100, 12345, 7, 99, 1, 2, 3, 4, 5, 137, 271, 314, 666, 1000, 2024, 4242, 0xDEAD, 0xBEEF,
    0xCAFE,
];

fn run_ty_seed(k: u32, n_pts: usize, seed: u64) -> Option<f64> {
    let solver = TianYauSolver;
    let spec = Cy3MetricSpec::TianYau {
        k,
        n_sample: n_pts,
        max_iter: 25,
        donaldson_tol: 1.0e-3,
        seed,
    };
    let r = solver.solve_metric(&spec).ok()?;
    let s = r.final_sigma_residual();
    if s.is_finite() && s > 0.0 { Some(s) } else { None }
}

fn run_schoen_seed(k: u32, n_pts: usize, seed: u64) -> Option<f64> {
    let (d_x, d_y, d_t) = schoen_tuple_for_k(k);
    let solver = SchoenSolver;
    let spec = Cy3MetricSpec::Schoen {
        d_x,
        d_y,
        d_t,
        n_sample: n_pts,
        max_iter: 25,
        donaldson_tol: 1.0e-3,
        seed,
    };
    let r = solver.solve_metric(&spec).ok()?;
    let s = r.final_sigma_residual();
    if s.is_finite() && s > 0.0 { Some(s) } else { None }
}

fn mean_stderr(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = if n > 1.0 {
        samples.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    let stderr = (var / n).sqrt();
    (mean, stderr)
}

/// Long-running 20-seed × 2-candidate ensemble at k=3, n_pts=25 000
/// (Path A: n_pts boost). Asserts n-σ > 5.0.
///
/// On failure the test prints the actual n-σ + a suggested budget bump so
/// the gap to 5σ is documented in CI output. The path-not-yet-cleared
/// fallback in `references/p5_10_5sigma_target.md` matches this output.
#[test]
#[ignore]
fn test_p5_10_ty_schoen_5sigma_at_higher_npts() {
    let n_pts = 25_000_usize;
    let k = 3_u32;

    let mut ty_vals = Vec::with_capacity(SEEDS_20.len());
    let mut schoen_vals = Vec::with_capacity(SEEDS_20.len());

    eprintln!(
        "[p5_10] running k={k}, n_pts={n_pts}, n_seeds={} (Path A: n_pts boost)",
        SEEDS_20.len()
    );

    for (i, &seed) in SEEDS_20.iter().enumerate() {
        if let Some(s) = run_ty_seed(k, n_pts, seed) {
            eprintln!(
                "  [{:2}/{}] TY     seed={:>6}: σ={:.6}",
                i + 1,
                SEEDS_20.len(),
                seed,
                s
            );
            ty_vals.push(s);
        }
        if let Some(s) = run_schoen_seed(k, n_pts, seed) {
            eprintln!(
                "  [{:2}/{}] Schoen seed={:>6}: σ={:.6}",
                i + 1,
                SEEDS_20.len(),
                seed,
                s
            );
            schoen_vals.push(s);
        }
    }

    assert!(
        !ty_vals.is_empty(),
        "no TY seeds succeeded — pipeline regression"
    );
    assert!(
        !schoen_vals.is_empty(),
        "no Schoen seeds succeeded — pipeline regression"
    );

    let (mu_ty, se_ty) = mean_stderr(&ty_vals);
    let (mu_sc, se_sc) = mean_stderr(&schoen_vals);
    let d_sigma = mu_ty - mu_sc;
    let se_comb = (se_ty * se_ty + se_sc * se_sc).sqrt();
    let n_sigma = if se_comb > 0.0 {
        d_sigma.abs() / se_comb
    } else {
        f64::INFINITY
    };

    eprintln!(
        "[p5_10 k={k}] <σ_TY>={:.6} ± {:.4e} (n={})  <σ_Schoen>={:.6} ± {:.4e} (n={})",
        mu_ty,
        se_ty,
        ty_vals.len(),
        mu_sc,
        se_sc,
        schoen_vals.len()
    );
    eprintln!(
        "[p5_10 k={k}] Δσ = {:.6}, SE_combined = {:.4e}, n-σ = {:.3}",
        d_sigma, se_comb, n_sigma
    );
    eprintln!(
        "[p5_10 verdict] 5σ achieved: {} (n-σ = {:.3} vs threshold 5.0)",
        n_sigma >= 5.0,
        n_sigma
    );

    if n_sigma < 5.0 {
        // Estimate additional n_pts needed assuming SE ∝ 1/√n_pts.
        let scale = (5.0 / n_sigma).powi(2);
        let needed_n_pts = (n_pts as f64 * scale).ceil() as usize;
        eprintln!(
            "[p5_10 budget] to clear 5σ at this k, estimated n_pts ≈ {} (= {} × current)",
            needed_n_pts, scale
        );
    }

    assert!(
        n_sigma.is_finite(),
        "n-σ must be finite, got {}",
        n_sigma
    );
    assert!(
        n_sigma > 5.0,
        "P5.10 5σ goal not met: n-σ = {:.3} (threshold 5.0). \
         See log above for budget needed.",
        n_sigma
    );
}

/// P5.5k regression guard. After raising the Donaldson iter cap from 50
/// to 100 (round-6 P5.5k), the Tier 0 strict-converged Schoen subset at
/// canonical n_pts=40k should contain at least 12 seeds — the floor
/// established by the iter_cap=50 baseline. cap=100 is expected to
/// recover up to 5 additional seeds (7, 314, 1000, 2024, 57005) that
/// previously hit the cap with residuals just above tol; falling below
/// the 12-seed floor would indicate a regression in the Donaldson
/// machinery (e.g. an over-aggressive new guard) rather than a budget
/// shortfall.
///
/// `#[ignore]`'d because the runtime is ~5–10 minutes for the strict-
/// converged subset alone (40k points × ~12 Schoen seeds × ≤100 iters).
#[test]
#[ignore]
fn test_p5_10_n40k_iter_cap_100_strict_converged_floor() {
    use crate::route34::cy3_metric_unified::{
        Cy3MetricSolver, Cy3MetricSpec, SchoenSolver,
    };

    let n_pts = 40_000_usize;
    let k = 3_u32;
    let donaldson_iters = 100_usize;
    let donaldson_tol = 1.0e-6_f64;
    let (d_x, d_y, d_t) = schoen_tuple_for_k(k);

    let mut strict_converged_count = 0_usize;
    let mut details: Vec<(u64, usize, f64)> = Vec::new();
    for &seed in &SEEDS_20 {
        let solver = SchoenSolver;
        let spec = Cy3MetricSpec::Schoen {
            d_x,
            d_y,
            d_t,
            n_sample: n_pts,
            max_iter: donaldson_iters,
            donaldson_tol,
            seed,
        };
        let r = match solver.solve_metric(&spec) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let s = r.summary();
        let res = s.final_donaldson_residual;
        let iters = s.iterations_run;
        let sigma = s.final_sigma_residual;
        details.push((seed, iters, res));
        if res.is_finite()
            && res < donaldson_tol
            && iters < donaldson_iters
            && sigma.is_finite()
            && sigma > 0.0
        {
            strict_converged_count += 1;
        }
    }
    eprintln!(
        "[p5_10 n40k iter_cap=100] strict_converged Schoen seeds: {} (floor: 12)",
        strict_converged_count
    );
    for (seed, iters, res) in &details {
        eprintln!(
            "  seed={:>6} iters={:>3} residual={:.3e}",
            seed, iters, res
        );
    }
    // P5.5l (round-7 hostile review): tightened from >= 12 to >= 15 to
    // catch silent 1-4 seed regressions. Tip-of-tree at b206abd4 produces
    // 16 strict-converged Schoen seeds at n_pts=40k, iter_cap=100. The
    // 4 systematic drop-outs are seeds 5 (σ=43 outlier), 100 (early
    // ambiguous, residual=4e-3), 4242 (early ambiguous, residual=2e-3),
    // and 57005 (cap-hit at 100, residual=8e-5). Allowing one slack seed
    // (15 not 16) protects against unrelated trajectory perturbations
    // that legitimately trade one seed for another.
    assert!(
        strict_converged_count >= 15,
        "P5.5l regression: Tier 0 strict-converged Schoen seed count = {} < 15 floor. \
         Raising iter_cap from 50 to 100 should monotonically recover seeds; if this \
         fires, a new guard or precision regression is bailing seeds that previously \
         converged. Expected 16 at tip-of-tree; one-seed slack to 15.",
        strict_converged_count
    );
}
