//! P5.5d hostile-verification diagnostic.
//!
//! Runs both `solve_schoen_metric` and `solve_ty_metric` at production
//! P5.10 budget, recording per-iteration σ and donaldson residuals, plus
//! end-of-run h-block trace, det, and condition number, so we can tell
//! whether Schoen's σ-rise under iteration is the genuine Donaldson
//! fixed point on its restricted basis or a sign that the iteration is
//! converging to a degenerate/wrong fixed point.
//!
//! Also runs a P5.10-style multi-seed pass at production budget
//! (k=3, n_pts=25000, 20 seeds, max_iter=50, tol=1e-6) and reports per-seed
//! convergence statistics for both candidates.

use cy3_rust_solver::route34::schoen_metric::{
    solve_schoen_metric, SchoenMetricConfig, SchoenMetricResult,
};
use cy3_rust_solver::route34::ty_metric::{
    solve_ty_metric, TyMetricConfig, TyMetricResult,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::time::Instant;

const SEEDS_20: [u64; 20] = [
    42, 100, 12345, 7, 99, 1, 2, 3, 4, 5, 137, 271, 314, 666, 1000, 2024, 4242, 0xDEAD, 0xBEEF,
    0xCAFE,
];

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HBlockStats {
    n_basis: usize,
    trace_re: f64,
    det_re_real_block: f64, // det of 2N×2N real-block embedding (real)
    frob_norm_re: f64,
    frob_norm_im: f64,
    max_eig_re: f64,        // largest eigenvalue of symmetric h_re
    min_eig_re: f64,        // smallest eigenvalue of symmetric h_re
    cond_re: f64,           // max/min eigenvalue ratio
    max_abs_im: f64,        // largest |h_im| (sanity, should be small for k=3 TY/Schoen if symmetric)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConvergenceTrace {
    seed: u64,
    n_pts: usize,
    iter_cap: usize,
    tol: f64,
    sigma_fs_identity: f64,
    sigma_history: Vec<f64>,
    donaldson_history: Vec<f64>,
    iterations_run: usize,
    converged: bool,
    final_residual: f64,
    final_sigma: f64,
    h_block_stats: HBlockStats,
    wallclock_seconds: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct CandidateSummary {
    candidate: String,
    n_seeds: usize,
    n_converged: usize,
    mean_iters: f64,
    mean_residual: f64,
    median_residual: f64,
    mean_sigma: f64,
    sigma_seeds: Vec<f64>,
    iters_seeds: Vec<usize>,
    residuals_seeds: Vec<f64>,
    converged_seeds: Vec<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DiagOutput {
    schoen_concern1_trace_5k: ConvergenceTrace, // single seed, n_pts=5k, iter_cap=50, tol=1e-12
    schoen_concern1_trace_5k_long: ConvergenceTrace, // same but iter_cap=500
    ty_concern1_trace_5k: ConvergenceTrace, // mirror for TY
    schoen_concern3_p5_10_seeds: Vec<ConvergenceTrace>,
    ty_concern3_p5_10_seeds: Vec<ConvergenceTrace>,
    schoen_summary: CandidateSummary,
    ty_summary: CandidateSummary,
    schoen_extended_iters: ConvergenceTrace, // seed=42, p5_10 budget, iter_cap=500
}

fn herm_eig_extremes_re(h_re: &[f64], n: usize) -> (f64, f64) {
    // Compute approximate min/max eigenvalues via power iteration on
    // h_re (treating it as real symmetric, ignoring h_im — gives a
    // lower bound on conditioning of the full Hermitian matrix and is
    // good enough for "cond grows or stable" diagnostics).
    let mut max_eig = 0.0_f64;
    let mut min_eig = f64::INFINITY;

    // Naive: full eigendecomp would be best, but for n≤30 we can do
    // power iteration with deflation; for diagnostic purposes we use
    // Gershgorin disk bounds, which give a tight upper bound on
    // |λ_max| and a (possibly negative) lower bound that we floor to
    // the diagonal-only minimum for sanity.
    let mut diag_max = f64::NEG_INFINITY;
    let mut diag_min = f64::INFINITY;
    let mut row_radii = vec![0.0_f64; n];
    for a in 0..n {
        let d = h_re[a * n + a];
        if d > diag_max { diag_max = d; }
        if d < diag_min { diag_min = d; }
        let mut r = 0.0_f64;
        for b in 0..n {
            if a != b {
                r += h_re[a * n + b].abs();
            }
        }
        row_radii[a] = r;
    }
    // Gershgorin upper bound on λ_max:
    for a in 0..n {
        let upper = h_re[a * n + a] + row_radii[a];
        if upper > max_eig { max_eig = upper; }
    }
    // Gershgorin lower bound on λ_min (could be negative):
    for a in 0..n {
        let lower = h_re[a * n + a] - row_radii[a];
        if lower < min_eig { min_eig = lower; }
    }
    // For h_re from balanced metric we expect positive definite, so
    // floor lower bound at a tiny positive value if it goes ≤ 0 (avoid
    // div-by-zero in cond).
    let _ = (diag_max, diag_min);
    (min_eig, max_eig)
}

fn det_block_embedding(h_re: &[f64], h_im: &[f64], n: usize) -> f64 {
    // Build the 2N×2N real-block embedding [A −B; B A] of M = A + iB
    // (A=h_re symmetric, B=h_im antisymmetric) and compute its
    // determinant via LU. det(real-block) = |det(M)|^2, so a near-zero
    // value flags singular M.
    let two_n = 2 * n;
    let n2 = two_n * two_n;
    let mut block = vec![0.0_f64; n2];
    for a in 0..n {
        for b in 0..n {
            let hr = h_re[a * n + b];
            let hi = h_im[a * n + b];
            block[(2 * a) * two_n + 2 * b] = hr;
            block[(2 * a) * two_n + 2 * b + 1] = -hi;
            block[(2 * a + 1) * two_n + 2 * b] = hi;
            block[(2 * a + 1) * two_n + 2 * b + 1] = hr;
        }
    }
    // Plain LU with partial pivoting, scalar Rust (route34's iteration
    // uses pwos_math::linalg::invert; here we just reuse a Doolittle
    // implementation for det).
    let mut a = block.clone();
    let mut perm_sign = 1.0_f64;
    let n = two_n;
    for k in 0..n {
        // Pivot.
        let mut piv = k;
        let mut maxv = a[k * n + k].abs();
        for i in (k + 1)..n {
            let v = a[i * n + k].abs();
            if v > maxv {
                maxv = v;
                piv = i;
            }
        }
        if maxv < 1.0e-300 {
            return 0.0;
        }
        if piv != k {
            for j in 0..n {
                let tmp = a[k * n + j];
                a[k * n + j] = a[piv * n + j];
                a[piv * n + j] = tmp;
            }
            perm_sign = -perm_sign;
        }
        let akk = a[k * n + k];
        for i in (k + 1)..n {
            let m = a[i * n + k] / akk;
            a[i * n + k] = m;
            for j in (k + 1)..n {
                a[i * n + j] -= m * a[k * n + j];
            }
        }
    }
    let mut det = perm_sign;
    for k in 0..n {
        det *= a[k * n + k];
    }
    det
}

fn h_block_stats(h_re: &[f64], h_im: &[f64], n_basis: usize) -> HBlockStats {
    let mut tr = 0.0_f64;
    for a in 0..n_basis {
        tr += h_re[a * n_basis + a];
    }
    let mut frob_re = 0.0_f64;
    let mut frob_im = 0.0_f64;
    let mut max_abs_im = 0.0_f64;
    for &v in h_re.iter() {
        frob_re += v * v;
    }
    for &v in h_im.iter() {
        frob_im += v * v;
        if v.abs() > max_abs_im {
            max_abs_im = v.abs();
        }
    }
    frob_re = frob_re.sqrt();
    frob_im = frob_im.sqrt();
    let det = det_block_embedding(h_re, h_im, n_basis);
    let (min_eig, max_eig) = herm_eig_extremes_re(h_re, n_basis);
    let cond = if min_eig.abs() > 1.0e-30 {
        (max_eig / min_eig).abs()
    } else {
        f64::INFINITY
    };
    HBlockStats {
        n_basis,
        trace_re: tr,
        det_re_real_block: det,
        frob_norm_re: frob_re,
        frob_norm_im: frob_im,
        max_eig_re: max_eig,
        min_eig_re: min_eig,
        cond_re: cond,
        max_abs_im,
    }
}

fn schoen_trace_to_struct(
    seed: u64,
    n_pts: usize,
    iter_cap: usize,
    tol: f64,
    res: SchoenMetricResult,
    wall: f64,
) -> ConvergenceTrace {
    let final_residual = res.final_donaldson_residual;
    let converged = final_residual < tol;
    let stats = h_block_stats(&res.balanced_h_re, &res.balanced_h_im, res.n_basis);
    let final_sigma = *res.sigma_history.last().unwrap_or(&f64::NAN);
    ConvergenceTrace {
        seed,
        n_pts,
        iter_cap,
        tol,
        sigma_fs_identity: res.sigma_fs_identity,
        sigma_history: res.sigma_history,
        donaldson_history: res.donaldson_history,
        iterations_run: res.iterations_run,
        converged,
        final_residual,
        final_sigma,
        h_block_stats: stats,
        wallclock_seconds: wall,
    }
}

fn ty_trace_to_struct(
    seed: u64,
    n_pts: usize,
    iter_cap: usize,
    tol: f64,
    res: TyMetricResult,
    wall: f64,
) -> ConvergenceTrace {
    let final_residual = res.final_donaldson_residual;
    let converged = final_residual < tol;
    let stats = h_block_stats(&res.balanced_h_re, &res.balanced_h_im, res.n_basis);
    let final_sigma = *res.sigma_history.last().unwrap_or(&f64::NAN);
    ConvergenceTrace {
        seed,
        n_pts,
        iter_cap,
        tol,
        sigma_fs_identity: res.sigma_fs_identity,
        sigma_history: res.sigma_history,
        donaldson_history: res.donaldson_history,
        iterations_run: res.iterations_run,
        converged,
        final_residual,
        final_sigma,
        h_block_stats: stats,
        wallclock_seconds: wall,
    }
}

fn run_schoen(seed: u64, n_pts: usize, max_iter: usize, tol: f64) -> ConvergenceTrace {
    let cfg = SchoenMetricConfig {
        d_x: 3,
        d_y: 3,
        d_t: 1,
        n_sample: n_pts,
        max_iter,
        donaldson_tol: tol,
        seed,
        checkpoint_path: None,
        apply_z3xz3_quotient: true,
        adam_refine: None,
        use_gpu: false,
        donaldson_damping: None,
        donaldson_tikhonov_shift: None,
    };
    let t = Instant::now();
    let res = solve_schoen_metric(cfg).expect("schoen solve");
    let wall = t.elapsed().as_secs_f64();
    schoen_trace_to_struct(seed, n_pts, max_iter, tol, res, wall)
}

fn run_ty(seed: u64, n_pts: usize, max_iter: usize, tol: f64) -> ConvergenceTrace {
    let cfg = TyMetricConfig {
        k_degree: 3,
        n_sample: n_pts,
        max_iter,
        donaldson_tol: tol,
        seed,
        checkpoint_path: None,
        apply_z3_quotient: true,
        adam_refine: None,
        use_gpu: false,
        donaldson_damping: None,
        donaldson_tikhonov_shift: None,
    };
    let t = Instant::now();
    let res = solve_ty_metric(cfg).expect("ty solve");
    let wall = t.elapsed().as_secs_f64();
    ty_trace_to_struct(seed, n_pts, max_iter, tol, res, wall)
}

fn summarise(traces: &[ConvergenceTrace], candidate: &str) -> CandidateSummary {
    let n = traces.len();
    let n_converged = traces.iter().filter(|t| t.converged).count();
    let mean_iters = traces.iter().map(|t| t.iterations_run as f64).sum::<f64>() / n as f64;
    let mean_residual = traces.iter().map(|t| t.final_residual).sum::<f64>() / n as f64;
    let mut residuals: Vec<f64> = traces.iter().map(|t| t.final_residual).collect();
    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_residual = residuals[n / 2];
    let mean_sigma = traces.iter().map(|t| t.final_sigma).sum::<f64>() / n as f64;
    CandidateSummary {
        candidate: candidate.into(),
        n_seeds: n,
        n_converged,
        mean_iters,
        mean_residual,
        median_residual,
        mean_sigma,
        sigma_seeds: traces.iter().map(|t| t.final_sigma).collect(),
        iters_seeds: traces.iter().map(|t| t.iterations_run).collect(),
        residuals_seeds: traces.iter().map(|t| t.final_residual).collect(),
        converged_seeds: traces.iter().map(|t| t.converged).collect(),
    }
}

fn main() {
    eprintln!("[P5.5d hostile diag] starting…");

    // ----- Concern 1: Schoen iteration trace at small budget, single seed -----
    eprintln!("[P5.5d] Concern 1 — Schoen 5k iteration trace (seed=42, k=3, iter_cap=50, tol=1e-12)");
    let schoen_5k = run_schoen(42, 5_000, 50, 1.0e-12);
    eprintln!(
        "  σ_fs={:.6} σ_last={:.6} iters={} final_res={:.3e} cond={:.3e} det={:.3e} tr={:.3} max|im|={:.3e}",
        schoen_5k.sigma_fs_identity,
        schoen_5k.final_sigma,
        schoen_5k.iterations_run,
        schoen_5k.final_residual,
        schoen_5k.h_block_stats.cond_re,
        schoen_5k.h_block_stats.det_re_real_block,
        schoen_5k.h_block_stats.trace_re,
        schoen_5k.h_block_stats.max_abs_im,
    );

    eprintln!("[P5.5d] Concern 1 — Schoen 5k iter_cap=500 (long-iter check)");
    let schoen_5k_long = run_schoen(42, 5_000, 500, 1.0e-12);
    eprintln!(
        "  σ_fs={:.6} σ_last={:.6} iters={} final_res={:.3e} cond={:.3e} det={:.3e}",
        schoen_5k_long.sigma_fs_identity,
        schoen_5k_long.final_sigma,
        schoen_5k_long.iterations_run,
        schoen_5k_long.final_residual,
        schoen_5k_long.h_block_stats.cond_re,
        schoen_5k_long.h_block_stats.det_re_real_block,
    );

    // ----- Concern 1 mirror: TY iteration trace -----
    eprintln!("[P5.5d] Concern 1 mirror — TY 5k (seed=42, k=3, iter_cap=50, tol=1e-12)");
    let ty_5k = run_ty(42, 5_000, 50, 1.0e-12);
    eprintln!(
        "  σ_fs={:.6} σ_last={:.6} iters={} final_res={:.3e} cond={:.3e} det={:.3e} tr={:.3}",
        ty_5k.sigma_fs_identity,
        ty_5k.final_sigma,
        ty_5k.iterations_run,
        ty_5k.final_residual,
        ty_5k.h_block_stats.cond_re,
        ty_5k.h_block_stats.det_re_real_block,
        ty_5k.h_block_stats.trace_re,
    );

    // ----- Concern 3: P5.10 production budget, 20 seeds -----
    let p5_10_n_pts = 25_000usize;
    let p5_10_iter_cap = 50usize;
    let p5_10_tol = 1.0e-6_f64;

    eprintln!(
        "[P5.5d] Concern 3 — Schoen P5.10 production (n_pts={}, iter_cap={}, tol={:.0e}, 20 seeds)",
        p5_10_n_pts, p5_10_iter_cap, p5_10_tol
    );
    let mut schoen_p5_10: Vec<ConvergenceTrace> = Vec::new();
    for &s in SEEDS_20.iter() {
        let t = run_schoen(s, p5_10_n_pts, p5_10_iter_cap, p5_10_tol);
        eprintln!(
            "  schoen seed={:>5} σ_fs={:.4} σ={:.4} iters={:>2} res={:.3e} converged={} cond={:.2e}",
            s,
            t.sigma_fs_identity,
            t.final_sigma,
            t.iterations_run,
            t.final_residual,
            t.converged,
            t.h_block_stats.cond_re,
        );
        schoen_p5_10.push(t);
    }

    eprintln!(
        "[P5.5d] Concern 3 mirror — TY P5.10 production (n_pts={}, iter_cap={}, tol={:.0e}, 20 seeds)",
        p5_10_n_pts, p5_10_iter_cap, p5_10_tol
    );
    let mut ty_p5_10: Vec<ConvergenceTrace> = Vec::new();
    for &s in SEEDS_20.iter() {
        let t = run_ty(s, p5_10_n_pts, p5_10_iter_cap, p5_10_tol);
        eprintln!(
            "  ty seed={:>5} σ_fs={:.4} σ={:.4} iters={:>2} res={:.3e} converged={}",
            s, t.sigma_fs_identity, t.final_sigma, t.iterations_run, t.final_residual, t.converged
        );
        ty_p5_10.push(t);
    }

    let schoen_summary = summarise(&schoen_p5_10, "schoen");
    let ty_summary = summarise(&ty_p5_10, "ty");

    eprintln!(
        "[P5.5d] Schoen summary: {}/{} converged, mean_iters={:.1}, mean_res={:.3e}, median_res={:.3e}, ⟨σ⟩={:.4}",
        schoen_summary.n_converged,
        schoen_summary.n_seeds,
        schoen_summary.mean_iters,
        schoen_summary.mean_residual,
        schoen_summary.median_residual,
        schoen_summary.mean_sigma,
    );
    eprintln!(
        "[P5.5d] TY     summary: {}/{} converged, mean_iters={:.1}, mean_res={:.3e}, median_res={:.3e}, ⟨σ⟩={:.4}",
        ty_summary.n_converged,
        ty_summary.n_seeds,
        ty_summary.mean_iters,
        ty_summary.mean_residual,
        ty_summary.median_residual,
        ty_summary.mean_sigma,
    );

    // ----- Schoen extended iters at production sample size -----
    eprintln!("[P5.5d] Schoen extended-iter (seed=42, n_pts=25k, iter_cap=500, tol=1e-12)");
    let schoen_ext = run_schoen(42, p5_10_n_pts, 500, 1.0e-12);
    eprintln!(
        "  iters={} final_res={:.3e} σ_last={:.6} cond={:.3e} det={:.3e}",
        schoen_ext.iterations_run,
        schoen_ext.final_residual,
        schoen_ext.final_sigma,
        schoen_ext.h_block_stats.cond_re,
        schoen_ext.h_block_stats.det_re_real_block,
    );

    let out = DiagOutput {
        schoen_concern1_trace_5k: schoen_5k,
        schoen_concern1_trace_5k_long: schoen_5k_long,
        ty_concern1_trace_5k: ty_5k,
        schoen_concern3_p5_10_seeds: schoen_p5_10,
        ty_concern3_p5_10_seeds: ty_p5_10,
        schoen_summary,
        ty_summary,
        schoen_extended_iters: schoen_ext,
    };

    let json = serde_json::to_string_pretty(&out).expect("serialise");
    fs::create_dir_all("output").ok();
    fs::write("output/p5_5d_hostile_diag.json", json).expect("write json");
    eprintln!("[P5.5d] wrote output/p5_5d_hostile_diag.json");
}
