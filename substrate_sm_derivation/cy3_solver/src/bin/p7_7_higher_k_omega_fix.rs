// P7.7 — Higher-k / larger-basis convergence study for ω_fix.
//
// P7.6 (commit 8add33b3) ran the Z/3 × Z/3 Wilson-line bundle
// Laplacian Δ_∂̄^V + H_4 (Z/5) sector projection at k=3 and got
// 0.81% off ω_fix = 123/248. P7.6's caveat: at k=3 the projected
// basis collapses to 3 surviving modes (24 → 3) after Z/3×Z/3 + H_4
// filtering. The 0.81% may include a sigmoid-saturation
// contribution. This binary sweeps k and (where applicable)
// test_degree to determine whether the residual converges to <100
// ppm (4-digit ω_fix verification) at higher resolution.
//
// We run two independent gateway-mode tests at each (k, test_degree):
//
//  (A) **Bundle Laplacian** (P7.6 path) — Δ_∂̄^V on the Z/3×Z/3 +
//      H_4 (Z/5) projected polynomial-seed sub-bundle. Basis is
//      controlled by the AKLP bundle's `b_lines` (degrees 0 and 1)
//      and is independent of `test_degree`, but the *metric
//      background* (Donaldson sample-cloud + balanced σ) depends on
//      `k`. So for the bundle Laplacian we sweep `k` only.
//
//  (B) **Scalar metric Laplacian, H_4-projected** (P7.5 path) —
//      Δ_g on the Z/3×Z/3 + H_4 (Z/5) projected bigraded test-
//      function basis with `max_total_degree = test_degree`. This
//      depends on BOTH `k` (via the metric background) and
//      `test_degree` (via the basis size). This is the full
//      (k, test_degree) sweep.
//
// Sweep grid:
//   (k, test_degree) ∈ {(3,4), (3,5), (4,4), (4,5), (5,4), (5,5)}
// with single converged seed (12345 — strict-converged Schoen
// seed from P5.10 ensemble) and the canonical n_pts=25000.
//
// For each cell we record:
//   - bundle Laplacian (raw λ, all 8 normalisations, ppm residual,
//     surviving basis size after each projection stage)
//   - H_4-projected metric Laplacian (raw λ, 5 normalisations,
//     ppm residual, surviving basis size at each stage)
//
// Reported normalisations (P7.5/P7.6 panel + per-task spec):
//   raw λ, λ/λ_min_nonzero, λ/λ_max, λ/mean(λ), λ/trace(λ),
//   λ/Vol(M), λ/(λ + λ_max) (sigmoid), λ/(λ + 1) (saturating
//   absolute), λ × Vol(M)^{1/3} (dimensional).

use clap::Parser;
use cy3_rust_solver::route34::cy3_metric_unified::{
    Cy3AdamOverride, Cy3MetricResultKind, Cy3MetricSolver, Cy3MetricSpec, SchoenSolver,
    TianYauSolver,
};
use cy3_rust_solver::route34::schoen_metric::AdamRefineConfig;
use cy3_rust_solver::route34::hym_hermitian::{solve_hym_metric, HymConfig, MetricBackground};
use cy3_rust_solver::route34::metric_laplacian::MetricLaplacianConfig;
use cy3_rust_solver::route34::metric_laplacian_projected::ProjectionKind;
use cy3_rust_solver::route34::sub_coxeter_h4_projector::compute_h4_projected_metric_laplacian_spectrum;
use cy3_rust_solver::route34::wilson_line_e8_z3xz3::Z3xZ3WilsonLines;
use cy3_rust_solver::route34::yukawa_pipeline::Cy3MetricResultBackground;
use cy3_rust_solver::route34::zero_modes_harmonic_z3xz3::{
    solve_z3xz3_bundle_laplacian, Z3xZ3BundleConfig, Z3xZ3Geometry,
};
use cy3_rust_solver::zero_modes::MonadBundle;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const E8_DIM: u32 = 248;

fn predicted_omega_fix() -> f64 {
    0.5 - 1.0 / (E8_DIM as f64)
}

fn ppm(observed: f64, predicted: f64) -> f64 {
    1.0e6 * (observed - predicted).abs() / predicted.abs().max(1.0e-300)
}

#[derive(Parser, Debug)]
#[command(about = "P7.7 higher-k / larger-basis ω_fix convergence sweep")]
struct Cli {
    /// Sample-cloud size for Donaldson.
    #[arg(long, default_value_t = 25_000)]
    n_pts: usize,

    /// Donaldson iteration cap.
    #[arg(long, default_value_t = 100)]
    max_iter: usize,

    /// Donaldson tolerance.
    #[arg(long, default_value_t = 1.0e-6)]
    donaldson_tol: f64,

    /// Single converged seed.
    #[arg(long, default_value_t = 12345)]
    seed: u64,

    /// Comma-separated k list (Donaldson section degree on Schoen).
    #[arg(long, default_value = "3,4,5")]
    k_list: String,

    /// Comma-separated test_degree list (max total degree of the
    /// scalar-metric-Laplacian test-function basis).
    #[arg(long, default_value = "4,5")]
    test_degree_list: String,

    /// Number of low non-zero eigenvalues to record per cell.
    #[arg(long, default_value_t = 12)]
    n_record: usize,

    /// Numerical-zero threshold for kernel classification (|λ| <
    /// thresh × λ_max counts as kernel).
    #[arg(long, default_value_t = 1.0e-3)]
    kernel_zero_thresh: f64,

    /// Skip the Schoen geometry (for fast TY-only debug runs).
    #[arg(long, default_value_t = false)]
    skip_schoen: bool,

    /// Skip the TY control geometry.
    #[arg(long, default_value_t = false)]
    skip_ty: bool,

    /// Skip the bundle Laplacian channel.
    #[arg(long, default_value_t = false)]
    skip_bundle: bool,

    /// Skip the H_4-projected scalar metric Laplacian channel.
    #[arg(long, default_value_t = false)]
    skip_metric: bool,

    /// Hard wallclock budget per Donaldson solve (seconds). If a
    /// k-cell's metric solve exceeds this estimate, the cell is
    /// skipped. 0 = no budget.
    #[arg(long, default_value_t = 0.0)]
    metric_budget_s: f64,

    /// P7.7-PROD Tikhonov regularisation strength applied to the
    /// Gram-matrix inversion in both the bundle-Laplacian and the
    /// H_4-projected metric-Laplacian channels. Larger values
    /// suppress the negative-eigenvalue artifacts that emerge once
    /// `td ≥ 3` makes the Galerkin Gram badly conditioned. The
    /// effective shift is `λ_T · ||G||_F` on the diagonal of `G`
    /// before inversion. Default `1e-10` reproduces the legacy
    /// behaviour.
    #[arg(long, default_value_t = 1.0e-10)]
    tikhonov_lambda: f64,

    /// P7.8 — orthogonalize the Z/3 × Z/3 + H_4 projected basis under
    /// the L²(M) inner product (modified Gram-Schmidt with deflation)
    /// BEFORE the Galerkin assembly. Eliminates the basis-redundancy
    /// pathology that emerges at `td ≥ 3` (Schoen Gram matrix becomes
    /// near-singular, eigensolve produces spurious negatives).
    /// Default `false` reproduces P7.7-PROD behaviour.
    #[arg(long, default_value_t = false)]
    orthogonalize: bool,

    /// Numerical null-space tolerance for the modified Gram-Schmidt
    /// deflation. Vectors whose squared L²(M) norm divided by the
    /// largest accepted vector's norm² is below this value are
    /// dropped. Default `1e-10`.
    #[arg(long, default_value_t = 1.0e-10)]
    orthogonalize_tol: f64,

    /// P7.7-Adam — number of post-Donaldson Adam σ-functional descent
    /// iterations. 0 disables Adam (legacy Donaldson-only behaviour).
    /// 50 is the recommended production value when k≥4 σ flatlines on
    /// the small invariant basis.
    #[arg(long, default_value_t = 0usize)]
    adam_iters: usize,

    /// P7.7-Adam — learning rate for the Adam σ-refinement.
    #[arg(long, default_value_t = 1.0e-3)]
    adam_lr: f64,

    /// P7.7-Adam — finite-difference step for the Adam σ-gradient.
    #[arg(long, default_value_t = 1.0e-3)]
    adam_fd_step: f64,

    /// P7.10 — use GPU σ-evaluator inside the FD-Adam loop. Requires
    /// the `gpu` feature; falls back to CPU if GPU init fails.
    #[arg(long, default_value_t = false)]
    use_gpu: bool,

    #[arg(long, default_value = "output/p7_7_higher_k_omega_fix.json")]
    output: PathBuf,
}

// P-INFRA Fix 3 — see p7_3_bundle_laplacian_omega_fix.rs for the
// full rationale; the sigmoid scheme produces false ~0.81% matches
// because λ / (λ + λ_max) saturates to 0.5 ≈ 123/248 = 0.4960. The
// JSON schema below has been updated to drop every sigmoid-related
// field; comparing post-fix runs to pre-fix outputs requires
// realising those fields are gone.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Normalised {
    raw: f64,
    by_lambda1_min: f64,
    by_lambda_max: f64,
    by_mean: f64,
    by_trace: f64,
    by_volume: f64,
    by_saturating_abs: f64,
    by_volume_dim: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EigInfo {
    rank: usize,
    eigenvalue: f64,
    normalised: Normalised,
    residual_ppm_raw: f64,
    residual_ppm_by_lambda1_min: f64,
    residual_ppm_by_lambda_max: f64,
    residual_ppm_by_mean: f64,
    residual_ppm_by_trace: f64,
    residual_ppm_by_volume: f64,
    residual_ppm_by_saturating_abs: f64,
    residual_ppm_by_volume_dim: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClosestPick {
    rank: usize,
    eigenvalue: f64,
    chosen_scheme: String,
    chosen_value: f64,
    residual_ppm: f64,
    residual_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CellResult {
    channel: String,
    geometry: String,
    k: u32,
    test_degree: Option<u32>,
    metric_iters: usize,
    metric_sigma_residual: f64,
    metric_seconds: f64,
    spectrum_seconds: f64,
    full_basis_dim: usize,
    gamma_basis_dim: usize,
    final_basis_dim: usize,
    gamma_survival_fraction: f64,
    final_survival_fraction: f64,
    n_points: usize,
    lambda_min_nonzero: Option<f64>,
    lambda_max: f64,
    lambda_mean: f64,
    lambda_trace: f64,
    volume_proxy: f64,
    full_spectrum_head: Vec<f64>,
    bottom_eigvalues: Vec<EigInfo>,
    closest_to_omega_fix: Option<ClosestPick>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WallclockEstimate {
    k: u32,
    estimated_donaldson_seconds_each_geometry: f64,
    estimated_spectrum_seconds_each_cell: f64,
}

// P-INFRA Fix 3 — `sigmoid_saturated` removed; the picker no longer
// reports `by_sigmoid` so a saturation artifact is impossible by
// construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConvergenceVerdict {
    pattern: String,
    explanation: String,
    best_residual_ppm: f64,
    best_cell_label: String,
    best_normalisation: String,
    cleared_100_ppm: bool,
    cleared_1_ppm: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiagnosticReport {
    label: &'static str,
    e8_dim: u32,
    predicted_omega_fix: f64,
    n_pts: usize,
    seed: u64,
    k_list: Vec<u32>,
    test_degree_list: Vec<u32>,
    wallclock_estimates: Vec<WallclockEstimate>,
    cells: Vec<CellResult>,
    convergence_verdict: ConvergenceVerdict,
    notes: Vec<String>,
    total_elapsed_s: f64,
}

// ----------------------------------------------------------------------
// Volume / metric helpers (mirror P7.5 / P7.6).
// ----------------------------------------------------------------------

fn volume_proxy(bg: &Cy3MetricResultBackground<'_>) -> f64 {
    let n = bg.n_points();
    let mut acc = 0.0_f64;
    for p in 0..n {
        let w = bg.weight(p);
        if !w.is_finite() || w <= 0.0 {
            continue;
        }
        let o = bg.omega(p);
        if !o.re.is_finite() || !o.im.is_finite() {
            continue;
        }
        acc += w * o.norm_sqr();
    }
    acc
}

// ----------------------------------------------------------------------
// Normalisation panel.
// ----------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn make_normalised(
    raw: f64,
    lambda_min_nonzero: f64,
    lambda_max: f64,
    lambda_mean: f64,
    lambda_trace: f64,
    vol: f64,
) -> Normalised {
    let safe = |denom: f64| -> Option<f64> {
        if denom.is_finite() && denom > 0.0 { Some(denom) } else { None }
    };
    let _ = lambda_max;
    Normalised {
        raw,
        by_lambda1_min: safe(lambda_min_nonzero).map(|d| raw / d).unwrap_or(f64::NAN),
        by_lambda_max: safe(lambda_max).map(|d| raw / d).unwrap_or(f64::NAN),
        by_mean: safe(lambda_mean).map(|d| raw / d).unwrap_or(f64::NAN),
        by_trace: safe(lambda_trace).map(|d| raw / d).unwrap_or(f64::NAN),
        by_volume: safe(vol).map(|d| raw / d).unwrap_or(f64::NAN),
        by_saturating_abs: if (raw + 1.0).abs() > 1.0e-300 {
            raw / (raw + 1.0)
        } else {
            f64::NAN
        },
        by_volume_dim: if vol.is_finite() && vol > 0.0 {
            raw * vol.powf(1.0 / 3.0)
        } else {
            f64::NAN
        },
    }
}

fn make_eig_info(rank: usize, raw: f64, n: Normalised) -> EigInfo {
    let omega_fix = predicted_omega_fix();
    EigInfo {
        rank,
        eigenvalue: raw,
        residual_ppm_raw: ppm(n.raw, omega_fix),
        residual_ppm_by_lambda1_min: ppm(n.by_lambda1_min, omega_fix),
        residual_ppm_by_lambda_max: ppm(n.by_lambda_max, omega_fix),
        residual_ppm_by_mean: ppm(n.by_mean, omega_fix),
        residual_ppm_by_trace: ppm(n.by_trace, omega_fix),
        residual_ppm_by_volume: ppm(n.by_volume, omega_fix),
        residual_ppm_by_saturating_abs: ppm(n.by_saturating_abs, omega_fix),
        residual_ppm_by_volume_dim: ppm(n.by_volume_dim, omega_fix),
        normalised: n,
    }
}

fn pick_closest(bottom: &[EigInfo]) -> Option<ClosestPick> {
    if bottom.is_empty() {
        return None;
    }
    let omega_fix = predicted_omega_fix();
    // P-INFRA Fix 3 — `by_sigmoid` deliberately omitted.
    let schemes: [(&'static str, fn(&Normalised) -> f64); 8] = [
        ("raw", |n| n.raw),
        ("by_lambda1_min", |n| n.by_lambda1_min),
        ("by_lambda_max", |n| n.by_lambda_max),
        ("by_mean", |n| n.by_mean),
        ("by_trace", |n| n.by_trace),
        ("by_volume", |n| n.by_volume),
        ("by_saturating_abs", |n| n.by_saturating_abs),
        ("by_volume_dim", |n| n.by_volume_dim),
    ];
    let mut best: Option<(f64, &EigInfo, &'static str, f64)> = None;
    for e in bottom {
        for (name, getter) in &schemes {
            let v = getter(&e.normalised);
            if !v.is_finite() {
                continue;
            }
            let res = (v - omega_fix).abs() / omega_fix.abs().max(1.0e-300);
            match &best {
                None => best = Some((res, e, name, v)),
                Some((b, _, _, _)) if res < *b => best = Some((res, e, name, v)),
                _ => {}
            }
        }
    }
    best.map(|(res, e, scheme, value)| ClosestPick {
        rank: e.rank,
        eigenvalue: e.eigenvalue,
        chosen_scheme: scheme.to_string(),
        chosen_value: value,
        residual_ppm: 1.0e6 * res,
        residual_pct: 100.0 * res,
    })
}

// ----------------------------------------------------------------------
// Spectrum analysis: turn ascending eigenvalue list + metric stats
// into a full CellResult.
// ----------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn build_cell_result(
    channel: &str,
    geometry: &str,
    k: u32,
    test_degree: Option<u32>,
    evals: &[f64],
    full_basis_dim: usize,
    gamma_basis_dim: usize,
    final_basis_dim: usize,
    n_points: usize,
    metric_iters: usize,
    metric_sigma: f64,
    metric_seconds: f64,
    spectrum_seconds: f64,
    vol: f64,
    n_record: usize,
    kernel_zero_thresh: f64,
) -> CellResult {
    if evals.is_empty() {
        return CellResult {
            channel: channel.to_string(),
            geometry: geometry.to_string(),
            k,
            test_degree,
            metric_iters,
            metric_sigma_residual: metric_sigma,
            metric_seconds,
            spectrum_seconds,
            full_basis_dim,
            gamma_basis_dim,
            final_basis_dim,
            gamma_survival_fraction: if full_basis_dim > 0 {
                gamma_basis_dim as f64 / full_basis_dim as f64
            } else {
                0.0
            },
            final_survival_fraction: if full_basis_dim > 0 {
                final_basis_dim as f64 / full_basis_dim as f64
            } else {
                0.0
            },
            n_points,
            lambda_min_nonzero: None,
            lambda_max: f64::NAN,
            lambda_mean: f64::NAN,
            lambda_trace: f64::NAN,
            volume_proxy: vol,
            full_spectrum_head: Vec::new(),
            bottom_eigvalues: Vec::new(),
            closest_to_omega_fix: None,
        };
    }

    let lambda_max = evals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lambda_mean = evals.iter().sum::<f64>() / (evals.len() as f64);
    let lambda_trace: f64 = evals.iter().sum();

    let cutoff = lambda_max * kernel_zero_thresh;
    let kernel_count_threshold = evals.iter().take_while(|&&v| v.abs() <= cutoff).count();
    let kernel_count = if kernel_count_threshold == 0 {
        // No threshold-classified kernel; treat the lowest 1 eigenvalue
        // as kernel by default.
        1.min(evals.len())
    } else {
        kernel_count_threshold
    };
    let nonzero: Vec<f64> = evals[kernel_count..].to_vec();
    let lambda_min_nonzero = nonzero.first().copied();
    let lambda_min_used = lambda_min_nonzero.unwrap_or(f64::NAN);

    let bottom: Vec<EigInfo> = nonzero
        .iter()
        .take(n_record)
        .enumerate()
        .map(|(rank, &ev)| {
            let n = make_normalised(ev, lambda_min_used, lambda_max, lambda_mean, lambda_trace, vol);
            make_eig_info(rank, ev, n)
        })
        .collect();

    let closest = pick_closest(&bottom);

    let head_n = evals.len().min(20);
    CellResult {
        channel: channel.to_string(),
        geometry: geometry.to_string(),
        k,
        test_degree,
        metric_iters,
        metric_sigma_residual: metric_sigma,
        metric_seconds,
        spectrum_seconds,
        full_basis_dim,
        gamma_basis_dim,
        final_basis_dim,
        gamma_survival_fraction: if full_basis_dim > 0 {
            gamma_basis_dim as f64 / full_basis_dim as f64
        } else {
            0.0
        },
        final_survival_fraction: if full_basis_dim > 0 {
            final_basis_dim as f64 / full_basis_dim as f64
        } else {
            0.0
        },
        n_points,
        lambda_min_nonzero,
        lambda_max,
        lambda_mean,
        lambda_trace,
        volume_proxy: vol,
        full_spectrum_head: evals[..head_n].to_vec(),
        bottom_eigvalues: bottom,
        closest_to_omega_fix: closest,
    }
}

/// Build the [`Cy3AdamOverride`] from CLI flags. `--adam-iters 0`
/// (the default) yields a no-op override that reproduces P7.7-PROD
/// pure-Donaldson behaviour. Any positive `--adam-iters` enables
/// post-Donaldson FD-Adam σ-functional descent.
fn build_adam_override(cli: &Cli) -> Cy3AdamOverride {
    if cli.adam_iters == 0 {
        return Cy3AdamOverride::default();
    }
    Cy3AdamOverride {
        adam_refine: Some(AdamRefineConfig {
            max_iters: cli.adam_iters,
            learning_rate: cli.adam_lr,
            fd_step: Some(cli.adam_fd_step),
            tol: 1.0e-7,
            use_gpu: cli.use_gpu,
        }),
        use_gpu_donaldson: false,
    }
}

// ----------------------------------------------------------------------
// Cached metric solver — solves Donaldson once per (k, geometry).
// ----------------------------------------------------------------------

struct MetricCache {
    schoen: std::collections::HashMap<u32, Result<CachedSchoen, String>>,
    ty: std::collections::HashMap<u32, Result<CachedTy, String>>,
}

struct CachedSchoen {
    result: cy3_rust_solver::route34::schoen_metric::SchoenMetricResult,
    iters: usize,
    sigma: f64,
    seconds: f64,
}

struct CachedTy {
    result: cy3_rust_solver::route34::ty_metric::TyMetricResult,
    iters: usize,
    sigma: f64,
    seconds: f64,
}

impl MetricCache {
    fn new() -> Self {
        Self {
            schoen: std::collections::HashMap::new(),
            ty: std::collections::HashMap::new(),
        }
    }

    fn schoen(&mut self, cli: &Cli, k: u32) -> Result<&CachedSchoen, String> {
        if !self.schoen.contains_key(&k) {
            let spec = Cy3MetricSpec::Schoen {
                d_x: k,
                d_y: k,
                d_t: 1,
                n_sample: cli.n_pts,
                max_iter: cli.max_iter,
                donaldson_tol: cli.donaldson_tol,
                seed: cli.seed,
            };
            let solver = SchoenSolver;
            let adam_override = build_adam_override(cli);
            eprintln!(
                "  [Donaldson] Schoen k={} n_pts={} seed={} adam_iters={} ...",
                k, cli.n_pts, cli.seed, cli.adam_iters
            );
            let t0 = Instant::now();
            let r = solver
                .solve_metric_with_adam(&spec, &adam_override)
                .map_err(|e| format!("Schoen k={k}: metric solve failed: {e}"));
            let entry: Result<CachedSchoen, String> = match r {
                Ok(kind) => {
                    let s = kind.summary();
                    let seconds = t0.elapsed().as_secs_f64();
                    let iters = s.iterations_run;
                    let sigma = s.final_sigma_residual;
                    match kind {
                        Cy3MetricResultKind::Schoen(t) => Ok(CachedSchoen {
                            result: *t,
                            iters,
                            sigma,
                            seconds,
                        }),
                        Cy3MetricResultKind::TianYau(_) => {
                            Err("Schoen solver returned TY result unexpectedly".to_string())
                        }
                    }
                }
                Err(e) => Err(e),
            };
            if let Ok(c) = &entry {
                eprintln!(
                    "  [Donaldson+Adam] Schoen k={} done: iters={} σ_donaldson={:.3e} σ_adam={:.3e} adam_iters={} t={:.1}s",
                    k,
                    c.iters,
                    c.result.sigma_after_donaldson,
                    c.result.sigma_after_adam,
                    c.result.adam_iters_run,
                    c.seconds
                );
            }
            self.schoen.insert(k, entry);
        }
        self.schoen
            .get(&k)
            .unwrap()
            .as_ref()
            .map_err(|e| e.clone())
    }

    fn ty(&mut self, cli: &Cli, k: u32) -> Result<&CachedTy, String> {
        if !self.ty.contains_key(&k) {
            let spec = Cy3MetricSpec::TianYau {
                k,
                n_sample: cli.n_pts,
                max_iter: cli.max_iter,
                donaldson_tol: cli.donaldson_tol,
                seed: cli.seed,
            };
            let solver = TianYauSolver;
            let adam_override = build_adam_override(cli);
            eprintln!(
                "  [Donaldson] TY     k={} n_pts={} seed={} adam_iters={} ...",
                k, cli.n_pts, cli.seed, cli.adam_iters
            );
            let t0 = Instant::now();
            let r = solver
                .solve_metric_with_adam(&spec, &adam_override)
                .map_err(|e| format!("TY k={k}: metric solve failed: {e}"));
            let entry: Result<CachedTy, String> = match r {
                Ok(kind) => {
                    let s = kind.summary();
                    let seconds = t0.elapsed().as_secs_f64();
                    let iters = s.iterations_run;
                    let sigma = s.final_sigma_residual;
                    match kind {
                        Cy3MetricResultKind::TianYau(t) => Ok(CachedTy {
                            result: *t,
                            iters,
                            sigma,
                            seconds,
                        }),
                        Cy3MetricResultKind::Schoen(_) => {
                            Err("TY solver returned Schoen result unexpectedly".to_string())
                        }
                    }
                }
                Err(e) => Err(e),
            };
            if let Ok(c) = &entry {
                eprintln!(
                    "  [Donaldson] TY     k={} done: iters={} σ={:.3e} t={:.1}s",
                    k, c.iters, c.sigma, c.seconds
                );
            }
            self.ty.insert(k, entry);
        }
        self.ty.get(&k).unwrap().as_ref().map_err(|e| e.clone())
    }
}

// ----------------------------------------------------------------------
// Bundle Laplacian channel (P7.6 path).
// ----------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_bundle_cell(
    label: &str,
    geometry_label: &str,
    k: u32,
    bg: &Cy3MetricResultBackground<'_>,
    metric_iters: usize,
    metric_sigma: f64,
    metric_seconds: f64,
    bundle: &MonadBundle,
    wilson: &Z3xZ3WilsonLines,
    geom: Z3xZ3Geometry,
    seed_max_total_degree: usize,
    tikhonov_lambda: f64,
    orthogonalize: bool,
    orthogonalize_tol: f64,
    n_record: usize,
    kernel_zero_thresh: f64,
) -> CellResult {
    let hym_cfg = HymConfig {
        max_iter: 8,
        damping: 0.5,
        ..HymConfig::default()
    };
    let h_v = solve_hym_metric(bundle, bg, &hym_cfg);

    let cfg = Z3xZ3BundleConfig {
        geometry: geom,
        apply_h4: true,
        seed_max_total_degree,
        tikhonov_lambda,
        orthogonalize_first: orthogonalize,
        orthogonalize_tol,
        ..Z3xZ3BundleConfig::default()
    };

    let t0 = Instant::now();
    let res = solve_z3xz3_bundle_laplacian(bundle, bg, &h_v, wilson, &cfg);
    let spectrum_seconds = t0.elapsed().as_secs_f64();
    let vol = volume_proxy(bg);

    eprintln!(
        "    [bundle ] {} k={}  basis: full={} → Z3×Z3={} (surv {:.4}) → final={} (surv {:.4})",
        label,
        k,
        res.full_seed_basis_dim,
        res.z3xz3_basis_dim,
        res.z3xz3_survival_fraction,
        res.final_basis_dim,
        res.final_survival_fraction
    );

    build_cell_result(
        "bundle_laplacian",
        geometry_label,
        k,
        None,
        &res.eigenvalues_full,
        res.full_seed_basis_dim,
        res.z3xz3_basis_dim,
        res.final_basis_dim,
        res.n_points,
        metric_iters,
        metric_sigma,
        metric_seconds,
        spectrum_seconds,
        vol,
        n_record,
        kernel_zero_thresh,
    )
}

// ----------------------------------------------------------------------
// H_4-projected scalar metric Laplacian channel (P7.5 path).
// ----------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_metric_cell(
    label: &str,
    geometry_label: &str,
    k: u32,
    test_degree: u32,
    bg: &Cy3MetricResultBackground<'_>,
    metric_iters: usize,
    metric_sigma: f64,
    metric_seconds: f64,
    projection: ProjectionKind,
    tikhonov_lambda: f64,
    n_record: usize,
    kernel_zero_thresh: f64,
) -> CellResult {
    let cfg = MetricLaplacianConfig {
        max_total_degree: test_degree,
        n_low_eigenvalues: 50,
        return_eigenvectors: false,
        mass_regularisation: tikhonov_lambda,
        ..MetricLaplacianConfig::default()
    };

    let t0 = Instant::now();
    let report = compute_h4_projected_metric_laplacian_spectrum(bg, &cfg, projection);
    let spectrum_seconds = t0.elapsed().as_secs_f64();
    let vol = volume_proxy(bg);

    eprintln!(
        "    [metric ] {} k={} td={}  basis: full={} → Γ={} (surv {:.4}) → H_4={} (surv {:.4}, rel {:.4})",
        label,
        k,
        test_degree,
        report.full_basis_dim,
        report.gamma_projected_basis_dim,
        report.gamma_survival_fraction,
        report.h4_projected_basis_dim,
        report.h4_survival_fraction,
        report.h4_relative_survival_fraction
    );

    build_cell_result(
        "metric_h4_projected",
        geometry_label,
        k,
        Some(test_degree),
        &report.spectrum.eigenvalues_full,
        report.full_basis_dim,
        report.gamma_projected_basis_dim,
        report.h4_projected_basis_dim,
        report.spectrum.n_points,
        metric_iters,
        metric_sigma,
        metric_seconds,
        spectrum_seconds,
        vol,
        n_record,
        kernel_zero_thresh,
    )
}

// ----------------------------------------------------------------------
// Verdict logic.
// ----------------------------------------------------------------------

// P-INFRA Fix 3 — `detect_sigmoid_artifact` removed alongside the
// sigmoid normalisation entry it depended on.

fn build_verdict(cells: &[CellResult]) -> ConvergenceVerdict {
    let mut best: Option<(f64, String, String)> = None;
    for c in cells {
        if c.geometry.starts_with("ty") {
            // Don't pick TY as the headline — it's the control.
            continue;
        }
        if let Some(pick) = &c.closest_to_omega_fix {
            let label = format!(
                "{}/{} k={} td={}",
                c.channel,
                c.geometry,
                c.k,
                c.test_degree
                    .map(|d| d.to_string())
                    .unwrap_or_else(|| "-".to_string())
            );
            match &best {
                None => best = Some((pick.residual_ppm, label, pick.chosen_scheme.clone())),
                Some((b, _, _)) if pick.residual_ppm < *b => {
                    best = Some((pick.residual_ppm, label, pick.chosen_scheme.clone()))
                }
                _ => {}
            }
        }
    }
    let (best_ppm, best_label, best_scheme) = best.unwrap_or_else(|| {
        (
            f64::INFINITY,
            "<no-schoen-cells>".to_string(),
            "<n/a>".to_string(),
        )
    });

    // Convergence pattern detection on Schoen bundle channel only.
    let mut schoen_bundle: Vec<(u32, f64)> = cells
        .iter()
        .filter(|c| c.geometry.starts_with("schoen") && c.channel == "bundle_laplacian")
        .filter_map(|c| c.closest_to_omega_fix.as_ref().map(|p| (c.k, p.residual_ppm)))
        .collect();
    schoen_bundle.sort_by_key(|(k, _)| *k);

    let monotone_decreasing = schoen_bundle.windows(2).all(|w| w[1].1 <= w[0].1);
    let last_ratio = if schoen_bundle.len() >= 2 {
        let n = schoen_bundle.len();
        schoen_bundle[n - 1].1 / schoen_bundle[n - 2].1.max(1.0e-300)
    } else {
        f64::NAN
    };

    let pattern = if best_ppm <= 100.0 {
        "(a) monotone toward zero — cleared 100 ppm".to_string()
    } else if monotone_decreasing && schoen_bundle.len() >= 2 && last_ratio < 0.9 {
        "(a) monotone toward zero — extrapolation needed for <100 ppm".to_string()
    } else if schoen_bundle.len() >= 2
        && last_ratio > 0.9
        && last_ratio < 1.1
        && best_ppm > 100.0
    {
        "(b) saturates above 100 ppm".to_string()
    } else {
        "indeterminate — insufficient k-sweep coverage".to_string()
    };

    let explanation = format!(
        "Best Schoen residual: {:.3} ppm at {} via {}. \
         k-sweep (Schoen bundle Laplacian): {:?}. \
         Monotone-decreasing: {}. Last ratio: {:.4}.",
        best_ppm,
        best_label,
        best_scheme,
        schoen_bundle,
        monotone_decreasing,
        last_ratio,
    );

    ConvergenceVerdict {
        pattern,
        explanation,
        best_residual_ppm: best_ppm,
        best_cell_label: best_label,
        best_normalisation: best_scheme,
        cleared_100_ppm: best_ppm <= 100.0,
        cleared_1_ppm: best_ppm <= 1.0,
    }
}

// ----------------------------------------------------------------------
// Main.
// ----------------------------------------------------------------------

fn parse_u32_csv(s: &str) -> Vec<u32> {
    s.split(',')
        .filter_map(|t| t.trim().parse::<u32>().ok())
        .collect()
}

fn estimate_wallclock(k_list: &[u32]) -> Vec<WallclockEstimate> {
    // Empirical scaling from P7.6 (k=3: ~90s) and chain-match
    // diagnostics (k=4 TY/189-basis: ~600s; k=5 estimated ~2400s).
    // Spectrum step ~10-120s.
    k_list
        .iter()
        .map(|&k| {
            let donaldson_seconds = match k {
                3 => 90.0,
                4 => 600.0,
                5 => 2400.0,
                _ => 120.0 * (k as f64).powi(3),
            };
            let spectrum_seconds = 10.0 * ((k as f64) / 3.0).powi(2);
            WallclockEstimate {
                k,
                estimated_donaldson_seconds_each_geometry: donaldson_seconds,
                estimated_spectrum_seconds_each_cell: spectrum_seconds,
            }
        })
        .collect()
}

fn print_cell(c: &CellResult) {
    eprintln!(
        "    [{}] geom={} k={} td={:?} basis(full→Γ→final)={}→{}→{}  iters={} σ={:.3e}  t_metric={:.1}s t_spec={:.2}s",
        c.channel,
        c.geometry,
        c.k,
        c.test_degree,
        c.full_basis_dim,
        c.gamma_basis_dim,
        c.final_basis_dim,
        c.metric_iters,
        c.metric_sigma_residual,
        c.metric_seconds,
        c.spectrum_seconds
    );
    eprintln!(
        "         λ_min_nonzero={}  λ_max={:.4e}  λ_mean={:.4e}  vol={:.4e}",
        match c.lambda_min_nonzero {
            Some(v) => format!("{:.4e}", v),
            None => "<none>".to_string(),
        },
        c.lambda_max,
        c.lambda_mean,
        c.volume_proxy
    );
    if let Some(pick) = &c.closest_to_omega_fix {
        eprintln!(
            "         closest-to-ω_fix:  rank={}  λ={:.4e}  via {}={:.6}  → {:.3} ppm  ({:.4} %)",
            pick.rank,
            pick.eigenvalue,
            pick.chosen_scheme,
            pick.chosen_value,
            pick.residual_ppm,
            pick.residual_pct
        );
    } else {
        eprintln!("         closest-to-ω_fix:  <empty spectrum>");
    }
}

fn main() {
    let cli = Cli::parse();
    let omega_fix = predicted_omega_fix();
    eprintln!("================================================");
    eprintln!("P7.7 — higher-k / larger-basis ω_fix convergence sweep");
    eprintln!("  predicted ω_fix = 1/2 - 1/dim(E_8) = 123/248 = {:.18}", omega_fix);
    eprintln!("================================================");

    let k_list = parse_u32_csv(&cli.k_list);
    let test_degree_list = parse_u32_csv(&cli.test_degree_list);
    eprintln!(
        "  n_pts={} seed={} k_list={:?} test_degree_list={:?}",
        cli.n_pts, cli.seed, k_list, test_degree_list
    );

    let wallclock_estimates = estimate_wallclock(&k_list);
    eprintln!("  Wallclock pre-estimates:");
    let mut total_est = 0.0_f64;
    for w in &wallclock_estimates {
        let geom_count = if cli.skip_schoen || cli.skip_ty { 1.0 } else { 2.0 };
        let bundle_t = if cli.skip_bundle { 0.0 } else { w.estimated_spectrum_seconds_each_cell };
        let metric_t = if cli.skip_metric {
            0.0
        } else {
            (test_degree_list.len() as f64) * w.estimated_spectrum_seconds_each_cell
        };
        let cell_t = geom_count * (w.estimated_donaldson_seconds_each_geometry + bundle_t + metric_t);
        total_est += cell_t;
        eprintln!(
            "    k={}: Donaldson ≈ {:.0}s/geom × {} geom + spectrum ({} cells × {:.0}s) ≈ {:.0}s",
            w.k,
            w.estimated_donaldson_seconds_each_geometry,
            geom_count,
            (if cli.skip_bundle { 0 } else { 1 }) + (if cli.skip_metric { 0 } else { test_degree_list.len() }),
            w.estimated_spectrum_seconds_each_cell,
            cell_t
        );
    }
    eprintln!("  Total estimated wallclock: ≈ {:.0}s ({:.1} min)", total_est, total_est / 60.0);

    let bundle = MonadBundle::anderson_lukas_palti_example();
    let wilson = Z3xZ3WilsonLines::canonical_aklp_schoen();

    // Quick Wilson sanity check (matches P7.6).
    let (q1, q2) = wilson.quantization_residuals();
    let comm = wilson.commutator_residual();
    eprintln!(
        "  Wilson Z/3×Z/3:  ω_2∨ residual={:.3e}  ω_5∨ residual={:.3e}  [W_1,W_2] residual={:.3e}",
        q1, q2, comm
    );
    if q1 > 1.0e-7 || q2 > 1.0e-7 {
        eprintln!("FATAL: Z/3 × Z/3 Wilson lines are not Z/3-quantized");
        std::process::exit(1);
    }

    let mut cache = MetricCache::new();
    let mut cells: Vec<CellResult> = Vec::new();
    let t_total = Instant::now();
    let mut notes: Vec<String> = Vec::new();
    notes.push(
        "P7.7 sweeps both bundle Laplacian (P7.6 path) and H_4-projected scalar metric \
         Laplacian (P7.5 path) over (k, test_degree)."
            .to_string(),
    );
    notes.push(
        "Bundle Laplacian basis is fixed by the AKLP bundle b_lines (degrees 0 and 1) and \
         is independent of test_degree; only k affects it (via the metric background)."
            .to_string(),
    );
    notes.push(
        "Scalar-metric Laplacian basis depends on both k (sample cloud) and test_degree \
         (max total degree of the test-function basis)."
            .to_string(),
    );

    for &k in &k_list {
        eprintln!("\n--- k = {} ---", k);

        // Schoen branch.
        if !cli.skip_schoen {
            match cache.schoen(&cli, k) {
                Ok(cached) => {
                    if cli.metric_budget_s > 0.0 && cached.seconds > cli.metric_budget_s {
                        notes.push(format!(
                            "Schoen k={k} Donaldson took {:.0}s, exceeds budget {:.0}s; downstream cells \
                             still computed since the metric is already paid for.",
                            cached.seconds, cli.metric_budget_s
                        ));
                    }
                    let bg = Cy3MetricResultBackground::from_schoen(&cached.result);

                    if !cli.skip_bundle {
                        // P-INFRA Fix 2 — sweep the bundle Laplacian
                        // over `test_degree` too (each `td` becomes
                        // `seed_max_total_degree`), so the bundle
                        // basis grows with the requested polynomial
                        // degree. Pre-fix the bundle was locked at
                        // 24 modes regardless of `td`.
                        for &td in &test_degree_list {
                            let cell = run_bundle_cell(
                                "Schoen/Z3xZ3-AKLP/H4",
                                "schoen_z3xz3",
                                k,
                                &bg,
                                cached.iters,
                                cached.sigma,
                                cached.seconds,
                                &bundle,
                                &wilson,
                                Z3xZ3Geometry::Schoen,
                                td as usize,
                                cli.tikhonov_lambda,
                                cli.orthogonalize,
                                cli.orthogonalize_tol,
                                cli.n_record,
                                cli.kernel_zero_thresh,
                            );
                            // Tag the test_degree on the bundle cell so
                            // the (k, td) sweep table reflects the
                            // basis-growth axis post-Fix-2.
                            let mut cell = cell;
                            cell.test_degree = Some(td);
                            print_cell(&cell);
                            cells.push(cell);
                        }
                    }

                    if !cli.skip_metric {
                        for &td in &test_degree_list {
                            let cell = run_metric_cell(
                                "Schoen/Γ∩H_4",
                                "schoen_z3xz3",
                                k,
                                td,
                                &bg,
                                cached.iters,
                                cached.sigma,
                                cached.seconds,
                                ProjectionKind::SchoenZ3xZ3,
                                cli.tikhonov_lambda,
                                cli.n_record,
                                cli.kernel_zero_thresh,
                            );
                            print_cell(&cell);
                            cells.push(cell);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("  SCHOEN k={k} SKIPPED: {e}");
                    notes.push(format!("Schoen k={k} metric solve failed: {e}"));
                }
            }
        }

        // TY control branch.
        if !cli.skip_ty {
            match cache.ty(&cli, k) {
                Ok(cached) => {
                    let bg = Cy3MetricResultBackground::from_ty(&cached.result);

                    if !cli.skip_bundle {
                        for &td in &test_degree_list {
                            let cell = run_bundle_cell(
                                "TY/Z3-AKLP/H4",
                                "ty_z3",
                                k,
                                &bg,
                                cached.iters,
                                cached.sigma,
                                cached.seconds,
                                &bundle,
                                &wilson,
                                Z3xZ3Geometry::TianYau,
                                td as usize,
                                cli.tikhonov_lambda,
                                cli.orthogonalize,
                                cli.orthogonalize_tol,
                                cli.n_record,
                                cli.kernel_zero_thresh,
                            );
                            let mut cell = cell;
                            cell.test_degree = Some(td);
                            print_cell(&cell);
                            cells.push(cell);
                        }
                    }

                    if !cli.skip_metric {
                        for &td in &test_degree_list {
                            let cell = run_metric_cell(
                                "TY/Γ∩H_4",
                                "ty_z3",
                                k,
                                td,
                                &bg,
                                cached.iters,
                                cached.sigma,
                                cached.seconds,
                                ProjectionKind::TianYauZ3,
                                cli.tikhonov_lambda,
                                cli.n_record,
                                cli.kernel_zero_thresh,
                            );
                            print_cell(&cell);
                            cells.push(cell);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("  TY k={k} SKIPPED: {e}");
                    notes.push(format!("TY k={k} metric solve failed: {e}"));
                }
            }
        }
    }

    let total_elapsed_s = t_total.elapsed().as_secs_f64();
    let convergence_verdict = build_verdict(&cells);

    eprintln!("\n================================================");
    eprintln!("CONVERGENCE VERDICT:");
    eprintln!("  pattern: {}", convergence_verdict.pattern);
    eprintln!("  best:    {} ppm at {} via {}",
        convergence_verdict.best_residual_ppm,
        convergence_verdict.best_cell_label,
        convergence_verdict.best_normalisation
    );
    eprintln!("  cleared 100 ppm: {}", convergence_verdict.cleared_100_ppm);
    eprintln!("  cleared 1 ppm:   {}", convergence_verdict.cleared_1_ppm);
    eprintln!("  {}", convergence_verdict.explanation);
    eprintln!("================================================");

    let report = DiagnosticReport {
        label: "p7_7_higher_k_omega_fix",
        e8_dim: E8_DIM,
        predicted_omega_fix: omega_fix,
        n_pts: cli.n_pts,
        seed: cli.seed,
        k_list,
        test_degree_list,
        wallclock_estimates,
        cells,
        convergence_verdict,
        notes,
        total_elapsed_s,
    };

    if let Some(parent) = cli.output.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let json = serde_json::to_string_pretty(&report).expect("serialise diagnostic report");
    fs::write(&cli.output, json).expect("write diagnostic JSON");
    eprintln!("\nWrote {}", cli.output.display());
    eprintln!("Total elapsed: {:.1}s", total_elapsed_s);
}
