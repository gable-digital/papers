//! P-BASIS-CONV — empirical diagnostic for the
//! "σ-channel discrimination is dominated by basis-size differences"
//! claim from P8.1e and P8.4.
//!
//! ## Why this exists
//!
//! P5.10 (and downstream P8.1e / P8.4 multi-channel BF analyses)
//! observed a wide TY-vs-Schoen σ gap at k=3:
//!
//! * TY (Z/3): k=3 native `n_basis = 87` (`enumerate_bigraded_kk_monomials`
//!   plus Z/3-character + LM-ideal reduction).
//! * Schoen (Z/3 × Z/3): bidegree (3,3,1) native `n_basis = 27`
//!   (`enumerate_bidegree_monomials` plus Γ = Z/3×Z/3 invariance + ideal
//!   reduction).
//!
//! The P8.1e thesis is that the σ gap is largely a basis-size artefact
//! (TY's σ converges to a smaller floor because Donaldson has more
//! degrees of freedom to balance against). If true, σ stays excluded
//! from the multi-channel Bayes Factor. If false, σ is real geometric
//! discrimination and could be reinstated.
//!
//! This binary measures the basis-size dependence directly by running
//! σ-eval at MATCHED basis sizes across both varieties at k=3, plus a
//! native-basis k-scan.
//!
//! ## Experiments
//!
//! ### Experiment A — matched-basis-size σ at k=3
//! For each `n_b ∈ {15, 20, 25, 27, 35, 50, 87}` (Schoen native = 27,
//! TY native = 87) we re-solve TY at k=3 with the section basis
//! truncated to the first `n_b` monomials (in the natural lex-sorted
//! order produced by `build_ty_invariant_reduced_basis`). Schoen at
//! bidegree (3,3,1) is similarly truncated to `n_b ∈ {15, 20, 25, 27}`.
//! At each `n_b` we run a fixed seed ensemble, record `mean σ ± SE`,
//! and plot σ_TY(n_b) vs σ_Schoen(n_b) on the same axis.
//!
//! Verdict question: at matched `n_b = 27`, is σ_TY(27) ≈ σ_Schoen(27)?
//!
//! ### Experiment B — k-scan at native basis sizes
//! σ_TY at `k ∈ {2, 3, 4}` (native `n_basis ≈ 25 / 87 / 200`) and
//! σ_Schoen at the corresponding bidegrees `{(2,2,1), (3,3,1),
//! (4,4,2)}` (native `n_basis ≈ 9 / 27 / 48`). Does Δσ scale as
//! `1/k²` (ABKO 2010 prediction for Donaldson convergence) or persist
//! at fixed magnitude?
//!
//! ## Truncation mechanism
//!
//! Truncation is delivered through the diagnostic-only thread-local in
//! `crate::route34::basis_truncation_diag`. Both `solve_ty_metric` and
//! `solve_schoen_metric` consult the override at exactly one site (the
//! line right after `build_*_invariant_reduced_basis`, before
//! workspace allocation). The override is set via an RAII guard for
//! each (CY3, k, n_b, seed) tuple and cleared on drop. Production
//! code paths never set the override; they remain bit-identical.
//!
//! ## Output
//! * `output/p_basis_convergence_diag.csv` — raw per-row data.
//! * `output/p_basis_convergence_diag.json` — summary tables.
//! * Stdout — verdict tables and basis-size-artefact assessment.

use clap::Parser;
use cy3_rust_solver::route34::basis_truncation_diag::TruncationGuard;
use cy3_rust_solver::route34::schoen_metric::{self, SchoenMetricConfig, SchoenMetricResult};
use cy3_rust_solver::route34::ty_metric::{self, TyMetricConfig, TyMetricResult};
use pwos_math::stats::bootstrap::{Bootstrap, BootstrapConfig};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    about = "P-BASIS-CONV: matched-basis-size σ comparison + k-scan diagnostic"
)]
struct Cli {
    /// Number of CY3 sample points per (CY3, k, n_b, seed) tuple.
    /// Smaller than the production P5.10 budget (10k) because we are
    /// running ~75 tuples and need wallclock < 1 hour total.
    #[arg(long, default_value_t = 10_000)]
    n_pts: usize,

    /// Donaldson max iterations per σ-eval. The diagnostic only needs
    /// "fast σ-eval" not converged-to-tol balancing; 25 iters is
    /// what P5.10 used for the Path-A budget.
    #[arg(long, default_value_t = 25)]
    donaldson_iters: usize,

    /// Donaldson tolerance. Loose (1e-3) so iters = max usually.
    #[arg(long, default_value_t = 1.0e-3)]
    donaldson_tol: f64,

    /// Comma-separated seed list. Default = 5 seeds (matches
    /// P5.7's economy-mode for fast scanning).
    #[arg(long, default_value = "42,100,12345,7,99")]
    seeds: String,

    /// Convenience for production runs: `--n-seeds 20` is shorthand
    /// for the P5.10 20-seed roster
    /// (`42,100,12345,7,99,1,2,3,4,5,137,271,314,666,1000,2024,4242,57005,48879,51966`).
    /// Other values just take the first N from that list. If not
    /// specified, the explicit `--seeds` flag wins.
    #[arg(long)]
    n_seeds: Option<usize>,

    /// Bootstrap resamples per group for percentile + BCa CIs.
    /// 10000 matches P5.10. Set 0 to skip CI computation.
    #[arg(long, default_value_t = 10_000)]
    boot_resamples: usize,

    /// Bootstrap PRNG seed (matches P5.10 default).
    #[arg(long, default_value_t = 12345)]
    boot_seed: u64,

    /// Bootstrap CI level (default = 0.95 → 95% CI).
    #[arg(long, default_value_t = 0.95)]
    boot_ci_level: f64,

    /// Output CSV path.
    #[arg(long, default_value = "output/p_basis_convergence_diag.csv")]
    csv_output: PathBuf,

    /// Output JSON summary path.
    #[arg(long, default_value = "output/p_basis_convergence_diag.json")]
    json_output: PathBuf,

    /// Convenience flag: if set, overrides `--json-output` and also
    /// derives a sibling `.csv` path (replacing the `.json` suffix).
    /// Mirrors `p5_10_ty_schoen_5sigma`'s single-output ergonomic.
    #[arg(long)]
    output: Option<PathBuf>,

    /// If true, route Donaldson through the GPU kernel (requires `gpu`).
    #[arg(long, default_value_t = false)]
    use_gpu: bool,

    /// Skip Experiment A (matched-basis-size at k=3).
    #[arg(long, default_value_t = false)]
    skip_exp_a: bool,

    /// Skip Experiment B (native-basis k-scan).
    #[arg(long, default_value_t = false)]
    skip_exp_b: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiagRow {
    candidate: String,
    k: u32,
    n_basis_truncated: i64, // -1 = native (no truncation)
    n_basis_native: usize,
    n_basis_actual: usize,
    seed: u64,
    sigma_final: f64,
    sigma_fs_identity: f64,
    residual_donaldson: f64,
    iterations_run: usize,
    elapsed_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GroupSummary {
    candidate: String,
    k: u32,
    n_basis_truncated: i64,
    n_basis_actual: usize,
    n_seeds: usize,
    sigma_mean: f64,
    sigma_se: f64,
    sigma_min: f64,
    sigma_max: f64,
    /// Bootstrap percentile CI (low, high) on the mean. None if
    /// `boot_resamples == 0` or n < 2.
    sigma_pct_ci_low: Option<f64>,
    sigma_pct_ci_high: Option<f64>,
    /// Bootstrap BCa CI (low, high) on the mean. None if BCa
    /// degenerates (e.g. all samples identical) or boot disabled.
    sigma_bca_ci_low: Option<f64>,
    sigma_bca_ci_high: Option<f64>,
    /// Number of bootstrap resamples actually run (0 = CI not computed).
    sigma_boot_resamples: usize,
    /// CI level used (0.95 → 95% CI).
    sigma_ci_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExperimentASummary {
    n_pts: usize,
    seeds: Vec<u64>,
    rows: Vec<GroupSummary>,
    /// Δσ = σ_TY(27) − σ_Schoen(27): the matched-basis gap.
    delta_sigma_at_n27: Option<f64>,
    /// Δσ_native = σ_TY(87) − σ_Schoen(27): the native-basis gap.
    delta_sigma_native: Option<f64>,
    /// Ratio. If matched-gap << native-gap, basis-size artefact wins.
    matched_to_native_ratio: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExperimentBSummary {
    n_pts: usize,
    seeds: Vec<u64>,
    rows: Vec<GroupSummary>,
    /// Per-k Δσ rows: (k, σ_TY native, σ_Schoen native, Δσ).
    delta_per_k: Vec<KScanRow>,
    /// 1/k² scaling check: ratio Δσ(k=3) / Δσ(k=2) compared with
    /// (2/3)² = 0.444 prediction.
    delta_ratio_k3_k2: Option<f64>,
    delta_ratio_k4_k2: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KScanRow {
    k: u32,
    sigma_ty: f64,
    se_ty: f64,
    n_basis_ty: usize,
    sigma_schoen: f64,
    se_schoen: f64,
    n_basis_schoen: usize,
    delta_sigma: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct DiagOutput {
    config: ConfigEcho,
    experiment_a: Option<ExperimentASummary>,
    experiment_b: Option<ExperimentBSummary>,
    raw_rows: Vec<DiagRow>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ConfigEcho {
    n_pts: usize,
    donaldson_iters: usize,
    donaldson_tol: f64,
    seeds: Vec<u64>,
    use_gpu: bool,
    boot_resamples: usize,
    boot_seed: u64,
    boot_ci_level: f64,
}

// -------------------------------------------------------------------
// σ-eval helpers: run TY / Schoen with optional truncation override
// -------------------------------------------------------------------

fn run_ty(
    k: u32,
    n_pts: usize,
    max_iter: usize,
    donaldson_tol: f64,
    seed: u64,
    n_basis_truncate: Option<usize>,
    use_gpu: bool,
) -> Result<(TyMetricResult, f64), String> {
    let t0 = Instant::now();
    let cfg = TyMetricConfig {
        k_degree: k,
        n_sample: n_pts,
        max_iter,
        donaldson_tol,
        seed,
        checkpoint_path: None,
        apply_z3_quotient: true,
        adam_refine: None,
        use_gpu,
        donaldson_damping: None,
        donaldson_tikhonov_shift: None,
    };
    // RAII guard: thread-local override is restored on drop.
    let _g = TruncationGuard::new(n_basis_truncate);
    let r = ty_metric::solve_ty_metric(cfg)
        .map_err(|e| format!("TY k={k} seed={seed} trunc={n_basis_truncate:?}: {e}"))?;
    let elapsed = t0.elapsed().as_secs_f64();
    Ok((r, elapsed))
}

fn schoen_bidegree_for_k(k: u32) -> (u32, u32, u32) {
    match k {
        2 => (2, 2, 1),
        3 => (3, 3, 1),
        4 => (4, 4, 2),
        other => panic!("unsupported k for Schoen bidegree: {other}"),
    }
}

fn run_schoen(
    k: u32,
    n_pts: usize,
    max_iter: usize,
    donaldson_tol: f64,
    seed: u64,
    n_basis_truncate: Option<usize>,
    use_gpu: bool,
) -> Result<(SchoenMetricResult, f64), String> {
    let t0 = Instant::now();
    let (d_x, d_y, d_t) = schoen_bidegree_for_k(k);
    let cfg = SchoenMetricConfig {
        d_x,
        d_y,
        d_t,
        n_sample: n_pts,
        max_iter,
        donaldson_tol,
        seed,
        checkpoint_path: None,
        apply_z3xz3_quotient: true,
        adam_refine: None,
        use_gpu,
        donaldson_damping: None,
        donaldson_tikhonov_shift: None,
    };
    let _g = TruncationGuard::new(n_basis_truncate);
    let r = schoen_metric::solve_schoen_metric(cfg).map_err(|e| {
        format!("Schoen k={k} (d={d_x},{d_y},{d_t}) seed={seed} trunc={n_basis_truncate:?}: {e}")
    })?;
    let elapsed = t0.elapsed().as_secs_f64();
    Ok((r, elapsed))
}

// Native basis sizes — measured empirically once for the verdict tables
// to avoid hardcoded surprises if the basis-construction code drifts.
fn ty_native_n_basis(k: u32) -> usize {
    use cy3_rust_solver::route34::basis_truncation_diag::TruncationGuard;
    let _g = TruncationGuard::new(None);
    // Run a tiny solve to read the native basis size. Doesn't have to
    // converge; we only need `n_basis`.
    let cfg = TyMetricConfig {
        k_degree: k,
        n_sample: 200,
        max_iter: 1,
        donaldson_tol: 1.0,
        seed: 0,
        checkpoint_path: None,
        apply_z3_quotient: true,
        adam_refine: None,
        use_gpu: false,
        donaldson_damping: None,
        donaldson_tikhonov_shift: None,
    };
    ty_metric::solve_ty_metric(cfg)
        .map(|r| r.n_basis)
        .unwrap_or(0)
}

fn schoen_native_n_basis(k: u32) -> usize {
    use cy3_rust_solver::route34::basis_truncation_diag::TruncationGuard;
    let _g = TruncationGuard::new(None);
    let (d_x, d_y, d_t) = schoen_bidegree_for_k(k);
    let cfg = SchoenMetricConfig {
        d_x,
        d_y,
        d_t,
        n_sample: 200,
        max_iter: 1,
        donaldson_tol: 1.0,
        seed: 0,
        checkpoint_path: None,
        apply_z3xz3_quotient: true,
        adam_refine: None,
        use_gpu: false,
        donaldson_damping: None,
        donaldson_tikhonov_shift: None,
    };
    schoen_metric::solve_schoen_metric(cfg)
        .map(|r| r.n_basis)
        .unwrap_or(0)
}

// -------------------------------------------------------------------
// Stats helpers
// -------------------------------------------------------------------

/// Run a percentile + BCa bootstrap on the mean of `xs`.
/// Returns `(pct_lo, pct_hi, bca_lo, bca_hi, resamples_used)`.
/// If `boot_resamples == 0`, `xs.len() < 2`, or the bootstrap
/// workspace fails to allocate, returns four `None`s and 0 resamples.
fn bootstrap_mean_ci(
    xs: &[f64],
    boot_resamples: usize,
    boot_seed: u64,
    ci_level: f64,
) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>, usize) {
    if boot_resamples == 0 || xs.len() < 2 {
        return (None, None, None, None, 0);
    }
    let cfg = BootstrapConfig {
        n_resamples: boot_resamples,
        seed: boot_seed,
        ci_level,
    };
    let mut boot = match Bootstrap::new(cfg, xs.len()) {
        Ok(b) => b,
        Err(_) => return (None, None, None, None, 0),
    };
    let result = match boot.run(xs, |s: &[f64]| -> f64 {
        s.iter().sum::<f64>() / s.len() as f64
    }) {
        Ok(r) => r,
        Err(_) => return (None, None, None, None, 0),
    };
    let (pct_lo, pct_hi) = result.percentile_ci;
    let (bca_lo, bca_hi) = match result.bca_ci {
        Some((lo, hi)) => (Some(lo), Some(hi)),
        None => (None, None),
    };
    (Some(pct_lo), Some(pct_hi), bca_lo, bca_hi, boot_resamples)
}

fn mean_se_min_max(xs: &[f64]) -> (f64, f64, f64, f64) {
    let n = xs.len();
    if n == 0 {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let sum: f64 = xs.iter().copied().sum();
    let mean = sum / n as f64;
    let mn = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let mx = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if n < 2 {
        return (mean, f64::NAN, mn, mx);
    }
    let var: f64 = xs
        .iter()
        .map(|&x| (x - mean) * (x - mean))
        .sum::<f64>()
        / (n - 1) as f64;
    let se = (var / n as f64).sqrt();
    (mean, se, mn, mx)
}

// -------------------------------------------------------------------
// Experiment runners
// -------------------------------------------------------------------

fn run_experiment_a(
    seeds: &[u64],
    n_pts: usize,
    donaldson_iters: usize,
    donaldson_tol: f64,
    use_gpu: bool,
    boot_resamples: usize,
    boot_seed: u64,
    boot_ci_level: f64,
    rows_out: &mut Vec<DiagRow>,
) -> ExperimentASummary {
    let ty_native = ty_native_n_basis(3);
    let schoen_native = schoen_native_n_basis(3);
    println!(
        "[exp A] native sizes: TY k=3 n_basis={ty_native}, Schoen k=3 n_basis={schoen_native}"
    );

    // Truncation grid. We always include a `None` (native) entry so
    // we can sanity-check truncated == native at n = native.
    let mut ty_truncs: Vec<Option<usize>> = vec![Some(15), Some(20), Some(25), Some(27), Some(35), Some(50)];
    if ty_native > 50 {
        ty_truncs.push(Some(ty_native)); // native via explicit-equal trunc
    }
    ty_truncs.push(None); // native via no-op
    let mut schoen_truncs: Vec<Option<usize>> = vec![Some(15), Some(20), Some(25)];
    if schoen_native > 25 {
        schoen_truncs.push(Some(schoen_native));
    }
    schoen_truncs.push(None);

    let mut group_summaries: Vec<GroupSummary> = Vec::new();

    for trunc in &ty_truncs {
        let label_n = match trunc {
            Some(n) => *n as i64,
            None => -1,
        };
        let mut sigmas: Vec<f64> = Vec::with_capacity(seeds.len());
        let mut n_basis_actual: usize = 0;
        for &seed in seeds {
            match run_ty(3, n_pts, donaldson_iters, donaldson_tol, seed, *trunc, use_gpu) {
                Ok((r, elapsed)) => {
                    n_basis_actual = r.n_basis;
                    sigmas.push(r.final_sigma_residual);
                    rows_out.push(DiagRow {
                        candidate: "TY".into(),
                        k: 3,
                        n_basis_truncated: label_n,
                        n_basis_native: ty_native,
                        n_basis_actual: r.n_basis,
                        seed,
                        sigma_final: r.final_sigma_residual,
                        sigma_fs_identity: r.sigma_fs_identity,
                        residual_donaldson: r.final_donaldson_residual,
                        iterations_run: r.iterations_run,
                        elapsed_s: elapsed,
                    });
                }
                Err(e) => {
                    eprintln!("[exp A] TY trunc={trunc:?} seed={seed} ERROR: {e}");
                }
            }
        }
        let (mean, se, mn, mx) = mean_se_min_max(&sigmas);
        let (pct_lo, pct_hi, bca_lo, bca_hi, boot_used) =
            bootstrap_mean_ci(&sigmas, boot_resamples, boot_seed, boot_ci_level);
        println!(
            "[exp A] TY      k=3 trunc={:>6} n_basis={:>3} n={} σ = {:.4e} ± {:.2e}  pct95=[{}, {}]  bca95=[{}, {}]",
            match trunc {
                Some(n) => n.to_string(),
                None => "native".into(),
            },
            n_basis_actual,
            sigmas.len(),
            mean,
            se,
            pct_lo.map(|x| format!("{x:.4e}")).unwrap_or_else(|| "n/a".into()),
            pct_hi.map(|x| format!("{x:.4e}")).unwrap_or_else(|| "n/a".into()),
            bca_lo.map(|x| format!("{x:.4e}")).unwrap_or_else(|| "n/a".into()),
            bca_hi.map(|x| format!("{x:.4e}")).unwrap_or_else(|| "n/a".into()),
        );
        group_summaries.push(GroupSummary {
            candidate: "TY".into(),
            k: 3,
            n_basis_truncated: label_n,
            n_basis_actual,
            n_seeds: sigmas.len(),
            sigma_mean: mean,
            sigma_se: se,
            sigma_min: mn,
            sigma_max: mx,
            sigma_pct_ci_low: pct_lo,
            sigma_pct_ci_high: pct_hi,
            sigma_bca_ci_low: bca_lo,
            sigma_bca_ci_high: bca_hi,
            sigma_boot_resamples: boot_used,
            sigma_ci_level: boot_ci_level,
        });
    }

    for trunc in &schoen_truncs {
        let label_n = match trunc {
            Some(n) => *n as i64,
            None => -1,
        };
        let mut sigmas: Vec<f64> = Vec::with_capacity(seeds.len());
        let mut n_basis_actual: usize = 0;
        for &seed in seeds {
            match run_schoen(3, n_pts, donaldson_iters, donaldson_tol, seed, *trunc, use_gpu) {
                Ok((r, elapsed)) => {
                    n_basis_actual = r.n_basis;
                    sigmas.push(r.final_sigma_residual);
                    rows_out.push(DiagRow {
                        candidate: "Schoen".into(),
                        k: 3,
                        n_basis_truncated: label_n,
                        n_basis_native: schoen_native,
                        n_basis_actual: r.n_basis,
                        seed,
                        sigma_final: r.final_sigma_residual,
                        sigma_fs_identity: r.sigma_fs_identity,
                        residual_donaldson: r.final_donaldson_residual,
                        iterations_run: r.iterations_run,
                        elapsed_s: elapsed,
                    });
                }
                Err(e) => {
                    eprintln!("[exp A] Schoen trunc={trunc:?} seed={seed} ERROR: {e}");
                }
            }
        }
        let (mean, se, mn, mx) = mean_se_min_max(&sigmas);
        let (pct_lo, pct_hi, bca_lo, bca_hi, boot_used) =
            bootstrap_mean_ci(&sigmas, boot_resamples, boot_seed, boot_ci_level);
        println!(
            "[exp A] Schoen  k=3 trunc={:>6} n_basis={:>3} n={} σ = {:.4e} ± {:.2e}  pct95=[{}, {}]  bca95=[{}, {}]",
            match trunc {
                Some(n) => n.to_string(),
                None => "native".into(),
            },
            n_basis_actual,
            sigmas.len(),
            mean,
            se,
            pct_lo.map(|x| format!("{x:.4e}")).unwrap_or_else(|| "n/a".into()),
            pct_hi.map(|x| format!("{x:.4e}")).unwrap_or_else(|| "n/a".into()),
            bca_lo.map(|x| format!("{x:.4e}")).unwrap_or_else(|| "n/a".into()),
            bca_hi.map(|x| format!("{x:.4e}")).unwrap_or_else(|| "n/a".into()),
        );
        group_summaries.push(GroupSummary {
            candidate: "Schoen".into(),
            k: 3,
            n_basis_truncated: label_n,
            n_basis_actual,
            n_seeds: sigmas.len(),
            sigma_mean: mean,
            sigma_se: se,
            sigma_min: mn,
            sigma_max: mx,
            sigma_pct_ci_low: pct_lo,
            sigma_pct_ci_high: pct_hi,
            sigma_bca_ci_low: bca_lo,
            sigma_bca_ci_high: bca_hi,
            sigma_boot_resamples: boot_used,
            sigma_ci_level: boot_ci_level,
        });
    }

    // Verdict numbers.
    let ty_at_27 = group_summaries
        .iter()
        .find(|g| g.candidate == "TY" && g.n_basis_actual == 27)
        .map(|g| g.sigma_mean);
    let schoen_at_27 = group_summaries
        .iter()
        .find(|g| g.candidate == "Schoen" && g.n_basis_actual == 27)
        .map(|g| g.sigma_mean);
    let delta_sigma_at_n27 = match (ty_at_27, schoen_at_27) {
        (Some(t), Some(s)) => Some(t - s),
        _ => None,
    };

    let ty_native_sigma = group_summaries
        .iter()
        .find(|g| g.candidate == "TY" && g.n_basis_truncated == -1)
        .map(|g| g.sigma_mean);
    let schoen_native_sigma = group_summaries
        .iter()
        .find(|g| g.candidate == "Schoen" && g.n_basis_truncated == -1)
        .map(|g| g.sigma_mean);
    let delta_sigma_native = match (ty_native_sigma, schoen_native_sigma) {
        (Some(t), Some(s)) => Some(t - s),
        _ => None,
    };

    let matched_to_native_ratio = match (delta_sigma_at_n27, delta_sigma_native) {
        (Some(m), Some(n)) if n.abs() > 1e-12 => Some(m / n),
        _ => None,
    };

    println!();
    println!("[exp A] VERDICT NUMBERS");
    println!(
        "  σ_TY(n=27)         = {:?}",
        ty_at_27.map(|x| format!("{x:.4e}"))
    );
    println!(
        "  σ_Schoen(n=27)     = {:?}",
        schoen_at_27.map(|x| format!("{x:.4e}"))
    );
    println!(
        "  Δσ(matched n=27)   = {:?}",
        delta_sigma_at_n27.map(|x| format!("{x:+.4e}"))
    );
    println!(
        "  σ_TY(native=87)    = {:?}",
        ty_native_sigma.map(|x| format!("{x:.4e}"))
    );
    println!(
        "  σ_Schoen(native=27)= {:?}",
        schoen_native_sigma.map(|x| format!("{x:.4e}"))
    );
    println!(
        "  Δσ(native)         = {:?}",
        delta_sigma_native.map(|x| format!("{x:+.4e}"))
    );
    println!(
        "  matched / native   = {:?}",
        matched_to_native_ratio.map(|x| format!("{x:.3}"))
    );

    ExperimentASummary {
        n_pts,
        seeds: seeds.to_vec(),
        rows: group_summaries,
        delta_sigma_at_n27,
        delta_sigma_native,
        matched_to_native_ratio,
    }
}

fn run_experiment_b(
    seeds: &[u64],
    n_pts: usize,
    donaldson_iters: usize,
    donaldson_tol: f64,
    use_gpu: bool,
    boot_resamples: usize,
    boot_seed: u64,
    boot_ci_level: f64,
    rows_out: &mut Vec<DiagRow>,
) -> ExperimentBSummary {
    let mut group_summaries: Vec<GroupSummary> = Vec::new();
    let mut delta_per_k: Vec<KScanRow> = Vec::new();

    for &k in &[2u32, 3, 4] {
        let ty_native = ty_native_n_basis(k);
        let schoen_native = schoen_native_n_basis(k);
        // TY native (no truncation).
        let mut ty_sigmas: Vec<f64> = Vec::new();
        for &seed in seeds {
            match run_ty(k, n_pts, donaldson_iters, donaldson_tol, seed, None, use_gpu) {
                Ok((r, elapsed)) => {
                    ty_sigmas.push(r.final_sigma_residual);
                    rows_out.push(DiagRow {
                        candidate: "TY".into(),
                        k,
                        n_basis_truncated: -1,
                        n_basis_native: ty_native,
                        n_basis_actual: r.n_basis,
                        seed,
                        sigma_final: r.final_sigma_residual,
                        sigma_fs_identity: r.sigma_fs_identity,
                        residual_donaldson: r.final_donaldson_residual,
                        iterations_run: r.iterations_run,
                        elapsed_s: elapsed,
                    });
                }
                Err(e) => eprintln!("[exp B] TY k={k} seed={seed} ERROR: {e}"),
            }
        }
        let (mean_ty, se_ty, mn_ty, mx_ty) = mean_se_min_max(&ty_sigmas);
        let (ty_pct_lo, ty_pct_hi, ty_bca_lo, ty_bca_hi, ty_boot_used) =
            bootstrap_mean_ci(&ty_sigmas, boot_resamples, boot_seed, boot_ci_level);
        println!(
            "[exp B] TY      k={k} n_basis(native)={ty_native} σ = {mean_ty:.4e} ± {se_ty:.2e}  bca95=[{}, {}]",
            ty_bca_lo.map(|x| format!("{x:.4e}")).unwrap_or_else(|| "n/a".into()),
            ty_bca_hi.map(|x| format!("{x:.4e}")).unwrap_or_else(|| "n/a".into()),
        );
        group_summaries.push(GroupSummary {
            candidate: "TY".into(),
            k,
            n_basis_truncated: -1,
            n_basis_actual: ty_native,
            n_seeds: ty_sigmas.len(),
            sigma_mean: mean_ty,
            sigma_se: se_ty,
            sigma_min: mn_ty,
            sigma_max: mx_ty,
            sigma_pct_ci_low: ty_pct_lo,
            sigma_pct_ci_high: ty_pct_hi,
            sigma_bca_ci_low: ty_bca_lo,
            sigma_bca_ci_high: ty_bca_hi,
            sigma_boot_resamples: ty_boot_used,
            sigma_ci_level: boot_ci_level,
        });

        // Schoen native (no truncation).
        let mut schoen_sigmas: Vec<f64> = Vec::new();
        for &seed in seeds {
            match run_schoen(k, n_pts, donaldson_iters, donaldson_tol, seed, None, use_gpu) {
                Ok((r, elapsed)) => {
                    schoen_sigmas.push(r.final_sigma_residual);
                    rows_out.push(DiagRow {
                        candidate: "Schoen".into(),
                        k,
                        n_basis_truncated: -1,
                        n_basis_native: schoen_native,
                        n_basis_actual: r.n_basis,
                        seed,
                        sigma_final: r.final_sigma_residual,
                        sigma_fs_identity: r.sigma_fs_identity,
                        residual_donaldson: r.final_donaldson_residual,
                        iterations_run: r.iterations_run,
                        elapsed_s: elapsed,
                    });
                }
                Err(e) => eprintln!("[exp B] Schoen k={k} seed={seed} ERROR: {e}"),
            }
        }
        let (mean_s, se_s, mn_s, mx_s) = mean_se_min_max(&schoen_sigmas);
        let (s_pct_lo, s_pct_hi, s_bca_lo, s_bca_hi, s_boot_used) =
            bootstrap_mean_ci(&schoen_sigmas, boot_resamples, boot_seed, boot_ci_level);
        println!(
            "[exp B] Schoen  k={k} n_basis(native)={schoen_native} σ = {mean_s:.4e} ± {se_s:.2e}  bca95=[{}, {}]",
            s_bca_lo.map(|x| format!("{x:.4e}")).unwrap_or_else(|| "n/a".into()),
            s_bca_hi.map(|x| format!("{x:.4e}")).unwrap_or_else(|| "n/a".into()),
        );
        group_summaries.push(GroupSummary {
            candidate: "Schoen".into(),
            k,
            n_basis_truncated: -1,
            n_basis_actual: schoen_native,
            n_seeds: schoen_sigmas.len(),
            sigma_mean: mean_s,
            sigma_se: se_s,
            sigma_min: mn_s,
            sigma_max: mx_s,
            sigma_pct_ci_low: s_pct_lo,
            sigma_pct_ci_high: s_pct_hi,
            sigma_bca_ci_low: s_bca_lo,
            sigma_bca_ci_high: s_bca_hi,
            sigma_boot_resamples: s_boot_used,
            sigma_ci_level: boot_ci_level,
        });

        delta_per_k.push(KScanRow {
            k,
            sigma_ty: mean_ty,
            se_ty,
            n_basis_ty: ty_native,
            sigma_schoen: mean_s,
            se_schoen: se_s,
            n_basis_schoen: schoen_native,
            delta_sigma: mean_ty - mean_s,
        });
    }

    // 1/k² check.
    let d2 = delta_per_k.iter().find(|r| r.k == 2).map(|r| r.delta_sigma);
    let d3 = delta_per_k.iter().find(|r| r.k == 3).map(|r| r.delta_sigma);
    let d4 = delta_per_k.iter().find(|r| r.k == 4).map(|r| r.delta_sigma);
    let delta_ratio_k3_k2 = match (d3, d2) {
        (Some(a), Some(b)) if b.abs() > 1e-12 => Some(a / b),
        _ => None,
    };
    let delta_ratio_k4_k2 = match (d4, d2) {
        (Some(a), Some(b)) if b.abs() > 1e-12 => Some(a / b),
        _ => None,
    };

    println!();
    println!("[exp B] K-SCAN VERDICT (Δσ = σ_TY - σ_Schoen at native bases)");
    for r in &delta_per_k {
        println!(
            "  k={}  Δσ = {:+.4e}  (TY n_basis={}, Schoen n_basis={})",
            r.k, r.delta_sigma, r.n_basis_ty, r.n_basis_schoen
        );
    }
    println!(
        "  Δσ(k=3)/Δσ(k=2) = {:?}  (1/k² scaling predicts ~{:.3})",
        delta_ratio_k3_k2.map(|x| format!("{x:.3}")),
        (2.0_f64 / 3.0).powi(2)
    );
    println!(
        "  Δσ(k=4)/Δσ(k=2) = {:?}  (1/k² scaling predicts ~{:.3})",
        delta_ratio_k4_k2.map(|x| format!("{x:.3}")),
        (2.0_f64 / 4.0).powi(2)
    );

    ExperimentBSummary {
        n_pts,
        seeds: seeds.to_vec(),
        rows: group_summaries,
        delta_per_k,
        delta_ratio_k3_k2,
        delta_ratio_k4_k2,
    }
}

// -------------------------------------------------------------------
// CSV writer
// -------------------------------------------------------------------

fn write_csv(path: &PathBuf, rows: &[DiagRow]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut f = fs::File::create(path)?;
    writeln!(
        f,
        "candidate,k,n_basis_truncated,n_basis_native,n_basis_actual,seed,sigma_final,sigma_fs_identity,residual_donaldson,iterations_run,elapsed_s"
    )?;
    for r in rows {
        writeln!(
            f,
            "{},{},{},{},{},{},{:.10e},{:.10e},{:.10e},{},{:.6}",
            r.candidate,
            r.k,
            r.n_basis_truncated,
            r.n_basis_native,
            r.n_basis_actual,
            r.seed,
            r.sigma_final,
            r.sigma_fs_identity,
            r.residual_donaldson,
            r.iterations_run,
            r.elapsed_s
        )?;
    }
    Ok(())
}

// -------------------------------------------------------------------
// main
// -------------------------------------------------------------------

/// P5.10's canonical 20-seed roster.
const P5_10_SEEDS: [u64; 20] = [
    42, 100, 12345, 7, 99, 1, 2, 3, 4, 5, 137, 271, 314, 666, 1000, 2024, 4242, 57005, 48879, 51966,
];

fn main() {
    let cli = Cli::parse();
    // Seed selection: --n-seeds (if set) overrides the default --seeds list,
    // taking the first N from the P5.10 roster. Explicit --seeds always wins
    // when --n-seeds is not provided.
    let seeds: Vec<u64> = match cli.n_seeds {
        Some(n) => {
            if n == 0 {
                panic!("--n-seeds must be ≥ 1");
            }
            if n > P5_10_SEEDS.len() {
                panic!(
                    "--n-seeds = {n} exceeds available P5.10 roster ({})",
                    P5_10_SEEDS.len()
                );
            }
            P5_10_SEEDS[..n].to_vec()
        }
        None => cli
            .seeds
            .split(',')
            .map(|s| s.trim().parse().expect("seed must be u64"))
            .collect(),
    };

    // Output path resolution: --output overrides --json-output and derives
    // a sibling .csv path.
    let json_output = match cli.output.as_ref() {
        Some(p) => p.clone(),
        None => cli.json_output.clone(),
    };
    let csv_output = match cli.output.as_ref() {
        Some(p) => {
            // Replace .json suffix with .csv (or append .csv).
            let mut q = p.clone();
            let stem = q
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "output".into());
            q.set_file_name(format!("{stem}.csv"));
            q
        }
        None => cli.csv_output.clone(),
    };

    println!("=== P-BASIS-CONV diagnostic ===");
    println!("n_pts           = {}", cli.n_pts);
    println!("donaldson_iters = {}", cli.donaldson_iters);
    println!("donaldson_tol   = {}", cli.donaldson_tol);
    println!("seeds (n={})    = {:?}", seeds.len(), seeds);
    println!("use_gpu         = {}", cli.use_gpu);
    println!("boot_resamples  = {}", cli.boot_resamples);
    println!("boot_seed       = {}", cli.boot_seed);
    println!("boot_ci_level   = {}", cli.boot_ci_level);
    println!("json_output     = {}", json_output.display());
    println!("csv_output      = {}", csv_output.display());
    println!();

    let mut rows: Vec<DiagRow> = Vec::new();
    let exp_a = if !cli.skip_exp_a {
        println!(">>> Experiment A: matched-basis-size σ at k=3");
        Some(run_experiment_a(
            &seeds,
            cli.n_pts,
            cli.donaldson_iters,
            cli.donaldson_tol,
            cli.use_gpu,
            cli.boot_resamples,
            cli.boot_seed,
            cli.boot_ci_level,
            &mut rows,
        ))
    } else {
        None
    };

    let exp_b = if !cli.skip_exp_b {
        println!();
        println!(">>> Experiment B: native-basis k-scan");
        Some(run_experiment_b(
            &seeds,
            cli.n_pts,
            cli.donaldson_iters,
            cli.donaldson_tol,
            cli.use_gpu,
            cli.boot_resamples,
            cli.boot_seed,
            cli.boot_ci_level,
            &mut rows,
        ))
    } else {
        None
    };

    if let Err(e) = write_csv(&csv_output, &rows) {
        eprintln!("CSV write failed for {:?}: {e}", csv_output);
    } else {
        println!();
        println!("CSV written: {}", csv_output.display());
    }

    let out = DiagOutput {
        config: ConfigEcho {
            n_pts: cli.n_pts,
            donaldson_iters: cli.donaldson_iters,
            donaldson_tol: cli.donaldson_tol,
            seeds: seeds.clone(),
            use_gpu: cli.use_gpu,
            boot_resamples: cli.boot_resamples,
            boot_seed: cli.boot_seed,
            boot_ci_level: cli.boot_ci_level,
        },
        experiment_a: exp_a,
        experiment_b: exp_b,
        raw_rows: rows,
    };
    if let Some(parent) = json_output.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = fs::create_dir_all(parent);
        }
    }
    match serde_json::to_string_pretty(&out) {
        Ok(s) => {
            if let Err(e) = fs::write(&json_output, s) {
                eprintln!("JSON write failed: {e}");
            } else {
                println!("JSON written: {}", json_output.display());
            }
        }
        Err(e) => eprintln!("JSON serialise failed: {e}"),
    }

    // Verdict summary.
    println!();
    println!("=== VERDICT ===");
    if let Some(a) = out.experiment_a.as_ref() {
        println!("Experiment A (matched-basis k=3):");
        if let (Some(matched), Some(native)) = (a.delta_sigma_at_n27, a.delta_sigma_native) {
            println!("  Δσ(matched n=27)  = {matched:+.4e}");
            println!("  Δσ(native)        = {native:+.4e}");
            if let Some(ratio) = a.matched_to_native_ratio {
                println!("  matched/native    = {ratio:.3}");
                if ratio.abs() < 0.25 {
                    println!("  ⇒ basis-size artefact DOMINATES (matched gap is < 25% of native gap)");
                } else if ratio.abs() < 0.5 {
                    println!("  ⇒ basis-size artefact is a MAJOR contributor (matched < 50% of native)");
                } else {
                    println!("  ⇒ matched gap PERSISTS — discrimination is at least partly real geometry");
                }
            }
        } else {
            println!("  (insufficient matched/native data to compute ratio)");
        }
    }
    if let Some(b) = out.experiment_b.as_ref() {
        println!("Experiment B (native-basis k-scan):");
        for r in &b.delta_per_k {
            println!(
                "  k={}  Δσ = {:+.4e}  (TY n_basis={}, Schoen n_basis={})",
                r.k, r.delta_sigma, r.n_basis_ty, r.n_basis_schoen
            );
        }
        if let Some(r3) = b.delta_ratio_k3_k2 {
            println!(
                "  Δσ(k=3)/Δσ(k=2) = {:.3}  (1/k² predicts {:.3})",
                r3,
                (2.0_f64 / 3.0).powi(2)
            );
        }
        if let Some(r4) = b.delta_ratio_k4_k2 {
            println!(
                "  Δσ(k=4)/Δσ(k=2) = {:.3}  (1/k² predicts {:.3})",
                r4,
                (2.0_f64 / 4.0).powi(2)
            );
        }
    }
}
