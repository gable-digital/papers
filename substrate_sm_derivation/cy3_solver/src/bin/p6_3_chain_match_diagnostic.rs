// This binary intentionally uses the deprecated chain_matcher /
// metric_laplacian APIs — it exists exclusively for diagnostic /
// re-run purposes against the retracted P6.3 result. The runtime
// banner below makes the retraction visible to anyone running the
// binary; the source-level retraction lives in the module-level
// `//!` doc and the `#[deprecated]` annotations the binary references.
#![allow(deprecated)]
//! P6.3 — Metric-Laplacian + chain-matcher discrimination diagnostic.
//!
//! For each candidate (TY/Z3 and Schoen/Z3xZ3):
//!   1. Run a Donaldson-balanced metric solve (k=3, n_pts=10000).
//!   2. Compute the lowest ~15 metric Laplacian eigenvalues via the
//!      Galerkin / Rayleigh-Ritz solver in
//!      `route34::metric_laplacian`.
//!   3. Match the spectrum against the (√2)-quark-chain predicted by
//!      the journal entry
//!      `book/journal/2026-04-29/2026-04-29-charged-fermion-spectrum-from-e8-sub-coxeter-structure.adoc`.
//!   4. Report the residual: the candidate with the lower residual
//!      better fits the predicted chain positions.
//!
//! Output:
//!   * Stdout: per-candidate eigenvalues, chain-match summary.
//!   * JSON: `output/p6_3_chain_match_diagnostic.json`.

use clap::Parser;
use cy3_rust_solver::route34::chain_matcher::{
    match_chain, match_chain_hungarian, ratio_pattern, ChainMatchResult, ChainType, RatioMatch,
};
use cy3_rust_solver::route34::cy3_metric_unified::{
    Cy3AdamOverride, Cy3MetricResultKind, Cy3MetricSolver, Cy3MetricSpec, SchoenSolver,
    TianYauSolver,
};
use cy3_rust_solver::route34::metric_laplacian::{
    compute_metric_laplacian_spectrum, MetricLaplacianConfig, MetricLaplacianSpectrum,
};
use cy3_rust_solver::route34::metric_laplacian_projected::{
    compute_projected_metric_laplacian_spectrum, ProjectionKind,
};
use cy3_rust_solver::route34::schoen_metric::AdamRefineConfig;
use cy3_rust_solver::route34::yukawa_pipeline::Cy3MetricResultBackground;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "P6.3 metric-Laplacian + chain-match discrimination diagnostic")]
struct Cli {
    /// Sample-cloud size for each candidate.
    #[arg(long, default_value_t = 10000)]
    n_pts: usize,

    /// Bigraded section-basis degree for the Donaldson metric solve.
    #[arg(long, default_value_t = 3)]
    k: u32,

    /// Donaldson iteration budget.
    #[arg(long, default_value_t = 8)]
    max_iter: usize,

    /// Single-seed PRNG seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Maximum total degree of the test-function basis.
    #[arg(long, default_value_t = 4)]
    test_degree: u32,

    /// Number of low eigenvalues to extract / report.
    #[arg(long, default_value_t = 15)]
    n_low: usize,

    /// Chain to match against. `quark` (default) corresponds to the
    /// (√2)-chain; `lepton` to the φ-chain.
    #[arg(long, default_value = "quark")]
    chain: String,

    #[arg(long, default_value = "output/p6_3_chain_match_diagnostic.json")]
    output: PathBuf,

    /// If set, run a basis-size convergence sweep: re-run the
    /// chain-match diagnostic at `test_degree` ∈ `--sweep-degrees` for
    /// both candidates and write
    /// `output/p6_3b_basis_convergence_sweep.json`.
    #[arg(long, default_value_t = false)]
    basis_sweep: bool,

    /// Comma-separated list of test_degree values to use under
    /// `--basis-sweep`. Default: 3,4,5.
    #[arg(long, default_value = "3,4,5")]
    sweep_degrees: String,

    /// Per-candidate, per-degree wall-clock cap (seconds). If a
    /// (candidate, degree) point exceeds this, it is skipped and
    /// flagged in the report. Default 300 (5 min).
    #[arg(long, default_value_t = 300.0)]
    sweep_time_cap_s: f64,

    /// Assignment algorithm for chain matching: `greedy` (P6.3 legacy)
    /// or `hungarian` (P7.2 optimal). Default `hungarian`.
    #[arg(long, default_value = "hungarian")]
    assign: String,

    /// P7.2 — when set under `--basis-sweep`, runs both quark and
    /// lepton chains for every (candidate, test_degree) combination
    /// and writes `output/p7_2_chain_sweep.json` (instead of the
    /// P6.3b `p6_3b_basis_convergence_sweep.json` filename, which is
    /// preserved for the legacy single-chain runs).
    #[arg(long, default_value_t = false)]
    p7_2: bool,

    /// Number of low eigenvalues to inspect for the ratio-pattern
    /// analysis at test_degree=4 (P7.2 §4). Default 30.
    #[arg(long, default_value_t = 30)]
    ratio_floor: usize,

    /// Top-N largest consecutive ratios to surface in the ratio
    /// pattern report. Default 5.
    #[arg(long, default_value_t = 5)]
    ratio_top: usize,

    /// P7.11 — orthogonalize the (projected) test basis under L²(M)
    /// inner product before the Galerkin assembly, and run the
    /// standard Hermitian EVP on the orthonormal basis. Eliminates
    /// the negative-eigenvalue / sign-flip pathology diagnosed in
    /// P7.2. When `--orthogonalize` is set, the binary also routes
    /// through the trivial-irrep projected basis
    /// (`metric_laplacian_projected`) — TY/Z3 → Z/3 filter, Schoen
    /// → Z/3×Z/3 filter — instead of the legacy bigraded basis.
    #[arg(long, default_value_t = false)]
    orthogonalize: bool,

    /// P7.11 — Donaldson convergence tolerance. Threaded through to
    /// the metric solver. Default 1e-9 (legacy behaviour); pass
    /// `1e-6` (looser) for faster but coarser solves on big sweeps.
    #[arg(long, default_value_t = 1.0e-9)]
    donaldson_tol: f64,

    /// P7.11 — alias for `--sweep-degrees` accepted for command-line
    /// parity with sibling P7.* binaries (which use the longer name).
    #[arg(long)]
    test_degree_list: Option<String>,

    /// P7.9 — number of post-Donaldson Adam σ-functional descent
    /// iterations applied inside the metric solve. 0 disables Adam
    /// (legacy Donaldson-only behaviour). 50 is the recommended
    /// production value for k≥4 where σ_Donaldson flatlines on the
    /// invariant basis.
    #[arg(long, default_value_t = 0usize)]
    adam_iters: usize,

    /// P7.9 — learning rate for the Adam σ-refinement.
    #[arg(long, default_value_t = 1.0e-3)]
    adam_lr: f64,

    /// P7.9 — finite-difference step for the Adam σ-gradient.
    #[arg(long, default_value_t = 1.0e-3)]
    adam_fd_step: f64,

    /// P-DONALDSON-GPU — use GPU Donaldson T-operator inside the
    /// balancing loop and, if Adam is enabled, the GPU σ-evaluator
    /// inside the FD-Adam loop. Requires the `gpu` feature; falls
    /// back to CPU if GPU init fails.
    #[arg(long, default_value_t = false)]
    use_gpu: bool,
}

/// Build the [`Cy3AdamOverride`] from CLI flags. `--adam-iters 0`
/// yields a default override (no Adam) but still propagates the
/// `--use-gpu` flag into the Donaldson balancing loop.
fn build_adam_override(cli: &Cli) -> Cy3AdamOverride {
    if cli.adam_iters == 0 {
        return Cy3AdamOverride {
            adam_refine: None,
            use_gpu_donaldson: cli.use_gpu,
        };
    }
    Cy3AdamOverride {
        adam_refine: Some(AdamRefineConfig {
            max_iters: cli.adam_iters,
            learning_rate: cli.adam_lr,
            fd_step: Some(cli.adam_fd_step),
            tol: 1.0e-7,
            use_gpu: cli.use_gpu,
        }),
        use_gpu_donaldson: cli.use_gpu,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AssignAlgo {
    Greedy,
    Hungarian,
}

fn parse_assign(s: &str) -> AssignAlgo {
    match s.to_ascii_lowercase().as_str() {
        "greedy" => AssignAlgo::Greedy,
        _ => AssignAlgo::Hungarian,
    }
}

fn match_chain_with(algo: AssignAlgo, eigs: &[f64], chain_type: ChainType) -> ChainMatchResult {
    match algo {
        AssignAlgo::Greedy => match_chain(eigs, chain_type),
        AssignAlgo::Hungarian => match_chain_hungarian(eigs, chain_type),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CandidateReport {
    label: String,
    spectrum: MetricLaplacianSpectrum,
    chain_match: ChainMatchResult,
    metric_iters: usize,
    metric_sigma: f64,
    metric_seconds: f64,
    spectrum_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiagnosticReport {
    label: &'static str,
    chain_type: String,
    n_pts: usize,
    k: u32,
    max_iter: usize,
    seed: u64,
    test_degree: u32,
    n_low: usize,
    /// P-REPRO-2 — compute path used for the Donaldson tree reduction
    /// ("cpu" | "gpu"). CPU and GPU paths are deterministic-but-bit-
    /// distinct; persisting this lets audit-time detect cross-path
    /// drift. Optional for backward compatibility.
    #[serde(default)]
    compute_path: Option<String>,
    candidates: Vec<CandidateReport>,
    /// Difference Schoen.residual - TY.residual. Negative = Schoen
    /// fits the predicted chain better; positive = TY fits better.
    /// `None` if either residual is non-finite.
    discrimination_delta: Option<f64>,
    total_elapsed_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SweepRow {
    test_degree: u32,
    basis_dim: usize,
    ty_residual: Option<f64>,
    schoen_residual: Option<f64>,
    delta: Option<f64>,
    ty_seconds: f64,
    schoen_seconds: f64,
    ty_skipped_reason: Option<String>,
    schoen_skipped_reason: Option<String>,
    ty_metric_iters: usize,
    ty_metric_sigma: f64,
    schoen_metric_iters: usize,
    schoen_metric_sigma: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SweepReport {
    label: &'static str,
    chain_type: String,
    n_pts: usize,
    k: u32,
    max_iter: usize,
    seed: u64,
    n_low: usize,
    sweep_time_cap_s: f64,
    /// P-REPRO-2 — compute path used for the Donaldson tree reduction
    /// ("cpu" | "gpu"). See `DiagnosticReport::compute_path`.
    #[serde(default)]
    compute_path: Option<String>,
    rows: Vec<SweepRow>,
    converged: bool,
    converged_test_degree: Option<u32>,
    converged_ty_residual: Option<f64>,
    converged_schoen_residual: Option<f64>,
    converged_delta: Option<f64>,
    convergence_notes: Vec<String>,
    total_elapsed_s: f64,
}

fn parse_chain(s: &str) -> ChainType {
    match s.to_ascii_lowercase().as_str() {
        "lepton" | "phi" | "leptons" => ChainType::Lepton,
        _ => ChainType::Quark,
    }
}

fn run_candidate(
    label: &str,
    spec: Cy3MetricSpec,
    solver: &dyn Cy3MetricSolver,
    chain_type: ChainType,
    test_degree: u32,
    n_low: usize,
    algo: AssignAlgo,
    orthogonalize: bool,
    adam_override: &Cy3AdamOverride,
) -> Result<CandidateReport, String> {
    let t_metric = Instant::now();
    let r = solver
        .solve_metric_with_adam(&spec, adam_override)
        .map_err(|e| format!("{label}: metric solve failed: {e}"))?;
    let summary = r.summary();
    let metric_seconds = t_metric.elapsed().as_secs_f64();

    let bg = match &r {
        Cy3MetricResultKind::TianYau(t) => Cy3MetricResultBackground::from_ty(t.as_ref()),
        Cy3MetricResultKind::Schoen(t) => Cy3MetricResultBackground::from_schoen(t.as_ref()),
    };

    let cfg = MetricLaplacianConfig {
        max_total_degree: test_degree,
        n_low_eigenvalues: n_low,
        orthogonalize_first: orthogonalize,
        ..MetricLaplacianConfig::default()
    };
    let t_spec = Instant::now();
    let spectrum = if orthogonalize {
        let projection = if label.starts_with("TY") {
            ProjectionKind::TianYauZ3
        } else {
            ProjectionKind::SchoenZ3xZ3
        };
        compute_projected_metric_laplacian_spectrum(&bg, &cfg, projection).spectrum
    } else {
        compute_metric_laplacian_spectrum(&bg, &cfg)
    };
    let spectrum_seconds = t_spec.elapsed().as_secs_f64();

    let chain_match = match_chain_with(algo, &spectrum.eigenvalues, chain_type);

    Ok(CandidateReport {
        label: label.to_string(),
        spectrum,
        chain_match,
        metric_iters: summary.iterations_run,
        metric_sigma: summary.final_sigma_residual,
        metric_seconds,
        spectrum_seconds,
    })
}

/// Compute Galerkin matrix condition diagnostic from the full eigenvalue
/// spectrum: ratio of largest to smallest positive eigenvalue. This is a
/// lower bound on the M^{-1}K condition; if the smallest eigenvalue is
/// near machine epsilon the Galerkin pencil is ill-conditioned.
fn galerkin_condition(spectrum: &MetricLaplacianSpectrum) -> Option<f64> {
    let mut pos: Vec<f64> = spectrum
        .eigenvalues_full
        .iter()
        .cloned()
        .filter(|&v| v.is_finite() && v > 1.0e-15)
        .collect();
    pos.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if pos.is_empty() {
        return None;
    }
    Some(pos[pos.len() - 1] / pos[0])
}

fn parse_sweep_degrees(s: &str) -> Vec<u32> {
    s.split(',')
        .filter_map(|t| t.trim().parse::<u32>().ok())
        .filter(|&d| d >= 1)
        .collect()
}

fn run_sweep(cli: &Cli, chain_type: ChainType, chain_label: &str) {
    let degrees_src = cli
        .test_degree_list
        .as_ref()
        .map(|s| s.as_str())
        .unwrap_or(cli.sweep_degrees.as_str());
    let degrees = parse_sweep_degrees(degrees_src);
    if degrees.is_empty() {
        eprintln!("No valid sweep degrees parsed from '{}'", degrees_src);
        std::process::exit(1);
    }
    eprintln!(
        "=== P6.3b basis-size convergence sweep ===\n  n_pts={}  k={}  max_iter={}  seed={}\n  n_low={}  degrees={:?}  chain={}\n  per-point time cap = {:.0}s",
        cli.n_pts, cli.k, cli.max_iter, cli.seed, cli.n_low, degrees, chain_label, cli.sweep_time_cap_s
    );

    let t_total = Instant::now();
    let mut rows: Vec<SweepRow> = Vec::new();
    let adam_override = build_adam_override(cli);

    for &deg in &degrees {
        eprintln!("\n--- test_degree = {} ---", deg);
        let ty_spec = Cy3MetricSpec::TianYau {
            k: cli.k,
            n_sample: cli.n_pts,
            max_iter: cli.max_iter,
            donaldson_tol: cli.donaldson_tol,
            seed: cli.seed,
        };
        let schoen_spec = Cy3MetricSpec::Schoen {
            d_x: cli.k,
            d_y: cli.k,
            d_t: 1.max(cli.k / 2),
            n_sample: cli.n_pts,
            max_iter: cli.max_iter,
            donaldson_tol: cli.donaldson_tol,
            seed: cli.seed,
        };

        let t_ty = Instant::now();
        let ty_report = run_candidate(
            "TY/Z3",
            ty_spec,
            &TianYauSolver,
            chain_type,
            deg,
            cli.n_low,
            parse_assign(&cli.assign),
            cli.orthogonalize,
            &adam_override,
        );
        let ty_seconds = t_ty.elapsed().as_secs_f64();
        let (ty_residual, ty_skip, ty_iters, ty_sigma, basis_dim_ty) = match &ty_report {
            Ok(c) => {
                print_candidate(c);
                let r = c.chain_match.residual_log_f64;
                (
                    if r.is_finite() { Some(r) } else { None },
                    if ty_seconds > cli.sweep_time_cap_s {
                        Some(format!(
                            "exceeded time cap ({:.1}s > {:.0}s)",
                            ty_seconds, cli.sweep_time_cap_s
                        ))
                    } else {
                        None
                    },
                    c.metric_iters,
                    c.metric_sigma,
                    c.spectrum.basis_dim,
                )
            }
            Err(e) => {
                eprintln!("  TY/Z3 SKIPPED: {e}");
                (None, Some(e.clone()), 0, f64::NAN, 0)
            }
        };

        let t_sch = Instant::now();
        let sch_report = run_candidate(
            "Schoen/Z3xZ3",
            schoen_spec,
            &SchoenSolver,
            chain_type,
            deg,
            cli.n_low,
            parse_assign(&cli.assign),
            cli.orthogonalize,
            &adam_override,
        );
        let schoen_seconds = t_sch.elapsed().as_secs_f64();
        let (sch_residual, sch_skip, sch_iters, sch_sigma, basis_dim_sch) = match &sch_report {
            Ok(c) => {
                print_candidate(c);
                let r = c.chain_match.residual_log_f64;
                (
                    if r.is_finite() { Some(r) } else { None },
                    if schoen_seconds > cli.sweep_time_cap_s {
                        Some(format!(
                            "exceeded time cap ({:.1}s > {:.0}s)",
                            schoen_seconds, cli.sweep_time_cap_s
                        ))
                    } else {
                        None
                    },
                    c.metric_iters,
                    c.metric_sigma,
                    c.spectrum.basis_dim,
                )
            }
            Err(e) => {
                eprintln!("  Schoen/Z3xZ3 SKIPPED: {e}");
                (None, Some(e.clone()), 0, f64::NAN, 0)
            }
        };

        let basis_dim = basis_dim_ty.max(basis_dim_sch);
        let delta = match (ty_residual, sch_residual) {
            (Some(t), Some(s)) => Some(s - t),
            _ => None,
        };

        eprintln!(
            "  >>> deg={} basis_dim={} TY={:?} Schoen={:?} delta={:?}",
            deg, basis_dim, ty_residual, sch_residual, delta
        );

        rows.push(SweepRow {
            test_degree: deg,
            basis_dim,
            ty_residual,
            schoen_residual: sch_residual,
            delta,
            ty_seconds,
            schoen_seconds,
            ty_skipped_reason: ty_skip,
            schoen_skipped_reason: sch_skip,
            ty_metric_iters: ty_iters,
            ty_metric_sigma: ty_sigma,
            schoen_metric_iters: sch_iters,
            schoen_metric_sigma: sch_sigma,
        });

        // Stop early on the next iteration if we just exceeded the
        // time cap on either candidate — higher degrees are strictly
        // more expensive.
        let last = rows.last().unwrap();
        if last.ty_skipped_reason.is_some() || last.schoen_skipped_reason.is_some() {
            eprintln!(
                "  Skipping remaining higher degrees (>{}) because time cap was exceeded.",
                deg
            );
            break;
        }
    }

    // Convergence verdict: consecutive deltas within 10% of each
    // other, sign of delta unchanged, and TY/Schoen residuals each
    // within 10% across consecutive degrees.
    let mut notes: Vec<String> = Vec::new();
    let mut converged = false;
    let mut converged_deg: Option<u32> = None;
    let mut converged_ty: Option<f64> = None;
    let mut converged_sch: Option<f64> = None;
    let mut converged_delta: Option<f64> = None;

    let usable: Vec<&SweepRow> = rows
        .iter()
        .filter(|r| r.ty_residual.is_some() && r.schoen_residual.is_some())
        .collect();
    if usable.len() < 2 {
        notes.push(format!(
            "Need at least 2 successful (TY, Schoen) pairs to assess convergence; got {}.",
            usable.len()
        ));
    } else {
        for w in usable.windows(2) {
            let a = w[0];
            let b = w[1];
            let ty_a = a.ty_residual.unwrap();
            let ty_b = b.ty_residual.unwrap();
            let sch_a = a.schoen_residual.unwrap();
            let sch_b = b.schoen_residual.unwrap();
            let d_a = a.delta.unwrap();
            let d_b = b.delta.unwrap();
            let ty_rel = (ty_b - ty_a).abs() / ty_a.abs().max(1e-12);
            let sch_rel = (sch_b - sch_a).abs() / sch_a.abs().max(1e-12);
            let delta_rel = (d_b - d_a).abs() / d_a.abs().max(1e-12);
            let sign_kept = (d_a >= 0.0) == (d_b >= 0.0);
            notes.push(format!(
                "deg {}->{}: TY rel-Δ={:.3} Schoen rel-Δ={:.3} delta rel-Δ={:.3} sign-kept={}",
                a.test_degree, b.test_degree, ty_rel, sch_rel, delta_rel, sign_kept
            ));
            if ty_rel < 0.10 && sch_rel < 0.10 && delta_rel < 0.10 && sign_kept {
                converged = true;
                converged_deg = Some(b.test_degree);
                converged_ty = Some(ty_b);
                converged_sch = Some(sch_b);
                converged_delta = Some(d_b);
            }
        }
        if !converged {
            notes.push(
                "No consecutive pair of degrees met the 10% convergence criterion."
                    .to_string(),
            );
        }
    }

    eprintln!("\n=== Convergence verdict ===");
    eprintln!("  converged = {}", converged);
    if let Some(d) = converged_deg {
        eprintln!(
            "  canonical test_degree = {}, TY={:?}, Schoen={:?}, delta={:?}",
            d, converged_ty, converged_sch, converged_delta
        );
    } else {
        eprintln!("  No converged value. Sweep results:");
        for r in &rows {
            eprintln!(
                "    deg={} basis_dim={} TY={:?} Schoen={:?} delta={:?}",
                r.test_degree, r.basis_dim, r.ty_residual, r.schoen_residual, r.delta
            );
        }
    }
    for n in &notes {
        eprintln!("  NOTE: {}", n);
    }

    let total_elapsed_s = t_total.elapsed().as_secs_f64();
    let report = SweepReport {
        label: "p6_3b_basis_convergence_sweep",
        chain_type: chain_label.to_string(),
        n_pts: cli.n_pts,
        k: cli.k,
        max_iter: cli.max_iter,
        seed: cli.seed,
        n_low: cli.n_low,
        sweep_time_cap_s: cli.sweep_time_cap_s,
        compute_path: Some(if cli.use_gpu { "gpu".to_string() } else { "cpu".to_string() }),
        rows,
        converged,
        converged_test_degree: converged_deg,
        converged_ty_residual: converged_ty,
        converged_schoen_residual: converged_sch,
        converged_delta,
        convergence_notes: notes,
        total_elapsed_s,
    };

    // P7.11 — when --output is set to a non-default path, route the
    // sweep JSON there (so callers can write `p7_11_quark_chain.json`
    // etc. without colliding on the legacy default).
    let default_out = PathBuf::from("output/p6_3_chain_match_diagnostic.json");
    let out_path = if cli.output != default_out {
        cli.output.clone()
    } else {
        PathBuf::from("output/p6_3b_basis_convergence_sweep.json")
    };
    if let Some(parent) = out_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let json = serde_json::to_string_pretty(&report).expect("serde_json::to_string_pretty");
    fs::write(&out_path, json).expect("write sweep output JSON");
    eprintln!("\nWrote {}", out_path.display());
    eprintln!("Total elapsed: {:.1}s", total_elapsed_s);
}

// ---------------------------------------------------------------------
// P7.2 — extended sweep with both chains and ratio-pattern.
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct P7_2Row {
    candidate: String,
    chain: String,
    test_degree: u32,
    basis_dim: usize,
    n_pos_eigvals: usize,
    galerkin_condition_log10: Option<f64>,
    smallest_eig: Option<f64>,
    largest_eig: Option<f64>,
    residual_log: Option<f64>,
    n_assigned: usize,
    max_individual_dev: f64,
    metric_iters: usize,
    metric_sigma: f64,
    metric_seconds: f64,
    spectrum_seconds: f64,
    skipped_reason: Option<String>,
    /// At test_degree==4, the top-N consecutive eigenvalue ratios and
    /// their closest-predicted match. Empty otherwise.
    ratio_pattern: Vec<RatioMatch>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct P7_2Verdict {
    candidate: String,
    chain: String,
    /// Whether the chain-match residual is stable across consecutive
    /// test_degrees (relative delta < 10%).
    residual_stable: bool,
    /// Sign of (Schoen - TY) at converged degree, if any.
    converged_test_degree: Option<u32>,
    converged_residual: Option<f64>,
    /// Per-pair stability notes.
    notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct P7_2Report {
    label: &'static str,
    n_pts: usize,
    k: u32,
    max_iter: usize,
    seed: u64,
    n_low: usize,
    sweep_time_cap_s: f64,
    /// P-REPRO-2 — compute path used for the Donaldson tree reduction
    /// ("cpu" | "gpu"). See `DiagnosticReport::compute_path`.
    #[serde(default)]
    compute_path: Option<String>,
    assign: String,
    rows: Vec<P7_2Row>,
    per_candidate_chain_verdicts: Vec<P7_2Verdict>,
    /// Per-(chain, test_degree) discrimination delta = Schoen - TY.
    /// Negative => Schoen wins.
    discrimination: Vec<P7_2Discrim>,
    overall_converged: bool,
    overall_headline: String,
    total_elapsed_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct P7_2Discrim {
    chain: String,
    test_degree: u32,
    ty_residual: Option<f64>,
    schoen_residual: Option<f64>,
    delta: Option<f64>,
    sign: String,
}

fn run_p7_2_sweep(cli: &Cli) {
    let degrees_src = cli
        .test_degree_list
        .as_ref()
        .map(|s| s.as_str())
        .unwrap_or(cli.sweep_degrees.as_str());
    let degrees = parse_sweep_degrees(degrees_src);
    if degrees.is_empty() {
        eprintln!("No valid sweep degrees parsed from '{}'", degrees_src);
        std::process::exit(1);
    }
    let algo = parse_assign(&cli.assign);
    eprintln!(
        "=== P7.2 chain-position eigenvalue clustering test ===\n  n_pts={}  k={}  max_iter={}  seed={}\n  n_low={}  degrees={:?}\n  assign={:?}  per-point time cap = {:.0}s",
        cli.n_pts, cli.k, cli.max_iter, cli.seed, cli.n_low, degrees, algo, cli.sweep_time_cap_s
    );

    let chains: [(&str, ChainType); 2] = [("quark", ChainType::Quark), ("lepton", ChainType::Lepton)];
    let candidates: [&str; 2] = ["TY/Z3", "Schoen/Z3xZ3"];

    let t_total = Instant::now();
    let mut rows: Vec<P7_2Row> = Vec::new();
    let adam_override = build_adam_override(cli);

    // For each candidate, solve the metric ONCE, then for each
    // test_degree solve the spectrum ONCE and run BOTH chains on the
    // shared spectrum. This avoids 4× redundant metric solves and
    // 2× redundant spectrum solves.
    for cand_label in &candidates {
        let spec_kind = if cand_label.starts_with("TY") {
            Cy3MetricSpec::TianYau {
                k: cli.k,
                n_sample: cli.n_pts,
                max_iter: cli.max_iter,
                donaldson_tol: cli.donaldson_tol,
                seed: cli.seed,
            }
        } else {
            Cy3MetricSpec::Schoen {
                d_x: cli.k,
                d_y: cli.k,
                d_t: 1.max(cli.k / 2),
                n_sample: cli.n_pts,
                max_iter: cli.max_iter,
                donaldson_tol: cli.donaldson_tol,
                seed: cli.seed,
            }
        };

        eprintln!("\n=== {} metric solve (once for all degrees) ===", cand_label);
        let t_metric = Instant::now();
        let solve_result = if cand_label.starts_with("TY") {
            TianYauSolver.solve_metric_with_adam(&spec_kind, &adam_override)
        } else {
            SchoenSolver.solve_metric_with_adam(&spec_kind, &adam_override)
        };
        let metric_seconds = t_metric.elapsed().as_secs_f64();

        let r = match solve_result {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  {} metric solve FAILED: {}", cand_label, e);
                for (chain_label, _) in &chains {
                    for &deg in &degrees {
                        rows.push(P7_2Row {
                            candidate: cand_label.to_string(),
                            chain: chain_label.to_string(),
                            test_degree: deg,
                            basis_dim: 0,
                            n_pos_eigvals: 0,
                            galerkin_condition_log10: None,
                            smallest_eig: None,
                            largest_eig: None,
                            residual_log: None,
                            n_assigned: 0,
                            max_individual_dev: f64::NAN,
                            metric_iters: 0,
                            metric_sigma: f64::NAN,
                            metric_seconds: 0.0,
                            spectrum_seconds: 0.0,
                            skipped_reason: Some(format!("metric solve failed: {}", e)),
                            ratio_pattern: Vec::new(),
                        });
                    }
                }
                continue;
            }
        };
        let summary = r.summary();
        let bg = match &r {
            Cy3MetricResultKind::TianYau(t) => Cy3MetricResultBackground::from_ty(t.as_ref()),
            Cy3MetricResultKind::Schoen(t) => Cy3MetricResultBackground::from_schoen(t.as_ref()),
        };
        eprintln!(
            "  metric: iters={} sigma={:.3e} t={:.1}s",
            summary.iterations_run, summary.final_sigma_residual, metric_seconds
        );

        let mut last_was_skipped = false;
        for &deg in &degrees {
            if last_was_skipped {
                eprintln!(
                    "  {} skipping degree {} (previous degree exceeded time cap)",
                    cand_label, deg
                );
                break;
            }
            let cfg = MetricLaplacianConfig {
                max_total_degree: deg,
                n_low_eigenvalues: cli.n_low,
                orthogonalize_first: cli.orthogonalize,
                ..MetricLaplacianConfig::default()
            };
            eprintln!(
                "  --- {} | spectrum solve at deg={} ---",
                cand_label, deg
            );
            let t_spec = Instant::now();
            let spectrum = if cli.orthogonalize {
                let projection = if cand_label.starts_with("TY") {
                    ProjectionKind::TianYauZ3
                } else {
                    ProjectionKind::SchoenZ3xZ3
                };
                compute_projected_metric_laplacian_spectrum(&bg, &cfg, projection).spectrum
            } else {
                compute_metric_laplacian_spectrum(&bg, &cfg)
            };
            let spectrum_seconds = t_spec.elapsed().as_secs_f64();
            eprintln!(
                "    basis_dim={} elapsed={:.1}s n_low_kept={}",
                spectrum.basis_dim, spectrum_seconds, spectrum.eigenvalues.len()
            );
            let cond = galerkin_condition(&spectrum);
            let (smallest, largest) = {
                let mut pos: Vec<f64> = spectrum
                    .eigenvalues_full
                    .iter()
                    .cloned()
                    .filter(|&v| v.is_finite() && v > 1.0e-15)
                    .collect();
                pos.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                (pos.first().copied(), pos.last().copied())
            };
            let n_pos = spectrum
                .eigenvalues_full
                .iter()
                .filter(|&&v| v.is_finite() && v > 1.0e-15)
                .count();
            let pat = if deg == 4 {
                ratio_pattern(&spectrum.eigenvalues, cli.ratio_floor, cli.ratio_top)
            } else {
                Vec::new()
            };
            let skip_reason = if spectrum_seconds > cli.sweep_time_cap_s {
                last_was_skipped = true;
                Some(format!(
                    "exceeded time cap ({:.1}s > {:.0}s) — skipping higher degrees",
                    spectrum_seconds, cli.sweep_time_cap_s
                ))
            } else {
                None
            };

            for (chain_label, chain_type) in &chains {
                let chain_match = match_chain_with(algo, &spectrum.eigenvalues, *chain_type);
                let r = chain_match.residual_log_f64;
                eprintln!(
                    "    chain={}: matched={} residual={:.4} max_dev={:.4}",
                    chain_label, chain_match.n_matched, r, chain_match.max_individual_dev_f64
                );
                rows.push(P7_2Row {
                    candidate: cand_label.to_string(),
                    chain: chain_label.to_string(),
                    test_degree: deg,
                    basis_dim: spectrum.basis_dim,
                    n_pos_eigvals: n_pos,
                    galerkin_condition_log10: cond.map(|v| v.log10()),
                    smallest_eig: smallest,
                    largest_eig: largest,
                    residual_log: if r.is_finite() { Some(r) } else { None },
                    n_assigned: chain_match.n_matched,
                    max_individual_dev: chain_match.max_individual_dev_f64,
                    metric_iters: summary.iterations_run,
                    metric_sigma: summary.final_sigma_residual,
                    metric_seconds,
                    spectrum_seconds,
                    skipped_reason: skip_reason.clone(),
                    ratio_pattern: pat.clone(),
                });
            }
        }
    }

    // Per-(candidate, chain) convergence verdict.
    let mut verdicts: Vec<P7_2Verdict> = Vec::new();
    for cand_label in &candidates {
        for (chain_label, _ct) in &chains {
            let mut series: Vec<&P7_2Row> = rows
                .iter()
                .filter(|r| r.candidate == *cand_label && &r.chain == chain_label && r.residual_log.is_some())
                .collect();
            series.sort_by_key(|r| r.test_degree);
            let mut notes: Vec<String> = Vec::new();
            let mut converged = false;
            let mut converged_deg: Option<u32> = None;
            let mut converged_res: Option<f64> = None;
            if series.len() < 2 {
                notes.push(format!("Only {} successful test_degree(s); cannot assess convergence.", series.len()));
            } else {
                for w in series.windows(2) {
                    let (a, b) = (w[0], w[1]);
                    let ra = a.residual_log.unwrap();
                    let rb = b.residual_log.unwrap();
                    let rel = (rb - ra).abs() / ra.abs().max(1e-12);
                    notes.push(format!(
                        "deg {}->{}: residual {:.4} -> {:.4} (rel-Δ={:.3})",
                        a.test_degree, b.test_degree, ra, rb, rel
                    ));
                    if rel < 0.10 {
                        converged = true;
                        converged_deg = Some(b.test_degree);
                        converged_res = Some(rb);
                    }
                }
            }
            verdicts.push(P7_2Verdict {
                candidate: cand_label.to_string(),
                chain: chain_label.to_string(),
                residual_stable: converged,
                converged_test_degree: converged_deg,
                converged_residual: converged_res,
                notes,
            });
        }
    }

    // Per-(chain, test_degree) discrimination delta.
    let mut discrim: Vec<P7_2Discrim> = Vec::new();
    for (chain_label, _ct) in &chains {
        for &deg in &degrees {
            let ty_r = rows
                .iter()
                .find(|r| r.candidate == "TY/Z3" && &r.chain == chain_label && r.test_degree == deg)
                .and_then(|r| r.residual_log);
            let sch_r = rows
                .iter()
                .find(|r| r.candidate == "Schoen/Z3xZ3" && &r.chain == chain_label && r.test_degree == deg)
                .and_then(|r| r.residual_log);
            let delta = match (ty_r, sch_r) {
                (Some(t), Some(s)) => Some(s - t),
                _ => None,
            };
            let sign = match delta {
                Some(d) if d < 0.0 => "Schoen wins".to_string(),
                Some(d) if d > 0.0 => "TY/Z3 wins".to_string(),
                Some(_) => "tie".to_string(),
                None => "n/a".to_string(),
            };
            discrim.push(P7_2Discrim {
                chain: chain_label.to_string(),
                test_degree: deg,
                ty_residual: ty_r,
                schoen_residual: sch_r,
                delta,
                sign,
            });
        }
    }

    // Overall verdict: converged-and-discriminating iff
    //   (a) signs of Δ agree across all basis sizes for a chain, AND
    //   (b) every per-candidate residual is "stable" (rel-Δ < 10%)
    //       between consecutive basis sizes.
    // Sign agreement alone is not enough — a residual that is moving
    // by 27% between bases is not yet converged even if it always
    // favours the same candidate.
    let mut overall_converged = true;
    let mut headline = String::new();
    for (chain_label, _ct) in &chains {
        let chain_disc: Vec<&P7_2Discrim> = discrim.iter().filter(|d| &d.chain == chain_label && d.delta.is_some()).collect();
        if chain_disc.is_empty() {
            overall_converged = false;
            headline += &format!("[{} chain: no usable data] ", chain_label);
            continue;
        }
        let signs: Vec<f64> = chain_disc.iter().map(|d| d.delta.unwrap().signum()).collect();
        let unanimous = signs.iter().all(|&s| s == signs[0]);
        let chain_verdicts: Vec<&P7_2Verdict> = verdicts.iter().filter(|v| &v.chain == chain_label).collect();
        let all_stable = chain_verdicts.iter().all(|v| v.residual_stable);
        if !(unanimous && all_stable) {
            overall_converged = false;
        }
        let last = chain_disc.last().unwrap();
        headline += &format!(
            "[{} chain @ deg={}: Δ = {} {:.3} ({}) sign-agree={} stable={}] ",
            chain_label,
            last.test_degree,
            if last.delta.unwrap() < 0.0 { "−" } else { "+" },
            last.delta.unwrap().abs(),
            last.sign,
            unanimous,
            all_stable,
        );
    }

    eprintln!("\n=== P7.2 verdict ===");
    eprintln!("  overall_converged = {}", overall_converged);
    eprintln!("  headline: {}", headline);
    for d in &discrim {
        eprintln!(
            "  {} chain @ deg={}: TY={:?} Schoen={:?} Δ={:?} ({})",
            d.chain, d.test_degree, d.ty_residual, d.schoen_residual, d.delta, d.sign
        );
    }
    for v in &verdicts {
        eprintln!(
            "  candidate={} chain={} stable={} converged_deg={:?} converged_res={:?}",
            v.candidate, v.chain, v.residual_stable, v.converged_test_degree, v.converged_residual
        );
        for n in &v.notes {
            eprintln!("    NOTE: {}", n);
        }
    }

    let report = P7_2Report {
        label: "p7_2_chain_sweep",
        n_pts: cli.n_pts,
        k: cli.k,
        max_iter: cli.max_iter,
        seed: cli.seed,
        n_low: cli.n_low,
        sweep_time_cap_s: cli.sweep_time_cap_s,
        compute_path: Some(if cli.use_gpu { "gpu".to_string() } else { "cpu".to_string() }),
        assign: format!("{:?}", algo),
        rows,
        per_candidate_chain_verdicts: verdicts,
        discrimination: discrim,
        overall_converged,
        overall_headline: headline,
        total_elapsed_s: t_total.elapsed().as_secs_f64(),
    };

    let out_path = PathBuf::from("output/p7_2_chain_sweep.json");
    if let Some(parent) = out_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let json = serde_json::to_string_pretty(&report).expect("serde_json::to_string_pretty");
    fs::write(&out_path, json).expect("write P7.2 sweep output JSON");
    eprintln!("\nWrote {}", out_path.display());
    eprintln!("Total elapsed: {:.1}s", report.total_elapsed_s);
}

fn main() {
    // P6.3b retraction banner — make the retraction visible to anyone
    // running the binary directly. Module-level `//!` docs and
    // `#[deprecated]` annotations are invisible at runtime.
    eprintln!("================================================");
    eprintln!("[RETRACTED] P6.3 chain-match result is retracted");
    eprintln!("            per P6.3b — see references/p6_3_chain_match.md");
    eprintln!("            Binary kept for diagnostic / re-runs only.");
    eprintln!("================================================");
    let cli = Cli::parse();
    let chain_type = parse_chain(&cli.chain);
    let chain_label = match chain_type {
        ChainType::Lepton => "lepton (phi)",
        ChainType::Quark => "quark (sqrt2)",
    };

    if cli.p7_2 {
        run_p7_2_sweep(&cli);
        return;
    }
    if cli.basis_sweep {
        run_sweep(&cli, chain_type, chain_label);
        return;
    }

    eprintln!(
        "=== P6.3 chain-match diagnostic ===\n  n_pts={}  k={}  max_iter={}  seed={}\n  test_degree={}  n_low={}  chain={}",
        cli.n_pts, cli.k, cli.max_iter, cli.seed, cli.test_degree, cli.n_low, chain_label
    );

    let t_total = Instant::now();
    let mut candidates: Vec<CandidateReport> = Vec::new();

    let ty_spec = Cy3MetricSpec::TianYau {
        k: cli.k,
        n_sample: cli.n_pts,
        max_iter: cli.max_iter,
        donaldson_tol: cli.donaldson_tol,
        seed: cli.seed,
    };
    let main_adam_override = build_adam_override(&cli);
    eprintln!("\n--- TY/Z3 ---");
    match run_candidate(
        "TY/Z3",
        ty_spec,
        &TianYauSolver,
        chain_type,
        cli.test_degree,
        cli.n_low,
        parse_assign(&cli.assign),
        cli.orthogonalize,
        &main_adam_override,
    ) {
        Ok(c) => {
            print_candidate(&c);
            candidates.push(c);
        }
        Err(e) => eprintln!("  SKIPPED: {e}"),
    }

    let schoen_spec = Cy3MetricSpec::Schoen {
        d_x: cli.k,
        d_y: cli.k,
        d_t: 1.max(cli.k / 2),
        n_sample: cli.n_pts,
        max_iter: cli.max_iter,
        donaldson_tol: cli.donaldson_tol,
        seed: cli.seed,
    };
    eprintln!("\n--- Schoen/Z3xZ3 ---");
    match run_candidate(
        "Schoen/Z3xZ3",
        schoen_spec,
        &SchoenSolver,
        chain_type,
        cli.test_degree,
        cli.n_low,
        parse_assign(&cli.assign),
        cli.orthogonalize,
        &main_adam_override,
    ) {
        Ok(c) => {
            print_candidate(&c);
            candidates.push(c);
        }
        Err(e) => eprintln!("  SKIPPED: {e}"),
    }

    let total_elapsed_s = t_total.elapsed().as_secs_f64();

    let discrimination_delta = if candidates.len() == 2 {
        let r0 = candidates[0].chain_match.residual_log_f64;
        let r1 = candidates[1].chain_match.residual_log_f64;
        if r0.is_finite() && r1.is_finite() {
            // candidates[0] = TY, candidates[1] = Schoen.
            Some(r1 - r0)
        } else {
            None
        }
    } else {
        None
    };

    eprintln!("\n=== Discrimination summary ===");
    if let Some(d) = discrimination_delta {
        eprintln!(
            "  delta = Schoen.residual - TY.residual = {:+.6}\n  ({} fits the {} chain better)",
            d,
            if d < 0.0 { "Schoen" } else { "TY/Z3" },
            chain_label
        );
    } else {
        eprintln!("  Insufficient data to discriminate.");
    }

    let report = DiagnosticReport {
        label: "p6_3_chain_match_diagnostic",
        chain_type: chain_label.to_string(),
        n_pts: cli.n_pts,
        k: cli.k,
        max_iter: cli.max_iter,
        seed: cli.seed,
        test_degree: cli.test_degree,
        n_low: cli.n_low,
        compute_path: Some(if cli.use_gpu { "gpu".to_string() } else { "cpu".to_string() }),
        candidates,
        discrimination_delta,
        total_elapsed_s,
    };

    if let Some(parent) = cli.output.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let json = serde_json::to_string_pretty(&report).expect("serde_json::to_string_pretty");
    fs::write(&cli.output, json).expect("write output JSON");
    eprintln!("\nWrote {}", cli.output.display());
    eprintln!("Total elapsed: {:.1}s", total_elapsed_s);
}

fn print_candidate(c: &CandidateReport) {
    eprintln!(
        "  metric: iters={} sigma={:.3e} t={:.1}s",
        c.metric_iters, c.metric_sigma, c.metric_seconds
    );
    eprintln!(
        "  spectrum: basis_dim={} t={:.2}s",
        c.spectrum.basis_dim, c.spectrum_seconds
    );
    let n_print = c.spectrum.eigenvalues.len().min(10);
    eprint!("  λ[0..{}] = ", n_print);
    for v in &c.spectrum.eigenvalues[..n_print] {
        eprint!("{:.4e} ", v);
    }
    eprintln!();
    eprintln!(
        "  chain-match: matched={}  residual_log={:.6}  max_dev={:.6}",
        c.chain_match.n_matched, c.chain_match.residual_log_f64, c.chain_match.max_individual_dev_f64
    );
    eprint!("  predicted exps = ");
    for k in &c.chain_match.predicted_exponents {
        eprint!("{} ", k);
    }
    eprintln!();
    eprint!("  assigned λ      = ");
    for v in &c.chain_match.assigned_eigvals {
        if v.is_finite() {
            eprint!("{:.4e} ", v);
        } else {
            eprint!("--- ");
        }
    }
    eprintln!();
}
