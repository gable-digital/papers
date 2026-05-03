// P7.3 — ω_fix gateway-eigenvalue diagnostic on the **bundle** Laplacian.
//
// Companion to P7.1 / P7.1b / P7.2b. P7.1b + P7.2b falsified the journal
// §L.2 prediction
//
//     ω_fix = 1/2 - 1/dim(E_8) = 123/248 = 0.495967741935...
//
// as the lowest non-zero eigenvalue of the **scalar metric Laplacian
// Δ_g** on the AKLP test-function basis (both with full character
// classification, P7.1, and on a Z/3- or Z/3×Z/3-projected basis,
// P7.2b).
//
// However, the journal language ("gateway mode", "fermion zero
// mode") is heterotic-string speak: in heterotic compactifications,
// the relevant operator is NOT the scalar metric Laplacian but the
// twisted-bundle Dolbeault Laplacian
//
//     Δ_∂̄^V := ∂̄_V ∂̄_V^* + ∂̄_V^* ∂̄_V    on    Ω^{0,1}(M, V)
//
// where V is the heterotic E_8 → E_6 × SU(3) bundle. P7.3 tests
// whether the lowest non-zero eigenvalue of THAT operator (suitably
// normalised) is 123/248. If yes, the journal's prediction was
// correctly stating a fact about the bundle Laplacian, and the
// metric-Laplacian falsification is a consequence of the wrong
// operator having been tested in P7.1/P7.1b/P7.2b.
//
// We reuse `route34::zero_modes_harmonic::solve_harmonic_zero_modes`
// — that builds the bundle-twisted Galerkin Laplacian
// L_{αβ} = ⟨D_V ψ_α, D_V ψ_β⟩ on the AKLP polynomial-seed ansatz, and
// `HarmonicZeroModeResult.eigenvalues_full` is the full ascending
// spectrum (NOT just the kernel). The BBW-9-mode kernel is at
// `eigenvalues_full[0..cohomology_dim_predicted]` (numerically near
// zero); the lowest non-zero eigenvalue is the next entry up.

use clap::Parser;
use cy3_rust_solver::route34::cy3_metric_unified::{
    Cy3MetricResultKind, Cy3MetricSolver, Cy3MetricSpec, SchoenSolver, TianYauSolver,
};
use cy3_rust_solver::route34::hym_hermitian::{
    solve_hym_metric, HymConfig, MetricBackground,
};
use cy3_rust_solver::route34::yukawa_pipeline::Cy3MetricResultBackground;
use cy3_rust_solver::route34::zero_modes_harmonic::{
    solve_harmonic_zero_modes, HarmonicConfig, HarmonicZeroModeResult,
};
use cy3_rust_solver::zero_modes::{AmbientCY3, MonadBundle};
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
#[command(about = "P7.3 ω_fix gateway-eigenvalue diagnostic on the bundle Laplacian Δ_∂̄^V")]
struct Cli {
    /// Sample-cloud size for Donaldson.
    #[arg(long, default_value_t = 40_000)]
    n_pts: usize,

    /// Bigraded section-basis degree (k for TY; d_x = d_y = k for Schoen).
    #[arg(long, default_value_t = 3)]
    k: u32,

    /// Donaldson iteration cap.
    #[arg(long, default_value_t = 100)]
    max_iter: usize,

    /// Donaldson tolerance.
    #[arg(long, default_value_t = 1.0e-6)]
    donaldson_tol: f64,

    /// PRNG seed. Default 12345 is one of the strict-converged Schoen seeds
    /// (residual ~6.5e-7, Donaldson iterations ~20).
    #[arg(long, default_value_t = 12345)]
    seed: u64,

    /// How many low non-zero eigenvalues to record per candidate.
    #[arg(long, default_value_t = 10)]
    n_record: usize,

    /// Numerical-zero threshold for kernel-mode classification.
    /// Eigenvalues with |λ| < kernel_zero_thresh * λ_max are treated as
    /// kernel modes (BBW = 9 expected).
    #[arg(long, default_value_t = 1.0e-3)]
    kernel_zero_thresh: f64,

    #[arg(long, default_value_t = false)]
    skip_schoen: bool,

    #[arg(long, default_value_t = false)]
    skip_ty: bool,

    #[arg(long, default_value = "output/p7_3_bundle_laplacian_omega_fix.json")]
    output: PathBuf,
}

// P-INFRA Fix 3 — `by_sigmoid` (λ / (λ + λ_max)) saturates to 0.5
// when λ ≪ λ_max, and 0.5 is fortuitously close to 123/248 = 0.4960.
// That created false ~0.81% matches across every cell of the P7.7
// sweep, regardless of whether the eigenvalue carried real ω_fix
// signal. The normalisation panel + picker now drop the sigmoid
// entry entirely; the JSON schema below no longer carries any
// sigmoid-related field.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Normalised {
    by_lambda1_min: f64,
    by_lambda_max: f64,
    by_mean_eigvalue: f64,
    by_trace: f64,
    by_volume: f64,
    /// λ_kernel_max sets the kernel-zero scale (the largest of the
    /// numerical-near-zero kernel eigenvalues). Useful as a baseline
    /// scale for "what counts as zero on this metric".
    by_kernel_max: f64,
    raw: f64,
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
    residual_ppm_by_kernel_max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClosestPick {
    rank: usize,
    eigenvalue: f64,
    chosen_scheme: String,
    chosen_value: f64,
    residual_ppm: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CandidateReport {
    label: String,
    geometry: String,
    bundle: String,
    metric_iters: usize,
    metric_sigma_residual: f64,
    metric_seconds: f64,
    spectrum_seconds: f64,
    seed_basis_dim: usize,
    n_points: usize,
    cohomology_dim_predicted: usize,
    /// Number of eigenvalues classified as kernel by the threshold
    /// rule (|λ| < thresh · λ_max).
    cohomology_dim_observed_threshold: usize,
    /// Kernel-count actually used to slice the spectrum (= threshold
    /// count when non-zero, else BBW prediction).
    kernel_count_used: usize,
    full_spectrum: Vec<f64>,
    kernel_eigenvalues: Vec<f64>,
    /// Lowest non-zero eigenvalue. `None` when the entire spectrum is
    /// classified as kernel (shouldn't happen on production sweeps).
    lambda_lowest_nonzero: Option<f64>,
    lambda_kernel_max: f64,
    lambda_max: f64,
    lambda_mean: f64,
    lambda_trace: f64,
    volume_proxy: f64,
    bottom_eigvalues: Vec<EigInfo>,
    closest_to_omega_fix: Option<ClosestPick>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiagnosticReport {
    label: &'static str,
    e8_dim: u32,
    predicted_omega_fix: f64,
    n_pts: usize,
    k: u32,
    max_iter: usize,
    seed: u64,
    kernel_zero_thresh: f64,
    candidates: Vec<CandidateReport>,
    verdict: String,
    notes: Vec<String>,
    total_elapsed_s: f64,
}

/// Compute the volume proxy `Σ_p w_p · |Ω(p)|²` exactly as in P7.2b.
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

#[allow(clippy::too_many_arguments)]
fn run_candidate(
    label: &str,
    geometry: &str,
    bundle_label: &str,
    spec: Cy3MetricSpec,
    solver: &dyn Cy3MetricSolver,
    bundle: &MonadBundle,
    ambient: &AmbientCY3,
    n_record: usize,
    kernel_zero_thresh: f64,
) -> Result<CandidateReport, String> {
    let t_metric = Instant::now();
    let r = solver
        .solve_metric(&spec)
        .map_err(|e| format!("{label}: metric solve failed: {e}"))?;
    let summary = r.summary();
    let metric_seconds = t_metric.elapsed().as_secs_f64();

    let bg = match &r {
        Cy3MetricResultKind::TianYau(t) => Cy3MetricResultBackground::from_ty(t.as_ref()),
        Cy3MetricResultKind::Schoen(t) => Cy3MetricResultBackground::from_schoen(t.as_ref()),
    };

    // Solve HYM with the same config used by the production Yukawa
    // pipeline (`bayes_discriminate.rs`). HYM provides the bundle
    // metric h_V that enters L_{αβ} = ⟨D_V ψ_α, D_V ψ_β⟩.
    let hym_cfg = HymConfig {
        max_iter: 8,
        damping: 0.5,
        ..HymConfig::default()
    };
    let h_v = solve_hym_metric(bundle, &bg, &hym_cfg);

    // Bundle-twisted Galerkin Laplacian. We use HarmonicConfig::default()
    // — the kernel-selection mechanism does not affect `eigenvalues_full`
    // (the full ascending spectrum is always populated). We classify
    // kernel/non-kernel ourselves by the |λ| < thresh · λ_max test.
    let cfg = HarmonicConfig::default();
    let t_spec = Instant::now();
    let res: HarmonicZeroModeResult =
        solve_harmonic_zero_modes(bundle, ambient, &bg, &h_v, &cfg);
    let spectrum_seconds = t_spec.elapsed().as_secs_f64();

    let evals = &res.eigenvalues_full;
    if evals.is_empty() {
        return Err(format!(
            "{label}: bundle Laplacian returned empty spectrum (seed_basis_dim = {})",
            res.seed_basis_dim
        ));
    }

    let lambda_max = evals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lambda_mean = evals.iter().sum::<f64>() / (evals.len() as f64);
    let lambda_trace: f64 = evals.iter().sum();
    let vol = volume_proxy(&bg);

    // Kernel classification. Two policies, applied in priority order:
    //
    //   1. THRESHOLD: |λ| < kernel_zero_thresh · λ_max counts as kernel.
    //      Works on FS-Gram-identity-style spectra where the BBW kernel
    //      truly sits at numerical zero.
    //   2. BBW-INDEX FALLBACK: if (1) finds *no* kernel modes (the
    //      Donaldson-balanced regime where the lowest eigenvalues sit
    //      at the Bergman-kernel numerical residual, NOT at zero — see
    //      the policy-priority comment in
    //      `zero_modes_harmonic::solve_harmonic_zero_modes`), fall
    //      back to skipping the lowest `cohomology_dim_predicted`
    //      eigenvalues.
    //
    // `evals` is ascending (`hermitian_jacobi_n` returns ascending
    // order), so the kernel is a contiguous prefix in either case.
    let cutoff = lambda_max * kernel_zero_thresh;
    let kernel_count_threshold = evals.iter().take_while(|&&v| v.abs() <= cutoff).count();
    let kernel_count = if kernel_count_threshold == 0 {
        // Donaldson regime — strip the BBW prefix instead.
        res.cohomology_dim_predicted.min(evals.len())
    } else {
        kernel_count_threshold
    };
    let kernel_eigenvalues: Vec<f64> = evals[..kernel_count].to_vec();
    let lambda_kernel_max = kernel_eigenvalues
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let lambda_kernel_max = if lambda_kernel_max.is_finite() {
        lambda_kernel_max
    } else {
        // No kernel detected — fall back to a very small positive
        // number so divisions don't NaN.
        1.0e-30
    };
    let nonzero: Vec<f64> = evals[kernel_count..].to_vec();
    let lambda_lowest_nonzero = nonzero.first().copied();
    let lambda_min_nonzero = lambda_lowest_nonzero.unwrap_or(f64::NAN);

    let make_norm = |raw: f64| -> Normalised {
        let by_lambda1_min = if lambda_min_nonzero.is_finite() && lambda_min_nonzero > 0.0 {
            raw / lambda_min_nonzero
        } else {
            f64::NAN
        };
        let by_lambda_max = if lambda_max.is_finite() && lambda_max > 0.0 {
            raw / lambda_max
        } else {
            f64::NAN
        };
        let by_mean_eigvalue = if lambda_mean.is_finite() && lambda_mean > 0.0 {
            raw / lambda_mean
        } else {
            f64::NAN
        };
        let by_trace = if lambda_trace.is_finite() && lambda_trace > 0.0 {
            raw / lambda_trace
        } else {
            f64::NAN
        };
        let by_volume = if vol.is_finite() && vol > 0.0 {
            raw / vol
        } else {
            f64::NAN
        };
        let by_kernel_max = if lambda_kernel_max.is_finite() && lambda_kernel_max > 0.0 {
            raw / lambda_kernel_max
        } else {
            f64::NAN
        };
        Normalised {
            by_lambda1_min,
            by_lambda_max,
            by_mean_eigvalue,
            by_trace,
            by_volume,
            by_kernel_max,
            raw,
        }
    };

    let omega_fix = predicted_omega_fix();

    let bottom: Vec<EigInfo> = nonzero
        .iter()
        .take(n_record)
        .enumerate()
        .map(|(rank, &ev)| {
            let n = make_norm(ev);
            EigInfo {
                rank,
                eigenvalue: ev,
                residual_ppm_raw: ppm(n.raw, omega_fix),
                residual_ppm_by_lambda1_min: ppm(n.by_lambda1_min, omega_fix),
                residual_ppm_by_lambda_max: ppm(n.by_lambda_max, omega_fix),
                residual_ppm_by_mean: ppm(n.by_mean_eigvalue, omega_fix),
                residual_ppm_by_trace: ppm(n.by_trace, omega_fix),
                residual_ppm_by_volume: ppm(n.by_volume, omega_fix),
                residual_ppm_by_kernel_max: ppm(n.by_kernel_max, omega_fix),
                normalised: n,
            }
        })
        .collect();

    let closest_to_omega_fix = if bottom.is_empty() {
        None
    } else {
        // P-INFRA Fix 3 — `by_sigmoid` deliberately omitted; see the
        // type-level note on `Normalised`.
        let schemes: [(&'static str, fn(&Normalised) -> f64); 7] = [
            ("raw", |n| n.raw),
            ("by_lambda1_min", |n| n.by_lambda1_min),
            ("by_lambda_max", |n| n.by_lambda_max),
            ("by_mean_eigvalue", |n| n.by_mean_eigvalue),
            ("by_trace", |n| n.by_trace),
            ("by_volume", |n| n.by_volume),
            ("by_kernel_max", |n| n.by_kernel_max),
        ];
        let mut best: Option<(f64, &EigInfo, &'static str, f64)> = None;
        for e in &bottom {
            for (name, getter) in &schemes {
                let v = getter(&e.normalised);
                if !v.is_finite() {
                    continue;
                }
                let res = (v - omega_fix).abs() / omega_fix.abs().max(1.0e-300);
                match &best {
                    None => best = Some((res, e, name, v)),
                    Some((b_res, _, _, _)) if res < *b_res => best = Some((res, e, name, v)),
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
        })
    };

    Ok(CandidateReport {
        label: label.to_string(),
        geometry: geometry.to_string(),
        bundle: bundle_label.to_string(),
        metric_iters: summary.iterations_run,
        metric_sigma_residual: summary.final_sigma_residual,
        metric_seconds,
        spectrum_seconds,
        seed_basis_dim: res.seed_basis_dim,
        n_points: bg.n_points(),
        cohomology_dim_predicted: res.cohomology_dim_predicted,
        cohomology_dim_observed_threshold: kernel_count_threshold,
        kernel_count_used: kernel_count,
        full_spectrum: evals.clone(),
        kernel_eigenvalues,
        lambda_lowest_nonzero,
        lambda_kernel_max,
        lambda_max,
        lambda_mean,
        lambda_trace,
        volume_proxy: vol,
        bottom_eigvalues: bottom,
        closest_to_omega_fix,
    })
}

fn print_candidate(c: &CandidateReport) {
    let omega_fix = predicted_omega_fix();
    eprintln!("\n=== {} ===", c.label);
    eprintln!(
        "  geometry: {}, bundle: {}",
        c.geometry, c.bundle
    );
    eprintln!(
        "  metric: iters={}  σ-residual={:.3e}  t={:.1}s",
        c.metric_iters, c.metric_sigma_residual, c.metric_seconds
    );
    eprintln!(
        "  bundle Laplacian: seed_basis_dim={}  n_pts={}  t={:.2}s",
        c.seed_basis_dim, c.n_points, c.spectrum_seconds
    );
    eprintln!(
        "  cohomology dim: predicted (BBW)={}, observed (|λ|<thresh·λ_max)={}, kernel_count_used={}",
        c.cohomology_dim_predicted, c.cohomology_dim_observed_threshold, c.kernel_count_used
    );
    eprintln!(
        "  λ_kernel_max={:.6e}   λ_lowest_nonzero={}   λ_max={:.6e}   λ_mean={:.6e}",
        c.lambda_kernel_max,
        match c.lambda_lowest_nonzero {
            Some(v) => format!("{:.6e}", v),
            None => "<none>".to_string(),
        },
        c.lambda_max,
        c.lambda_mean
    );
    eprintln!(
        "  λ_trace={:.6e}  vol_proxy={:.6e}",
        c.lambda_trace, c.volume_proxy
    );
    let n_print = c.full_spectrum.len().min(20);
    eprint!("  full spectrum[0..{}]: ", n_print);
    for v in &c.full_spectrum[..n_print] {
        eprint!("{:.3e} ", v);
    }
    eprintln!();
    eprintln!(
        "  bottom-{} non-zero eigenvalues vs ω_fix = {:.12} = 123/248:",
        c.bottom_eigvalues.len(),
        omega_fix
    );
    for e in &c.bottom_eigvalues {
        eprintln!(
            "    [{}] λ={:.6e}  raw_ppm={:8.1}  /λ1={:.6} ({:8.1} ppm)  /λmax={:.6} ({:8.1} ppm)  /vol={:.6} ({:8.1} ppm)  /kernel={:.6} ({:8.1} ppm)",
            e.rank,
            e.eigenvalue,
            e.residual_ppm_raw,
            e.normalised.by_lambda1_min,
            e.residual_ppm_by_lambda1_min,
            e.normalised.by_lambda_max,
            e.residual_ppm_by_lambda_max,
            e.normalised.by_volume,
            e.residual_ppm_by_volume,
            e.normalised.by_kernel_max,
            e.residual_ppm_by_kernel_max,
        );
    }
    if let Some(pick) = &c.closest_to_omega_fix {
        eprintln!(
            "  closest-to-ω_fix:  rank={} λ={:.6e} scheme={} value={:.6} residual={:.3} ppm",
            pick.rank, pick.eigenvalue, pick.chosen_scheme, pick.chosen_value, pick.residual_ppm
        );
    }
}

fn build_verdict(candidates: &[CandidateReport]) -> String {
    if candidates.is_empty() {
        return "NO RESULT — both candidates skipped".to_string();
    }
    let mut s = String::new();
    for c in candidates {
        match &c.closest_to_omega_fix {
            Some(pick) => {
                let tier = if pick.residual_ppm <= 1.0 {
                    "VERIFIED at <=1 ppm (6+ digits)"
                } else if pick.residual_ppm <= 1.0e2 {
                    "VERIFIED at <=100 ppm (4-digit match)"
                } else if pick.residual_ppm <= 1.0e4 {
                    "MARGINAL (<=1% but >100 ppm)"
                } else {
                    "FAILED (>1% off ω_fix)"
                };
                s.push_str(&format!(
                    "{}: rank={} λ={:.6e} via {} -> {:.6} ({:.3} ppm = {:.4}% off ω_fix) — {}\n",
                    c.label,
                    pick.rank,
                    pick.eigenvalue,
                    pick.chosen_scheme,
                    pick.chosen_value,
                    pick.residual_ppm,
                    100.0 * pick.residual_ppm / 1.0e6,
                    tier
                ));
            }
            None => {
                s.push_str(&format!(
                    "{}: NO non-zero eigenvalues survived kernel filter (kernel saturated the spectrum)\n",
                    c.label
                ));
            }
        }
    }
    s
}

fn main() {
    let cli = Cli::parse();
    let omega_fix = predicted_omega_fix();
    eprintln!("================================================");
    eprintln!("P7.3 — ω_fix gateway-eigenvalue diagnostic on Δ_∂̄^V (bundle Laplacian)");
    eprintln!("  predicted ω_fix = 1/2 - 1/dim(E_8) = 123/248");
    eprintln!("                  = {:.18}", omega_fix);
    eprintln!("================================================");
    eprintln!(
        "  n_pts={}  k={}  max_iter={}  seed={}  kernel_zero_thresh={:.0e}",
        cli.n_pts, cli.k, cli.max_iter, cli.seed, cli.kernel_zero_thresh
    );

    // Bundle and ambient pair: AKLP monad on the upstairs ambient.
    // For Schoen, we use schoen_z3xz3_upstairs (the canonical Schoen
    // upstairs ambient); for TY we use tian_yau_upstairs. The bundle
    // (AKLP) is the same in both candidates — this is the standard
    // E_8 → E_6 × SU(3) construction, NOT the more specific Z_3×Z_3
    // Wilson-line breaking that the journal §F.1.5/§F.1.6 prescribes
    // (see the limitation note in the verdict).
    let bundle = MonadBundle::anderson_lukas_palti_example();

    let t_total = Instant::now();
    let mut candidates: Vec<CandidateReport> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    notes.push(
        "Bundle Laplacian = ⟨D_V ψ_α, D_V ψ_β⟩ on AKLP polynomial-seed basis;".to_string(),
    );
    notes.push(
        "h_V is the HYM-balanced bundle metric (route34::hym_hermitian).".to_string(),
    );
    notes.push(
        "Bundle is the canonical AKLP / E_8 → E_6 × SU(3) example, NOT the".to_string(),
    );
    notes.push(
        "Z_3×Z_3 Wilson-line-respecting bundle the journal §F.1.5 prescribes.".to_string(),
    );
    notes.push(
        "If this test does not verify ω_fix, the limitation may be due to bundle choice;".to_string(),
    );
    notes.push(
        "see references/p7_3_bundle_laplacian_omega_fix.md for the discussion.".to_string(),
    );

    if !cli.skip_schoen {
        let schoen_spec = Cy3MetricSpec::Schoen {
            d_x: cli.k,
            d_y: cli.k,
            d_t: 1,
            n_sample: cli.n_pts,
            max_iter: cli.max_iter,
            donaldson_tol: cli.donaldson_tol,
            seed: cli.seed,
        };
        let ambient = AmbientCY3::schoen_z3xz3_upstairs();
        eprintln!("\n--- running Schoen / AKLP bundle ---");
        match run_candidate(
            "Schoen/AKLP",
            "schoen_z3xz3",
            "anderson_lukas_palti_example",
            schoen_spec,
            &SchoenSolver,
            &bundle,
            &ambient,
            cli.n_record,
            cli.kernel_zero_thresh,
        ) {
            Ok(c) => {
                print_candidate(&c);
                candidates.push(c);
            }
            Err(e) => eprintln!("  SKIPPED: {e}"),
        }
    }

    if !cli.skip_ty {
        let ty_spec = Cy3MetricSpec::TianYau {
            k: cli.k,
            n_sample: cli.n_pts,
            max_iter: cli.max_iter,
            donaldson_tol: cli.donaldson_tol,
            seed: cli.seed,
        };
        let ambient = AmbientCY3::tian_yau_upstairs();
        eprintln!("\n--- running TY / AKLP bundle (control) ---");
        match run_candidate(
            "TY/AKLP",
            "tian_yau_z3",
            "anderson_lukas_palti_example",
            ty_spec,
            &TianYauSolver,
            &bundle,
            &ambient,
            cli.n_record,
            cli.kernel_zero_thresh,
        ) {
            Ok(c) => {
                print_candidate(&c);
                candidates.push(c);
            }
            Err(e) => eprintln!("  SKIPPED: {e}"),
        }
    }

    let total_elapsed_s = t_total.elapsed().as_secs_f64();
    let verdict = build_verdict(&candidates);
    eprintln!("\n================================================");
    eprintln!("VERDICT:");
    eprintln!("{}", verdict);
    eprintln!("================================================");

    let report = DiagnosticReport {
        label: "p7_3_bundle_laplacian_omega_fix",
        e8_dim: E8_DIM,
        predicted_omega_fix: omega_fix,
        n_pts: cli.n_pts,
        k: cli.k,
        max_iter: cli.max_iter,
        seed: cli.seed,
        kernel_zero_thresh: cli.kernel_zero_thresh,
        candidates,
        verdict,
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
