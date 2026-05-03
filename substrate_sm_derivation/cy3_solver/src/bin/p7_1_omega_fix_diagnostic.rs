// P7.1 — ω_fix gateway-electron diagnostic.
//
// Tests the journal-§L.2 prediction
//
//     ω_fix = 1/2 − 1/dim(E_8) = 123/248 = 0.495967741935...
//
// against the Schoen Z₃×Z₃ Donaldson-balanced metric Laplacian.
// Strategy (b): decompose the metric Laplacian's eigenfunctions into
// Z₃×Z₃ irreps via the test-basis monomials' (α, β)-characters, then
// extract the lowest trivial-representation eigenvalue.
//
// We retain the deprecated `compute_metric_laplacian_spectrum` API for
// the Galerkin solve (the deprecation is over the chain-match consumer,
// not the eigenvalue solver itself).
#![allow(deprecated)]

use clap::Parser;
use cy3_rust_solver::route34::cy3_metric_unified::{
    Cy3MetricResultKind, Cy3MetricSolver, Cy3MetricSpec, SchoenSolver, TianYauSolver,
};
use cy3_rust_solver::route34::metric_laplacian::{
    compute_metric_laplacian_spectrum, MetricLaplacianConfig,
};
use cy3_rust_solver::route34::yukawa_pipeline::Cy3MetricResultBackground;
use cy3_rust_solver::route34::z3xz3_projector::{alpha_character, beta_character};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Predicted gateway-electron eigenvalue: `1/2 − 1/dim(E_8) = 123/248`.
const E8_DIM: u32 = 248;
fn predicted_omega_fix() -> f64 {
    0.5 - 1.0 / (E8_DIM as f64)
}

#[derive(Parser, Debug)]
#[command(about = "P7.1 ω_fix gateway-electron eigenvalue diagnostic")]
struct Cli {
    /// Sample-cloud size for the Donaldson metric solve.
    /// Default 40_000 matches P5.10's converged-seed configuration.
    #[arg(long, default_value_t = 40_000)]
    n_pts: usize,

    /// Bigraded section-basis degree for Donaldson (k for TY, d_x=d_y=k
    /// for Schoen with d_t=1).
    #[arg(long, default_value_t = 3)]
    k: u32,

    /// Donaldson iteration cap. P5.5k uses 100.
    #[arg(long, default_value_t = 100)]
    max_iter: usize,

    /// Donaldson convergence tolerance.
    #[arg(long, default_value_t = 1.0e-6)]
    donaldson_tol: f64,

    /// Single-seed PRNG seed. seed=42 is one of the 16 Schoen seeds
    /// that converge at n_pts=40k under the P5.5k iter_cap=100 setup.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Maximum total degree of the test-function basis (P6.3b: 4).
    #[arg(long, default_value_t = 4)]
    test_degree: u32,

    /// Number of low trivial-rep eigenvalues to record per candidate.
    #[arg(long, default_value_t = 5)]
    n_trivial_record: usize,

    /// Skip Schoen (TY-only run).
    #[arg(long, default_value_t = false)]
    skip_schoen: bool,

    /// Skip TY (Schoen-only run).
    #[arg(long, default_value_t = false)]
    skip_ty: bool,

    /// Output JSON path.
    #[arg(long, default_value = "output/p7_1_omega_fix_diagnostic.json")]
    output: PathBuf,
}

/// Bundle of normalisations we report. The journal does not specify
/// one explicitly; we report several so the closest match is
/// transparent.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Normalised {
    /// `λ / λ_min_overall_nonzero` — ratio to the lowest non-zero
    /// eigenvalue of the full spectrum (all reps).
    by_lambda1_overall: f64,
    /// `λ / λ_min_trivial_nonzero` — ratio to the lowest non-zero
    /// trivial-rep eigenvalue.
    by_lambda1_trivial: f64,
    /// `λ / max_full_eigvalue` — top-of-spectrum scale.
    by_lambda_max: f64,
    /// `λ / mean_full_eigvalue` — mean-eigenvalue scale.
    by_mean_eigvalue: f64,
    /// Raw eigenvalue (no normalisation).
    raw: f64,
}

fn ppm(observed: f64, predicted: f64) -> f64 {
    1.0e6 * (observed - predicted).abs() / predicted.abs().max(1.0e-300)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EigInfo {
    /// Index in the |.|-sorted full spectrum.
    full_index: usize,
    /// Raw eigenvalue.
    eigenvalue: f64,
    /// L²-fraction of |coefficient|² supported on trivial-rep
    /// basis monomials (= 1.0 if the spectrum cleanly block-diagonalises
    /// over the symmetry, < 1.0 if there is character mixing from
    /// Donaldson noise / numerical roundoff).
    trivial_rep_purity: f64,
    /// Dominant character (α, β) by `Σ |c_i|²` over basis monomials.
    dominant_character: (u32, u32),
    /// `Σ |c_i|²` carried by each (α, β) character class, length-9.
    /// Order: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2).
    character_weights: [f64; 9],
    /// All five normalisations.
    normalised: Normalised,
    /// `|normalised.by_X − ω_fix| / ω_fix` in ppm for each scheme.
    residual_ppm_raw: f64,
    residual_ppm_by_lambda1_overall: f64,
    residual_ppm_by_lambda1_trivial: f64,
    residual_ppm_by_lambda_max: f64,
    residual_ppm_by_mean_eigvalue: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CandidateReport {
    label: String,
    /// Per-symmetry character function in use: "z3xz3" or "z3" or
    /// "z3xz3_on_ty" (Schoen's Z₃×Z₃ formulas applied to TY's basis,
    /// which is informative as a control).
    symmetry: String,
    metric_iters: usize,
    metric_sigma_residual: f64,
    metric_seconds: f64,
    spectrum_seconds: f64,
    basis_dim: usize,
    n_points: usize,
    /// Bottom of the full (all-reps) eigenvalue spectrum.
    lambda_min_full_nonzero: f64,
    /// Bottom of the trivial-rep eigenvalue spectrum.
    lambda_min_trivial_nonzero: Option<f64>,
    lambda_max_full: f64,
    lambda_mean_full: f64,
    /// Lowest-`n` trivial-rep eigenfunctions.
    trivial_eigvalues: Vec<EigInfo>,
    /// Lowest-`n` overall eigenfunctions for context.
    bottom_eigvalues: Vec<EigInfo>,
    /// Closest-to-ω_fix eigenfunction across all reps under each scheme.
    closest_to_omega_fix_overall: ClosestPick,
    /// Closest-to-ω_fix in the trivial rep (the predicted location).
    closest_to_omega_fix_trivial: Option<ClosestPick>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClosestPick {
    full_index: usize,
    eigenvalue: f64,
    /// Scheme that achieved this pick (e.g. "by_lambda1_trivial").
    chosen_scheme: String,
    /// Normalised value under the winning scheme.
    chosen_value: f64,
    residual_ppm: f64,
    dominant_character: (u32, u32),
    trivial_rep_purity: f64,
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
    test_degree: u32,
    candidates: Vec<CandidateReport>,
    /// Verdict in plain English.
    verdict: String,
    total_elapsed_s: f64,
    /// Reproducibility manifest collected at run start: rust toolchain,
    /// target triple, CPU SIMD features, hostname, RFC 3339 UTC timestamp,
    /// command line, rayon thread count. See `route34::repro::ReproManifest`.
    #[serde(default)]
    repro_manifest: Option<cy3_rust_solver::route34::repro::ReproManifest>,
}

/// Sum-and-classify a full-basis eigenvector by (α, β)-character
/// using a per-monomial `character` lookup. Returns
/// `(weights[9], dominant, trivial_purity)`.
fn classify_eigenvector(
    evec_real_imag: &[(f64, f64)],
    n_b: usize,
    col: usize,
    chi: &[(u32, u32)],
) -> ([f64; 9], (u32, u32), f64) {
    let mut weights = [0.0_f64; 9];
    let mut total = 0.0_f64;
    for i in 0..n_b {
        let (re, im) = evec_real_imag[i * n_b + col];
        let mag2 = re * re + im * im;
        if !mag2.is_finite() || mag2 == 0.0 {
            continue;
        }
        let (a, b) = chi[i];
        let bucket = (a * 3 + b) as usize;
        weights[bucket] += mag2;
        total += mag2;
    }
    if total > 0.0 {
        for w in weights.iter_mut() {
            *w /= total;
        }
    }
    let mut best_b = 0usize;
    let mut best_w = weights[0];
    for k in 1..9 {
        if weights[k] > best_w {
            best_w = weights[k];
            best_b = k;
        }
    }
    let dom = ((best_b / 3) as u32, (best_b % 3) as u32);
    (weights, dom, weights[0])
}

fn pick_closest(
    triv: &[EigInfo],
    overall: &[EigInfo],
    target: f64,
    use_trivial: bool,
) -> Option<ClosestPick> {
    let pool = if use_trivial { triv } else { overall };
    if pool.is_empty() {
        return None;
    }
    let mut best: Option<(f64, &EigInfo, &'static str, f64)> = None;
    let schemes: [(&'static str, fn(&Normalised) -> f64); 5] = [
        ("raw", |n| n.raw),
        ("by_lambda1_trivial", |n| n.by_lambda1_trivial),
        ("by_lambda1_overall", |n| n.by_lambda1_overall),
        ("by_lambda_max", |n| n.by_lambda_max),
        ("by_mean_eigvalue", |n| n.by_mean_eigvalue),
    ];
    for e in pool {
        for (name, getter) in &schemes {
            let v = getter(&e.normalised);
            if !v.is_finite() {
                continue;
            }
            let res = (v - target).abs() / target.abs().max(1.0e-300);
            match &best {
                None => best = Some((res, e, name, v)),
                Some((b_res, _, _, _)) if res < *b_res => best = Some((res, e, name, v)),
                _ => {}
            }
        }
    }
    let (res, e, scheme, value) = best?;
    Some(ClosestPick {
        full_index: e.full_index,
        eigenvalue: e.eigenvalue,
        chosen_scheme: scheme.to_string(),
        chosen_value: value,
        residual_ppm: 1.0e6 * res,
        dominant_character: e.dominant_character,
        trivial_rep_purity: e.trivial_rep_purity,
    })
}

fn run_candidate(
    label: &str,
    spec: Cy3MetricSpec,
    solver: &dyn Cy3MetricSolver,
    test_degree: u32,
    n_trivial_record: usize,
    use_z3xz3_chars: bool,
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

    let cfg = MetricLaplacianConfig {
        max_total_degree: test_degree,
        n_low_eigenvalues: 50,
        return_eigenvectors: true,
        ..MetricLaplacianConfig::default()
    };
    let t_spec = Instant::now();
    let spectrum = compute_metric_laplacian_spectrum(&bg, &cfg);
    let spectrum_seconds = t_spec.elapsed().as_secs_f64();

    let basis = spectrum
        .basis_exponents
        .as_ref()
        .ok_or_else(|| format!("{label}: basis_exponents missing — return_eigenvectors=true should populate it"))?;
    let evecs = spectrum
        .eigenvectors_full
        .as_ref()
        .ok_or_else(|| format!("{label}: eigenvectors_full missing"))?;

    let n_b = spectrum.basis_dim;
    if n_b == 0 {
        return Err(format!("{label}: empty basis after Galerkin assembly"));
    }
    if spectrum.eigenvalues_full.len() != n_b {
        return Err(format!(
            "{label}: eigenvalue/basis length mismatch ({} vs {})",
            spectrum.eigenvalues_full.len(),
            n_b
        ));
    }
    if evecs.len() != n_b * n_b {
        return Err(format!(
            "{label}: eigenvector matrix size mismatch ({} vs {}*{})",
            evecs.len(),
            n_b,
            n_b
        ));
    }

    // Per-basis-monomial (α, β)-character lookup.
    // For Schoen we use the canonical Z₃×Z₃ formulas. For TY we use
    // the Z/3 character formula (encoded as α-only, with β=0 for all
    // monomials so the trivial rep matches the Z/3 invariant subspace).
    let chi: Vec<(u32, u32)> = if use_z3xz3_chars {
        basis
            .iter()
            .map(|m| (alpha_character(m), beta_character(m)))
            .collect()
    } else {
        // TY/Z3 character: (m[1] + 2 m[2] + m[5] + 2 m[6]) mod 3
        // (note the index shift: positions 0..4 are z, 4..8 are w, so
        // z₁ z₂ → m[1] m[2], w₁ w₂ → m[5] m[6]). β is identically 0
        // (no second Z/3 factor).
        basis
            .iter()
            .map(|m| {
                let alpha = (m[1] + 2 * m[2] + m[5] + 2 * m[6]) % 3;
                (alpha, 0u32)
            })
            .collect()
    };

    let symmetry = if use_z3xz3_chars { "z3xz3" } else { "z3_ty" };

    // Build per-eigenvalue info.
    let evals = &spectrum.eigenvalues_full;
    let nonzero: Vec<f64> = evals.iter().cloned().filter(|&v| v > 1.0e-10).collect();
    let lambda_min_full_nonzero = nonzero.first().copied().unwrap_or(f64::NAN);
    let lambda_max_full = evals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lambda_mean_full = if !evals.is_empty() {
        evals.iter().sum::<f64>() / (evals.len() as f64)
    } else {
        f64::NAN
    };

    // First pass: classify every eigenvector, compute trivial-rep
    // purity and dominant character.
    let mut classified: Vec<(usize, f64, [f64; 9], (u32, u32), f64)> = (0..n_b)
        .map(|j| {
            let (w, dom, triv) = classify_eigenvector(evecs, n_b, j, &chi);
            (j, evals[j], w, dom, triv)
        })
        .collect();

    // Trivial-rep eigenvalues: dominant character (0,0) AND raw eigvalue
    // > tiny floor (filter constant-mode-like noise).
    // The basis already excludes the constant function (P6.3b), so
    // λ_0 won't be exactly zero, but it can be very small; keep the
    // floor at 1e-10.
    let trivial_filter = |&(_, ev, _, dom, _): &(usize, f64, [f64; 9], (u32, u32), f64)| -> bool {
        dom == (0, 0) && ev > 1.0e-10
    };
    let trivial_pool: Vec<&(usize, f64, [f64; 9], (u32, u32), f64)> =
        classified.iter().filter(|c| trivial_filter(c)).collect();
    let lambda_min_trivial_nonzero = trivial_pool.first().map(|c| c.1);

    let make_norm = |raw: f64| -> Normalised {
        Normalised {
            by_lambda1_overall: if lambda_min_full_nonzero.is_finite()
                && lambda_min_full_nonzero > 0.0
            {
                raw / lambda_min_full_nonzero
            } else {
                f64::NAN
            },
            by_lambda1_trivial: match lambda_min_trivial_nonzero {
                Some(v) if v > 0.0 => raw / v,
                _ => f64::NAN,
            },
            by_lambda_max: if lambda_max_full.is_finite() && lambda_max_full > 0.0 {
                raw / lambda_max_full
            } else {
                f64::NAN
            },
            by_mean_eigvalue: if lambda_mean_full.is_finite() && lambda_mean_full > 0.0 {
                raw / lambda_mean_full
            } else {
                f64::NAN
            },
            raw,
        }
    };

    let omega_fix = predicted_omega_fix();

    let to_eiginfo = |c: &(usize, f64, [f64; 9], (u32, u32), f64)| -> EigInfo {
        let (j, ev, weights, dom, triv) = c;
        let n = make_norm(*ev);
        EigInfo {
            full_index: *j,
            eigenvalue: *ev,
            trivial_rep_purity: *triv,
            dominant_character: *dom,
            character_weights: *weights,
            residual_ppm_raw: ppm(n.raw, omega_fix),
            residual_ppm_by_lambda1_overall: ppm(n.by_lambda1_overall, omega_fix),
            residual_ppm_by_lambda1_trivial: ppm(n.by_lambda1_trivial, omega_fix),
            residual_ppm_by_lambda_max: ppm(n.by_lambda_max, omega_fix),
            residual_ppm_by_mean_eigvalue: ppm(n.by_mean_eigvalue, omega_fix),
            normalised: n,
        }
    };

    // Sort classified by eigenvalue ascending (already mostly sorted by
    // |.|-sort but be defensive).
    classified.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let trivial_eigvalues: Vec<EigInfo> = classified
        .iter()
        .filter(|c| trivial_filter(*c))
        .take(n_trivial_record)
        .map(to_eiginfo)
        .collect();
    let bottom_eigvalues: Vec<EigInfo> = classified
        .iter()
        .filter(|c| c.1 > 1.0e-10)
        .take(n_trivial_record)
        .map(to_eiginfo)
        .collect();

    let closest_to_omega_fix_overall = pick_closest(
        &trivial_eigvalues,
        &bottom_eigvalues,
        omega_fix,
        false,
    )
    .ok_or_else(|| format!("{label}: failed to pick closest eigvalue"))?;
    let closest_to_omega_fix_trivial = pick_closest(
        &trivial_eigvalues,
        &bottom_eigvalues,
        omega_fix,
        true,
    );

    Ok(CandidateReport {
        label: label.to_string(),
        symmetry: symmetry.to_string(),
        metric_iters: summary.iterations_run,
        metric_sigma_residual: summary.final_sigma_residual,
        metric_seconds,
        spectrum_seconds,
        basis_dim: n_b,
        n_points: spectrum.n_points,
        lambda_min_full_nonzero,
        lambda_min_trivial_nonzero,
        lambda_max_full,
        lambda_mean_full,
        trivial_eigvalues,
        bottom_eigvalues,
        closest_to_omega_fix_overall,
        closest_to_omega_fix_trivial,
    })
}

fn print_candidate(c: &CandidateReport) {
    let omega_fix = predicted_omega_fix();
    eprintln!("\n=== {} ===", c.label);
    eprintln!("  symmetry: {}", c.symmetry);
    eprintln!(
        "  metric: iters={}  σ-residual={:.3e}  t={:.1}s",
        c.metric_iters, c.metric_sigma_residual, c.metric_seconds
    );
    eprintln!(
        "  spectrum: basis_dim={}  n_pts={}  t={:.2}s",
        c.basis_dim, c.n_points, c.spectrum_seconds
    );
    eprintln!(
        "  λ_min_full = {:.6e}  λ_max_full = {:.6e}  λ_mean_full = {:.6e}",
        c.lambda_min_full_nonzero, c.lambda_max_full, c.lambda_mean_full
    );
    if let Some(v) = c.lambda_min_trivial_nonzero {
        eprintln!("  λ_min_trivial-rep = {:.6e}", v);
    } else {
        eprintln!("  λ_min_trivial-rep = (none — no trivial-rep modes found)");
    }
    eprintln!("  trivial-rep eigenvalues (lowest {}):", c.trivial_eigvalues.len());
    for (i, e) in c.trivial_eigvalues.iter().enumerate() {
        eprintln!(
            "    [{}] λ={:.6e}  norm/(λ_min_triv)={:.6}  norm/(λ_min_overall)={:.6}  triv-purity={:.4}  dom-χ={:?}",
            i,
            e.eigenvalue,
            e.normalised.by_lambda1_trivial,
            e.normalised.by_lambda1_overall,
            e.trivial_rep_purity,
            e.dominant_character
        );
    }
    eprintln!("  bottom (all-rep) eigenvalues (lowest {}):", c.bottom_eigvalues.len());
    for (i, e) in c.bottom_eigvalues.iter().enumerate() {
        eprintln!(
            "    [{}] λ={:.6e}  norm/(λ_min_overall)={:.6}  triv-purity={:.4}  dom-χ={:?}",
            i,
            e.eigenvalue,
            e.normalised.by_lambda1_overall,
            e.trivial_rep_purity,
            e.dominant_character
        );
    }
    eprintln!(
        "  closest-to-ω_fix (trivial pool):  {:?}",
        c.closest_to_omega_fix_trivial
    );
    eprintln!(
        "  closest-to-ω_fix (all-rep pool):  scheme={} value={:.6}  residual={:.3} ppm  dom-χ={:?}  triv-purity={:.4}",
        c.closest_to_omega_fix_overall.chosen_scheme,
        c.closest_to_omega_fix_overall.chosen_value,
        c.closest_to_omega_fix_overall.residual_ppm,
        c.closest_to_omega_fix_overall.dominant_character,
        c.closest_to_omega_fix_overall.trivial_rep_purity
    );
    eprintln!("  predicted ω_fix = {:.12} (= 123/248)", omega_fix);
}

fn main() {
    let cli = Cli::parse();
    let omega_fix = predicted_omega_fix();
    eprintln!("================================================");
    eprintln!("P7.1 — ω_fix gateway-electron diagnostic");
    eprintln!("  predicted ω_fix = 1/2 − 1/dim(E_8) = 123/248");
    eprintln!("                  = {:.18}", omega_fix);
    eprintln!("================================================");
    eprintln!(
        "  n_pts={}  k={}  max_iter={}  seed={}  test_degree={}",
        cli.n_pts, cli.k, cli.max_iter, cli.seed, cli.test_degree
    );

    let t_total = Instant::now();
    let mut candidates: Vec<CandidateReport> = Vec::new();

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
        eprintln!("\n--- running Schoen/Z₃×Z₃ ---");
        match run_candidate(
            "Schoen/Z3xZ3",
            schoen_spec,
            &SchoenSolver,
            cli.test_degree,
            cli.n_trivial_record,
            true, // use Z₃×Z₃ characters
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
        eprintln!("\n--- running TY/Z₃ (control) ---");
        match run_candidate(
            "TY/Z3",
            ty_spec,
            &TianYauSolver,
            cli.test_degree,
            cli.n_trivial_record,
            false, // use TY's single Z/3 character
        ) {
            Ok(c) => {
                print_candidate(&c);
                candidates.push(c);
            }
            Err(e) => eprintln!("  SKIPPED: {e}"),
        }
    }

    let total_elapsed_s = t_total.elapsed().as_secs_f64();

    // Construct verdict.
    let verdict = build_verdict(&candidates, omega_fix);
    eprintln!("\n================================================");
    eprintln!("VERDICT: {}", verdict);
    eprintln!("================================================");

    let report = DiagnosticReport {
        label: "p7_1_omega_fix_diagnostic",
        e8_dim: E8_DIM,
        predicted_omega_fix: omega_fix,
        n_pts: cli.n_pts,
        k: cli.k,
        max_iter: cli.max_iter,
        seed: cli.seed,
        test_degree: cli.test_degree,
        candidates,
        verdict,
        total_elapsed_s,
        repro_manifest: Some(cy3_rust_solver::route34::repro::ReproManifest::collect()),
    };

    if let Some(parent) = cli.output.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let json = serde_json::to_string_pretty(&report).expect("serialise diagnostic report");
    fs::write(&cli.output, json).expect("write diagnostic JSON");
    eprintln!("\nWrote {}", cli.output.display());
    eprintln!("Total elapsed: {:.1}s", total_elapsed_s);
}

fn build_verdict(candidates: &[CandidateReport], omega_fix: f64) -> String {
    if candidates.is_empty() {
        return "NO RESULT — both candidates skipped".to_string();
    }
    let schoen = candidates.iter().find(|c| c.label == "Schoen/Z3xZ3");
    let ty = candidates.iter().find(|c| c.label == "TY/Z3");

    let summary = |c: &CandidateReport| -> String {
        let pick = c
            .closest_to_omega_fix_trivial
            .as_ref()
            .unwrap_or(&c.closest_to_omega_fix_overall);
        format!(
            "{}: closest-trivial-rep value = {:.6} via {} → residual = {:.3} ppm  ({:.5}% of ω_fix)",
            c.label,
            pick.chosen_value,
            pick.chosen_scheme,
            pick.residual_ppm,
            100.0 * pick.residual_ppm / 1.0e6
        )
    };

    let mut s = String::new();
    if let Some(c) = schoen {
        s.push_str(&summary(c));
        s.push('\n');
    }
    if let Some(c) = ty {
        s.push_str(&summary(c));
        s.push('\n');
    }
    // Pass/fail relative to the journal's "high precision" claim
    // (4+ digit match → ≤ 100 ppm relative residual).
    let one_pct = 1.0e4_f64; // ppm
    let four_digit = 1.0e2_f64; // ppm
    let test = |c: &CandidateReport| -> (f64, &'static str) {
        let pick = c
            .closest_to_omega_fix_trivial
            .as_ref()
            .unwrap_or(&c.closest_to_omega_fix_overall);
        let r = pick.residual_ppm;
        let tier = if r <= four_digit {
            "VERIFIED (≤100 ppm = 4-digit match)"
        } else if r <= one_pct {
            "MARGINAL (≤1% but >100 ppm)"
        } else {
            "FAILED (>1% off ω_fix)"
        };
        (r, tier)
    };
    if let Some(c) = schoen {
        let (_r, tier) = test(c);
        s.push_str(&format!("Schoen tier: {}\n", tier));
    }
    if let Some(c) = ty {
        let (_r, tier) = test(c);
        s.push_str(&format!("TY tier:     {} (control — should NOT match)\n", tier));
    }
    let _ = omega_fix; // referenced for context
    s
}
