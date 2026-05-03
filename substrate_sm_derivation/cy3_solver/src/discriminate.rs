//! End-to-end discrimination binary.
//!
//! Builds a `Vec<Candidate>` for the substrate-physics CY3
//! candidates listed in chapter 8 (Tian-Yau Z/3, Schoen Z/3 × Z/3)
//! and runs them through
//! [`cy3_rust_solver::pipeline::sweep_candidates`], emitting a JSON
//! ranking report on stdout.
//!
//! Earlier revisions of this binary called the toy
//! `sample_points` / `donaldson_solve` / `yukawa_tensor` polysphere
//! kernels and labelled both runs as `"TY/Z3"` / `"Schoen/Z3xZ3"`
//! without any actual geometry switch — the two runs were
//! arithmetically identical. This rewrite replaces that with the
//! real `compute_5sigma_score_for_candidate` pipeline, which uses
//! the candidate's `CicyGeometry` field to drive the Tian-Yau
//! line-intersection sampler. The Schoen path is wired through
//! `sweep_candidates` so the candidate appears in the report with
//! the `geometry not currently sampler-supported` error message
//! until the route34 dispatch (task #33) is plumbed.
//!
//! Output schema (one JSON object on stdout):
//!
//! ```json
//! {
//!   "binary": "discriminate",
//!   "config": { "n_sample_points": 200, "sampler_seed": 42, ... },
//!   "candidates": [
//!     {
//!       "rank": 1,
//!       "candidate_id": 1,
//!       "candidate_short_name": "tian-yau-z3-default",
//!       "geometry_label": "Tian-Yau Z/3",
//!       "fundamental_group": "Z3",
//!       "elapsed_seconds": 4.83,
//!       "total_loss": 1.23e25,
//!       "passes_5_sigma": false,
//!       "breakdown": { "n_samples_accepted": 200, "n_27_generations": 9, ... },
//!       "error": null
//!     },
//!     {
//!       "rank": 2,
//!       "candidate_id": 2,
//!       "geometry_label": "Schoen Z/3 × Z/3 fiber-product",
//!       "total_loss": Infinity,
//!       "error": "compute_5sigma_score: only Tian-Yau Z/3 ... is currently sampler-supported"
//!     }
//!   ]
//! }
//! ```
//!
//! Run:
//!
//! ```text
//!   cargo run --release --bin discriminate
//!   cargo run --release --bin discriminate --features gpu     # GPU paths
//! ```

extern crate cy3_rust_solver;

use clap::{Parser, ValueEnum};
use cy3_rust_solver::geometry::CicyGeometry;
use cy3_rust_solver::pipeline::{
    format_ranking_report_markdown, sweep_candidates, Candidate, CandidateRanking,
    FiveSigmaConfig,
};
use serde::Serialize;

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum OutputFormat {
    /// Structured JSON only (single object on stdout).
    Json,
    /// Markdown table only (suitable for chapter / PR / notebook).
    Markdown,
    /// Both — markdown table on stdout first, then a `---` divider,
    /// then the JSON object.
    Both,
}

#[derive(Parser, Debug)]
#[command(name = "discriminate", about = "End-to-end CY3 candidate discrimination")]
struct Cli {
    /// Output format on stdout. The eprintln summary always prints
    /// regardless of this flag.
    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    format: OutputFormat,
    /// CY3 sample-point count for the Tian-Yau line-intersection
    /// sampler. Higher = tighter Donaldson convergence + slower
    /// per-candidate evaluation.
    #[arg(long, default_value_t = 500)]
    n_sample_points: usize,
    /// PRNG seed for sampling reproducibility.
    #[arg(long, default_value_t = 42)]
    sampler_seed: u64,
    /// Compute the Route 3 (η-integral) χ² contribution for each
    /// candidate. Adds 30s–10min per candidate. Default off; the
    /// standalone `eta_discriminate` binary is preferred for tighter
    /// uncertainty bands.
    #[arg(long, default_value_t = false)]
    compute_eta: bool,
}

/// Build the chapter-8 candidate list: Tian-Yau Z/3 and Schoen
/// Z/3 × Z/3 fiber product. Geometry comes from
/// [`CicyGeometry::tian_yau_z3`] and [`CicyGeometry::schoen_z3xz3`]
/// respectively (both first-principles topological data; no free
/// parameters). Moduli vectors are left empty since
/// `compute_5sigma_score_for_candidate` doesn't currently consume
/// them on the Tian-Yau path (the Donaldson balancer derives its
/// own metric from the geometry's intersection form).
fn build_candidate_list() -> Vec<Candidate> {
    vec![
        Candidate {
            id: 1,
            candidate_short_name: "tian-yau-z3".to_string(),
            euler_characteristic: -6,
            fundamental_group: "Z3".to_string(),
            kahler_moduli: Vec::new(),
            complex_moduli_real: Vec::new(),
            complex_moduli_imag: Vec::new(),
            bundle_moduli: Vec::new(),
            parent_id: None,
            geometry: CicyGeometry::tian_yau_z3(),
        },
        Candidate {
            id: 2,
            candidate_short_name: "schoen-z3xz3".to_string(),
            euler_characteristic: 0,
            fundamental_group: "Z3xZ3".to_string(),
            kahler_moduli: Vec::new(),
            complex_moduli_real: Vec::new(),
            complex_moduli_imag: Vec::new(),
            bundle_moduli: Vec::new(),
            parent_id: None,
            geometry: CicyGeometry::schoen_z3xz3(),
        },
    ]
}

#[derive(Debug, Serialize)]
struct ConfigOut {
    n_sample_points: usize,
    sampler_seed: u64,
    mu_init_gev: f64,
}

#[derive(Debug, Serialize)]
struct RankedCandidate<'a> {
    rank: usize,
    #[serde(flatten)]
    inner: &'a CandidateRanking,
}

#[derive(Debug, Serialize)]
struct DiscriminationReport<'a> {
    binary: &'static str,
    config: ConfigOut,
    candidates: Vec<RankedCandidate<'a>>,
}

fn main() {
    let cli = Cli::parse();
    let cfg = FiveSigmaConfig {
        n_sample_points: cli.n_sample_points,
        sampler_seed: cli.sampler_seed,
        mu_init_gev: 1.0e16,
        compute_eta_chi2: cli.compute_eta,
    };

    let candidates = build_candidate_list();

    eprintln!(
        "discriminate: scoring {} candidates ({})...",
        candidates.len(),
        if cfg!(feature = "gpu") {
            "gpu feature ON; CUDA dispatch will be attempted with CPU fallback"
        } else {
            "CPU-only build"
        }
    );

    let rankings = sweep_candidates(&candidates, &cfg);

    // Human-readable summary to stderr (one line per candidate).
    eprintln!();
    eprintln!("Ranking (lowest total loss first):");
    for (i, r) in rankings.iter().enumerate() {
        eprintln!(
            "{}",
            cy3_rust_solver::pipeline::ranking_summary_line(r, i + 1)
        );
    }
    eprintln!();

    // Stdout output: format-dependent.
    let ranked: Vec<RankedCandidate> = rankings
        .iter()
        .enumerate()
        .map(|(i, r)| RankedCandidate {
            rank: i + 1,
            inner: r,
        })
        .collect();
    let report = DiscriminationReport {
        binary: "discriminate",
        config: ConfigOut {
            n_sample_points: cfg.n_sample_points,
            sampler_seed: cfg.sampler_seed,
            mu_init_gev: cfg.mu_init_gev,
        },
        candidates: ranked,
    };

    let emit_markdown = || {
        print!("{}", format_ranking_report_markdown(&rankings));
    };
    let emit_json = || -> std::process::ExitCode {
        match serde_json::to_string_pretty(&report) {
            Ok(s) => {
                println!("{s}");
                std::process::ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: failed to serialize ranking report: {e}");
                std::process::ExitCode::FAILURE
            }
        }
    };
    match cli.format {
        OutputFormat::Json => {
            std::process::exit(if emit_json() == std::process::ExitCode::SUCCESS { 0 } else { 1 });
        }
        OutputFormat::Markdown => {
            emit_markdown();
        }
        OutputFormat::Both => {
            emit_markdown();
            println!();
            println!("---");
            println!();
            std::process::exit(if emit_json() == std::process::ExitCode::SUCCESS { 0 } else { 1 });
        }
    }
}
