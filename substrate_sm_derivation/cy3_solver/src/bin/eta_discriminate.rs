//! Standalone η-discrimination binary.
//!
//! Runs the η evaluator on both Tian-Yau Z/3 and Schoen Z/3×Z/3
//! candidates, computes χ² against the chapter-21 observed value
//! `η_obs = (6.115 ± 0.038) × 10⁻¹⁰`, emits a JSON discrimination
//! report, a Markdown summary, and a reproducibility log.
//!
//! Usage:
//!
//! ```text
//!   cargo run --release --bin eta_discriminate -- \
//!     --n-metric-samples 5000 --n-integrand-samples 50000 \
//!     --output-dir ./eta_discrimination_$(date +%Y%m%d_%H%M%S)
//! ```

use clap::Parser;
use cy3_rust_solver::route34::eta_evaluator::{
    evaluate_eta_schoen, evaluate_eta_tian_yau, EtaEvaluatorConfig, EtaResult,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

/// Observed η value from chapter-21 substrate-physics analysis. Quoted
/// to one part in `~10^4`; the uncertainty is the propagated error from
/// the Saturn-hexagon / Jupiter-pole input data.
const ETA_OBSERVED_CENTRAL: f64 = 6.115e-10;
const ETA_OBSERVED_SIGMA: f64 = 0.038e-10;

#[derive(Parser, Debug, Clone)]
#[command(
    author,
    version,
    about = "Run Tian-Yau and Schoen η evaluators and emit a chi-squared discrimination report."
)]
struct Cli {
    /// Number of CY3 sample points used for the metric balance.
    #[arg(long, default_value_t = 1000)]
    n_metric_samples: usize,

    /// Number of integrand-stage samples (drives the MC uncertainty).
    #[arg(long, default_value_t = 5000)]
    n_integrand_samples: usize,

    /// Number of Donaldson balancing iterations.
    #[arg(long, default_value_t = 20)]
    n_metric_iters: usize,

    /// PRNG seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Output directory (will be created).
    #[arg(long)]
    output_dir: PathBuf,

    /// Maximum wall-clock seconds per candidate.
    #[arg(long, default_value_t = 1800)]
    max_wallclock_seconds: u64,

    /// Comma-separated list of candidate labels to run (default: both).
    /// Valid: `tian_yau`, `schoen`.
    #[arg(long, default_value = "tian_yau,schoen")]
    candidates: String,

    /// Optional checkpoint base path (per-candidate suffix appended).
    #[arg(long)]
    checkpoint_path_base: Option<PathBuf>,

    /// Tian-Yau Kähler moduli (2 entries, comma-separated).
    #[arg(long, default_value = "1.0,1.0")]
    kahler_ty: String,

    /// Schoen Kähler moduli (3 entries, comma-separated).
    #[arg(long, default_value = "1.0,1.0,1.0")]
    kahler_schoen: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CandidateBlock {
    label: String,
    eta_predicted: f64,
    eta_uncertainty: f64,
    numerator_value: f64,
    numerator_uncertainty: f64,
    denominator_value: f64,
    denominator_uncertainty: f64,
    donaldson_residual: f64,
    chi_squared: f64,
    sigma_distance: f64,
    run_metadata: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct DiscriminationReport {
    schema_version: u32,
    timestamp_utc: String,
    git_sha: String,
    cli_args: serde_json::Value,
    eta_observed_central: f64,
    eta_observed_sigma: f64,
    candidates: Vec<CandidateBlock>,
    winner: Option<String>,
    notes: String,
}

fn parse_kahler(s: &str) -> Vec<f64> {
    s.split(',')
        .filter(|t| !t.trim().is_empty())
        .filter_map(|t| t.trim().parse().ok())
        .collect()
}

fn git_sha() -> String {
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output();
    match out {
        Ok(o) if o.status.success() => {
            String::from_utf8_lossy(&o.stdout).trim().to_string()
        }
        _ => "unknown".into(),
    }
}

fn host_info() -> String {
    let host = std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "unknown".into());
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    format!("host={host}, os={os}, arch={arch}")
}

fn timestamp_utc() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("unix_seconds={secs}")
}

/// SHA-256 hex of a JSON-serialised value (used to fingerprint
/// intermediate results for the reproducibility log).
fn sha256_json(value: &serde_json::Value) -> String {
    let mut h = Sha256::new();
    h.update(value.to_string().as_bytes());
    hex::encode(h.finalize())
}

fn build_candidate_block(label: &str, res: &EtaResult) -> CandidateBlock {
    let combined_sigma = (res.eta_uncertainty.powi(2) + ETA_OBSERVED_SIGMA.powi(2)).sqrt();
    let chi2 = if combined_sigma > 0.0 {
        ((res.eta_predicted - ETA_OBSERVED_CENTRAL) / combined_sigma).powi(2)
    } else {
        f64::INFINITY
    };
    let sigma_dist = chi2.sqrt();
    CandidateBlock {
        label: label.to_string(),
        eta_predicted: res.eta_predicted,
        eta_uncertainty: res.eta_uncertainty,
        numerator_value: res.numerator_value,
        numerator_uncertainty: res.numerator_uncertainty,
        denominator_value: res.denominator_value,
        denominator_uncertainty: res.denominator_uncertainty,
        donaldson_residual: res.donaldson_residual,
        chi_squared: chi2,
        sigma_distance: sigma_dist,
        run_metadata: serde_json::to_value(&res.run_metadata)
            .unwrap_or(serde_json::Value::Null),
    }
}

fn winner(blocks: &[CandidateBlock]) -> Option<String> {
    if blocks.len() < 2 {
        return None;
    }
    // Smaller χ² wins, but we only declare a winner if its σ-distance
    // is < 3 and the loser's σ-distance is > 3 (roughly: one within
    // 3σ, the other clearly excluded).
    let mut sorted = blocks.iter().collect::<Vec<_>>();
    sorted.sort_by(|a, b| {
        a.chi_squared
            .partial_cmp(&b.chi_squared)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let best = sorted[0];
    let runner_up = sorted[1];
    if best.sigma_distance < 3.0 && runner_up.sigma_distance > 3.0 {
        Some(best.label.clone())
    } else {
        None
    }
}

fn write_markdown(path: &PathBuf, report: &DiscriminationReport) -> std::io::Result<()> {
    let mut s = String::new();
    s.push_str("# Route-3 η-discrimination report\n\n");
    s.push_str(&format!("- **Timestamp**: {}\n", report.timestamp_utc));
    s.push_str(&format!("- **Git SHA**: `{}`\n", report.git_sha));
    s.push_str(&format!(
        "- **Observed η**: ({:.4e} ± {:.4e})\n",
        report.eta_observed_central, report.eta_observed_sigma
    ));
    s.push_str(&format!(
        "- **Winner**: {}\n\n",
        report
            .winner
            .clone()
            .unwrap_or_else(|| "inconclusive".into())
    ));

    s.push_str("## CLI parameters\n\n```json\n");
    s.push_str(
        &serde_json::to_string_pretty(&report.cli_args).unwrap_or_default(),
    );
    s.push_str("\n```\n\n");

    s.push_str("## Per-candidate results\n\n");
    s.push_str(
        "| Candidate | η_pred | σ_η | numerator | denominator | Donaldson resid | χ² | σ-distance |\n",
    );
    s.push_str(
        "|-----------|--------|-----|-----------|-------------|-----------------|----|-----------|\n",
    );
    for c in &report.candidates {
        s.push_str(&format!(
            "| {} | {:.6e} | {:.3e} | {:.6e} | {:.6e} | {:.3e} | {:.3} | {:.3} |\n",
            c.label,
            c.eta_predicted,
            c.eta_uncertainty,
            c.numerator_value,
            c.denominator_value,
            c.donaldson_residual,
            c.chi_squared,
            c.sigma_distance,
        ));
    }
    s.push('\n');

    s.push_str("## Per-candidate run metadata\n\n");
    for c in &report.candidates {
        s.push_str(&format!("### {}\n\n```json\n", c.label));
        s.push_str(
            &serde_json::to_string_pretty(&c.run_metadata).unwrap_or_default(),
        );
        s.push_str("\n```\n\n");
    }

    s.push_str("## Citations\n\n");
    s.push_str(
        "- Tian-Yau visible bundle: Anderson, Gray, Lukas, Palti, *Two hundred heterotic standard models on smooth Calabi-Yau threefolds*, arXiv:1106.4804.\n",
    );
    s.push_str("- Schoen visible bundle: Donagi, He, Ovrut, Reinbacher, *The particle spectrum of heterotic compactifications*, JHEP 06 (2006) 039 (arXiv:hep-th/0512149).\n");
    s.push_str(
        "- Donaldson balancing: Donaldson, *Some numerical results in complex differential geometry*, PAMQ 5 (2009) 571.\n",
    );
    s.push_str(
        "- η integrand form: book chapter 21, `08-choosing-a-substrate.adoc` lines 233-263.\n",
    );
    s.push_str(
        "- Schoen geometry: Schoen, *On fiber products of rational elliptic surfaces with section*, Math. Z. 197 (1988) 177.\n",
    );

    s.push_str(&format!("\n## Notes\n\n{}\n", report.notes));
    fs::write(path, s)
}

fn write_repro_log(
    path: &PathBuf,
    cli: &Cli,
    git_sha: &str,
    blocks: &[CandidateBlock],
) -> std::io::Result<()> {
    let mut f = fs::File::create(path)?;
    writeln!(f, "# Reproducibility log")?;
    writeln!(f, "git_sha={}", git_sha)?;
    writeln!(f, "host={}", host_info())?;
    writeln!(f, "{}", timestamp_utc())?;
    writeln!(f, "cli_args={}", serde_json::to_string(cli).unwrap_or_default())?;
    for c in blocks {
        let h_meta = sha256_json(&c.run_metadata);
        writeln!(
            f,
            "candidate={} eta={:.16e} sigma={:.16e} numerator={:.16e} denominator={:.16e} donaldson_residual={:.16e} chi_squared={:.16e} run_metadata_sha256={}",
            c.label,
            c.eta_predicted,
            c.eta_uncertainty,
            c.numerator_value,
            c.denominator_value,
            c.donaldson_residual,
            c.chi_squared,
            h_meta,
        )?;
    }
    Ok(())
}

// `Cli` does not derive `Serialize`. We only need to serialise the
// argv-style record into the report; do it manually.
impl serde::Serialize for Cli {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;
        let mut m = serializer.serialize_map(Some(10))?;
        m.serialize_entry("n_metric_samples", &self.n_metric_samples)?;
        m.serialize_entry("n_integrand_samples", &self.n_integrand_samples)?;
        m.serialize_entry("n_metric_iters", &self.n_metric_iters)?;
        m.serialize_entry("seed", &self.seed)?;
        m.serialize_entry("output_dir", &self.output_dir)?;
        m.serialize_entry("max_wallclock_seconds", &self.max_wallclock_seconds)?;
        m.serialize_entry("candidates", &self.candidates)?;
        m.serialize_entry("checkpoint_path_base", &self.checkpoint_path_base)?;
        m.serialize_entry("kahler_ty", &self.kahler_ty)?;
        m.serialize_entry("kahler_schoen", &self.kahler_schoen)?;
        m.end()
    }
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    fs::create_dir_all(&cli.output_dir)?;

    let kahler_ty = parse_kahler(&cli.kahler_ty);
    let kahler_schoen = parse_kahler(&cli.kahler_schoen);
    if kahler_ty.len() != 2 {
        return Err(format!(
            "--kahler-ty must have exactly 2 entries; got {}",
            kahler_ty.len()
        )
        .into());
    }
    if kahler_schoen.len() != 3 {
        return Err(format!(
            "--kahler-schoen must have exactly 3 entries; got {}",
            kahler_schoen.len()
        )
        .into());
    }

    let candidates: Vec<&str> = cli.candidates.split(',').map(|s| s.trim()).collect();

    let mut blocks: Vec<CandidateBlock> = Vec::new();
    let mut notes = String::new();

    if candidates.contains(&"tian_yau") {
        let cfg = EtaEvaluatorConfig {
            n_metric_iters: cli.n_metric_iters,
            n_metric_samples: cli.n_metric_samples,
            n_integrand_samples: cli.n_integrand_samples,
            kahler_moduli: kahler_ty.clone(),
            seed: cli.seed,
            checkpoint_path: cli
                .checkpoint_path_base
                .as_ref()
                .map(|p| p.with_extension("ty.ckpt")),
            max_wallclock_seconds: cli.max_wallclock_seconds,
        };
        eprintln!("[eta_discriminate] Running Tian-Yau Z/3 evaluator…");
        match evaluate_eta_tian_yau(&cfg) {
            Ok(r) => {
                eprintln!(
                    "[eta_discriminate] TY η = {:.6e} ± {:.3e}, donaldson_residual = {:.3e}",
                    r.eta_predicted, r.eta_uncertainty, r.donaldson_residual
                );
                blocks.push(build_candidate_block("TY/Z3", &r));
            }
            Err(e) => {
                notes.push_str(&format!("Tian-Yau evaluation failed: {e}\n"));
                eprintln!("[eta_discriminate] TY failed: {e}");
            }
        }
    }

    if candidates.contains(&"schoen") {
        let cfg = EtaEvaluatorConfig {
            n_metric_iters: cli.n_metric_iters,
            n_metric_samples: cli.n_metric_samples,
            n_integrand_samples: cli.n_integrand_samples,
            kahler_moduli: kahler_schoen.clone(),
            seed: cli.seed,
            checkpoint_path: cli
                .checkpoint_path_base
                .as_ref()
                .map(|p| p.with_extension("schoen.ckpt")),
            max_wallclock_seconds: cli.max_wallclock_seconds,
        };
        eprintln!("[eta_discriminate] Running Schoen Z/3xZ/3 evaluator…");
        match evaluate_eta_schoen(&cfg) {
            Ok(r) => {
                eprintln!(
                    "[eta_discriminate] Schoen η = {:.6e} ± {:.3e}, donaldson_residual = {:.3e}",
                    r.eta_predicted, r.eta_uncertainty, r.donaldson_residual
                );
                blocks.push(build_candidate_block("Schoen/Z3xZ3", &r));
            }
            Err(e) => {
                notes.push_str(&format!("Schoen evaluation failed: {e}\n"));
                eprintln!("[eta_discriminate] Schoen failed: {e}");
            }
        }
    }

    let winning = winner(&blocks);

    let report = DiscriminationReport {
        schema_version: 1,
        timestamp_utc: timestamp_utc(),
        git_sha: git_sha(),
        cli_args: serde_json::to_value(&cli).unwrap_or(serde_json::Value::Null),
        eta_observed_central: ETA_OBSERVED_CENTRAL,
        eta_observed_sigma: ETA_OBSERVED_SIGMA,
        candidates: blocks.clone(),
        winner: winning.clone(),
        notes: notes.clone(),
    };

    let json_path = cli.output_dir.join("discrimination_report.json");
    fs::write(&json_path, serde_json::to_string_pretty(&report)?)?;
    eprintln!("[eta_discriminate] Wrote {}", json_path.display());

    let md_path = cli.output_dir.join("discrimination_report.md");
    write_markdown(&md_path, &report)?;
    eprintln!("[eta_discriminate] Wrote {}", md_path.display());

    let repro_path = cli.output_dir.join("reproducibility_log.txt");
    write_repro_log(&repro_path, &cli, &report.git_sha, &blocks)?;
    eprintln!("[eta_discriminate] Wrote {}", repro_path.display());

    if let Some(w) = winning {
        eprintln!("[eta_discriminate] Discrimination verdict: {w} favoured");
    } else {
        eprintln!(
            "[eta_discriminate] Discrimination verdict: inconclusive (both candidates within 3σ or both excluded)"
        );
    }

    Ok(())
}
