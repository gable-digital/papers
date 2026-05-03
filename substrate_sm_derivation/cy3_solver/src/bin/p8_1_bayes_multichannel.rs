//! # P8.1 — Multichannel Bayes-factor diagnostic binary.
//!
//! Loads per-channel log-Bayes-factor evidence from the P5.10 σ-channel
//! (`output/p5_10_ty_schoen_5sigma.json`), the P7.11 chain-match channels
//! (`output/p7_11_quark_chain.json`, `output/p7_11_lepton_chain.json`),
//! and — when available — the in-flight P8.2 Hodge-consistency channel and
//! P8.3 Yukawa-spectrum channel, combines them under the independence
//! assumption, and writes a JSON summary plus a human-readable terminal
//! report.
//!
//! Channels missing from disk are skipped with a console note (the Hodge
//! and Yukawa channels are still being built by sibling agents).
//!
//! Output:
//! * stdout — per-channel log-BF table + combined verdict
//! * `output/p8_1_bayes_multichannel.json` — full
//!   [`MultichannelBayesResult`] for downstream consumers.

use std::fs;
use std::path::Path;

use cy3_rust_solver::route34::bayes_factor_multichannel::{
    combine_channels, from_chain_match, from_hodge_consistency,
    from_sigma_distribution_with_breakdown, from_yukawa_spectrum, ChainType, ChannelEvidence,
    DiscriminabilityReport, MultichannelBayesResult, SigmaChannelBreakdown,
};

/// Default output path for the JSON summary. Relative to the package root
/// (matches the pattern used by the other P5/P6/P7 diagnostic binaries).
const OUTPUT_JSON: &str = "output/p8_1_bayes_multichannel.json";

/// Default input paths.
const SIGMA_INPUT: &str = "output/p5_10_ty_schoen_5sigma.json";
const QUARK_CHAIN_INPUT: &str = "output/p7_11_quark_chain.json";
const LEPTON_CHAIN_INPUT: &str = "output/p7_11_lepton_chain.json";
const HODGE_INPUT: &str = "output/p8_2_hodge_channel.json";
const YUKAWA_INPUT: &str = "output/p8_3_yukawa_channel.json";

fn try_load_optional<F>(label: &str, path: &str, loader: F) -> Option<ChannelEvidence>
where
    F: FnOnce(&str) -> Result<ChannelEvidence, String>,
{
    if !Path::new(path).exists() {
        eprintln!(
            "[skip] {label} channel: input file not present at {path} \
             (in-flight; sibling agent has not landed yet)"
        );
        return None;
    }
    match loader(path) {
        Ok(c) => {
            eprintln!(
                "[load] {label} channel from {path}: ln BF = {:+.4} (n_obs = {})",
                c.log_bf_ty_vs_schoen, c.n_observations
            );
            Some(c)
        }
        Err(e) => {
            eprintln!("[skip] {label} channel: load failed: {e}");
            None
        }
    }
}

fn verdict_label(combined_log_bf: f64) -> &'static str {
    if combined_log_bf > 0.0 {
        "TY-favored"
    } else if combined_log_bf < 0.0 {
        "Schoen-favored"
    } else {
        "inconclusive"
    }
}

fn print_result(r: &MultichannelBayesResult, sigma_breakdown: Option<&SigmaChannelBreakdown>) {
    println!();
    println!("==============================================================");
    println!(" P8.1 — Multichannel Bayes-factor combiner (P8.1e)");
    println!("==============================================================");
    println!();
    if let Some(bd) = sigma_breakdown {
        println!(" σ-channel Welch t-test breakdown (discriminability):");
        println!("   k                       = {}", bd.k);
        println!(
            "   n_TY / n_Schoen         = {} / {}",
            bd.n_ty, bd.n_schoen
        );
        println!(
            "   sigma_pop_TY            = {:.6e}  (sample SD)",
            bd.sigma_pop_ty
        );
        println!(
            "   sigma_pop_Schoen        = {:.6e}  (sample SD)",
            bd.sigma_pop_schoen
        );
        println!(
            "   pooled SE               = {:.6e}",
            bd.pooled_sd
        );
        println!(
            "   Welch t-statistic       = {:+.4}",
            bd.t_statistic
        );
        println!(
            "   σ-channel ln BF (BIC)   = {:+.4}  (DIAGNOSTIC ONLY — not in combined BF)",
            bd.total_log_bf
        );
        println!();
    }
    println!(" Per-channel evidence:");
    println!(
        "   {:<14} {:>14} {:>14} {:>10} {:>8}",
        "channel", "ln BF_TY:Schoen", "log10 BF", "n_obs", "in_BF?"
    );
    for c in &r.channels {
        let log10_bf = c.log_bf_ty_vs_schoen / core::f64::consts::LN_10;
        let in_bf = if c.for_combination { "yes" } else { "no" };
        println!(
            "   {:<14} {:>+14.4} {:>+14.4} {:>10} {:>8}",
            c.name, c.log_bf_ty_vs_schoen, log10_bf, c.n_observations, in_bf
        );
    }
    println!();
    println!("--------------------------------------------------------------");
    println!(" HEADLINE 1 — Discriminability (σ-channel, basis-size-saturated):");
    if let Some(disc) = r.discriminability.as_ref() {
        println!(
            "   t = {:+.4}, n-σ = {:.2}, n_TY = {}, n_Schoen = {}",
            disc.sigma_t_statistic,
            disc.sigma_n_sigma,
            disc.sigma_n_ty,
            disc.sigma_n_schoen
        );
    } else {
        println!("   (σ-channel not loaded)");
    }
    println!();
    println!(" HEADLINE 2 — Physics preference (chain + Hodge + Yukawa, σ excluded):");
    let physics_channels: Vec<&ChannelEvidence> = r
        .channels
        .iter()
        .filter(|c| c.for_combination)
        .collect();
    println!(
        "   physics channels combined = {}",
        physics_channels
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>()
            .join(" + ")
    );
    println!("   combined ln BF         = {:+.4} nats", r.combined_log_bf);
    println!("   combined log10 BF      = {:+.4}", r.combined_log10_bf);
    println!("   posterior odds TY:S    = {:.4e}", r.posterior_odds_ty);
    println!(
        "   n-σ-equivalent (CCGV)  = {:.3} σ",
        r.n_sigma_equivalent
    );
    println!(
        "   Jeffreys-Trotta strength = {}",
        r.combined_strength_label
    );
    println!("   verdict                = {}", verdict_label(r.combined_log_bf));
    println!("--------------------------------------------------------------");
    println!();
    println!(
        " Total observations across channels: {} (provenance count, includes σ)",
        r.total_observations
    );
    println!("==============================================================");
}

fn write_json(
    path: &str,
    r: &MultichannelBayesResult,
    sigma_breakdown: Option<&SigmaChannelBreakdown>,
) -> Result<(), String> {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("p8_1: cannot create {}: {}", parent.display(), e))?;
        }
    }
    // Compose a wrapper object so we can surface the P8.1e split
    // explicitly: top-level `discriminability` (σ-channel headline,
    // |t|/n-σ — diagnostic only) and `physics_preference` (combined BF
    // over chain + Hodge + Yukawa, σ excluded). The full result and the
    // σ-channel per-batch breakdown remain available for audit.
    let physics_channels: Vec<&ChannelEvidence> = r
        .channels
        .iter()
        .filter(|c| c.for_combination)
        .collect();
    let physics_per_channel: Vec<serde_json::Value> = physics_channels
        .iter()
        .map(|c| {
            serde_json::json!({
                "name": c.name,
                "log_bf_ty_vs_schoen": c.log_bf_ty_vs_schoen,
                "n_observations": c.n_observations,
                "source": c.source,
            })
        })
        .collect();
    let physics_preference = serde_json::json!({
        "channels": physics_per_channel,
        "combined_log_bf": r.combined_log_bf,
        "combined_log10_bf": r.combined_log10_bf,
        "posterior_odds_ty": r.posterior_odds_ty,
        "n_sigma_equivalent": r.n_sigma_equivalent,
        "combined_strength_label": r.combined_strength_label,
        "verdict": verdict_label(r.combined_log_bf),
        "convention": "positive ln BF = TY-favored, negative = Schoen-favored",
    });
    let wrapper = serde_json::json!({
        "discriminability": r.discriminability,
        "physics_preference": physics_preference,
        "result": r,
        "sigma_channel_breakdown": sigma_breakdown,
        "p8_1e_note":
            "σ-channel is reported as discriminability only (|t|, n-σ). The model-\
             comparison Bayes factor (`physics_preference.combined_log_bf`) excludes σ \
             because the σ-gap is dominated by basis-size differences (n_TY=87 vs \
             n_Schoen=27 at k=3), not by physics observables. See \
             references/p5_10_5sigma_target.md for the full justification.",
    });
    let json = serde_json::to_string_pretty(&wrapper)
        .map_err(|e| format!("p8_1: JSON serialise failed: {}", e))?;
    fs::write(path, json).map_err(|e| format!("p8_1: cannot write {}: {}", path, e))?;
    eprintln!("[ok] wrote {}", path);
    Ok(())
}

fn main() {
    eprintln!("[start] P8.1 multichannel Bayes-factor combiner");

    let mut channels: Vec<ChannelEvidence> = Vec::new();
    let sigma_breakdown: Option<SigmaChannelBreakdown>;

    // Required channel: σ-distribution (P5.10 → Welch two-sample t-test).
    // History:
    //   * P8.1   self-anchor at the TY mean — tautological inflation.
    //   * P8.1b  per-seed Δ_i with SE_mean as predictive width — 10⁹-nats
    //            inflation under SE asymmetry.
    //   * P8.1c  Welch's t on the means, σ_pop from per-seed vectors,
    //            Laplace/BIC ln BF.
    //   * P8.1d  σ-channel sign convention "lower σ = candidate favored".
    //   * P8.1e  σ excluded from model-comparison BF (basis-size
    //            artifact, n_TY=87 vs n_Schoen=27); retained as
    //            *discriminability* metric (|t|, n-σ). Physics verdict
    //            comes from chain + Hodge + Yukawa channels only. See
    //            references/p5_10_5sigma_target.md.
    match from_sigma_distribution_with_breakdown(SIGMA_INPUT) {
        Ok((c, bd)) => {
            eprintln!(
                "[load] sigma channel from {}: ln BF = {:+.4} (n_obs = {}; \
                 Welch t = {:+.4}, n_TY = {}, n_Schoen = {}, σ_pop_TY = {:.3e}, \
                 σ_pop_Schoen = {:.3e})",
                SIGMA_INPUT,
                c.log_bf_ty_vs_schoen,
                c.n_observations,
                bd.t_statistic,
                bd.n_ty,
                bd.n_schoen,
                bd.sigma_pop_ty,
                bd.sigma_pop_schoen
            );
            channels.push(c);
            sigma_breakdown = Some(bd);
        }
        Err(e) => {
            eprintln!(
                "[fatal] sigma channel is the canonical P5.8 input but failed to load: {e}"
            );
            std::process::exit(2);
        }
    }

    // Required channel: chain-match quark (P7.11).
    match from_chain_match(QUARK_CHAIN_INPUT, ChainType::Quark) {
        Ok(c) => {
            eprintln!(
                "[load] chain_quark channel from {}: ln BF = {:+.4} (n_obs = {})",
                QUARK_CHAIN_INPUT, c.log_bf_ty_vs_schoen, c.n_observations
            );
            channels.push(c);
        }
        Err(e) => {
            eprintln!("[fatal] chain_quark channel failed to load: {e}");
            std::process::exit(2);
        }
    }

    // Required channel: chain-match lepton (P7.11).
    match from_chain_match(LEPTON_CHAIN_INPUT, ChainType::Lepton) {
        Ok(c) => {
            eprintln!(
                "[load] chain_lepton channel from {}: ln BF = {:+.4} (n_obs = {})",
                LEPTON_CHAIN_INPUT, c.log_bf_ty_vs_schoen, c.n_observations
            );
            channels.push(c);
        }
        Err(e) => {
            eprintln!("[fatal] chain_lepton channel failed to load: {e}");
            std::process::exit(2);
        }
    }

    // Optional channels: Hodge (P8.2) + Yukawa (P8.3) — still being built.
    if let Some(c) = try_load_optional("hodge", HODGE_INPUT, from_hodge_consistency) {
        channels.push(c);
    }
    if let Some(c) = try_load_optional("yukawa", YUKAWA_INPUT, from_yukawa_spectrum) {
        channels.push(c);
    }

    let mut result = combine_channels(channels);
    // Surface σ-channel as discriminability headline alongside the
    // (σ-excluded) physics-preference BF.
    result.discriminability = sigma_breakdown
        .as_ref()
        .map(DiscriminabilityReport::from_sigma_breakdown);
    print_result(&result, sigma_breakdown.as_ref());

    if let Err(e) = write_json(OUTPUT_JSON, &result, sigma_breakdown.as_ref()) {
        eprintln!("[warn] {e}");
        // Don't fail the process on JSON write failure — the report has
        // already been printed to stdout.
    }
}
