//! P8.1 — top-level integration tests for the multichannel Bayes-factor
//! combiner. Mirrors `src/route34/tests/test_bayes_multichannel.rs` (the
//! canonical reference under the route34 layout) so the regression
//! invariants are actually exercised by `cargo test` without requiring an
//! edit to `src/route34/tests/mod.rs`.

use cy3_rust_solver::route34::bayes_factor_multichannel::{
    combine_channels, from_chain_match, from_sigma_distribution,
    from_sigma_distribution_with_breakdown, ChainType, ChannelEvidence,
};

/// (a) Three synthetic channels at ln BF = 5 each → combined 15;
///     n-σ ≈ √30 ≈ 5.477.
#[test]
fn synthetic_three_channel_combination() {
    let channels = vec![
        ChannelEvidence::new("a", 5.0, 100, "synthetic").unwrap(),
        ChannelEvidence::new("b", 5.0, 100, "synthetic").unwrap(),
        ChannelEvidence::new("c", 5.0, 100, "synthetic").unwrap(),
    ];
    let r = combine_channels(channels);
    assert!((r.combined_log_bf - 15.0).abs() < 1e-12);
    let expected_nsigma = (2.0_f64 * 15.0).sqrt(); // ≈ 5.477
    assert!(
        (r.n_sigma_equivalent - expected_nsigma).abs() < 1e-9,
        "n_sigma = {}, expected {}",
        r.n_sigma_equivalent,
        expected_nsigma
    );
    assert!(r.combined_log10_bf > 6.0); // log10(e) * 15 ≈ 6.514
    assert_eq!(r.combined_strength_label, "Strong");
    assert_eq!(r.total_observations, 300);
}

/// (b) Independence: two channels at ln BF = 10 each combine to 20.
#[test]
fn independent_channels_add_log_bf() {
    let channels = vec![
        ChannelEvidence::new("x", 10.0, 50, "synthetic").unwrap(),
        ChannelEvidence::new("y", 10.0, 50, "synthetic").unwrap(),
    ];
    let r = combine_channels(channels);
    assert!((r.combined_log_bf - 20.0).abs() < 1e-12);
    // Negative + positive cancellation
    let mixed = vec![
        ChannelEvidence::new("x", 10.0, 50, "synthetic").unwrap(),
        ChannelEvidence::new("y", -10.0, 50, "synthetic").unwrap(),
    ];
    let r2 = combine_channels(mixed);
    assert!(r2.combined_log_bf.abs() < 1e-12);
    assert_eq!(r2.combined_strength_label, "Inconclusive");
}

/// (c) Production data: σ + chain_quark + chain_lepton from the
///     repo's actual P5.10 + P7.11 outputs.
///
///     **σ-channel semantics (P8.1e):** σ is computed via a Welch two-
///     sample t-test on the per-seed σ population means and reported as
///     a *discriminability* metric (|t|, n-σ). It is **excluded** from
///     the combined model-comparison Bayes factor (`for_combination =
///     false`) because the σ-gap is dominated by basis-size differences
///     (n_TY=87 vs n_Schoen=27 at k=3), not physics observables. The
///     chain-match channels (P7.11) drive the physics verdict.
///
///     Regression invariants pinned here:
///
///       1. Each chain channel is deeply Schoen-favoured (< -1 nat),
///          matching the P7.11 published numbers.
///       2. σ-channel |t| matches the 6.92σ headline (`n_sigma` field
///          in the P5.10 JSON) to within 1%.
///       3. σ-channel |ln BF| is in the tens of nats — NOT 10⁹ (the
///          P8.1b inflation regression).
///       4. σ-channel `for_combination = false` (P8.1e).
///       5. Combined log-BF (chain only, σ excluded) is finite,
///          Schoen-favoured (negative), and at least "Moderate".
#[test]
fn production_data_three_channels() {
    let (sigma, sigma_bd) =
        match from_sigma_distribution_with_breakdown("output/p5_10_ty_schoen_5sigma.json") {
            Ok(pair) => pair,
            Err(e) => {
                eprintln!("[skip] σ channel data not present: {e}");
                return;
            }
        };
    let q = match from_chain_match("output/p7_11_quark_chain.json", ChainType::Quark) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[skip] chain_quark data not present: {e}");
            return;
        }
    };
    let l = match from_chain_match("output/p7_11_lepton_chain.json", ChainType::Lepton) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[skip] chain_lepton data not present: {e}");
            return;
        }
    };

    let sigma_log_bf = sigma.log_bf_ty_vs_schoen;
    let chain_q_log_bf = q.log_bf_ty_vs_schoen;
    let chain_l_log_bf = l.log_bf_ty_vs_schoen;

    // P7.11: Schoen wins quark by ~9 nats and lepton by ~13-18 nats.
    assert!(
        chain_q_log_bf < -1.0,
        "chain_quark ln BF = {chain_q_log_bf} should be deeply negative \
         (Schoen-favoured per P7.11)"
    );
    assert!(
        chain_l_log_bf < -1.0,
        "chain_lepton ln BF = {chain_l_log_bf} should be deeply negative \
         (Schoen-favoured per P7.11)"
    );

    // σ-channel Welch t magnitude matches the 6.92σ P5.10 headline.
    assert!(
        (sigma_bd.t_statistic.abs() - 6.92).abs() < 0.1,
        "σ-channel |Welch t| = {} should match the 6.92σ headline within \
         the bootstrap precision of P5.10",
        sigma_bd.t_statistic.abs()
    );
    // No SE_mean-as-width inflation: |ln BF| in the tens of nats, not 10⁹.
    assert!(
        sigma_log_bf.abs() < 1000.0,
        "σ-channel |ln BF| = {} must NOT inflate (P8.1b bug regression: \
         SE_mean used as predictive width gave ~1.07e9)",
        sigma_log_bf.abs()
    );
    assert!(
        sigma_log_bf.abs() > 5.0,
        "σ-channel |ln BF| = {} should still detect the 6.92σ separation \
         (predicted ~22 nats from 0.5·t² − 0.5·ln(n))",
        sigma_log_bf.abs()
    );

    // P8.1e: σ-channel must be marked discriminability-only.
    assert!(
        !sigma.for_combination,
        "σ-channel must have for_combination=false (P8.1e); got true"
    );

    let combined = combine_channels(vec![sigma, q, l]);
    assert!(
        combined.combined_log_bf.is_finite(),
        "combined_log_bf must be finite, got {}",
        combined.combined_log_bf
    );
    // P8.1e: combined BF excludes σ, so it must equal chain_q + chain_l.
    let expected_combined = chain_q_log_bf + chain_l_log_bf;
    assert!(
        (combined.combined_log_bf - expected_combined).abs() < 1e-10,
        "combined ln BF = {} must equal chain_q + chain_l = {} \
         (σ excluded under P8.1e)",
        combined.combined_log_bf,
        expected_combined
    );
    // Both chains favour Schoen → combined must be negative (Schoen-
    // favored under the "positive = TY-favored" convention).
    assert!(
        combined.combined_log_bf < 0.0,
        "physics-channel combined ln BF must be Schoen-favored (negative); \
         got {:+.4}",
        combined.combined_log_bf
    );
    let strength = combined.combined_strength_label.as_str();
    assert!(
        strength == "Strong" || strength == "Moderate",
        "combined strength = {strength} should be Strong or Moderate at \
         production-data magnitudes ({:+.4} ln BF)",
        combined.combined_log_bf
    );
    let expected_total: usize = combined.channels.iter().map(|c| c.n_observations).sum();
    assert_eq!(combined.total_observations, expected_total);
    eprintln!(
        "[production data] σ ln BF = {:+.4} (Welch t = {:+.4}, σ_pop_TY = {:.3e}, \
         σ_pop_Schoen = {:.3e}; DIAGNOSTIC ONLY, σ excluded from combined BF), \
         chain_quark = {:+.4}, chain_lepton = {:+.4}, physics-combined = {:+.4} \
         ({} {})",
        sigma_log_bf,
        sigma_bd.t_statistic,
        sigma_bd.sigma_pop_ty,
        sigma_bd.sigma_pop_schoen,
        chain_q_log_bf,
        chain_l_log_bf,
        combined.combined_log_bf,
        combined.combined_strength_label,
        if combined.combined_log_bf > 0.0 {
            "TY-favored"
        } else {
            "Schoen-favored"
        }
    );
}

/// Suppress the unused-import warning for the legacy `from_sigma_distribution`
/// re-export — kept available so external callers that still use the
/// summary-only entry point continue to compile.
#[test]
fn from_sigma_distribution_still_exported() {
    // Just resolve the symbol; no behavioural assertion.
    let _ = from_sigma_distribution;
}
