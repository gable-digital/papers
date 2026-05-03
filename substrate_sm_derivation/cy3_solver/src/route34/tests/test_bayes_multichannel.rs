//! P8.1 — multichannel Bayes-factor combiner regression tests.
//!
//! NOTE on wiring: `src/route34/tests/mod.rs` is owned by the parent
//! crate's test registry and was deliberately not modified by this
//! agent (per the work-unit constraint). The same test cases are
//! mirrored as a top-level integration test at
//! `tests/test_bayes_multichannel_integration.rs` (Cargo auto-discovers
//! `tests/*.rs`), which is what gets run by `cargo test`. This file is
//! kept as the canonical reference under the standard route34 layout
//! so future maintainers landing alongside the µ.2/µ.3 sister-channel
//! work can flip a single line in `src/route34/tests/mod.rs` to wire
//! it in without rewriting the body.
//!
//! See [`crate::route34::bayes_factor_multichannel`] for the API under
//! test.

#[cfg(test)]
mod tests {
    use crate::route34::bayes_factor_multichannel::{
        combine_channels, from_chain_match, from_sigma_distribution_with_breakdown,
        ChainType, ChannelEvidence,
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
    ///     **P8.1e:** σ is reported as discriminability (|t|=6.92) and
    ///     excluded from the model-comparison BF sum (basis-size
    ///     artifact, not a physics observable). Combined BF is the
    ///     chain-only sum; physics verdict is Schoen-favored.
    #[test]
    fn production_data_three_channels() {
        let (sigma, sigma_bd) = match from_sigma_distribution_with_breakdown(
            "output/p5_10_ty_schoen_5sigma.json",
        ) {
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

        assert!(
            chain_q_log_bf < -1.0,
            "chain_quark ln BF = {chain_q_log_bf} should be Schoen-favoured per P7.11"
        );
        assert!(
            chain_l_log_bf < -1.0,
            "chain_lepton ln BF = {chain_l_log_bf} should be Schoen-favoured per P7.11"
        );

        // σ-channel Welch t magnitude tracks the 6.92σ P5.10 headline.
        assert!(
            (sigma_bd.t_statistic.abs() - 6.92).abs() < 0.1,
            "σ-channel |Welch t| = {} should match the 6.92σ headline",
            sigma_bd.t_statistic.abs()
        );
        // No SE_mean-as-width inflation: |ln BF| in the tens of nats.
        assert!(
            sigma_log_bf.abs() < 1000.0,
            "σ-channel |ln BF| = {} must NOT inflate past 10³ nats \
             (P8.1b regression: SE_mean used as width gave ~1.07e9)",
            sigma_log_bf.abs()
        );
        assert!(
            sigma_log_bf.abs() > 5.0,
            "σ-channel |ln BF| = {} should still detect the 6.92σ separation",
            sigma_log_bf.abs()
        );

        // P8.1e: σ excluded from combined BF.
        assert!(!sigma.for_combination, "σ-channel must be discriminability-only");
        let combined = combine_channels(vec![sigma, q, l]);
        assert!(combined.combined_log_bf.is_finite());
        let expected = chain_q_log_bf + chain_l_log_bf;
        assert!(
            (combined.combined_log_bf - expected).abs() < 1e-10,
            "combined ln BF must equal chain sum (σ excluded): {} vs {}",
            combined.combined_log_bf,
            expected
        );
        assert!(combined.combined_log_bf < 0.0, "physics verdict must be Schoen-favored");
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
             σ_pop_Schoen = {:.3e}), chain_quark = {:+.4}, chain_lepton = {:+.4}, \
             combined = {:+.4} ({})",
            sigma_log_bf,
            sigma_bd.t_statistic,
            sigma_bd.sigma_pop_ty,
            sigma_bd.sigma_pop_schoen,
            chain_q_log_bf,
            chain_l_log_bf,
            combined.combined_log_bf,
            combined.combined_strength_label
        );
    }
}
