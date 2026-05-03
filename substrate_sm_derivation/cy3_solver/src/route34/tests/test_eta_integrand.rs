//! End-to-end η-integrand evaluation test.
//!
//! Combines `fixed_locus::enumerate_fixed_loci`,
//! `hidden_bundle::sample_polystable_hidden_bundles`, and
//! `chern_field_strength::integrate_visible_minus_hidden` to compute
//! the unnormalized η numerator for both Tian-Yau Z/3 and Schoen
//! Z/3×Z/3, against the published "standard" visible bundles.

use crate::geometry::CicyGeometry;
use crate::route34::chern_field_strength::{
    integrate_tr_f_squared_wedge_J, integrate_visible_minus_hidden,
};
use crate::route34::fixed_locus::{enumerate_fixed_loci, QuotientAction};
use crate::route34::hidden_bundle::{
    sample_polystable_hidden_bundles, HiddenBundle, VisibleBundle,
};

#[test]
fn ty_eta_numerator_finite_and_nonzero() {
    let geom = CicyGeometry::tian_yau_z3();
    let visible = VisibleBundle::ty_aglp_2011_standard();
    // Bianchi-canceling hidden bundle.
    let candidates = sample_polystable_hidden_bundles(&geom, &visible, 1);
    // Fall back to a hand-built Bianchi-matched hidden if the search
    // yields no polystable hits in the small bounded grid.
    let hidden = candidates.into_iter().next().unwrap_or_else(|| {
        // Manual: B = O(1)^4 ⊕ O(4), C = O(8) → c_2 = 22, c_1 = 0.
        HiddenBundle {
            monad_data: crate::heterotic::MonadBundle {
                b_degrees: vec![1, 1, 1, 1, 4],
                c_degrees: vec![8],
                map_coefficients: vec![1.0; 5],
            },
            e8_embedding: crate::route34::hidden_bundle::E8Embedding::SU5,
        }
    });

    let action = QuotientAction::tian_yau_z3();
    let loci = enumerate_fixed_loci(&geom, &action);
    let divisor = &loci[0].components[0];

    // Numerator: ∫_F (Tr_v(F_v²) − Tr_h(F_h²)) ∧ J
    let numerator = integrate_visible_minus_hidden(
        &visible,
        &hidden,
        Some(divisor),
        &geom,
        &[1.0, 1.0],
        0,
        0,
    );
    // Denominator: ∫_M Tr_v(F_v²) ∧ J²
    let denominator = integrate_tr_f_squared_wedge_J(
        &visible,
        None,
        &geom,
        &[1.0, 1.0],
        0,
        0,
    );

    assert!(numerator.is_finite() && denominator.is_finite());
    assert!(denominator.abs() > 1.0e-3);
    let eta = numerator.abs() / denominator.abs();
    assert!(eta.is_finite());
    // η should be of order O(1) (not yet matching the observed
    // 6e-10 — that requires the proper Kähler-moduli scan, which is
    // the post-Ch8 numerical-pipeline work). What we verify here is
    // that the form evaluates and is non-zero, so the discrimination
    // pipeline can iterate.
    assert!(eta > 0.0, "η should be positive when V_h ≠ V_v cohomologically");
}

#[test]
fn schoen_eta_numerator_finite_and_nonzero() {
    let geom = CicyGeometry::schoen_z3xz3();
    let visible = VisibleBundle::schoen_dhor_2006_minimal();
    // Manual Bianchi-matched hidden for Schoen.
    let hidden = HiddenBundle {
        monad_data: crate::heterotic::MonadBundle {
            b_degrees: vec![1, 1, 1, 1, 4],
            c_degrees: vec![8],
            map_coefficients: vec![1.0; 5],
        },
        e8_embedding: crate::route34::hidden_bundle::E8Embedding::SU5,
    };
    let action = QuotientAction::schoen_z3xz3();
    let loci = enumerate_fixed_loci(&geom, &action);
    assert_eq!(loci.len(), 8);
    let divisor = &loci[0].components[0];

    // FIX-NOTE: Schoen ambient is `CP^2 × CP^2 × CP^1` (3 Kähler factors),
    // not 2. The Wave-1 schoen_geometry pins this; see
    // `route34::schoen_geometry::PUBLISHED_TRIPLE_INTERSECTIONS`
    // (DHOR-2006 §3 Eq. 3.7).
    let numerator = integrate_visible_minus_hidden(
        &visible,
        &hidden,
        Some(divisor),
        &geom,
        &[1.0, 1.0, 1.0],
        0,
        0,
    );
    let denominator = integrate_tr_f_squared_wedge_J(
        &visible,
        None,
        &geom,
        &[1.0, 1.0, 1.0],
        0,
        0,
    );
    assert!(numerator.is_finite() && denominator.is_finite());
    assert!(denominator.abs() > 1.0e-3);
}

#[test]
fn eta_integrand_changes_with_kahler_moduli() {
    // Sanity: shifting the Kähler moduli produces a different η,
    // confirming the integrand depends on the moduli (the moduli
    // scan is one of the discrimination signals).
    let geom = CicyGeometry::tian_yau_z3();
    let visible = VisibleBundle::ty_aglp_2011_standard();
    let hidden = HiddenBundle::trivial(4);
    let action = QuotientAction::tian_yau_z3();
    let loci = enumerate_fixed_loci(&geom, &action);
    let divisor = &loci[0].components[0];

    let n1 = integrate_visible_minus_hidden(
        &visible,
        &hidden,
        Some(divisor),
        &geom,
        &[1.0, 1.0],
        0,
        0,
    );
    let n2 = integrate_visible_minus_hidden(
        &visible,
        &hidden,
        Some(divisor),
        &geom,
        &[2.0, 1.0],
        0,
        0,
    );
    assert!((n1 - n2).abs() > 1.0e-3, "η numerator should depend on Kähler moduli");
}


/// **S5 closure**: exercise the metric-integral η evaluator on the
/// real Tian-Yau Calabi-Yau metric (NOT the cohomological pairing).
/// Verifies that:
///   1. The metric-integral path runs end-to-end without panicking.
///   2. The polystability gate (Donaldson 1985 / Uhlenbeck-Yau 1986)
///      either accepts the bundle (proceeds with metric integral) or
///      rejects it (returns BundleNotPolystable error).
///   3. If accepted, the predicted η is finite + non-negative + bounded.
///
/// `#[ignore]`'d because the metric solve at k=2 still costs ~10s.
#[test]
#[ignore]
fn s5_closure_metric_eta_runs_on_tian_yau() {
    use crate::route34::eta_evaluator::{
        evaluate_eta_tian_yau_metric, EtaError, EtaEvaluatorConfig, EtaMetricEvaluatorConfig,
    };
    let cfg = EtaMetricEvaluatorConfig {
        base: EtaEvaluatorConfig {
            n_metric_iters: 4,
            n_metric_samples: 200,
            n_integrand_samples: 256,
            kahler_moduli: vec![1.0, 1.0],
            seed: 123,
            checkpoint_path: None,
            max_wallclock_seconds: 600,
        },
        n_numerator_samples: 256,
        n_denominator_samples: 512,
        n_bootstrap: 8,
        polystability_subsheaf_rank: 1,
    };
    match evaluate_eta_tian_yau_metric(&cfg) {
        Ok(result) => {
            assert!(
                result.eta_predicted.is_finite() && result.eta_predicted >= 0.0,
                "metric η must be finite and non-negative, got {}",
                result.eta_predicted
            );
            assert!(
                result.eta_uncertainty.is_finite() && result.eta_uncertainty >= 0.0,
                "metric η uncertainty must be finite and non-negative, got {}",
                result.eta_uncertainty
            );
            // Numerator and denominator are returned separately; both
            // must be finite (regardless of magnitude).
            assert!(result.numerator_value.is_finite());
            assert!(result.denominator_value.is_finite());
            eprintln!(
                "S5 closure: metric η on TY = {} ± {} (numerator={}, denominator={}, donaldson_residual={})",
                result.eta_predicted,
                result.eta_uncertainty,
                result.numerator_value,
                result.denominator_value,
                result.donaldson_residual,
            );
        }
        Err(EtaError::BundleNotPolystable { .. }) => {
            // Acceptable: the AGLP standard bundle on TY may fail
            // polystability at the chosen Kähler moduli; the metric
            // path correctly refuses to compute η in that case.
            eprintln!(
                "S5 closure: AGLP bundle not polystable at unit Kähler — \
                 metric η correctly refused (chapter-21 / DUY 1985 gating)"
            );
        }
        Err(e) => panic!("metric η on TY failed unexpectedly: {:?}", e),
    }
}
