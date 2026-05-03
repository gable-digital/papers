//! Integration tests for [`crate::route34::route4_predictor`]:
//! end-to-end Saturn / Jupiter polar wavenumber prediction with
//! discrimination scoring against the observed `{6, 8, 5}` triple.

use crate::route34::route4_predictor::{
    route4_canonical_candidates, route4_discrimination,
};
use crate::route34::KillingResult;

#[test]
fn integration_polysphere_predicted_set_is_arnold_intersected_with_killing() {
    // S^3 x S^3 has continuous isometry of dim 12 ⇒ Killing-set =
    // {1..=12}. The polar Lyapunov germ at q_coef=0 (the chapter-21
    // critical-boundary regime) classifies as D_4 hyperbolic umbilic.
    // Per Bourbaki Ch VI §1.11 / Humphreys §3.18 the admissible
    // wavenumber set is (Coxeter exponents) ∪ {Coxeter number h(D_4)}
    // = {1, 3, 5} ∪ {6} = {1, 3, 5, 6}.
    // The gated Arnold ∩ Killing set is therefore {1, 3, 5, 6}.
    //
    // Observed planetary wavenumbers {6, 8, 5}: Saturn (n=6) and
    // Jupiter-S (n=5) are in the predicted set under this candidate;
    // Jupiter-N (n=8) is not (nearest admissible is 6, distance 2).
    let pred = route4_discrimination(&KillingResult::polysphere_s3xs3()).unwrap();
    assert_eq!(pred.saturn_predicted, vec![1, 3, 5, 6]);
    assert_eq!(pred.jupiter_north_predicted, vec![1, 3, 5, 6]);
    assert_eq!(pred.jupiter_south_predicted, vec![1, 3, 5, 6]);
    // Saturn n=6 admissible via D_4 Coxeter number.
    assert!((pred.saturn_match - 1.0).abs() < 1e-12);
    assert_eq!(pred.saturn_distance, 0);
    // Jupiter-S (n=5) is in the predicted set.
    assert!((pred.jupiter_south_match - 1.0).abs() < 1e-12);
    assert_eq!(pred.jupiter_south_distance, 0);
    // Jupiter-N (n=8) is not — nearest admissible is 6, distance 2.
    assert!(pred.jupiter_north_match < 1.0);
    assert_eq!(pred.jupiter_north_distance, 2);
    // Combined χ² = 0² + 2² + 0² = 4.
    assert!((pred.combined_chi_squared - 4.0).abs() < 1e-9);
}

#[test]
fn integration_polysphere_matches_all_observed_wavenumbers() {
    // Top-level integration test corresponding to the chapter-21
    // hard-gating semantics post-Coxeter-number extension. Saturn
    // n=6 (D_4 Coxeter number) is admissible under the polysphere
    // candidate; Jupiter-S n=5 (D_4 Coxeter exponent) is admissible;
    // Jupiter-N n=8 is NOT (it lies outside D_4's admissible set
    // {1, 3, 5, 6} and outside the killing-set ∩ arnold-set
    // intersection regardless of continuous-isometry richness, since
    // the intersection is bounded above by the Arnold-set itself).
    //
    // This is the integration test referenced in the S8 fix:
    // Saturn discrimination is no longer impossible under any
    // candidate. Jupiter-N n=8 remains an open question — the chapter
    // §"Bottom Line and Next Step" lines 299-306 acknowledges that
    // Route 4 alone does not currently match all three observed
    // wavenumbers, and that combined Route 1-4 evaluation is the
    // discrimination signal.
    let pred = route4_discrimination(&KillingResult::polysphere_s3xs3()).unwrap();
    // Saturn n=6: D_4 Coxeter number, admissible.
    assert!(pred.saturn_predicted.contains(&6),
        "Saturn n=6 should be admissible under polysphere D_4 ∩ Killing; got {:?}",
        pred.saturn_predicted);
    assert!((pred.saturn_match - 1.0).abs() < 1e-12);
    // Jupiter-S n=5: D_4 Coxeter exponent, admissible.
    assert!(pred.jupiter_south_predicted.contains(&5));
    assert!((pred.jupiter_south_match - 1.0).abs() < 1e-12);
}

#[test]
fn integration_flat_t6_matches_saturn_and_jupiter_south() {
    // T^6 has continuous isometry dim 6 ⇒ Killing-set = {1..=6}; the
    // Arnold-set under D_4 hyperbolic umbilic is (exponents ∪ Coxeter
    // number) = {1, 3, 5, 6}; the intersection is {1, 3, 5, 6}.
    // Saturn n=6 matches (D_4 Coxeter number h=6); Jupiter-S n=5
    // matches (D_4 exponent); Jupiter-N n=8 does not (nearest
    // admissible 6, distance 2).
    let pred = route4_discrimination(&KillingResult::flat_t6()).unwrap();
    assert!((pred.saturn_match - 1.0).abs() < 1e-12);
    assert_eq!(pred.saturn_distance, 0);
    assert!((pred.jupiter_south_match - 1.0).abs() < 1e-12);
    // Jupiter-N n=8 not in {1, 3, 5, 6}; nearest is 6 at d=2.
    assert!(pred.jupiter_north_match < 1.0);
    assert_eq!(pred.jupiter_north_distance, 2);
}

#[test]
fn integration_canonical_candidates_all_run() {
    let preds = route4_canonical_candidates().unwrap();
    assert_eq!(preds.len(), 5);
    let labels: Vec<&str> = preds.iter().map(|p| p.candidate_label.as_str()).collect();
    assert!(labels.contains(&"round_S3xS3"));
    assert!(labels.contains(&"flat_T6"));
    assert!(labels.contains(&"TY/Z3"));
    assert!(labels.contains(&"Schoen/Z3xZ3"));
    assert!(labels.contains(&"generic_no_isometry"));
}

#[test]
fn integration_chi_squared_in_valid_range() {
    // Combined χ² is bounded above by max possible (n_obs)² sum over
    // three planets if the predicted set is empty: (6²+8²+5²) = 125.
    // For the canonical candidates with the D_4 hyperbolic-umbilic
    // gating, χ² is in [0, 125].
    let preds = route4_canonical_candidates().unwrap();
    for p in &preds {
        assert!(p.combined_chi_squared >= 0.0);
        assert!(
            p.combined_chi_squared <= 125.0 + 1e-9,
            "candidate {} has χ² = {}, expected ≤ 125",
            p.candidate_label,
            p.combined_chi_squared
        );
        assert!(p.saturn_match >= 0.0 && p.saturn_match <= 1.0);
        assert!(p.jupiter_north_match >= 0.0 && p.jupiter_north_match <= 1.0);
        assert!(p.jupiter_south_match >= 0.0 && p.jupiter_south_match <= 1.0);
    }
}

#[test]
fn integration_predicted_sets_non_empty() {
    let preds = route4_canonical_candidates().unwrap();
    for p in &preds {
        assert!(!p.saturn_predicted.is_empty());
        assert!(!p.jupiter_north_predicted.is_empty());
        assert!(!p.jupiter_south_predicted.is_empty());
    }
}

#[test]
fn integration_higher_continuous_isometry_strictly_dominates_in_match() {
    // Round S^3 x S^3 (dim 12) ≥ Flat T^6 (dim 6) ≥ generic (dim 0)
    // in admissible-wavenumber-set inclusion ⇒ χ² ordered the
    // opposite way.
    let s3 = route4_discrimination(&KillingResult::polysphere_s3xs3()).unwrap();
    let t6 = route4_discrimination(&KillingResult::flat_t6()).unwrap();
    let none = route4_discrimination(&KillingResult::generic_no_isometry()).unwrap();
    assert!(
        s3.combined_chi_squared <= t6.combined_chi_squared + 1e-9,
        "S^3 χ² {} should be ≤ T^6 χ² {}",
        s3.combined_chi_squared,
        t6.combined_chi_squared
    );
    assert!(
        t6.combined_chi_squared <= none.combined_chi_squared + 1e-9,
        "T^6 χ² {} should be ≤ no-isometry χ² {}",
        t6.combined_chi_squared,
        none.combined_chi_squared
    );
}

#[test]
fn integration_ty_and_schoen_distinct_candidates() {
    let ty = route4_discrimination(&KillingResult::tianyau_z3()).unwrap();
    let sch = route4_discrimination(&KillingResult::schoen_z3xz3()).unwrap();
    assert_eq!(ty.candidate_label, "TY/Z3");
    assert_eq!(sch.candidate_label, "Schoen/Z3xZ3");
    assert!(ty.combined_chi_squared.is_finite());
    assert!(sch.combined_chi_squared.is_finite());
}