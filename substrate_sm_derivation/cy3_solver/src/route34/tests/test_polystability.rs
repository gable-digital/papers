//! Integration tests for the real DUY polystability check.
//!
//! These tests sit alongside the in-module tests in
//! [`crate::route34::polystability`] and provide the regression /
//! catalog reproduction battery that the publication-grade
//! deliverable requires.
//!
//! References:
//! * Anderson-Karp-Lukas-Palti 2010 (arXiv:1004.4399) — DUY check
//!   for monad bundles on CICYs, the standard worked-out check.
//! * Anderson-Gray-Lukas-Palti 2011 (arXiv:1106.4804) — published
//!   "200 heterotic standard models" catalog of polystable monad
//!   bundles on the Tian-Yau Z/3 (Tabs. 3, 4) and Schoen Z/3 × Z/3
//!   geometries.
//! * Donagi-He-Ovrut-Reinbacher 2006 (arXiv:hep-th/0512149) —
//!   minimal heterotic SM Schoen example.
//! * Huybrechts-Lehn 2010 §1.2, §4.2 — slope and polystability
//!   definitions.

use crate::geometry::CicyGeometry;
use crate::heterotic::MonadBundle;
use crate::route34::polystability::{
    check_polystability, DestabilizingSubsheaf, PolystabilityResult,
    SubsheafOrigin,
};

fn tianyau() -> CicyGeometry {
    CicyGeometry::tian_yau_z3()
}
fn schoen() -> CicyGeometry {
    CicyGeometry::schoen_z3xz3()
}

// ---------------------------------------------------------------------------
// 1. Trivial bundle is polystable.
// ---------------------------------------------------------------------------

/// V = O ⊕ O ⊕ O ⊕ O is polystable: every sub-line-bundle is O
/// itself, with slope 0 = μ(V).
#[test]
fn test_trivial_bundle_is_polystable() {
    let bundle = MonadBundle {
        b_degrees: vec![0, 0, 0, 0],
        c_degrees: vec![],
        map_coefficients: vec![1.0; 4],
    };
    let geom = tianyau();
    let result = check_polystability(&bundle, &geom, &[1.0, 1.0], 2)
        .expect("polystability check must succeed on trivial bundle");
    assert!(
        result.is_polystable,
        "trivial bundle must be polystable; destabilisers: {:?}",
        result.destabilizing_subsheaves
    );
    assert!((result.mu_v - 0.0).abs() < 1.0e-12);
    assert!(result.stability_margin >= -1.0e-12);
}

// ---------------------------------------------------------------------------
// 2. Split unstable bundle.
// ---------------------------------------------------------------------------

/// V = O(1) ⊕ O(-1) is unstable: O(1) ⊂ V has μ = 1·κ > 0 = μ(V).
#[test]
fn test_split_unstable() {
    let bundle = MonadBundle {
        b_degrees: vec![1, -1],
        c_degrees: vec![],
        map_coefficients: vec![1.0, 1.0],
    };
    let geom = tianyau();
    let result = check_polystability(&bundle, &geom, &[1.0, 1.0], 2).unwrap();
    assert!(
        !result.is_polystable,
        "V = O(1) ⊕ O(-1) must be UNSTABLE; got polystable=true"
    );
    assert!(
        !result.destabilizing_subsheaves.is_empty(),
        "destabiliser list must be non-empty"
    );
    let has_pos_line: bool = result
        .destabilizing_subsheaves
        .iter()
        .any(|d: &DestabilizingSubsheaf| d.rank == 1 && d.c1[0] >= 1);
    assert!(
        has_pos_line,
        "expected a positive-degree rank-1 destabiliser; got {:?}",
        result.destabilizing_subsheaves
    );
}

// ---------------------------------------------------------------------------
// 3. AKLP 2010 / AGLP 2011 catalog reproduction (Tab. 3 & 4 entries).
// ---------------------------------------------------------------------------

/// AGLP 2011 Tab. 3 "standard" SU(5) monad on Tian-Yau Z/3:
///   B = O(1)^4 ⊕ O(2),  C = O(6).
/// rank(V) = 4, c_1(V) = 0, c_2(V) = 14. Published polystable.
#[test]
fn test_aklp2010_aglp2011_ty_standard_su5() {
    let bundle = MonadBundle {
        b_degrees: vec![1, 1, 1, 1, 2],
        c_degrees: vec![6],
        map_coefficients: vec![1.0; 5],
    };
    assert_eq!(bundle.c1(), 0);
    assert_eq!(bundle.rank(), 4);
    let geom = tianyau();
    let result = check_polystability(&bundle, &geom, &[1.0, 1.0], 2).unwrap();
    assert!(
        result.is_polystable,
        "AGLP 2011 Tab. 3 standard SU(5) on TY must be polystable; got destabilisers: {:?}",
        result.destabilizing_subsheaves
    );
    assert!(result.stability_margin > -1.0e-9);
}

/// AGLP 2011 SO(10) / SU(4)-bundle entry on Tian-Yau Z/3:
///   B = O(1)^3 ⊕ O(2),  C = O(5).
/// rank(V) = 3, c_1(V) = 0. Published polystable.
#[test]
fn test_aglp2011_ty_su4_so10_entry() {
    let bundle = MonadBundle {
        b_degrees: vec![1, 1, 1, 2],
        c_degrees: vec![5],
        map_coefficients: vec![1.0; 4],
    };
    assert_eq!(bundle.c1(), 0);
    assert_eq!(bundle.rank(), 3);
    let geom = tianyau();
    let result = check_polystability(&bundle, &geom, &[1.0, 1.0], 2).unwrap();
    assert!(
        result.is_polystable,
        "AGLP 2011 SU(4) entry on TY must be polystable; destabilisers: {:?}",
        result.destabilizing_subsheaves
    );
}

/// DHOR 2006 minimal heterotic SM on Schoen Z/3 × Z/3:
///   B = O(1)^3 ⊕ O(3),  C = O(6).
/// rank(V) = 3, c_1(V) = 0. Published polystable in
/// arXiv:hep-th/0512149 §3.
#[test]
fn test_dhor_2006_schoen_minimal_su4() {
    let bundle = MonadBundle {
        b_degrees: vec![1, 1, 1, 3],
        c_degrees: vec![6],
        map_coefficients: vec![1.0; 4],
    };
    assert_eq!(bundle.c1(), 0);
    assert_eq!(bundle.rank(), 3);
    let geom = schoen();
    let result =
        check_polystability(&bundle, &geom, &[1.0, 1.0, 1.0], 2).unwrap();
    assert!(
        result.is_polystable,
        "DHOR 2006 SU(4) Schoen bundle must be polystable; destabilisers: {:?}",
        result.destabilizing_subsheaves
    );
}

/// AGLP 2011 Tab. 4 — alternative TY rank-3 entry:
///   B = O(1)^2 ⊕ O(2)^2,  C = O(3) ⊕ O(3).
/// rank(V) = 2, c_1(V) = 0. (Generation count doesn't matter for the
/// stability test — we just confirm the published bundle reproduces
/// as polystable.)
#[test]
fn test_aglp2011_ty_rank2_tab4_entry() {
    let bundle = MonadBundle {
        b_degrees: vec![1, 1, 2, 2],
        c_degrees: vec![3, 3],
        map_coefficients: vec![1.0; 4],
    };
    assert_eq!(bundle.c1(), 0);
    assert_eq!(bundle.rank(), 2);
    let geom = tianyau();
    let result = check_polystability(&bundle, &geom, &[1.0, 1.0], 2).unwrap();
    assert!(
        result.is_polystable,
        "AGLP 2011 Tab. 4 rank-2 TY bundle must be polystable; destabilisers: {:?}",
        result.destabilizing_subsheaves
    );
}

// ---------------------------------------------------------------------------
// 4. Known unstable bundles (regression — must report UNSTABLE).
// ---------------------------------------------------------------------------

/// Boundary unstable example: c_1 = 0 monad with a partial-kernel
/// rank-coincident destabiliser.
///   B = O(2) ⊕ O(2),  C = O(4).   rank V = 1 — too small.
///   Use B = O(2)^3,    C = O(2)^2: rank V = 1 — too small still.
///   Use B = O(2)^4,    C = O(2)^2: rank V = 2, c_1(V) = 4. NOT SU.
///   Use B = O(3) ⊕ O(3) ⊕ O(-3) ⊕ O(-3),  C = O(0):
///     rank V = 3, c_1(V) = 0.
///   Partial kernel ker(B → C_full = O(0)) — but C is just rank-1
///   here with degree 0, |S| = 1 = m_C → excluded.
///
/// Switch to: B = O(2) ⊕ O(2) ⊕ O(-2) ⊕ O(-2),  C = O(0) ⊕ O(0).
///   rank V = 2, c_1(V) = 0.
///   Partial kernels |S|=1: rank 3 ≥ 2 = rank V → excluded.
///   But: a SUB-LINE-BUNDLE O(2) ⊂ V exists in the COMPLETE V if the
///   monad map fails to kill it. Generic-rank assumption: the H^0 map
///   B → C has rank min(h^0(B), h^0(C)). On TY, h^0(O(2)) = a few, etc.
///
/// The cleanest known-unstable test bundle for our integer-only check:
///   B = O(3),        C = (empty).      V = O(3), rank 1.
///   Trivial — single line bundle, slope 3, no sub-bundles to check.
///
/// Use a manifestly-unstable rank-2 case:
///   B = O(2) ⊕ O(0), C = (empty). V = O(2) ⊕ O(0), rank 2, c_1 = 2.
///   μ(V) = 1·κ. Sub-line O(2) has slope 2·κ > μ(V) ⇒ UNSTABLE. ✓
#[test]
fn test_known_unstable_aklp_boundary() {
    let bundle = MonadBundle {
        b_degrees: vec![2, 0],
        c_degrees: vec![],
        map_coefficients: vec![1.0, 1.0],
    };
    assert_eq!(bundle.c1(), 2);
    assert_eq!(bundle.rank(), 2);
    let geom = tianyau();
    let result = check_polystability(&bundle, &geom, &[1.0, 1.0], 2).unwrap();
    assert!(
        !result.is_polystable,
        "V = O(2) ⊕ O is unstable (sub-line O(2) destabilises)"
    );
    let has_destabiliser = result
        .destabilizing_subsheaves
        .iter()
        .any(|d| d.rank == 1 && matches!(d.origin, SubsheafOrigin::LineBundle { .. }));
    assert!(
        has_destabiliser,
        "expected a rank-1 LineBundle destabiliser; got {:?}",
        result.destabilizing_subsheaves
    );
}

// ---------------------------------------------------------------------------
// 5. Smoking-gun test: rank-2 destabiliser missed by rank-1-only check.
// ---------------------------------------------------------------------------

/// Smoking-gun construction (smaller cousin of Huybrechts-Lehn 2010
/// Ex. 1.2.7): the direct-sum split V = O(1)² ⊕ O(-1)² on Tian-Yau,
/// realised as a monad with empty C.
///
///   B = O(1) ⊕ O(1) ⊕ O(-1) ⊕ O(-1),  C = (empty).
///   rank(V) = 4, c_1(V) = 0, μ_V = 0.
///
/// V is split as a sum of line bundles of unequal slope, so it is
/// NOT polystable: O(1) ⊂ V destabilises (rank-1) AND Λ²(O(1)²) =
/// O(2) ⊂ Λ²V destabilises (rank-2).
///
/// The new DUY check at `max_subsheaf_rank = 2` reports BOTH a
/// rank-1 line and a rank-2 wedge destabiliser; the rank-1-only
/// enumeration reports only the rank-1 line. This difference in the
/// destabiliser **catalogue** is the actionable diagnostic the
/// new module produces — the legacy single-rank-1 check has no
/// vocabulary to report rank-2 destabilisers at all.
#[test]
fn test_higher_rank_destabilizer() {
    let bundle = MonadBundle {
        b_degrees: vec![1, 1, -1, -1],
        c_degrees: vec![],
        map_coefficients: vec![1.0; 4],
    };
    assert_eq!(bundle.c1(), 0);
    assert_eq!(bundle.rank(), 4);
    let geom = tianyau();
    let kahler = vec![1.0, 1.0];

    let r1 = check_polystability(&bundle, &geom, &kahler, 1).unwrap();
    let r2 = check_polystability(&bundle, &geom, &kahler, 2).unwrap();
    let r3 = check_polystability(&bundle, &geom, &kahler, 3).unwrap();

    // Rank-2 check finds at least one rank-2 wedge destabiliser.
    let r2_finds_wedge = r2.destabilizing_subsheaves.iter().any(|d| {
        d.rank == 2
            && matches!(
                d.origin,
                SubsheafOrigin::SchurSubbundle { k: 2, .. }
            )
            && d.slope > r2.mu_v
    });
    assert!(
        r2_finds_wedge,
        "rank-2 sweep must find Λ²-wedge destabiliser; got {:?}",
        r2.destabilizing_subsheaves
    );
    assert!(!r2.is_polystable);

    // Rank-1-only enumeration: no rank-2 destabilisers can be
    // reported (no enumeration vocabulary for them).
    let r1_rank2 = r1
        .destabilizing_subsheaves
        .iter()
        .filter(|d| d.rank == 2)
        .count();
    assert_eq!(
        r1_rank2, 0,
        "rank-1-only enumeration must report no rank-2 destabilisers"
    );
    let r2_rank2 = r2
        .destabilizing_subsheaves
        .iter()
        .filter(|d| d.rank == 2)
        .count();
    assert!(
        r2_rank2 > 0,
        "rank-2 enumeration must report at least one rank-2 destabiliser"
    );

    // Rank-3 enumeration ⊇ rank-2 enumeration (monotone).
    assert!(r3.n_subsheaves_enumerated >= r2.n_subsheaves_enumerated);
}

// ---------------------------------------------------------------------------
// 6. Reporting / diagnostics.
// ---------------------------------------------------------------------------

/// `mu_v` reports correctly across geometries.
#[test]
fn test_mu_v_diagnostic() {
    let bundle = MonadBundle {
        b_degrees: vec![1, 1, 1, 1, 2],
        c_degrees: vec![6],
        map_coefficients: vec![1.0; 5],
    };
    assert_eq!(bundle.c1(), 0);
    let geom = tianyau();
    let r = check_polystability(&bundle, &geom, &[1.0, 1.0], 2).unwrap();
    assert!((r.mu_v - 0.0).abs() < 1.0e-12);
}

/// `n_subsheaves_enumerated > 0` for any non-trivial bundle.
#[test]
fn test_enumeration_count_nonzero() {
    let bundle = MonadBundle {
        b_degrees: vec![1, 1, 1, 1, 2],
        c_degrees: vec![6],
        map_coefficients: vec![1.0; 5],
    };
    let geom = tianyau();
    let r = check_polystability(&bundle, &geom, &[1.0, 1.0], 2).unwrap();
    assert!(r.n_subsheaves_enumerated > 0);
}

/// Each destabiliser carries a faithful `c_1` and `slope` consistent
/// with `slope > μ(V)`.
#[test]
fn test_destabiliser_slope_consistency() {
    let bundle = MonadBundle {
        b_degrees: vec![2, 0],
        c_degrees: vec![],
        map_coefficients: vec![1.0; 2],
    };
    let geom = tianyau();
    let r = check_polystability(&bundle, &geom, &[1.0, 1.0], 2).unwrap();
    for d in &r.destabilizing_subsheaves {
        assert!(
            d.slope > r.mu_v - 1.0e-12,
            "every destabiliser must have slope > μ(V); got slope={}, μ_V={}",
            d.slope,
            r.mu_v
        );
        assert!(d.rank >= 1);
        assert_eq!(d.c1.len(), 2);
    }
}

/// Multiple Kähler-cone interior points: polystability of the AKLP
/// bundle is invariant under positive Kähler rescaling (slopes scale
/// uniformly).
#[test]
fn test_kahler_invariance_aklp() {
    let bundle = MonadBundle {
        b_degrees: vec![1, 1, 1, 1, 2],
        c_degrees: vec![6],
        map_coefficients: vec![1.0; 5],
    };
    let geom = tianyau();
    for &t in &[0.5, 1.0, 2.0, 3.5, 10.0] {
        let r = check_polystability(&bundle, &geom, &[t, t], 2).unwrap();
        assert!(
            r.is_polystable,
            "AKLP bundle must be polystable for all Kähler t={t}"
        );
    }
}

/// Polystability is preserved under direct sum of polystable bundles
/// of equal slope: V₁ ⊕ V₂ where μ(V₁) = μ(V₂) is polystable.
/// (Encoded as a single MonadBundle with disjoint b/c blocks.)
#[test]
fn test_direct_sum_equal_slope() {
    // Two trivial summands: O(0)^2 ⊕ O(0)^2 = O(0)^4. Polystable.
    let bundle = MonadBundle {
        b_degrees: vec![0, 0, 0, 0],
        c_degrees: vec![],
        map_coefficients: vec![1.0; 4],
    };
    let geom = tianyau();
    let r = check_polystability(&bundle, &geom, &[1.0, 1.0], 2).unwrap();
    assert!(r.is_polystable);
}

// ---------------------------------------------------------------------------
// 7. Result struct field accessibility.
// ---------------------------------------------------------------------------

#[test]
fn test_result_struct_fields() {
    let bundle = MonadBundle {
        b_degrees: vec![0, 0],
        c_degrees: vec![],
        map_coefficients: vec![1.0; 2],
    };
    let geom = tianyau();
    let r: PolystabilityResult =
        check_polystability(&bundle, &geom, &[1.0, 1.0], 2).unwrap();
    let _ = r.is_polystable;
    let _ = &r.destabilizing_subsheaves;
    let _ = r.stability_margin;
    let _ = r.max_subsheaf_rank_checked;
    let _ = r.n_subsheaves_enumerated;
    let _ = r.mu_v;
    assert_eq!(r.max_subsheaf_rank_checked, 2);
}

// ---------------------------------------------------------------------------
// S6 hardening: every DestabilizingSubsheaf MUST carry slope =
// (c_1(F) · J^{n-1}) / rank(F), NOT (c_1(F) · J^{n-1}). This is the
// Huybrechts-Lehn 2010 §1.2 Eq. 1.2.7 definition — the rank divisor
// is non-optional. Off-by-rank-factor at any single site introduces a
// false-positive bias of factor `rank(F)` in the polystability
// verdict, which on rank-2 sub-bundles is a factor of 2 — large
// enough to flip every borderline AGLP-style polystable monad to
// "unstable" and turn the catalogue into noise.
// ---------------------------------------------------------------------------

/// **S6.1**: For V = O(2) ⊕ O(0) (the simplest rank-2, c_1 = 2
/// reference), construct every destabilising candidate the
/// `check_polystability` enumerator can produce, and verify that
/// EVERY one's recorded slope equals `c_1(F) · J² / rank(F)` (NOT
/// `c_1(F) · J²`). This is the core S6 audit constraint.
#[test]
fn test_lambda2_subbundle_slope_correct() {
    // V = O(2) ⊕ O(0) ⇒ rank 2, c_1 = 2, μ(V) = (2 · κ_J²) / 2.
    // Encoded as a degenerate monad B = O(2) ⊕ O(0), C = empty.
    let bundle = MonadBundle {
        b_degrees: vec![2, 0],
        c_degrees: vec![],
        map_coefficients: vec![1.0; 2],
    };
    let geom = tianyau();
    // Unit Kähler form on Tian-Yau: μ(V) = (2 · J²) / 2 = J² .
    // From `slope_pairing_unit_kahler_tianyau` (in the in-module
    // test of polystability.rs), the unit-Kähler J² · H_0 pairing
    // is 27, so μ(V) = (2 · 27) / 2 = 27 in the unit-Kähler basis.
    let kahler = vec![1.0, 1.0];
    let mu_v_expected = 27.0; // (2 · 27) / 2 = 27
    let r = check_polystability(&bundle, &geom, &kahler, 2)
        .expect("polystability check must run");
    assert!(
        (r.mu_v - mu_v_expected).abs() < 1.0e-9,
        "μ(V) for V = O(2) ⊕ O(0) at unit Kähler should be \
         (c_1 · J²) / rank = (2 · 27) / 2 = 27; got {} \
         (mismatch suggests slope is not divided by rank — see S6 audit)",
        r.mu_v
    );
    // Audit: every recorded DestabilizingSubsheaf must have
    // slope = (c1 · J²) / rank — recompute and compare.
    for d in &r.destabilizing_subsheaves {
        // c_1(F) · J²  =  d.c1 · J²  =  d.c1[0] · 27  in unit Kähler.
        let c1_dot_j2: f64 = (d.c1[0] as f64) * 27.0;
        let expected = c1_dot_j2 / (d.rank as f64);
        assert!(
            (d.slope - expected).abs() < 1.0e-9,
            "destabiliser {:?}: recorded slope = {}, expected \
             c_1·J² / rank = {} (S6 audit: slope MUST be divided \
             by rank — Huybrechts-Lehn 2010 §1.2 Eq. 1.2.7)",
            d.origin,
            d.slope,
            expected
        );
    }
}

/// **S6.2**: For V = O(3) ⊕ O(-1) (the textbook rank-2,
/// c_1 = 2, BUT genuinely-unstable example), the rank-1 sub-line-
/// bundle O(3) has slope `3·J²` ≫ μ(V) = `J²` and MUST be reported
/// as a destabiliser. The polystability verdict MUST be unstable.
/// This catches a regression where a slope-by-rank bug would lower
/// the recorded slope of every sub-bundle uniformly and silently
/// turn this textbook-unstable bundle into a "polystable" one.
#[test]
fn test_unstable_O3_O_minus_1_correctly_flagged_in_decomposition() {
    let bundle = MonadBundle {
        b_degrees: vec![3, -1],
        c_degrees: vec![],
        map_coefficients: vec![1.0; 2],
    };
    let geom = tianyau();
    let kahler = vec![1.0, 1.0];
    let r = check_polystability(&bundle, &geom, &kahler, 2)
        .expect("polystability check must run on V = O(3) ⊕ O(-1)");

    // μ(V) = (c_1 · J²) / rank = (2 · 27) / 2 = 27.
    assert!((r.mu_v - 27.0).abs() < 1.0e-9, "μ(V) must be 27, got {}", r.mu_v);

    assert!(
        !r.is_polystable,
        "V = O(3) ⊕ O(-1) is NOT polystable: O(3) ⊂ V has slope \
         3·J² = 81 > 27 = μ(V). Verdict polystable means a slope \
         miscomputation has occurred. Destabilisers: {:?}",
        r.destabilizing_subsheaves
    );

    // The specific destabiliser must be identified: a rank-1
    // sub-line-bundle with c_1 ≥ 3 and slope ≥ 3 · 27 = 81.
    let has_o3_destab = r
        .destabilizing_subsheaves
        .iter()
        .any(|d: &DestabilizingSubsheaf| {
            d.rank == 1
                && d.c1[0] >= 3
                && (d.slope - 81.0).abs() < 1.0e-9
                && matches!(d.origin, SubsheafOrigin::LineBundle { .. })
        });
    assert!(
        has_o3_destab,
        "Expected a rank-1 destabiliser O(3) ⊂ V with c_1 = 3, \
         slope = 81 (= 3 · 27); got destabilisers: {:?}",
        r.destabilizing_subsheaves
    );
}

/// **S6.3**: At every recorded destabiliser, recompute slope by
/// hand and compare. Run this against the smoking-gun bundle from
/// the in-module suite (V = O(1)² ⊕ O(-1)², rank 4, c_1 = 0); the
/// rank-2 Λ²-wedge has c_1 = 2 (the wedge of the two O(1) summands)
/// and slope must be `(2 · 27) / 2 = 27`, NOT `2 · 27 = 54`.
#[test]
fn test_smoking_gun_lambda2_slope_divided_by_rank() {
    let bundle = MonadBundle {
        b_degrees: vec![1, 1, -1, -1],
        c_degrees: vec![],
        map_coefficients: vec![1.0; 4],
    };
    let geom = tianyau();
    let kahler = vec![1.0, 1.0];
    let r = check_polystability(&bundle, &geom, &kahler, 2)
        .expect("polystability check must succeed");
    // μ(V) = 0.
    assert!(r.mu_v.abs() < 1.0e-9);
    // The rank-2 Λ²-wedge destabiliser must exist with slope > 0
    // but MUST equal (c_1 · J²) / 2 (rank divisor!), not c_1 · J².
    let lambda2_destab = r
        .destabilizing_subsheaves
        .iter()
        .find(|d| {
            d.rank == 2
                && matches!(d.origin, SubsheafOrigin::SchurSubbundle { .. })
        })
        .expect("Λ²-wedge destabiliser must be present");
    // Compute c_1 · J^2 directly from the geometry's triple-intersection
    // numbers, so the test is independent of a particular (c_1[0], c_1[1])
    // distribution: μ(F) := (c_1 · J^2) / rank.  Huybrechts–Lehn 2010
    // §1.2 Eq. 1.2.7 — the rank divisor is the load-bearing line.
    let mut c1_dot_j2 = 0.0_f64;
    for a in 0..lambda2_destab.c1.len() {
        for b in 0..kahler.len() {
            for c in 0..kahler.len() {
                let mut va = vec![0i32; kahler.len()];
                va[a] = 1;
                let mut vb = vec![0i32; kahler.len()];
                vb[b] = 1;
                let mut vc = vec![0i32; kahler.len()];
                vc[c] = 1;
                let kappa = geom.triple_intersection(&va, &vb, &vc) as f64;
                c1_dot_j2 +=
                    (lambda2_destab.c1[a] as f64) * kahler[b] * kahler[c] * kappa;
            }
        }
    }
    let expected = c1_dot_j2 / (lambda2_destab.rank as f64);
    assert!(
        (lambda2_destab.slope - expected).abs() < 1.0e-9,
        "Λ²-wedge slope {} should equal (c_1 · J²) / rank = {}; \
         c_1 = {:?}, c_1 · J² = {}, rank = {} — \
         a value of c_1 · J² (no rank divisor) would be a bug — \
         see S6 audit / Huybrechts-Lehn 2010 §1.2 Eq. 1.2.7",
        lambda2_destab.slope,
        expected,
        lambda2_destab.c1,
        c1_dot_j2,
        lambda2_destab.rank
    );
}

/// **S6.4**: Same audit as S6.1 but on Schoen `Z/3 × Z/3`. The
/// rank-divisor convention must hold uniformly across geometries,
/// including the multi-Kähler case where the slope pairing
/// involves the full `triple_intersection` table rather than a
/// single integer. Construct V = O(2) ⊕ O(0), check μ(V) is the
/// rank-divided slope, and verify every destabiliser respects the
/// per-geometry triple-intersection / rank divisor convention.
#[test]
fn test_lambda2_subbundle_slope_divided_by_rank_on_schoen() {
    let bundle = MonadBundle {
        b_degrees: vec![2, 0],
        c_degrees: vec![],
        map_coefficients: vec![1.0; 2],
    };
    let geom = schoen();
    let kahler = vec![1.0, 1.0, 1.0];
    let r = check_polystability(&bundle, &geom, &kahler, 2)
        .expect("polystability on Schoen V = O(2) ⊕ O(0) must succeed");

    // μ(V) = (c_1 · J²) / rank = (2 · ⟨H_0² · (J_1 + J_2 + J_t)⟩) / 2.
    // On Schoen the per-cover triple intersections are
    // κ_{1,1,2} = 3, κ_{1,2,2} = 3, κ_{1,2,t} = 9 (DHOR 2006 Eq. 3.7);
    // contracting H_0² · (J_1 + J_2 + J_t) at unit Kähler in basis
    // (J_1, J_2, J_t) gives the same coefficient appearing in r.mu_v.
    // We don't hand-compute it here; we audit consistency: every
    // destabiliser's slope must equal (c1·J²)/rank under whatever
    // J² = ⟨[J]² · ·⟩ pairing the geometry exposes.
    for d in &r.destabilizing_subsheaves {
        // Reconstruct slope from c_1 vector and rank.
        let mut c1_dot_j2 = 0.0_f64;
        // Use the geometry's triple_intersection to recompute c1 · J²
        // at unit Kähler.
        let nf = d.c1.len();
        for a in 0..nf {
            if d.c1[a] == 0 {
                continue;
            }
            for b in 0..nf {
                for c in 0..nf {
                    let mut va = vec![0i32; nf];
                    va[a] = 1;
                    let mut vb = vec![0i32; nf];
                    vb[b] = 1;
                    let mut vc = vec![0i32; nf];
                    vc[c] = 1;
                    use crate::route34::fixed_locus::CicyGeometryTrait;
                    let kappa = CicyGeometryTrait::triple_intersection(
                        &geom, &va, &vb, &vc,
                    ) as f64;
                    c1_dot_j2 += (d.c1[a] as f64) * 1.0 * 1.0 * kappa;
                }
            }
        }
        let expected = c1_dot_j2 / (d.rank as f64);
        assert!(
            (d.slope - expected).abs() < 1.0e-7,
            "destabiliser {:?} on Schoen: recorded slope = {}, \
             expected (c_1 · J²) / rank = {} — rank divisor lost",
            d.origin,
            d.slope,
            expected
        );
    }
}
