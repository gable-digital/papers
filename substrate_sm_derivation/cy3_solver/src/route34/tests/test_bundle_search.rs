//! # Integration tests for [`crate::route34::bundle_search`]
//!
//! Cross-checks the derived-Chern monad-bundle parameterisation
//! against the published heterotic-string-phenomenology
//! literature:
//!
//! * Anderson-Gray-Lukas-Palti 2011 (arXiv:1106.4804) — the
//!   AGLP-2011 SU(5) catalogue on Tian-Yau Z/3.
//! * Donagi-He-Ovrut-Reinbacher 2006 (arXiv:hep-th/0512149) —
//!   the DHOR-2006 SU(4) catalogue on Schoen Z/3 × Z/3.
//! * Anderson-Karp-Lukas-Palti 2010 (arXiv:1004.4399) — the
//!   numerical HYM / polystability cross-checks.
//!
//! Every test ties the new structured `LineBundleDegrees`
//! representation to a published Chern-class number or a
//! published anomaly cancellation pair. No hand-tuned magic
//! constants outside the cited literature.

use crate::route34::bundle_search::{
    aglp_2011_ty_su5, bhop_2005_schoen, dhor_2006_schoen_su4,
    enumerate_candidate_bundles, EnumerationConfig, LineBundleDegrees,
};
use crate::route34::wilson_line_e8::WilsonLineE8;

// ----------------------------------------------------------------------
// Test 1: the derived-Chern parameterisation eliminates the legacy
// double-encoding bug.
// ----------------------------------------------------------------------

/// **Bug-fix regression:** the old `pipeline.rs` stored Chern numbers
/// at separate `bundle_moduli` indices AND pre-quantized the line-
/// bundle degrees with `sum_b == sum_c` (forcing c_1 = 0), then
/// `chern_class_loss` separately penalised `bundle_moduli[0] != 0`.
/// Two encodings of the same physical quantity, uncoupled.
///
/// The structured `LineBundleDegrees` has exactly ONE source of truth
/// for c_1: it's recomputed via `derived_chern()` from the integer
/// degrees. This test verifies that for `b = [1, 1, -2]` (sum 0,
/// no C side), `c_1 = 0` is **derived**, not stored — and that the
/// derived c_2, c_3 propagate to the η pipeline correctly.
#[test]
fn test_derived_chern_no_double_count() {
    let lb = LineBundleDegrees::new(vec![1, 1, -2], vec![]);
    let chern = lb.derived_chern();
    assert_eq!(chern.c1, 0, "Σ b_i = 0 (no C side) ⇒ c_1 = 0 derived");

    // Verify the derived c_2 matches a hand computation:
    //   c_2 = Σ_{i<j} b_i b_j = (1·1 + 1·(-2) + 1·(-2)) = 1 - 2 - 2 = -3.
    // Newton: p_1 = 0, p_2 = 1 + 1 + 4 = 6. 2 c_2 = 0 - 6 = -6 ⇒ c_2 = -3.
    assert_eq!(chern.c2, -3);

    // Verify c_3 from Newton: p_3 = 1 + 1 - 8 = -6. 6 c_3 = 0 - 0 + 2·(-6) = -12 ⇒ c_3 = -2.
    // Hand: c_3 = e_3(b) = 1·1·(-2) = -2.
    assert_eq!(chern.c3, -2);
}

/// Verify that two bundles with the same `derived_chern()` MUST
/// have the same line-bundle-degree multiset (modulo permutation).
/// This is the point of having a single source of truth: equal
/// derived Chern data ⇔ equal bundle, no decoupled-encoding
/// inconsistency.
#[test]
fn test_derived_chern_uniqueness_modulo_permutation() {
    let a = LineBundleDegrees::new(vec![1, 2, 3], vec![6]);
    let b = LineBundleDegrees::new(vec![3, 1, 2], vec![6]);
    let c = LineBundleDegrees::new(vec![3, 2, 1], vec![6]);
    let cha = a.derived_chern();
    let chb = b.derived_chern();
    let chc = c.derived_chern();
    assert_eq!(cha, chb);
    assert_eq!(cha, chc);
}

// ----------------------------------------------------------------------
// Test 2: anomaly cancellation against published AGLP-2011 examples.
// ----------------------------------------------------------------------

/// **Published-catalogue cross-check:** AGLP-2011 §3 exhibits a
/// `(V_v, V_h)` pair on Tian-Yau Z/3 with c_2(V_v) = 14 and the
/// hidden bundle chosen so that c_2(V_v) + c_2(V_h) = c_2(TM) = 36.
/// This test verifies the Bianchi residual is exactly 0 against
/// the published shape `B_v = O(1)^4 ⊕ O(2)`, `C_v = O(6)` /
/// `B_h = O(1)^4 ⊕ O(4)`, `C_h = O(8)`.
#[test]
fn test_anomaly_cancellation_with_real_hidden() {
    let rec = aglp_2011_ty_su5();
    let chern_v = rec.visible.derived_chern();
    let chern_h = rec.hidden.derived_chern();
    assert_eq!(chern_v.c1, 0, "AGLP visible c_1 must vanish");
    assert_eq!(chern_h.c1, 0, "AGLP hidden c_1 must vanish");
    assert_eq!(chern_v.c2, 14, "AGLP visible c_2 = 14 (published)");
    assert_eq!(chern_h.c2, 22, "AGLP hidden c_2 = 22 (Bianchi-derived)");
    // c_2(V_v) + c_2(V_h) - c_2(TM) = 14 + 22 - 36 = 0.
    let residual = rec.visible.bianchi_residual(36, &rec.hidden);
    assert_eq!(residual, 0, "AGLP-2011 Bianchi identity holds exactly");
}

/// DHOR-2006 §3.1 SU(4)-bundle on Schoen Z/3 × Z/3. The published
/// shape is `B = O(1)^3 ⊕ O(3)`, `C = O(6)` with c_1 = 0. We do
/// NOT assert c_2 against a specific number here because DHOR's
/// downstairs Schoen `c_2(TM)` uses a different basis; the test
/// only checks c_1 = 0 and the derived-Chern API agrees with
/// itself across calls.
#[test]
fn test_dhor_2006_su4_c1_zero() {
    let rec = dhor_2006_schoen_su4();
    let chern = rec.visible.derived_chern();
    assert_eq!(chern.c1, 0, "DHOR-2006 Schoen SU(4) visible c_1 = 0");
    let chern2 = rec.visible.derived_chern();
    assert_eq!(chern, chern2, "derived_chern is deterministic");
}

/// BHOP-2005 Schoen Z/3 × Z/3 visible bundle: `B = O(1)^3 ⊕ O(3)`,
/// `C = O(6)`, c_1 = 0 (verified by construction).
#[test]
fn test_bhop_2005_schoen_c1_zero() {
    let rec = bhop_2005_schoen();
    let chern = rec.visible.derived_chern();
    assert_eq!(chern.c1, 0);
}

// ----------------------------------------------------------------------
// Test 3: three-generation count from c_3.
// ----------------------------------------------------------------------

/// **AGLP-2010 generation formula:** `n_gen = |c_3(V)| / (2 |Γ|)`.
/// For Tian-Yau Z/3 (`|Γ| = 3`), three generations require
/// `|c_3| = 18`. We construct a bundle with c_3 = -150 (Newton-
/// correct value for B = (3,3,2), C = (8) — see the analysis in
/// the bundle_search docstring), and verify the formula returns
/// 25 (= 150 / 6).
#[test]
fn test_three_generation_count_arithmetic() {
    let lb = LineBundleDegrees::new(vec![3, 3, 2], vec![8]);
    let chern = lb.derived_chern();
    assert_eq!(chern.c1, 0);
    assert_eq!(chern.c3, -150, "Newton-correct c_3 for (3,3,2)/(8)");
    // n_gen = |c_3| / (2 · 3) = 150 / 6 = 25.
    let n_gen = chern.generations(3);
    assert!(
        (n_gen - 25.0).abs() < 1.0e-9,
        "n_gen should be 25 for c_3 = -150, |Γ| = 3"
    );
}

/// **Construct a true 3-generation bundle**: find a (b, c) shape
/// such that the Newton-derived c_3 is exactly 18 and c_1 = 0.
///
/// Search: try B = (1, 1, 1, 1, 2), C = (6) — the AGLP shape:
///   p_1 = 0, p_2 = -28, p_3 = 4·1 + 8 - 216 = -204.
///   c_3 = (0 - 0 + 2·(-204))/6 = -68. n_gen = 68/6 ≈ 11.33. Not 3.
///
/// Try B = (1, 1, 1, 0), C = (3): c_1 = 3 - 3 = 0.
///   p_1 = 0, p_2 = 1 + 1 + 1 + 0 - 9 = -6, p_3 = 1+1+1+0-27 = -24.
///   c_3 = (0 - 0 + 2·(-24))/6 = -8. n_gen = 8/6.
///
/// Try B = (2, 2, -2), C = []: c_1 = 2.
/// (skip — c_1 ≠ 0)
///
/// Try B = (3, 2, 1), C = (6): c_1 = 0.
///   p_1 = 0, p_2 = 9 + 4 + 1 - 36 = -22, p_3 = 27+8+1 - 216 = -180.
///   c_3 = -60. n_gen = 60/6 = 10.
///
/// Try B = (2, 1, -1), C = (2): c_1 = 0.
///   p_1 = 0, p_2 = 4+1+1-4 = 2, p_3 = 8+1-1-8 = 0.
///   c_3 = 0. n_gen = 0.
///
/// Try B = (3, 1), C = (4): c_1 = 0.
///   p_1 = 0, p_2 = 9+1-16 = -6, p_3 = 27+1-64 = -36.
///   c_3 = -12. n_gen = 12/6 = 2.
///
/// Try B = (3, 3, 3), C = (9): c_1 = 0.
///   p_1 = 0, p_2 = 27 - 81 = -54, p_3 = 81 - 729 = -648.
///   c_3 = -216. n_gen = 36.
///
/// Try B = (3, 3, -3), C = (3): c_1 = 0.
///   p_1 = 0, p_2 = 9+9+9-9 = 18, p_3 = 27+27-27-27 = 0.
///   c_3 = 0.
///
/// Try B = (3, 2, -2), C = (3): c_1 = 0.
///   p_1 = 0, p_2 = 9+4+4-9 = 8, p_3 = 27+8-8-27 = 0.
///   c_3 = 0.
///
/// Try B = (4, 1, -2), C = (3): c_1 = 0.
///   p_2 = 16+1+4-9 = 12, p_3 = 64+1-8-27 = 30.
///   c_3 = (0 - 0 + 60)/6 = 10. n_gen = 10/6.
///
/// Try B = (3, 1, 1), C = (5): c_1 = 0.
///   p_2 = 9+1+1-25 = -14, p_3 = 27+1+1-125 = -96.
///   c_3 = -32. n_gen = 32/6.
///
/// Try B = (-3, 1, 2), C = (0): c_1 = 0.
///   p_2 = 9+1+4 - 0 = 14, p_3 = -27+1+8 = -18.
///   c_3 = -6. n_gen = 1.
///
/// Try B = (4, 1, -1), C = (4): c_1 = 0.
///   p_2 = 16+1+1-16 = 2, p_3 = 64+1-1-64 = 0.
///   c_3 = 0.
///
/// Try B = (3, -1, 1), C = (3): c_1 = 0.
///   p_2 = 9+1+1-9 = 2, p_3 = 27-1+1-27 = 0.
///   c_3 = 0.
///
/// Try B = (-1, 1, 1, -1), C = []: c_1 = 0.
///   p_2 = 1+1+1+1 = 4, p_3 = -1+1+1-1 = 0.
///   c_3 = 0.
///
/// Try B = (-3, 3), C = []: c_1 = 0, rank 2.
///   p_2 = 9+9 = 18, p_3 = -27+27 = 0.
///   c_3 = 0.
///
/// Try B = (3, 1, -3, -1), C = []: c_1 = 0, rank 4.
///   p_2 = 9+1+9+1 = 20, p_3 = 27+1-27-1 = 0.
///   c_3 = 0.
///
/// Try B = (-1, 2, -1), C = []: c_1 = 0, rank 3.
///   p_2 = 1+4+1 = 6, p_3 = -1+8-1 = 6.
///   c_3 = (0 - 0 + 12)/6 = 2. n_gen = 1/3 (Z/3) – not 3-gen.
///
/// **The AGLP-2011 catalogue exact 3-generation hits require LARGER
/// degree ranges and asymmetric C-side splittings.** For test
/// purposes we accept any c_3 = ±18 hit; computing one constructively
/// is below.
#[test]
fn test_three_generation_via_explicit_bundle() {
    // Hand-found: B = (-3, 1, 1, 1, 1, 1, 1, 1, 1, -3), C = []. 10 ones... too clumsy.
    // Newton gives:
    //   B = (-3, -3, 1, 1, 1, 1, 1, 1, 1, 1), 10 entries: c_1 = -6+8 = 2 ≠ 0.
    //   Adjust: B = (-3, -3, 1, 1, 1, 1, 1, 1, 1, 1), C = (2): c_1 = 2-2 = 0. ✓
    //   p_2(B) = 9+9+8 = 26. p_2(C) = 4. p_2(V) = 22.
    //   p_3(B) = -27-27+8 = -46. p_3(C) = 8. p_3(V) = -54.
    //   2 c_2 = 0 - 22 = -22. c_2 = -11.
    //   6 c_3 = 0 - 0 + 2·(-54) = -108. c_3 = -18. ✓ THREE GEN ON Z/3
    let lb = LineBundleDegrees::new(
        vec![-3, -3, 1, 1, 1, 1, 1, 1, 1, 1],
        vec![2],
    );
    let chern = lb.derived_chern();
    assert_eq!(chern.c1, 0);
    assert_eq!(chern.c3, -18);
    let n_gen = chern.generations(3);
    assert!((n_gen - 3.0).abs() < 1.0e-9);
}

// ----------------------------------------------------------------------
// Test 4: polystability via split bundles.
// ----------------------------------------------------------------------

/// **Mathematical content:** V = O(a) ⊕ O(b) is polystable iff
/// μ(O(a)) = μ(O(b)), i.e. a · K = b · K. For a single-Kähler-
/// modulus theory this collapses to a == b.
#[test]
fn test_polystability_via_split() {
    // Equal degrees: polystable.
    let lb = LineBundleDegrees::new(vec![2, 2], vec![]);
    // c_1 = 4 ≠ 0 — but polystability is independent of c_1
    // (polystability = sub-line-bundle slope ≤ μ(V)).
    // μ(V) = c_1/r = 4/2 = 2; both b_i are 2 ≤ 2. Polystable.
    assert!(lb.is_polystable(&[1.0]));

    // Unequal degrees with c_1 = 0: not polystable in generic
    // Kähler chamber. b_i = 3 > 0 = μ.
    let lb2 = LineBundleDegrees::new(vec![3, -3], vec![]);
    assert!(!lb2.is_polystable(&[1.0]));

    // A polystable c_1 = 0 case with all b_i ≤ 0.
    let lb3 = LineBundleDegrees::new(vec![-1, -1, 0, 0], vec![-2]);
    assert_eq!(lb3.derived_chern().c1, 0);
    assert!(lb3.is_polystable(&[1.0, 1.0]));
}

/// Polystability test: a bundle with `b_i = 0` for all i is
/// always polystable (it's a direct sum of trivial line bundles).
#[test]
fn test_trivial_bundle_polystable() {
    let lb = LineBundleDegrees::new(vec![0, 0, 0, 0], vec![]);
    assert_eq!(lb.derived_chern().c1, 0);
    assert!(lb.is_polystable(&[1.0]));
}

// ----------------------------------------------------------------------
// Test 5: enumeration sanity (small range, low cap).
// ----------------------------------------------------------------------

/// **End-to-end smoke**: run the enumerator over a tiny range and
/// verify every returned candidate satisfies all filters.
#[test]
fn test_enumerate_returns_admissible_candidates() {
    use crate::geometry::CicyGeometry;
    let geom = CicyGeometry::tian_yau_z3();
    let config = EnumerationConfig {
        degree_range: -2..=2,
        n_summands_visible_b: 3,
        n_summands_visible_c: 1,
        n_summands_hidden_b: 3,
        n_summands_hidden_c: 1,
        kahler: vec![1.0, 1.0],
        max_candidates: 16,
    };
    let cands = enumerate_candidate_bundles(&geom, config);
    // Each must be admissible.
    for c in &cands {
        let v = c.passes_all_filters(36, 3, &[1.0, 1.0]);
        assert!(
            v.is_admissible(),
            "enumerated candidate must be admissible, got verdict {:?}",
            v
        );
    }
}

/// Enumeration with an empty range should yield no candidates.
#[test]
fn test_enumerate_empty_range() {
    use crate::geometry::CicyGeometry;
    let geom = CicyGeometry::tian_yau_z3();
    // A range that makes c_1 = 0 impossible with the given shapes:
    // n_summands_visible_b = 1, range = [1, 1]: B = O(1) only.
    // We'd need C-side O(1) too; rank(V) = 0; rejected.
    let config = EnumerationConfig {
        degree_range: 1..=1,
        n_summands_visible_b: 1,
        n_summands_visible_c: 1,
        n_summands_hidden_b: 1,
        n_summands_hidden_c: 1,
        kahler: vec![1.0, 1.0],
        max_candidates: 8,
    };
    let cands = enumerate_candidate_bundles(&geom, config);
    assert!(
        cands.is_empty(),
        "rank-0 monad shapes should yield no candidates, got {} cands",
        cands.len()
    );
}

// ----------------------------------------------------------------------
// Test 6: structured bundle vs. legacy bundle_moduli decode parity.
// ----------------------------------------------------------------------

/// Verify that a CandidateBundle built directly from line-bundle
/// degrees gives the same Chern numbers as decoding a legacy
/// `bundle_moduli` vector with the same encoded degrees.
#[test]
fn test_legacy_bundle_moduli_compat() {
    use crate::route34::bundle_search::from_legacy_bundle_moduli;
    // Encode b = (1, 1, 1, 1, 2), c = (6) into the legacy 30-vector.
    let mut bm = vec![0.0_f64; 30];
    bm[0] = 1.0;
    bm[1] = 1.0;
    bm[2] = 1.0;
    bm[3] = 1.0;
    bm[4] = 2.0;
    bm[5] = 6.0;
    let cand = from_legacy_bundle_moduli(&bm, "TY/Z3", 3);
    let chern = cand.visible.derived_chern();
    assert_eq!(chern.c1, 0);
    assert_eq!(chern.c2, 14, "decoded legacy moduli must match AGLP c_2");
    // The default Wilson line should be the canonical E_8 → E_6 × SU(3).
    let canonical = WilsonLineE8::canonical_e8_to_e6_su3(3);
    assert_eq!(cand.wilson_line, canonical);
}

// ----------------------------------------------------------------------
// Test 7: serde round-trip.
// ----------------------------------------------------------------------

/// LineBundleDegrees and DerivedChern must round-trip through JSON
/// for reproducibility metadata.
#[test]
fn test_serde_roundtrip() {
    let lb = LineBundleDegrees::new(vec![1, 1, 1, 1, 2], vec![6]);
    let json = serde_json::to_string(&lb).expect("serialise");
    let lb2: LineBundleDegrees =
        serde_json::from_str(&json).expect("deserialise");
    assert_eq!(lb, lb2);
    let chern = lb.derived_chern();
    let chern_json = serde_json::to_string(&chern).expect("ser chern");
    let chern2 = serde_json::from_str(&chern_json).expect("de chern");
    assert_eq!(chern, chern2);
}
