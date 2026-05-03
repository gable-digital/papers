//! Integration-level tests for [`crate::route34::schoen_geometry`].
//!
//! Cross-checks the topological data and intersection numbers against
//! the published values in:
//!
//! * Schoen 1988, *Math. Z.* **197** 177 (DOI 10.1007/BF01215653)
//! * Donagi-He-Ovrut-Reinbacher 2006 JHEP **06** 039
//!   (DOI 10.1088/1126-6708/2006/06/039)
//! * Anderson-Gray-Lukas-Palti 2011, arXiv:1106.4804

use crate::route34::schoen_geometry::{
    SchoenGeometry, J1, J2, JT, PUBLISHED_C2_TM_INTEGRALS, PUBLISHED_TRIPLE_INTERSECTIONS,
};

/// All published triple-intersection numbers reproduce.
#[test]
fn published_triple_intersections_reproduce_to_machine_precision() {
    let g = SchoenGeometry::schoen_z3xz3();
    for (a, b, c, expected) in PUBLISHED_TRIPLE_INTERSECTIONS.iter() {
        let mut exp = [0u32; 3];
        exp[*a] += 1;
        exp[*b] += 1;
        exp[*c] += 1;
        let computed = g.intersection_number(exp).expect("intersection eval");
        assert_eq!(
            computed, *expected,
            "triple intersection at indices ({a},{b},{c}) = {expected}, got {computed}"
        );
    }
}

/// `c_2(TM) ∧ J_a` integrals reproduce the DHOR 2006 §3 Eq. 3.13–3.15.
#[test]
fn c2_tm_integrals_match_dhor_2006() {
    let g = SchoenGeometry::schoen_z3xz3();
    for (a, &expected) in PUBLISHED_C2_TM_INTEGRALS.iter().enumerate() {
        let computed = g.c2_tm_dot_j(a).expect("c2 dot J eval");
        assert_eq!(
            computed, expected,
            "c_2(TM) · J_{} = {expected}, got {computed}",
            a
        );
    }
}

/// Bianchi anomaly residual vanishes when c_2(V_v) + c_2(V_h) = c_2(TM).
#[test]
fn bianchi_residual_zero_at_anomaly_cancellation() {
    let g = SchoenGeometry::schoen_z3xz3();
    let target = PUBLISHED_C2_TM_INTEGRALS;
    let visible = vec![target[0] / 2, target[1] / 2, target[2] / 2];
    let hidden = vec![
        target[0] - visible[0],
        target[1] - visible[1],
        target[2] - visible[2],
    ];
    let r = g.bianchi_residual(&visible, &hidden).expect("bianchi");
    assert!(r < 1e-12, "Bianchi residual at exact cancellation = {r}");
}

/// Calabi-Yau condition `(3, 0, 1) + (0, 3, 1) = (3, 3, 2) = (n+1)`.
#[test]
fn cy_condition_holds() {
    let g = SchoenGeometry::schoen_z3xz3();
    assert!(g.satisfies_calabi_yau_condition());
    assert_eq!(g.h11_upstairs, 19);
    assert_eq!(g.h21_upstairs, 19);
    assert_eq!(g.chi_upstairs, 0);
    assert_eq!(g.h11_quotient, 3);
    assert_eq!(g.h21_quotient, 3);
    assert_eq!(g.chi_quotient(), 0);
    assert_eq!(g.quotient_order, 9);
}

/// `slope_pair` returns positive numerator and denominator for a positive
/// polarisation against an effective line bundle.
#[test]
fn slope_pair_positive_polarisation() {
    let g = SchoenGeometry::schoen_z3xz3();
    let polarisation = [1, 1, 1];
    // The line bundle `O(0, 0, 1)` has c_1 = J_t.
    let (num, denom) = g.slope_pair([0, 0, 1], polarisation).expect("slope");
    // c_1 · J^2 = J_t · (J_1 + J_2 + J_t)^2 = J_t (J_1^2 + 2 J_1 J_2 + ...)
    //   = 0 + 2 · ∫ J_1 J_2 J_t = 2 · 9 = 18 (the J_1^2 J_t and J_2^2 J_t
    //   terms vanish by truncation).
    assert_eq!(num, 18, "c_1(O(0,0,1)) · J^2 should be 18; got {num}");
    // J^3 = (J_1+J_2+J_t)^3 = expand. Most terms truncated. Non-zero
    // contributions from J_1^2 J_2 (3 ways), J_1 J_2^2 (3 ways), and
    // 6 · J_1 J_2 J_t.
    //   = 3·3 + 3·3 + 6·9 = 9 + 9 + 54 = 72.
    assert_eq!(denom, 72, "J^3 with polarisation (1,1,1) should be 72; got {denom}");
}

/// Self-mirror structure: `h^{1,1} = h^{2,1}` on cover.
#[test]
fn self_mirror_invariant() {
    let g = SchoenGeometry::schoen_z3xz3();
    assert_eq!(g.h11_upstairs, g.h21_upstairs);
    assert_eq!(2 * (g.h11_upstairs as i32 - g.h21_upstairs as i32), g.chi_upstairs);
}

/// Sanity: J_1 J_2 J_t = 9 (computed from `[X̃] = (3 H_1 + H_t)(3 H_2 + H_t)`).
#[test]
fn intersection_j1_j2_jt_is_nine() {
    let g = SchoenGeometry::schoen_z3xz3();
    // Using triple_intersection: a = (1, 0, 0), b = (0, 1, 0), c = (0, 0, 1).
    let (j1v, j2v, jtv) = (vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]);
    let v = g.triple_intersection(&j1v, &j2v, &jtv).expect("triple");
    assert_eq!(v, 9);
}

#[test]
fn intersection_j1_squared_j2_is_three() {
    let g = SchoenGeometry::schoen_z3xz3();
    let v = g.intersection_number([2, 1, 0]).expect("eval");
    assert_eq!(v, 3);
}

#[test]
fn intersection_j2_squared_j1_is_three() {
    let g = SchoenGeometry::schoen_z3xz3();
    let v = g.intersection_number([1, 2, 0]).expect("eval");
    assert_eq!(v, 3);
}

/// Indices J1, J2, JT are 0, 1, 2.
#[test]
fn kahler_indices_are_well_defined() {
    assert_eq!(J1, 0);
    assert_eq!(J2, 1);
    assert_eq!(JT, 2);
}
