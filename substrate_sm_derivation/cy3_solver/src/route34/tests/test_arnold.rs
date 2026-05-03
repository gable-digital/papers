//! Integration tests for [`crate::route34::arnold_normal_form`].
//!
//! These exercise the published Arnold normal forms (codimensions
//! up to 7) and verify that the classifier returns the correct ADE
//! type AND the correct Milnor number AND the published admissible-
//! wavenumber set for each.

use crate::route34::arnold_normal_form::{
    admissible_wavenumber_set, classify_singularity, classify_singularity_batch,
    corank, milnor_number, ArnoldType, GermError, Sign, SmoothFunctionGerm,
};

fn a_k_normal_form(k: u32) -> SmoothFunctionGerm {
    // V(x) = x^(k+1)
    let mut g = SmoothFunctionGerm::zeros(1, (k + 1) as usize).unwrap();
    g.set_coeff(&[k + 1], 1.0).unwrap();
    g
}

fn d4_plus_normal_form() -> SmoothFunctionGerm {
    // V(x, y) = x^3 + y^3 (hyperbolic umbilic)
    let mut g = SmoothFunctionGerm::zeros(2, 3).unwrap();
    g.set_coeff(&[3, 0], 1.0).unwrap();
    g.set_coeff(&[0, 3], 1.0).unwrap();
    g
}

fn d4_minus_normal_form() -> SmoothFunctionGerm {
    // V(x, y) = x^3 - 3 x y^2 (elliptic umbilic)
    let mut g = SmoothFunctionGerm::zeros(2, 3).unwrap();
    g.set_coeff(&[3, 0], 1.0).unwrap();
    g.set_coeff(&[1, 2], -3.0).unwrap();
    g
}

fn e6_normal_form() -> SmoothFunctionGerm {
    // V = x^3 + y^4
    let mut g = SmoothFunctionGerm::zeros(2, 4).unwrap();
    g.set_coeff(&[3, 0], 1.0).unwrap();
    g.set_coeff(&[0, 4], 1.0).unwrap();
    g
}

fn e7_normal_form() -> SmoothFunctionGerm {
    // V = x^3 + x y^3
    let mut g = SmoothFunctionGerm::zeros(2, 4).unwrap();
    g.set_coeff(&[3, 0], 1.0).unwrap();
    g.set_coeff(&[1, 3], 1.0).unwrap();
    g
}

fn e8_normal_form() -> SmoothFunctionGerm {
    // V = x^3 + y^5
    let mut g = SmoothFunctionGerm::zeros(2, 5).unwrap();
    g.set_coeff(&[3, 0], 1.0).unwrap();
    g.set_coeff(&[0, 5], 1.0).unwrap();
    g
}

#[test]
fn integration_a2_through_a5_classify_with_correct_milnor() {
    for k in 2..=5u32 {
        let g = a_k_normal_form(k);
        let t = classify_singularity(&g).unwrap();
        assert_eq!(t, ArnoldType::A(k), "wrong ADE label for x^(k+1)");
        let mu = milnor_number(&g).unwrap();
        assert_eq!(mu, k as usize, "wrong Milnor number for A_{}", k);
    }
}

#[test]
fn integration_d4_plus_minus_distinguishes_signs() {
    let plus = d4_plus_normal_form();
    let minus = d4_minus_normal_form();
    let tp = classify_singularity(&plus).unwrap();
    let tm = classify_singularity(&minus).unwrap();
    match (tp, tm) {
        (ArnoldType::D(4, Sign::Hyperbolic), ArnoldType::D(4, Sign::Elliptic)) => {}
        _ => panic!("expected D_4^+ and D_4^-, got {:?} and {:?}", tp, tm),
    }
}

#[test]
fn integration_e6_e7_e8_classify() {
    assert_eq!(classify_singularity(&e6_normal_form()).unwrap(), ArnoldType::E6);
    assert_eq!(classify_singularity(&e7_normal_form()).unwrap(), ArnoldType::E7);
    assert_eq!(classify_singularity(&e8_normal_form()).unwrap(), ArnoldType::E8);
}

#[test]
fn integration_admissible_wavenumbers_match_bourbaki_coxeter_exponents() {
    // Bourbaki Ch VI §1.11; Humphreys §3.18 / Table 3.1.
    // Admissible set is (Coxeter exponents) ∪ {Coxeter number h}.
    // A_n: exponents 1..=n, h = n+1.
    assert_eq!(admissible_wavenumber_set(ArnoldType::A(2)), vec![1, 2, 3]);
    assert_eq!(
        admissible_wavenumber_set(ArnoldType::A(5)),
        vec![1, 2, 3, 4, 5, 6]
    );
    // E_6: exponents {1,4,5,7,8,11}, h = 12.
    assert_eq!(
        admissible_wavenumber_set(ArnoldType::E6),
        vec![1, 4, 5, 7, 8, 11, 12]
    );
    // E_7: exponents {1,5,7,9,11,13,17}, h = 18.
    assert_eq!(
        admissible_wavenumber_set(ArnoldType::E7),
        vec![1, 5, 7, 9, 11, 13, 17, 18]
    );
    // E_8: exponents {1,7,11,13,17,19,23,29}, h = 30.
    assert_eq!(
        admissible_wavenumber_set(ArnoldType::E8),
        vec![1, 7, 11, 13, 17, 19, 23, 29, 30]
    );
}

#[test]
fn integration_unfolding_a3_cusp_is_morse_at_nonzero_a() {
    // V(x) = x^4 + a x^2: Hessian 2a at origin. a != 0 ⇒ Morse.
    for a in [-2.0, -0.5, 0.5, 2.0] {
        let mut g = SmoothFunctionGerm::zeros(1, 4).unwrap();
        g.set_coeff(&[4], 1.0).unwrap();
        g.set_coeff(&[2], a).unwrap();
        let t = classify_singularity(&g).unwrap();
        assert_eq!(t, ArnoldType::MorseRegular, "x^4 + {} x^2 should be Morse", a);
    }
    // a = 0 ⇒ pure cusp x^4 ⇒ A_3.
    let mut g0 = SmoothFunctionGerm::zeros(1, 4).unwrap();
    g0.set_coeff(&[4], 1.0).unwrap();
    let t0 = classify_singularity(&g0).unwrap();
    assert_eq!(t0, ArnoldType::A(3));
}

#[test]
fn integration_d4_hyperbolic_unfolding_a_xy_term() {
    // V(x, y) = x^3 + y^3 + a x y. Hessian off-diagonal a, eigenvalues
    // ±a. a = 0 ⇒ D_4^+; a != 0 ⇒ Morse.
    for a in [-1.0, -0.1, 0.1, 1.0] {
        let mut g = SmoothFunctionGerm::zeros(2, 3).unwrap();
        g.set_coeff(&[3, 0], 1.0).unwrap();
        g.set_coeff(&[0, 3], 1.0).unwrap();
        g.set_coeff(&[1, 1], a).unwrap();
        let t = classify_singularity(&g).unwrap();
        assert_eq!(
            t,
            ArnoldType::MorseRegular,
            "x^3+y^3+{} xy should be Morse at origin",
            a
        );
    }
    let pure = d4_plus_normal_form();
    assert!(matches!(
        classify_singularity(&pure).unwrap(),
        ArnoldType::D(4, Sign::Hyperbolic)
    ));
}

#[test]
fn integration_corank_recovers_arnold_table_codimensions() {
    // Per Arnold 1974 §3 / AGZV Vol I §15.1:
    //   A_k     codim k - 1, corank 1
    //   D_4     codim 3,     corank 2
    //   E_6     codim 5,     corank 2
    //   E_7     codim 6,     corank 2
    //   E_8     codim 7,     corank 2
    assert_eq!(corank(&a_k_normal_form(2)).unwrap(), 1);
    assert_eq!(corank(&a_k_normal_form(5)).unwrap(), 1);
    assert_eq!(corank(&d4_plus_normal_form()).unwrap(), 2);
    assert_eq!(corank(&e6_normal_form()).unwrap(), 2);
    assert_eq!(corank(&e7_normal_form()).unwrap(), 2);
    assert_eq!(corank(&e8_normal_form()).unwrap(), 2);
}

#[test]
fn integration_batch_classification_returns_consistent_with_serial() {
    let germs = vec![
        a_k_normal_form(2),
        a_k_normal_form(3),
        a_k_normal_form(4),
        a_k_normal_form(5),
        d4_plus_normal_form(),
        d4_minus_normal_form(),
        e6_normal_form(),
        e7_normal_form(),
        e8_normal_form(),
    ];
    let parallel = classify_singularity_batch(&germs);
    for (i, g) in germs.iter().enumerate() {
        let serial = classify_singularity(g).unwrap();
        assert_eq!(*parallel[i].as_ref().unwrap(), serial);
    }
}

#[test]
fn integration_milnor_higher_returns_err() {
    // A flat-to-high-order germ with no leading term within the
    // Taylor truncation is "Higher" — Milnor number is undefined
    // and should error.
    let g = SmoothFunctionGerm::zeros(2, 4).unwrap();
    let mu = milnor_number(&g);
    assert!(matches!(mu, Err(GermError::NumericalFailure(_))));
}
