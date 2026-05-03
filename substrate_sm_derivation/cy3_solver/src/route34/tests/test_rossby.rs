//! Integration tests for [`crate::route34::rossby_polar`]: Saturn /
//! Jupiter polar Lyapunov-functional germ assembly + interaction with
//! the Arnold classifier.

use crate::route34::arnold_normal_form::{ArnoldType, classify_singularity};
use crate::route34::rossby_polar::{
    arnold_type_cutoff, linearised_lyapunov, predict_wavenumber_set,
    published_jupiter_north_polar, published_jupiter_south_polar,
    published_saturn_polar, PolarBasicState,
};
use crate::route34::CyclicSubgroup;

fn at_critical_boundary(s: &mut PolarBasicState) {
    let l_r = s.rossby_radius_published("auto");
    s.jet_shear = s.potential_vorticity_gradient * l_r * l_r;
}

#[test]
fn integration_saturn_parameters_match_published_cassini_values() {
    let s = published_saturn_polar();
    // Period 38018 s ⇒ ω = 2π/38018 ≈ 1.6526e-4 rad/s.
    let expected_omega = 2.0 * std::f64::consts::PI / 38018.0;
    assert!((s.planet_omega - expected_omega).abs() < 1e-12);
    // Equatorial radius 60,268 km exactly.
    assert!((s.planet_radius - 60_268_000.0).abs() < 1.0);
    // Hexagon at 78°N.
    let expected_lat = 78.0_f64.to_radians();
    assert!((s.jet_latitude - expected_lat).abs() < 1e-12);
    // β at 78°N positive.
    assert!(s.potential_vorticity_gradient > 0.0);
    // Jet shear ~1 m/s/km = 1e-3 1/s.
    assert!((s.jet_shear - 1.0e-3).abs() < 1e-12);
}

#[test]
fn integration_jupiter_parameters_match_published_juno_values() {
    let n = published_jupiter_north_polar();
    let s = published_jupiter_south_polar();
    // Same rotation period 35729.71 s.
    let expected_omega = 2.0 * std::f64::consts::PI / 35729.71;
    assert!((n.planet_omega - expected_omega).abs() < 1e-12);
    assert!((s.planet_omega - expected_omega).abs() < 1e-12);
    // Equatorial radius 71,492 km.
    assert!((n.planet_radius - 71_492_000.0).abs() < 1.0);
    // Both polar regions at 85° latitude (north + south).
    assert!((n.jet_latitude - 85.0_f64.to_radians()).abs() < 1e-12);
    assert!((s.jet_latitude - 85.0_f64.to_radians()).abs() < 1e-12);
    // North pole has stronger jet shear than south pole.
    assert!(n.jet_shear > s.jet_shear);
}

#[test]
fn integration_lyapunov_at_critical_boundary_has_vanishing_quadratic() {
    let mut state = published_saturn_polar();
    at_critical_boundary(&mut state);
    let g = linearised_lyapunov(&state, 2).unwrap();
    let cx2 = g.coeff(&[2, 0]).unwrap();
    let cy2 = g.coeff(&[0, 2]).unwrap();
    assert!(cx2.abs() < 1e-10);
    assert!(cy2.abs() < 1e-10);
}

#[test]
fn integration_lyapunov_classifies_at_polar_critical_boundary() {
    // At the critical boundary the cubic part dominates; we should
    // land in the D_4 family (or its degenerate completion). The
    // classifier returns one of D_4^±, E_6, E_7, E_8, or Higher.
    let mut state = published_saturn_polar();
    at_critical_boundary(&mut state);
    let g = linearised_lyapunov(&state, 2).unwrap();
    let t = classify_singularity(&g).unwrap();
    match t {
        ArnoldType::D(_, _)
        | ArnoldType::E6
        | ArnoldType::E7
        | ArnoldType::E8
        | ArnoldType::Higher
        | ArnoldType::A(_) => {}
        ArnoldType::MorseRegular => {
            panic!("at the critical boundary the germ should NOT be Morse");
        }
        ArnoldType::Inadmissible => panic!("inadmissible — bug"),
    }
}

#[test]
fn integration_round_s3xs3_admits_all_low_wavenumbers() {
    // Continuous isometry dim 12 ⇒ wavenumbers 1..=12 admissible.
    let mut state = published_saturn_polar();
    at_critical_boundary(&mut state);
    let nums = predict_wavenumber_set(&state, &[], 12, 2).unwrap();
    for k in 1..=12u32 {
        assert!(
            nums.contains(&k),
            "round S^3 x S^3 should admit wavenumber {} (got set {:?})",
            k,
            nums
        );
    }
}

#[test]
fn integration_flat_t6_admits_low_wavenumbers() {
    let mut state = published_saturn_polar();
    at_critical_boundary(&mut state);
    let nums = predict_wavenumber_set(&state, &[], 6, 2).unwrap();
    for k in 1..=6u32 {
        assert!(nums.contains(&k));
    }
}

#[test]
fn integration_no_isometry_admits_only_arnold_set() {
    let mut state = published_saturn_polar();
    at_critical_boundary(&mut state);
    let nums = predict_wavenumber_set(&state, &[], 0, 2).unwrap();
    let nums_perm = predict_wavenumber_set(&state, &[], 12, 2).unwrap();
    // No-isometry set is a subset of the high-isometry set.
    for k in &nums {
        assert!(nums_perm.contains(k));
    }
    // And it's strictly smaller (most permissive case adds at
    // least the wavenumbers 1..=12 from continuous isometry).
    assert!(nums.len() < nums_perm.len());
}

#[test]
fn integration_z3_cyclic_factor_adds_multiples_of_3() {
    let mut state = published_saturn_polar();
    at_critical_boundary(&mut state);
    let z3 = vec![CyclicSubgroup::new(3)];
    let nums = predict_wavenumber_set(&state, &z3, 0, 2).unwrap();
    // Should contain 3, 6, ... up to the type cutoff.
    assert!(nums.contains(&3));
    assert!(nums.contains(&6));
}

#[test]
fn integration_arnold_type_cutoff_monotone_in_label_complexity() {
    // E_8 has the largest Coxeter exponents (up to 29); cutoff
    // should reflect that.
    let c8 = arnold_type_cutoff(ArnoldType::E8);
    let c2 = arnold_type_cutoff(ArnoldType::A(2));
    assert!(c8 > c2);
}
