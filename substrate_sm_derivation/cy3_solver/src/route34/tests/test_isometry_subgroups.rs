//! Integration tests for cyclic-subgroup detection and the
//! admissible-wavenumber map consumed by the Arnold classifier.

use crate::route34::isometry_subgroups::*;
use crate::route34::killing_solver::*;
use crate::route34::lichnerowicz::*;

#[test]
fn flat_translation_admits_all_wavenumbers() {
    let d = 3;
    let metric = FlatMetric { d };
    let basis = PolynomialVectorBasis::coordinate_polynomial_basis(d, 0);
    let n_pts = 50;
    let mut pts = Vec::with_capacity(n_pts * d);
    let mut rng_state = 99u64;
    for _ in 0..(n_pts * d) {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
        pts.push(bits * 2.0 - 1.0);
    }
    let weights = vec![1.0; n_pts];
    let opts = KillingSolveOptions::default();
    let result = killing_algebra(&metric, &basis, &pts, &weights, &opts).unwrap();
    let killing_basis: Vec<_> = result.basis.iter().take(result.dim).cloned().collect();
    let structure = isometry_structure(&killing_basis, None, 1e-6);
    assert_eq!(structure.killing_dim, d);
    assert!(structure.has_continuous_s1);
    let wn = polyhedral_admissible_wavenumbers(&structure);
    assert_eq!(wn.len(), N_MAX_PHYSICAL as usize);
    for n in 1..=N_MAX_PHYSICAL {
        assert!(wn.contains(&n));
    }
}

#[test]
fn no_killing_admits_only_n1() {
    let killing_basis: Vec<KillingVectorField> = Vec::new();
    let structure = isometry_structure(&killing_basis, None, 1e-6);
    assert_eq!(structure.killing_dim, 0);
    assert_eq!(structure.abelian_rank, 0);
    assert!(!structure.has_continuous_s1);
    let wn = polyhedral_admissible_wavenumbers(&structure);
    assert_eq!(wn, vec![1u32]);
}

#[test]
fn custom_n_max_works() {
    let basis = vec![KillingVectorField {
        coefficients: vec![1.0],
        eigenvalue: 0.0,
        residual: 0.0,
    }];
    let structure = isometry_structure(&basis, None, 1e-6);
    let wn = polyhedral_admissible_wavenumbers_with_bound(&structure, 5);
    assert_eq!(wn, vec![1, 2, 3, 4, 5]);
}

#[test]
fn abelian_rank_so3_levi_civita() {
    // ε_abc structure constants (so(3)). Greedy abelian rank should be 1.
    let k = 3;
    let mut f = vec![0.0; k * k * k];
    let eps = |a: usize, b: usize, c: usize| -> f64 {
        match (a, b, c) {
            (0, 1, 2) | (1, 2, 0) | (2, 0, 1) => 1.0,
            (1, 0, 2) | (0, 2, 1) | (2, 1, 0) => -1.0,
            _ => 0.0,
        }
    };
    for a in 0..k {
        for b in 0..k {
            for c in 0..k {
                f[a * k * k + b * k + c] = eps(a, b, c);
            }
        }
    }
    assert_eq!(abelian_rank(&f, k, 0.5), 1);
}
