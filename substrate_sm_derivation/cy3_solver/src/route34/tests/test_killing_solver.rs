//! Integration tests for [`crate::route34::killing_solver`]:
//! Cholesky-whitening + Jacobi-diagonalisation pipeline against
//! analytic eigenvalue spectra.

use crate::route34::killing_solver::*;
use crate::route34::lichnerowicz::*;

// ---------------------------------------------------------------------------
// Generalised eigenproblem self-consistency tests
// ---------------------------------------------------------------------------

#[test]
fn small_diagonal_killing_problem_recovers_exact_eigenvalues() {
    // L = diag(2, 4, 6), G = I; expected eigenvalues = {2, 4, 6}.
    let n = 3;
    let mut l = vec![0.0; n * n];
    l[0] = 2.0;
    l[4] = 4.0;
    l[8] = 6.0;
    let mut g = vec![0.0; n * n];
    g[0] = 1.0;
    g[4] = 1.0;
    g[8] = 1.0;
    let op = LichnerowiczOperator {
        n_basis: n,
        d: 1,
        n_sample: 1,
        total_weight: 1.0,
        l_matrix: l,
        gram_matrix: g,
        asymmetry: 0.0,
    };
    let opts = KillingSolveOptions {
        tol_abs: 0.0,
        tol_rel: 1e-12,
        gram_regularisation: 0.0,
        max_jacobi_sweeps: 32,
        jacobi_tol: 1e-14,
    };
    let result = solve_killing_kernel(&op, &opts).unwrap();
    assert_eq!(result.dim, 0);
    assert!((result.spectrum[0] - 2.0).abs() < 1e-10);
    assert!((result.spectrum[1] - 4.0).abs() < 1e-10);
    assert!((result.spectrum[2] - 6.0).abs() < 1e-10);
}

#[test]
fn singular_l_matrix_gives_correct_kernel() {
    // L = diag(0, 0, 4), G = I → kernel dim = 2.
    let n = 3;
    let mut l = vec![0.0; n * n];
    l[8] = 4.0;
    let mut g = vec![0.0; n * n];
    g[0] = 1.0;
    g[4] = 1.0;
    g[8] = 1.0;
    let op = LichnerowiczOperator {
        n_basis: n,
        d: 1,
        n_sample: 1,
        total_weight: 1.0,
        l_matrix: l,
        gram_matrix: g,
        asymmetry: 0.0,
    };
    let opts = KillingSolveOptions {
        tol_abs: 1e-9,
        tol_rel: 1e-12,
        gram_regularisation: 0.0,
        max_jacobi_sweeps: 32,
        jacobi_tol: 1e-14,
    };
    let result = solve_killing_kernel(&op, &opts).unwrap();
    assert_eq!(result.dim, 2);
}

#[test]
fn killing_residuals_are_small_in_kernel_large_outside() {
    // Flat R^3 with degree-≤1 basis → 6-dim Killing algebra of iso(R^3).
    let d = 3;
    let metric = FlatMetric { d };
    let basis = PolynomialVectorBasis::coordinate_polynomial_basis(d, 1);
    let n_pts = 200;
    let mut pts = Vec::with_capacity(n_pts * d);
    let mut rng_state = 7u64;
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
    let expected = d + d * (d - 1) / 2;
    assert_eq!(result.dim, expected);
    // Residuals on the kernel modes are tiny.
    for j in 0..result.dim {
        assert!(
            result.basis[j].residual < 1e-6,
            "kernel mode {j} residual = {} too large",
            result.basis[j].residual
        );
    }
    // Residuals on the non-kernel modes are non-trivial.
    if result.basis.len() > expected {
        assert!(
            result.basis[expected].residual > 1e-3,
            "first non-kernel mode residual {} too small",
            result.basis[expected].residual
        );
    }
}

// ---------------------------------------------------------------------------
// Bracket structure tests
// ---------------------------------------------------------------------------

#[test]
fn flat_translation_brackets_vanish() {
    // Translations on flat R^3 commute. Their Lie brackets should be ~0.
    let d = 3;
    let metric = FlatMetric { d };
    let basis = PolynomialVectorBasis::coordinate_polynomial_basis(d, 0);
    assert_eq!(basis.n_basis(), 3);
    let n_pts = 100;
    let mut pts = Vec::with_capacity(n_pts * d);
    let mut rng_state = 11u64;
    for _ in 0..(n_pts * d) {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
        pts.push(bits * 2.0 - 1.0);
    }
    let weights = vec![1.0; n_pts];
    let opts = KillingSolveOptions::default();
    let op = assemble_lichnerowicz_matrix(&metric, &basis, &pts, &weights).unwrap();
    let result = solve_killing_kernel(&op, &opts).unwrap();
    let killing_basis: Vec<_> = result.basis.iter().take(result.dim).cloned().collect();
    let f = killing_bracket_structure_constants(
        &op,
        &killing_basis,
        &metric,
        &basis,
        &pts,
        &weights,
    )
    .unwrap();
    let max_abs: f64 = f.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    assert!(
        max_abs < 1e-8,
        "translation Lie brackets should vanish, max |f| = {max_abs}"
    );
}
