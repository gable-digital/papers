//! Integration-level tests for [`crate::route34::z3xz3_projector`].
//!
//! Cross-checks character-table behaviour, idempotency, and orbit-canonical
//! invariants against published representation theory of `Z/3 × Z/3`.

use crate::route34::z3xz3_projector::{
    alpha_character, beta_act, beta_orbit, beta_orbit_canonical, enumerate_bidegree_monomials,
    Monomial, Z3xZ3Projector, N_EXP,
};

/// β has order 3 on **all** monomials of degree ≤ 4.
#[test]
fn beta_has_order_three_on_all_monomials() {
    for d_x in 0..=4u32 {
        for d_y in 0..=(4u32 - d_x) {
            for d_t in 0..=2u32 {
                let mons = enumerate_bidegree_monomials(d_x, d_y, d_t);
                for m in mons {
                    let m1 = beta_act(&m);
                    let m2 = beta_act(&m1);
                    let m3 = beta_act(&m2);
                    assert_eq!(m, m3, "β³ should = identity on {m:?}");
                }
            }
        }
    }
}

/// α-character is constant on β-orbits (because αβ = βα).
#[test]
fn alpha_char_constant_on_beta_orbits() {
    for d_x in 0..=4u32 {
        for d_y in 0..=4u32 {
            for d_t in 0..=2u32 {
                let mons = enumerate_bidegree_monomials(d_x, d_y, d_t);
                for m in mons {
                    let chi = alpha_character(&m);
                    for k in beta_orbit(&m).iter() {
                        assert_eq!(alpha_character(k), chi);
                    }
                }
            }
        }
    }
}

/// Projector idempotency: P · P = P (machine precision).
#[test]
fn projector_idempotent_on_full_basis_3_3_1() {
    let p = Z3xZ3Projector::new();
    let mons = enumerate_bidegree_monomials(3, 3, 1);
    let r = p.idempotency_residual(&mons);
    assert!(r < 1e-12, "idempotency residual = {r}");
}

/// At bidegree (3, 3, 1) the invariant subspace contains the canonical
/// Schoen defining-polynomial monomials. Specifically the monomials
/// `x_a^3 y_b^3 t_c` for any a, b, c are α-invariant (α-character =
/// 0 mod 3 since the cubed exponent kills phases) and lie in distinct
/// β-orbits or the same ones depending on the indices.
#[test]
fn defining_polynomial_monomials_are_invariant() {
    let p = Z3xZ3Projector::new();
    // x_0^3 y_0^3 t_0 — manifestly α-invariant
    let m: Monomial = [3, 0, 0, 3, 0, 0, 1, 0];
    assert_eq!(alpha_character(&m), 0);
    let canon = beta_orbit_canonical(&m);
    let inv = p.project_invariant_basis(&[m]);
    assert!(inv.contains(&canon));
}

/// Trivial monomial is invariant.
#[test]
fn trivial_monomial_invariant() {
    let p = Z3xZ3Projector::new();
    let m: Monomial = [0; N_EXP];
    let inv = p.project_invariant_basis(&[m]);
    assert_eq!(inv, vec![m]);
}

/// Projection-matrix column-sum is in [1/3, 1] for every invariant column
/// (this is the Reynolds-operator coefficient; orbits of length 3 give 1,
/// fixed points give 1/3 — depending on whether the orbit is free or fixed).
#[test]
fn projection_matrix_column_sums_are_bounded() {
    let p = Z3xZ3Projector::new();
    let mons = enumerate_bidegree_monomials(3, 3, 1);
    let (mat, basis) = p.projection_matrix(&mons);
    for j in 0..basis.len() {
        let mut s = 0.0_f64;
        for i in 0..mons.len() {
            s += mat[i * basis.len() + j];
        }
        assert!(
            (s - 1.0_f64 / 3.0_f64).abs() < 1e-12 || (s - 1.0_f64).abs() < 1e-12,
            "column-sum should be 1/3 or 1; got {s}"
        );
    }
}

/// `project_gram` block-diagonal property: applying to a non-trivial
/// rank-1 cross-character projector matrix zeros the cross block
/// entirely. We construct a rank-1 outer product `m_a m_b^T` between
/// monomials of distinct character classes; project_gram should zero
/// every entry.
#[test]
fn project_gram_zeros_cross_class_outer_product() {
    use crate::route34::z3xz3_projector::beta_character;
    let p = Z3xZ3Projector::new();
    let mons = enumerate_bidegree_monomials(3, 3, 1);
    let n = mons.len();

    // Find one α-invariant β-zero monomial and one with χ_α=1 (different class).
    let mut idx_invariant = None;
    let mut idx_chi1 = None;
    for (i, m) in mons.iter().enumerate() {
        if alpha_character(m) == 0 && beta_character(m) == 0 && idx_invariant.is_none() {
            idx_invariant = Some(i);
        }
        if alpha_character(m) == 1 && idx_chi1.is_none() {
            idx_chi1 = Some(i);
        }
        if idx_invariant.is_some() && idx_chi1.is_some() {
            break;
        }
    }
    let i = idx_invariant.expect("need invariant monomial");
    let j = idx_chi1.expect("need χ_α=1 monomial");
    assert_ne!(i, j);

    let mut h = vec![0.0_f64; n * n];
    h[i * n + j] = 1.0;
    h[j * n + i] = 1.0;
    p.project_gram(&mut h, &mons).expect("gram project");
    assert_eq!(h[i * n + j], 0.0, "cross-class entry not zeroed");
    assert_eq!(h[j * n + i], 0.0, "cross-class entry not zeroed");
}

/// Group order is exactly 9.
#[test]
fn group_order_is_9() {
    let p = Z3xZ3Projector::new();
    assert_eq!(p.order, 9);
}

/// Cube-root-of-unity satisfies `1 + ω + ω² = 0`.
#[test]
fn sum_of_cube_roots_is_zero() {
    let p = Z3xZ3Projector::new();
    let mut sr = 0.0_f64;
    let mut si = 0.0_f64;
    for (r, i) in &p.cube_roots {
        sr += r;
        si += i;
    }
    assert!(sr.abs() < 1e-12 && si.abs() < 1e-12, "1 + ω + ω² = ({sr}, {si})");
}
