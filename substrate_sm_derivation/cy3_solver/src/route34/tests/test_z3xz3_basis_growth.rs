//! P-INFRA Fix 2 regression test.
//!
//! `Z3xZ3BundleConfig::seed_max_total_degree` controls the maximum
//! total polynomial degree of the AKLP b_lines seed-basis expansion.
//!
//! Pre-fix the field did not exist and the basis was locked at the
//! degrees encoded in `bundle.b_lines` (mostly 0/1 → 24 modes total).
//! Post-fix the field exists with default `1` (back-compat with P7.6),
//! and higher values strictly grow the basis.

use crate::route34::wilson_line_e8_z3xz3::Z3xZ3WilsonLines;
use crate::route34::zero_modes_harmonic_z3xz3::{
    expanded_seed_basis_for_test, Z3xZ3BundleConfig,
};
use crate::zero_modes::MonadBundle;

#[test]
fn z3xz3_bundle_basis_grows_with_seed_degree() {
    let bundle = MonadBundle::anderson_lukas_palti_example();
    let _wilson = Z3xZ3WilsonLines::canonical_aklp_schoen();

    let cfg_low = Z3xZ3BundleConfig {
        seed_max_total_degree: 1,
        ..Z3xZ3BundleConfig::default()
    };
    let cfg_med = Z3xZ3BundleConfig {
        seed_max_total_degree: 2,
        ..Z3xZ3BundleConfig::default()
    };
    let cfg_high = Z3xZ3BundleConfig {
        seed_max_total_degree: 3,
        ..Z3xZ3BundleConfig::default()
    };

    let basis_low = expanded_seed_basis_for_test(&bundle, cfg_low.seed_max_total_degree);
    let basis_med = expanded_seed_basis_for_test(&bundle, cfg_med.seed_max_total_degree);
    let basis_high = expanded_seed_basis_for_test(&bundle, cfg_high.seed_max_total_degree);

    assert!(
        basis_low.len() > 0,
        "basis at degree 1 must be non-empty"
    );
    assert!(
        basis_med.len() > basis_low.len(),
        "basis at degree 2 ({}) must exceed degree 1 ({})",
        basis_med.len(),
        basis_low.len()
    );
    assert!(
        basis_high.len() > basis_med.len(),
        "basis at degree 3 ({}) must exceed degree 2 ({})",
        basis_high.len(),
        basis_med.len()
    );
    // Quadrature growth: degree 3 should be at least 4× degree 1.
    assert!(
        basis_high.len() >= 4 * basis_low.len(),
        "basis at degree 3 ({}) must be at least 4× degree 1 ({})",
        basis_high.len(),
        basis_low.len()
    );
}

#[test]
fn z3xz3_bundle_default_seed_degree_is_one() {
    // Back-compat: default must reproduce the P7.6 locked-basis
    // behaviour (degrees 0/1 only).
    let cfg = Z3xZ3BundleConfig::default();
    assert_eq!(
        cfg.seed_max_total_degree, 1,
        "default seed_max_total_degree must be 1 for P7.6 back-compat"
    );
}
