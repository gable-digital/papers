//! P7.8b — Galerkin subspace-inclusion regression test.
//!
//! P7.8 added orthogonal-basis assembly via modified Gram-Schmidt with
//! deflation. The hostile review demonstrated that per-cap deflation is
//! applied independently — the orthonormal basis at `td = 2` is NOT a
//! subspace of the orthonormal basis at `td = 3`, because different
//! vectors get deflated at different cap levels. On production Schoen
//! data the lowest five eigenvalues at `td = 3` were strictly larger
//! than at `td = 2`, which is impossible for a true Galerkin
//! refinement (Courant-Fischer min-max forces lower-or-equal
//! eigenvalues on a refined subspace).
//!
//! P7.8b restores subspace inclusion by switching the seed enumeration
//! to a **cumulative** bigraded basis ordered canonically by
//! `(td_first_appears, b_line, exponents_lex)` where `td_first_appears
//! = max(deg_z, deg_w)`. Under this ordering the basis at cap = k is
//! exactly the prefix of the basis at cap = k+1 where `td_first ≤ k`.
//! Modified Gram-Schmidt with deflation in this order then produces
//! `Q_k ⊆ Q_{k+1}` by construction, restoring the min-max guarantee.
//!
//! This test exercises the invariant directly: it solves the
//! Z/3 × Z/3 + H_4 bundle Laplacian on the same Schoen background at
//! `td = 2` and `td = 3` with `orthogonalize_first = true`, then asserts
//! the lowest five eigenvalues at `td = 2` are bounded below by the
//! lowest five at `td = 3` (within tolerance). Pre-fix this would fail.

use crate::route34::cy3_metric_unified::{
    Cy3MetricResultKind, Cy3MetricSolver, Cy3MetricSpec, SchoenSolver,
};
use crate::route34::hym_hermitian::{solve_hym_metric, HymConfig};
use crate::route34::metric_laplacian::MetricLaplacianConfig;
use crate::route34::metric_laplacian_projected::{
    compute_projected_metric_laplacian_spectrum, ProjectionKind,
};
use crate::route34::wilson_line_e8_z3xz3::Z3xZ3WilsonLines;
use crate::route34::yukawa_pipeline::Cy3MetricResultBackground;
use crate::route34::zero_modes_harmonic_z3xz3::{
    solve_z3xz3_bundle_laplacian, Z3xZ3BundleConfig, Z3xZ3Geometry,
};
use crate::zero_modes::MonadBundle;

/// Cheap Schoen Donaldson background suitable for the regression test.
/// Mirrors the small-Schoen factory in test_z3xz3_orthogonal_basis.rs.
fn small_schoen_background()
-> Result<crate::route34::schoen_metric::SchoenMetricResult, String>
{
    let spec = Cy3MetricSpec::Schoen {
        d_x: 3,
        d_y: 3,
        d_t: 1,
        n_sample: 1500,
        max_iter: 8,
        donaldson_tol: 1.0e-2,
        seed: 12345,
    };
    let solver = SchoenSolver;
    let kind = solver
        .solve_metric(&spec)
        .map_err(|e| format!("test schoen solve: {e}"))?;
    match kind {
        Cy3MetricResultKind::Schoen(t) => Ok(*t),
        Cy3MetricResultKind::TianYau(_) => {
            Err("Schoen solver returned TY result".to_string())
        }
    }
}

#[test]
fn galerkin_subspace_inclusion_via_min_max() {
    let result = match small_schoen_background() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("SKIP galerkin_subspace_inclusion test: {e}");
            return;
        }
    };
    let bg = Cy3MetricResultBackground::from_schoen(&result);
    let bundle = MonadBundle::anderson_lukas_palti_example();
    let wilson = Z3xZ3WilsonLines::canonical_aklp_schoen();
    let hym_cfg = HymConfig {
        max_iter: 4,
        damping: 0.5,
        ..HymConfig::default()
    };
    let h_v = solve_hym_metric(&bundle, &bg, &hym_cfg);

    let cfg_td2 = Z3xZ3BundleConfig {
        geometry: Z3xZ3Geometry::Schoen,
        apply_h4: true,
        seed_max_total_degree: 2,
        tikhonov_lambda: 0.0,
        orthogonalize_first: true,
        orthogonalize_tol: 1.0e-10,
        ..Z3xZ3BundleConfig::default()
    };
    let cfg_td3 = Z3xZ3BundleConfig {
        seed_max_total_degree: 3,
        ..cfg_td2.clone()
    };

    let r2 = solve_z3xz3_bundle_laplacian(&bundle, &bg, &h_v, &wilson, &cfg_td2);
    let r3 = solve_z3xz3_bundle_laplacian(&bundle, &bg, &h_v, &wilson, &cfg_td3);

    assert!(
        !r2.eigenvalues_full.is_empty(),
        "td=2 spectrum is empty"
    );
    assert!(
        !r3.eigenvalues_full.is_empty(),
        "td=3 spectrum is empty"
    );

    // Min-max / Courant-Fischer: lowest k eigenvalues at td=3 ≤ lowest
    // k eigenvalues at td=2 for every k ≤ rank(td=2). We allow a
    // small additive slack to absorb f64 / Donaldson-noise floor; the
    // hostile-review failure mode was an order-of-magnitude flip
    // (5.97 -> 11.43), not a sub-ε wobble.
    let scale = r2
        .eigenvalues_full
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1.0);
    let slack = 1.0e-3 * scale;
    let n_check = r2.eigenvalues_full.len().min(r3.eigenvalues_full.len()).min(5);
    assert!(
        n_check > 0,
        "no overlap to check; r2 dim {}, r3 dim {}",
        r2.eigenvalues_full.len(),
        r3.eigenvalues_full.len()
    );
    for k in 0..n_check {
        let lam2 = r2.eigenvalues_full[k];
        let lam3 = r3.eigenvalues_full[k];
        assert!(
            lam3 <= lam2 + slack,
            "Galerkin refinement violated at index {}: \
             td=2 λ_{} = {:.6e}, td=3 λ_{} = {:.6e} (slack {:.3e}, scale {:.3e}). \
             td=2 head: {:?}, td=3 head: {:?}",
            k,
            k,
            lam2,
            k,
            lam3,
            slack,
            scale,
            &r2.eigenvalues_full[..r2.eigenvalues_full.len().min(5)],
            &r3.eigenvalues_full[..r3.eigenvalues_full.len().min(5)],
        );
    }
}

/// Analogous regression for the metric Laplacian path
/// (`compute_projected_metric_laplacian_spectrum` with
/// `orthogonalize_first = true`). Pre-P7.8b the projected basis
/// inherited the `(dz, dw)` lex order from `build_test_basis`, which
/// interleaves higher-`dz` (and therefore higher-total-degree)
/// monomials between lower-`dz` ones — so a td=2 basis was NOT a
/// prefix of the td=3 basis and per-cap deflation broke subspace
/// inclusion.
///
/// Post-fix `compute_projected_metric_laplacian_spectrum` re-sorts
/// the projected basis by `(total_degree, exponents)`, restoring the
/// prefix property and the Courant-Fischer min-max guarantee.
#[test]
fn galerkin_subspace_inclusion_metric_laplacian() {
    let result = match small_schoen_background() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("SKIP galerkin_subspace_inclusion_metric_laplacian test: {e}");
            return;
        }
    };
    let bg = Cy3MetricResultBackground::from_schoen(&result);

    let cfg_td2 = MetricLaplacianConfig {
        max_total_degree: 2,
        n_low_eigenvalues: 5,
        mass_regularisation: 0.0,
        orthogonalize_first: true,
        orthogonalize_tol: 1.0e-10,
        ..MetricLaplacianConfig::default()
    };
    let cfg_td3 = MetricLaplacianConfig {
        max_total_degree: 3,
        ..cfg_td2.clone()
    };

    let r2 = compute_projected_metric_laplacian_spectrum(
        &bg,
        &cfg_td2,
        ProjectionKind::SchoenZ3xZ3,
    );
    let r3 = compute_projected_metric_laplacian_spectrum(
        &bg,
        &cfg_td3,
        ProjectionKind::SchoenZ3xZ3,
    );

    let s2 = &r2.spectrum.eigenvalues_full;
    let s3 = &r3.spectrum.eigenvalues_full;

    assert!(!s2.is_empty(), "td=2 metric-Laplacian spectrum is empty");
    assert!(!s3.is_empty(), "td=3 metric-Laplacian spectrum is empty");

    let scale = s2.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(1.0);
    let slack = 1.0e-3 * scale;
    let n_check = s2.len().min(s3.len()).min(5);
    assert!(
        n_check > 0,
        "no overlap to check; r2 dim {}, r3 dim {}",
        s2.len(),
        s3.len()
    );
    for k in 0..n_check {
        assert!(
            s3[k] <= s2[k] + slack,
            "metric-Laplacian Galerkin refinement violated at index {}: \
             td=2 λ_{} = {:.6e}, td=3 λ_{} = {:.6e} (slack {:.3e}, scale {:.3e}). \
             td=2 head: {:?}, td=3 head: {:?}",
            k,
            k,
            s2[k],
            k,
            s3[k],
            slack,
            scale,
            &s2[..s2.len().min(5)],
            &s3[..s3.len().min(5)],
        );
    }
}
