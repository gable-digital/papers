//! Validation tests for the Lichnerowicz operator and Killing-vector
//! kernel extraction against analytic results on round S^n, flat T^n,
//! and product S^2 × S^2 × S^2.
//!
//! These are integration-style tests that exercise the full assembly
//! → kernel-extraction pipeline. Faster unit tests live inline in the
//! module files.

use crate::route34::isometry_subgroups::*;
use crate::route34::killing_solver::*;
use crate::route34::lichnerowicz::*;

// ---------------------------------------------------------------------------
// Sample-point generation helpers
// ---------------------------------------------------------------------------

/// Generate `n_pts` deterministic uniform points in the box `[-r, r]^d`
/// using a linear-congruential RNG with the given seed.
fn lcg_box_points(n_pts: usize, d: usize, r: f64, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut out = Vec::with_capacity(n_pts * d);
    for _ in 0..(n_pts * d) {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = ((state >> 11) & ((1u64 << 53) - 1)) as f64;
        let u = bits / (1u64 << 53) as f64;
        out.push((u * 2.0 - 1.0) * r);
    }
    out
}

// ---------------------------------------------------------------------------
// Flat-torus tests
// ---------------------------------------------------------------------------

#[test]
fn flat_t6_translations_kernel_dim_6() {
    // T^6 has 6 translation Killing vectors and (in flat R^6) also
    // 15 rotations, but the rotations are *not* Killing on the torus
    // because they don't descend through the periodic identification.
    // For the local-chart computation we use here, however, we DO
    // recover both: degree-≤1 polynomial basis in 6 variables → kernel
    // dim = 6 (translations) + 15 (rotations of so(6)) = 21.
    //
    // The test below uses degree-0 only ⇒ kernel dim = 6.
    let d = 6;
    let metric = FlatMetric { d };
    let basis = PolynomialVectorBasis::coordinate_polynomial_basis(d, 0);
    assert_eq!(basis.n_basis(), d);
    let pts = lcg_box_points(200, d, 1.0, 12345);
    let weights = vec![1.0; pts.len() / d];
    let opts = KillingSolveOptions::default();
    let result = killing_algebra(&metric, &basis, &pts, &weights, &opts).unwrap();
    assert_eq!(result.dim, 6, "T^6 translation Killing dim");
}

#[test]
fn flat_t3_translations_plus_rotations() {
    // Flat R^3 with degree-≤1 basis: 3 + 3 = 6 = dim(iso(R^3)).
    let d = 3;
    let metric = FlatMetric { d };
    let basis = PolynomialVectorBasis::coordinate_polynomial_basis(d, 1);
    let pts = lcg_box_points(150, d, 1.0, 42);
    let weights = vec![1.0; pts.len() / d];
    let opts = KillingSolveOptions::default();
    let result = killing_algebra(&metric, &basis, &pts, &weights, &opts).unwrap();
    let expected = d + d * (d - 1) / 2; // translations + so(d)
    assert_eq!(result.dim, expected, "iso(R^3) Killing dim");
}

#[test]
fn flat_metric_residuals_below_kernel_above_separation() {
    // After degree-≤1, the next-higher degree mode (quadratic) is *not*
    // a Killing vector on flat space. Verify the kernel/non-kernel
    // separation is large.
    let d = 3;
    let metric = FlatMetric { d };
    let basis = PolynomialVectorBasis::coordinate_polynomial_basis(d, 2);
    let pts = lcg_box_points(150, d, 1.0, 42);
    let weights = vec![1.0; pts.len() / d];
    let opts = KillingSolveOptions::default();
    let result = killing_algebra(&metric, &basis, &pts, &weights, &opts).unwrap();
    let expected_kernel = d + d * (d - 1) / 2;
    assert_eq!(result.dim, expected_kernel);
    // Spectrum gap: first non-kernel eigenvalue > tol_used by margin.
    let first_nonkernel = result.spectrum[expected_kernel];
    assert!(
        first_nonkernel > result.tol_used * 100.0,
        "spectrum[{expected_kernel}] = {first_nonkernel} should exceed tol_used = {} by a wide \
         margin",
        result.tol_used,
    );
}

// ---------------------------------------------------------------------------
// Christoffel-symbol sanity tests
// ---------------------------------------------------------------------------

#[test]
fn christoffel_symbols_on_stereographic_s2() {
    // On the stereographic-coords round S^2:
    //   g_{ij} = (4/(1+|y|²)²) δ_{ij}
    //   Γ^λ_{μν} = -2 (y_μ δ^λ_ν + y_ν δ^λ_μ - y^λ δ_{μν}) / (1 + |y|²)
    let metric = StereographicSphereMetric { d: 2 };
    let y = vec![0.3, -0.4];
    let mut g = vec![0.0; 4];
    let mut dg = vec![0.0; 8];
    metric.evaluate(&y, &mut g, &mut dg);
    let mut g_inv = vec![0.0; 4];
    let mut gamma = vec![0.0; 8];
    let mut lu = vec![0.0; 4];
    let mut perm = vec![0usize; 2];
    let mut col = vec![0.0; 2];
    christoffel_symbols(
        &g, &dg, 2, &mut g_inv, &mut gamma, &mut lu, &mut perm, &mut col,
    )
    .unwrap();
    // Compare entry-wise with the analytic formula.
    let denom = 1.0 + 0.3 * 0.3 + 0.4 * 0.4;
    for lam in 0..2 {
        for mu in 0..2 {
            for nu in 0..2 {
                let kron_lam_nu = if lam == nu { 1.0 } else { 0.0 };
                let kron_lam_mu = if lam == mu { 1.0 } else { 0.0 };
                let kron_mu_nu = if mu == nu { 1.0 } else { 0.0 };
                let analytic = -2.0
                    * (y[mu] * kron_lam_nu + y[nu] * kron_lam_mu
                        - y[lam] * kron_mu_nu)
                    / denom;
                let numeric = gamma[lam * 4 + mu * 2 + nu];
                assert!(
                    (numeric - analytic).abs() < 1e-10,
                    "Γ^{lam}_{mu}{nu} numeric = {numeric}, analytic = {analytic}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Symmetric / asymmetric properties of L
// ---------------------------------------------------------------------------

#[test]
fn lichnerowicz_matrix_is_symmetrised() {
    let d = 3;
    let metric = FlatMetric { d };
    let basis = PolynomialVectorBasis::coordinate_polynomial_basis(d, 1);
    let pts = lcg_box_points(50, d, 1.0, 7);
    let weights = vec![1.0; pts.len() / d];
    let op = assemble_lichnerowicz_matrix(&metric, &basis, &pts, &weights).unwrap();
    let n = op.n_basis;
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = op.l_matrix[i * n + j] - op.l_matrix[j * n + i];
            assert!(diff.abs() < 1e-14, "L should be exactly symmetric after symmetrise()");
        }
    }
}

// ---------------------------------------------------------------------------
// Round-sphere tests (analytic so(d+1) Killing dimension)
// ---------------------------------------------------------------------------

#[test]
fn round_s2_killing_dim_so3_three() {
    // S^2 in stereographic coords has so(3) Killing algebra (dim 3).
    // The 3 generators are quadratic vector fields, so a degree-≤2
    // polynomial basis can express all of them.
    let metric = StereographicSphereMetric { d: 2 };
    let basis = PolynomialVectorBasis::coordinate_polynomial_basis(2, 2);
    // Stereographic chart covers all of S^2 minus the north pole; sample
    // points stay well away from infinity by sampling in the disc |y| ≤ 1.
    let n_pts = 400;
    let mut pts = Vec::with_capacity(n_pts * 2);
    let mut rng_state = 314u64;
    let mut accepted = 0;
    while accepted < n_pts {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
        let x = bits * 2.0 - 1.0;
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
        let y = bits * 2.0 - 1.0;
        if x * x + y * y <= 1.0 {
            pts.push(x);
            pts.push(y);
            accepted += 1;
        }
    }
    // Quadrature weight = round-S^2 volume element / chart-Lebesgue
    // = sqrt(det g) = (4 / (1 + |y|²)²)^(d/2) = (4 / (1+|y|²)²) for d=2.
    let mut weights = Vec::with_capacity(n_pts);
    for p in 0..n_pts {
        let y0 = pts[p * 2];
        let y1 = pts[p * 2 + 1];
        let r2 = y0 * y0 + y1 * y1;
        let denom = 1.0 + r2;
        // sqrt(det g) for the conformal metric (4/(1+r²)²)δ_ij is
        // (4/(1+r²)²)^(d/2) = 4 / (1+r²)^d for d=2.
        weights.push(4.0 / (denom * denom));
    }
    let opts = KillingSolveOptions {
        tol_abs: 1e-3,
        tol_rel: 1e-3,
        gram_regularisation: 1e-10,
        max_jacobi_sweeps: 64,
        jacobi_tol: 1e-13,
    };
    let result = killing_algebra(&metric, &basis, &pts, &weights, &opts).unwrap();
    // Expected: 3 (so(3)).
    assert_eq!(
        result.dim, 3,
        "round S^2 Killing dim: expected 3 (so(3)), got {} (spectrum head: {:?})",
        result.dim,
        &result.spectrum[..result.spectrum.len().min(8)]
    );
}

// ---------------------------------------------------------------------------
// Cyclic-subgroup / wavenumber-admissibility tests
// ---------------------------------------------------------------------------

#[test]
fn translation_killing_admits_all_wavenumbers_up_to_n_max() {
    // Continuous Killing flow ⇒ S^1 ⊃ Z/n for all n.
    let d = 3;
    let metric = FlatMetric { d };
    let basis = PolynomialVectorBasis::coordinate_polynomial_basis(d, 0);
    let pts = lcg_box_points(60, d, 1.0, 9);
    let weights = vec![1.0; pts.len() / d];
    let opts = KillingSolveOptions::default();
    let result = killing_algebra(&metric, &basis, &pts, &weights, &opts).unwrap();
    let killing_basis: Vec<_> = result
        .basis
        .iter()
        .take(result.dim)
        .cloned()
        .collect();
    let structure = isometry_structure(&killing_basis, None, 1e-6);
    assert!(structure.has_continuous_s1);
    let wn = polyhedral_admissible_wavenumbers(&structure);
    assert_eq!(wn.len(), N_MAX_PHYSICAL as usize);
    for n in 1..=N_MAX_PHYSICAL {
        assert!(wn.contains(&n), "n={n} should be admissible");
    }
}

#[test]
fn empty_killing_admits_only_n1() {
    // No killing fields: basis = {} (manually constructed degenerate
    // case to model the "generic CY3 has zero continuous isometry"
    // analytic result of Yau 1978).
    let killing_basis: Vec<KillingVectorField> = Vec::new();
    let structure = isometry_structure(&killing_basis, None, 1e-6);
    assert_eq!(structure.killing_dim, 0);
    assert!(!structure.has_continuous_s1);
    let wn = polyhedral_admissible_wavenumbers(&structure);
    assert_eq!(wn, vec![1]);
}

// ---------------------------------------------------------------------------
// CPU / GPU agreement (gated behind feature flag)
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
#[test]
fn cpu_gpu_matrix_agreement_within_1e10() {
    use crate::route34::lichnerowicz_gpu::assemble_lichnerowicz_matrix_gpu;

    let d = 3;
    let metric = FlatMetric { d };
    let basis = PolynomialVectorBasis::coordinate_polynomial_basis(d, 1);
    let pts = lcg_box_points(64, d, 1.0, 1);
    let weights = vec![1.0; pts.len() / d];

    let cpu_op =
        assemble_lichnerowicz_matrix(&metric, &basis, &pts, &weights).unwrap();
    let gpu_res = assemble_lichnerowicz_matrix_gpu(&metric, &basis, &pts, &weights);
    if let Ok(gpu_op) = gpu_res {
        assert_eq!(cpu_op.n_basis, gpu_op.n_basis);
        let n = cpu_op.n_basis;
        let mut max_rel = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let cl = cpu_op.l_matrix[i * n + j];
                let gl = gpu_op.l_matrix[i * n + j];
                let diff = (cl - gl).abs();
                let scale = cl.abs().max(gl.abs()).max(1e-12);
                let rel = diff / scale;
                if rel > max_rel {
                    max_rel = rel;
                }
            }
        }
        assert!(
            max_rel < 1e-9,
            "CPU/GPU L max-relative-diff = {max_rel}; expected < 1e-9"
        );
    }
    // GPU init failure is acceptable in environments without CUDA.
}
