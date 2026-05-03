//! P7.2b — projected-basis metric Laplacian.
//!
//! Sister to [`crate::route34::metric_laplacian`]: same Galerkin assembly
//! and Hermitian generalised eigensolve, but on a basis that has been
//! pre-projected onto the **trivial irrep of the discrete symmetry group**
//! before the mass / stiffness matrices are built.
//!
//! ## Why
//!
//! The full bigraded-monomial test basis used in
//! [`crate::route34::metric_laplacian::build_test_basis`] is *not* a
//! direct sum of `Z/3` (TY) or `Z/3 × Z/3` (Schoen) isotypic components.
//! When the underlying CY3 is genuinely Z/3-(×Z/3-)symmetric, the metric
//! Laplacian commutes with the symmetry and ought to block-diagonalise,
//! but in the f64 Galerkin matrix the block structure is broken by:
//!
//!   * Donaldson-noise on the metric / mass / stiffness matrices
//!     (per-point sampling weights are not exactly invariant under the
//!     orbit-averaged measure — they are only *consistent estimators* of
//!     it),
//!   * f64 roundoff on the eigensolve.
//!
//! Empirically (P7.1, commit db1b4ca1) this leaks ~74 % of an eigenvector's
//! L²-mass into non-trivial character buckets even when the dominant
//! character is trivial — the irrep classification at fixed test_degree=4,
//! basis_dim=494 has only ~26 % purity. That noise is large enough to
//! swamp the ω_fix = 1 − 1/dim(E_8) = 123/248 prediction we are testing.
//!
//! ## How
//!
//! Both the Schoen Z/3×Z/3 and the TY Z/3 actions on the bigraded
//! ambient are **diagonal** (each monomial maps to itself up to a phase
//! determined by an integer character — see
//! [`crate::route34::z3xz3_projector`]). The trivial-character projector
//!
//!     P(m) = m  if χ(m) = 0;  P(m) = 0  otherwise
//!
//! therefore reduces to **filtering** the basis to the χ = 0 subset.
//! The surviving basis is exact in the trivial rep by construction, so
//! every eigenvalue of `M^{-1} K v = λ v` on this projected basis is in
//! the trivial irrep with 100 % purity (no character mixing possible —
//! the basis simply does not contain any non-trivial-rep functions).
//!
//! Mathematically this is the **Reynolds-projected Galerkin space**
//! `V^{⟨G⟩}`, the same trivial-isotypic subspace that gets tested in P7.1
//! via post-hoc character classification. The projected-basis solve is
//! cheaper (basis_dim ≈ 50–80 vs 494 at test_degree=4) and clean: the
//! "purity 26 %" pathology cannot arise.
//!
//! ## Caveat: orthogonality to non-trivial reps in the *measure*
//!
//! The Galerkin pencil `(M, K)` involves the metric / volume measure,
//! which is approximately G-invariant on the converged Donaldson-balanced
//! candidate but only exactly so in the continuum limit. Hence non-zero
//! `M_{αβ}` between two trivial-rep monomials is *NOT* guaranteed to be
//! orthogonal to a non-trivial-rep contamination (Donaldson noise mixes
//! all reps when integrated against the noisy weight). However, the
//! eigenfunctions reported are guaranteed in the trivial rep at the
//! basis level — they live in the span of trivial-rep monomials by
//! construction. The Donaldson noise floor manifests as a small shift in
//! the eigenvalues but does *not* leak character mass.
//!
//! ## Public surface
//!
//! Mirrors [`metric_laplacian`]:
//!
//!   * [`build_projected_basis_schoen`] — Z/3 × Z/3 trivial-rep filter.
//!   * [`build_projected_basis_ty`] — Z/3 trivial-rep filter.
//!   * [`compute_projected_metric_laplacian_spectrum`] — Galerkin solve
//!     on the projected basis.
//!
//! All eigenvalues / eigenvectors are returned with the same conventions
//! as [`crate::route34::metric_laplacian::MetricLaplacianSpectrum`].
//!
//! ## P7.2b consumer
//!
//! [`crate::bin::p7_2b_omega_fix_localized`] (binary) compares the
//! lowest projected-basis eigenvalue to the ω_fix prediction on a
//! single converged seed.

#![allow(clippy::needless_range_loop)]

use crate::route34::hym_hermitian::MetricBackground;
use crate::route34::metric_laplacian::{build_test_basis, MetricLaplacianConfig, MetricLaplacianSpectrum, TestMonomial};
use crate::route34::z3xz3_projector::{alpha_character, beta_character};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Z/3 × Z/3 trivial-character filter for the **Schoen** ambient
/// `[x_0, x_1, x_2, y_0, y_1, y_2, t_0, t_1]`.
///
/// Returns the subset of `basis` whose `(alpha, beta)`-character is
/// `(0, 0)` — i.e. monomials that are pointwise invariant under the
/// diagonal Z/3 × Z/3 action.
pub fn build_projected_basis_schoen(basis: &[TestMonomial]) -> Vec<TestMonomial> {
    basis
        .iter()
        .filter(|m| alpha_character(&m.exponents) == 0 && beta_character(&m.exponents) == 0)
        .cloned()
        .collect()
}

/// Z/3 trivial-character filter for the **Tian-Yau** ambient
/// `[z_0, z_1, z_2, z_3, w_0, w_1, w_2, w_3]`.
///
/// The TY/Z3 character is `(m[1] + 2*m[2] + m[5] + 2*m[6]) mod 3`
/// (matches the formula used in `bin/p7_1_omega_fix_diagnostic.rs`).
/// Returns the χ = 0 subset.
pub fn build_projected_basis_ty(basis: &[TestMonomial]) -> Vec<TestMonomial> {
    basis
        .iter()
        .filter(|m| ty_z3_character(&m.exponents) == 0)
        .cloned()
        .collect()
}

/// TY/Z3 character `(m[1] + 2*m[2] + m[5] + 2*m[6]) mod 3`.
///
/// Mirrors the formula used in P7.1 (`bin/p7_1_omega_fix_diagnostic.rs`,
/// circa line 350). Treats indices 0..4 as the `z`-block and 4..8 as the
/// `w`-block, both with the Z/3 phase on the (1, 2)-entries of each
/// block.
#[inline]
pub fn ty_z3_character(exps: &[u32; 8]) -> u32 {
    (exps[1] + 2 * exps[2] + exps[5] + 2 * exps[6]) % 3
}

/// Identifier for which discrete-symmetry projection to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectionKind {
    /// Schoen: full Z/3 × Z/3 (both `α` and `β` characters must be zero).
    SchoenZ3xZ3,
    /// Tian-Yau: single Z/3 (only `α` is non-trivial).
    TianYauZ3,
}

impl ProjectionKind {
    pub fn label(&self) -> &'static str {
        match self {
            Self::SchoenZ3xZ3 => "schoen_z3xz3",
            Self::TianYauZ3 => "ty_z3",
        }
    }
}

// ---------------------------------------------------------------------
// Public driver: project basis, then call the standard Galerkin solver.
// ---------------------------------------------------------------------

/// Result of [`compute_projected_metric_laplacian_spectrum`]. Reuses
/// the data layout from [`MetricLaplacianSpectrum`] for downstream
/// compatibility, plus a few fields specific to the projected path.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProjectedSpectrumReport {
    /// Standard Galerkin spectrum on the projected basis.
    pub spectrum: MetricLaplacianSpectrum,
    /// Symmetry projection that was applied.
    pub projection: String,
    /// Original (full) basis dimension before the trivial-character filter.
    pub full_basis_dim: usize,
    /// Projected (trivial-rep) basis dimension.
    pub projected_basis_dim: usize,
    /// `projected_basis_dim / full_basis_dim` — fraction of the full basis
    /// that survived the projection. For a generic Z/3 quotient at high
    /// degree this is ~ 1/3; for Z/3 × Z/3 ~ 1/9.
    pub survival_fraction: f64,
}

/// Compute the metric-Laplacian spectrum on the trivial-isotypic
/// component of the test basis under the Z/3 (TY) or Z/3 × Z/3 (Schoen)
/// action. The spectrum is in the trivial irrep by construction —
/// no character classification is needed downstream.
pub fn compute_projected_metric_laplacian_spectrum(
    metric: &dyn MetricBackground,
    config: &MetricLaplacianConfig,
    projection: ProjectionKind,
) -> ProjectedSpectrumReport {
    let started = Instant::now();
    let full = build_test_basis(config.max_total_degree);
    let mut projected = match projection {
        ProjectionKind::SchoenZ3xZ3 => build_projected_basis_schoen(&full),
        ProjectionKind::TianYauZ3 => build_projected_basis_ty(&full),
    };
    // P7.8b — canonical hierarchical ordering. Sort the projected
    // basis by `(total_degree, exponents)` so that the basis at
    // `max_total_degree = k` is exactly the prefix of the basis at
    // `max_total_degree = k+1` where `total_degree ≤ k`. Modified
    // Gram-Schmidt with deflation in this order then produces an
    // orthonormal basis Q_k that is an L²(M)-subspace of Q_{k+1},
    // restoring the Courant-Fischer min-max guarantee that the lowest
    // `n` eigenvalues are monotone non-increasing in `k` (Galerkin
    // refinement).
    //
    // Pre-fix `build_test_basis` enumerated `(dz, dw)` lexicographically,
    // so e.g. (0, 3) appeared in the td=3 basis BEFORE (1, 0) — a
    // td=2 monomial. The td=2 basis was therefore not a prefix of the
    // td=3 basis, breaking subspace inclusion under per-cap
    // deflation. Sorting by `total_degree` first restores the prefix
    // property.
    projected.sort_by(|a, b| {
        a.total_degree
            .cmp(&b.total_degree)
            .then_with(|| a.exponents.cmp(&b.exponents))
    });
    let n_b = projected.len();
    let n_pts = metric.n_points();
    let full_basis_dim = full.len();
    let survival_fraction = if full_basis_dim > 0 {
        n_b as f64 / full_basis_dim as f64
    } else {
        0.0
    };

    if n_b == 0 || n_pts == 0 {
        return ProjectedSpectrumReport {
            spectrum: MetricLaplacianSpectrum {
                eigenvalues: Vec::new(),
                eigenvalues_full: Vec::new(),
                basis_dim: n_b,
                n_points: n_pts,
                wall_clock_seconds: started.elapsed().as_secs_f64(),
                basis_exponents: None,
                eigenvectors_full: None,
                orthogonalized_basis_dim: 0,
            },
            projection: projection.label().to_string(),
            full_basis_dim,
            projected_basis_dim: n_b,
            survival_fraction,
        };
    }

    let pts = metric.sample_points();

    // Evaluate the projected basis at every sample point, plus all 8
    // partial derivatives. Layouts mirror metric_laplacian:
    //   basis_at[α * n_pts + p]                = s_α(p)
    //   d_basis_at[(α * 8 + i) * n_pts + p]    = ∂_{z_i} s_α(p)
    let mut basis_at = vec![Complex64::new(0.0, 0.0); n_b * n_pts];
    let mut d_basis_at = vec![Complex64::new(0.0, 0.0); n_b * 8 * n_pts];
    for alpha in 0..n_b {
        let exps = projected[alpha].exponents;
        for p in 0..n_pts {
            basis_at[alpha * n_pts + p] = eval_monomial(&pts[p], &exps);
        }
        for i in 0..8 {
            let (coeff, dexps) = d_dz(&exps, i);
            if coeff == 0.0 {
                continue;
            }
            let cz = Complex64::new(coeff, 0.0);
            for p in 0..n_pts {
                d_basis_at[(alpha * 8 + i) * n_pts + p] = cz * eval_monomial(&pts[p], &dexps);
            }
        }
    }

    let m_mat = build_mass_matrix(&basis_at, n_b, metric);
    let k_mat = build_stiffness_matrix(&d_basis_at, n_b, metric);

    let (full_eigs, full_indexed, evec_storage, evec_dim) = if config.orthogonalize_first {
        // P7.11 path — modified Gram-Schmidt with deflation under L²(M)
        // inner product (the projected mass matrix `m_mat`), then
        // standard Hermitian EVP on K transformed to the orthonormal
        // basis. Mirrors P7.8 (`zero_modes_harmonic_z3xz3::run_orthogonalized`).
        let (eigvals, _evecs_orth, rank) = run_orthogonalized_metric_laplacian(
            &m_mat,
            &k_mat,
            n_b,
            config.orthogonalize_tol,
            config.jacobi_max_sweeps,
            config.jacobi_tol,
        );

        // |.|-sort the eigenvalues. With the orthogonal-basis path this
        // is a sort of the genuine Hermitian spectrum; no sign-flip
        // pathology.
        let mut full_indexed: Vec<(usize, f64)> = eigvals
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        full_indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let full_eigs: Vec<f64> = full_indexed.iter().map(|(_, v)| *v).collect();
        // Eigenvectors live in the rank-dimensional orthonormal space;
        // returning them un-pulled-back to the full basis would
        // mismatch `basis_exponents`. We therefore omit them from the
        // orthogonalized path (consumer chain_matcher only reads
        // eigenvalues anyway).
        (full_eigs, full_indexed, Vec::new(), rank)
    } else {
        // Legacy path: M^{-1} K Galerkin with Hermitian projection.
        let m_inv = invert_hermitian_regularised(&m_mat, n_b, config.mass_regularisation);
        let l_eff = matmul_complex(&m_inv, &k_mat, n_b);

        // Hermitian-project to remove antihermitian roundoff before Jacobi.
        let mut l_h = l_eff;
        for i in 0..n_b {
            let z = l_h[i * n_b + i];
            l_h[i * n_b + i] = Complex64::new(z.re, 0.0);
            for j in (i + 1)..n_b {
                let a = l_h[i * n_b + j];
                let b = l_h[j * n_b + i].conj();
                let avg = (a + b) * 0.5;
                l_h[i * n_b + j] = avg;
                l_h[j * n_b + i] = avg.conj();
            }
        }
        let (eigvals, evecs) =
            hermitian_jacobi_n(&l_h, n_b, config.jacobi_max_sweeps, config.jacobi_tol);

        let mut full_indexed: Vec<(usize, f64)> = eigvals
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        full_indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let full_eigs: Vec<f64> = full_indexed.iter().map(|(_, v)| *v).collect();

        (full_eigs, full_indexed, evecs, n_b)
    };

    let n_low = config.n_low_eigenvalues.min(full_eigs.len());
    let low: Vec<f64> = full_eigs.iter().take(n_low).cloned().collect();

    let (basis_exponents, eigenvectors_full) = if config.return_eigenvectors && !config.orthogonalize_first {
        let mut sorted_evecs = vec![(0.0_f64, 0.0_f64); n_b * n_b];
        for (new_j, (old_j, _)) in full_indexed.iter().enumerate() {
            for i in 0..n_b {
                let z = evec_storage[i * n_b + *old_j];
                sorted_evecs[i * n_b + new_j] = (z.re, z.im);
            }
        }
        let exps: Vec<[u32; 8]> = projected.iter().map(|m| m.exponents).collect();
        (Some(exps), Some(sorted_evecs))
    } else {
        (None, None)
    };

    ProjectedSpectrumReport {
        spectrum: MetricLaplacianSpectrum {
            eigenvalues: low,
            eigenvalues_full: full_eigs,
            basis_dim: n_b,
            n_points: n_pts,
            wall_clock_seconds: started.elapsed().as_secs_f64(),
            basis_exponents,
            eigenvectors_full,
            orthogonalized_basis_dim: evec_dim,
        },
        projection: projection.label().to_string(),
        full_basis_dim,
        projected_basis_dim: n_b,
        survival_fraction,
    }
}

// ---------------------------------------------------------------------
// P7.11 — modified Gram-Schmidt under L²(M) on the projected basis,
// followed by standard Hermitian EVP on the orthonormal basis.
// Mirrors P7.8's `run_orthogonalized` in
// `zero_modes_harmonic_z3xz3.rs`, adapted to the metric-Laplacian
// (function-space) Galerkin pencil (M, K) instead of the bundle
// (G, L) pencil.
// ---------------------------------------------------------------------

/// Orthogonalize the projected test basis under the L²(M) inner
/// product, deflate the numerical null space, then build K on the
/// orthonormal basis (M = identity by construction) and solve the
/// standard Hermitian EVP `K_orth v = λ v`.
///
/// Returns `(ascending eigenvalues, eigenvectors-in-orth-basis-row-major,
/// rank)`. The eigenvector storage uses `rank * rank` elements; callers
/// that need eigenvectors back in the full projected-basis coordinate
/// system should `Q · v_orth` themselves (we don't carry `Q` out of
/// this function — the chain matcher consumes only eigenvalues).
fn run_orthogonalized_metric_laplacian(
    m_mat: &[Complex64],
    k_mat: &[Complex64],
    n_b: usize,
    tol: f64,
    jacobi_max_sweeps: usize,
    jacobi_tol: f64,
) -> (Vec<f64>, Vec<Complex64>, usize) {
    if n_b == 0 {
        return (Vec::new(), Vec::new(), 0);
    }

    // `q[k]` is the coefficient vector of the k-th orthonormal basis
    // function in the original projected basis. `mq[k]` caches M · q[k]
    // so re-projection is fast.
    let mut q: Vec<Vec<Complex64>> = Vec::new();
    let mut mq: Vec<Vec<Complex64>> = Vec::new();
    let mut max_norm2_accepted: f64 = 0.0;

    for col in 0..n_b {
        // Start with the unit vector e_col.
        let mut v = vec![Complex64::new(0.0, 0.0); n_b];
        v[col] = Complex64::new(1.0, 0.0);

        // Two passes of modified Gram-Schmidt for numerical stability.
        for _pass in 0..2 {
            for k in 0..q.len() {
                // <q_k, v>_M = q_k^H · M · v = (M q_k)^H · v = mq[k]^H v
                let mut proj = Complex64::new(0.0, 0.0);
                for i in 0..n_b {
                    proj += mq[k][i].conj() * v[i];
                }
                for i in 0..n_b {
                    v[i] -= proj * q[k][i];
                }
            }
        }

        // Compute M v for residual norm and caching.
        let mut mv = vec![Complex64::new(0.0, 0.0); n_b];
        for i in 0..n_b {
            let mut acc = Complex64::new(0.0, 0.0);
            for j in 0..n_b {
                acc += m_mat[i * n_b + j] * v[j];
            }
            mv[i] = acc;
        }

        // Squared L²(M) norm: <v, v>_M = v^H M v = v^H · mv.
        let mut norm2 = Complex64::new(0.0, 0.0);
        for i in 0..n_b {
            norm2 += v[i].conj() * mv[i];
        }
        let n2 = norm2.re.max(0.0);

        // Deflate the numerical null space.
        let cutoff = if max_norm2_accepted > 0.0 {
            tol * max_norm2_accepted
        } else {
            tol
        };
        if n2 <= cutoff || !n2.is_finite() {
            continue;
        }

        let inv_n = 1.0 / n2.sqrt();
        let scale = Complex64::new(inv_n, 0.0);
        for i in 0..n_b {
            v[i] *= scale;
            mv[i] *= scale;
        }
        if n2 > max_norm2_accepted {
            max_norm2_accepted = n2;
        }
        q.push(v);
        mq.push(mv);
    }

    let rank = q.len();
    if rank == 0 {
        return (Vec::new(), Vec::new(), 0);
    }

    // Build K_orth[i, j] = q[i]^H · K · q[j] (the orthonormal-basis
    // restriction of K). With M = identity in this basis, the
    // generalised EVP collapses to the standard Hermitian EVP.
    let mut k_orth = vec![Complex64::new(0.0, 0.0); rank * rank];
    let mut kq = vec![Complex64::new(0.0, 0.0); n_b];
    for j in 0..rank {
        for i in 0..n_b {
            let mut acc = Complex64::new(0.0, 0.0);
            for kk in 0..n_b {
                acc += k_mat[i * n_b + kk] * q[j][kk];
            }
            kq[i] = acc;
        }
        for i in 0..rank {
            let mut acc = Complex64::new(0.0, 0.0);
            for kk in 0..n_b {
                acc += q[i][kk].conj() * kq[kk];
            }
            k_orth[i * rank + j] = acc;
        }
    }

    // Hermitian projection (kill antihermitian roundoff).
    for i in 0..rank {
        let z = k_orth[i * rank + i];
        k_orth[i * rank + i] = Complex64::new(z.re, 0.0);
        for j in (i + 1)..rank {
            let a = k_orth[i * rank + j];
            let b = k_orth[j * rank + i].conj();
            let avg = (a + b) * 0.5;
            k_orth[i * rank + j] = avg;
            k_orth[j * rank + i] = avg.conj();
        }
    }

    let (eigvals, evecs) =
        hermitian_jacobi_n(&k_orth, rank, jacobi_max_sweeps, jacobi_tol);
    (eigvals, evecs, rank)
}

// ---------------------------------------------------------------------
// Local copies of the Galerkin / linalg helpers from
// `metric_laplacian`. We duplicate (rather than `pub`-export the
// originals) to keep the projected path self-contained while the
// `chain_matcher` consumer of the original module is in deprecation /
// research-only status (see metric_laplacian.rs module-level note).
//
// Identical logic to metric_laplacian; any future refactor that
// consolidates the n×n complex Hermitian Jacobi + invert into a
// shared crate helper should fold both copies in.
// ---------------------------------------------------------------------

#[inline]
fn eval_monomial(point: &[Complex64; 8], exps: &[u32; 8]) -> Complex64 {
    let mut acc = Complex64::new(1.0, 0.0);
    for k in 0..8 {
        let e = exps[k];
        if e == 0 {
            continue;
        }
        let mut base = point[k];
        let mut ee = e;
        let mut term = Complex64::new(1.0, 0.0);
        while ee > 0 {
            if ee & 1 == 1 {
                term *= base;
            }
            ee >>= 1;
            if ee > 0 {
                base = base * base;
            }
        }
        acc *= term;
    }
    acc
}

#[inline]
fn d_dz(exps: &[u32; 8], k: usize) -> (f64, [u32; 8]) {
    if exps[k] == 0 {
        return (0.0, *exps);
    }
    let mut new_exps = *exps;
    let coeff = new_exps[k] as f64;
    new_exps[k] -= 1;
    (coeff, new_exps)
}

fn build_mass_matrix(
    basis_at: &[Complex64],
    n_b: usize,
    metric: &dyn MetricBackground,
) -> Vec<Complex64> {
    let n_pts = metric.n_points();
    let mut m = vec![Complex64::new(0.0, 0.0); n_b * n_b];
    let mut total_w = 0.0f64;
    for p in 0..n_pts {
        let w = metric.weight(p);
        if !w.is_finite() || w <= 0.0 {
            continue;
        }
        let omega = metric.omega(p);
        if !omega.re.is_finite() || !omega.im.is_finite() {
            continue;
        }
        let pw = Complex64::new(omega.norm_sqr() * w, 0.0);
        for alpha in 0..n_b {
            let sa = basis_at[alpha * n_pts + p];
            if sa.norm_sqr() == 0.0 {
                continue;
            }
            let sa_c = sa.conj();
            for beta in alpha..n_b {
                let sb = basis_at[beta * n_pts + p];
                let inc = sa_c * sb * pw;
                m[alpha * n_b + beta] += inc;
            }
        }
        total_w += w;
    }
    if total_w > 0.0 {
        let inv = 1.0 / total_w;
        for z in m.iter_mut() {
            *z *= Complex64::new(inv, 0.0);
        }
    }
    for i in 0..n_b {
        let z = m[i * n_b + i];
        m[i * n_b + i] = Complex64::new(z.re, 0.0);
        for j in (i + 1)..n_b {
            m[j * n_b + i] = m[i * n_b + j].conj();
        }
    }
    m
}

fn build_stiffness_matrix(
    d_basis_at: &[Complex64],
    n_b: usize,
    metric: &dyn MetricBackground,
) -> Vec<Complex64> {
    let n_pts = metric.n_points();
    let mut k = vec![Complex64::new(0.0, 0.0); n_b * n_b];
    let mut total_w = 0.0f64;
    for p in 0..n_pts {
        let w = metric.weight(p);
        if !w.is_finite() || w <= 0.0 {
            continue;
        }
        let omega = metric.omega(p);
        if !omega.re.is_finite() || !omega.im.is_finite() {
            continue;
        }
        let pw = Complex64::new(omega.norm_sqr() * w, 0.0);
        for i in 0..8 {
            for alpha in 0..n_b {
                let da = d_basis_at[(alpha * 8 + i) * n_pts + p];
                if da.norm_sqr() == 0.0 {
                    continue;
                }
                let da_c = da.conj();
                for beta in alpha..n_b {
                    let db = d_basis_at[(beta * 8 + i) * n_pts + p];
                    if db.norm_sqr() == 0.0 {
                        continue;
                    }
                    let inc = da_c * db * pw;
                    k[alpha * n_b + beta] += inc;
                }
            }
        }
        total_w += w;
    }
    if total_w > 0.0 {
        let inv = 1.0 / total_w;
        for z in k.iter_mut() {
            *z *= Complex64::new(inv, 0.0);
        }
    }
    for i in 0..n_b {
        let z = k[i * n_b + i];
        k[i * n_b + i] = Complex64::new(z.re.max(0.0), 0.0);
        for j in (i + 1)..n_b {
            k[j * n_b + i] = k[i * n_b + j].conj();
        }
    }
    k
}

fn matmul_complex(a: &[Complex64], b: &[Complex64], n: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = Complex64::new(0.0, 0.0);
            for k in 0..n {
                acc += a[i * n + k] * b[k * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn invert_hermitian_regularised(h: &[Complex64], n: usize, eps: f64) -> Vec<Complex64> {
    if n == 0 {
        return Vec::new();
    }
    let frob = h.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    let lambda = (frob * eps).max(1.0e-30);
    let two_n = 2 * n;
    let mut m = vec![Complex64::new(0.0, 0.0); n * two_n];
    for i in 0..n {
        for j in 0..n {
            m[i * two_n + j] = h[i * n + j];
        }
        m[i * two_n + i] += Complex64::new(lambda, 0.0);
        m[i * two_n + (n + i)] = Complex64::new(1.0, 0.0);
    }
    for k in 0..n {
        let mut pivot = k;
        let mut best = m[k * two_n + k].norm();
        for r in (k + 1)..n {
            let v = m[r * two_n + k].norm();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < 1.0e-300 {
            let mut id = vec![Complex64::new(0.0, 0.0); n * n];
            let inv = 1.0 / lambda.max(1.0e-30);
            for i in 0..n {
                id[i * n + i] = Complex64::new(inv, 0.0);
            }
            return id;
        }
        if pivot != k {
            for j in 0..two_n {
                m.swap(k * two_n + j, pivot * two_n + j);
            }
        }
        let pinv = Complex64::new(1.0, 0.0) / m[k * two_n + k];
        for j in 0..two_n {
            m[k * two_n + j] = m[k * two_n + j] * pinv;
        }
        for r in 0..n {
            if r == k {
                continue;
            }
            let factor = m[r * two_n + k];
            if factor.norm() == 0.0 {
                continue;
            }
            for j in 0..two_n {
                let kv = m[k * two_n + j];
                m[r * two_n + j] = m[r * two_n + j] - factor * kv;
            }
        }
    }
    let mut out = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = m[i * two_n + (n + j)];
        }
    }
    out
}

fn hermitian_jacobi_n(
    a_in: &[Complex64],
    n: usize,
    max_sweeps: usize,
    tol: f64,
) -> (Vec<f64>, Vec<Complex64>) {
    let mut a: Vec<Complex64> = a_in.to_vec();
    let mut v: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        v[i * n + i] = Complex64::new(1.0, 0.0);
    }
    for _sweep in 0..max_sweeps {
        let mut off = 0.0f64;
        for p in 0..n {
            for q in 0..n {
                if p != q {
                    off += a[p * n + q].norm_sqr();
                }
            }
        }
        if off.sqrt() < tol {
            break;
        }
        for p in 0..(n - 1) {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.norm() < tol {
                    continue;
                }
                let app = a[p * n + p].re;
                let aqq = a[q * n + q].re;
                let abs_pq = apq.norm();
                let phi = if abs_pq < 1.0e-300 { 0.0 } else { apq.arg() };
                let theta_arg = if (app - aqq).abs() < 1.0e-300 {
                    std::f64::consts::FRAC_PI_4
                } else {
                    0.5 * (2.0 * abs_pq / (app - aqq)).atan()
                };
                let c = theta_arg.cos();
                let s = theta_arg.sin();
                let cs_phi = Complex64::new(0.0, phi).exp() * Complex64::new(s, 0.0);
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = akp * Complex64::new(c, 0.0) + akq * cs_phi.conj();
                    a[k * n + q] = akq * Complex64::new(c, 0.0) - akp * cs_phi;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = apk * Complex64::new(c, 0.0) + aqk * cs_phi;
                    a[q * n + k] = aqk * Complex64::new(c, 0.0) - apk * cs_phi.conj();
                }
                a[p * n + q] = Complex64::new(0.0, 0.0);
                a[q * n + p] = Complex64::new(0.0, 0.0);
                a[p * n + p] = Complex64::new(a[p * n + p].re, 0.0);
                a[q * n + q] = Complex64::new(a[q * n + q].re, 0.0);
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = vkp * Complex64::new(c, 0.0) + vkq * cs_phi.conj();
                    v[k * n + q] = vkq * Complex64::new(c, 0.0) - vkp * cs_phi;
                }
            }
        }
    }
    let mut eig: Vec<(usize, f64)> = (0..n).map(|i| (i, a[i * n + i].re)).collect();
    eig.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut eig_sorted = vec![0.0f64; n];
    let mut v_sorted = vec![Complex64::new(0.0, 0.0); n * n];
    for (new_k, (old_k, val)) in eig.iter().enumerate() {
        eig_sorted[new_k] = *val;
        for i in 0..n {
            v_sorted[i * n + new_k] = v[i * n + *old_k];
        }
    }
    (eig_sorted, v_sorted)
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route34::metric_laplacian::build_test_basis;

    /// Schoen Z/3 × Z/3 projection survives ~1/9 of basis at large degree.
    #[test]
    fn schoen_projection_survival_at_d4() {
        let basis = build_test_basis(4);
        let projected = build_projected_basis_schoen(&basis);
        let frac = projected.len() as f64 / basis.len() as f64;
        // Expect 1/9 ≈ 0.111. Allow [0.05, 0.30] given finite-degree
        // edge effects (small bigrades with very few monomials may skew
        // the ratio in either direction).
        assert!(
            (0.05..=0.30).contains(&frac),
            "schoen projection survival {frac} outside [0.05, 0.30] at d=4 (basis_dim {}, projected {})",
            basis.len(),
            projected.len()
        );
        // Sanity: every survivor has zero α and β characters.
        for m in &projected {
            assert_eq!(alpha_character(&m.exponents), 0);
            assert_eq!(beta_character(&m.exponents), 0);
        }
    }

    /// TY Z/3 projection survives ~1/3 of basis at large degree.
    #[test]
    fn ty_projection_survival_at_d4() {
        let basis = build_test_basis(4);
        let projected = build_projected_basis_ty(&basis);
        let frac = projected.len() as f64 / basis.len() as f64;
        assert!(
            (0.20..=0.50).contains(&frac),
            "ty projection survival {frac} outside [0.20, 0.50] at d=4 (basis_dim {}, projected {})",
            basis.len(),
            projected.len()
        );
        for m in &projected {
            assert_eq!(ty_z3_character(&m.exponents), 0);
        }
    }

    /// Idempotency: reapplying the projector to its own output is a no-op.
    #[test]
    fn projector_idempotent_schoen() {
        let basis = build_test_basis(3);
        let p1 = build_projected_basis_schoen(&basis);
        let p2 = build_projected_basis_schoen(&p1);
        assert_eq!(p1.len(), p2.len());
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_eq!(a.exponents, b.exponents);
        }
    }

    /// P7.11 — modified Gram-Schmidt with deflation drops the
    /// numerical null space and yields a non-negative spectrum.
    ///
    /// We construct a rank-2 subspace embedded in a 3-dim ambient by
    /// taking M = diag(1, 1, ε) (third basis vector has tiny mass)
    /// and K = diag(2, 5, 100). The legacy M^{-1} K path would assign
    /// the third coordinate eigenvalue 100/ε ~ 1e10 and dominate the
    /// spectrum; the orthogonalized path drops it (norm² ε ~ 1e-12
    /// below cutoff) and reports {2, 5} cleanly.
    #[test]
    fn p7_11_orthogonalize_drops_null_space() {
        // n_b = 3.
        let n_b = 3;
        let mut m_mat = vec![Complex64::new(0.0, 0.0); n_b * n_b];
        let mut k_mat = vec![Complex64::new(0.0, 0.0); n_b * n_b];
        m_mat[0] = Complex64::new(1.0, 0.0);
        m_mat[4] = Complex64::new(1.0, 0.0);
        m_mat[8] = Complex64::new(1.0e-12, 0.0); // null direction
        k_mat[0] = Complex64::new(2.0, 0.0);
        k_mat[4] = Complex64::new(5.0, 0.0);
        k_mat[8] = Complex64::new(100.0, 0.0);

        let (eigvals, _evecs, rank) =
            run_orthogonalized_metric_laplacian(&m_mat, &k_mat, n_b, 1.0e-10, 128, 1.0e-12);
        assert_eq!(rank, 2, "deflation should drop the ε-mass coordinate");
        // K_orth = diag(2, 5) on the orthonormal basis (M is identity
        // on its non-null subspace, q vectors are e_0, e_1).
        assert!(
            (eigvals[0] - 2.0).abs() < 1.0e-9 && (eigvals[1] - 5.0).abs() < 1.0e-9,
            "expected {{2, 5}} on the rank-2 subspace, got {:?}",
            eigvals
        );
        // No spurious negative eigenvalue should appear.
        assert!(eigvals.iter().all(|&v| v > -1.0e-9), "spectrum has negative eigenvalues: {:?}", eigvals);
    }

    /// P7.11 — on a non-degenerate basis (M positive-definite, no
    /// near-null direction) the orthogonalized path agrees with the
    /// straight Hermitian eigenproblem of M^{-1/2} K M^{-1/2}.
    /// We compare to a hand-computed reference for diagonal M, K.
    #[test]
    fn p7_11_orthogonalize_agrees_on_well_conditioned_diag() {
        let n_b = 3;
        let mut m_mat = vec![Complex64::new(0.0, 0.0); n_b * n_b];
        let mut k_mat = vec![Complex64::new(0.0, 0.0); n_b * n_b];
        // M = diag(2, 3, 5), K = diag(8, 12, 25). Eigenvalues of M^{-1}K
        // are {4, 4, 5}. Orthogonalized path should report the same.
        m_mat[0] = Complex64::new(2.0, 0.0);
        m_mat[4] = Complex64::new(3.0, 0.0);
        m_mat[8] = Complex64::new(5.0, 0.0);
        k_mat[0] = Complex64::new(8.0, 0.0);
        k_mat[4] = Complex64::new(12.0, 0.0);
        k_mat[8] = Complex64::new(25.0, 0.0);
        let (eigvals, _evecs, rank) =
            run_orthogonalized_metric_laplacian(&m_mat, &k_mat, n_b, 1.0e-10, 128, 1.0e-12);
        assert_eq!(rank, 3);
        let mut sorted = eigvals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 4.0).abs() < 1.0e-9, "λ_0: {}", sorted[0]);
        assert!((sorted[1] - 4.0).abs() < 1.0e-9, "λ_1: {}", sorted[1]);
        assert!((sorted[2] - 5.0).abs() < 1.0e-9, "λ_2: {}", sorted[2]);
    }

    /// TY/Z3 character matches the in-line formula in
    /// `bin/p7_1_omega_fix_diagnostic.rs`.
    #[test]
    fn ty_character_matches_p7_1_inline_formula() {
        for &exps in &[
            [0u32, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [3, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
        ] {
            let p7_1 = (exps[1] + 2 * exps[2] + exps[5] + 2 * exps[6]) % 3;
            assert_eq!(ty_z3_character(&exps), p7_1);
        }
    }
}
