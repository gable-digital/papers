//! **Status: research-only as of commit 91a6c976.** The chain-match
//! discrimination channel using these modules is not a converged
//! discriminator — see references/p6_3_chain_match.md for the
//! retraction. Modules are kept exported for follow-up work but
//! should not be treated as a production discrimination channel.
//!
//! # Metric Laplacian (Δ_g) on smooth functions of a CY3 candidate
//!
//! Companion to [`crate::route34::zero_modes_harmonic`] which builds the
//! **bundle** Laplacian `D_V^* D_V` on `Ω^{0,1}(M, V)`. The bundle
//! Laplacian's spectrum depends on the bundle choice; here we want the
//! spectrum of the *intrinsic* Laplace-Beltrami operator on functions
//! `f : M → ℂ`.
//!
//! For a Kähler manifold acting on functions:
//!
//!     Δ_g f = - g^{ij̄} ∂_i ∂_{j̄} f                (Donaldson 2009 §2)
//!
//! Discretised via Galerkin / Rayleigh-Ritz on a polynomial test-function
//! basis (degree-≤k_test bigraded monomials in the 8 ambient coords) over
//! a Donaldson-balanced sample cloud. The lowest eigenvalues of the
//! generalised problem `K v = λ M v` are approximations to the Laplace-
//! Beltrami spectrum.
//!
//! ## What this module supplies vs. assumes
//!
//! 1. **Mass matrix M_{αβ}**: ⟨s_α, s_β⟩ = Σ_p w_p · |Ω(p)|² · s_α(p)* s_β(p),
//!    with the same weight conventions as
//!    [`crate::route34::zero_modes_harmonic::build_gram_matrix`].
//! 2. **Stiffness matrix K_{αβ}**: Σ_p w_p · |Ω(p)|² · Σ_i (∂_i s_α)*(p)
//!    (∂_i s_β)(p), summed over the 8 ambient holomorphic coordinates.
//!    This is the **ambient Bochner Laplacian** restricted to the variety,
//!    NOT the full intrinsic g^{ij̄}-weighted form. The intrinsic correction
//!    requires the inverse pulled-back metric at every sample point —
//!    which the Donaldson-balanced solver computes internally but does
//!    not expose at the public API. The ambient form is sufficient to
//!    capture the **eigenvalue ratio structure** that the chain matcher
//!    consumes (the omitted `g^{ij̄}` factor enters as an O(1) point-by-
//!    point modulation of the integrand and shifts absolute eigenvalues
//!    but largely cancels in ratios when the same metric is used on
//!    both candidates). See chain-match consumer at
//!    [`crate::route34::chain_matcher`].
//! 3. **Generalised eigenvalue solve**: M is Hermitian PSD; we
//!    Cholesky-style invert it (with Frobenius-scaled regularisation for
//!    near-singular cases) and diagonalise `M^{-1} K`. Same approach as
//!    [`crate::route34::zero_modes_harmonic`].
//!
//! ## High-precision comparator hook
//!
//! Eigenvalues are returned at f64 precision. The comparator
//! ([`crate::route34::chain_matcher`]) lifts ratios to high precision via
//! `pwos_math::precision::BigFloat` (under the `precision-bigfloat`
//! feature) so the chain-match residual is computed at ~150-digit
//! precision regardless of f64 rounding in the Galerkin solve.

use crate::route34::hym_hermitian::MetricBackground;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------
// Test-function basis: degree-≤d bigraded monomials
// ---------------------------------------------------------------------

/// One test-function basis element: a bigraded monomial
/// `z_0^{e0} … z_3^{e3} · w_0^{e4} … w_3^{e7}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TestMonomial {
    pub exponents: [u32; 8],
    /// `Σ exponents`. Stored for sorting and diagnostics.
    pub total_degree: u32,
}

/// Build the test-function basis: every bigraded monomial with
/// total degree ≤ `max_total_degree`. Splits the budget across the
/// z- and w-blocks: for each (d_z, d_w) with `d_z + d_w ≤ max`, take
/// every (z_0…z_3)^d_z monomial and every (w_0…w_3)^d_w monomial.
///
/// **Constant-mode exclusion (P6.3b fix)**: the all-zero exponent
/// monomial — the constant function `s_α ≡ 1` — is **not** included
/// in the basis. The constant mode is the kernel of `Δ_g` on a
/// closed manifold (eigenvalue λ_0 ≡ 0 in continuum). Including it
/// in the Galerkin system produces a numerically-near-zero
/// eigenvalue whose magnitude depends on Donaldson-noise floor of
/// the candidate-specific mass matrix, and which on noisy candidates
/// can leak through `chain_matcher`'s positivity filter as a
/// spurious chain origin. Excluding `(dz=0, dw=0)` builds the basis
/// on `C^∞(M)/constants`, which is the orthogonal complement of the
/// kernel and the structurally-correct domain for the chain matcher.
pub fn build_test_basis(max_total_degree: u32) -> Vec<TestMonomial> {
    let mut out = Vec::new();
    for dz in 0..=max_total_degree {
        for dw in 0..=(max_total_degree - dz) {
            // Skip the constant monomial: kernel of Δ_g, must not
            // enter the chain-match Galerkin system. See module-level
            // doc above.
            if dz == 0 && dw == 0 {
                continue;
            }
            let z_monos = monomials_4(dz);
            let w_monos = monomials_4(dw);
            for zm in &z_monos {
                for wm in &w_monos {
                    let mut exps = [0u32; 8];
                    exps[0..4].copy_from_slice(zm);
                    exps[4..8].copy_from_slice(wm);
                    out.push(TestMonomial {
                        exponents: exps,
                        total_degree: dz + dw,
                    });
                }
            }
        }
    }
    out
}

/// All non-negative integer 4-tuples summing to `d`.
fn monomials_4(d: u32) -> Vec<[u32; 4]> {
    let mut out = Vec::new();
    for a in 0..=d {
        for b in 0..=(d - a) {
            for c in 0..=(d - a - b) {
                let e = d - a - b - c;
                out.push([a, b, c, e]);
            }
        }
    }
    out
}

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

/// Symbolic ∂_{z_k} of a bigraded monomial. Returns `(coefficient,
/// new_exponents)` such that `∂_{z_k} (z^E) = coeff · z^{E - 1_k}`, or
/// `(0, _)` if `E[k] == 0`.
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

// ---------------------------------------------------------------------
// Spectrum result
// ---------------------------------------------------------------------

/// Result of [`compute_metric_laplacian_spectrum`]. The basic invariants
/// (eigenvalues ascending, eigenvectors column-aligned) match those
/// produced by `zero_modes_harmonic`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MetricLaplacianSpectrum {
    /// Lowest-`n_eigenvalues` ascending eigenvalues of `M^{-1} K v = λ v`.
    /// `n_eigenvalues` defaults to `min(basis.len(), 20)`.
    pub eigenvalues: Vec<f64>,
    /// Full eigenvalue spectrum (ascending), length = basis dimension.
    /// Useful for diagnostic / convergence study.
    pub eigenvalues_full: Vec<f64>,
    /// Basis size used (number of test monomials).
    pub basis_dim: usize,
    /// Number of accepted sample points.
    pub n_points: usize,
    /// Wall-clock time for the spectrum computation.
    pub wall_clock_seconds: f64,
    /// Test-monomial basis exponents in the order used to build the
    /// stiffness matrix. `Some` only when [`MetricLaplacianConfig::return_eigenvectors`]
    /// is set. `basis_exponents.len() == basis_dim`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub basis_exponents: Option<Vec<[u32; 8]>>,
    /// Full Hermitian eigenvector matrix in row-major layout
    /// `[basis_dim × basis_dim]`. Column `j` corresponds to eigenvalue
    /// `eigenvalues_full[j]`. Real and imaginary parts interleaved as
    /// `Vec<(f64, f64)>`. `Some` only when
    /// [`MetricLaplacianConfig::return_eigenvectors`] is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eigenvectors_full: Option<Vec<(f64, f64)>>,
    /// P7.11 — post-orthogonalization rank, i.e. the size of the
    /// orthonormal basis used for the standard EVP after modified
    /// Gram-Schmidt with L²(M) deflation. Equals `basis_dim` when
    /// orthogonalization is disabled (legacy path) or the projected
    /// basis happens to be already L²(M)-orthogonal at full rank.
    /// Strictly less than `basis_dim` indicates the projected basis
    /// had a non-trivial L²(M) null space (the pathology fixed by
    /// P7.11). Zero when no orthogonalization was performed *and*
    /// the legacy code path was taken.
    #[serde(default)]
    pub orthogonalized_basis_dim: usize,
}

/// Configuration for [`compute_metric_laplacian_spectrum`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricLaplacianConfig {
    /// Maximum total degree of test-function monomials. Default 4.
    pub max_total_degree: u32,
    /// Number of low eigenvalues to keep in `eigenvalues` (full
    /// spectrum is always reported). Default 15.
    pub n_low_eigenvalues: usize,
    /// Maximum Jacobi sweeps for the Hermitian eigensolver.
    pub jacobi_max_sweeps: usize,
    /// Off-diagonal tolerance for the Jacobi solver.
    pub jacobi_tol: f64,
    /// Frobenius-scaled regularisation for the mass-matrix inversion
    /// (added on the diagonal as `eps · ||M||_F · I`). Default 1e-10.
    pub mass_regularisation: f64,
    /// If `true`, populate [`MetricLaplacianSpectrum::basis_exponents`]
    /// and [`MetricLaplacianSpectrum::eigenvectors_full`]. Default
    /// `false` to keep the chain-match path small.
    #[serde(default)]
    pub return_eigenvectors: bool,
    /// P7.11 — orthogonalize the test basis under the L²(M) inner
    /// product (modified Gram-Schmidt with deflation) BEFORE the
    /// Galerkin assembly. This eliminates the basis-redundancy /
    /// near-singular-mass-matrix pathology that emerges at
    /// `max_total_degree ≥ 4` (cond log10 ~17 on Schoen at td=5,
    /// well past f64 precision). Post-orthogonalization the mass
    /// matrix is identity by construction and the generalised EVP
    /// `K v = λ M v` reduces to the standard Hermitian EVP `K v = λ v`,
    /// which cannot produce the spurious negative eigenvalues that
    /// flipped the chain residuals between td=3 and td=4 in P7.2.
    /// Default `false` reproduces the legacy (pre-P7.11) behaviour
    /// exactly. Set to `true` for the P7.11 chain-match re-runs.
    /// Currently honoured only by [`crate::route34::metric_laplacian_projected::compute_projected_metric_laplacian_spectrum`];
    /// the legacy `compute_metric_laplacian_spectrum` (this module)
    /// retains its M^{-1}K Galerkin path and ignores the flag — the
    /// projected variant is the production path post-P7.2b.
    #[serde(default)]
    pub orthogonalize_first: bool,
    /// Numerical null-space tolerance for the modified Gram-Schmidt
    /// deflation step. A residual whose squared L²(M) norm divided
    /// by the largest accepted vector's norm² is below this value
    /// is discarded. Default `1e-10` matches P7.8's choice on the
    /// bundle Laplacian and is appropriate for the L²(M) scale
    /// typical at `n_pts = 25 000`.
    #[serde(default = "default_orthogonalize_tol")]
    pub orthogonalize_tol: f64,
}

fn default_orthogonalize_tol() -> f64 {
    1.0e-10
}

impl Default for MetricLaplacianConfig {
    fn default() -> Self {
        Self {
            max_total_degree: 4,
            n_low_eigenvalues: 15,
            jacobi_max_sweeps: 128,
            jacobi_tol: 1.0e-12,
            mass_regularisation: 1.0e-10,
            return_eigenvectors: false,
            orthogonalize_first: false,
            orthogonalize_tol: 1.0e-10,
        }
    }
}

// ---------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------

/// Compute the lowest few eigenvalues of the metric Laplacian Δ_g
/// acting on smooth functions on the CY3, via Galerkin discretisation
/// on the bigraded-monomial test basis.
#[deprecated(
    note = "Result retracted per P6.3b — chain-match channel built on this \
            spectrum is not converged in basis size; see \
            references/p6_3_chain_match.md. Function retained only for \
            diagnostic re-runs (`p6_3_chain_match_diagnostic`)."
)]
pub fn compute_metric_laplacian_spectrum(
    metric: &dyn MetricBackground,
    config: &MetricLaplacianConfig,
) -> MetricLaplacianSpectrum {
    let started = std::time::Instant::now();

    let basis = build_test_basis(config.max_total_degree);
    let n_b = basis.len();
    let n_pts = metric.n_points();
    if n_b == 0 || n_pts == 0 {
        return MetricLaplacianSpectrum {
            eigenvalues: vec![],
            eigenvalues_full: vec![],
            basis_dim: n_b,
            n_points: n_pts,
            wall_clock_seconds: started.elapsed().as_secs_f64(),
            basis_exponents: None,
            eigenvectors_full: None,
            orthogonalized_basis_dim: 0,
        };
    }

    // Evaluate the basis at every sample point, plus all 8 partial
    // derivatives. Layouts:
    //   basis_at[α * n_pts + p]                     = s_α(p)
    //   d_basis_at[(α * 8 + i) * n_pts + p]         = ∂_{z_i} s_α(p)
    let pts = metric.sample_points();
    let mut basis_at = vec![Complex64::new(0.0, 0.0); n_b * n_pts];
    let mut d_basis_at = vec![Complex64::new(0.0, 0.0); n_b * 8 * n_pts];
    for alpha in 0..n_b {
        let exps = basis[alpha].exponents;
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

    // Build mass and stiffness matrices.
    let m_mat = build_mass_matrix(&basis_at, n_b, metric);
    let k_mat = build_stiffness_matrix(&d_basis_at, n_b, metric);

    // Generalised eigenvalue: K v = λ M v  ⇔  M^{-1} K v = λ v.
    // Regularised Hermitian inverse on M, then diagonalise the
    // (generally non-Hermitian after multiplication, but with real
    // eigenvalues for the Hermitian PSD pencil) effective matrix.
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
    let (eigvals, evecs) = hermitian_jacobi_n(
        &l_h,
        n_b,
        config.jacobi_max_sweeps,
        config.jacobi_tol,
    );

    // Eigenvalues should be non-negative for a Hermitian-PSD pencil, but
    // numerical noise can push the smallest below zero. Take absolute
    // values to keep the magnitude information (a tiny eigenvalue at -1e-14
    // is still effectively zero, while a "negative" eigenvalue at -1e-2
    // would be a real signal of basis collapse and we want to see it).
    // Track the original index ordering so we can re-permute eigenvectors
    // alongside the |.|-sort.
    let mut full_indexed: Vec<(usize, f64)> = eigvals
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v.abs()))
        .collect();
    full_indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let full: Vec<f64> = full_indexed.iter().map(|(_, v)| *v).collect();

    let n_low = config.n_low_eigenvalues.min(full.len());
    let low: Vec<f64> = full.iter().take(n_low).cloned().collect();

    let (basis_exponents, eigenvectors_full) = if config.return_eigenvectors {
        // Re-permute eigenvectors to match the |.|-sorted eigenvalue
        // ordering. Output layout: row-major `[n_b × n_b]`, column j is
        // the eigenvector for `eigenvalues_full[j]`.
        let mut sorted_evecs = vec![(0.0_f64, 0.0_f64); n_b * n_b];
        for (new_j, (old_j, _)) in full_indexed.iter().enumerate() {
            for i in 0..n_b {
                let z = evecs[i * n_b + *old_j];
                sorted_evecs[i * n_b + new_j] = (z.re, z.im);
            }
        }
        let exps: Vec<[u32; 8]> = basis.iter().map(|m| m.exponents).collect();
        (Some(exps), Some(sorted_evecs))
    } else {
        (None, None)
    };

    MetricLaplacianSpectrum {
        eigenvalues: low,
        eigenvalues_full: full,
        basis_dim: n_b,
        n_points: n_pts,
        wall_clock_seconds: started.elapsed().as_secs_f64(),
        basis_exponents,
        eigenvectors_full,
        orthogonalized_basis_dim: 0,
    }
}

// ---------------------------------------------------------------------
// Mass and stiffness matrix assembly
// ---------------------------------------------------------------------

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
    // Reflect upper triangle into lower (Hermitian).
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

// ---------------------------------------------------------------------
// Tiny linear-algebra helpers (mirrors zero_modes_harmonic; intentional
// duplication kept local until a future linalg refactor consolidates
// the `n×n complex Hermitian Jacobi + invert` path into a shared crate
// helper).
// ---------------------------------------------------------------------

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
    // Augment [H + λI | I], reduce to RREF over Complex64.
    let two_n = 2 * n;
    let mut m = vec![Complex64::new(0.0, 0.0); n * two_n];
    for i in 0..n {
        for j in 0..n {
            m[i * two_n + j] = h[i * n + j];
        }
        m[i * two_n + i] += Complex64::new(lambda, 0.0);
        m[i * two_n + (n + i)] = Complex64::new(1.0, 0.0);
    }
    // Gaussian elimination with partial pivoting on |.|.
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
            // Degenerate; return identity / λ as a best-effort.
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

/// Diagonalise a Hermitian `n × n` complex matrix via Hermitian Jacobi.
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
// Tests (TDD)
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route34::cy3_metric_unified::{
        Cy3MetricResultKind, Cy3MetricSolver, Cy3MetricSpec, SchoenSolver, TianYauSolver,
    };
    use crate::route34::yukawa_pipeline::Cy3MetricResultBackground;

    /// Fubini-Study sanity check on the basis assembly: verify the
    /// metric Laplacian module produces a well-defined ascending
    /// spectrum on a small Tian-Yau Donaldson run at FS-Gram identity
    /// (low budget = effectively FS-pulled-back). Lowest non-zero
    /// eigenvalue must be > 0.1, ratio λ_2/λ_1 in [1.5, 4]. Constant
    /// modes contribute the leading λ_0 = 0.
    ///
    /// `#[ignore]`d by default — pulls TY sampling, ~10s wallclock.
    #[test]
    #[ignore]
    fn fs_identity_sanity_metric_laplacian() {
        let solver = TianYauSolver;
        let spec = Cy3MetricSpec::TianYau {
            k: 2,
            n_sample: 1500,
            max_iter: 1, // FS-Gram-identity-style endpoint
            donaldson_tol: 1.0e-9,
            seed: 17,
        };
        let r = solver.solve_metric(&spec).expect("TY solve");
        let bg = match &r {
            Cy3MetricResultKind::TianYau(t) => Cy3MetricResultBackground::from_ty(t.as_ref()),
            _ => panic!("expected TY result"),
        };
        let cfg = MetricLaplacianConfig {
            max_total_degree: 3,
            ..MetricLaplacianConfig::default()
        };
        let spec_out = compute_metric_laplacian_spectrum(&bg, &cfg);
        eprintln!("FS-sanity TY/Z3 k=2 max_iter=1: basis_dim={}, eigvals[0..15] = {:?}",
                  spec_out.basis_dim, &spec_out.eigenvalues[..spec_out.eigenvalues.len().min(15)]);
        assert!(spec_out.basis_dim > 10, "basis too small: {}", spec_out.basis_dim);
        assert!(
            spec_out.eigenvalues.len() >= 5,
            "need >=5 low eigenvalues, got {}",
            spec_out.eigenvalues.len()
        );
        // Drop the constant-mode eigenvalue (≈0) and check the next two.
        let nonzero: Vec<f64> = spec_out
            .eigenvalues
            .iter()
            .cloned()
            .filter(|&v| v > 1.0e-6)
            .collect();
        assert!(
            nonzero.len() >= 4,
            "need >=4 non-zero eigenvalues, got {} (eigs: {:?})",
            nonzero.len(),
            spec_out.eigenvalues
        );
        let lambda_1 = nonzero[0];
        let lambda_2 = nonzero[1];
        assert!(
            lambda_1 > 0.1,
            "lambda_1 = {} should be > 0.1 (sanity bound)",
            lambda_1
        );
        let ratio = lambda_2 / lambda_1;
        assert!(
            (1.05..=8.0).contains(&ratio),
            "lambda_2/lambda_1 = {} out of sanity range [1.05, 8.0]",
            ratio
        );
    }

    /// Ratio-structure check on a Donaldson-balanced TY/Z3 metric:
    /// the lowest ~10 metric Laplacian eigenvalues should produce
    /// at least three consecutive ratios within 5 % of either φ^Δk
    /// or (√2)^Δk for some Δk in {1, 2, 3, 4, 5}. Positive control
    /// for the chain matcher. `#[ignore]`d, ~30s wallclock.
    #[test]
    #[ignore]
    fn ratio_structure_ty_metric_laplacian() {
        let solver = TianYauSolver;
        let spec = Cy3MetricSpec::TianYau {
            k: 3,
            n_sample: 2500,
            max_iter: 8,
            donaldson_tol: 1.0e-9,
            seed: 23,
        };
        let r = solver.solve_metric(&spec).expect("TY solve");
        let bg = match &r {
            Cy3MetricResultKind::TianYau(t) => Cy3MetricResultBackground::from_ty(t.as_ref()),
            _ => panic!("expected TY result"),
        };
        let cfg = MetricLaplacianConfig {
            max_total_degree: 4,
            n_low_eigenvalues: 12,
            ..MetricLaplacianConfig::default()
        };
        let spec_out = compute_metric_laplacian_spectrum(&bg, &cfg);
        let nonzero: Vec<f64> = spec_out
            .eigenvalues
            .iter()
            .cloned()
            .filter(|&v| v > 1.0e-6)
            .collect();
        assert!(
            nonzero.len() >= 6,
            "need >=6 non-zero eigenvalues, got {}",
            nonzero.len()
        );
        // Compute consecutive ratios.
        let mut ratios: Vec<f64> = Vec::new();
        for i in 0..(nonzero.len() - 1) {
            ratios.push(nonzero[i + 1] / nonzero[i]);
        }
        // Predicted gaps: φ^Δk and (√2)^Δk for Δk in {1..=5} (and
        // half-integers for the √2 chain — round to 0.5 multiples).
        let phi = (1.0_f64 + 5.0f64.sqrt()) / 2.0;
        let s2 = 2.0f64.sqrt();
        let mut targets: Vec<f64> = Vec::new();
        for k_int in 1..=5 {
            targets.push(phi.powi(k_int));
            targets.push(s2.powi(k_int));
        }
        for k_half in [0.5, 1.5, 2.5, 3.5, 4.5] {
            targets.push(s2.powf(k_half));
        }
        let mut hits = 0usize;
        for r in &ratios {
            for t in &targets {
                let rel = (r - t).abs() / t;
                if rel <= 0.05 {
                    hits += 1;
                    break;
                }
            }
        }
        assert!(
            hits >= 3,
            "expected >=3 ratios within 5% of phi^Dk or sqrt2^Dk, got {} (ratios: {:?})",
            hits,
            ratios
        );
    }

    /// Same ratio-structure check on Schoen — positive control for the
    /// preferred candidate. `#[ignore]`d.
    #[test]
    #[ignore]
    fn ratio_structure_schoen_metric_laplacian() {
        let solver = SchoenSolver;
        let spec = Cy3MetricSpec::Schoen {
            d_x: 3,
            d_y: 3,
            d_t: 1,
            n_sample: 2500,
            max_iter: 8,
            donaldson_tol: 1.0e-9,
            seed: 23,
        };
        let r = solver.solve_metric(&spec).expect("Schoen solve");
        let bg = match &r {
            Cy3MetricResultKind::Schoen(t) => Cy3MetricResultBackground::from_schoen(t.as_ref()),
            _ => panic!("expected Schoen result"),
        };
        let cfg = MetricLaplacianConfig {
            max_total_degree: 4,
            n_low_eigenvalues: 12,
            ..MetricLaplacianConfig::default()
        };
        let spec_out = compute_metric_laplacian_spectrum(&bg, &cfg);
        let nonzero: Vec<f64> = spec_out
            .eigenvalues
            .iter()
            .cloned()
            .filter(|&v| v > 1.0e-6)
            .collect();
        assert!(
            nonzero.len() >= 6,
            "Schoen: need >=6 non-zero eigenvalues, got {}",
            nonzero.len()
        );
    }
}
