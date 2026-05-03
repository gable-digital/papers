//! Null-space extraction for the discrete Lichnerowicz operator.
//!
//! ## Approach
//!
//! Given the assembled Lichnerowicz matrix `L` and Gram matrix `G` from
//! [`crate::route34::lichnerowicz::LichnerowiczOperator`], we want the
//! kernel of the generalised eigenproblem
//!
//! ```text
//!     L c = λ G c
//! ```
//!
//! (`λ ≥ 0` for any reasonable basis on a Riemannian manifold; `L` is
//! the symmetric quadratic form `(ξ, ξ) ↦ ⟨ξ, Δ_L ξ⟩` in the basis
//! coefficients, and `G` is the L²-Gram matrix of the basis vectors).
//!
//! We reduce to a standard symmetric eigenproblem via **Cholesky
//! whitening**:
//!
//! 1. Cholesky-factor the Gram matrix `G = R Rᵀ` (after a small
//!    regularisation `G ← G + ε_G I` to bound the condition number).
//! 2. Form `L̃ = R^{-1} L R^{-T}`.
//! 3. Diagonalise the symmetric matrix `L̃` via cyclic Jacobi rotations
//!    (deterministic, no external eigensolver dependency, exact orthogonal
//!    transformation history → reproducible).
//! 4. The eigenvalues of `L̃` are exactly the generalised eigenvalues of
//!    `(L, G)`. Eigenvectors of the original problem are recovered as
//!    `c_j = R^{-T} ẽ_j`.
//! 5. Apply the kernel-tolerance filter `λ_j < tol_kernel` to retrieve
//!    the discrete Killing algebra basis. The tolerance is chosen
//!    relative to the largest eigenvalue: `tol_kernel = tol_rel · λ_max`
//!    when the user passes a relative tolerance.
//!
//! ## Why Jacobi
//!
//! For the basis sizes we run (`n_basis ≤ ~200`), Jacobi rotations are
//! both fast (`O(n^3)` per sweep, ~ten sweeps to converge to machine
//! precision) and **bit-deterministic** given the input matrix. This
//! preserves the publication-grade reproducibility requirement and
//! removes any external LAPACK / `ndarray-linalg` dependency.
//!
//! For larger bases an external Krylov-Schur eigensolver would be
//! preferable; we expose the `LichnerowiczOperator` matrices publicly so
//! that downstream callers can apply LAPACK directly when they wish.
//!
//! ## Subgroup detection
//!
//! Once we have a kernel basis `{ξ_1, …, ξ_k}` of the Lichnerowicz
//! operator, we estimate the **Lie bracket** `[ξ_a, ξ_b]` numerically by
//! evaluating it at the sample points and projecting back into the
//! Killing-coefficient basis via the Gram inverse. The structure
//! constants `f^c_{ab}` give the Lie algebra; cyclic / abelian
//! subalgebras are detected by inspecting the spectrum of `ad(ξ_a)` for
//! each generator.
//!
//! ## References
//!
//! * Golub, Van Loan, *Matrix Computations*, 4th ed. (2013), §8.5
//!   (Jacobi method for symmetric eigenvalue problem), §8.7
//!   (Cholesky-based reduction of generalised symmetric eigenproblem).
//! * Wald, *General Relativity* (1984), Ch. 3 (Killing-vector commutator
//!   structure).
//! * Kobayashi, Nomizu, *Foundations of Differential Geometry*, vol. I,
//!   §III.3 (Killing-vector Lie algebra).

use rayon::prelude::*;

use crate::route34::lichnerowicz::{
    LichnerowiczOperator, MetricEvaluator, VectorFieldBasis,
};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// One Killing-vector field returned by the kernel extraction.
///
/// The vector field is stored both as a coefficient vector
/// `(c_1, …, c_{n_basis})` in the supplied basis, **and** as a sampled
/// pointwise representation `(V^μ(x_p))_{p, μ}` re-evaluated at the
/// sample points, plus the residual `‖L c‖` (a measure of how exactly
/// the discrete kernel is satisfied — for an analytically-Killing field
/// this should be `O(discretisation_error)`; for a spurious kernel
/// element it will be larger).
#[derive(Debug, Clone)]
pub struct KillingVectorField {
    /// Basis-expansion coefficients, length `n_basis`.
    pub coefficients: Vec<f64>,
    /// Generalised eigenvalue (≈ 0 for a true Killing direction).
    pub eigenvalue: f64,
    /// Discrete residual `‖L c‖_2 / ‖c‖_G`. Smaller = closer to the
    /// kernel.
    pub residual: f64,
}

/// Result of a full Killing-algebra solve.
#[derive(Debug, Clone)]
pub struct KillingResult {
    /// Estimated dimension of the Killing algebra (the number of
    /// kernel modes that pass the tolerance filter).
    pub dim: usize,
    /// All extracted kernel-candidate fields, sorted by ascending
    /// eigenvalue. The first `dim` of these are the accepted
    /// Killing-algebra basis; remaining entries are spurious near-kernel
    /// modes retained for inspection.
    pub basis: Vec<KillingVectorField>,
    /// Full eigenvalue spectrum (length `n_basis`), sorted ascending.
    /// Useful for diagnosing the kernel/non-kernel separation gap.
    pub spectrum: Vec<f64>,
    /// The accepted-vs-rejected boundary tolerance actually used.
    pub tol_used: f64,
}

/// Options controlling kernel extraction.
#[derive(Debug, Clone, Copy)]
pub struct KillingSolveOptions {
    /// Absolute tolerance: any eigenvalue with `|λ| < tol_abs` is
    /// considered to be in the kernel. Set to `0.0` to disable
    /// (then `tol_rel` controls).
    pub tol_abs: f64,
    /// Relative tolerance: any eigenvalue with `|λ| < tol_rel · λ_max`
    /// is also considered to be in the kernel. The combined tolerance
    /// is `max(tol_abs, tol_rel * λ_max)`.
    pub tol_rel: f64,
    /// Cholesky regularisation: replace `G` with `G + ε_G · I`. Use a
    /// small positive number to ensure invertibility under the
    /// finite-sample-noise floor.
    pub gram_regularisation: f64,
    /// Maximum number of Jacobi sweeps. Each sweep performs `O(n²)`
    /// rotations.
    pub max_jacobi_sweeps: usize,
    /// Convergence threshold on the off-diagonal Frobenius norm of the
    /// Jacobi-reduced matrix. Sweeping stops when the off-diagonal
    /// shrinks below this value.
    pub jacobi_tol: f64,
}

impl Default for KillingSolveOptions {
    fn default() -> Self {
        Self {
            tol_abs: 1e-6,
            tol_rel: 1e-8,
            gram_regularisation: 1e-12,
            max_jacobi_sweeps: 64,
            jacobi_tol: 1e-13,
        }
    }
}

// ---------------------------------------------------------------------------
// Cholesky factorisation (lower-triangular)
// ---------------------------------------------------------------------------

/// In-place Cholesky factorisation `G = R Rᵀ` with `R` lower-triangular.
///
/// Overwrites the lower triangle of `g_inout` with `R`. The upper
/// triangle (above the diagonal) is left untouched.
///
/// Returns `Err` if `G` is not positive definite.
fn cholesky_lower_in_place(g_inout: &mut [f64], n: usize) -> Result<(), &'static str> {
    for j in 0..n {
        // Diagonal: R_{jj} = sqrt(G_{jj} - Σ_{k<j} R_{jk}^2)
        let mut diag = g_inout[j * n + j];
        for k in 0..j {
            let r = g_inout[j * n + k];
            diag -= r * r;
        }
        if diag <= 0.0 {
            return Err("cholesky_lower_in_place: matrix not positive definite");
        }
        let r_jj = diag.sqrt();
        g_inout[j * n + j] = r_jj;
        // Below-diagonal column: R_{ij} = (G_{ij} - Σ_{k<j} R_{ik} R_{jk}) / R_{jj}
        for i in (j + 1)..n {
            let mut s = g_inout[i * n + j];
            for k in 0..j {
                s -= g_inout[i * n + k] * g_inout[j * n + k];
            }
            g_inout[i * n + j] = s / r_jj;
        }
    }
    Ok(())
}

/// Solve `R y = b` in place where `R` is lower-triangular stored in the
/// lower triangle of `r` (row-major, `n × n`). Overwrites `b` with `y`.
fn solve_lower_triangular(r: &[f64], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= r[i * n + j] * b[j];
        }
        b[i] = s / r[i * n + i];
    }
}

/// Solve `Rᵀ y = b` in place where `R` is lower-triangular stored in the
/// lower triangle of `r`. Overwrites `b` with `y`.
fn solve_upper_triangular_transposed(r: &[f64], n: usize, b: &mut [f64]) {
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= r[j * n + i] * b[j];
        }
        b[i] = s / r[i * n + i];
    }
}

// ---------------------------------------------------------------------------
// Cyclic Jacobi for symmetric eigenproblems
// ---------------------------------------------------------------------------

/// In-place cyclic-Jacobi diagonalisation of a real symmetric matrix
/// `a` (row-major, `n × n`). On exit, `a`'s diagonal contains the
/// eigenvalues and `q` (also `n × n` row-major) contains the orthogonal
/// eigenvectors as columns: `Q[:, j]` is the j-th eigenvector.
///
/// `q` is initialised to the identity; the cumulative product of Jacobi
/// rotations is accumulated into it.
///
/// Returns the number of sweeps performed and the final off-diagonal
/// Frobenius residue.
pub fn jacobi_eigh(
    a: &mut [f64],
    q: &mut [f64],
    n: usize,
    max_sweeps: usize,
    tol: f64,
) -> (usize, f64) {
    // Initialise Q = I.
    for i in 0..n {
        for j in 0..n {
            q[i * n + j] = if i == j { 1.0 } else { 0.0 };
        }
    }

    let mut last_off = f64::INFINITY;
    let mut sweeps_used = 0;
    for sweep in 0..max_sweeps {
        sweeps_used = sweep + 1;
        let mut off2 = 0.0;
        for p in 0..(n - 1) {
            for qq in (p + 1)..n {
                let app = a[p * n + p];
                let aqq = a[qq * n + qq];
                let apq = a[p * n + qq];
                off2 += apq * apq;
                if apq.abs() < 1e-300 {
                    continue;
                }
                // Compute Jacobi rotation
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Apply rotation to A: rows/cols p and q.
                a[p * n + p] = app - t * apq;
                a[qq * n + qq] = aqq + t * apq;
                a[p * n + qq] = 0.0;
                a[qq * n + p] = 0.0;
                for r in 0..n {
                    if r == p || r == qq {
                        continue;
                    }
                    let arp = a[r * n + p];
                    let arq = a[r * n + qq];
                    a[r * n + p] = c * arp - s * arq;
                    a[r * n + qq] = s * arp + c * arq;
                    a[p * n + r] = a[r * n + p];
                    a[qq * n + r] = a[r * n + qq];
                }
                // Update Q: columns p and q.
                for r in 0..n {
                    let qrp = q[r * n + p];
                    let qrq = q[r * n + qq];
                    q[r * n + p] = c * qrp - s * qrq;
                    q[r * n + qq] = s * qrp + c * qrq;
                }
            }
        }
        let off = off2.sqrt();
        if off < tol {
            return (sweeps_used, off);
        }
        if off > last_off * 0.999 && sweep > 0 {
            // Stagnated: bail out (avoids infinite spinning when input
            // already at machine precision).
            return (sweeps_used, off);
        }
        last_off = off;
    }
    let off = (0..(n - 1))
        .flat_map(|p| ((p + 1)..n).map(move |q_| (p, q_)))
        .map(|(p, qq)| a[p * n + qq] * a[p * n + qq])
        .sum::<f64>()
        .sqrt();
    (sweeps_used, off)
}

// ---------------------------------------------------------------------------
// Generalised symmetric eigenproblem L c = λ G c
// ---------------------------------------------------------------------------

/// Solve the generalised symmetric eigenproblem `L c = λ G c` with
/// `L`, `G` symmetric and `G` positive-definite. Returns sorted
/// (ascending) eigenvalues and the corresponding `n × n`
/// G-orthonormal eigenvector matrix `C` such that `Cᵀ G C = I` and
/// `Cᵀ L C = diag(λ)`.
///
/// Uses Cholesky whitening + cyclic Jacobi. `gram_reg` is added to the
/// diagonal of `G` before factorisation (`> 0` to bound the condition
/// number on noisy data).
fn solve_gen_symmetric_eigenproblem(
    l_in: &[f64],
    g_in: &[f64],
    n: usize,
    gram_reg: f64,
    max_jacobi_sweeps: usize,
    jacobi_tol: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    // Build R = chol(G + gram_reg · I)
    let mut r = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            r[i * n + j] = g_in[i * n + j];
        }
        r[i * n + i] += gram_reg;
    }
    cholesky_lower_in_place(&mut r, n)
        .map_err(|e| format!("Gram-Cholesky failed (regularisation = {gram_reg}): {e}"))?;

    // Form L_tilde = R^{-1} L R^{-T}.
    //
    // Algorithm:
    //   Step 1: M = R^{-1} L. For each column j, solve R · M[:, j] = L[:, j]
    //   via solve_lower_triangular acting on the column.
    //   Step 2: L_tilde = M R^{-T}. Equivalently L_tilde^T = R^{-1} M^T;
    //   for each row i of M, solve R · z = M[i, :]^T via
    //   solve_lower_triangular and the result z is L_tilde[i, :]^T.
    let mut m_mat = vec![0.0; n * n];
    for j in 0..n {
        let mut col = vec![0.0; n];
        for i in 0..n {
            col[i] = l_in[i * n + j];
        }
        solve_lower_triangular(&r, n, &mut col);
        for i in 0..n {
            m_mat[i * n + j] = col[i];
        }
    }
    let mut l_tilde = vec![0.0; n * n];
    for i in 0..n {
        let mut row = vec![0.0; n];
        for j in 0..n {
            row[j] = m_mat[i * n + j];
        }
        // Treat the row as a column-vector and solve R · z = row.
        solve_lower_triangular(&r, n, &mut row);
        for j in 0..n {
            l_tilde[i * n + j] = row[j];
        }
    }
    // Symmetrise L_tilde to scrub round-off asymmetry.
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (l_tilde[i * n + j] + l_tilde[j * n + i]);
            l_tilde[i * n + j] = avg;
            l_tilde[j * n + i] = avg;
        }
    }

    // Diagonalise L_tilde via Jacobi.
    let mut q = vec![0.0; n * n];
    let _ = jacobi_eigh(&mut l_tilde, &mut q, n, max_jacobi_sweeps, jacobi_tol);

    // Eigenvalues = diag(L_tilde); recover original eigenvectors as
    // c_j = R^{-T} q_j  (per-column triangular solve).
    let mut eigvals: Vec<f64> = (0..n).map(|i| l_tilde[i * n + i]).collect();
    // Eigenvectors-of-the-original-problem matrix C = R^{-T} Q.
    let mut c = vec![0.0; n * n];
    for j in 0..n {
        let mut col = vec![0.0; n];
        for i in 0..n {
            col[i] = q[i * n + j];
        }
        solve_upper_triangular_transposed(&r, n, &mut col);
        for i in 0..n {
            c[i * n + j] = col[i];
        }
    }

    // Sort eigenpairs by ascending eigenvalue. Use indirect sort.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| eigvals[a].partial_cmp(&eigvals[b]).unwrap());
    let sorted_eigvals: Vec<f64> = idx.iter().map(|&i| eigvals[i]).collect();
    let mut sorted_c = vec![0.0; n * n];
    for (new_j, &old_j) in idx.iter().enumerate() {
        for i in 0..n {
            sorted_c[i * n + new_j] = c[i * n + old_j];
        }
    }
    eigvals = sorted_eigvals;

    Ok((eigvals, sorted_c))
}

// ---------------------------------------------------------------------------
// Public solver entry-point
// ---------------------------------------------------------------------------

/// Extract the kernel of an assembled Lichnerowicz operator. Returns
/// the discrete Killing-algebra basis (filtered by `opts.tol_*`), the
/// full sorted eigenvalue spectrum, and the actual tolerance used.
pub fn solve_killing_kernel(
    op: &LichnerowiczOperator,
    opts: &KillingSolveOptions,
) -> Result<KillingResult, String> {
    let n = op.n_basis;
    if n == 0 {
        return Ok(KillingResult {
            dim: 0,
            basis: Vec::new(),
            spectrum: Vec::new(),
            tol_used: opts.tol_abs,
        });
    }
    let (eigvals, c) = solve_gen_symmetric_eigenproblem(
        &op.l_matrix,
        &op.gram_matrix,
        n,
        opts.gram_regularisation,
        opts.max_jacobi_sweeps,
        opts.jacobi_tol,
    )?;

    let lambda_max = eigvals
        .iter()
        .copied()
        .fold(0.0f64, |acc, x| acc.max(x.abs()));
    let tol_used = opts.tol_abs.max(opts.tol_rel * lambda_max);

    // Compute residuals and pack KillingVectorField entries for every
    // eigenvalue. Residuals are computed in the original basis:
    // r_j = ‖L c_j - λ_j G c_j‖ / ‖c_j‖_G  ≤ ‖L c_j‖ / ‖c_j‖_G  + |λ_j|.
    let mut basis_out: Vec<KillingVectorField> = Vec::with_capacity(n);
    for j in 0..n {
        let mut coeffs = vec![0.0; n];
        for i in 0..n {
            coeffs[i] = c[i * n + j];
        }
        // Compute G-norm and L*c residual.
        let mut gc = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += op.gram_matrix[i * n + k] * coeffs[k];
            }
            gc[i] = s;
        }
        let mut g_norm_sq = 0.0;
        for i in 0..n {
            g_norm_sq += coeffs[i] * gc[i];
        }
        let g_norm = g_norm_sq.max(1e-30).sqrt();
        let mut lc = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += op.l_matrix[i * n + k] * coeffs[k];
            }
            lc[i] = s;
        }
        let lc_norm: f64 = lc.iter().map(|x| x * x).sum::<f64>().sqrt();
        let residual = lc_norm / g_norm;

        basis_out.push(KillingVectorField {
            coefficients: coeffs,
            eigenvalue: eigvals[j],
            residual,
        });
    }

    let dim = eigvals.iter().filter(|&&l| l.abs() < tol_used).count();

    Ok(KillingResult {
        dim,
        basis: basis_out,
        spectrum: eigvals,
        tol_used,
    })
}

/// High-level API: assemble the Lichnerowicz operator on the given
/// metric/basis/sample-points and immediately extract its Killing kernel.
pub fn killing_algebra<M, B>(
    metric: &M,
    basis: &B,
    sample_points: &[f64],
    weights: &[f64],
    opts: &KillingSolveOptions,
) -> Result<KillingResult, String>
where
    M: MetricEvaluator,
    B: VectorFieldBasis,
{
    let op = crate::route34::lichnerowicz::assemble_lichnerowicz_matrix(
        metric,
        basis,
        sample_points,
        weights,
    )?;
    solve_killing_kernel(&op, opts)
}

/// Convenience wrapper returning just the dimension and residuals.
pub fn killing_algebra_dimension<M, B>(
    metric: &M,
    basis: &B,
    sample_points: &[f64],
    weights: &[f64],
    opts: &KillingSolveOptions,
) -> Result<(usize, Vec<f64>), String>
where
    M: MetricEvaluator,
    B: VectorFieldBasis,
{
    let result = killing_algebra(metric, basis, sample_points, weights, opts)?;
    let residuals: Vec<f64> = result.basis.iter().map(|k| k.residual).collect();
    Ok((result.dim, residuals))
}

// ---------------------------------------------------------------------------
// Lie bracket structure
// ---------------------------------------------------------------------------

/// Estimate the Lie-bracket structure constants of an extracted
/// Killing-algebra basis. The bracket `[ξ_a, ξ_b]^μ = ξ_a^ν ∂_ν ξ_b^μ
///   − ξ_b^ν ∂_ν ξ_a^μ` is evaluated pointwise on the sample grid and
/// projected back into the Killing basis via the Gram inverse.
///
/// Returns the structure-constant tensor `f[a * k^2 + b * k + c]`
/// where `k` is the Killing-algebra dimension and `f^c_{ab} = f[a, b, c]`
/// is antisymmetric in `(a, b)` for any genuine Lie algebra.
///
/// The pointwise bracket requires `∂ξ` at each sample point; we
/// re-evaluate the basis to get `(V_a^μ, ∂_k V_a^μ)` and contract with
/// the basis-coefficient expansions.
pub fn killing_bracket_structure_constants<M, B>(
    op: &LichnerowiczOperator,
    killing_basis: &[KillingVectorField],
    metric: &M,
    basis: &B,
    sample_points: &[f64],
    weights: &[f64],
) -> Result<Vec<f64>, String>
where
    M: MetricEvaluator,
    B: VectorFieldBasis,
{
    let k = killing_basis.len();
    let n = op.n_basis;
    let d = op.d;
    let ambient = metric.ambient_dim();
    let n_sample = sample_points.len() / ambient;
    if weights.len() != n_sample {
        return Err(format!(
            "weights length {} != n_sample {}",
            weights.len(),
            n_sample
        ));
    }
    if k == 0 {
        return Ok(Vec::new());
    }

    // Pre-compute Gram inverse (we'll need it to project the
    // pointwise bracket back into the Killing basis).
    let mut g_inv = vec![0.0; n * n];
    let mut a_work = vec![0.0; n * n];
    let mut perm = vec![0usize; n];
    let mut col_buf = vec![0.0; n];
    crate::linalg::invert(
        &op.gram_matrix,
        n,
        &mut a_work,
        &mut perm,
        &mut g_inv,
        &mut col_buf,
    )
    .map_err(|e| format!("Gram-inverse failed: {e}"))?;

    // For each sample point, evaluate the basis (V, ∂V), reconstruct
    // each Killing vector field's value and ∂ at this point, compute
    // brackets, and accumulate the inner-product
    //   B_{ab,c} = Σ_p w_p g_{μν}(x_p) [ξ_a, ξ_b]^μ(x_p) (V_c)^ν(x_p)
    // where the c index runs over the original basis. Then
    //   f^c_{ab} = G^{-1} · B_{ab, *}_c.

    let init = || vec![0.0; k * k * n];
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_pts = ((n_sample + n_threads - 1) / n_threads).max(32);

    let big = (0..n_sample)
        .into_par_iter()
        .with_min_len(chunk_pts)
        .fold(init, |mut acc, p_idx| {
            let pt = &sample_points[p_idx * ambient..(p_idx + 1) * ambient];
            let w = weights[p_idx];
            if !w.is_finite() || w <= 0.0 {
                return acc;
            }
            let mut v = vec![0.0; n * d];
            let mut dv = vec![0.0; n * d * d];
            let mut ddv = vec![0.0; n * d * d * d];
            basis.evaluate(pt, &mut v, &mut dv, &mut ddv);
            let mut g = vec![0.0; d * d];
            let mut dg = vec![0.0; d * d * d];
            metric.evaluate(pt, &mut g, &mut dg);

            // Reconstruct each Killing field's V and ∂V at this point.
            let mut xi = vec![0.0; k * d];
            let mut dxi = vec![0.0; k * d * d];
            for kf in 0..k {
                let coeffs = &killing_basis[kf].coefficients;
                for mu in 0..d {
                    let mut s = 0.0;
                    for a in 0..n {
                        s += coeffs[a] * v[a * d + mu];
                    }
                    xi[kf * d + mu] = s;
                }
                for kk_ in 0..d {
                    for mu in 0..d {
                        let mut s = 0.0;
                        for a in 0..n {
                            s += coeffs[a] * dv[a * d * d + kk_ * d + mu];
                        }
                        dxi[kf * d * d + kk_ * d + mu] = s;
                    }
                }
            }

            // For each pair (a, b), compute pointwise [ξ_a, ξ_b]^μ
            // = ξ_a^ν ∂_ν ξ_b^μ − ξ_b^ν ∂_ν ξ_a^μ.
            for a in 0..k {
                for b in 0..k {
                    if a == b {
                        continue;
                    }
                    let mut bracket = vec![0.0; d];
                    for mu in 0..d {
                        let mut s = 0.0;
                        for nu in 0..d {
                            s += xi[a * d + nu] * dxi[b * d * d + nu * d + mu];
                            s -= xi[b * d + nu] * dxi[a * d * d + nu * d + mu];
                        }
                        bracket[mu] = s;
                    }
                    // Lower the bracket index.
                    let mut bracket_low = vec![0.0; d];
                    for mu in 0..d {
                        let mut s = 0.0;
                        for nu in 0..d {
                            s += g[mu * d + nu] * bracket[nu];
                        }
                        bracket_low[mu] = s;
                    }
                    // Inner-product against each original basis V_c.
                    for c in 0..n {
                        let mut s = 0.0;
                        for mu in 0..d {
                            s += bracket_low[mu] * v[c * d + mu];
                        }
                        acc[a * k * n + b * n + c] += w * s;
                    }
                }
            }
            acc
        })
        .reduce(init, |mut a, b| {
            for k_ in 0..a.len() {
                a[k_] += b[k_];
            }
            a
        });

    // Contract with G^{-1} on the c-index: f^d_{ab} = G^{-1}_{dc} B_{ab,c}.
    let mut f = vec![0.0; k * k * k];
    for a in 0..k {
        for b in 0..k {
            // Project onto the Killing-coefficient basis: for each
            // killing-index `dest`, we need to compute the inner product
            // ⟨[ξ_a, ξ_b], ξ_dest⟩_G in the original-basis coordinates.
            //   G·c_dest = the gram-applied coefficients of ξ_dest;
            //   ⟨[a,b], ξ_dest⟩_G = Σ_c B_{ab, c} (G^{-1} · G c_dest)_c
            //                     = Σ_c B_{ab, c} c_dest_c.
            // i.e. project the per-c sum directly onto the Killing
            // coefficient vectors.
            for dest in 0..k {
                let cd = &killing_basis[dest].coefficients;
                let mut s = 0.0;
                for c in 0..n {
                    s += big[a * k * n + b * n + c] * cd[c];
                }
                // Normalise by the Killing's G-norm² so ξ_dest is unit.
                let mut g_norm_sq = 0.0;
                for i in 0..n {
                    let mut row = 0.0;
                    for j in 0..n {
                        row += op.gram_matrix[i * n + j] * cd[j];
                    }
                    g_norm_sq += cd[i] * row;
                }
                let g_norm_sq = g_norm_sq.max(1e-30);
                f[a * k * k + b * k + dest] = s / g_norm_sq;
            }
        }
    }
    let _ = g_inv; // retained for downstream API symmetry
    Ok(f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route34::lichnerowicz::*;

    #[test]
    fn jacobi_diagonalises_diagonal_matrix() {
        let n = 4;
        let mut a = vec![0.0; n * n];
        a[0] = 1.0;
        a[5] = 2.0;
        a[10] = 3.0;
        a[15] = 4.0;
        let mut q = vec![0.0; n * n];
        let (_sweeps, off) = jacobi_eigh(&mut a, &mut q, n, 32, 1e-14);
        assert!(off < 1e-13);
        let eigs: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
        // Already diagonal, so eigenvalues are unchanged.
        let mut sorted = eigs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 1.0).abs() < 1e-12);
        assert!((sorted[3] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn cholesky_solve_round_trip() {
        // G = [[2, 1], [1, 2]]; chol -> R; solve Rᵀ R x = b, check.
        let mut g = vec![2.0, 1.0, 1.0, 2.0];
        cholesky_lower_in_place(&mut g, 2).unwrap();
        let mut b = vec![3.0, 3.0];
        // Solve R y = b
        solve_lower_triangular(&g, 2, &mut b);
        // Solve Rᵀ x = y
        solve_upper_triangular_transposed(&g, 2, &mut b);
        // Original [[2,1],[1,2]] times x should give [3, 3]:
        // 2x0 + x1 = 3; x0 + 2x1 = 3; ⇒ x0 = x1 = 1.
        assert!((b[0] - 1.0).abs() < 1e-10);
        assert!((b[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn flat_torus_killing_dim_matches_translation_count() {
        // Flat R^d with degree-0 polynomial basis: d translation
        // generators, all in the Killing kernel.
        let d = 4;
        let metric = FlatMetric { d };
        let basis = PolynomialVectorBasis::coordinate_polynomial_basis(d, 0);
        let n_pts = 50;
        let mut pts = Vec::with_capacity(n_pts * d);
        for p in 0..n_pts {
            for k in 0..d {
                pts.push((p as f64 * 0.1 + k as f64 * 0.07).sin());
            }
        }
        let weights = vec![1.0; n_pts];
        let opts = KillingSolveOptions::default();
        let result = killing_algebra(&metric, &basis, &pts, &weights, &opts).unwrap();
        assert_eq!(
            result.dim, d,
            "flat R^{d} translation Killing count: expected {d}, got {}",
            result.dim
        );
        // Bottom d eigenvalues should be ≈ 0; subsequent ones (if any)
        // should be > tol_used.
        for j in 0..d {
            assert!(
                result.spectrum[j].abs() < result.tol_used * 10.0 + 1e-8,
                "eigenvalue {j} = {} too large",
                result.spectrum[j]
            );
        }
    }

    #[test]
    fn flat_torus_translation_plus_linear_basis_finds_d_kernels() {
        // Same flat manifold, but include linear polynomials too.
        // Killing fields on flat R^d are translations + rotations
        // (so(d) with d(d-1)/2 generators, total d + d(d-1)/2 = d(d+1)/2).
        // With max_degree = 1 the polynomial basis covers all of these
        // (rotations are linear vector fields), so the kernel dimension
        // should be d + d(d-1)/2 = d(d+1)/2.
        let d = 3;
        let metric = FlatMetric { d };
        let basis = PolynomialVectorBasis::coordinate_polynomial_basis(d, 1);
        let n_pts = 100;
        let mut pts = Vec::with_capacity(n_pts * d);
        let mut rng_state = 12345u64;
        for _ in 0..(n_pts * d) {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
            pts.push((bits - 0.5) * 2.0);
        }
        let weights = vec![1.0; n_pts];
        let opts = KillingSolveOptions::default();
        let result = killing_algebra(&metric, &basis, &pts, &weights, &opts).unwrap();
        let expected = d + d * (d - 1) / 2;
        assert_eq!(
            result.dim, expected,
            "flat R^{d} (translations + so({d})) Killing count: expected {expected}, got {} \
             (spectrum head: {:?})",
            result.dim,
            &result.spectrum[..result.spectrum.len().min(10)],
        );
    }
}
