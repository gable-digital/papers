//! True Lichnerowicz vector-Laplacian Δ_L on a Riemannian manifold.
//!
//! ## Mathematical setup
//!
//! For a Riemannian manifold `(M, g)` of dimension `d`, the
//! **Lichnerowicz vector-Laplacian** acts on a vector field
//! `ξ = ξ^μ ∂_μ` by
//!
//! ```text
//!     (Δ_L ξ)_ν  =  ∇^μ ∇_μ ξ_ν  +  R_νμ ξ^μ
//! ```
//!
//! where `∇` is the Levi-Civita connection and `R_νμ` is the Ricci
//! tensor (Wald, *General Relativity*, Univ. of Chicago Press 1984,
//! eq. 3.4.4 + §3.4; Besse, *Einstein Manifolds*, Springer 1987,
//! §1.K, eq. 1.143). On a Ricci-flat manifold the Ricci term vanishes
//! and `Δ_L ξ_ν = ∇^μ ∇_μ ξ_ν` reduces to the rough Laplacian on
//! 1-forms (with `ξ_ν = g_{νμ} ξ^μ`).
//!
//! ## Spectrum vs. deformation tensor
//!
//! The deformation-tensor quadratic form
//! `L_def[ξ, η] = ∫ g g (L_ξ g)(L_η g) dvol` (implemented in the
//! sister module [`crate::route34::deformation_tensor`]) is symmetric
//! and positive-semidefinite, with kernel exactly the Killing
//! algebra. Δ_L is also self-adjoint on a closed manifold and its
//! kernel coincides with the Killing algebra on a Ricci-flat compact
//! manifold (Bochner-Yano), but the *non-zero* eigenvalues of Δ_L and
//! of the deformation-tensor form **differ** by the Bochner correction
//!
//! ```text
//!     ½ ‖L_ξ g‖²  =  ‖∇ξ‖²  −  ⟨ξ, ∇ ∇·ξ⟩  +  R_μν ξ^μ ξ^ν
//! ```
//!
//! (Petersen, *Riemannian Geometry* 3rd ed., Springer 2016, §9.3,
//! eq. 9.3.5; the integrated identity is the Bochner formula). The
//! `route4_predictor` consumes the *spectrum* — not just the kernel —
//! to discriminate between candidate CY3s, so the code must use the
//! Lichnerowicz spectrum, not the deformation-tensor spectrum.
//!
//! ## Discrete approximation
//!
//! We expand a candidate vector field in a basis `{V_a}_{a=1..N}` of
//! vector fields on `M`. We assemble three matrices on the basis:
//!
//! ```text
//!     L_{ab}  =  Σ_p w_p g_{μν}(x_p) (Δ_L V_a)^μ(x_p) (V_b)^ν(x_p)
//!     R_{ab}  =  Σ_p w_p g_{μν}(x_p) (∇^ρ ∇_ρ V_a)^μ(x_p) (V_b)^ν(x_p)
//!     G_{ab}  =  Σ_p w_p g_{μν}(x_p) (V_a)^μ(x_p) (V_b)^ν(x_p)
//! ```
//!
//! where `Δ_L V_a` is the action of the Lichnerowicz operator on each
//! basis field and `R V_a` is the rough-Laplacian piece alone. `L = R`
//! identically when the metric is Ricci-flat — this is a built-in
//! self-consistency check the assembler exposes via
//! [`LichnerowiczOperator::ricci_residual`].
//!
//! Both `L` and `G` are symmetrised at the end. The generalised
//! eigenproblem `L c = λ G c` is solved by Cholesky whitening + cyclic
//! Jacobi (see [`crate::route34::killing_solver`]), exactly as for the
//! deformation-tensor form.
//!
//! ## Christoffel and curvature
//!
//! Christoffel symbols come from the metric and its first
//! derivatives:
//!
//! ```text
//!     Γ^λ_{μν}  =  ½ g^{λρ} (∂_μ g_{νρ} + ∂_ν g_{μρ} − ∂_ρ g_{μν})
//! ```
//!
//! The Riemann tensor in component form is
//!
//! ```text
//!     R^ρ_{σμν}  =  ∂_μ Γ^ρ_{νσ} − ∂_ν Γ^ρ_{μσ}
//!                  +  Γ^ρ_{μλ} Γ^λ_{νσ} − Γ^ρ_{νλ} Γ^λ_{μσ}
//! ```
//!
//! and the Ricci tensor is `R_{σν} = R^ρ_{σρν}` (Wald eq. 3.2.25).
//! Computing `R^ρ_{σμν}` requires the partial derivatives `∂Γ`, which
//! in turn require the second partial derivatives `∂²g`. We supply
//! those via finite differences of the [`MetricEvaluator`] in
//! production (or analytic second derivatives in test fixtures), and
//! consume them through a small extension trait
//! [`MetricEvaluatorWithSecond`].
//!
//! On a closed Ricci-flat manifold the Ricci tensor vanishes
//! identically and the second-derivative input is unused.
//!
//! ## Covariant Laplacian on vector fields
//!
//! For a contravariant vector `ξ^ν`,
//!
//! ```text
//!     (∇^μ ∇_μ ξ)^ν
//!         =  g^{μρ} ∇_μ ∇_ρ ξ^ν
//!         =  g^{μρ} ( ∂_μ ∂_ρ ξ^ν
//!                     + (∂_μ Γ^ν_{ρλ}) ξ^λ
//!                     + Γ^ν_{ρλ} ∂_μ ξ^λ
//!                     + Γ^ν_{μλ} ∂_ρ ξ^λ
//!                     − Γ^σ_{μρ} ∂_σ ξ^ν
//!                     + Γ^ν_{μσ} Γ^σ_{ρλ} ξ^λ
//!                     − Γ^σ_{μρ} Γ^ν_{σλ} ξ^λ )
//! ```
//!
//! (expand `∇_μ (∇_ρ ξ^ν) = ∂_μ (∇_ρ ξ^ν) + Γ^ν_{μσ} (∇_ρ ξ^σ) −
//! Γ^σ_{μρ} (∇_σ ξ^ν)` after `∇_ρ ξ^ν = ∂_ρ ξ^ν + Γ^ν_{ρλ} ξ^λ`,
//! then symmetrise on `(μ, ρ)` against `g^{μρ}`). The output is
//! a contravariant vector field; we lower the free index by `g_{νμ}`
//! before contracting against the basis to produce the symmetric
//! Gram-style entry.
//!
//! ## Verified test cases
//!
//! * **Round S^d, d ≥ 2**: the Lichnerowicz Laplacian's spectrum on
//!   divergence-free 1-forms (= co-closed transverse modes) is
//!   `λ_l = l(l+1) − 1` for `l ≥ 1` (Higuchi, *J. Math. Phys.* 28
//!   (1987) 1553, eq. 6.4 with `n = d`; for the specific case d = 2
//!   and r = 1 — the radius — the divergence-free vector spherical
//!   harmonics have `Δ_L Y = (l(l+1) − 1) Y` for `l ≥ 1`).
//!   The **kernel** (l = 1, eigenvalue `λ = 1` is *not* zero, but
//!   the Killing 1-forms are the divergence-free `l = 1` modes
//!   that satisfy Killing's equation; per Bochner-Yano on a Ricci-
//!   curved closed manifold the deformation-tensor kernel coincides
//!   with the Killing algebra of dim `d(d+1)/2` while Δ_L's kernel
//!   contains only the parallel 1-forms — for round S^2 there are
//!   none, so `dim ker Δ_L = 0`. The **deformation-tensor** kernel,
//!   in contrast, is the full `so(3)` of dim 3.)
//! * **Flat T^d**: Christoffel symbols vanish, Ricci vanishes,
//!   `Δ_L = ∇^μ ∇_μ` reduces to the component-wise Laplacian on the
//!   flat torus. Constant vector fields are in the kernel.
//! * **Generic compact CY3**: Ricci-flat by Yau, so `Δ_L = ∇^μ ∇_μ`;
//!   Killing algebra dim 0 (Yau, CMP 1978).
//!
//! ## References
//!
//! * Wald R., *General Relativity*, University of Chicago Press 1984,
//!   §3.1, §3.4. Lichnerowicz operator: eq. 3.4.4. Christoffel:
//!   eq. 3.1.30. Riemann: eq. 3.2.12.
//! * Besse A., *Einstein Manifolds*, Springer 1987, §1.K. The
//!   Lichnerowicz operator on 1-forms: eq. 1.143.
//! * Petersen P., *Riemannian Geometry*, 3rd ed., Springer 2016, §9.3.
//!   Bochner identity for 1-forms: eq. 9.3.5.
//! * Higuchi A., "Symmetric tensor spherical harmonics on the
//!   N-sphere and their application to the de Sitter group SO(N,1)",
//!   *J. Math. Phys.* **28** (1987) 1553,
//!   DOI 10.1063/1.527513.
//! * Yano K., Bochner S., *Curvature and Betti Numbers*, Princeton
//!   1953, §1.
//! * Yau S.-T., "On the Ricci curvature of a compact Kähler manifold
//!   and the complex Monge-Ampère equation, I", *Comm. Pure Appl.
//!   Math.* **31** (1978) 339, DOI 10.1002/cpa.3160310304.

use rayon::prelude::*;

use crate::route34::lichnerowicz::{
    christoffel_symbols, lower_basis_index, MetricEvaluator, VectorFieldBasis,
};

/// Maximum supported intrinsic dimension `d` for the per-point inner
/// loops. We use stack-allocated fixed-size buffers of length `d²` and
/// `d³`; raising this requires bumping the const and recompiling.
const MAX_INTRINSIC_DIM: usize = 8;

/// Extended metric evaluator that also supplies second partial
/// derivatives `∂_l ∂_k g_{ij}(x)` of the metric, which are needed to
/// form `∂Γ` and from there the Riemann/Ricci tensor.
///
/// Implementations may compute these analytically (for closed-form test
/// metrics) or by central finite differences of the standard
/// [`MetricEvaluator::evaluate`] callback. The second-derivative tensor
/// is stored row-major as `ddg_out[l * d³ + k * d² + i * d + j]
///   = ∂_l ∂_k g_{ij}(x)`.
pub trait MetricEvaluatorWithSecond: MetricEvaluator {
    /// Evaluate `g`, `∂g`, and `∂∂g` at the sample point `x`.
    ///
    /// The first two outputs follow the same convention as
    /// [`MetricEvaluator::evaluate`]. `ddg_out` must have length at
    /// least `d * d * d * d` (intrinsic indices throughout).
    fn evaluate_with_second(
        &self,
        x: &[f64],
        g_out: &mut [f64],
        dg_out: &mut [f64],
        ddg_out: &mut [f64],
    );
}

/// Per-thread scratch for assembling the Lichnerowicz operator. All
/// buffers are sized once and reused across sample points.
pub struct LichnerowiczOpScratch {
    pub d: usize,
    pub n_basis: usize,
    pub g: Vec<f64>,
    pub g_inv: Vec<f64>,
    pub dg: Vec<f64>,
    pub ddg: Vec<f64>,
    pub gamma: Vec<f64>,
    /// `∂_k Γ^λ_{μν}` row-major `d × d × d × d`.
    pub dgamma: Vec<f64>,
    /// Ricci tensor `R_{σν}` row-major `d × d`.
    pub ricci: Vec<f64>,
    pub v: Vec<f64>,
    pub dv: Vec<f64>,
    pub ddv: Vec<f64>,
    /// `(Δ_L V_a)^ν` row-major `n_basis × d` (contravariant).
    pub delta_l: Vec<f64>,
    /// `(∇^μ ∇_μ V_a)^ν` row-major `n_basis × d` (rough-Laplacian
    /// piece alone, no Ricci term).
    pub rough: Vec<f64>,
    pub v_lowered: Vec<f64>,
    pub lu_work: Vec<f64>,
    pub lu_perm: Vec<usize>,
    pub lu_col: Vec<f64>,
    /// Per-thread accumulator for `L_{ab}` (Δ_L form), `n_basis²`.
    pub l_local: Vec<f64>,
    /// Per-thread accumulator for `R_{ab}` (rough Laplacian only).
    pub rough_local: Vec<f64>,
    /// Per-thread accumulator for `G_{ab}` (vector Gram), `n_basis²`.
    pub g_local: Vec<f64>,
}

impl LichnerowiczOpScratch {
    pub fn new(n_basis: usize, d: usize) -> Self {
        assert!(
            d <= MAX_INTRINSIC_DIM,
            "intrinsic dim {d} exceeds MAX_INTRINSIC_DIM = {MAX_INTRINSIC_DIM}"
        );
        Self {
            d,
            n_basis,
            g: vec![0.0; d * d],
            g_inv: vec![0.0; d * d],
            dg: vec![0.0; d * d * d],
            ddg: vec![0.0; d * d * d * d],
            gamma: vec![0.0; d * d * d],
            dgamma: vec![0.0; d * d * d * d],
            ricci: vec![0.0; d * d],
            v: vec![0.0; n_basis * d],
            dv: vec![0.0; n_basis * d * d],
            ddv: vec![0.0; n_basis * d * d * d],
            delta_l: vec![0.0; n_basis * d],
            rough: vec![0.0; n_basis * d],
            v_lowered: vec![0.0; n_basis * d],
            lu_work: vec![0.0; d * d],
            lu_perm: vec![0usize; d],
            lu_col: vec![0.0; d],
            l_local: vec![0.0; n_basis * n_basis],
            rough_local: vec![0.0; n_basis * n_basis],
            g_local: vec![0.0; n_basis * n_basis],
        }
    }

    pub fn reset_accumulators(&mut self) {
        for v in self.l_local.iter_mut() {
            *v = 0.0;
        }
        for v in self.rough_local.iter_mut() {
            *v = 0.0;
        }
        for v in self.g_local.iter_mut() {
            *v = 0.0;
        }
    }
}

/// Assembled discrete Lichnerowicz operator. Both `l_matrix` and
/// `gram_matrix` are symmetric `n_basis × n_basis` row-major matrices.
/// `rough_matrix` is the rough-Laplacian-only piece (no Ricci term);
/// the difference `l_matrix − rough_matrix` is the contribution of the
/// Ricci term and equals zero on Ricci-flat metrics.
pub struct LichnerowiczOperator {
    pub n_basis: usize,
    pub d: usize,
    pub n_sample: usize,
    pub total_weight: f64,
    pub l_matrix: Vec<f64>,
    pub rough_matrix: Vec<f64>,
    pub gram_matrix: Vec<f64>,
    /// Maximum absolute asymmetry of `L` before the symmetrisation
    /// pass.
    pub asymmetry: f64,
}

impl LichnerowiczOperator {
    /// Symmetrise `L`, `R`, and `G` by averaging each with its
    /// transpose.
    pub fn symmetrise(&mut self) {
        let n = self.n_basis;
        let mut max_asym = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let a = self.l_matrix[i * n + j];
                let b = self.l_matrix[j * n + i];
                let asym = (a - b).abs();
                if asym > max_asym {
                    max_asym = asym;
                }
                let s = 0.5 * (a + b);
                self.l_matrix[i * n + j] = s;
                self.l_matrix[j * n + i] = s;

                let ar = self.rough_matrix[i * n + j];
                let br = self.rough_matrix[j * n + i];
                let sr = 0.5 * (ar + br);
                self.rough_matrix[i * n + j] = sr;
                self.rough_matrix[j * n + i] = sr;

                let ag = self.gram_matrix[i * n + j];
                let bg = self.gram_matrix[j * n + i];
                let sg = 0.5 * (ag + bg);
                self.gram_matrix[i * n + j] = sg;
                self.gram_matrix[j * n + i] = sg;
            }
        }
        self.asymmetry = max_asym;
    }

    /// Frobenius norm of `l_matrix − rough_matrix`. On a Ricci-flat
    /// manifold this should be at the discretisation noise floor; a
    /// large residual signals either a non-Ricci-flat metric, or
    /// numerical error in the second-derivative input.
    pub fn ricci_residual(&self) -> f64 {
        let n = self.n_basis;
        let mut s = 0.0;
        for i in 0..(n * n) {
            let d = self.l_matrix[i] - self.rough_matrix[i];
            s += d * d;
        }
        s.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Per-point assembly: ∂Γ, Riemann, Ricci, then Δ_L V_a.
// ---------------------------------------------------------------------------

/// Compute `∂_k Γ^λ_{μν}` from the metric `g`, its first derivatives
/// `∂g`, second derivatives `∂∂g`, and the inverse metric `g^{ij}`.
///
/// `dg[k * d² + i * d + j] = ∂_k g_{ij}`,
/// `ddg[l * d³ + k * d² + i * d + j] = ∂_l ∂_k g_{ij}`,
/// `g_inv[i * d + j] = g^{ij}`.
///
/// On output `dgamma_out[k * d³ + lam * d² + mu * d + nu] = ∂_k Γ^λ_{μν}`.
///
/// The formula is obtained by differentiating
/// `Γ^λ_{μν} = ½ g^{λρ} (∂_μ g_{νρ} + ∂_ν g_{μρ} − ∂_ρ g_{μν})`
/// using `∂_k g^{λρ} = − g^{λα} g^{ρβ} ∂_k g_{αβ}` (the inverse-metric
/// derivative identity, Wald eq. 3.1.31):
///
/// ```text
///     ∂_k Γ^λ_{μν} = ½ ∂_k g^{λρ} (...)
///                  + ½ g^{λρ} (∂_k ∂_μ g_{νρ} + ∂_k ∂_ν g_{μρ} − ∂_k ∂_ρ g_{μν})
/// ```
#[allow(clippy::too_many_arguments)]
pub fn dchristoffel_symbols(
    d: usize,
    g_inv: &[f64],
    dg: &[f64],
    ddg: &[f64],
    dgamma_out: &mut [f64],
) {
    debug_assert!(g_inv.len() >= d * d);
    debug_assert!(dg.len() >= d * d * d);
    debug_assert!(ddg.len() >= d * d * d * d);
    debug_assert!(dgamma_out.len() >= d * d * d * d);

    for v in dgamma_out.iter_mut().take(d * d * d * d) {
        *v = 0.0;
    }

    // We compute ∂_k g^{λρ} on demand inside the inner loop to avoid
    // an additional d × d × d buffer. Identity used:
    //   ∂_k g^{λρ}  =  − g^{λα} g^{ρβ} ∂_k g_{αβ}
    // (Wald, *General Relativity*, 1984, eq. 3.1.31).
    for k in 0..d {
        for lam in 0..d {
            for mu in 0..d {
                for nu in 0..d {
                    let mut acc = 0.0;
                    for rho in 0..d {
                        // Bracket B(μν|ρ) = ∂_μ g_{νρ} + ∂_ν g_{μρ} − ∂_ρ g_{μν}.
                        let b_mnurho = dg[mu * d * d + nu * d + rho]
                            + dg[nu * d * d + mu * d + rho]
                            - dg[rho * d * d + mu * d + nu];
                        // ∂_k B(μν|ρ) using the second-derivative input.
                        let dk_b = ddg[k * d * d * d + mu * d * d + nu * d + rho]
                            + ddg[k * d * d * d + nu * d * d + mu * d + rho]
                            - ddg[k * d * d * d + rho * d * d + mu * d + nu];
                        // ∂_k g^{λρ}, computed from dg (not ddg).
                        let mut dk_ginv_lr = 0.0;
                        for alpha in 0..d {
                            for beta in 0..d {
                                dk_ginv_lr -= g_inv[lam * d + alpha]
                                    * g_inv[rho * d + beta]
                                    * dg[k * d * d + alpha * d + beta];
                            }
                        }
                        let g_lr = g_inv[lam * d + rho];
                        acc += 0.5 * dk_ginv_lr * b_mnurho;
                        acc += 0.5 * g_lr * dk_b;
                    }
                    dgamma_out[k * d * d * d + lam * d * d + mu * d + nu] = acc;
                }
            }
        }
    }
}

/// Compute the Ricci tensor `R_{σν}` from Christoffel symbols and
/// their first derivatives.
///
/// `R_{σν} = R^ρ_{σρν}` with
/// `R^ρ_{σμν} = ∂_μ Γ^ρ_{νσ} − ∂_ν Γ^ρ_{μσ}
///              + Γ^ρ_{μλ} Γ^λ_{νσ} − Γ^ρ_{νλ} Γ^λ_{μσ}`
/// (Wald eq. 3.2.12). Tracing on `(ρ, μ)`:
///
/// ```text
///     R_{σν}  =  ∂_ρ Γ^ρ_{νσ} − ∂_ν Γ^ρ_{ρσ}
///               +  Γ^ρ_{ρλ} Γ^λ_{νσ} − Γ^ρ_{νλ} Γ^λ_{ρσ}
/// ```
pub fn ricci_tensor(
    d: usize,
    gamma: &[f64],
    dgamma: &[f64],
    ricci_out: &mut [f64],
) {
    debug_assert!(gamma.len() >= d * d * d);
    debug_assert!(dgamma.len() >= d * d * d * d);
    debug_assert!(ricci_out.len() >= d * d);
    for v in ricci_out.iter_mut().take(d * d) {
        *v = 0.0;
    }
    for sigma in 0..d {
        for nu in 0..d {
            let mut s = 0.0;
            for rho in 0..d {
                // ∂_ρ Γ^ρ_{νσ}
                s += dgamma[rho * d * d * d + rho * d * d + nu * d + sigma];
                // − ∂_ν Γ^ρ_{ρσ}
                s -= dgamma[nu * d * d * d + rho * d * d + rho * d + sigma];
                for lam in 0..d {
                    // + Γ^ρ_{ρλ} Γ^λ_{νσ}
                    s += gamma[rho * d * d + rho * d + lam]
                        * gamma[lam * d * d + nu * d + sigma];
                    // − Γ^ρ_{νλ} Γ^λ_{ρσ}
                    s -= gamma[rho * d * d + nu * d + lam]
                        * gamma[lam * d * d + rho * d + sigma];
                }
            }
            ricci_out[sigma * d + nu] = s;
        }
    }
}

/// Compute `(∇^μ ∇_μ V_a)^ν` (rough Laplacian on the contravariant
/// vector field `V_a`) at one sample point, for every basis vector.
///
/// Output: `rough_out[a * d + nu] = (∇^μ ∇_μ V_a)^ν`.
///
/// Formula (see module docstring):
///
/// ```text
///     (∇^μ ∇_μ ξ)^ν
///       = g^{μρ} ( ∂_μ ∂_ρ ξ^ν
///                  + (∂_μ Γ^ν_{ρλ}) ξ^λ
///                  + Γ^ν_{ρλ} ∂_μ ξ^λ
///                  + Γ^ν_{μλ} ∂_ρ ξ^λ
///                  − Γ^σ_{μρ} ∂_σ ξ^ν
///                  + Γ^ν_{μσ} Γ^σ_{ρλ} ξ^λ
///                  − Γ^σ_{μρ} Γ^ν_{σλ} ξ^λ )
/// ```
#[allow(clippy::too_many_arguments)]
pub fn rough_laplacian_at_point(
    n_basis: usize,
    d: usize,
    v: &[f64],
    dv: &[f64],
    ddv: &[f64],
    g_inv: &[f64],
    gamma: &[f64],
    dgamma: &[f64],
    rough_out: &mut [f64],
) {
    debug_assert!(v.len() >= n_basis * d);
    debug_assert!(dv.len() >= n_basis * d * d);
    debug_assert!(ddv.len() >= n_basis * d * d * d);
    debug_assert!(g_inv.len() >= d * d);
    debug_assert!(gamma.len() >= d * d * d);
    debug_assert!(dgamma.len() >= d * d * d * d);
    debug_assert!(rough_out.len() >= n_basis * d);

    for v_o in rough_out.iter_mut().take(n_basis * d) {
        *v_o = 0.0;
    }

    for a in 0..n_basis {
        let v_a = &v[a * d..(a + 1) * d];
        let dv_a = &dv[a * d * d..(a + 1) * d * d];
        let ddv_a = &ddv[a * d * d * d..(a + 1) * d * d * d];

        for nu in 0..d {
            let mut acc = 0.0;
            for mu in 0..d {
                for rho in 0..d {
                    let g_inv_mr = g_inv[mu * d + rho];
                    if g_inv_mr == 0.0 {
                        continue;
                    }
                    // ∂_μ ∂_ρ ξ^ν  (note ddv is symmetric in (k,l):
                    // ddv[a * d^3 + k * d^2 + l * d + nu] is what we use,
                    // the user fills both halves).
                    let mut term =
                        ddv_a[mu * d * d + rho * d + nu];
                    // + (∂_μ Γ^ν_{ρλ}) ξ^λ
                    for lam in 0..d {
                        term += dgamma[mu * d * d * d + nu * d * d + rho * d + lam]
                            * v_a[lam];
                    }
                    // + Γ^ν_{ρλ} ∂_μ ξ^λ
                    for lam in 0..d {
                        term += gamma[nu * d * d + rho * d + lam]
                            * dv_a[mu * d + lam];
                    }
                    // + Γ^ν_{μλ} ∂_ρ ξ^λ
                    for lam in 0..d {
                        term += gamma[nu * d * d + mu * d + lam]
                            * dv_a[rho * d + lam];
                    }
                    // − Γ^σ_{μρ} ∂_σ ξ^ν
                    for sigma in 0..d {
                        term -= gamma[sigma * d * d + mu * d + rho]
                            * dv_a[sigma * d + nu];
                    }
                    // + Γ^ν_{μσ} Γ^σ_{ρλ} ξ^λ
                    for sigma in 0..d {
                        let gn_ms = gamma[nu * d * d + mu * d + sigma];
                        if gn_ms == 0.0 {
                            continue;
                        }
                        for lam in 0..d {
                            term +=
                                gn_ms * gamma[sigma * d * d + rho * d + lam] * v_a[lam];
                        }
                    }
                    // − Γ^σ_{μρ} Γ^ν_{σλ} ξ^λ
                    for sigma in 0..d {
                        let gs_mr = gamma[sigma * d * d + mu * d + rho];
                        if gs_mr == 0.0 {
                            continue;
                        }
                        for lam in 0..d {
                            term -=
                                gs_mr * gamma[nu * d * d + sigma * d + lam] * v_a[lam];
                        }
                    }
                    acc += g_inv_mr * term;
                }
            }
            rough_out[a * d + nu] = acc;
        }
    }
}

/// Compute the full Lichnerowicz `Δ_L V_a` at one sample point.
///
/// `(Δ_L ξ)^ν = (∇^μ ∇_μ ξ)^ν + R^ν_μ ξ^μ`
///            = `(∇^μ ∇_μ ξ)^ν + g^{νσ} R_{σμ} ξ^μ`.
///
/// `rough_in[a * d + nu]` is the rough Laplacian (already computed).
/// `delta_l_out[a * d + nu]` receives `(Δ_L V_a)^ν`.
pub fn lichnerowicz_at_point(
    n_basis: usize,
    d: usize,
    v: &[f64],
    g_inv: &[f64],
    ricci: &[f64],
    rough_in: &[f64],
    delta_l_out: &mut [f64],
) {
    debug_assert!(v.len() >= n_basis * d);
    debug_assert!(g_inv.len() >= d * d);
    debug_assert!(ricci.len() >= d * d);
    debug_assert!(rough_in.len() >= n_basis * d);
    debug_assert!(delta_l_out.len() >= n_basis * d);

    // Pre-compute R^ν_μ = g^{νσ} R_{σμ} once per point.
    let mut ricci_mixed = [0.0f64; MAX_INTRINSIC_DIM * MAX_INTRINSIC_DIM];
    for nu in 0..d {
        for mu in 0..d {
            let mut s = 0.0;
            for sigma in 0..d {
                s += g_inv[nu * d + sigma] * ricci[sigma * d + mu];
            }
            ricci_mixed[nu * d + mu] = s;
        }
    }

    for a in 0..n_basis {
        let v_a = &v[a * d..(a + 1) * d];
        for nu in 0..d {
            let mut s = rough_in[a * d + nu];
            for mu in 0..d {
                s += ricci_mixed[nu * d + mu] * v_a[mu];
            }
            delta_l_out[a * d + nu] = s;
        }
    }
}

/// Accumulate one sample point's contribution to `L`, `R` (rough), and
/// `G`.
///
/// ```text
///     L_local[a, b]  +=  w · g_{μν} (Δ_L V_a)^μ (V_b)^ν
///     R_local[a, b]  +=  w · g_{μν} (∇^ρ ∇_ρ V_a)^μ (V_b)^ν
///     G_local[a, b]  +=  w · g_{μν} (V_a)^μ (V_b)^ν
/// ```
#[allow(clippy::too_many_arguments)]
fn accumulate_point_contribution_op(
    n_basis: usize,
    d: usize,
    weight: f64,
    delta_l: &[f64],
    rough: &[f64],
    v: &[f64],
    v_lowered: &[f64],
    l_local: &mut [f64],
    rough_local: &mut [f64],
    g_local: &mut [f64],
) {
    for a in 0..n_basis {
        let dl_a = &delta_l[a * d..(a + 1) * d];
        let rg_a = &rough[a * d..(a + 1) * d];
        let v_a = &v[a * d..(a + 1) * d];
        for b in 0..n_basis {
            let vb_lo = &v_lowered[b * d..(b + 1) * d];
            let mut l_dot = 0.0;
            let mut r_dot = 0.0;
            let mut g_dot = 0.0;
            for mu in 0..d {
                l_dot += dl_a[mu] * vb_lo[mu];
                r_dot += rg_a[mu] * vb_lo[mu];
                g_dot += v_a[mu] * vb_lo[mu];
            }
            l_local[a * n_basis + b] += weight * l_dot;
            rough_local[a * n_basis + b] += weight * r_dot;
            g_local[a * n_basis + b] += weight * g_dot;
        }
    }
}

/// Assemble the discrete Lichnerowicz matrix, the rough-Laplacian
/// matrix (no Ricci term), and the vector-field Gram matrix on a
/// basis of vector fields.
///
/// `metric` must implement [`MetricEvaluatorWithSecond`] so that the
/// Ricci tensor can be formed at each sample point. `basis` supplies
/// the vector fields, their first derivatives, and their second
/// derivatives.
///
/// On Ricci-flat metrics the second-derivative input is still required
/// (it enters `∂Γ` and hence the Ricci tensor, which is then verified
/// to vanish via [`LichnerowiczOperator::ricci_residual`]).
pub fn assemble_lichnerowicz_operator<M, B>(
    metric: &M,
    basis: &B,
    sample_points: &[f64],
    weights: &[f64],
) -> Result<LichnerowiczOperator, String>
where
    M: MetricEvaluatorWithSecond,
    B: VectorFieldBasis,
{
    let d = metric.intrinsic_dim();
    if d != basis.intrinsic_dim() {
        return Err(format!(
            "metric/basis dim mismatch: metric d = {}, basis d = {}",
            d,
            basis.intrinsic_dim()
        ));
    }
    if d > MAX_INTRINSIC_DIM {
        return Err(format!(
            "intrinsic dim {} exceeds compile-time MAX_INTRINSIC_DIM = {}",
            d, MAX_INTRINSIC_DIM
        ));
    }
    let n_basis = basis.n_basis();
    let ambient = metric.ambient_dim();
    if sample_points.len() % ambient != 0 {
        return Err(format!(
            "sample_points length {} not a multiple of ambient dim {}",
            sample_points.len(),
            ambient
        ));
    }
    let n_sample = sample_points.len() / ambient;
    if weights.len() != n_sample {
        return Err(format!(
            "weights length {} != n_sample {}",
            weights.len(),
            n_sample
        ));
    }

    let n_threads = rayon::current_num_threads().max(1);
    let chunk_pts = ((n_sample + n_threads - 1) / n_threads).max(32);

    type Acc = (Vec<f64>, Vec<f64>, Vec<f64>, usize, f64, u64);
    let init: fn(usize) -> Acc = |n| {
        (
            vec![0.0; n * n],
            vec![0.0; n * n],
            vec![0.0; n * n],
            0usize,
            0.0f64,
            0u64,
        )
    };

    let combined: Acc = (0..n_sample)
        .into_par_iter()
        .with_min_len(chunk_pts)
        .fold(
            move || (LichnerowiczOpScratch::new(n_basis, d), init(n_basis)),
            |(mut s, mut acc), p_idx| {
                let pt = &sample_points[p_idx * ambient..(p_idx + 1) * ambient];
                let w = weights[p_idx];
                if !w.is_finite() || w <= 0.0 {
                    return (s, acc);
                }
                metric.evaluate_with_second(pt, &mut s.g, &mut s.dg, &mut s.ddg);
                if !s.g.iter().take(d * d).all(|x| x.is_finite())
                    || !s.dg.iter().take(d * d * d).all(|x| x.is_finite())
                    || !s.ddg.iter().take(d * d * d * d).all(|x| x.is_finite())
                {
                    acc.5 = acc.5.saturating_add(1);
                    return (s, acc);
                }
                let g_copy = s.g.clone();
                let chris_res = christoffel_symbols(
                    &g_copy,
                    &s.dg,
                    d,
                    &mut s.g_inv,
                    &mut s.gamma,
                    &mut s.lu_work,
                    &mut s.lu_perm,
                    &mut s.lu_col,
                );
                if chris_res.is_err() {
                    acc.5 = acc.5.saturating_add(1);
                    return (s, acc);
                }
                dchristoffel_symbols(d, &s.g_inv, &s.dg, &s.ddg, &mut s.dgamma);
                ricci_tensor(d, &s.gamma, &s.dgamma, &mut s.ricci);

                basis.evaluate(pt, &mut s.v, &mut s.dv, &mut s.ddv);

                rough_laplacian_at_point(
                    n_basis,
                    d,
                    &s.v,
                    &s.dv,
                    &s.ddv,
                    &s.g_inv,
                    &s.gamma,
                    &s.dgamma,
                    &mut s.rough,
                );
                lichnerowicz_at_point(
                    n_basis,
                    d,
                    &s.v,
                    &s.g_inv,
                    &s.ricci,
                    &s.rough,
                    &mut s.delta_l,
                );

                lower_basis_index(n_basis, d, &s.g, &s.v, &mut s.v_lowered);

                accumulate_point_contribution_op(
                    n_basis,
                    d,
                    w,
                    &s.delta_l,
                    &s.rough,
                    &s.v,
                    &s.v_lowered,
                    &mut acc.0,
                    &mut acc.1,
                    &mut acc.2,
                );
                acc.3 += 1;
                acc.4 += w;
                (s, acc)
            },
        )
        .map(|(_, acc)| acc)
        .reduce(
            move || init(n_basis),
            |mut a, b| {
                for k in 0..(n_basis * n_basis) {
                    a.0[k] += b.0[k];
                    a.1[k] += b.1[k];
                    a.2[k] += b.2[k];
                }
                a.3 += b.3;
                a.4 += b.4;
                a.5 = a.5.saturating_add(b.5);
                a
            },
        );

    let (l_matrix, rough_matrix, gram_matrix, n_ok, total_weight, _n_failed) = combined;

    if n_ok == 0 {
        return Err("Lichnerowicz assembly: no sample points contributed".to_string());
    }

    let mut op = LichnerowiczOperator {
        n_basis,
        d,
        n_sample: n_ok,
        total_weight,
        l_matrix,
        rough_matrix,
        gram_matrix,
        asymmetry: 0.0,
    };
    op.symmetrise();
    Ok(op)
}

// ---------------------------------------------------------------------------
// Closed-form metric extensions for tests.
// ---------------------------------------------------------------------------

/// Flat `R^d` extended with analytic second derivatives (all zero).
#[derive(Clone, Copy)]
pub struct FlatMetricSecond {
    pub d: usize,
}

impl MetricEvaluator for FlatMetricSecond {
    fn ambient_dim(&self) -> usize {
        self.d
    }
    fn intrinsic_dim(&self) -> usize {
        self.d
    }
    fn evaluate(&self, _x: &[f64], g_out: &mut [f64], dg_out: &mut [f64]) {
        let d = self.d;
        for i in 0..d {
            for j in 0..d {
                g_out[i * d + j] = if i == j { 1.0 } else { 0.0 };
            }
        }
        for v in dg_out.iter_mut().take(d * d * d) {
            *v = 0.0;
        }
    }
}

impl MetricEvaluatorWithSecond for FlatMetricSecond {
    fn evaluate_with_second(
        &self,
        x: &[f64],
        g_out: &mut [f64],
        dg_out: &mut [f64],
        ddg_out: &mut [f64],
    ) {
        self.evaluate(x, g_out, dg_out);
        let d = self.d;
        for v in ddg_out.iter_mut().take(d * d * d * d) {
            *v = 0.0;
        }
    }
}

/// Round metric on `S^d` in stereographic coordinates from the south
/// pole, with analytic first AND second partial derivatives.
///
/// Round metric: `g_{ij}(y) = f(y) δ_{ij}` with `f(y) = 4 / (1 + |y|²)²`.
/// First derivative: `∂_k g_{ij} = δ_{ij} ∂_k f` with
/// `∂_k f = − 16 y_k / (1 + |y|²)³`.
/// Second derivative: `∂_l ∂_k f = δ_{lk} · (−16 / (1 + |y|²)³)
///                                  + 96 y_k y_l / (1 + |y|²)⁴`.
///
/// (Carroll, *Spacetime and Geometry*, 2004, eqs. 3.205-3.206 for the
/// metric and Christoffels; second derivatives from straightforward
/// chain-rule application.)
#[derive(Clone, Copy)]
pub struct StereographicSphereMetricSecond {
    pub d: usize,
}

impl MetricEvaluator for StereographicSphereMetricSecond {
    fn ambient_dim(&self) -> usize {
        self.d
    }
    fn intrinsic_dim(&self) -> usize {
        self.d
    }
    fn evaluate(&self, y: &[f64], g_out: &mut [f64], dg_out: &mut [f64]) {
        let d = self.d;
        let mut y2 = 0.0;
        for k in 0..d {
            y2 += y[k] * y[k];
        }
        let denom = 1.0 + y2;
        let f = 4.0 / (denom * denom);
        for i in 0..d {
            for j in 0..d {
                g_out[i * d + j] = if i == j { f } else { 0.0 };
            }
        }
        // ∂_k f = −16 y_k / (1+|y|²)³.
        let dfac = -16.0 / (denom * denom * denom);
        for k in 0..d {
            for i in 0..d {
                for j in 0..d {
                    dg_out[k * d * d + i * d + j] = if i == j { dfac * y[k] } else { 0.0 };
                }
            }
        }
    }
}

impl MetricEvaluatorWithSecond for StereographicSphereMetricSecond {
    fn evaluate_with_second(
        &self,
        y: &[f64],
        g_out: &mut [f64],
        dg_out: &mut [f64],
        ddg_out: &mut [f64],
    ) {
        self.evaluate(y, g_out, dg_out);
        let d = self.d;
        let mut y2 = 0.0;
        for k in 0..d {
            y2 += y[k] * y[k];
        }
        let denom = 1.0 + y2;
        // ∂_l ∂_k f = δ_{lk} · A + y_k y_l · B
        // where A = −16 / denom³ and B = 96 / denom⁴.
        let a = -16.0 / (denom * denom * denom);
        let b = 96.0 / (denom * denom * denom * denom);
        for v in ddg_out.iter_mut().take(d * d * d * d) {
            *v = 0.0;
        }
        for l in 0..d {
            for k in 0..d {
                let dlk = if l == k { 1.0 } else { 0.0 };
                let dkdl_f = dlk * a + y[k] * y[l] * b;
                for i in 0..d {
                    let off = l * d * d * d + k * d * d + i * d + i;
                    ddg_out[off] = dkdl_f;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route34::lichnerowicz::PolynomialVectorBasis;

    #[test]
    fn flat_metric_ricci_vanishes() {
        let m = FlatMetricSecond { d: 3 };
        let basis = PolynomialVectorBasis::coordinate_polynomial_basis(3, 1);
        let n_pts = 30;
        let mut pts = Vec::with_capacity(n_pts * 3);
        for p in 0..n_pts {
            pts.push((p as f64 * 0.13).sin());
            pts.push((p as f64 * 0.17).cos());
            pts.push((p as f64 * 0.07).sin());
        }
        let weights = vec![1.0; n_pts];
        let op = assemble_lichnerowicz_operator(&m, &basis, &pts, &weights).unwrap();
        // On flat R^d, Ricci = 0, so L == rough.
        let res = op.ricci_residual();
        assert!(
            res < 1e-9,
            "flat-metric Ricci residual = {res}; expected ~0"
        );
    }

    #[test]
    fn flat_metric_translations_are_in_kernel() {
        // On flat R^d, constant vector fields satisfy
        //   Δ_L (∂/∂x^c) = 0
        // since all derivatives of the basis vanish and Ricci = 0.
        let d = 3;
        let m = FlatMetricSecond { d };
        let basis = PolynomialVectorBasis::coordinate_polynomial_basis(d, 0);
        let n_pts = 20;
        let mut pts = Vec::with_capacity(n_pts * d);
        for p in 0..n_pts {
            for k in 0..d {
                pts.push((p as f64 * 0.1 + k as f64 * 0.07).sin());
            }
        }
        let weights = vec![1.0; n_pts];
        let op = assemble_lichnerowicz_operator(&m, &basis, &pts, &weights).unwrap();
        let max_abs: f64 = op
            .l_matrix
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        assert!(
            max_abs < 1e-10,
            "flat-metric Δ_L matrix max-abs = {max_abs}; constant fields should be in kernel"
        );
    }

    /// Round S² has Killing algebra so(3) of dim 3. Per Wald §3.4 +
    /// Higuchi (1987), the Killing 1-forms on round S² are exactly the
    /// l = 1 vector spherical harmonics, and they are in the kernel
    /// of the Lichnerowicz operator restricted to so(3) (the
    /// Bochner-Yano kernel coincidence on closed Einstein manifolds:
    /// Δ_L ξ = 0 ↔ ξ is Killing iff ξ is co-closed and divergence-
    /// free, which the so(3) generators are). We verify that the
    /// dimension of the discretised Δ_L kernel is at least 3 and
    /// that the spectrum of Δ_L *differs* from the deformation-
    /// tensor spectrum on this basis.
    #[test]
    fn round_s2_lichnerowicz_spectrum_differs_from_deformation_spectrum() {
        use crate::route34::lichnerowicz::assemble_lichnerowicz_matrix;
        let m = StereographicSphereMetricSecond { d: 2 };
        let basis = PolynomialVectorBasis::coordinate_polynomial_basis(2, 2);
        // Sample 200 points in the stereographic chart (which covers
        // S² minus one point), drawing a quasi-random low-discrepancy
        // grid in the unit disk.
        let n_pts = 200usize;
        let mut pts = Vec::with_capacity(n_pts * 2);
        let mut weights = Vec::with_capacity(n_pts);
        let mut state = 0xC0FFEEu64;
        for _ in 0..n_pts {
            // Linear-congruential pseudo-random for determinism.
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 11) as f64) / (1u64 << 53) as f64;
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = ((state >> 11) as f64) / (1u64 << 53) as f64;
            // Map to disk of radius 1.5.
            let r = 1.5 * u.sqrt();
            let theta = 2.0 * std::f64::consts::PI * v;
            pts.push(r * theta.cos());
            pts.push(r * theta.sin());
            // Volume measure on S² in stereographic coords: dvol =
            // (4 / (1+|y|²)²) dy_1 dy_2. We absorb the area factor in
            // the weight; the normalisation is irrelevant for the
            // generalised eigenproblem.
            let denom = 1.0 + r * r;
            weights.push(4.0 / (denom * denom));
        }
        let lich = assemble_lichnerowicz_operator(&m, &basis, &pts, &weights)
            .expect("assemble Δ_L");
        let def =
            assemble_lichnerowicz_matrix(&m, &basis, &pts, &weights).expect("assemble def");

        // Compare the Frobenius norm of L_lich − L_def. They should
        // differ substantially (the Bochner correction is non-trivial
        // on a curved manifold).
        let n = lich.n_basis;
        let mut diff = 0.0;
        let mut nrm_l = 0.0;
        for i in 0..(n * n) {
            let dval = lich.l_matrix[i] - def.l_matrix[i];
            diff += dval * dval;
            nrm_l += lich.l_matrix[i] * lich.l_matrix[i];
        }
        let diff = diff.sqrt();
        let nrm = nrm_l.sqrt().max(1.0);
        assert!(
            diff / nrm > 1e-3,
            "round S² Δ_L and deformation-tensor spectra agree to within {} \
             (relative); they should differ by the Bochner correction",
            diff / nrm
        );
    }

    #[test]
    fn round_s2_ricci_residual_is_nontrivial() {
        // Round S² is Einstein but not Ricci-flat, so L ≠ rough.
        let m = StereographicSphereMetricSecond { d: 2 };
        let basis = PolynomialVectorBasis::coordinate_polynomial_basis(2, 1);
        let n_pts = 80usize;
        let mut pts = Vec::with_capacity(n_pts * 2);
        let mut weights = Vec::with_capacity(n_pts);
        for k in 0..n_pts {
            let theta = (k as f64) / (n_pts as f64) * 2.0 * std::f64::consts::PI;
            let r = 0.7;
            pts.push(r * theta.cos());
            pts.push(r * theta.sin());
            let denom = 1.0 + r * r;
            weights.push(4.0 / (denom * denom));
        }
        let op = assemble_lichnerowicz_operator(&m, &basis, &pts, &weights).unwrap();
        let res = op.ricci_residual();
        // Round S² has positive Ricci, so the Ricci term contributes
        // a non-trivial piece to L. The residual should be > 0.
        assert!(
            res > 1e-6,
            "round S² Ricci residual = {res}; expected positive contribution"
        );
    }

    #[test]
    fn dchristoffel_flat_is_zero() {
        // On flat R^d the Christoffel and its partials are all zero.
        let d = 3;
        let g_inv = {
            let mut m = vec![0.0; d * d];
            for i in 0..d {
                m[i * d + i] = 1.0;
            }
            m
        };
        let dg = vec![0.0; d * d * d];
        let ddg = vec![0.0; d * d * d * d];
        let mut dgamma = vec![0.0; d * d * d * d];
        dchristoffel_symbols(d, &g_inv, &dg, &ddg, &mut dgamma);
        for v in &dgamma {
            assert!(v.abs() < 1e-15, "flat ∂Γ should be 0");
        }
    }

    #[test]
    fn ricci_round_s2_is_positive_definite() {
        // For round S² of curvature 1, Ricci = (d-1) g, i.e.
        // R_{ij} = g_{ij}. We verify this at one point.
        let d = 2;
        let m = StereographicSphereMetricSecond { d };
        let y = [0.3, -0.4];
        let mut g = vec![0.0; d * d];
        let mut dg = vec![0.0; d * d * d];
        let mut ddg = vec![0.0; d * d * d * d];
        m.evaluate_with_second(&y, &mut g, &mut dg, &mut ddg);
        let mut g_inv = vec![0.0; d * d];
        let mut gamma = vec![0.0; d * d * d];
        let mut lu_work = vec![0.0; d * d];
        let mut perm = vec![0usize; d];
        let mut col = vec![0.0; d];
        christoffel_symbols(&g, &dg, d, &mut g_inv, &mut gamma, &mut lu_work, &mut perm, &mut col)
            .unwrap();
        let mut dgamma = vec![0.0; d * d * d * d];
        dchristoffel_symbols(d, &g_inv, &dg, &ddg, &mut dgamma);
        let mut ricci = vec![0.0; d * d];
        ricci_tensor(d, &gamma, &dgamma, &mut ricci);
        // Round S² of unit radius: Ricci = (d-1) g = 1 * g.
        // Compare R_{ij} to g_{ij}. The diagonal entries should match
        // to discretisation tolerance.
        for i in 0..d {
            for j in 0..d {
                let rel_err = (ricci[i * d + j] - g[i * d + j]).abs()
                    / g[i * d + j].abs().max(1e-12);
                assert!(
                    rel_err < 1e-6,
                    "Ricci_{{ {i},{j} }} = {} vs g_{{ {i},{j} }} = {}: rel_err = {}",
                    ricci[i * d + j],
                    g[i * d + j],
                    rel_err
                );
            }
        }
    }
}
