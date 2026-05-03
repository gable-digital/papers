//! Discrete **deformation-tensor** quadratic form on vector fields,
//! whose kernel is precisely the Killing algebra of a Riemannian
//! manifold.
//!
//! NOTE: this module is **not** the Lichnerowicz vector-Laplacian
//! `Δ_L = ∇^μ ∇_μ + R^ν_μ` itself; it is a different (positive-
//! semidefinite) symmetric form built from the deformation tensor
//! `L_ξ g = ∇_μ ξ_ν + ∇_ν ξ_μ`. The two have the **same kernel** on
//! any Riemannian manifold (both vanish iff `ξ` is Killing — Wald
//! 1984 eq. 3.4.1) but **different non-zero spectra**, related by the
//! Bochner identity (Petersen, *Riemannian Geometry* 3rd ed., §9.3,
//! eq. 9.3.5). For the proper Δ_L (with its own non-zero spectrum)
//! see [`crate::route34::lichnerowicz_operator`]. The module is
//! re-exported under the name `deformation_tensor` from the parent
//! mod.
//!
//! ## Mathematical setup
//!
//! For a Riemannian manifold `(M, g)` of real dimension `d`, the
//! **deformation tensor** of a vector field `ξ` is
//!
//! ```text
//!     (Lξ g)_{μν}  =  ∇_μ ξ_ν + ∇_ν ξ_μ
//!                  =  2 ∇_(μ ξ_ν)
//! ```
//!
//! and `ξ` is **Killing** iff `(Lξ g)_{μν} = 0` (Wald 1984, eq. 3.4.1).
//!
//! We assemble the symmetric quadratic form
//!
//! ```text
//!     L[ξ, η]  =  ∫_M  g^{μρ} g^{νσ} (Lξ g)_{μν} (Lη g)_{ρσ}  dvol_g
//! ```
//!
//! (the L²-norm-squared of the deformation tensor pairing two vector
//! fields). `L` is symmetric and positive-semidefinite; its kernel is
//! precisely the set of pairs `(ξ, η)` for which `Lξ g = Lη g = 0`,
//! i.e. the Killing algebra. This is the correct discretisation
//! whose kernel matches the analytic Killing dimension on **any**
//! Riemannian manifold (compact or not), not only on compact Ricci-
//! flat manifolds where the Lichnerowicz vector-Laplacian's kernel
//! coincides with the Killing algebra by Bochner-Yano.
//!
//! The relation to the more familiar Lichnerowicz vector-Laplacian
//! `Δ_L = −∇^μ ∇_μ + R^μ_ν` is via the (Bochner) identity
//!
//! ```text
//!     ⟨ξ, Δ_L ξ⟩  +  ⟨ξ, ∇ ∇·ξ⟩
//!         =  ½ ‖Lξ g‖²  +  ‖∇ξ‖²  −  ‖∇·ξ‖²
//! ```
//!
//! (Petersen, *Riemannian Geometry* 3rd ed., §9.3). For a divergence-
//! free `ξ` on a compact manifold, the right-hand side reduces to the
//! deformation-tensor norm and the kernel coincides with the Killing
//! algebra (which also includes the constraint that the divergence
//! vanishes). For our purposes the deformation-tensor form is more
//! direct: it has the correct kernel on any Riemannian manifold
//! without requiring compactness or Ricci-flatness assumptions, and
//! second derivatives of the basis are not needed (only first).
//!
//! ## Test-case kernels (analytic)
//!
//! * Round `S^d` ⊂ R^{d+1}: `dim Killing = d(d+1)/2 = dim so(d+1)`.
//! * Flat `R^d` (or its quotient `T^d`): `dim Killing = d + d(d-1)/2`
//!   (translations + rotations) for the local solver, restricted to
//!   `d` translations on the global torus. Our local-chart solver sees
//!   the full `d(d+1)/2` algebra of iso(R^d).
//! * Generic compact CY3: `dim Killing = 0` (Yau, CMP 1978).
//!
//! ## Discrete approximation
//!
//! We expand a candidate vector field in a basis `{V_a}_{a=1..N_basis}`
//! of vector fields on `M`, write
//!
//! ```text
//!     ξ^μ(x)  =  Σ_a c_a (V_a)^μ(x),
//! ```
//!
//! and compute the symmetric Gram-style matrix
//!
//! ```text
//!     L_{ab}  =  Σ_p w_p g_{μν}(x_p) (Δ_L V_a)^μ(x_p) (V_b)^ν(x_p)
//! ```
//!
//! at a finite collection of sample points `{x_p}` with quadrature
//! weights `{w_p}`. (We also assemble the vector-field Gram matrix
//! `G_{ab}` and reduce the generalised eigenproblem `L c = λ G c` to a
//! standard eigenproblem via Cholesky-like whitening; see
//! [`crate::route34::killing_solver`].) The kernel of `L` (relative to
//! `G`) is then the discrete Killing algebra.
//!
//! Christoffel symbols are computed from `g` and `∂g`:
//!
//! ```text
//!     Γ^λ_{μν}  =  (1/2) g^{λρ} ( ∂_μ g_{νρ} + ∂_ν g_{μρ} − ∂_ρ g_{μν} )
//! ```
//!
//! (Wald 1984, eq. 3.1.30). The covariant Laplacian on a vector field
//! in components is
//!
//! ```text
//!     (∇^μ ∇_μ ξ)^ν  =  g^{μρ} ( ∂_μ ∂_ρ ξ^ν
//!                                +  Γ^ν_{μλ} ∂_ρ ξ^λ
//!                                +  ∂_μ Γ^ν_{ρλ} · ξ^λ
//!                                +  Γ^ν_{ρλ} ∂_μ ξ^λ
//!                                −  Γ^σ_{μρ} ∂_σ ξ^ν
//!                                −  Γ^σ_{μρ} Γ^ν_{σλ} ξ^λ
//!                                +  Γ^ν_{μσ} Γ^σ_{ρλ} ξ^λ
//!                                +  Γ^ν_{ρσ} Γ^σ_{μλ} ξ^λ )
//! ```
//!
//! (expanded form of `g^{μρ} ∇_μ ∇_ρ ξ^ν` after writing
//! `∇_ρ ξ^ν = ∂_ρ ξ^ν + Γ^ν_{ρλ} ξ^λ` and applying `∇_μ` again,
//! then symmetrising on `(μ, ρ)` via the inverse metric). We compute
//! `∂g` analytically (when the metric is given by closed-form
//! evaluators in test cases) or by central differences on the sample
//! grid in production.
//!
//! ## Coordinate domain
//!
//! The solver is parameterised by an ambient Euclidean dimension `D`
//! and an intrinsic manifold dimension `d ≤ D`. The metric and basis
//! evaluators receive `D`-dimensional points; `d` is the dimension of
//! the metric tensor (the indices `μ, ν, λ, ρ, σ` run over `1..=d`).
//! For an embedded `S^3 ⊂ R^4` we have `D = 4`, `d = 3`; for a flat
//! `T^d` we have `D = d`. For the polysphere ambient `S^3 × S^3` of
//! the discrimination pipeline, `D = 8` and `d = 6`.
//!
//! ## Numerical conventions
//!
//! * All matrices are stored row-major as flat `Vec<f64>` (cache-
//!   friendly Struct-of-Arrays).
//! * No allocation in hot loops; per-thread scratch buffers are
//!   carried as `LichnerowiczScratch` and reused across all sample
//!   points handled by that thread.
//! * Symmetry of `L` is enforced by averaging `(L + Lᵀ)/2` at the end.
//! * Determinism: given the same metric/basis/sampler/seed, the
//!   assembled matrix is bit-identical across runs (the rayon
//!   reductions use `fold`+`reduce` with associative summation, so
//!   ordering can vary by ULP across thread counts; we explicitly
//!   `+`/`-` accumulate in a single thread for the final reduction
//!   step to make the output bit-deterministic for a fixed thread
//!   count).
//!
//! ## References
//!
//! * Wald, "General Relativity", University of Chicago Press, 1984,
//!   eqs. 3.1.30, 3.4.1.
//! * Carroll, "Spacetime and Geometry", Addison-Wesley, 2004, §3.2.
//! * Besse, "Einstein Manifolds", Springer, 1987, §1.K.
//! * Yau, "On the Ricci curvature of a compact Kähler manifold and the
//!   complex Monge-Ampère equation, I", *Comm. Pure Appl. Math.* 31
//!   (1978) 339–411.
//! * Bochner-Yano, "Curvature and Betti Numbers", Princeton, 1953.

use rayon::prelude::*;

use crate::linalg::{gemm, invert};

// ---------------------------------------------------------------------------
// Public traits and data structures
// ---------------------------------------------------------------------------

/// A Riemannian metric evaluator on an ambient Euclidean space `R^D`,
/// restricted to a (possibly embedded) `d`-dimensional manifold.
///
/// Implementations supply, at any sample point `x ∈ R^D`,
///
/// * the metric tensor `g_{ij}(x)` as a flat `d × d` row-major array,
///   stored in `g_out[..d * d]`,
/// * its first partial derivatives `∂_k g_{ij}(x)` as a `d × d × d`
///   array stored in `dg_out[k * d * d + i * d + j]`,
///
/// where the index `k` ranges over the **ambient** coordinate index
/// (`0..D`) when `D > d` (the embedded case) but the metric components
/// `i, j` are intrinsic-frame indices (`0..d`).
///
/// For test cases with `D == d` (flat torus, intrinsic coordinates on a
/// chart) the ambient and intrinsic indices coincide and the partials
/// `∂_k g_{ij}` are the usual chart-coordinate partials.
///
/// ### Embedded case (`D > d`)
///
/// When the manifold is embedded (e.g. `S^3 ⊂ R^4`), implementations
/// must return `g_{ij}` as the **pulled-back induced metric in an
/// orthonormal tangent frame** at `x`. The sample-time tangent frame
/// is selected by the caller via [`MetricEvaluator::tangent_frame`],
/// which returns an orthonormal `D × d` matrix `e^a_μ` (`a` ambient,
/// `μ` intrinsic). The metric components in that frame are
/// `g_{μν}(x) = δ_{μν}` for an intrinsic embedding (Riemannian
/// pullback into an orthonormal frame), but the partials of `g` are
/// nonzero because the frame itself rotates with `x`.
///
/// In practice we therefore compute Christoffel symbols directly from
/// `g` and `∂g` in whatever chart / frame the implementation provides,
/// without assuming the metric is diagonal.
pub trait MetricEvaluator: Sync {
    /// Ambient (Euclidean) dimension `D`.
    fn ambient_dim(&self) -> usize;
    /// Intrinsic (manifold) dimension `d`.
    fn intrinsic_dim(&self) -> usize;
    /// Evaluate `g_{ij}(x)` and `∂_k g_{ij}(x)` at the sample point `x`.
    ///
    /// `x.len() == ambient_dim()`; `g_out.len() >= d*d`;
    /// `dg_out.len() >= d*d*d` (intrinsic-coord partials only).
    ///
    /// The partial-derivative index `k` ranges over `0..d` (intrinsic).
    /// Implementations whose chart is an embedding `R^D ↪ M` must
    /// project the ambient `∂g/∂x^k` onto the intrinsic tangent frame
    /// before writing into `dg_out`.
    fn evaluate(&self, x: &[f64], g_out: &mut [f64], dg_out: &mut [f64]);

    /// Optional per-point tangent frame `e^a_μ` (ambient × intrinsic).
    /// Used by [`VectorFieldBasis`] implementations that work in
    /// ambient coordinates and need to project to intrinsic indices.
    /// Return `None` if the chart is intrinsic (`D == d`).
    fn tangent_frame(&self, x: &[f64], frame_out: &mut [f64]) -> bool {
        let _ = (x, frame_out);
        false
    }
}

/// A finite basis of vector fields on the manifold.
///
/// Implementations supply, at any sample point `x`,
///
/// * the basis vectors `(V_a)^μ(x)` as a `n_basis × d` array,
/// * their first partial derivatives `∂_k (V_a)^μ(x)` as a
///   `n_basis × d × d` array (`k` runs over intrinsic),
/// * their second partial derivatives `∂_k ∂_l (V_a)^μ(x)` as a
///   `n_basis × d × d × d` array (`k, l` intrinsic; symmetric in the
///   two derivative indices, but the implementation may freely write
///   the full square slot).
pub trait VectorFieldBasis: Sync {
    /// Number of basis vector fields.
    fn n_basis(&self) -> usize;
    /// Intrinsic dimension `d` (must agree with the metric's).
    fn intrinsic_dim(&self) -> usize;
    /// Evaluate the basis at the sample point `x`.
    ///
    /// `v_out.len() >= n_basis * d`,
    /// `dv_out.len() >= n_basis * d * d`,
    /// `ddv_out.len() >= n_basis * d * d * d`.
    fn evaluate(
        &self,
        x: &[f64],
        v_out: &mut [f64],
        dv_out: &mut [f64],
        ddv_out: &mut [f64],
    );
}

/// Pre-allocated workspace for a single thread's contribution to the
/// Lichnerowicz matrix assembly. All buffers are sized once at
/// construction and reused across all sample points handled by that
/// thread.
pub struct LichnerowiczScratch {
    pub d: usize,
    pub n_basis: usize,
    /// `g_{ij}` row-major `d × d`.
    pub g: Vec<f64>,
    /// `g^{ij}` row-major `d × d`.
    pub g_inv: Vec<f64>,
    /// `∂_k g_{ij}` row-major `d × d × d`.
    pub dg: Vec<f64>,
    /// `Γ^λ_{μν}` row-major `d × d × d`.
    pub gamma: Vec<f64>,
    /// `(V_a)^μ` row-major `n_basis × d`.
    pub v: Vec<f64>,
    /// `∂_k (V_a)^μ` row-major `n_basis × d × d`.
    pub dv: Vec<f64>,
    /// `∂_k ∂_l (V_a)^μ` row-major `n_basis × d × d × d`.
    pub ddv: Vec<f64>,
    /// `(L V_a g)_{μν}` row-major `n_basis × d × d` (lowered deformation
    /// tensor of each basis field; symmetric in μν).
    pub lkv: Vec<f64>,
    /// `g_{μν} (V_b)^ν` row-major `n_basis × d` (lowered basis).
    pub v_lowered: Vec<f64>,
    /// LU workspace for inverting `g_{ij}`.
    pub lu_work: Vec<f64>,
    pub lu_perm: Vec<usize>,
    pub lu_col: Vec<f64>,
    /// Per-thread accumulator for the local L matrix `n_basis^2`.
    pub l_local: Vec<f64>,
    /// Per-thread accumulator for the local Gram matrix `n_basis^2`.
    pub g_local: Vec<f64>,
}

impl LichnerowiczScratch {
    /// Allocate scratch buffers for an `(n_basis, d)` problem.
    pub fn new(n_basis: usize, d: usize) -> Self {
        Self {
            d,
            n_basis,
            g: vec![0.0; d * d],
            g_inv: vec![0.0; d * d],
            dg: vec![0.0; d * d * d],
            gamma: vec![0.0; d * d * d],
            v: vec![0.0; n_basis * d],
            dv: vec![0.0; n_basis * d * d],
            ddv: vec![0.0; n_basis * d * d * d],
            lkv: vec![0.0; n_basis * d * d],
            v_lowered: vec![0.0; n_basis * d],
            lu_work: vec![0.0; d * d],
            lu_perm: vec![0usize; d],
            lu_col: vec![0.0; d],
            l_local: vec![0.0; n_basis * n_basis],
            g_local: vec![0.0; n_basis * n_basis],
        }
    }

    /// Zero the per-thread accumulators (`l_local`, `g_local`).
    /// Other buffers are overwritten on each sample point and need not
    /// be cleared.
    pub fn reset_accumulators(&mut self) {
        for v in self.l_local.iter_mut() {
            *v = 0.0;
        }
        for v in self.g_local.iter_mut() {
            *v = 0.0;
        }
    }
}

/// Assembled discrete Lichnerowicz operator on a vector-field basis.
///
/// Both `l_matrix` and `gram_matrix` are symmetric `n_basis × n_basis`
/// row-major matrices. The discrete generalised eigenproblem
///
/// ```text
///     L c = λ G c
/// ```
///
/// has eigenvalue 0 (within the discretisation tolerance) precisely on
/// the discrete Killing algebra; non-zero eigenvalues correspond to
/// non-Killing vector fields in the basis.
///
/// `n_sample` and `total_weight` are reproducibility metadata.
pub struct LichnerowiczOperator {
    pub n_basis: usize,
    pub d: usize,
    pub n_sample: usize,
    pub total_weight: f64,
    pub l_matrix: Vec<f64>,
    pub gram_matrix: Vec<f64>,
    /// Maximum absolute asymmetry of `L` (informational; assembly
    /// symmetrises before returning).
    pub asymmetry: f64,
}

impl LichnerowiczOperator {
    /// Sanitise the matrices: enforce exact symmetry by averaging
    /// `(M + Mᵀ)/2`. Updates `asymmetry` to the pre-symmetrisation
    /// maximum off-diagonal `|L_ab − L_ba|`.
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
                let sym = 0.5 * (a + b);
                self.l_matrix[i * n + j] = sym;
                self.l_matrix[j * n + i] = sym;
                let ga = self.gram_matrix[i * n + j];
                let gb = self.gram_matrix[j * n + i];
                let gsym = 0.5 * (ga + gb);
                self.gram_matrix[i * n + j] = gsym;
                self.gram_matrix[j * n + i] = gsym;
            }
        }
        self.asymmetry = max_asym;
    }
}

// ---------------------------------------------------------------------------
// Christoffel symbols and covariant Laplacian
// ---------------------------------------------------------------------------

/// Compute Christoffel symbols `Γ^λ_{μν}` in place from `g` and `∂g`.
///
/// `g` is `d × d` row-major.
/// `dg` is `d × d × d` with `dg[k * d * d + i * d + j] = ∂_k g_{ij}`.
/// `gamma_out[lambda * d * d + mu * d + nu] = Γ^λ_{μν}`, written by
/// the standard formula `(1/2) g^{λρ} ( ∂_μ g_{νρ} + ∂_ν g_{μρ} − ∂_ρ g_{μν} )`.
///
/// `g_inv_out` is filled with `g^{ij}`. The LU workspace `lu_work`,
/// `lu_perm`, `lu_col` are passed through to [`crate::linalg::invert`].
///
/// Returns `Err` if `g` is exactly singular at this sample point.
pub fn christoffel_symbols(
    g: &[f64],
    dg: &[f64],
    d: usize,
    g_inv_out: &mut [f64],
    gamma_out: &mut [f64],
    lu_work: &mut [f64],
    lu_perm: &mut [usize],
    lu_col: &mut [f64],
) -> Result<(), &'static str> {
    invert(g, d, lu_work, lu_perm, g_inv_out, lu_col)?;
    for lam in 0..d {
        for mu in 0..d {
            for nu in 0..d {
                let mut s = 0.0;
                for rho in 0..d {
                    let dmu_g_nurho = dg[mu * d * d + nu * d + rho];
                    let dnu_g_murho = dg[nu * d * d + mu * d + rho];
                    let drho_g_munu = dg[rho * d * d + mu * d + nu];
                    s += g_inv_out[lam * d + rho]
                        * (dmu_g_nurho + dnu_g_murho - drho_g_munu);
                }
                gamma_out[lam * d * d + mu * d + nu] = 0.5 * s;
            }
        }
    }
    Ok(())
}

/// Compute the lowered deformation tensor `(L V_a g)_{μν}` for every
/// basis vector `V_a` at one sample point, using `V`, `∂V`,
/// `g_{ij}`, and `Γ^λ_{μν}`.
///
/// The deformation tensor is `(L V g)_{μν} = ∇_μ V_ν + ∇_ν V_μ` where
///
/// ```text
///     ∇_μ V_ν  =  ∂_μ V_ν − Γ^λ_{μν} V_λ
///              =  g_{νλ} ∂_μ V^λ + V^λ ∂_μ g_{λν}
///                  − Γ^λ_{μν} g_{λρ} V^ρ
/// ```
///
/// We compute `(L V g)_{μν}` directly from `V^λ` and its partials by
///
/// ```text
///     (L V g)_{μν}  =  ∂_μ (g_{νλ} V^λ)  +  ∂_ν (g_{μλ} V^λ)
///                       − 2 Γ^λ_{μν} g_{λρ} V^ρ
/// ```
///
/// (Wald 1984 eq. 3.1.30 + 3.4.1, Carroll 2004 §3.2). Equivalently
///
/// ```text
///     (L V g)_{μν}  =  g_{νλ} ∂_μ V^λ + g_{μλ} ∂_ν V^λ
///                       + (∂_μ g_{νλ} + ∂_ν g_{μλ}) V^λ
///                       − 2 Γ^λ_{μν} g_{λρ} V^ρ
/// ```
///
/// The result is symmetric in `(μ, ν)`.
///
/// Storage: `lkv_out[a * d * d + mu * d + nu] = (L V_a g)_{μν}`.
/// `lkv_out` must be at least `n_basis * d * d` long.
#[allow(clippy::too_many_arguments)]
pub fn killing_deformation_at_point(
    n_basis: usize,
    d: usize,
    v: &[f64],
    dv: &[f64],
    g: &[f64],
    dg: &[f64],
    gamma: &[f64],
    lkv_out: &mut [f64],
) {
    debug_assert!(v.len() >= n_basis * d);
    debug_assert!(dv.len() >= n_basis * d * d);
    debug_assert!(g.len() >= d * d);
    debug_assert!(dg.len() >= d * d * d);
    debug_assert!(gamma.len() >= d * d * d);
    debug_assert!(lkv_out.len() >= n_basis * d * d);

    for v_out in lkv_out.iter_mut().take(n_basis * d * d) {
        *v_out = 0.0;
    }

    for a in 0..n_basis {
        let v_a = &v[a * d..(a + 1) * d];
        let dv_a = &dv[a * d * d..(a + 1) * d * d];
        // Pre-compute (∇_μ V^λ) for this basis vector: nabla[mu * d + lam].
        // ∇_μ V^λ = ∂_μ V^λ + Γ^λ_{μρ} V^ρ.
        let mut nabla = [0.0f64; 64]; // d ≤ 8 cap; widen if needed
        debug_assert!(d * d <= 64, "killing_deformation_at_point: d ≤ 8 supported");
        for mu in 0..d {
            for lam in 0..d {
                let mut s = dv_a[mu * d + lam];
                for rho in 0..d {
                    s += gamma[lam * d * d + mu * d + rho] * v_a[rho];
                }
                nabla[mu * d + lam] = s;
            }
        }
        // (L V g)_{μν} = g_{νλ} ∇_μ V^λ + g_{μλ} ∇_ν V^λ
        for mu in 0..d {
            for nu in 0..d {
                let mut s = 0.0;
                for lam in 0..d {
                    s += g[nu * d + lam] * nabla[mu * d + lam];
                    s += g[mu * d + lam] * nabla[nu * d + lam];
                }
                lkv_out[a * d * d + mu * d + nu] = s;
            }
        }
    }
    // Suppress unused: dg passed in for API symmetry / future
    // diagnostics; metric compatibility guarantees the alternative
    // formula can use g and Γ alone.
    let _ = dg;
}

/// Lower the basis index: `(V_a)_μ = g_{μν} (V_a)^ν`.
pub fn lower_basis_index(
    n_basis: usize,
    d: usize,
    g: &[f64],
    v: &[f64],
    v_lowered: &mut [f64],
) {
    for a in 0..n_basis {
        for mu in 0..d {
            let mut s = 0.0;
            for nu in 0..d {
                s += g[mu * d + nu] * v[a * d + nu];
            }
            v_lowered[a * d + mu] = s;
        }
    }
}

// ---------------------------------------------------------------------------
// Per-point contributions to L and G
// ---------------------------------------------------------------------------

/// Accumulate one sample point's contribution to the local L and Gram
/// matrices.
///
/// `weight` is the quadrature weight `w_p` for this sample point.
///
/// ```text
///   L_local[a, b]  +=  w · g^{μρ} g^{νσ} (L V_a g)_{μν} (L V_b g)_{ρσ}
///   G_local[a, b]  +=  w · g_{μν} (V_a)^μ (V_b)^ν
/// ```
///
/// `lkv` is the basis-deformation tensor `n_basis × d × d` from
/// [`killing_deformation_at_point`]; `g_inv` is the inverse metric;
/// `v_lowered_b_μ = g_{μν} V_b^ν` for the Gram matrix.
#[allow(clippy::too_many_arguments)]
fn accumulate_point_contribution(
    n_basis: usize,
    d: usize,
    weight: f64,
    lkv: &[f64],
    g_inv: &[f64],
    v: &[f64],
    v_lowered: &[f64],
    l_local: &mut [f64],
    g_local: &mut [f64],
) {
    debug_assert!(d * d <= 64, "accumulate_point_contribution: d ≤ 8 supported");

    // Pre-raise the deformation tensor for each basis vector once:
    // (L V_a)^{μν} = g^{μρ} g^{νσ} (L V_a)_{ρσ}. We do this for each `a`
    // upfront (cost O(n_basis · d^4)), then the (a, b) contraction is
    // O(n_basis^2 · d^2).
    let mut lkv_up = vec![0.0f64; n_basis * d * d];
    for a in 0..n_basis {
        let lk_a = &lkv[a * d * d..(a + 1) * d * d];
        // Step 1: half = g^{μρ} (L V_a)_{ρσ}. Stored as half[μ * d + σ].
        let mut half = [0.0f64; 64];
        for mu in 0..d {
            for sigma in 0..d {
                let mut s = 0.0;
                for rho in 0..d {
                    s += g_inv[mu * d + rho] * lk_a[rho * d + sigma];
                }
                half[mu * d + sigma] = s;
            }
        }
        // Step 2: (L V_a)^{μν} = half[μ, σ] · g^{σν}.
        for mu in 0..d {
            for nu in 0..d {
                let mut s = 0.0;
                for sigma in 0..d {
                    s += half[mu * d + sigma] * g_inv[sigma * d + nu];
                }
                lkv_up[a * d * d + mu * d + nu] = s;
            }
        }
    }

    for a in 0..n_basis {
        let lkv_up_a = &lkv_up[a * d * d..(a + 1) * d * d];
        let v_a = &v[a * d..(a + 1) * d];
        for b in 0..n_basis {
            let lkv_b = &lkv[b * d * d..(b + 1) * d * d];
            let vlow_b = &v_lowered[b * d..(b + 1) * d];
            let mut l_dot = 0.0;
            for mu in 0..d {
                for nu in 0..d {
                    l_dot += lkv_up_a[mu * d + nu] * lkv_b[mu * d + nu];
                }
            }
            let mut g_dot = 0.0;
            for mu in 0..d {
                g_dot += v_a[mu] * vlow_b[mu];
            }
            l_local[a * n_basis + b] += weight * l_dot;
            g_local[a * n_basis + b] += weight * g_dot;
        }
    }
}

// ---------------------------------------------------------------------------
// High-level matrix assembly
// ---------------------------------------------------------------------------

/// Assemble the discrete Lichnerowicz matrix and vector-field Gram
/// matrix on a basis of vector fields, using a multi-threaded pool
/// over sample points.
///
/// `metric` and `basis` must agree on the intrinsic dimension `d`.
///
/// `sample_points` is `n_sample × D` row-major (D = ambient dimension);
/// `weights` is the per-point quadrature weight (length `n_sample`).
///
/// Returns the assembled [`LichnerowiczOperator`] (already
/// symmetrised). Returns `Err` if any sample point produces a singular
/// metric tensor.
pub fn assemble_lichnerowicz_matrix<M, B>(
    metric: &M,
    basis: &B,
    sample_points: &[f64],
    weights: &[f64],
) -> Result<LichnerowiczOperator, String>
where
    M: MetricEvaluator,
    B: VectorFieldBasis,
{
    let d_metric = metric.intrinsic_dim();
    let d_basis = basis.intrinsic_dim();
    if d_metric != d_basis {
        return Err(format!(
            "metric/basis dim mismatch: metric d = {}, basis d = {}",
            d_metric, d_basis
        ));
    }
    let _d = d_metric;
    let _n_basis = basis.n_basis();
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

    assemble_lichnerowicz_matrix_impl(metric, basis, sample_points, weights)
}

/// Internal implementation of [`assemble_lichnerowicz_matrix`].
///
/// Split out so the public function can present a clean error API while
/// keeping the rayon `fold`+`reduce` pipeline easy to maintain.
fn assemble_lichnerowicz_matrix_impl<M, B>(
    metric: &M,
    basis: &B,
    sample_points: &[f64],
    weights: &[f64],
) -> Result<LichnerowiczOperator, String>
where
    M: MetricEvaluator,
    B: VectorFieldBasis,
{
    let d = metric.intrinsic_dim();
    let n_basis = basis.n_basis();
    let ambient = metric.ambient_dim();
    let n_sample = sample_points.len() / ambient;
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_pts = ((n_sample + n_threads - 1) / n_threads).max(32);

    // Per-thread contributions: (l_chunk, g_chunk, accepted, wsum, failed).
    type Acc = (Vec<f64>, Vec<f64>, usize, f64, u64);
    let init: fn(usize, usize) -> Acc = |n_basis, _d| {
        (
            vec![0.0; n_basis * n_basis],
            vec![0.0; n_basis * n_basis],
            0usize,
            0.0f64,
            0u64,
        )
    };

    let combined: Acc = (0..n_sample)
        .into_par_iter()
        .with_min_len(chunk_pts)
        .fold(
            move || {
                (
                    LichnerowiczScratch::new(n_basis, d),
                    init(n_basis, d),
                )
            },
            |(mut scratch, mut acc), p_idx| {
                let pt = &sample_points[p_idx * ambient..(p_idx + 1) * ambient];
                let w = weights[p_idx];
                if !w.is_finite() || w <= 0.0 {
                    return (scratch, acc);
                }
                metric.evaluate(pt, &mut scratch.g, &mut scratch.dg);
                if !scratch.g.iter().take(d * d).all(|x| x.is_finite())
                    || !scratch.dg.iter().take(d * d * d).all(|x| x.is_finite())
                {
                    acc.4 = acc.4.saturating_add(1);
                    return (scratch, acc);
                }
                // Clone g into a temporary so christoffel_symbols can
                // freely use lu_work without aliasing g_inv. (g itself is
                // re-read after the inversion to lower the basis index.)
                let g_copy = scratch.g.clone();
                let chris_res = christoffel_symbols(
                    &g_copy,
                    &scratch.dg,
                    d,
                    &mut scratch.g_inv,
                    &mut scratch.gamma,
                    &mut scratch.lu_work,
                    &mut scratch.lu_perm,
                    &mut scratch.lu_col,
                );
                if chris_res.is_err() {
                    acc.4 = acc.4.saturating_add(1);
                    return (scratch, acc);
                }
                basis.evaluate(
                    pt,
                    &mut scratch.v,
                    &mut scratch.dv,
                    &mut scratch.ddv,
                );
                killing_deformation_at_point(
                    n_basis,
                    d,
                    &scratch.v,
                    &scratch.dv,
                    &scratch.g,
                    &scratch.dg,
                    &scratch.gamma,
                    &mut scratch.lkv,
                );
                lower_basis_index(
                    n_basis,
                    d,
                    &scratch.g,
                    &scratch.v,
                    &mut scratch.v_lowered,
                );
                accumulate_point_contribution(
                    n_basis,
                    d,
                    w,
                    &scratch.lkv,
                    &scratch.g_inv,
                    &scratch.v,
                    &scratch.v_lowered,
                    &mut acc.0,
                    &mut acc.1,
                );
                acc.2 += 1;
                acc.3 += w;
                (scratch, acc)
            },
        )
        .map(|(_scratch, acc)| acc)
        .reduce(
            move || init(n_basis, d),
            |mut a, b| {
                for k in 0..(n_basis * n_basis) {
                    a.0[k] += b.0[k];
                    a.1[k] += b.1[k];
                }
                a.2 += b.2;
                a.3 += b.3;
                a.4 = a.4.saturating_add(b.4);
                a
            },
        );

    let (l_matrix, gram_matrix, n_ok, total_weight, n_failed) = combined;

    if n_ok == 0 {
        return Err(format!(
            "Lichnerowicz assembly: no sample points contributed (n_failed = {})",
            n_failed
        ));
    }

    let mut op = LichnerowiczOperator {
        n_basis,
        d,
        n_sample: n_ok,
        total_weight,
        l_matrix,
        gram_matrix,
        asymmetry: 0.0,
    };
    op.symmetrise();
    Ok(op)
}

// ---------------------------------------------------------------------------
// Self-contained metric / basis evaluators for tests and validation
// ---------------------------------------------------------------------------

/// Flat `R^d` metric in standard coordinates (Christoffel = 0).
/// Used for the flat-torus validation case.
#[derive(Clone, Copy)]
pub struct FlatMetric {
    pub d: usize,
}

impl MetricEvaluator for FlatMetric {
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

/// Round metric on `S^d ⊂ R^{d+1}` of unit radius, in **stereographic
/// coordinates** from the south pole. The chart covers `S^d`
/// minus one point, which is sufficient for sampling-based tests.
///
/// In stereographic coordinates `y ∈ R^d`, the round metric is
///
/// ```text
///     g_{ij}(y)  =  (4 / (1 + |y|²)²) δ_{ij}
/// ```
///
/// so `g` is conformally flat. The partial derivatives are
///
/// ```text
///     ∂_k g_{ij}  =  −16 y_k / (1 + |y|²)³ · δ_{ij}.
/// ```
///
/// The Christoffel symbols are
/// `Γ^λ_{μν} = −2 (y_μ δ^λ_ν + y_ν δ^λ_μ − y^λ δ_{μν}) / (1 + |y|²)`
/// (Carroll 2004, eq. 3.205-3.206).
#[derive(Clone, Copy)]
pub struct StereographicSphereMetric {
    pub d: usize,
}

impl MetricEvaluator for StereographicSphereMetric {
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
        let factor = 4.0 / (denom * denom);
        for i in 0..d {
            for j in 0..d {
                g_out[i * d + j] = if i == j { factor } else { 0.0 };
            }
        }
        // ∂_k g_{ij} = -16 y_k / (1+|y|²)^3  for i==j
        let dfactor = -16.0 / (denom * denom * denom);
        for k in 0..d {
            for i in 0..d {
                for j in 0..d {
                    dg_out[k * d * d + i * d + j] = if i == j { dfactor * y[k] } else { 0.0 };
                }
            }
        }
    }
}

/// Direct sum of two metrics: the manifold is a product `M_1 × M_2`,
/// the metric is `g = g_1 ⊕ g_2`, and the chart is the cartesian
/// product of the two factor charts. Used for the `S^2 × S^2 × S^2`
/// test (build by composing two `ProductMetric`s).
pub struct ProductMetric<A: MetricEvaluator, B: MetricEvaluator> {
    pub left: A,
    pub right: B,
}

impl<A: MetricEvaluator, B: MetricEvaluator> MetricEvaluator
    for ProductMetric<A, B>
{
    fn ambient_dim(&self) -> usize {
        self.left.ambient_dim() + self.right.ambient_dim()
    }
    fn intrinsic_dim(&self) -> usize {
        self.left.intrinsic_dim() + self.right.intrinsic_dim()
    }
    fn evaluate(&self, x: &[f64], g_out: &mut [f64], dg_out: &mut [f64]) {
        let da = self.left.intrinsic_dim();
        let db = self.right.intrinsic_dim();
        let d = da + db;
        let aa = self.left.ambient_dim();
        let (xa, xb) = x.split_at(aa);

        let mut g_left = vec![0.0; da * da];
        let mut dg_left = vec![0.0; da * da * da];
        let mut g_right = vec![0.0; db * db];
        let mut dg_right = vec![0.0; db * db * db];
        self.left.evaluate(xa, &mut g_left, &mut dg_left);
        self.right.evaluate(xb, &mut g_right, &mut dg_right);

        // Block-diagonal g.
        for v in g_out.iter_mut().take(d * d) {
            *v = 0.0;
        }
        for i in 0..da {
            for j in 0..da {
                g_out[i * d + j] = g_left[i * da + j];
            }
        }
        for i in 0..db {
            for j in 0..db {
                g_out[(da + i) * d + (da + j)] = g_right[i * db + j];
            }
        }

        // Block-diagonal ∂g, with the partial-derivative index running
        // over the joint intrinsic space.
        for v in dg_out.iter_mut().take(d * d * d) {
            *v = 0.0;
        }
        // Left block: nonzero only when k < da and (i,j) both in left.
        for k in 0..da {
            for i in 0..da {
                for j in 0..da {
                    dg_out[k * d * d + i * d + j] = dg_left[k * da * da + i * da + j];
                }
            }
        }
        // Right block: nonzero only when k >= da and (i,j) both in right.
        for k in 0..db {
            for i in 0..db {
                for j in 0..db {
                    dg_out[(da + k) * d * d + (da + i) * d + (da + j)] =
                        dg_right[k * db * db + i * db + j];
                }
            }
        }
    }
}

/// A finite-dimensional basis of vector fields whose components are
/// polynomials in the chart coordinates. Used for the analytic
/// validation tests (Killing fields on a torus / sphere are
/// polynomial in the right charts).
///
/// Stored as a flat list of monomials. Each basis vector field is one
/// pair `(component_index, exponent_vector)`: the field `(c, e)`
/// represents the vector field
///
/// ```text
///     V^μ(y)  =  δ^μ_c · Π_k y_k^{e_k}.
/// ```
///
/// The `n_basis` count is the length of `entries`. Up to second
/// derivatives are computed analytically per call.
#[derive(Clone)]
pub struct PolynomialVectorBasis {
    pub d: usize,
    /// Each entry: `(component_index, exponent_vector_of_length_d)`.
    pub entries: Vec<(usize, Vec<u32>)>,
}

impl PolynomialVectorBasis {
    /// Build the standard degree-≤k coordinate-polynomial basis: every
    /// monomial of total degree ≤ `max_degree` in `d` variables, paired
    /// with each of the `d` component slots. Total size:
    /// `d * C(d + max_degree, max_degree)`.
    pub fn coordinate_polynomial_basis(d: usize, max_degree: u32) -> Self {
        let mut monomials: Vec<Vec<u32>> = Vec::new();
        let mut current = vec![0u32; d];
        Self::enumerate_monomials(d, 0, max_degree as i32, &mut current, &mut monomials);
        let mut entries = Vec::with_capacity(d * monomials.len());
        for c in 0..d {
            for m in &monomials {
                entries.push((c, m.clone()));
            }
        }
        Self { d, entries }
    }

    fn enumerate_monomials(
        d: usize,
        idx: usize,
        remaining: i32,
        current: &mut [u32],
        out: &mut Vec<Vec<u32>>,
    ) {
        if idx == d {
            out.push(current.to_vec());
            return;
        }
        for e in 0..=remaining {
            current[idx] = e as u32;
            Self::enumerate_monomials(d, idx + 1, remaining - e, current, out);
        }
    }
}

impl VectorFieldBasis for PolynomialVectorBasis {
    fn n_basis(&self) -> usize {
        self.entries.len()
    }
    fn intrinsic_dim(&self) -> usize {
        self.d
    }
    fn evaluate(
        &self,
        x: &[f64],
        v_out: &mut [f64],
        dv_out: &mut [f64],
        ddv_out: &mut [f64],
    ) {
        let d = self.d;
        let n = self.entries.len();
        // Pre-compute power tables y_k^e for e in 0..=kmax.
        let mut kmax = 0u32;
        for (_, exp) in &self.entries {
            for &e in exp {
                if e > kmax {
                    kmax = e;
                }
            }
        }
        let kmax = kmax as usize;
        let stride = kmax + 1;
        let mut pow = vec![1.0f64; d * stride];
        for k in 0..d {
            pow[k * stride] = 1.0;
            for e in 1..=kmax {
                pow[k * stride + e] = pow[k * stride + e - 1] * x[k];
            }
        }

        // Zero outputs.
        for v in v_out.iter_mut().take(n * d) {
            *v = 0.0;
        }
        for v in dv_out.iter_mut().take(n * d * d) {
            *v = 0.0;
        }
        for v in ddv_out.iter_mut().take(n * d * d * d) {
            *v = 0.0;
        }

        for (a, (c, exp)) in self.entries.iter().enumerate() {
            // V_a^μ = δ^μ_c · ∏ x_k^{e_k}
            let mut p_full = 1.0;
            for k in 0..d {
                p_full *= pow[k * stride + exp[k] as usize];
            }
            v_out[a * d + *c] = p_full;

            // ∂_k V_a^μ = δ^μ_c · e_k · x_k^{e_k - 1} · ∏_{l != k} x_l^{e_l}
            for k in 0..d {
                if exp[k] == 0 {
                    continue;
                }
                let factor = exp[k] as f64
                    * pow[k * stride + (exp[k] - 1) as usize];
                let mut prod_other = 1.0;
                for l in 0..d {
                    if l == k {
                        continue;
                    }
                    prod_other *= pow[l * stride + exp[l] as usize];
                }
                dv_out[a * d * d + k * d + *c] = factor * prod_other;
            }

            // ∂_k ∂_l V_a^μ
            for k in 0..d {
                for l in 0..d {
                    if k == l {
                        if exp[k] < 2 {
                            continue;
                        }
                        let f = exp[k] as f64
                            * (exp[k] as f64 - 1.0)
                            * pow[k * stride + (exp[k] - 2) as usize];
                        let mut prod_other = 1.0;
                        for m in 0..d {
                            if m == k {
                                continue;
                            }
                            prod_other *= pow[m * stride + exp[m] as usize];
                        }
                        ddv_out[a * d * d * d + k * d * d + l * d + *c] =
                            f * prod_other;
                    } else {
                        if exp[k] == 0 || exp[l] == 0 {
                            continue;
                        }
                        let f1 = exp[k] as f64
                            * pow[k * stride + (exp[k] - 1) as usize];
                        let f2 = exp[l] as f64
                            * pow[l * stride + (exp[l] - 1) as usize];
                        let mut prod_other = 1.0;
                        for m in 0..d {
                            if m == k || m == l {
                                continue;
                            }
                            prod_other *= pow[m * stride + exp[m] as usize];
                        }
                        ddv_out[a * d * d * d + k * d * d + l * d + *c] =
                            f1 * f2 * prod_other;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CY3 metric adapter
// ---------------------------------------------------------------------------

/// Adapter that exposes the **Donaldson-balanced Kähler metric**
/// derived from the section-basis coefficient matrix `h` as a
/// [`MetricEvaluator`] in real ambient `R^D` coordinates (`D = 8` for
/// the polysphere `S^3 × S^3` ambient of the discrimination pipeline,
/// `D = 6` post-tangent-projection).
///
/// The Kähler potential is `K(z) = sᵀ h s` and the real symmetric
/// metric in tangent-projected real coordinates is the Hessian of
/// `log K`, evaluated and projected by exactly the same code path as
/// [`crate::refine::monge_ampere_residual`]. We expose it here as a
/// [`MetricEvaluator`] so the Killing-vector solver can treat it
/// uniformly with the closed-form test cases.
///
/// **Caveat**: this evaluator currently uses the Hessian of `log K`
/// in the ambient `R^8` and projects to the `6`-dim tangent space of
/// `S^3 × S^3` at each sample point. The metric's chart-coordinate
/// partials `∂g` are computed by a **central finite difference** on
/// the ambient `R^8` coordinates with a fixed step `h_fd` (default
/// `1e-4`); the projection introduces an additional step error of
/// `O(h_fd²)` due to the moving tangent frame. Until a closed-form
/// `∂g` from the polynomial Kähler potential is wired through, this
/// is the unavoidable cost of the off-the-shelf adapter.
pub struct DonaldsonMetricAdapter<'a> {
    pub h: &'a [f64],
    pub n_basis: usize,
    pub k_degree: u32,
    /// Step for central finite-difference of `g`. `1e-4` is a good
    /// trade-off between truncation `O(h²) ≈ 1e-8` and round-off
    /// `O(ε / h) ≈ 1e-12 / h`.
    pub h_fd: f64,
}

impl<'a> DonaldsonMetricAdapter<'a> {
    pub fn new(h: &'a [f64], n_basis: usize, k_degree: u32) -> Self {
        Self {
            h,
            n_basis,
            k_degree,
            h_fd: 1e-4,
        }
    }
}

impl<'a> MetricEvaluator for DonaldsonMetricAdapter<'a> {
    fn ambient_dim(&self) -> usize {
        8
    }
    fn intrinsic_dim(&self) -> usize {
        6
    }
    fn evaluate(&self, x: &[f64], g_out: &mut [f64], dg_out: &mut [f64]) {
        // Compute g(x), then g(x ± h e_k) for each ambient coord k,
        // central-difference for ∂g/∂x_k, and project both g and ∂g
        // onto the 6-dim tangent space at x.
        let h_fd = self.h_fd;
        let mut x_buf = [0.0f64; 8];
        x_buf.copy_from_slice(&x[..8]);
        let g_at_x = donaldson_metric_at_ambient(&x_buf, self.h, self.n_basis, self.k_degree);
        // Project g_at_x (6x6) into g_out.
        for i in 0..6 {
            for j in 0..6 {
                g_out[i * 6 + j] = g_at_x[i * 6 + j];
            }
        }
        // Finite-difference partials w.r.t. the 6 *intrinsic*
        // (tangent) directions. We pick a fixed orthonormal tangent
        // frame at x and step along it to get ∂g/∂y^μ in that frame.
        let frame = polysphere_tangent_basis_8x6(&x_buf);
        for k in 0..6 {
            let mut x_plus = x_buf;
            let mut x_minus = x_buf;
            for a in 0..8 {
                x_plus[a] += h_fd * frame[a * 6 + k];
                x_minus[a] -= h_fd * frame[a * 6 + k];
            }
            // Re-normalise the two CP^3-factor radial constraints to
            // stay on S^3 × S^3.
            renormalise_polysphere(&mut x_plus);
            renormalise_polysphere(&mut x_minus);
            let g_plus = donaldson_metric_at_ambient(&x_plus, self.h, self.n_basis, self.k_degree);
            let g_minus =
                donaldson_metric_at_ambient(&x_minus, self.h, self.n_basis, self.k_degree);
            for i in 0..6 {
                for j in 0..6 {
                    let dval = (g_plus[i * 6 + j] - g_minus[i * 6 + j]) / (2.0 * h_fd);
                    dg_out[k * 6 * 6 + i * 6 + j] = dval;
                }
            }
        }
    }
}

/// Compute the Hessian of `log K(z)` at one polysphere point and
/// project it onto the 6-dim tangent frame, returning the 6x6 metric.
fn donaldson_metric_at_ambient(z: &[f64; 8], h: &[f64], n_basis: usize, k_deg: u32) -> [f64; 36] {
    let monomials = crate::refine::degree_k_monomials(k_deg);
    debug_assert_eq!(monomials.len(), n_basis);
    let (s, ds, dds) = crate::refine::evaluate_section_basis_with_derivs(z, monomials);
    // K = sᵀ h s
    let mut k_val = 0.0;
    let mut h_s = vec![0.0; n_basis];
    for a in 0..n_basis {
        let mut row_sum = 0.0;
        for b in 0..n_basis {
            row_sum += h[a * n_basis + b] * s[b];
        }
        h_s[a] = row_sum;
        k_val += s[a] * row_sum;
    }
    let k_safe = k_val.max(1e-30);

    let mut dk = [0.0f64; 8];
    for k_i in 0..8 {
        let dsk = &ds[k_i * n_basis..(k_i + 1) * n_basis];
        let mut sum = 0.0;
        for a in 0..n_basis {
            sum += dsk[a] * h_s[a];
        }
        dk[k_i] = 2.0 * sum;
    }

    let mut hess = [0.0f64; 64];
    for i in 0..8 {
        for j in i..8 {
            let pij = pack_ij_8(i, j);
            let dsi = &ds[i * n_basis..(i + 1) * n_basis];
            let dsj = &ds[j * n_basis..(j + 1) * n_basis];
            let dds_ij = &dds[pij * n_basis..(pij + 1) * n_basis];
            let mut term1 = 0.0;
            for a in 0..n_basis {
                term1 += dds_ij[a] * h_s[a];
            }
            let mut h_dsj = vec![0.0; n_basis];
            for a in 0..n_basis {
                let mut rs = 0.0;
                for b in 0..n_basis {
                    rs += h[a * n_basis + b] * dsj[b];
                }
                h_dsj[a] = rs;
            }
            let mut term2 = 0.0;
            for a in 0..n_basis {
                term2 += dsi[a] * h_dsj[a];
            }
            let val = 2.0 * (term1 + term2) / k_safe - (dk[i] * dk[j]) / (k_safe * k_safe);
            hess[i * 8 + j] = val;
            hess[j * 8 + i] = val;
        }
    }

    // Project 8x8 hess onto 6x6 tangent space.
    let p = polysphere_tangent_basis_8x6(z);
    let mut hp = [0.0f64; 48];
    for i in 0..8 {
        for j in 0..6 {
            let mut s2 = 0.0;
            for k_i in 0..8 {
                s2 += hess[i * 8 + k_i] * p[k_i * 6 + j];
            }
            hp[i * 6 + j] = s2;
        }
    }
    let mut h_tan = [0.0f64; 36];
    for i in 0..6 {
        for j in 0..6 {
            let mut s2 = 0.0;
            for k_i in 0..8 {
                s2 += p[k_i * 6 + i] * hp[k_i * 6 + j];
            }
            h_tan[i * 6 + j] = s2;
        }
    }
    h_tan
}

fn pack_ij_8(i: usize, j: usize) -> usize {
    if i == j {
        i
    } else {
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        let mut idx = 8;
        for a in 0..lo {
            idx += 7 - a;
        }
        idx += hi - lo - 1;
        idx
    }
}

fn polysphere_tangent_basis_8x6(z: &[f64; 8]) -> [f64; 48] {
    let mut candidates = [[0.0f64; 8]; 8];
    let za_dot = |v: &[f64; 8]| v[0] * z[0] + v[1] * z[1] + v[2] * z[2] + v[3] * z[3];
    let zb_dot = |v: &[f64; 8]| v[4] * z[4] + v[5] * z[5] + v[6] * z[6] + v[7] * z[7];
    for k in 0..8 {
        let mut v = [0.0f64; 8];
        v[k] = 1.0;
        let da = za_dot(&v);
        let db = zb_dot(&v);
        for i in 0..4 {
            v[i] -= da * z[i];
        }
        for i in 4..8 {
            v[i] -= db * z[i];
        }
        candidates[k] = v;
    }
    let mut p = [0.0f64; 48];
    let mut col = 0usize;
    for k in 0..8 {
        if col >= 6 {
            break;
        }
        let mut v = candidates[k];
        for j in 0..col {
            let mut dot = 0.0;
            for i in 0..8 {
                dot += p[i * 6 + j] * v[i];
            }
            for i in 0..8 {
                v[i] -= dot * p[i * 6 + j];
            }
        }
        let nrm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nrm > 1e-10 {
            for i in 0..8 {
                p[i * 6 + col] = v[i] / nrm;
            }
            col += 1;
        }
    }
    p
}

fn renormalise_polysphere(z: &mut [f64; 8]) {
    let n1: f64 = (z[0] * z[0] + z[1] * z[1] + z[2] * z[2] + z[3] * z[3]).sqrt();
    let n2: f64 = (z[4] * z[4] + z[5] * z[5] + z[6] * z[6] + z[7] * z[7]).sqrt();
    if n1 > 1e-12 {
        for i in 0..4 {
            z[i] /= n1;
        }
    }
    if n2 > 1e-12 {
        for i in 0..4 {
            z[4 + i] /= n2;
        }
    }
}

// ---------------------------------------------------------------------------
// Run-metadata
// ---------------------------------------------------------------------------

/// Run-metadata capturing reproducibility-relevant parameters for one
/// Lichnerowicz assembly.
#[derive(Debug, Clone)]
pub struct LichnerowiczRunMetadata {
    pub git_sha: String,
    pub seed: u64,
    pub n_sample: usize,
    pub n_basis: usize,
    pub d: usize,
    pub total_weight: f64,
    pub asymmetry: f64,
}

impl LichnerowiczRunMetadata {
    pub fn from_operator(op: &LichnerowiczOperator, seed: u64, git_sha: &str) -> Self {
        Self {
            git_sha: git_sha.to_string(),
            seed,
            n_sample: op.n_sample,
            n_basis: op.n_basis,
            d: op.d,
            total_weight: op.total_weight,
            asymmetry: op.asymmetry,
        }
    }
}

// ---------------------------------------------------------------------------
// Inline helper: scalar matmul for small matrices (used by tests)
// ---------------------------------------------------------------------------

#[doc(hidden)]
pub fn small_matmul(a: &[f64], a_rows: usize, a_cols: usize, b: &[f64], b_cols: usize, out: &mut [f64]) {
    gemm(1.0, a, a_rows, a_cols, b, a_cols, b_cols, 0.0, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_metric_christoffel_zero() {
        let m = FlatMetric { d: 3 };
        let x = vec![0.5, -0.3, 0.7];
        let mut g = vec![0.0; 9];
        let mut dg = vec![0.0; 27];
        m.evaluate(&x, &mut g, &mut dg);
        let mut g_inv = vec![0.0; 9];
        let mut gamma = vec![0.0; 27];
        let mut lu = vec![0.0; 9];
        let mut perm = vec![0usize; 3];
        let mut col = vec![0.0; 3];
        christoffel_symbols(
            &g, &dg, 3, &mut g_inv, &mut gamma, &mut lu, &mut perm, &mut col,
        )
        .unwrap();
        for v in &gamma {
            assert!(v.abs() < 1e-12, "flat Christoffel must be zero, got {v}");
        }
    }

    #[test]
    fn polynomial_basis_degree_zero_one() {
        // d = 2, max_degree = 1: monomials {1, x_0, x_1}, paired with each
        // of 2 component slots → 6 basis fields.
        let basis = PolynomialVectorBasis::coordinate_polynomial_basis(2, 1);
        assert_eq!(basis.n_basis(), 6);
        let n = basis.n_basis();
        let d = 2;
        let mut v = vec![0.0; n * d];
        let mut dv = vec![0.0; n * d * d];
        let mut ddv = vec![0.0; n * d * d * d];
        let x = [3.0, 4.0];
        basis.evaluate(&x, &mut v, &mut dv, &mut ddv);
        // Monomial enumeration order is [0,0], [0,1], [1,0] = {1, x_1, x_0}.
        // First three entries (component_index = 0): V^0 ∈ {1, x_1, x_0}.
        // At x = (3, 4): V_0 = (1, 0); V_1 = (x_1, 0) = (4, 0); V_2 = (x_0, 0) = (3, 0)
        assert!((v[0 * d + 0] - 1.0).abs() < 1e-12);
        assert!((v[1 * d + 0] - 4.0).abs() < 1e-12);
        assert!((v[2 * d + 0] - 3.0).abs() < 1e-12);
        // Fourth entry: (component_index = 1, monomial 1) ⇒ V^1 = 1
        assert!((v[3 * d + 1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn flat_torus_lichnerowicz_kernel_constants() {
        // On flat R^d, the constant vector fields ∂/∂x^c are all
        // Killing (translations). With a polynomial basis of
        // max_degree = 0 we have exactly d basis fields, all in the
        // Lichnerowicz kernel.
        let d = 3;
        let metric = FlatMetric { d };
        let basis = PolynomialVectorBasis::coordinate_polynomial_basis(d, 0);
        // Three polysphere-irrelevant sample points (any chart points
        // work for a flat metric).
        let pts = [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0, 0.0, 0.0,
        ];
        let weights = vec![1.0; pts.len() / d];
        let op = assemble_lichnerowicz_matrix(&metric, &basis, &pts, &weights).unwrap();
        // L should be exactly 0 (within FP noise) on this basis.
        let max_abs: f64 = op
            .l_matrix
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        assert!(
            max_abs < 1e-10,
            "flat-torus L max-abs = {max_abs}; expected ~0"
        );
        // Gram matrix should be d × identity (each constant V_c has
        // ‖V_c‖² = 1 at every point with sample-weight sum = n_sample).
        for c in 0..d {
            assert!((op.gram_matrix[c * d + c] - (pts.len() / d) as f64).abs() < 1e-9);
        }
    }
}
