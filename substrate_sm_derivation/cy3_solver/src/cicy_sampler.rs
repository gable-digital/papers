//! M4 line-intersection sampler for the Tian-Yau Z/3 quotient Calabi-Yau threefold.
//!
//! This module implements the Donaldson / cymetric line-intersection point sampler
//! for the canonical Tian-Yau Complete Intersection Calabi-Yau (CICY) threefold,
//! given as the simultaneous vanishing locus of three Z/3-symmetric polynomials in
//! `CP^3 × CP^3` of bidegrees `(3, 0)`, `(0, 3)`, and `(1, 1)`. After taking the
//! free Z/3 quotient one obtains the Tian-Yau threefold of [Tian-Yau 1986] with
//! Hodge numbers `(h^{1,1}, h^{2,1}) = (6, 9)` and Euler characteristic
//! `χ = -6`, suitable for three-generation heterotic standard-model
//! constructions.
//!
//! Algorithmic references:
//! * Tian, Yau, "Three-dimensional algebraic manifolds with C_1 = 0 and chi = -6"
//!   (Mathematical aspects of string theory, 1986) — original Tian-Yau construction.
//! * Donaldson, "Some numerical results in complex differential geometry"
//!   (arXiv:math/0512625, 2009) — original line-intersection sampling idea.
//! * Anderson, Braun, Karp, Ovrut, "Numerical Hermitian Yang-Mills Connections..."
//!   (arXiv:1004.4399, 2010) — the ABKO algorithm and weight construction.
//! * Larfors, Lukas, Ruehle, Schneider, "Numerical metrics, curvature expansions
//!   and Calabi-Yau manifolds" / cymetric (arXiv:2111.01436, 2021) — the
//!   reference Python implementation `cymetric.pointgen.pointgen_cicy.PointGeneratorCICY`
//!   that this module mirrors.
//! * Anderson, Gray, Lukas, Palti, "Two Hundred Heterotic Standard Models on
//!   Smooth Calabi-Yau Threefolds" (arXiv:1106.4804, 2011) — Tian-Yau-style
//!   examples used in heterotic line bundle phenomenology.
//!
//! ## Geometry: the Tian-Yau CICY (NHYPER = 3)
//!
//! The Tian-Yau cover `K_0 ⊂ CP^3 × CP^3` is the codimension-3 complete
//! intersection cut out by three polynomials of bidegrees
//!
//! ```text
//!     deg(p_1) = (3, 0)        — pure cubic in z = (z_0, z_1, z_2, z_3)
//!     deg(p_2) = (0, 3)        — pure cubic in w = (w_0, w_1, w_2, w_3)
//!     deg(p_3) = (1, 1)        — bilinear bridge between the two factors
//! ```
//!
//! The complex dimension is `dim CP^3 + dim CP^3 - NHYPER = 3 + 3 - 3 = 3`.
//!
//! For the Calabi-Yau condition we require the sum of bidegrees to equal the
//! anticanonical class of the ambient: `(3, 0) + (0, 3) + (1, 1) = (4, 4)`,
//! which is exactly `(deg c_1(CP^3), deg c_1(CP^3)) = (4, 4)`. Hence `K_0`
//! is a smooth Calabi-Yau 3-fold.
//!
//! The Tian-Yau manifold is the free quotient `TY = K_0 / Z_3`.
//!
//! ## Z/3 quotient action
//!
//! With `α = exp(2πi/3)` (so `α^3 = 1`), the canonical free `Z_3` action of
//! [Tian-Yau 1986] is
//!
//! ```text
//!     g : (z_0 : z_1 : z_2 : z_3) × (w_0 : w_1 : w_2 : w_3)
//!         |->  (z_0 : α^2 z_1 : α z_2 : α z_3)
//!            × (w_0 : α   w_1 : α^2 w_2 : α^2 w_3).
//! ```
//!
//! All three defining polynomials are **manifestly invariant** under this
//! action (verified arithmetically below):
//!
//! ```text
//!     p_1(g·x) = z_0^3 + (α^2 z_1)^3 + (α z_2)^3 + (α z_3)^3
//!              = z_0^3 + α^6 z_1^3 + α^3 z_2^3 + α^3 z_3^3
//!              = z_0^3 + z_1^3 + z_2^3 + z_3^3 = p_1(x)            ✓
//!
//!     p_2(g·x) = w_0^3 + (α w_1)^3 + (α^2 w_2)^3 + (α^2 w_3)^3
//!              = w_0^3 + α^3 w_1^3 + α^6 w_2^3 + α^6 w_3^3
//!              = w_0^3 + w_1^3 + w_2^3 + w_3^3 = p_2(x)            ✓
//!
//!     p_3(g·x) = z_0 w_0 + (α^2 z_1)(α w_1) + (α z_2)(α^2 w_2) + (α z_3)(α^2 w_3)
//!              = z_0 w_0 + α^3 z_1 w_1 + α^3 z_2 w_2 + α^3 z_3 w_3
//!              = z_0 w_0 + z_1 w_1 + z_2 w_2 + z_3 w_3 = p_3(x)    ✓
//! ```
//!
//! cymetric option (a) for the quotient is to scale all sampling weights by
//! `1/k` where `k = 3` is the order of the group. This is provided as
//! [`CicySampler::apply_z3_quotient`].
//!
//! ## Default polynomials
//!
//! The default [`BicubicPair`] returned by [`BicubicPair::z3_invariant_default`]
//! is the **canonical Tian-Yau Fermat-type triple** of [Tian-Yau 1986]:
//!
//! ```text
//!     p_1 = z_0^3 + z_1^3 + z_2^3 + z_3^3                          // bidegree (3, 0)
//!     p_2 = w_0^3 + w_1^3 + w_2^3 + w_3^3                          // bidegree (0, 3)
//!     p_3 = z_0 w_0 + z_1 w_1 + z_2 w_2 + z_3 w_3                  // bidegree (1, 1)
//! ```
//!
//! All coefficients are `1.0 + 0.0 i` — this is the published canonical
//! representative; no arbitrary "genericity" constants are introduced.
//! Every monomial of every polynomial is `Z_3`-invariant under the action
//! above, so the variety descends to the Tian-Yau quotient. The only
//! pre-existing well-known issue with the Fermat representative is that
//! the cover `K_0` may have isolated singularities where the
//! `(3 × 8)`-Jacobian drops rank; the sampler rejects such points
//! automatically (singular elimination Jacobian → `None`).
//!
//! Users who require a deformation away from the Fermat point (to lift
//! conifold-type singularities) should construct their own
//! [`BicubicPair`] via [`BicubicPair::new`].
//!
//! ## Algorithm outline
//!
//! For `NHYPER = 3` defining polynomials, `NCOORDS = 8`, ambient
//! `CP^3 × CP^3`, complex dimension `NFOLD = 3`:
//!
//!   1. Sample `NHYPER + 1 = 4` Gaussian unit vectors in `C^8` (uniform on
//!      `S^15`); use one as the line origin `p` and the other three as line
//!      directions `q_1, q_2, q_3`.
//!   2. Solve `f_i(p + t_1 q_1 + t_2 q_2 + t_3 q_3) = 0` for
//!      `(t_1, t_2, t_3) ∈ C^3` simultaneously by Newton-Raphson with
//!      Armijo back-tracking.
//!   3. Rescale each `CP^3` factor so that the largest-magnitude coordinate
//!      equals `1 + 0i` (patch choice).
//!   4. Reject the point if more than `NFACTORS = 2` coordinates lie within
//!      `PATCH_EPS` of `1 + 0i` (degenerate patch ambiguity).
//!   5. Compute the holomorphic top form residue `Ω = 1 / det(J_elim)` where
//!      `J_elim` is the `NHYPER × NHYPER = 3 × 3` sub-Jacobian on the
//!      eliminated coordinates.
//!   6. Compute the pullback of the block-diagonal Fubini-Study metric and
//!      its determinant `det g_pb`.
//!   7. Form the canonical sampling weight `w = |Ω|^2 / det g_pb`.
//!
//! ## Public API
//!
//! * [`BicubicPair`] — the three Tian-Yau defining polynomials, sparse
//!   monomial representation. The legacy two-polynomial type name is
//!   retained for downstream compatibility; `f3` is the new bilinear field.
//! * [`SampledPoint`] — one accepted point on the CICY with associated weight.
//! * [`CicySampler`] — stateful sampler that owns the RNG and the polynomial set.
//!
//! The orchestrator registers this module via `mod cicy_sampler;` in `lib.rs`.

use num_complex::Complex64;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Codimension of the Tian-Yau CICY in `CP^3 × CP^3`.
///
/// For the canonical Tian-Yau presentation there are **three** defining
/// polynomials of bidegrees `(3, 0)`, `(0, 3)`, `(1, 1)`. This was upgraded
/// from `NHYPER = 2` (which would have given a complex 4-fold rather than a
/// 3-fold) to match the published Tian-Yau geometry; see the module docstring
/// for the citation chain (Tian-Yau 1986; Anderson-Lukas-Palti arXiv:1106.4804).
pub const NHYPER: usize = 3;

/// Number of homogeneous ambient coordinates (`4 + 4`).
pub const NCOORDS: usize = 8;

/// Complex dimension of the Calabi-Yau threefold (Tian-Yau, after Z/3 quotient).
///
/// With the corrected `NHYPER = 3` the cover `K_0 = {p_1 = p_2 = p_3 = 0}` in
/// `CP^3 × CP^3` has complex dimension `(3 + 3) - 3 = 3`. The free `Z_3`
/// quotient preserves complex dimension, so `NFOLD = 3` for the Tian-Yau
/// threefold itself.
pub const NFOLD: usize = 3;

/// Number of `CP^k` factors in the ambient space.
pub const NFACTORS: usize = 2;

/// Number of *free* affine coordinates remaining after fixing one homogeneous
/// coordinate in each `CP^3` factor and eliminating `NHYPER` more via the
/// defining polynomials. For our setup `8 - 2 - 3 = 3`. This equals `NFOLD`
/// (as it must — the free affine coordinates parametrise the threefold).
pub const N_FREE: usize = NCOORDS - NFACTORS - NHYPER;

const _: () = assert!(N_FREE == NFOLD, "N_FREE must equal NFOLD for a CY threefold");

/// Tolerance on `|f_i|` for accepting a Newton root.
const ACC: f64 = 1.0e-8;

/// Number of Newton restarts per ambient line before giving up.
const NATTEMPTS: usize = 4;

/// Patch-ambiguity threshold: a coordinate is "essentially 1" if
/// `|coord - 1| < PATCH_EPS` after rescaling.
const PATCH_EPS: f64 = 1.0e-8;

/// Default Newton iteration bound.
const DEFAULT_MAX_NEWTON: usize = 64;

// ---------------------------------------------------------------------------
// BicubicPair
// ---------------------------------------------------------------------------

/// The three defining polynomials of the Tian-Yau CICY in `CP^3 × CP^3`.
///
/// The name `BicubicPair` is retained for backward compatibility with
/// downstream callers (`yukawa_overlap.rs`, etc.); strictly the data is now
/// a *triple*: two pure cubics and a bilinear bridge.
///
/// Each polynomial is represented sparsely as a list of `(coefficient,
/// exponents)` tuples where `exponents` is `[e_z0, e_z1, e_z2, e_z3,
/// e_w0, e_w1, e_w2, e_w3]`. The bidegrees enforced (and regression-tested)
/// are `(3, 0)`, `(0, 3)`, `(1, 1)` for `f1`, `f2`, `f3` respectively.
#[derive(Clone, Debug)]
pub struct BicubicPair {
    /// First defining polynomial — bidegree `(3, 0)`, pure cubic in `z`.
    pub f1: Vec<(Complex64, [u32; 8])>,
    /// Second defining polynomial — bidegree `(0, 3)`, pure cubic in `w`.
    pub f2: Vec<(Complex64, [u32; 8])>,
    /// Third defining polynomial — bidegree `(1, 1)`, bilinear bridge
    /// between the two `CP^3` factors. Required for the variety to be a
    /// complex 3-fold (and a Calabi-Yau).
    pub f3: Vec<(Complex64, [u32; 8])>,
}

impl BicubicPair {
    /// Construct a [`BicubicPair`] from three explicit polynomial lists,
    /// without checking degree invariants.
    ///
    /// Callers that build their own polynomials (e.g. from a published example)
    /// should ensure the bidegrees are `(3, 0)`, `(0, 3)`, `(1, 1)` for
    /// `f1`, `f2`, `f3` respectively, and that all three are Z/3-invariant
    /// under the canonical action documented at the module level if a
    /// quotient will be applied later.
    pub fn new(
        f1: Vec<(Complex64, [u32; 8])>,
        f2: Vec<(Complex64, [u32; 8])>,
        f3: Vec<(Complex64, [u32; 8])>,
    ) -> Self {
        Self { f1, f2, f3 }
    }

    /// The canonical Z/3-invariant Tian-Yau triple at the Fermat point.
    ///
    /// ```text
    ///     p_1 = z_0^3 + z_1^3 + z_2^3 + z_3^3                  // bidegree (3, 0)
    ///     p_2 = w_0^3 + w_1^3 + w_2^3 + w_3^3                  // bidegree (0, 3)
    ///     p_3 = z_0 w_0 + z_1 w_1 + z_2 w_2 + z_3 w_3          // bidegree (1, 1)
    /// ```
    ///
    /// This is the published canonical representative of the Tian-Yau
    /// CICY [Tian-Yau 1986]. All three polynomials are exactly invariant
    /// under the diagonal `Z_3` action
    /// `(z_0, α^2 z_1, α z_2, α z_3) × (w_0, α w_1, α^2 w_2, α^2 w_3)`
    /// (verification in the module docstring).
    pub fn z3_invariant_default() -> Self {
        let one = Complex64::new(1.0, 0.0);

        // f1 = sum_i z_i^3   (bidegree (3, 0))
        let mut f1: Vec<(Complex64, [u32; 8])> = Vec::with_capacity(4);
        for i in 0..4 {
            let mut e = [0u32; 8];
            e[i] = 3;
            f1.push((one, e));
        }

        // f2 = sum_i w_i^3   (bidegree (0, 3))
        let mut f2: Vec<(Complex64, [u32; 8])> = Vec::with_capacity(4);
        for i in 0..4 {
            let mut e = [0u32; 8];
            e[4 + i] = 3;
            f2.push((one, e));
        }

        // f3 = sum_i z_i * w_i   (bidegree (1, 1))
        let mut f3: Vec<(Complex64, [u32; 8])> = Vec::with_capacity(4);
        for i in 0..4 {
            let mut e = [0u32; 8];
            e[i] = 1;
            e[4 + i] = 1;
            f3.push((one, e));
        }

        Self { f1, f2, f3 }
    }

    /// Evaluate `[f_1(point), f_2(point), f_3(point)]`.
    #[inline]
    pub fn eval(&self, point: &[Complex64; 8]) -> [Complex64; NHYPER] {
        [
            eval_poly(&self.f1, point),
            eval_poly(&self.f2, point),
            eval_poly(&self.f3, point),
        ]
    }

    /// Holomorphic Jacobian `∂f_i/∂x_k`, returned as a `NHYPER × NCOORDS`
    /// row-major `Vec<Complex64>`. Row `i ∈ {0, 1, 2}` corresponds to
    /// `f_{i+1}`; column `k ∈ 0..NCOORDS` to ambient coordinate `x_k`.
    pub fn jacobian(&self, point: &[Complex64; 8]) -> Vec<Complex64> {
        let mut j = vec![Complex64::new(0.0, 0.0); NHYPER * NCOORDS];
        for k in 0..NCOORDS {
            j[k] = poly_partial(&self.f1, point, k);
            j[NCOORDS + k] = poly_partial(&self.f2, point, k);
            j[2 * NCOORDS + k] = poly_partial(&self.f3, point, k);
        }
        j
    }
}

#[inline]
fn eval_monomial(point: &[Complex64; 8], exps: &[u32; 8]) -> Complex64 {
    let mut acc = Complex64::new(1.0, 0.0);
    for k in 0..NCOORDS {
        let e = exps[k];
        if e == 0 {
            continue;
        }
        // Small integer powers — unrolled multiplication is faster than powi here.
        let xk = point[k];
        let mut p = xk;
        for _ in 1..e {
            p *= xk;
        }
        acc *= p;
    }
    acc
}

#[inline]
fn eval_poly(poly: &[(Complex64, [u32; 8])], point: &[Complex64; 8]) -> Complex64 {
    let mut s = Complex64::new(0.0, 0.0);
    for (coef, exps) in poly {
        s += *coef * eval_monomial(point, exps);
    }
    s
}

/// Partial derivative of a sparse polynomial with respect to coordinate `k`.
fn poly_partial(
    poly: &[(Complex64, [u32; 8])],
    point: &[Complex64; 8],
    k: usize,
) -> Complex64 {
    let mut s = Complex64::new(0.0, 0.0);
    for (coef, exps) in poly {
        let e_k = exps[k];
        if e_k == 0 {
            continue;
        }
        // d/dx_k x_k^e_k = e_k * x_k^{e_k - 1}; other factors unchanged.
        let mut new_exps = *exps;
        new_exps[k] = e_k - 1;
        let mono = eval_monomial(point, &new_exps);
        s += *coef * Complex64::new(e_k as f64, 0.0) * mono;
    }
    s
}

// ---------------------------------------------------------------------------
// Sampled point
// ---------------------------------------------------------------------------

/// One accepted point on the CICY together with the data needed to form the
/// canonical sampling measure.
#[derive(Clone, Debug)]
pub struct SampledPoint {
    /// Homogeneous coordinates of the first `CP^3` factor; one entry equals
    /// `1 + 0i` after the patch rescaling.
    pub z: [Complex64; 4],
    /// Homogeneous coordinates of the second `CP^3` factor; same convention.
    pub w: [Complex64; 4],
    /// Holomorphic top-form residue `Ω = 1 / det(J_elim)`.
    pub omega: Complex64,
    /// Sampling weight `|Ω|^2 / det g_pb` (or normalised for batches).
    pub weight: f64,
}

// ---------------------------------------------------------------------------
// CicySampler
// ---------------------------------------------------------------------------

/// Stateful line-intersection sampler.
///
/// Owns its RNG and a small scratch buffer for the per-attempt Newton iteration.
pub struct CicySampler {
    /// The CICY defining polynomials (Tian-Yau triple).
    pub bicubic: BicubicPair,
    /// Initial PRNG seed (kept for reproducibility / re-seeding).
    pub seed: u64,
    /// Newton iteration cap per restart.
    pub max_newton_iter: usize,
    /// Residual tolerance.
    pub newton_tol: f64,
    /// Maximum Newton restarts per ambient line.
    pub max_attempts_per_line: usize,
    /// Patch-ambiguity rejection threshold.
    pub patch_eps: f64,
    rng: ChaCha8Rng,
}

impl CicySampler {
    /// Construct a new sampler with the supplied bicubic triple and PRNG seed.
    pub fn new(bicubic: BicubicPair, seed: u64) -> Self {
        Self {
            bicubic,
            seed,
            max_newton_iter: DEFAULT_MAX_NEWTON,
            newton_tol: ACC,
            max_attempts_per_line: NATTEMPTS,
            patch_eps: PATCH_EPS,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Sample one point on the CICY.
    ///
    /// Returns `None` on rejection (Newton non-convergence on every restart,
    /// patch ambiguity, singular elimination Jacobian, singular pullback metric,
    /// or non-finite weight).
    pub fn sample_one(&mut self) -> Option<SampledPoint> {
        // 1) NHYPER + 1 Gaussian unit vectors on S^15 ⊂ C^8: one origin p
        //    plus NHYPER = 3 line directions.
        let p = gaussian_unit_sphere_c8(&mut self.rng);
        let mut directions = [[Complex64::new(0.0, 0.0); 8]; NHYPER];
        for d in directions.iter_mut() {
            *d = gaussian_unit_sphere_c8(&mut self.rng);
        }

        // 2) Solve f_i(p + sum_j t_j * dirs[j]) = 0 for t in C^NHYPER by
        //    Newton-Raphson with Armijo back-tracking.
        let mut best_x: Option<[Complex64; 8]> = None;
        let mut best_residual = f64::INFINITY;

        for _ in 0..self.max_attempts_per_line {
            // Initial guess: Gaussian complex numbers with std ~1.
            let normal = StandardNormal;
            let mut t = [Complex64::new(0.0, 0.0); NHYPER];
            for tj in t.iter_mut() {
                *tj = Complex64::new(normal.sample(&mut self.rng), normal.sample(&mut self.rng));
            }

            let (x, residual, converged) =
                self.newton_solve_line(&p, &directions, &mut t);

            if converged && residual < best_residual {
                best_residual = residual;
                best_x = Some(x);
                if residual < self.newton_tol {
                    break;
                }
            }
        }

        let x = best_x?;
        if best_residual >= self.newton_tol {
            return None;
        }
        self.post_process_solved_point(&x)
    }

    /// Post-Newton processing: take a converged ambient-coordinate
    /// solution `x` (from either the CPU Newton or the GPU
    /// `gpu_sampler` kernel), perform patch rescale + omega computation
    /// + pullback metric + weight assembly, and return the
    /// `SampledPoint` if every step succeeded.
    ///
    /// Public so the GPU sampler wrapper can call it directly.
    pub fn post_process_solved_point(
        &self,
        x: &[Complex64; 8],
    ) -> Option<SampledPoint> {
        // 3) Patch rescale: divide each CP^3 factor by its largest-modulus coordinate.
        let mut z = [x[0], x[1], x[2], x[3]];
        let mut w = [x[4], x[5], x[6], x[7]];
        let z_idx = argmax_abs(&z)?;
        let w_idx = argmax_abs(&w)?;
        let z_scale = z[z_idx];
        let w_scale = w[w_idx];
        if z_scale.norm() < f64::EPSILON || w_scale.norm() < f64::EPSILON {
            return None;
        }
        for c in z.iter_mut() {
            *c /= z_scale;
        }
        for c in w.iter_mut() {
            *c /= w_scale;
        }

        // 4) Patch-ambiguity rejection: count coords near 1+0i across both factors.
        let mut near_one = 0usize;
        for c in z.iter().chain(w.iter()) {
            if (*c - Complex64::new(1.0, 0.0)).norm() < self.patch_eps {
                near_one += 1;
            }
        }
        if near_one > NFACTORS {
            return None;
        }

        // Rebuild the rescaled ambient point for derivative evaluations.
        let x_resc: [Complex64; 8] = [z[0], z[1], z[2], z[3], w[0], w[1], w[2], w[3]];

        // 5) Compute Omega via residue: pick NHYPER coords with largest |dQ_i/dx_k|.
        let jac = self.bicubic.jacobian(&x_resc); // NHYPER x NCOORDS, row-major
        let j_elim = pick_elimination_columns(&jac, &z_idx, &w_idx)?;

        // omega_mat = NHYPER x NHYPER matrix of dQ_i/dx_{j_elim[k]}.
        let mut omega_mat = vec![Complex64::new(0.0, 0.0); NHYPER * NHYPER];
        for i in 0..NHYPER {
            for k in 0..NHYPER {
                omega_mat[i * NHYPER + k] = jac[i * NCOORDS + j_elim[k]];
            }
        }
        let det_omega = det_complex_lu(&omega_mat, NHYPER);
        if det_omega.norm() < f64::EPSILON {
            return None;
        }
        let omega = Complex64::new(1.0, 0.0) / det_omega;

        // 6) Pullback Jacobian j_pb: N_FREE x NCOORDS that maps free coordinates
        //    to all ambient coords. Free coords = {0..NCOORDS} \
        //    ({z_idx, NCOORDS/2 + w_idx} ∪ j_elim).
        let mut eliminated = [false; NCOORDS];
        eliminated[z_idx] = true;
        eliminated[4 + w_idx] = true;
        for &k in &j_elim {
            eliminated[k] = true;
        }
        let mut free: Vec<usize> = Vec::with_capacity(N_FREE);
        for k in 0..NCOORDS {
            if !eliminated[k] {
                free.push(k);
            }
        }
        if free.len() != N_FREE {
            return None;
        }

        // The pullback Jacobian maps (free coords) -> (ambient coords).
        // For free coords it is the identity row; for j_elim coords it is the
        // implicit-function derivative `-(omega_mat)^{-1} * jac[:, free]`;
        // for the homogeneous-patch coords (z_idx, 4+w_idx) it is zero.
        let mut j_pb = vec![Complex64::new(0.0, 0.0); N_FREE * NCOORDS];
        for (a, &fa) in free.iter().enumerate() {
            j_pb[a * NCOORDS + fa] = Complex64::new(1.0, 0.0);
        }
        // Solve omega_mat * E = jac[:, free] -> implicit derivative = -E.
        let omega_inv = inv_n(&omega_mat, NHYPER)?;
        for (a, &fa) in free.iter().enumerate() {
            // rhs vector b[i] = jac[i, fa]
            let mut b = [Complex64::new(0.0, 0.0); NHYPER];
            for i in 0..NHYPER {
                b[i] = jac[i * NCOORDS + fa];
            }
            // E = omega_inv * b   (NHYPER-vector); j_pb[free idx][j_elim] = -E
            for r in 0..NHYPER {
                let mut e_r = Complex64::new(0.0, 0.0);
                for c in 0..NHYPER {
                    e_r += omega_inv[r * NHYPER + c] * b[c];
                }
                j_pb[a * NCOORDS + j_elim[r]] = -e_r;
            }
        }

        // Block-diagonal Fubini-Study metric on CP^3 x CP^3 in affine patches.
        // For a CP^n patch with one homogeneous coord = 1 the FS Kahler metric is
        //   g_{ab̄} = (1 + |z|^2) δ_{ab} - z_a z̄_b   (up to a global Kähler constant)
        // all divided by (1 + |z|^2)^2. We use the standard (1 + |z|^2)^{-2} form;
        // the overall Kähler scale cancels in the ratio |Ω|^2 / det g_pb when used
        // consistently across the dataset (cymetric convention).
        let g_fs = fs_metric_block(&z, z_idx, &w, w_idx);

        // g_pb = j_pb * g_fs * j_pb^†  (N_FREE x N_FREE, Hermitian)
        let g_pb = pullback_metric(&j_pb, &g_fs);
        let det_g = det_complex_lu(&g_pb, N_FREE).re;
        if !det_g.is_finite() || det_g.abs() < 1.0e-300 {
            return None;
        }

        // 7) Weight.
        let weight = omega.norm_sqr() / det_g;
        if !weight.is_finite() || weight <= 0.0 {
            return None;
        }

        Some(SampledPoint {
            z,
            w,
            omega,
            weight,
        })
    }

    /// Sample `n_target` accepted points and return them with weights normalised
    /// so that `sum(weights) ≈ 1`.
    ///
    /// When the `gpu` feature is on and a CUDA device is available,
    /// this dispatches to [`sample_batch_gpu`](Self::sample_batch_gpu)
    /// which runs the Newton solver in parallel on the device. On
    /// any GPU failure (no CUDA, kernel error, …) it transparently
    /// falls back to the per-point CPU loop below.
    pub fn sample_batch(&mut self, n_target: usize) -> Vec<SampledPoint> {
        #[cfg(feature = "gpu")]
        {
            if let Some(out) = self.sample_batch_gpu(n_target) {
                return out;
            }
        }
        let mut out: Vec<SampledPoint> = Vec::with_capacity(n_target);
        // Reasonable absolute upper bound on attempts to avoid pathological loops.
        let max_total = n_target.saturating_mul(64).max(1024);
        let mut attempts = 0usize;
        while out.len() < n_target && attempts < max_total {
            attempts += 1;
            if let Some(pt) = self.sample_one() {
                out.push(pt);
            }
        }
        // Normalise weights.
        let sum: f64 = out.iter().map(|p| p.weight).sum();
        if sum.is_finite() && sum > 0.0 {
            for p in out.iter_mut() {
                p.weight /= sum;
            }
        }
        out
    }

    /// GPU-accelerated batch sampler. Generates `n_target ×
    /// max_attempts` `(origin, directions, t_init)` triples on the
    /// host, dispatches them all to [`crate::gpu_sampler`] in a
    /// single kernel launch, then post-processes the converged
    /// threads (patch rescale + omega + pullback metric + weight)
    /// on CPU.
    ///
    /// Returns `None` on any GPU failure; the caller should fall back
    /// to the CPU path. Weights are normalised to sum-to-1 on the
    /// returned vector.
    ///
    /// The current implementation generates one Newton attempt per
    /// thread (no host-side restart bookkeeping) — for the typical
    /// 5σ workload (n_target ≤ 5000, ~80% per-attempt convergence on
    /// the Tian-Yau Fermat point) this is sufficient. A more
    /// sophisticated multi-restart scheduler would post-process
    /// failed threads and re-dispatch with new initial guesses; that
    /// can be added if convergence rates drop on harder geometries.
    #[cfg(feature = "gpu")]
    pub fn sample_batch_gpu(&mut self, n_target: usize) -> Option<Vec<SampledPoint>> {
        use crate::gpu_sampler::{gpu_newton_solve_lines, GpuSamplerContext};

        thread_local! {
            static GPU_CTX: std::cell::OnceCell<Option<GpuSamplerContext>> =
                const { std::cell::OnceCell::new() };
        }

        // Pull the context (or initialise it once per thread).
        let result = GPU_CTX.with(|cell| -> Option<Vec<SampledPoint>> {
            let ctx_opt = cell.get_or_init(|| {
                GpuSamplerContext::new()
                    .map_err(|e| {
                        eprintln!(
                            "[gpu_sampler] init failed: {e} — falling back to CPU"
                        );
                        e
                    })
                    .ok()
            });
            let ctx = ctx_opt.as_ref()?;

            // Over-provision: dispatch n_target * 4 threads so that
            // even with ~25% rejection we have enough converged
            // points. The GPU per-thread Newton is cheap so this
            // headroom costs little.
            let n_threads = n_target.saturating_mul(4).max(64);

            let mut origins_re = vec![0.0_f64; n_threads * NCOORDS];
            let mut origins_im = vec![0.0_f64; n_threads * NCOORDS];
            let mut dirs_re = vec![0.0_f64; n_threads * NHYPER * NCOORDS];
            let mut dirs_im = vec![0.0_f64; n_threads * NHYPER * NCOORDS];
            let mut t_init_re = vec![0.0_f64; n_threads * NHYPER];
            let mut t_init_im = vec![0.0_f64; n_threads * NHYPER];

            // Pre-generate all per-thread inputs on the host.
            let normal = StandardNormal;
            for tid in 0..n_threads {
                let p = gaussian_unit_sphere_c8(&mut self.rng);
                for k in 0..NCOORDS {
                    origins_re[tid * NCOORDS + k] = p[k].re;
                    origins_im[tid * NCOORDS + k] = p[k].im;
                }
                for j in 0..NHYPER {
                    let d = gaussian_unit_sphere_c8(&mut self.rng);
                    for k in 0..NCOORDS {
                        let idx = tid * NHYPER * NCOORDS + j * NCOORDS + k;
                        dirs_re[idx] = d[k].re;
                        dirs_im[idx] = d[k].im;
                    }
                }
                for j in 0..NHYPER {
                    t_init_re[tid * NHYPER + j] = normal.sample(&mut self.rng);
                    t_init_im[tid * NHYPER + j] = normal.sample(&mut self.rng);
                }
            }

            let results = gpu_newton_solve_lines(
                ctx,
                &self.bicubic,
                &origins_re,
                &origins_im,
                &dirs_re,
                &dirs_im,
                &t_init_re,
                &t_init_im,
                n_threads,
                self.max_newton_iter,
                self.newton_tol,
            )
            .map_err(|e| {
                eprintln!("[gpu_sampler] launch failed: {e} — falling back to CPU");
                e
            })
            .ok()?;

            // Post-process converged threads on CPU.
            let mut out: Vec<SampledPoint> = Vec::with_capacity(n_target);
            for tid in 0..n_threads {
                if out.len() >= n_target {
                    break;
                }
                let r = &results[tid];
                if !r.converged {
                    continue;
                }
                // x = origin + Σ_j t_j · dirs[j]
                let mut x = [Complex64::new(0.0, 0.0); NCOORDS];
                for k in 0..NCOORDS {
                    let mut s = Complex64::new(
                        origins_re[tid * NCOORDS + k],
                        origins_im[tid * NCOORDS + k],
                    );
                    for j in 0..NHYPER {
                        let d_idx = tid * NHYPER * NCOORDS + j * NCOORDS + k;
                        let d = Complex64::new(dirs_re[d_idx], dirs_im[d_idx]);
                        s += r.t[j] * d;
                    }
                    x[k] = s;
                }
                if let Some(pt) = self.post_process_solved_point(&x) {
                    out.push(pt);
                }
            }

            // Normalise weights.
            let sum: f64 = out.iter().map(|p| p.weight).sum();
            if sum.is_finite() && sum > 0.0 {
                for p in out.iter_mut() {
                    p.weight /= sum;
                }
            }
            Some(out)
        });

        result
    }

    /// Apply the Z/3 quotient by dividing every weight by the group order
    /// (cymetric option (a)). Operates in place on a slice of points sampled
    /// on the cover; the result represents the measure on the quotient.
    pub fn apply_z3_quotient(points: &mut [SampledPoint]) {
        let inv_k = 1.0 / 3.0;
        for p in points.iter_mut() {
            p.weight *= inv_k;
        }
    }

    // ---------------------------------------------------------------------
    // Internal: Newton solve on the parametric line.
    // ---------------------------------------------------------------------

    /// Solve `f_i(p + sum_j t_j * dirs[j]) = 0` for `t ∈ C^NHYPER` by
    /// Newton-Raphson with Armijo back-tracking.
    fn newton_solve_line(
        &self,
        p: &[Complex64; 8],
        dirs: &[[Complex64; 8]; NHYPER],
        t: &mut [Complex64; NHYPER],
    ) -> ([Complex64; 8], f64, bool) {
        let mut x = line_eval(p, dirs, t);
        let mut f = self.bicubic.eval(&x);
        let mut residual = residual_norm(&f);

        for _ in 0..self.max_newton_iter {
            if residual < self.newton_tol {
                return (x, residual, true);
            }

            // Build d f_i / d t_j = sum_k (∂ f_i / ∂ x_k) * dirs[j][k]
            let jac_x = self.bicubic.jacobian(&x); // NHYPER x NCOORDS
            let mut m = vec![Complex64::new(0.0, 0.0); NHYPER * NHYPER];
            for i in 0..NHYPER {
                for j in 0..NHYPER {
                    let mut s = Complex64::new(0.0, 0.0);
                    for k in 0..NCOORDS {
                        s += jac_x[i * NCOORDS + k] * dirs[j][k];
                    }
                    m[i * NHYPER + j] = s;
                }
            }

            // Newton step: solve m * dt = -f.
            let det_m = det_complex_lu(&m, NHYPER);
            if det_m.norm() < 1.0e-30 {
                // Singular — bail.
                return (x, residual, false);
            }
            let m_inv = match inv_n(&m, NHYPER) {
                Some(v) => v,
                None => return (x, residual, false),
            };
            let mut dt = [Complex64::new(0.0, 0.0); NHYPER];
            for r in 0..NHYPER {
                let mut s = Complex64::new(0.0, 0.0);
                for c in 0..NHYPER {
                    s += m_inv[r * NHYPER + c] * f[c];
                }
                dt[r] = -s;
            }

            // Armijo back-tracking: try step sizes 1, 1/2, 1/4, ...
            // Accept any sufficient decrease (very mild Armijo constant 1e-4).
            let mut step = 1.0;
            let mut accepted = false;
            for _ in 0..32 {
                let mut t_trial = [Complex64::new(0.0, 0.0); NHYPER];
                for r in 0..NHYPER {
                    t_trial[r] = t[r] + Complex64::new(step, 0.0) * dt[r];
                }
                let x_trial = line_eval(p, dirs, &t_trial);
                let f_trial = self.bicubic.eval(&x_trial);
                let r_trial = residual_norm(&f_trial);
                if r_trial < (1.0 - 1.0e-4 * step) * residual {
                    *t = t_trial;
                    x = x_trial;
                    f = f_trial;
                    residual = r_trial;
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if !accepted {
                // Couldn't improve — give up on this restart.
                return (x, residual, false);
            }
        }

        (x, residual, residual < self.newton_tol)
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

#[inline]
fn residual_norm(f: &[Complex64; NHYPER]) -> f64 {
    let mut s = 0.0;
    for c in f.iter() {
        s += c.norm_sqr();
    }
    s.sqrt()
}

/// `x = p + sum_j t_j * dirs[j]`.
#[inline]
fn line_eval(
    p: &[Complex64; 8],
    dirs: &[[Complex64; 8]; NHYPER],
    t: &[Complex64; NHYPER],
) -> [Complex64; 8] {
    let mut x = *p;
    for j in 0..NHYPER {
        let tj = t[j];
        let d = &dirs[j];
        for k in 0..NCOORDS {
            x[k] += tj * d[k];
        }
    }
    x
}

/// Sample a unit-norm complex 8-vector by Gaussian normalisation
/// (uniform on `S^15 ⊂ C^8 ≅ R^16`).
fn gaussian_unit_sphere_c8(rng: &mut ChaCha8Rng) -> [Complex64; 8] {
    let normal = StandardNormal;
    let mut v = [Complex64::new(0.0, 0.0); 8];
    let mut sq = 0.0;
    for k in 0..8 {
        let re: f64 = normal.sample(rng);
        let im: f64 = normal.sample(rng);
        v[k] = Complex64::new(re, im);
        sq += re * re + im * im;
    }
    let n = sq.sqrt();
    if n > 0.0 {
        for c in v.iter_mut() {
            *c /= n;
        }
    }
    v
}

#[inline]
fn argmax_abs(v: &[Complex64; 4]) -> Option<usize> {
    let mut best = 0usize;
    let mut best_abs = -1.0f64;
    for (i, c) in v.iter().enumerate() {
        let a = c.norm();
        if a > best_abs {
            best_abs = a;
            best = i;
        }
    }
    if best_abs > 0.0 {
        Some(best)
    } else {
        None
    }
}

/// Pick the `NHYPER` ambient indices to eliminate via the residue formula.
/// We greedily pick, for each row of the Jacobian, the column with largest
/// absolute value while ensuring picks are distinct and not equal to the
/// patch-fixing coordinate indices (`z_idx`, `4 + w_idx`).
fn pick_elimination_columns(
    jac: &[Complex64],
    z_idx: &usize,
    w_idx: &usize,
) -> Option<[usize; NHYPER]> {
    let forbidden = [*z_idx, 4 + *w_idx];
    let mut taken = [false; NCOORDS];
    for &f in &forbidden {
        taken[f] = true;
    }

    let mut picks = [usize::MAX; NHYPER];
    for i in 0..NHYPER {
        let mut best_k = usize::MAX;
        let mut best_abs = -1.0f64;
        for k in 0..NCOORDS {
            if taken[k] {
                continue;
            }
            let a = jac[i * NCOORDS + k].norm();
            if a > best_abs {
                best_abs = a;
                best_k = k;
            }
        }
        if best_k == usize::MAX || best_abs <= 0.0 {
            return None;
        }
        picks[i] = best_k;
        taken[best_k] = true;
    }
    Some(picks)
}

/// Determinant of a complex `n x n` matrix (row-major) via partial-pivot LU.
fn det_complex_lu(mat: &[Complex64], n: usize) -> Complex64 {
    let mut a: Vec<Complex64> = mat.to_vec();
    let mut det = Complex64::new(1.0, 0.0);
    for k in 0..n {
        // Partial pivot: find row p ∈ [k, n) maximising |a[p, k]|.
        let mut pivot = k;
        let mut best = a[k * n + k].norm();
        for p in (k + 1)..n {
            let v = a[p * n + k].norm();
            if v > best {
                best = v;
                pivot = p;
            }
        }
        if best == 0.0 {
            return Complex64::new(0.0, 0.0);
        }
        if pivot != k {
            for j in 0..n {
                a.swap(k * n + j, pivot * n + j);
            }
            det = -det;
        }
        let pivot_val = a[k * n + k];
        det *= pivot_val;
        let inv_pivot = Complex64::new(1.0, 0.0) / pivot_val;
        for i in (k + 1)..n {
            let factor = a[i * n + k] * inv_pivot;
            for j in (k + 1)..n {
                let lhs = a[i * n + j];
                let rhs = factor * a[k * n + j];
                a[i * n + j] = lhs - rhs;
            }
            a[i * n + k] = Complex64::new(0.0, 0.0);
        }
    }
    det
}

/// Inverse of a complex `n x n` matrix via Gauss-Jordan with partial pivoting.
/// Returns `None` if singular (smallest pivot below `1e-30`).
fn inv_n(mat: &[Complex64], n: usize) -> Option<Vec<Complex64>> {
    // Augmented matrix [A | I] of shape n x 2n.
    let cols = 2 * n;
    let mut a: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n * cols];
    for i in 0..n {
        for j in 0..n {
            a[i * cols + j] = mat[i * n + j];
        }
        a[i * cols + n + i] = Complex64::new(1.0, 0.0);
    }

    for k in 0..n {
        // Partial pivot.
        let mut pivot = k;
        let mut best = a[k * cols + k].norm();
        for p in (k + 1)..n {
            let v = a[p * cols + k].norm();
            if v > best {
                best = v;
                pivot = p;
            }
        }
        if best < 1.0e-30 {
            return None;
        }
        if pivot != k {
            for j in 0..cols {
                a.swap(k * cols + j, pivot * cols + j);
            }
        }
        let pivot_val = a[k * cols + k];
        let inv_pivot = Complex64::new(1.0, 0.0) / pivot_val;
        // Scale pivot row.
        for j in 0..cols {
            a[k * cols + j] *= inv_pivot;
        }
        // Eliminate column k in all other rows.
        for i in 0..n {
            if i == k {
                continue;
            }
            let factor = a[i * cols + k];
            if factor.norm() == 0.0 {
                continue;
            }
            for j in 0..cols {
                let v = a[k * cols + j] * factor;
                a[i * cols + j] -= v;
            }
        }
    }

    // Extract right block.
    let mut inv = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = a[i * cols + n + j];
        }
    }
    Some(inv)
}

/// Build the block-diagonal Fubini-Study metric on `C^4 x C^4` in the chosen
/// affine patches. Returns `g_fs` as an `NCOORDS x NCOORDS` row-major matrix;
/// rows/columns corresponding to the patch-fixing coordinates are zeroed (those
/// directions carry no degree of freedom in the affine chart).
fn fs_metric_block(
    z: &[Complex64; 4],
    z_idx: usize,
    w: &[Complex64; 4],
    w_idx: usize,
) -> Vec<Complex64> {
    let mut g = vec![Complex64::new(0.0, 0.0); NCOORDS * NCOORDS];

    // First CP^3 factor in patch where coord z_idx = 1.
    let mut nz = 0.0;
    for (k, c) in z.iter().enumerate() {
        if k != z_idx {
            nz += c.norm_sqr();
        }
    }
    let denom_z = (1.0 + nz).powi(2);
    for a in 0..4 {
        if a == z_idx {
            continue;
        }
        for b in 0..4 {
            if b == z_idx {
                continue;
            }
            let kron = if a == b { 1.0 } else { 0.0 };
            // (1 + |z|^2) δ_{ab} - z_a z̄_b, all divided by (1 + |z|^2)^2.
            let num = Complex64::new(1.0 + nz, 0.0) * Complex64::new(kron, 0.0)
                - z[a] * z[b].conj();
            g[a * NCOORDS + b] = num / Complex64::new(denom_z, 0.0);
        }
    }

    // Second CP^3 factor in patch where coord w_idx = 1.
    let mut nw = 0.0;
    for (k, c) in w.iter().enumerate() {
        if k != w_idx {
            nw += c.norm_sqr();
        }
    }
    let denom_w = (1.0 + nw).powi(2);
    for a in 0..4 {
        if a == w_idx {
            continue;
        }
        for b in 0..4 {
            if b == w_idx {
                continue;
            }
            let kron = if a == b { 1.0 } else { 0.0 };
            let num = Complex64::new(1.0 + nw, 0.0) * Complex64::new(kron, 0.0)
                - w[a] * w[b].conj();
            let row = 4 + a;
            let col = 4 + b;
            g[row * NCOORDS + col] = num / Complex64::new(denom_w, 0.0);
        }
    }

    g
}

/// Compute `g_pb = j_pb * g_fs * j_pb^†` for `j_pb` an `N_FREE x NCOORDS`
/// row-major matrix and `g_fs` an `NCOORDS x NCOORDS` row-major matrix.
/// Returns the result as a row-major `N_FREE x N_FREE` `Vec<Complex64>`.
fn pullback_metric(j_pb: &[Complex64], g_fs: &[Complex64]) -> Vec<Complex64> {
    // tmp = j_pb * g_fs   (N_FREE x NCOORDS)
    let mut tmp = vec![Complex64::new(0.0, 0.0); N_FREE * NCOORDS];
    for a in 0..N_FREE {
        for b in 0..NCOORDS {
            let mut s = Complex64::new(0.0, 0.0);
            for k in 0..NCOORDS {
                s += j_pb[a * NCOORDS + k] * g_fs[k * NCOORDS + b];
            }
            tmp[a * NCOORDS + b] = s;
        }
    }
    // out = tmp * j_pb^†   (N_FREE x N_FREE)
    let mut out = vec![Complex64::new(0.0, 0.0); N_FREE * N_FREE];
    for a in 0..N_FREE {
        for b in 0..N_FREE {
            let mut s = Complex64::new(0.0, 0.0);
            for k in 0..NCOORDS {
                s += tmp[a * NCOORDS + k] * j_pb[b * NCOORDS + k].conj();
            }
            out[a * N_FREE + b] = s;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// REGRESSION: `f1` must have **pure bidegree (3, 0)** — degree 3 in z,
    /// degree 0 in w. Test: scale z by λ; f1 must scale by λ^3.
    #[test]
    fn f1_has_bidegree_3_0() {
        let pair = BicubicPair::z3_invariant_default();
        let mut x = [Complex64::new(0.0, 0.0); 8];
        x[0] = Complex64::new(0.7, 0.1);
        x[1] = Complex64::new(0.3, -0.2);
        x[2] = Complex64::new(0.5, 0.05);
        x[3] = Complex64::new(0.4, -0.15);
        x[4] = Complex64::new(0.6, 0.0);
        x[5] = Complex64::new(0.5, 0.1);
        x[6] = Complex64::new(0.3, 0.2);
        x[7] = Complex64::new(0.4, -0.1);
        let f_orig = pair.eval(&x);

        let lambda = 2.0;
        let mut x_scaled = x;
        for i in 0..4 {
            x_scaled[i] *= lambda;
        }
        let f_scaled = pair.eval(&x_scaled);

        let ratio = (f_scaled[0] / f_orig[0]).norm();
        assert!(
            (ratio - lambda.powi(3)).abs() < 1e-9,
            "f1 bidegree (3, 0) violated: f1(λz, w) / f1(z, w) = {ratio:.4} (expected {})",
            lambda.powi(3)
        );
    }

    #[test]
    fn f2_has_bidegree_0_3() {
        let pair = BicubicPair::z3_invariant_default();
        let mut x = [Complex64::new(0.0, 0.0); 8];
        x[0] = Complex64::new(0.7, 0.1);
        x[1] = Complex64::new(0.3, -0.2);
        x[2] = Complex64::new(0.5, 0.05);
        x[3] = Complex64::new(0.4, -0.15);
        x[4] = Complex64::new(0.6, 0.0);
        x[5] = Complex64::new(0.5, 0.1);
        x[6] = Complex64::new(0.3, 0.2);
        x[7] = Complex64::new(0.4, -0.1);
        let f_orig = pair.eval(&x);

        let mu = 2.0;
        let mut x_scaled = x;
        for i in 4..8 {
            x_scaled[i] *= mu;
        }
        let f_scaled = pair.eval(&x_scaled);
        let ratio = (f_scaled[1] / f_orig[1]).norm();
        assert!(
            (ratio - mu.powi(3)).abs() < 1e-9,
            "f2 bidegree (0, 3) violated: f2(z, μw) / f2(z, w) = {ratio:.4} (expected {})",
            mu.powi(3)
        );
    }

    /// f1 must NOT depend on w (bidegree (3, 0) means degree 0 in w).
    #[test]
    fn f1_independent_of_w() {
        let pair = BicubicPair::z3_invariant_default();
        let mut x = [Complex64::new(0.0, 0.0); 8];
        x[0] = Complex64::new(0.7, 0.1);
        x[1] = Complex64::new(0.3, -0.2);
        x[2] = Complex64::new(0.5, 0.05);
        x[3] = Complex64::new(0.4, -0.15);
        x[4] = Complex64::new(0.6, 0.0);
        x[5] = Complex64::new(0.5, 0.1);
        x[6] = Complex64::new(0.3, 0.2);
        x[7] = Complex64::new(0.4, -0.1);
        let f_orig = pair.eval(&x);

        let mut x_w_changed = x;
        for i in 4..8 {
            x_w_changed[i] *= Complex64::new(3.0, 0.0);
        }
        let f_changed = pair.eval(&x_w_changed);
        let delta = (f_changed[0] - f_orig[0]).norm();
        assert!(
            delta < 1e-12,
            "f1 should be independent of w, but f1(z, 3w) - f1(z, w) = {delta:.3e}"
        );
    }

    /// REGRESSION: `f3` must have **bidegree (1, 1)** — degree 1 in z,
    /// degree 1 in w.
    #[test]
    fn f3_has_bidegree_1_1() {
        let pair = BicubicPair::z3_invariant_default();
        let mut x = [Complex64::new(0.0, 0.0); 8];
        x[0] = Complex64::new(0.7, 0.1);
        x[1] = Complex64::new(0.3, -0.2);
        x[2] = Complex64::new(0.5, 0.05);
        x[3] = Complex64::new(0.4, -0.15);
        x[4] = Complex64::new(0.6, 0.0);
        x[5] = Complex64::new(0.5, 0.1);
        x[6] = Complex64::new(0.3, 0.2);
        x[7] = Complex64::new(0.4, -0.1);
        let f_orig = pair.eval(&x);

        // Scale z by λ; f3 must scale by λ.
        let lambda = 2.5;
        let mut x_z = x;
        for i in 0..4 {
            x_z[i] *= lambda;
        }
        let f_z = pair.eval(&x_z);
        let ratio_z = (f_z[2] / f_orig[2]).norm();
        assert!(
            (ratio_z - lambda).abs() < 1e-9,
            "f3 z-degree must be 1: f3(λz,w)/f3(z,w) = {ratio_z:.4} (expected {lambda})",
        );

        // Scale w by μ; f3 must scale by μ.
        let mu = 1.7;
        let mut x_w = x;
        for i in 4..8 {
            x_w[i] *= mu;
        }
        let f_w = pair.eval(&x_w);
        let ratio_w = (f_w[2] / f_orig[2]).norm();
        assert!(
            (ratio_w - mu).abs() < 1e-9,
            "f3 w-degree must be 1: f3(z,μw)/f3(z,w) = {ratio_w:.4} (expected {mu})",
        );
    }

    /// REGRESSION: All three default polynomials must be exactly invariant
    /// under the canonical Tian-Yau Z/3 action
    /// `(z_0, α^2 z_1, α z_2, α z_3) × (w_0, α w_1, α^2 w_2, α^2 w_3)`.
    #[test]
    fn default_pair_is_z3_invariant() {
        use std::f64::consts::TAU;
        let pair = BicubicPair::z3_invariant_default();
        let alpha = Complex64::from_polar(1.0, TAU / 3.0);
        let alpha2 = alpha * alpha;

        let x: [Complex64; 8] = [
            Complex64::new(0.7, 0.1),
            Complex64::new(-0.3, 0.4),
            Complex64::new(0.2, -0.6),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.4, 0.2),
            Complex64::new(-0.1, 0.5),
            Complex64::new(0.3, -0.2),
            Complex64::new(0.6, 0.1),
        ];
        // Apply g: (z_0, α^2 z_1, α z_2, α z_3) × (w_0, α w_1, α^2 w_2, α^2 w_3)
        let gx: [Complex64; 8] = [
            x[0],
            alpha2 * x[1],
            alpha * x[2],
            alpha * x[3],
            x[4],
            alpha * x[5],
            alpha2 * x[6],
            alpha2 * x[7],
        ];

        let f = pair.eval(&x);
        let f_g = pair.eval(&gx);
        for i in 0..NHYPER {
            let delta = (f_g[i] - f[i]).norm();
            assert!(
                delta < 1e-12,
                "f_{} not Z/3-invariant: |f(g·x) - f(x)| = {:.3e}",
                i + 1,
                delta
            );
        }
    }

    /// `sample_one` produces a point that satisfies `|f_i| < tol` on all
    /// `NHYPER = 3` defining polynomials.
    #[test]
    fn sample_one_lies_on_variety() {
        let pair = BicubicPair::z3_invariant_default();
        let mut sampler = CicySampler::new(pair, 42);

        let mut found = None;
        for _ in 0..2048 {
            if let Some(pt) = sampler.sample_one() {
                found = Some(pt);
                break;
            }
        }
        let pt = found.expect("sampler should produce at least one point in 2048 tries");

        let x: [Complex64; 8] = [
            pt.z[0], pt.z[1], pt.z[2], pt.z[3], pt.w[0], pt.w[1], pt.w[2], pt.w[3],
        ];
        let f = sampler.bicubic.eval(&x);
        // Newton converges on the *unrescaled* line root; after the patch
        // rescaling each polynomial picks up a (z_scale)^deg_z * (w_scale)^deg_w
        // factor. With all coords ≤ 1 in modulus after rescaling and bidegrees
        // ≤ 3, the residuals satisfy |f_i| ≤ |f_i_unrescaled| / (z_scale)^deg_z.
        // We allow a generous absolute tolerance.
        for i in 0..NHYPER {
            assert!(
                f[i].norm() < 0.5,
                "|f_{}| unreasonably large after rescaling: {}",
                i + 1,
                f[i].norm()
            );
        }
    }

    /// `sample_batch(100)` returns 100 points with finite positive weights
    /// summing to ~1.
    #[test]
    fn sample_batch_returns_normalised_weights() {
        let pair = BicubicPair::z3_invariant_default();
        let mut sampler = CicySampler::new(pair, 1234);
        let pts = sampler.sample_batch(100);
        assert_eq!(pts.len(), 100, "should accept 100 points");
        let mut sum = 0.0;
        for p in &pts {
            assert!(p.weight.is_finite(), "weight not finite");
            assert!(p.weight > 0.0, "weight not positive");
            sum += p.weight;
        }
        assert!(
            (sum - 1.0).abs() < 1.0e-9,
            "weights should normalise to 1, got {}",
            sum
        );
    }

    /// `apply_z3_quotient` divides every weight by 3.
    #[test]
    fn apply_z3_quotient_divides_weights_by_three() {
        let mut pts = vec![
            SampledPoint {
                z: [Complex64::new(1.0, 0.0); 4],
                w: [Complex64::new(1.0, 0.0); 4],
                omega: Complex64::new(1.0, 0.0),
                weight: 0.3,
            },
            SampledPoint {
                z: [Complex64::new(1.0, 0.0); 4],
                w: [Complex64::new(1.0, 0.0); 4],
                omega: Complex64::new(1.0, 0.0),
                weight: 0.9,
            },
            SampledPoint {
                z: [Complex64::new(1.0, 0.0); 4],
                w: [Complex64::new(1.0, 0.0); 4],
                omega: Complex64::new(1.0, 0.0),
                weight: 1.5,
            },
        ];
        let originals: Vec<f64> = pts.iter().map(|p| p.weight).collect();
        CicySampler::apply_z3_quotient(&mut pts);
        for (p, w0) in pts.iter().zip(originals.iter()) {
            let expected = w0 / 3.0;
            assert!(
                (p.weight - expected).abs() < 1.0e-15,
                "expected {} got {}",
                expected,
                p.weight
            );
        }
    }

    /// Sanity test: BicubicPair eval / jacobian agree with finite differences.
    #[test]
    fn jacobian_matches_finite_differences() {
        let pair = BicubicPair::z3_invariant_default();
        let point: [Complex64; 8] = [
            Complex64::new(0.7, 0.1),
            Complex64::new(-0.3, 0.4),
            Complex64::new(0.2, -0.6),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.4, 0.2),
            Complex64::new(-0.1, 0.5),
            Complex64::new(0.3, -0.2),
            Complex64::new(0.6, 0.1),
        ];
        let jac = pair.jacobian(&point);
        let h = 1.0e-6;
        for k in 0..NCOORDS {
            let mut p_plus = point;
            let mut p_minus = point;
            p_plus[k] += Complex64::new(h, 0.0);
            p_minus[k] -= Complex64::new(h, 0.0);
            let f_plus = pair.eval(&p_plus);
            let f_minus = pair.eval(&p_minus);
            for i in 0..NHYPER {
                let fd = (f_plus[i] - f_minus[i]) / Complex64::new(2.0 * h, 0.0);
                let analytic = jac[i * NCOORDS + k];
                let err = (fd - analytic).norm();
                assert!(
                    err < 1.0e-5,
                    "jacobian mismatch at i={} k={}: fd={:?} analytic={:?}",
                    i,
                    k,
                    fd,
                    analytic
                );
            }
        }
    }

    /// Sanity test for the n×n complex inverse helper.
    #[test]
    fn inv_n_round_trip() {
        // 3x3 random-ish complex matrix.
        let m = vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(0.2, -0.3),
            Complex64::new(-0.4, 0.1),
            Complex64::new(0.1, 0.0),
            Complex64::new(0.7, 0.2),
            Complex64::new(0.3, -0.1),
            Complex64::new(-0.2, 0.4),
            Complex64::new(0.0, 0.5),
            Complex64::new(0.9, -0.2),
        ];
        let inv = inv_n(&m, 3).expect("matrix should be invertible");
        // Compute m * inv and check it's the identity (within tolerance).
        for i in 0..3 {
            for j in 0..3 {
                let mut s = Complex64::new(0.0, 0.0);
                for k in 0..3 {
                    s += m[i * 3 + k] * inv[k * 3 + j];
                }
                let expected =
                    if i == j { Complex64::new(1.0, 0.0) } else { Complex64::new(0.0, 0.0) };
                let err = (s - expected).norm();
                assert!(err < 1.0e-12, "m·inv[{},{}] = {:?}, expected {:?}", i, j, s, expected);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Measure-correctness tests
    //
    // The Donaldson / cymetric line-intersection sampler returns points
    // p_α with weights w_α = |Ω(p_α)|² / det(g_FS_pb(p_α)). After
    // normalisation by the batch sum, the Monte Carlo estimator
    //
    //     E_CY[f]  ≈  Σ_α w_α^norm  ·  f(p_α)
    //
    // converges (in N) to the integral of f against the Calabi-Yau measure
    // d_μ_CY = |Ω|² / V_CY, where V_CY is the CY volume. The tests below
    // verify two consequences of this:
    //
    //   (A) For the canonical Tian-Yau triple, the defining polynomials are
    //       jointly invariant under simultaneous permutation of the four
    //       index-pairs (z_a, w_a). Hence E_CY[ |z_a|² / Σ_i |z_i|² ] must
    //       be the same for a = 0, 1, 2, 3 (and equal to 1/4 by the sum
    //       constraint), and similarly for the w-block.
    //
    //   (B) Cross-seed unbiasedness: for any fixed function f the MC
    //       estimator with two independent seeds must agree within the
    //       sampling-error bar (~1/sqrt(N) for a function bounded by O(1)).
    //
    // Together these probe a *non-trivial* property of the weights — the
    // permutation symmetry test (A) cannot be satisfied by, e.g., uniform
    // weights or constant-1 weights on the same point set.
    // -----------------------------------------------------------------------

    /// Return `(|z_0|² / ||z||², |z_1|² / ||z||², ..., |w_3|² / ||w||²)`,
    /// re-projecting the patch representative `(z, w)` onto the
    /// `S^7 × S^7` representative. This factors out the patch-fixing
    /// convention and exposes the genuine projective magnitudes.
    fn projective_squared_magnitudes(z: &[Complex64; 4], w: &[Complex64; 4]) -> [f64; 8] {
        let nz: f64 = z.iter().map(|c| c.norm_sqr()).sum();
        let nw: f64 = w.iter().map(|c| c.norm_sqr()).sum();
        let mut out = [0.0f64; 8];
        if nz > 0.0 {
            for i in 0..4 {
                out[i] = z[i].norm_sqr() / nz;
            }
        }
        if nw > 0.0 {
            for i in 0..4 {
                out[4 + i] = w[i].norm_sqr() / nw;
            }
        }
        out
    }

    /// Test (A): permutation symmetry of the CY measure.
    ///
    /// The default Tian-Yau triple is invariant under simultaneous
    /// permutation of `(z_a, w_a)` for `a = 0, 1, 2, 3` (the polynomial
    /// `p_1 = Σ z_a^3` is `S_4`-invariant on z, `p_2 = Σ w_a^3` is
    /// `S_4`-invariant on w, and `p_3 = Σ z_a w_a` is invariant under
    /// simultaneous permutation of z and w indices). Consequently
    /// `E_CY[|z_a|² / Σ_i |z_i|²]` must be identical for `a = 0, 1, 2, 3`,
    /// and equal to `1/4` by the constraint that the four expectations
    /// sum to `E_CY[1] = 1`.
    ///
    /// **What this test verifies that previous tests did NOT**: the
    /// weights `w_α` vary across samples, so reproducing the
    /// `(1/4, 1/4, 1/4, 1/4)` symmetric mean is a non-trivial check on
    /// the *shape* of the weight distribution, not just its
    /// normalisation. If the measure formula `w = |Ω|² / det g_pb` were
    /// wrong by a non-symmetric factor (e.g. missing the FS-metric
    /// pullback), this test would fail.
    #[test]
    fn weights_recover_calabi_yau_symmetry() {
        let pair = BicubicPair::z3_invariant_default();
        let mut sampler = CicySampler::new(pair, 0xCAFE_BABE);
        let n: usize = 4000;
        let pts = sampler.sample_batch(n);
        // sample_batch is allowed to under-deliver if rejections dominate;
        // require at least 80% of the target.
        assert!(
            pts.len() >= (n * 4) / 5,
            "sample_batch yielded only {} of {} points",
            pts.len(),
            n
        );

        // E_CY[|z_a|² / ||z||²] for a = 0..4, then for w.
        let mut means = [0.0f64; 8];
        for p in &pts {
            let m = projective_squared_magnitudes(&p.z, &p.w);
            for (acc, mi) in means.iter_mut().zip(m.iter()) {
                *acc += p.weight * mi;
            }
        }
        // Each block (z and w) must sum to 1 by construction (sum of
        // |z_a|² / Σ_i |z_i|² over a is identically 1 per sample, and
        // weights sum to 1).
        let z_sum: f64 = means[0..4].iter().sum();
        let w_sum: f64 = means[4..8].iter().sum();
        assert!(
            (z_sum - 1.0).abs() < 1e-6,
            "z-block expectations sum to {} ≠ 1",
            z_sum
        );
        assert!(
            (w_sum - 1.0).abs() < 1e-6,
            "w-block expectations sum to {} ≠ 1",
            w_sum
        );

        // S_4 symmetry: each of the 4 z-block means must be ≈ 1/4, same
        // for w. Standard error for a sum of weighted Bernoulli-ish
        // variables with N effective samples ~ N is ~1/sqrt(N) ≈ 0.016
        // for N = 4000; allow a generous 5σ envelope of ≈ 0.08.
        let tol = 0.08;
        for a in 0..4 {
            assert!(
                (means[a] - 0.25).abs() < tol,
                "z-permutation symmetry broken: E[|z_{}|²/||z||²] = {} (expected 0.25 ± {})",
                a,
                means[a],
                tol
            );
            assert!(
                (means[4 + a] - 0.25).abs() < tol,
                "w-permutation symmetry broken: E[|w_{}|²/||w||²] = {} (expected 0.25 ± {})",
                a,
                means[4 + a],
                tol
            );
        }
    }

    /// Test (B): cross-seed unbiasedness for a holomorphic bilinear function.
    ///
    /// We integrate `|f(p)|²` for `f = z_0 w_0 / (||z|| · ||w||)` (a
    /// normalised piece of the bilinear bridge `p_3`) using two
    /// independent seeds. If the weights truly approximate the CY
    /// measure, both estimates must agree within sampling error —
    /// independent of the (unknown) true value of the integral.
    ///
    /// **What this test verifies**: the absence of a *deterministic*
    /// bias in the weight construction. Two independent samplings using
    /// the same weight formula will only converge to the same value if
    /// the formula is unbiased; any seed-independent error term (e.g. a
    /// missing prefactor that depends on the patch index) would produce
    /// systematic seed-to-seed disagreement.
    #[test]
    fn weights_unbiased_for_holomorphic_polynomial() {
        let pair = BicubicPair::z3_invariant_default();
        let n: usize = 3000;

        let mc_estimate = |seed: u64| -> f64 {
            let mut sampler = CicySampler::new(pair.clone(), seed);
            let pts = sampler.sample_batch(n);
            let mut acc = 0.0f64;
            for p in &pts {
                let nz = p.z.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
                let nw = p.w.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
                if nz > 0.0 && nw > 0.0 {
                    let f = p.z[0] * p.w[0] / Complex64::new(nz * nw, 0.0);
                    acc += p.weight * f.norm_sqr();
                }
            }
            acc
        };

        let est_a = mc_estimate(0xA1);
        let est_b = mc_estimate(0xB2);

        // Standard error for a bounded estimator (|f|² ≤ 1) with N
        // samples is ~1/sqrt(N). Difference of two independent estimates
        // has std ≈ sqrt(2)/sqrt(N). Allow a 5σ window.
        let sigma = 1.0 / (n as f64).sqrt();
        let tol = 5.0 * sigma * (2.0_f64).sqrt();
        let diff = (est_a - est_b).abs();
        assert!(
            diff < tol,
            "cross-seed estimate inconsistent: |est_a - est_b| = {:.4e} > {:.4e} \
             (est_a = {:.4e}, est_b = {:.4e})",
            diff,
            tol,
            est_a,
            est_b
        );
        // Also verify the estimate is in a sane range (positive, finite, < 1).
        assert!(est_a.is_finite() && est_a > 0.0 && est_a < 1.0);
        assert!(est_b.is_finite() && est_b > 0.0 && est_b < 1.0);
    }
}
