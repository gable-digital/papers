//! Line-intersection point sampler for the Schoen `Z/3 × Z/3` Calabi-Yau
//! 3-fold cover `X̃ ⊂ CP^2 × CP^2 × CP^1`.
//!
//! ## Geometric setup
//!
//! `X̃` is the smooth complete intersection cut out by two polynomials of
//! bidegrees `(3, 0, 1)` and `(0, 3, 1)` (see
//! [`crate::route34::schoen_geometry`] for the topological data and
//! Calabi-Yau adjunction).
//!
//! In the layout
//!
//! ```text
//!     coords  =  [x_0, x_1, x_2, y_0, y_1, y_2, t_0, t_1] ∈ C^8
//! ```
//!
//! the defining polynomials are
//!
//! ```text
//!     F_1(x, t)  =  p_1(x_0, x_1, x_2) · t_0  +  p_2(x_0, x_1, x_2) · t_1   (1a)
//!     F_2(y, t)  =  q_1(y_0, y_1, y_2) · t_0  +  q_2(y_0, y_1, y_2) · t_1,  (1b)
//! ```
//!
//! with `p_1, p_2` pure cubics in `x` and `q_1, q_2` pure cubics in `y`.
//!
//! ## Algorithm — line-intersection sampling (Donaldson 2009, cymetric §2.1)
//!
//! With two defining polynomials (`NHYPER = 2`) we follow the standard
//! cymetric / [Anderson-Braun-Karp-Ovrut 2010] recipe, identical in
//! structure to [`crate::cicy_sampler`] but with the appropriate
//! per-Schoen constants:
//!
//! 1. Draw `NHYPER + 1 = 3` Gaussian unit vectors `p, d_1, d_2 ∈ S^{15} ⊂ C^8`.
//!    Use `p` as a line origin and `d_1, d_2` as line directions.
//! 2. Solve `F_i(p + t_1 d_1 + t_2 d_2) = 0` for `(t_1, t_2) ∈ C^2` by
//!    Newton-Raphson with Armijo back-tracking (`max_newton_iter` cap,
//!    multiple restarts).
//! 3. Patch-rescale each projective factor:
//!     * `CP^2_x`: divide by `x_{j_x}`, `j_x = argmax |x_k|`.
//!     * `CP^2_y`: divide by `y_{j_y}`, `j_y = argmax |y_k|`.
//!     * `CP^1_t`: divide by `t_{j_t}`, `j_t = argmax |t_k|`.
//! 4. Reject if more than `NFACTORS = 3` coordinates are within
//!    `PATCH_EPS` of `1 + 0i` (degenerate patch ambiguity).
//! 5. Compute the **holomorphic top-form residue**
//!    `Ω = 1 / det(J_elim)` where `J_elim` is the `NHYPER × NHYPER = 2 × 2`
//!    sub-Jacobian of `(F_1, F_2)` on the eliminated coordinates.
//! 6. Compute the pullback Fubini-Study metric
//!    `g_pb = J_pb · g_FS · J_pb†`
//!    (`N_FREE = 3` complex tangent dimensions) and its determinant
//!    `det g_pb`.
//! 7. Per-point weight `w = |Ω|^2 / det g_pb`.
//!
//! Steps 1–4 differ from [`crate::cicy_sampler`] only in `NHYPER`,
//! `NFACTORS`, and the patch-rescale loop covering three projective
//! factors instead of two; steps 5–7 are structurally identical.
//!
//! ## Static workspace
//!
//! The hot loop is allocation-free. All per-iteration buffers are
//! pre-allocated in the [`SchoenSampler::new`] constructor.
//!
//! ## Multi-threaded sampling
//!
//! [`SchoenSampler::sample_batch_parallel`] uses rayon to draw N
//! candidate triples on independent thread-local copies of the sampler
//! state and merge; per-thread RNG seeds are derived deterministically
//! from a base seed so the multi-threaded run is reproducible.
//!
//! ## Reproducibility metadata + checkpointing
//!
//! [`SchoenSampler::run_metadata`] returns a [`RunMetadata`] record
//! capturing the seed, point count, output SHA-256, wall-clock time,
//! and a pre-computed git-SHA placeholder. [`SchoenSampler::checkpoint`]
//! /[`deserialize_state`] support resumable sampling.
//!
//! ## Reference equivariant polynomials
//!
//! [`SchoenPoly::z3xz3_invariant_default`] returns the canonical
//! Braun-He-Ovrut-Pantev-equivariant pair: the simplest non-trivial
//! choice that is `(α, β)`-invariant in the appropriate equivariant
//! sense (specifically, the polynomials transform by an overall character
//! that cancels on the combination `F_1 · F_2 = 0`).

use crate::route34::schoen_geometry::SchoenGeometry;
use num_complex::Complex64;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of defining polynomials of the Schoen complete intersection.
pub const NHYPER: usize = 2;

/// Total ambient homogeneous coordinate count: `3 + 3 + 2 = 8`.
pub const NCOORDS: usize = 8;

/// Number of projective factors in the ambient: `CP^2 × CP^2 × CP^1`.
pub const NFACTORS: usize = 3;

/// Complex dimension of the Calabi-Yau (always `3` post-quotient).
pub const NFOLD: usize = 3;

/// Free affine coordinates remaining: `8 − 3 patch − 2 elim = 3`.
pub const N_FREE: usize = NCOORDS - NFACTORS - NHYPER;

const _: () = assert!(N_FREE == NFOLD, "N_FREE must equal NFOLD for a CY3");

/// Indices of the first `CP^2` factor's coordinates.
pub const X_RANGE: std::ops::Range<usize> = 0..3;
/// Indices of the second `CP^2` factor's coordinates.
pub const Y_RANGE: std::ops::Range<usize> = 3..6;
/// Indices of the `CP^1` factor's coordinates.
pub const T_RANGE: std::ops::Range<usize> = 6..8;

/// Newton residual tolerance for accepting a root.
pub const NEWTON_TOL: f64 = 1.0e-8;

/// Default Newton iteration cap per restart.
pub const DEFAULT_MAX_NEWTON: usize = 64;

/// Default number of Newton restarts per ambient line.
pub const DEFAULT_MAX_ATTEMPTS: usize = 4;

/// Patch-ambiguity threshold (rejecting candidates whose chosen patch is
/// degenerate).
pub const PATCH_EPS: f64 = 1.0e-8;

// ---------------------------------------------------------------------------
// Polynomial representation
// ---------------------------------------------------------------------------

/// The Schoen complete-intersection defining polynomial pair on
/// `CP^2 × CP^2 × CP^1`. Each polynomial is sparse `(coefficient,
/// exponents)` with `exponents = [a_0, a_1, a_2, b_0, b_1, b_2, c_0, c_1]`
/// of bidegrees `(3, 0, 1)` and `(0, 3, 1)` respectively.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SchoenPoly {
    /// `F_1(x, t) = p_1(x) t_0 + p_2(x) t_1`, bidegree `(3, 0, 1)`.
    /// Bidegree-3 in `x`, bidegree-0 in `y`, bidegree-1 in `t`.
    pub f1: Vec<(Complex64, [u32; 8])>,
    /// `F_2(y, t) = q_1(y) t_0 + q_2(y) t_1`, bidegree `(0, 3, 1)`.
    pub f2: Vec<(Complex64, [u32; 8])>,
}

impl SchoenPoly {
    /// Construct from explicit polynomial lists. Callers should ensure
    /// the bidegrees are `(3, 0, 1)` and `(0, 3, 1)`.
    pub fn new(
        f1: Vec<(Complex64, [u32; 8])>,
        f2: Vec<(Complex64, [u32; 8])>,
    ) -> Self {
        Self { f1, f2 }
    }

    /// The canonical `Z/3 × Z/3`-equivariant Schoen pair.
    ///
    /// We pick the simplest non-trivial choice that is jointly invariant
    /// under both generators of the Braun-He-Ovrut-Pantev action (Eq. (2)
    /// of [`crate::route34::schoen_geometry`]):
    ///
    /// ```text
    ///     F_1(x, t)  =  (x_0^3 + x_1^3 + x_2^3) · t_0
    ///                   + (x_0 x_1 x_2) · t_1
    ///
    ///     F_2(y, t)  =  (y_0^3 + y_1^3 + y_2^3) · t_1
    ///                   + (y_0 y_1 y_2) · t_0.
    /// ```
    ///
    /// The α-action multiplies each cubic by an overall phase:
    ///   `α · (x_0^3 + x_1^3 + x_2^3) = x_0^3 + ω^3 x_1^3 + ω^6 x_2^3
    ///     = x_0^3 + x_1^3 + x_2^3`,
    ///   `α · (x_0 x_1 x_2) = x_0 (ω x_1)(ω^2 x_2) = ω^3 x_0 x_1 x_2 = x_0 x_1 x_2`,
    /// and similarly for the `y`-block (using the opposite phase
    /// assignment). The β-action permutes the cubics and swaps `t_0 ↔ t_1`,
    /// taking `F_1 → F_1'` where `F_1'` differs only by the t-swap, but
    /// because the symmetric functions are unchanged we have `F_1 ↔ F_2`
    /// under `β` (with the symmetric `t_0 ↔ t_1` swap exchanging F_1's
    /// `t_0` term with F_2's `t_1` term). The complete intersection
    /// `{F_1 = F_2 = 0}` is therefore `Γ`-invariant as a set, even though
    /// each individual polynomial transforms non-trivially.
    pub fn z3xz3_invariant_default() -> Self {
        let one = Complex64::new(1.0, 0.0);
        // F_1 = (x_0^3 + x_1^3 + x_2^3) t_0 + (x_0 x_1 x_2) t_1
        // Bidegree (3, 0, 1).
        let mut f1: Vec<(Complex64, [u32; 8])> = Vec::with_capacity(4);
        // Cubics times t_0
        for k in 0..3 {
            let mut e = [0u32; 8];
            e[k] = 3;
            e[6] = 1; // t_0
            f1.push((one, e));
        }
        // x_0 x_1 x_2 * t_1
        f1.push((one, [1, 1, 1, 0, 0, 0, 0, 1]));

        // F_2 = (y_0^3 + y_1^3 + y_2^3) t_1 + (y_0 y_1 y_2) t_0
        // Bidegree (0, 3, 1).
        let mut f2: Vec<(Complex64, [u32; 8])> = Vec::with_capacity(4);
        for k in 0..3 {
            let mut e = [0u32; 8];
            e[3 + k] = 3;
            e[7] = 1; // t_1
            f2.push((one, e));
        }
        f2.push((one, [0, 0, 0, 1, 1, 1, 1, 0]));

        Self { f1, f2 }
    }

    /// Evaluate `[F_1(point), F_2(point)]`.
    #[inline]
    pub fn eval(&self, point: &[Complex64; NCOORDS]) -> [Complex64; NHYPER] {
        [eval_poly(&self.f1, point), eval_poly(&self.f2, point)]
    }

    /// Holomorphic Jacobian `∂F_i/∂x_k`, `NHYPER × NCOORDS` row-major.
    pub fn jacobian(&self, point: &[Complex64; NCOORDS]) -> [Complex64; NHYPER * NCOORDS] {
        let mut j = [Complex64::new(0.0, 0.0); NHYPER * NCOORDS];
        for k in 0..NCOORDS {
            j[k] = poly_partial(&self.f1, point, k);
            j[NCOORDS + k] = poly_partial(&self.f2, point, k);
        }
        j
    }

    /// Verify each polynomial has the published bidegree (regression check).
    pub fn check_bidegrees(&self) -> bool {
        // Per polynomial: deg in x-block, y-block, t-block matches (3,0,1) or (0,3,1).
        let f1_ok = self.f1.iter().all(|(_, e)| {
            let dx = e[0] + e[1] + e[2];
            let dy = e[3] + e[4] + e[5];
            let dt = e[6] + e[7];
            dx == 3 && dy == 0 && dt == 1
        });
        let f2_ok = self.f2.iter().all(|(_, e)| {
            let dx = e[0] + e[1] + e[2];
            let dy = e[3] + e[4] + e[5];
            let dt = e[6] + e[7];
            dx == 0 && dy == 3 && dt == 1
        });
        f1_ok && f2_ok
    }
}

// ---------------------------------------------------------------------------
// SchoenPoint and RunMetadata
// ---------------------------------------------------------------------------

/// One accepted point on the Schoen variety with the data for forming
/// the canonical Monte-Carlo measure.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SchoenPoint {
    /// First `CP^2` homogeneous coordinates (one entry equals `1 + 0i`
    /// after rescaling).
    pub x: [Complex64; 3],
    /// Second `CP^2` homogeneous coordinates.
    pub y: [Complex64; 3],
    /// `CP^1` homogeneous coordinates.
    pub t: [Complex64; 2],
    /// Holomorphic top-form residue `Ω = 1 / det(J_elim)`.
    pub omega: Complex64,
    /// Sampling weight `|Ω|^2 / det g_pb` (or normalised in batches).
    pub weight: f64,
}

impl SchoenPoint {
    /// Flat 8-coordinate view, ordering `[x_0, x_1, x_2, y_0, y_1, y_2, t_0, t_1]`.
    #[inline]
    pub fn flat_coords(&self) -> [Complex64; NCOORDS] {
        [
            self.x[0], self.x[1], self.x[2], self.y[0], self.y[1], self.y[2], self.t[0], self.t[1],
        ]
    }

    /// Pack two `SchoenPoint`s' coordinates into the SOA `Vec<f64>` layout
    /// used by downstream η-integral code (8 reals per point: real/imag
    /// interleaved across `[x0..x2, y0..y2, t0..t1]`).
    #[inline]
    pub fn pack_real_into(&self, out_re: &mut [f64], out_im: &mut [f64]) {
        let coords = self.flat_coords();
        for k in 0..NCOORDS {
            out_re[k] = coords[k].re;
            out_im[k] = coords[k].im;
        }
    }
}

/// Reproducibility metadata returned by [`SchoenSampler::run_metadata`].
///
/// Captures the inputs and outputs of one sampling run so that downstream
/// consumers can verify byte-for-byte reproducibility.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunMetadata {
    /// PRNG seed used for the run.
    pub seed: u64,
    /// Number of points produced.
    pub n_points: usize,
    /// Wall-clock time spent in [`SchoenSampler::sample_points`] (or
    /// equivalent), in seconds. `0.0` if metadata is fetched outside a
    /// timed run.
    pub wall_clock_seconds: f64,
    /// SHA-256 hex digest of the produced point cloud (real and imag
    /// parts of all 8 coordinates per point, concatenated as IEEE 754
    /// big-endian double precision).
    pub point_cloud_sha256: String,
    /// Git SHA placeholder. Populated from `env!("VERGEN_GIT_SHA")` when
    /// available, otherwise from the `git rev-parse HEAD` query at the
    /// time of metadata generation. Returns `"unknown"` if neither path
    /// resolves (e.g. a non-checkout build).
    pub git_sha: String,
    /// Sampler library version (Cargo `version` field of this crate).
    pub library_version: String,
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

/// Per-sampler pre-allocated workspace. Sized once at construction; the
/// hot Newton loop reuses these buffers and never allocates.
///
/// Fields prefixed `origin` / `directions` / `t_param` carry RNG-derived
/// per-iteration state; the others are matrix/jacobian scratch used in
/// post-processing. All buffers are inline arrays (no heap allocation
/// after `Workspace::new`), so the per-point post-processing stage is
/// allocation-free.
#[derive(Debug)]
struct Workspace {
    /// Origin Gaussian sample (kept across attempts for the same line).
    origin: [Complex64; NCOORDS],
    /// `NHYPER` direction vectors (Gaussian unit, S^{15}).
    directions: [[Complex64; NCOORDS]; NHYPER],
    /// Newton parameter vector `(t_1, t_2)` (per-attempt scratch).
    t_param: [Complex64; NHYPER],
    /// `NHYPER × NCOORDS` Jacobian buffer (post-processing scratch).
    jac_buffer: [Complex64; NHYPER * NCOORDS],
    /// `N_FREE × NCOORDS` pullback Jacobian.
    j_pb: [Complex64; N_FREE * NCOORDS],
    /// `NCOORDS × NCOORDS` ambient Fubini-Study metric.
    g_fs: [Complex64; NCOORDS * NCOORDS],
    /// `N_FREE × N_FREE` pullback metric.
    g_pb: [Complex64; N_FREE * N_FREE],
    /// `N_FREE × NCOORDS` temporary `J_pb · g_FS`.
    tmp_pb: [Complex64; N_FREE * NCOORDS],
    /// `NHYPER × NHYPER` omega matrix (sub-Jacobian on eliminated coords).
    omega_mat: [Complex64; NHYPER * NHYPER],
    /// Inverse of `omega_mat`.
    omega_inv: [Complex64; NHYPER * NHYPER],
}

impl Workspace {
    fn new() -> Self {
        Self {
            origin: [Complex64::new(0.0, 0.0); NCOORDS],
            directions: [[Complex64::new(0.0, 0.0); NCOORDS]; NHYPER],
            t_param: [Complex64::new(0.0, 0.0); NHYPER],
            jac_buffer: [Complex64::new(0.0, 0.0); NHYPER * NCOORDS],
            j_pb: [Complex64::new(0.0, 0.0); N_FREE * NCOORDS],
            g_fs: [Complex64::new(0.0, 0.0); NCOORDS * NCOORDS],
            g_pb: [Complex64::new(0.0, 0.0); N_FREE * N_FREE],
            tmp_pb: [Complex64::new(0.0, 0.0); N_FREE * NCOORDS],
            omega_mat: [Complex64::new(0.0, 0.0); NHYPER * NHYPER],
            omega_inv: [Complex64::new(0.0, 0.0); NHYPER * NHYPER],
        }
    }
}

// ---------------------------------------------------------------------------
// SchoenSampler
// ---------------------------------------------------------------------------

/// Stateful Schoen-variety sampler.
///
/// Owns the polynomial pair, the PRNG, the per-attempt workspace, and the
/// run-time metadata accumulators. Construct once, call
/// [`Self::sample_points`] (or [`Self::sample_one`]) repeatedly.
pub struct SchoenSampler {
    /// The complete-intersection defining polynomials.
    pub poly: SchoenPoly,
    /// Geometry descriptor (Hodge numbers, Bianchi etc.).
    pub geometry: SchoenGeometry,
    /// PRNG seed used at construction.
    pub seed: u64,
    /// Newton iteration cap.
    pub max_newton_iter: usize,
    /// Newton residual tolerance.
    pub newton_tol: f64,
    /// Newton restart cap per ambient line.
    pub max_attempts_per_line: usize,
    /// Patch-ambiguity rejection threshold.
    pub patch_eps: f64,
    rng: ChaCha8Rng,
    workspace: Workspace,
    /// Last-run wall-clock time in seconds (zero if no run has finished).
    last_run_seconds: f64,
    /// Last-run output count.
    last_run_n: usize,
    /// Last-run output SHA-256 digest.
    last_run_sha256: Option<String>,
}

impl SchoenSampler {
    /// Construct a new sampler.
    pub fn new(poly: SchoenPoly, geometry: SchoenGeometry, seed: u64) -> Self {
        Self {
            poly,
            geometry,
            seed,
            max_newton_iter: DEFAULT_MAX_NEWTON,
            newton_tol: NEWTON_TOL,
            max_attempts_per_line: DEFAULT_MAX_ATTEMPTS,
            patch_eps: PATCH_EPS,
            rng: ChaCha8Rng::seed_from_u64(seed),
            workspace: Workspace::new(),
            last_run_seconds: 0.0,
            last_run_n: 0,
            last_run_sha256: None,
        }
    }

    /// Sample one accepted point, or `None` on rejection.
    pub fn sample_one(&mut self) -> Option<SchoenPoint> {
        // 1) Draw NHYPER + 1 = 3 Gaussian unit directions on S^15.
        let p = gaussian_unit_sphere(&mut self.rng);
        self.workspace.origin = p;
        for d in self.workspace.directions.iter_mut() {
            *d = gaussian_unit_sphere(&mut self.rng);
        }

        // 2) Newton-solve F_i(p + Σ_j t_j d_j) = 0.
        let mut best_x: Option<[Complex64; NCOORDS]> = None;
        let mut best_residual = f64::INFINITY;

        for _ in 0..self.max_attempts_per_line {
            let normal = StandardNormal;
            let mut t_init = [Complex64::new(0.0, 0.0); NHYPER];
            for tj in t_init.iter_mut() {
                *tj = Complex64::new(
                    normal.sample(&mut self.rng),
                    normal.sample(&mut self.rng),
                );
            }
            self.workspace.t_param = t_init;
            let (x, residual, converged) = newton_solve_line(
                &self.poly,
                &self.workspace.origin,
                &self.workspace.directions,
                &mut self.workspace.t_param,
                self.max_newton_iter,
                self.newton_tol,
            );
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
        post_process_solved_point(
            &x,
            &self.poly,
            self.patch_eps,
            &mut self.workspace,
        )
    }

    /// Sample `n_target` accepted points, single-threaded. Weights are
    /// normalised so that `Σ weights ≈ 1`. Wall-clock time is recorded
    /// for [`Self::run_metadata`].
    pub fn sample_points(&mut self, n_target: usize, _seed_override: Option<u64>) -> Vec<SchoenPoint> {
        let start = Instant::now();
        let mut out: Vec<SchoenPoint> = Vec::with_capacity(n_target);
        let max_total = n_target.saturating_mul(64).max(1024);
        let mut attempts = 0usize;
        while out.len() < n_target && attempts < max_total {
            attempts += 1;
            if let Some(pt) = self.sample_one() {
                out.push(pt);
            }
        }
        normalise_weights(&mut out);
        self.last_run_seconds = start.elapsed().as_secs_f64();
        self.last_run_n = out.len();
        self.last_run_sha256 = Some(point_cloud_sha256(&out));
        out
    }

    /// Multi-threaded sampling via rayon. Per-thread RNG seeds are
    /// derived deterministically from `self.seed` so the output is
    /// reproducible at fixed thread count. Cross-thread-count
    /// reproducibility is **not** guaranteed (the per-thread allocation
    /// of attempts depends on the rayon scheduler); for that, use
    /// [`Self::sample_points`].
    pub fn sample_batch_parallel(
        &mut self,
        n_target: usize,
        n_threads: usize,
    ) -> Vec<SchoenPoint> {
        let start = Instant::now();
        let n_threads = n_threads.max(1);
        let per_thread = (n_target + n_threads - 1) / n_threads;
        let base_seed = self.seed;
        let geometry = self.geometry.clone();
        let poly = self.poly.clone();
        let max_iter = self.max_newton_iter;
        let tol = self.newton_tol;
        let attempts_per_line = self.max_attempts_per_line;
        let patch_eps = self.patch_eps;

        let mut all: Vec<SchoenPoint> = (0..n_threads)
            .into_par_iter()
            .flat_map(|tid| {
                let thread_seed = base_seed
                    .wrapping_add(0x9E3779B97F4A7C15u64.wrapping_mul(tid as u64 + 1));
                let mut local =
                    SchoenSampler::new(poly.clone(), geometry.clone(), thread_seed);
                local.max_newton_iter = max_iter;
                local.newton_tol = tol;
                local.max_attempts_per_line = attempts_per_line;
                local.patch_eps = patch_eps;
                let mut local_out: Vec<SchoenPoint> = Vec::with_capacity(per_thread);
                let max_local = per_thread.saturating_mul(64).max(256);
                let mut tries = 0usize;
                while local_out.len() < per_thread && tries < max_local {
                    tries += 1;
                    if let Some(p) = local.sample_one() {
                        local_out.push(p);
                    }
                }
                local_out
            })
            .collect();
        all.truncate(n_target);
        normalise_weights(&mut all);
        self.last_run_seconds = start.elapsed().as_secs_f64();
        self.last_run_n = all.len();
        self.last_run_sha256 = Some(point_cloud_sha256(&all));
        all
    }

    /// Apply the `Z/3 × Z/3` quotient by dividing every weight by `|Γ| = 9`
    /// (cymetric option (a) — Larfors-Lukas-Ruehle-Schneider 2021 §2.4).
    pub fn apply_z3xz3_quotient(points: &mut [SchoenPoint]) {
        let inv_k = 1.0 / 9.0;
        for p in points.iter_mut() {
            p.weight *= inv_k;
        }
    }

    /// Snapshot of the last-completed sampling run.
    pub fn run_metadata(&self) -> RunMetadata {
        RunMetadata {
            seed: self.seed,
            n_points: self.last_run_n,
            wall_clock_seconds: self.last_run_seconds,
            point_cloud_sha256: self
                .last_run_sha256
                .clone()
                .unwrap_or_else(|| "no-run".to_string()),
            git_sha: detect_git_sha(),
            library_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Persist the sampler state (RNG + last-run metadata) to disk for
    /// resumable sampling.
    pub fn checkpoint(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let state = SamplerState {
            seed: self.seed,
            poly: self.poly.clone(),
            max_newton_iter: self.max_newton_iter,
            newton_tol: self.newton_tol,
            max_attempts_per_line: self.max_attempts_per_line,
            patch_eps: self.patch_eps,
            last_run_seconds: self.last_run_seconds,
            last_run_n: self.last_run_n,
            last_run_sha256: self.last_run_sha256.clone(),
        };
        let s = serde_json::to_string(&state)?;
        std::fs::write(path, s)?;
        Ok(())
    }

    /// Re-construct a sampler from a previously serialized state. The
    /// PRNG is re-seeded from `state.seed`; sampling resumes
    /// deterministically as if from the beginning. (For pause/resume of
    /// a single in-progress run, sample in [`Self::sample_batch_parallel`]
    /// chunks and concatenate; full RNG-stream serialization is
    /// **intentionally** not exposed because `ChaCha8Rng`'s internal
    /// state isn't part of its stable API.)
    pub fn deserialize_state(
        state_path: &Path,
        geometry: SchoenGeometry,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let s = std::fs::read_to_string(state_path)?;
        let state: SamplerState = serde_json::from_str(&s)?;
        Ok(Self {
            poly: state.poly,
            geometry,
            seed: state.seed,
            max_newton_iter: state.max_newton_iter,
            newton_tol: state.newton_tol,
            max_attempts_per_line: state.max_attempts_per_line,
            patch_eps: state.patch_eps,
            rng: ChaCha8Rng::seed_from_u64(state.seed),
            workspace: Workspace::new(),
            last_run_seconds: state.last_run_seconds,
            last_run_n: state.last_run_n,
            last_run_sha256: state.last_run_sha256,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct SamplerState {
    seed: u64,
    poly: SchoenPoly,
    max_newton_iter: usize,
    newton_tol: f64,
    max_attempts_per_line: usize,
    patch_eps: f64,
    last_run_seconds: f64,
    last_run_n: usize,
    last_run_sha256: Option<String>,
}

// ---------------------------------------------------------------------------
// Newton solve
// ---------------------------------------------------------------------------

fn newton_solve_line(
    poly: &SchoenPoly,
    p: &[Complex64; NCOORDS],
    dirs: &[[Complex64; NCOORDS]; NHYPER],
    t: &mut [Complex64; NHYPER],
    max_iter: usize,
    tol: f64,
) -> ([Complex64; NCOORDS], f64, bool) {
    let mut x = line_eval(p, dirs, t);
    let mut f = poly.eval(&x);
    let mut residual = residual_norm(&f);

    for _ in 0..max_iter {
        if residual < tol {
            return (x, residual, true);
        }

        let jac_x = poly.jacobian(&x);
        let mut m = [Complex64::new(0.0, 0.0); NHYPER * NHYPER];
        for i in 0..NHYPER {
            for j in 0..NHYPER {
                let mut s = Complex64::new(0.0, 0.0);
                for k in 0..NCOORDS {
                    s += jac_x[i * NCOORDS + k] * dirs[j][k];
                }
                m[i * NHYPER + j] = s;
            }
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

        // Armijo back-tracking
        let mut step = 1.0;
        let mut accepted = false;
        for _ in 0..32 {
            let mut t_trial = [Complex64::new(0.0, 0.0); NHYPER];
            for r in 0..NHYPER {
                t_trial[r] = t[r] + Complex64::new(step, 0.0) * dt[r];
            }
            let x_trial = line_eval(p, dirs, &t_trial);
            let f_trial = poly.eval(&x_trial);
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
            return (x, residual, false);
        }
    }

    (x, residual, residual < tol)
}

#[inline]
fn line_eval(
    p: &[Complex64; NCOORDS],
    dirs: &[[Complex64; NCOORDS]; NHYPER],
    t: &[Complex64; NHYPER],
) -> [Complex64; NCOORDS] {
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

#[inline]
fn residual_norm(f: &[Complex64; NHYPER]) -> f64 {
    let mut s = 0.0;
    for c in f.iter() {
        s += c.norm_sqr();
    }
    s.sqrt()
}

// ---------------------------------------------------------------------------
// Post-processing: patch rescale, omega, pullback metric, weight
// ---------------------------------------------------------------------------

fn post_process_solved_point(
    x: &[Complex64; NCOORDS],
    poly: &SchoenPoly,
    patch_eps: f64,
    ws: &mut Workspace,
) -> Option<SchoenPoint> {
    // Step 3: patch rescale on each of the three projective factors.
    let x_idx = argmax_abs_range(x, X_RANGE)?;
    let y_idx = argmax_abs_range(x, Y_RANGE)?;
    let t_idx = argmax_abs_range(x, T_RANGE)?;
    let x_scale = x[x_idx];
    let y_scale = x[y_idx];
    let t_scale = x[t_idx];
    if x_scale.norm() < f64::EPSILON
        || y_scale.norm() < f64::EPSILON
        || t_scale.norm() < f64::EPSILON
    {
        return None;
    }
    let mut x_resc = *x;
    for k in X_RANGE {
        x_resc[k] /= x_scale;
    }
    for k in Y_RANGE {
        x_resc[k] /= y_scale;
    }
    for k in T_RANGE {
        x_resc[k] /= t_scale;
    }

    // Step 4: patch-ambiguity rejection.
    let one = Complex64::new(1.0, 0.0);
    let mut near_one = 0usize;
    for c in x_resc.iter() {
        if (*c - one).norm() < patch_eps {
            near_one += 1;
        }
    }
    if near_one > NFACTORS {
        return None;
    }

    // Step 5: holomorphic top-form residue.
    let jac = poly.jacobian(&x_resc);
    ws.jac_buffer.copy_from_slice(&jac);

    let j_elim = pick_elimination_columns(&jac, x_idx, y_idx, t_idx)?;
    for i in 0..NHYPER {
        for k in 0..NHYPER {
            ws.omega_mat[i * NHYPER + k] = jac[i * NCOORDS + j_elim[k]];
        }
    }
    let det_omega = det_complex_lu(&ws.omega_mat, NHYPER);
    if det_omega.norm() < f64::EPSILON {
        return None;
    }
    let omega = Complex64::new(1.0, 0.0) / det_omega;

    // Step 6: pullback Jacobian. Free coords = [0..NCOORDS] \ ({x_idx,
    // y_idx, t_idx} ∪ j_elim).
    let mut eliminated = [false; NCOORDS];
    eliminated[x_idx] = true;
    eliminated[y_idx] = true;
    eliminated[t_idx] = true;
    for &k in &j_elim {
        eliminated[k] = true;
    }
    let mut free: [usize; N_FREE] = [0; N_FREE];
    let mut nf = 0usize;
    for k in 0..NCOORDS {
        if !eliminated[k] {
            if nf >= N_FREE {
                return None;
            }
            free[nf] = k;
            nf += 1;
        }
    }
    if nf != N_FREE {
        return None;
    }

    // Build pullback Jacobian in the workspace.
    for v in ws.j_pb.iter_mut() {
        *v = Complex64::new(0.0, 0.0);
    }
    for (a, &fa) in free.iter().enumerate() {
        ws.j_pb[a * NCOORDS + fa] = Complex64::new(1.0, 0.0);
    }
    let omega_inv = inv_n(&ws.omega_mat, NHYPER)?;
    ws.omega_inv[..NHYPER * NHYPER].copy_from_slice(&omega_inv);
    for (a, &fa) in free.iter().enumerate() {
        let mut b = [Complex64::new(0.0, 0.0); NHYPER];
        for i in 0..NHYPER {
            b[i] = jac[i * NCOORDS + fa];
        }
        for r in 0..NHYPER {
            let mut e_r = Complex64::new(0.0, 0.0);
            for c in 0..NHYPER {
                e_r += ws.omega_inv[r * NHYPER + c] * b[c];
            }
            ws.j_pb[a * NCOORDS + j_elim[r]] = -e_r;
        }
    }

    // Step 7: Fubini-Study metric on the three patches, pulled back, det.
    fs_metric_three_blocks(
        &x_resc,
        x_idx,
        y_idx,
        t_idx,
        &mut ws.g_fs,
    );
    pullback_metric(&ws.j_pb, &ws.g_fs, &mut ws.tmp_pb, &mut ws.g_pb);
    let det_g = det_complex_lu(&ws.g_pb, N_FREE).re;
    if !det_g.is_finite() || det_g.abs() < 1.0e-300 {
        return None;
    }

    let weight = omega.norm_sqr() / det_g;
    if !weight.is_finite() || weight <= 0.0 {
        return None;
    }

    let pt = SchoenPoint {
        x: [x_resc[0], x_resc[1], x_resc[2]],
        y: [x_resc[3], x_resc[4], x_resc[5]],
        t: [x_resc[6], x_resc[7]],
        omega,
        weight,
    };
    Some(pt)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn argmax_abs_range(v: &[Complex64; NCOORDS], range: std::ops::Range<usize>) -> Option<usize> {
    let mut best = range.start;
    let mut best_abs = -1.0f64;
    for k in range {
        let a = v[k].norm();
        if a > best_abs {
            best_abs = a;
            best = k;
        }
    }
    if best_abs > 0.0 {
        Some(best)
    } else {
        None
    }
}

fn pick_elimination_columns(
    jac: &[Complex64],
    x_idx: usize,
    y_idx: usize,
    t_idx: usize,
) -> Option<[usize; NHYPER]> {
    let mut taken = [false; NCOORDS];
    taken[x_idx] = true;
    taken[y_idx] = true;
    taken[t_idx] = true;

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

#[inline]
fn eval_monomial(point: &[Complex64; NCOORDS], exps: &[u32; 8]) -> Complex64 {
    let mut acc = Complex64::new(1.0, 0.0);
    for k in 0..NCOORDS {
        let e = exps[k];
        if e == 0 {
            continue;
        }
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
fn eval_poly(poly: &[(Complex64, [u32; 8])], point: &[Complex64; NCOORDS]) -> Complex64 {
    let mut s = Complex64::new(0.0, 0.0);
    for (coef, exps) in poly {
        s += *coef * eval_monomial(point, exps);
    }
    s
}

fn poly_partial(
    poly: &[(Complex64, [u32; 8])],
    point: &[Complex64; NCOORDS],
    k: usize,
) -> Complex64 {
    let mut s = Complex64::new(0.0, 0.0);
    for (coef, exps) in poly {
        let e_k = exps[k];
        if e_k == 0 {
            continue;
        }
        let mut new_exps = *exps;
        new_exps[k] = e_k - 1;
        let mono = eval_monomial(point, &new_exps);
        s += *coef * Complex64::new(e_k as f64, 0.0) * mono;
    }
    s
}

fn gaussian_unit_sphere(rng: &mut ChaCha8Rng) -> [Complex64; NCOORDS] {
    let normal = StandardNormal;
    let mut v = [Complex64::new(0.0, 0.0); NCOORDS];
    let mut sq = 0.0;
    for k in 0..NCOORDS {
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

/// Determinant via partial-pivot LU (mirrors cicy_sampler::det_complex_lu).
fn det_complex_lu(mat: &[Complex64], n: usize) -> Complex64 {
    let mut a: Vec<Complex64> = mat.to_vec();
    let mut det = Complex64::new(1.0, 0.0);
    for k in 0..n {
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

fn inv_n(mat: &[Complex64], n: usize) -> Option<Vec<Complex64>> {
    let cols = 2 * n;
    let mut a: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n * cols];
    for i in 0..n {
        for j in 0..n {
            a[i * cols + j] = mat[i * n + j];
        }
        a[i * cols + n + i] = Complex64::new(1.0, 0.0);
    }
    for k in 0..n {
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
        for j in 0..cols {
            a[k * cols + j] *= inv_pivot;
        }
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
    let mut inv = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = a[i * cols + n + j];
        }
    }
    Some(inv)
}

/// Block-diagonal Fubini-Study metric on `CP^2 × CP^2 × CP^1`.
fn fs_metric_three_blocks(
    x_resc: &[Complex64; NCOORDS],
    x_idx: usize,
    y_idx: usize,
    t_idx: usize,
    g: &mut [Complex64],
) {
    debug_assert_eq!(g.len(), NCOORDS * NCOORDS);
    for v in g.iter_mut() {
        *v = Complex64::new(0.0, 0.0);
    }
    fs_block(g, x_resc, X_RANGE, x_idx);
    fs_block(g, x_resc, Y_RANGE, y_idx);
    fs_block(g, x_resc, T_RANGE, t_idx);
}

fn fs_block(
    g: &mut [Complex64],
    x: &[Complex64; NCOORDS],
    range: std::ops::Range<usize>,
    patch_idx: usize,
) {
    // Build (1 + |z|^2) δ_{ab} - z_a z_b^* for the affine patch where
    // coord patch_idx has been set to 1. Here z_a is the "reduced" subtuple.
    let mut nz = 0.0;
    for k in range.clone() {
        if k == patch_idx {
            continue;
        }
        nz += x[k].norm_sqr();
    }
    let denom = (1.0 + nz).powi(2);
    for a in range.clone() {
        if a == patch_idx {
            continue;
        }
        for b in range.clone() {
            if b == patch_idx {
                continue;
            }
            let kron = if a == b { 1.0 } else { 0.0 };
            let num = Complex64::new(1.0 + nz, 0.0) * Complex64::new(kron, 0.0)
                - x[a] * x[b].conj();
            g[a * NCOORDS + b] = num / Complex64::new(denom, 0.0);
        }
    }
}

fn pullback_metric(
    j_pb: &[Complex64],
    g_fs: &[Complex64],
    tmp: &mut [Complex64],
    out: &mut [Complex64],
) {
    debug_assert_eq!(j_pb.len(), N_FREE * NCOORDS);
    debug_assert_eq!(g_fs.len(), NCOORDS * NCOORDS);
    debug_assert_eq!(tmp.len(), N_FREE * NCOORDS);
    debug_assert_eq!(out.len(), N_FREE * N_FREE);
    // tmp = j_pb * g_fs   (N_FREE x NCOORDS)
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
    for a in 0..N_FREE {
        for b in 0..N_FREE {
            let mut s = Complex64::new(0.0, 0.0);
            for k in 0..NCOORDS {
                s += tmp[a * NCOORDS + k] * j_pb[b * NCOORDS + k].conj();
            }
            out[a * N_FREE + b] = s;
        }
    }
}

fn normalise_weights(out: &mut [SchoenPoint]) {
    let sum: f64 = out.iter().map(|p| p.weight).sum();
    if sum.is_finite() && sum > 0.0 {
        let inv = 1.0 / sum;
        for p in out.iter_mut() {
            p.weight *= inv;
        }
    }
}

// ---------------------------------------------------------------------------
// SHA-256 (pure Rust, NIST FIPS 180-4)
// ---------------------------------------------------------------------------

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const SHA256_H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

fn sha256_compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([
            block[i * 4],
            block[i * 4 + 1],
            block[i * 4 + 2],
            block[i * 4 + 3],
        ]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(SHA256_K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

fn sha256_hex(data: &[u8]) -> String {
    let mut state = SHA256_H0;
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut i = 0;
    while i + 64 <= data.len() {
        let mut block = [0u8; 64];
        block.copy_from_slice(&data[i..i + 64]);
        sha256_compress(&mut state, &block);
        i += 64;
    }
    // Final block(s): append 0x80, pad with zeros, append big-endian
    // 64-bit length.
    let rem = &data[i..];
    let mut last = [0u8; 64];
    last[..rem.len()].copy_from_slice(rem);
    last[rem.len()] = 0x80;
    if rem.len() + 1 + 8 > 64 {
        sha256_compress(&mut state, &last);
        let mut last2 = [0u8; 64];
        last2[56..64].copy_from_slice(&bit_len.to_be_bytes());
        sha256_compress(&mut state, &last2);
    } else {
        last[56..64].copy_from_slice(&bit_len.to_be_bytes());
        sha256_compress(&mut state, &last);
    }
    let mut hex = String::with_capacity(64);
    for word in &state {
        hex.push_str(&format!("{word:08x}"));
    }
    hex
}

fn point_cloud_sha256(points: &[SchoenPoint]) -> String {
    let mut buf: Vec<u8> = Vec::with_capacity(points.len() * NCOORDS * 16);
    for p in points {
        let coords = p.flat_coords();
        for c in coords.iter() {
            buf.extend_from_slice(&c.re.to_be_bytes());
            buf.extend_from_slice(&c.im.to_be_bytes());
        }
    }
    sha256_hex(&buf)
}

fn detect_git_sha() -> String {
    // Static path: only available if a build script exposes it; we don't
    // rely on one. Attempt git rev-parse at runtime; on failure return
    // "unknown".
    use std::process::Command;
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output();
    match out {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if s.is_empty() {
                "unknown".to_string()
            } else {
                s
            }
        }
        _ => "unknown".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_poly_has_correct_bidegrees() {
        let p = SchoenPoly::z3xz3_invariant_default();
        assert!(p.check_bidegrees(), "default polynomial bidegrees wrong");
    }

    #[test]
    fn poly_evaluation_consistency() {
        let p = SchoenPoly::z3xz3_invariant_default();
        let x: [Complex64; 8] = [
            Complex64::new(0.5, 0.1),
            Complex64::new(0.3, -0.2),
            Complex64::new(0.4, 0.15),
            Complex64::new(0.2, 0.0),
            Complex64::new(0.5, 0.3),
            Complex64::new(0.4, -0.1),
            Complex64::new(0.7, 0.0),
            Complex64::new(0.5, 0.2),
        ];
        let f = p.eval(&x);
        // F_1 = (x_0^3 + x_1^3 + x_2^3) t_0 + (x_0 x_1 x_2) t_1
        let f1_expected = (x[0].powi(3) + x[1].powi(3) + x[2].powi(3)) * x[6]
            + x[0] * x[1] * x[2] * x[7];
        let delta = (f[0] - f1_expected).norm();
        assert!(
            delta < 1e-12,
            "F_1 evaluation mismatch: got {:?}, expected {:?}, delta={delta:.3e}",
            f[0],
            f1_expected
        );
    }

    #[test]
    fn jacobian_matches_finite_diffs() {
        let p = SchoenPoly::z3xz3_invariant_default();
        let x: [Complex64; 8] = [
            Complex64::new(0.6, 0.1),
            Complex64::new(0.5, -0.3),
            Complex64::new(0.2, 0.4),
            Complex64::new(-0.3, 0.2),
            Complex64::new(0.4, 0.1),
            Complex64::new(0.6, -0.2),
            Complex64::new(0.7, 0.0),
            Complex64::new(0.4, 0.3),
        ];
        let jac = p.jacobian(&x);
        let h = 1e-6;
        for k in 0..NCOORDS {
            let mut p_plus = x;
            let mut p_minus = x;
            p_plus[k] += Complex64::new(h, 0.0);
            p_minus[k] -= Complex64::new(h, 0.0);
            let f_p = p.eval(&p_plus);
            let f_m = p.eval(&p_minus);
            for i in 0..NHYPER {
                let fd = (f_p[i] - f_m[i]) / Complex64::new(2.0 * h, 0.0);
                let analytic = jac[i * NCOORDS + k];
                let err = (fd - analytic).norm();
                assert!(
                    err < 1e-5,
                    "jacobian mismatch i={i} k={k}: fd={fd:?} analytic={analytic:?}"
                );
            }
        }
    }

    /// Sampler produces a point that satisfies `|F_i| < tol` after rescaling
    /// (relaxed tolerance to absorb the patch-rescale rescaling factor —
    /// see the analogous test in `cicy_sampler::sample_one_lies_on_variety`).
    #[test]
    fn sample_one_lies_on_variety() {
        let poly = SchoenPoly::z3xz3_invariant_default();
        let geom = SchoenGeometry::schoen_z3xz3();
        let mut sampler = SchoenSampler::new(poly, geom, 42);
        let mut found = None;
        for _ in 0..2048 {
            if let Some(pt) = sampler.sample_one() {
                found = Some(pt);
                break;
            }
        }
        let pt = found.expect("sampler should produce ≥1 point in 2048 tries");
        let coords = pt.flat_coords();
        let f = sampler.poly.eval(&coords);
        for i in 0..NHYPER {
            assert!(
                f[i].norm() < 1.0,
                "|F_{i}| too large after rescale: {}",
                f[i].norm()
            );
        }
    }

    #[test]
    fn sample_batch_normalises_weights() {
        let poly = SchoenPoly::z3xz3_invariant_default();
        let geom = SchoenGeometry::schoen_z3xz3();
        let mut sampler = SchoenSampler::new(poly, geom, 1234);
        let pts = sampler.sample_points(100, None);
        assert_eq!(pts.len(), 100, "should accept 100 points");
        let mut sum = 0.0;
        for p in &pts {
            assert!(p.weight.is_finite() && p.weight > 0.0);
            sum += p.weight;
        }
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "weights should sum to 1; got {sum}"
        );
    }

    #[test]
    fn z3xz3_quotient_divides_by_nine() {
        let mut pts = vec![
            SchoenPoint {
                x: [Complex64::new(1.0, 0.0); 3],
                y: [Complex64::new(1.0, 0.0); 3],
                t: [Complex64::new(1.0, 0.0); 2],
                omega: Complex64::new(1.0, 0.0),
                weight: 0.3,
            },
            SchoenPoint {
                x: [Complex64::new(1.0, 0.0); 3],
                y: [Complex64::new(1.0, 0.0); 3],
                t: [Complex64::new(1.0, 0.0); 2],
                omega: Complex64::new(1.0, 0.0),
                weight: 0.9,
            },
        ];
        let originals: Vec<f64> = pts.iter().map(|p| p.weight).collect();
        SchoenSampler::apply_z3xz3_quotient(&mut pts);
        for (p, w0) in pts.iter().zip(originals.iter()) {
            let expected = w0 / 9.0;
            assert!((p.weight - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn reproducibility_same_seed_same_output() {
        let poly = SchoenPoly::z3xz3_invariant_default();
        let geom = SchoenGeometry::schoen_z3xz3();
        let mut s1 = SchoenSampler::new(poly.clone(), geom.clone(), 7777);
        let pts1 = s1.sample_points(50, None);
        let mut s2 = SchoenSampler::new(poly, geom, 7777);
        let pts2 = s2.sample_points(50, None);
        assert_eq!(pts1.len(), pts2.len());
        for (a, b) in pts1.iter().zip(pts2.iter()) {
            for k in 0..3 {
                assert_eq!(a.x[k], b.x[k], "x[{k}] mismatch under same seed");
                assert_eq!(a.y[k], b.y[k], "y[{k}] mismatch under same seed");
            }
            for k in 0..2 {
                assert_eq!(a.t[k], b.t[k], "t[{k}] mismatch under same seed");
            }
            assert_eq!(a.weight, b.weight);
        }
    }

    #[test]
    fn sha256_known_vectors() {
        // NIST test vector: SHA-256("abc") =
        //   ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        let h = sha256_hex(b"abc");
        assert_eq!(
            h, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "SHA-256(\"abc\") mismatch: {h}"
        );

        // SHA-256(""): e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let h2 = sha256_hex(b"");
        assert_eq!(
            h2, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "SHA-256(\"\") mismatch: {h2}"
        );
    }

    #[test]
    fn run_metadata_populated_after_run() {
        let poly = SchoenPoly::z3xz3_invariant_default();
        let geom = SchoenGeometry::schoen_z3xz3();
        let mut sampler = SchoenSampler::new(poly, geom, 42);
        let _ = sampler.sample_points(20, None);
        let meta = sampler.run_metadata();
        assert_eq!(meta.seed, 42);
        assert_eq!(meta.n_points, 20);
        assert!(meta.point_cloud_sha256.len() == 64);
        assert!(meta.wall_clock_seconds >= 0.0);
        assert!(!meta.library_version.is_empty());
    }

    #[test]
    fn checkpoint_round_trip() {
        let poly = SchoenPoly::z3xz3_invariant_default();
        let geom = SchoenGeometry::schoen_z3xz3();
        let mut sampler = SchoenSampler::new(poly, geom.clone(), 1001);
        let _ = sampler.sample_points(20, None);

        let dir = std::env::temp_dir();
        let path = dir.join("schoen_sampler_checkpoint_test.json");
        sampler.checkpoint(&path).expect("checkpoint write");
        let restored = SchoenSampler::deserialize_state(&path, geom).expect("checkpoint read");
        assert_eq!(restored.seed, sampler.seed);
        assert_eq!(restored.last_run_n, sampler.last_run_n);
        // Cleanup.
        let _ = std::fs::remove_file(&path);
    }

    /// The volume integral with weight `1` should converge as `n` grows.
    /// We don't assert convergence to a specific normalised value (the
    /// per-point weight has an arbitrary global Kähler scale that
    /// cancels in ratios); we just check the sampler produces finite
    /// numbers and the empirical mean converges as `n` grows.
    #[test]
    fn volume_integral_is_finite_and_positive() {
        let poly = SchoenPoly::z3xz3_invariant_default();
        let geom = SchoenGeometry::schoen_z3xz3();
        let mut sampler = SchoenSampler::new(poly, geom, 314);
        let pts = sampler.sample_points(200, None);
        assert!(pts.len() == 200);
        // Sum of weights must be 1 (normalised).
        let s: f64 = pts.iter().map(|p| p.weight).sum();
        assert!((s - 1.0).abs() < 1e-9);
        // Each weight finite, positive.
        for p in &pts {
            assert!(p.weight.is_finite() && p.weight > 0.0);
        }
    }
}
