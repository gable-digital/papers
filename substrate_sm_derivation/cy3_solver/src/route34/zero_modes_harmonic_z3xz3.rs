//! P7.6 — Bundle-twisted Laplacian on the **Z/3 × Z/3 Wilson-line +
//! H_4 (icosahedral Z/5)** projected polynomial-seed basis.
//!
//! Companion to [`crate::route34::zero_modes_harmonic`] (the canonical
//! single-Wilson-line / E_6 × SU(3) bundle Laplacian) and
//! [`crate::route34::sub_coxeter_h4_projector`] (the H_4 Coxeter
//! sector projection on the **scalar** metric Laplacian). This module
//! combines the two:
//!
//! 1. Each polynomial-seed monomial `s_α = z^I w^J` (or
//!    `x^I y^J t^K` on Schoen) is tagged by its parent B-summand
//!    `b_line` and the Z/3 × Z/3 fiber character
//!    `(g_1, g_2) ∈ {0,1,2}²` from
//!    [`crate::route34::wilson_line_e8_z3xz3::Z3xZ3WilsonLines`].
//!
//! 2. We **filter** the seeds to the Z/3 × Z/3 trivial-rep sub-bundle:
//!    the combined base + fiber character `(α_base + g_1, β_base + g_2)`
//!    must be `(0, 0)`. These are the seeds in the modded-out bundle
//!    `V / (Z/3 × Z/3)` prescribed by the journal §F.1.5 / §F.1.6.
//!
//! 3. Optionally we additionally filter on the H_4 (icosahedral Z/5)
//!    character `χ_5(s_α) = 0` (Klein 1884; matches the formula in
//!    [`crate::route34::sub_coxeter_h4_projector::icosa_z5_character`]).
//!
//! 4. Build the bundle-twisted Bochner Laplacian
//!    `L_{αβ} = Σ_i ⟨∂_{z_i} s_α, ∂_{z_i} s_β⟩_{h_V, ω}` on the
//!    filtered basis with the HYM metric `h_V`, exactly as in
//!    [`crate::route34::zero_modes_harmonic::solve_harmonic_zero_modes`]
//!    but on the smaller sub-bundle.
//!
//! 5. Solve the generalised eigenproblem `L v = λ G v` and report the
//!    full spectrum (kernel + above-kernel eigenvalues).
//!
//! ## Design rationale
//!
//! Why not modify `zero_modes_harmonic.rs` directly? Two reasons:
//!
//! 1. The single-Wilson-line spectrum is consumed by the production
//!    Yukawa pipeline (P5.x bayes_discriminate) — adding optional
//!    arguments to the public `solve_harmonic_zero_modes` would change
//!    its API surface and risk breaking the AKLP-on-TY/Z3 control.
//! 2. The Z/3 × Z/3 filter changes the seed basis dimension (drops it
//!    by ~1/9 on Schoen, ~1/3 on TY), so the kernel-dimension and
//!    cohomology-prediction logic in the canonical solver doesn't
//!    apply directly. The P7.6 spectrum is interpreted differently
//!    (as the gateway-mode test, not as a Yukawa cohomology basis).
//!
//! The duplication between this module and `zero_modes_harmonic.rs`
//! is intentional, mirrors the
//! [`crate::route34::metric_laplacian_projected`] /
//! [`crate::route34::sub_coxeter_h4_projector`] pattern, and keeps
//! P7.6's mathematical contract independent.
//!
//! ## References
//!
//! * Anderson, Karp, Lukas, Palti, arXiv:1004.4399 §5
//!   (polynomial-seed bundle Laplacian).
//! * Donagi, He, Ovrut, Reinbacher, JHEP **06** (2006) 039,
//!   arXiv:hep-th/0512149 §3 (Z/3 × Z/3 Wilson-line breaking on
//!   Schoen).
//! * Klein, *Vorlesungen über das Ikosaeder* (Teubner 1884) §I.7
//!   (icosahedral Z/5 weights).
//!
//! ## OOM-protection cap on `seed_max_total_degree`
//!
//! The cumulative bigraded enumeration in [`expanded_seed_basis`]
//! grows roughly as `O(cap^4)` because it iterates over the full
//! Cartesian product of `(d_z, d_w)` bidegrees with each component
//! contributing a 4-variable monomial count of `binom(d+3, 3)`.
//! Empirically (canonical AKLP bundle, 6 b_lines):
//!   * `cap = 1` → 24 modes
//!   * `cap = 2` → ~600 modes
//!   * `cap = 3` → ~6 000 modes
//!   * `cap = 4` → ~30 000 modes
//!   * `cap = 5` → ~94 500 modes
//!   * `cap = 6` → ~263 000 modes
//!   * `cap = 7` → ~640 000 modes (Gram + Laplacian dense storage at
//!     this size is ~6 TiB of `Complex64` — pre-filter — and would
//!     OOM any reasonable host).
//!
//! P8.6 hostile review flagged the absence of an upper bound as an
//! OOM-vulnerability. We enforce a hard cap of
//! [`Z3XZ3_MAX_TOTAL_DEGREE`] (= 6) at solver-entry and via the
//! [`Z3xZ3BundleConfig::validate`] method. All in-tree callers
//! currently use `cap ≤ 4`; the cap leaves headroom for legitimate
//! refinement studies while making accidental cap=8/10 settings
//! impossible.

#![allow(clippy::needless_range_loop)]

use crate::route34::hym_hermitian::{HymHermitianMetric, MetricBackground};
use crate::route34::sub_coxeter_h4_projector::{
    icosa_z5_character, icosa_z5_character_ty,
};
use crate::route34::wilson_line_e8_z3xz3::{
    base_alpha_character_schoen, base_alpha_character_ty, base_beta_character_schoen,
    Z3xZ3WilsonLines,
};
use crate::zero_modes::MonadBundle;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------
// OOM-protection cap (P8.6-followup-E).
// ---------------------------------------------------------------------

/// Hard upper bound on [`Z3xZ3BundleConfig::seed_max_total_degree`].
///
/// The cumulative bigraded enumeration in [`expanded_seed_basis`]
/// grows roughly as `O(cap^4)`; at `cap = 6` the unfiltered seed
/// basis is already ~263 000 modes (Gram + Laplacian dense storage
/// ~1.1 TiB of `Complex64`). At `cap ≥ 7` the host would OOM.
///
/// All in-tree callers use `cap ≤ 4`. This cap leaves a 2× headroom
/// for legitimate refinement studies while preventing accidental
/// catastrophic settings.
pub const Z3XZ3_MAX_TOTAL_DEGREE: usize = 6;

// ---------------------------------------------------------------------
// Geometry / projection enum.
// ---------------------------------------------------------------------

/// Which Calabi-Yau geometry is in use. Determines the per-coordinate
/// Z/3 (and Z/5) character formula on polynomial-seed monomials.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Z3xZ3Geometry {
    /// Schoen `Z/3 × Z/3` Calabi-Yau on `CP^2 × CP^2 × CP^1`.
    /// Sample-point layout: `[x_0, x_1, x_2, y_0, y_1, y_2, t_0, t_1]`.
    Schoen,
    /// Tian-Yau `Z/3` Calabi-Yau on `CP^3 × CP^3`.
    /// Sample-point layout: `[z_0, z_1, z_2, z_3, w_0, w_1, w_2, w_3]`.
    /// Only the **first** Z/3 is non-trivial on TY (the second factor
    /// is the identity); the second-coweight Wilson line on TY is a
    /// no-op (TY has no Z/3 × Z/3 free quotient — its quotient group
    /// is just Z/3). For TY we therefore filter only on the first
    /// character + Z/5.
    TianYau,
}

// ---------------------------------------------------------------------
// Expanded polynomial-seed basis (duplicated from
// zero_modes_harmonic.rs with the addition of fiber-character tags).
// ---------------------------------------------------------------------

#[derive(Clone, Debug)]
struct ExpandedSeedZ3 {
    /// Index into `bundle.b_lines`.
    b_line: usize,
    /// Bigraded monomial exponent vector
    /// `[e_0, e_1, e_2, e_3, e_4, e_5, e_6, e_7]`.
    exponents: [u32; 8],
}

/// Enumerate all 4-variable monomials of total degree `d`.
fn monomials_of_degree_4(d: u32) -> Vec<[u32; 4]> {
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

/// Build the full expanded polynomial-seed basis (no Z/3 × Z/3 / Z/5
/// filter applied yet). One element per bigraded monomial in every
/// non-negative B-summand.
///
/// `max_total_degree` (P-INFRA Fix 2 / P7.8b): caller-controlled
/// per-CP² block degree cap.
///   * `0` or `1` reproduces the P7.6 locked-basis behaviour
///     (`degree per block = b_lines[i][·]` → 24 modes total on the
///     canonical AKLP bundle). Single bidegree per block, no
///     cumulative enumeration.
///   * `>= 2` enumerates the **cumulative** bigraded basis: every
///     bidegree `(d_z, d_w)` with `b[0] ≤ d_z ≤ max(b[0], cap)` and
///     `b[1] ≤ d_w ≤ max(b[1], cap)`. Seeds are then sorted by
///     `(td_first_appears, b_line, lex_exponents)` where
///     `td_first_appears = max(d_z, d_w)` is the smallest cap value
///     at which the seed first enters the basis.
///
///     This canonical ordering is the **P7.8b hierarchical
///     orthogonalization** invariant: the basis at cap=k is exactly
///     the prefix of the basis at cap=k+1 where `td_first ≤ k`. When
///     fed into modified Gram-Schmidt with deflation in this order,
///     the orthonormal basis at cap=k is a true L²-subspace of the
///     orthonormal basis at cap=k+1. By the Courant-Fischer min-max
///     principle this guarantees the lowest `n` eigenvalues are
///     monotone non-increasing in cap — the Galerkin refinement
///     property required for P7.11 chain-match convergence.
fn expanded_seed_basis(bundle: &MonadBundle, max_total_degree: usize) -> Vec<ExpandedSeedZ3> {
    let mut seeds = Vec::new();
    let cap = max_total_degree as u32;
    // Only override the canonical b_line degree when the caller
    // explicitly requests expansion (cap >= 2). cap <= 1 reproduces
    // the legacy basis exactly.
    let expand = cap >= 2;
    for (b_idx, b) in bundle.b_lines.iter().enumerate() {
        if b[0] < 0 || b[1] < 0 {
            continue;
        }
        let b_z = b[0] as u32;
        let b_w = b[1] as u32;
        if expand {
            // P7.8b — cumulative enumeration. For each bidegree
            // `(d_z, d_w)` with `b_z ≤ d_z ≤ max(b_z, cap)` and
            // `b_w ≤ d_w ≤ max(b_w, cap)`, include all bigraded
            // monomials of that bidegree. The enumeration order here
            // is provisional; we re-sort below by
            // `(td_first_appears, b_line, exponents)`.
            let d_z_max = b_z.max(cap);
            let d_w_max = b_w.max(cap);
            for d_z in b_z..=d_z_max {
                for d_w in b_w..=d_w_max {
                    let z_monos = monomials_of_degree_4(d_z);
                    let w_monos = monomials_of_degree_4(d_w);
                    for zm in &z_monos {
                        for wm in &w_monos {
                            let mut exps = [0u32; 8];
                            exps[0..4].copy_from_slice(zm);
                            exps[4..8].copy_from_slice(wm);
                            seeds.push(ExpandedSeedZ3 {
                                b_line: b_idx,
                                exponents: exps,
                            });
                        }
                    }
                }
            }
        } else {
            // Legacy single-bidegree path (cap ≤ 1).
            let z_monos = monomials_of_degree_4(b_z);
            let w_monos = monomials_of_degree_4(b_w);
            for zm in &z_monos {
                for wm in &w_monos {
                    let mut exps = [0u32; 8];
                    exps[0..4].copy_from_slice(zm);
                    exps[4..8].copy_from_slice(wm);
                    seeds.push(ExpandedSeedZ3 {
                        b_line: b_idx,
                        exponents: exps,
                    });
                }
            }
        }
    }

    // P7.8b — canonical hierarchical ordering. Sort by
    //   (td_first_appears, b_line, exponents)
    // where `td_first_appears` is the smallest `cap` value at which
    // the seed first enters the cumulative basis. Concretely:
    //   td_first(seed) = max(deg_z(seed), deg_w(seed))
    // with `deg_z = e_0+e_1+e_2+e_3`, `deg_w = e_4+e_5+e_6+e_7`.
    //
    // Under this ordering the basis at cap=k is exactly the prefix
    // of the cap=(k+1) basis where `td_first ≤ k`. Modified
    // Gram-Schmidt with deflation in this order then produces an
    // orthonormal basis Q_k that is an L²-subspace of Q_{k+1}, which
    // is the Galerkin refinement invariant.
    seeds.sort_by(|a, b| {
        let td_a = seed_td_first_appears(a);
        let td_b = seed_td_first_appears(b);
        td_a.cmp(&td_b)
            .then_with(|| a.b_line.cmp(&b.b_line))
            .then_with(|| a.exponents.cmp(&b.exponents))
    });
    seeds
}

/// `td_first_appears` for a bigraded seed: smallest `cap` value at
/// which this seed enters the cumulative `expanded_seed_basis`. Equal
/// to `max(deg_z, deg_w)` since `expanded_seed_basis(cap)` includes
/// every bidegree `(d_z, d_w)` with `d_z, d_w ≤ max(b[i], cap)`, and
/// each component of `(d_z, d_w)` is bounded below by the
/// corresponding `b[i]`.
#[inline]
fn seed_td_first_appears(s: &ExpandedSeedZ3) -> u32 {
    let deg_z: u32 = s.exponents[0..4].iter().sum();
    let deg_w: u32 = s.exponents[4..8].iter().sum();
    deg_z.max(deg_w)
}

/// P-INFRA Fix 2 test-only accessor for [`expanded_seed_basis`]. The
/// basis itself is a private implementation detail (no
/// non-trivial public API consumes `ExpandedSeedZ3` directly), but
/// regression tests need to verify size growth as a function of
/// `seed_max_total_degree`.
#[doc(hidden)]
pub fn expanded_seed_basis_for_test(
    bundle: &MonadBundle,
    seed_max_total_degree: usize,
) -> Vec<[u32; 8]> {
    expanded_seed_basis(bundle, seed_max_total_degree)
        .into_iter()
        .map(|s| s.exponents)
        .collect()
}

/// Apply the Z/3 × Z/3 trivial-rep filter (combined base + fiber
/// character is `(0, 0)`). Optionally additionally apply the
/// icosahedral Z/5 filter `χ_5 = 0`.
fn filter_seeds_z3xz3_h4(
    seeds: &[ExpandedSeedZ3],
    wilson: &Z3xZ3WilsonLines,
    geometry: Z3xZ3Geometry,
    apply_h4: bool,
) -> Vec<ExpandedSeedZ3> {
    seeds
        .iter()
        .filter(|s| {
            let (g1, g2) = wilson.fiber_character(s.b_line);
            match geometry {
                Z3xZ3Geometry::Schoen => {
                    let combined_alpha =
                        (base_alpha_character_schoen(&s.exponents) + g1) % 3;
                    let combined_beta =
                        (base_beta_character_schoen(&s.exponents) + g2) % 3;
                    let z3xz3_ok = combined_alpha == 0 && combined_beta == 0;
                    if !z3xz3_ok {
                        return false;
                    }
                    if apply_h4 && icosa_z5_character(&s.exponents) != 0 {
                        return false;
                    }
                    true
                }
                Z3xZ3Geometry::TianYau => {
                    let combined_alpha =
                        (base_alpha_character_ty(&s.exponents) + g1) % 3;
                    if combined_alpha != 0 {
                        return false;
                    }
                    if apply_h4 && icosa_z5_character_ty(&s.exponents) != 0 {
                        return false;
                    }
                    true
                }
            }
        })
        .cloned()
        .collect()
}

// ---------------------------------------------------------------------
// Public result type.
// ---------------------------------------------------------------------

/// Result of the Z/3 × Z/3 + H_4 projected bundle Laplacian solve.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Z3xZ3BundleSpectrumResult {
    /// Geometry (Schoen / TianYau).
    pub geometry: Z3xZ3Geometry,
    /// Whether the H_4 (icosa-Z/5) filter was applied.
    pub h4_applied: bool,
    /// Full expanded seed basis dimension before any filter.
    pub full_seed_basis_dim: usize,
    /// Seed basis dimension after Z/3 × Z/3 trivial-rep filter only.
    pub z3xz3_basis_dim: usize,
    /// Seed basis dimension after Z/3 × Z/3 + (optional) H_4 filter.
    pub final_basis_dim: usize,
    /// `z3xz3_basis_dim / full_seed_basis_dim`.
    pub z3xz3_survival_fraction: f64,
    /// `final_basis_dim / full_seed_basis_dim`.
    pub final_survival_fraction: f64,
    /// Number of sample points used.
    pub n_points: usize,
    /// Full ascending eigenvalue spectrum on the filtered basis.
    pub eigenvalues_full: Vec<f64>,
    /// Wall-clock seconds for the full solve.
    pub wall_clock_seconds: f64,
    /// P7.8 — post-orthogonalization rank, i.e. the size of the
    /// orthonormal basis used for the standard EVP after modified
    /// Gram-Schmidt with deflation. Equals `final_basis_dim` when
    /// `orthogonalize_first = false`.
    pub orthogonalized_basis_dim: usize,
}

/// Configuration for the Z/3 × Z/3 + H_4 bundle Laplacian solve.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Z3xZ3BundleConfig {
    /// Geometry (Schoen or TY).
    pub geometry: Z3xZ3Geometry,
    /// Apply the H_4 (Z/5) filter on top of the Z/3 × Z/3 filter.
    pub apply_h4: bool,
    /// Hermitian-Jacobi eigensolver max sweeps.
    pub jacobi_max_sweeps: usize,
    /// Hermitian-Jacobi off-diagonal tolerance.
    pub jacobi_tol: f64,
    /// P-INFRA Fix 2 — maximum total polynomial degree (per CP²
    /// block) of the AKLP b_lines seed-basis expansion. Default `1`
    /// reproduces the P7.6 locked-basis behaviour. Larger values let
    /// the bundle Laplacian see higher harmonic modes:
    ///   * `1` → 24 modes  (degrees 0/1, P7.6 setting)
    ///   * `2` → ~150 modes
    ///   * `3` → ~600 modes
    /// The cap stacks with `bundle.b_lines[i]` — each line uses
    /// `max(bundle.b_lines[i], seed_max_total_degree)` per block.
    pub seed_max_total_degree: usize,
    /// P7.7-PROD Tikhonov regularisation strength applied to the
    /// Gram-matrix inversion `G^{-1}` in `L_eff = G^{-1} L`. The
    /// effective shift is `λ_T · ||G||_F` added to the diagonal of
    /// `G` before inversion. The legacy floor `1e-10` is enforced as
    /// a hard minimum so existing callers keep their behaviour. At
    /// `seed_max_total_degree ≥ 3` the basis condition number can
    /// span 8+ decades and a stronger shift is required to suppress
    /// negative-eigenvalue artifacts in the resulting `L_eff`.
    pub tikhonov_lambda: f64,
    /// P7.8 — orthogonalize the projected basis under the L²(M) inner
    /// product (modified Gram-Schmidt with deflation) BEFORE the
    /// Galerkin assembly. This eliminates the basis-redundancy
    /// pathology that emerges at `seed_max_total_degree ≥ 3`, where
    /// the Z/3×Z/3 + H_4 projector maps multiple distinct monomials
    /// onto near-collinear vectors and the Gram matrix becomes
    /// near-singular. Post-orthogonalization the Gram is identity by
    /// construction and the generalised EVP `K v = λ M v` reduces to
    /// the standard Hermitian EVP `K v = λ v`. Default `false`
    /// reproduces the P7.7-PROD behaviour exactly. Set to `true` for
    /// the production sweep at td ≥ 3.
    pub orthogonalize_first: bool,
    /// Numerical null-space tolerance for the modified Gram-Schmidt
    /// deflation step. A residual whose squared norm divided by the
    /// largest accepted vector's norm² is below this value is
    /// discarded. Default `1e-10` is appropriate for the L²(M)
    /// inner-product scale typical at `n_pts = 25 000`.
    pub orthogonalize_tol: f64,
}

impl Default for Z3xZ3BundleConfig {
    fn default() -> Self {
        Self {
            geometry: Z3xZ3Geometry::Schoen,
            apply_h4: true,
            jacobi_max_sweeps: 128,
            jacobi_tol: 1.0e-12,
            seed_max_total_degree: 1,
            tikhonov_lambda: 1.0e-10,
            orthogonalize_first: false,
            orthogonalize_tol: 1.0e-10,
        }
    }
}

impl Z3xZ3BundleConfig {
    /// P8.6-followup-E — OOM-protection validator.
    ///
    /// Returns `Err` if `seed_max_total_degree` exceeds
    /// [`Z3XZ3_MAX_TOTAL_DEGREE`]. The cumulative bigraded enumeration
    /// scales as `O(cap^4)`, so values above the cap can exhaust host
    /// memory before any filter is applied. See module-level docs for
    /// the empirical mode-count table.
    pub fn validate(&self) -> Result<(), String> {
        if self.seed_max_total_degree > Z3XZ3_MAX_TOTAL_DEGREE {
            return Err(format!(
                "Z3xZ3BundleConfig::seed_max_total_degree = {} exceeds \
                 hard cap Z3XZ3_MAX_TOTAL_DEGREE = {} (OOM protection; \
                 unfiltered seed basis grows as O(cap^4))",
                self.seed_max_total_degree, Z3XZ3_MAX_TOTAL_DEGREE
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Public driver.
// ---------------------------------------------------------------------

/// Solve the bundle-twisted Bochner Laplacian on the Z/3 × Z/3
/// Wilson-line + H_4 sub-bundle / sub-Coxeter sector.
pub fn solve_z3xz3_bundle_laplacian(
    bundle: &MonadBundle,
    metric: &dyn MetricBackground,
    h_v: &HymHermitianMetric,
    wilson: &Z3xZ3WilsonLines,
    config: &Z3xZ3BundleConfig,
) -> Z3xZ3BundleSpectrumResult {
    let started = std::time::Instant::now();

    // P8.6-followup-E — OOM protection. Reject configs whose seed
    // basis would exceed the host-memory budget before any filter is
    // applied. See [`Z3XZ3_MAX_TOTAL_DEGREE`].
    config
        .validate()
        .expect("Z3xZ3BundleConfig validation failed");

    let full = expanded_seed_basis(bundle, config.seed_max_total_degree);
    let full_dim = full.len();

    // Step 1: Z/3 × Z/3 only (for survival-fraction reporting).
    let z3xz3_only = filter_seeds_z3xz3_h4(&full, wilson, config.geometry, false);
    let z3xz3_dim = z3xz3_only.len();

    // Step 2: Z/3 × Z/3 + (optional) H_4.
    let final_seeds = if config.apply_h4 {
        filter_seeds_z3xz3_h4(&full, wilson, config.geometry, true)
    } else {
        z3xz3_only.clone()
    };
    let n_seeds = final_seeds.len();
    let n_pts = metric.n_points();

    let z3_survival = if full_dim > 0 {
        z3xz3_dim as f64 / full_dim as f64
    } else {
        0.0
    };
    let final_survival = if full_dim > 0 {
        n_seeds as f64 / full_dim as f64
    } else {
        0.0
    };

    if n_seeds == 0 || n_pts == 0 {
        return Z3xZ3BundleSpectrumResult {
            geometry: config.geometry,
            h4_applied: config.apply_h4,
            full_seed_basis_dim: full_dim,
            z3xz3_basis_dim: z3xz3_dim,
            final_basis_dim: n_seeds,
            z3xz3_survival_fraction: z3_survival,
            final_survival_fraction: final_survival,
            n_points: n_pts,
            eigenvalues_full: Vec::new(),
            wall_clock_seconds: started.elapsed().as_secs_f64(),
            orthogonalized_basis_dim: 0,
        };
    }

    // Evaluate seed basis at every sample point.
    let pts = metric.sample_points();
    let basis_at = evaluate_basis(&final_seeds, pts);

    let (eigvals, n_orth) = if config.orthogonalize_first {
        // P7.8 — orthogonalize the projected basis under the L²(M)
        // inner product, deflate the numerical null space, then build
        // K on the orthonormal basis (M = identity by construction)
        // and solve the standard Hermitian EVP.
        run_orthogonalized(
            &final_seeds,
            &basis_at,
            metric,
            h_v,
            config.orthogonalize_tol,
            config.jacobi_max_sweeps,
            config.jacobi_tol,
        )
    } else {
        // Build Gram and Laplacian.
        let g = build_gram_matrix(&final_seeds, &basis_at, metric, h_v);
        let l = build_laplacian_matrix(&final_seeds, metric, h_v);

        // Generalised eigenproblem L v = λ G v → standard L_eff =
        // G^{-1} L. Direct invert with Tikhonov regularisation.
        // P7.7-PROD: shift driven by `config.tikhonov_lambda` (legacy
        // floor 1e-10 enforced inside `invert_hermitian`).
        let g_inv = invert_hermitian(&g, n_seeds, config.tikhonov_lambda);
        let l_eff = matmul_complex(&g_inv, &l, n_seeds);

        // Hermitian projection (kill antihermitian roundoff from inverse).
        let mut l_h = l_eff;
        for i in 0..n_seeds {
            let z = l_h[i * n_seeds + i];
            l_h[i * n_seeds + i] = Complex64::new(z.re, 0.0);
            for j in (i + 1)..n_seeds {
                let a = l_h[i * n_seeds + j];
                let b = l_h[j * n_seeds + i].conj();
                let avg = (a + b) * 0.5;
                l_h[i * n_seeds + j] = avg;
                l_h[j * n_seeds + i] = avg.conj();
            }
        }
        let (eigvals, _evecs) =
            hermitian_jacobi_n(&l_h, n_seeds, config.jacobi_max_sweeps, config.jacobi_tol);
        (eigvals, n_seeds)
    };

    Z3xZ3BundleSpectrumResult {
        geometry: config.geometry,
        h4_applied: config.apply_h4,
        full_seed_basis_dim: full_dim,
        z3xz3_basis_dim: z3xz3_dim,
        final_basis_dim: n_seeds,
        z3xz3_survival_fraction: z3_survival,
        final_survival_fraction: final_survival,
        n_points: n_pts,
        eigenvalues_full: eigvals,
        wall_clock_seconds: started.elapsed().as_secs_f64(),
        orthogonalized_basis_dim: n_orth,
    }
}

// ---------------------------------------------------------------------
// P7.8 — modified Gram-Schmidt under L²(M) on the projected basis,
// followed by standard Hermitian EVP on the orthonormal basis.
// ---------------------------------------------------------------------

/// Run modified Gram-Schmidt with deflation on the projected basis
/// under the L²(M) inner product, then build K on the orthonormal
/// basis and solve the standard Hermitian EVP.
///
/// Returns `(ascending eigenvalues, rank)`.
fn run_orthogonalized(
    seeds: &[ExpandedSeedZ3],
    basis_at: &[Complex64],
    metric: &dyn MetricBackground,
    h_v: &HymHermitianMetric,
    tol: f64,
    jacobi_max_sweeps: usize,
    jacobi_tol: f64,
) -> (Vec<f64>, usize) {
    let n_seeds = seeds.len();
    let n_pts = metric.n_points();
    if n_seeds == 0 || n_pts == 0 {
        return (Vec::new(), 0);
    }

    // Compute the full Gram matrix once — modified Gram-Schmidt on
    // coefficient vectors only needs the Gram, not the per-point
    // basis evaluation. This avoids the O(rank · n_seeds² · n_pts)
    // recomputation of `inner_product_m` and reduces orthogonalization
    // cost to O(rank · n_seeds²).
    let g = build_gram_matrix(seeds, basis_at, metric, h_v);

    // `q[k]` is the coefficient vector of the k-th orthonormal basis
    // function. `gq[k][.]` caches G * q[k] so re-projection is fast.
    let mut q: Vec<Vec<Complex64>> = Vec::new();
    let mut gq: Vec<Vec<Complex64>> = Vec::new();
    let mut max_norm2_accepted: f64 = 0.0;

    for col in 0..n_seeds {
        // Start with the unit vector e_col.
        let mut v = vec![Complex64::new(0.0, 0.0); n_seeds];
        v[col] = Complex64::new(1.0, 0.0);

        // Subtract projections onto previously accepted q[k].
        for k in 0..q.len() {
            // <q[k], v>_G = q[k]^H · G · v = (gq[k])^H · v ... but
            // gq[k] = G q[k], and q[k]^H G v = (G q[k])^H v ONLY if G
            // is Hermitian (which it is). We want
            //     <q_k, v>_G = q_k^H G v.
            // Using gq[k] := G q[k] (a column), we have
            //     q_k^H G v = (G^H q_k)^H v = (G q_k)^H v = gq[k]^H v.
            let mut proj = Complex64::new(0.0, 0.0);
            for i in 0..n_seeds {
                proj += gq[k][i].conj() * v[i];
            }
            for i in 0..n_seeds {
                v[i] -= proj * q[k][i];
            }
        }

        // Re-orthogonalize once for stability (modified Gram-Schmidt
        // pass 2; necessary when the projector creates near-collinear
        // vectors).
        for k in 0..q.len() {
            let mut proj = Complex64::new(0.0, 0.0);
            for i in 0..n_seeds {
                proj += gq[k][i].conj() * v[i];
            }
            for i in 0..n_seeds {
                v[i] -= proj * q[k][i];
            }
        }

        // Compute G v for residual norm and caching.
        let mut gv = vec![Complex64::new(0.0, 0.0); n_seeds];
        for i in 0..n_seeds {
            let mut acc = Complex64::new(0.0, 0.0);
            for j in 0..n_seeds {
                acc += g[i * n_seeds + j] * v[j];
            }
            gv[i] = acc;
        }

        // Squared L²(M) norm: <v, v>_G = v^H G v = v^H gv.
        let mut norm2 = Complex64::new(0.0, 0.0);
        for i in 0..n_seeds {
            norm2 += v[i].conj() * gv[i];
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
        for i in 0..n_seeds {
            v[i] *= scale;
            gv[i] *= scale;
        }
        if n2 > max_norm2_accepted {
            max_norm2_accepted = n2;
        }
        q.push(v);
        gq.push(gv);
    }

    let rank = q.len();
    if rank == 0 {
        return (Vec::new(), 0);
    }

    // Build K on the original projected basis (this is exactly the
    // matrix `build_laplacian_matrix` returns), then transform to the
    // orthonormal basis via K_orth[i,j] = q[i]^H · K · q[j].
    let l = build_laplacian_matrix(seeds, metric, h_v);
    let mut k_orth = vec![Complex64::new(0.0, 0.0); rank * rank];
    let mut lq = vec![Complex64::new(0.0, 0.0); n_seeds];
    for j in 0..rank {
        for i in 0..n_seeds {
            let mut acc = Complex64::new(0.0, 0.0);
            for k in 0..n_seeds {
                acc += l[i * n_seeds + k] * q[j][k];
            }
            lq[i] = acc;
        }
        for i in 0..rank {
            let mut acc = Complex64::new(0.0, 0.0);
            for kk in 0..n_seeds {
                acc += q[i][kk].conj() * lq[kk];
            }
            k_orth[i * rank + j] = acc;
        }
    }

    // Hermitian projection.
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

    let (eigvals, _evecs) =
        hermitian_jacobi_n(&k_orth, rank, jacobi_max_sweeps, jacobi_tol);
    (eigvals, rank)
}

#[doc(hidden)]
/// Test-only handle to exercise the modified Gram-Schmidt deflation
/// directly on a synthetic Gram matrix (without going through the
/// full Donaldson + bundle-Laplacian stack).
///
/// `gram` is `n × n` row-major (Hermitian-positive-semi-definite).
/// Returns the rank (number of vectors retained) and the orthonormal
/// coefficient matrix `Q` packed row-major as `n × rank`.
pub fn modified_gram_schmidt_for_test(
    gram: &[Complex64],
    n: usize,
    tol: f64,
) -> (usize, Vec<Complex64>) {
    if n == 0 {
        return (0, Vec::new());
    }
    let mut q: Vec<Vec<Complex64>> = Vec::new();
    let mut gq: Vec<Vec<Complex64>> = Vec::new();
    let mut max_norm2_accepted: f64 = 0.0;
    for col in 0..n {
        let mut v = vec![Complex64::new(0.0, 0.0); n];
        v[col] = Complex64::new(1.0, 0.0);
        for _pass in 0..2 {
            for k in 0..q.len() {
                let mut proj = Complex64::new(0.0, 0.0);
                for i in 0..n {
                    proj += gq[k][i].conj() * v[i];
                }
                for i in 0..n {
                    v[i] -= proj * q[k][i];
                }
            }
        }
        let mut gv = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..n {
            let mut acc = Complex64::new(0.0, 0.0);
            for j in 0..n {
                acc += gram[i * n + j] * v[j];
            }
            gv[i] = acc;
        }
        let mut norm2 = Complex64::new(0.0, 0.0);
        for i in 0..n {
            norm2 += v[i].conj() * gv[i];
        }
        let n2 = norm2.re.max(0.0);
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
        for i in 0..n {
            v[i] *= scale;
            gv[i] *= scale;
        }
        if n2 > max_norm2_accepted {
            max_norm2_accepted = n2;
        }
        q.push(v);
        gq.push(gv);
    }
    let rank = q.len();
    let mut packed = vec![Complex64::new(0.0, 0.0); n * rank];
    for j in 0..rank {
        for i in 0..n {
            packed[i * rank + j] = q[j][i];
        }
    }
    (rank, packed)
}

// ---------------------------------------------------------------------
// Local helpers (duplicated from zero_modes_harmonic.rs — same
// algorithm, self-contained per the duplication-note pattern).
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

fn evaluate_basis(seeds: &[ExpandedSeedZ3], pts: &[[Complex64; 8]]) -> Vec<Complex64> {
    let n_seeds = seeds.len();
    let n_pts = pts.len();
    let mut out = vec![Complex64::new(0.0, 0.0); n_seeds * n_pts];
    for alpha in 0..n_seeds {
        let exps = seeds[alpha].exponents;
        for p in 0..n_pts {
            out[alpha * n_pts + p] = eval_monomial(&pts[p], &exps);
        }
    }
    out
}

fn build_gram_matrix(
    seeds: &[ExpandedSeedZ3],
    basis_at: &[Complex64],
    metric: &dyn MetricBackground,
    h_v: &HymHermitianMetric,
) -> Vec<Complex64> {
    let n_seeds = seeds.len();
    let n_pts = metric.n_points();
    let mut g = vec![Complex64::new(0.0, 0.0); n_seeds * n_seeds];

    let h_dim = h_v.n.max(1);
    let mut h_pair = vec![Complex64::new(0.0, 0.0); n_seeds * n_seeds];
    for alpha in 0..n_seeds {
        let ba = seeds[alpha].b_line.min(h_dim - 1);
        for beta in 0..n_seeds {
            let bb = seeds[beta].b_line.min(h_dim - 1);
            h_pair[alpha * n_seeds + beta] = h_v.entry(ba, bb);
        }
    }

    let mut total_w = 0.0_f64;
    for p in 0..n_pts {
        let w = metric.weight(p);
        if !w.is_finite() || w <= 0.0 {
            continue;
        }
        let omega = metric.omega(p);
        if !omega.re.is_finite() || !omega.im.is_finite() {
            continue;
        }
        let pweight = Complex64::new(omega.norm_sqr() * w, 0.0);
        for alpha in 0..n_seeds {
            let sa = basis_at[alpha * n_pts + p];
            if sa.norm_sqr() == 0.0 {
                continue;
            }
            let sa_c = sa.conj();
            for beta in 0..n_seeds {
                let sb = basis_at[beta * n_pts + p];
                let h_ab = h_pair[alpha * n_seeds + beta];
                let inc = sa_c * h_ab * sb * pweight;
                g[alpha * n_seeds + beta] += inc;
            }
        }
        total_w += w;
    }
    if total_w > 0.0 {
        let inv = 1.0 / total_w;
        for z in g.iter_mut() {
            *z *= Complex64::new(inv, 0.0);
        }
    }
    // Hermitian projection.
    for i in 0..n_seeds {
        let z = g[i * n_seeds + i];
        g[i * n_seeds + i] = Complex64::new(z.re, 0.0);
        for j in (i + 1)..n_seeds {
            let a = g[i * n_seeds + j];
            let b = g[j * n_seeds + i].conj();
            let avg = (a + b) * 0.5;
            g[i * n_seeds + j] = avg;
            g[j * n_seeds + i] = avg.conj();
        }
    }
    g
}

fn build_laplacian_matrix(
    seeds: &[ExpandedSeedZ3],
    metric: &dyn MetricBackground,
    h_v: &HymHermitianMetric,
) -> Vec<Complex64> {
    let n_seeds = seeds.len();
    let n_pts = metric.n_points();
    let pts = metric.sample_points();
    let mut l = vec![Complex64::new(0.0, 0.0); n_seeds * n_seeds];
    if n_seeds == 0 || n_pts == 0 {
        return l;
    }

    let mut d_seeds: Vec<(f64, [u32; 8])> = Vec::with_capacity(n_seeds * 8);
    for alpha in 0..n_seeds {
        for i in 0..8 {
            d_seeds.push(d_dz(&seeds[alpha].exponents, i));
        }
    }

    let mut dvals = vec![Complex64::new(0.0, 0.0); n_seeds * 8 * n_pts];
    for alpha in 0..n_seeds {
        for i in 0..8 {
            let (coeff, exps) = d_seeds[alpha * 8 + i];
            if coeff == 0.0 {
                continue;
            }
            let cz = Complex64::new(coeff, 0.0);
            for p in 0..n_pts {
                let v = eval_monomial(&pts[p], &exps);
                dvals[(alpha * 8 + i) * n_pts + p] = cz * v;
            }
        }
    }

    let h_dim = h_v.n.max(1);
    let mut h_pair = vec![Complex64::new(0.0, 0.0); n_seeds * n_seeds];
    for alpha in 0..n_seeds {
        let ba = seeds[alpha].b_line.min(h_dim - 1);
        for beta in 0..n_seeds {
            let bb = seeds[beta].b_line.min(h_dim - 1);
            h_pair[alpha * n_seeds + beta] = h_v.entry(ba, bb);
        }
    }

    let mut total_w = 0.0_f64;
    for p in 0..n_pts {
        let w = metric.weight(p);
        if !w.is_finite() || w <= 0.0 {
            continue;
        }
        let omega = metric.omega(p);
        if !omega.re.is_finite() || !omega.im.is_finite() {
            continue;
        }
        let pweight = Complex64::new(omega.norm_sqr() * w, 0.0);
        for i in 0..8 {
            for alpha in 0..n_seeds {
                let da = dvals[(alpha * 8 + i) * n_pts + p];
                if da.norm_sqr() == 0.0 {
                    continue;
                }
                let da_c = da.conj();
                for beta in 0..n_seeds {
                    let db = dvals[(beta * 8 + i) * n_pts + p];
                    if db.norm_sqr() == 0.0 {
                        continue;
                    }
                    let h_ab = h_pair[alpha * n_seeds + beta];
                    l[alpha * n_seeds + beta] += da_c * h_ab * db * pweight;
                }
            }
        }
        total_w += w;
    }
    if total_w > 0.0 {
        let inv = 1.0 / total_w;
        for z in l.iter_mut() {
            *z *= Complex64::new(inv, 0.0);
        }
    }
    for i in 0..n_seeds {
        let z = l[i * n_seeds + i];
        l[i * n_seeds + i] = Complex64::new(z.re.max(0.0), 0.0);
        for j in (i + 1)..n_seeds {
            let a = l[i * n_seeds + j];
            let b = l[j * n_seeds + i].conj();
            let avg = (a + b) * 0.5;
            l[i * n_seeds + j] = avg;
            l[j * n_seeds + i] = avg.conj();
        }
    }
    l
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

fn invert_hermitian(h: &[Complex64], n: usize, tikhonov_lambda: f64) -> Vec<Complex64> {
    if n == 0 {
        return Vec::new();
    }
    let frob = h.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    // P7.7-PROD: Tikhonov shift = max(λ_T · ||G||_F, 1e-10·||G||_F, 1e-12).
    // Legacy `1e-10 * ||G||_F` floor is preserved so default behaviour is
    // unchanged when callers don't override.
    let lam = tikhonov_lambda.max(1.0e-10);
    let eps = (frob * lam).max(1.0e-12);

    let two_n = 2 * n;
    let mut a = vec![0.0_f64; two_n * two_n];
    for i in 0..n {
        for j in 0..n {
            let z = h[i * n + j]
                + if i == j {
                    Complex64::new(eps, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
            a[i * two_n + j] = z.re;
            a[i * two_n + (n + j)] = -z.im;
            a[(n + i) * two_n + j] = z.im;
            a[(n + i) * two_n + (n + j)] = z.re;
        }
    }
    let mut aug = vec![0.0_f64; two_n * (2 * two_n)];
    for i in 0..two_n {
        for j in 0..two_n {
            aug[i * (2 * two_n) + j] = a[i * two_n + j];
        }
        aug[i * (2 * two_n) + (two_n + i)] = 1.0;
    }
    for col in 0..two_n {
        let mut pivot_row = col;
        let mut pivot_abs = aug[col * (2 * two_n) + col].abs();
        for r in (col + 1)..two_n {
            let v = aug[r * (2 * two_n) + col].abs();
            if v > pivot_abs {
                pivot_abs = v;
                pivot_row = r;
            }
        }
        if pivot_abs < 1.0e-30 {
            let mut out = vec![Complex64::new(0.0, 0.0); n * n];
            for i in 0..n {
                out[i * n + i] = Complex64::new(1.0, 0.0);
            }
            return out;
        }
        if pivot_row != col {
            for j in 0..(2 * two_n) {
                aug.swap(col * (2 * two_n) + j, pivot_row * (2 * two_n) + j);
            }
        }
        let pv = aug[col * (2 * two_n) + col];
        let inv = 1.0 / pv;
        for j in 0..(2 * two_n) {
            aug[col * (2 * two_n) + j] *= inv;
        }
        for r in 0..two_n {
            if r == col {
                continue;
            }
            let factor = aug[r * (2 * two_n) + col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..(2 * two_n) {
                aug[r * (2 * two_n) + j] -= factor * aug[col * (2 * two_n) + j];
            }
        }
    }
    let mut a_inv = vec![0.0_f64; two_n * two_n];
    for i in 0..two_n {
        for j in 0..two_n {
            a_inv[i * two_n + j] = aug[i * (2 * two_n) + two_n + j];
        }
    }
    let mut out = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let re = a_inv[i * two_n + j];
            let im = a_inv[(n + i) * two_n + j];
            out[i * n + j] = Complex64::new(re, im);
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
        let mut off = 0.0_f64;
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
        for p in 0..(n.saturating_sub(1)) {
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
    eig.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut eig_sorted = vec![0.0_f64; n];
    let mut v_sorted = vec![Complex64::new(0.0, 0.0); n * n];
    for (new_k, (old_k, val)) in eig.iter().enumerate() {
        eig_sorted[new_k] = *val;
        for i in 0..n {
            v_sorted[i * n + new_k] = v[i * n + *old_k];
        }
    }
    (eig_sorted, v_sorted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aklp_full_seed_basis_dim_24() {
        let bundle = MonadBundle::anderson_lukas_palti_example();
        let full = expanded_seed_basis(&bundle, 1);
        // Same count as zero_modes_harmonic: 6 b_lines × 4 monomials each
        // (3 z-monomials × 1 w-monomial for the (1,0) triple, and
        // 1 z-monomial × 3 w-monomials for the (0,1) triple, but each
        // monomial-of-degree-d-in-4-vars at d=1 gives 4 monomials,
        // not 3, so 6 × 4 = 24).
        assert_eq!(full.len(), 24, "AKLP expanded basis dim");
    }

    #[test]
    fn z3xz3_filter_reduces_basis_schoen() {
        let bundle = MonadBundle::anderson_lukas_palti_example();
        let wilson = Z3xZ3WilsonLines::canonical_aklp_schoen();
        let full = expanded_seed_basis(&bundle, 1);
        let filtered = filter_seeds_z3xz3_h4(&full, &wilson, Z3xZ3Geometry::Schoen, false);
        assert!(
            filtered.len() < full.len(),
            "Z/3 × Z/3 filter must shrink the basis (got {} from {})",
            filtered.len(),
            full.len()
        );
    }

    #[test]
    fn validate_seed_max_total_degree_cap() {
        // P8.6-followup-E — `validate()` must accept any value at or
        // below `Z3XZ3_MAX_TOTAL_DEGREE` and reject anything above it.
        let mut cfg = Z3xZ3BundleConfig::default();

        // Production-relevant value (matches existing test suite).
        cfg.seed_max_total_degree = 5;
        assert!(
            cfg.validate().is_ok(),
            "cap=5 must be accepted (current ceiling of in-tree usage)"
        );

        // Boundary: exactly at the cap is valid.
        cfg.seed_max_total_degree = Z3XZ3_MAX_TOTAL_DEGREE;
        assert!(
            cfg.validate().is_ok(),
            "cap == Z3XZ3_MAX_TOTAL_DEGREE must be accepted"
        );

        // Boundary: one above the cap must be rejected.
        cfg.seed_max_total_degree = Z3XZ3_MAX_TOTAL_DEGREE + 1;
        let err = cfg.validate().expect_err(
            "cap > Z3XZ3_MAX_TOTAL_DEGREE must be rejected (OOM protection)",
        );
        assert!(
            err.contains("Z3XZ3_MAX_TOTAL_DEGREE"),
            "error message must reference the cap constant: {err}"
        );

        // Sanity: comically high value is also rejected.
        cfg.seed_max_total_degree = 20;
        assert!(cfg.validate().is_err(), "cap=20 must be rejected");
    }

    #[test]
    fn h4_filter_further_reduces_basis() {
        let bundle = MonadBundle::anderson_lukas_palti_example();
        let wilson = Z3xZ3WilsonLines::canonical_aklp_schoen();
        let full = expanded_seed_basis(&bundle, 1);
        let z3 = filter_seeds_z3xz3_h4(&full, &wilson, Z3xZ3Geometry::Schoen, false);
        let z3_h4 = filter_seeds_z3xz3_h4(&full, &wilson, Z3xZ3Geometry::Schoen, true);
        assert!(
            z3_h4.len() <= z3.len(),
            "H_4 filter must not enlarge basis"
        );
    }
}
