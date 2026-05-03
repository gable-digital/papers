//! Route 3-4: Schoen `Z/3 × Z/3` Calabi-Yau three-fold infrastructure.
//!
//! Sister module to [`crate::cicy_sampler`] / [`crate::quotient`], which
//! handle the **Tian-Yau** branch (`Z/3` quotient of a complete intersection
//! `(3,0)+(0,3)+(1,1)` in `CP^3 × CP^3`). This module supplies the
//! analogous machinery for the **Schoen fiber-product** branch:
//!
//! * [`schoen_geometry`] — topological / intersection-number data for the
//!   bidegree-`(3,3,1)` Schoen 3-fold on `CP^2 × CP^2 × CP^1`, plus
//!   `c_2(TM)` and a Bianchi-anomaly residual function.
//! * [`schoen_sampler`] — line-intersection point sampler on the Schoen
//!   variety via Newton-projection from a parametric ambient line, with
//!   per-point holomorphic-volume / pullback-metric weight; multi-threaded
//!   via rayon, optional GPU backend behind the `gpu` feature.
//! * [`z3xz3_projector`] — character-table-based projector that maps a
//!   monomial basis to its `Z/3 × Z/3`-invariant subspace.
//! * [`schoen_sampler_gpu`] (under `#[cfg(feature = "gpu")]`) — CUDA
//!   kernel for the Newton step, mirroring [`crate::gpu_sampler`].
//!
//! The naming `route34` matches the substrate-physics chapter-21 enumeration
//! (Routes 3 / 4 are the Schoen-side discrimination paths, contrasted with
//! Routes 1 / 2 which use the Tian-Yau geometry). Downstream code in the
//! η-integral pipeline imports `crate::route34::{schoen_geometry::*,
//! schoen_sampler::*, z3xz3_projector::*}`.
//!
//! ## Mathematical references
//!
//! * Schoen, "On fiber products of rational elliptic surfaces with section",
//!   *Math. Z.* **197** (1988) 177–199, DOI 10.1007/BF01215653.
//! * Donagi, He, Ovrut, Reinbacher, "The particle spectrum of heterotic
//!   compactifications", JHEP **06** (2006) 039, arXiv:hep-th/0512149,
//!   DOI 10.1088/1126-6708/2006/06/039.
//! * Braun, He, Ovrut, Pantev, "A heterotic standard model",
//!   *Phys. Lett. B* **618** (2005) 252–258, arXiv:hep-th/0501070,
//!   DOI 10.1016/j.physletb.2005.05.007.
//! * Anderson, Gray, Lukas, Palti, "Two hundred heterotic standard models on
//!   smooth Calabi-Yau threefolds", arXiv:1106.4804.
//! * Larfors, Lukas, Ruehle, Schneider (cymetric) arXiv:2111.01436 — the
//!   reference Python `pointgen_cicy` against which the sampler is
//!   cross-validated.

pub mod schoen_geometry;
pub mod schoen_sampler;
pub mod z3xz3_projector;
pub mod basis_truncation_diag;

#[cfg(feature = "gpu")]
pub mod schoen_sampler_gpu;

// ----------------------------------------------------------------------
// Route 3 η-integral pipeline support (added by η-integral agent).
//
// The four modules below build the geometric / topological / bundle
// infrastructure required to evaluate the chapter-21 η-integral:
//
//     η = | ∫_F (Tr_v(F_v²) − Tr_h(F_h²)) ∧ J | / ∫_M Tr_v(F_v²) ∧ J²
//
// where F is the Z/3-fixed (or, equivalently for free quotients, the
// Wilson-line orbifold-coupling) divisor in the post-quotient CY3 M.
//
// References:
//   - Anderson, Karp, Lukas, Palti, "Numerical Hermitian Yang-Mills
//     connections and vector bundle stability" arXiv:1004.4399 (2010).
//   - Anderson, Gray, Lukas, Palti, "Two hundred heterotic standard
//     models on smooth Calabi-Yau threefolds" arXiv:1106.4804 (2011).
//   - Donagi, He, Ovrut, Reinbacher, JHEP 06 (2006) 039,
//     DOI 10.1088/1126-6708/2006/06/039 (arXiv:hep-th/0512149).
//   - Braun, He, Ovrut, Pantev, PLB 618 (2005) 252,
//     DOI 10.1016/j.physletb.2005.05.007 (arXiv:hep-th/0501070).
// ----------------------------------------------------------------------

pub mod fixed_locus;
pub mod divisor_integration;
pub mod hidden_bundle;
pub mod chern_field_strength;
pub mod eta_evaluator;

// Derived-Chern monad-bundle parameterisation (replaces the
// dummy-vector `Candidate::bundle_moduli: Vec<f64>` in
// `pipeline.rs` with a structured `CandidateBundle` whose only
// free parameters are line-bundle degrees, with all Chern
// classes derived via the splitting principle, and a structured
// E_8 Wilson-line element with the canonical E_8 → E_6 × SU(3)
// embedding from Slansky 1981 / AGLP-2011 / BHOP-2005).
pub mod bundle_search;
pub mod wilson_line_e8;
pub mod wilson_line_e8_z3xz3;

#[cfg(feature = "gpu")]
pub mod divisor_integration_gpu;

/// CUDA-accelerated section evaluator for the BHOP-2005 §6 SU(4)
/// extension visible bundle on the Schoen Z/3 × Z/3 Calabi-Yau.
/// See [`hidden_bundle::VisibleBundle::schoen_bhop2005_su4_extension`].
#[cfg(feature = "gpu")]
pub mod hidden_bundle_gpu;

// ----------------------------------------------------------------------
// Real Donaldson-Uhlenbeck-Yau polystability check (added by the
// polystability agent).
//
// `bbw_cohomology`     — self-contained Bott-Borel-Weil + Koszul-chase
//                        line-bundle cohomology helper, independent of
//                        `crate::zero_modes`.
// `polystability`      — full DUY check via coherent sub-sheaf
//                        enumeration: sub-line-bundles via
//                        H^0(V ⊗ O(-d)), partial monad-kernel
//                        sub-bundles, and Schur-functor (∧^k V) sub-
//                        bundles up to rank `max_subsheaf_rank`.
//                        Replaces the legacy single-rank-1 check.
// `polystability_gpu`  — rayon-batched polystability sweep across
//                        many candidate bundles, bit-exact with the
//                        sequential CPU path (cohomology integers
//                        admit no round-off).
//
// References:
//   - Donaldson 1985 / Uhlenbeck-Yau 1986 — DUY theorem.
//   - Huybrechts-Lehn 2010, Geometry of Moduli Spaces of Sheaves,
//     §1.2 (slope), §4.2 (polystability).
//   - Anderson-Karp-Lukas-Palti 2010 (arXiv:1004.4399) §2.4.
//   - Anderson-Constantin-Lukas-Palti 2017 (arXiv:1707.03442).
//   - Anderson-Gray-Lukas-Palti 2011 (arXiv:1106.4804) Tabs. 3-5.
// ----------------------------------------------------------------------

pub mod bbw_cohomology;
pub mod polystability;

#[cfg(feature = "gpu")]
pub mod polystability_gpu;

#[cfg(test)]
mod tests;

// ----------------------------------------------------------------------
// Route 4 Killing-vector / continuous-isometry solver (added by Killing
// agent).
//
// Step 4 of the Arnold-ADE → wavenumber-prediction chain in chapter 21:
// given a converged Donaldson-balanced Ricci-flat metric on a CY3
// candidate, enumerate its Killing-vector algebra by solving
//
//     Δ_L ξ_ν = ∇^μ ∇_μ ξ_ν + R_νμ ξ^μ = 0
//
// (Lichnerowicz vector-Laplacian; Ricci-flat ⇒ second term vanishes).
// Submodules:
//
//   - `lichnerowicz`        — assembly of the discrete vector-Laplacian
//                             on a basis of vector fields, Christoffel
//                             symbols from the metric, multi-threaded
//                             matrix construction via rayon.
//   - `killing_solver`      — null-space extraction (subspace
//                             iteration, deflation against trivial
//                             modes), high-level API
//                             `killing_algebra_dimension`.
//   - `isometry_subgroups`  — cyclic-subgroup detection from the Lie
//                             algebra structure; consumed by the Arnold
//                             classifier (separate agent's deliverable)
//                             via `polyhedral_admissible_wavenumbers`.
//   - `lichnerowicz_gpu`    — CUDA path for the Lichnerowicz matrix
//                             assembly (`#[cfg(feature = "gpu")]`).
//
// References:
//   - Wald, "General Relativity" (1984), Ch. 3.
//   - Carroll, "Spacetime and Geometry" (2004), Ch. 3.
//   - Besse, "Einstein Manifolds" (1987), §1.K.
//   - Yau, "On the Ricci curvature of a compact Kähler manifold and
//     the complex Monge-Ampère equation, I", CMP 1978.
//   - Wang, "Hermitian forms and locally symmetric spaces",
//     J. Differential Geom. 46 (1997) 580.
//   - Anderson, Karp, Lukas, Palti, arXiv:1004.4399.
// ----------------------------------------------------------------------

pub mod lichnerowicz;
/// Real Lichnerowicz vector-Laplacian Δ_L = ∇^μ ∇_μ + R^ν_μ on
/// vector fields (Wald 1984 §3.4 eq. 3.4.4; Besse 1987 §1.K
/// eq. 1.143). On a Ricci-flat manifold the kernel coincides with
/// the kernel of [`lichnerowicz`]'s deformation-tensor form
/// (= Killing algebra by Bochner-Yano), but the non-zero spectra
/// differ by the Bochner correction (Petersen 2016 §9.3 eq. 9.3.5).
/// `route4_predictor` consumes the spectrum, not just the kernel,
/// so it must use this module.
pub mod lichnerowicz_operator;
/// Alias re-export of [`lichnerowicz`] under its more accurate name.
/// The `lichnerowicz` module is the **deformation-tensor** form
/// `L_def[ξ, η] = ∫ g g (L_ξ g)(L_η g) dvol`; `Δ_L` proper lives
/// in [`lichnerowicz_operator`].
pub mod deformation_tensor {
    pub use crate::route34::lichnerowicz::*;
}
pub mod killing_solver;
pub mod isometry_subgroups;

#[cfg(feature = "gpu")]
pub mod lichnerowicz_gpu;

// ----------------------------------------------------------------------
// Route 4 Arnold catastrophe-classifier and Rossby-polar predictor
// (added by the Arnold-Rossby agent).
//
// Steps 2 and 3 of the chapter-21 four-step prediction chain:
//   Step 1 (fluid emergence):       structural background (discharged in
//                                   `book/chapters/part3/proofs/
//                                   hyp_substrate_fluid_emergence.tex`).
//   Step 2 (Arnold ADE classify):   `arnold_normal_form` — Splitting
//                                   Lemma + Arnold's classification
//                                   theorem applied to a smooth-function
//                                   germ, returning the ADE singularity
//                                   type plus its Milnor number plus the
//                                   admissible-wavenumber set published
//                                   by Arnold-Gusein-Zade-Varchenko
//                                   1985 vol. I.
//   Step 3 (Rossby polar):          `rossby_polar` — assembles the
//                                   linearised quasi-geostrophic
//                                   Rossby-wave Lyapunov functional at
//                                   the polar critical-boundary regime
//                                   for Saturn / Jupiter using published
//                                   Cassini / Juno measurements, and
//                                   feeds it into the Arnold classifier.
//   Step 4 (Killing projection):    `route4_predictor` — combines the
//                                   Killing-spectrum input from the
//                                   Killing agent's `killing_solver`
//                                   with the Arnold classification to
//                                   produce the final wavenumber-set
//                                   prediction; computes a chi-squared
//                                   discrimination score against the
//                                   observed { Saturn n=6, Jupiter
//                                   north n=8, Jupiter south n=5 }.
//
// References (verified):
//   - Arnold, "Normal forms of functions in neighbourhoods of degenerate
//     critical points", Russian Math. Surveys 29 (1974) 10-50,
//     DOI 10.1070/RM1974v029n02ABEH002889.
//   - Arnold, Gusein-Zade, Varchenko, "Singularities of Differentiable
//     Maps", Volume I (Birkhäuser 1985), ISBN 0817632433.
//   - Poston, Stewart, "Catastrophe Theory and Its Applications"
//     (Pitman 1978), ISBN 0273010298.
//   - Sánchez-Lavega et al., "The long-term steady motion of Saturn's
//     hexagon and the stability of its enclosed jet stream under
//     seasonal changes", Geophys. Res. Lett. 41 (2014) 1425-1431,
//     DOI 10.1002/2013GL058783.
//   - Adriani et al., "Clusters of cyclones encircling Jupiter's poles",
//     Nature 555 (2018) 216-219, DOI 10.1038/nature25491.
//   - Pedlosky, "Geophysical Fluid Dynamics" (Springer 1987), §3.7.
//   - Vallis, "Atmospheric and Oceanic Fluid Dynamics" (Cambridge
//     University Press 2017, 2nd ed.), §6.4.
// ----------------------------------------------------------------------

pub mod arnold_normal_form;
pub mod rossby_polar;
pub mod route4_predictor;

#[cfg(feature = "gpu")]
pub mod arnold_normal_form_gpu;

// ----------------------------------------------------------------------
// Real Yukawa pipeline (added by the Yukawa-pipeline agent, Apr 2026).
//
// The seven modules below replace the polynomial-seed / identity-metric
// / Monte-Carlo-without-error-bars / hardcoded-SU(5)-sector / no-RG
// pipeline (zero_modes.rs / yukawa_overlap.rs / yukawa_sectors.rs in
// the legacy crate root) with a publication-grade pipeline:
//
//   - hym_hermitian:           HYM Hermitian metric h_V on the bundle V
//                              (T-operator iteration of AKLP 2010 §3).
//   - zero_modes_harmonic:     Genuine harmonic representatives of
//                              H^1(M, V ⊗ R) via twisted Dirac kernel.
//   - yukawa_overlap_real:     Triple overlap integrals with
//                              Shiffman-Zelditch quadrature, MC error
//                              bars, convergence test, and the HYM
//                              metric (not identity) entering the
//                              normalisation.
//   - yukawa_sectors_real:     E_8 → E_6 × SU(3) decomposition and
//                              dynamic sector-to-cohomology assignment
//                              (replaces the hardcoded SU(5) [0,1,2]/
//                              [3,4,5]/[6,7,8] split).
//   - rg_running:              SM 1-loop RGEs (Machacek-Vaughn 1984)
//                              from M_GUT down to M_Z.
//   - yukawa_pipeline:         End-to-end driver
//                              (metric → bundle → HYM → harmonic
//                              modes → Yukawas → RG → fermion masses
//                              + CKM).
//
// References (verified):
//   - Anderson-Karp-Lukas-Palti, "Numerical Hermitian Yang-Mills
//     connections and vector bundle stability", arXiv:1004.4399 (2010).
//   - Anderson-Constantin-Lukas-Palti, "Yukawa couplings in heterotic
//     Calabi-Yau models", arXiv:1707.03442 (2017).
//   - Donagi-He-Ovrut-Reinbacher, JHEP 06 (2006) 039,
//     arXiv:hep-th/0512149.
//   - Machacek-Vaughn, "Two loop renormalization group equations in a
//     general quantum field theory", Nucl. Phys. B 222 (1984) 83;
//     ibid. 236 (1984) 221; ibid. 249 (1985) 70.
//   - Bednyakov-Pikelner-Velizhanin, "Three-loop SM beta-functions for
//     matrix Yukawa couplings", arXiv:1303.4364 (2013).
//   - Donaldson, Proc. London Math. Soc. 50 (1985) 1.
//   - Uhlenbeck-Yau, Comm. Pure Appl. Math. 39 (1986) S257.
//   - Wang, J. Differential Geom. 70 (2005) 393.
//   - Butbaia et al., arXiv:2401.15078 (2024).
// ----------------------------------------------------------------------

pub mod hym_hermitian;
pub mod zero_modes_harmonic;
pub mod zero_modes_harmonic_z3xz3;

// Reproducibility manifest + chained-SHA replog writer shared across the
// CY3 discriminator binaries (P5.9, P5.10, P7.1).
pub mod repro;
pub mod yukawa_overlap_real;
pub mod yukawa_sectors_real;
pub mod rg_running;
pub mod yukawa_pipeline;

// P6.3 — metric-Laplacian + chain-matcher discrimination channel
// (charged-fermion chain positions on E_8 / D_8 sub-Coxeter structure;
// see book/journal/2026-04-29/2026-04-29-charged-fermion-spectrum-from-e8-sub-coxeter-structure.adoc).
pub mod metric_laplacian;
pub mod metric_laplacian_projected;
pub mod chain_matcher;

// P7.5 — H_4 (icosahedral) sub-Coxeter projection on top of the
// Z/3 × Z/3 (Schoen) / Z/3 (Tian-Yau) trivial-rep filter, for the
// ω_fix gateway-eigenvalue test on the lepton sector.
pub mod sub_coxeter_h4_projector;

#[cfg(feature = "gpu")]
pub mod zero_modes_harmonic_gpu;

#[cfg(feature = "gpu")]
pub mod yukawa_overlap_real_gpu;

// Shared types between the Arnold/Rossby/predictor modules and the
// parallel Killing-solver agent. Defined here (not in
// `isometry_subgroups`) so this module compiles independently of the
// Killing agent's deliverable; once the Killing agent lands, its
// `IsometrySubgroups::cyclic_factors()` should return `&[CyclicSubgroup]`
// using these very types.
//
// A `CyclicSubgroup` records a single cyclic factor of the candidate
// CY3's continuous-isometry group's torsion structure, parameterised
// by its order. A `ContinuousIsometryDim` records the dimension of the
// continuous part (= dim Killing algebra). Together with the ADE
// McKay-graph labelling already in `automorphism::McKayGraphKind`,
// these provide the Step-4 input that the Arnold classification must be
// projected onto.

/// A cyclic factor `Z/n` of the candidate's discrete-isometry group.
///
/// `order = n`. The trivial subgroup is `order = 1`. Generators are
/// not represented at this granularity (the Arnold classifier only
/// needs the order); for explicit Lie-algebra generators see the
/// Killing agent's `IsometrySubgroups`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CyclicSubgroup {
    pub order: u32,
}

impl CyclicSubgroup {
    pub fn new(order: u32) -> Self {
        Self {
            order: order.max(1),
        }
    }
    pub fn trivial() -> Self {
        Self { order: 1 }
    }
}

/// The Killing-spectrum result produced by the Killing-solver agent.
///
/// `continuous_isometry_dim` is the dimension of the Killing algebra
/// (= dimension of the continuous-isometry Lie group). For a generic
/// CY3 in the Yau sense this is `0`; non-generic loci with extra
/// continuous symmetry (e.g. `S^3 x S^3` polysphere ambient or
/// flat-torus degenerate metrics) have positive dimension.
///
/// `cyclic_factors` are the discrete cyclic subgroups detected.
///
/// `candidate_label` is a string tag such as `"TY/Z3"`, `"Schoen/Z3xZ3"`,
/// `"flat_T6"`, `"round_S3xS3"` — used for downstream report
/// formatting and reproducibility metadata.
#[derive(Debug, Clone)]
pub struct KillingResult {
    pub candidate_label: String,
    pub continuous_isometry_dim: u32,
    pub cyclic_factors: Vec<CyclicSubgroup>,
}

impl KillingResult {
    /// Convenience: the round-`S^3 x S^3` polysphere ambient. Continuous
    /// isometry algebra `so(4) + so(4)` of dimension 12; trivial discrete
    /// part. This is the most permissive case (admits all wavenumbers).
    pub fn polysphere_s3xs3() -> Self {
        Self {
            candidate_label: "round_S3xS3".to_string(),
            continuous_isometry_dim: 12,
            cyclic_factors: vec![CyclicSubgroup::trivial()],
        }
    }

    /// Flat `T^6`: continuous isometry `R^6` of dim 6 (translations);
    /// discrete part empty at the level we care about. Admits even
    /// wavenumbers preferentially (parity-symmetric resonance).
    pub fn flat_t6() -> Self {
        Self {
            candidate_label: "flat_T6".to_string(),
            continuous_isometry_dim: 6,
            cyclic_factors: vec![CyclicSubgroup::trivial()],
        }
    }

    /// A generic CY3 (no continuous isometry) with a single `Z/3`
    /// discrete factor — the Tian-Yau Z/3 quotient case.
    pub fn tianyau_z3() -> Self {
        Self {
            candidate_label: "TY/Z3".to_string(),
            continuous_isometry_dim: 0,
            cyclic_factors: vec![CyclicSubgroup::new(3)],
        }
    }

    /// A generic CY3 with `Z/3 x Z/3` — Schoen.
    pub fn schoen_z3xz3() -> Self {
        Self {
            candidate_label: "Schoen/Z3xZ3".to_string(),
            continuous_isometry_dim: 0,
            cyclic_factors: vec![CyclicSubgroup::new(3), CyclicSubgroup::new(3)],
        }
    }

    /// A truly featureless CY3 — generic Yau-theorem manifold with no
    /// non-trivial isometries at all. Only Arnold classification
    /// constrains the wavenumbers.
    pub fn generic_no_isometry() -> Self {
        Self {
            candidate_label: "generic_no_isometry".to_string(),
            continuous_isometry_dim: 0,
            cyclic_factors: Vec::new(),
        }
    }
}

// ----------------------------------------------------------------------
// Calabi-Yau metric solvers on the actual sub-varieties (Tian-Yau and
// Schoen). These replace the polysphere-ambient sigma-functional
// pipeline in `crate::refine` for downstream code that needs a
// publication-grade Donaldson-balanced metric on the CY3 itself.
//
// Each module follows the gold-standard pattern in `crate::quintic`:
// Newton-projected sample points on the variety, affine-chart tangent
// frame from the implicit-function theorem, sigma = weighted L1-MAD of
// eta = |det g_tan| / |Omega|^2 (canonical Douglas-Karp-Lukic-
// Reinbacher 2006 / Larfors-Schneider-Strominger 2020 convention).
//
// References:
//   - Donaldson, "Some numerical results in complex differential
//     geometry", Pure Appl. Math. Q. 5 (2009) 571, arXiv:math/0512625.
//   - Headrick-Wiseman, "Numerical Ricci-flat metrics on K3", Class.
//     Quantum Grav. 22 (2005) 4931, arXiv:hep-th/0506129.
//   - Anderson-Karp-Lukas-Palti arXiv:1004.4399; Anderson-Gray-Lukas-
//     Palti arXiv:1106.4804.
//   - Larfors-Schneider-Strominger arXiv:2012.04656.
//   - Donagi-He-Ovrut-Reinbacher 2006 §3 (Schoen intersection numbers).
// ----------------------------------------------------------------------

pub mod groebner;
// P-REPRO-2-fix-BC — GPU-tree-matched CPU h_pair summation. Used by
// both ty_metric and schoen_metric Donaldson T-operator construction
// to keep CPU and GPU paths bit-identical.
pub mod donaldson_h_pair_sum;
pub mod ty_metric;
pub mod schoen_metric;
pub mod cy3_metric_unified;
pub mod metric_cache;

#[cfg(feature = "gpu")]
pub mod cy3_metric_gpu;

// P7.10 — GPU σ-evaluator for Schoen + TY (NCOORDS=8). Used in
// the σ-FD-Adam refinement loop after Donaldson balancing.
pub mod cy3_sigma_gpu;

// P7.11 — GPU Donaldson T-operator for Schoen + TY (NCOORDS=8).
// Used inside the Donaldson balancing iteration; replaces the
// CPU rayon outer-product accumulator at production scale.
pub mod cy3_donaldson_gpu;

// ----------------------------------------------------------------------
// Bayesian discrimination layer (likelihood -> prior -> nested-sampling
// evidence -> Bayes factor -> verdict). Replaces the legacy "best-
// individual-candidate constraint-satisfaction ranking" with rigorous
// model selection: each candidate family's posterior moduli space is
// marginalised, the per-family evidence Z_c = int pi(theta) L(D|theta) d theta
// is computed via Skilling-2004 nested sampling, and the Bayes factor
// B = Z_TY / Z_Schoen is classified against the Jeffreys-1961 thresholds.
//
// References:
//   - Jeffreys, H. "An invariant form for the prior probability in
//     estimation problems", Proc. Roy. Soc. A 186 (1946) 453.
//   - Jeffreys, H. "Theory of Probability", 3rd ed. (Oxford 1961),
//     Appendix B (discrimination thresholds).
//   - Skilling, J. "Nested sampling for general Bayesian computation",
//     AIP Conf. Proc. 735 (2004) 395.
//   - Feroz, F.; Hobson, M.P. "Multimodal nested sampling: an efficient
//     and robust alternative to MCMC for cosmological parameter
//     estimation", MNRAS 384 (2008) 449, arXiv:0704.3704.
// ----------------------------------------------------------------------

pub mod likelihood;
pub mod prior;
pub mod nested_sampling;
pub mod bayes_factor;
// P5.8 — σ-distribution Bayes-factor formalisation (TY-vs-Schoen). Sits one
// level above `bayes_factor` (which compares nested-sampling evidences) and
// directly consumes the per-seed σ ensembles produced by P5.4 / P5.7 / P5.10.
pub mod bayes_factor_sigma;
// P8.1 — Multichannel Bayes-factor combiner. Aggregates per-channel log-BF
// values (σ from P5.8/P5.10, chain-match from P7.11, Hodge-consistency from
// P8.2, Yukawa-spectrum from P8.3) into a single posterior odds + n-σ
// equivalent under the channel-independence assumption.
pub mod bayes_factor_multichannel;
pub mod discrimination;

// ----------------------------------------------------------------------
// Route 1 / Route 2 substrate-physical evaluators (chapter 21,
// `book/chapters/part3/08-choosing-a-substrate.adoc` lines 216-231).
//
//   - `route1` — boundary-condition chi-squared from the photon-mediator
//                radial fall-off (Coulomb 1/r²), W/Z-mediator zero-mode
//                range, and gluon-class long-range strain tail.
//   - `route2` — Yukawa-coupling-determinant chi-squared from the
//                fermion-mass hierarchy + det-sign + magnitude pattern.
//
// Both consume Wave-1/Wave-4 outputs read-only and produce a scalar
// chi-squared + per-component breakdown for the
// [`crate::route34::likelihood::ChiSquaredBreakdown`] aggregator.
//
// References (verified):
//   - Particle Data Group, Workman R. L. *et al.*, "Review of Particle
//     Physics", Prog. Theor. Exp. Phys. 2022 (2022) 083C01,
//     2024 update, doi:10.1093/ptep/ptac097.
//   - Anderson J., Karp R., Lukas A., Palti E., "Numerical Hermitian-
//     Yang-Mills connections and vector bundle stability",
//     arXiv:1004.4399 (2010).
//   - Bjorken J. D., Drell S. D., Relativistic Quantum Mechanics
//     (McGraw-Hill 1964), §1 — Coulomb 1/r² as photon-propagator FT.
//   - Wilczek F., "Asymptotic Freedom: From Paradox to Paradigm",
//     Rev. Mod. Phys. 77 (2005) 857, doi:10.1103/RevModPhys.77.857.
// ----------------------------------------------------------------------

pub mod route1;
pub mod route2;

// ----------------------------------------------------------------------
// P8.2 — Hodge-number consistency discrimination channel.
//
// Counts kernel modes of the bundle Laplacian Δ_∂̄^V on the
// `Z/3 × Z/3` (Schoen) / `Z/3` (TY) trivial-rep sub-bundle, splits
// the kernel total by Hodge symmetry, and compares against the
// journal-predicted `(h^{1,1}, h^{2,1}, χ) = (3, 3, -6)` downstairs
// via a Gaussian likelihood. The per-candidate log-likelihood is the
// channel's contribution to the multi-channel Bayes factor.
//
// References:
//   - Griffiths-Harris 1978 §0.7 (Hodge ↔ harmonic forms).
//   - Anderson-Karp-Lukas-Palti 2010 (arXiv:1004.4399).
//   - Anderson-Constantin-Lukas-Palti 2017 (arXiv:1707.03442) §4.
//   - Donagi-He-Ovrut-Reinbacher 2006 (arXiv:hep-th/0512149).
// ----------------------------------------------------------------------
pub mod hodge_channel;
