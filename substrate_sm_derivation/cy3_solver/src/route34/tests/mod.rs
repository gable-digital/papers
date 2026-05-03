//! Route-3 integration tests combining `fixed_locus`,
//! `divisor_integration`, `hidden_bundle`, and `chern_field_strength`
//! into a single end-to-end η-integrand evaluation.
//!
//! These run only under `cargo test`.

#[cfg(test)]
pub mod test_eta_integrand;

// Route-4 integration tests (Arnold catastrophe classifier + Rossby-
// polar Lyapunov germ + Route-4 discrimination predictor).
#[cfg(test)]
pub mod test_arnold;
#[cfg(test)]
pub mod test_rossby;
#[cfg(test)]
pub mod test_route4;

// Schoen-side sampler / projector / geometry integration tests.
#[cfg(test)]
pub mod test_schoen_geometry;
#[cfg(test)]
pub mod test_schoen_sampler;
#[cfg(test)]
pub mod test_z3xz3_projector;
#[cfg(test)]
pub mod test_schoen_integration;

// Killing-vector / Lichnerowicz solver tests (round S^n, flat T^n,
// product spheres, generic CY3).
#[cfg(test)]
pub mod test_lichnerowicz;
#[cfg(test)]
pub mod test_killing_solver;
#[cfg(test)]
pub mod test_isometry_subgroups;

// End-to-end η-integral evaluator tests (Wave-2 integration).
#[cfg(test)]
pub mod test_eta_evaluator;

// Real Donaldson-Uhlenbeck-Yau polystability check tests
// (replaces the legacy single-rank-1 check). See
// `route34::polystability` and `route34::bbw_cohomology`.
#[cfg(test)]
pub mod test_polystability;

// Buchberger / Gröbner-basis tests.
#[cfg(test)]
pub mod test_groebner;

// Derived-Chern monad-bundle parameterisation tests
// (`bundle_search` + `wilson_line_e8`).
#[cfg(test)]
pub mod test_bundle_search;
#[cfg(test)]
pub mod test_wilson_line_e8;
#[cfg(test)]
pub mod test_bundle_integration;

// §5.4 — first σ values on the Tian-Yau (TY/Z3) physics candidate.
// Up to this wave every published σ was on the Fermat quintic test case;
// these tests demonstrate the σ pipeline runs on the actual TY/Z3 CY3.
#[cfg(test)]
pub mod test_p5_4_ty_sigma;

// §5.7 — first TY-vs-Schoen σ-discrimination result. Schoen baseline
// + 20-seed multi-candidate ensemble + n-σ discrimination summary.
#[cfg(test)]
pub mod test_p5_7_ty_schoen_discrimination;

// §5.10 — TY-vs-Schoen σ-discrimination at higher n_pts (5σ target).
// Path A: n_pts boost from P5.7's 10 000 to 25 000, k=3, 20 seeds.
#[cfg(test)]
pub mod test_p5_10_ty_schoen_5sigma;

// P-INFRA Fix 1 — `Cy3MetricResultBackground::from_schoen` must plumb
// the Donaldson-balanced metric so that the bundle Laplacian sees a
// k-dependent metric. Pre-fix the entire ω_fix sweep (P7.7) was
// bit-identical across k=3,4,5; post-fix the metric varies with k.
#[cfg(test)]
pub mod test_metric_background_k_dependence;

// P-INFRA Fix 2 — `Z3xZ3BundleConfig::seed_max_total_degree` must
// expand the AKLP b_lines polynomial seed basis at higher degrees.
// Pre-fix the basis was locked at 24 modes (degrees 0/1) regardless
// of caller intent.
#[cfg(test)]
pub mod test_z3xz3_basis_growth;

// P-INFRA Fix 3 — the closest-to-ω_fix picker must NEVER select the
// `by_sigmoid` normalisation. Pre-fix sigmoid saturated to 0.5 ≈
// 123/248 = 0.4960 and produced false ~0.81% matches.
#[cfg(test)]
pub mod test_no_sigmoid_in_picker;

// P7.8 — orthogonalize the Z/3 × Z/3 + H_4 projected basis under the
// L²(M) inner product BEFORE the Galerkin solve, to eliminate the
// basis-redundancy bottleneck that produces negative eigenvalues at
// td ≥ 3.
#[cfg(test)]
pub mod test_z3xz3_orthogonal_basis;

// P7.8b — Galerkin subspace-inclusion regression. Hierarchical
// orthogonalization via canonical (td_first_appears, b_line, lex)
// ordering must produce Q_k ⊆ Q_{k+1}, restoring the Courant-Fischer
// min-max guarantee (lowest n eigenvalues monotone non-increasing in
// cap). Pre-fix per-cap deflation broke this on production Schoen
// data: td=3 lowest 5 eigenvalues were strictly larger than td=2.
#[cfg(test)]
pub mod test_galerkin_subspace_inclusion;

// P7.10 — CPU↔GPU parity for σ-evaluator (NCOORDS=8, Schoen + TY).
#[cfg(test)]
pub mod test_cy3_sigma_gpu_parity;

// P7.11 — CPU↔GPU parity for Donaldson T-operator (NCOORDS=8, Schoen + TY).
#[cfg(test)]
pub mod test_cy3_donaldson_gpu_parity;

// P8.2 — Hodge-number consistency discrimination channel
// (kernel-count of bundle Laplacian Δ_∂̄^V on Z/3 × Z/3 trivial-rep
// sub-bundle vs predicted (3, 3, -6) downstairs).
#[cfg(test)]
pub mod test_hodge_channel;

// P7.12 — ω_fix as an *exact algebraic invariant* of E_8, not as a
// measurable eigenvalue. Pins the journal identity ω_fix = 1/2 - 1/dim
// = 123/248 at f64 + BigFloat(500-bit) precision, the dual-anchor
// self-consistency at ~30 ppb, and the empirical content (perturbation
// by ±1 in the numerator breaks the dual-anchor by >1000 ppm). Closes
// the P7.1-P7.10 series — those eight tests were testing an eigenvalue
// hypothesis the framework does not actually make.
#[cfg(test)]
pub mod test_omega_fix_algebraic_identity;
