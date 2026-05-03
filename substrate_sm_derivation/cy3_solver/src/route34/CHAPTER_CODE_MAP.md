# Chapter 21 ("In Search of a Substrate Topology: Lost in Space") -> Route 34 Code Map

This file links every claim, equation, hypothesis-citation, or
discrimination signal in `book/chapters/part3/08-choosing-a-substrate.adoc`
to the file/function/test in `src/route34/` that implements it. Every
entry includes the chapter line range, the route34 file:function:line, the
published reference (DOI/arXiv id) the implementation cites, and the
test that exercises it.

All file paths are absolute paths within the rust_solver crate. Line
numbers are pinned to the commit at which this map was authored;
re-run `git log -p <file>` to verify.

---

## 1. Heterotic E_8 x E_8 commitment

**Chapter range**: lines 6, 18, 52-56, 68-79 ("The Commitment", "Two
Sectors Built Into the Lie-Algebra Factors").

**Claim**: the substrate's mathematical-object structure is heterotic
`E_8 x E_8`, with the visible sector carrying the Standard-Model gauge
content via the Wilson-line breaking chain `E_8 -> E_6 -> SU(3) x SU(2) x U(1)`.

**Implementation**:

| File | Function | Line | What it does |
|------|----------|------|--------------|
| `src/route34/wilson_line_e8.rs` | `e8_roots` | 136 | Constructs the 240 root vectors of `E_8` (120 norm-2 + 120 half-integer co-set) per Bourbaki Lie Groups VI.4, used as the basis for the canonical embedding. |
| `src/route34/wilson_line_e8.rs` | `WilsonLineE8::canonical_e8_to_e6_su3` | 280 | Returns the canonical Cartan-phase vector that produces the `E_6 x SU(3)` unbroken subgroup; cites Slansky 1981 *Phys. Rep.* 79:1 Tab. 23 (DOI 10.1016/0370-1573(81)90092-2). |
| `src/route34/wilson_line_e8.rs` | `UnbrokenSubgroup::is_e6_times_su3` | 490 | Verifies (Lie-algebra dim 78 + 8 = 86) that the Wilson line breaks to exactly `E_6 x SU(3)`. |
| `src/route34/hidden_bundle.rs` | `HiddenBundle::trivial_e8` | ~95 | Returns the trivial hidden-`E_8` configuration; matches the chapter's "the hidden sector ... observable to our region only through gravitational effects" (line 95). |

**Tests**:

* `src/route34/tests/test_wilson_line_e8.rs::canonical_wilson_line_unbroken_dim_86` — asserts the unbroken-subgroup Lie dim is 86.
* `src/route34/tests/test_wilson_line_e8.rs::canonical_wilson_line_quantization_zero` — asserts the Z/3 quantization residual is zero.

**References cited in code**: Slansky 1981; Bourbaki Lie Groups VI;
Anderson-Gray-Lukas-Palti arXiv:1106.4804; Braun-He-Ovrut-Pantev
arXiv:hep-th/0501070.

---

## 2. Z/3 quotient on Tian-Yau

**Chapter range**: lines 127-141 ("Topological-Protection Requirement",
"The Z/3 Doing Double Duty"), 186 (Tian-Yau definition).

**Claim**: the Tian-Yau manifold is the freely-acting `Z/3` quotient
of a complete intersection of two cubic hypersurfaces in
`CP^3 x CP^3`; non-trivial `pi_1 = Z/3` provides topological protection
for the entanglement encoding and the three-fermion-generation count.

**Implementation**:

| File | Function | Line | What it does |
|------|----------|------|--------------|
| `src/quotient.rs` | `Z3QuotientGroup` | (legacy) | Defines the freely-acting `Z/3` action on the bicubic CICY; preserved as legacy because the post-quotient downstream pipeline reads from this module. |
| `src/route34/fixed_locus.rs` | `QuotientAction::tian_yau_z3` | 223 | The `Z/3` action on the (3,0) + (0,3) bicubic; returns the character data and the divisor representation needed to integrate the eta-form on the fixed locus. |
| `src/route34/fixed_locus.rs` | `enumerate_fixed_loci` | 377 | Enumerates the connected components of the fixed-point locus; for the freely-acting Tian-Yau Z/3 the locus is empty after quotient (the orbifold-coupling locus lives in the upstairs variety as a divisor at the Z/3 fixed-points-on-the-pre-quotient). |
| `src/route34/z3xz3_projector.rs` | (cross-validates the Z/3 part of the Schoen Z/3 x Z/3 projector at restricted bidegrees) | 541-line module | Z/3 sector of the larger projector. |

**Tests**:

* `src/route34/tests/test_z3xz3_projector.rs::beta_has_order_three` — Z/3 generator has order 3.
* `src/route34/tests/test_schoen_geometry.rs::quotient_consistency` — checks the Z/3 character table's idempotents.

**References cited in code**: Tian-Yau 1986 (the original construction);
Anderson-Gray-Lukas-Palti arXiv:1106.4804 §3.2.

---

## 3. Z/3 x Z/3 quotient on Schoen

**Chapter range**: lines 188 (Schoen definition), 137-141 (Z/3 doing
double-duty applies for either candidate).

**Claim**: Schoen's CICY is a fiber-product of two rational elliptic
surfaces with appropriate divisor structure quotiented by a freely-
acting `Z/3 x Z/3`, with `chi = -6` and `pi_1` containing `Z/3`
subgroups.

**Implementation**:

| File | Function | Line | What it does |
|------|----------|------|--------------|
| `src/route34/schoen_geometry.rs` | `SchoenCY3::canonical` | (top) | Schoen 1988 *Math. Z.* 197:177 fiber-product construction (DOI 10.1007/BF01215653) at bidegree (3,3,1) on `CP^2 x CP^2 x CP^1`. |
| `src/route34/z3xz3_projector.rs` | `Z3xZ3Projector::projection_matrix` | 541-line module | Character-table-based projector that maps a monomial basis to its `Z/3 x Z/3`-invariant subspace; the projector's order-9 idempotency is a regression test. |
| `src/route34/fixed_locus.rs` | `QuotientAction::schoen_z3xz3` | 280 | The `Z/3 x Z/3` action data on the Schoen fiber-product; returns the per-character divisor representation. |
| `src/route34/schoen_metric.rs` | `SchoenSolver::solve_metric` | (top) | Donaldson sigma-functional solver on the Schoen sub-variety; cites Donagi-He-Ovrut-Reinbacher JHEP 06 (2006) 039 (DOI 10.1088/1126-6708/2006/06/039, arXiv:hep-th/0512149) §3 for intersection numbers. |

**Tests**:

* `src/route34/tests/test_z3xz3_projector.rs::projector_order_is_9` — projector has order 9.
* `src/route34/tests/test_schoen_geometry.rs::full_pipeline_geometry_sampler_projector` — end-to-end Schoen integration.
* `src/route34/schoen_metric::tests::test_schoen_metric_converges` — Donaldson solver converges on the Schoen sub-variety.

**References cited in code**: Schoen 1988 DOI 10.1007/BF01215653;
Donagi-He-Ovrut-Reinbacher arXiv:hep-th/0512149; Braun-He-Ovrut-Pantev
arXiv:hep-th/0501070.

---

## 4. Wilson-line E_8 -> E_6 x SU(3) breaking

**Chapter range**: lines 71-72 ("the well-studied `E_8 -> E_6 -> SU(3) x SU(2) x U(1)`
embedding chain via Wilson-line breaking"), 186 (Tian-Yau Wilson-line
breaking around Z/3 generator), 198 (gauge bundle).

**Claim**: the canonical Wilson-line element `W in E_8` for a `Z/3`
quotient satisfies `W^3 = 1` and breaks `E_8 -> E_6 x SU(3)` with the
unbroken subgroup of Lie-dim 86 (= 78 for `E_6` + 8 for `SU(3)`).

**Implementation**:

| File | Function | Line | What it does |
|------|----------|------|--------------|
| `src/route34/wilson_line_e8.rs` | `WilsonLineE8::canonical_e8_to_e6_su3` | 280 | Constructs the canonical Wilson-line element with Cartan phases pinned to the Slansky 1981 Tab. 23 simple-root basis; quantization residual zero by construction. |
| `src/route34/wilson_line_e8.rs` | `WilsonLineE8::quantization_residual` | 306 | Computes `||W^Gamma - 1||` for the discrete-quotient order `Gamma`; zero iff the Cartan phases are Z/Gamma-valued. |
| `src/route34/wilson_line_e8.rs` | `WilsonLineE8::unbroken_subalgebra` | 414 | Returns the unbroken `E_6 x SU(3)` subgroup data with Lie-dim 86. |
| `src/route34/wilson_line_e8.rs` | `WilsonLineE8::embeds_for_se` | 444 | Verifies the Wilson line is compatible with a chosen line-bundle structure-group embedding into `E_8`. |

**Tests**:

* `src/route34/tests/test_wilson_line_e8.rs::canonical_wilson_line_unbroken_dim_86`
* `src/route34/tests/test_wilson_line_e8.rs::canonical_wilson_line_quantization_zero`
* `src/route34/wilson_line_e8::unit_tests::spinor_part_count` — counts the 27 + 1 + 27bar decomposition of the `E_6 x SU(3)` matter content.

**References cited in code**: Slansky 1981 Tab. 23 (DOI 10.1016/0370-1573(81)90092-2);
Anderson-Gray-Lukas-Palti arXiv:1106.4804; Braun-He-Ovrut-Pantev
arXiv:hep-th/0501070.

---

## 5. The eta integral form (Route 3)

**Chapter range**: lines 233-263 ("Pinning Down Route 3").

**Claim**: the matter-antimatter asymmetry `eta` has the substrate-
physical form

```
eta = | int_F (Tr_v(F_v^2) - Tr_h(F_h^2)) /\ J | / int_M Tr_v(F_v^2) /\ J^2
```

where `M` is the heterotic CY3, `F` is the `Z/3`-fixed divisor, `J` is
the Kahler form, `F_v` and `F_h` are the visible and hidden sector
gauge field strengths.

**Implementation**:

| File | Function | Line | What it does |
|------|----------|------|--------------|
| `src/route34/eta_evaluator.rs` | `evaluate_eta_tian_yau` | 357 | End-to-end evaluation of the eta form on the Tian-Yau Z/3 candidate, returning the predicted eta plus an MC error bar. |
| `src/route34/eta_evaluator.rs` | `evaluate_eta_schoen` | 536 | Same for the Schoen Z/3 x Z/3 candidate. |
| `src/route34/chern_field_strength.rs` | `tr_f_squared` | 101 | Computes `Tr(F^2)` for a monad bundle as a `(2,2)`-form integrand at a sample point. |
| `src/route34/chern_field_strength.rs` | `integrate_tr_f_squared_wedge_J` | 151 | Integrates `Tr(F^2) wedge J` over a divisor or the full CY3 via Shiffman-Zelditch quadrature. |
| `src/route34/chern_field_strength.rs` | `integrate_visible_minus_hidden` | 186 | The numerator of the eta form: `int_F (Tr_v(F_v^2) - Tr_h(F_h^2)) wedge J`. |
| `src/route34/divisor_integration.rs` | (full module) | top | Sample-point integration restricted to the `Z/3`-fixed divisor `F`. |
| `src/route34/fixed_locus.rs` | `enumerate_fixed_loci` | 377 | Identifies the divisor `F` as the fixed locus of the discrete quotient action. |

**Tests**:

* `src/route34/tests/test_eta_evaluator.rs` — integration tests for both Tian-Yau and Schoen eta predictions.
* `src/route34/tests/test_eta_integrand.rs` — verifies the integrand pieces (`Tr_v`, `Tr_h`, wedge product) against analytic baselines on toy bundles.
* `src/route34/tests/test_bundle_integration.rs` — checks that the integration is bundle-aware (different bundles produce different eta).

**References cited in code**:
* Anderson-Karp-Lukas-Palti arXiv:1004.4399 §2 (Chern-character integrand).
* Anderson-Gray-Lukas-Palti arXiv:1106.4804 (heterotic anomaly cancellation).
* Donagi-He-Ovrut-Reinbacher arXiv:hep-th/0512149 (orbifold-coupling locus).

The integral form is exactly the chapter's display equation at line 249.

---

## 6. Killing-vector solver via Lichnerowicz operator

**Chapter range**: lines 265-298 (Step 4 of the Arnold-ADE chain).

**Claim**: a converged Donaldson-balanced Ricci-flat metric on a CY3
candidate has its Killing-vector algebra extracted by solving the
Lichnerowicz vector-Laplacian equation `Delta_L xi_nu = 0`. On a
Ricci-flat manifold the Ricci term vanishes so this reduces to
`grad^mu grad_mu xi_nu = 0`.

**Implementation**:

| File | Function | Line | What it does |
|------|----------|------|--------------|
| `src/route34/lichnerowicz.rs` | `LichnerowiczOperator` | 327 | Discrete vector-Laplacian on a basis of vector fields, multi-threaded matrix construction via rayon, Christoffel symbols from the metric. |
| `src/route34/killing_solver.rs` | `solve_killing_kernel` | 422 | Null-space extraction (subspace iteration, deflation against trivial modes) for the Lichnerowicz operator. |
| `src/route34/killing_solver.rs` | `killing_algebra_dimension` | 524 | High-level API: returns `dim(Killing algebra)` for a candidate metric. |
| `src/route34/killing_solver.rs` | `killing_bracket_structure_constants` | 556 | Lie-bracket structure constants of the Killing algebra. |
| `src/route34/isometry_subgroups.rs` | `IsometrySubgroups::cyclic_factors` | (top) | Cyclic-subgroup detection from the Lie-algebra structure; consumed by the Arnold classifier via `polyhedral_admissible_wavenumbers`. |
| `src/route34/lichnerowicz_gpu.rs` | (full module) | top | CUDA path for the Lichnerowicz matrix assembly (gated on `feature = "gpu"`). |

**Tests**:

* `src/route34/tests/test_killing_solver.rs` — null-space extraction on synthetic Ricci-flat Lichnerowicz operators.
* `src/route34/tests/test_lichnerowicz.rs` — operator assembly correctness against analytic baseline (Ricci tensor formula, Bianchi identity).
* `src/route34/tests/test_isometry_subgroups.rs` — cyclic-subgroup detection.

**References cited in code**:
* Wald, "General Relativity" (1984), Ch. 3.
* Carroll, "Spacetime and Geometry" (2004), Ch. 3.
* Besse, "Einstein Manifolds" (1987), §1.K.
* Yau, "On the Ricci curvature of a compact Kahler manifold...", CMP 1978.

---

## 7. Arnold catastrophe-theory ADE classification

**Chapter range**: lines 273-274 (Step 2 of the Arnold-ADE chain).

**Claim**: Arnold's classification theorem (Arnold 1974,
*Russian Math. Surveys* 29:10, DOI 10.1070/RM1974v029n02ABEH002889;
Arnold-Gusein-Zade-Varchenko 1985) classifies elementary singularities
of smooth maps as exactly the simply-laced ADE Dynkin diagrams. The
seven elementary catastrophes (`A_2, A_3, A_4, A_5, D_4^pm, D_5`)
plus the higher series `A_n, D_n, E_6, E_7, E_8` exhaust the catalogue.

**Implementation**:

| File | Function | Line | What it does |
|------|----------|------|--------------|
| `src/route34/arnold_normal_form.rs` | `corank` | 527 | Computes the corank of a smooth-function germ at a critical point (= dim(ker Hessian)). |
| `src/route34/arnold_normal_form.rs` | `splitting_lemma_reduce` | 554 | Implements the Splitting Lemma: separates a germ into a Morse part plus a corank-d residual. |
| `src/route34/arnold_normal_form.rs` | `classify_singularity` | 731 | Returns the ADE type (and its Milnor number) of a smooth-function germ. Implements Arnold's classification table directly. |
| `src/route34/arnold_normal_form.rs` | `milnor_number` | 1014 | Milnor number from the germ Hessian + the Arnold table. |
| `src/route34/arnold_normal_form.rs` | `admissible_wavenumber_set` | 1041 | The published-Arnold-Gusein-Zade-Varchenko 1985 (vol. I) table mapping each ADE type to its admissible polyhedral-pattern wavenumber set. |
| `src/route34/arnold_normal_form_gpu.rs` | (full module) | top | CUDA path for the classify-singularity batch (gated on `feature = "gpu"`). |

**Tests**:

* `src/route34/tests/test_arnold.rs` — classification correctness on canonical-form singularities for `A_2, A_3, A_4, A_5, D_4^pm, D_5, E_6, E_7, E_8`.

**References cited in code**:
* Arnold 1974 *Russian Math. Surveys* 29:10 DOI 10.1070/RM1974v029n02ABEH002889.
* Arnold-Gusein-Zade-Varchenko 1985, "Singularities of Differentiable Maps" Vol. I (Birkhauser, ISBN 0817632433).
* Poston-Stewart 1978, "Catastrophe Theory and Its Applications" (Pitman, ISBN 0273010298).

---

## 8. Rossby-wave Lyapunov germ at polar critical boundary

**Chapter range**: lines 271, 275-276 (Steps 1 + 3 of the Arnold-ADE chain).

**Claim**: a rotating planet's polar atmosphere sits at a stability-
critical boundary where the linearised quasi-geostrophic Rossby-wave
dispersion relation hits one of Arnold's smooth-map ADE singularities.
The basic-state shear profile + rotation rate + substrate-amplitude
resonance constraints fix the local Arnold-ADE type.

**Implementation**:

| File | Function | Line | What it does |
|------|----------|------|--------------|
| `src/route34/rossby_polar.rs` | `published_saturn_polar` | 169 | Saturn's polar basic-state from Sanchez-Lavega et al. 2014 GRL 41:1425 (DOI 10.1002/2013GL058783) — observed wavenumber `n=6`. |
| `src/route34/rossby_polar.rs` | `published_jupiter_north_polar` | 196 | Jupiter's north polar basic-state from Adriani et al. 2018 *Nature* 555:216 (DOI 10.1038/nature25491) — observed wavenumber `n=8`. |
| `src/route34/rossby_polar.rs` | `published_jupiter_south_polar` | 220 | Jupiter's south polar basic-state from the same Adriani et al. 2018 — observed wavenumber `n=5`. |
| `src/route34/rossby_polar.rs` | `linearised_lyapunov` | 281 | Assembles the linearised Rossby-wave Lyapunov functional germ at the polar critical-boundary regime. |
| `src/route34/rossby_polar.rs` | `predict_wavenumber_set` | 408 | Combines the linearised Lyapunov germ with `arnold_normal_form::admissible_wavenumber_set` to predict the polyhedral wavenumber set. |
| `src/route34/rossby_polar.rs` | `arnold_type_cutoff` | 458 | Maximum wavenumber compatible with each ADE type. |

**Tests**:

* `src/route34/tests/test_rossby.rs` — predictions for Saturn (n=6), Jupiter north (n=8), Jupiter south (n=5) against observation.
* `src/route34/tests/test_route4.rs` — full route-4 chain (Killing -> Arnold -> Rossby) with chi-squared discrimination.

**References cited in code**:
* Sanchez-Lavega et al. 2014 GRL 41:1425 DOI 10.1002/2013GL058783.
* Adriani et al. 2018 *Nature* 555:216 DOI 10.1038/nature25491.
* Pedlosky, "Geophysical Fluid Dynamics" (Springer 1987), §3.7.
* Vallis, "Atmospheric and Oceanic Fluid Dynamics" 2nd ed. (CUP 2017), §6.4.

---

## 9. Calabi-Yau metric on the actual sub-variety

**Chapter range**: lines 199-211 ("Bottom Line and Next Step", part of
the eigenvalue-computation pipeline step 1 at line 196).

**Claim**: a publication-grade fermion-mass extraction requires the
Calabi-Yau metric to be solved on the actual CY3 sub-variety (not the
polysphere ambient), via a Donaldson-balanced sigma-functional solver
on Newton-projected sample points with an affine-chart tangent frame
from the implicit-function theorem.

**Implementation**:

| File | Function | Line | What it does |
|------|----------|------|--------------|
| `src/route34/ty_metric.rs` | `TianYauSolver::solve_metric` | (top) | Donaldson sigma-functional solver on the Tian-Yau bicubic sub-variety in `CP^3 x CP^3`. Sigma is the weighted L1-MAD of `eta = |det g_tan| / |Omega|^2` (Donaldson-Karp-Lukic-Reinbacher 2006 / Larfors-Schneider-Strominger 2020 convention). |
| `src/route34/schoen_metric.rs` | `SchoenSolver::solve_metric` | (top) | Same sigma-functional approach on the bidegree-(3,3,1) Schoen sub-variety in `CP^2 x CP^2 x CP^1`. |
| `src/route34/cy3_metric_unified.rs` | `Cy3MetricSpec` + `Cy3MetricSolver` | top | Trait dispatch unifying both candidates under a single solve interface. |
| `src/route34/cy3_metric_gpu.rs` | `solve_cy3_metric_gpu` | 44 | GPU path (Phase-1 CPU-fallback scaffold; Phase-2 NVRTC kernel deferred — see file docstring). |

**Tests**:

* `src/route34/ty_metric::tests::test_ty_metric_converges_at_k4` — sigma decreases monotonically.
* `src/route34/ty_metric::tests::test_ty_sigma_functional_decreases` — Donaldson balancing produces a sigma decrease.
* `src/route34/ty_metric::tests::test_ty_metric_volume_invariant` — total volume preserved across iteration.
* `src/route34/ty_metric::tests::test_ty_metric_vs_polysphere` — output differs from the legacy polysphere-ambient metric.
* `src/route34/schoen_metric::tests::test_schoen_metric_converges`
* `src/route34/schoen_metric::tests::test_schoen_sigma_decreases`
* `src/route34/cy3_metric_gpu::tests::gpu_matches_cpu_to_tolerance_ty` (1e-10)
* `src/route34/cy3_metric_gpu::tests::gpu_matches_cpu_to_tolerance_schoen` (1e-10)

**References cited in code**:
* Donaldson, "Some numerical results in complex differential geometry", Pure Appl. Math. Q. 5 (2009) 571, arXiv:math/0512625.
* Headrick-Wiseman, "Numerical Ricci-flat metrics on K3", Class. Quantum Grav. 22 (2005) 4931, arXiv:hep-th/0506129.
* Anderson-Karp-Lukas-Palti arXiv:1004.4399.
* Anderson-Gray-Lukas-Palti arXiv:1106.4804.
* Larfors-Schneider-Strominger arXiv:2012.04656.
* Donagi-He-Ovrut-Reinbacher 2006 §3 (Schoen intersection numbers).

The `refine.rs` legacy module (now labeled `// LEGACY-SUPERSEDED-BY-ROUTE34`)
runs on the polysphere ambient and only weighs the variety-defining
polynomials as soft constraints — that approach is superseded for
publication-grade work.

---

## 10. Hermitian-Yang-Mills metric on the bundle

**Chapter range**: line 196 (eigenvalue-pipeline step 1, "the gauge
bundle"; the HYM metric is the substrate-physical content of the
bundle's polystability that the chapter cites at lines 198, 206).

**Claim**: a publication-grade Yukawa extraction requires the HYM
Hermitian metric `h_V` on the bundle `V` (not the identity Hermitian
form). `h_V` is determined by the HYM equation
`Lambda_omega F_h = const * 1` where `Lambda_omega` is the Kahler-
form contraction; numerically solved by the T-operator iteration of
Anderson-Karp-Lukas-Palti 2010.

**Implementation**:

| File | Function | Line | What it does |
|------|----------|------|--------------|
| `src/route34/hym_hermitian.rs` | `solve_hym_metric` | 532 | T-operator iteration solver for `h_V`; cites AKLP 2010 §3. |
| `src/route34/hym_hermitian.rs` | `HymHermitianMetric` | 152 | The output metric struct: an `n_seeds x n_seeds` Hermitian matrix per sample point, plus convergence metadata. |
| `src/route34/hym_hermitian.rs` | `MetricBackground` trait | 211 | Sample-cloud abstraction so HYM works against either Tian-Yau or Schoen quadrature. |

**Tests**:

* `src/route34/hym_hermitian::tests::hym_iteration_converges` — T-operator convergence below tolerance.
* `src/route34/hym_hermitian::tests::trivial_bundle_returns_identity` — sanity check on the trivial bundle.

**References cited in code**:
* Donaldson, Proc. London Math. Soc. 50 (1985) 1.
* Uhlenbeck-Yau, Comm. Pure Appl. Math. 39 (1986) S257.
* Anderson-Karp-Lukas-Palti arXiv:1004.4399 §3.
* Wang, J. Differential Geom. 70 (2005) 393.

---

## 11. Harmonic Dirac zero modes

**Chapter range**: line 200 (eigenvalue-pipeline step 3).

**Claim**: the matter-field zero modes on the CY3 are the harmonic
representatives of `H^1(M, V tensor R)`, computed as the kernel of
the twisted Dirac operator `D_V = dbar_V + dbar_V^*`. Each generation
is a triple of zero modes (one per quark colour) at a specific
cohomology level.

**Implementation**:

| File | Function | Line | What it does |
|------|----------|------|--------------|
| `src/route34/zero_modes_harmonic.rs` | `solve_harmonic_zero_modes` | 499 | Genuine harmonic representatives via twisted Dirac kernel. Builds the section-basis Hermitian matrix `L_{alpha beta} = <D_V psi_alpha, D_V psi_beta>`, finds the low-eigenvalue subspace via Hermitian Jacobi rotations, takes the smallest-eigenvalue eigenvectors as the harmonic basis, and verifies residuals + orthonormality. |
| `src/route34/zero_modes_harmonic.rs` | `HarmonicZeroModeResult` | 164 | The result struct: harmonic modes, residual norms, orthonormality residual, observed-vs-predicted cohomology dimensions, run metadata. |
| `src/route34/zero_modes_harmonic.rs` | `polynomial_seed_modes` | 837 | The legacy polynomial-seed baseline (preserved for the seed-vs-harmonic difference test). |
| `src/route34/bbw_cohomology.rs` | (full module) | top | Bott-Borel-Weil + Koszul-chase line-bundle cohomology; predicts the dimension of `H^1(M, V)` so the harmonic solver knows how many kernel directions to expect. |
| `src/route34/zero_modes_harmonic_gpu.rs` | `solve_harmonic_zero_modes_gpu` | 60 | GPU path (Phase-1 CPU-fallback scaffold; Phase-2 NVRTC kernel deferred). |

**Tests**:

* `src/route34/zero_modes_harmonic::tests::harmonic_kernel_nonempty`
* `src/route34/zero_modes_harmonic::tests::harmonic_orthonormality_below_tol`
* `src/route34/zero_modes_harmonic::tests::harmonic_residuals_finite`
* `src/route34/zero_modes_harmonic::tests::polynomial_seed_vs_harmonic_differ`
* `src/route34/zero_modes_harmonic::tests::trivial_bundle_kernel_bounded`
* `src/route34/zero_modes_harmonic_gpu::tests::gpu_matches_cpu_to_tolerance` (1e-10)

**References cited in code**:
* Griffiths-Harris, "Principles of Algebraic Geometry" (Wiley 1978), Ch. 0 §6 (Hodge theorem on bundle-valued forms).
* Anderson-Karp-Lukas-Palti arXiv:1004.4399 §4.
* Anderson-Constantin-Lukas-Palti arXiv:1707.03442.
* Butbaia et al. arXiv:2401.15078 (2024) §5.2.

---

## 12. Polystability via DUY sub-sheaf enumeration

**Chapter range**: line 198 ("the bundle's structure group must embed
into `E_8` ... slope-stability conditions").

**Claim**: the heterotic gauge bundle `V` must be holomorphic and
polystable. The Donaldson-Uhlenbeck-Yau theorem identifies polystable
bundles with those admitting an HYM Hermitian metric. Numerical
verification enumerates rank-1, rank-2 Schur-functor, and partial-
monad-kernel sub-sheaves and verifies the slope inequality on each.

**Implementation**:

| File | Function | Line | What it does |
|------|----------|------|--------------|
| `src/route34/polystability.rs` | `check_polystability` | 478 | Full DUY check via coherent sub-sheaf enumeration: sub-line-bundles via `H^0(V (-d))`, partial monad-kernel sub-bundles, and Schur-functor (`wedge^k V`) sub-bundles up to rank `max_subsheaf_rank`. Replaces the legacy single-rank-1 check. |
| `src/route34/polystability.rs` | `PolystabilityResult` | 174 | Output: stability verdict, destabilizing sub-sheaves (if any), slope diagnostics. |
| `src/route34/polystability_gpu.rs` | (full module) | top | Rayon-batched polystability sweep across many candidate bundles, bit-exact with sequential CPU path. |
| `src/route34/bbw_cohomology.rs` | (full module) | top | Self-contained BBW + Koszul-chase line-bundle cohomology, independent of `crate::zero_modes`. |
| `src/route34/bundle_search.rs` | `enumerate_candidate_bundles` | 544 | Generates the candidate bundle catalogue (line-bundle degrees, monad map data) on which polystability is checked. |
| `src/route34/bundle_search.rs` | `LineBundleDegrees::is_polystable` | 238 | Quick polystability check at the line-bundle slope level. |

**Tests**:

* `src/route34/tests/test_polystability.rs` — DUY check on the canonical AKLP example (passes), on the MonadBundle::stable_example (passes), on a destabilizing sub-line-bundle case (fails with diagnostic).
* `src/route34/tests/test_bundle_search.rs` — candidate enumeration + filter pipeline.

**References cited in code**:
* Donaldson 1985 / Uhlenbeck-Yau 1986 — the DUY theorem.
* Huybrechts-Lehn 2010, "Geometry of Moduli Spaces of Sheaves", §1.2 (slope), §4.2 (polystability).
* Anderson-Karp-Lukas-Palti arXiv:1004.4399 §2.4.
* Anderson-Constantin-Lukas-Palti arXiv:1707.03442.
* Anderson-Gray-Lukas-Palti arXiv:1106.4804 Tabs. 3-5.

---

## 13. Yukawa triple overlap with quadrature error bars

**Chapter range**: line 202 (eigenvalue-pipeline step 4).

**Claim**: the Yukawa couplings are triple overlap integrals on the
CY3 against the matter-field zero-modes and the Higgs zero-mode.
Publication-grade values require the HYM metric in the normalisation
(not the identity) plus per-entry Monte-Carlo error bars from
bootstrap resampling, with a convergence test `n_pts -> 2 n_pts ->
4 n_pts` to confirm the integrator has converged.

**Implementation**:

| File | Function | Line | What it does |
|------|----------|------|--------------|
| `src/route34/yukawa_overlap_real.rs` | `compute_yukawa_couplings` | 326 | Triple overlap with HYM metric in the normalisation, Shiffman-Zelditch quadrature, bootstrap MC error bars, and convergence test. Returns `Y_{ijk}`, per-entry uncertainty `sigma(Y_{ijk})`, and convergence ratio. |
| `src/route34/yukawa_overlap_real.rs` | `YukawaResult` | 127 | Output struct. |
| `src/route34/yukawa_sectors_real.rs` | (full module) | top | E_8 -> E_6 x SU(3) decomposition + dynamic sector-to-cohomology assignment. Replaces the hardcoded SU(5) `[0,1,2]/[3,4,5]/[6,7,8]` slot split. |
| `src/route34/yukawa_overlap_real_gpu.rs` | `compute_yukawa_couplings_gpu` | 75 | GPU path (Phase-1 CPU-fallback scaffold; Phase-2 NVRTC kernel deferred). |

**Tests**:

* `src/route34/yukawa_overlap_real::tests::yukawa_uncertainty_decreases_with_n` — variance scales as `1/sqrt(N)`.
* `src/route34/yukawa_overlap_real_gpu::tests::gpu_matches_cpu_to_tolerance` (1e-10).

**References cited in code**:
* Anderson-Karp-Lukas-Palti arXiv:1004.4399 §5.
* Anderson-Constantin-Lukas-Palti arXiv:1707.03442 §3.
* Butbaia et al. arXiv:2401.15078 §5.3.
* Shiffman-Zelditch, Comm. Math. Phys. 200 (1999) 661.

---

## 14. RG flow GUT -> M_Z

**Chapter range**: line 204 (eigenvalue-pipeline step 5).

**Claim**: the fermion mass spectrum is computed from the Yukawas
`Y_{ijk}` evaluated at the GUT scale (where heterotic compactification
naturally lives) plus the Higgs VEV, with the Yukawas RG-evolved down
to `M_Z` via the Standard-Model 1-loop renormalization-group
equations of Machacek-Vaughn 1984.

**Implementation**:

| File | Function | Line | What it does |
|------|----------|------|--------------|
| `src/route34/rg_running.rs` | `run_yukawas_to_mz` | 46 | SM 1-loop RGE solver from `M_GUT = 2e16 GeV` down to `M_Z = 91.1876 GeV`. Cites Machacek-Vaughn 1984 *Nucl. Phys. B* 222:83 + 236:221 + 249:70. |
| `src/route34/rg_running.rs` | `top_yukawa_running_ratio` | 64 | Convenience: the dominant top-Yukawa running ratio, used as a sanity check. |

**Tests**:

* `src/route34/rg_running::tests::top_yukawa_running_within_observed_range` — top-Yukawa running matches the observed `y_t(M_GUT) / y_t(M_Z) ~= 0.45-0.55` band.

**References cited in code**:
* Machacek-Vaughn, "Two loop renormalization group equations in a general quantum field theory", *Nucl. Phys. B* 222 (1984) 83; 236 (1984) 221; 249 (1985) 70.
* Bednyakov-Pikelner-Velizhanin, "Three-loop SM beta-functions for matrix Yukawa couplings", arXiv:1303.4364 (2013).

---

## 15. End-to-end Yukawa pipeline

**Chapter range**: lines 192-206 (the "Eigenvalue-Computation Pipeline"
subsection).

**Implementation**:

| File | Function | Line | What it does |
|------|----------|------|--------------|
| `src/route34/yukawa_pipeline.rs` | `predict_fermion_masses` | 166 | End-to-end driver: metric -> bundle -> HYM -> harmonic modes -> Yukawas -> RG -> fermion masses + CKM. |
| `src/route34/yukawa_pipeline.rs` | `pipeline_chi_squared` | 276 | Chi-squared score against observed fermion masses + CKM. |
| `src/route34/yukawa_pipeline.rs` | `FermionMassPrediction` | 65 | Predicted spectrum struct: `(m_e, m_mu, m_tau, m_u, m_d, m_s, m_c, m_b, m_t)` + CKM angles. |
| `src/route34/yukawa_pipeline.rs` | `ObservablesSnapshot` | 99 | Reference observables from PDG 2024. |

**Tests**:

* `src/route34/yukawa_pipeline::tests::end_to_end_pipeline_runs`
* `src/route34/yukawa_pipeline::tests::aklp_pipeline_demo_print`

---

## 16. Bayesian discrimination

**Chapter range**: implicit in the chapter's "discrimination" framing
(see lines 281, 297, 301-303, 326-329 — the `~10^17` evaluation count
and the explicit "discrimination signal between the two candidates"
language).

**Claim**: the canonical discrimination output is the Bayes factor
`ln(Z_TY / Z_Schoen)` from a Skilling 2004 nested-sampling evidence
integral over the heterotic moduli space, plus a Jeffreys-scale
verdict + an equivalent N-sigma value derived from the chi-squared
survival function.

**Implementation**:

| File | Function | Line | What it does |
|------|----------|------|--------------|
| `src/route34/prior.rs` | `Prior` trait + `LogUniformPrior`, `DiscreteUniformPrior`, `ProductPrior`, `UniformPrior`, `GaussianPrior` | top | Prior distributions over moduli (Jeffreys 1946 log-uniform for Kahler scale parameters; uniform measure over discrete enumerations for line-bundle and Wilson-line choices). |
| `src/route34/likelihood.rs` | `evaluate_log_likelihood` | 174 | Combines route-1/2/3/4 chi-squared contributions plus the Yukawa-PDG comparison into a single log-likelihood. |
| `src/route34/likelihood.rs` | `chi2_sf` | 300 | Chi-squared survival function (1 - CDF). |
| `src/route34/likelihood.rs` | `p_value_to_n_sigma` | 359 | Converts a one-sided p-value to the equivalent two-sided N-sigma value via the standard normal quantile. |
| `src/route34/likelihood.rs` | `breakdown_from_route_results` | 382 | Aggregates the per-route chi-squared breakdown into the likelihood input. |
| `src/route34/nested_sampling.rs` | `compute_evidence` | 211 | Skilling 2004 nested-sampling evidence calculation; returns `ln Z` plus the information `H` and convergence diagnostics. |
| `src/route34/bayes_factor.rs` | `compute_bayes_factor` | 126 | Computes `ln(Z_A / Z_B)` plus Jeffreys-class verdict (Inconclusive / Substantial / Strong / Decisive) and the equivalent N-sigma. |
| `src/route34/discrimination.rs` | `run_full_discrimination` | 171 | High-level driver: enumerates candidate models, runs nested sampling on each, computes pairwise Bayes factors, and produces a `DiscriminationVerdict`. |
| `src/bin/bayes_discriminate.rs` | `main` | top | CLI entry point. Supports `toy` (analytic Gaussian) and `production` (full route-1/2/3/4 wiring) likelihood modes. |

**Tests**:

* `src/route34/likelihood::tests::*` — chi-squared survival function correctness against scipy reference values; p-value-to-N-sigma round-trip.
* `src/route34/nested_sampling::tests::*` — analytic-evidence correctness on a Gaussian likelihood (compares to `ln Z = -d/2 ln(2 pi sigma^2)` to 1% relative).
* `src/route34/bayes_factor::tests::*` — Jeffreys-class boundaries.
* `src/route34/discrimination::tests::*` — end-to-end run on a synthetic toy problem with known winner.

**Smoke run** (recorded in `FINAL_DISCRIMINATION_RESULT.md` at the
crate root):

```
./target/release/bayes_discriminate.exe \
    --candidates tian_yau,schoen --n-live 200 --n-metric-samples 500 \
    --seed 42 --output-dir /tmp/bayes_smoke_42 --likelihood toy
```

returns at toy-mode sample size:

```
tian_yau:  ln Z = -1.8344 +/- 0.0829  (H = 1.375 nats; iters = 725)
schoen:    ln Z = -1.2750 +/- 0.0626  (H = 0.784 nats; iters = 614)
schoen vs tian_yau: |ln B| = 0.5593 +/- 0.1039; class = Inconclusive
                                                eq. n-sigma = 1.06
```

The toy-mode output is by design (the toy likelihood is a 1-D Gaussian
centred at zero for both candidates; only at production-mode sample
size does the route-3 + route-4 + Yukawa-PDG signal differentiate the
candidates). The smoke run validates the wiring end-to-end; the
production-mode run is gated by the moduli-space discretisation budget
the chapter discusses at lines 326-329.

**Python orchestrator integration**:

`tier_bc/discrimination_runner.py::_try_bayes_discrimination` calls
the Rust binary when `GDS_USE_BAYES_DISCRIMINATE=1` is set, parses the
emitted `bayes_report.json`, and folds the Bayes factor + Jeffreys
class + equivalent N-sigma into the discrimination verdict. The
chi^2 verdict from Routes 3+4 is preserved alongside the Bayesian one
for cross-checking.

**References cited in code**:
* Skilling, "Nested sampling for general Bayesian computation",
  AIP Conf. Proc. 735 (2004) 395, DOI 10.1063/1.1835238.
* Jeffreys, "An invariant form for the prior probability in estimation
  problems", Proc. Roy. Soc. A 186 (1946) 453, DOI 10.1098/rspa.1946.0056.
* Kass-Raftery, "Bayes factors", J. Amer. Statist. Assoc. 90 (1995) 773
  (Jeffreys scale).
* Buchner, "A statistical test for Nested Sampling algorithms",
  Stat. Comput. 26 (2016) 383, arXiv:1407.5459.
* Berger J.O., "Statistical Decision Theory and Bayesian Analysis"
  (Springer 1985), §3.3 (objective priors).

---

## 17. Falsification conditions

**Chapter range**: lines 166-180.

**Claim**: five specific empirical signatures that would falsify the
heterotic `E_8 x E_8` commitment.

**Implementation status**: the discrimination pipeline does not
*directly* implement falsification tests (the falsification conditions
are predictions about future observations, not current outputs).
However, the route34 modules listed below produce the predicted
quantities that any future falsification observation would have to
contradict:

| Falsification condition | Predicted quantity (which route34 produces) |
|-------------------------|---------------------------------------------|
| 1. New gauge symmetry that does not embed in `E_8` | Implicit in `wilson_line_e8::e8_roots` enumeration — any future-discovered Lie algebra outside the `E_8` decomposition would be a falsification. |
| 2. Cosmology requiring more than two parent regions | Implicit in `chern_field_strength::integrate_visible_minus_hidden` — the formula assumes exactly 2 sectors. |
| 3. Dark-matter non-gravitational coupling | Outside the route34 scope (this is an experimental-physics question). |
| 4. Extended-locus or fibration-style merger-birth | Outside the route34 scope (this requires a different topology candidate). |
| 5. Non-ADE polyhedral wavenumber | `arnold_normal_form::admissible_wavenumber_set` returns only ADE-compatible wavenumbers. Any future observation of a 7-fold, 11-fold, or 13-fold stable polyhedral pattern would be a falsification. |

---

## 18. Per-claim cross-reference index

For reviewers who want to walk the chapter top-to-bottom:

| Chapter line(s) | Claim | Implementation entry above |
|----------------|-------|----------------------------|
| 6, 18, 52-79 | Heterotic `E_8 x E_8` commitment | §1, §4 |
| 117-141 | `Z/3` quotient + topological protection | §2, §3 |
| 156-164 | CY3 selection criteria | §1-§4 (combined) |
| 166-180 | Falsification conditions | §17 |
| 184-190 | Tian-Yau and Schoen as the two surviving candidates | §2, §3, §9 |
| 192-206 | Eigenvalue-computation pipeline | §15 (driver) + §9, §10, §11, §12, §13, §14 |
| 216-229 | Four routes (1: boundary conditions; 2: cross-term sign; 3: eta; 4: polyhedral) | §5 (Route 3), §6, §7, §8 (Route 4) |
| 233-263 | Route 3 `eta` integral form | §5 |
| 245-254 | The pinned `eta` form display equation (line 249) | §5 |
| 265-298 | Route 4 polyhedral wavenumber chain | §6 (Killing) + §7 (Arnold) + §8 (Rossby) |
| 299-305 | Bottom Line and Next Step (full discrimination pipeline) | §15 + §16 |
| 311-329 | Computational savings and the residual gap | §16 (Bayesian discrimination is the proposed closer) |

---

## 19. Test counts (route34, release profile)

* Total route34 tests at Wave-6 close: **317 passed, 0 failed, 2 ignored**.
* Total cross-cutting tests in `cy3_rust_solver`: **490 passed, 1 failed (legacy `refine.rs` debug-assert test, outside route34), 20 ignored**.
* GPU CPU/GPU agreement tests: **5 passed (cy3_metric_gpu x 2, divisor_integration_gpu, zero_modes_harmonic_gpu, yukawa_overlap_real_gpu)**, all at 1.0e-10 relative tolerance.

---

## 20. How to extend this map

When new functionality lands in route34:

1. Identify the chapter range it implements (file: `book/chapters/part3/08-choosing-a-substrate.adoc`).
2. Add a new numbered section above with: chapter range + claim + implementation table + tests + references.
3. Update the per-claim cross-reference index in §18.
4. If a published reference is cited in code, add the DOI/arXiv id to the section.
5. Bump the test counts in §19 from `cargo test --release route34`.

When Wave 5's Bayesian-discrimination files land, the §16 placeholder
above should be replaced with a full implementation table mirroring
the §5-§15 format.
