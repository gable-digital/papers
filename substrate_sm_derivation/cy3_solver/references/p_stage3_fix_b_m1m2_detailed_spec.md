# Stage 3 Fix B — M1 + M2 detailed implementation spec

**Status:** research spec, doc-only. Compatible companion to the G1
research draft `references/p_stage3_g1_fix.md` §5 and to the parallel
upstream-spike investigation (task `ad680663b0400d131`).

**Scope:** flesh out milestones M1 (genuine `(0,1)`-form representative
basis) and M2 (bundle-twisted derivative `∂̄_V` with HYM `(0,1)`-
connection coefficients) to enough precision that the user can decide
between (a) bespoke Rust, (b) Python adapter wrapping ACLP / Butbaia
upstream, or (c) hybrid.

**Critical finding up front (§2 below).** The G1 spec asserts:

> "the HYM connection `(0,1)`-coefficients
> `A^I_{ī J}(z, w)` from `crate::route34::hym_hermitian`."

This is **not what the code currently provides.** `hym_hermitian.rs`
solves for the *Hermitian metric matrix* `H_{αβ}` on the section basis;
it never extracts the **Chern connection 1-form**, and in particular
never decomposes it into `(1,0)` and `(0,1)` parts. The
`A^I_{ī J} = (h^{-1} ∂_{z̄_i} h)^I_{\ J}` data the G1 spec quotes
**must be added as a new infrastructure milestone (M0)** before M2 can
be assembled. The G1 estimate of "1 week" for M2 omits this missing
piece. Realistic effort with M0 included: **5–6 weeks** for the
bespoke path.

---

## Table of contents

1. M1 — `(0,1)`-form representative basis (per ACLP 2017 §4)
2. M2 — bundle-twisted derivative `∂̄_V` (with M0 prerequisite)
3. Concrete implementation outline (Rust types, file layout, tests, LOC)
4. Compatibility with the upstream-spike work — bypass map for path (b)
5. Recommendation

---

## 1. Section 1 — M1: genuine `(0,1)`-form representative basis

### 1.1 Why the polynomial-seed basis is many-to-one onto `H¹`

The G1 diagnosis (audit §A.5 / §C.2) is this:

* The legacy ansatz space spans `H^0(B) = ⊕ H^0(O(b_α))` — a space
  of *holomorphic sections* of the splitting line bundles.
* The harmonic-representative space we want is `H¹(X, V)` — the
  Dolbeault cohomology of `V`-valued `(0,1)`-forms modulo `∂̄_V`.
* The polynomial-seed map sends `s_α ∈ H^0(B_α)` to the *constant*
  Dolbeault class `[0]` because `∂̄ s_α = 0` for any holomorphic
  monomial in the ambient coordinates. Different `s_α` map to the
  same class — this is the "many-to-one" collapse.

The Bochner Laplacian `L_{αβ} = Σ_i ⟨∂_{z_i} s_α, ∂_{z_i} s_β⟩_{h_V}`
that the current `build_laplacian_matrix`
(`src/route34/zero_modes_harmonic.rs:687–795`) constructs is
*holomorphic-derivative*-based. It probes `H^0` curvature, not `H¹`
co-closedness — so its kernel never sees the non-trivial Dolbeault
direction.

### 1.2 ACLP 2017 §4 — the `(0,1)`-form representative basis

Anderson–Constantin–Lukas–Palti, *"Yukawa couplings in heterotic
Calabi–Yau models"*, arXiv:1707.03442 (2017), §4 gives the
construction this spec builds on. The methodology, summarised:

1. **Test sections live in `Ω^{0,1}(X, V)`**, not in `Γ(X, V)`.
   Each test object is a `V`-valued `(0,1)`-form
   ```
       ω = Σ_{a, I}  f^I_{ a}(z, z̄)  dz̄^a  ⊗  e_I
   ```
   where:
   - `a ∈ {1, 2, 3}` runs over **antiholomorphic CY3 directions**
     (only 3 because we are on a complex 3-fold);
   - `I` runs over a frame `{e_I}` of `V` (so `I` has `rank(V)`
     values — 3 for SU(3), 4 for SU(4)/BHOP);
   - `f^I_{ a}(z, z̄)` is a smooth coefficient function (not
     holomorphic — anti-holomorphic dependence is allowed and is what
     gives `H¹` non-trivial content).

2. **Concrete realisation on a monad bundle.** ACLP §4 (eqs. (4.7)–
   (4.12)) realises `f^I_{a}` as a **rational function** on the
   ambient projective space, of bidegree chosen so that `ω` lives in
   the right Čech-class. For a monad `0 → V → B → C → 0`, ACLP take
   the basis
   ```
       ω^{(α, a)}  =  s_α(z)  ·  (z̄^a / |z|²)  ·  dz̄^a  ⊗  e^B_α
   ```
   where `s_α` is a polynomial seed in `H^0(B_α)`, the rational
   `(z̄^a / |z|²)` factor is the Beltrami-differential-style
   `∂̄`-non-trivial component (this is exactly what the
   polynomial-seed map kills by setting `f^I_a ≡ 0`), and `e^B_α` is
   the frame element of `V` corresponding to the α-th `B`-summand.

3. **Anti-holomorphic-degree filtering.** ACLP §4.2 require the
   coefficient `f^I_a` to be of total bidegree `(d_z, d_z̄)` with
   `d_z̄ ≥ 1` (otherwise the form is `∂̄`-exact, i.e. trivial in
   cohomology) and bounded above by a cap chosen so the basis
   stabilises (their Table 2 reports stabilisation at total degree
   ≈ 4 for the `B = O(1,0)^3 ⊕ O(0,1)^3` example).

4. **`∂̄_V`-closedness projection.** Not every `(α, a)` pair gives a
   `∂̄_V`-closed form. The basis is reduced to the
   `∂̄_V`-closed subspace by solving a small linear system **before**
   the Laplacian is assembled. ACLP §4.3 give the explicit projection
   formula (their eq. (4.18)).

### 1.3 Basis-enumeration formula at total polynomial degree `d`

Combining the above, for a monad with `rank(V) = r`, on CY3 (`a = 1, 2,
3`), at completion degree `d`:

```
N_{(0,1)-basis}(d, r)  =  3  ·  Σ_α  D_α(d)
```

where `D_α(d)` is the number of bigraded monomials of total degree
`≤ d` in `H^0(O(b_α[0], b_α[1]))` weighted by the antiholomorphic
extension space (the `(z̄^a / |z|²)` factor consumes one
antiholomorphic degree, so `D_α(d) = #{(d_z, d_w) : b_α[0] ≤ d_z ≤ d,
b_α[1] ≤ d_w ≤ d, d_z + d_w ≤ d}` × `binomial(d_z + 3, 3) ·
binomial(d_w + 3, 3)`).

For the AKLP example bundle (`B = O(1,0)^3 ⊕ O(0,1)^3`, rank 3) at
`d = 4`, `N_{(0,1)-basis} ≈ 3 · 6 · 35 = 630` raw basis elements
**before** `∂̄_V`-closedness reduction, dropping to ~150–200 after
projection. (ACLP Table 2, §4.4.)

For BHOP rank-4 SU(4) the prefactor scales by `4/3` so ~840 raw and
~200–270 closed.

### 1.4 Atiyah–Singer cohomology dimension expectation

The G1 spec quotes "3 generations × 16 = 48 modes after Wilson-line
projection" for the SO(10) chiral-matter cohomology. **This number is
the post-Wilson-line Atiyah–Singer count and is the right kernel
target for M1's eigensolver convergence test on Schoen Z/3×Z/3 + BHOP
SU(4).** The corresponding upstairs (pre-Wilson) count is
`48 · |Γ| = 48 · 9 = 432` modes for `Γ = Z/3 × Z/3`, which is what
the bare `∂̄_V` kernel should reproduce *before* any Wilson-line
projection is applied.

For the AKLP rank-3 SU(3) bundle on Tian-Yau the corresponding
target is the audit's `n_27 = 9` chiral modes (3 generations of `27`
of E_6) after the Z/3 quotient.

### 1.5 What M1 specifically delivers

1. A struct `OneFormSeed` extending the existing `ExpandedSeed`
   (`zero_modes_harmonic.rs:155–164`) with:
   - `antihol_index: u8` (the `a ∈ {1, 2, 3}` index of the `dz̄^a`
     factor — encoded as `a ∈ {0, 4} | {1, 5} | {2, 6} | {3, 7}` in
     the 8-coordinate convention, paired so the antihol index matches
     the holomorphic ambient direction);
   - `pole_normalisation: f64` (the `1/|z|²` Beltrami denominator's
     resolution — see §3.1);
   - inherits `b_line` (= the frame index `I`) and `exponents` (= the
     `(d_z, d_w)` bidegree).

2. A `(0,1)`-closedness projector that takes the raw enumerated basis
   and produces the `∂̄_V`-closed subspace via a small QR-on-rectangular
   linear system (ACLP eq. (4.18)).

3. A reproduction of the legacy basis as the `d = 0` degenerate slice
   (which is `∂̄_V`-exact and therefore drops to dimension 0 — the
   correct cohomological answer that explains why Fix A failed).

---

## 2. Section 2 — M2: bundle-twisted derivative `∂̄_V`

### 2.1 The operator we must assemble

Acting on a `V`-valued `(0,1)`-form `ω = f^I_a dz̄^a ⊗ e_I` we want

```
    (∂̄_V ω)^I_{a b}  =  ∂_{z̄_a} f^I_b  -  ∂_{z̄_b} f^I_a
                          +  A^I_{ ā J} f^J_b  -  A^I_{ b̄ J} f^J_a
```

where `A^I_{ā J}(z, z̄)` is the **(0,1)-component of the Chern
connection 1-form** on `V`, dual to the HYM Hermitian metric. The
matrix-element on the seed basis is
`L_{(α a),(β b)} = ⟨∂̄_V ω^{(α a)}, ∂̄_V ω^{(β b)}⟩_{h_V, ω}` evaluated
on the Shiffman-Zelditch sample cloud.

### 2.2 What is currently in `hym_hermitian.rs` — and what is NOT

I read `src/route34/hym_hermitian.rs` end-to-end (1230 lines) and
inventoried:

**Present:**
- `HymHermitianMetric.h_coefficients` — the converged Hermitian
  matrix `H_{αβ}` on the **section basis**, indexed by B-summand pair
  (file:184–212, accessor at 235–237).
- `solve_hym_metric` — the AKLP balanced-functional iteration that
  converges `H` (file:804–896).
- `compute_hym_residual` — `||σ(H)·H^{-1} − cI||_F` for diagnostics
  (file:479–583).
- A `MetricBackground` trait giving the sample cloud (file:257–294).

**Absent (the M0 gap):**
- No function returning the Chern connection `A = h^{-1} ∂h`.
- No function returning the `(0,1)`-component `A^{(0,1)} = h^{-1}
  ∂̄ h`. (For a Chern connection on a holomorphic bundle the
  `(0,1)`-part of the connection is identically zero in the
  *holomorphic frame* — the non-trivial connection component is
  in the `(1,0)` direction. The thing that couples different bundle
  frames is the curvature `F = ∂̄(h^{-1} ∂ h)` and its `(1,1)`-
  component.)
- No discretised representation of `∂_{z̄_a} h` (the antiholomorphic
  derivative of the metric matrix at sample points).

**Subtle technical note that blunts the M0 ask.** For a Chern
connection on a holomorphic vector bundle, the standard convention
(Griffiths-Harris 1978 Ch. 0 §5) splits the Chern 1-form
`θ = h^{-1} ∂ h` into
- `θ^{(1,0)} = h^{-1} ∂ h` (non-trivial),
- `θ^{(0,1)} = 0` (in the holomorphic frame!).

So the symbol `A^I_{ā J}` in the G1 spec, **in the standard Chern
gauge, is identically zero**. The non-trivial `(0,1)` content lives in
the curvature `F^{(1,1)} = ∂̄ θ^{(1,0)} = ∂̄(h^{-1} ∂ h)`, and the
`∂̄_V` operator on `(0,1)`-forms only differs from the flat `∂̄` by a
**curvature term** when one commutes derivatives (Bianchi).

This recasts M2 / M0 as: we do not need raw `(0,1)`-connection
coefficients; we need the **curvature `(1,1)`-form
`F^I_{ a b̄ J}(z, z̄)` evaluated on the sample cloud**, which IS the
non-trivial coupling between frame components. Concretely:

```
    F^I_{ a b̄ J}(p)  =  ∂_{z̄_b}  (h^{-1} ∂_{z_a} h)^I_{ J}  (p)
```

This is what M0 must add. M2 then assembles the bundle-twisted
Bochner-Lichnerowicz Laplacian
```
    Δ_{∂̄_V}  =  ∂̄_V^* ∂̄_V + ∂̄_V ∂̄_V^*
              =  flat term  +  F^{(1,1)} · (·)        (Weitzenböck)
```
on the closed `(0,1)`-form basis from M1.

### 2.3 Sample-point convention

M0's curvature evaluation must use the **same sample cloud** as the
HYM solve, i.e. the `MetricBackground` instance that drove
`solve_hym_metric`. The Donaldson points / Shiffman-Zelditch weights
are the canonical quadrature, and re-using them lets `H_{αβ}(p_k)`
(the pointwise Bergman kernel) be cached from the solve.

The M0 evaluator computes `∂_{z̄_b}(h^{-1} ∂_{z_a} h)` symbolically
in the section basis (the basis sections `s_α(z) = z_0^{b_α[0]}
w_0^{b_α[1]}` have closed-form holomorphic and antiholomorphic
derivatives), then contracts with the converged `H` and `H^{-1}` at
each sample point `p_k`.

### 2.4 Schoen + BHOP rank-4 status

The Schoen Z/3×Z/3 sampler exists (`schoen_sampler.rs`, 1561 lines, 77
tests; per MEMORY.md "Schoen sampler is REAL"). The BHOP rank-4 SU(4)
bundle is representable in the existing `MonadBundle` struct
(`zero_modes.rs:218–260`); its `b_lines` configuration just makes the
section basis 4-dimensional per quadrature point instead of 3.

`solve_hym_metric` is bundle-dimension-agnostic — it works for any
`MonadBundle` whose section basis has positive dimension. So **no new
infrastructure is required to run the existing HYM solver on Schoen +
BHOP**; the only gap is M0 (extracting `F^{(1,1)}`).

---

## 3. Section 3 — Concrete implementation outline

### 3.1 New module layout (bespoke Rust path)

```
src/route34/
├── hym_hermitian.rs                # existing — unchanged
├── hym_curvature.rs                # NEW (M0): F^{(1,1)} extractor
├── one_form_basis.rs               # NEW (M1): Ω^{0,1}(V) basis + closedness
├── twisted_dirac.rs                # NEW (M2): ∂̄_V matrix assembly
├── zero_modes_harmonic.rs          # existing — gain a v2 entry point
└── ...
```

The legacy `solve_harmonic_zero_modes` entry point is preserved
bit-identically. A new sibling
`solve_harmonic_zero_modes_twisted_dirac` is added that consumes the
M0/M1/M2 infrastructure and is the only path the production binaries
need to switch to.

### 3.2 M0 — `hym_curvature.rs` (new file, ~400 LOC)

```rust
//! Chern curvature `F^{(1,1)}` on a polystable monad bundle.

use crate::route34::hym_hermitian::{HymHermitianMetric, MetricBackground};
use crate::zero_modes::MonadBundle;
use num_complex::Complex64;

/// Pointwise (1,1)-curvature of the Chern connection on V, evaluated
/// on a sample cloud.  `entries[(a, b̄, p)] = F^{·}_{a b̄ ·}(p)` with
/// each entry an `n × n` complex matrix in the B-section basis.
pub struct ChernCurvatureCloud {
    /// 8 holomorphic coordinate directions × 8 antiholomorphic ×
    /// n_pts × n×n matrix.  Stored row-major over (a, b, p) outer,
    /// (i, j) inner.
    pub entries: Vec<Complex64>,
    /// Side length n of the bundle's section basis
    /// (equal to `bundle.b_lines.len()`).
    pub n: usize,
    /// Number of sample points.
    pub n_pts: usize,
}

/// Evaluate `F^{(1,1)} = ∂̄(h^{-1} ∂h)` at every accepted sample
/// point of `metric`.
pub fn evaluate_chern_curvature(
    bundle: &MonadBundle,
    metric: &dyn MetricBackground,
    h_v: &HymHermitianMetric,
) -> ChernCurvatureCloud { /* ... */ }

/// Diagnostics: trace and Frobenius norm of `F^{(1,1)}` averaged
/// over the cloud.  For an HYM-converged metric `i Λ_ω F = 0` in
/// the SU(n) case, so the integrated `Tr F` should be near-zero.
pub fn curvature_diagnostics(
    cloud: &ChernCurvatureCloud,
    metric: &dyn MetricBackground,
) -> CurvatureDiagnostics { /* ... */ }
```

**Tests (M0):**
1. `trivial_bundle_has_zero_curvature` — for `b_lines = [[0,0]; n]`
   (constant frame), `F ≡ 0`.
2. `rank_one_line_bundle_curvature_is_pure_trace` — `F^{(1,1)}` is
   diagonal scalar (`O(1)` line bundle has FS curvature).
3. `aklp_bundle_hym_converged_curvature_traceless` — `Tr F` < 1e-2
   (AKLP convergence floor).
4. `curvature_evaluator_uses_same_quadrature_as_hym` — pin: passing
   the same `MetricBackground` and `HymHermitianMetric` produces the
   expected reuse of cached Bergman kernels.

**Risks:**
- HYM convergence floor of ~1e-2 (AKLP Fig. 1) means the residual
  `(1,1)`-curvature carries ~10⁻² relative noise — this propagates
  into the M2 Laplacian's lowest eigenvalues and must be quantified
  per-degree, not blamed on M2.
- Antiholomorphic derivative of `H` requires either (a) symbolic
  per-monomial derivatives reusing the existing `d_dz` machinery
  (`zero_modes_harmonic.rs:325`) trivially conjugated, or (b) a
  finite-difference cloud at the sample points. Choice (a) is exact
  but more code; (b) is shorter but introduces O(h²) error. Spec
  recommends (a).

### 3.3 M1 — `one_form_basis.rs` (new file, ~600 LOC)

```rust
//! Genuine Ω^{0,1}(V)-form representative basis (ACLP 2017 §4).

use crate::zero_modes::MonadBundle;
use num_complex::Complex64;

/// A single test (0,1)-form: f(z, z̄) · dz̄^a ⊗ e^B_{b_line}.
#[derive(Clone, Debug)]
pub struct OneFormSeed {
    pub b_line: usize,        // frame index I
    pub antihol_dir: u8,      // a ∈ {0..7}: which dz̄_a
    pub holo_exponents: [u32; 8],  // |I|·|J| as before
    pub antihol_exponents: [u32; 8],  // antihol bidegree extras
    pub td_first_appears: u32, // for hierarchical-prefix sort
}

/// Enumerate the raw (0,1)-form basis at completion degree `d` per
/// ACLP eq. (4.7)-(4.12).  Returns elements in canonical hierarchical
/// order so the cap=k basis is a prefix of cap=(k+1).
pub fn enumerate_raw_one_form_basis(
    bundle: &MonadBundle,
    completion_degree: u32,
) -> Vec<OneFormSeed> { /* ... */ }

/// Project the raw basis onto the ∂̄_V-closed subspace (ACLP eq.
/// 4.18).  Returns the projected basis + a full-rank-witness QR
/// decomposition for downstream debug.
pub fn project_to_closed(
    raw: &[OneFormSeed],
    /* curvature info needed for closedness test */
    cloud: &ChernCurvatureCloud,
) -> ClosedOneFormBasis { /* ... */ }
```

**Tests (M1):**
1. `basis_size_matches_aclp_table_2_aklp_example` — at `d = 4` on the
   AKLP `B = O(1,0)^3 ⊕ O(0,1)^3` bundle, raw basis ≈ 630, closed
   basis dimension matches ACLP Table 2 (~150).
2. `prefix_property_d3_is_prefix_of_d4` — sorted bases preserve the
   Galerkin-refinement invariant (existing `expanded_seed_basis`
   already pins this for the legacy basis).
3. `closed_basis_dim_matches_bbw_dimension_aklp` — at sufficiently
   high `d`, dim ker `∂̄_V` matches the `n_27` BBW prediction from
   `compute_zero_mode_spectrum` (= 9 for AKLP).
4. `closed_basis_dim_matches_atiyah_singer_schoen_bhop` — at `d = 4`
   on Schoen Z/3×Z/3 + BHOP rank-4, dim matches the 432 (or 48
   post-Wilson) Atiyah–Singer count, ±10%.
5. `legacy_basis_d0_collapses_to_zero_kernel` — pin: the polynomial-
   seed basis at the reduced `d = 0` slice gives a `∂̄_V`-closed
   subspace of dimension 0 (because every constant section is
   `∂̄_V`-exact). This is the cohomological certification of why
   Fix A failed.

**Risks:**
- ACLP §4.4 caveat their basis with "we have not verified
  completeness in degree" — the closure projection sometimes drops
  mass. Mitigation: degree-by-degree convergence sweep with
  bookkeeping of dropped vs. retained ranks.
- The Beltrami-style `1/|z|²` denominator introduces poles at the
  ambient origin `z = 0`. The Schoen and Tian-Yau samplers exclude
  origin points by construction, but a defensive `panic_if_origin`
  guard is required.

### 3.4 M2 — `twisted_dirac.rs` (new file, ~700 LOC)

```rust
//! Bundle-twisted Bochner-Lichnerowicz Laplacian Δ_{∂̄_V} on
//! Ω^{0,1}(V) basis.

/// Assemble the n × n Hermitian Δ_{∂̄_V} matrix on the closed basis,
/// using the M0 curvature cloud and the M1 closed basis.
pub fn build_twisted_dirac_laplacian(
    closed_basis: &ClosedOneFormBasis,
    metric: &dyn MetricBackground,
    h_v: &HymHermitianMetric,
    cloud: &ChernCurvatureCloud,
) -> Vec<Complex64> { /* returns n×n in row-major */ }

/// Public entry point — replaces solve_harmonic_zero_modes for the
/// twisted-Dirac path.
pub fn solve_harmonic_zero_modes_twisted_dirac(
    bundle: &MonadBundle,
    ambient: &AmbientCY3,
    metric: &dyn MetricBackground,
    h_v: &HymHermitianMetric,
    config: &HarmonicConfig,
) -> HarmonicZeroModeResult { /* ... */ }
```

**Tests (M2):**
1. `twisted_dirac_kernel_dim_matches_bbw_aklp_d4` — kernel rank = 9
   on AKLP at `d = 4`. **This is the test that distinguishes a
   real ∂̄_V from the rank-1-collapse Bochner Laplacian.**
2. `twisted_dirac_kernel_dim_matches_atiyah_singer_schoen_bhop_d4` —
   kernel rank ∈ [432 ± 10%] on Schoen Z/3×Z/3 + BHOP rank-4 at
   `d = 4` (pre-Wilson-line).
3. `twisted_dirac_residuals_below_tol` — for each kernel basis
   vector ψ, `||∂̄_V ψ||_{L²}` < 1e-6 on a converged HYM (residual
   floor is set by the M0 curvature noise, not by M2).
4. `weitzenbock_consistency` — the assembled Laplacian = flat
   `∂̄^* ∂̄ + ∂̄ ∂̄^*` + curvature-coupling term to within 1e-10
   (consistency check on the formula).

**Risks:**
- Memory at `d ≥ 3`: the closed basis at Schoen + BHOP rank-4 reaches
  ~10000 modes; the `n × n` Hermitian Laplacian is 1.6 GB. **M3
  Lanczos solver remains on the critical path** — M2's matrix is
  build-time feasible but not diagonalisation-feasible without M3.
- Hermitian-projection averaging (existing convention at
  `zero_modes_harmonic.rs:782–793`) suppresses the antihermitian
  roundoff; M2 must use the same convention.

### 3.5 LOC + effort summary (bespoke path)

| Milestone | New LOC | Tests | Effort  |
|-----------|---------|-------|---------|
| M0 (curvature)         | ~400 | 4 | 1.5 weeks |
| M1 (one-form basis)    | ~600 | 5 | 1.5 weeks |
| M2 (twisted Laplacian) | ~700 | 4 | 1.5 weeks |
| M3 (Lanczos eigsolve)  | ~500 | 3 | 1 week    |
| M4 (PDG regression)    | ~200 | 6 | 0.5 week  |
| **Total bespoke**      | ~2400| 22| **~6 weeks** |

The G1 spec's 3–4 weeks omits M0 and underestimates Lanczos. **The
realistic bespoke estimate is 5–6 weeks.**

---

## 4. Section 4 — Compatibility with the upstream-spike work

The parallel agent (task `ad680663b0400d131`) is investigating whether
ACLP 2017's PyTorch reference code (or Butbaia 2024's) can be wrapped
as a Python adapter. The two upstream candidates are:

1. **ACLP 2017** — arXiv:1707.03442, code at
   `https://github.com/...` (verify via parallel-agent report).
   Targets explicit `(0,1)`-form basis on monad bundles; rank-3 SU(3)
   only in published examples.
2. **Butbaia 2024** — arXiv:2401.15078, code at
   `https://github.com/cyclam/heterotic-yukawa-numerics` (verify).
   Targets neural-network HYM + bundle-Dirac discretisation;
   includes rank-4 SU(4) examples per §5.3.

### 4.1 Path (b) — Python wrap — milestone bypass map

If the upstream wrap is viable, the Python adapter replaces M0 + M1 +
M2 + M3 wholesale (the upstream code already implements all four).
**Bespoke milestones bypassed:** M0, M1, M2, M3 (the full ~5 weeks of
core work). **Bespoke milestones still needed:**

- **A new milestone M2.5 (1 week)**: Rust ↔ Python interop layer.
  Marshal `MonadBundle`, `HymHermitianMetric`, `MetricBackground` into
  the upstream's input format (likely numpy `.npy` files for the
  sample cloud, JSON for the bundle config). Marshal kernel basis
  back. PyO3 or subprocess+JSON.
- **M4 (PDG regression, 0.5 week)**: unchanged.

**Net effort under path (b): ~1.5 weeks** (down from ~6 weeks bespoke).

### 4.2 Risks of path (b) that the upstream-spike must verify

1. **Bundle representation match.** Does the upstream encode
   `MonadBundle` in a compatible scheme? If they hard-code the AKLP
   `O(1,0)^3 ⊕ O(0,1)^3` example, BHOP rank-4 will need a fork.
2. **Sample-cloud convention match.** Does the upstream accept
   externally-supplied sample points + weights, or does it run its
   own DKLR / Donaldson sampler? Mismatch breaks the framework's
   convergence story.
3. **HYM solver match.** Does the upstream solve HYM internally
   (and how — same AKLP balanced-functional, or NN gradient descent
   à la Butbaia 2024?), or does it accept an externally-converged
   `H`? If internal-only, our hard-won AKLP-2010 σ-fit (memory entry
   `project_cy3_aklp_2010_match.md`) gets bypassed.
4. **Schoen geometry match.** The upstream may only support
   complete-intersection CY3s; Schoen Z/3×Z/3 is fiber-product on
   `CP² × CP² × CP¹` (DHOR-2006 §3) and may not be in any upstream
   library out of the box.

### 4.3 Hybrid option

Hybrid: prototype with the upstream wrap (path b) to **validate the
Atiyah–Singer kernel-rank prediction** and the **rank-3 Yukawa
hierarchy** at low risk; then port to bespoke Rust (path a) for
production deployment, because:
- Rust homogeneity is a maintenance win (no PyO3 ABI fragility).
- The HYM solver in `hym_hermitian.rs` is already AKLP-converged
  and σ-pinned — re-implementing the rest of the stack in the same
  language has low ongoing-cost.
- The Lanczos solver from `pwos_math::linalg` is reusable.

**Hybrid effort: ~1.5 weeks (prototype) + ~5 weeks (production).**

---

## 5. Section 5 — Recommendation

### Decision drivers

| Factor | Bespoke (a) | Wrap (b) | Hybrid |
|--------|-------------|----------|--------|
| **HYM connection coeff availability** | ❌ M0 needed (~1.5 wk) | ✅ upstream provides | prototype: ✅ / prod: ❌ |
| **Upstream wrap viability** | n/a | ⚠ unverified — depends on §4.2 spike | ⚠ same |
| **BHOP rank-4 SU(4) support** | ✅ (we control the bundle struct) | ⚠ depends on upstream | ✅ |
| **Schoen Z/3×Z/3 sampler** | ✅ (already real, 1561 LOC) | ⚠ likely needs forking | ✅ |
| **Maintenance cost** | low (Rust homogeneity) | high (Python ABI) | medium (transitional) |
| **Time-to-first-result** | ~6 weeks | ~1.5 weeks (if viable) | ~1.5 weeks for prototype |
| **Time-to-publication-quality** | ~6 weeks | ~1.5–8 weeks (depending on Schoen / BHOP fork cost) | ~6.5 weeks |

### Recommendation

**Hybrid.** Run the upstream-spike (path b prototype) **first** to:

1. Confirm the Atiyah–Singer kernel rank fires correctly on at least
   the AKLP rank-3 SU(3) example (a 1–2 day milestone in the wrap).
   This validates that the M0+M1+M2 mathematical content is right.
2. Get a **first PDG-hierarchy data point** at low cost — even if it's
   only on the AKLP bundle, it tells us whether the rank-3 Yukawa
   hierarchy emerges at all from a genuine `∂̄_V` (vs. a rank-1
   collapse, which would indicate a deeper structural issue with the
   framework's bundle, not the Bochner approximation).

**Then port to bespoke Rust.** Specifically:

- If the wrap shows the rank-3 hierarchy emerging on AKLP at
  ~10⁻¹ accuracy: **green-light bespoke M0+M1+M2+M3+M4** (5–6 weeks),
  using the upstream's eigenvalues as numerical ground-truth for the
  bespoke regression tests.
- If the wrap is non-viable for BHOP rank-4 / Schoen geometry:
  **bespoke is forced** (5–6 weeks) and the prototype's only role is
  the AKLP-rank-3 sanity check.
- If the wrap shows persistent rank-1 collapse even on a genuine
  `∂̄_V`: **stop and re-audit the bundle physics**, because this
  would mean the framework's claim to a chiral SO(10) bundle is
  upstream-broken, not numerics-broken.

The hybrid path **front-loads the highest-risk question (does the
mathematical Fix B even break the rank-1 collapse on this framework's
bundle?) for the lowest cost (~1.5 weeks)**, before committing 6
weeks of Rust implementation. This is the responsible bet given the
framework's publication-quality posture (per
`project_cy3_5sigma_discrimination_achieved.md`).

---

## 6. Cross-references

- G1 research draft: `references/p_stage3_g1_fix.md` §5
- Audit: `references/p_stage3_eigenmode_audit.md` §A.5 / §C.2
- HYM module (M0 host): `src/route34/hym_hermitian.rs:184–212` (struct),
  `:804–896` (solver)
- Existing harmonic solver (M1/M2 sibling target):
  `src/route34/zero_modes_harmonic.rs:687–795` (current Bochner build),
  `:818–824` (entry point)
- ACLP 2017: arXiv:1707.03442 §4 (basis), §4.3 (closure projection)
- Butbaia 2024: arXiv:2401.15078 §5 (Dirac discretisation)
- AKLP 2010 σ-fit pin: `project_cy3_aklp_2010_match.md`
- Schoen sampler (real, not stub):
  `src/route34/schoen_sampler.rs` (1561 LOC)
- Stage 3 publication target: chain-match (Schoen) + 6.92σ
  discrimination, both standing — Fix B is what unlocks the **Yukawa**
  publication rail (Stage 4+), not the σ-channel rail (Stage 3).
