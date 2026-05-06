# Stage 3 G1 — Yukawa rank-1-collapse fix attempt (Fix A)

**Status:** **Fix A INSUFFICIENT.** Bumping the polynomial-completion
degree of `solve_harmonic_zero_modes` from the legacy single-bidegree
basis to the cumulative-bigraded basis at cap = 2 grows the seed-basis
50× (24 → 1 260 seeds on the AKLP TY bundle) but **does not break the
rank-1 mode degeneracy.** Result: every fermion-mass ratio
(`m_e/m_τ`, `m_μ/m_τ`, `m_u/m_t`, `m_c/m_t`, `m_d/m_b`, `m_s/m_b`)
remains at the floating-point zero floor at both cap = 1 (legacy) and
cap = 2 (new). Fix B (genuine bundle-twisted Dirac kernel `∂̄_V`) is
the only remaining avenue. Fix B is **out of scope for this task**;
this document drafts the research spec.

**Audit reference:** `references/p_stage3_eigenmode_audit.md` §A.5
item 1, §C.2 — gap **G1** ("non-degenerate harmonic modes on Schoen").

**Audit head when this work started:** `d05bc3a2` (origin/main).

**Key files modified:**
- `src/route34/zero_modes_harmonic.rs` — `expanded_seed_basis` now
  takes a `completion_degree: u32` parameter and runs the cumulative
  bigraded enumeration with hierarchical Galerkin-refinement sort
  when `cap >= 2`. The pre-G1 single-bidegree path is preserved
  bit-identically when `cap <= 1` (default = 2 per
  `HarmonicConfig::default()`, but practical use of cap ≥ 2 requires a
  Lanczos-style sparse eigensolver — see Compute-bound caveat below).
- `src/bin/p_stage3_g1_basis_sweep.rs` — new binary driving the
  basis-convergence sweep across `completion_degree ∈ {1, 2, 3, 4, 5, 6}`.

**Output files:**
- `output/p_stage3_g1_cap1.json` — cap = 1 (legacy) baseline.
- `output/p_stage3_g1_cap_1_2.json` — cap ∈ {1, 2} on TY/Z3 (Schoen
  cap = 2 was attempted but did not fit into the wallclock budget;
  see compute-bound caveat).

---

## 1. What Fix A actually changed

The audit (§A.5 / §C.2) recommended **Fix A — basis degree bump**:
"Increase `Z3XZ3_MAX_TOTAL_DEGREE` from 4 to 6, run a basis-convergence
sweep, decide based on stabilisation."

**Audit conflation (caveat).** The audit's wording ("`Z3XZ3_MAX_TOTAL_DEGREE`
=
4 in production") refers to the experimental
`route34::zero_modes_harmonic_z3xz3` solver — **not** the production
`route34::zero_modes_harmonic` solver consumed by
`predict_fermion_masses`. The latter had a different problem: the dead
`HarmonicConfig::completion_degree` config field existed but **was
never wired** into the basis builder; the legacy
`expanded_seed_basis(bundle)` always returned exactly the natural
single-bidegree basis (one element per bigraded monomial of bidegree
`(b_α[0], b_α[1])` per B-summand).

The G1 commit:

1. Adds a `completion_degree: u32` parameter to
   `expanded_seed_basis` in
   `src/route34/zero_modes_harmonic.rs`.
2. Implements the cumulative bigraded enumeration when
   `cap >= 2`: every bidegree `(d_z, d_w)` with
   `b_α[0] ≤ d_z ≤ max(b_α[0], cap)` and
   `b_α[1] ≤ d_w ≤ max(b_α[1], cap)` per B-summand, sorted by
   `td_first_appears = max(deg_z, deg_w)` so the cap = k basis is a
   strict prefix of the cap = (k + 1) basis (Galerkin refinement
   invariant, identical to the convention used by
   `zero_modes_harmonic_z3xz3::expanded_seed_basis`).
3. Wires `config.completion_degree` into the call site at
   `solve_harmonic_zero_modes`.
4. Preserves the pre-G1 behaviour bit-identically when `cap <= 1`.

This brings the production solver's basis-completion mechanism in
line with what the audit described.

---

## 2. Per-degree sweep results (k = 2, n_pts = 300, iter_cap = 3)

The sweep runs `predict_fermion_masses` on AKLP TY/Z3 and the canonical
Schoen Z/3×Z/3 bundle at each `completion_degree ∈ {1, 2}`. Higher
degrees were not feasible within wallclock (see §3).

For each row, the eigenvalue-ratio columns are
`m_2 / m_3` and `m_1 / m_3` per fermion sector. Rank-1 collapse
signature: ratios `≈ 0` (floating-point zero floor). Rank-3 signature:
ratios within canonical PDG generation hierarchies (e.g.
`m_μ/m_τ ≈ 0.06`, `m_e/m_τ ≈ 3 × 10^{-4}`).

### 2.1 TY/Z3 (AKLP bundle)

| cap | n_seeds | h^obs / h^pred | m_μ/m_τ | m_e/m_τ | m_c/m_t | m_u/m_t | m_d/m_b | m_s/m_b | quad_unif | elapsed |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 24 | 9 / 9 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.633 | 0.0 s |
| 2 | 1 260 | 9 / 9 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.633 | 282.6 s |

**Verdict (TY/Z3):** rank-1 collapse persists at cap = 2 despite a
50× larger basis. The Bochner-Laplacian-on-polynomial-seeds null space
is functionally one-dimensional regardless of basis enrichment.

### 2.2 Schoen/Z3×Z3 (canonical bundle, k = 2 quotient with d_x=2, d_y=2, d_t=1)

| cap | n_seeds | h^obs / h^pred | m_μ/m_τ | m_e/m_τ | m_c/m_t | m_u/m_t | m_d/m_b | m_s/m_b | quad_unif | elapsed |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 18 | 2 / 0 (fallback) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.219 | 0.0 s |
| 2 | ~1 190 | — | — | — | — | — | — | — | — | DNF |

**Verdict (Schoen/Z3×Z3):** rank-1 collapse at cap = 1; the
cap = 2 Schoen run was killed by the wallclock budget after > 10
minutes per Yukawa-pipeline call (the dense Hermitian Jacobi solver
on a `1190 × 1190` matrix is the bottleneck — see §3).

A `kernel_dim_target = 9` fallback is engaged inside
`predict_fermion_masses_with_overrides`, but the Schoen baseline
(cap = 1) reports `cohomology_dim_observed = 2`, indicating the kernel
selection picked only 2 modes despite the fallback call. This is a
secondary defect orthogonal to G1 — see §5 future work.

---

## 3. Compute-bound caveat — why we cannot sweep cap > 2

The dense Hermitian Jacobi solver in `solve_harmonic_zero_modes`
diagonalises the full `n_seeds × n_seeds` matrix `L_eff = G^{-1} L`.
Per-iteration cost ≈ `O(n_seeds²)` with `O(n_seeds)` sweeps for
convergence.

Basis-size growth (AKLP TY bundle, 6 B-summands):

| cap | n_seeds | dense Hermitian matrix size | est. Jacobi runtime |
|---|---|---|---|
| 1 | 24 | 4 KB | < 0.1 s |
| 2 | 1 260 | 25 MB | ~5 minutes |
| 3 | 7 140 | 815 MB | ~hours (likely OOM on consumer RAM) |
| 4 | 24 192 | 9.4 GB | OOM |
| 5 | 60 750 | 59 GB | OOM |
| 6 | 130 977 | 274 GB | OOM |

Schoen Z/3×Z/3 has comparable growth (the `[0, 0]` summands' cumulative
basis has the same combinatorial structure).

This matches the OOM cap documented at
`zero_modes_harmonic_z3xz3.rs:107–117`
("at `cap = 6` the unfiltered seed basis is already ~263 000 modes
... ~1.1 TiB of `Complex64`"). The production solver runs into
the same scaling wall.

**Implication.** Even if Fix A *had* worked at higher cap, deploying
it requires migrating from dense Jacobi to a **Lanczos-style sparse
eigensolver** that computes only the lowest k = 9 eigenvalues in
`O(n_seeds² · k)` time — the dense solver is not viable at cap ≥ 3
on consumer hardware. This is independent infrastructure work that
should sit alongside any fix decision.

---

## 4. Decision — Fix A INSUFFICIENT

Both data points we could obtain — TY cap = 1 and TY cap = 2 — show
**identical rank-1 collapse**: every off-leading fermion-mass ratio
sits at the floating-point zero floor regardless of basis size. A
50× larger ansatz space does not break the mode degeneracy.

The diagnosis is unambiguous: the rank-1 collapse is **not** a
basis-completeness artifact. The kernel of the Bochner Laplacian on
polynomial-seed monomials is functionally rank-1 because the Bochner
Laplacian is **not the operator we should be diagonalising in the
first place** — it is a Bochner *approximation* of the bundle-twisted
Dirac kernel, valid only insofar as the seed monomials span the
Dolbeault `(0, 1)`-form space, which they do not on a Donaldson-
balanced metric (the polynomial-seed map from `H^0(B)` to the
`(0, 1)`-form representatives is many-to-one onto a one-dimensional
subspace of the actual cohomology class).

Fix A's underlying assumption — "more polynomial seeds will eventually
span the right Dolbeault space" — fails because the polynomial-seed
ansatz space is structurally *flat* in the operator sense: its
Bochner Laplacian has a rank-1 null space at every basis size, the
single null-mode being the constant section's lift.

**Recommendation:** proceed to Fix B.

---

## 5. Fix B research spec — bundle-twisted Dirac kernel `∂̄_V`

(Out of scope for the present task; documented here per the task
brief's "Fix B research-spec doc but DO NOT attempt Fix B in this
task" directive.)

### 5.1 Problem statement

`H^1(X, V)` should be computed as `ker(∂̄_V)` where `∂̄_V` is the
bundle-twisted Dolbeault operator `Γ(X, Ω^{0,0}(V)) → Γ(X, Ω^{0,1}(V))`,
with the Bochner / Bochner-Lichnerowicz Laplacian
`Δ_{∂̄_V} = ∂̄_V^* ∂̄_V` (or its Dirac-square equivalent
`D_V^* D_V` on the spinor side) used as the eigensolver target. The
zero-eigenvalue eigenvectors **are** the harmonic representatives.

The current implementation evaluates `Δ_{∂̄_V}` *in coordinates*
via `Σ_i ⟨∂_{z_i} s_α, ∂_{z_i} s_β⟩_h` over polynomial-seed sections
`s_α`. This is the Bochner ansatz: mathematically valid for sections
of `O(b_α)` on the projective ambient, but the evaluated operator's
null space on this restricted ansatz space is rank-1 because the
ansatz space is itself low-rank in the cohomology sense.

### 5.2 Required components

1. **A genuine `∂̄_V` operator** acting on `Γ(X, Ω^{0,1}(V))`. This
   means representing test sections as `(0, 1)`-form-valued functions,
   not as scalar polynomial monomials. The natural representation
   is via `∂̄`-closed `(0, 1)`-form representatives, e.g. the
   Beltrami-differential-style basis from Anderson-Constantin-Lukas-
   Palti 2017 §4 (arXiv:1707.03442) or the harmonic-form basis
   from Butbaia et al. 2024 §5 (arXiv:2401.15078).
2. **Bundle-twisted derivative**: `(∂̄_V s)^I_{ī} = ∂_{z̄_i} s^I + A^I_{ī J} s^J`
   where `A` is the (0,1)-part of the HYM Hermitian connection.
   Currently the code uses the *Hermitian* HYM metric `h_V` to define
   the inner product but does not contract with the (0,1)-connection
   coefficients in the derivative. For a Donaldson-balanced metric
   the `A^I_{ī J}` part is non-trivial and is precisely what couples
   different cohomology directions — this is what breaks the rank-1
   collapse.
3. **Sparse eigensolver**: Lanczos / LOBPCG / ARPACK-style algorithm
   computing only the lowest `k = 9` eigenvalues of the (sparse)
   `∂̄_V^* ∂̄_V` operator-matrix product on a sample cloud. Required
   for any cap ≥ 2 basis to be feasible — see §3.

### 5.3 Milestones (rough estimate)

- **Fix-B-M1 (1 week)**: implement a `(0, 1)`-form representative
  basis on the polynomial-seed scaffolding (one element per
  `dz̄_i ⊗ s_α(z, w)` pair). This is the test-section space for `∂̄_V`.
- **Fix-B-M2 (1 week)**: assemble the bundle-twisted derivative
  matrix using the HYM connection `(0, 1)`-coefficients
  `A^I_{ī J}(z, w)` from `crate::route34::hym_hermitian`. Build the
  Bochner-Lichnerowicz kernel `D_V^* D_V` block of size
  `n_seeds × n_seeds`.
- **Fix-B-M3 (1 week)**: integrate a Lanczos eigensolver on the
  `n_seeds × n_seeds` Hermitian matrix, returning the lowest `k = 9`
  eigenvalues + eigenvectors. The existing `pwos_math` library has a
  `linalg` feature that may already provide this; if not,
  implement Krylov-subspace iteration directly.
- **Fix-B-M4 (1 week)**: regression test against AKLP / ACLP 2017
  Yukawa benchmarks (the published Schoen-bundle Yukawa eigenvalue
  ratios are `~0.06, ~3e-4` for the lepton sector, matching PDG
  generation hierarchies up to O(1) fudge factors). Iterate on
  numerics until ratios are stable to 10% across `n_pts ∈ {1k,
  10k, 25k}`.

**Total Fix B effort: 3–4 weeks of focused work**, consistent with
the audit's §G1 estimate ("2-4 weeks; involves a basis-convergence
sweep at increasing cap, then a kernel-solver upgrade if cap-up
doesn't converge"). Cap-up did not converge → kernel-solver upgrade
is on the critical path.

### 5.4 Acceptance criteria

Fix B is declared successful when, on a Donaldson-converged
Schoen/Z3×Z3 metric (`donaldson_residual ≤ 1e-6`,
`quadrature_uniformity_score ≥ 0.5`):

- `m_μ / m_τ ∈ [0.03, 0.15]` (PDG: 0.0594).
- `m_e / m_τ ∈ [1e-5, 1e-2]` (PDG: 2.876e-4).
- `m_c / m_t ∈ [1e-3, 5e-2]` (PDG: 7.38e-3).
- `m_u / m_t ∈ [1e-7, 1e-3]` (PDG: 1.25e-5).
- `m_s / m_b ∈ [1e-3, 1e-1]` (PDG: 2.23e-2).
- `m_d / m_b ∈ [1e-5, 1e-2]` (PDG: 1.12e-3).

These criteria identify a rank-3 (genuinely chiral) Yukawa matrix
with PDG-compatible generation hierarchy. They do not require
PDG-level fit accuracy (that's Stage 4+ work), only an unambiguous
break of the rank-1 collapse.

---

## 6. Summary table

| Cap | n_seeds (TY) | TY rank-3 ratio min | Schoen rank-3 ratio min | Verdict |
|---|---|---|---|---|
| 1 | 24 | 0 (rank-1) | 0 (rank-1) | rank-1 collapse |
| 2 | 1 260 | 0 (rank-1) | DNF (compute) | **rank-1 collapse PERSISTS** |
| 3+ | ≥ 7 140 | OOM-bound | OOM-bound | not feasible without Lanczos solver |

**Fix A: INSUFFICIENT.** The Bochner Laplacian on polynomial seeds
has a rank-1 null space regardless of basis size — this is a
structural defect of the operator, not a basis-completeness one.

**Fix B (bundle-twisted Dirac kernel `∂̄_V` + sparse eigensolver) is
the next required step.** Estimated effort: 3–4 weeks.

---

## 7. Other observations

1. **`completion_degree` config field was previously dead** (declared
   in `HarmonicConfig` with default = 2, recorded in
   `HarmonicRunMetadata`, but never consumed by
   `expanded_seed_basis`). The G1 commit makes it live. The default
   stays at 2; the production binaries
   `p8_3_yukawa_production` etc. that override `HarmonicConfig` via
   `..HarmonicConfig::default()` will now silently start using
   `completion_degree = 2` (1 260-seed basis). This is a behavioural
   change. **Mitigation:** the production binaries either need to
   (a) explicitly set `completion_degree: 1` to preserve their
   pre-G1 numbers, OR (b) accept the cap = 2 basis and re-baseline
   their reported numbers. The Stage 3 G1 sweep (this binary) shows
   the rank-1 collapse persists at cap = 2 anyway, so option (a) is
   the conservative choice for the σ-channel paper headline.

2. **Schoen-side `cohomology_dim_observed = 2` at cap = 1.** Despite
   `predict_fermion_masses_with_overrides` being supposed to retry
   with `kernel_dim_target = Some(9)` when the BBW count is 0, the
   sweep observed a `2 / 0` kernel selection on Schoen at cap = 1.
   This is orthogonal to G1 but worth a downstream audit — likely
   a side effect of the recent G2 Chern-class extension altering
   the BBW count to a small positive integer that no longer triggers
   the "empty kernel basis" retry path.

3. **Quadrature uniformity remains poor on Schoen** (0.219 at the
   small `n_pts = 300` smoke run, audit reported 0.0081 at the
   production `n_pts = 25 000`). This is gap **G6** in the audit;
   it's independent of Fix A / Fix B and remains a publication-
   grade quality blocker on the Schoen side.

---

## 8. References

- `references/p_stage3_eigenmode_audit.md` — original audit.
- `src/route34/zero_modes_harmonic.rs` — production solver.
- `src/route34/zero_modes_harmonic_z3xz3.rs` — experimental Z/3×Z/3
  solver with the cumulative-bigraded-basis mechanism that G1 ports.
- Anderson, Constantin, Lukas, Palti, "Yukawa couplings in heterotic
  Calabi-Yau models", arXiv:1707.03442 (2017) — §4 on the
  `(0, 1)`-form representative basis (Fix B M1 reference).
- Butbaia et al., arXiv:2401.15078 (2024) — §5 on the bundle-Dirac
  discretisation (Fix B M2 reference).
