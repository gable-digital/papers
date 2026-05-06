# Stage 3 Fix B falsifiability test — Lanczos + cap=3 sweep (DEFERRED)

**Status:** **DEFERRED — sparse Lanczos infrastructure does not yet
exist in our codebase.** The cap=3 falsifiability test cannot be run
on the existing dense Jacobi solver within the 4-hour compute budget,
and the production `pwos-math::linalg` module does not expose a sparse
symmetric/Hermitian eigensolver that the cy3 stack could wire into the
existing `solve_harmonic_zero_modes` path.

This document records what was investigated, what was found, and the
~1-week implementation plan needed before the falsifiability test can
be executed. Without that wiring, the structural-degeneracy diagnosis
from G1 (commit `4bea323f`) cannot be cheaply falsified, so Fix B
remains the recommended path on the prior structural evidence (rank-1
collapse persists across the cap = 1 → cap = 2 50× basis growth, in
the same dense polynomial-seeded Bochner Laplacian).

**Test rationale (recap):** Hostile review on task `a7e1e8d7be27051bf`
flagged that the cap=3 sweep is *operator-agnostic with sparse
Lanczos* but *compute-bound with dense Jacobi*. If the rank-1 collapse
is structural (G1's diagnosis), cap=3-with-Lanczos still collapses
→ Fix B confirmed needed. If unexpectedly cap=3 breaks the rank-1
collapse, the diagnosis was a basis-size artefact, not structural,
and bumping production basis to cap=3 would be sufficient (saves
3-4 weeks of bespoke Fix B implementation).

This is the cheap *hours-not-weeks* test that should run BEFORE
committing to Fix B. It cannot run today; this doc records the
blocker.

---

## 1. What was investigated

### 1.1 Existing sweep harness

`src/bin/p_stage3_g1_basis_sweep.rs` already drives the
basis-convergence sweep across `HarmonicConfig::completion_degree ∈
{2, 3, 4, 5, 6}`. The harness is operator-agnostic — it just calls
`predict_fermion_masses` / `predict_fermion_masses_with_overrides` at
each cap, parses the resulting eigenvalue ratios, and applies the
rank-3 + 10% stabilisation criterion. No code changes were needed
to the binary itself; the blocker is downstream in
`solve_harmonic_zero_modes`.

### 1.2 Eigensolver inventory

Greps for `lanczos`, `arpack`, `eigsh`, `sparse_eigen`, `lobpcg`,
`krylov`, `implicitly_restarted` across:

- `book/scripts/cy3_substrate_discrimination/rust_solver/src/`
- `products/pwos/libs/pwos-math/src/`

returned **only** Lanczos-coefficient hits for the gamma-function
log-Γ approximation (Numerical Recipes §6.1) in
`route34/likelihood.rs`, `pdg.rs`, and `pwos-math/stats/chi2.rs`. The
single substantive cy3-stack reference is a `// would be` TODO in
`route34/killing_solver.rs`:

> "For larger bases an external Krylov-Schur eigensolver would be
> preferable…"

`products/pwos/libs/pwos-math/src/linalg/` exposes only:

- `gemm.rs` — dense GEMM
- `gpu.rs` — GPU dispatch
- `invert.rs` — dense matrix inverse
- `lu.rs` — LU factorisation
- `mod.rs` — linalg trait surface
- `tests/`

No symmetric/Hermitian dense or sparse eigensolver, no SVD, no
Krylov solver. `Cargo.toml` for the `cy3_rust_solver` crate adds no
external eigensolver crate either — there's no `spectra`, `arpack`,
`lobpcg`, `nalgebra-lapack`, `nalgebra-sparse`, or `faer`
dependency.

### 1.3 Current `solve_harmonic_zero_modes` eigensolver

`src/route34/zero_modes_harmonic.rs:486` defines
`hermitian_jacobi_n` — a dense complex Hermitian Jacobi rotation
solver. Configured via `HarmonicConfig` with
`jacobi_max_sweeps = 128`, `jacobi_tol = 1e-12`. Cost per call is
O(sweeps · n³) on an n × n complex Hermitian matrix.

**Cap=3 cost estimate:** with ≈ 7 140 seeds on AKLP/TY:

- Dense Hermitian matrix size: 7 140 × 7 140 × 16 B ≈ **815 MB**
  (fits in RAM but cache-thrashes hard).
- Jacobi work: 128 sweeps · 7 140³ ≈ **4.7 × 10¹³ FLOPs** for the
  L matrix alone, doubled for the Gram pre-conditioning. Single-
  threaded, this is in the 10-hour-plus regime; even with naive
  thread parallelism in the rotation loop (which the current code
  does not exploit) it is well outside the 4-hour budget. And we
  need to do this for **two** candidates (TY/Z3 and Schoen/Z3xZ3)
  at the production `n_pts = 25 000` quadrature size.

**Schoen at cap=3 is even worse.** With `quotient_order = 9` and
the Schoen basis growth pattern (see `p_stage3_g1_fix.md` Compute-
Bound Caveat), Schoen cap=2 already did not fit into the wallclock
budget — cap=3 there is squarely in dense-Jacobi-OOM territory.

### 1.4 Why Lanczos solves this

The Bochner Laplacian L on the polynomial-seed basis is sparse —
many bigraded monomial pairs have vanishing inner product because
their bidegree difference exceeds the bundle's quadrature support.
At cap=3 the L matrix is ~95% sparse (the cumulative bigraded
enumeration densely populates only the diagonal block of overlapping
bidegrees per B-summand pair).

Sparse implicitly-restarted Lanczos (or LOBPCG / Krylov-Schur):

- Stores L as a sparse CSR/CSC matrix → ~40 MB at cap=3 instead of
  815 MB.
- Computes only the smallest k eigenvalues (here k = 9 for AKLP
  AKLP, k = 9 for Schoen Z/3 × Z/3 also) → O(k · n · iters) work,
  not O(n³).
- Estimated cap=3 cost: < 30 minutes per candidate at production
  quadrature, well inside the budget.

The bottom k eigenpairs are exactly what
`solve_harmonic_zero_modes` needs (`auto_use_predicted_dim`'s
"smallest k" or `kernel_eigenvalue_ratio`'s "below cutoff"). Both
selection modes degrade gracefully to Lanczos's k-smallest output
without changing semantics.

---

## 2. Implementation plan (~1 week)

### 2.1 Wire a sparse Hermitian eigensolver into `pwos-math::linalg`

**Option A (recommended): pure-Rust Lanczos in `pwos-math`.**

1. Add `pwos-math/src/linalg/lanczos.rs` implementing
   *implicitly-restarted Lanczos with deflation* on a complex
   Hermitian sparse matrix represented as
   `pwos_math::sparse::CsrMatrix<Complex64>` (or, if no sparse
   matrix abstraction exists yet, also add one — start with CSR).

2. The standard reference is Lehoucq, Sorensen, Yang — *ARPACK
   Users' Guide* §4.1 (implicitly-restarted Arnoldi/Lanczos). For a
   pure Hermitian implementation we want IRLM (Implicitly-Restarted
   Lanczos Method) targeting the smallest-k eigenpairs, with
   selective full reorthogonalisation against converged Ritz
   vectors (Parlett-Scott 1979) to retain numerical stability under
   the cap=3 ill-conditioning observed in the dense Gram step.

3. Validate against the existing `hermitian_jacobi_n` on cap=1 and
   cap=2 problems — eigenvalues must agree to 1e-10 absolute, and
   the recovered nullspace must give the same Yukawa overlaps to
   the same tolerance. This validation is essential because the
   subsequent G1 sweep result hinges on getting the kernel
   subspace right.

4. Add a `pwos-math::linalg::SparseEigensolver` trait so future
   code (Lichnerowicz, hidden-bundle, Killing) can plug in too.

**Option B (faster, less control): add `spectra-rs` (or `faer-eig`)
as a dependency.**

`spectra-rs` is a Rust port of the Spectra C++ ARPACK alternative;
it exposes `Sparse-Symmetric-Eig::compute_smallest_k`. Adding it as
a Cargo dependency would skip the Lanczos write-up entirely — but
introduces a third-party dep that hasn't been HOL-validated or
subjected to our other compliance/audit pipelines, and `spectra-rs`
is real-symmetric only (would need a Hermitian-to-real-block
expansion, doubling the matrix size). Faer-eig has a sparse
Hermitian path but is similarly upstream.

**Recommendation:** Option A. The pwos-math discipline is to keep
core numerics in-house with HOL `#[proven]` annotations; pulling in
a third-party Krylov solver would break that invariant.

### 2.2 Refactor `solve_harmonic_zero_modes` to use it

`zero_modes_harmonic.rs` builds the Gram matrix and the L matrix
densely today. The conversion to sparse needs:

1. A sparsity-preserving Gram-orthogonalisation step. The current
   path inverts the Gram density via Cholesky (in
   `apply_gram_inverse`) and applies `U^{-†} L U^{-1}`. For sparse
   Gram, switch to either (a) generalised eigensolve `L v = λ G v`
   directly through a generalised-Hermitian Lanczos (preferred), or
   (b) sparse Cholesky / ICho + Krylov shift-invert.

2. A `SparseSeedBasisBuilder` that constructs L as sparse from the
   start (current code densely accumulates triple-overlap
   contributions; many of those entries are below numerical
   threshold and could be skipped at assembly time).

3. Cap-conditional dispatch: keep the dense Jacobi path bit-
   identically when `n_seeds < 1500` (so cap=1 and cap=2 reproduce
   exactly) and switch to Lanczos when `n_seeds >= 1500` or when a
   new `HarmonicConfig::eigensolver = SparseLanczos` flag is set.

### 2.3 Run the falsifiability sweep

Once 2.1 and 2.2 land:

```
cargo run --release --bin p_stage3_g1_basis_sweep -- \
  --n-pts 25000 \
  --k 3 \
  --iter-cap 100 \
  --donaldson-tol 1e-6 \
  --seed 12345 \
  --degrees 1,2,3 \
  --output output/p_stage3_lanczos_cap3.json
```

Cap=1 / cap=2 results must reproduce the existing
`p_stage3_g1_basis_sweep.json` bit-identically (validation). Cap=3
is the new data point.

### 2.4 Decision tree at completion

- **Cap=3 still rank-1** → confirms G1's structural-degeneracy
  diagnosis; commit to Fix B as currently scoped.
- **Cap=3 breaks rank-1** → falsifies the diagnosis; reroute to
  "production basis = cap=3 with sparse Lanczos" and skip Fix B's
  bespoke ∂̄_V kernel build.

Either outcome is valuable; the test is exactly the kind of
~1-week-effort-to-save-3-weeks gating decision worth running before
committing to Fix B.

---

## 3. Why this can't be skipped by running dense Jacobi at cap=3

We considered just running dense Jacobi at cap=3 with the existing
binary and a bigger compute budget, but:

1. The TY/Z3 candidate alone needs ≈ 10 hours of single-threaded
   Jacobi sweeps at cap=3 with `n_pts = 25 000`. Schoen is worse.
   This blows past both the 4-hour test budget and the
   dev-environment policy.

2. Even if we tolerate the wallclock, Jacobi at this size is
   numerically ill-conditioned: the off-diagonal Frobenius norm
   gets dominated by 7 140² ≈ 5 × 10⁷ near-zero entries that the
   sweep tries to rotate, yielding catastrophic cancellation
   accumulation. The current `jacobi_tol = 1e-12` is set for cap≤2
   problem sizes and would need its own validation pass at cap=3.

3. The hostile-review framing ("operator-agnostic with sparse
   Lanczos") is the *correct* test framing; running dense Jacobi
   would muddle the signal because we'd be reading both rank-1
   collapse (operator) and Jacobi-vs-Lanczos numeric drift
   (algorithm) at once. The point of the test is to isolate the
   former.

---

## 4. Recommendation for path planning

1. **Today:** keep Fix B on the critical path. The structural-
   degeneracy diagnosis from G1 remains the best available
   evidence (cap = 1 → cap = 2 50× basis growth did not break
   rank-1).

2. **Insert ~1 week of pwos-math Lanczos work BEFORE Fix B's
   bespoke ∂̄_V build.** This is the decision-of-record gate. If
   cap=3-with-Lanczos breaks rank-1, Fix B's 3-4-week bespoke
   build is unnecessary. If it doesn't, Fix B continues with
   stronger justification.

3. **Track this as a falsifiability gate**, not as an add-on.
   The pwos-math Lanczos is also useful downstream
   (Lichnerowicz, Killing, hidden-bundle eigensolvers all flagged
   "would be preferable" in their respective files), so the
   investment compounds beyond just this Fix B gate.

---

## 5. Files referenced

- `src/bin/p_stage3_g1_basis_sweep.rs` — existing sweep harness;
  no changes needed.
- `src/route34/zero_modes_harmonic.rs:486` — `hermitian_jacobi_n`
  dense eigensolver (current bottleneck at cap=3).
- `src/route34/zero_modes_harmonic.rs:895` — eigensolve call site
  (the swap-in point for the future Lanczos path).
- `products/pwos/libs/pwos-math/src/linalg/` — target location for
  the new `lanczos.rs`.
- `references/p_stage3_g1_fix.md` — original Fix A writeup
  (Compute-Bound Caveat there flags the same blocker).
- `references/p_stage3_eigenmode_audit.md` §A.5 / §C.2 — original
  G1 diagnosis.
- `references/p_stage3_fix_b_m1m2_detailed_spec.md` — the Fix B
  spec being gated by this test.

---

## 6. Commit and audit trail

This document is a *deferral note*, not a result. No source code
was modified. The next agent picking up this thread should:

1. Read this document end-to-end.
2. Confirm the eigensolver inventory is still accurate (greps in
   §1.2 above).
3. Pick Option A (pwos-math in-house Lanczos) per the
   recommendation in §2.1.
4. Implement, validate against dense Jacobi at cap≤2, then run
   the cap=3 sweep with `p_stage3_g1_basis_sweep`.
5. Replace this deferral document with one of:
   - `references/p_stage3_lanczos_cap3_confirms_fix_b.md`
   - `references/p_stage3_lanczos_cap3_falsifies_diagnosis.md`

Decision-tree outcomes B and C in the original task spec.
