# Stage 3 Fix B M1a — Raw enumeration scaffolding (post hostile-review reframing)

**Status:** implementation landed. Companion to the M0 reference at
`p_stage3_fix_b_m0_chern_curvature.md` and to the full M1+M2 design
spec at `p_stage3_fix_b_m1m2_detailed_spec.md`.

**Module:** [`src/route34/zero_one_form_basis.rs`].

**Tests:** [`tests/test_zero_one_form_basis.rs`] (integration);
in-module unit tests (lib).

## Honest scope statement (post hostile-review)

A hostile review of the prior framing of this milestone returned a
verdict of "structurally-incomplete". The work landed at commit
`2baa8dc3` is reframed here as **M1a — raw enumeration scaffolding**.
The original M1 framing claimed delivery of the bundle-twisted
(0,1)-form Dolbeault representative basis on Schoen `Z/3 × Z/3` with a
polystable monad bundle. That claim is rescinded.

What this module actually delivers:

1. Polynomial-seed enumeration on the Schoen ambient.
2. Raw (0,1)-form basis enumeration: products of seeds × `dz̄^a` ×
   frame indices.
3. Raw (0,0)-form basis enumeration for coboundary candidates.
4. Flat-coordinate ∂̄ kernel/image dimension computation via
   sample-value Vandermonde-style ranks.

What this module does NOT deliver (deferred to M1b/M2):

- **Bundle twist absent.** `MonadBundle.b_lines` is consulted only for
  its length (rank `n`); the line-bundle degrees `[d_1, d_2]` are not
  used in the ∂̄ action.
- **Curvature unused.** The `_curvature: Option<&ChernCurvatureCloud>`
  parameter on `compute_h1_basis` is plumbed through but never read.
- **No Schoen `Z/3 × Z/3` quotient.** Sample points are generic 8-D
  random complex; CICY constraints `(3,0,1)/(0,3,1)` and `Z/3 × Z/3`
  orbit identification are absent.
- **No real Wilson-line action.** The `(g_1, g_2)` label assignment is
  pure index arithmetic (`v_frame_idx % 3`, `antihol_dir % 3`), not
  the BHOP §6 `(W_1, W_2)` Cartan-vector action.
- **BHOP proxy is decorative.** The "BHOP rank-4 proxy" is four
  arbitrary line-bundle degrees `[[1,0],[0,1],[1,1],[2,0]]` chosen to
  fix the frame rank — it does not reproduce BHOP physics.

The raw enumeration + ∂̄-kernel-rank algorithm is combinatorially
sound and the existing tests pass honestly as algorithmic-invariant
checks. The work is reusable scaffolding for M2; the misleading
framing is what needed to change.

## Summary of what M1a delivers

For Fix B Milestone 1 (M1) the solver now provides:

* A polynomial-seed enumeration `PolynomialSeedSet` over the 8 ambient
  coordinates of the Schoen `CP^2 × CP^2 × CP^1` ambient.
* A raw (0,1)-form basis `ZeroOneFormBasis` of size `n_seeds × 3 × n`
  (3 antiholomorphic CY3 directions, `n = bundle.b_lines.len()` frame
  indices) that enumerates `ω = s_p(z) · z̄^a · dz̄^a ⊗ e_I`.
* A raw (0,0)-form basis `ZeroZeroFormBasis` whose `∂̄_V` image lives in
  a sample-value space that the (0,1)-basis values are projected onto
  (per ACLP 2017 §4 methodology).
* A driver `compute_h1_basis` that returns:
    - the H¹ Dolbeault representatives expressed as coefficient vectors
      on the raw (0,1)-basis;
    - per-representative Wilson-line character `(g_1, g_2) ∈ {0,1,2}²`
      via majority-vote over the rep's support;
    - per-representative SO(10) rep label (currently `Unassigned` —
      proper assignment requires the full Wilson-line projection that
      lives downstream of M2);
    - diagnostic ranks `n_kernel_d01`, `n_image_d00`, `n_raw_basis`.

## Algorithm

### Raw basis enumeration

```
ω_{(p, a, I)}(z, z̄)  =  s_p(z)  ·  z̄^a  ·  dz̄^a  ⊗  e_I
```

with `s_p` a holomorphic monomial of total degree ≤ d, `a ∈ {0, 4, 7}`
(the lowest-index coord of each Schoen CP factor — these are the three
antiholomorphic directions on the CY3), and `I ∈ {0, …, n-1}` the
B-line index of the section basis.

### `∂̄_V` action (holomorphic frame)

For ω with f^{I'}_{a'} = s_p(z) · z̄^{a'} · δ_{I, I'}:

```
(∂̄_V ω)^I_{a, b}  =  ∂_{z̄_a} f^I_b  −  ∂_{z̄_b} f^I_a
                  =  δ_{I, I'} (δ_{a, b} − δ_{b, a}) · s_p(z)
                  =  0 for all a ≠ b.
```

The diagonal-pair identity makes every raw basis element automatically
∂̄_V-closed. So `Z¹ ≃ raw basis` (kernel of D_01 = full raw basis).

### Coboundary image

For (0,0)-form `s = s_p(z) · z̄^c · e_I`:

```
(∂̄_V s)^I_b  =  s_p(z) · δ_{c, b}  · e_I
```

This produces a (0,1)-form WITH polynomial-only coefficient (no z̄
factor). The image of `∂̄_V` on Ω^{0,0} therefore lies in a different
function-space slice than the raw (0,1)-basis (which has z̄^a factors).

### `H¹` via sample-value space

We embed both spaces into a common `C^{n_pts × n × 3}` sample-value
space:

* `Φ`: row (k, I, b) = δ_{a, b} · z̄_k^a · s_p(z_k) (raw (0,1)-basis
  values).
* `B¹`: row (k, I, b) = δ_{c, b} · s_p(z_k) (image-of-coboundary
  values).

Compute `dim H¹ = rank(Φ) − rank(Φ ∩ B¹)` via:
1. SVD-based column-space basis of `Φ` (orthonormal `phi_basis`).
2. SVD-based column-space basis of `B¹` (orthonormal `b1_basis`).
3. Project each `phi_basis` column against `b1_basis` and against
   already-accepted H¹ reps; accept residuals with norm above
   `rank_tolerance`.

Numerical implementation (post Fix B M1.5) uses **sparse Lanczos via
`pwos-math::linalg::LanczosSolver`** with the matrix-free Pattern-A
strategy: the operator `−M† M` is applied without ever materialising
the Gram matrix, with two streaming passes over `M` per `apply_op`
invocation. The complex Hermitian operator is real-encoded via the
standard `[[Re, −Im], [Im, Re]]` blocking; eigenvalues of `M† M` are
recovered as the negatives of Lanczos's lowest-k Ritz returns.
`LanczosSolver::solve_generalised` is plumbed for the M2 Bochner
Laplacian where the inner-product Gram is non-trivial, but is unused
at M1 (`B = I`).

The previous Stage-3 implementation used dense Hermitian Jacobi
diagonalisation of an explicit `M† M` Gram (cost `O(N³)` plus
significant cache-unfriendly index updates per sweep).

**M1.5 status (Apr 2026)**: the Lanczos refactor delivers correct
results at d ∈ {0, 1} verified against the dense Jacobi reference
(all 11 lib unit tests + 9 active integration tests pass without
modification of test expectations). At d ≥ 2 the current
`k_target = 2 · n_cols` full-spectrum target does not terminate within
the default `max_iters` Krylov budget — the convergence-check cost
`O(m³)` per check after `2 · k_target` iterations dominates, and tail
eigenvalues near round-off cannot reach the relative residual bound.
The d=2 BHOP-rank-4 proxy and d=3 single-line bundle integration tests
are therefore re-`#[ignore]`d post-M1.5 with a clear M1.5b deferral
note. **M1.5b** (parameter tuning) will switch the SVD path to top-K
Lanczos with adaptive K growth — request only the eigenvalues we
actually consume (`λ > tol²`), grow K until smallest returned ≤ tol²,
which avoids the full-spectrum convergence struggle.

### Coefficient back-projection

H¹ representatives in sample-value space are converted back to raw-basis
coefficients via the normal-equation least-squares solve
`(Φ† Φ) c = Φ† ψ` with Tikhonov regularisation. M2 will consume these
coefficient vectors directly.

## Test results

All 11 unit tests + 9 integration tests pass as **algorithmic-invariant
checks** — they verify no crashes, monotonic dim growth, sensible
diagnostic output, and that error paths fire correctly. They do NOT
assert geometric `H¹` counts.

### Diagnostic counts (from `diagnostic_dump_h1_counts`)

At polynomial degree d=1, sample cloud n_pts=64. The "dim H¹" column
below is named for backward compatibility with the API field; per the
honest scope statement above, it is the **Vandermonde-style rank of
the sample-evaluation column space**, not the geometric Dolbeault
dimension of the bundle's first cohomology.

| Bundle | n_raw_basis | n_kernel_d01 | n_image_d00 | dim (Vandermonde rank) | geometric H¹ (NOT measured) |
|--------|-------------|--------------|-------------|------------------------|-----------------------------|
| trivial (n=4) | 108 | 108 | 108 | 108 | 12 = h¹(O^4) (would be) |
| O(1,0) (n=1) | 27 | 27 | 27 | 27 | not measured |
| BHOP-rank-4 proxy (n=4) | 108 | 108 | 108 | 108 | 48 (target — not measured) |

**Important**: the `dim` column is NOT a finite-d truncation of the
geometric H¹ converging to it as d grows. It is the rank of the
sample-evaluation column space of an ansatz whose linear algebra
ignores the bundle twist and the Schoen quotient entirely. By
construction (no bundle action, every raw element ∂̄-closed in flat
coordinates), the kernel is the full raw basis. The reported dim
equals the raw basis size minus the (also full) coboundary-image
overlap, which collapses to the raw basis size at low d because the
two value-space slices have a different `z̄^a`-factor structure.

**ACLP §4.4 does NOT apply.** That stabilisation argument addresses
under-counting at low d (missing high-degree representatives reaching
the geometric limit from below). This module **over-counts** by
reporting the full sample-evaluation rank with no bundle action — a
fundamentally different gap that no degree sweep will close until M2
introduces the bundle twist and the Schoen quotient.

The genuine 48-mode (post-Wilson) Atiyah-Singer count for BHOP
requires the bundle-twisted ∂̄_V (M2), the genuine BHOP section basis
(M2 adapter), the Schoen `Z/3 × Z/3` quotient (M2), and the proper
Wilson-line action (M2/M4). The work landed at M1a is reusable
scaffolding. **Fix B M1.5 (Apr 2026)** swapped the dense Hermitian
Jacobi solver for sparse matrix-free Lanczos
(`pwos-math::linalg::LanczosSolver`); correctness verified at
d ∈ {0, 1} against the dense Jacobi reference. The d ≥ 2 path is
infrastructure-ready but blocked on a full-spectrum convergence-target
parameter mismatch — M1.5b will switch to top-K adaptive growth.
**M2** then adds the geometric content (curvature term + bundle
twist + Schoen quotient).

### Wilson-line label distribution (index arithmetic, BHOP-rank-4 proxy at d=1)

The 108 raw modes distribute across the 9 `(g_1, g_2) ∈ {0,1,2}²`
label classes as below. **These are NOT BHOP §6 Wilson-line
characters.** They come from the index-arithmetic assignment
`g_1 = v_frame_idx % 3, g_2 = antihol_dir % 3`:

| (g_1, g_2) | count |
|-----------|-------|
| (0, 0) | 18 |
| (0, 1) | 18 |
| (0, 2) | 18 |
| (1, 0) | 9 |
| (1, 1) | 9 |
| (1, 2) | 9 |
| (2, 0) | 9 |
| (2, 1) | 9 |
| (2, 2) | 9 |

The 18:9:9 imbalance is a direct combinatorial consequence of the
index-arithmetic assignment with `n_frame = 4` (two frames share
`mod 3 = 0`: `v_frame = 0` and `v_frame = 3`). It is not a physical
character-rep distribution. The genuine BHOP rank-4 SU(4) Wilson-line
spectrum per BHOP-2005 §6 acts via the joint `(W_1, W_2)` Cartan
vectors on the bundle's E_8 representation, which is a different
object entirely. The 3:3:3 SO(10) generation pinning is M2/M4 work.

## Forward compatibility with M2

The `_curvature: Option<&ChernCurvatureCloud>` parameter on
`compute_h1_basis` is plumbed through but unused at M1a. The
underscore prefix on the parameter name records this honestly. M2 must
turn this into a USED parameter. M2's Weitzenböck Laplacian will
consume:

1. The raw-basis coefficient vectors from `representatives` —
   M2's bundle-twisted Bochner-Lichnerowicz Laplacian will act on this
   coordinate system. (Note: the coefficient vectors themselves are
   currently a Vandermonde-rank object, not a geometric ∂̄_V-cocycle
   coefficient set; M2 must add the bundle action that makes them
   geometric.)
2. The M0 `ChernCurvatureCloud` — M2's curvature term `i Λ_ω F · ψ`
   will couple through the M0 cloud at sample points. **This is the
   coupling that is currently absent from M1a.**
3. The Schoen `Z/3 × Z/3` quotient via CICY constraint enforcement
   and orbit identification on the sample cloud (also absent at M1a).
4. A BHOP-`MonadBundle` adapter or direct `BhopExtensionBundle`
   consumer (no representation today).

The raw-enumeration scaffolding from M1a is reusable as the basis
data-structure that M2 will populate with the missing physics.

## Risks and follow-ups

1. **Polynomial degree convergence**: M1 has been exercised at d=0
   and d=1 (post-M1.5 — both verified to match the dense Jacobi
   reference). Higher degrees scale as O(d^8 · n) raw basis size; the
   matrix-free Lanczos backend handles the operator-application cost
   well, but the full-spectrum convergence target stalls at
   `n_cols ≥ 540`. The d=2 BHOP proxy and d=3 line-bundle integration
   tests are `#[ignore]`d pending M1.5b's top-K adaptive K-growth.
   d=4 (where ACLP §4.4 predicts H¹ stabilisation) is the long-term
   target after M2 lands the bundle-twist content.

2. **Genuine BHOP bundle**: this M1 implementation works against any
   `MonadBundle` with a positive section basis. The genuine BHOP
   rank-4 SU(4) extension bundle (`BhopExtensionBundle` in
   `hidden_bundle.rs`) is purely topological data and does not have a
   matching `MonadBundle` representation in the current code. M2 will
   need either:
     * A new monad-style representation of the BHOP extension bundle
       in `MonadBundle` form, with a 4-element `b_lines`; or
     * A direct extension of M1 to consume `BhopExtensionBundle` via
       a "bundle adapter" trait.
   The proxy used in M1 tests (`b_lines = [[1,0],[0,1],[1,1],[2,0]]`)
   stands in for this — it has the right rank-4 frame structure but
   differs in detailed character assignment.

3. **Wilson-line label is index arithmetic, not a projection**: the
   `(g_1, g_2)` assignment in M1a is `v_frame_idx % 3`,
   `antihol_dir % 3`. This is **not** a Wilson-line projection in any
   geometric sense; it is a placeholder that lets the data-structure
   carry a `(g_1, g_2) ∈ {0,1,2}²` field. The genuine Wilson-line
   projection requires acting on the coefficient vector with the joint
   `(W_1, W_2)` representation from
   [`crate::route34::wilson_line_e8_z3xz3`], which lives at the M2/M4
   stage of the pipeline.

4. **Numerical dense-SVD scaling** [partially addressed by Fix B
   M1.5, Apr 2026; remainder deferred to M1.5b]:
   the Hermitian Jacobi at the heart of `column_basis_via_svd`
   previously scaled as `O(N³ · sweeps)`. M1.5 replaced both call
   sites (`column_basis_via_svd` for SVD rank determination,
   `solve_phi_coefficients` for the Tikhonov LS solve) with sparse
   matrix-free Lanczos via `pwos-math::linalg::LanczosSolver`,
   following Pattern A from the M1.5 spec (matrix-free `−M† M`,
   real-encoded complex Hermitian). Verified at d ∈ {0, 1}:
   * Lanczos results match the dense Jacobi reference to working
     precision; all 9 integration tests + 11 unit tests pass without
     modification of test expectations.
   * d=1 BHOP-rank-4 proxy (`n01 = 108`): ≪ 1 s.
   **Open at M1.5**: full-spectrum Lanczos at `n_cols ≥ 540` does not
   converge within the default `max_iters` budget — the
   `try_extract_converged` check cost (O(m³) per check after 2·k_target
   iterations) dominates and tail-eigenvalue residuals never reach the
   relative tol. d=2 BHOP and d=3 line-bundle tests are therefore
   `#[ignore]`d at M1.5 commit. **M1.5b plan**: switch SVD to top-K
   Lanczos with adaptive K growth (request only eigenvalues we keep,
   stop growing when smallest returned drops below `tol²`). This
   avoids requesting the full noise-floor end of the spectrum entirely
   and aligns the cost model with rank, not dimension.

## References

* ACLP 2017: arXiv:1707.03442, §4 (basis construction), §4.3 (closure
  projection), §4.4 (degree-stabilisation table).
* M0 doc: `references/p_stage3_fix_b_m0_chern_curvature.md`.
* Full M1+M2 spec: `references/p_stage3_fix_b_m1m2_detailed_spec.md`.
* G1 spec (precursor): `references/p_stage3_g1_fix.md`.
* Audit (rank-1 collapse diagnosis):
  `references/p_stage3_eigenmode_audit.md` §A.5 / §C.2.
