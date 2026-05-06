# Stage 3 G2 — `MonadBundle::chern_classes` 3-factor Schoen extension

**Status**: LANDED. The Whitney-decomposition Chern-class accessor
[`zero_modes::MonadBundle::chern_classes`] now handles 3-factor
ambients (Schoen `CP² × CP² × CP¹`) in addition to the existing
2-factor (Tian-Yau `CP³ × CP³`) path. A new accessor
[`MonadBundle::chern_class_2_per_kahler`] returns the full per-Kähler-
generator integrals `∫_X c_2(V) ∧ J_a` for the Bianchi residual
diagnostic.

**Author**: Stage 3 G2 implementation pass, 2026-05-04.

## What was done

1. **New struct field** `MonadBundle::c_lines_3factor: Option<Vec<[i32;
   3]>>` mirrors the existing `b_lines_3factor` on the C side of the
   monad, populated for `schoen_z3xz3_canonical` with the documented
   `[1,1,0], [0,1,1], [1,0,1]` line-bundle classes.

2. **`chern_classes` rewrite** — replaced the gated 2-factor Whitney-
   cofactor expansion with a uniform implementation that uses the
   geometry's `triple_intersection` form directly. The new
   implementation factorizes `c_2(V)` as a list of line-class pairs
   instead of expanding into a monomial basis, which makes the formula
   independent of the ambient factor count `nf`. For nf == 2 the
   integer summary is `∫_X c_2(V) ∧ (J_1 + J_2)` (semantically
   different from the prior "scalar proxy" `Σ_{i<j} d_i · d_j` but
   physically more meaningful). For nf == 3 the implementation
   delivers the previously-stubbed Schoen integers.

3. **New accessor** `chern_class_2_per_kahler` returns `[∫ c_2(V) ∧
   J_1, ∫ c_2(V) ∧ J_2, ∫ c_2(V) ∧ J_T]` for direct comparison against
   the published `c_2(TX̃) = [36, 36, 24]`
   (`route34::schoen_geometry::PUBLISHED_C2_TM_INTEGRALS`,
   Donagi-He-Ovrut-Reinbacher 2006 Eq. 3.13–3.15).

4. **Five regression tests** in
   `tests/test_chern_classes_3factor_schoen.rs` pin every Whitney-
   decomposition output against an explicit hand computation
   reproduced in the test doc-comment.

## Numerical results — `MonadBundle::schoen_z3xz3_canonical`

Bundle data:

```text
B = O(1,0,0)² ⊕ O(0,1,0)² ⊕ O(0,0,1)²    (rank 6)
C = O(1,1,0)  ⊕ O(0,1,1)  ⊕ O(1,0,1)     (rank 3)
rank(V) = 3, c_1(V) = (0, 0, 0)          (SU(3) ✓)
```

Schoen non-zero triple intersections (DHOR 2006 Eq. 3.7):
`J_1²J_2 = J_1 J_2² = 3`, `J_1 J_2 J_T = 9`.

Computed Chern data:

| Quantity                              | Computed   | Framework commitment        |
|---------------------------------------|------------|-----------------------------|
| `c_1(V)` summary                      | `0`        | `0` (SU(3) check)           | ✓
| `∫ c_2(V) ∧ J_1`                      | `12`       | `36` (= c_2(TX̃)_1)          | ✗
| `∫ c_2(V) ∧ J_2`                      | `12`       | `36` (= c_2(TX̃)_2)          | ✗
| `∫ c_2(V) ∧ J_T`                      | `9`        | `24` (= c_2(TX̃)_T)          | ✗
| `∫_X c_3(V)`                          | `-6`       | `±54` (3 generations after Z/3×Z/3) | ✗
| `n_gen = \|c_3(V)\| / (2 \|Γ\|)`      | `6/18 = 1/3` | `3` (heterotic SM target) | ✗
| Bianchi residual `c_2(V) − c_2(TX̃)`   | `(-24, -24, -15)` | `(0, 0, 0)` (standard embedding) | ✗

Hand computations reproduced verbatim in the test doc-comments at
`tests/test_chern_classes_3factor_schoen.rs`.

## Discrepancy analysis

### `c_2(V) ≠ c_2(TX̃)`

Standard embedding requires `c_2(V) = c_2(TX̃)` exactly. The bundle
above gives `(12, 12, 9)` against published `(36, 36, 24)` — exactly
1/3 on the J_1, J_2 entries and 9/24 = 3/8 on J_T. The bundle is NOT
a standard embedding of the tangent bundle.

### `|c_3(V)| ≠ 54`

`src/heterotic.rs:283` and `src/heterotic.rs:572` commit to
`|c_3(V)| = 54` so that `n_gen = 54 / (2 · 9) = 3` after the free
Z/3 × Z/3 quotient. The bundle above gives `|c_3(V)| = 6`, yielding
`n_gen = 1/3` — a fractional generation count, which is unphysical.

### Bianchi cannot close with a physical 5-brane class

The residual `c_2(V) − c_2(TX̃) = (-24, -24, -15)` is NEGATIVE in every
Kähler direction. The heterotic Bianchi identity allows
`c_2(V) − c_2(TX̃) − [W] = 0` only when `[W]` is a positive-tension
class (5-branes wrapping holomorphic curves carry positive charge).
A negative residual cannot be cancelled by physical 5-branes.

## Resolution paths (author decision required)

The bundle as currently coded does not realize the framework's
heterotic standard-model commitments on Schoen `Z/3 × Z/3`. The
honest paths forward, in priority order:

1. **Replace `MonadBundle::schoen_z3xz3_canonical` with the
   BHOP-2005 §6 rank-4 SU(4) extension bundle.** Its Chern data
   (already in `route34::hidden_bundle::BhopExtensionBundle::published`)
   gives `c_2(V) = -2 τ_1² + 7 τ_2² + 4 τ_1 τ_2`, `c_2(TX̃) = 12 τ_1²
   + 12 τ_2²`, residual `c_2(TX̃) − c_2(V) = 6 τ_1²` — POSITIVE,
   supporting the published `[W] = 6 τ_1²` 5-brane class (BHOP Eq.
   99). The polystability witness is already in
   `tests/test_schoen_polystability.rs`. This is the path the existing
   audit (`references/p_anomaly_cancellation_audit.md`) most strongly
   endorses.

2. **Search the rank-3 SU(3) monad space on the 3-factor Schoen
   ambient** for a bundle that achieves `c_2(V) = (36, 36, 24)`
   exactly (the standard-embedding path). The current 6-line-bundle
   configuration (`O(1,0,0)² ⊕ O(0,1,0)² ⊕ O(0,0,1)²` for B,
   `O(1,1,0) ⊕ O(0,1,1) ⊕ O(1,0,1)` for C) does not satisfy that;
   richer line-bundle catalogues might.

3. **Retract the `n_gen = 3` framework commitment on this specific
   bundle** and re-label `schoen_z3xz3_canonical` as a "Yukawa-pipeline
   structural placeholder" only — its current role in the codebase
   is to provide Wilson Z/3×Z/3 phase-class buckets for the harmonic-
   mode counting pipeline, NOT to be a heterotic compactification.

The five regression tests in
`tests/test_chern_classes_3factor_schoen.rs` lock in the bundle's
ACTUAL Chern data (not the framework's claimed values), so any future
attempt to "fix the value" by tweaking line-bundle multiplicities
without changing the bundle's structural content will be caught
immediately. When the author decision lands and a new bundle replaces
`schoen_z3xz3_canonical`, the tests should be updated to assert the
new bundle's actual values.

## Polystability of the rank-3 SU(3) bundle

Polystability for `MonadBundle::schoen_z3xz3_canonical` is **not**
exercised in this commit. The existing `check_polystability` machinery
(`route34::polystability::check_polystability`) consumes the
`heterotic::MonadBundle` type with scalar `b_degrees`/`c_degrees` —
not the bidegree-encoded `zero_modes::MonadBundle` type that holds
the 3-factor data. Plumbing the rank-3 bundle through requires
either translating the bidegrees lossily to single integers (which
loses the per-factor structure essential to polystability on a
multifactor ambient) or extending `check_polystability` to consume
the bidegree representation.

Either extension is substantial and is deferred. The bundle's failure
to satisfy Bianchi cancellation with a positive 5-brane class makes
the rank-3 SU(3) polystability question moot anyway: a bundle that
does not close anomaly cancellation is not a heterotic compactification
candidate, regardless of whether its slope conditions hold on some
sub-cone of the Kähler moduli space.

## Files changed

* `src/zero_modes.rs` — extended `MonadBundle::chern_classes` to
  arbitrary `nf`; added `MonadBundle::c_lines_3factor` field and
  `MonadBundle::chern_class_2_per_kahler` accessor.
* `src/route34/eta_evaluator.rs`,
  `src/route34/hym_hermitian.rs`,
  `src/route34/zero_modes_harmonic.rs` — added
  `c_lines_3factor: None` to MonadBundle constructors at every
  site (5 + 2 + 2 = 9 sites).
* `tests/test_chern_classes_3factor_schoen.rs` — five regression
  tests (this commit).
* `references/p8_3_followup_b_chern_3factor_extension.md` — this
  reference (this commit).
