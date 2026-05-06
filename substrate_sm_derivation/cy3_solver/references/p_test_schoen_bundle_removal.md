# Removal of `tests/test_schoen_bundle.rs` (wave-3 unblock)

## Summary

`tests/test_schoen_bundle.rs` was deleted as part of the wave-3 hostile-review
unblock pass. It called `MonadBundle::aglp_2011_schoen_su4_example()`, a
constructor that has never existed in `src/zero_modes.rs` (verified via
`git log --all -S 'aglp_2011_schoen_su4_example' -- src/zero_modes.rs`,
which returns no results — the only commit touching that symbol anywhere
in the tree is `7bd0bcef`, which introduced the test file itself with
the unimplemented call).

The test was an aspirational stub: it asserted properties of a hypothetical
rank-4 SU(4) Schoen monad bundle that was never constructed, and was
written against an earlier API design where `MonadBundle.b_lines` /
`c_lines` had variable arity (`Vec<Vec<i32>>`-style). The current type
is `Vec<[i32; 2]>` (compile-time-fixed arity 2) plus an optional
`b_lines_3factor: Option<Vec<[i32; 3]>>` lift — so the test's
`assert_eq!(b.len(), 3)` is unsatisfiable at the type level.

## What the test asserted

1. `schoen_su4_bundle_has_rank_four` — `MonadBundle::aglp_2011_schoen_su4_example().rank() == 4`
2. `schoen_su4_bundle_c1_vanishes` — `c_1(V) == 0` on the Schoen ambient
3. `schoen_su4_bundle_arity_three` — `b_lines[i].len() == 3` and
   `c_lines[i].len() == 3` (3-factor multidegrees)
4. `schoen_su4_bundle_b_c_count` — 6 B-summands and 2 C-summands (rank 6 − 2 = 4)
5. `ty_and_schoen_bundles_are_distinct` — TY and Schoen bundles type-distinct via arity

## Why removal is safe

The Schoen bundle structural properties are covered by:

- **`src/zero_modes.rs::tests::schoen_z3xz3_canonical_returns_three_factor_bundle`**
  (lib unit test, line ~2436). Asserts:
  * `b_lines_3factor` is `Some(_)` and length matches `b_lines.len()`
  * At least 3 distinct Wilson phase classes under `(a − b) mod 3`
  * `rank() == 3` (SU(3), the actual Schoen bundle constructed —
    not the SU(4) the deleted test wrongly asserted)
  * `c_1(B) == c_1(C)` on the 2-factor projection (SU(n) condition)
- **`tests/test_schoen_hodge.rs`** (POS-CY3-1) — covers the Schoen
  Calabi-Yau topological invariants (`h^{1,1}`, `h^{2,1}`, `χ`)
  with primary-source citations to Schoen 1988, BHOP 2005
  (hep-th/0501070), and DOPR.

The current real Schoen bundle is `schoen_z3xz3_canonical`, an SU(3)
monad with b_lines / c_lines structured as:

```text
B = O(1,0,0)² ⊕ O(0,1,0)² ⊕ O(0,0,1)²            (rank 6)
C = O(1,1,0)  ⊕ O(0,1,1)  ⊕ O(1,0,1)             (rank 3)
rank V = 6 − 3 = 3                                (SU(3))
```

Per `47cf3f06 feat(cy3): adaptive damping + 3-factor Schoen bundle +
Yukawa diagnostics` (P8.3-followup-B). The deleted test's claim of
"6 B-summands and 2 C-summands (rank 6 − 2 = 4)" is therefore
inconsistent with the current bundle anyway.

## Why removal (Path B) over migration (Path A)

Migration would require:

1. Changing `MonadBundle::b_lines` / `c_lines` from `Vec<[i32; 2]>` to
   `Vec<Vec<i32>>` — wide-blast type refactor across the route34
   pipeline.
2. Either constructing a new genuine rank-4 SU(4) Schoen bundle
   (a research task with no published source for the exact bidegrees
   the test asserts) or rewriting every assertion against the current
   rank-3 SU(3) bundle, with no clear physics motivation.
3. Replacing the never-cited "AGLP 2011" reference with primary
   sources. The legitimate Schoen-bundle reference is BHOP 2005
   ("A Heterotic Standard Model", hep-th/0501070), which §4 explicitly
   constructs a **rank-4 vector bundle V with structure group SU(4)**
   whose commutant in E_8 is SO(10) — matching the SU(4) rank the
   deleted test asserted, but with substantively different bidegrees.
   The residual breaking SO(10) → G_SM is then driven by Wilson lines
   on the freely-acting Z_3 × Z_3 quotient. (Cross-reference:
   `references/p_e6_su3_breaking_audit.md` §"Fix path option (b)",
   which correctly identifies BHOP §3-4 as a rank-4 SU(4) construction.)
   Earlier text in this doc that called BHOP "an SU(5) bundle" was a
   citation error and has been corrected here.

Path B is the maintainer choice because the test is structurally
broken at the type level AND its physics target (SU(4) Schoen for
SO(10)×SU(4) embedding) is not what the current codebase models.

## Wave-3 hostile reviewers who flagged the blockage

Three independent reviewers identified
`cargo test --release --tests --no-run` failing on this file as
blocking the wave-2 NEG-BF regression suite (per the wave-3 unblock
brief). The compile error was masking the 4 expected NEG-BF failures
and any other regressions.

## Verification

After deletion:
- `cargo test --release --tests --no-run` compiles cleanly.
- `cargo test --release --lib` continues to pass (Schoen bundle
  coverage retained via `schoen_z3xz3_canonical_returns_three_factor_bundle`).
- The wave-2 NEG-BF regression suite is now visible.
