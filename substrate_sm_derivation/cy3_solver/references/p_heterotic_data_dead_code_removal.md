# P-HETEROTIC-DATA — Dead-code removal: `CY3TopologicalData::{h11, h21, euler_characteristic}`

## Summary

Removed three internally-inconsistent dead-code fields from
`cy3_rust_solver::heterotic::CY3TopologicalData`. They had no effect on
the production pipeline but their values libelled the published
literature. Surviving fields: `c2_tm`, `quotient_order`.

## Bug (wave-1 hostile review)

`CY3TopologicalData::tian_yau_z3()` stored
`(h11=14, h21=23, euler_characteristic=-6)`. For any Calabi-Yau
three-fold, `χ = 2(h^{1,1} − h^{2,1})`. Substituting:
`2·(14 − 23) = -18 ≠ -6`. The `(14, 23)` pair is the upstairs cover
spectrum (Tian-Yau 1986); the χ = -6 is the downstairs free `Z/3`
quotient (Greene-Kirklin-Miron-Ross 1987, Nucl. Phys. B **278**), which
on its own has `(h^{1,1}, h^{2,1}) = (6, 9)`. The constructor mixed
the two conventions.

`CY3TopologicalData::schoen_z3xz3()` stored
`(h11=19, h21=19, euler_characteristic=-6)`. The Schoen Calabi-Yau
three-fold has `χ_top = 0` per Schoen 1988 (Math. Z. **197**),
Theorem 0, BOTH on the upstairs fiber-product variety AND on the free
`Z/3 × Z/3` quotient (the quotient has `(h^{1,1}, h^{2,1}) = (3, 3)`,
giving `χ = 0`). The stored `χ = -6` directly contradicts the
primary source.

## Grep evidence: fields were dead

Field-level grep for `\.h11\b`, `\.h21\b`, `\.euler_characteristic\b`
inside the three live consumers of `CY3TopologicalData`:

```
src/route34/hidden_bundle.rs   : 0 matches
src/route34/bundle_search.rs   : 0 matches
src/bench_pipeline.rs          : 7 matches, all on `c.<field>` where
                                  `c` is a `Candidate`, NOT `CY3TopologicalData`
                                  (lines 116, 120, 246, 329, 429, 547, 648-666:
                                  CLI-arg overrides + Candidate-struct reads)
```

The `bench_pipeline.rs` matches `c.euler_characteristic`, `args.h11`,
and `family.h11()`/`family.h21()` are on the `Candidate` struct (lines
85-100 of bench_pipeline.rs) and a `family` helper, not on
`CY3TopologicalData`. The only `CY3TopologicalData`-typed values in
that file (lines 196-203) call only the constructors. No live consumer
read the bad fields.

## Fix applied

Per CLAUDE.md "Handling Unused Code": verified, prior state already in
Git (commit `9b22b354` is HEAD), removed the three fields:

- `src/heterotic.rs`:
  - Removed `pub euler_characteristic: i32`, `pub h11: i32`,
    `pub h21: i32` from `struct CY3TopologicalData`.
  - Removed corresponding initializers from `tian_yau_z3()` and
    `schoen_z3xz3()`.
  - Updated doc-comment to reflect the field-set change and cite
    `geometry::CicyGeometry` as the authoritative Hodge / Euler source.

- `tests/test_heterotic_data_consistency.rs` (NEW, 75 lines):
  - Three regression tests on the surviving (`c2_tm`, `quotient_order`)
    field set.
  - Compile-time guard via struct-literal construction: re-introducing
    `h11`/`h21`/`euler_characteristic` would cause this test to fail
    to compile.
  - File-level doc-comment describes the bug history with primary-source
    citations (Schoen 1988 Theorem 0; GKMR 1987 §3) and the rationale
    for removal vs correction.

Authoritative downstairs Hodge / Euler data is sourced from
`geometry::CicyGeometry`, exercised by the `test_ty_hodge.rs` and
`test_z3xz3_invariants.rs` integration tests.

## Verification

- `cargo build --release`: ok (only pre-existing dead-code warnings on
  unrelated binaries).
- `cargo test --release --lib`: **745 passed; 0 failed; 62 ignored**.
- `cargo test --release --test test_heterotic_data_consistency`:
  **3 passed; 0 failed**.
- All other integration tests pass except two pre-existing failures
  (NOT caused by this change — verified by `git stash` reproduction):
  - `test_schoen_bundle.rs` references the long-removed
    `MonadBundle::aglp_2011_schoen_su4_example()` constructor.
  - `test_bf_respects_convergence_flag.rs` is a wave-1 NEG-BF test
    awaiting a `from_chain_match` `converged` check (task #270).
  - `test_p5_10_determinism.rs` is the rayon-scheduling
    bit-exactness drift (task #243).

## Out-of-scope observations (NOT fixed in this commit)

- The doc-comment in the original struct cited "Braun-He-Ovrut-Pantev
  2005" for Schoen `(h11=19, h21=19)`. BHOP 2005 in fact discusses the
  Z/3 × Z/3 quotient with downstairs `(3, 3)`. The combined `(19, 19)`
  upstairs values appear in the broader Schoen-fiber-product literature
  (DHOR 2006, Donagi-He-Ovrut-Reinbacher). Not patched because the
  field is now gone.
- `bench_pipeline.rs` keeps a `Candidate.euler_characteristic` field of
  its own. Audit did not flag it. Out of scope.

## References

- Tian-Yau 1986: *Three-dimensional algebraic manifolds with `c_1 = 0`
  and `χ = -6`*, in *Math. Aspects of String Theory* (S.-T. Yau, ed.),
  World Scientific (1987) pp. 543-559.
- Greene, Kirklin, Miron, Ross 1987: *A three-generation superstring
  model. I.*, Nucl. Phys. B **278** 667-693, §3.
- Schoen 1988: *On fiber products of rational elliptic surfaces with
  section*, Math. Z. **197** (1988), Theorem 0.
- Braun, He, Ovrut, Pantev 2005: heterotic standard model on Schoen
  Z/3 × Z/3 quotient.
