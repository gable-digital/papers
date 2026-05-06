# Audit: REM-OPT-B P1.1 anomaly-cancellation regression test BLOCKED

**Status**: BLOCKED. The asserted regression test
`tests/test_schoen_anomaly_cancellation.rs` cannot be written truthfully
against the existing `MonadBundle::schoen_z3xz3_canonical()` — the
bundle has no programmatic `c_2(V)` accessible, and there is no
asserted 5-brane class `[W]` data in code for it.

**Auditor**: REM-OPT-B P1.1 implementation pass, 2026-05-04.

## Plan vs. reality

The plan calls for:

> Add `tests/test_schoen_anomaly_cancellation.rs`:
> - Load the canonical Schoen monad `MonadBundle::schoen_z3xz3_canonical()`.
> - Compute `ch_2(V)` from the monad (programmatic), `ch_2(TX)` from the
>   Schoen tangent bundle, and the asserted 5-brane class `[W]`.
> - Assert `ch_2(V) - ch_2(TX) - [W] = 0` cohomologically, with a
>   numerical tolerance documented.
> - The test should fail if the bundle is altered in a way that violates
>   Bianchi.

Each input listed is, in this codebase, in the following state:

### `c_2(V)` from `schoen_z3xz3_canonical` — not programmatically available

`zero_modes::MonadBundle::schoen_z3xz3_canonical()` builds a rank-3
SU(3) monad on the Schoen 3-factor ambient `CP² × CP² × CP¹` with

```text
B = O(1,0,0)² ⊕ O(0,1,0)² ⊕ O(0,0,1)²
C = O(1,1,0)  ⊕ O(0,1,1)  ⊕ O(1,0,1)
```

The Whitney-decomposition Chern formula
`zero_modes::MonadBundle::chern_classes(&AmbientCY3)` is gated to
2-factor ambients (`if nf != 2 { return (0, 0, 0); }` at
`src/zero_modes.rs:803-805`), so calling it with
`AmbientCY3::schoen_z3xz3_upstairs()` (which is 3-factor) returns
the stub triple `(0, 0, 0)`.

The constructor's own doc-comment (`src/zero_modes.rs:398-408`)
flags this as a follow-up:

> The 2-factor `Self::chern_classes` formula is gated to `nf == 2`
> and silently returns `(0, 0, 0)` for 3-factor ambients, so the
> c₁ / c₂ / c₃ integers reported by
> `chern_classes(&schoen_z3xz3_upstairs())` on this bundle are
> `(0, 0, 0)` — a stub. … a follow-up task **P8.3-followup-B-Chern**
> is required to extend the Whitney-decomposition formula to
> 3-factor ambients before any topology-dependent invariant
> (generation count, stability, Bianchi gauge anomaly) on this
> bundle is trusted.

That follow-up has not landed. There is therefore no honest
programmatic `c_2(V)` to feed into a Bianchi residual.

### `c_2(TX)` for Schoen X̃ — available

`route34::schoen_geometry::SchoenGeometry::c2_tm_vector()` returns
`[36, 36, 24]` in the `(J_1, J_2, J_T)` basis, matching
Donagi-He-Ovrut-Reinbacher 2006 Eq. 3.13–3.15
(`src/route34/schoen_geometry.rs:592-629`). This input is solid.

### Asserted 5-brane class `[W]` — does not exist for this bundle

A grep of the workspace for
`w_class | ns5 | brane_class | fivebrane_class | brane_charge` yields
only doc-comment mentions and the BHOP rank-4 `BhopExtensionBundle`
machinery (see below). There is no `5_brane_class()`,
`fivebrane_class`, or any equivalent accessor on
`zero_modes::MonadBundle::schoen_z3xz3_canonical()` or its containing
type. The Tian-Yau research log
(`references/p_ty_bundle_research_log.md:486`) cites
"5-brane W charges = (36, 18)" for a TY/Z3 search and BHOP-style
"(27, 27)" for a different bundle, but neither is the Schoen canonical
case and neither is encoded in `MonadBundle`.

### What DOES carry a Bianchi-closing dataset on Schoen

`route34::hidden_bundle::BhopExtensionBundle::published()`
(`src/route34/hidden_bundle.rs:741-755`) carries the genuine
**BHOP-2005 §6.1-6.2** rank-4 SU(4) extension bundle Chern data:

```text
c_2(V)  =  -2 τ_1²  +  7 τ_2²  +  4 τ_1 τ_2     (BHOP Eq. 98)
c_2(H)  =   8 τ_1²  +  5 τ_2²  -  4 τ_1 τ_2     (BHOP Eq. 95-96)
c_2(TX̃) = 12 τ_1²  + 12 τ_2²                   (BHOP Eq. 94)
```

with `BhopExtensionBundle::anomaly_residual()` returning
`c_2(TX̃) - c_2(V) - c_2(H) = 6 τ_1²` (BHOP Eq. 99 — the residue
"cancelled by 5-branes wrapping `PD(τ_1²)`"). This is a different
bundle (rank-4 SU(4), not rank-3 SU(3)) on the same geometry. It
does close anomaly cancellation modulo a published, named 5-brane
class, but it is **not** what `MonadBundle::schoen_z3xz3_canonical()`
returns.

### Cross-reference: existing audits reach the same conclusion

- `references/p_e6_su3_breaking_audit.md` (round-3 hostile review,
  2026-05-03): documents that the bundle is "rank-3 SU(3)
  standard-embedding on Schoen geometry" — author decision **P1.3**
  (rank-3 vs rank-4 BHOP vs rank-5 DOPR vs line-bundle SM) is open.
- `references/p_schoen_variant_pin.md`: pins
  `SCHOEN_BUNDLE_VARIANT = "TY-standard-embedding-on-Schoen-geometry
  (P1.3 author-decision-pending)"`.
- `references/p_ty_bundle_audit.md`: parallel finding for the TY/Z3
  side — no published 3-generation line-bundle or monad SM bundle
  exists on Picard-rank-2 TY/Z3 (AGLP-2012 explicitly excluded that
  Picard rank).

The honest reading: production builds `schoen_z3xz3_canonical` as a
**structurally appropriate placeholder for the Yukawa-bucket pipeline**
(it spans the right Wilson Z/3 phase classes and feeds harmonic-mode
counting), not as a compactification certified by Bianchi. Its
Chern-class machinery is intentionally stubbed out and its `[W]`
class is not asserted.

## Why a regression test is not write-able right now

Three options to write `test_schoen_anomaly_cancellation.rs`:

1. **Use `MonadBundle::chern_classes(&schoen_z3xz3_upstairs())`** to
   get `c_2(V)`. This returns `(0, 0, 0)` (the documented stub),
   so the residual `c_2(TX̃) - c_2(V) - [W]` reduces to
   `c_2(TX̃) - [W] = 36 τ_1² + 36 τ_2² + 24 τ_1 τ_2 - [W]`. Any
   `[W]` we pick to make that vanish is a value invented by the
   author of this test — that is "rigging the test to pass" and is
   forbidden by the plan ("Do not commit a test rigged to pass").

2. **Compute `c_2(V)` by hand** from the splitting principle on
   the 3-factor ambient and hardcode it in the test. That is
   re-deriving the machinery the production code does not have.
   It would replicate the P8.3-followup-B-Chern work outside the
   production code and lock the answer in a test fixture, which
   is precisely the wrong direction (the test is supposed to
   exercise production code, not duplicate it). It would also
   fabricate an `[W]` for which there is no primary-source
   citation on this specific bundle.

3. **Substitute `BhopExtensionBundle::published()`** for
   `MonadBundle::schoen_z3xz3_canonical()` in the test. That tests
   a different bundle (rank-4 SU(4) BHOP-2005, not rank-3 SU(3)
   standard-embedding), so the test does not exercise the bundle
   the plan names. It also does not satisfy the plan's
   "fail if the bundle is altered in a way that violates Bianchi"
   acceptance criterion for `schoen_z3xz3_canonical`.

None of (1)/(2)/(3) is a faithful implementation of the plan.

## What would unblock REM-OPT-B P1.1

Either (in priority order):

1. **Land P8.3-followup-B-Chern** — extend
   `MonadBundle::chern_classes` to 3-factor ambients via the
   Whitney-cofactor expansion (already sketched at
   `src/zero_modes.rs:858-913` for `nf == 2`), AND choose / cite
   an asserted `[W]` 5-brane class on this specific bundle, AND
   verify Bianchi closes against `c2_tm_vector() = [36, 36, 24]`.
   Then write the regression test against the resulting
   `chern_classes()` output.

2. **Resolve P1.3** in favour of (b) BHOP-2005 rank-4 SU(4):
   replace `schoen_z3xz3_canonical()` with a constructor that
   returns the BHOP `BhopExtensionBundle`-equivalent monad data
   (the line-bundle catalogue side; numerical Chern data already
   in `BhopExtensionBundle::published()`). The Bianchi residual is
   then `BhopExtensionBundle::anomaly_residual() = (6, 0, 0)` in
   the `(τ_1², τ_2², τ_1 τ_2)` basis, with `[W] = 6 τ_1²` as the
   asserted 5-brane class (BHOP Eq. 99). The regression test is
   then well-defined and can be written against
   `BhopExtensionBundle::published()` — but the plan must be
   updated to say so explicitly (the function name
   `MonadBundle::schoen_z3xz3_canonical` would be a misnomer in
   that world).

3. **Drop REM-OPT-B P1.1 from this round** — keep
   `schoen_z3xz3_canonical` as the documented Yukawa-pipeline
   placeholder, treat the Bianchi gauge anomaly as out of scope
   for the plan that names this bundle, and fold the regression
   test into a future P-cycle that follows whichever of (1)/(2)
   the author chooses.

## Summary

- No test was written.
- No test was committed.
- No production code was modified.
- This audit (`references/p_anomaly_cancellation_audit.md`) was
  added.

The bundle math, as currently in code for
`MonadBundle::schoen_z3xz3_canonical()`, neither closes nor
demonstrably fails Bianchi cancellation — its `c_2(V)` accessor is
explicitly stubbed and its `[W]` class is not asserted. Until the
P8.3-followup-B-Chern work or P1.3 lands, the regression test the
plan requests has no honest implementation.
