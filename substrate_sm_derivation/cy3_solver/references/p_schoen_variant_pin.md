# Audit 3: Schoen variant pin

**Status (round-5 hostile review, 2026-05-03)**: REVISED. The literal
`"BHOP-2005"` pin overstated primary-source conjugacy. The production
β acts as a diagonal phase `t_1 ↦ ω · t_1` while fixing `t_0`; the
projective line `{t_1 = 0} ⊂ CP¹` is point-wise FIXED by β.
BHOP-2005 §3 uses an action that mixes the elliptic-fibration base
non-diagonally (translation by a 3-torsion section), and is NOT
conjugate to a diagonal phase on the `(t_0, t_1)` block — the DFT
conjugacy claim earlier in this doc covers only the `(x, y)`-block
(BHOP `(x_0:x_1:x_2)`-cyclic vs `(0,1,2)`-diagonal on each `CP^2`),
not the `t`-block.

`SCHOEN_GEOMETRY_VARIANT` was rewritten to a self-documenting
longer string naming the Schoen Z/3 × Z/3 quotient AND the freeness
gap. The literal-equality test in `tests/test_schoen_hodge.rs::
schoen_variant_pin_bhop_2005` was loosened to a contract test that
asserts the variant string contains "Schoen" and one of
"diagonal phase" / "freeness" / "unverified". A new sibling test
`schoen_diagonal_action_freeness_gap_documented` witnesses the
β-fixed locus at the projector level
(`beta_character([3,0,0,3,0,0,1,0]) == 0` — β acts as identity on
the monomial `x_0³ y_0³ t_0`).

The bundle side is tracked separately in `SCHOEN_BUNDLE_VARIANT`
(wave-4 finding NEW-2).

**Status (wave-3, historical)**: ADDRESSED. Constant
`SCHOEN_GEOMETRY_VARIANT = "BHOP-2005"` lives at
`src/route34/schoen_geometry.rs` immediately after `QUOTIENT_ORDER`.

**Hostile reviewer**: heterotic-CY3 specialist, wave-3 review (2026-05-03).

**Update (round-5 hostile review, NEW-6, 2026-05-03)**: the bare
`SCHOEN_VARIANT` `#[deprecated]` backward-compatibility alias has been
removed — it had no live callers. All references below to the bare
`SCHOEN_VARIANT` name are historical; new code must use
`SCHOEN_GEOMETRY_VARIANT` (geometry) or `SCHOEN_BUNDLE_VARIANT` (bundle).

## Finding

The hostile review wanted a code-level marker pinning which heterotic-SM
*variant* of the Schoen geometry the production code matches:

- **BHOP-2005** (Braun-He-Ovrut-Pantev, `hep-th/0501070`)
- **DOPR-2005** (Donagi-Ovrut-Pantev-Reinbacher, `hep-th/0411156`)
- **Bouchard-Donagi 2005** (`hep-th/0512149`)

All three use the same underlying Schoen `Z/3 × Z/3` 3-fold but differ
in (a) the explicit `Z/3 × Z/3` group action on the cover, and (b) the
choice of holomorphic bundle V on top.

## Decision

`SchoenGeometry::schoen_z3xz3` in `src/route34/schoen_geometry.rs:342`
implements the **BHOP-2005** *geometry*:

- Module-level doc-comment Eq. (2)–(3): diagonal cyclic-permutation
  generators on the two `CP^2` factors with `ω = exp(2πi/3)`.
- Cover Hodge data `(h^{1,1}, h^{2,1}) = (19, 19)`,
  `χ_upstairs = 0`.
- Quotient Hodge data `(h^{1,1}, h^{2,1})_X = (3, 3)`,
  `χ(X̃/Γ) = 0`.
- Free `Γ ≃ Z/3 × Z/3`, `|Γ| = 9`.

This matches BHOP-2005 §3 (Phys. Lett. B 618 (2005) 252-258).

## Action taken

1. Added `pub const SCHOEN_VARIANT: &str = "BHOP-2005";` to
   `src/route34/schoen_geometry.rs` (right after `QUOTIENT_ORDER`),
   with a doc-comment that:
   - Names all three candidate variants.
   - Explains why BHOP-2005 was picked (geometry equations match).
   - Notes the open caveat: the **bundle** V on top is currently
     rank-3 SU(3), NOT BHOP's rank-4 SU(4). See
     `references/p_e6_su3_breaking_audit.md`.

2. Updated the doc-comment of `SchoenGeometry::schoen_z3xz3()` to
   cite the variant and refer to `SCHOEN_VARIANT`.

3. Added `schoen_variant_pin_bhop_2005` test in
   `tests/test_schoen_hodge.rs` that asserts
   `SCHOEN_VARIANT == "BHOP-2005"`. This test PASSES — it is a
   regression pin, not an audit-fail.

## What this does NOT fix

The bundle V is still mismatched (Audit 2). `SCHOEN_VARIANT` pins
the *geometry* only; the *bundle* variant is tracked separately in
`MonadBundle::schoen_z3xz3_canonical`, which the hostile review
flagged as rank-3 SU(3) (TY/standard-embedding) instead of BHOP-2005
rank-4 SU(4). That fix is blocked on author decision P1.3.
