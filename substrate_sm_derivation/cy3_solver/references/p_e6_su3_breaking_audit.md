# Audit: `e6_su3_breaking_residual` measures the wrong Wilson breaking pattern

**Status**: open. Production-code semantic change BLOCKED on author decision
P1.3 (bundle structure group: rank-3 SU(3) vs rank-4 SU(4) vs rank-5 SU(5)).

**Hostile reviewer**: heterotic-CY3 specialist, wave-3 review (2026-05-03).

## Finding

`src/heterotic.rs:397-418` defines:

```rust
/// Embedding-consistency residual: for E_8 -> E_6 x SU(3) breaking
/// via Wilson line, the SU(3) factor must commute with the W^3 = 1
/// constraint. In Cartan basis, the SU(3) sub-block is the last 3
/// phases (a particular embedding choice). We require the first 5
/// phases to be 0 (E_6 directions unbroken) and the last 3 phases
/// to sum to 0 mod 2π (SU(3) traceless).
pub fn e6_su3_breaking_residual(&self) -> f64 { ... }
```

invoked symmetrically at `src/heterotic.rs:468`:

```rust
let wilson_breaking = wilson.e6_su3_breaking_residual();
```

inside `heterotic_bundle_loss`, regardless of which CY3 candidate
(`CY3TopologicalData::tian_yau_z3` or `::schoen_z3xz3`) is being
scored. The function name and doc-comment commit to the
`E_8 → E_6 × SU(3) → G_SM` Wilson breaking scheme — the **standard
embedding** (rank-3 SU(3) bundle V whose commutant in `E_8` is
`E_6 × SU(3)`).

## Canonical Wilson breaking patterns per primary source

| Reference | Bundle V | rank | Commutant in E_8 | Wilson breaks |
|---|---|---|---|---|
| BHOP-2005 (`hep-th/0501070`) §4 | SU(4) on Schoen | 4 | SO(10) | SO(10) → G_SM |
| DOPR-2005 (`hep-th/0411156`) §2.2-2.3 | SU(5) on Schoen | 5 | SU(5) | SU(5) → G_SM |
| Bouchard-Donagi 2005 (`hep-th/0512149`) §3 | SU(5) on Schoen | 5 | SU(5) | SU(5) → G_SM |
| AGLP-2012 (`arXiv:1202.1757`) | line-bundle SM | 5 (split) | U(1)^5 | U(1)^5 → G_SM |
| Tian-Yau 1986 / GKMR 1987 standard embedding | TX (rank-3 SU(3)) on TY/Z₃ | 3 | E_6 | E_6 × SU(3) → E_6 × discrete |

The `E_6 × SU(3)` pattern is the **standard-embedding** scheme,
correct ONLY when the bundle V has structure group SU(3) (commutant
of SU(3) in E_8 is E_6×SU(3) with the SU(3) factor being the
Wilson-line target). It is the WRONG breaking pattern for any of
BHOP / DOPR / Bouchard-Donagi.

## Cross-check: which bundles does production actually build?

`src/zero_modes.rs::MonadBundle::anderson_lukas_palti_example` (TY/Z₃):

```text
B = O(1,0)³ ⊕ O(0,1)³  (rank 6, c_1(B) = (3, 3))
C = O(1,1)³            (rank 3, c_1(C) = (3, 3))
rank(V) = 6 - 3 = 3    (SU(3))
```

`src/zero_modes.rs::MonadBundle::schoen_z3xz3_canonical`
(`zero_modes.rs:419`):

```text
B = O(1,0,0)² ⊕ O(0,1,0)² ⊕ O(0,0,1)²  (rank 6)
C = O(1,1,0) ⊕ O(0,1,1) ⊕ O(1,0,1)     (rank 3)
rank V = 6 - 3 = 3                       (SU(3))
```

**Both** production bundles are rank-3 SU(3). So
`e6_su3_breaking_residual` is **internally** consistent with the
production bundle choice — the SU(3) commutes with itself, the
E_6 stays unbroken on the first 5 Cartan directions, etc.

But the production bundle choice does **not** match BHOP-2005
(which uses rank-4 SU(4)) or DOPR-2005 / Bouchard-Donagi 2005
(both rank-5 SU(5)) — the three "canonical Schoen heterotic
SM" references in the literature. Production has implicitly
adopted the **Tian-Yau / standard-embedding** scheme on the
Schoen geometry, which is a non-standard pairing.

## The actual mismatch

Two distinct issues braid here:

**Issue 1** (function vs current bundle): no mismatch.
`e6_su3_breaking_residual` correctly scores the Wilson breaking for
a rank-3 SU(3) bundle. Both `anderson_lukas_palti_example` and
`schoen_z3xz3_canonical` are rank-3 SU(3). The function is
self-consistent with the bundle.

**Issue 2** (current bundle vs canonical Schoen heterotic-SM
literature): mismatch. Production does not implement BHOP/DOPR/
Bouchard-Donagi. It implements a TY-style standard-embedding
bundle on the Schoen geometry. If the publication intends to
test BHOP-2005 specifically, the bundle structure group must be
upgraded to SU(4) and `e6_su3_breaking_residual` becomes the
WRONG residual to score (correct one would be an SO(10)-breaking
residual on the **commutant** of SU(4), with Wilson lines in
SO(10), not E_6×SU(3)).

The hostile-reviewer claim, restated:

> If production claims to test BHOP-2005 (per the doc-comment in
> `src/geometry.rs:307` referring to Braun-He-Ovrut-Pantev 2005,
> arXiv:hep-th/0501070), the rank-3 SU(3) Schoen bundle is the
> wrong V and `e6_su3_breaking_residual` is the wrong residual.

## Fix path (BLOCKED — author decision required)

**Author decision P1.3**: Pick the bundle structure group:

- **(a) rank-3 SU(3) standard-embedding** on Schoen — keeps current
  code; deviates from BHOP/DOPR/Bouchard-Donagi; needs primary-source
  support for "TY-on-Schoen" being the publication's claim.
- **(b) rank-4 SU(4) BHOP-2005** — replace `schoen_z3xz3_canonical`
  with a rank-4 SU(4) bundle from BHOP §3 Table 1; replace
  `e6_su3_breaking_residual` with `so10_breaking_residual` (Wilson
  lines as elements of SO(10) ⊂ E_8, breaking SO(10) → G_SM).
- **(c) rank-5 SU(5) DOPR / Bouchard-Donagi** — replace bundle with
  rank-5 SU(5); replace residual with `su5_breaking_residual`.
- **(d) line-bundle SM (AGLP-2012)** — bundle is direct-sum of 5
  line bundles, structure group U(1)^5; residual measures the
  hypercharge Wilson line.

Whichever (a)-(d) the author chooses, the audit also requires:
- A `SCHOEN_VARIANT` marker constant pinning that production matches
  the chosen primary source — see `references/p_schoen_variant_pin.md`
  (Audit 3 in this batch).

## Audit artefact

`tests/test_e6_su3_breaking_pattern_audit.rs` — three tests:

1. `current_implementation_assumes_rank_3_su3_standard_embedding` —
   passes today (production matches its own assumption); will need
   reframing if P1.3 lands at (b)/(c)/(d).
2. `schoen_canonical_bundle_actual_rank_and_structure_group` —
   reports the actual bundle data: `rank = 3`, `c_1(V) = (0,0)`,
   `B-rank = 6, C-rank = 3`. Documents finding for P1.3.
3. `schoen_bundle_does_not_match_bhop_2005_rank_4_su4` — FAILS
   (asserts production matches BHOP-2005 rank-4); regression pin
   for the "publication-claims-BHOP" interpretation.

## Cross-test consistency post-wave-4

Wave-4 finding NEW-7 (post-round-3 hostile review, 2026-05-03)
flagged a superficial conflict between two tests:

- `tests/test_schoen_hodge.rs::schoen_variant_pin_bhop_2005`
  asserted `SCHOEN_VARIANT == "BHOP-2005"`, which a reviewer could
  read as "the production *bundle* is BHOP rank-4 SU(4)".
- `src/zero_modes.rs::tests::schoen_z3xz3_canonical_returns_three_factor_bundle`
  asserts `schoen_z3xz3_canonical().rank() == 3`, contradicting the
  rank-4 reading.

The reconciliation introduced by wave-4 NEW-2 splits the single
constant into two:

- `route34::schoen_geometry::SCHOEN_GEOMETRY_VARIANT = "BHOP-2005"`
  pins the *geometry* (Z/3 × Z/3 quotient action, cover Hodge
  `(19, 19)`, quotient Hodge `(3, 3)`). Asserted in
  `tests/test_schoen_hodge.rs::schoen_variant_pin_bhop_2005` (now
  reading the renamed constant).
- `route34::schoen_geometry::SCHOEN_BUNDLE_VARIANT =
  "TY-standard-embedding-on-Schoen-geometry (P1.3 author-decision-pending)"`
  pins the *bundle*: rank-3 SU(3) standard-embedding while P1.3 is
  open. Asserted in
  `tests/test_schoen_hodge.rs::schoen_bundle_variant_pin_documents_actual_bundle`.
- `tests/test_schoen_canonical_bundle_rank_policy.rs::
  schoen_canonical_bundle_rank_policy_is_three` (wave-4 NEW-6)
  pins `schoen_z3xz3_canonical().rank() == 3` as a committed-to
  policy and cross-references both constants.

After P1.3 lands at (a) rank-3 SU(3) / (b) rank-4 SU(4) BHOP /
(c) rank-5 SU(5) DOPR / (d) line-bundle SM AGLP-2012, all three
must be updated together:

- `SCHOEN_BUNDLE_VARIANT` literal value
- `tests/test_schoen_canonical_bundle_rank_policy.rs` rank assertion
- `tests/test_schoen_hodge.rs::schoen_bundle_variant_pin_*` content

The lib unit test `schoen_z3xz3_canonical_returns_three_factor_bundle`
will need its `bundle.rank() == 3` assertion updated as well, but
*only if* P1.3 chooses (b)/(c)/(d) — option (a) keeps it at 3.

## Round-6 audit update (2026-05-03, informational)

Round-6 audit note: `BhopExtensionBundle::published()`
(`src/route34/hidden_bundle.rs`) carries the genuine BHOP-2005 §6
Eq. 98 Chern data (c_2 = (-2, 7, 4)) and Eq. 88 index = -27, but is
not currently consumed end-to-end by the `bayes_discriminate` /
`p8_*` Bayes-factor pipeline. The constructor is exercised at the
unit-test level via
`tests/test_published_bundle_shape_collision.rs::bhop_extension_bundle_published_has_non_trivial_chern_data`.
Wiring into the BF pipeline is interlocked with author decision
P1.3 (bundle structure group). Treat as informational, not a
blocker.
