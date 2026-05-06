# Audit: Stage 3 bundle physics requirements vs. standard-embedding over-claim

**Status**: BUNDLE FAILS — fails Bianchi (no asserted hidden bundle / 5-brane on
this geometry), fails N_gen = 3 (∫_X̃ c_3(V) = −6, not a multiple of 2|Γ|=18),
not wired to polystability, no character-decomposition data on H¹.

**Auditor**: REM-OPT-B follow-up, 2026-05-04.

**Subject**: `zero_modes::MonadBundle::schoen_z3xz3_canonical()` evaluated
against the FULL set of physical heterotic-SM requirements on
`X = X̃/(Z/3 × Z/3)` where X̃ is the Schoen fiber-product CY3 in
CP² × CP² × CP¹.

**Key reframe**: the wave-28 substrate paper's commitment
`c_2(V) = c_2(TX̃)` is a standard-embedding-flavoured over-claim. The correct
physical requirement is the **full Bianchi identity**
`c_2(V_vis) + c_2(V_hid) + [W] = c_2(TX̃)`. This audit replaces the standard-
embedding test with the full-Bianchi test, plus N_gen, polystability, and
H¹ decomposition.

---

## Section 1 — The user's correction

The wave-28 substrate paper Step 0.7 / 3.3 currently claims:

> non-standard rank-3 SU(3) bundle V on the Schoen cover with
> `c_2(V) = c_2(TX̃)` and `|c_3(V_cover)| = 54`.

Both halves of that sentence inherit standard-embedding assumptions. In a
non-standard heterotic E_8 × E_8 compactification:

- `c_2(V) = c_2(TX̃)` is **only required** if `V_hid = 0` and `[W] = 0`. Once
  the framework's hidden-E_8' commitment turns on, or 5-branes wrap holomorphic
  curves, the Bianchi identity is the *full* anomaly equation
  `c_2(V_vis) + c_2(V_hid) + [W] = c_2(TX̃)`, with `[W]` effective (component-
  wise non-negative in the Mori cone — anti-5-branes break SUSY) and
  `c_2(V_hid) ≥ 0` for a holomorphic hidden bundle.
- The cover-vs-quotient distinction on `c_3(V)` is the load-bearing piece for
  three generations: by Atiyah-Singer on the free quotient,
  `N_gen = |c_3(V_cover)| / (2 |Γ|)`. With `|Γ| = 9`, three generations
  requires `|c_3(V_cover)| = 54`. The agent's reported `c_3 = −6` does not
  satisfy this.

What MATTERS is whether the geometry admits the SM. Concretely (Sec. 2 below):
full Bianchi closes, N_gen = 3, polystability, H¹ decomposes 3:3:3 across
Wilson characters.

---

## Section 2 — The four physical requirements

### Requirement 1 — Anomaly cancellation (Bianchi identity)

`c_2(V_vis) + c_2(V_hid) + [W] = c_2(TX̃)` in `H⁴(X̃, ℤ)`, with `[W] ≥ 0`
(effective, holomorphic 5-branes), `c_2(V_hid) ≥ 0` (holomorphic hidden
bundle). Any single component-wise violation in the `(J_1, J_2, J_t)` basis
falsifies anomaly closure.

### Requirement 2 — N_gen = 3 via Atiyah-Singer

Free `Z/3 × Z/3` quotient: `N_gen = |c_3(V_cover)| / (2 |Γ|) = |c_3| / 18`.
Three generations ⟺ `|c_3(V_cover)| = 54`. Equivalently: the bundle must be
`Z/3 × Z/3`-equivariant in a way that descends with integer index −3 on
the quotient.

### Requirement 3 — Polystability (Mumford-Takemoto / DUY)

`V` must admit a Hermitian-Einstein connection on some open Kähler-cone
region — equivalent to slope-polystability for some `J ∈ K(X̃)` invariant
under `Z/3 × Z/3`.

### Requirement 4 — Cohomology decomposition (3:3:3 across Wilson characters)

For Wilson-line breaking `E_8 → E_6 × Z/3 × Z/3 → SU(3)⁴` trinification,
`H¹(X/Γ, V)` must split as a 9-dimensional space distributing 3:3:3 across
the three irreducible Wilson characters of the surviving Z/3 quotient that
breaks `E_6 → SU(3)³`. (`H¹(X/Γ, ad V)` provides Higgs doublets / triplets.)

---

## Section 3 — Per-requirement audit of `schoen_z3xz3_canonical`

### Bundle data (recovered from `src/zero_modes.rs:431-501`)

```text
0  →  V  →  B  →  C  →  0
B = O(1,0,0)² ⊕ O(0,1,0)² ⊕ O(0,0,1)²       (rank 6)
C = O(1,1,0) ⊕ O(0,1,1) ⊕ O(1,0,1)          (rank 3)
rank V = 3   (SU(3))
c_1(B) = (2, 2, 2),  c_1(C) = (2, 2, 2),
c_1(V) = (0, 0, 0)   ✓ SU(3)
```

Geometry (Schoen Z/3 × Z/3 fiber-product, `geometry.rs:321-335`): triple
intersections on X̃ ⊂ CP² × CP² × CP¹:

- ∫ J_1² J_2 = 3
- ∫ J_1 J_2² = 3
- ∫ J_1 J_2 J_t = 9
- All other top monomials = 0

Quotient order |Γ| = 9.

`c_2(TX̃) = (36, 36, 24)` per `route34::schoen_geometry::SchoenGeometry::c2_tm_vector`
(DHOR-2006 Eq. 3.13–3.15).

### Computed Chern data (Whitney decomposition, `MonadBundle::chern_classes`)

The 3-factor branch landed in commit `ce01d749` (no longer a stub). Direct
computation via the geometry's `triple_intersection`:

| Class                  | Value (per Kähler / scalar)             |
|------------------------|-----------------------------------------|
| `c_1(V)`               | `(0, 0, 0)` ✓ SU(3)                     |
| `c_2(V) ∧ J_1`         | 51 (B) − 39 (C) = **12**                |
| `c_2(V) ∧ J_2`         | 12 (by 1↔2 symmetry of B and C)         |
| `c_2(V) ∧ J_t`         | 36 (B) − 27 (C) = **9**                 |
| `c_2(V)` vector        | **(12, 12, 9)**                         |
| `c_3(B)`               | 84 (8·9 + 2·3 + 2·3)                    |
| `c_3(C)`               | 24 (single triple `(J_1+J_2)(J_2+J_t)(J_1+J_t)`) |
| `∫ c_2(V) · c_1(C)`    | 2·(12+12+9) = 66                        |
| `c_3(V)`               | 84 − 24 − 66 − 0 = **−6** (cover)       |

These match the agent's reported `c_2 = (12, 12, 9)`, `c_3 = −6`.

The integration is performed against `geometry.triple_intersection`, which uses
the cover's defining-relation form (no division by `|Γ| = 9`). Therefore
**`c_3(V) = −6` is the cover value `∫_X̃ c_3(V)`**, not the quotient index.

### Requirement 1 audit — Bianchi: FAILS

`c_2(TX̃) − c_2(V_vis) = (36 − 12, 36 − 12, 24 − 9) = (24, 24, 15)`.

For full Bianchi closure we need `c_2(V_hid) + [W] = (24, 24, 15)` with both
summands `≥ 0` component-wise.

**The framework has no asserted hidden bundle on this geometry in this basis.**
The closest candidate, `route34::hidden_bundle::BhopExtensionBundle::published()`
(BHOP-2005 §6.1-6.2, cited at `hidden_bundle.rs:741-755`), works in the
`(τ_1², τ_2², τ_1 τ_2)` BHOP base-divisor basis — NOT the `(J_1, J_2, J_t)`
ambient-pullback basis used by the canonical Schoen monad. The two bases are
related by a non-trivial change-of-basis through Schoen's projection
`X̃ → CP¹`, and the BHOP bundle is **rank-4 SU(4)** (paired with rank-4 V_vis,
not the rank-3 SU(3) of the canonical Schoen monad). A rank-4 hidden + rank-3
visible decomposition is non-standard and not documented anywhere in the
codebase.

`route34::bundle_search::LineBundleDegrees::bianchi_residual_vector`
(`bundle_search.rs:319-336`) implements the per-Kähler check, but it expects
a `LineBundleDegrees` for both V and H. `MonadBundle::schoen_z3xz3_canonical()`
is not reduced to `LineBundleDegrees` form, and there is no asserted hidden
`LineBundleDegrees` for it.

**Verdict**: Requirement 1 unverifiable in current code; the structurally-
named hidden-E_8' bundle that would close `(24, 24, 15) = c_2(V_hid) + [W]`
does not exist in the codebase as data on this bundle. There is no
positive evidence Bianchi closes; there is structural evidence it cannot
close trivially (`(24, 24, 15) = c_2(V_hid)` alone is excluded by the
absence of any BHOP-equivalent hidden bundle in the `(J_1, J_2, J_t)` basis).

### Requirement 2 audit — N_gen = 3: FAILS HARD

Atiyah-Singer: `N_gen = |c_3(V_cover)| / (2 |Γ|) = |−6| / 18 = 1/3`.
**Fractional generation count is unphysical.** The bundle either:

- Is not `Z/3 × Z/3`-equivariant (no equivariant lift exists for these
  particular B, C summands under the canonical free Z/3 × Z/3 action on
  the affine `CP²` coordinates) — the constructor's `b_lines_3factor` is
  Wilson phase-class data for the Yukawa-bucket pipeline (`(a − b) mod 3`
  projection), NOT a constructed equivariant structure on the bundle, and
- Has `c_3 = −6` not divisible by 18, so even if a lift existed the index
  would still be fractional.

`compute_zero_mode_spectrum` (`zero_modes.rs:1481-1530`) hides this by integer
division: `|c_3| / 2 = 3`, then `3 / |Γ|=9` truncates to `0`. **The current
spectrum-counter reports zero generations downstairs.** The constructor's
own doc-comment (lines 422-430) flags this: *"compute_zero_mode_spectrum still
returns 0 for 3-factor ambients pending P8.3-followup-B-Chern."*

**Verdict**: Fails. `c_3(V_cover) = ±54` (i.e. 9× larger) is required for
3 generations.

### Requirement 3 audit — Polystability: NOT WIRED

`route34::polystability::check_polystability` is the production
HYM/Mumford-Takemoto checker. It accepts `heterotic::MonadBundle` (a separate
struct from `zero_modes::MonadBundle`, see `references/p_anomaly_cancellation_audit.md:43-58`).
The BHOP rank-4 SU(4) `BhopExtensionBundle::published()` passes the test
(commit `a4d1bc72`, 4/4 tests). The rank-3 SU(3)
`zero_modes::MonadBundle::schoen_z3xz3_canonical()` is not connected — there
is no `to_heterotic_monad()` adapter, no test, no pre-computed verdict.

**Verdict**: Insufficient information without further work. Cannot conclude
either way until the adapter is plumbed and the polystability checker runs.

### Requirement 4 audit — H¹ decomposition: PARTIAL / NEGATIVE

`assign_sectors_dynamic` (`yukawa_sectors_real.rs:82-256`) classifies harmonic
modes by Wilson phase class via the projection `(a − b) mod 3` on the
3-factor lift `b_lines_3factor`. For the canonical Schoen B-summands
`{[1,0,0], [0,1,0], [0,0,1]}` with multiplicity 2 each:

| Bidegree    | (a − b) mod 3 | Class |
|-------------|---------------|-------|
| `[1,0,0]` ×2| +1            | 1     |
| `[0,1,0]` ×2| −1 ≡ 2        | 2     |
| `[0,0,1]` ×2| 0             | 0     |

Distribution on B: 2:2:2 across classes (0:1:2). This is **not 3:3:3**.
It would be the correct multiplicity pattern for a *rank-9* bundle delivering
9 generations on the cover (2 + 2 + 2 = 6 modes per class × 9 classes is
inconsistent with rank 6 and 3 classes), but for rank 6 with 2 modes per
class delivering 3 generations downstairs, the pattern only checks out if
`H¹` cohomology lifts to a multiplicity that becomes 3 on the quotient via
Wilson-line invariant projection. With `c_3 = −6` this lift is structurally
impossible (Sec. R2 above).

The `compute_zero_mode_spectrum` code does NOT actually decompose `H¹(X/Γ, V)`
by Wilson character — it returns a single integer `n_27 = |c_3|/(2|Γ|)`
truncated to zero on this bundle. The Wilson character data is consumed
downstream by `assign_sectors_dynamic` for Yukawa-bucket assignment, but never
fed back as a generation-count-per-character cross-check.

**Verdict**: The 2:2:2 phase distribution is structurally inconsistent with
3 generations (which requires 3:3:3 per character on the rank-9 H¹). The
absence of a per-character `H¹` count in `compute_zero_mode_spectrum` masks
this — the pipeline never makes the 3:3:3 claim because it never measures it.

---

## Section 4 — Verdict

**Verdict: B (bundle FAILS)** with an audit-trail caveat that R3 is
unverifiable without further work.

| Req | Test                                | Status | Evidence                                  |
|-----|-------------------------------------|--------|-------------------------------------------|
| 1   | Full Bianchi closes                 | FAIL   | No hidden bundle in `(J_1,J_2,J_t)` basis; gap `(24, 24, 15)` unaccounted |
| 2   | N_gen = 3                           | FAIL   | `c_3(V_cover) = −6`, not 54; 1/3 generation = unphysical |
| 3   | Polystability                       | UNK    | Not wired to `check_polystability`; no certificate |
| 4   | 3:3:3 H¹ decomposition              | FAIL   | Distribution is 2:2:2 on B, never measured on H¹     |

The wave-28 paper's `c_2(V) = c_2(TX̃)` claim is independently false on this
bundle (`c_2(V) = (12, 12, 9) ≠ (36, 36, 24)`), but per the user's correction
that's a non-defect — the standard-embedding identity is not what physics
requires. The genuine defects are (R1) no closing hidden + 5-brane data,
(R2) wrong `c_3` for 3 generations, (R4) wrong character distribution.

This is consistent with the prior-audit conclusion at
`references/p_anomaly_cancellation_audit.md:113-119`:

> production builds `schoen_z3xz3_canonical` as a structurally appropriate
> placeholder for the Yukawa-bucket pipeline (it spans the right Wilson Z/3
> phase classes and feeds harmonic-mode counting), not as a compactification
> certified by Bianchi.

Re-reading that with the user's reframe: it is also not a compactification
certified by Atiyah-Singer N_gen, nor by the per-character H¹ count. The
bundle is a rank-3 SU(3) **placeholder** that the Yukawa pipeline can chase
buckets through — it is not a working SM heterotic compactification.

---

## Section 5 — Recommended paper-prose update

### Current (over-claimed) wording

> non-standard rank-3 SU(3) bundle V on the Schoen cover with
> `c_2(V) = c_2(TX̃)` and `|c_3(V_cover)| = 54`.

### Honest replacement (option A — the bundle is a placeholder)

> non-standard rank-3 SU(3) bundle V on the Schoen Z/3 × Z/3 cover, used in
> this work as a *Wilson-character-aligned placeholder* for the harmonic
> Yukawa pipeline (B-summand bidegrees populate three distinct
> `(a − b) mod 3` Wilson phase classes, sufficient for non-degenerate
> Yu/Yd/Ye sector buckets in `assign_sectors_dynamic`). The full
> heterotic-SM physical requirements — Bianchi closure
> `c_2(V_vis) + c_2(V_hid) + [W] = c_2(TX̃)` with effective `[W]`, the
> Atiyah-Singer count `|c_3(V_cover)| = 2 |Γ| = 18` for three generations,
> Mumford-Takemoto polystability, and the 3:3:3 distribution of
> `H¹(X, V)` across Wilson characters — are open follow-ups. The
> σ-discrimination result of this paper is a **geometry test** (Schoen
> versus Tian-Yau via the σ-channel) and is independent of the visible-
> bundle SM completion; it does not assert that V realises the Standard
> Model spectrum on this geometry.

### Honest replacement (option B — adopt BHOP rank-4 SU(4) as the Schoen visible bundle)

If the paper is willing to switch to the BHOP-2005 rank-4 SU(4) extension
bundle (already implemented at `BhopExtensionBundle::published()`):

> rank-4 SU(4) extension bundle V on the Schoen Z/3 × Z/3 cover, taken
> verbatim from Braun-He-Ovrut-Pantev 2005 (arXiv:hep-th/0505041 §6.1-6.2):
> `c_2(V) = -2 τ_1² + 7 τ_2² + 4 τ_1 τ_2`,
> `c_2(H) = 8 τ_1² + 5 τ_2² - 4 τ_1 τ_2`,
> `c_2(TX̃) = 12 (τ_1² + τ_2²)`,
> Bianchi residual `6 τ_1²` cancelled by 5-branes wrapping `PD(τ_1²)`
> (BHOP Eq. 99), `Index(V) = -27` on the cover ⟹ `N_gen = -27 / 9 = -3`
> on the quotient (BHOP Eq. 88-89). This is a published rank-4 SM
> compactification on Schoen Z/3 × Z/3.

Option B requires:
- Replacing `MonadBundle::schoen_z3xz3_canonical()` callers with the BHOP
  bundle (the function name becomes a misnomer).
- Re-running the Yukawa pipeline against rank-4 BHOP data instead of
  rank-3 monad data.
- The σ-discrimination already runs against the *geometry*, not the bundle,
  so the 6.92σ result at `references/cy3_publication_summary.md` is
  unaffected.

Either replacement is honest. Option A is minimal-invasive and faithful to
the paper's actual scope (it is a σ-discrimination paper, not a generation-
count paper). Option B promotes the paper's claims to actually deliver an
SM compactification on Schoen, at the cost of a code refactor.

---

## Section 6 — Recommended next action

Given the σ-discrimination result is the load-bearing scientific claim and
already stands at 6.92σ Tier 0 (`references/cy3_publication_summary.md`,
`project_cy3_5sigma_discrimination_achieved.md`), and given the bundle's
role in *that* result is purely to provide a 9-mode harmonic-Laplacian
kernel basis (decoupled from σ via P5.6b's BBW-correct `kernel_dim_target`,
`project_cy3_yukawa_kernel_basis_bug.md`), **option A is the recommended
fix**:

1. **Update wave-28 paper Step 0.7 / 3.3** to the option-A wording above
   (the bundle is a Wilson-character-aligned harmonic-pipeline placeholder,
   not a certified SM compactification; the σ-discrimination claim is a
   geometry test independent of bundle SM completion).
2. **Pin** `references/p_schoen_variant_pin.md` accordingly:
   `SCHOEN_BUNDLE_VARIANT = "Wilson-character-aligned-placeholder
   (R1/R2/R4 open; not an SM compactification)"`.
3. **Defer** the genuine BHOP-rank-4-SU(4) port (option B) to a follow-up
   paper or a separate appendix that owns the rank-4 SM-completion claim
   in its own right.
4. **Add a regression test** asserting the audit's Chern numbers on
   `schoen_z3xz3_canonical`: `c_2(V) = (12, 12, 9)`, `c_3(V) = -6` —
   so any future change to the bundle that drifts the placeholder is
   surfaced. The test does NOT assert Bianchi or N_gen = 3 (those would
   be rigging — the bundle does not satisfy them).
5. **Document in code** at `MonadBundle::schoen_z3xz3_canonical()` doc-
   comment: this function returns a placeholder, NOT a certified SM
   compactification. Update the existing 9-line WARNING already in lines
   411-420 to be a top-level WARNING block.

This path keeps the paper's actual scientific claim intact while removing
the over-claim. Recommended.

---

## Appendix — calculation traces

### Pairs in B used for c_2(V) ∧ J_a

B has 6 lines: `[1,0,0]` ×2, `[0,1,0]` ×2, `[0,0,1]` ×2. Pair multiplicities:

| Pair classes                   | Count |
|--------------------------------|-------|
| `([1,0,0], [1,0,0])`           | 1     |
| `([1,0,0], [0,1,0])`           | 4     |
| `([1,0,0], [0,0,1])`           | 4     |
| `([0,1,0], [0,1,0])`           | 1     |
| `([0,1,0], [0,0,1])`           | 4     |
| `([0,0,1], [0,0,1])`           | 1     |
| **Total**                      | **15** = C(6,2) ✓ |

### c_2(V) ∧ J_1 (J_a = [1,0,0])

Σ_B = 1·I(3,0,0) + 4·I(2,1,0) + 4·I(2,0,1) + 1·I(1,2,0) + 4·I(1,1,1) + 1·I(1,0,2)
    = 0 + 4·3 + 0 + 3 + 4·9 + 0 = **51**.

Σ_C: pairs of `[1,1,0], [0,1,1], [1,0,1]`, contracted with `J_1`:
- `([1,1,0],[0,1,1],J_1)` → `(J_1+J_2)(J_2+J_t)·J_1` = J_1²J_2 + J_1²J_t + J_1J_2² + J_1J_2J_t = 3 + 0 + 3 + 9 = 15
- `([1,1,0],[1,0,1],J_1)` → `(J_1+J_2)(J_1+J_t)·J_1` = 0 + 0 + 3 + 9 = 12
- `([0,1,1],[1,0,1],J_1)` → `(J_2+J_t)(J_1+J_t)·J_1` = 3 + 9 + 0 + 0 = 12

Σ_C = 39. **c_2(V) ∧ J_1 = 51 − 39 = 12**.

### c_3 triples in B

| Multiset of classes               | # triples | I(triple)      |
|-----------------------------------|-----------|----------------|
| (a, a, b) [1,0,0]²·[0,1,0]        | 2         | 3 = I(2,1,0)   |
| (a, a, c) [1,0,0]²·[0,0,1]        | 2         | 0 = I(2,0,1)   |
| (b, b, a) [0,1,0]²·[1,0,0]        | 2         | 3 = I(1,2,0)   |
| (b, b, c) [0,1,0]²·[0,0,1]        | 2         | 0 = I(0,2,1)   |
| (c, c, a) [0,0,1]²·[1,0,0]        | 2         | 0 = I(1,0,2)   |
| (c, c, b) [0,0,1]²·[0,1,0]        | 2         | 0 = I(0,1,2)   |
| (a, b, c) all-distinct            | 8         | 9 = I(1,1,1)   |

Σ_B = 2·3 + 2·3 + 8·9 = 6 + 6 + 72 = **84**.

Σ_C (single triple): `(J_1+J_2)(J_2+J_t)(J_1+J_t)`. Expand:
J_1²J_2 + J_1²J_t + J_1J_2J_t + J_1J_t² + J_1J_2² + J_1J_2J_t + J_2²J_t + J_2J_t²
= 3 + 0 + 9 + 0 + 3 + 9 + 0 + 0 = **24**.

`c_3(V) = 84 − 24 − 66 − 0 = −6` (cover).
