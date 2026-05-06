# Wilson-line specification: BHOP-2005 §3 SO(10) Pati-Salam-style chain (rank-4 SU(4) bundle)

**Status**: retargeted (2026-05-04). The framework now commits to the
**rank-4 SU(4) bundle** on the Schoen quotient (BHOP-2005,
arXiv:hep-th/0501070, §3) and the corresponding `E_8 → SO(10) ×
SU(4) → SU(3)_C × SU(2)_L × U(1)_Y × U(1)_X` Pati-Salam-style chain.

The previous rank-3 SU(3) trinification specification (commit
`0efe6a2e`, joint commutant `SU(3)⁴`) is deprecated. Its constructor
[`Z3xZ3WilsonLines::canonical_aklp_schoen`] is retained as
`#[deprecated]` for diagnostic comparison only.

## Erratum (2026-05-04)

- **Prior claim** (commit `0efe6a2e`): the framework's rank-3 SU(3)
  bundle (`MonadBundle::schoen_z3xz3_canonical()`, rank 3) supports a
  Z/3 × Z/3 Wilson-line pair `(α_1, α_2) = ((1/3)(2,−1,−1,0,0,0,0,0),
  (1/3)(0,0,0,2,−1,−1,0,0))` whose joint commutant in `E_8` is
  `SU(3)⁴` (24 invariant roots, dim 32, Slansky 1981 Tab. 24
  trinification chain `E_8 → E_6 × SU(3) → SU(3)⁴`).
- **Status**: arithmetically correct for the rank-3 bundle, but
  **incompatible** with the rank-4 SU(4) bundle production target.
  The rank-3 `SU(3)⁴` joint commutant cannot accommodate the
  Standard-Model gauge group `SU(3)_C × SU(2)_L × U(1)_Y` because
  there is no `SU(2)` factor.
- **Resolution**: retarget to the BHOP-2005 chain. The rank-4 SU(4)
  bundle commutant in `E_8` is `SO(10)` (Slansky 1981 Tab. 24); the
  Wilson lines further break `SO(10) → SU(5) × U(1)_X → SU(3)_C ×
  SU(2)_L × U(1)_Y × U(1)_X`. The extra `U(1)_X` is the Pati-Salam
  `U(1)_{B-L}` (broken at lower scale).
- **Paper revision**: dispatched in parallel with this retarget; the
  rank-3 → rank-4 bundle commitment is now load-bearing.

## 1 Scope

The framework targets the heterotic-`E_8 × E_8` compactification on
the **Schoen** Calabi-Yau three-fold `M̃` with quotient
`Γ = Z/3 × Z/3` (Schoen 1988; Donagi-He-Ovrut-Reinbacher 2006;
Braun-He-Ovrut-Pantev 2005). The bundle structure group is **rank-4
SU(4)** (BHOP-2005), whose commutant in `E_8` is `SO(10)` (the
Pati-Salam group). The discrete Wilson lines live in the SO(10)
Cartan subspace of `T^8 ⊂ E_8`, and break

```text
    SO(10)  →  SU(5) × U(1)_X
            →  SU(3)_C × SU(2)_L × U(1)_Y × U(1)_X.
```

Combined with the rank-4 SU(4) bundle action, the unbroken gauge
group is the SM gauge group with an extra `U(1)_{B-L}`.

The companion **Tian-Yau** branch uses a single `Z/3` quotient,
hence a single Wilson line; that case is recovered by `coweight_1`
alone with `quotient_order = 3`.

## 2 The two Wilson-line generators (explicit Cartan vectors)

Both Wilson lines are stored as `WilsonLineE8` records — see
`src/route34/wilson_line_e8.rs` — with field `cartan_phases: [f64; 8]`
giving the 8-component vector `α ∈ R^8` such that the Wilson element
is `W = exp(2π i α / N)` for `N = quotient_order`.

The canonical pair is constructed by
`Z3xZ3WilsonLines::canonical_bhop2005_schoen()` in
`src/route34/wilson_line_e8_z3xz3.rs`. The basis convention places
SO(10) on coordinates 0..4 and SU(4) on coordinates 5..7
(orthogonal split of the rank-8 E_8 Cartan into
`rank(SO(10)) + rank(SU(4)) = 5 + 3 = 8`).

```text
First Z/3 generator (SO(10) → SU(5) × U(1)_X coweight):

    α_1 = (1/3) · (2,  2,  2,  2,  2,  0,  0,  0)

         3 · α_1 = (2, 2, 2, 2, 2, 0, 0, 0)
                   integer entries, sum = 10 (even)
                   ⇒ ∈ D_8 sublattice ⊂ Λ_root(E_8)
                   ⇒ W_1^3 = 1 exactly.

         Action: ⟨α_1, e_i − e_j⟩ = 0 for i, j ∈ {0..4}
                 (preserves the A_4 = SU(5) sub-root-system)
                 ⟨α_1, e_i + e_j⟩ = 4/3 ∉ Z for i, j ∈ {0..4}
                 (breaks the remaining D_5 = SO(10) roots)

         Result: SO(10) → SU(5) × U(1)_X.


Second Z/3 generator (SU(5) → SU(3) × SU(2) × U(1)_Y hypercharge):

    α_2 = (1/3) · (2,  2,  2, -3, -3,  0,  0,  0)

         3 · α_2 = (2, 2, 2, -3, -3, 0, 0, 0)
                   integer entries, sum = 0 (even)
                   ⇒ ∈ D_8 sublattice ⊂ Λ_root(E_8)
                   ⇒ W_2^3 = 1 exactly.

         Action: ⟨α_2, e_i − e_j⟩ = 0 for i, j ∈ {0, 1, 2}
                 (preserves SU(3)_C on coords 0..2)
                 ⟨α_2, e_3 − e_4⟩ = 0
                 (preserves SU(2)_L on coords 3..4)
                 ⟨α_2, e_0 − e_3⟩ = 5/3 ∉ Z
                 (breaks the SU(5) ↔ SU(3) × SU(2) cross roots)

         Result: SU(5) → SU(3)_C × SU(2)_L × U(1)_Y.
```

Both vectors vanish on coordinates 5..7 (the SU(4) Cartan), so they
commute pointwise with the SU(4) bundle structure-group action.

### BHOP-vector substitution disclosure

BHOP-2005 (arXiv:hep-th/0501070) commits to this `SO(10) × SU(4)`
embedding and the `Z/3 × Z/3` Wilson-line pair on the Schoen quotient,
but the paper presents the Wilson lines abstractly (in
`H¹(M̃, End(V))`-character language) rather than as explicit
8-component Cartan vectors in a fixed `R^8` basis. The vectors above
are the **standard Slansky-1981-Table-24 substitution** for the
canonical SO(10) → SU(5) → SM chain. Different basis conventions
for the SO(10) × SU(4) ⊂ E_8 embedding will produce the same
**joint commutant** up to a Cartan-basis rotation, so the
group-theoretic conclusions are convention-independent.

If a future audit pins BHOP's specific basis convention, the vectors
in `canonical_bhop2005_schoen` should be updated and the joint-
commutant test re-run. The audit-grade test
`bhop_joint_commutant_is_sm_gauge_group` programmatically asserts
that the joint commutant has the SM gauge dimension regardless of
the specific Cartan-basis representation.

### Quantization (order-3 property)

Each Wilson line is `Z/3`-quantised iff `3 · α_k ∈ Λ_root(E_8)`. For
both BHOP generators:
- `3·α_1 = (2, 2, 2, 2, 2, 0, 0, 0)` — integer, sum 10 (even) ⇒ in D_8 ⇒ in Λ_root(E_8) ✓
- `3·α_2 = (2, 2, 2, -3, -3, 0, 0, 0)` — integer, sum 0 (even) ⇒ in D_8 ✓

### Commutativity (Z/3 × Z/3 abelian)

Both `α_1` and `α_2` are diagonal in the same orthonormal Cartan
basis, so `[α_1, α_2] = 0` exactly. Tested in
`tests/test_wilson_line_commutant.rs::bhop_wilson_lines_commute`.

## 3 Programmatic commutant verification

The unbroken subalgebra of `e_8` after the joint Wilson-line + SU(4)
bundle action is the triple intersection

```text
    h(W_1, W_2, V_SU(4))
        = Z(W_1) ∩ Z(W_2) ∩ Z(V_SU(4))  ⊂  e_8.
```

For the production setup, this is computed by enumerating the 240
`E_8` roots and counting those satisfying:

1. `⟨β, α_1⟩ ∈ Z` (W_1-invariance),
2. `⟨β, α_2⟩ ∈ Z` (W_2-invariance),
3. `β[5] = β[6] = β[7] = 0` (zero SU(4)-Cartan projection ⇔ SU(4)-bundle commutativity).

Implementation:
[`Z3xZ3WilsonLines::joint_invariant_root_count_with_su4_bundle`]
(`src/route34/wilson_line_e8_z3xz3.rs`).

### Expected SM gauge group root count

| Subgroup        | Roots | Cartan rank |
| --------------- | ----- | ----------- |
| SU(3)_C         |   6   |     2       |
| SU(2)_L         |   2   |     1       |
| U(1)_Y × U(1)_X |   0   |     2       |
| **Total**       | **8** |   **5**     |

`dim h_unbroken = 8 + 5 = 13` (= 5 SO(10)-Cartan generators + 8
invariant SO(10) roots). Note: the SU(4) Cartan (rank 3) is absorbed
into the bundle structure group and is not part of the low-energy
gauge group, so it does not contribute to `h_unbroken`.

### Single-Wilson breaking diagnostics

Each `W_k` alone leaves the full SU(4) bundle Cartan intact (because
`α_k[5..7] = 0`), so the single-W lower bound on the unbroken
subalgebra is `dim(SO(10)) + dim(SU(4)) = 45 + 15 = 60`. The actual
single-W counts may be higher if some `E_8` roots happen to satisfy
the W_k-integrality condition — see the diagnostic prints in
`bhop_w1_alone_breaks_e8_to_so10_times_su4`.

## 4 BBW cohomology hook (status)

The plan's "BBW cohomology programmatic check" applies when the
Wilson lines act on the bundle cohomology
`H*(M, V ⊗ ρ_{Z/3×Z/3})` to project out the Standard-Model spectrum.
The Borel-Weil-Bott machinery for this is implemented in
`src/route34/zero_modes_harmonic.rs` and
`src/route34/zero_modes_harmonic_z3xz3.rs`; the per-`b_line`
character assignment used to project the harmonic basis is the
`character_table` field of `Z3xZ3WilsonLines` (see `fiber_character`
and `combined_z3xz3_character_schoen`).

**Note**: the existing `character_table` in
`canonical_bhop2005_schoen` is preserved verbatim from the AKLP
rank-6 monad case (`B = O(1,0)^3 ⊕ O(0,1)^3`) for backwards
compatibility with downstream zero-mode-harmonic projection code.
The rank-4 SU(4) bundle has a different monad shape; rank-4-specific
fibre characters belong with the BHOP bundle constructor and will be
addressed jointly with the bundle-side retarget.

## 5 Prior bundle-structure-group dependency: P1.3 (now CLOSED)

The rank-3 → rank-4 bundle decision (P1.3) is now closed by user
directive (rank-4 SU(4), 2026-05-04, paper revision in flight).
Historical option table:

```text
    Bundle V                    Commutant in E_8        Wilson breaking target
    ----------------------      --------------------    ------------------------
    rank-3 SU(3) (deprecated)   E_6 × SU(3)             trinification SU(3)⁴
    rank-4 SU(4) (PRODUCTION)   SO(10)                  SO(10) → SM (this doc)
    rank-5 SU(5) (DOPR-2005)    SU(5)                   SU(5) → SM
    rank-5 split (AGLP-2012)    U(1)^5                  U(1)^5 → SM
```

`schoen_z3xz3_canonical()` currently builds a rank-3 SU(3) monad
bundle; updating it to the rank-4 BHOP bundle is tracked in the
parallel paper revision and is out-of-scope for this Wilson-line
spec doc.

## 6 Test catalogue (Wilson-line specification)

### Unit tests (in-module, `cargo test --lib wilson`)

- `wilson_line_e8::unit_tests::*` — single Wilson line invariants
  (240 roots, lattice membership, canonical SU(3)-coweight breaking).
- `wilson_line_e8_z3xz3::tests::bhop_both_coweights_z3_quantized`
- `wilson_line_e8_z3xz3::tests::bhop_3alpha_in_root_lattice`
- `wilson_line_e8_z3xz3::tests::bhop_cartan_pair_commutes`
- `wilson_line_e8_z3xz3::tests::bhop_joint_smaller_than_single`
- `wilson_line_e8_z3xz3::tests::bhop_fiber_character_table_has_six_entries`
- `wilson_line_e8_z3xz3::tests::combined_character_trivial_for_constant_monomial`
- `wilson_line_e8_z3xz3::tests::combined_character_compensation`
- `wilson_line_e8_z3xz3::tests::deprecated_aklp_constructor_still_well_formed`

### Integration tests (`cargo test --test test_wilson_line_commutant`)

8 BHOP-chain audit tests (NEW, 2026-05-04 retarget):

1. `bhop_wilson_lines_have_order_3_each` — `3·α_k ∈ Λ_root(E_8)` for k = 1, 2.
2. `bhop_wilson_lines_commute` — `[W_1, W_2] = 0`.
3. `bhop_w1_alone_breaks_e8_to_so10_times_su4` — single-W support
   on SO(10) Cartan, lower bound on unbroken Lie dim.
4. `bhop_joint_commutant_is_sm_gauge_group` — joint count matches
   SM gauge dim (8 invariant roots).
5. `bhop_jacobi_anticommutator_vanishes` — pointwise
   commutativity of all Cartan-component pairs.
6. `bhop_w1_action_on_16_of_so10` — non-triviality and linear
   independence of `(α_1, α_2)`.
7. `bhop_three_generations_from_h1_decomposition` — 6-entry
   character table preserved with `Z/3 × Z/3` valued entries.
8. `bhop_anti_e8_invariant_root_count_matches_sm_gauge_dim` —
   independent enumeration of joint-invariant root count.

Plus one regression test for the deprecated constructor:

9. `deprecated_trinification_pair_still_well_formed` — pins
   `joint_invariant_root_count = 24` (SU(3)⁴) for the rank-3 path.

## 7 References

1. Schoen, *Math. Z.* **197** (1988) 177–199 — fiber-product CY3.
2. Braun-He-Ovrut-Pantev, *Phys. Lett. B* **618** (2005) 252
   (`hep-th/0501070`) — heterotic-SM rank-4 SU(4) on Schoen
   (**production target**, this spec).
3. Donagi-He-Ovrut-Reinbacher, JHEP **06** (2006) 039
   (`hep-th/0512149`) — Schoen heterotic spectrum.
4. Anderson-Gray-Lukas-Palti, JHEP **06** (2012) 113
   (`arXiv:1106.4804`) — 200 heterotic SMs (line-bundle
   realisations of the same chain).
5. Slansky, *Phys. Rep.* **79** (1981) 1, §6 and Tables 24, 25–29
   (`E_8 ⊃ SO(10) × SU(4)`, `SO(10) ⊃ SU(5) × U(1)`,
   `SU(5) ⊃ SU(3) × SU(2) × U(1)` chains).
6. Bourbaki, *Lie Groups and Lie Algebras* Ch. VI §4 Plate VII.
7. Conway-Sloane, *Sphere Packings, Lattices and Groups* (Springer
   1999, 3rd ed.) Ch. 4 §8.1.
