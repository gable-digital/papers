# Audit: `c2_tm = 36` collapse for both TY and Schoen candidates

**Status (round-5 hostile review, 2026-05-03)**: CONFIRMED OPEN.
Reviewer asked whether the production fix should be implemented or
the NEG test strengthened to a hard CI gate. Verification: the
correct multi-component vector requires the structure-group decision
P1.3 (rank-3 SU(3) vs rank-4 SU(4) BHOP vs rank-5 SU(5) DOPR vs (d)
line-bundle SM) — the right vector shape and target values depend
on which `V` was chosen. The Schoen vector `(36, 36, 24)` is fixed
at the cover (DHOR Eq. 3.13–3.15), but the TY vector under matching
conventions is per-bundle and is NOT yet pinned in production. The
NEG test `tests/test_c2_tm_distinguishes_candidates.rs::
c2_tm_distinct_between_ty_and_schoen` is **not** `#[ignore]`'d and
is **not** gated on a feature flag — running `cargo test --release`
from the project root will FAIL on this test, blocking any
release-build CI gate that runs the full test suite. The audit
artefact therefore is already a hard test-suite failure
(see verbatim cargo output below). No additional production-code
semantic change in round-5: the fix remains interlocked with P1.3.

**Status (wave-3, historical)**: open. Production-code semantic change
BLOCKED on author decision P1.3 (bundle structure group /
Bianchi-anomaly granularity).

**Hostile reviewer**: heterotic-CY3 specialist, wave-3 review (2026-05-03).

## Finding

`src/heterotic.rs:99-110` defines two constructors that store the SAME scalar
`c_2(TM) = 36` for **both** the Tian-Yau `Z/3` candidate and the Schoen
`Z/3 × Z/3` candidate:

```rust
pub fn tian_yau_z3() -> Self {
    Self { c2_tm: 36, quotient_order: 3 }
}
pub fn schoen_z3xz3() -> Self {
    Self { c2_tm: 36, quotient_order: 9 }
}
```

The struct doc-comment at `src/heterotic.rs:80-91` already admits the value
is fragile:

> `c_2 lives in a richer cohomology lattice and 36/9=4 is one summand in our
> convention. The 36 we use here is the "single H-component" downstairs
> aggregate from AGLP-2010 conventions.`

The downstream consumer is `anomaly_residual` at `src/heterotic.rs:424-427`:

```rust
pub fn anomaly_residual(bundle: &MonadBundle, topo: &CY3TopologicalData) -> f64 {
    let diff = bundle.c2() - topo.c2_tm;
    (diff as f64).powi(2)
}
```

invoked symmetrically by `heterotic_bundle_loss` (line 465) for **both**
TY and Schoen. Two distinct CY3 candidates with genuinely different
H^4 cohomology lattices are scored against an identical scalar target.
The "anomaly" channel of `HeteroticBundleLoss` therefore has zero
discriminating power between TY and Schoen — confirmed by the existing
test `tests/test_heterotic_data_consistency.rs:46,55`, which pins
`c2_tm == 36` for both candidates without any independent derivation.

## Correct values per primary sources

### Schoen `Z/3 × Z/3`

`src/route34/schoen_geometry.rs:287` already encodes the published triple:

```rust
pub const PUBLISHED_C2_TM_INTEGRALS: [i64; N_KAHLER] = [36, 36, 24];
```

with citation:

> Donagi-He-Ovrut-Reinbacher, JHEP 06 (2006) 039,
> DOI 10.1088/1126-6708/2006/06/039, Eq. (3.13–3.15).

So the correct cover-level (upstairs) datum is the **3-vector**
`(∫ c_2(T) ∧ J_1, ∫ c_2(T) ∧ J_2, ∫ c_2(T) ∧ J_t) = (36, 36, 24)`.
Reducing all three components to a single scalar `36` discards the
`J_t` direction (24 ≠ 36).

Anderson-Gray-Lukas-Palti 2011 (`arXiv:1106.4804`) §3 quotes the same
integers for the AGLP convention used by `heterotic.rs`.

### Tian-Yau `Z/3`

Cover `K_0 ⊂ CP^3 × CP^3` cut by three relations of bidegrees
`(3, 0), (0, 3), (1, 1)`. The **upstairs** Hodge numbers are
`(h^{1,1}_↑, h^{2,1}_↑) = (14, 23)`, χ_↑ = 2(14 − 23) = −18, as
pinned in production by `tests/test_ty_hodge.rs::ty_z3_hodge_*` and
in `src/route34/cicy.rs`.

A naive Lefschetz-hyperplane count would give only the two ambient
hyperplane classes (one per `CP^3`), making `h^{1,1}_↑` look like 2.
That argument applies to a transversal *bidegree (3,3)* hypersurface
in `CP^3 × CP^3` (one defining equation, codim 1, dim 5 — not a
CY3). The CICY-list TY 3-fold is **codim 3** (three defining
equations: (3,0), (0,3), (1,1)), so the (1,1)-divisor breaks the
transversality assumption of Lefschetz and additional Kähler classes
descend from blow-ups along the singular loci of the (3,0) and (0,3)
cones — yielding the 14 stored in production.

`c_2(T) ∧ J_a` is therefore a 14-vector, not a scalar (and not a
2-vector).

Primary references:
- **Tian-Yau 1986** (Comm. Math. Phys. **106**, 137-145): original
  construction of the CY3 from the bidegree-`(3,0)/(0,3)/(1,1)` CI.
- **Candelas-Dale-Lütken-Schimmrigk 1988** (Nucl. Phys. B **298**,
  493-525), CICY catalogue: lists TY as CICY entry with
  `(h^{1,1}, h^{2,1}) = (14, 23)`.
- **Greene-Kirklin-Miron-Ross 1987** (Nucl. Phys. B **278**, 667-693),
  Table 1: Chern data for the TY/Z₃ bicubic, `h^{1,1}(TY/Z_3) = 6`,
  `h^{2,1}(TY/Z_3) = 9` downstairs (character-formula breakdown
  reproduced in `tests/test_ty_hodge.rs::ty_z3_character_formula_*`).
- **Anderson-Gray-Lukas-Palti 2012** (`arXiv:1202.1757`), Table 2:
  AGLP convention for the TY-bicubic c_2 integrals.

Even on the simpler 2-Kähler-class projection used by the
[`MonadBundle::anderson_lukas_palti_example`] (rank-3 SU(3) bundle
with `B = O(1,0)³ ⊕ O(0,1)³`, `C = O(1,1)³`), the natural
2-component pair `(∫ c_2(T) ∧ H_1², ∫ c_2(T) ∧ H_2²)` is **not**
guaranteed to coincide with `(36, 36)`. The "single-H-component
aggregate" used in the constructor collapses both directions
into one number, hiding the candidate-dependent off-diagonal
structure.

## Downstream impact

`anomaly_residual` and `heterotic_bundle_loss` produce **identical**
anomaly-channel scores for the two candidates whenever
`bundle.c2()` is the same. This silently equates the heterotic
Bianchi identity for two genuinely different CY3 manifolds. Any
downstream Bayes-factor or χ² combiner that includes the anomaly
channel will see zero contribution from this channel to TY-vs-Schoen
discrimination. The σ-discrimination, chain-match, and Hodge
channels are unaffected.

`src/route34/bundle_search.rs:267-273` already implements the
correct **multi-component** Bianchi residual using the
`(36, 36, 24)` triple from Donagi-He-Ovrut-Reinbacher. This
infrastructure exists and is wired for the Schoen Bianchi check;
the bug is that `heterotic.rs::CY3TopologicalData` discards it in
favour of a scalar.

## Fix path (BLOCKED — author decision required)

Two options, both interlocked with **author decision P1.3** (bundle
structure-group choice):

**Option A** — replace scalar `c2_tm: i32` with a vector
`c2_tm_components: Vec<i64>` matching `h^{1,1}` of the cover. TY/Z₃
gets a 14-vector (or 2-vector under the AGLP projection); Schoen
gets `[36, 36, 24]`. `anomaly_residual` becomes vector-valued and
sums squared deviations component-wise. This is the same shape as
`route34::bundle_search::bianchi_residual_vector` already uses.

**Option B** — replace `c2_tm: 36` for TY with a *distinct* scalar
derived from the GKMR 1987 / AGLP 2012 Table 2 convention (e.g. the
top-form integral `∫ c_2(T) ∧ ω` against the Kähler form), so at
least the two candidates have different anomaly targets. Lower
fidelity than A; preserves the scalar API.

**Why blocked.** The "right" multi-component shape depends on
which structure-group `V` is chosen (rank-3 SU(3) vs rank-4 SU(4)
vs rank-5 SU(5); see `p_e6_su3_breaking_audit.md`), because the
anomaly equation `c_2(V_v) + c_2(V_h) = c_2(TM)` reads off
different Kähler-class components for different bundles. Author
must resolve P1.3 first.

## Audit artefact

`tests/test_c2_tm_distinguishes_candidates.rs` — NEG test that
fails on the current code, asserting `tian_yau_z3().c2_tm !=
schoen_z3xz3().c2_tm`. Doc-comment cites DHOR 2006, AGLP 2010,
GKMR 1987.
