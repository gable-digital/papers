# P-Wilson-Fix — Cartan-phase indexing bug in legacy 2-factor projection

**Date:** 2026-04-30
**Path:** `src/route34/yukawa_sectors_real.rs::assign_sectors_dynamic`,
legacy 2-factor branch (lines 169-174 pre-fix).
**Discovered by:** P8.3-followup-B side discovery while wiring the
3-factor lift.
**Cross-references:**
* `references/p8_3_followup_a2_tensor_sparsity_diagnostic.md`
* `src/route34/wilson_line_e8.rs::canonical_e8_to_e6_su3`
* `src/route34/wilson_line_e8_z3xz3.rs::fiber_character`

## Symptom

For TY/Z3 (AKLP `O(1,0)^3 ⊕ O(0,1)^3` bundle, `b_lines_3factor: None`)
the legacy 2-factor sector projection always returned **phase class 0**
for every harmonic mode, regardless of which B-summand it was dominant
on. Consequently `up_quark`, `down_quark`, and `lepton` buckets were
empty, the round-robin fallback at line 222 fired, and the resulting
sector-buckets landed off the non-zero T-tensor support — producing the
documented Y_d = 6/9, Y_e = 4/9 bucket-hit numbers.

## Root cause (option-a in the task framing)

The legacy 2-factor projection computed
```rust
let inner = wilson.cartan_phases[6] * (b[0] as f64)
          + wilson.cartan_phases[7] * (b[1] as f64);
let class = (n as f64 * inner).round() as i32;
```

The canonical Wilson element from `WilsonLineE8::canonical_e8_to_e6_su3`
has
```text
cartan_phases = (2/3, -1/3, -1/3, 0, 0, 0, 0, 0)
```
in the orthonormal `R^8` realisation of the E_8 Cartan (Bourbaki
Plate VII / Slansky 1981 Tab. 23). Components `[6]` and `[7]` are
**exactly zero**, so `inner = 0` for every B-summand bidegree, every
mode collapsed to class 0, and the round-robin fallback fired.

The deeper problem is that this expression is **dimensionally
inconsistent**: `cartan_phases[k]` is a coefficient of the
orthonormal Cartan basis vector `e_k ∈ R^8`, while `b[0], b[1]` are
divisor degrees on the CY3 ambient `CP^3 × CP^3`. The two live in
**different spaces** and pairing their coordinates index-for-index
(`cartan_phases[k] * b[k]`) is not a meaningful inner product. There
is no choice of indices `[6]/[7]` (or `[0]/[1]` / `[3]/[4]` / any
other pair) that fixes this — the formulation itself is broken.

## Physical fix

The Wilson Z/3 character on AKLP-style B-summands is determined by
the **splitting principle**: under the `E_8 → E_6 × SU(3)` Wilson
breaking, the rank-3 SU(3) bundle splits as `V = ⊕_a L_a` and each
B-summand index `a` carries a distinct SU(3)-Cartan weight. The
canonical assignment (per `wilson_line_e8_z3xz3::fiber_character`'s
character table at lines 144-151) is

```text
b_line index | g_1 (SU(3) flavour)
     0       |  0
     1       |  1
     2       |  2
     3       |  0
     4       |  1
     5       |  2
```

i.e. `g_1 = (b_line index) mod 3`, with the `[1,0]` and `[0,1]` blocks
mapping to fundamental and antifundamental Z/3 characters
respectively (under the second Wilson factor for Z/3×Z/3, but that's
3-factor territory). For the single-Z/3 TY case, only `g_1` matters
and the Wilson character is simply the dominant B-summand index
modulo 3.

This is the correct fix and it matches the convention already
established by `wilson_line_e8_z3xz3.rs` for the 3-factor Schoen
path. Applied to the AKLP TY/Z3 example:

| B-summand index | bidegree `b` | Wilson class `g_1 = a mod 3` |
| --------------- | ------------ | ---------------------------- |
| 0               | `[1, 0]`     | 0 (lepton)                   |
| 1               | `[1, 0]`     | 1 (up_quark)                 |
| 2               | `[1, 0]`     | 2 (down_quark)               |
| 3               | `[0, 1]`     | 0 (lepton)                   |
| 4               | `[0, 1]`     | 1 (up_quark)                 |
| 5               | `[0, 1]`     | 2 (down_quark)               |

— **three distinct phase classes**, two B-summands each. The
round-robin fallback no longer fires.

## Code change

`src/route34/yukawa_sectors_real.rs`, legacy 2-factor branch:

```rust
// Before (broken — produces class = 0 for every mode):
let b = bundle.b_lines[dom];
let inner = wilson.cartan_phases[6] * (b[0] as f64)
          + wilson.cartan_phases[7] * (b[1] as f64);
let raw = n as f64 * inner;
raw.round() as i32

// After (correct — splitting-principle SU(3) flavour by index):
let _trace = wilson.cartan_phases[0]
           + wilson.cartan_phases[1]
           + wilson.cartan_phases[2];
debug_assert!(_trace.abs() < 1.0e-12,
    "canonical SU(3)-Cartan must be traceless");
(dom as i32).rem_euclid(3.min(n).max(1))
```

The `debug_assert` keeps the dependency on the canonical Wilson
element live: if a future revision changes `canonical_e8_to_e6_su3`
to a non-traceless 8-tuple, the assertion fires and the consumer is
forced to revisit the splitting-principle convention.

The mod is `3.min(n).max(1)` rather than `3` so a hypothetical
trivial-quotient case (`n = 1`) still gets a single class.

## Regression test

`assign_sectors_dynamic_2factor_projection_distinguishes_classes`
(in `yukawa_sectors_real.rs` test module) constructs a 6-mode synthetic
harmonic result on the AKLP TY/Z3 bundle (one mode per B-summand) and
asserts:

* Three non-empty class buckets (no round-robin fallback).
* `up_quark = {1, 4}`, `down_quark = {2, 5}`, `lepton = {0, 3}` (the
  exact splitting-principle assignment).
* `>= 2` distinct phase classes — the headline assertion motivating
  the fix.

The legacy bug would fail every assertion: all 6 modes would land in
class 0, the up/down buckets would be empty, and the round-robin
fallback would scramble into `up = [0, 3], down = [1, 4],
lepton = [2, 5]` — a clearly-distinguishable signature.

## Y_d / Y_e bucket-hit numbers (TY/Z3)

The P8.3-followup-A2 diagnostic binary cannot be run from this work
session because the library is currently in a broken intermediate
state from the in-flight P8.4-fix-d work on `schoen_metric.rs` /
`ty_metric.rs` / `cy3_metric_unified.rs` (struct field
`donaldson_tikhonov_shift` referenced in initialisers but not yet
declared in the configs). The Wilson-fix change is independent of
those errors and `cargo check` reports zero errors in
`yukawa_sectors_real.rs`.

Predicted hit-rate improvement (from P8.3-followup-A2 sparsity
data with the new bucket assignment):

* **Pre-fix:** all 9 modes in class 0 → round-robin
  `up = [0, 3, 6], down = [1, 4, 7], lepton = [2, 5, 8]` → Y_d 6/9,
  Y_e 4/9 (these are the documented baseline numbers — sectors land
  on a mix of populated and zero T-tensor entries, T[k=4], T[k=5]
  slices are numerically zero).

* **Post-fix:** modes are bucketed by their dominant B-summand index
  modulo 3. The actual harmonic-mode-to-B-summand binding depends on
  the upstream solver (`solve_harmonic_zero_modes`); for
  `dim H^1(V) = 9` modes spread across 6 B-summands (typical AKLP
  spectrum), expected sector populations are non-uniform but
  three-way split. Bucket-hit rate is expected to improve toward
  9/9 when the per-class buckets land on populated T-tensor entries
  (the populated h_0 = mode 0 slice has rank 7 with 49 non-zero
  entries — most index-pairs are populated, so any non-trivial
  three-way bucket split should raise both Y_d and Y_e hit-rates).

A precise re-run of the P8.3-followup-A2 binary requires P8.4-fix-d
to complete first. The Wilson-fix is correct independent of that
re-run; the sector buckets are now physically meaningful (three
distinct Z/3 classes) instead of all-zero collapsed.

## Build status

* `cargo check --release --features gpu --lib`: 2 errors, both pre-
  existing in P8.4-fix-d in-flight files (`schoen_metric.rs` /
  `cy3_metric_unified.rs` / `ty_metric.rs`), zero errors in
  `yukawa_sectors_real.rs`.
* `cargo check --release --features gpu --tests --lib`: same — no
  new errors introduced by the Wilson-fix or its regression test.

## What this fixes / does not fix

* **Fixes:** the legacy 2-factor projection now produces three
  distinct phase classes for the AKLP TY/Z3 bundle, matching the
  splitting-principle convention already in use by the 3-factor
  Schoen path. The round-robin fallback no longer fires for any
  bundle whose B-summand count exceeds the quotient order.

* **Does not fix:** the upstream P8.3-followup-A issue
  (rank-1-along-third-index `T_{ijk}` from harmonic-mode Galerkin
  coalescence). With a generically full-rank `T`, the Wilson-fix
  alone delivers correctly-bucketed sectors; with the upstream rank
  collapse still present, downstream hit-rates depend jointly on
  this fix and the upstream resolution.

---

## Empirical re-run (post-build-clean, 2026-04-30)

After P8.4-fix-d completed and the build went clean, I re-ran the
P8.3-followup-A2 diagnostic at HEAD (commit `2e456d73` + subsequent
clean-build commits) with the canonical settings used in the original
P8.3-followup-A2 baseline:

```
n_pts=200, seed=42, k_or_dx=3, max_iter=50, donaldson_tol=1e-9
MAG_FLOOR = 1e-10
```

### TY/Z3 (k=3) — bucket-hit table

| Channel | Pre-fix (round-robin) | Post-fix (Wilson-correct) |
|---------|-----------------------|---------------------------|
| Y_u (up x up)            | 9/9 | **1/9** |
| Y_d (up x down)          | 6/9 | **2/9** |
| Y_e (lepton x lepton)    | 4/9 | **4/9** |

Sector partition under the new projection:

```
up_quark   : [2]               (1 mode — only Y_u(2,2) hits)
down_quark : [1, 3, 4, 7]      (4 modes)
lepton     : [0, 5, 6, 8]      (4 modes)
higgs      : [0, 5, 6, 8]      (overlaps lepton — same bucket)
h_0 = 0
```

T-tensor h_0=0 slice has 49 / 81 non-zero (i,j) entries (rank 7).

### Schoen Z/3xZ/3 (d=(3,3,1)) — bucket-hit table (control)

| Channel | P8.3-followup-B2 baseline | This re-run |
|---------|--------------------------|-------------|
| Y_u | 9/9 | **9/9** |
| Y_d | 9/9 | **9/9** |
| Y_e | 1/9 | **1/9** |

Schoen is **unchanged** — confirming the Wilson-fix touches only the
legacy 2-factor projection path used by TY/Z3 and leaves the 3-factor
Schoen path untouched. Y_e=1/9 on Schoen is the previously-documented
padding-honest result (only 2 lepton modes available).

### Diagnosis: TY hit-rate dropped, did not improve

The pre-fix prediction was wrong. Three contributing causes:

1. **Sector-size mismatch.** The Wilson-fix correctly produces three
   Z/3 classes, but the AKLP `[1,0]³ ⊕ [0,1]³` bundle yields a class
   distribution `up=[2]` (one mode), `down=[1,3,4,7]` (four), 
   `lepton=[0,5,6,8]` (four). With only one up-type mode, the Y_u 3x3
   sector bucket has just one valid (li, rj) = (2, 2) cell — so
   Y_u hit-rate collapses from the round-robin's 9/9 to 1/9.
   Likewise Y_d has 1x4=4 valid (li, rj) cells of which only 3 fit in
   the 3x3 bucket print, capping the apparent hit at 3/9 max; only 2
   are non-zero ((2,1) and (2,3)) so hit=2/9.

2. **Round-robin's "9/9" was an artifact, not real geometry.** Pre-fix,
   round-robin populated all 3 sectors with 3 modes each, so the 3x3
   bucket grid was always fully evaluated. The high hit-rate reflected
   the dense 49/81 T-tensor non-zero pattern, not physically correct
   sector assignment. Post-fix hit-rates are lower numerically but
   geometrically meaningful.

3. **Y_e is unchanged at 4/9** because the lepton sector now has 4
   modes (vs 3 under round-robin), and the 3x3 Y_e bucket evaluates
   `(li, rj)` for `li, rj` drawn from `[0, 5, 6]` (first 3 lepton
   indices). Of those, 4 cells exceed MAG_FLOOR — same count as the
   pre-fix round-robin's `[0, 1, 2]` slice happened to land on.

### Verdict

The Wilson-fix is **mathematically correct** and the diagnostic confirms
three distinct Z/3 classes are now produced (up=1, down=4, lepton=4
distinct sector indices). However, the **hit-rate metric drops** because:

* The previous "9/9" Y_u was inflated by round-robin uniformity, not
  geometric content.
* The AKLP `[1,0]³ ⊕ [0,1]³` bundle's natural Wilson decomposition
  produces a **highly imbalanced** sector partition (1 up, 4 down,
  4 lepton) under the splitting-principle projection — not the 3:3:3
  partition the round-robin fallback faked.
* The bucket-hit metric (3x3 grid) is structurally biased toward 3:3:3
  sector partitions and undercounts when classes are imbalanced.

**Real progress**: the projection is now physically meaningful and the
fallback no longer fires. **Apparent regression**: the bucket-hit
metric falls because round-robin's 9/9 was a measurement artifact.

**Recommendation**: bucket-hit at h_0 in a 3x3 grid is no longer the
right diagnostic for the Wilson-corrected TY path. A better metric:
sum |T_{ij,h_0}| over the full Cartesian product of (up_sector x
down_sector) cells (including k>3 modes), normalized by sector size.
Defer that re-instrumentation to a future P-Wilson-validate pass.

### Cargo health

* `cargo test --release --features gpu --lib yukawa`: 42 passed,
  0 failed, 2 ignored, 795 filtered out — **no regression**.
* Build clean at HEAD (no errors in any of the recently-changed files).

### Production-code policy

`yukawa_sectors_real.rs` was **not modified** during this verification
pass. Only `references/p_wilson_fix_diagnostic.md` and
`references/p8_3_yukawa_production_results.md` were updated.
