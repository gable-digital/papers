# P7.4 — ω_fix ratio-interpretation sweep (exit-door 2)

**Status:** FALSIFIED. No eigenvalue ratio interpretation reproduces
ω_fix = 123/248 to <100 ppm robustly. Recommend P7.5 (sub-Coxeter basis)
as the next exit-door test.

## Hypothesis

P7.1/P7.1b/P7.2b falsified ω_fix = 123/248 as a single normalized
metric-Laplacian eigenvalue on Schoen/Z3xZ3. Hypothesis (2): the
journal's ω_fix is a *ratio* between two specific eigenvalues, not a
single one.

## Data sources

Available outputs (canonical settings n_pts=40k, iter_cap=100, k=3,
test_degree=4):

| File | Seeds | Spectrum size | Characters |
|---|---|---|---|
| `p7_1_omega_fix_diagnostic.json` | 42 (retracted) | 8 distinct (full_idx 0-7) | yes (Z3xZ3 dom. character + weights) |
| `p7_2b_omega_fix_localized.json` | 12345 | 5 projected | no |
| `p7_2b_omega_fix_localized_seed42.json` | 42 | 5 projected | no |
| `p7_1b_omega_fix_converged_seeds.json` | 48879, 2, 12345, 51966, 4 | 1 (`lambda_lowest_trivial` only) | n/a |

Cross-seed *full-spectrum* data is unavailable: only seed=42 (P7.1) and
seed=12345 (P7.2b) have ≥5 eigenvalues. The 5-seed P7.1b file stores
only `lambda_lowest_trivial` per seed, so true 5-seed full-spectrum
cross-validation cannot be performed from existing JSON outputs. To
perform it, P7.1b would need to be re-run with full spectrum dumps.

## Methodology

For each available spectrum (8 + 5 + 5 eigenvalues), computed:

1. **All pairwise raw ratios** λ_i / λ_j  →  64 + 20 + 20 = 104 ratios
2. **All sigmoid ratios** λ_i / (λ_i + λ_j)  →  64 + 20 + 20 = 104 ratios
3. **Gap-fraction ratios** (λ_{i+1} − λ_i)/λ_{i+1} and λ_i/(λ_i+1)
4. **Character-aware ratios** on P7.1 (only file with character data):
   - λ_lowest_(0,0) / λ_lowest_(non-trivial)
   - λ_lowest_(0,0) / λ_max_(0,0)
   - λ_2nd_(0,0) / λ_lowest_(0,0)
   - All four sigmoid variants
5. **Offset check**: (1/2 − λ) × dim(E_8) — would test whether ω_fix
   encodes the offset 1/dim(E_8) from 1/2 directly
6. **Cross-seed consistency** (n=2: seed 42 vs 12345) on shared (i,j) indices

## Results

### Best raw-ratio matches to 123/248

| Rank | Kind | Seed | (i, j) | Value | ppm to 123/248 |
|---|---|---|---|---|---|
| 1 | sigmoid λ_2/(λ_2+λ_3) | 12345 | (2, 3) | 0.494885 | **2 183** |
| 2 | λ_2/(λ_2+1) | 42 | (2, —) | 0.487236 | 17 606 |
| 3 | sigmoid λ_3/(λ_2+λ_3) | 12345 | (3, 2) | 0.505115 | 18 444 |
| 4 | raw λ_2/λ_5 | 42 | (2, 5) | 0.506937 | 22 116 |
| 5 | raw λ_1/λ_4 | 42 | (1, 4) | 0.484248 | 23 629 |

Best single match is at 0.22% (2 183 ppm) — far above the 100 ppm
threshold needed to call this a confirmation.

### Best character-aware ratios (seed=42 only)

| Tag | Value | ppm |
|---|---|---|
| sigmoid λ_low(non-triv)/(λ_low(triv)+λ_low(non-triv)) | 0.460769 | 70 969 |
| sigmoid λ_low(triv)/(λ_low(triv)+λ_low(non-triv)) | 0.539231 | 87 229 |
| λ_low(triv)/λ_max(triv) | 0.213 | 571 433 |

The Schoen/Z3xZ3 lowest non-trivial character is the (1,0) representation
sitting *below* the lowest (0,0) trivial — yielding sigmoid ≈ 0.46 / 0.54
(7 % off ω_fix). Plausible-looking but no clean match.

### Cross-seed consistency (seed 42 vs 12345 only)

The 10 most consistent (i,j) raw/sigmoid pairs:

| (kind, i, j) | Mean | Spread | ppm to 123/248 |
|---|---|---|---|
| raw λ_0/λ_1 | 0.5245 | 150.4 % | 57 607 |
| raw λ_1/λ_2 | 0.5465 | 128.8 % | 101 964 |
| sigmoid λ_2/(λ_2+λ_3) | 0.4354 | 28.2 % | 122 142 |
| raw λ_2/λ_4 | 0.5621 | 12.3 % | 133 332 |
| sigmoid λ_3/(λ_2+λ_3) | 0.5646 | 21.8 % | 138 402 |

The lowest-ppm pair (`raw λ_0/λ_1`) has 150 % cross-seed spread —
i.e. the two seeds disagree on the value by more than 100 % of their
mean. The pair with smallest spread (`sigmoid λ_4/λ_2`, 4.4 %) is at
291 213 ppm (29 % off). **No (i,j) combination is simultaneously
near 123/248 AND consistent across seeds.**

### Offset check

`(1/2 − λ) × dim(E_8)` should be near a small integer (esp. ±1) if ω_fix
encodes a 1/dim(E_8) offset from 1/2. Closest hit:

- P7.1 seed=42, full_idx=1 (the lowest trivial λ): (1/2 − 0.5055)·248
  = **−1.366**, distance 0.366 from −1. Not integer.
- All other λ are far from 1/2, producing values from −832 to +94.

No integer-offset interpretation works.

### λ_lowest_trivial across all 5 P7.1b seeds (sanity check)

| Seed | λ_lowest_triv | ppm to 123/248 (raw) |
|---|---|---|
| 48879 | 0.3142 | 366 447 |
| 2 | 0.1152 | 767 766 |
| 12345 | 0.1424 | 712 881 |
| 51966 | 0.6362 | 282 757 |
| 4 | 0.0951 | 808 176 |

Spread is 0.10–0.64 — over 6× variation. Neither λ alone, λ/(λ+1),
nor 1−λ produces anything near 0.4960 across seeds. This re-confirms
P7.1b's verdict and also rules out the simplest single-seed
sigmoid/offset interpretations.

## Verdict

**Hypothesis (2) is FALSIFIED.** No eigenvalue ratio of the
metric-Laplacian spectrum on Schoen/Z3xZ3 reproduces ω_fix = 123/248
robustly (cross-seed-consistent at <100 ppm) under any of the
interpretations attempted (raw pairwise, sigmoid, gap-fraction,
character-aware lowest pairs, offset-from-half).

The best single-spectrum hit (sigmoid λ_2/(λ_2+λ_3) at seed=12345,
ppm = 2 183) is too noisy to survive cross-seed averaging — the same
(i,j) at seed=42 lands at sigmoid value 0.372 (P7.2b projected) and
0.439 (P7.1 raw), giving a 28 % cross-seed spread on the (i,j)=(2,3)
sigmoid family.

The gateway-formula falsification stands.

## Caveat — incomplete cross-seed coverage

Only 2 of the 5 strict-converged seeds have full bottom-spectrum data
captured. A definitive multi-seed ratio sweep would require re-running
P7.1b (or running a new P7.4-like Rust harness) with full bottom-15
eigenvalue dumps per seed, plus character weights. If that re-run shows
robust <100 ppm cluster on a specific (i,j) pair across all 5 seeds,
this verdict would need to be revisited. Based on the wild scatter of
λ_lowest_trivial across seeds (factor 6×), such a robust ratio is very
unlikely, but cannot be analytically ruled out from existing JSON.

## Recommendation

Proceed to **P7.5 — sub-Coxeter basis**: investigate whether ω_fix
arises naturally from the E_8 sub-algebra structure (e.g. Coxeter
number h = 30, dual h^∨ = 30, or breaking into D_8/E_7 sub-roots) rather
than from the metric Laplacian eigenvalue spectrum at all. The factor
123 = 248/2 − 1 strongly suggests a half-rank-minus-one combinatorial
identity, not a geometric eigenvalue.

## Files

- `output/p7_4_omega_fix_ratio_sweep.json` — full sweep results
- `scripts/p7_4_omega_fix_ratio_sweep.py` — analysis script
- `references/p7_4_omega_fix_ratio_search.md` — this file


## Closure (P7.12, 2026-04-29)

The "half-rank-minus-one combinatorial identity" hypothesis floated
above turned out to be exactly the journal's actual position. P7.12
verifies that

```
ω_fix = 1/2 − 1/dim(E_8) = (dim − 2)/(2·dim) = 246/496 = 123/248
```

is a simply-laced-Lie-algebra invariant (the gateway-mode amplitude
coupling fraction, journal §F.2.1 lines 2192-2194), **not** a
measurable eigenvalue of any operator on the Schoen Donaldson
background. The P7.5–P7.10 follow-on tests all failed to verify it as
an eigenvalue because the framework doesn't claim it as one.

See [`p7_12_omega_fix_reconciliation.md`](p7_12_omega_fix_reconciliation.md)
for the full classification of every journal appearance of `ω_fix` /
`123/248` and the regression tests that close the P7 series.
