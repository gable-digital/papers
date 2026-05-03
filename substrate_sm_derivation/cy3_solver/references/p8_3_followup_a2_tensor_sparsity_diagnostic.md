# P8.3-followup-A2 — T-tensor sparsity diagnostic

## Goal

P8.3-followup-A established that the harmonic zero modes ψ_α are
full rank (9/9 at machine precision). The 8/9 zero-mass collapse
must therefore originate downstream of the harmonic basis. Two
candidate causes were left open:

* **(a) Geometric (T-sparse).** The triple-overlap T_{ijk} produced
  by `compute_yukawa_couplings` is itself sparse / low-rank at the
  Higgs slice (a property of the bundle/metric and the overlap
  integral that no choice of sector assignment can fix).
* **(b) Assignment-driven.** T_{ijk} has many non-zero entries, but
  `assign_sectors_dynamic`'s round-robin sector buckets pull from
  disjoint mode pools at the chosen Higgs slice h_0, zeroing 8/9
  entries through index mismatch.

The diagnostic binary
`src/bin/p8_3_followup_a2_tensor_sparsity_diag.rs` localises the
cause by computing the full T-tensor on TY/Z3 (k=3) and Schoen/Z3xZ3
(d=(3,3,1)) and cross-referencing the round-robin sector buckets
against the non-zero entries of the T_{i,j,h_0} slice.

Run:

```
cargo run --release --features "gpu precision-bigfloat" \
  --bin p8_3_followup_a2_tensor_sparsity_diag \
  2>&1 | tee output/p8_3_followup_a2_tensor_sparsity.log
```

CSV dumps of the full 9×9×9 (TY) / 12×12×12 (Schoen) tensors are
written to `output/p8_3_followup_a2_tensor_TY.csv` and
`output/p8_3_followup_a2_tensor_Schoen.csv` (one row per entry:
`i,j,k,abs,arg,re,im`).

## Sparsity statistics

| label        | n_modes | n_nonzero | total | sparsity (zeros/N) | max\|T\|   | min nz\|T\| | mean nz\|T\| |
|--------------|--------:|----------:|------:|-------------------:|-----------:|------------:|-------------:|
| TY/Z3        |       9 |       343 |   729 |             0.5295 | 4.626e+00  | 2.567e-02   | 2.947e-01    |
| Schoen/Z3xZ3 |      12 |      1271 |  1728 |             0.2645 | 2.676e+05  | 1.014e-10   | 6.759e+03    |

Both candidates yield T-tensors that are very far from being rank-1.
The TY tensor is roughly 47 % filled, the Schoen tensor 73 % filled.

## Per-Higgs-slice rank (n × n submatrix at fixed k)

### TY/Z3 (n=9)

| k | rank | nz_count | max\|T\| | fro_norm |
|---|-----:|---------:|---------:|---------:|
| 0 | **7** |  49 |  1.209e+0 | 2.516e+0 |
| 1 |    7 |  49 |  3.672e-1 | 1.444e+0 |
| 2 |    7 |  49 |  9.422e-1 | 2.678e+0 |
| 3 |    7 |  49 |  6.375e-1 | 1.149e+0 |
| 4 |    7 |   0 |       ~0  | 6.93e-12 |
| 5 |    7 |   0 |       ~0  | 5.04e-12 |
| 6 |    7 |  49 |  1.479e+0 | 3.738e+0 |
| 7 |    7 |  49 |  1.064e+0 | 2.302e+0 |
| 8 |    7 |  49 |  4.626e+0 | 5.672e+0 |

Note: slices k=4 and k=5 are numerically zero (likely a Wilson-line
selection rule), but the h_0 = mode 0 slice has rank 7 with 49
non-zero (i,j) entries out of 81. **Far from rank-1.**

### Schoen/Z3xZ3 (n=12)

| k  | rank | nz_count | max\|T\| | fro_norm |
|----|-----:|---------:|---------:|---------:|
|  0 | **8** | 122 | 2.145e+1 | 3.852e+1 |
|  1 |    7 |  63 | 1.220e-5 | 2.091e-5 |
|  2 |    7 |  64 | 1.283e-3 | 2.586e-3 |
|  3 |    7 | 128 | 5.232e+4 | 1.137e+5 |
|  4 |    7 | 128 | 9.640e+4 | 2.652e+5 |
|  5 |    7 | 128 | 9.640e+4 | 1.834e+5 |
|  6 |    7 | 128 | 1.066e+5 | 1.522e+5 |
|  7 |    7 |  63 | 9.078e-5 | 1.554e-4 |
|  8 |    7 |  63 | 3.444e-5 | 6.364e-5 |
|  9 |    7 | 128 | 9.739e+4 | 2.086e+5 |
| 10 |    7 | 128 | 2.676e+5 | 4.307e+5 |
| 11 |    7 | 128 | 1.436e+5 | 2.790e+5 |

The Schoen h_0 (mode 0) slice has rank 8 with 122 non-zero (i,j)
entries out of 144. Again, very far from rank-1.

## Sector assignment vs sparsity cross-reference

`assign_sectors_dynamic` returns the round-robin fallback for both
candidates, because the AKLP-aliased bundle exposes only one Wilson
Z/N phase class (everything maps to the trivial / zero phase under
the canonical Wilson-line projection). The buckets are:

* **TY/Z3:** up=[0,3,6], down=[1,4,7], lepton=[2,5,8],
  higgs=[0,1,2,3,4,5,6,7,8] sorted by ascending eigenvalue (h_0=0).
* **Schoen/Z3xZ3:** up=[0,3,6,9], down=[1,4,7,10], lepton=[2,5,8,11],
  higgs=[0..11] (h_0=0).

Cross-referencing the 9 (li, rj) buckets that
`extract_3x3_from_tensor` evaluates at h_0 against the non-zero
(i, j) coordinates on the T_{:,:,h_0} slice:

| candidate | sector pair          | buckets evaluated at h_0                                  | hit / 9 |
|-----------|----------------------|-----------------------------------------------------------|--------:|
| TY/Z3     | Y_u (up × up)        | (0,0)(0,3)(0,6)(3,0)(3,3)(3,6)(6,0)(6,3)(6,6)             | **9/9** |
| TY/Z3     | Y_d (up × down)      | (0,1)(0,4)(0,7)(3,1)(3,4)(3,7)(6,1)(6,4)(6,7)             | **6/9** |
| TY/Z3     | Y_e (lepton × lepton)| (2,2)(2,5)(2,8)(5,2)(5,5)(5,8)(8,2)(8,5)(8,8)             | **4/9** |
| Schoen    | Y_u (up × up)        | (0,0)(0,3)(0,6)(3,0)(3,3)(3,6)(6,0)(6,3)(6,6)             | **9/9** |
| Schoen    | Y_d (up × down)      | (0,1)(0,4)(0,7)(3,1)(3,4)(3,7)(6,1)(6,4)(6,7)             | **7/9** |
| Schoen    | Y_e (lepton × lepton)| (2,2)(2,5)(2,8)(5,2)(5,5)(5,8)(8,2)(8,5)(8,8)             | **5/9** |

The actual `Y_u` block on TY/Z3 at h_0 reads (magnitudes):

```
Y_u (up x up)
           rj=0       rj=3       rj=6
li=0  : +6.171e-1 +8.407e-2 +2.184e-1
li=3  : +8.407e-2 +4.397e-2 +1.909e-1
li=6  : +2.184e-1 +1.909e-1 +5.621e-1
```

All 9 entries are O(0.05–0.6) — every bucket has a non-trivial
value. The other two sectors (`Y_d`, `Y_e`) hit fewer cells because
the round-robin assignment pairs columns in the j=4 and j=5 columns
which lie in the all-zero j-indices the Wilson selection rule kills
within those cross-sector buckets.

## Verdict — cause (b) ASSIGNMENT-DRIVEN

Both candidates produce non-sparse, near-full-rank T-tensors at the
Higgs slice (TY rank 7, Schoen rank 8 out of n=9 / n=12). The
8/9-zero collapse documented in P8.3b therefore did **not** come
from a geometric T-sparsity. It came from the round-robin sector
assignment (which only fires because the AKLP-aliased bundle
collapses every mode into a single Wilson phase class) pairing
left/right indices that do not coincide with the non-zero entries of
the h_0 slice in the Y_d and Y_e blocks.

Crucially, **Y_u clears 9/9 buckets on both TY and Schoen** with
realistic magnitudes — confirming the diagnostic that the issue is
limited to the cross-sector blocks where round-robin index pairing
collides with Wilson-line selection rules.

## Recommended next task — P8.3-followup-B

Dispatch P8.3-followup-B (real Schoen 3-factor bundle providing 3
Wilson Z/3 phase classes). Once `assign_sectors_dynamic` sees three
non-empty by-class buckets, the round-robin fallback at line 183 of
`yukawa_sectors_real.rs` disengages and the journal-canonical
class-0 / class-1 / class-2 assignment takes over. With the real
Schoen bundle the up-quark, down-quark, and lepton sectors will be
populated by modes from the corresponding Wilson phase classes
rather than by interleaved index-mod-3 fallbacks; bucket coordinates
will then naturally land on the non-zero T-tensor coordinates the
geometry already provides, lifting Y_d and Y_e to full rank.

The geometric content (T-tensor) is already adequate. The fix is
upstream of the contraction layer, in the bundle-side phase-class
structure.

## Provenance / non-modifications

* **No production code modified.** Only added:
  * `src/bin/p8_3_followup_a2_tensor_sparsity_diag.rs`
  * `[[bin]]` stanza in `Cargo.toml`
  * this reference doc.
* PID 2270068 (`p5_10_ty_schoen_5sigma.exe`) verified alive both
  before and after the build (started 4/30/2026 9:52:00 PM).
* P8.4-fix-c is in flight modifying `schoen_metric.rs` and
  `ty_metric.rs`; this diagnostic does not touch either file.
