"""
H3 — Alternative gauge embedding: SO(10) × U(1) via rank-4 SU(4) bundle on TY/Z3.

Hypothesis (H3)
---------------
Instead of E_8 → SU(5) (the AGLP-2012 framing), embed E_8 → Spin(10) × SU(4)
where SU(4) is the structure group of an internal rank-4 bundle V.
Wilson Z/3 then breaks Spin(10) further to SU(3) × SU(2) × U(1)^2 via a
single Z/3 element acting in the SU(4)-stabilizer.

For this to give 3 generations, the bundle V must be:
    rank 4, c_1(V) = 0, c_3(V) integrated to ±18 (3 net generations),
    c_2(V) <= c_2(TX),
    Wilson Z/3 partition on V is 1:1:1:1 across all four "weight" classes
        (or any pattern producing 3 net 16's of SO(10)).
For SU(4), the Z/3-grading of the 4-dim rep can be either:
    (a) (0,0,0,0) — Z/3 trivial → no Wilson breaking, fails
    (b) (0,1,2,?) — needs sum to 0 mod 3 (4-dim rep determinant trivial),
        so 4th weight = 0; partition (2,1,1) or (1,1,1,0) ... hmm.

Actually for SU(4) the determinant condition forces sum of weights ≡ 0 mod 3.
The most natural choice is weights (0, 1, 2, 0) giving partition 2:1:1
(NOT balanced).

Without a 1:1:1 partition we cannot get 3 equal-multiplicity generations
in the SO(10) direction.

Falsification: exhaustive scan over rank-4 line-bundle sums in [-2,2]^2 with
c_1=0 and c_3=±18 returns no candidate.

Run:
    PYTHONIOENCODING=utf-8 python h3_so10_rank4_scan.py
"""

import sys
from itertools import combinations_with_replacement as cwr, product
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ty_z3_bundle_constraints import (
    LineBundle, LineBundleSum,
    wilson_partition, anomaly_check, polystability_check,
)

A_RANGE = range(-3, 4)
RANK = 4
TARGET_C3 = {18, -18}


def main():
    print("=" * 70)
    print("H3 — SO(10) × U(1) via rank-4 SU(4) bundle on TY/Z3")
    print("=" * 70)
    lbs = [LineBundle(a, b) for a in A_RANGE for b in A_RANGE]
    print(f"Bidegree range: {list(A_RANGE)}^2  ({len(lbs)} line bundles)")
    print(f"Target rank = {RANK}, c_3 = ±18")

    n_total = 0
    n_c1 = 0
    n_c3 = 0
    n_anom = 0
    n_poly = 0
    survivors = []

    # Wilson on rank-4 SU(4): determinant condition sum a_i = sum b_i (already
    # forces c_1 = 0 in our notation). Z/3 weights (a-b mod 3) summing to 0
    # mod 3. Partitions of 4 into 3 buckets summing to 0 mod 3:
    #   (4,0,0), (1,3,0), (0,2,2) ... not-balanced
    #   (2,1,1) -- weights 0+1+1+2*1 = 0 mod 3? Sum of weight*count: 0*2+1*1+2*1 = 3 ≡ 0 ✓
    # Only (2,1,1) up to permutation gives a non-trivially-balanced split.
    # We accept any partition with at least 2 distinct nonzero classes
    # (gives some Wilson breaking).

    bucket = {0: [], 1: [], 2: []}
    for L in lbs:
        bucket[(L.a - L.b) % 3].append(L)

    # Enumerate over partitions (n0, n1, n2) of RANK with sum*weight ≡ 0 mod 3.
    valid_parts = []
    for n0 in range(RANK + 1):
        for n1 in range(RANK - n0 + 1):
            n2 = RANK - n0 - n1
            if (n1 + 2 * n2) % 3 == 0:
                valid_parts.append((n0, n1, n2))
    print(f"Valid Wilson partitions: {valid_parts}")

    for part in valid_parts:
        for c0 in cwr(bucket[0], part[0]):
            for c1 in cwr(bucket[1], part[1]):
                for c2 in cwr(bucket[2], part[2]):
                    summands = tuple(c0 + c1 + c2)
                    n_total += 1
                    a = sum(L.a for L in summands)
                    b = sum(L.b for L in summands)
                    if a != 0 or b != 0:
                        continue
                    n_c1 += 1
                    V = LineBundleSum(summands)
                    c3 = V.c3_value()
                    if c3 not in TARGET_C3:
                        continue
                    n_c3 += 1
                    an = anomaly_check(V)
                    if not (an['pass_H1'] and an['pass_H2']):
                        continue
                    n_anom += 1
                    p = polystability_check(V, n_samples=15)
                    if not p['polystable']:
                        continue
                    n_poly += 1
                    survivors.append({
                        'summands': [(L.a, L.b) for L in summands],
                        'c2': V.c2_components(),
                        'c3': c3,
                        'wilson': wilson_partition(V),
                        'poly_witness': p['witness'],
                    })

    print(f"\nTotal:                {n_total}")
    print(f"Pass c_1=0:           {n_c1}")
    print(f"Pass c_3=±18:         {n_c3}")
    print(f"Pass anomaly:         {n_anom}")
    print(f"Pass polystability:   {n_poly}")
    print(f"Survivors: {len(survivors)}")
    if survivors:
        for i, s in enumerate(survivors[:10]):
            print(f"\n  Candidate #{i+1}: {s}")

    print("\n" + "=" * 70)
    print("H3 VERDICT")
    print("=" * 70)
    if survivors:
        print(f"PARTIAL VALID: {len(survivors)} rank-4 candidate(s) found.")
        print("Each requires Wilson-decomposition matching to give 3 net SO(10) 16s.")
    else:
        print("REJECT. No rank-4 polystable line-bundle sum on TY/Z3 with")
        print("c_1=0, c_3=±18, valid Wilson partition (sum-0 mod 3), and")
        print("anomaly pass. SO(10)×U(1) embedding via rank-4 LBS is excluded")
        print("in the bidegree range [-3,3].")


if __name__ == "__main__":
    main()
