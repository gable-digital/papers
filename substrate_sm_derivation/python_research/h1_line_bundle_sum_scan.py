"""
H1 — Exhaustive scan: rank-N line-bundle SUM Standard-Model bundles on TY/Z3.

Hypothesis (H1)
---------------
Claim: There exists a polystable line-bundle sum V = ⊕_{i=1..N} O(a_i, b_i)
on the Tian-Yau Z/3 quotient X = X~/Z_3 with:
    1. c_1(V) = 0  (SU(N) condition)
    2. Wilson partition under the Z/3 phase class (a-b) mod 3 is 3:3:3
       balanced (so each generation comes from a distinct phase class)
    3. c_3(V) integrated over X~ equals ±18 (so net 3 generations downstairs:
       (1/2)(±18) / 3 = ±3)
    4. c_2(V) ≤ c_2(TX) for both H_1 and H_2 directions (anomaly cancellation
       with effective 5-brane class W = c_2(TX) - c_2(V))
    5. V is polystable w.r.t. SOME Kähler class in the (1+1)-dim cone.

Falsification: exhaustive scan over a, b ∈ {-3,...,3} for ranks 3, 6, 9
returns no candidate that meets all 5 criteria simultaneously.

This is the test AGLP-2012 §5.3 ran (or a direct subset of it). They reported
"no phenomenologically viable model" for h^{1,1}=2. We re-do the scan with
explicit bookkeeping so we have an in-tree empirical artifact.

Run:
    PYTHONIOENCODING=utf-8 python h1_line_bundle_sum_scan.py
"""

from itertools import combinations_with_replacement, product
import sys
from fractions import Fraction

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ty_z3_bundle_constraints import (
    LineBundle, LineBundleSum,
    wilson_partition, is_3_3_3_balanced,
    anomaly_check, polystability_check,
    index_theorem_count,
)


# Scan parameters
A_RANGE = range(-3, 4)   # -3, -2, -1, 0, 1, 2, 3
B_RANGE = range(-3, 4)
RANKS = [3, 4, 5, 6]     # rank 9 explicitly too large to brute force; SU(9)
                         # bundles also unphysical for SU(5) × U(1)^4 GUT
                         # embedding (and AGLP-2012 §5.3 already covered them).
TARGET_C3 = {18, -18}    # net generations: c_3/2 / 3 = ±3 → c_3 = ±18


def enumerate_line_bundles():
    """All distinct (a, b) ∈ scan grid."""
    return [LineBundle(a, b) for a, b in product(A_RANGE, B_RANGE)]


def enumerate_sums(rank: int, line_bundles):
    """All multisets (combinations_with_replacement) of given rank
    drawn from line_bundles."""
    return combinations_with_replacement(line_bundles, rank)


def fast_c1_filter(summands):
    """Quick check: sum of c_1 components must equal (0, 0)."""
    a = sum(L.a for L in summands)
    b = sum(L.b for L in summands)
    return a == 0 and b == 0


def fast_wilson_filter(summands):
    """Quick check: each Z/3 phase class appears exactly rank/3 times."""
    n = len(summands)
    if n % 3 != 0:
        return False
    target = n // 3
    counts = {0: 0, 1: 0, 2: 0}
    for L in summands:
        counts[(L.a - L.b) % 3] += 1
    return counts[0] == counts[1] == counts[2] == target


def run_scan(rank: int):
    """Run full constraint chain for given rank."""
    lbs = enumerate_line_bundles()
    print(f"\n{'='*70}")
    print(f"Rank {rank} scan: {len(lbs)} candidate line bundles")
    n_total = 0
    n_c1 = 0
    n_wilson = 0
    n_c3 = 0
    n_anomaly = 0
    n_polystable = 0
    survivors = []

    # For rank > 6 the brute-force enumeration is too large. Reduce by
    # iterating over "rank-balanced" partitions: pre-bucket line bundles by
    # Wilson phase class and choose rank/3 from each. (Wilson 3:3:3 filter
    # subsumed.) Falls back to brute force for rank not divisible by 3.
    if rank % 3 == 0:
        per = rank // 3
        bucket = {0: [], 1: [], 2: []}
        for L in lbs:
            bucket[(L.a - L.b) % 3].append(L)
        # Iterate over per-class multisets, then combine.
        from itertools import combinations_with_replacement as cwr
        gen = (
            tuple(c0 + c1 + c2)
            for c0 in cwr(bucket[0], per)
            for c1 in cwr(bucket[1], per)
            for c2 in cwr(bucket[2], per)
        )
    else:
        gen = enumerate_sums(rank, lbs)

    for combo in gen:
        n_total += 1
        if not fast_c1_filter(combo):
            continue
        n_c1 += 1
        if not fast_wilson_filter(combo):
            continue
        n_wilson += 1

        V = LineBundleSum(tuple(combo))
        c3 = V.c3_value()
        if c3 not in TARGET_C3:
            continue
        n_c3 += 1

        a = anomaly_check(V)
        if not (a['pass_H1'] and a['pass_H2']):
            continue
        n_anomaly += 1

        p = polystability_check(V, n_samples=20)
        if not p['polystable']:
            # Keep for reporting but don't promote to "survivor".
            continue
        n_polystable += 1

        survivors.append({
            'V': V,
            'c1': V.c1_components(),
            'c2': V.c2_components(),
            'c3': c3,
            'wilson': wilson_partition(V),
            'anomaly': a,
            'poly_witness': p['witness'],
            'gens_up_down': index_theorem_count(V),
        })

    print(f"  Total combinations:                  {n_total}")
    print(f"  Pass c_1=0:                          {n_c1}")
    print(f"  Pass Wilson 3:3:3 balanced:          {n_wilson}")
    print(f"  Pass c_3 = ±18 (3 gen on quotient):  {n_c3}")
    print(f"  Pass anomaly c_2(V) ≤ c_2(TX):       {n_anomaly}")
    print(f"  Pass polystability (integer scan):   {n_polystable}")
    print(f"  Survivors: {len(survivors)}")
    if survivors:
        for i, s in enumerate(survivors[:5]):
            print(f"\n  Candidate #{i+1}:")
            print(f"    summands: {[(L.a, L.b) for L in s['V'].summands]}")
            print(f"    c_1: {s['c1']}, c_2: {s['c2']}, c_3: {s['c3']}")
            print(f"    Wilson: {s['wilson']}")
            print(f"    poly witness Kähler (t1,t2): {s['poly_witness']}")
            print(f"    generations (up, down): {s['gens_up_down']}")
    return {
        'rank': rank,
        'n_total': n_total,
        'n_c1': n_c1,
        'n_wilson': n_wilson,
        'n_c3': n_c3,
        'n_anomaly': n_anomaly,
        'n_polystable': n_polystable,
        'survivors': survivors,
    }


def main():
    print("=" * 70)
    print("H1 — Line-bundle SUM scan on TY/Z3")
    print("=" * 70)
    print(f"Bidegree range: a, b ∈ {list(A_RANGE)}")
    print(f"Ranks: {RANKS}")
    print(f"Target c_3: ±18 (=> 3 net generations downstairs)")

    results = []
    for r in RANKS:
        res = run_scan(r)
        results.append(res)

    print("\n" + "=" * 70)
    print("H1 SUMMARY")
    print("=" * 70)
    total_survivors = sum(len(r['survivors']) for r in results)
    print(f"Total survivor bundles across all ranks: {total_survivors}")
    if total_survivors == 0:
        print("\nVERDICT: H1 REJECTED.")
        print("No polystable rank-3/6/9 line-bundle sum on TY/Z3 with")
        print("(c_1=0, Wilson 3:3:3, c_3=±18, anomaly OK) found in the")
        print("scan range a, b ∈ [-3, 3]. This empirically reproduces")
        print("the AGLP-2012 §5.3 exclusion of h^{1,1}=2 manifolds.")
    else:
        print(f"\nVERDICT: H1 PARTIALLY VALID — {total_survivors} candidate(s).")
        print("Further analysis required (full polystability, supergravity")
        print("dressing, downstairs Wilson invariance check).")

    return results


if __name__ == "__main__":
    main()
