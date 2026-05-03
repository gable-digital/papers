"""
Cycle 5 — Basis-rank-corrected H2 monad bundle scan on TY/Z3.

Hypothesis (cycle 5)
--------------------
There exists a rank-3 monad bundle V = ker(B -> C) on TY/Z_3 with
rank(B) >= 6, satisfying:
    1. c_1(V) = 0  (i.e. c_1(B) = c_1(C))
    2. Wilson Z_3 partition on V's full upstairs basis has >= 3 modes
       per phase class (chi(V)_p in absolute value, computed as
       sum of ch_3 contributions per class)
    3. int c_3(V) = +/- 18  (3 net generations downstairs)
    4. c_2(V) <= c_2(TX)  (anomaly cancellation)
    5. V is poly-stable (line-destabilizer test on integer Kähler classes)
    6. The monad map B -> C exists generically (necessary feeders heuristic)

Implementation note
-------------------
Naive enumeration of rank-7 and rank-8 line-bundle multisets blows up
(C(32, 8) ~ 10.5M for rank 8). We pre-bucket by Wilson class AND apply
the cycle-5 |chi(V)_p| >= 3 gate AT THE WILSON-CLASS LEVEL (not on the
cross product B x C). Chi(V)_p depends only on the per-class bidegree
multiset, so we can filter Wilson-resolved sub-multisets independently
and only join the survivors.

Then we apply: c_1(V) = 0 (fast), c_3(V) = +/-18 (fast), polystability
(no strictly-positive B-summand — fast), anomaly, surjectivity.

Run:
    python h2_basis_rank_corrected_scan.py 2>&1 | tee output/h2_basis_rank_corrected_scan.log
"""

import sys
import time
from itertools import combinations_with_replacement as cwr
from fractions import Fraction
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Force unbuffered stdout so tee shows progress in real time
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, write_through=True, encoding='utf-8', errors='replace')

from ty_z3_bundle_constraints import (
    LineBundle, LineBundleSum, D,
    wilson_z3_phase, integrate_c2_against,
    is_3_generation_basis_compatible, wilson_partition_modes_per_class,
    _ch3_int_of_lb,
)
from h2_monad_bundle_scan import (
    monad_chern, wilson_partition_monad, anomaly_check_monad,
    map_existence_check,
)


# ---- Polystability (necessary): every B-summand has slope <= 0 EVERYWHERE ----
#
# Slope of L = O(a, b) at Kähler class J = t1 H1 + t2 H2:
#   mu(L) = 9 [a Q1(t1,t2) + b Q2(t1,t2)]
#   Q1 = t2(2t1+t2),  Q2 = t1(t1+2t2)   (both strictly positive for t1, t2 > 0)
#
# As (t1, t2) varies over the open positive quadrant, the ratio Q2/Q1 sweeps
# the entire open interval (0, infinity). So sign(mu) sweeps:
#   - if a > 0 AND b > 0: mu > 0 always
#   - if a >= 0 AND b >= 0 with (a,b) != (0,0): mu > 0 always (boundary still)
#   - if a < 0 AND b < 0: mu < 0 always
#   - if a, b have opposite signs OR one of them is positive while the other
#     is negative: mu changes sign — there exists (t1, t2) > 0 with mu > 0.
#
# Therefore the truly conservative line-destabilizer rejection is:
#   ANY B-summand with a > 0 OR b > 0 destabilizes V at SOME Kähler class.
# The only safe B-summands are those with a <= 0 AND b <= 0 (closed negative
# cone, including O(0,0) which has mu = 0 always).
#
# Cycle 1-3 used the looser "strictly positive a>0 AND b>0" filter; this was
# verified by cycle 5 drilldown to be inadequate (boundary destabilizers like
# O(2, 0), O(0, 1), and even mixed O(-1, 2), O(2, -1) all show mu > 0 at
# some random Kähler class).
def line_potentially_positive(L: LineBundle) -> bool:
    """True iff slope(L) > 0 at SOME Kähler class in the open positive cone.
    Equivalent to: a > 0 OR b > 0 (with at least one strict)."""
    return L.a > 0 or L.b > 0


def line_uniformly_positive(L: LineBundle) -> bool:
    """True iff slope(L) > 0 at EVERY Kähler class. a >= 0, b >= 0, not (0,0)."""
    if L.a == 0 and L.b == 0:
        return False
    return L.a >= 0 and L.b >= 0


def line_strictly_positive(L: LineBundle) -> bool:
    """Legacy alias (strict open positive cone)."""
    return L.a > 0 and L.b > 0


def polystability_necessary(B_summands, C_summands):
    """Reject if any B-summand has potentially-positive slope at some Kähler class."""
    bad = [L for L in B_summands if line_potentially_positive(L)]
    if bad:
        return False, f"B has potentially-positive-slope summands {[(L.a, L.b) for L in bad]}"
    return True, "all B-summands have slope <= 0 at every Kähler class"


# ---- Strict surjectivity heuristic ----
def feeders_for_C_summand(B, c_summand):
    return sum(1 for L in B if L.a <= c_summand.a and L.b <= c_summand.b)


def strict_surjectivity(B_summands, C_summands):
    r_B = len(B_summands)
    r_C = len(C_summands)
    for cL in C_summands:
        f = feeders_for_C_summand(B_summands, cL)
        if f < 2:
            return False, f"C-summand ({cL.a},{cL.b}) has only {f} feeders (need >= 2)"
        if f * r_C < r_B:
            return False, f"C-summand ({cL.a},{cL.b}) has {f} feeders, " \
                          f"insufficient for rank balance r_B/r_C = {r_B}/{r_C}"
    n_total = sum(feeders_for_C_summand(B_summands, cL) for cL in C_summands)
    if n_total < 2 * r_C:
        return False, f"only {n_total} total feeders, need >= {2 * r_C}"
    return True, f"strict surjectivity passes ({n_total} feeders)"


# ---- Scan parameters ----
A_RANGE = range(-2, 3)
B_RANGE = range(-2, 3)
RANK_V = 3
RANK_B_C_PAIRS = [(6, 3), (7, 4), (8, 5)]
TARGET_C3 = {18, -18}
WILSON_CLASS_TARGET = 3


def enumerate_lbs_safe_for_B():
    """Closed negative cone (a <= 0 AND b <= 0). Conservative line-destabilizer
    polystability filter: these are the only B-summands whose slope is <= 0
    at every Kähler class in the open positive quadrant."""
    return [LineBundle(a, b) for a in A_RANGE for b in B_RANGE
            if a <= 0 and b <= 0]


def enumerate_lbs_excluding_uniformly_positive():
    """Looser: closed positive cone minus origin excluded (a>=0, b>=0, not 0,0)."""
    return [LineBundle(a, b) for a in A_RANGE for b in B_RANGE
            if not (a >= 0 and b >= 0 and not (a == 0 and b == 0))]


def enumerate_lbs_excluding_strictly_positive():
    """Legacy cycle 1-3 (strict open positive cone excluded)."""
    return [LineBundle(a, b) for a in A_RANGE for b in B_RANGE
            if not (a > 0 and b > 0)]


def enumerate_lbs_full():
    return [LineBundle(a, b) for a in A_RANGE for b in B_RANGE]


def per_class_chi(combo):
    """Per-Wilson-class integrated ch_3 contribution of a multiset of
    line bundles. Returns dict {0,1,2} -> Fraction."""
    chi = {0: Fraction(0), 1: Fraction(0), 2: Fraction(0)}
    for L in combo:
        chi[(L.a - L.b) % 3] += _ch3_int_of_lb(L)
    return chi


def per_class_c1(combo):
    """Per-Wilson-class (a, b) sums."""
    out = {0: (0, 0), 1: (0, 0), 2: (0, 0)}
    for L in combo:
        p = (L.a - L.b) % 3
        a, b = out[p]
        out[p] = (a + L.a, b + L.b)
    return out


def main():
    print("=" * 78, flush=True)
    print("Cycle 5 — Basis-rank-corrected H2 monad scan on TY/Z3", flush=True)
    print("=" * 78, flush=True)
    lbs_B = enumerate_lbs_safe_for_B()
    lbs_C = enumerate_lbs_full()
    print(f"B alphabet: {len(lbs_B)} line bundles "
          f"(closed negative cone a<=0 AND b<=0; conservative line-destabilizer filter)", flush=True)
    print(f"C alphabet: {len(lbs_C)} line bundles", flush=True)
    print(f"Rank pairs (B, C): {RANK_B_C_PAIRS}  (V = B - C, rank V = 3)", flush=True)
    print(f"Target c_3(V) = +/- 18", flush=True)
    print(f"Cycle-5 gate: |chi(V)_p| >= {WILSON_CLASS_TARGET} per Wilson class p", flush=True)
    print(flush=True)

    # Bucket line bundles by Wilson class for B and C separately
    bucket_B = {0: [], 1: [], 2: []}
    for L in lbs_B:
        bucket_B[(L.a - L.b) % 3].append(L)
    bucket_C = {0: [], 1: [], 2: []}
    for L in lbs_C:
        bucket_C[(L.a - L.b) % 3].append(L)
    for p in (0, 1, 2):
        print(f"  B class {p}: {len(bucket_B[p])} | C class {p}: {len(bucket_C[p])}", flush=True)
    print(flush=True)

    survivors = []
    counters = defaultdict(int)

    t0 = time.time()

    for r_B, r_C in RANK_B_C_PAIRS:
        print(f"\n--- Scanning (rank B, rank C) = ({r_B}, {r_C}) ---", flush=True)
        t_pair = time.time()

        # Iterate over Wilson partitions of B and C such that
        # b_part - c_part has all entries >= 0 and sums to RANK_V = 3.
        # For each per-class pair (B_p, C_p), pre-build all possible per-class
        # combos and pre-filter by per-class chi gates.

        # Per-class B sub-multisets: for each class p and each size n,
        # enumerate cwr(bucket_B[p], n) and store with its chi and c_1.
        # Similarly for C.
        B_sub = {p: defaultdict(list) for p in (0, 1, 2)}  # B_sub[p][n] = list of (combo, chi_p, c1_p)
        C_sub = {p: defaultdict(list) for p in (0, 1, 2)}

        max_size_B_per_class = r_B  # could be all in one class
        max_size_C_per_class = r_C

        for p in (0, 1, 2):
            for n in range(0, max_size_B_per_class + 1):
                for combo in cwr(bucket_B[p], n):
                    chi_p = sum((_ch3_int_of_lb(L) for L in combo), Fraction(0))
                    a_sum = sum(L.a for L in combo)
                    b_sum = sum(L.b for L in combo)
                    B_sub[p][n].append((combo, chi_p, (a_sum, b_sum)))
            for n in range(0, max_size_C_per_class + 1):
                for combo in cwr(bucket_C[p], n):
                    chi_p = sum((_ch3_int_of_lb(L) for L in combo), Fraction(0))
                    a_sum = sum(L.a for L in combo)
                    b_sum = sum(L.b for L in combo)
                    C_sub[p][n].append((combo, chi_p, (a_sum, b_sum)))

        n_B_total = sum(len(B_sub[p][n]) for p in (0,1,2) for n in B_sub[p])
        n_C_total = sum(len(C_sub[p][n]) for p in (0,1,2) for n in C_sub[p])
        print(f"  Per-class B sub-multisets cached: {n_B_total}", flush=True)
        print(f"  Per-class C sub-multisets cached: {n_C_total}", flush=True)

        # Iterate Wilson-partition skeleton
        for n0_B in range(r_B + 1):
            for n1_B in range(r_B - n0_B + 1):
                n2_B = r_B - n0_B - n1_B
                for n0_C in range(min(n0_B, r_C) + 1):
                    for n1_C in range(min(n1_B, r_C - n0_C) + 1):
                        n2_C = r_C - n0_C - n1_C
                        if n2_C < 0 or n2_C > n2_B:
                            continue
                        # V's Wilson rank distribution per class:
                        v0 = n0_B - n0_C
                        v1 = n1_B - n1_C
                        v2 = n2_B - n2_C
                        if v0 + v1 + v2 != RANK_V:
                            continue

                        # Iterate per-class B combos (with per-class chi)
                        for B0, chi0_B, c1_0_B in B_sub[0][n0_B]:
                            for C0, chi0_C, c1_0_C in C_sub[0][n0_C]:
                                chi0_V = chi0_B - chi0_C
                                if abs(chi0_V) < WILSON_CLASS_TARGET:
                                    continue
                                a0_V = c1_0_B[0] - c1_0_C[0]
                                b0_V = c1_0_B[1] - c1_0_C[1]
                                for B1, chi1_B, c1_1_B in B_sub[1][n1_B]:
                                    for C1, chi1_C, c1_1_C in C_sub[1][n1_C]:
                                        chi1_V = chi1_B - chi1_C
                                        if abs(chi1_V) < WILSON_CLASS_TARGET:
                                            continue
                                        a1_V = c1_1_B[0] - c1_1_C[0]
                                        b1_V = c1_1_B[1] - c1_1_C[1]
                                        for B2, chi2_B, c1_2_B in B_sub[2][n2_B]:
                                            for C2, chi2_C, c1_2_C in C_sub[2][n2_C]:
                                                chi2_V = chi2_B - chi2_C
                                                if abs(chi2_V) < WILSON_CLASS_TARGET:
                                                    continue
                                                counters['post_class_chi'] += 1

                                                # c_1(V) = 0
                                                a_V = a0_V + a1_V + (c1_2_B[0] - c1_2_C[0])
                                                b_V = b0_V + b1_V + (c1_2_B[1] - c1_2_C[1])
                                                if a_V != 0 or b_V != 0:
                                                    continue
                                                counters['pass_c1'] += 1

                                                # c_3(V) = sum chi_p * 2 (since c_3 = 2*ch_3 when c_1=0)
                                                # Actually chi_p here is integrated ch_3 of summands.
                                                # c_3(V) integrated = 2 * (chi0_V + chi1_V + chi2_V)
                                                c3_V = 2 * (chi0_V + chi1_V + chi2_V)
                                                if c3_V not in TARGET_C3:
                                                    continue
                                                counters['pass_c3'] += 1

                                                B_combo = B0 + B1 + B2
                                                C_combo = C0 + C1 + C2

                                                # Compute c_2 fully
                                                m = monad_chern(B_combo, C_combo)
                                                a = anomaly_check_monad(m['c2_V'])
                                                if not (a['pass_H1'] and a['pass_H2']):
                                                    continue
                                                counters['pass_anomaly'] += 1

                                                # Polystability necessary (B has no strictly positive)
                                                # — already enforced by alphabet
                                                counters['pass_polystab'] += 1

                                                # Map existence
                                                if not map_existence_check(B_combo, C_combo):
                                                    continue
                                                counters['pass_map_exist'] += 1

                                                # Strict surjectivity
                                                ok_surj, surj_reason = strict_surjectivity(
                                                    B_combo, C_combo
                                                )
                                                if not ok_surj:
                                                    continue
                                                counters['pass_strict_surj'] += 1

                                                survivors.append({
                                                    'r_B': r_B,
                                                    'r_C': r_C,
                                                    'B': [(L.a, L.b) for L in B_combo],
                                                    'C': [(L.a, L.b) for L in C_combo],
                                                    'c1_V': m['c1_V'],
                                                    'c2_V': m['c2_V'],
                                                    'c3_V_int': m['c3_V_int'],
                                                    'wilson_V_rank': (v0, v1, v2),
                                                    'chi_per_class': {
                                                        0: float(chi0_V),
                                                        1: float(chi1_V),
                                                        2: float(chi2_V),
                                                    },
                                                    'anomaly': a,
                                                    'surj_reason': surj_reason,
                                                })
                                                if len(survivors) <= 5:
                                                    print(f"    [survivor #{len(survivors)}] "
                                                          f"r_B={r_B}, B={[(L.a,L.b) for L in B_combo]}, "
                                                          f"C={[(L.a,L.b) for L in C_combo]}, "
                                                          f"c3={c3_V}, "
                                                          f"chi=({float(chi0_V)},{float(chi1_V)},{float(chi2_V)})",
                                                          flush=True)

        dt = time.time() - t_pair
        print(f"  pair done in {dt:.1f}s, post-chi-gate joins: {counters['post_class_chi']:,}, "
              f"running survivors: {len(survivors)}", flush=True)

    dt_total = time.time() - t0
    print(flush=True)
    print("=" * 78, flush=True)
    print("Scan summary", flush=True)
    print("=" * 78, flush=True)
    print(f"  Total scan time:                     {dt_total:.1f}s", flush=True)
    print(f"  Joins post per-class chi gate:       {counters['post_class_chi']:,}", flush=True)
    print(f"  Pass c_1(V) = 0:                     {counters['pass_c1']:,}", flush=True)
    print(f"  Pass c_3(V) = +/-18:                 {counters['pass_c3']:,}", flush=True)
    print(f"  Pass anomaly:                        {counters['pass_anomaly']:,}", flush=True)
    print(f"  Pass polystability:                  {counters['pass_polystab']:,}", flush=True)
    print(f"  Pass map existence:                  {counters['pass_map_exist']:,}", flush=True)
    print(f"  Pass strict surjectivity:            {counters['pass_strict_surj']:,}", flush=True)
    print(f"  FINAL SURVIVORS: {len(survivors)}", flush=True)

    if survivors:
        print(flush=True)
        print("=" * 78, flush=True)
        print(f"Survivors (showing first 10 of {len(survivors)})", flush=True)
        print("=" * 78, flush=True)
        for i, s in enumerate(survivors[:10]):
            print(f"\nSurvivor #{i + 1}: rank-({s['r_B']},{s['r_C']}) monad", flush=True)
            print(f"  B = O{tuple(s['B'])}", flush=True)
            print(f"  C = O{tuple(s['C'])}", flush=True)
            print(f"  c_1(V) = {s['c1_V']}", flush=True)
            print(f"  c_2(V) = {s['c2_V'][0]} H1^2 + {s['c2_V'][1]} H1H2 + {s['c2_V'][2]} H2^2", flush=True)
            print(f"  c_3(V) = {s['c3_V_int']}", flush=True)
            print(f"  V Wilson rank partition: {s['wilson_V_rank']}", flush=True)
            print(f"  chi(V) per class: {s['chi_per_class']}", flush=True)
            print(f"  Anomaly delta H1={s['anomaly']['c2_TX_dot_H1'] - s['anomaly']['c2_V_dot_H1']}, "
                  f"H2={s['anomaly']['c2_TX_dot_H2'] - s['anomaly']['c2_V_dot_H2']}", flush=True)
            print(f"  Surj: {s['surj_reason']}", flush=True)

    return survivors, counters


if __name__ == "__main__":
    survivors, counters = main()
    print(flush=True)
    print("=" * 78, flush=True)
    print("CYCLE 5 VERDICT", flush=True)
    print("=" * 78, flush=True)
    if not survivors:
        print("REJECT: No rank-3 monad V = ker(B -> C) with rank(B) in {6,7,8}", flush=True)
        print("on TY/Z_3 in bidegree range [-2,2]^2 satisfies all cycle-5 constraints", flush=True)
        print("(c_1=0, |chi(V)_p|>=3 per Wilson class, c_3=+/-18, anomaly,", flush=True)
        print("polystability necessary, map existence, strict surjectivity).", flush=True)
    else:
        print(f"PARTIAL VALID: {len(survivors)} survivor(s) pass all gates.", flush=True)
        print("Each requires drilldown verification before proposing as V_min2.", flush=True)
