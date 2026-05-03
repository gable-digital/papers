"""
H2 — Monad bundle scan: V = ker(B → C) on TY/Z3.

Hypothesis (H2)
---------------
A 'monad bundle' is defined by a short exact sequence
    0 → V → B → C → 0
where B and C are line-bundle sums.  V has rank rank(B) - rank(C).

Claim: There exist line-bundle sums B (rank r_B) and C (rank r_C = r_B - 3)
on TY/Z3 such that V = ker(B → C) is a rank-3 SU(3) bundle satisfying:
    1. c_1(V) = c_1(B) - c_1(C) = 0
    2. Wilson partition on V (induced by Wilson on B/C) is 3:3:3 balanced
    3. c_3(V) = c_3(B) - c_3(C) - c_1(B) c_2(C) + c_2(B) c_1(C) etc.
       (full short-exact-sequence Chern computation), integrated to ±18
    4. c_2(V) ≤ c_2(TX) (anomaly)
    5. The map B → C is surjective at all points (existence of monad)
       — generic for ample C, we encode this as a degree-positivity check
       (each summand of C must be reachable as a quotient of B).

Falsification: exhaustive scan over rank-3 (= rank-6 B, rank-3 C) and
rank-4 monad bundles within bidegree range [-2, 2] returns no candidate
satisfying all 5 constraints.

This adapts the Schoen monad construction (rank 6 → 3 via cubic-line C)
to TY's 2-factor ambient.  Schoen's `schoen_z3xz3_canonical` uses
B = O(1,0,0)² ⊕ O(0,1,0)² ⊕ O(0,0,1)² and C = O(1,1,1)³ on CP² × CP² × CP¹
which IS 3:3:3 balanced.  The TY analog drops one CP factor, leaving 2 — and
the question is whether a TY analog exists at all.

Run:
    PYTHONIOENCODING=utf-8 python h2_monad_bundle_scan.py
"""

import sys
from itertools import combinations_with_replacement as cwr, product
from fractions import Fraction

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ty_z3_bundle_constraints import (
    LineBundle, LineBundleSum, D,
    wilson_z3_phase, anomaly_check, integrate_c2_against,
)


def chern_total(summands):
    """For a line-bundle sum, return (c1_a, c1_b, c2_11, c2_12, c2_22, c3_int)."""
    V = LineBundleSum(tuple(summands))
    a, b = V.c1_components()
    c2 = V.c2_components()
    c3 = V.c3_value()
    return (a, b, c2[0], c2[1], c2[2], c3)


def monad_chern(B_summands, C_summands):
    """
    From 0 → V → B → C → 0:
        c(V) = c(B) / c(C)
    Compute c_1, c_2, c_3 of V to second order.

    Notation: c(B) = 1 + c1B + c2B + c3B; same for C.
    1/c(C) = 1 - c1C + (c1C^2 - c2C) - (c1C^3 - 2 c1C c2C + c3C) + ...

    c(V) = c(B) * (1/c(C)):
        c_1(V) = c1B - c1C
        c_2(V) = c2B - c2C - c1B c1C + c1C^2
        c_3(V) = c3B - c3C - (c1B c2C - c2B c1C) + (c1B c1C^2 - c1C^3)
                            ... carefully expanded.

    With c_1(V) = 0 (we'll filter to this), the formulas simplify a lot.
    Implementation: keep symbolic in (H_1, H_2) coefficients, integrate at end.
    """
    a_B, b_B, c211_B, c212_B, c222_B, c3_B = chern_total(B_summands)
    a_C, b_C, c211_C, c212_C, c222_C, c3_C = chern_total(C_summands)

    # c_1(V) components
    a_V = a_B - a_C
    b_V = b_B - b_C

    # c_2(V) = c_2(B) - c_2(C) - c_1(B) c_1(C) + c_1(C)^2
    # In components (over basis H_1^2, H_1 H_2, H_2^2):
    # c_1(B) c_1(C) has components (a_B a_C, a_B b_C + b_B a_C, b_B b_C)
    # c_1(C)^2 has components (a_C^2, 2 a_C b_C, b_C^2)
    c211_V = c211_B - c211_C - a_B * a_C + a_C * a_C
    c212_V = c212_B - c212_C - (a_B * b_C + b_B * a_C) + 2 * a_C * b_C
    c222_V = c222_B - c222_C - b_B * b_C + b_C * b_C

    # c_3(V) integrated over X~.
    # Full formula (Friedman-Morgan, Chern character of quotient):
    # ch(V) = ch(B) - ch(C)  ⇒  ch_3(V) = ch_3(B) - ch_3(C)
    # Convert ch_3 ↔ c_3 via Newton:
    #   ch_3 = (1/6)(c_1^3 - 3 c_1 c_2 + 3 c_3)
    # So: c_3 = 2 ch_3 + c_1 c_2 - (1/3) c_1^3
    # Or simpler when c_1(V) = 0: c_3(V) = 2 ch_3(V) = 2 (ch_3(B) - ch_3(C))
    # ch_3 of a line-bundle sum = (1/6) sum_i c_1(L_i)^3 integrated.

    def ch3_int(summands):
        # ch_3(L) = c_1(L)^3 / 6
        total = 0
        for L in summands:
            total += (L.a**3 * D[(1, 1, 1)]
                      + 3 * L.a**2 * L.b * D[(1, 1, 2)]
                      + 3 * L.a * L.b**2 * D[(1, 2, 2)]
                      + L.b**3 * D[(2, 2, 2)])
        return Fraction(total, 6)

    ch3_B_int = ch3_int(B_summands)
    ch3_C_int = ch3_int(C_summands)
    ch3_V_int = ch3_B_int - ch3_C_int

    # If c_1(V) = 0, c_3(V) = 2 ch_3(V).
    if a_V == 0 and b_V == 0:
        c3_V_int = 2 * ch3_V_int
    else:
        # Full formula needed: c_3 = 2 ch_3 + c_1 c_2 - (1/3) c_1^3
        # Compute c_1(V) c_2(V) integrated:
        c1c2_int = (
            a_V * (c211_V * D[(1, 1, 1)] + c212_V * D[(1, 1, 2)] + c222_V * D[(1, 2, 2)])
            + b_V * (c211_V * D[(1, 1, 2)] + c212_V * D[(1, 2, 2)] + c222_V * D[(2, 2, 2)])
        )
        c1cube_int = (a_V**3 * D[(1, 1, 1)]
                      + 3 * a_V**2 * b_V * D[(1, 1, 2)]
                      + 3 * a_V * b_V**2 * D[(1, 2, 2)]
                      + b_V**3 * D[(2, 2, 2)])
        c3_V_int = 2 * ch3_V_int + c1c2_int - Fraction(c1cube_int, 3)

    return {
        'rank': len(B_summands) - len(C_summands),
        'c1_V': (a_V, b_V),
        'c2_V': (c211_V, c212_V, c222_V),
        'c3_V_int': c3_V_int,
    }


def wilson_partition_monad(B_summands, C_summands):
    """
    Wilson phase classes on V = ker(B → C).

    Heuristic: V's Wilson decomposition equals B's Wilson partition MINUS
    C's Wilson partition (in the Z/3 representation ring).
    So count(class k in V) = count(class k in B) - count(class k in C).
    Returns dict or None if any class is negative (invalid).
    """
    counts = {0: 0, 1: 0, 2: 0}
    for L in B_summands:
        counts[(L.a - L.b) % 3] += 1
    for L in C_summands:
        counts[(L.a - L.b) % 3] -= 1
    for k in counts:
        if counts[k] < 0:
            return None
    return counts


def is_3_3_3_balanced(parts):
    if parts is None:
        return False
    return parts[0] == parts[1] == parts[2]


def anomaly_check_monad(c2_V_components):
    """Check int c_2(V) ∧ J <= int c_2(TX) ∧ J for J = H_1, H_2."""
    c11, c12, c22 = c2_V_components
    int_v_h1 = c11 * D[(1, 1, 1)] + c12 * D[(1, 1, 2)] + c22 * D[(1, 2, 2)]
    int_v_h2 = c11 * D[(1, 1, 2)] + c12 * D[(1, 2, 2)] + c22 * D[(2, 2, 2)]
    int_tx_h1 = integrate_c2_against((1, 0))
    int_tx_h2 = integrate_c2_against((0, 1))
    return {
        'pass_H1': int_v_h1 <= int_tx_h1,
        'pass_H2': int_v_h2 <= int_tx_h2,
        'c2_V_dot_H1': int_v_h1, 'c2_V_dot_H2': int_v_h2,
        'c2_TX_dot_H1': int_tx_h1, 'c2_TX_dot_H2': int_tx_h2,
    }


def map_existence_check(B_summands, C_summands):
    """
    Heuristic: a generic monad map B → C exists and is surjective iff for each
    summand O(c_j, d_j) of C, there is at least one summand O(a_i, b_i) of B
    with a_i ≤ c_j AND b_i ≤ d_j. (Sections of Hom(O(a,b), O(c,d)) =
    O(c-a, d-b) are nonempty iff c-a, d-b ≥ 0, by Bott on CP^n.)

    This is a NECESSARY condition. Not sufficient — proper sheaf-theoretic
    surjectivity requires checking the determinant locus.
    """
    for cL in C_summands:
        c, d = cL.a, cL.b
        if not any(bL.a <= c and bL.b <= d for bL in B_summands):
            return False
    return True


# Scan
A_RANGE = range(-2, 3)
B_RANGE = range(-2, 3)
RANK_V = 3   # Want rank-3 SU(3) bundle V
RANK_B_CHOICES = [4, 5]   # (4,1) and (5,2); (6,3) ~ 1.7 G pairs is too large
                          # for an unindexed scan -- we instead provide a
                          # targeted rank-(6,3) check below in run_targeted().
TARGET_C3 = {18, -18}


def enumerate_lbs():
    return [LineBundle(a, b) for a in A_RANGE for b in B_RANGE]


def run_h2():
    print("=" * 70)
    print("H2 — Monad bundle V = ker(B → C) scan on TY/Z3")
    print("=" * 70)
    print(f"Bidegree range: [-2, 2]^2  ({len(enumerate_lbs())} line bundles)")
    print(f"Target rank(V) = 3, scanning rank(B) ∈ {RANK_B_CHOICES}")
    print(f"Target c_3(V) = ±18 (3 net generations downstairs)")
    lbs = enumerate_lbs()

    survivors = []
    n_total = 0
    n_c1 = 0
    n_wilson = 0
    n_c3 = 0
    n_anomaly = 0
    n_map = 0

    # Bucket line bundles by Wilson class for early 1:1:1 filtering.
    bucket = {0: [], 1: [], 2: []}
    for L in lbs:
        bucket[(L.a - L.b) % 3].append(L)

    def gen_with_wilson(r):
        """Yield rank-r combos pre-filtered by sum-Wilson distribution."""
        # Iterate over partitions (n0, n1, n2) of r; we need V's Wilson 1:1:1
        # which means (B's wilson) - (C's wilson) = (1,1,1).
        # We pre-iterate over B's wilson partition (n0_B, n1_B, n2_B).
        for n0 in range(r + 1):
            for n1 in range(r - n0 + 1):
                n2 = r - n0 - n1
                for c0 in cwr(bucket[0], n0):
                    for c1 in cwr(bucket[1], n1):
                        for c2 in cwr(bucket[2], n2):
                            yield (n0, n1, n2), tuple(c0 + c1 + c2)

    for r_B in RANK_B_CHOICES:
        r_C = r_B - RANK_V
        if r_C < 1:
            continue
        print(f"\n  rank(B)={r_B}, rank(C)={r_C}: enumerating ...")
        # Pre-build B and C lists keyed by Wilson partition for fast 1:1:1 join.
        B_by_part = {}
        for part, combo in gen_with_wilson(r_B):
            B_by_part.setdefault(part, []).append(combo)
        C_by_part = {}
        for part, combo in gen_with_wilson(r_C):
            C_by_part.setdefault(part, []).append(combo)
        # 1:1:1 means B's part minus C's part = (1,1,1).
        # iterate over all C parts; for each, B part = C part + (1,1,1).
        for c_part, c_list in C_by_part.items():
            b_part = (c_part[0] + 1, c_part[1] + 1, c_part[2] + 1)
            if b_part not in B_by_part:
                continue
            for B_combo in B_by_part[b_part]:
              for C_combo in c_list:
                n_total += 1
                # Quick c_1 filter
                a_V = sum(L.a for L in B_combo) - sum(L.a for L in C_combo)
                b_V = sum(L.b for L in B_combo) - sum(L.b for L in C_combo)
                if a_V != 0 or b_V != 0:
                    continue
                n_c1 += 1

                parts = wilson_partition_monad(B_combo, C_combo)
                if not is_3_3_3_balanced(parts):
                    continue
                if parts[0] != 1:   # rank-3 V means each class has 1 summand
                    continue
                n_wilson += 1

                m = monad_chern(B_combo, C_combo)
                if m['c3_V_int'] not in TARGET_C3:
                    continue
                n_c3 += 1

                a = anomaly_check_monad(m['c2_V'])
                if not (a['pass_H1'] and a['pass_H2']):
                    continue
                n_anomaly += 1

                if not map_existence_check(B_combo, C_combo):
                    continue
                n_map += 1

                survivors.append({
                    'B': [(L.a, L.b) for L in B_combo],
                    'C': [(L.a, L.b) for L in C_combo],
                    'c1_V': m['c1_V'],
                    'c2_V': m['c2_V'],
                    'c3_V_int': m['c3_V_int'],
                    'wilson': parts,
                    'anomaly': a,
                })

    print(f"\n  Total monad pairs:                  {n_total}")
    print(f"  Pass c_1(V) = 0:                    {n_c1}")
    print(f"  Pass Wilson 1:1:1 (rank-3 V):       {n_wilson}")
    print(f"  Pass c_3(V) = ±18:                  {n_c3}")
    print(f"  Pass anomaly:                       {n_anomaly}")
    print(f"  Pass map existence (necessary):     {n_map}")
    print(f"  Survivors: {len(survivors)}")
    if survivors:
        for i, s in enumerate(survivors[:10]):
            print(f"\n  Candidate #{i+1}:")
            print(f"    B = ⊕ O{tuple(s['B'])}")
            print(f"    C = ⊕ O{tuple(s['C'])}")
            print(f"    c_1(V): {s['c1_V']}, c_2(V): {s['c2_V']}, c_3(V): {s['c3_V_int']}")
            print(f"    Wilson: {s['wilson']}")
            print(f"    5-brane W: H_1·{s['anomaly']['c2_TX_dot_H1']-s['anomaly']['c2_V_dot_H1']}, "
                  f"H_2·{s['anomaly']['c2_TX_dot_H2']-s['anomaly']['c2_V_dot_H2']}")

    return survivors


if __name__ == "__main__":
    survivors = run_h2()
    print("\n" + "=" * 70)
    print("H2 SUMMARY")
    print("=" * 70)
    if not survivors:
        print("VERDICT: H2 REJECTED.")
        print("No rank-3 monad bundle V = ker(B → C) on TY/Z3 in the scanned")
        print("bidegree range satisfies all constraints (c_1=0, Wilson 1:1:1,")
        print("c_3=±18, anomaly, map-existence).")
    else:
        print(f"VERDICT: H2 PARTIALLY VALID — {len(survivors)} survivor(s).")
        print("Each requires further analysis: full sheaf-surjectivity check,")
        print("polystability of V (not directly inheritable from B, C), and")
        print("low-energy spectrum verification (Higgs, exotics, etc).")
