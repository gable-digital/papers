"""
Drill down on a single H2 monad candidate to verify it carefully.

We pick the cleanest candidate from the polystability filter and verify:
- Chern-class arithmetic by hand
- Map surjectivity heuristic
- Stability behavior under different Kähler classes
- Index-theorem generation count

This is the "do the math by hand" step before promoting to a publication
candidate.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ty_z3_bundle_constraints import (
    LineBundle, LineBundleSum, D, integrate_c2_against,
    wilson_z3_phase,
)
from h2_monad_bundle_scan import monad_chern, anomaly_check_monad


def slope_at(L: LineBundle, t1, t2):
    """mu(L) = int c_1(L) ∧ J^2 with J = t1 H_1 + t2 H_2.
    On TY: D_111=D_222=0, D_112=D_122=9.
    int H_1 ∧ J^2 = 0 + 2 t1 t2 * 9 + t2^2 * 9 = 9 t2 (2 t1 + t2)
    int H_2 ∧ J^2 = 9 t1^2 + 18 t1 t2 + 0 = 9 t1 (t1 + 2 t2)
    """
    Q1 = 9 * t2 * (2 * t1 + t2)
    Q2 = 9 * t1 * (t1 + 2 * t2)
    return L.a * Q1 + L.b * Q2


def drilldown(B_specs, C_specs, label="Candidate"):
    print("=" * 70)
    print(f"H2 drilldown: {label}")
    print("=" * 70)
    B = [LineBundle(*ab) for ab in B_specs]
    C = [LineBundle(*ab) for ab in C_specs]
    print(f"B = {B_specs}  (rank {len(B)})")
    print(f"C = {C_specs}  (rank {len(C)})")
    print(f"V = ker(B -> C)  rank {len(B) - len(C)}")

    m = monad_chern(B, C)
    print(f"\nChern data of V:")
    print(f"  c_1(V) = {m['c1_V']}")
    print(f"  c_2(V) = {m['c2_V'][0]} H_1^2 + {m['c2_V'][1]} H_1H_2 + {m['c2_V'][2]} H_2^2")
    print(f"  c_3(V) integrated = {m['c3_V_int']}")
    print(f"  net generations downstairs = c_3(V)/(2*3) = {m['c3_V_int']}/6 = "
          f"{m['c3_V_int']/6}")

    a = anomaly_check_monad(m['c2_V'])
    print(f"\nAnomaly: int c_2(V) ∧ H_1 = {a['c2_V_dot_H1']} (≤ {a['c2_TX_dot_H1']}?)")
    print(f"         int c_2(V) ∧ H_2 = {a['c2_V_dot_H2']} (≤ {a['c2_TX_dot_H2']}?)")
    print(f"  Pass H_1: {a['pass_H1']}, Pass H_2: {a['pass_H2']}")

    print(f"\nWilson phase classes (Z/3) of B-summands:")
    for L in B:
        print(f"  O({L.a}, {L.b}) -> phase {wilson_z3_phase(L)}, "
              f"slope at (1,1) = {slope_at(L, 1, 1)}")
    print(f"\nWilson phase classes of C-summands:")
    for L in C:
        print(f"  O({L.a}, {L.b}) -> phase {wilson_z3_phase(L)}, "
              f"slope at (1,1) = {slope_at(L, 1, 1)}")

    print(f"\nSlope analysis at sample Kähler classes:")
    print(f"  J = (1,1):   sum mu(B_i) = {sum(slope_at(L, 1, 1) for L in B)}")
    print(f"               sum mu(C_j) = {sum(slope_at(L, 1, 1) for L in C)}")
    print(f"  J = (1,2):   sum mu(B_i) = {sum(slope_at(L, 1, 2) for L in B)}")
    print(f"  J = (2,1):   sum mu(B_i) = {sum(slope_at(L, 2, 1) for L in B)}")

    # Map surjectivity: list O(c_j - a_i, d_j - b_i) for each (i,j).
    # The map B → C in component (i,j) is a section of O(c_j-a_i, d_j-b_i).
    # Sections are nonzero iff both indices are >= 0.
    # Number of sections h^0 = (c-a+3 choose 3) * (d-b+3 choose 3) on CP^3xCP^3
    # (well, restricted to TY -- a parent count is an upper bound).
    print(f"\nMap-component degree matrix (c_j - a_i, d_j - b_i):")
    for j, cL in enumerate(C):
        for i, bL in enumerate(B):
            da = cL.a - bL.a
            db = cL.b - bL.b
            sect = "OK" if (da >= 0 and db >= 0) else "0-sect"
            print(f"  Hom(B_{i+1}={bL.a},{bL.b} → C_{j+1}={cL.a},{cL.b}) = "
                  f"O({da}, {db})  [{sect}]")


# A cleanest-looking candidate from the filter pass:
# B = [(-2,1), (-1,1), (0,1), (1,-1)],  C = [(-2,2)]
# Wilson partitions: B has phases [(-2-1)%3=0, (-1-1)%3=2, (0-1)%3=2, (1-(-1))%3=2]
# Wait that's 1:0:3 partition, not 1:1:1. Let me recompute
# (-2-1)%3 = -3%3 = 0
# (-1-1)%3 = -2%3 = 1
# (0-1)%3 = -1%3 = 2
# (1-(-1))%3 = 2%3 = 2
# B: classes [0,1,2,2] -> {0:1, 1:1, 2:2}
# C: phase (-2-2)%3 = -4%3 = 2 -> {2:1}
# V = B - C = {0:1, 1:1, 2:1}  ✓ 1:1:1

if __name__ == "__main__":
    # Best candidate: smallest |c_2| values, balanced bidegrees.
    drilldown([(-2, 1), (-1, 1), (0, 1), (1, -1)],
              [(-2, 2)],
              label="V_1: rank-(4,1) monad with small c_2")

    print("\n\n")

    # Another aesthetically pleasing one:
    drilldown([(-1, 1), (-1, 1), (1, -1), (1, -1)],
              [(0, 0)],
              label="V_2: symmetric monad attempt")
