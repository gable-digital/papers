"""
Cycle 5 drilldown: verify the leading survivor V_min2 candidate.

Leading survivor (cycle 5 scan, rank-(6,3)):
  B = O(-2,1) + O(-1,-2) + O(-2,-1)^2 + O(0,1) + O(1,-1)
  C = O(-2,-1)^3
  c_3(V) = +18, chi per Wilson class = (9, -27, 27)

We verify:
  1. Chern numbers via independent intersection-theory recomputation
  2. Polystability across at least 5 random integer Kähler classes
  3. Per-Wilson-class harmonic mode count (chi splits)
"""

import sys
import random
from itertools import product

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ty_z3_bundle_constraints import (
    LineBundle, LineBundleSum, D, integrate_c2_against,
    wilson_z3_phase, _ch3_int_of_lb,
)
from h2_monad_bundle_scan import monad_chern, anomaly_check_monad
from h2_basis_rank_corrected_scan import (
    polystability_necessary, strict_surjectivity, line_strictly_positive,
)


def slope_at(L, t1, t2):
    """mu(L) = int c_1(L) * J^2 with J = t1 H1 + t2 H2."""
    return (L.a * (t1*t1*D[(1,1,1)] + 2*t1*t2*D[(1,1,2)] + t2*t2*D[(1,2,2)])
            + L.b * (t1*t1*D[(1,1,2)] + 2*t1*t2*D[(1,2,2)] + t2*t2*D[(2,2,2)]))


def independent_chern_check(B_specs, C_specs):
    """Recompute c_1, c_2, c_3 via direct splitting principle on V's
    Chern class c(V) = c(B) / c(C), to all orders."""
    B = [LineBundle(*x) for x in B_specs]
    C = [LineBundle(*x) for x in C_specs]
    # c_1
    c1B_a = sum(L.a for L in B)
    c1B_b = sum(L.b for L in B)
    c1C_a = sum(L.a for L in C)
    c1C_b = sum(L.b for L in C)
    c1V = (c1B_a - c1C_a, c1B_b - c1C_b)

    # c_2 = c_2(B) - c_2(C) - c_1(B)c_1(C) + c_1(C)^2
    # c_2(SUM) = sum_{i<j} c_1(L_i) c_1(L_j)  (rank>=2, else 0)
    def c2_sum(LBs):
        c11 = c12 = c22 = 0
        n = len(LBs)
        for i in range(n):
            for j in range(i+1, n):
                ai, bi = LBs[i].a, LBs[i].b
                aj, bj = LBs[j].a, LBs[j].b
                c11 += ai*aj
                c22 += bi*bj
                c12 += ai*bj + aj*bi
        return (c11, c12, c22)
    c2B = c2_sum(B)
    c2C = c2_sum(C)
    # c_1(B) c_1(C)
    c1B_c1C_11 = c1B_a * c1C_a
    c1B_c1C_12 = c1B_a * c1C_b + c1B_b * c1C_a
    c1B_c1C_22 = c1B_b * c1C_b
    # c_1(C)^2
    c1C_sq_11 = c1C_a * c1C_a
    c1C_sq_12 = 2 * c1C_a * c1C_b
    c1C_sq_22 = c1C_b * c1C_b
    c2V = (
        c2B[0] - c2C[0] - c1B_c1C_11 + c1C_sq_11,
        c2B[1] - c2C[1] - c1B_c1C_12 + c1C_sq_12,
        c2B[2] - c2C[2] - c1B_c1C_22 + c1C_sq_22,
    )

    # c_3 via ch_3(V) = ch_3(B) - ch_3(C) (when c_1(V)=0, c_3 = 2 ch_3)
    ch3B = sum(_ch3_int_of_lb(L) for L in B)
    ch3C = sum(_ch3_int_of_lb(L) for L in C)
    ch3V = ch3B - ch3C
    if c1V == (0, 0):
        c3V = 2 * ch3V
    else:
        c3V = None  # not handled here

    return {'c1_V': c1V, 'c2_V': c2V, 'c3_V_int': c3V,
            'ch3_B': ch3B, 'ch3_C': ch3C, 'ch3_V': ch3V}


def polystability_random_kahler(B_specs, C_specs, n_samples=20, seed=42):
    """Test polystability necessary condition (no destabilizing line sub)
    across n_samples random integer Kähler classes (t1, t2) >= 1.
    Necessary condition (since c_1=0, mu(V)=0 everywhere):
       no B-summand has slope > 0 at any (t1, t2) > 0.
    For TY where D_111=D_222=0, D_112=D_122=9:
       mu(L) = 9 [a t2 (2t1+t2) + b t1 (t1+2t2)]
    Sign analysis:
      - If a > 0 AND b > 0: mu always > 0 (destabilizes)
      - If a < 0 AND b < 0: mu always < 0
      - If a >= 0, b <= 0 or vice versa: mu sign depends on (t1,t2)
        Find roots: a t2 (2t1+t2) + b t1 (t1+2t2) = 0.
    """
    B = [LineBundle(*x) for x in B_specs]
    rng = random.Random(seed)

    results = []
    for _ in range(n_samples):
        t1 = rng.randint(1, 50)
        t2 = rng.randint(1, 50)
        slopes = {(L.a, L.b): slope_at(L, t1, t2) for L in B}
        max_pos = max(slopes.values())
        results.append({
            't1': t1, 't2': t2,
            'max_slope': max_pos,
            'destabilized': max_pos > 0,
            'slopes': slopes,
        })

    n_destab = sum(1 for r in results if r['destabilized'])
    return {
        'n_samples': n_samples,
        'n_destabilized': n_destab,
        'samples': results,
        'all_pass': n_destab == 0,
    }


def per_class_modes(B_specs, C_specs):
    """For each Wilson class p, list:
       - B-summands of class p with their ch_3 contribution
       - C-summands of class p with their ch_3 contribution
       - chi(V)_p = sum_B - sum_C
    """
    B = [LineBundle(*x) for x in B_specs]
    C = [LineBundle(*x) for x in C_specs]
    out = {}
    for p in (0, 1, 2):
        B_p = [(L.a, L.b, float(_ch3_int_of_lb(L))) for L in B
               if (L.a - L.b) % 3 == p]
        C_p = [(L.a, L.b, float(_ch3_int_of_lb(L))) for L in C
               if (L.a - L.b) % 3 == p]
        chi = sum(_ch3_int_of_lb(L) for L in B if (L.a-L.b)%3 == p) \
              - sum(_ch3_int_of_lb(L) for L in C if (L.a-L.b)%3 == p)
        out[p] = {'B_summands': B_p, 'C_summands': C_p, 'chi_V_p': float(chi)}
    return out


def drilldown_cycle5(B_specs, C_specs, label):
    print("=" * 78)
    print(f"Cycle 5 drilldown: {label}")
    print("=" * 78)
    print(f"  B = {B_specs}  (rank {len(B_specs)})")
    print(f"  C = {C_specs}  (rank {len(C_specs)})")
    print(f"  V = ker(B -> C), rank = {len(B_specs) - len(C_specs)}")
    print()

    # 1. Independent Chern check
    print("1. Independent Chern recomputation")
    print("-" * 78)
    chern_a = independent_chern_check(B_specs, C_specs)
    chern_b = monad_chern([LineBundle(*x) for x in B_specs],
                          [LineBundle(*x) for x in C_specs])
    print(f"  c_1(V): direct = {chern_a['c1_V']}, monad_chern = {chern_b['c1_V']}")
    print(f"  c_2(V): direct = {chern_a['c2_V']}, monad_chern = {chern_b['c2_V']}")
    print(f"  c_3(V) integrated: direct = {chern_a['c3_V_int']}, "
          f"monad_chern = {chern_b['c3_V_int']}")
    consistent = (chern_a['c1_V'] == chern_b['c1_V']
                  and chern_a['c2_V'] == chern_b['c2_V']
                  and chern_a['c3_V_int'] == chern_b['c3_V_int'])
    print(f"  Consistent: {consistent}")
    print()

    # 2. Anomaly
    print("2. Anomaly check c_2(V) <= c_2(TX)")
    print("-" * 78)
    a = anomaly_check_monad(chern_b['c2_V'])
    print(f"  int c_2(V) * H1 = {a['c2_V_dot_H1']}, c_2(TX) * H1 = {a['c2_TX_dot_H1']}")
    print(f"  int c_2(V) * H2 = {a['c2_V_dot_H2']}, c_2(TX) * H2 = {a['c2_TX_dot_H2']}")
    print(f"  Pass H1: {a['pass_H1']}, Pass H2: {a['pass_H2']}")
    print(f"  Effective 5-brane W charges: H1*{a['c2_TX_dot_H1']-a['c2_V_dot_H1']}, "
          f"H2*{a['c2_TX_dot_H2']-a['c2_V_dot_H2']}")
    print()

    # 3. Polystability across random Kähler classes
    print("3. Polystability (line-destabilizer test, 20 random Kähler classes)")
    print("-" * 78)
    poly = polystability_random_kahler(B_specs, C_specs, n_samples=20, seed=42)
    print(f"  Samples checked: {poly['n_samples']}")
    print(f"  Destabilized: {poly['n_destabilized']}")
    print(f"  All pass: {poly['all_pass']}")
    if not poly['all_pass']:
        # Show first destabilizing sample
        for r in poly['samples']:
            if r['destabilized']:
                print(f"  First destabilizing: t1={r['t1']}, t2={r['t2']}, "
                      f"max slope = {r['max_slope']}")
                # Identify destabilizer line bundle
                for ab, s in r['slopes'].items():
                    if s == r['max_slope']:
                        print(f"    Destabilizer: O{ab} with slope {s}")
                        break
                break
    # Also check: all B-summands not strictly positive (necessary, faster)
    B = [LineBundle(*x) for x in B_specs]
    bad = [L for L in B if line_strictly_positive(L)]
    print(f"  Strictly-positive B-summands (a>0, b>0): {[(L.a,L.b) for L in bad]}")
    print()

    # 4. Per-class modes
    print("4. Per-Wilson-class mode count (chi(V)_p)")
    print("-" * 78)
    pcm = per_class_modes(B_specs, C_specs)
    for p in (0, 1, 2):
        print(f"  Class {p}:")
        print(f"    B-summands ({len(pcm[p]['B_summands'])}): "
              f"{pcm[p]['B_summands']}")
        print(f"    C-summands ({len(pcm[p]['C_summands'])}): "
              f"{pcm[p]['C_summands']}")
        print(f"    chi(V)_p = {pcm[p]['chi_V_p']}")
    chi_total = sum(pcm[p]['chi_V_p'] for p in (0, 1, 2))
    print(f"  Sum: {chi_total} (should match c_3/2 = {chern_b['c3_V_int']}/2 = "
          f"{float(chern_b['c3_V_int'])/2})")
    print()

    return {
        'consistent_chern': consistent,
        'anomaly_pass': a['pass_H1'] and a['pass_H2'],
        'poly_pass': poly['all_pass'],
        'per_class': pcm,
    }


if __name__ == "__main__":
    # Cycle 5 v3 survivors (closed-negative-cone B, rank-(7,4))
    candidates = [
        ([(0,0),(0,0),(-1,-2),(-2,-1),(-2,-1),(-1,0),(-1,0)],
         [(-1,-1),(-1,-2),(-1,-2),(-1,-2)],
         "V_min2 candidate (v3 survivor #1): rank-(7,4), c_3=+18"),
        ([(-2,-2),(-2,-2),(0,-1),(0,-1),(-1,0),(-1,0),(-1,0)],
         None,  # need to look up C
         "v3 survivor #2 — C lookup needed"),
        ([(0,0),(0,0),(-1,-2),(-1,-2),(0,-1),(0,-1),(-2,-1)],
         None,
         "v3 survivor #4 — c_3=+18, simpler structure"),
    ]
    candidates = [
        ([(0,0),(0,0),(-1,-2),(-2,-1),(-2,-1),(-1,0),(-1,0)],
         [(-1,-1),(-2,-1),(-2,-1),(-2,-1)],
         "V_min2 leading candidate (v3 survivor #1): rank-(7,4), c_3=+18, Wilson V=(1,1,1)"),
        ([(0,0),(0,0),(-1,-2),(-1,-2),(0,-1),(0,-1),(-2,-1)],
         [(-1,-1),(-1,-2),(-1,-2),(-1,-2)],
         "V_min2 alt candidate (v3 survivor #4): rank-(7,4), c_3=+18, Wilson V=(1,1,1)"),
    ]
    for B_specs, C_specs, label in candidates:
        drilldown_cycle5(B_specs, C_specs, label)
        print("\n\n")
