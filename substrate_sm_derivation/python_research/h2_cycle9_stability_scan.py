"""
Cycle 9 — Re-scan cycle-5 v3 monad survivors with the H^0(V) = 0
Mumford-Takemoto stability filter (computed via BBW + monad LES).

Hypothesis (cycle 9, verbatim from task brief)
----------------------------------------------
Adding the H^0(V) = 0 stability constraint (concretely:
Σ h^0(B_α) ≤ Σ h^0(C_β), with strict inequality only if the LES
connecting map H^0(B) → H^0(C) is not full rank) eliminates ALL 10
v3 survivors from cycle 5.

Falsification: at least one survivor passes the new constraint AND has
Σ h^1(B_α) > 0 (so the H^1 branch could potentially populate Yukawa
modes).

Validation: the H^0(V) computation matches V_min2's known h^0(V) = 2.
(Validated separately in h_zero_validate.py — PASS.)

This file replays the cycle-5 v3 scan but inserts the new stability
filter immediately after polystability (so we still see how many
post-poly survivors there are AND how many pass the new filter).
"""

import sys
import time
from itertools import combinations_with_replacement as cwr
from fractions import Fraction
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

from ty_z3_bundle_constraints import (
    LineBundle, LineBundleSum, D,
    wilson_z3_phase, integrate_c2_against,
    is_3_generation_basis_compatible, wilson_partition_modes_per_class,
    _ch3_int_of_lb,
    h_zero_of_V, h0_of_line_bundle_sum_TY, h1_of_line_bundle_sum_TY,
    h_star_X_line_TY,
)
from h2_monad_bundle_scan import (
    monad_chern, wilson_partition_monad, anomaly_check_monad,
    map_existence_check,
)
from h2_basis_rank_corrected_scan import (
    enumerate_lbs_safe_for_B, enumerate_lbs_full,
    strict_surjectivity, line_potentially_positive,
    A_RANGE, B_RANGE, RANK_V, TARGET_C3, WILSON_CLASS_TARGET,
)
# Match cycle-5 v3 scope exactly: (6,3) and (7,4) only.
RANK_B_C_PAIRS = [(6, 3), (7, 4)]


def main():
    print("=" * 78, flush=True)
    print("Cycle 9 — H^0(V) stability filter on cycle-5 v3 monad survivors",
          flush=True)
    print("=" * 78, flush=True)

    lbs_B = enumerate_lbs_safe_for_B()
    lbs_C = enumerate_lbs_full()
    print(f"B alphabet: {len(lbs_B)} (closed neg cone)", flush=True)
    print(f"C alphabet: {len(lbs_C)}", flush=True)
    print(f"Rank pairs: {RANK_B_C_PAIRS}", flush=True)
    print(f"Filter chain (in order):", flush=True)
    print(f"  Wilson-class chi gate: |chi(V)_p| >= {WILSON_CLASS_TARGET}", flush=True)
    print(f"  c_1(V) = 0", flush=True)
    print(f"  c_3(V) = +/-18", flush=True)
    print(f"  Anomaly c_2(V) <= c_2(TX)", flush=True)
    print(f"  Polystability (B all <= 0,0)  [enforced by alphabet]", flush=True)
    print(f"  Map existence", flush=True)
    print(f"  Strict surjectivity (>=2 feeders per C)", flush=True)
    print(f"  ** NEW (cycle 9): H^0(V) = 0 stability **", flush=True)
    print(flush=True)

    bucket_B = {0: [], 1: [], 2: []}
    for L in lbs_B:
        bucket_B[(L.a - L.b) % 3].append(L)
    bucket_C = {0: [], 1: [], 2: []}
    for L in lbs_C:
        bucket_C[(L.a - L.b) % 3].append(L)

    survivors_pre_h0 = []
    survivors_final = []
    counters = defaultdict(int)
    h1_branch_candidates = []

    t0 = time.time()

    for r_B, r_C in RANK_B_C_PAIRS:
        print(f"\n--- Scanning (rank B, rank C) = ({r_B}, {r_C}) ---", flush=True)
        t_pair = time.time()

        B_sub = {p: defaultdict(list) for p in (0, 1, 2)}
        C_sub = {p: defaultdict(list) for p in (0, 1, 2)}

        for p in (0, 1, 2):
            for n in range(0, r_B + 1):
                for combo in cwr(bucket_B[p], n):
                    chi_p = sum((_ch3_int_of_lb(L) for L in combo), Fraction(0))
                    a_sum = sum(L.a for L in combo)
                    b_sum = sum(L.b for L in combo)
                    B_sub[p][n].append((combo, chi_p, (a_sum, b_sum)))
            for n in range(0, r_C + 1):
                for combo in cwr(bucket_C[p], n):
                    chi_p = sum((_ch3_int_of_lb(L) for L in combo), Fraction(0))
                    a_sum = sum(L.a for L in combo)
                    b_sum = sum(L.b for L in combo)
                    C_sub[p][n].append((combo, chi_p, (a_sum, b_sum)))

        for n0_B in range(r_B + 1):
            for n1_B in range(r_B - n0_B + 1):
                n2_B = r_B - n0_B - n1_B
                for n0_C in range(min(n0_B, r_C) + 1):
                    for n1_C in range(min(n1_B, r_C - n0_C) + 1):
                        n2_C = r_C - n0_C - n1_C
                        if n2_C < 0 or n2_C > n2_B:
                            continue
                        v0 = n0_B - n0_C
                        v1 = n1_B - n1_C
                        v2 = n2_B - n2_C
                        if v0 + v1 + v2 != RANK_V:
                            continue

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

                                                a_V = a0_V + a1_V + (c1_2_B[0] - c1_2_C[0])
                                                b_V = b0_V + b1_V + (c1_2_B[1] - c1_2_C[1])
                                                if a_V != 0 or b_V != 0:
                                                    continue
                                                counters['pass_c1'] += 1

                                                c3_V = 2 * (chi0_V + chi1_V + chi2_V)
                                                if c3_V not in TARGET_C3:
                                                    continue
                                                counters['pass_c3'] += 1

                                                B_combo = B0 + B1 + B2
                                                C_combo = C0 + C1 + C2

                                                m = monad_chern(B_combo, C_combo)
                                                a = anomaly_check_monad(m['c2_V'])
                                                if not (a['pass_H1'] and a['pass_H2']):
                                                    continue
                                                counters['pass_anomaly'] += 1
                                                counters['pass_polystab'] += 1

                                                if not map_existence_check(B_combo, C_combo):
                                                    continue
                                                counters['pass_map_exist'] += 1

                                                ok_surj, surj_reason = strict_surjectivity(
                                                    B_combo, C_combo
                                                )
                                                if not ok_surj:
                                                    continue
                                                counters['pass_strict_surj'] += 1

                                                # Pre-H0 survivor (cycle 5 v3 reproduction)
                                                pre_record = {
                                                    'r_B': r_B, 'r_C': r_C,
                                                    'B': [(L.a, L.b) for L in B_combo],
                                                    'C': [(L.a, L.b) for L in C_combo],
                                                    'c3_V': int(c3_V),
                                                }
                                                survivors_pre_h0.append(pre_record)

                                                # === NEW cycle-9 H^0(V) filter ===
                                                lower, upper, stable, info = h_zero_of_V(
                                                    B_combo, C_combo
                                                )
                                                if not stable:
                                                    counters['fail_h0_V'] += 1
                                                    continue
                                                counters['pass_h0_V'] += 1

                                                # Stable survivor — record + h^1 branch info
                                                h1_B = h1_of_line_bundle_sum_TY(B_combo)
                                                h1_C = h1_of_line_bundle_sum_TY(C_combo)
                                                rec = {
                                                    'r_B': r_B, 'r_C': r_C,
                                                    'B': [(L.a, L.b) for L in B_combo],
                                                    'C': [(L.a, L.b) for L in C_combo],
                                                    'c3_V': int(c3_V),
                                                    'h0_B': info['h0_B'],
                                                    'h0_C': info['h0_C'],
                                                    'h1_B': h1_B,
                                                    'h1_C': h1_C,
                                                    'chi_per_class': (
                                                        float(chi0_V),
                                                        float(chi1_V),
                                                        float(chi2_V),
                                                    ),
                                                }
                                                survivors_final.append(rec)
                                                if h1_B > 0:
                                                    h1_branch_candidates.append(rec)
                                                if len(survivors_final) <= 10:
                                                    print(
                                                        f"    [stable survivor #{len(survivors_final)}] "
                                                        f"B={rec['B']} C={rec['C']} "
                                                        f"c_3={rec['c3_V']} h0(B)={info['h0_B']} "
                                                        f"h0(C)={info['h0_C']} h1(B)={h1_B}",
                                                        flush=True,
                                                    )

        dt = time.time() - t_pair
        print(f"  pair done in {dt:.1f}s, "
              f"pre-H0 survivors: {len(survivors_pre_h0)}, "
              f"final survivors: {len(survivors_final)}", flush=True)

    dt_total = time.time() - t0
    print(flush=True)
    print("=" * 78, flush=True)
    print("Cycle 9 scan summary", flush=True)
    print("=" * 78, flush=True)
    print(f"  Total scan time:                          {dt_total:.1f}s", flush=True)
    print(f"  Joins post per-class chi gate:            {counters['post_class_chi']:,}", flush=True)
    print(f"  Pass c_1(V) = 0:                          {counters['pass_c1']:,}", flush=True)
    print(f"  Pass c_3(V) = +/-18:                      {counters['pass_c3']:,}", flush=True)
    print(f"  Pass anomaly:                             {counters['pass_anomaly']:,}", flush=True)
    print(f"  Pass polystability:                       {counters['pass_polystab']:,}", flush=True)
    print(f"  Pass map existence:                       {counters['pass_map_exist']:,}", flush=True)
    print(f"  Pass strict surjectivity:                 {counters['pass_strict_surj']:,}", flush=True)
    print(f"  --- New cycle-9 stability filter ---", flush=True)
    print(f"  Pre-H0 v3 survivors (cycle-5 reproduction): {len(survivors_pre_h0)}", flush=True)
    print(f"  Fail H^0(V) = 0 stability:                {counters['fail_h0_V']:,}", flush=True)
    print(f"  Pass H^0(V) = 0 stability:                {counters['pass_h0_V']:,}", flush=True)
    print(f"  FINAL CYCLE-9 SURVIVORS:                  {len(survivors_final)}", flush=True)
    print(f"  --- of which: h^1(B) > 0 (H^1-branch eligible) ---", flush=True)
    print(f"  H^1(B) > 0 candidates:                    {len(h1_branch_candidates)}", flush=True)

    if survivors_final:
        print(flush=True)
        print("=" * 78, flush=True)
        print(f"All cycle-9 survivors ({len(survivors_final)}):", flush=True)
        print("=" * 78, flush=True)
        for i, s in enumerate(survivors_final):
            print(f"\n#{i+1}: rank-({s['r_B']},{s['r_C']})", flush=True)
            print(f"  B = {s['B']}", flush=True)
            print(f"  C = {s['C']}", flush=True)
            print(f"  c_3(V) = {s['c3_V']}", flush=True)
            print(f"  h^0(B) = {s['h0_B']}, h^0(C) = {s['h0_C']}  -> stable", flush=True)
            print(f"  h^1(B) = {s['h1_B']}, h^1(C) = {s['h1_C']}", flush=True)
            print(f"  chi per class: {s['chi_per_class']}", flush=True)

    return survivors_pre_h0, survivors_final, h1_branch_candidates, counters


if __name__ == "__main__":
    pre, final, h1_cand, counters = main()
    print(flush=True)
    print("=" * 78, flush=True)
    print("CYCLE 9 VERDICT", flush=True)
    print("=" * 78, flush=True)
    if not final:
        print("SCENARIO A: 0 survivors pass H^0(V) = 0 stability filter.", flush=True)
        print(f"  Cycle-5 reported {len(pre)} pre-stability survivors;", flush=True)
        print(f"  ALL eliminated by Mumford-Takemoto stability.", flush=True)
        print(flush=True)
        print("This is the 9th independent failure mode for the substrate-Schoen", flush=True)
        print("uniqueness conjecture's empirical evidence — strengthening the", flush=True)
        print("case from '8-cycle empirical exclusion' to '9-cycle exclusion", flush=True)
        print("with mathematically airtight stability filter'.", flush=True)
    else:
        print(f"SCENARIO B: {len(final)} stable survivor(s) pass.", flush=True)
        if h1_cand:
            print(f"  Of these, {len(h1_cand)} have h^1(B) > 0 — i.e. the", flush=True)
            print(f"  H^1-branch could populate additional Yukawa modes.", flush=True)
            print(f"  These are cycle-10 LES_full implementation candidates.", flush=True)
        else:
            print(f"  None have h^1(B) > 0 — the H^1-branch contributes zero", flush=True)
            print(f"  extra modes, so the cycle-7-proposed LES_full implementation", flush=True)
            print(f"  cannot rescue any of these survivors.", flush=True)
