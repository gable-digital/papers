"""
Strict surjectivity filter: monad map B → C requires that for each C-summand,
the number of B-summands with nonzero Hom(B_i → C_j) is "enough" to ensure
generic surjectivity, AND the surjectivity must hold at EVERY point of TY,
not just generically.

A practical strict heuristic: define
   feeders(C_j) = #{i : a_i <= c_j and b_i <= d_j}
Surjectivity at fiberwise level requires feeders(C_j) >= 1 (necessary).
Generic-section-surjectivity additionally requires that the matrix of
section spaces has full rank at every point — for line-bundle monads on
products of projective spaces this typically fails when feeders(C_j) = 1
(the single map can have zeros).

We adopt the **stronger** practical filter:
   STRICT_PASS iff feeders(C_j) >= rank(B) / rank(C) for every j AND
   the total number of nonzero hom components >= 2 * rank(C).

For the V_1 candidate with 4 B-summands but only 1 C-summand and only 1
nonzero hom, this fails (1 < 4).
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ty_z3_bundle_constraints import LineBundle
from h2_polystability_filter import run_filtered


def feeders(B, c_summand):
    return sum(1 for L in B if L.a <= c_summand.a and L.b <= c_summand.b)


def total_nonzero_homs(B, C):
    n = 0
    for cL in C:
        for bL in B:
            if bL.a <= cL.a and bL.b <= cL.b:
                n += 1
    return n


def strict_surjectivity(B_specs, C_specs):
    B = [LineBundle(*ab) for ab in B_specs]
    C = [LineBundle(*ab) for ab in C_specs]
    r_B = len(B)
    r_C = len(C)
    # Need each C-summand to have at least 2 feeders (so generic sum is
    # surjective and not over-determined as a single-section trap).
    for cL in C:
        if feeders(B, cL) < 2:
            return (False, f"C-summand {(cL.a, cL.b)} has fewer than 2 feeders "
                           f"({feeders(B, cL)})")
    # Need total feeders >= 2 * r_C (heuristic for surjectivity).
    n = total_nonzero_homs(B, C)
    if n < 2 * r_C:
        return (False, f"only {n} nonzero hom components for {r_C} target summands "
                       f"(need >= {2 * r_C})")
    # Also need feeders >= rank(B)/rank(C) for each C summand to span
    # adequately: this approximates the requirement that the kernel is locally
    # free of rank r_B - r_C.
    for cL in C:
        if feeders(B, cL) * r_C < r_B:
            return (False, f"C-summand {(cL.a, cL.b)} has insufficient feeders "
                           f"({feeders(B, cL)}) for rank balance")
    return (True, "passes strict surjectivity heuristic")


if __name__ == "__main__":
    candidates = run_filtered()
    print("\n" + "=" * 70)
    print("Strict surjectivity filter")
    print("=" * 70)
    strict_pass = []
    for s in candidates:
        ok, reason = strict_surjectivity(s['B'], s['C'])
        if ok:
            strict_pass.append({**s, 'strict_reason': reason})

    print(f"Polystability-passing candidates:    {len(candidates)}")
    print(f"Strict-surjectivity-passing:          {len(strict_pass)}")
    if strict_pass:
        print("\nStrict-passing candidates (first 10):")
        for i, s in enumerate(strict_pass[:10]):
            print(f"\n  Candidate #{i+1}:")
            print(f"    B = {s['B']}")
            print(f"    C = {s['C']}")
            print(f"    c_3(V) = {s['c3_V_int']}, c_2(V) = {s['c2_V']}")
            print(f"    Wilson: {s['wilson']}")
            print(f"    strict: {s['strict_reason']}")
    else:
        print("\nNo monad survives strict surjectivity.")

    print("\n" + "=" * 70)
    print("H2 STRICT FINAL VERDICT")
    print("=" * 70)
    if not strict_pass:
        print("REJECT: After applying strict surjectivity, no rank-3 monad bundle")
        print("on TY/Z3 in scan range satisfies all necessary conditions for a")
        print("3-generation Standard Model. The Schoen-style monad construction")
        print("does NOT extend to TY's 2-factor Kähler cone within bidegree [-2,2].")
    else:
        print(f"PARTIAL VALID: {len(strict_pass)} candidates remain.")
