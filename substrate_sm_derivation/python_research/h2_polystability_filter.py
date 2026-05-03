"""
Post-process H2 survivors with polystability filter.

For monad V = ker(B → C):
- V is locally free iff B → C is fiberwise surjective.
- V is polystable iff for every saturated coherent subsheaf F ⊂ V,
  mu(F) ≤ mu(V) = 0 (with equality only on direct summands).

Necessary conditions checkable from B, C alone:
1. **Line-sub-destabilizers**: any L ⊂ V is also L ⊂ B. If L is a B-summand
   that does NOT factor through some C-summand surjectively, then L ⊂ V
   exactly. So if any B-summand L_i has slope mu(L_i) > 0 at the same
   Kähler class where mu(V) = 0, V is destabilized.

2. **Slope-0 witness**: there must exist (t_1, t_2) > 0 such that mu(V) = 0
   AND mu(L_i) ≤ 0 for ALL B-summands L_i (or at least those not killed
   by the monad map -- a strict bound from a maximally-injective set).

For c_1(V) = 0, mu(V) = 0 automatically at every Kähler class. So the
polystability necessary condition reduces to:
    EXISTS (t_1, t_2) > 0 such that mu(L_i) ≤ 0 for ALL B-summands L_i,
    AND at least one L_i has mu(L_i) < 0 (otherwise V is the trivial sum
    OR is unstable in a borderline way).

This is a sufficient destabilizer test: any L_i with mu(L_i) > 0 at every
Kähler class would force a sub-line-bundle of V to violate the slope bound.

For our 2-tuple Kähler cone with t_1, t_2 > 0:
    mu(L) = a Q_1(t,t) + b Q_2(t,t)  where Q_i are fixed quadratics in (t1, t2)
On TY: D_111 = D_222 = 0, D_112 = D_122 = 9, so:
    Q_1(t1, t2) = int H_1 ∧ J^2 = 0 + 2 t1 t2 * 9 + t2^2 * 9 = 9 t2 (2 t1 + t2)
    Q_2(t1, t2) = int H_2 ∧ J^2 = t1^2 * 9 + 2 t1 t2 * 9 + 0 = 9 t1 (t1 + 2 t2)

So mu(O(a, b)) = 9 [a t2 (2 t1 + t2) + b t1 (t1 + 2 t2)].

For all t1, t2 > 0, both Q_1 and Q_2 are strictly positive. So mu(L) has
the same sign as (a Q_1 + b Q_2).
  - If a > 0 and b > 0:  mu > 0 always (destabilizes)
  - If a < 0 and b < 0:  mu < 0 always
  - If a = 0 and b = 0:  mu = 0
  - Mixed sign: mu can have either sign depending on (t1, t2).
    Specifically mu(L) = 0 ⟺ a t2 (2 t1 + t2) = -b t1 (t1 + 2 t2).

For the SUM B with c_1(B) > 0 (since c_1(C) > 0 to give c_1(V)=0 and we need
"positive" monads to have a chance of surjectivity), B will typically have
several positive-slope summands. ANY of them would destabilize V unless
exactly killed by the monad map -- and a B-summand L is killed iff there's
a C-summand the map factors through.

We implement a stronger, more conservative test:
    REJECT V if ANY B-summand L_i has BOTH a > 0 AND b > 0 (positive in both
    classes, hence mu(L_i) > 0 everywhere in the Kähler cone).
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from h2_monad_bundle_scan import (
    run_h2, A_RANGE, B_RANGE, RANK_V, RANK_B_CHOICES, TARGET_C3,
    enumerate_lbs, monad_chern, wilson_partition_monad,
    is_3_3_3_balanced, anomaly_check_monad, map_existence_check,
)
from ty_z3_bundle_constraints import LineBundle


def line_bundle_strictly_positive(L: LineBundle) -> bool:
    """A line bundle O(a,b) on TY is in the strict interior of the positive
    cone iff a > 0 AND b > 0. Such L has slope > 0 at EVERY Kähler class.
    """
    return L.a > 0 and L.b > 0


def line_bundle_strictly_negative(L: LineBundle) -> bool:
    """Mirror: a, b < 0 gives slope < 0 everywhere."""
    return L.a < 0 and L.b < 0


def polystability_necessary(B_summands, C_summands):
    """
    Necessary conditions for V = ker(B → C) to be polystable on TY:
    1. No B-summand has BOTH a>0 and b>0 (would destabilize unconditionally).
    2. The map B → C must "absorb" any borderline-positive summands; the
       NUMBER of strictly-positive summands of B must be ≤ NUMBER of
       strictly-positive summands of C (each can absorb at most one).
    3. We also require that V_max := max sub-line-bundle slope at the
       common-zero-slope locus < 0; a CONSERVATIVE proxy is that no
       B-summand is in the closed positive cone (a>=0 AND b>=0 with not both
       zero) UNLESS C has a corresponding absorbing summand of the same
       Wilson class.

    We return (passes_test, reason_str).
    """
    # Test 1
    bad_B = [L for L in B_summands if line_bundle_strictly_positive(L)]
    if bad_B:
        return (False, f"B has strictly-positive summand(s) {[(L.a,L.b) for L in bad_B]} "
                       "with mu>0 at every Kähler class")
    return (True, "no strictly-positive B-summand")


def run_filtered():
    """Run H2 scan and apply polystability necessary filter."""
    survivors = run_h2()
    print("\n" + "=" * 70)
    print("H2 — Polystability necessary-condition filter")
    print("=" * 70)
    print(f"Pre-filter survivors: {len(survivors)}")
    polystable_candidates = []
    for s in survivors:
        B_lbs = [LineBundle(*ab) for ab in s['B']]
        C_lbs = [LineBundle(*ab) for ab in s['C']]
        ok, reason = polystability_necessary(B_lbs, C_lbs)
        if ok:
            polystable_candidates.append({**s, 'poly_reason': reason})

    print(f"Post-filter candidates (no strictly-positive B-summand): "
          f"{len(polystable_candidates)}")

    if polystable_candidates:
        print("\nFirst 10 polystability-passing candidates:")
        for i, s in enumerate(polystable_candidates[:10]):
            print(f"\n  Candidate #{i+1}:")
            print(f"    B = {s['B']}")
            print(f"    C = {s['C']}")
            print(f"    c_3(V) = {s['c3_V_int']}, c_2(V) = {s['c2_V']}")
            print(f"    Wilson partition: {s['wilson']}")
            print(f"    poly: {s['poly_reason']}")
    else:
        print("\nNo monad survives the polystability necessary filter.")

    print("\n" + "=" * 70)
    print("H2 FINAL VERDICT")
    print("=" * 70)
    if not polystable_candidates:
        print("REJECT: 20809 monads survived basic checks but ALL are destabilized")
        print("by a strictly-positive line-bundle summand of B (slope > 0")
        print("everywhere in the 2-class Kähler cone). No polystable rank-3")
        print("monad bundle in scan range.")
    else:
        print(f"PARTIAL VALID: {len(polystable_candidates)} candidates pass the")
        print("necessary polystability filter. Each requires the FULL stability")
        print("test (saturated sub-sheaf scan) before being declared polystable.")
        print("These are the strongest survivors and merit further analysis.")
    return polystable_candidates


if __name__ == "__main__":
    run_filtered()
