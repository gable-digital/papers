"""Cycle 9 validation: BBW h^* values + V_min2/AKLP h^0(V) check."""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ty_z3_bundle_constraints import (
    LineBundle, h_star_X_line_TY, h_zero_of_V,
    h0_of_line_bundle_sum_TY, h1_of_line_bundle_sum_TY,
)

# Reference values from cycle-8 probe (probe_h1_v_min2.rs, lines 1037-1064)
EXPECTED = {
    (0, 0):   [1, 0, 0, 1],
    (-1, -2): [0, 0, 0, 36],
    (-2, -1): [0, 0, 0, 36],
    (-1, 0):  [0, 0, 1, 4],
    (-1, -1): [0, 0, 0, 15],
    (1, 0):   [4, 1, 0, 0],
    (0, 1):   [4, 1, 0, 0],
    (1, 1):   [15, 0, 0, 0],
}

print("=" * 70)
print("Cycle 9: BBW h^*(X_TY, O(a, b)) cross-check vs cycle-8 probe values")
print("=" * 70)
all_match = True
for (a, b), expected in EXPECTED.items():
    got = h_star_X_line_TY(a, b)
    ok = got == expected
    all_match &= ok
    flag = "OK" if ok else "MISMATCH"
    print(f"  O({a:2d},{b:2d}): got={got}  expected={expected}  [{flag}]")

print()
if all_match:
    print("ALL h^* VALUES MATCH cycle-8 Rust probe. BBW Python mirror VALIDATED.")
else:
    print("MISMATCH DETECTED — fix Python BBW code before proceeding.")
    sys.exit(1)

print()
print("=" * 70)
print("V_min2 H^0(V) stability check")
print("=" * 70)
B_vmin2 = [LineBundle(0, 0), LineBundle(0, 0),
           LineBundle(-1, -2),
           LineBundle(-2, -1), LineBundle(-2, -1),
           LineBundle(-1, 0), LineBundle(-1, 0)]
C_vmin2 = [LineBundle(-1, -1),
           LineBundle(-2, -1), LineBundle(-2, -1), LineBundle(-2, -1)]
lower, upper, stable, info = h_zero_of_V(B_vmin2, C_vmin2)
print(f"  Σ h^0(B) = {info['h0_B']}    (cycle-8 probe: 2)")
print(f"  Σ h^0(C) = {info['h0_C']}    (cycle-8 probe: 0)")
print(f"  h^0(V) bounds: [{lower}, {upper}]   Stable? {stable}")
print(f"  EXPECTED: lower=2, upper=2, stable=False  (V_min2 known unstable)")
v_match = (info['h0_B'] == 2 and info['h0_C'] == 0 and lower == 2 and not stable)
print(f"  Validation: {'PASS' if v_match else 'FAIL'}")

print()
print("=" * 70)
print("AKLP H^0(V) stability check (positive control)")
print("=" * 70)
B_aklp = [LineBundle(1, 0)] * 3 + [LineBundle(0, 1)] * 3
C_aklp = [LineBundle(1, 1)] * 3
lower, upper, stable, info = h_zero_of_V(B_aklp, C_aklp)
print(f"  Σ h^0(B) = {info['h0_B']}    (cycle-8 probe: 24)")
print(f"  Σ h^0(C) = {info['h0_C']}    (cycle-8 probe: 45)")
print(f"  h^0(V) bounds: [{lower}, {upper}]   Stable? {stable}")
print(f"  EXPECTED: lower=0, stable=True (AKLP known stable, gives 9/27)")
a_match = (info['h0_B'] == 24 and info['h0_C'] == 45 and lower == 0 and stable)
print(f"  Validation: {'PASS' if a_match else 'FAIL'}")

print()
print("=" * 70)
print(f"OVERALL: BBW={all_match}, V_min2={v_match}, AKLP={a_match}")
print("=" * 70)
if all_match and v_match and a_match:
    print("Cycle 9 stability filter is READY for v3 survivor re-scan.")
    sys.exit(0)
else:
    sys.exit(1)
