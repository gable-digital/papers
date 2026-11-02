# -*- coding: ascii -*-
"""Generate every number, commit reference and claim table used by the two papers.

The papers papers/acorr_derivation and papers/two_center_born_infeld quote no
hand-typed number or commit hash. This script reads each quoted value from a
corpus blob at a recorded commit (git show), recomputes the closed-form
constants with mpmath, independently recomputes the order-18 indicial
multisets from the stored coefficient arrays, and writes:

  numbers.tex            (identical copy in both paper folders)
  claims_table.tex       (per paper: the claim, status and source table)
  timeline_table.tex     (acorr paper: the dated commit history)
  REVIEW_PACKET.md       (per paper: the generated section between markers)

Usage (from anywhere; the monorepo is the papers submodule's parent):
  python generate_numbers.py            write outputs, reusing provenance_refs.json
  python generate_numbers.py --check    regenerate in memory, exit 1 on any drift
  python generate_numbers.py --refresh  re-resolve the branch refs first

Requires sympy and mpmath. Reads git objects only; never writes the corpus.
"""
import argparse
import gzip
import io
import json
import os
import re
import subprocess
import sys
from contextlib import redirect_stdout
from fractions import Fraction

import mpmath as mp
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
PAPERS = os.path.dirname(HERE)
MONO = os.path.dirname(PAPERS)
ACORR_DIR = os.path.join(PAPERS, "acorr_derivation")
TWO_DIR = HERE
REFS_FILE = os.path.join(HERE, "provenance_refs.json")
SY = "book/gr_qm_unification/sympy/"
FO = SY + "derivations/rtail_frozen_ode/"

# Branch aliases cited by the papers. Resolved once to full hashes and stored.
REF_NAMES = {
    "dev": "origin/dev",
    "ode": "origin/math/rtail-frozen-ode-1-10",
    "c27": "origin/review/order17-divisor-c27-20261101",
    "bond": "origin/substrate-bond-generator-20261101",
    "acorr28": "origin/research/acorr-newton-g-c28",
    "ion": "origin/research/ionization-generator-period3-20261101",
}


GIT_ENV = dict(os.environ, TZ="UTC")


def git(*args, binary=False):
    out = subprocess.run(["git", "-C", MONO] + list(args), capture_output=True, check=True,
                         env=GIT_ENV).stdout
    return out if binary else out.decode("utf-8")


def resolve_refs(refresh):
    if os.path.exists(REFS_FILE) and not refresh:
        with open(REFS_FILE, encoding="ascii") as fh:
            refs = json.load(fh)
    else:
        refs = {}
    # Stored references stay fixed; an alias added later is resolved once and then stored.
    for alias, name in REF_NAMES.items():
        if alias not in refs:
            refs[alias] = {"name": name, "commit": git("rev-parse", "--verify", name + "^{commit}").strip()}
    for alias, rec in refs.items():
        if not re.fullmatch(r"[0-9a-f]{40}", rec["commit"]):
            raise SystemExit("bad recorded commit for %s" % alias)
        git("cat-file", "-e", rec["commit"] + "^{commit}")
    return refs


def blob(refs, alias, path):
    raw = git("show", refs[alias]["commit"] + ":" + path, binary=True)
    return gzip.decompress(raw) if path.endswith(".gz") else raw


def jblob(refs, alias, path):
    return json.loads(blob(refs, alias, path).decode("utf-8"))


def last_commit(refs, alias, path):
    line = git("log", "-1", "--date=format-local:%Y-%m-%dT%H:%MZ", "--format=%H%x09%cd",
               refs[alias]["commit"], "--", path).strip()
    if not line:
        raise SystemExit("no commit touches %s at %s" % (path, alias))
    h, date = line.split("\t")
    return h, date


def is_ancestor(commit, target):
    return subprocess.run(["git", "-C", MONO, "merge-base", "--is-ancestor", commit, target]).returncode == 0


def exec_corpus_script(refs, alias, path):
    """Run a light corpus script from its blob in an isolated namespace; return stdout."""
    src = blob(refs, alias, path).decode("ascii")
    sympy_dir = os.path.join(MONO, SY)
    ns = {"__name__": "__corpus__", "__file__": os.path.join(sympy_dir, "derivations", "x.py")}
    buf = io.StringIO()
    old_path = list(sys.path)
    try:
        with redirect_stdout(buf):
            exec(compile(src, path, "exec"), ns)
    finally:
        sys.path[:] = old_path
    return buf.getvalue()


# ---------------------------------------------------------------- numbers
def fmt(x, digits):
    return mp.nstr(x, digits, strip_zeros=False)


def trunc(s, digits_after_point):
    s = s.strip()
    sign = "-" if s.startswith("-") else ""
    s = s.lstrip("-")
    whole, _, frac = s.partition(".")
    return sign + whole + "." + frac[:digits_after_point]


def sci(s, digits=2):
    return mp.nstr(mp.mpf(s), digits, min_fixed=1, max_fixed=0)


def tex_sci(s, digits=3):
    v = mp.mpf(s)
    if v == 0:
        return "0"
    e = int(mp.floor(mp.log10(abs(v))))
    m = v / mp.mpf(10) ** e
    return "\\ensuremath{%s\\times10^{%d}}" % (mp.nstr(m, digits), e)


def indicial_multiset(coeffs, at_zero):
    """Exact indicial roots of sum_j p_j(d) D^j at d=0 or at infinity (z=1/d)."""
    rho = sp.Symbol("rho")
    polys = [[Fraction(int(c)) for c in row] for row in coeffs]
    if at_zero:
        vals = [next(i for i, c in enumerate(row) if c != 0) for row in polys]
        m = min(v - j for j, v in enumerate(vals))
        terms = [(j, polys[j][j + m]) for j in range(len(polys)) if vals[j] - j == m]
        ind = sum(sp.Rational(c.numerator, c.denominator) * sp.ff(rho, j) for j, c in terms)
    else:
        degs = [max(i for i, c in enumerate(row) if c != 0) for row in polys]
        m = max(dg - j for j, dg in enumerate(degs))
        terms = [(j, polys[j][j + m]) for j in range(len(polys)) if degs[j] - j == m]
        ind = sum(sp.Rational(c.numerator, c.denominator) * sp.ff(-rho, j) for j, c in terms)
    poly = sp.Poly(sp.expand(ind), rho)
    roots = sp.roots(poly, filter="Q")
    total = sum(roots.values())
    if total != poly.degree():
        raise SystemExit("indicial polynomial has non-rational roots (%d of %d)" % (total, poly.degree()))
    out = []
    for r, k in roots.items():
        out += [sp.Rational(r)] * k
    return sorted(out)


def tex_multiset(vals):
    parts = []
    for v in vals:
        v = sp.Rational(v)
        parts.append(str(v.p) if v.q == 1 else "%s%d/%d" % ("-" if v < 0 else "", abs(v.p), v.q))
    return "\\{" + ",".join(parts) + "\\}"


def local_remainder_counterexample():
    """Paper A, contact block: the split a = sum mu_k G_k + R_loc + R_out needs the local Taylor remainder
    R_loc = int_0^{r*} (G - sum_k G_k r^{p_k}) w dr. For G = r^10 e^-r, w = 1, r* = 1 and retained exponents
    3..9 every retained coefficient is zero while R_loc = 10! - 9864101/e > 0. Checked symbolically here;
    the identity contact + R_loc + R_out = total is checked exactly as well."""
    r = sp.Symbol("r", positive=True)
    g = r**10 * sp.exp(-r)
    ps = range(3, 10)
    ser = sp.series(g, r, 0, 10).removeO()
    gk = [ser.coeff(r, p) for p in ps]
    if any(c != 0 for c in gk):
        raise SystemExit("counterexample: a retained coefficient is nonzero: %s" % gk)
    r_loc = sp.integrate(g, (r, 0, 1))
    if sp.simplify(r_loc - (sp.factorial(10) - sp.Integer(9864101) / sp.E)) != 0:
        raise SystemExit("counterexample: R_loc is not 10! - 9864101/e: %s" % r_loc)
    r_out = sp.integrate(g, (r, 1, sp.oo))
    if sp.simplify(r_loc + r_out - sp.integrate(g, (r, 0, sp.oo))) != 0:
        raise SystemExit("counterexample: contact + R_loc + R_out differs from the total")
    val = mp.mpf(sp.N(r_loc, 40))
    if not val > 0:
        raise SystemExit("counterexample: R_loc is not positive")
    return mp.nstr(val, 6)


def local_remainder_zero_witness():
    """Paper A, contact block: pointwise equality with the retained expansion is sufficient for R_loc = 0 but
    not necessary. For G = r^3 + r^10 (r - 11/12), w = 1, r* = 1 and retained exponents 3..9 the retained
    expansion is r^3 and R_loc = int_0^1 r^10 (r - 11/12) dr = 0, while the residual at r = 1/2 is -5/12288
    and G > 0 on (0, 1]. Checked exactly here; returns the residual at r = 1/2 as an exact rational."""
    r = sp.Symbol("r", positive=True)
    g = r**3 + r**10 * (r - sp.Rational(11, 12))
    poly = sp.Poly(sp.expand(g), r)
    retained = sum(poly.coeff_monomial(r**p) * r**p for p in range(3, 10))
    if sp.expand(retained - r**3) != 0:
        raise SystemExit("zero witness: retained expansion is not r^3: %s" % retained)
    residual = sp.expand(g - retained)
    if sp.integrate(residual, (r, 0, 1)) != 0:
        raise SystemExit("zero witness: R_loc is not zero")
    at_half = residual.subs(r, sp.Rational(1, 2))
    if at_half != sp.Rational(-5, 12288):
        raise SystemExit("zero witness: residual at 1/2 is not -5/12288: %s" % at_half)
    # G / r^3 = 1 + r^7 (r - 11/12) >= 1 - (11/12) r^7 >= 1/12 on (0, 1], so G > 0 there.
    if sp.minimum(sp.expand(g / r**3), r, sp.Interval.Lopen(0, 1)) <= 0:
        raise SystemExit("zero witness: G is not positive on (0, 1]")
    return "%d/%d" % (at_half.p, at_half.q)


def compute_numbers(refs):
    mp.mp.dps = 60
    g = mp.gamma(mp.mpf(1) / 4) ** 2
    K = g / (6 * mp.sqrt(mp.pi))
    C0 = g / (4 * mp.sqrt(mp.pi))
    n = {}
    n["KVal"] = fmt(K, 16)
    n["CzeroVal"] = fmt(C0, 16)
    n["TwoKVal"] = fmt(2 * K, 16)
    n["ContactTotalVal"] = fmt(2 * mp.sqrt(2) * K, 16)
    n["VzeroVal"] = fmt((2 * mp.sqrt(2) - 2) * K, 16)
    n["EndpointRatioVal"] = fmt(4 * (mp.sqrt(2) - 1) / 3, 16)
    n["TestChargeGapVal"] = fmt((mp.mpf(7) / 2 - 2 * mp.sqrt(2)) * K, 16)
    n["FrozenCfourVal"] = fmt(3 * K / 4, 16)
    eta2 = mp.mpf(9) / 32 * mp.log(mp.mpf(3) / 4) - mp.mpf(13) / 432
    n["EtaTwoVal"] = fmt(eta2, 21)

    # eps_2 literal from the corpus source (a published value, read not typed)
    src = blob(refs, "dev", SY + "imported_theorems/second_order_partial_waves.py").decode("ascii")
    m = re.search(r'def published_eps2\(\):.*?eps_2 = (-0\.[0-9]{20})', src, re.S)
    if not m:
        raise SystemExit("published eps_2 literal not found")
    eps2 = mp.mpf(m.group(1))
    n["EpsTwoVal"] = m.group(1)
    n["AcorrDiagVal"] = fmt(eps2 - eta2, 20)
    n["AcorrMagVal"] = fmt(abs(eps2 - eta2), 12)

    # The particle-expression route: values recorded in FORWARD_ELEMENT_GENERATOR.md.
    feg = blob(refs, "dev", "book/scripts/FORWARD_ELEMENT_GENERATOR.md").decode("utf-8")
    vm = re.search(r"Z=2: (0\.[0-9]+), Z=4: (0\.[0-9]+), Z=8: (0\.[0-9]+), Z=16: (0\.[0-9]+) Ha", feg)
    if not vm:
        raise SystemExit("VMC values not found in FORWARD_ELEMENT_GENERATOR.md")
    vmc = {2: vm.group(1), 4: vm.group(2), 8: vm.group(3), 16: vm.group(4)}
    n["VmcValuesVal"] = ", ".join("$Z=%d$: %s" % (z, vmc[z]) for z in (2, 4, 8, 16))
    # the script's own extrapolation: least squares a + b/Z over Z = 4, 8, 16
    zs = [4, 8, 16]
    A = sp.Matrix([[1, sp.Rational(1, z)] for z in zs])
    y = sp.Matrix([sp.Rational(vmc[z]) for z in zs])
    sol = (A.T * A).inv() * A.T * y
    n["VmcExtrapVal"] = mp.nstr(mp.mpf(sp.N(sol[0], 30)), 4)
    n["ScreenedConstVal"] = mp.nstr(mp.mpf(25) / 256, 8)
    n["ExactMinusScreenedVal"] = mp.nstr(abs(eps2 + mp.mpf(25) / 256), 6)
    n["TwentyOneDevVal"] = "%.2f" % float(100 * (mp.mpf(1) / 21 / abs(eps2 - eta2) - 1))
    n["SixtyFourDevVal"] = "%.2f" % float(100 * (mp.mpf(3) / 64 / abs(eps2 - eta2) - 1))
    n["RlocCounterVal"] = local_remainder_counterexample()
    n["RlocZeroResidualVal"] = local_remainder_zero_witness()

    # partial-wave split: run the corpus script from its blob
    out = exec_corpus_script(refs, "dev", SY + "derivations/acorr_partialwave_decomposition.py")
    fr = re.findall(r"fraction of a_corr = ([0-9.]+)", out)
    s_wave = re.search(r"l=0 correlation\s+eps_2\(0\)-eta_2 = (-?[0-9.]+)", out)
    tail0 = re.search(r"cusp law summed from l=0\s+:\s+([0-9.]+)", out)
    eps20 = re.search(r"TRUE eps_2\(0\) = (-?[0-9.]+)", out)
    if len(fr) < 2 or not (s_wave and tail0 and eps20):
        raise SystemExit("partial-wave script output not recognized")
    n["SwaveFracVal"] = "%.1f" % (100 * float(fr[0]))
    n["HigherlFracVal"] = "%.1f" % (100 * float(fr[1]))
    n["SwaveCorrVal"] = s_wave.group(1)
    n["CuspTailSumVal"] = "%.2f" % float(tail0.group(1))
    n["EpsTwoZeroVal"] = eps20.group(1)

    # nonsuperposition witness: run from its blob
    out = exec_corpus_script(refs, "dev", SY + "derivations/bi_two_center_nonsuperposition_witness.py")
    w = re.search(r"obstruction \(grad\|D\| x D\)_z at off-axis point.*?: (-?[0-9.]+)", out)
    if not w or "WITNESS OK" not in out:
        raise SystemExit("nonsuperposition witness output not recognized")
    n["WitnessVal"] = mp.nstr(mp.mpf(w.group(1)), 12)

    # frozen-ODE receipts (dev)
    init = jblob(refs, "dev", FO + "initial-data.json")
    n["WoneVal"] = trunc(init["derivatives"][0], 30)
    n["WprimeOneVal"] = trunc(init["derivatives"][1], 30)
    ep = jblob(refs, "dev", FO + "endpoint-data.json")
    k192 = [r for r in ep["contact_quadrature"] if r["nodes"] == 192][0]["kappa_super"]
    k96 = [r for r in ep["contact_quadrature"] if r["nodes"] == 96][0]["kappa_super"]
    n["KappaSuperVal"] = trunc(k192, 36)
    with mp.workdps(200):
        n["KappaChangeVal"] = tex_sci(mp.nstr(abs(mp.mpf(k192) - mp.mpf(k96)), 5))
    ivp = jblob(refs, "dev", FO + "ivp-replay.json")
    if ivp["verdict"] != "PASS" or ivp["wrong_initial_data"] != "REJECTED":
        raise SystemExit("ivp replay receipt not PASS")
    n["IvpErrVal"] = tex_sci(ivp["absolute_error"])
    n["IvpThreshVal"] = tex_sci(ivp["threshold"], 1)
    n["IvpTargetVal"] = ivp["target"]
    n["IvpOracleVal"] = tex_sci(ivp["oracle_refinement"])
    n["IvpWrongVal"] = tex_sci(ivp["wrong_initial_data_error"], 1)
    nv = jblob(refs, "dev", FO + "numeric-verification.json")
    rows = []
    worst = mp.mpf(0)
    for c in nv["accepted_cases"]:
        if not (c["pass"] and c["wrong_rejected"]):
            raise SystemExit("numeric verification case not passing")
        worst = max(worst, mp.mpf(c["W_residual"]))
        rows.append("%s & %s & $%s$ & $%s$ \\\\" % (c["d"], c["nodes"], tex_sci(c["W_residual"]),
                                                    tex_sci(c["wrong_c0_one_percent"])))
    n["ResidualRows"] = "\n".join(rows)
    n["ResidualThreshVal"] = tex_sci(nv["threshold"], 1)
    n["ResidualWorstVal"] = tex_sci(str(worst))
    mc = jblob(refs, "dev", FO + "master-comparison.json")
    n["MasterMaxErrVal"] = tex_sci(str(max(mp.mpf(c["absolute_error"]) for c in mc["cases"])))
    ev = jblob(refs, "dev", FO + "exact-verification.json")["verdict"]
    n["EnergyOrderVal"] = str(ev["energy_order"])
    n["ScalarOrderVal"] = str(ev["scalar_order"])
    n["DivergenceColsVal"] = str(ev["divergence_columns"])
    n["ApparentDegVal"] = str(ev["apparent_factor_degree"])
    eop = jblob(refs, "dev", FO + "energy-operator-char0.json")
    coeffs = eop["coefficients"]
    if len(coeffs) - 1 != int(eop["order"]) or str(eop["minimal_order_asserted"]) != "False":
        raise SystemExit("energy operator receipt shape changed")
    degs = [max(i for i, c in enumerate(row) if int(c) != 0) for row in coeffs]
    n["EnergyDegMinVal"] = str(min(degs))
    n["EnergyDegMaxVal"] = str(max(degs))
    hop = jblob(refs, "dev", FO + "operator-char0.json")
    hco = hop["coefficients"] if isinstance(hop, dict) else hop
    hdegs = [max(i for i, c in enumerate(row) if int(c) != 0) for row in hco]
    n["HDegMinVal"] = str(min(hdegs))
    n["HDegMaxVal"] = str(max(hdegs))
    z18 = indicial_multiset(coeffs, True)
    i18 = indicial_multiset(coeffs, False)
    want_z = [sp.Rational(x) for x in "-1 0 1 1 1 7/3 3 11/3 5 5 6 19/3 23/3 9 9 13 13 17".split()]
    want_i = [sp.Rational(x) for x in "-3 -1 -1/3 -1/3 0 1/3 1 1 5/3 2 7/3 7/3 4 5 6 7 9 9".split()]
    if z18 != sorted(want_z) or i18 != sorted(want_i):
        raise SystemExit("recomputed order-18 indicial multisets disagree with the FINDINGS text")
    n["IndZeroEighteen"] = tex_multiset(z18)
    n["IndInfEighteen"] = tex_multiset(i18)
    spb = jblob(refs, "dev", FO + "selected-period-bound.json")
    n["SelectedLowerVal"] = str(spb["selected_W_order_lower_bound"])
    if spb["selected_W_algebraic"] is not False:
        raise SystemExit("selected bound receipt changed")

    # order-17 receipts (frozen-ode branch)
    r17 = jblob(refs, "ode", FO + "radial-scalar-checks/result.json")
    if r17["order"] != 17 or r17["minimum_asserted"] is not False:
        raise SystemExit("order17 receipt changed")
    n["OrderSeventeenVal"] = str(r17["order"])
    dv = jblob(refs, "ode", FO + "radial-divisor-checks/result.json")
    n["OldFactorDegVal"] = str(dv["old_exceptional_factor_degree"])
    n["NewFactorDegVal"] = str(dv["new_exceptional_factor_degree"])
    if not (dv["old_exceptional_roots_are_ordinary_for_new_scalar_operator"]
            and dv["new_exceptional_factor_squarefree_and_off_ambient_poles"]):
        raise SystemExit("order17 divisor receipt changed")
    ind = jblob(refs, "ode", FO + "radial-divisor-checks/indicial-result.json")
    for side, key in (("zero", "IndZeroSeventeen"), ("infinity", "IndInfSeventeen")):
        if ind[side]["remaining_factors"]:
            raise SystemExit("order17 indicial factors not all rational")
        vals = sorted(4 * sp.Rational(x) for x in ind[side]["rational_exponents"])
        n[key] = tex_multiset(vals)

    # true-minimizer certificates (dev)
    d1 = jblob(refs, "dev", SY + "derivations/rtail_step2_d1_certificate.json")
    lo, hi = Fraction(d1["lower"]), Fraction(d1["upper"])
    n["DoneLowerVal"] = mp.nstr(mp.floor(mp.mpf(lo.numerator) / lo.denominator * 10**8) / 10**8, 8)
    n["DoneUpperVal"] = mp.nstr(mp.ceil(mp.mpf(hi.numerator) / hi.denominator * 10**8) / 10**8, 8)
    n["DoneRelGapVal"] = mp.nstr(mp.mpf(Fraction(d1["relative_gap_to_upper"]).numerator)
                                 / Fraction(d1["relative_gap_to_upper"]).denominator * 100, 4)
    ct = jblob(refs, "dev", SY + "derivations/rtail_contact_asymptotic_certificate.json")
    if ct["contact_energy_right_derivative"] != "-1/2" or ct["coalescence_premise_consumed"] is not False:
        raise SystemExit("contact certificate changed")
    # The paper transcribes this enclosure; refuse if the certified statement differs.
    if (ct["remainder"] != "0 <= V(d)-V(0)+d/2 <= (31/20)*(d/2)^(5/3)" or ct["domain"] != "0<=d<=1/2"
            or ct["negative_right_derivative"] != "1/2"):
        raise SystemExit("contact enclosure statement differs from the transcribed form")
    n["ContactBitsVal"] = str(ct["bits"])
    n["ContactWidthHundredthVal"] = mp.nstr(mp.mpf(31) / 20 * (mp.mpf(1) / 200) ** (mp.mpf(5) / 3), 3)
    v_sup1 = mp.mpf(init["derivatives"][0]) - 2 * K
    n["FrozenVoneVal"] = mp.nstr(v_sup1, 12)
    n["FrozenMinusUpperVal"] = mp.nstr(mp.floor((v_sup1 - mp.mpf(hi.numerator) / hi.denominator) * 10**5) / 10**5, 5)
    n["KappaHalfGapVal"] = mp.nstr(mp.mpf(1) / 2 - mp.mpf(k192), 6)

    # bond-lane receipts (unreviewed author lane)
    sc = jblob(refs, "bond", SY + "derivations/diatomic_frozen_pair_series_check.json")
    if sc["certified"] is not False:
        raise SystemExit("bond series receipt changed")
    brow = []
    for key in ("1.25 upper", "1.5 upper", "2 upper", "3 upper", "4 upper"):
        r = sc["rows"][key]
        brow.append("%s & %s & $%s$ \\\\" % (r["target"], r["steps"], tex_sci(r["abs_difference"])))
    n["BondSeriesRows"] = "\n".join(brow)
    diffs = [mp.mpf(sc["rows"][k]["abs_difference"]) for k in sc["rows"] if "abs_difference" in sc["rows"][k]]
    n["BondDiffMinVal"] = tex_sci(mp.nstr(min(diffs), 5), 2)
    n["BondDiffMaxVal"] = tex_sci(mp.nstr(max(diffs), 5), 2)
    n["BondSeriesDpsVal"] = str(sc["dps"])
    n["BondImagVal"] = tex_sci(sc["rows"]["1.25 upper"]["W_ode_imag"].lstrip("-"))
    n["BondHalfVal"] = trunc(sc["rows"]["0.5 real-left"]["W_ode"], 29)
    fe = jblob(refs, "bond", SY + "derivations/diatomic_static_pair_fences.json")
    n["BondFenceRows"] = bond_fence_rows(fe)
    bs = [b for b in fe["bond_scale"] if b["bond_length_m"] == "1e-10"][0]
    n["BondScaleDVal"] = bs["d_core_units"]
    n["BondScaleRelVal"] = tex_sci(bs["relative_departure_from_inverse_d"])
    n["BondCoreRadiusVal"] = tex_sci(bs["core_radius_m"], 5)
    return n


def bond_fence_rows(fe):
    rows = fe.get("rows") if isinstance(fe, dict) else None
    if not isinstance(rows, list):
        raise SystemExit("bond fence receipt shape not recognized")
    if fe.get("certified_lower") is not True:
        raise SystemExit("bond fence receipt is not a certified lower fence")
    out = []
    for r in rows:
        d = str(r["d"])
        if d in ("1/100", "1/4", "1", "4", "100"):
            dv = mp.mpf(Fraction(d).numerator) / Fraction(d).denominator
            low = mp.mpf(r["V_lower"])
            out.append("%s & %s & %s \\\\" % (d, mp.nstr(low, 10), mp.nstr(low * dv, 6)))
    if len(out) != 5:
        raise SystemExit("bond fence rows missing")
    return "\n".join(out)


# ---------------------------------------------------------------- claims
# (id, paper, statement, status, ref alias, path, function or None)
CLAIMS = [
    ("A1", "A", "a_corr is defined as eps_2 - eta_2 of the Coulomb comparison model (high-Z, same centre, two electrons)", "DEFINITION", "dev", SY + "substrate_theorems/atomic_correlation_constant_not_forced.py", "verdict"),
    ("A2", "A", "eta_2 = (9/32)ln(3/4) - 13/432 by the Dalgarno-Lewis reduction", "DERIVED (imported Coulomb operator)", "dev", SY + "substrate_theorems/atomic_correlation_constant_not_forced.py", "eta2_symbolic_derivation"),
    ("A3", "A", "eps_2 = published value (Wang 2013), not a substrate result; with it the a_corr value is a cross-check oracle", "IMPORTED (literature; cross-check oracle only)", "dev", SY + "imported_theorems/second_order_partial_waves.py", "published_eps2"),
    ("A4", "A", "s-wave and l>=1 split of a_corr from literature per-l increments", "DIAGNOSTIC (literature inputs)", "dev", SY + "derivations/acorr_partialwave_decomposition.py", None),
    ("A5", "A", "the s-wave crossover correction lies outside the elementary plus Ei span (non-membership only)", "DERIVED (conditional)", "dev", SY + "substrate_theorems/atomic_correlation_constant_not_forced.py", "eps2_swave_crossover_is_non_elementary"),
    ("A6", "A", "coincidence match is a square system on the retained contact coefficients; it fixes them, not a_corr", "DERIVED", "dev", SY + "substrate_theorems/atomic_correlation_constant_not_forced.py", "substrate_correlation_coincidence_match_underdetermines"),
    ("A7", "A", "proximity cap rows add zero rank; (a_corr, remainder) block stays rank 1 in 2 unknowns", "DERIVED (finite contact-coefficient model only)", "dev", SY + "derivations/acorr_proximity_cap_rank_analysis.py", None),
    ("A8", "A", "test-charge contact ratio V(1,1;0)/U(0) = 4(sqrt2-1)/3, not 1", "CERTIFIED-EXACT", "dev", SY + "substrate_theorems/bi_test_charge_pair_energy.py", "contact_interaction_certificate"),
    ("A9", "A", "scalar point-source potential: exact substitution, quartic range [1/8,1], Coulomb bracket", "DERIVED (scalar route only)", "dev", SY + "derivations/acorr_charge_tail_dependency_FINDINGS.md", None),
    ("A10", "A", "static linear strain tail is l=0 and the wake is O(alpha^2) inside the Coulomb comparison only", "CONDITIONAL DIAGNOSTIC", "dev", SY + "substrate_theorems/atomic_correlation_constant_not_forced.py", "substrate_correlation_strain_tail_and_wake_not_the_resisting_piece"),
    ("A11", "A", "static two-centre energy to a_corr identification", "COMMITMENT (M11-gated)", "dev", SY + "substrate_theorems/rtail_two_center_synthesis.py", "two_center_synthesis"),
    ("A12", "A", "M11 true-minimizer route: physical moving-mode functional and bound mode missing", "OPEN", "dev", SY + "derivations/acorr_m11_true_minimizer_route_validation.py", None),
    ("A13", "A", "former order-4 Picard-Fuchs non-elementarity claim", "WITHDRAWN", "dev", SY + "derivations/rtail_known_problems_context.md", None),
    ("A14", "A", "a_corr derived from the Born-Infeld Lagrangian and the particle equations (the required route)", "OPEN", "dev", SY + "substrate_theorems/atomic_correlation_constant_not_forced.py", "verdict"),
    ("A15", "A", "electron mass from the fixed closed-form particle equations (default structural anchor)", "FIXED CLOSED FORM (framework premise)", "dev", "book/scripts/substrate_mass_spectrum.py", "derive_constants"),
    ("A16", "A", "energy unit alpha^2 m_e c^2 that expresses a_corr in eV, from the particle module", "CONDITIONAL (module marks a_0 and Ry not forward-closed)", "dev", "book/scripts/forward_p_row_generator.py", "hartree_eV"),
    ("A17", "A", "r12 Jastrow VMC 'forward a': Monte Carlo of the ordinary Coulomb problem, screened reference, extrapolated", "IMPORTED (cross-check only; not a derivation)", "dev", "book/scripts/derive_pair_correlation_constant_forward.py", "corr_at_Z"),
    ("A18", "A", "b/a = -3/4 identified with the spin-1/2 Casimir of the particle sector", "OPEN (lead, to confirm)", "dev", "book/scripts/FORWARD_ELEMENT_GENERATOR.md", None),
    ("A19", "A", "element generator installs a_corr = 4667/100000 as a legacy diagnostic literal", "IMPORTED (literature value installed)", "dev", "book/scripts/forward_generator_complete.py", "legacy_orbital_ie_diagnostic"),
    ("A20", "A", "occupation shape k(3k-11)/8 and b/a = -3/4 found by least-squares fit of the period-2 residual", "FITTED", "dev", "book/scripts/forward_missing_terms_generator.py", None),
    ("A21", "A", "legacy optimizer minimized a removed-orbital expression, not the atom's energy (September correction)", "CERTIFIED-EXACT (defect)", "dev", SY + "derivations/atomic_generator_energy_consistency_FINDINGS.md", None),
    ("A22", "A", "period 2 under 1% with zero fits superseded; corrected objective gives B to Ne -16.1 to -53.6 percent", "UNREVIEWED (other lane probe)", "ion", "work_journal/ionization-generator-period3-open.md", None),
    ("A23", "A", "proximity cap consistency sum_k L_k r_min^p_k = 0 is a condition on r_min; a positive root and its uniqueness are not proved", "OPEN (result left unset)", "dev", SY + "substrate_theorems/atomic_correlation_constant_not_forced.py", "substrate_correlation_proximity_cap_is_contact_local_no_rank"),
    ("B1", "B","superposed D is divergence-correct but curl E is nonzero at (3/5,7/5)", "CERTIFIED-EXACT", "dev", SY + "derivations/bi_two_center_nonsuperposition_witness.py", None),
    ("B2", "B", "the frozen trial lies strictly above the static infimum for every d>0", "CERTIFIED", "dev", SY + "substrate_theorems/rtail_trial_strict_upper_bound.py", None),
    ("B3", "B", "unique minimizer and weak-duality bracket", "DERIVED + LITERATURE (admissible-class commitment)", "dev", SY + "substrate_theorems/rtail_convex_minimizer_uniqueness.py", None),
    ("B4", "B", "order-16 H operator and order-18 W annihilator with exact certificate", "CERTIFIED-EXACT", "dev", SY + "substrate_theorems/rtail_frozen_superposition_ode.py", "frozen_certificate"),
    ("B5", "B", "indicial multisets of the order-18 operator at d=0 and infinity", "CERTIFIED-EXACT (recomputed here)", "dev", FO + "energy-operator-char0.json", None),
    ("B6", "B", "ODE residual test at eight separations, 150 digits", "OBSERVED", "dev", FO + "numeric-verification.json", None),
    ("B7", "B", "exact algebraic-period germ at d=1; displayed values", "DEFINITION + OBSERVED", "dev", FO + "initial-data.json", None),
    ("B8", "B", "continuation to d=1.005 agrees with fresh quadrature", "OBSERVED", "dev", FO + "ivp-replay.json", None),
    ("B9", "B", "frozen contact slope kappa_super by dominated convergence", "DERIVED + OBSERVED value", "dev", FO + "contact_limit.py", None),
    ("B10", "B", "frozen large-d law with C4 = -3K/4 (global proof)", "CERTIFIED", "dev", FO + "global_c4_proof.py", None),
    ("B11", "B", "selected physical W: order >= 4 after finite covers, not algebraic", "CERTIFIED", "dev", FO + "selected-period-bound.json", None),
    ("B12", "B", "order-17 upper-bound annihilator", "SOURCE-CLEAR, NOT LANDED", "ode", FO + "radial-scalar-checks/result.json", None),
    ("B13", "B", "order-17 divisors: degree-39 factor ordinary, degree-35 factor apparent", "AUTHOR + INDEPENDENT CHECK, NOT LANDED", "ode", FO + "radial-divisor-checks/result.json", None),
    ("B14", "B", "independent order-17 divisor and endpoint check", "SOURCE-CLEAR (review)", "c27", "work_journal/order17-divisor-review-c27-open.md", None),
    ("B15", "B", "formal radial cyclic rank 17 (minimal for the formal module)", "AUTHOR-QUALIFIED", "ode", FO + "radial-euler-reduction-spec.md", None),
    ("B16", "B", "series continuation over [1/2,4] agrees with quadrature", "UNREVIEWED (other lane)", "bond", SY + "derivations/diatomic_frozen_pair_series_check.json", None),
    ("B17", "B", "true minimum contact enclosure, F0 = 1/2", "CERTIFIED-INTERVAL", "dev", SY + "derivations/rtail_contact_asymptotic_certificate.json", None),
    ("B18", "B", "true minimum interval at d=1", "CERTIFIED-INTERVAL", "dev", SY + "derivations/rtail_step2_d1_certificate.json", None),
    ("B19", "B", "maximum-potential dual lower bound", "DERIVED (weak duality)", "dev", SY + "derivations/rtail_maximum_potential_lower_bound.md", None),
    ("B20", "B", "dual fence values at sample separations", "UNREVIEWED (other lane)", "bond", SY + "derivations/diatomic_static_pair_fences.json", None),
    ("B21", "B", "true large-d law V = 1/d - C0/d^4 (alpha = C0 exact)", "CONDITIONAL (induced-dipole commitment)", "dev", SY + "substrate_theorems/rtail_large_d_polarizability_exact.py", None),
    ("B22", "B", "first boundary period: Meijer G form and Sp4 monodromy", "CERTIFIED", "dev", FO + "first-boundary-period.json", None),
]

# Formula strings the papers transcribe from the corpus: (ref alias, path, exact substring).
AT = SY + "substrate_theorems/atomic_correlation_constant_not_forced.py"
FF = SY + "derivations/rtail_frozen_superposition_ode_FINDINGS.md"
TRANSCRIPTIONS = [
    ("dev", AT, "(9/32) ln(3/4) - 13/432"),
    ("dev", AT, "5 r_>/8 - (3/8) ln r_> + 3/(16 r_>)"),
    ("dev", AT, "-(45/256)(l+1/2)^-4 + (225/1024)(l+1/2)^-6"),
    ("dev", AT, "-0.125334(1)"),
    ("dev", AT, "J = 5/8 Z"),
    ("dev", AT, "O(alpha^2) ~ 5e-5"),
    ("dev", AT, '"physical_generator_complete": False'),
    ("dev", AT, '"a_corr": None'),
    ("dev", SY + "derivations/acorr_m11_true_minimizer_route_validation.py", "full_elastic_observer_energy is None"),
    ("dev", SY + "derivations/acorr_m11_true_minimizer_route_validation.py", "O(r0^2) ~ 1e-9"),
    ("dev", SY + "derivations/acorr_proximity_cap_rank_analysis.py", "a_corr = sum_k L_k mu_k + R_tail"),
    ("dev", SY + "derivations/acorr_proximity_cap_rank_analysis.py", "200 random instances"),
    ("dev", SY + "derivations/acorr_charge_tail_dependency_FINDINGS.md", "U(s) = 1/(s+r0) * integral_0^1 du/sqrt((1-a*u)^4+(a*u)^4)."),
    ("dev", SY + "derivations/acorr_charge_tail_dependency_FINDINGS.md", "1/s-r0^4/(10*s^5)<=U(s)<=1/s"),
    ("dev", SY + "derivations/bi_test_charge_pair_energy_FINDINGS.md", "V(1,1;0) / U(0) = 4(sqrt(2)-1)/3"),
    ("dev", SY + "derivations/bi_test_charge_pair_energy_FINDINGS.md", "U(0) - V(1,1;0) = (7/2-2sqrt(2)) K > 0"),
    ("dev", SY + "derivations/rtail_known_problems_context.md", "The former claim of a proved order-four non-elementary physical energy is"),
    ("dev", SY + "derivations/README.md", "now PROVEN NON-ELEMENTARY"),
    ("dev", SY + "substrate_theorems/rtail_two_center_synthesis.py", "rtail_two_center_m11_physical_identification"),
    ("acorr28", "work_journal/acorr-and-newton-G-open.md", "Original 2p diagnostic installs 0.04667 Ha"),
    ("dev", FF, "d^16*(d^4-27)*(d^4-27/16)*(d^4+1)*(d^4+256/27)*(d^4+64)"),
    ("dev", FF, "W18 has the exact Euler solutions d and d^3"),
    ("dev", FF, "(3/2)*|b|^2*min(|a|^2,1)"),
    ("dev", FF, "G_{4,4}^{2,4}(64s/27;"),
    ("dev", FF, "upper=(1/2,3/4,1/2,1/4), lower=(0,0;2/3,1/3))"),
    ("dev", FF, "NS=U+D4(-1)^2+A5(-1)"),
    ("dev", FF, "T=U(2)^2+A2(-1)+A1(-1)"),
    ("dev", FF, "H16=Q9*L7"),
    ("dev", SY + "substrate_theorems/rtail_large_d_polarizability_exact.py", "alpha = -B/A = 3(C0 - K) = C0"),
    ("dev", SY + "substrate_theorems/rtail_large_d_polarizability_exact.py", "rtail_large_d_induced_dipole_structure"),
    ("dev", SY + "substrate_theorems/rtail_convex_minimizer_uniqueness.py", "rtail_finite_energy_admissible_class"),
    ("dev", SY + "derivations/rtail_maximum_potential_lower_bound.md", "q * [C0 + I(d/(2*sqrt(q)))] = 2*C0."),
    ("ode", FO + "radial-euler-reduction-spec.md", "F(m)=-(2m+11)(3m+16)(3m+17)(6m+31)^2(6m+35)^2/23328."),
    ("ode", FO + "radial-euler-reduction-spec.md", "(2*s*D_s+1)*A-1/8=q*L16"),
    ("ode", FO + "radial-euler-reduction-spec.md", "lead(N17_W)=s^16*(27s+4)*(64s+1)*(s+1)*(64s-27)*(1024s-27)*Q35."),
    ("ode", FO + "radial-euler-reduction-spec.md", "SOURCE COMPONENT CLEAR C0H0M0L1"),
    ("c27", "work_journal/order17-divisor-review-c27-open.md", "C0/H0/M0/L0 new candidate findings"),
    ("dev", "book/scripts/derive_pair_correlation_constant_forward.py", "pot = -Z / r1 - Z / r2 + 1.0 / r12"),
    ("dev", "book/scripts/derive_pair_correlation_constant_forward.py", "E_corr(Z) = -(Z-5/16)^2 - min E(zeta,c;Z)"),
    ("dev", "book/scripts/derive_pair_correlation_constant_forward.py", "a_generator = 0.04738"),
    ("dev", "book/scripts/derive_pair_correlation_constant_forward.py", "assert abs(a_lim - a_generator) < 0.008"),
    ("dev", "book/scripts/derive_helium_correlation_forward.py", "pot = -Z / r1 - Z / r2 + 1.0 / r12"),
    ("dev", "book/scripts/FORWARD_ELEMENT_GENERATOR.md", "generator / E8 double-leakage / anomalous-moment work"),
    ("dev", "book/scripts/FORWARD_ELEMENT_GENERATOR.md", "TO CONFIRM: derive the 3/4 from the spin algebra"),
    ("dev", "book/scripts/forward_p_row_generator.py", "alpha_sub = ce['lambda_C_e'] / (2 * mp.pi * ce['a_0'])"),
    ("dev", "book/scripts/forward_p_row_generator.py", "return 27.211386245988, 'Hartree (fallback)'"),
    ("dev", "book/scripts/substrate_mass_spectrum.py", "'a_0': 'NOT forward-closed: inherits alpha_g2_forward"),
    ("dev", "book/scripts/substrate_mass_spectrum.py", "default='structural'"),
    ("dev", "book/scripts/forward_generator_complete.py", "LEGACY_DIAGNOSTIC_A_CORR=sp.Rational(4667,100000)"),
    ("dev", "book/scripts/forward_missing_terms_generator.py", "A_CORR = 0.04667"),
    ("dev", "book/scripts/forward_missing_terms_generator.py", "This script DISCOVERS the RATIO pattern (fits to expose the 3:4 lock and the single constant a)."),
    ("dev", "book/scripts/forward_missing_terms_generator.py", "sol, *_ = np.linalg.lstsq(B2, resHa, rcond=None)"),
    ("dev", SY + "derivations/atomic_generator_energy_consistency_FINDINGS.md", "`[0, -21/64, -63/64, -585/256, -921/256, -1341/256]`"),
    ("dev", SY + "derivations/atomic_p_relaxed_energy_FINDINGS.md", "Selecting the common energy fixes that objective"),
    ("ion", "work_journal/ionization-generator-period3-open.md", "SUPERSEDED: \"period 2 complete to <1% with zero fits\"."),
    ("ion", "work_journal/ionization-generator-period3-open.md", "B..Ne at -16.1, -23.4, -28.1, -53.6, -44.5, -37.9 %"),
    ("ion", SY + "derivations/atomic_shell_generator_FINDINGS.md", "the shape does not transfer to the corrected generator."),
]

# Commit-message transcriptions: (exact subject, substring of the commit body).
COMMIT_TRANSCRIPTIONS = [
    ("feat(gr_qm): a_corr is measured, not derivable -- and the corpus disagrees with itself by 2",
     "3/64 and 1/21 are both rejected by the error bar, at +0.45% and +2.0%."),
    ("feat(gr_qm): a_corr is measured, not derivable -- and the corpus disagrees with itself by 2",
     "a_corr is the Z->infinity limit of E_corr(Z) = E_exact - E_HF."),
    ("feat(book): forward-derive the missing-terms constant a = pair-correlation energy",
     "extrapolating to ~0.05 Ha"),
]


def check_commit_transcriptions():
    for subject, text in COMMIT_TRANSCRIPTIONS:
        hashes = [line.split("\t", 1)[0] for line in git("log", "--remotes=origin", "--since=2026-08-01",
                                                         "--format=%H\t%s").splitlines()
                  if line.split("\t", 1)[-1] == subject]
        if not hashes:
            raise SystemExit("commit subject not found: %r" % subject[:50])
        if not any(" ".join(text.split()) in " ".join(git("show", "-s", "--format=%B", h).split()) for h in hashes):
            raise SystemExit("commit body lacks transcribed text: %r" % text[:50])
    return len(COMMIT_TRANSCRIPTIONS)


def check_transcriptions(refs):
    for alias, path, text in TRANSCRIPTIONS:
        # Join wrapped prose: compare whitespace-normalized text.
        if " ".join(text.split()) not in " ".join(blob(refs, alias, path).decode("utf-8").split()):
            raise SystemExit("transcribed string not found in %s: %r" % (path, text[:60]))
    return len(TRANSCRIPTIONS)


EM = chr(0x2014)  # the em dash two historical subjects carry

# Dated history used by the acorr paper: exact commit subjects, looked up in git.
TIMELINE = [
    "feat(book): the COMPLETE forward generator " + EM + " zero fitted constants, 2p row <1%",
    "fix: correct the E_c=a_corr over-claim " + EM + " verified they are different quantities",
    "feat(abi): forward helium correlation -- the r12 term closes the mean-field residual to +0.42%",
    "feat(book): forward-derive the missing-terms constant a = pair-correlation energy",
    "feat(gr_qm): a_corr is measured, not derivable -- and the corpus disagrees with itself by 2",
    "fix(gr-qm): reconcile conditional atomic energy and ionization",
    "fix(gr-qm): consume exact relaxed occupation certificates",
    "feat(gr-qm): eps_2(0) does not close by the Green's-function assembly -- the crossover correction is non-elementary",
    "research(gr-qm): prove the rigid-limit source non-elementary; the substrate answer is the deformed metric",
    "research(gr-qm): the substrate a_corr is a global field energy; the saturation root resums globally",
    "research(gr-qm): name the two-center Born-Infeld field energy as the open a_corr residual",
    "research(sympy): land a_corr coincidence match derivations and two open verdicts",
    "research(sympy): land proximity-cap contact-local no-rank verdict for a_corr",
    "research: land two-center Born-Infeld field energy R_tail closed-form bracket",
    "research: prove the two-center Born-Infeld field energy R_tail is non-elementary",
    "research: downgrade R_tail two-center BI structure to a rank-6 upper bound with open obligations",
    "fix(physics): derive R_tail trial bounds and correct correlation claims",
    "feat(gr-qm): two-center Born-Infeld minimizer is unique and two-sided bracketed",
    "docs(gr-qm): label the lambda-reduction as the frozen superposition energy, not the true R_tail",
    "feat(gr-qm): validate the M11 true-minimizer route to a_corr - OPEN, blocking antecedent named",
    "docs: record acorr derivation history and remaining antecedents",
    "feat(gr-qm): establish selected physical energy order lower bound",
    "feat: derive order17 frozen energy annihilator",
    "feat: classify order17 apparent divisors",
    "test: independently check order17 divisor and endpoint data",
]


def tex_escape(s):
    return (s.replace("\\", "/").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
            .replace("#", "\\#").replace("^", "\\^{}").replace("<", "$<$").replace(">", "$>$")
            .replace("--", "-{}-"))


def claim_rows(refs, paper):
    tex, md = [], []
    for cid, p, stmt, status, alias, path, func in CLAIMS:
        if p != paper:
            continue
        h, date = last_commit(refs, alias, path)
        if func is not None:
            text = blob(refs, alias, path).decode("utf-8")
            if not re.search(r"^def %s\(" % re.escape(func), text, re.M):
                raise SystemExit("function %s not found in %s" % (func, path))
        on_dev = is_ancestor(h, refs["dev"]["commit"])
        loc = path.replace(SY, "") + ("::" + func if func else "")
        tex.append("\\cell{0.04}{%s} & \\cell{0.33}{%s} & \\cell{0.17}{%s} & \\cell{0.29}{\\path{%s}} & "
                   "\\cell{0.1}{\\texttt{%s}%s} \\\\" % (
                       cid, tex_escape(stmt), tex_escape(status), loc, h[:10],
                       "" if on_dev else "$^{\\dagger}$"))
        md.append("| %s | %s | %s | `%s` | `%s` (%s) | %s |" % (
            cid, stmt, status, loc, h[:10], date[:10], "dev" if on_dev else REF_NAMES[alias]))
    # The closing rule lives here: a \bottomrule after a non-expandable \input is misplaced.
    return "\n".join(tex) + "\n\\bottomrule\n", "\n".join(md) + "\n"


def timeline_rows(refs):
    log = git("log", "--remotes=origin", "--since=2026-08-01", "--date=format-local:%Y-%m-%dT%H:%MZ",
              "--format=%H%x09%cd%x09%s")
    by_subject = {}
    for line in log.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            by_subject.setdefault(parts[2], []).append((parts[0], parts[1]))
    rows = []
    for subject in TIMELINE:
        hits = by_subject.get(subject)
        if not hits:
            raise SystemExit("timeline subject not found: %r" % subject[:60])
        on_dev = [h for h in hits if is_ancestor(h[0], refs["dev"]["commit"])]
        h, date = sorted(on_dev or hits, key=lambda x: x[1])[0]
        utc = git("log", "-1", "--date=format-local:%Y-%m-%d %H:%MZ", "--format=%ad", h).strip()
        safe = subject.replace(EM, ";")
        rows.append((utc, "\\cell{0.17}{%s} & \\cell{0.14}{\\texttt{%s}%s} & \\cell{0.62}{%s} \\\\" % (
            utc, h[:10], "" if on_dev else "$^{\\dagger}$", tex_escape(safe))))
    return "\n".join(r for _, r in sorted(rows)) + "\n\\bottomrule\n"


def numbers_tex(n, refs):
    lines = ["% Generated by papers/two_center_born_infeld/generate_numbers.py; do not edit."]
    for alias in sorted(refs):
        lines.append("%% ref %s = %s at %s" % (alias, refs[alias]["name"], refs[alias]["commit"]))
    for alias in sorted(refs):
        name = "Ref" + alias.capitalize().replace("2", "Two").replace("8", "Eight").replace("7", "Seven")
        name = re.sub(r"[^A-Za-z]", "", name)
        lines.append("\\newcommand{\\%s}{\\texttt{%s}}" % (name, refs[alias]["commit"][:12]))
    for k in sorted(n):
        if not re.fullmatch(r"[A-Za-z]+", k):
            raise SystemExit("bad macro name " + k)
        lines.append("\\newcommand{\\%s}{%s}" % (k, n[k]))
    return "\n".join(lines) + "\n"


MARK_BEGIN = "<!-- BEGIN GENERATED CLAIMS -->"
MARK_END = "<!-- END GENERATED CLAIMS -->"


def packet_with(path, table_md):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    if MARK_BEGIN not in text or MARK_END not in text:
        raise SystemExit("markers missing in " + path)
    head, rest = text.split(MARK_BEGIN, 1)
    _, tail = rest.split(MARK_END, 1)
    header = ("| id | claim | status | source (file::function) | commit (date) | ref |\n"
              "| --- | --- | --- | --- | --- | --- |\n")
    return head + MARK_BEGIN + "\n" + header + table_md + MARK_END + tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    refs = resolve_refs(args.refresh)
    n = compute_numbers(refs)
    n["TranscriptionCountVal"] = str(check_transcriptions(refs) + check_commit_transcriptions())
    num = numbers_tex(n, refs)
    a_tex, a_md = claim_rows(refs, "A")
    b_tex, b_md = claim_rows(refs, "B")
    outputs = {
        os.path.join(ACORR_DIR, "numbers.tex"): num,
        os.path.join(TWO_DIR, "numbers.tex"): num,
        os.path.join(ACORR_DIR, "claims_table.tex"): a_tex,
        os.path.join(TWO_DIR, "claims_table.tex"): b_tex,
        os.path.join(ACORR_DIR, "timeline_table.tex"): timeline_rows(refs),
        os.path.join(ACORR_DIR, "REVIEW_PACKET.md"): packet_with(os.path.join(ACORR_DIR, "REVIEW_PACKET.md"), a_md),
        os.path.join(TWO_DIR, "REVIEW_PACKET.md"): packet_with(os.path.join(TWO_DIR, "REVIEW_PACKET.md"), b_md),
        REFS_FILE: json.dumps(refs, indent=1, sort_keys=True) + "\n",
    }
    drift = []
    for path, text in sorted(outputs.items()):
        old = open(path, encoding="utf-8").read() if os.path.exists(path) else None
        if old != text:
            drift.append(os.path.relpath(path, PAPERS))
            if not args.check:
                with open(path, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(text)
    if args.check:
        print("CHECK %s: %d drifted file(s) %s" % ("FAIL" if drift else "PASS", len(drift), ",".join(drift)))
        return 1 if drift else 0
    print("WROTE %d file(s): %s" % (len(drift), ",".join(drift)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
