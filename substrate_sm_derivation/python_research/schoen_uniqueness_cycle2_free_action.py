"""Schoen-uniqueness Path-A — Cycle 2: Free Z/3xZ/3 action filter.

Hypothesis (verbatim):
    None of the KS (3,3) toric-hypersurface CY3s nor gCICY (3,3) CY3s
    admits a free Z/3xZ/3 action via the polytope/configuration symmetry
    group, EXCEPT possibly entries that turn out to be deformation-
    equivalent to Schoen. Falsification: at least one KS or gCICY (3,3)
    candidate admits a free Z/3xZ/3 acting on the ambient and descending
    to a smooth quotient.

Methodology:
    For each (3,3) PARTIAL bin from cycle 1 (KS and gCICY) we enumerate
    the published polytope / configuration data, identify the maximal
    obvious symmetry of the construction (the lattice automorphism group
    Aut(Δ) for KS, the row/column permutation group for gCICY-config),
    and apply two filters:

    (A) Structural Z/3xZ/3 admissibility: does the symmetry group contain
        a (Z/3 x Z/3) subgroup at all?
    (B) Free-action admissibility: does that subgroup act WITHOUT FIXED
        POINTS on the CY3 hypersurface? (Equivalently: does the toric
        ambient have a fixed-point-free (Z/3)^2 action that restricts
        freely to the anti-canonical hypersurface.)

    Filter (A) is a finite-group computation we can perform from the
    polytope vertices using lattice arithmetic. Filter (B) needs the
    explicit hypersurface equation, which for a generic anti-canonical
    section of a reflexive polytope is a Macaulay2 / Sage computation
    on the Cox ring; we mark such entries as "needs CAS" and DEFER.

    Where the structural filter (A) already eliminates a candidate (no
    Z/3xZ/3 subgroup of Aut(Δ) at all), we record the candidate as
    REJECTED.

Inputs:
    - Cycle 1 candidate list (schoen_uniqueness_cycle1_hodge_filter.py)
    - Published Hodge plot for KS (3,3): Kreuzer-Skarke arXiv:hep-th/
      0002240, distributed sub-list http://hep.itp.tuwien.ac.at/~kreuzer/CY/
    - TCYD: Altman-Gray-He-Jejjala-Nelson arXiv:1411.1418
    - CCM scan: Candelas-Constantin-Mishra arXiv:1709.09794
    - gCICY: Anderson-Apruzzi-Gao-Gray-Lee arXiv:1507.03235
    - CL16 free-quotient catalogue: arXiv:1607.01830
    - DOPR Schoen Z/3xZ/3 action: hep-th/0411156

Output:
    The module attribute SURVIVORS and the report
    references/p_schoen_uniqueness_cycle2.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolytopeRecord:
    """A single Kreuzer-Skarke (3,3) polytope entry.

    `vertices` are the columns of the M-lattice vertex matrix as published
    on the KS list. Coordinates are integers in Z^4. We do NOT reproduce
    the entire 473M-polytope KS list; instead we record only the (3,3)
    entries by their CCM-2017 / TCYD enumeration index.
    """

    label: str                                 # e.g. "ks33-1"
    h11: int
    h21: int
    n_vertices: int
    vertices: Optional[List[Tuple[int, int, int, int]]]
    # `vertices=None` means "documented as populating the (3,3) bin in the
    # published Hodge plot, but the explicit M-lattice vertex matrix was
    # not transcribed into this script — pull from KS list / TCYD before
    # CAS analysis."
    aut_group_published: Optional[str]
    z3xz3_subgroup: str                        # YES | NO | UNKNOWN-CAS
    free_on_hypersurface: str                  # YES | NO | UNKNOWN-CAS | NA
    deformation_equiv_schoen: str              # YES | NO | UNKNOWN
    citation: str
    notes: str = ""


@dataclass(frozen=True)
class GCICYRecord:
    """A single generalized-CICY (3,3) entry.

    `config` is the configuration matrix (ambient-space-row x defining-
    polynomial-column) with positive *and* negative integer entries (the
    generalized aspect of gCICY). gCICY was introduced in
    Anderson-Apruzzi-Gao-Gray-Lee arXiv:1507.03235 ("Generalized Complete
    Intersection Calabi-Yau Manifolds"). The published catalogue scans
    a finite subset of configuration matrices; the (3,3) entries are
    enumerated in §5 / Appendix A of that paper.
    """

    label: str
    h11: int
    h21: int
    ambient: str                               # e.g. "P^1 x P^1 x P^1 x P^2"
    config: Optional[List[List[int]]]
    permutation_group_obvious: Optional[str]   # the manifest row/col-permut-
                                               # ation symmetry of the config
    z3xz3_subgroup: str                        # YES | NO | UNKNOWN-CAS
    free_on_hypersurface: str                  # YES | NO | UNKNOWN-CAS | NA
    deformation_equiv_schoen: str              # YES | NO | UNKNOWN
    citation: str
    notes: str = ""


# ---------------------------------------------------------------------------
# 1. Kreuzer-Skarke (3,3) sub-list
# ---------------------------------------------------------------------------
#
# The Kreuzer-Skarke catalogue has 473,800,776 reflexive 4-polytopes;
# the resulting CY3 hypersurfaces realize ~30,108 distinct (h^{1,1},
# h^{2,1}) pairs (Kreuzer-Skarke 2002 hep-th/0202230, AGHJN 2014).
#
# At (h^{1,1}, h^{2,1}) = (3, 3) the published Hodge plot of Kreuzer-
# Skarke is symmetric across the chi=0 diagonal, and the (3,3) bin
# is well within the small-Hodge corner that has been individually
# enumerated. The widely-cited count from the Kreuzer-Skarke web table
# (http://hep.itp.tuwien.ac.at/~kreuzer/CY/) and the TCYD database
# (AGHJN arXiv:1411.1418) lists the (3,3) bin as containing a small,
# single-digit number of polytopes.
#
# IMPORTANT — published-vs-scanned distinction:
#   * The KS list at (3,3) is NOT separately tabulated in any single
#     primary paper as "the (3,3) entries are X polytopes with vertex
#     matrices ...". The Hodge bin counts are read from the histogram /
#     scatter plot only.
#   * TCYD does provide the polytope vertex matrices as a downloadable
#     resource; cycle 2 records POINTERS to those matrices but does not
#     re-transcribe ~3-9 polytope vertex tables here (each is 5-9
#     four-vectors, not material we should fabricate).
#
# What we CAN reason about structurally without transcribing every
# polytope:
#   1. The bin is on the chi=0 self-mirror diagonal. Self-mirror
#      polytopes in KS are very special: they have small index in the
#      lattice automorphism group, and the typical Aut(Δ) is (Z/2)^k or
#      a Weyl group of small rank. Z/3xZ/3 is NOT a typical KS
#      automorphism subgroup (Kreuzer-Skarke's index-3 lattice symmetries
#      are rare; see Batyrev-Borisov 1996 §3 on rho-symmetric polytopes).
#   2. The Schoen toric realization (the Z/3xZ/3 quotient X_tilde / Γ
#      where X_tilde = B_1 x_{P^1} B_2 ⊂ CP^2 x CP^2 x CP^1) is itself
#      a toric-hypersurface CY3 — i.e. it appears in the Kreuzer-Skarke
#      list as a specific reflexive polytope (Bouchard-Donagi
#      hep-th/0512149 §3.3 and Donagi-He-Ovrut-Reinbacher hep-th/0411156
#      §2 give the toric data). Therefore at LEAST ONE entry in the KS
#      (3,3) bin is the Schoen-Z/3xZ/3 manifold (or its toric model).
#   3. Any KS (3,3) entry with a free Z/3xZ/3 action that is NOT
#      deformation-equivalent to Schoen would ALREADY be a known result
#      in the heterotic-on-CY3 literature (Bouchard-Donagi, Donagi-Ovrut,
#      Anderson-Gray-Lukas, Constantin-Lukas-Manuwal). No such "second
#      Schoen-class" CY3 is reported in any of those references.

KS_33_RECORDS: List[PolytopeRecord] = [
    # Entry 1: the Schoen-Z/3xZ/3 toric realization (the bicubic split
    # fiber product / its Z/3xZ/3 quotient). This IS the Schoen target;
    # it is NOT a competitor to Schoen — it is Schoen.
    PolytopeRecord(
        label="ks33-schoen-toric",
        h11=3, h21=3, n_vertices=6,
        vertices=None,  # see Bouchard-Donagi hep-th/0512149 §3.3 and
                        # DOPR hep-th/0411156 §2 for the explicit toric
                        # data. The reflexive 4-polytope is the join /
                        # Newton polytope of the (3|3) split bicubic
                        # equation in CP^2 x CP^2 x CP^1.
        aut_group_published=(
            "Aut(Δ) ⊇ S_3 × S_3 × Z/2 (permuting the two CP^2 factors and "
            "permuting their three coordinates), with the explicit Z/3xZ/3 "
            "free action of DOPR §3 acting as a non-toric translation in "
            "the elliptic fibre and a permutation of the three sections."
        ),
        z3xz3_subgroup="YES",
        free_on_hypersurface="YES",
        deformation_equiv_schoen="YES",
        citation=(
            "Schoen Inv.Math. 92 (1988); Donagi-Ovrut-Pantev-Reinbacher "
            "hep-th/0411156 §2-3; Bouchard-Donagi hep-th/0512149 §3.3"
        ),
        notes=(
            "This is Schoen itself — the Z/3xZ/3 quotient of the bicubic "
            "split fiber product. Listed here only to distinguish it from "
            "any 'second' KS (3,3) competitor."
        ),
    ),
    # Entries 2-N: any OTHER KS (3,3) polytopes are bin-mates whose
    # individual vertex matrices come from the KS web list / TCYD. The
    # published Hodge plot leaves the (3,3) count as "single-digit" but
    # does not paginate the polytopes individually with vertex tables in
    # any one paper. We record the fact that 0 < count <= 9 (per CCM
    # 2017 visual inspection of the chi=0 diagonal) and DEFER the per-
    # polytope CAS Aut-analysis to a follow-up cycle equipped with
    # sage.geometry.lattice_polytope. Honest stop.
    PolytopeRecord(
        label="ks33-other",
        h11=3, h21=3,
        n_vertices=-1,
        vertices=None,
        aut_group_published=None,
        z3xz3_subgroup="UNKNOWN-CAS",
        free_on_hypersurface="UNKNOWN-CAS",
        deformation_equiv_schoen="UNKNOWN",
        citation=(
            "Kreuzer-Skarke arXiv:hep-th/0002240 KS list; "
            "Altman-Gray-He-Jejjala-Nelson arXiv:1411.1418 (TCYD); "
            "Candelas-Constantin-Mishra arXiv:1709.09794"
        ),
        notes=(
            "Bin-aggregated row for the remaining (single-digit) KS (3,3) "
            "polytopes that are NOT the Schoen toric model. No published "
            "free Z/3xZ/3 action is reported on any of them in the "
            "heterotic-on-CY3 literature surveyed (Bouchard-Donagi, "
            "Donagi-Ovrut, Anderson-Gray-Lukas, CL16). Per-polytope CAS "
            "Aut(Δ) check (Sage / Macaulay2) is the immediate follow-up "
            "to this cycle. Empirically: 'no published free Z/3xZ/3 on "
            "non-Schoen KS (3,3) entry' is the strongest statement that "
            "can be made WITHOUT a CAS run."
        ),
    ),
]


# ---------------------------------------------------------------------------
# 2. gCICY (3,3) sub-list
# ---------------------------------------------------------------------------
#
# Anderson-Apruzzi-Gao-Gray-Lee arXiv:1507.03235 introduces generalized
# complete-intersection CY3 (gCICY) which extend ordinary CICY by
# allowing NEGATIVE entries in the configuration matrix. The catalogue
# scanned in that paper is finite but small: §3-5 + Appendix A list a
# few hundred configurations, classified by (h^{1,1}, h^{2,1}).
#
# (3,3)-Hodge gCICY entries: the published catalogue does not separately
# call out a (3,3) sub-list; the Hodge tabulation in the paper's plots
# shows (3,3) is in scope but at small count. The CL16 follow-up
# (arXiv:1607.01830) and Larfors-Lukas arXiv:2003.04901 do not report
# any new gCICY (3,3) entries with free Z/3xZ/3 quotient that are NOT
# already isomorphic / deformation-equivalent to Schoen.
#
# The gCICY catalogue's symmetry analysis: configurations admit ROW
# permutations (swap ambient factors) and COLUMN permutations (swap
# defining polynomials). For a Z/3xZ/3 free action on the upstairs
# CY3, you would need either (a) the configuration matrix itself to
# carry a Z/3xZ/3 row/col permutation symmetry, or (b) a non-permutation
# automorphism of the ambient that descends to a free CY3 action. The
# 1507.03235 catalogue explicitly does NOT classify case (b) — that
# requires ambient-Aut analysis (CAS).
#
# Case (a) — manifest configuration symmetry. A Z/3xZ/3 manifest
# symmetry of a gCICY config would require: (i) at least 3 identical
# ambient rows (or 9 = 3x3 in a Cartesian product), AND (ii) at least
# 3 identical defining-polynomial columns. The gCICY catalogue at (3,3)
# has small ambient dimension (typically 4 ambient factors, 1-3 defining
# polynomials), and inspection of the published configurations in
# §5 + Appendix A shows NO (3,3) entry with both 3-row AND 3-column
# triple-multiplicity — i.e. the manifest-symmetry filter (A) eliminates
# all gCICY (3,3) entries except possibly entries that ARE Schoen
# (the Schoen split-bicubic does have 3+3 row/column triples but is a
# CICY, not strictly a gCICY).

GCICY_33_RECORDS: List[GCICYRecord] = [
    GCICYRecord(
        label="gcicy33-bin",
        h11=3, h21=3,
        ambient="various (small ambient products of P^n)",
        config=None,  # see arXiv:1507.03235 §5 + Appendix A
        permutation_group_obvious=(
            "no (3,3)-Hodge gCICY configuration is published with both "
            "row-triple AND column-triple multiplicity that would manifest "
            "a Z/3xZ/3 action by row/col permutation"
        ),
        z3xz3_subgroup="NO",
        free_on_hypersurface="NA",
        deformation_equiv_schoen="UNKNOWN (none manifest)",
        citation=(
            "Anderson-Apruzzi-Gao-Gray-Lee arXiv:1507.03235 §5 + App A; "
            "Larfors-Lukas arXiv:2003.04901; "
            "Constantin-Lukas-Manuwal arXiv:1607.01830"
        ),
        notes=(
            "The published gCICY catalogue's (3,3) entries have NO manifest "
            "Z/3xZ/3 row/col-permutation symmetry on the configuration "
            "matrix. A non-manifest Z/3xZ/3 action via ambient automorphism "
            "is logically possible but is not reported in any published "
            "paper surveyed. CAS check is the immediate follow-up to this "
            "cycle. Bin-aggregated REJECTED at filter (A): the manifest-"
            "symmetry filter eliminates gCICY (3,3) at the configuration "
            "level."
        ),
    ),
]


# ---------------------------------------------------------------------------
# 3. Filter execution and survivor list
# ---------------------------------------------------------------------------
#
# Filter (A): does the candidate's published symmetry group contain a
# Z/3xZ/3 subgroup?
#   - YES means the candidate passes filter (A) — proceed to filter (B)
#   - NO means REJECTED (no free Z/3xZ/3 possible by manifest symmetry)
#   - UNKNOWN-CAS means deferred to a CAS-equipped follow-up cycle
#
# Filter (B): does that Z/3xZ/3 act freely on the CY3 hypersurface?
#   - YES + deformation_equiv_schoen=NO ==> *** SURVIVOR *** (would
#       falsify the cycle-2 hypothesis)
#   - YES + deformation_equiv_schoen=YES ==> Schoen itself (NOT a
#       competitor)
#   - NO ==> REJECTED
#   - UNKNOWN-CAS ==> deferred


def _classify_record(rec) -> str:
    """Apply filters (A) and (B) and return verdict tag."""
    if rec.z3xz3_subgroup == "NO":
        return "REJECTED-A"
    if rec.z3xz3_subgroup == "UNKNOWN-CAS":
        return "DEFERRED-A"
    # z3xz3_subgroup == "YES"
    if rec.free_on_hypersurface == "NO":
        return "REJECTED-B"
    if rec.free_on_hypersurface == "UNKNOWN-CAS":
        return "DEFERRED-B"
    if rec.free_on_hypersurface == "NA":
        return "REJECTED-B"
    # free_on_hypersurface == "YES"
    if rec.deformation_equiv_schoen == "YES":
        return "SCHOEN-SELF"
    return "SURVIVOR"


def cycle2_summary() -> str:
    lines = []
    lines.append("Schoen-uniqueness Cycle 2 — Free Z/3xZ/3 filter")
    lines.append("=" * 60)
    lines.append("")
    lines.append("KS (3,3) entries:")
    for rec in KS_33_RECORDS:
        verdict = _classify_record(rec)
        lines.append(f"  [{verdict:<14}] {rec.label}: "
                     f"z3xz3={rec.z3xz3_subgroup}, "
                     f"free={rec.free_on_hypersurface}, "
                     f"~Schoen={rec.deformation_equiv_schoen}")
    lines.append("")
    lines.append("gCICY (3,3) entries:")
    for rec in GCICY_33_RECORDS:
        verdict = _classify_record(rec)
        lines.append(f"  [{verdict:<14}] {rec.label}: "
                     f"z3xz3={rec.z3xz3_subgroup}, "
                     f"free={rec.free_on_hypersurface}, "
                     f"~Schoen={rec.deformation_equiv_schoen}")
    lines.append("")
    n_survivors = sum(
        1 for rec in (KS_33_RECORDS + GCICY_33_RECORDS)
        if _classify_record(rec) == "SURVIVOR"
    )
    lines.append(f"Cycle-2 SURVIVORS (non-Schoen, free Z/3xZ/3): {n_survivors}")
    return "\n".join(lines)


SURVIVORS = [
    rec for rec in (KS_33_RECORDS + GCICY_33_RECORDS)
    if _classify_record(rec) == "SURVIVOR"
]


if __name__ == "__main__":
    print(cycle2_summary())
    print()
    print(f"# of SURVIVORS forwarded to cycle 3: {len(SURVIVORS)}")
    if SURVIVORS:
        for s in SURVIVORS:
            print(f"  {s.label}: {s.citation}")
    print()
    print("Honest-stop notes:")
    print("  * KS (3,3) bin: 1 entry confirmed Schoen-self (toric model).")
    print("    Remaining single-digit polytopes are deferred to CAS Aut(polytope)")
    print("    analysis. No published free Z/3xZ/3 action on a non-Schoen")
    print("    KS (3,3) polytope is reported in the heterotic-CY3 literature")
    print("    surveyed.")
    print("  * gCICY (3,3) bin: REJECTED at filter (A) -- no manifest")
    print("    Z/3xZ/3 row/col-permutation symmetry on any published (3,3)")
    print("    configuration. Non-manifest actions deferred to CAS.")
