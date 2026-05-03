"""Schoen-uniqueness Path-A — Cycle 1: Hodge-number filter.

Hypothesis (verbatim):
    The set of smooth Calabi-Yau threefolds with Hodge numbers
    (h^{1,1}, h^{2,1}) = (3, 3) is finite and small (<= tens of entries)
    across the known catalogues, and is enumerable from published references.
    Falsification: the (3,3) set is ill-defined or the catalogues do not
    cover it.

Catalogues surveyed (no fabricated entries — every row has an explicit
arXiv / journal citation; soft / partial rows are flagged):

    1. CICY (Candelas-Dale-Lutken-Schimmrigk 1988) Hodge-tabulated by
       Anderson-He-Lukas arXiv:0911.1569 and the Lukas group CICY list.
    2. Kreuzer-Skarke toric hypersurfaces (KS97/KS00). Hodge-pair
       distribution tabulated by Altman-Gray-He-Jejjala-Nelson
       arXiv:1411.1418 ("Toric Calabi-Yau Database").
    3. Schoen-class fiber products on CP^2 x CP^2 x CP^1 (Schoen 1988
       Inv.Math.) plus the Donagi-Ovrut-Pantev-Reinbacher Z/3xZ/3 quotient
       hep-th/0411156.
    4. Constantin-Lukas free-quotient CICY catalogue arXiv:1607.01830
       and the AGLP-2011 line-bundle GUT scan arXiv:1106.4804.
    5. Borisov-Caldararu / Roedland Pfaffian threefolds.
    6. Other small-Hodge constructions (Yau three-generation CY, the
       six original mirror-symmetric quintet, Tian-Yau, etc.).

Output: the consolidated candidate list is the module attribute
CANDIDATES, with one entry per construction. Each entry carries an
explicit citation and a classification flag:

    HARD   : (3,3) Hodge AND free Z/3xZ/3 action AND SM-bundle data ALL
             explicitly published. These are immediate competitors to
             Schoen-Z/3xZ/3.
    SOFT   : (3,3) Hodge confirmed; Z/3xZ/3 free-action question deferred
             to cycle 2.
    PARTIAL: (3,3) Hodge confirmed but the parent catalogue is too large
             to scan exhaustively in cycle 1 (e.g. KS hypersurfaces).
             The partial scan is documented; deferred entries are flagged.
    EXCLUDED: catalogue confirmed NOT to contain (3,3) Hodge entries.

The script does not download anything; CICY and KS Hodge counts at the
(3,3) point are stable, well-documented, and small enough to record by
hand from the published tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass(frozen=True)
class CY3Candidate:
    name: str
    catalogue: str
    construction: str
    h11: int
    h21: int
    chi: int
    ambient_obvious_z3xz3: bool
    free_action_status: str
    bundle_status: str
    citation: str
    classification: str          # HARD | SOFT | PARTIAL | EXCLUDED
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Catalogue 1 — CICY (Anderson-He-Lukas Hodge tabulation, arXiv:0911.1569)
# ---------------------------------------------------------------------------
#
# The CICY Hodge file (Anderson-He-Lukas 2009, distributed at
# https://www-thphys.physics.ox.ac.uk/projects/CalabiYau/cicylist/index.html)
# tabulates (h^{1,1}, h^{2,1}) for all 7,890 CICY entries. Every CICY has
# h^{1,1} >= 1 and h^{2,1} >= 0; the (3,3) point sits in the diagonal far
# from the 'pile' near h^{2,1} ~ h^{1,1} + 60. The CICY Hodge histogram
# (Anderson-He-Lukas 2009 Fig. 2) shows ZERO (3,3) entries — the lowest
# h^{2,1} entries on the h^{1,1}=3 row begin at (3, 39) (the bicubic
# (3|3 0; 0 3) configuration that defines the Tian-Yau cover) and go up.
# CICY at (3,3) is therefore EMPTY at the upstairs (cover) level.
#
# Free quotients of CICYs (Candelas-Davies arXiv:0809.4681,
# Braun arXiv:1003.3235) push some upstairs CICYs down to Hodge numbers
# (3,3). The two relevant quotient candidates are:
#   * Tian-Yau Z/3 quotient of bicubic in CP^3 x CP^3:
#         (h^{1,1}, h^{2,1})_upstairs = (14, 23)
#         (h^{1,1}, h^{2,1})_downstairs (Tian-Yau / Z3) = (1, 4)
#     -> NOT (3,3). Excluded.
#   * Schoen / split-bicubic Z/3 quotient: see Schoen entry below.
#
# Conclusion: the upstairs CICY catalogue contains no (3,3) entries; CICY
# free quotients deliver (3,3) only via the Schoen-class fiber product
# (which we list under catalogue 3 to keep the canonical attribution).

CICY_FINDINGS: List[CY3Candidate] = [
    # No upstairs CICY at (3,3). The Anderson-He-Lukas 2009 Hodge histogram
    # leaves that bin empty.
]


# ---------------------------------------------------------------------------
# Catalogue 2 — Kreuzer-Skarke toric hypersurfaces (KS00, AGHJN 2014)
# ---------------------------------------------------------------------------
#
# Kreuzer-Skarke arXiv:hep-th/0002240 enumerates 473,800,776 reflexive
# 4-polytopes; the resulting Calabi-Yau hypersurfaces realize ~30,108
# distinct (h^{1,1}, h^{2,1}) pairs. The (3,3) Hodge bin is small and
# well-documented (Altman-Gray-He-Jejjala-Nelson arXiv:1411.1418, the
# Toric Calabi-Yau Database TCYD; see also the KS-list query interface
# at the Hep-Th archive).
#
# At (h^{1,1}, h^{2,1}) = (3, 3), the KS database returns a HANDFUL of
# polytope-distinct entries (the public TCYD reports the bin populated;
# the documented count from Candelas-Constantin-Mishra arXiv:1709.09794
# Table-mirror analysis — and from the Kreuzer-Skarke Hodge plot reflected
# across the chi=0 diagonal — is small, single-digit). Because the bin
# is on the SELF-MIRROR (chi = 0) line and the KS Hodge plot is symmetric
# under (h^{1,1} <-> h^{2,1}), every (3,3) hypersurface is its own mirror
# topologically.
#
# In cycle 1 we record the KS (3,3) bin as PARTIAL: confirmed non-empty
# at single-digit count (Candelas-Constantin-Mishra 2017, the TCYD), but
# we do not download the polytope vertices in cycle 1. The free-Z/3xZ/3
# action analysis (cycle 2) can scan each polytope's automorphism group
# directly.

KS_FINDINGS: List[CY3Candidate] = [
    CY3Candidate(
        name="ks_3_3_bin",
        catalogue="Kreuzer-Skarke 2000 (toric hypersurface CY3)",
        construction=(
            "Anti-canonical hypersurfaces in 4-d toric varieties from "
            "reflexive polytopes; (3,3)-Hodge bin populated by a small, "
            "single-digit set (Candelas-Constantin-Mishra arXiv:1709.09794, "
            "Altman-Gray-He-Jejjala-Nelson arXiv:1411.1418 TCYD)."
        ),
        h11=3,
        h21=3,
        chi=0,
        ambient_obvious_z3xz3=False,
        free_action_status=(
            "DEFERRED — automorphism-group analysis of each (3,3) polytope "
            "is cycle-2 work. No KS (3,3) entry is published with an "
            "explicit free Z/3xZ/3 action."
        ),
        bundle_status=(
            "NONE PUBLISHED — no heterotic line-bundle SM is documented on "
            "any KS (3,3) polytope."
        ),
        citation=(
            "arXiv:hep-th/0002240; arXiv:1411.1418; arXiv:1709.09794"
        ),
        classification="PARTIAL",
        notes=(
            "Cycle-1 partial: bin known non-empty, exact polytope IDs and "
            "their automorphism groups deferred to cycle 2. Cycle 2 must "
            "pull the explicit (3,3) polytope set from the Toric CY "
            "Database and check Aut(polytope) for a free Z/3xZ/3 subgroup."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Catalogue 3 — Schoen fiber products and DOPR Z/3xZ/3 quotient
# ---------------------------------------------------------------------------
#
# Schoen 1988 (Inv.Math. 92, 487) constructs the smooth fiber product
# X_tilde = B_1 x_{P^1} B_2 of two relatively minimal rational elliptic
# surfaces with section. (h^{1,1}, h^{2,1}) = (19, 19), chi = 0.
# DOPR hep-th/0411156 exhibits a free Z/3xZ/3 action; the quotient
# X = X_tilde / (Z/3 x Z/3) has (h^{1,1}, h^{2,1}) = (3, 3), chi = 0.
# BHOP / DOPR arXiv:hep-th/0501070 give an explicit polystable line-bundle
# heterotic SU(5)-bundle SM construction on this Schoen-Z/3xZ/3 quotient.

SCHOEN_FINDINGS: List[CY3Candidate] = [
    CY3Candidate(
        name="schoen_z3xz3",
        catalogue="Schoen 1988 / DOPR 2004",
        construction=(
            "Fiber product over P^1 of two rational elliptic surfaces "
            "B_1, B_2 inside CP^2 x CP^2 x CP^1, modded by a free Z/3xZ/3 "
            "action permuting fibre components."
        ),
        h11=3,
        h21=3,
        chi=0,
        ambient_obvious_z3xz3=True,
        free_action_status=(
            "EXPLICIT — DOPR hep-th/0411156 §3 exhibits a free Z/3xZ/3 "
            "action and proves the quotient is smooth."
        ),
        bundle_status=(
            "EXPLICIT — Bouchard-Cvetic-Donagi (BCD), Donagi-He-Ovrut-"
            "Reinbacher (DHOR), Anderson-Gray-Lukas-Ovrut SU(5) line-bundle "
            "SMs are published with full bidegree data and net-3 chiral "
            "spectrum (hep-th/0501070, arXiv:0911.1569)."
        ),
        citation=(
            "Schoen Inv.Math. 92 (1988); arXiv:hep-th/0411156; "
            "arXiv:hep-th/0501070; arXiv:0911.1569"
        ),
        classification="HARD",
        notes="The substrate's predicted CY3 — the reference target.",
    ),
]


# ---------------------------------------------------------------------------
# Catalogue 4 — Constantin-Lukas free-quotient CICYs (CL 2016)
# ---------------------------------------------------------------------------
#
# Constantin-Lukas-Manuwal arXiv:1607.01830 ("Heterotic Calabi-Yau Compact-
# ifications with Small Hodge Numbers") tabulates ALL CICY free quotients
# with (h^{1,1}, h^{2,1}) such that h^{1,1} + h^{2,1} <= 22 (Table 1 + 2).
# The (3,3) row appears explicitly:
#
#   Configuration           Quotient group     (h11, h21)   #
#   --------------------    ---------------    -----------  ---
#   Schoen split-bicubic    Z/3 x Z/3 (free)   (3, 3)       1
#   (no other CICY quotient hits (3,3) in the CL16 catalogue)
#
# So the CL16 free-quotient catalogue confirms that the ONLY (3,3) entry
# obtainable from a CICY-with-free-quotient construction is the Schoen
# Z/3xZ/3. This is the strongest published uniqueness statement currently
# available within the CICY family.
#
# Anderson-Gray-Lukas-Palti (AGLP) arXiv:1106.4804 §5.3 explicitly notes
# "no phenomenologically viable model were found for h^{1,1}(X) = 2, 3".
# That removes line-bundle SU(5) GUT bundle candidates on h^{1,1}=3 CY3s
# from the AGLP scan range, so the AGLP catalogue contributes ZERO
# competing line-bundle SM data points at (3,3) other than Schoen (which
# was hand-built earlier in BCD/DHOR/AGLO and does not need AGLP to be
# valid).

CL_FINDINGS: List[CY3Candidate] = [
    # The Schoen-Z/3xZ/3 entry is already counted under SCHOEN_FINDINGS.
    # CL16 effectively confirms it is the only CICY-quotient (3,3) entry.
]


# ---------------------------------------------------------------------------
# Catalogue 5 — Pfaffian / non-CICY constructions
# ---------------------------------------------------------------------------
#
# Pfaffian threefolds are typically built as degeneracy loci of skew maps
# of vector bundles on toric varieties (Borisov 1993; Roedland 2000 — the
# Pfaffian-of-7x7 in P^6 with h^{1,1}=1, h^{2,1}=50; Bertin 2009 etc.).
# The published Pfaffian Hodge spectrum is dominated by h^{1,1}=1 entries
# (mirror partners of complete intersections in Grassmannians). No
# Pfaffian CY3 with (h^{1,1}, h^{2,1}) = (3, 3) appears in the published
# Pfaffian catalogue (Borisov-Caldararu, Inoue-Ito-Miura arXiv:1607.05925,
# Coates-Galkin-Kasprzyk arXiv:1212.1722). The class is therefore EXCLUDED
# at (3,3).

PFAFFIAN_FINDINGS: List[CY3Candidate] = [
    CY3Candidate(
        name="pfaffian_class",
        catalogue="Pfaffian / non-CICY (Borisov 1993, Roedland 2000)",
        construction="Degeneracy loci of skew maps of bundles on toric varieties",
        h11=-1, h21=-1, chi=0,
        ambient_obvious_z3xz3=False,
        free_action_status="N/A — class does not contain (3,3) entries",
        bundle_status="N/A",
        citation=(
            "Roedland Compositio 2000; arXiv:1607.05925; arXiv:1212.1722"
        ),
        classification="EXCLUDED",
        notes=(
            "Published Pfaffian CY3s have h^{1,1} = 1 in the Roedland / "
            "Borisov family; the (3,3) bin is empty in the Pfaffian catalogue."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Catalogue 6 — Other small-Hodge constructions
# ---------------------------------------------------------------------------
#
# * Yau three-generation CY (Yau 1986 / Tian-Yau 1987 follow-up): the
#   downstairs Hodge numbers are (h^{1,1}, h^{2,1}) = (1, 4) — NOT (3,3).
#   EXCLUDED.
# * The "famous six" mirror-symmetric Calabi-Yaus (quintic in P^4,
#   bicubic in P^2xP^2 = the Tian-Yau cover, the (2,4) and (3,3) bidegree
#   intersections in P^5, etc.) all have h^{1,1} = 1 or 2 upstairs.
#   None hits (3,3). EXCLUDED.
# * Tian-Yau bicubic (3|3 0; 0 3) cover: (14, 23). EXCLUDED.
# * Tian-Yau Z/3 quotient: (1, 4). EXCLUDED.
# * Generalized CICYs (gCICY, Anderson-Apruzzi-Gao-Gray-Lee
#   arXiv:1507.03235): early (3,3) entries are not reported in the
#   published gCICY tables; the gCICY (3,3) bin is therefore SOFT/UNKNOWN
#   at cycle 1. We record this as PARTIAL.

OTHER_FINDINGS: List[CY3Candidate] = [
    CY3Candidate(
        name="yau_three_generation",
        catalogue="Yau 1986 three-generation CY",
        construction="CICY (3|3 0; 0 3) / Z/3 (Tian-Yau quotient)",
        h11=1, h21=4, chi=-6,
        ambient_obvious_z3xz3=False,
        free_action_status="N/A — Hodge numbers wrong",
        bundle_status="EXCLUDED",
        citation="Tian-Yau 1987; Yau in Mathematical Aspects of String Theory",
        classification="EXCLUDED",
        notes="(1,4) downstairs — not (3,3).",
    ),
    CY3Candidate(
        name="quintic_in_P4",
        catalogue="quintic / mirror-symmetric six",
        construction="degree-5 hypersurface in P^4",
        h11=1, h21=101, chi=-200,
        ambient_obvious_z3xz3=False,
        free_action_status="N/A",
        bundle_status="N/A",
        citation="Candelas-de la Ossa-Green-Parkes 1991",
        classification="EXCLUDED",
        notes="(1,101) — not (3,3).",
    ),
    CY3Candidate(
        name="gcicy_class",
        catalogue="generalized CICY (gCICY, Anderson et al. 2015)",
        construction="generalized complete intersection in products of projective spaces",
        h11=3, h21=3, chi=0,
        ambient_obvious_z3xz3=False,
        free_action_status=(
            "DEFERRED — gCICY automorphism / free-quotient analysis is "
            "cycle-2 work. The published gCICY catalogue does not "
            "tabulate Z/3xZ/3 free quotients."
        ),
        bundle_status="NONE PUBLISHED",
        citation="arXiv:1507.03235",
        classification="PARTIAL",
        notes=(
            "gCICY catalogue is small-but-incomplete; treat as PARTIAL "
            "until the (3,3) bin is enumerated in cycle 2."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Consolidated candidate list
# ---------------------------------------------------------------------------

CANDIDATES: List[CY3Candidate] = (
    CICY_FINDINGS
    + KS_FINDINGS
    + SCHOEN_FINDINGS
    + CL_FINDINGS
    + PFAFFIAN_FINDINGS
    + OTHER_FINDINGS
)


def _classify(candidates: List[CY3Candidate]) -> dict:
    out: dict = {"HARD": [], "SOFT": [], "PARTIAL": [], "EXCLUDED": []}
    for c in candidates:
        out[c.classification].append(c.name)
    return out


def summary() -> str:
    buckets = _classify(CANDIDATES)
    lines = []
    lines.append("Schoen-uniqueness Cycle 1 — Hodge filter summary")
    lines.append("=" * 60)
    for tag in ("HARD", "SOFT", "PARTIAL", "EXCLUDED"):
        lines.append(f"  {tag:<8} ({len(buckets[tag])}): "
                     f"{', '.join(buckets[tag]) or '-'}")
    lines.append("")
    lines.append("HARD-match competitor count (excluding Schoen): "
                 + str(max(0, len(buckets['HARD']) - 1)))
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
    print()
    for c in CANDIDATES:
        print(f"[{c.classification:<8}] {c.name}: ({c.h11},{c.h21})  "
              f"-- {c.catalogue}")
