# Review packet: provenance of a_corr

Tier HIGH (derivation; public-claims care). This packet lists every claim of
`acorr_derivation.tex` with its corpus source and status, how to verify it, the author's review
rounds, and the open items.

## How to verify

1. `python ../two_center_born_infeld/generate_numbers.py --check` (exit 0 means every macro, the claim
   and timeline tables and this packet's claim table regenerate byte-identically from the recorded
   commits; see `../two_center_born_infeld/REVIEW_PACKET.md` for the generator's refusals).
2. The partial-wave split and the nonsuperposition witness are produced by executing the corpus
   scripts `acorr_partialwave_decomposition.py` and `bi_two_center_nonsuperposition_witness.py` from
   their blobs at the recorded origin/dev commit inside the generator.
3. `latexmk -pdf acorr_derivation.tex` (TeX Live 2026 used; 8 pages).

## Claims

<!-- BEGIN GENERATED CLAIMS -->
| id | claim | status | source (file::function) | commit (date) | ref |
| --- | --- | --- | --- | --- | --- |
| A1 | a_corr is defined as eps_2 - eta_2 of the Coulomb comparison model (high-Z, same centre, two electrons) | DEFINITION | `substrate_theorems/atomic_correlation_constant_not_forced.py::verdict` | `56d7c16cfe` (2026-09-23) | dev |
| A2 | eta_2 = (9/32)ln(3/4) - 13/432 by the Dalgarno-Lewis reduction | DERIVED (imported Coulomb operator) | `substrate_theorems/atomic_correlation_constant_not_forced.py::eta2_symbolic_derivation` | `56d7c16cfe` (2026-09-23) | dev |
| A3 | eps_2 = published value (Wang 2013), not a substrate result | LITERATURE | `imported_theorems/second_order_partial_waves.py::published_eps2` | `56d7c16cfe` (2026-09-23) | dev |
| A4 | s-wave and l>=1 split of a_corr from literature per-l increments | DIAGNOSTIC (literature inputs) | `derivations/acorr_partialwave_decomposition.py` | `674a28b5fd` (2026-09-18) | dev |
| A5 | the s-wave crossover correction lies outside the elementary plus Ei span (non-membership only) | DERIVED (conditional) | `substrate_theorems/atomic_correlation_constant_not_forced.py::eps2_swave_crossover_is_non_elementary` | `56d7c16cfe` (2026-09-23) | dev |
| A6 | coincidence match is a square system; it fixes contact data, not a_corr | DERIVED | `substrate_theorems/atomic_correlation_constant_not_forced.py::substrate_correlation_coincidence_match_underdetermines` | `56d7c16cfe` (2026-09-23) | dev |
| A7 | proximity cap adds zero rank; (a_corr, R_tail) block stays rank 1 in 2 unknowns | DERIVED | `derivations/acorr_proximity_cap_rank_analysis.py` | `56d7c16cfe` (2026-09-23) | dev |
| A8 | test-charge contact ratio V(1,1;0)/U(0) = 4(sqrt2-1)/3, not 1 | CERTIFIED-EXACT | `substrate_theorems/bi_test_charge_pair_energy.py::contact_interaction_certificate` | `b436d403d6` (2026-09-20) | dev |
| A9 | scalar point-source potential: exact substitution, quartic range [1/8,1], Coulomb bracket | DERIVED (scalar route only) | `derivations/acorr_charge_tail_dependency_FINDINGS.md` | `cfa43b6430` (2026-10-31) | dev |
| A10 | static linear strain tail is l=0 and the wake is O(alpha^2) inside the Coulomb comparison only | CONDITIONAL DIAGNOSTIC | `substrate_theorems/atomic_correlation_constant_not_forced.py::substrate_correlation_strain_tail_and_wake_not_the_resisting_piece` | `56d7c16cfe` (2026-09-23) | dev |
| A11 | static two-centre energy to a_corr identification | COMMITMENT (M11-gated) | `substrate_theorems/rtail_two_center_synthesis.py::two_center_synthesis` | `56d7c16cfe` (2026-09-23) | dev |
| A12 | M11 true-minimizer route: physical moving-mode functional and bound mode missing | OPEN | `derivations/acorr_m11_true_minimizer_route_validation.py` | `56d7c16cfe` (2026-09-23) | dev |
| A13 | former order-4 Picard-Fuchs non-elementarity claim | WITHDRAWN | `derivations/rtail_known_problems_context.md` | `745293d9c5` (2026-09-19) | dev |
| A14 | physical a_corr value from the master Lagrangian | OPEN | `substrate_theorems/atomic_correlation_constant_not_forced.py::verdict` | `56d7c16cfe` (2026-09-23) | dev |
<!-- END GENERATED CLAIMS -->

## Assessment of the owner's statement

"a_corr was derived into two parts with R_tail closed as a series" partly aligns with the record.
Aligns: the corpus splits a_corr into two parts (three different senses), and the frozen two-centre
trial energy is closed as an exact differential equation with an exact germ, computable by series.
Conflicts: no split determines a_corr from the master Lagrangian; what is closed as a series is the
frozen trial (an upper bound), not the true static minimum and not the moving-mode correlation; the
corpus a_corr number inherits the published eps_2. The physical a_corr is open.

## Literature, checked against Crossref on 2026-11-01

Born-Infeld 1934; Wang 2013 (eps_2); Loos-Gill 2009 (eta_2 oracle); Dalgarno-Lewis 1955; Kato 1957;
Schwartz 1962; Carroll-Silverstone-Metzger 1979 (per-l increments); Bonheure-d'Avenia-Pomponio 2016;
Bonheure-Colasuonno-Foldes 2019.

## Author review rounds (substrate-review-math and substrate-review-physics lenses)

Round 1 (2026-11-01, during drafting):

| severity | finding | disposition |
| --- | --- | --- |
| MEDIUM | The 1/Z expansion and Hartree-Fock reference are perturbative constructs (quarantine) | stated as an imported comparison problem only |
| MEDIUM | Owner's phrase "R_tail closed as a series" conflates three objects named R_tail | separate symbols; each object's status stated |
| MEDIUM | Derivations README headlined a withdrawn order-4 non-elementarity claim | corpus README corrected; paper records the history |
| MEDIUM | Corpus misattribution of the Bonheure-Colasuonno-Foldes paper | corrected in corpus and cited correctly |
| LOW | Numbers in abstract typed by hand | replaced by generated macros |

Round 2 (2026-11-01, whole-text re-read after fixes): no CRITICAL, HIGH or MEDIUM finding.

## Open items

Listed in Sec. VIII of the paper: the moving-mode two-electron correlation functional and bound mode;
the charge sector of the cross term; a closed form or impossibility proof for eps_2; the remainder
R_tail as a constructed integral; the static two-centre items of the companion paper; and unverified
corpus citation fields.
