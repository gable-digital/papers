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
3. `latexmk -pdf acorr_derivation.tex` (TeX Live 2026 used; 11 pages).

## Claims

<!-- BEGIN GENERATED CLAIMS -->
| id | claim | status | source (file::function) | commit (date) | ref |
| --- | --- | --- | --- | --- | --- |
| A1 | a_corr is defined as eps_2 - eta_2 of the Coulomb comparison model (high-Z, same centre, two electrons) | DEFINITION | `substrate_theorems/atomic_correlation_constant_not_forced.py::verdict` | `56d7c16cfe` (2026-09-23) | dev |
| A2 | eta_2 = (9/32)ln(3/4) - 13/432 by the Dalgarno-Lewis reduction | DERIVED (imported Coulomb operator) | `substrate_theorems/atomic_correlation_constant_not_forced.py::eta2_symbolic_derivation` | `56d7c16cfe` (2026-09-23) | dev |
| A3 | eps_2 = published value (Wang 2013), not a substrate result; with it the a_corr value is a cross-check oracle | IMPORTED (literature; cross-check oracle only) | `imported_theorems/second_order_partial_waves.py::published_eps2` | `56d7c16cfe` (2026-09-23) | dev |
| A4 | s-wave and l>=1 split of a_corr from literature per-l increments | DIAGNOSTIC (literature inputs) | `derivations/acorr_partialwave_decomposition.py` | `674a28b5fd` (2026-09-18) | dev |
| A5 | the s-wave crossover correction lies outside the elementary plus Ei span (non-membership only) | DERIVED (conditional) | `substrate_theorems/atomic_correlation_constant_not_forced.py::eps2_swave_crossover_is_non_elementary` | `56d7c16cfe` (2026-09-23) | dev |
| A6 | coincidence match is a square system on the retained contact coefficients; it fixes them, not a_corr | DERIVED | `substrate_theorems/atomic_correlation_constant_not_forced.py::substrate_correlation_coincidence_match_underdetermines` | `56d7c16cfe` (2026-09-23) | dev |
| A7 | proximity cap rows add zero rank; (a_corr, remainder) block stays rank 1 in 2 unknowns | DERIVED (finite contact-coefficient model only) | `derivations/acorr_proximity_cap_rank_analysis.py` | `56d7c16cfe` (2026-09-23) | dev |
| A8 | test-charge contact ratio V(1,1;0)/U(0) = 4(sqrt2-1)/3, not 1 | CERTIFIED-EXACT | `substrate_theorems/bi_test_charge_pair_energy.py::contact_interaction_certificate` | `b436d403d6` (2026-09-20) | dev |
| A9 | scalar point-source potential: exact substitution, quartic range [1/8,1], Coulomb bracket | DERIVED (scalar route only) | `derivations/acorr_charge_tail_dependency_FINDINGS.md` | `cfa43b6430` (2026-10-31) | dev |
| A10 | static linear strain tail is l=0 and the wake is O(alpha^2) inside the Coulomb comparison only | CONDITIONAL DIAGNOSTIC | `substrate_theorems/atomic_correlation_constant_not_forced.py::substrate_correlation_strain_tail_and_wake_not_the_resisting_piece` | `56d7c16cfe` (2026-09-23) | dev |
| A11 | static two-centre energy to a_corr identification | COMMITMENT (M11-gated) | `substrate_theorems/rtail_two_center_synthesis.py::two_center_synthesis` | `56d7c16cfe` (2026-09-23) | dev |
| A12 | M11 true-minimizer route: physical moving-mode functional and bound mode missing | OPEN | `derivations/acorr_m11_true_minimizer_route_validation.py` | `56d7c16cfe` (2026-09-23) | dev |
| A13 | former order-4 Picard-Fuchs non-elementarity claim | WITHDRAWN | `derivations/rtail_known_problems_context.md` | `745293d9c5` (2026-09-19) | dev |
| A14 | a_corr derived from the Born-Infeld Lagrangian and the particle equations (the required route) | OPEN | `substrate_theorems/atomic_correlation_constant_not_forced.py::verdict` | `56d7c16cfe` (2026-09-23) | dev |
| A15 | electron mass from the fixed closed-form particle equations (default structural anchor) | FIXED CLOSED FORM (framework premise) | `book/scripts/substrate_mass_spectrum.py::derive_constants` | `d90bb8280a` (2026-09-22) | dev |
| A16 | energy unit alpha^2 m_e c^2 that expresses a_corr in eV, from the particle module | CONDITIONAL (module marks a_0 and Ry not forward-closed) | `book/scripts/forward_p_row_generator.py::hartree_eV` | `de463350ee` (2026-09-20) | dev |
| A17 | r12 Jastrow VMC 'forward a': Monte Carlo of the ordinary Coulomb problem, screened reference, extrapolated | IMPORTED (cross-check only; not a derivation) | `book/scripts/derive_pair_correlation_constant_forward.py::corr_at_Z` | `79ab032f5a` (2026-08-11) | dev |
| A18 | b/a = -3/4 identified with the spin-1/2 Casimir of the particle sector | OPEN (lead, to confirm) | `book/scripts/FORWARD_ELEMENT_GENERATOR.md` | `7fd6e5c283` (2026-08-11) | dev |
| A19 | element generator installs a_corr = 4667/100000 as a legacy diagnostic literal | IMPORTED (literature value installed) | `book/scripts/forward_generator_complete.py::legacy_orbital_ie_diagnostic` | `82ead9b8d9` (2026-09-20) | dev |
| A20 | occupation shape k(3k-11)/8 and b/a = -3/4 found by least-squares fit of the period-2 residual | FITTED | `book/scripts/forward_missing_terms_generator.py` | `c0610401eb` (2026-08-11) | dev |
| A21 | legacy optimizer minimized a removed-orbital expression, not the atom's energy (September correction) | CERTIFIED-EXACT (defect) | `derivations/atomic_generator_energy_consistency_FINDINGS.md` | `de463350ee` (2026-09-20) | dev |
| A22 | period 2 under 1% with zero fits superseded; corrected objective gives B to Ne -16.1 to -53.6 percent | UNREVIEWED (other lane probe) | `work_journal/ionization-generator-period3-open.md` | `40cd2c13f7` (2026-11-01) | origin/research/ionization-generator-period3-20261101 |
| A23 | proximity cap consistency sum_k L_k r_min^p_k = 0 is a condition on r_min; a positive root and its uniqueness are not proved | OPEN (result left unset) | `substrate_theorems/atomic_correlation_constant_not_forced.py::substrate_correlation_proximity_cap_is_contact_local_no_rank` | `56d7c16cfe` (2026-09-23) | dev |
<!-- END GENERATED CLAIMS -->

## Assessment of the owner's statements

"a_corr was derived into two parts with R_tail closed as a series" partly aligns with the record.
Aligns: the corpus splits a_corr into two parts (three different senses), and the frozen two-centre
trial energy is closed as an exact differential equation with an exact germ, computable by series.
Conflicts: no split determines a_corr; what is closed as a series is the frozen trial (an upper
bound), not the true static minimum and not the moving-mode correlation; the corpus a_corr number
inherits the published eps_2.

Owner correction (relayed by the coordinator, verbatim): "a_corr is not fixed merely from the
lagrangian. It is also built based on the fixed closed form expressions for the particles". Aligns:
the particle equations of book/scripts/substrate_mass_spectrum.py enter, through the electron mass
(the hartree and Bohr units; fixed closed form, framework premise), the coupling used for the eV
conversion (conditional: the module marks a_0 and Ry not forward-closed) and the Born-Infeld core
scale relative to the atom (conditional on the W_single = m_e c^2 identification). Does not establish:
that these expressions fix the dimensionless value. The August route
(derive_pair_correlation_constant_forward.py) is a Coulomb-model Jastrow VMC of a screened-reference
quantity (recorded values extrapolate to 0.04995 by the script's own fit); the generators install the
literature value 0.04667; the occupation shape and the 3:4 ratio were fitted; and the period-2
under-one-percent closure was superseded in September (removed-orbital objective; see A20 to A22 and
paper Sec. IV). The physical a_corr is open.

Owner ruling (relayed by the coordinator, verbatim): "We shouldn't be running a monte carlo calculation
of the ordinary two-electron Coulomb problem. It has to come from the BIL and particle equations, not be
imported." Applied: the a_corr value 0.04667 (eps_2 from Wang 2013) and the Monte Carlo route are
labelled IMPORTED, cross-check oracles only and not derivations (A3, A17, A19); the derivation from the
Born-Infeld Lagrangian and the particle equations is the required route and is OPEN (A14, first open
item); no wording presents the Monte Carlo route as a forward derivation.

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

Round 3 (2026-11-01, revision for the owner's correction and the coordinator's relay of READY 128594):

| severity | finding | disposition |
| --- | --- | --- |
| HIGH | Paper framed a_corr as not determined "from the Lagrangian" alone and omitted how the closed-form particle expressions enter | new Sec. IV traces steps P1 to P7 with statuses; abstract and assessment revised |
| MEDIUM | The August "forward a" route was not analysed: Coulomb Hamiltonian, screened (not HF) reference, MC without error bars, extrapolation 0.04995 vs the documented 0.047, validation gated on the fitted 0.04738 | stated in Sec. IV, claim A17 |
| MEDIUM | The period-2 "under 1% with zero fits" claim needed its September correction and fit provenance | claims A20 to A22, timeline adds both September commits |
| LOW | Email line 14 pt overfull | follow-up |

Round 3 re-read: no CRITICAL, HIGH or MEDIUM finding remains.

Round 4 (2026-11-01, remediation of Codex4-2 VERDICT FIX FIRST C0 H0 M2 L2 on request 128554, and the
owner ruling above; remediation by the acct3 lead):

| severity | finding | disposition |
| --- | --- | --- |
| MEDIUM (M1) | The exact split kept only finite contact moments and defined R_tail as the outer integral, omitting the local Taylor remainder (counterexample G = r^10 e^-r, w = 1, r* = 1, p = 3..9); "no local condition reaches a_corr" overclaimed the finite-matrix result | Sec. III A: R_tail = R_loc + R_out with R_loc defined; the counterexample is checked symbolically by `generate_numbers.py::local_remainder_counterexample` (macro RlocCounterVal); the claim narrowed to conditions on the retained contact coefficients, with local data plus a global field equation kept open; A6 and A7 narrowed; source script docstring corrected in the corpus |
| MEDIUM (M2) | "the cap's consistency fixes the minimum separation" while the source derives only sum_k L_k r_min^p_k = 0 and leaves the result None | stated as a consistency condition with no proved positive or unique root; claim A23 (OPEN); routes table and open items updated |
| LOW | 1/s bounds applied at s = 0 | bounds stated for s > 0 |
| LOW | Paper B said every number is a generated macro, though model constants and coordinates are literals | narrowed to generated numerical results; the same narrowing applied to paper A's purpose paragraph |
| HIGH (owner ruling) | Value and Monte Carlo route presented as diagnostic rather than imported; required route not stated | IMPORTED status added and applied (abstract, Secs. II C and IV, certified list, assessment, premise ledger, A3, A17, A19); required BI-plus-particle-equations route OPEN (A14, open item 1) |

## Open items

Listed in Sec. IX of the paper: the required derivation of a_corr from the Born-Infeld Lagrangian and
the particle equations (imported values as cross-checks only); the moving-mode two-electron
correlation functional and bound mode; the charge sector of the cross term; a closed form or
impossibility proof for eps_2; the remainder R_tail (local Taylor remainder and outer integral) as
constructed integrals, and whether the cap consistency condition has a positive, unique root; the
static two-centre items of the companion paper; and unverified corpus citation fields.
