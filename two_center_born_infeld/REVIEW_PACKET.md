# Review packet: the static two-centre Born-Infeld problem

Tier HIGH (derivation; public-claims care). This packet lists every claim of
`two_center_born_infeld.tex` with its corpus source and status, how to verify it, the author's
review rounds, and the open items.

## How to verify

1. `python generate_numbers.py --check` (exit 0 means every macro, table and this packet's claim table
   regenerate byte-identically from the recorded commits). The generator refuses to run if a receipt's
   verdict or shape changed, if a cited function is missing, if any of the transcribed formula strings
   is absent from its source, or if the recomputed order-18 indicial multisets differ from the corpus.
2. Replay the frozen-ODE quick verifiers on a scratch extract of the recorded origin/dev commit (from
   `book/gr_qm_unification/sympy/derivations/rtail_frozen_ode/`, with python-flint):
   `verify_exact.py`, `identity_probe.py`, `check_controls.py`, `global_c4_proof.py`. The author's
   replay at the recorded commit returned PASS for all four (verify_exact: divergence columns 19,
   observable rank 16, energy order 18, wrong operator REJECTED).
3. `latexmk -pdf two_center_born_infeld.tex` (TeX Live 2026 used; 11 pages).

Status vocabulary is defined in Sec. I C of the paper. A dagger in the paper's claim table, or a ref
other than dev below, marks a source not contained in origin/dev.

## Claims

<!-- BEGIN GENERATED CLAIMS -->
| id | claim | status | source (file::function) | commit (date) | ref |
| --- | --- | --- | --- | --- | --- |
| B1 | superposed D is divergence-correct but curl E is nonzero at (3/5,7/5) | CERTIFIED-EXACT | `derivations/bi_two_center_nonsuperposition_witness.py` | `56d7c16cfe` (2026-09-23) | dev |
| B2 | the frozen trial lies strictly above the static infimum for every d>0 | CERTIFIED | `substrate_theorems/rtail_trial_strict_upper_bound.py` | `745293d9c5` (2026-09-19) | dev |
| B3 | unique minimizer and weak-duality bracket | DERIVED + LITERATURE (admissible-class commitment) | `substrate_theorems/rtail_convex_minimizer_uniqueness.py` | `56d7c16cfe` (2026-09-23) | dev |
| B4 | order-16 H operator and order-18 W annihilator with exact certificate | CERTIFIED-EXACT | `substrate_theorems/rtail_frozen_superposition_ode.py::frozen_certificate` | `cce5e5baac` (2026-10-29) | dev |
| B5 | indicial multisets of the order-18 operator at d=0 and infinity | CERTIFIED-EXACT (recomputed here) | `derivations/rtail_frozen_ode/energy-operator-char0.json` | `8301bc6870` (2026-10-29) | dev |
| B6 | ODE residual test at eight separations, 150 digits | OBSERVED | `derivations/rtail_frozen_ode/numeric-verification.json` | `cce5e5baac` (2026-10-29) | dev |
| B7 | exact algebraic-period germ at d=1; displayed values | DEFINITION + OBSERVED | `derivations/rtail_frozen_ode/initial-data.json` | `cce5e5baac` (2026-10-29) | dev |
| B8 | continuation to d=1.005 agrees with fresh quadrature | OBSERVED | `derivations/rtail_frozen_ode/ivp-replay.json` | `cce5e5baac` (2026-10-29) | dev |
| B9 | frozen contact slope kappa_super by dominated convergence | DERIVED + OBSERVED value | `derivations/rtail_frozen_ode/contact_limit.py` | `cce5e5baac` (2026-10-29) | dev |
| B10 | frozen large-d law with C4 = -3K/4 (global proof) | CERTIFIED | `derivations/rtail_frozen_ode/global_c4_proof.py` | `f38660a070` (2026-10-29) | dev |
| B11 | selected physical W: order >= 4 after finite covers, not algebraic | CERTIFIED | `derivations/rtail_frozen_ode/selected-period-bound.json` | `ef75344b6c` (2026-11-01) | dev |
| B12 | order-17 upper-bound annihilator | SOURCE-CLEAR, NOT LANDED | `derivations/rtail_frozen_ode/radial-scalar-checks/result.json` | `e3b4470853` (2026-11-01) | origin/math/rtail-frozen-ode-1-10 |
| B13 | order-17 divisors: degree-39 factor ordinary, degree-35 factor apparent | AUTHOR + INDEPENDENT CHECK, NOT LANDED | `derivations/rtail_frozen_ode/radial-divisor-checks/result.json` | `9188e5a275` (2026-11-01) | origin/math/rtail-frozen-ode-1-10 |
| B14 | independent order-17 divisor and endpoint check | SOURCE-CLEAR (review) | `work_journal/order17-divisor-review-c27-open.md` | `74e922a118` (2026-11-01) | origin/review/order17-divisor-c27-20261101 |
| B15 | formal radial cyclic rank 17 (minimal for the formal module) | AUTHOR-QUALIFIED | `derivations/rtail_frozen_ode/radial-euler-reduction-spec.md` | `3b8c265399` (2026-11-01) | origin/math/rtail-frozen-ode-1-10 |
| B16 | series continuation over [1/2,4] agrees with quadrature | UNREVIEWED (other lane) | `derivations/diatomic_frozen_pair_series_check.json` | `01b2512a98` (2026-11-01) | origin/substrate-bond-generator-20261101 |
| B17 | true minimum contact enclosure, F0 = 1/2 | CERTIFIED-INTERVAL | `derivations/rtail_contact_asymptotic_certificate.json` | `ad71d45ef8` (2026-10-29) | dev |
| B18 | true minimum interval at d=1 | CERTIFIED-INTERVAL | `derivations/rtail_step2_d1_certificate.json` | `ad71d45ef8` (2026-10-29) | dev |
| B19 | maximum-potential dual lower bound | DERIVED (weak duality) | `derivations/rtail_maximum_potential_lower_bound.md` | `745293d9c5` (2026-09-19) | dev |
| B20 | dual fence values at sample separations | UNREVIEWED (other lane) | `derivations/diatomic_static_pair_fences.json` | `01b2512a98` (2026-11-01) | origin/substrate-bond-generator-20261101 |
| B21 | true large-d law V = 1/d - C0/d^4 (alpha = C0 exact) | CONDITIONAL (induced-dipole commitment) | `substrate_theorems/rtail_large_d_polarizability_exact.py` | `cce5e5baac` (2026-10-29) | dev |
| B22 | first boundary period: Meijer G form and Sp4 monodromy | CERTIFIED | `derivations/rtail_frozen_ode/first-boundary-period.json` | `9aafd4d392` (2026-11-01) | dev |
<!-- END GENERATED CLAIMS -->

## Literature, checked against Crossref or arXiv on 2026-11-01

- Bonheure, d'Avenia, Pomponio, Commun. Math. Phys. 346 (2016) 877: unique minimizer for point
  charges (Theorem 1.3); classical Euler-Lagrange solution off the segments joining charges, and off
  the charges under extra conditions (Theorem 1.6, Section 4). No closed form.
- Bonheure, Colasuonno, Foldes, Ann. Mat. Pura Appl. 198 (2019) 749 (DOI 10.1007/s10231-018-0796-y):
  sufficient conditions for the minimizer to solve the Euler-Lagrange equation; singularity analysis.
  The corpus had attributed this DOI to Bonheure, d'Avenia and Pomponio with volume 197 and pages
  1025-1042; corrected in the accompanying monorepo commit.
- Yang, Ann. Phys. 458 (2023) 169463: exact multicentred BI solutions with induced static currents;
  a different problem.
- Carley, Kiessling, PRL 96 (2006) 030402: nonperturbative BI hydrogen calculation; disputed by
  Smolyakov (arXiv:2206.07611), reply arXiv:2501.00613.
- Breev, Shabad, EPJC 78 (2018) 39: a nonlinear model quadratic in the field invariant, not
  Born-Infeld; its contact law a + bR^(1/3) is a rough estimate in that model. The corpus forcing node
  had listed it among Born-Infeld results; corrected.
- Chyzak 2000; Beukers-Heckman 1989; van der Hoeven 1999; Mezzarobba 2010; DLMF.

## Author review rounds (substrate-review-math and substrate-review-physics lenses)

Round 1 (2026-11-01, during drafting): findings and dispositions.

| severity | finding | disposition |
| --- | --- | --- |
| MEDIUM | Corpus misattribution of DOI 10.1007/s10231-018-0796-y (authors, volume, pages) in `atomic_correlation_constant_not_forced.py` and the forcing node, carried into the task brief as "Bonheure, d'Avenia and Pomponio 2018" | cited correctly in the paper; corpus fixed |
| MEDIUM | Breev-Shabad cited in corpus as a Born-Infeld result (FM4 over-scope) | paper states the model; corpus claim strings rescoped |
| MEDIUM | True large-d law presented as exact in `rtail_two_center_synthesis.py` while it rests on the registered commitment `rtail_large_d_induced_dipole_structure` | paper marks it conditional; corpus docstring and statement corrected |
| MEDIUM | Physics: "retarded updates propagate at c" ignored the compression speed | rewritten with both speeds and the open characteristic-speed derivation |
| MEDIUM | Generated timeline printed local time, not UTC | generator forces TZ=UTC |
| LOW | "strain patterns" wording for the static flux D | reworded |
| LOW | Title line 5 pt overfull | follow-up |

Round 2 (2026-11-01, whole-text re-read after fixes): no CRITICAL, HIGH or MEDIUM finding. Remaining
LOW: title overfull; the forcing node's unverified "establishes" text for Carley and Kiessling (linear
Bopp-Podolsky remark) is left as a follow-up.

## Open items

Listed in Sec. IX of the paper. Values marked unreviewed come from the bond-research branch, whose
independent review was pending when this packet was written. Order-17 results are source-clear on
their branches and not landed on origin/dev. The forcing-graph `cli.py validate` run of the corrected
node is pending fleet memory headroom (the SCC diagnostic and JSON parse pass).
