# The static two-centre Born-Infeld problem

**Status: working draft, not submitted for peer review.**

`two_center_born_infeld.pdf` (built from `two_center_born_infeld.tex`) records what the substrate
corpus establishes about the static energy of two equal point sources in the Born-Infeld medium, and
what it does not:

- Born-Infeld does not superpose: the superposed field has nonzero curl at an explicit point, and the
  frozen superposition trial lies strictly above the true static minimum for every separation.
- The frozen trial energy has an exact order-18 annihilator with polynomial coefficients (certified,
  replayed here), a smaller order-17 annihilator (independent source clearance, not yet on dev), a
  certified lower bound of order 4 and nonalgebraicity, and an exact algebraic-period germ at d = 1.
  It is therefore computable by series continuation; the recorded agreements with independent
  quadrature are observations, not error bars.
- The true static minimum has certified fences only: a contact enclosure giving the exact contact
  force 1/2, a certified interval at d = 1, and a dual lower bound. Its large-separation law rests on a
  registered commitment.
- A frozen trial is an upper bound, not the energy. The static energy of like sources is a repulsion;
  a bond needs the moving, bound two-mode state, which is outside this problem.

The companion paper `../acorr_derivation/` is the provenance record for the correlation constant
a_corr, in which this static energy appears as the two-centre part.

## Files

| file | content |
| --- | --- |
| `two_center_born_infeld.tex` | paper source (REVTeX 4.2, single column) |
| `two_center_born_infeld.bib`, `two_center_born_infeldNotes.bib`, `two_center_born_infeld.bbl` | bibliography source, REVTeX control file, compiled bibliography |
| `two_center_born_infeld.pdf` | built PDF |
| `numbers.tex`, `claims_table.tex` | generated; every number and the claim table |
| `generate_numbers.py` | generator for both papers' numbers, claim tables, timeline and review-packet tables |
| `provenance_refs.json` | the full commit hashes of the cited branch references |
| `REVIEW_PACKET.md` | every claim with its corpus source and status |

## Reproduce

From this folder, with sympy and mpmath, inside a monorepo checkout whose object store contains the
recorded commits:

```sh
python generate_numbers.py --check   # regenerates in memory; exit 0 means no drift
latexmk -pdf two_center_born_infeld.tex
```

`generate_numbers.py` reads each value from a corpus receipt at the commit recorded in
`provenance_refs.json`, recomputes closed forms, independently recomputes the order-18 indicial
multisets, verifies every transcribed formula string against its source, and refuses to run if a
receipt's verdict or shape has changed. The papers were built with TeX Live 2026 (pdfTeX, BibTeX,
latexmk). REVTeX 4.2 with this LaTeX kernel rejects `p{}` table columns, so wrapped cells use a
`\cell` macro over `l` columns.
