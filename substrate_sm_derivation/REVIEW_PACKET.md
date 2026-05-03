# Empirical CY3 substrate identification — review packet

This packet contains the empirical trace data, source code, and per-cycle research logs that back the *Empirical CY3 substrate identification* section of `substrate_sm_derivation.tex`. The companion `README.md` in this same directory describes the paper itself.

Total size: ~11 MB (essential) + 148 MB optional pipeline checkpoints (not included; available on request).

## Layout

```
papers/substrate_sm_derivation/
├── README.md                                 (paper overview)
├── REVIEW_PACKET.md                          (this file)
├── substrate_sm_derivation.tex               (the paper source, scrubbed)
├── substrate_sm_derivation.pdf               (built PDF, 31 pages)
├── substrate_sm_derivation.bbl               (compiled bibliography)
├── substrate_sm_derivationNotes.bib          (bib source)
├── tables/                                   (13 referenced .tex tables)
├── cy3_solver/                               (Rust solver, ~9 MB)
│   ├── Cargo.toml, Cargo.lock, build.rs     (pinned build)
│   ├── src/                                  (182 .rs files; 114 with #[test])
│   ├── tests/                                (integration tests)
│   ├── output/                               (118 production-sweep JSONs + .replog)
│   └── references/                           (28 .md per-phase audit + literature extracts)
└── python_research/                          (40 .py scripts)
    ├── h{1,2,3}_*.py                         (TY/Z3 9-cycle bundle scan, Track 3)
    ├── schoen_uniqueness_cycle{1,2,2_5,2_6}*.py  (Path-A catalogue scan, Track 4)
    └── _anderson_2015_extracted.txt          (PDF-extracted source for cycle 2.6)
```

## Per-claim trace map

The paper's empirical claims (§ "Empirical CY3 substrate identification") cash out to four converging tracks. Each track has an independent trace.

**Track 1 — chain-channel Bayesian preference at 5.43σ** (Tab. cy3-bayesian-channels)
- Code: `cy3_solver/src/{discriminate,harmonic,chain_match,bayes_factor}.rs`
- Output: `cy3_solver/output/p_basis_convergence_prod.json` + chain-match JSONs
- Methodology: `cy3_solver/references/p_basis_convergence_prod_launch.md`

**Track 2 — σ-channel basis-size sign-reversal defence** (Tab. cy3-sigma-artifact)
- Output: `cy3_solver/output/p_basis_convergence_diag.{json,csv}`
- Methodology: `cy3_solver/references/p_basis_convergence_diagnostic.md`
- Code: `cy3_solver/src/{donaldson,bootstrap}.rs`

**Track 3 — TY/Z3 Yukawa nine-cycle structural exclusion** (Tab. cy3-ty-exclusion)
- Code: `python_research/h{1,2,3}_*.py` (9 scripts, one per cycle)
- Per-cycle log: `cy3_solver/references/p_ty_bundle_research_log.md`
- Audit: `cy3_solver/references/p_ty_bundle_audit.md`

**Track 4 — Path-A catalogue uniqueness at h=(3,3)** (new in revision)
- Code: `python_research/schoen_uniqueness_cycle{1,2,2_5_cas_sweep,2_6_gcicy_nonmanifest}.py`
- Per-cycle reports: `cy3_solver/references/p_schoen_uniqueness_cycle{1,2,2_5,2_6}.md`
- Cycle 2.6 source-of-truth extract: `python_research/_anderson_2015_extracted.txt`

## Cross-cutting

- **Reproducibility manifest**: each `output/*.json` embeds a SHA-256 chained event log (`.replog` sidecar files) covering binary identity, Rust toolchain, target triple, CPU features, hostname (set to a generic identifier), UTC timestamp, command-line, and per-config inputs.
- **Literature reference extracts**: `cy3_solver/references/{aklp_2010, headrick_wiseman_2005, donaldson_2009, dklr_2006, anderson_he_lukas_2007, ashmore_he_ovrut_2019, larfors_2022, dhhkw_2008_dp3}.md` — bit-exact σ values and convention conventions extracted from each cited paper, used to ground the convention-match h-block test (P3.13) and the σ ∝ 1/k² scaling regression test.
- **Tests**: 114 `#[test]` files in `cy3_solver/src/` cover convention-match (FS-Gram identity), σ ∝ 1/k², CPU↔GPU bit-exactness, gradient finite-difference parity, Donaldson FS-identity, Yukawa sum-rule, basis-size convergence, and reproducibility-log determinism.

## Build instructions

**Manuscript PDF** (already built; rebuild if desired):
```
pdflatex substrate_sm_derivation.tex
bibtex substrate_sm_derivation
pdflatex substrate_sm_derivation.tex
pdflatex substrate_sm_derivation.tex
```

**Rust solver** (~10–30 min for full test suite, GPU optional):
```
cd cy3_solver/
cargo build --release
cargo test --release          # 114 tests
# Production sweeps require ~50 GPU-h + ~30 CPU-h total; see references/cy3_publication_summary.md §6.
```

**Python research scripts** (Schoen-uniqueness Path-A reproduction):
```
cd python_research/
python3 schoen_uniqueness_cycle1_hodge_filter.py
python3 schoen_uniqueness_cycle2_free_action.py
python3 schoen_uniqueness_cycle2_5_cas_sweep.py        # queries Kreuzer-Skarke CGI; needs internet
python3 schoen_uniqueness_cycle2_6_gcicy_nonmanifest.py
```

## Honest scope notes

- **CAS verification deferred for cycle 2.6.** Sage/Macaulay2 were not installed in the build environment. Cycle 2.6 closes the gCICY non-manifest residue by literature-empirical means (PDF extraction + cross-paper search returning 0 hits for a (3,3) entry), not by a CAS-grade automorphism computation. A future paper extending the gCICY codim ≥ (3,1) catalogue with a new (3,3) entry would warrant a re-test.
- **σ-discriminability is not a likelihood.** The σ-channel is reported as a separate diagnostic |t|-statistic, not as a Bayes-factor likelihood contribution, because of the basis-size sign-reversal documented in Track 2. Reviewers should not interpret the 6.92σ k=3 number as additive to the 5.43σ chain-channel result.
- **TY 9-cycle exclusion is bundle-construction-specific.** It rules out rank-3 SU(3) line-bundle-sum and monad bundles in the searched bidegree range with the stability + balance conditions enumerated. It is not a proof that no SM-compatible bundle of any kind can exist on TY/Z3, only that none has been constructed in the catalogues searched (consistent with AGLP 2012 §5.3).
