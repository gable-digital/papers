# P-K5-postprocess (Filed Task)

**Filed:** 2026-04-30 by P-K5-launch
**Status:** PENDING — blocked on production sweep completion (~6-10 hr ETA from launch).
**Trigger:** When `output/p5_10_k5_damped.json` exists and is finalized
(the binary writes JSON only on clean exit; partial runs will leave
just `.log` and `.replog`).

## Inputs

- `output/p5_10_k5_damped.json` — full per-seed records.
- `output/p5_10_k5_damped.log` — wallclock + sigma per seed.
- `output/p5_10_k5_damped.kernel.replog` — kernel-level repro events.

Cross-comparators:
- `output/p5_10_ty_schoen_5sigma.json` (k=3, 6.92σ Tier 0).
- `output/p5_10_k4_damped.json` (k=4 static-α=0.5, 3.82σ supporting).
- `output/p5_10_k4_adaptive.json` (k=4 adaptive, 1.83σ — for adaptive
  vs static comparison at k=5).
- `references/p_basis_convergence_diag.csv` (basis-size-artifact data).

## Acceptance criteria

1. Read JSON, classify Tier 0 strict (residual<tol AND iters<cap AND
   n_Schoen ≥ 16 AND no guard snapshots AND no Tukey trim) per the
   P5.10 v7 protocol used at k=3.
2. Report n-σ point estimate, percentile CI, and BCa CI on Tier 0.
3. Compare adaptive (this run) vs an eventual static-α=0.5 rerun if
   adaptive under-performs (file P-K5-static-damping in that case).
4. Update `references/cy3_publication_summary.md`:
   - §1 Headlines: add k=5 supporting line.
   - §4 Supporting evidence: append k=5 row.
5. Update `references/p_basis_convergence_diagnostic.md` with the k=5
   data point — does the σ trend at {k=3, 4, 5} survive a basis-size
   correction, or does the artifact dominate?

## Decision gates

- **If k=5 σ ≥ 5σ Tier 0**: continuum extrapolation succeeds. File
  as publication-grade supporting evidence at higher k. Update the
  publication §1 to claim a "monotone-in-k" or "stable across
  k∈{3,4,5}" trend (whichever the data supports).
- **If k=5 σ < 5σ but TY > Schoen at significance**: file as
  qualitative continuum-confirming. Headline stays k=3.
- **If k=5 inverts (Schoen > TY) or collapses under basis-size
  correction**: file as evidence that the ABKO-style normalization
  is required at this k. Re-evaluate publication strategy.
- **If many Tier 0 seeds hit the iter cap or trip the regression
  guard**: file P-K5-static-damping (add `--donaldson-damping <f64>`
  CLI flag to p5_10_ty_schoen_5sigma binary, rerun with α=0.5).

## Out of scope

- The forward-model / Bayes-multichannel agent has separate work
  in flight on `bayes_factor_multichannel.rs` and
  `p8_1_bayes_multichannel.rs`. Do NOT touch those files in
  P-K5-postprocess.
- Memory updates and git commits are NOT part of P-K5-postprocess
  (the launch task explicitly forbade them; the same applies here
  unless escalated by the user).
