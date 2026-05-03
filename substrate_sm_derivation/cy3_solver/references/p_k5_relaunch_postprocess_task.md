# P-K5-RELAUNCH-POSTPROCESS (Filed Task)

**Filed:** 2026-04-30 by P-K5-RELAUNCH (see `p_k5_relaunch.md`).
**Status:** PENDING — blocked on production sweep completion.
**Trigger:** When `output/p5_10_k5_damped_relaunch.json` exists and is
finalized (the binary writes JSON only on clean exit; partial runs leave
just `.log` and `.replog`).
**Production PID:** 832016 (k=5, 40 000 pts, donaldson_iters=200, GPU).
**ETA:** 2026-05-02 ~01:00–03:00 local (~7–9 h from launch at 17:54).

## Inputs

- `output/p5_10_k5_damped_relaunch.json` — full per-seed records.
- `output/p5_10_k5_damped_relaunch.log` — wallclock + sigma per seed.
- `output/p5_10_k5_damped_relaunch.kernel.replog` — kernel-level repro
  events (if produced).

Cross-comparators:
- `output/p5_10_ty_schoen_5sigma.json` (k=3, 6.92σ Tier 0).
- `output/p5_10_k4_damped.json` (k=4 static-α=0.5, 3.82σ supporting).
- `references/p_basis_convergence_diag.csv` (basis-size-artifact data).
- Partial output of the dead PID 388840 run (13/20 TY seeds at σ ≈ 0.29)
  — for spot-check that the relaunch reproduces the same TY trajectory.

## Acceptance criteria

1. Read JSON, classify Tier 0 strict (residual < tol AND iters < cap AND
   n_Schoen ≥ 16 AND no guard snapshots AND no Tukey trim) per the
   P5.10 v7 protocol used at k=3.
2. Report n-σ point estimate, percentile CI, and BCa CI on Tier 0.
3. Compare relaunch TY-side σ trajectory against the dead-run partial
   (13/20 seeds at σ ≈ 0.29) — they should agree to within MC noise.
4. If relaunch-σ differs significantly from the partial: file
   P-K5-NONDETERMINISM.
5. Update `references/p8_4_k4_production.md` (or create
   `references/p_k5_production_results.md`) with the supporting-tier σ.
6. If σ ≥ 5σ Tier 0 OR if the k=3/k=4/k=5 picture meaningfully
   strengthens, update `references/cy3_publication_summary.md`:
   - §1 Headlines: add k=5 supporting line.
   - §4 Supporting evidence: append k=5 row.
7. Update `references/p_basis_convergence_diagnostic.md` with the k=5
   data point — does the σ trend at {k=3, 4, 5} survive a basis-size
   correction, or does the artifact dominate?

## Decision gates

- **σ ≥ 5σ Tier 0**: continuum extrapolation succeeds. File as
  publication-grade supporting evidence at higher k. Update
  publication §1 with monotone-in-k or stable-across-k claim.
- **σ < 5σ but TY > Schoen at significance**: file as qualitative
  continuum-confirming. Headline stays k=3.
- **Schoen > TY or σ collapses under basis-size correction**: file
  as evidence that ABKO-style normalization is required at k=5.
  Re-evaluate publication strategy.
- **Many Tier 0 seeds hit iter cap (200) or trip regression guards**:
  file P-K5-static-damping (add `--donaldson-damping <f64>` CLI flag
  to the binary, rerun with α=0.5).

## Out of scope

- Forward-model / Bayes-multichannel work
  (`bayes_factor_multichannel.rs`, `p8_1_bayes_multichannel.rs`).
- Memory updates and git commits (per launch task constraints).
- Source modifications to the production binary.
