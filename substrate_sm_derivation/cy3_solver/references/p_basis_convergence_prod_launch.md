# P-BASIS-CONVERGENCE-prod — Production launch record

**Status:** detached background run in progress (launched 2026-04-30)
**Binary:** `target/release/p_basis_convergence_diag.exe` (commit-tracked source: `src/bin/p_basis_convergence_diag.rs`)
**Predecessor:** `references/p_basis_convergence_diagnostic.md` (5 seeds × 10k pts, commit d416a6d6)
**Task spec:** P-BASIS-CONVERGENCE-prod (#257)

## 1. Why this rerun

The original diagnostic (5 seeds × n_pts=10 000 × 25 Donaldson iters × tol=1e-3)
found that **86% of the native TY-vs-Schoen σ gap collapses at matched basis
size n_b=27 at k=3**, and the k-scan does not obey the ABKO 1/k² Donaldson
prediction. That finding underwrites the P8.1e/P8.4 decision to keep σ
out of the multi-channel Bayes Factor. For manuscript inclusion the
numbers need to be:

* tightened to the P5.10 production budget (20 seeds, n_pts=40 000,
  Donaldson iters=100, tol=1e-6),
* equipped with full bootstrap percentile + BCa CIs per measurement
  (n_resamples=10 000, seed=12345 — matches P5.10).

This run delivers exactly that.

## 2. Production settings (concrete values)

| Parameter | Value | Provenance |
| --- | --- | --- |
| `--n-seeds` | 20 | P5.10 roster: `[42, 100, 12345, 7, 99, 1, 2, 3, 4, 5, 137, 271, 314, 666, 1000, 2024, 4242, 57005, 48879, 51966]` |
| `--n-pts` | 40 000 | P5.10 production-canonical |
| `--donaldson-iters` | 100 | P5.10 production-canonical |
| `--donaldson-tol` | 1e-6 | P5.10 production-canonical |
| `--boot-resamples` | 10 000 | P5.10 production-canonical |
| `--boot-seed` | 12345 | P5.10 production-canonical |
| `--boot-ci-level` | 0.95 | default (95% CI) |
| `--use-gpu` | false (CPU) | matches predecessor; GPU path not exercised here |
| Features | `gpu precision-bigfloat` | matches predecessor build flags |

## 3. Smoke-test runtime (single seed at production settings)

A 1-seed warm-up at `--n-pts 40000 --donaldson-iters 100 --donaldson-tol 1e-6`
(seed 42, `--boot-resamples 1000`) wall-clocked at:

* **488 s ≈ 8.13 min per seed** (CPU only, no GPU).

That covers all 14 σ-evals per seed (Experiment A: 8 TY trunc rows + 5
Schoen trunc rows; Experiment B: 6 native k-scan rows; the k=3 native
TY/Schoen pairs are reused by both experiments — Exp B duplicates 1
TY pair and 1 Schoen pair, so per-seed work is closer to 14 unique evals).

The smoke output `output/p_basis_convergence_smoke.json` confirms the
plumbing end-to-end (bootstrap CIs report `n/a` correctly for n=1, and
all numerical values land in the expected range).

## 4. Production extrapolation

* **Per-seed** = 488 s (smoke wall-clock).
* **20 seeds × 488 s** = 9 760 s ≈ **2.71 hours** σ-eval wall-clock.
* **+ Bootstrap pass** (B=10 000 × 65 groups × mean over ≤20 samples) ≈ negligible (< 30 s in p5_10's analogous pipeline).
* **Expected total wall-clock ≈ 2.75 h.**

The smoke binary is single-threaded for the σ-eval driver but uses
rayon-parallel sample integration internally; CPU pegged at ~700 % during
the warm-up. Seed-level parallelism is NOT exercised (each seed runs
sequentially) so there's no need to throttle for memory.

## 5. Process / launch metadata

| Field | Value |
| --- | --- |
| **Launch wallclock** | 2026-04-30 17:55:06 (local) |
| **Bash wrapper PID** | 17269 (Git-Bash subshell that spawned the binary) |
| **Bash subshell PID** | 17270 (proxy under nohup) |
| **Windows process PID** | **1904708** (`p_basis_convergence_diag.exe`) |
| **Log path** | `logs/p_basis_convergence_prod.log` |
| **JSON output target** | `output/p_basis_convergence_prod.json` |
| **CSV output target** | `output/p_basis_convergence_prod.csv` (auto-derived from `--output`) |
| **Estimated completion** | 2026-04-30 ~20:40 (≈ 2:45 from launch) |
| **Disowned?** | yes — survives this agent's termination |

Launch command (canonical, reproduce-with):

```bash
cd book/scripts/cy3_substrate_discrimination/rust_solver
nohup ./target/release/p_basis_convergence_diag.exe \
    --n-seeds 20 --n-pts 40000 \
    --donaldson-iters 100 --donaldson-tol 1e-6 \
    --boot-resamples 10000 --boot-seed 12345 \
    --output output/p_basis_convergence_prod.json \
    > logs/p_basis_convergence_prod.log 2>&1 &
disown
```

## 6. CLI changes vs predecessor

The binary was extended (commit-pending) with three production knobs:

* `--n-seeds N` — convenience shorthand for the first `N` seeds of the
  P5.10 roster. `--seeds` (explicit comma list) still wins when
  `--n-seeds` is unset.
* `--boot-resamples` (default 10 000), `--boot-seed` (default 12345),
  `--boot-ci-level` (default 0.95) — bootstrap config for percentile +
  BCa CIs on every `GroupSummary`.
* `--output PATH` — single-output ergonomic mirroring p5_10. If set,
  overrides `--json-output` and auto-derives a sibling `.csv` path.

`GroupSummary` JSON now carries `sigma_pct_ci_low/high`, `sigma_bca_ci_low/high`,
`sigma_boot_resamples`, `sigma_ci_level`. `ConfigEcho` records the boot
config so downstream analysis can reproduce CI computation from the
JSON alone.

No changes to production source modules (`schoen_metric.rs`, `ty_metric.rs`,
`cy3_donaldson_gpu.rs`, `bayes_factor_multichannel.rs`, `yukawa_*.rs`).
The truncation override path in `route34::basis_truncation_diag` is
unchanged. Production code remains bit-identical when the override is
unset (the universal case for non-diagnostic callers).

## 7. Post-process task (P-BASIS-CONVERGENCE-prod-postprocess)

**Trigger:** when `output/p_basis_convergence_prod.json` lands AND the
log file ends with `JSON written: ...`.

**Inputs:**

* `output/p_basis_convergence_prod.json` — manuscript-grade per-group
  stats (mean, SE, percentile + BCa 95% CI, raw seed sigmas).
* `output/p_basis_convergence_prod.csv` — raw 280-row table (20 seeds ×
  14 (cand, k, n_b) groups, ± k-scan duplicate native rows).
* `logs/p_basis_convergence_prod.log` — verdict-section console output.

**Steps:**

1. **Update `references/p_basis_convergence_diagnostic.md`** Sections
   3.1, 3.2, 4 with new tables that include `mean σ`, `SE`, `pct95 CI`,
   `BCa95 CI` columns. Add a "Production rerun (20 seeds × n_pts=40 000,
   Donaldson tol=1e-6, B=10 000)" header so the predecessor numbers
   remain visible for reproducibility audit.
2. **Recompute the matched/native ratio** from the new means; report
   it together with a bootstrap-derived CI on the ratio (resample
   seeds → recompute Δσ ratio → BCa). The 86% reduction figure becomes
   "X% reduction, 95% BCa CI [lo, hi]".
3. **Recompute Δσ(k=3)/Δσ(k=2) and Δσ(k=4)/Δσ(k=2)** with bootstrap
   CIs and re-state the 1/k² scaling-violation conclusion accordingly.
4. **Update `references/cy3_publication_summary.md` Section 2** with
   the manuscript-ready quote — exact form pending the new numbers,
   but template:

   > Direct measurement at matched basis size (n_b=27, k=3) places
   > **{X}% (95% BCa CI [{lo}, {hi}])** of the native TY-vs-Schoen σ
   > gap in the basis-size-artefact bucket. The native k-scan ratio
   > Δσ(k=3)/Δσ(k=2) = **{r3}** (95% BCa CI [{r3_lo}, {r3_hi}])
   > strongly violates the 1/k² ABKO 2010 prediction (0.444),
   > corroborating the artefact-dominated interpretation.

5. **Bayes Factor follow-through:** confirm that the Section 5.1
   "σ stays excluded" verdict still stands. If for any reason the
   tightened numbers shift the verdict (matched/native ratio > 0.5),
   escalate before merging — that would be a manuscript-significant
   reversal.
6. **Verify CSV row count** == 280 (20 seeds × 14 groups). If lower,
   inspect the log for any per-seed `ERROR:` lines and document them.
7. **Commit** the updated reference docs with message:

   ```
   docs(cy3): production-grade P-BASIS-CONVERGENCE numbers

   Refreshes p_basis_convergence_diagnostic.md and
   cy3_publication_summary.md Section 2 with 20-seed × n_pts=40k ×
   tol=1e-6 results plus full BCa 95% CIs (B=10 000, seed=12345).
   ```

   Do **not** force-push. Single commit, no amend.

**Artefacts to NOT modify in this post-process:**

* the binary source (already production-ready),
* `route34/basis_truncation_diag.rs` (RAII guard),
* any production solver under `route34/`.

## 8. Failure modes / babysit checklist

* If the log accumulates `ERROR:` lines for specific (cand, k, n_b,
  seed) tuples, the corresponding sigma is dropped from that group's
  ensemble (n_seeds<20 in the JSON). The post-process step 6 catches
  this; one or two seed losses don't invalidate the run.
* If the binary OOMs (TY k=4 native is the heaviest at n_basis=200 ×
  40 000 pts), the process will exit non-zero and the JSON will not
  land. Re-trigger with `RAYON_NUM_THREADS=4` (or smaller) and a fresh
  output path.
* If wall-clock exceeds 4 h with no completion, suspect a single-seed
  Donaldson stall — the per-seed cap is roughly 2× the smoke baseline.
  `tail -f logs/p_basis_convergence_prod.log` will reveal which group
  is stuck.

## 9. Reproducibility hash anchors

* Smoke run wall-clock: 488 s (Δσ matched-n=27 = +4.3846 at single seed=42; Δσ native = -7.928).
  These numbers will move under the 20-seed ensemble — noted here only
  to confirm the smoke binary is producing real σ-eval output, not a
  trivial pass-through.
* Cargo build: `cargo build --release --features "gpu precision-bigfloat" --bin p_basis_convergence_diag` (44.69 s on the launch host).
* Cargo test: `cargo test --release --features "gpu precision-bigfloat" --lib basis_truncation_diag` — 6/6 pass.
