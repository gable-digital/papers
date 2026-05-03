# P-K5-RELAUNCH

**Filed:** 2026-04-30 (task issuer) — launched 2026-05-01 17:54 local.
**Status:** RUNNING.
**Predecessor:** PID 388840 (`p5_10_k5_damped.json` target) died between
cycles 7 and 8 of the TY-bundle research before completion. Partial output
captured 13/20 TY seeds at σ ≈ 0.29 — consistent with the k=3 (~0.27) and
k=4 (~0.28) trajectory; both prior k's were bundle-size-saturated. No
Schoen k=5 data was captured, so n-σ at k=5 is currently not computable.

## This relaunch

- **Binary:** `target/release/p5_10_ty_schoen_5sigma.exe`
  (rebuilt from current source with `--features "gpu precision-bigfloat"`,
  build clean, 14 lib warnings — pre-existing).
- **PID:** 832016 (verified alive; ~2.33 GB RSS at start, GPU active).
- **Launch flags:**
  ```
  --ks 5 --n-pts 40000 --donaldson-iters 200 --donaldson-tol 1e-6 --use-gpu
  ```
  Matches the previous k=4 damped sweep that produced 3.82σ Tier 0,
  except `donaldson-iters` is 200 instead of 100 to give the larger k=5
  basis (n_basis = 385 TY, ~75 Schoen) headroom under the 1e-6 tolerance.
  Bootstrap defaults (B=10000, boot_seed=12345) inherited from CLI default.
- **Output JSON target:** `output/p5_10_k5_damped_relaunch.json`
  (binary writes JSON only on clean exit; partial runs leave only
  `.log` and `.replog`).
- **Stdout/stderr log:** `output/p5_10_k5_damped_relaunch.log`.
- **Seeds:** the binary's hard-coded n=20 list:
  `[42, 100, 12345, 7, 99, 1, 2, 3, 4, 5, 137, 271, 314, 666, 1000, 2024,
  4242, 57005, 48879, 51966]`. Same list as the prior P-K5-launch.

## Smoke test (before production launch)

- Settings: `--ks 5 --n-pts 5000 --donaldson-iters 30 --donaldson-tol 1e-4 --use-gpu`.
- Result (1st seed): seed=42, σ=0.349108, n_basis=385, iters=25,
  res_donaldson=7.084e-5, **tier=converged**, wallclock 58.14 s.
- Verdict: binary still converges at k=5 with the post-2bc342b1
  `schoen_tuple_for_k` extension. Smoke killed after seed 1 — production
  launched.

## Wall-clock estimate

- Prior PID 388840 averaged ~600 s/seed before death. Expected total:
  - 20 TY × ~600 s + 20 Schoen × ~600 s = **~6 h 40 min** (lower bound).
  - With `donaldson-iters=200` headroom, some seeds may push to ~800 s,
    so plan for **~7–9 h** end-to-end.
- ETA window (launched 2026-05-01 17:54 local): **2026-05-02 ~01:00–03:00 local**.
- The binary is single-threaded on the CPU side; GPU does the T-operator.

## Concurrency

- `p_basis_convergence_diag.exe` (PID 1904708, basis-convergence prod,
  CPU-only) verified intact post-launch — `448 K` → `657 K` RSS, normal
  growth, no crash. CPU vs GPU resource split, no contention.
- No other GPU competitors detected at launch time.

## Forward pointer

- Post-process task: `references/p_k5_postprocess_task.md` — extend its
  inputs to include `output/p5_10_k5_damped_relaunch.json` (and
  `.log`/`.replog`) when the JSON lands. The postprocess task ID for
  this relaunch is filed below as **P-K5-RELAUNCH-POSTPROCESS**.
- On completion, update `references/p8_4_k4_production.md` (or a new
  `p_k5_production_results.md` doc) with the supporting-tier σ.
- If σ strengthens the picture, update
  `references/cy3_publication_summary.md` §1 Headlines.

## Out of scope

- No source modifications.
- No git commits, no push, no MEMORY.md update (per task constraints).
