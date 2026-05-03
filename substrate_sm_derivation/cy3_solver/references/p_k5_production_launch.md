# P-K5: k=5 σ-Discrimination Production Sweep — Launch Record

**Date:** 2026-04-30
**Status:** Production sweep launched (in flight, ~6-10 hr ETA).
**Output target:** `output/p5_10_k5_damped.json` + `output/p5_10_k5_damped.log`

## 1. Motivation

Continuum-extrapolation argument for the basis-size-artifact concern
(P-BASIS-CONVERGENCE Experiment B showed Δσ at k=2,3,4 = {−1.22,
−46.87, −9.40} — non-monotonic, basis-size-artifact-dominated). A
successful k=5 supporting tier alongside the standing k=3 (6.92σ)
headline and k=4 (3.82σ supporting) gives a 3-point trend that addresses
the artifact concern at higher k where conditioning is harder.

k=5 was previously deferred ("budget exhausted" per P8.4). The
P8.4-fix and P8.4-fix-c-d-e damping/Tikhonov machinery is now in place
and auto-engages at k≥4, so k=5 is plausibly feasible.

## 2. Code changes (this launch)

`src/bin/p5_10_ty_schoen_5sigma.rs`: extended `schoen_tuple_for_k`
to map k=5 → (d_x=5, d_y=5, d_t=2). Rationale:

- Preserves d_t=⌊k/2⌋ scaling from k=4 → k=5.
- Total bidegree d_x+d_y+d_t = 12 stays in the k=4-equivalent stall
  regime trigger band (≥10), so `auto_schoen_tikhonov` /
  `auto_schoen_damping` / `auto_schoen_gated_tikhonov` rules continue
  to fire as designed.

No other binary or library changes. The TY k-parameterization
(`enumerate_bigraded_kk_monomials(k)`) was already k-agnostic. The
auto-rules in `route34::ty_metric` and `route34::schoen_metric` fire
at `k_degree ≥ 4` and `d_x+d_y+d_t ≥ 10` respectively, so k=5 inherits
the k=4 damping/Tikhonov machinery automatically.

## 3. Native basis sizes (measured)

| Manifold | k | Bidegree    | n_basis (Z/3 ∩ Gröbner-reduced) | Source                        |
|----------|---|-------------|---------------------------------|-------------------------------|
| TY       | 5 | (5,5)       | **385**                         | Smoke seed 42, log line 7     |
| Schoen   | 5 | (5,5,2)     | not measured pre-launch         | Will appear in production log |

Reference scaling: TY k=4 → n_basis=200, k=5 → 385 (1.93× larger,
consistent with C(k+3,3)² growth less Z/3 + ideal projection).

## 4. Smoke test verdict

**Command:**
```
./target/release/p5_10_ty_schoen_5sigma.exe \
  --ks 5 --n-pts 5000 --donaldson-iters 30 --donaldson-tol 1e-4 \
  --use-gpu --output output/p5_10_k5_smoke.json
```

**Result:** TY k=5, seed 42, n_pts=5000:
- σ = 0.349108
- n_basis = 385
- iters = 25 (within the 30 cap)
- res_donaldson = 7.084e-5 (below tol=1e-4)
- tier = converged
- elapsed = 53.07 s

**Verdict:** ALIVE / CONVERGING. No NaN, no immediate divergence, no
hang. Auto-rule's adaptive damping (k≥4 trigger) successfully drives
TY k=5 to a converged state on the first seed at the smoke n_pts.
Smoke killed after the first seed to free the GPU for the production
sweep — Schoen k=5 basis size will appear in the production log
once TY's 20 seeds finish.

## 5. Production launch

**Command:**
```
nohup ./target/release/p5_10_ty_schoen_5sigma.exe \
  --ks 5 --n-pts 40000 --donaldson-iters 200 --donaldson-tol 1e-6 \
  --use-gpu --output output/p5_10_k5_damped.json \
  > output/p5_10_k5_damped.log 2>&1 &
disown
```

- `--ks 5`: only k=5 (k=3 and k=4 already in `output/p5_10_*`).
- `--n-pts 40000`: matches k=3 / k=4 production for cross-comparability.
- `--donaldson-iters 200`: bumped from k=4's 100 to absorb harder
  conditioning at the larger n_basis.
- `--donaldson-tol 1e-6`: matches k=3 / k=4 strict-converged criterion.
- `--use-gpu`: required for tractable wallclock at n_basis=385.
- `--donaldson-damping` flag is **not exposed by this binary**;
  damping is set by the auto-rule (Adaptive at k≥4). The instruction
  to "pass `--donaldson-damping 0.5` if supported" is moot here. If
  the adaptive ramp under-performs at k=5 (as it did at k=4 — 1.83σ
  vs static-α=0.5's 3.82σ), a P-K5-followup will rerun with a
  static-α override added to the binary. For now the auto-rule is the
  only path.

**Process:**
- Binary PID: **388840** (verified via `tasklist`).
- Working memory: ~2.3 GB at startup, expected to climb to ~4-6 GB.
- Log: `output/p5_10_k5_damped.log` (started, "--- TY k=5 (n_pts=40000) ---" header written, first seed in flight).

## 6. Expected runtime

Extrapolation from k=4 production (`p5_10_k4_damped.log`):
- TY k=4: n_basis=200, ~195-300 s/seed at n_pts=40000 (most converged in 47-50 iters).
- TY k=5: n_basis=385 (1.93× larger). Donaldson T-op solve cost
  scales O(n_basis²-n_basis³); sample evaluation O(n_basis × n_pts).
  Per-seed wallclock estimate: 400-800 s.
- 20 TY seeds + 20 Schoen seeds at k=5. Schoen basis is much smaller
  (estimated ~50-100), per-seed Schoen wallclock ~150-300 s.
- Total: 20×(400-800) + 20×(150-300) ≈ 11 000 – 22 000 s ≈ **3-6 hr** wallclock.
- Plus bootstrap (~seconds) and JSON write.

If the adaptive damping under-performs at k=5 the way it did at k=4
(stuck residual on Schoen, or σ blow-up), individual seeds may hit
the 200-iter cap and push runtime to ~10 hr. Watch the log for any
"hit iter cap" / "regression guard" patterns.

## 7. Forward pointer: P-K5-postprocess

When `output/p5_10_k5_damped.json` is finalized (the binary writes
JSON only on clean exit), spawn a P-K5-postprocess task to:

1. Read `output/p5_10_k5_damped.json`. Extract Tier 0 strict-converged
   ensemble (`final_donaldson_residual < tol AND iterations_run < cap
   AND n_Schoen ≥ 16 AND no guard snapshots AND no Tukey trim`).
2. Compute n-σ point estimate, percentile CI, and BCa CI on the
   strict tier (matches P5.10 v7 protocol).
3. Cross-check against k=3 (6.92σ Tier 0) and k=4 (3.82σ static-α=0.5
   damped). Compare basis sizes (k=3: ~50, k=4: 200, k=5: 385) and
   n-σ trend — does a basis-size-artifact correction collapse the
   trend the way Experiment B suggested at k=2,3,4?
4. Update `references/cy3_publication_summary.md`:
   - **§1 Headlines**: add k=5 supporting-tier n-σ alongside the
     existing k=3 publication headline.
   - **§4 Supporting evidence**: add k=5 row to the continuum table.
5. If k=5 σ has the same form as k=3 / k=4 (TY > Schoen, monotone
   in basis size), file the result as continuum-supporting evidence
   for the publication. If k=5 inverts (Schoen > TY) or collapses
   under basis-size correction, file as evidence that the ABKO-style
   normalization is required.
6. Re-evaluate adaptive vs static damping at k=5 — if adaptive
   under-performs, file a P-K5-static-damping followup that adds a
   `--donaldson-damping <f64>` CLI flag to `p5_10_ty_schoen_5sigma`
   and re-runs with α=0.5.

## 8. Verification

- Cargo build: clean (release + features `gpu precision-bigfloat`).
- Cargo check: clean (release + features `gpu precision-bigfloat`).
- Lib tests: not re-run (no library changes; only binary edit was
  extending a panic match-arm). The damping/Tikhonov auto-rule machinery
  was unchanged.
- Smoke test: 1 TY seed converged at n_pts=5000.

## 9. Files touched

- `src/bin/p5_10_ty_schoen_5sigma.rs` — `schoen_tuple_for_k` extended
  for k=5 → (5,5,2).
- `output/p5_10_k5_smoke.json` + `.log` — smoke artifacts (1 seed, killed early).
- `output/p5_10_k5_damped.json` (pending) + `.log` (in flight) — production artifacts.
- `references/p_k5_production_launch.md` — this file.

No commits in this session per task instructions.
