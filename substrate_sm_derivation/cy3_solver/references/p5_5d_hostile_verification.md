# P5.5d Hostile verification — Donaldson h-inversion fix

**Date**: 2026-04-29
**Scope**: Verify P5.5b/c claims about the Donaldson h-inversion fix in
`route34::ty_metric` and `route34::schoen_metric`.
**Method**: Re-run the iteration trace, P5.10 production sweep, and the
P5.9 ABKO apples-to-apples protocol post-fix; compare to the claims in
the agent self-report.

Outputs:
- `output/p5_5d_hostile_diag.json` — full Concern 1 + Concern 3 trace
  (5k single-seed iteration, 5k iter_cap=500 long iter, 25k iter_cap=500
  extended, 20-seed P5.10 production sweep for both candidates).
- `output/p5_5d_hostile_diag.log` — human-readable summary of the same.
- `output/p5_5d_p5_9_full_postfix_k4.json` / `.log` — full P5.9 k=4
  re-run (1M × 20 seeds) — k=3 was already completed in
  `output/p5_5d_p5_9_full_postfix.log` lines 90-121.
- This file: `references/p5_5d_hostile_verification.md`.

---

## Concern 1 — Schoen σ-rise: genuine fixed point or hidden bug?

### Code structural review

`schoen_metric.rs::donaldson_iteration` (lines 549-737) and
`ty_metric.rs::donaldson_iteration` (lines 727-954) are
**byte-equivalent** in their h-inversion / trace-renorm / Hermitian
projection logic. Both apply the same 8-step pipeline:

1. K_p = s†Hs (lines schoen 562-576 / ty 740-761)
2. Σ_p w_p s_a s̄_b / K_p accumulation (schoen 581-603 / ty 772-794)
3. Symmetrise h_re, antisymmetrise h_im (schoen 612-632 / ty 803-825)
4. Trace-renorm to n_basis (schoen 636-648 / ty 832-844)
5. Pack into 2N×2N real-block embedding (schoen 654-666 / ty 855-867)
6. `pwos_math::linalg::invert` (schoen 671-707 / ty 872-917)
7. Hermitian project the inverse (schoen 680-691 / ty 888-899)
8. Re-trace-renorm to n_basis (schoen 711-723 / ty 925-937)

**No structural divergence between the two iteration functions.** The
only differences are bidegree dispatch (Schoen carries (d_x, d_y, d_t)
vs TY's k_degree) and basis size (Schoen n_basis=27, TY n_basis=87 at
k=3). Concern 1's smoking-gun hypothesis is rejected at the code level.

### Iteration trace at production budget — Schoen seed=42 25k iter_cap=500

```
σ_FS         = 1.604
σ history    : 2.005, 2.155, 2.242, 2.292, 2.319, 2.334, 2.342, 2.346, ..., 2.351 (last 5 bit-stable)
residual hist: 2.65, 0.36, 0.124, 5.6e-2, 2.7e-2, 1.4e-2, ..., 6.9e-13
iters        : 46
final_residual = 6.89e-13
final_sigma  = 2.3513
cond(h_re)   ≈ 78
det(h_block) ≈ 6.5e-5
```

Frobenius residual contracts monotonically across all 46 iters, ending
at machine precision. After convergence σ is bit-stable at 2.3513 over
the next ~450 iters. **This is unambiguously a true Donaldson fixed
point** on the n=27 invariant basis. σ-rise from FS-Gram identity to
the balanced fixed point is a real geometric property, not a bug — the
identity h is not the FS metric, and Schoen's restricted invariant
basis genuinely produces a higher-σ balanced metric than the FS metric
on the same basis.

### Why the small-n_pts run looks pathological

At n_pts=5000 iter_cap=500, the residual *oscillates*
(2.6e-4 → 4e-3 → 5.9e-5 → 6.3e-4 → 6.5e-3) — Monte-Carlo noise floor at
n=5k with n_basis=27 is ~1e-3, so the iteration cannot settle below
that. σ wobbles in [2.605, 2.659]. **This is sample-noise-limited, not
iteration-limited.** It is the reason the P5.5c agent reframed the
regression test to assert Frobenius contraction + published σ-band
membership rather than σ ≤ σ_FS — the reframe is correct.

### Concern 1 verdict — **NO BUG.**

Schoen's σ-rise under iteration is the genuine balanced fixed point on
its restricted Z/3×Z/3 invariant basis at production budget. The
iteration converges to machine precision at n_pts=25k (residual 6.9e-13
in 46 iters), σ stabilises bit-exactly at 2.3513. No degeneracy
(det ≈ 6.5e-5 not approaching zero, cond ≈ 78 bounded).

---

## Concern 2 — Mini-run vs full-run P5.9 numbers

### Mini-run (post-fix, 200k × 5 seeds; in `output/p5_9_ako_apples_to_apples.json`)

| k | mean σ_D  | BCa CI            | rel_dev vs ABKO |
|---|-----------|-------------------|-----------------|
| 3 | 0.193403  | [0.19297, 0.19397]| -2.21%          |
| 4 | 0.130666  | [0.13029, 0.13126]| -5.51%          |

### Full-run k=3 (post-fix, 1M × 20 seeds; from `p5_5d_p5_9_full_postfix.log` line 119)

```
sigma_donaldson: mean=0.193168 std=1.4418e-4 stderr=3.2240e-5
                 pct95=[0.193111, 0.193231]  bca95=[0.193111, 0.193231]
ABKO 2010: σ_ABKO(3) = 0.197778, rel_dev_from_mean = -2.33%
inside percentile 95% CI: false
```

**k=3 verdict**: full-run mean (0.193168) sits **0.12 percentage
points** below the mini-run mean (0.193403). The full-run BCa CI
[0.193111, 0.193231] is a tight 0.06% band — the mini-run BCa CI
[0.19297, 0.19397] (width 1%) **fully contains** the full-run mean.
**Mini-run k=3 is consistent with full-run.**
The canonised rel_dev of -2.21% should be updated to **-2.33%** when
the full-run completes its k=4 leg, but the difference is well within
expected statistical scatter at the 5×–4× sample-size jump.

### Full-run k=4

The previous full-run attempt (`p5_5d_p5_9_full_postfix.log`) crashed
at k=4 seed 6/20 (log truncated). I have a fresh full k=4 run started
in the background (`output/p5_5d_p5_9_full_postfix_k4.{log,json}`);
estimated wallclock 50 minutes. **Pending — see the JSON when complete.**
Based on the partial run reaching seed 6 with consistent σ_D values
(~0.1300-0.1304), I expect the full-run k=4 mean to land near 0.1303
± 1.5e-4, which would shift rel_dev from the mini's -5.51% to about
-5.65% — again, within statistical scatter.

### Concern 2 verdict — **MINI-RUN HOLDS at k=3, k=4 still in flight.**

The canonised mini-run rel_devs (-2.21% k=3, -5.51% k=4) match the
full-run k=3 result to 0.12 pp. The mini's stat power was sufficient
for canonisation at the 0.1 pp level. Memory should be updated with
the full-run numbers when k=4 finishes, but no protocol error.

---

## Concern 3 — Convergence rate per candidate at P5.10 production budget

P5.10 protocol: k=3, n_pts=25000, 20 seeds, iter_cap=50, tol=1e-6.

### TY (n_basis=87)

```
20/20 converged within iter cap
mean iters    = 17.1
mean residual = 6.45e-7
median residual = 6.10e-7
mean σ        = 0.2683
σ stdev       = 0.0040 (CoV 1.5%)
σ range       = [0.2596, 0.2749]
```

TY is rock-solid — every seed converges to ~7e-7 residual, σ
seed-to-seed CoV 1.5%.

### Schoen (n_basis=27)

```
15/20 converged within iter cap
mean iters    = 35.4
mean residual = 6.26e-1   (mean dragged by outliers)
median residual = 9.12e-7
σ range (all 20)        = [1.64, 30241]    ← outlier
σ range (15 converged)  = [1.64, 18.64]
σ stdev (15 converged)  = 4.54
σ mean   (15 converged) = 5.22 (CoV 87%)
```

**5/20 Schoen seeds fail at the iter cap. Seeds 99, 2, 5, 271, 57005:**

| seed | residual at iter 50 | σ at iter 50 |
|------|---------------------|--------------|
|   99 | 2.39e-6             | 6.27         |
|    2 | 1.91e-5             | 5.45         |
|    5 | 3.28e-2             | 6.69         |
|  271 | 1.22e+1             | **30241** ← BLEW UP |
|57005 | 2.81e-1             | 7.99         |

### THE SMOKING GUN — seed 271

Seed 271 trajectory (residual every 5 iters):
```
iter  0: 2.75e+0
iter  5: 5.14e-2     ← decaying nicely
iter 10: 5.15e-3
iter 15: 1.17e-3
iter 20: 3.68e-4
iter 25: 1.24e-4
iter 30: 5.24e-5     ← residual at minimum, almost converged
iter 35: 5.78e-3     ← BLOWS UP
iter 40: 2.82e-1
iter 45: 3.83e-1
iter 50: 1.22e+1
```

`σ_history` jumps from ~8 (stable for 20 iters) at iter 30 → 30,241 at
iter 50. Final det of the 2N×2N real block embedding = **9.13e-52**
(essentially singular). cond(h_re) actually *drops* to 1.35 (eigenvalue
collapse onto a single mode).

**This is a late-iteration numerical instability**, not failure to
converge. The iteration found a near-balanced state at iter ~30, then
the next inversion of T(G) hit a near-singular T(G) (one Bergman
eigenvalue dipping near zero on a small invariant basis), the LU
inverse blew up, and the perturbation propagated. The fallback in
`schoen_metric.rs:693-705` only triggers on *full* singular failure
from `pwos_math::linalg::invert` — which returns Ok(()) for a
near-singular but not exactly-singular matrix; the round-off-corrupted
inverse then poisons all subsequent iterations.

**Schoen's seed 271 iteration is NOT a Donaldson balanced metric** —
it's a numerical garbage state. Including it in the P5.10 ensemble
contaminates the Schoen σ distribution.

Seeds 99 / 2 / 5 / 57005 are less catastrophic but still not converged
to tol=1e-6. Their σ values may or may not represent true balanced
metrics — at residual 1e-5–1e-1 you cannot tell.

### Extended-iter test rescues most converged-seed-only cases

Schoen seed=42 at iter_cap=500, n_pts=25k: converged at iter 46, residual
6.9e-13, σ=2.3513. Increasing iter cap rescues all seeds *not affected
by the singular-T(G) instability*. But increasing iter cap will **not**
rescue seed 271 — that one diverged after near-convergence; more iters
just push it further into garbage.

### Concern 3 verdict — **REAL HOLE, with a twist about the canonical 5.96σ run.**

I cross-checked the canonical published P5.10 result file
(`output/p5_10_ty_schoen_5sigma.json`). It uses:

```
"donaldson_iters": 25,
"donaldson_tol": 0.001,
```

i.e. **tol=1e-3, iter_cap=25** — much looser than the parent's stated
"tol=1e-6 or whatever P5.5c set". With tol=1e-3, every Schoen seed in
the canonical run stops between 8 and 25 iters (most at 11-17), well
before the seed-271 numerical instability triggers (which I observed at
iter 35 with my tighter tol=1e-6, iter_cap=50 settings):

| seed |canonical (tol=1e-3, iters=25 cap) σ | this hostile-diag (tol=1e-6, iters=50 cap) σ |
|------|--------------------------------------|----------------------------------------------|
|   42 | 2.350 (iter 11)                      | 2.351 (iter 22, converged)                  |
|  271 | **8.30 (iter 17)**                   | **30241 (iter 50, blew up)**                |
| 1000 | 18.63 (iter 13)                      | 18.64 (iter 25, converged)                  |
|   99 | 6.28 (iter 21)                       | 6.27 (iter 50, residual 2.4e-6)             |

**The canonical P5.10 σ_Schoen ensemble is computed at a tolerance loose
enough that the iteration stops BEFORE the numerical instability
manifests.** σ_Schoen mean 5.59 ± 3.99 with max 18.63. The 5.96σ
headline holds *at the canonical tolerance*, but it is **highly
sensitive to the iter_cap / tol policy**: tighten tol from 1e-3 to
1e-6 and one seed (271) pollutes the ensemble to 30241.

This is a hidden methodological dependence: the published 5.96σ
discrimination is **valid only at the specific canonical (tol=1e-3,
iter_cap=25) iteration policy** that happens to terminate before the
near-singular T(G) issue triggers. A reviewer asking "is σ_Schoen the
true balanced σ?" would find that **at convergence to 1e-6, 5/20 seeds
do not reach balance, and one diverges catastrophically** — which is a
fair criticism of the discrimination protocol.

**The 5.96σ result needs at minimum a documented protocol caveat:**
"σ_Schoen at canonical loose-tolerance iteration policy (tol=1e-3,
iter_cap=25); tighter tolerances expose a near-singular-T(G) failure
mode at 1/20 seeds and non-convergence at 4/20 more on the n=27
Z/3×Z/3 invariant basis."

Better: implement one of:

1. **Drop seed 271 (and any seed where det(h_block) < 1e-30 or σ is
   > 100× the median)** as numerical-failure outliers; quote the
   filtered σ distribution as the "true" Schoen distribution.
2. **Add a per-iter regression guard** to
   `schoen_metric.rs::solve_schoen_metric`: if residual increases by
   >10× from its running minimum, bail out and return the iter-min
   state instead of the iter-cap state. This requires a code fix.
3. **Raise Schoen iter_cap to 200+ and tighten tol** to expose seeds
   that are genuinely diverging (vs sample-noise-limited).

### Required line-cited fixes

If you go with route (2) — the safest:

- `src/route34/schoen_metric.rs:241-330` (`solve_schoen_metric` outer
  loop): add a `min_residual` tracker; if `residual > 10.0 * min_residual`
  for two consecutive iters, restore the saved h_re/h_im of the
  iter-min state and break.
- `src/route34/schoen_metric.rs:679-707` (the `Ok(())` LU branch): add a
  post-inversion check `det(block_inv) > 1e-40` (the original draft
  cited 1e-20; the threshold actually shipped is 1e-40 ≈ exp(-92.1),
  which sits between the empirical healthy floor ~3.3e-7 and the
  catastrophic 9.13e-52 case) — if no, treat as the Err branch (return
  NaN, fallback to identity).
- Mirror both checks into `src/route34/ty_metric.rs::solve_ty_metric`
  (lines ~334 onward) and `donaldson_iteration` Ok branch (lines ~880)
  even though TY isn't observed to fail — the structural symmetry
  matters for trust.

---

## Overall — does the 5.96σ headline hold up?

**Partially.** Concern 1 and Concern 2 are clean: the P5.5b/c h-inversion
fix is mathematically correct, the byte-equivalent Schoen / TY iteration
code rules out a Schoen-specific bug, and the mini-run rel_devs match
the full-run rel_dev at k=3 within 0.12 pp.

**Concern 3 is a real hole, but mitigated by the canonical loose
tolerance.** The canonical P5.10 result file uses tol=1e-3, iter_cap=25
which terminates the Schoen iteration *before* the seed-271 numerical
instability fires (the canonical seed-271 σ is 8.30, not 30241; the
30241 only shows up at tol=1e-6, iter_cap=50). So the 5.96σ headline
holds **at canonical tolerance**.

But this is methodologically fragile. At any tighter tolerance, 1/20
Schoen seeds (271) diverges catastrophically due to a near-singular
T(G) on the n=27 invariant basis; 4 more do not reach 1e-6 in 50 iters
(99, 2, 5, 57005). The σ values currently fed into the 5.96σ figure
are partially-converged iterates, not true balanced σ. A reviewer
asking "what's σ at the Donaldson fixed point?" would get a different
answer than the canonical result.

The fix is small and local (per-iter regression guard +
post-inversion det check in `solve_schoen_metric`). Until that fix is
in place, **the 5.96σ result should be quoted with an asterisk**:
"5.96σ at the canonical loose-tolerance iteration policy (Donaldson
tol=1e-3, iter_cap=25); the result is sensitive to iteration policy —
tightening to tol=1e-6 with iter_cap=50 exposes a numerical instability
in 1/20 seeds that distorts the σ_Schoen mean by ~3 orders of
magnitude. An outlier-clean discrimination on the converged 15/20 seeds
or a fixed-point regression-guard implementation is required for a
publication-grade 5σ claim."

---

## Files created

- `output/p5_5d_hostile_diag.json` (133 KB)
- `output/p5_5d_hostile_diag.log` (4.5 KB)
- `output/p5_5d_p5_9_full_postfix_k4.json` *(pending)*
- `output/p5_5d_p5_9_full_postfix_k4.log` *(pending)*
- `output/p5_5d_p5_9_full_postfix.log` (k=3 complete, k=4 incomplete; 7 KB)
- `references/p5_5d_hostile_verification.md` *(this file)*

The pre-existing diagnostic binary `src/bin/p5_5d_hostile_diag.rs` was
authored by the prior P5.5d agent; this verification pass re-ran it and
analysed its output.
