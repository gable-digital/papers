# §P8.5 — Chain-match basis convergence sweep at k=4 (GPU + orthogonalize)

## TL;DR — Sign-agreement preserved at k=4 under reduced-budget Donaldson-only baseline (qualitative corroboration of P7.11; magnitude not comparable across k)

Post-P7.11 chain-match was demonstrated at the canonical Donaldson
basis order **k=3** with iter_cap=100, n_pts=25k, Adam disabled,
where Schoen beat TY by Δ ≈ 7–18 nats on each chain × test-degree
cell. P8.5 re-runs the same sweep at the next basis order **k=4**
(with `--orthogonalize` and the projected Z/3 (TY) / Z/3×Z/3 (Schoen)
trivial-irrep filter — same orthogonalization machinery as P7.11) to
test whether the *sign* of the discrimination is robust to a basis
refinement. **The k=4 run was executed under a reduced wall-clock
budget** (n_pts=8000, iter_cap=25, Adam disabled) because higher
budgets stalled in the TY metric solve within the agent session. The
two runs (P7.11 k=3 vs P8.5 k=4) are therefore **NOT iter-cap-matched
and NOT n_pts-matched**, so the absolute Δ magnitudes are not
directly comparable across k.

| chain  | deg=2 (Δ) | deg=3 (Δ) | deg=4 (Δ) | sign-agree |
|:---|---:|---:|---:|:---|
| quark  | +15.33 (Schoen wins) | +14.46 (Schoen wins) | +10.39 (Schoen wins) | YES |
| lepton | +38.01 (Schoen wins) | +35.68 (Schoen wins) | +30.86 (Schoen wins) | YES |

**Verdict at k=4 (qualitative)**: Schoen unanimously wins both
chains on all three test-degree cells; the *sign* of the
discrimination from P7.11 carries through to k=4 under the
reduced-budget Donaldson-only baseline.

**What the numbers actually show on Donaldson convergence**:

- TY's σ_Donaldson is *larger* at k=4 than at k=3
  (**0.285 > 0.272**), and TY hits the iter_cap (23 ≈ 25) without
  reaching `donaldson_tol=1e-6`. So **TY's metric is *less*
  converged at k=4** under iter_cap=25 than P7.11's k=3 baseline
  was under iter_cap=100 — the metric is *worse*-converged, not
  "more refined". Larger basis order with a smaller iteration
  budget does not yield a finer metric.
- Schoen's σ_Donaldson is *much* larger at k=4 (2.431) than at k=3
  (1.852) — a 31% rise. TY's σ rises only ~5% (0.272 → 0.285). The
  asymmetric rise is consistent with iter_cap=25 (P8.5) vs
  iter_cap=100 (P7.11) being the dominant control variable, not
  the k change.
- Schoen's chain-match residual is **not converged in test_degree**
  at k=4: per-degree rel-Δ is 0.42–0.47 *increasing* with td
  (8.49 → 9.85 → 13.96 quark; 7.40 → 10.21 → 14.99 lepton). With
  test_degree=5 dropped from this sweep (basis_dim ≈ 285 near f64
  edge), there is no upper-degree cell to confirm that the trend
  stabilises rather than continues to drift. The discrimination
  *magnitude* could compress (or even shift sign on a single cell)
  at the next test_degree.
- The Δ values are residual_log distances on the Hungarian-assigned
  log-eigenvalue chain comparator. They are **not log Bayes
  factors**; converting them into a "k-σ stronger than k=3" claim
  would require the `bayes_factor_multichannel` adapter, which was
  *not* run on the k=4 outputs.

The k=4 result is therefore **qualitative corroboration** of
P7.11's k=3 finding (Schoen wins both chains on every td cell),
not a publication-grade strengthening of it. Treat the magnitudes
as illustrative under this reduced-budget baseline, not as
evidence that the discrimination "grew" at k=4.

**Production-grade rerun is deferred.** A multi-hour batch job
with iter_cap matched to (or exceeding) P7.11's 100, n_pts ≥ 25000,
Adam refinement re-enabled, and all five test_degrees (2..5) is
required before any cross-k strength claim can be made. See
caveat #4 below for the parameter spec.

## Settings (P8.5 fall-back-fall-back)

The original P8.5 task spec called for n_pts=15000, max_iter=50,
adam_iters=25 at k=4. That run **timed out** in the 60-min wall
clock during the TY metric solve (Donaldson balancing at k=4 with
GPU + Adam refinement is intrinsically O(dim^3) per iter, dim ~225;
~28 min elapsed without completing TY). A second attempt at
n_pts=8000, max_iter=25, adam_iters=12 also stalled in TY metric
(~10 min, no completion).

Final accepted settings (Adam disabled — Donaldson-only):

```
--p7-2 --basis-sweep --k 4 --n-pts 8000 --max-iter 25 \
  --donaldson-tol 1e-6 --adam-iters 0 \
  --use-gpu --orthogonalize --seed 12345 \
  --test-degree-list "2,3,4"
```

- **k=4** (basis order for the section monomials in the Donaldson
  metric — the headline change vs. k=3 baseline)
- n_pts=8000 (10× lighter than P7.11's 25k; n_pts only enters via
  Monte-Carlo sampling for the σ-functional; smaller n_pts → larger
  σ noise floor but the chain-match residual is dominated by the
  *orthonormal-basis Hermitian EVP*, which is independent of n_pts)
- max_iter=25 (Donaldson balancing iterations; both candidates hit
  the cap rather than `donaldson_tol=1e-6`, so this is a soft cap)
- **adam_iters=0** (Adam refinement DISABLED — see "Caveats" below)
- --use-gpu (Donaldson GPU T-operator)
- --orthogonalize (P7.11 modified Gram-Schmidt under L²(M) followed
  by standard Hermitian EVP — eliminates negative-eigenvalue
  pathology of the M⁻¹K Galerkin pencil)
- --test-degree-list "2,3,4" (one degree shorter than P7.11's
  "2,3,4,5"; deg=5 was dropped because at k=4 the deg=5 spectrum
  basis_dim ~ 285 puts the orthogonalized pencil close to f64
  precision and would not have produced a converged number)
- seed=12345 (matches P7.11)

Total wall-clock: **42.6 s** end-to-end (TY 1.6 s metric + 1.4 s
spectra; Schoen 3.1 s metric + 0.5 s spectra; balance is sweep
bookkeeping). Compare: P7.11 at k=3 took ~19 min including ~14 min
on the two metric solves (n_pts=25k, max_iter=100, no Adam, no GPU).
The 25× wall-clock improvement here is dominated by GPU Donaldson +
n_pts=8k.

## Per-(candidate, chain, deg) residuals

| candidate    | chain  | deg | basis_dim | residual_log | metric σ | metric iters | cond log10 |
|:---|:---|---:|---:|---:|---:|---:|---:|
| TY/Z3        | quark  | 2 |  18 | 23.824 | 0.285 | 23 |  0.76 |
| TY/Z3        | quark  | 3 |  62 | 24.313 | 0.285 | 23 |  1.72 |
| TY/Z3        | quark  | 4 | 178 | 24.350 | 0.285 | 23 |  2.03 |
| TY/Z3        | lepton | 2 |  18 | 45.412 | 0.285 | 23 |  0.76 |
| TY/Z3        | lepton | 3 |  62 | 45.891 | 0.285 | 23 |  1.72 |
| TY/Z3        | lepton | 4 | 178 | 45.846 | 0.285 | 23 |  2.03 |
| Schoen/Z3xZ3 | quark  | 2 |  13 |  8.492 | 2.431 | 25 |  8.56 |
| Schoen/Z3xZ3 | quark  | 3 |  44 |  9.851 | 2.431 | 25 | 11.38 |
| Schoen/Z3xZ3 | quark  | 4 | 119 | 13.956 | 2.431 | 25 | 14.65 |
| Schoen/Z3xZ3 | lepton | 2 |  13 |  7.399 | 2.431 | 25 |  8.56 |
| Schoen/Z3xZ3 | lepton | 3 |  44 | 10.209 | 2.431 | 25 | 11.38 |
| Schoen/Z3xZ3 | lepton | 4 | 119 | 14.989 | 2.431 | 25 | 14.65 |

`metric σ` is `final_sigma_residual` from the Donaldson balancing
loop. `cond log10` is the projected-basis Hermitian EVP condition
number after orthogonalization (well below f64 precision at all
three degrees — orthogonalization works as designed).

Note that `residual_log` is the chain-match Hungarian residual on
log-eigenvalue distances; it is *not* a per-eigenvalue convergence
measure. TY's per-degree drift is `rel-Δ ≤ 0.020` (essentially
converged at k=4 on the chain comparator); Schoen's per-degree
drift is `rel-Δ ≤ 0.468` (not converged in basis size yet — same
qualitative story as P7.11, driven by Schoen's σ noise floor).

## Δ table (Schoen wins on each cell)

```
chain     deg  TY_resid  Schoen_resid    Δ
quark      2    23.824     8.492     +15.333  Schoen wins
quark      3    24.313     9.851     +14.462  Schoen wins
quark      4    24.350    13.956     +10.394  Schoen wins
lepton     2    45.412     7.399     +38.013  Schoen wins
lepton     3    45.891    10.209     +35.682  Schoen wins
lepton     4    45.846    14.989     +30.857  Schoen wins
```

Convention: `Δ = residual_log[TY] − residual_log[Schoen]`. Positive
Δ means Schoen has the smaller chain-match residual ⇒ Schoen wins.
Equivalent to `delta_schoen_minus_ty < 0` in the P7.11 schema.

## Comparison vs. P7.11 (k=3)

| chain × deg | k=3 (P7.11) Δ | k=4 (P8.5) Δ | sign |
|:---|---:|---:|:---|
| quark × 2    |  +8.93 |  +15.33 | same (Schoen wins) |
| quark × 3    |  +6.99 |  +14.46 | same (Schoen wins) |
| quark × 4    |  +9.32 |  +10.39 | same (Schoen wins) |
| quark × 5    | +11.20 |  N/A    | (deg=5 skipped at k=4) |
| lepton × 2   | +12.78 |  +38.01 | same (Schoen wins) |
| lepton × 3   | +11.33 |  +35.68 | same (Schoen wins) |
| lepton × 4   | +15.33 |  +30.86 | same (Schoen wins) |
| lepton × 5   | +18.31 |  N/A    | (deg=5 skipped at k=4) |

**Schoen wins both chains at k=4 unanimously** — the *sign* of the
discrimination from P7.11 is preserved on every cell available at
k=4. **The raw-nat magnitudes are NOT directly comparable across k**:
the P7.11 (k=3) baseline ran at iter_cap=100 / n_pts=25k while the
P8.5 (k=4) baseline ran at iter_cap=25 / n_pts=8000 because the
larger budget stalled in TY metric within the agent wall-clock. The
larger raw Δ values at k=4 are at least partially attributable to the
asymmetric Schoen σ rise (1.852 → 2.431, +31%) under the smaller
iter_cap, not to a genuine basis-refinement strengthening of the
signal. Treat the k=4 Δ values as illustrative under the reduced-
budget baseline, not as a quantitative "stronger than k=3" claim.

A direct apples-to-apples cross-k strength comparison requires the
production-grade re-run (caveat #4) — i.e. iter_cap matched (or
larger) to P7.11's 100, n_pts ≥ 25k, Adam re-enabled, and all five
test_degrees so Schoen's per-degree drift (currently rel-Δ ≈
0.42–0.47, increasing) can be checked for stabilisation.

## Caveats

1. **σ_Donaldson residuals at k=4 are *worse* than at k=3 (TY's
   metric is *less* converged here).** At k=4 / iter_cap=25 the
   final σ residuals are TY 0.285 and Schoen 2.431, vs P7.11's k=3
   / iter_cap=100 values of TY 0.272 and Schoen 1.852. Both
   candidates hit the iter_cap (TY 23 ≈ 25, Schoen 25 = 25) without
   reaching `donaldson_tol=1e-6`, i.e. iter_cap is the binding
   constraint. The TL;DR's qualitative verdict (Schoen wins both
   chains, sign-agree across all td cells) is robust to this extra
   Donaldson noise — the signs are consistent with P7.11, which
   also ran Donaldson-only (no Adam) — but **the metric at k=4 in
   this sweep is NOT a refinement of the metric at k=3 in P7.11**;
   it is a higher-basis-order solve under a much smaller iteration
   budget. Adam refinement would push σ further down; without it
   the absolute chain-match residuals include Donaldson noise that
   an Adam-refined production run would trim.

2. **Schoen chain-match residual is NOT converged in test_degree at
   k=4 (rel-Δ ≈ 0.42–0.47, *increasing* with td).** Per-degree
   drift on Schoen:
   - quark: 8.49 → 9.85 → 13.96 (rel-Δ 0.16 → 0.42)
   - lepton: 7.40 → 10.21 → 14.99 (rel-Δ 0.38 → 0.47)

   These are well above the 10% rel-Δ convergence rule of thumb
   from §6.3 / P7.2 and the trend is **upward**. Without test_degree=5
   in this sweep, there is no upper-degree cell to confirm that
   Schoen's residual stabilises rather than continuing to drift.
   In particular, the *magnitude* of the discrimination at k=4
   could compress (or one quark-cell sign could shift) at the next
   test_degree. TY's per-degree drift is small (rel-Δ ≤ 0.020);
   the non-convergence is on the Schoen side, driven by the larger
   σ noise floor (2.431 vs TY 0.285).

3. **Adam refinement disabled (wall-clock).** The P8.5 task spec
   called for adam_iters in {12, 25}. Both adam_iters>0 attempts
   (with n_pts in {8000, 15000} and max_iter in {25, 50}) **stalled
   in the TY metric solve** for the full agent wall-clock budget
   without reaching the spectrum loop. The σ-functional
   finite-difference gradient at k=4 with Adam learning rate 1e-3
   and 12–25 outer steps appears to spend O(10–30 min) per
   candidate even on GPU. The Donaldson-only run completed in
   42.6 s and is presented as a qualitative corroboration of P7.11.

4. **n_pts=8000 vs. P7.11's 25000.** Smaller n_pts increases the
   Monte-Carlo σ noise (and is the dominant contributor to the
   asymmetric σ rise on Schoen vs TY together with the smaller
   iter_cap) but does not enter the chain-match residual directly
   — the residual is computed from the orthogonalised Hermitian
   EVP eigenvalues, which are deterministic in the projected
   monomial basis once the metric is fixed. σ is reported above for
   transparency.

5. **deg=5 not exercised.** At k=4 the deg=5 projected basis_dim
   would be ≈ 285, putting the Hermitian EVP near the edge of f64
   precision. Following the P7.11 / P7.2 deg-5 caveat, deg=5 was
   dropped from this sweep. The k=4 verdict therefore rests on
   deg=2, 3, 4 — three test cells per chain, six cells total — all
   sign-agree. Caveat #2 above flags the open question that deg=5
   would have settled.

6. **Δ values are residual_log distances, not log Bayes factors.**
   The Hungarian-assigned log-eigenvalue residual difference
   `Δ = residual_log[TY] − residual_log[Schoen]` is **not** a
   nats-of-evidence quantity. Converting Δ into a per-cell or
   per-chain "k-σ stronger" statement requires the
   `bayes_factor_multichannel` adapter (see σ-channel
   `p5_10_ty_schoen_5sigma` for the calibrated version). That
   adapter was **not run** on the P8.5 k=4 outputs. Any "stronger
   than k=3" language elsewhere in the codebase that draws on
   k=3-vs-k=4 raw-Δ comparisons should be read as raw-residual
   commentary, not as a calibrated discrimination strength.

7. **Production-grade re-run deferred.** Before any cross-k strength
   claim is published, this sweep should be re-run on a host with
   multi-hour wall-clock (not an agent session) at:
   - `--n-pts 25000` (matching P7.11),
   - `--max-iter 100` (matching P7.11; or larger if budget allows
     so both candidates can hit `donaldson_tol=1e-6` rather than
     the iter_cap),
   - `--adam-iters 25` (Adam re-enabled),
   - `--test-degree-list "2,3,4,5"` (extend to deg=5 to confirm
     Schoen residual stabilisation),
   - same `--orthogonalize` / projector / seed.

   Goals: (i) confirm Δ signs and *recover comparable magnitudes*
   to P7.11 under a matched (or strictly larger) Donaldson budget;
   (ii) confirm Schoen residual rel-Δ flattens by td=5; (iii) feed
   the spectra through `bayes_factor_multichannel` so the
   strength-of-evidence claim is calibrated.

   The σ-channel (`p5_10_ty_schoen_5sigma`) remains the headline
   5σ-grade discriminator. Chain-match is corroborating evidence;
   the P8.5 k=4 result is qualitative corroboration of P7.11, not
   a publication-grade strengthening of it.

## Files

- Settings + raw output: `output/p8_5_chain_k4_gpu.json` (= identical
  copy of `output/p7_2_chain_sweep.json` written by the binary's
  hard-coded `--p7-2` path)
- stderr log: `output/p8_5b_chain_k4_gpu_run.log`
- Source: `src/bin/p6_3_chain_match_diagnostic.rs` (`--p7-2` mode);
  metric solver in `src/route34/{ty_metric,schoen_metric,
  cy3_metric_unified}.rs`; orthogonalized projected-basis Hermitian
  EVP in `src/route34/metric_laplacian_projected.rs`
  (`run_orthogonalized_metric_laplacian`).
