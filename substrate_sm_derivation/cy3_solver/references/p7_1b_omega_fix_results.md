# P7.1b — ω_fix gateway-eigenvalue multi-seed sweep on converged Schoen metrics

## Question

Journal §L.2 predicts a gateway-electron eigenvalue

    ω_fix = 1/2 − 1/dim(E_8) = 123/248 = 0.495967741935...

P7.1 (commit `db1b4ca1`) reported a Schoen Z₃×Z₃ lowest dominant-trivial-rep
eigenvalue of 0.5055 on seed=42 — a 1.92% miss from 123/248. The follow-up
question: is that 1.92% a property of seed=42's Donaldson noise, or is the
~ω_fix-sized eigenvalue a robust feature of the Schoen Donaldson metric
across the converged-seed ensemble?

## Protocol audit (Step 1)

Reading `src/bin/p7_1_omega_fix_diagnostic.rs`, the P7.1 binary defaults
already match the canonical P5.5k strict-converged settings:

| Parameter        | P7.1 default | P5.10 canonical | match? |
|------------------|--------------|-----------------|--------|
| `n_pts`          | 40 000       | 40 000          | yes    |
| `k`              | 3            | 3               | yes    |
| `max_iter`       | 100          | 100             | yes    |
| `donaldson_tol`  | 1e-6         | 1e-6            | yes    |
| `test_degree`    | 4            | (n/a; spectrum) | (P6.3b)|

Cross-checked seed=42 in `output/p5_10_ty_schoen_5sigma.json::candidates[Schoen]::per_seed`:

    seed=42  iters=26  final_donaldson_residual=9.346e-7  sigma_final=8.195

So seed=42 is in the strict-converged set (residual < tol AND iters < cap).

P7.1's reported "σ_residual=8.20" is `summary.final_sigma_residual` — i.e.
the Monge-Ampère σ-spread on the converged metric (the natural η-statistic
scale on Schoen — 2-13 across the converged ensemble), NOT a Donaldson
non-convergence indicator. **There is no settings bug.** seed=42 was a
properly converged metric. The 1.92% residual stands as a real number on a
real metric — but the question of robustness is what P7.1b answers.

## Method (Step 2)

`src/bin/p7_1b_omega_fix_multi_seed.rs` clones the P7.1 spectrum-and-
classification logic, but iterates over the five cleanest strict-converged
Schoen seeds (lowest Donaldson residual + lowest iteration count):

    seed=48879  iters=20  residual=5.04e-7
    seed=    2  iters=20  residual=5.39e-7
    seed=12345  iters=20  residual=5.50e-7
    seed=51966  iters=18  residual=5.69e-7
    seed=    4  iters=22  residual=5.82e-7

For each: solve Donaldson, build metric-Laplacian Galerkin basis at
`test_degree=4`, classify each eigenvector by Z₃×Z₃ (α,β) character,
extract the lowest dominant-trivial-rep raw eigenvalue, and record the
trivial-rep purity (the L²-fraction of |c_i|² supported on (0,0) monomials).

## Results (Step 3)

| seed  | iters | λ_lowest_triv | purity | ppm-from-ω_fix | donald-res |
|-------|-------|---------------|--------|----------------|------------|
| 48879 | 20    | 0.3142220424  | 0.2524 |     366 446.6  | 5.04e-7    |
| 2     | 20    | 0.1151804926  | 0.2667 |     767 766.2  | 5.39e-7    |
| 12345 | 20    | 0.1424019278  | 0.2630 |     712 880.7  | 5.50e-7    |
| 51966 | 18    | 0.6362059292  | 0.2440 |     282 756.7  | 5.69e-7    |
| 4     | 22    | 0.0951384761  | 0.2654 |     808 176.1  | 5.82e-7    |

Mean ± SE: **λ̄_min(triv) = 0.2606 ± 0.1016**

- spread (max−min)/mean = **207.6 %**
- mean per-seed residual from ω_fix = **587 605 ppm (~58.8 %)**
- residual of the mean from ω_fix = **474 503 ppm (~47.5 %)**

Trivial-rep purity is **0.24 — 0.27 across all five seeds** — i.e. the
"trivial-rep" eigenvector only carries about a quarter of its L²-mass on
(α=0, β=0) monomials. The metric Laplacian's spectrum does *not* cleanly
block-diagonalise over the Z₃×Z₃ irrep decomposition at `test_degree=4`.
This is the same ~26 % purity P7.1 reported on seed=42. It is now confirmed
to be a feature of the basis, not a feature of seed=42's particular noise.

## Verdict (Step 4)

**FAILED — the ω_fix prediction is not verified by the metric Laplacian on
Schoen at the basis sizes accessible to us.**

The lowest dominant-trivial-rep eigenvalue is **not a robust feature** of
the Schoen Z₃×Z₃ Donaldson metric at `test_degree=4`. It is essentially a
random pick out of a noisy spectrum: across five strict-converged seeds it
spans **0.095 — 0.636**, more than a factor of six. Seed=42 happened to
land near 0.5055 by coincidence; the same protocol on equally converged
metrics gives anything between 0.1 and 0.6.

Two reasons to expect this:

1. **Trivial-rep purity ≈ 0.25.** With the test-degree-4 polynomial basis,
   the (0,0) representation block is only ~9 monomials out of ~36, and
   Donaldson noise mixes characters by ~75 %. A "dominant-trivial-rep"
   eigenvector is mostly *not* trivial-rep. Its eigenvalue is therefore
   not a clean reading of the Z₃×Z₃-invariant spectrum of the metric
   Laplacian — it's a near-arbitrary mode that happens to weigh slightly
   more on (0,0) monomials than any of the other eight character classes.

2. **Basis-truncation residual dominates.** Even if we had perfect Z₃×Z₃
   decoupling, `test_degree=4` does not span the gateway mode. The lowest
   "trivial-rep" mode at test_degree=4 is a Galerkin projection artifact,
   not the true λ₁ of the metric Laplacian on the trivial-rep subspace.

P7.1's seed=42 value of 0.5055 (1.92% from 123/248) was therefore a
**lucky alignment**, not a verification. P7.1's high-precision claim is
**RETRACTED** pending a localized-basis approach (P7.2b track) or a
larger truncation degree.

The ω_fix formula may still be correct — P7.1b doesn't disprove it; it
only shows the existing P7.1 basis cannot test it. The 0.26 ± 0.10
ensemble mean is consistent with a noisy basis, not a real eigenvalue.

## Cross-reference to P7.1's stale data

`output/p7_1_omega_fix_diagnostic.json` retains P7.1's seed=42-only result
(0.5055, 1.92%). The result is real but **not robust** in the sense
required by the prediction; the per-seed-42 alignment with 123/248 does
not survive ensemble averaging.

## Files

- created: `src/bin/p7_1b_omega_fix_multi_seed.rs`
- created: `output/p7_1b_omega_fix_converged_seeds.json`
- created: `output/p7_1b_omega_fix_multi_seed.log`
- created: `references/p7_1b_omega_fix_results.md` (this file)
- not modified: `src/bin/p7_1_omega_fix_diagnostic.rs` (preserved for history)
- not modified: `output/p7_1_omega_fix_diagnostic.json` (preserved; superseded by P7.1b)
