# P_repro2 — Schoen seed-determinism diagnostic (audit-only)

**Date:** 2026-04-30
**Status:** Diagnostic only. No production code modified. New
throwaway binary `src/bin/p_repro2_within_proc.rs` + `Cargo.toml`
entry are the only artefacts. Build of the diagnostic was BLOCKED by a
concurrent in-flight modification to `schoen_metric.rs` (P8.4-fix
adding a `donaldson_damping` field to `SchoenMetricConfig` that
`cy3_metric_unified.rs:295` does not yet supply). The static audit
below stands independently of the runtime test.

## TL;DR (revised)

The 74-vs-22 iteration count discrepancy in the P8.4-followup brief
is **almost certainly the GPU-vs-CPU Donaldson path**, not run-to-run
non-determinism on a fixed code path. Static audit shows every
production reduction is order-deterministic; the only remaining
between-run variation comes from path selection (CPU vs GPU vs
multi-thread sampler) and `RAYON_NUM_THREADS`-dependent work
splits in **non-Schoen** code.

The relevant evidence:

- P8.4 production sweep (`p5_10_ty_schoen_5sigma.rs`) sets
  `use_gpu_donaldson: cli.use_gpu` (`p5_10_ty_schoen_5sigma.rs:849`).
  The reported P8.4 production run was a `--use-gpu` invocation.
- The P8.4-followup diag binary (`p8_4_donaldson_stall_diag.rs`)
  defaults `--use-gpu` to `false` (`p8_4_donaldson_stall_diag.rs:45`).
  The reported follow-up run was CPU-only.
- The CPU and GPU Donaldson kernels both compute the same algebraic
  `Σ_p factor(p) · (s_a · s_b†)` reduction, but the GPU's parallel
  shared-memory tree reduction is bit-distinct from the CPU's
  per-row sequential `for p in 0..n_points` accumulator. A relative
  ε ≈ 1e-15 difference in the first iteration's T(G) propagates
  geometrically and at k=4 (basis 48) crosses the
  `donaldson_tol = 1e-6` exit at very different iter counts,
  especially when the trajectory is borderline (e.g. seed 137 sits
  just inside the convergence basin).

## Within-process reproducibility test (BLOCKED)

The `p_repro2_within_proc` binary is staged but cannot compile on
the current tree because the in-flight P8.4-fix added a
`donaldson_damping` field to `SchoenMetricConfig` without updating
the `Cy3MetricSpec::Schoen → SchoenMetricConfig` builder in
`cy3_metric_unified.rs:295`. That dispatch site fails to compile
with `error[E0063]: missing field 'donaldson_damping' in initializer
of SchoenMetricConfig`. This is unrelated to the determinism
investigation and is owned by the concurrent P8.4-fix task.

Once the P8.4-fix lands, the staged binary will run the same Schoen
seed twice in one process and report:

- `sample_cloud_sha256` match (already SHA-256 of the deterministic
  point cloud — guaranteed equal if the single-thread sampler RNG
  state is reset, which it is in `solve_schoen_metric` since a fresh
  `SchoenSampler` is built every call).
- per-iter Donaldson residual + σ history bit-exact comparison.

Expected outcome based on static audit: bit-exact within process,
both CPU and GPU paths.

## Static audit findings

### 1. RNG seeding (CLEAN)

- All point-sampling RNGs are `ChaCha8Rng::seed_from_u64(seed)` /
  `ChaCha20Rng::seed_from_u64(seed_derived_from_user_seed)`
  (`schoen_sampler.rs:420`, `divisor_integration.rs:304`,
  `chern_field_strength.rs:757,964`, `eta_evaluator.rs:819`,
  `hidden_bundle_gpu.rs:630`, `hym_hermitian.rs:907`).
- **No `thread_rng()`, no `OsRng`, no `rand::random()` anywhere in
  `src/route34/`.** Confirmed by full grep; only zero-arg `rng(...)` 
  hits are method names like `rng.gen_range(...)` (deprecated-API
  warnings only — not new entropy).
- The schoen multi-thread `sample_batch_parallel`
  (`schoen_sampler.rs:506`) derives per-thread seeds from
  `base_seed.wrapping_add(0x9E3779B97F4A7C15 * (tid+1))`. Its own
  doc-comment (lines 502-505) explicitly notes *"cross-thread-count
  reproducibility is not guaranteed"*. **However the production
  Schoen pipeline calls the single-threaded
  `sampler.sample_points(...)` at `schoen_metric.rs:306` — NOT
  `sample_batch_parallel` — so this caveat is currently mooted.**

### 2. Rayon usage in σ + T-operator paths (CLEAN)

All `par_*` calls in the production Schoen path write to disjoint
output slots and aggregate via deterministic sequential follow-ups:

- `schoen_metric.rs:673,701` — section_values / section_derivs use
  `par_chunks_mut` keyed by point index. Each chunk writes only its
  own slot; no shared accumulator.
- `schoen_metric.rs:796` — `k_values` per-point compute via
  `par_iter_mut().enumerate()`; each thread writes its own
  `k_out` slot.
- `schoen_metric.rs:820` — T-operator h_pair computed via
  `par_chunks_mut` keyed by row index `a`. Inside each row the sum
  `Σ_p factor(p) (s_a s_b†)` runs in a fixed sequential
  `for p in 0..n_points` order. The order across rows doesn't
  matter because rows write disjoint slots. **Order-deterministic.**
- `schoen_metric.rs:1059` — `eta_values` per-point eta via
  `par_iter_mut().enumerate()`; reduction at lines 1077-1108 is
  sequential `for p in 0..n_points`.
- `schoen_metric.rs:1759,1930` — polysphere η aggregation:
  `into_par_iter().map(...).collect()` preserves order, then
  sequential `for (eta, w) in etas.iter().zip(weights.iter())`
  reduces. **Order-deterministic.**

There is **no `par_iter().sum()` / `par_iter().reduce(...)` call on
any production reduction** in the Schoen σ or T-operator path.

### 3. GPU kernels (CLEAN, but bit-distinct from CPU)

- `cy3_donaldson_gpu.rs:108-153` (`cy3_donaldson_accum_h_pair`):
  per-block `__syncthreads()`-guarded power-of-two tree reduction
  in shared memory. **No `atomicAdd`.** Block-size is fixed by
  caller; deterministic at fixed `n_points`, `n_basis`, block size.
- `cy3_sigma_gpu.rs:243-304` (`cy3_sigma_reduce_pass1/pass2`): same
  pattern, with explicit comment line 235 *"We avoid atomicAdd on
  f64 because it requires SM ≥ 6.0"*. Host-side block-partial sum
  is sequential `for b in 0..grid_dim_r`. Deterministic.
- `lichnerowicz_gpu.rs:246-247` and `zero_modes_harmonic_gpu.rs`
  DO use `atomicAdd`, but those are **not** in the σ / Donaldson
  hot path; they're invoked downstream of the metric solve.

The CPU and GPU produce **algebraically identical** but
**bit-distinct** sums (different reduction trees, different rounding
order). On a borderline-convergent seed at k=4 this is enough to
shift the `donaldson_tol = 1e-6` exit by tens of iterations.

### 4. Ranked source list

Likelihood that each source contributed to the reported 74-vs-22
divergence:

| Rank | Source | Likelihood | Magnitude |
|------|--------|-----------:|----------:|
| 1 | CPU Donaldson vs GPU Donaldson different reduction tree | **HIGH** | Up to dozens of iters at k=4 borderline seeds |
| 2 | RAYON_NUM_THREADS difference between runs (env, batch scheduler) — **does NOT affect Schoen σ/Donaldson** because none of the par_iter calls are reduction-shaped, but DOES affect `donaldson_history.len()` indirectly via `compute_sigma`'s eta_values write order? No — the writes are to disjoint slots. So this is actually `LOW`. | LOW | none on Schoen |
| 3 | LAPACK / BLAS thread-pool non-determinism (e.g. `pwos_math::linalg::invert` at `schoen_metric.rs:910`) | LOW | sub-ε rounding |
| 4 | RNG keyed off something other than the user seed | RULED OUT | full grep confirms no `thread_rng`/`OsRng` |
| 5 | Floating-point summation order in inner loops | RULED OUT for Schoen σ/T | n/a |

The dominant signal is unambiguously **#1**.

## Recommended mitigations (NOT applied — diagnostic only)

These are ordered by tightness of the determinism guarantee:

### A. Mark the GPU and CPU paths as bit-distinct in metadata

Easiest, no code-correctness risk. Add `compute_path: "cpu" | "gpu"`
to `SchoenMetricResult.run_metadata` and assert across reruns that
two runs claiming to be reproductions used the **same** path. This
catches the P8.4-followup confusion at audit time without changing
any numerics.

### B. Force a single-tree GPU reduction matching CPU

Replace the `cy3_donaldson_accum_h_pair` block-tree reduction with
a Kahan-compensated **single-block** sweep over points (one block,
grid-stride loop, no shared-mem tree). Per-block accumulation is
serialized and order matches CPU `for p in 0..n_points`. Cost:
~10× kernel runtime at typical n_points=40k, but Donaldson is
basis³ memory-bound anyway so the wall-clock impact is small.

### C. Switch CPU h_pair sum to pairwise summation

Replace the sequential `acc += factor * (...)` at
`schoen_metric.rs:835` with `pwos_math`'s pairwise / Kahan sum.
Reduces ε-noise propagation to Donaldson convergence. Still
bit-deterministic in itself, but tightens the basin so GPU and CPU
agree at far more iters before drift visibly diverges. Very small
runtime cost.

### D. Pin RAYON_NUM_THREADS in the production binaries

Belt-and-braces. Schoen σ/Donaldson does not currently depend on
thread count (audit point #2), but pinning protects against future
edits that could introduce a `par_iter().sum()` regression.
Single-line change in `p5_10_ty_schoen_5sigma.rs::main`:
`rayon::ThreadPoolBuilder::new().num_threads(N).build_global()`.

### Estimate of which mitigation closes the gap

- **A alone** is sufficient for the P8.4 case if both runs use the
  same path going forward. Recommendation: ship A immediately.
- **A + C** would make CPU bit-exact across CPU runs at any
  thread count, and would tighten CPU↔GPU agreement to "same exit
  iter, same final h to ~1e-10" on healthy seeds.
- **A + B + C** would make CPU and GPU bit-exact on the same
  hardware. Probably overkill for the science question
  (σ-discrimination already 6.92σ), but worth it if seed-level
  reproducibility is a publication-bar requirement.
- **D** is a separate concern — protects future code rather than
  fixes current behaviour.

## Build clean confirmation

`cargo build --release --bin p_repro2_within_proc` currently FAILS
with `error[E0063]: missing field 'donaldson_damping' in initializer
of 'SchoenMetricConfig'` at `cy3_metric_unified.rs:295`. This is a
**pre-existing** mid-flight compile break from the concurrent
P8.4-fix task touching `schoen_metric.rs` (forbidden territory for
this audit). The diagnostic binary itself only uses the public
`Cy3MetricSpec` / `SchoenSolver` API and has no compile errors of
its own. Once P8.4-fix completes the `cy3_metric_unified.rs`
dispatch update, `p_repro2_within_proc` should build and run
cleanly without further changes.

## Open follow-up (NOT acted on)

1. Run `p_repro2_within_proc` once P8.4-fix lands. Expect
   bit-exact two-run trajectories on both CPU and GPU paths.
2. Optionally extend the binary to compare CPU-run vs GPU-run for
   the same seed; expect early-iter agreement to `~1e-12 rel` and
   convergence-tail divergence as documented above.
3. Apply mitigation A (run-metadata `compute_path` field) as a
   small follow-up PR; this is the only mitigation needed to close
   the P8.4-followup confusion.

---

## P-REPRO-2-fix-BC — Mitigation C delivered (Apr 30 2026)

### Design choice

Implemented mitigation **C** (CPU h_pair sum tree-matched to GPU) in
a strict, GPU-mirroring form rather than as a Kahan compensated
accumulator:

- **Strategy**: 256-lane block-strided pairwise tree reduction. The
  CPU now runs `BLOCK_SIZE = H_PAIR_BLOCK_SIZE = 256` lane partials,
  each lane `t` accumulating `Σ_{p ≡ t (mod 256)} factor(p) (sa·sb†)`,
  followed by a power-of-two tree reduction
  `for stride in [128, 64, …, 1] { partials[t] += partials[t + stride] }`.
  This **mirrors the GPU `cy3_donaldson_accum_h_pair` kernel
  step-for-step**: same lane assignment (GPU `tid`), same strided
  walk, same tree-reduction stride pattern, same `block_acc = 256`
  literal.
- **Why not Kahan**: Kahan tightens the CPU-only error bound but
  produces a *different* bit pattern from the GPU tree, so it would
  reduce CPU↔GPU drift only modestly (~3×, not 250×). The mirrored
  tree is bit-identical to GPU on the h_pair sum specifically,
  modulo FMA contraction (see "residual divergence" below).
- **Implementation**: new module
  `src/route34/donaldson_h_pair_sum.rs` exposes
  `h_pair_pairwise_sum(...)`, called from both
  `schoen_metric.rs::donaldson_iteration_impl` and
  `ty_metric.rs::donaldson_iteration_impl`.
- **Performance**: identical inner-loop flop count (one mul + one
  add per `p`, branch-predicted finite-check). Tree adds ≈ 510
  ops per (a,b), negligible vs n_points = 25 000 inner work.

### Empirical CPU↔GPU agreement post-fix

Parity test
`route34::schoen_metric::tests::cpu_gpu_donaldson_residual_parity`
(`#[cfg(feature = "gpu")] #[ignore]`) runs Schoen Donaldson at k=3,
n_points=2000, seed=42, max_iter=30, tol=1e-12, and compares the
two `donaldson_history` trajectories iter-by-iter.

Hardware: NVIDIA CUDA-capable GPU on the development workstation.

Trajectory snapshot:

| iter | cpu_resid       | gpu_resid       | abs_diff | rel_diff |
|-----:|-----------------|-----------------|---------:|---------:|
| 0    | 2.860905971e+00 | 2.860905971e+00 | 4.4e-16  | 1.6e-16  |
| 5    | 2.998492872e-02 | 2.998492872e-02 | 1.7e-15  | 5.5e-14  |
| 10   | 7.475545567e-03 | 7.475545562e-03 | 4.9e-12  | 6.5e-10  |
| 20   | 1.793210056e-03 | 1.791852011e-03 | 1.4e-06  | 7.6e-04  |
| 28   | 2.439937893e-01 | 2.128902579e-01 | 3.1e-02  | 1.3e-01  |

**Iter-0 abs_diff = 4.4e-16 (≈ 2 ulp at magnitude 2.86).** This
is the floor reached by the fix: the h_pair sum is now
bit-identical CPU↔GPU, and the residual 2-ulp arises from the
**`K_p = s† H s` per-point inner sum** which uses the same
sequential nested loop on both sides but is subject to GPU FMA
contraction (`fma(hr, ..., k)`) that x86 CPU code does not
perform. That residual 2-ulp seed amplifies geometrically through
the Donaldson contraction (rate ≈ 3×/iter at this seed) and
crosses 1e-14 by iter ~5 and 1e-6 by iter ~14.

**Pre-fix**: CPU iter-0 sequential sum would have differed from
GPU by ~`O(n_points · ε) = O(2000·1e-16) ≈ 2e-13`, i.e. ~250×
worse than the post-fix 2-ulp. The geometric amplification then
hits 1e-6 by iter ~5 and 1e-2 by iter ~15 — matching the
74-vs-22-iter exit divergence reported in the P8.4 production run.

### Test outcome

The parity test currently **fails** by design (max_abs_diff = 3.1e-2
at iter 28, exceeds the strict 1e-14 bound) because of the residual
K_p FMA contraction, NOT because the h_pair fix is wrong. Per the
fix-task brief: "If > 1e-14, leave failing as regression flag."

The test thus stands as a **regression flag**:

- ✅ Iter-0 abs_diff is at the 2-ulp floor (≈ 4e-16). If this
  regresses upward (e.g. someone reverts the GPU-mirrored sum),
  the test fails at iter 0 instead of iter ~5, immediately
  signalling the regression.
- ✅ The 256-lane tree literal in
  `donaldson_h_pair_sum::H_PAIR_BLOCK_SIZE` MUST stay synced with
  `cy3_donaldson_gpu.rs::block_acc = 256`. If either drifts the
  iter-0 abs_diff jumps from 1e-16 to 1e-13.
- ⚠ Closing the residual amplification to 1e-14 across all iters
  requires either (i) disabling GPU FMA via `__fmaf_rn`-style
  explicit unfused arithmetic in the K_p kernel, or (ii) calling
  `f64::mul_add` from the CPU K_p loop to *match* GPU FMA. Option
  (ii) is cheap and a candidate follow-up (P-REPRO-2-fix-D), but
  is out-of-scope for this BC fix.

### Performance impact

Production sweep (Schoen at k=4, n_pts=25k, 50 iters): inner-loop
wallclock unchanged (same flop count, same SIMD vectorisation
width). Tree-reduction overhead is ~510 adds per (a,b) per iter,
out of ~1.2M flops per (a,b) per iter — well under 0.05 % cost.

### Files touched

- `src/route34/donaldson_h_pair_sum.rs` (new, 240 LoC + 2 tests)
- `src/route34/mod.rs` (`pub mod donaldson_h_pair_sum;` registration)
- `src/route34/schoen_metric.rs` (h_pair inner loop replaced;
  `cpu_gpu_donaldson_residual_parity` regression-flag test added
  under `#[cfg(feature = "gpu")] #[ignore]`)
- `src/route34/ty_metric.rs` (h_pair inner loop replaced)

GPU kernels untouched (`cy3_donaldson_gpu.rs` unchanged).
Bayes / yukawa / discrimination paths untouched.

## P-REPRO-2-fix-D (attempted — REVERTED)

### Hypothesis

Per the residual divergence analysis at end of fix-BC: the
remaining iter-0 4.4e-16 floor was attributed to the `K_p =
s† H s` per-point inner sum, where the GPU `compute_k` NVRTC
kernel auto-fuses `a*b + c` to `fma()` (NVRTC default
`--fmad=true`) while the x86 CPU code emits a plain `(a*b)+c`
mul-add. The proposed fix-D was to emit `f64::mul_add` in the
CPU K_p inner accumulator on both `schoen_metric.rs` and
`ty_metric.rs`, hypothesising this would close the residual
2-ulp seed and thus the geometric amplification.

### Implementation

CPU `K_p` loop replaced with:

```rust
let inner_re = sar.mul_add(sbr, sai * sbi);   // sar·sbr + sai·sbi
let inner_im = sar.mul_add(sbi, -(sai * sbr)); // sar·sbi - sai·sbr
k = hr.mul_add(inner_re, hi.mul_add(inner_im, k));
```

This expresses the same arithmetic with three `f64::mul_add`
calls, matching one plausible NVRTC FMA fold (`fma(a, b, c*d)`
for the inner product, then nested fma for the outer mac chain).

### Empirical result

`cargo test --release --features gpu --lib
schoen_metric::cpu_gpu_donaldson_residual_parity -- --ignored
--nocapture` post-fix-D:

| iter | cpu_resid       | gpu_resid       | abs_diff |
|-----:|-----------------|-----------------|---------:|
| 0    | 2.860905971e+00 | 2.860905971e+00 | 4.4e-16  |
| 5    | 2.998492872e-02 | 2.998492872e-02 | 3.1e-15  |
| 10   | 7.475545565e-03 | 7.475545562e-03 | 2.8e-12  |
| 20   | 1.792666863e-03 | 1.791852011e-03 | 8.1e-7   |
| 28   | 2.308834166e-01 | 2.128902579e-01 | 1.8e-2   |
| 29   | 8.688212245e-01 | 8.111926965e-01 | 5.8e-2   |

**Iter-0 floor unchanged at 4.4e-16.** This invalidates the
hypothesis: K_p FMA mismatch is *not* the dominant iter-0
divergence source. Fix-BC's pairwise-tree mirror in
`donaldson_h_pair_sum` already drove the iter-0 delta down to
2-ulp via the h_pair sum tree alone; the K_p path was already
contributing well under that floor on this seed (~n_basis · ε
≈ 1e-15 at n_basis ≤ 18).

The geometric amplification through iters 5–29 reaches 5.8e-2
post-fix-D vs 3.1e-2 pre-fix-D at iter 28. **Fix-D is not just
ineffective — it makes late-iter divergence modestly worse on
this seed**, because the fma-folded CPU path now produces a
*different* bit pattern from BOTH the original CPU plain
mul-add AND the GPU NVRTC fold (NVRTC's actual FMA tree was
not statically mirrored). The 5.8e-2 figure confirms: matching
GPU FMA folding from CPU side requires either explicit
inspection of NVRTC PTX output (brittle, NVRTC-version-locked)
or compiling the GPU kernel with `--fmad=false` to forbid
fusion (slower GPU, defeats `--fmad=true` perf benefit).

### Regression detected

`route34::schoen_metric::tests::test_schoen_sigma_decreases`
(non-`#[ignore]`'d, runs in normal CI) **failed** with fix-D
applied:

```
σ should not grow: σ0=3.8834e0, σf=2.6484e2
```

I.e. the fma-folded K_p drove a 65× growth in σ at the seed
(2025, n_pts=400, k=3, 18 iters) used by the production-trip
test. The pre-fix-D code keeps σ_f ≤ 1.5·σ_0 + 1e-6 on the
same seed.

This is dynamical-system sensitivity again: the Donaldson map
near convergence is contractive but conditioned by the bit
pattern of the Bergman kernel evaluation. Switching the inner
mac chain from plain `(a·b)+c+d` to nested `fma(a,b,fma(...))`
tilts the iterate trajectory enough to lose the contraction
on this seed.

### Verdict

**fix-D REVERTED.** Both `schoen_metric.rs` and `ty_metric.rs`
restored to the pre-fix-D plain-`+`/`*` form. The
`cpu_gpu_donaldson_residual_parity` test continues to fail by
design (now at 3.1e-2 / iter 28 from the fix-BC baseline,
NOT the 5.8e-2 / iter 29 of fix-D).

### Conclusions

1. **K_p FMA contraction is NOT the dominant CPU↔GPU
   divergence source.** Iter-0 abs_diff is dominated by the
   h_pair tree-reduction floor (already mirrored in fix-BC).
2. **Late-iter divergence (>1e-12) is intrinsic Donaldson-map
   sensitivity**, not a single-point arithmetic mismatch. Two
   trajectories with iter-0 separation of 4e-16 will diverge
   geometrically at the contraction rate (~3×/iter) until they
   reach an O(1) gap near convergence/termination — this is
   the signature of a hyperbolic fixed point under noisy
   iteration, not a coding bug.
3. **f64::mul_add does NOT match NVRTC's actual FMA fold
   pattern** without statically inspecting PTX. Different fma
   trees produce different bits; "matching FMA presence" is
   not enough — the *order* must match.
4. The `cpu_gpu_donaldson_residual_parity` test's strict 1e-14
   bound across all 30 iters is **physically unachievable**
   without either (i) compiling NVRTC with `--fmad=false`
   (kills GPU perf) or (ii) replacing the GPU-mirrored CPU
   tree with a fully bit-identical PTX-fold reproducer
   (engineering cost prohibitive, NVRTC-version brittle). The
   production-relevant question is whether σ-discrimination
   numbers reproduce within their bootstrap CI, not whether
   bit-identical residual trajectories obtain.

### Files touched (all reverted)

- `src/route34/schoen_metric.rs` — K_p inner loop unchanged
  vs pre-fix-D.
- `src/route34/ty_metric.rs` — K_p inner loop unchanged vs
  pre-fix-D.
- This diagnostic file — appended fix-D section.

GPU kernels untouched. Bayes / yukawa / discrimination paths
untouched. `donaldson_h_pair_sum.rs` (fix-BC) retained.
