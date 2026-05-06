# P-Stage 3 — Bundle-Laplacian Eigenmode Infrastructure Audit (unblock plan)

**Status:** Read-only audit. No production code modified.
**Audit head:** `eb04d05915a71a2c8e133c1e3788ffd16742446c` (origin/main).
**Scope:** Framework Stage 3 = `{harmonic representatives H^p(X, V^q), bundle-Laplacian Box_V eigenvalue spectrum, triple overlaps y_{ij,k̄}}` on Schoen `Z_3 × Z_3` (with Tian-Yau `Z_3` as the comparison candidate). Stages 4-7 (Lagrangian dim-reduction → closure check → PPN → cosmological constant) are mechanical once Stage 3 outputs are in hand and are explicitly out of scope for this audit.

**Primary references in tree consulted:**
- `src/route34/CHAPTER_CODE_MAP.md` — chapter 21 → file:line map (this is the canonical orientation document for the route34 tree).
- `REMEDIATION_PLAN_OPTION_B.md` — current 6–12 week plan to bring the pipeline up to publication grade. Phase 2 is the live work plan for what this audit is auditing.
- `references/p8_3_yukawa_production_results.md` — most recent production sweep through `predict_fermion_masses` on TY/Z_3 + Schoen/Z_3×Z_3.
- `references/p_chain_matcher_deprecation.md` — formal deprecation of the previous chain-eigenvalue path.

---

## A — Harmonic representatives of `H^p(X, V^q)`

### A.1 Where the codebase computes harmonic representatives

| File | Function / type | Lines | What it actually produces |
|---|---|---|---|
| `src/route34/zero_modes_harmonic.rs` | `solve_harmonic_zero_modes` | 722–905 (entry at 722) | Genuine numerical-harmonic-representative solver for `H^1(M, V)` on the **2-factor (Tian-Yau bicubic) ambient**. Builds the expanded polynomial-seed Dolbeault ansatz, assembles Hermitian Gram `G` (line 491) and bundle-twisted Bochner Laplacian `L` (line 591) with the **HYM** Hermitian metric (not identity), solves the generalised pencil `L v = λ G v`, takes the lowest-eigenvalue eigenvectors as the harmonic basis, returns `HarmonicZeroModeResult` (struct at 164) with per-mode residual norms + orthonormality residual + observed-vs-predicted cohomology dimension. |
| `src/route34/zero_modes_harmonic_z3xz3.rs` | (whole module) | 1488 lines total | Same algorithm projected to the **Z_3×Z_3 Wilson-line + H_4 (icosahedral Z_5)** invariant sub-bundle. Consumed by P7.6 / P7.8b gateway-mode tests. Has an OOM-protection cap `Z3XZ3_MAX_TOTAL_DEGREE = 6` (line 117). |
| `src/route34/zero_modes_harmonic_gpu.rs` | `solve_harmonic_zero_modes_gpu` | 60 | GPU adapter (Phase-1 CPU-fallback scaffold; Phase-2 NVRTC kernel deferred — per file docstring). Tested bit-identical to CPU at 1e-10. |
| `src/zero_modes.rs` | `compute_zero_mode_spectrum` | 1332–1380 | **Cohomology dimension predictor (NOT mode producer).** Returns `ZeroModeSpectrum { n_27, n_27bar, n_1_singlets, generation_count }`. Net generation count `n_27 − n_27bar` from Atiyah-Singer index theorem `χ(V) = c_3(V)/2`; the absolute count comes from the Koszul + BBW path in `monad_h1` / `monad_h2`. Used by the harmonic solver to set `kernel_dim_target` when `auto_use_predicted_dim = true`. |
| `src/route34/bbw_cohomology.rs` | (whole module) | 564 lines | Self-contained Bott-Borel-Weil + Künneth + Koszul-resolution chase for `h^p(X, O_X(L))` on the line-bundle CICY case. |
| `src/zero_modes.rs` | `evaluate_polynomial_seeds`, `polynomial_seed_modes` | (in `zero_modes.rs`) and `route34/zero_modes_harmonic.rs` line 837 | LEGACY polynomial-seed Dolbeault cocycles. Module doc (line 5) explicitly flags these as carrying `O(1)` systematic error in absolute Yukawa scale. Preserved only for the seed-vs-harmonic regression test. |

### A.2 Which `(p, q, V_power)` triples are computable today?

**Computable end-to-end** (i.e. `solve_harmonic_zero_modes` returns a kernel basis with finite residuals and an orthonormality residual `< 1e-9`):

- `(p=1, q=1, V^1)`, i.e. `H^1(X, V)` — the **27 generation modes**. Primary output. `cohomology_dim_predicted` = `n_27 = c_3(V) / (2|Γ|)` from `compute_zero_mode_spectrum`. Production sweep (`p8_3b_yukawa_production.json`) reports observed dim = 9 on both TY/Z_3 (BBW count 9) and Schoen/Z_3×Z_3 (BBW degrades to 0; the kernel-dim fallback in `predict_fermion_masses_with_overrides` injects `kernel_dim_target = 9`).

**Predictable but NOT mode-produced** by the current solver:
- `(p=1, V^*)` — `H^1(X, V^*)` (27̄ generations). Currently asserted = 0 by design for the canonical positive monad (`zero_modes.rs:1368: n_27bar = 0u32`). The 27̄ modes are simply not enumerated; `solve_harmonic_zero_modes` is wired to `H^1(X, V)` only.
- `(p=1, End V)` — `H^1(X, V ⊗ V^*)` (singlet / Higgs sector for SU(3)-bundle visibility). Currently estimated as `h^{2,1}(X)/|Γ|` (line 1372 of `zero_modes.rs`) — a lower-bound proxy via the complex-structure-moduli contribution. The full End-V cohomology requires a **double Koszul chase** that the comment at line 1325–1327 explicitly flags as out-of-scope. The "Higgs" assignment in the production pipeline (`yukawa_sectors_real::assign_sectors_dynamic`, line 273+) is therefore a phase-class-0 **proxy** — it picks the lowest-eigenvalue mode of the trivial Wilson-line phase class out of the same `H^1(V)` solve, NOT a separately computed `H^1(End V)` mode.
- `(p=2, V)` — `H^2(X, V)`. The comment at `zero_modes.rs:1349–1351` invokes Serre duality and asserts `h^2(V) = 0` for stable monads with `c_1 = 0`; this is checked by `monad_h2` for sanity but not solved for as a numerical kernel.
- `(p=0,3, V)` — `H^0` and `H^3` are both 0 by stability + Serre duality, asserted, not computed.

**NOT computable today:**
- `H^p(X, V^q)` for `q ≥ 2` (i.e. `Sym^q V`, `∧^q V`). The `polystability.rs:478` `check_polystability` does enumerate `wedge^k V` sub-bundles for slope checks (line 412), but only at the level of slope-stability slope numerics; harmonic representatives of `H^p(X, ∧^q V)` are not solved for.

### A.3 Which sectors / Wilson-line characters are projected onto?

The `Z_3 × Z_3` Wilson-line action gives 9 character classes `(g_1, g_2) ∈ {0,1,2}^2`. The codebase covers:

- **2-factor (TY) projection**: `assign_sectors_dynamic` (`yukawa_sectors_real.rs:82+`). Uses the per-summand SU(3)-Cartan weight projection `class = dom mod 3` (post-`p_wilson_fix_diagnostic.md` correction; before that, the `cartan_phases` formulation collapsed every mode to phase 0 — see comments at line 99–129). Three classes: `{0 → lepton/Higgs, 1 → up-quark, 2 → down-quark}`.
- **3-factor (Schoen) projection**: same function, taking the `(b[0] − b[1]) mod 3` branch (line 181) when `bundle.b_lines_3factor.is_some()`. Three classes again via DHOR-2006 / ALP-2011 dictionary (line 230–233).
- **Full Z_3 × Z_3 sub-bundle filter**: `zero_modes_harmonic_z3xz3.rs` filters seeds to combined `(α_base + g_1, β_base + g_2) = (0, 0)` (module doc line 18–19). This is the **journal §F.1.5/§F.1.6 modded-out V/(Z_3×Z_3) bundle** — the gateway-mode test sub-bundle. Optional additional filter on the H_4 Z_5 character (line 22–23).

Note: the `wilson_line_e8_z3xz3.rs:91` per-b_line `(g_1, g_2)` table only assigns **3 of 9** characters explicitly (the diagonal pattern `(0,0), (1,1), (2,2)` plus the conjugate-flipped pattern for the `(0,1)` block). The remaining 6 characters are **not** enumerated as separate sub-bundles — the production solver collapses to a 3-class assignment in both 2-factor and 3-factor branches.

### A.4 Precision / Donaldson convergence tier

The harmonic-mode solver is fed by a `MetricBackground` adapter (`Cy3MetricResultBackground`, `yukawa_pipeline.rs:213+`) that wraps either a `TyMetricResult` or `SchoenMetricResult`. The Donaldson balanced-metric residual at the latest production sweep (`p8_3_yukawa_production.json`, n_pts = 25k, k = 3, iter_cap = 100, donaldson_tol = 1e-6, seed = 12345):

- TY/Z_3: `donaldson_residual = 6.355e-7`, `sigma_final = 0.272`, 16 iterations, basis dim = 87.
- Schoen/Z_3×Z_3: `donaldson_residual = 7.546e-7`, `sigma_final = 1.852`, 24 iterations, basis dim = 27.

The reported harmonic-mode residual diagnostic (`harmonic_orthonormality_residual`):

- TY/Z_3: `7.97e-10` — at numerical-zero floor.
- Schoen/Z_3×Z_3: `2.40e-10` — at numerical-zero floor.

But: `quadrature_uniformity_score`:
- TY/Z_3: `0.600` (acceptable).
- Schoen/Z_3×Z_3: `0.0081` (essentially failed — the Shiffman-Zelditch quadrature on the Schoen sample cloud is non-uniform).

And `cohomology_dim_predicted` for Schoen is `0` because `MonadBundle::chern_classes()` is hard-coded to 2-factor ambients (per `p8_3_yukawa_production_results.md` §1) and the canonical Schoen bundle data was being passed in via the AKLP TY-bundle `MonadBundle::anderson_lukas_palti_example()` placeholder. `kernel_dim_target = 9` is the manual fallback that recovers a non-empty kernel.

### A.5 What is MISSING for a publication-grade catalogue

For Stage 3 to count as **done**, we need a complete `(p, q, V_power)` tableau on Schoen/Z_3×Z_3 with the right Wilson-line decomposition. Gaps:

1. **`H^1(X, V^*)` (27̄) computation.** Currently asserted = 0 by stability convention; needs a programmatic check (the LES of the monad `0 → V → B → C → 0` provides the numerical avenue via `h^2(V)` from `monad_h2`). For a Schoen-canonical bundle this matters because anti-generations enter the chiral index.
2. **`H^1(X, End V)` (singlets / Higgs).** The "Higgs is the lowest-eigenvalue mode of phase-class 0 of `H^1(V)`" proxy is **structurally wrong** — singlets live in `End V`, not `V`. The `extract_3x3_from_tensor` Higgs contraction (`yukawa_sectors_real.rs:416+`) currently uses this proxy. This is one of the reasons the Yukawa 3×3 matrix collapses to rank-1 in the production sweep (`p8_3_yukawa_production_results.md` §6).
3. **`H^1(X, ∧^2 V)` and `H^1(X, ∧^3 V)`.** Required for a full SU(4) bundle decomposition into SO(10) reps (16 + 10 + ...). The `polystability.rs` Schur-functor enumeration only checks slopes; doesn't produce harmonic representatives.
4. **Schoen-canonical monad bundle.** `MonadBundle::schoen_z3xz3_canonical()` was added in P8.3b (per `p8_3_yukawa_production_results.md` §P8.3b.1 Blocker 2) but it currently returns "the same line-bundle data as the AKLP example, but with journal §F.1.5 / §F.1.6 commentary on the Z/3 × Z/3 Wilson-line decomposition". It is **not yet a genuine Schoen bundle** — the 3-factor `chern_classes()` machinery is still unimplemented (`zero_modes.rs` 2-factor hard-code per line 412 cohomology comment).
5. **9-character Wilson-line sector decomposition.** Only 3 of 9 (g_1, g_2) classes are explicitly assigned; the rest collapse via mod-3 reduction. For a true SU(4) → SO(10) × U(1) decomposition you need all 9 to be addressable.
6. **Quadrature uniformity on Schoen.** 0.0081 is unacceptable for publication-grade Yukawa overlaps. Needs either more sample points (n_pts → 100k+) or a re-weighting fix in the `Cy3MetricResultBackground::from_schoen` adapter (the `donaldson_k` rebalance at `yukawa_pipeline.rs:313` is the likely culprit).

---

## B — Bundle-Laplacian eigenvalue spectrum

### B.1 Where the bundle Laplacian is computed

| File | Function | Line | What it computes |
|---|---|---|---|
| `src/route34/zero_modes_harmonic.rs` | `build_laplacian_matrix` | 591–699 | The genuine **bundle-twisted Bochner / Dolbeault Laplacian** `L_{αβ} = Σ_i ⟨∂_{z_i} s_α, ∂_{z_i} s_β⟩_{h_V, ω}`. Sums over the 8 ambient holomorphic coordinates, weighted by HYM `h_V` and the CY measure `|Ω|² · w_p`. **This is `Box_V` for the polynomial-seed ansatz space.** |
| `src/route34/zero_modes_harmonic.rs` | `solve_harmonic_zero_modes` | 770–842 | Generalised eigensolve `L v = λ G v`, returning the full eigenvalue list `eigenvalues_full` (line 873-ish in the result struct, wired through to JSON in p5_6b output). |
| `src/route34/zero_modes_harmonic_z3xz3.rs` | `solve_z3xz3_harmonic` (entry near top of file) | — | Same algorithm projected to the Z_3 × Z_3 + H_4 sub-bundle. **This is the `Box_V` for the gateway-mode (P7.6) sub-bundle.** |
| `src/route34/metric_laplacian.rs` | `compute_metric_laplacian_spectrum` | 296+ | The **scalar metric Laplacian** Δ_g on functions `f: M → ℂ`, NOT the bundle Laplacian. Module doc line 1 says "research-only, chain-match channel not converged" — this is the operator the deprecated `chain_matcher` consumed. Excludes the constant mode (line 78–98). |
| `src/route34/metric_laplacian_projected.rs` | `compute_projected_metric_laplacian_spectrum` | (whole module 948 lines) | Scalar metric Laplacian projected to a sub-Coxeter-invariant basis. Same caveat (research-only). |
| `src/route34/lichnerowicz.rs`, `lichnerowicz_operator.rs` | `LichnerowiczOperator` and friends | 327, 1182 lines | **Vector** Laplacian for the Killing-equation solver on the metric (Route 4). Not the bundle Laplacian. |

### B.2 What eigenvalues are produced and at what precision?

The bundle Laplacian's full ascending spectrum is exposed via `HarmonicZeroModeResult.eigenvalues_full`. Captured production data:

- **`output/p5_6b_eigenvalue_spectrum.json`** (TY/Z_3, n_pts = 200, k = 2, AKLP bundle, 24-mode basis): full 24-eigenvalue spectrum, range `[1.186, 1.954]`, largest lower-half gap of 1.159 at index 5. Notable: the spectrum is **bit-identical across budgets 1, 10, 50** (lines 18–43 vs 59–83 vs 100–124) — the eigenvalue computation is independent of Donaldson iteration count beyond the first iteration on this small-n cloud. (Likely a polynomial-seed-only contribution where the `h_V` contribution is negligible at this k.)
- **`output/p7_3_bundle_laplacian_omega_fix.json`** (Schoen/Z_3×Z_3 + AKLP, n_pts = 40k, k = 3): full 24-eigenvalue spectrum, range `[0.722, 12.997]`. `lambda_lowest_nonzero = 0.722`, `lambda_max = 12.997`, `lambda_mean = 4.684`, `lambda_trace = 112.41`. **No kernel modes detected** at the configured threshold (kernel_count_used = 0); `cohomology_dim_predicted = 0` (the 3-factor BBW degradation again).

Eigenvalue precision: f64 throughout the assembly + Jacobi solve. The `chain_matcher` documentation comments (`chain_matcher.rs:43–46`) note that ratios are lifted to `pwos_math::precision::BigFloat` (~150-digit) for chain residual comparison, but the underlying eigenvalues themselves are f64.

### B.3 Are eigenvalues mapped to chain positions or only used internally?

**Both, but the chain-mapping path is officially deprecated.**

The existing chain-mapping infrastructure:

- `src/route34/chain_matcher.rs::ChainType::predicted_exponents()` (lines 74–79) returns the predicted exponents:
  - `Lepton (φ-chain)`: `[1, 7, 11, 13, 17, 19, 23, 29]` (E_8 Coxeter exponents).
  - `Quark (√2-chain)`: `[4, 6.5, 15, 22.5, 26, 36.5]` (D_8 integer / half-integer steps).

- `src/route34/chain_matcher.rs::match_chain` and `match_chain_hungarian` (~line 130+) — greedy + Hungarian assignment of computed eigenvalues to chain exponents in log-space. **All `pub` items in this module are `#[deprecated]`** (per `references/p_chain_matcher_deprecation.md`).

- `src/bin/p7_11_quark_chain.rs` / `p7_11_lepton_chain.rs` (and their post-fix variants `p7_8b_*`) drive this matcher. Their JSON outputs (`output/p7_11_quark_chain.json`, `output/p7_11_lepton_chain.json`, `output/p7_8b_*_post_fix.json`) all use the same struct as `p6_3b_basis_convergence_sweep` — they ARE the basis-convergence sweep outputs, just relabelled. All have `"converged": false, "convergence_notes": ["No consecutive pair of degrees met the 10% convergence criterion."]`.

- `src/bin/p8_5_chain_k4_gpu.rs` — most recent chain attempt at k=4 GPU, `output/p8_5_chain_k4_gpu.json` 542 lines. Same Hungarian-assignment mechanics on `metric_laplacian` eigenvalues (NOT on the bundle Laplacian — this binary uses the **scalar** `compute_metric_laplacian_spectrum`).

### B.4 Is the framework's predicted chain `{0, 11, 17, 4, 13/2, 15, 22.5, 26, 36.5, 34.5, 1072/30, 1132/30}` actually compared against computed eigenvalues anywhere?

**No, not in this exact form.** The framework's 12-position chain in the task brief is a **superset** of what the codebase compares against:

| Codebase chain | Framework chain (task brief) | Notes |
|---|---|---|
| Lepton: `1, 7, 11, 13, 17, 19, 23, 29` (8 positions) | `0, 11, 17` (3 positions: e, μ, τ) | Codebase uses the full E_8 Coxeter exponent set; framework's "lepton chain positions" `(k_e, k_μ, k_τ) = (0, 11, 17)` are the actual generation slots. The codebase predicted exponent list is the **infrastructure** that the slots live in, not the slots themselves. |
| Quark: `4, 6.5, 15, 22.5, 26, 36.5` (6 positions) | `4, 13/2, 15, 22.5, 26, 36.5, 34.5, 1072/30, 1132/30` (9 positions) | Codebase has the canonical D_8 integer + half-integer pattern; framework adds `34.5, 1072/30, 1132/30` and identifies `13/2 = 6.5` (codebase uses decimal). The extra three positions are the **dim-2/(2·dim) ω_fix-related slots** that come up in P7.x diagnostics (see `references/p7_12_omega_fix_reconciliation.md`). |

The comparison logic that exists, file:line:

- `src/route34/chain_matcher.rs:130–270` (`match_chain` greedy) and `~270–500` (`match_chain_hungarian`). Both consume f64 eigenvalues and return a `ChainMatchResult` with `assigned_eigvals: Vec<f64>`, `predicted_exponents: Vec<f64>`, and a `residual_log_str` for high-precision comparison.

- `src/bin/p8_5_chain_k4_gpu.rs` (~lines 100–300, `output/p8_5_chain_k4_gpu.log` for run details) — invokes the matcher on `compute_metric_laplacian_spectrum` outputs, NOT on `solve_harmonic_zero_modes` outputs. This is **the scalar metric Laplacian, not the bundle Laplacian.** A sample of post-fix output (`output/p8_5_chain_k4_gpu.json:104–123` from earlier read) reports `ratio_pattern` entries with explicit `closest_harmonic`, `closest_predicted`, `rel_residual` per ratio — at test_degree = 4, n_pts = 8000, k = 4, the smallest ratio rel_residual is `0.020` (i.e. 2% off the √2 prediction).

- The bundle-Laplacian `eigenvalues_full` from `solve_harmonic_zero_modes` is **NOT** plugged into `chain_matcher`. The wiring is: bundle Laplacian → (consumed inside) `predict_fermion_masses` → `FermionMassPrediction` → consumed by `bayes_factor_multichannel::from_chain_match` (which still reads JSONs from the deprecated `metric_laplacian` path). Per `references/p_chain_matcher_deprecation.md`, this is **Phase 2.1 work**, not yet landed: there is no `from_yukawa_pipeline_prediction` adapter.

### B.5 Per-eigenvalue residual at the latest converged production sweep

The chain-residual numbers from `output/p7_8b_quark_chain_post_fix.json` (n_pts = 25k, k = 3, max_iter = 100, seed = 12345; this is the "production-grade" basis-sweep, NOT a converged run):

- TY/Z3, quark chain, basis-dim 18 (test_degree 2): residual_log = 22.94, n_assigned = 6, max_individual_dev = 9.73.
- TY/Z3, quark chain, basis-dim 178 (test_degree 4): residual_log = 23.69, max_individual_dev = 9.97.
- Schoen, quark chain, basis-dim 18: residual_log = 14.01, max_individual_dev not in the truncated row.
- Schoen, quark chain, basis-dim 450 (test_degree 5): residual_log = 12.19.

`"converged": false` across **all** rows — convergence_notes confirms "No consecutive pair of degrees met the 10% convergence criterion." Per residual: log-eigenvalue distance per-position is `O(1)` in nats, i.e. the predicted chain is **not** matched within a factor of `~e^(residual / n_assigned)` ≈ `e^(23.7/6) = e^3.9 ≈ 50`. **The chain is not currently a discriminator.** The `p_chain_matcher_deprecation.md` document is the formal retraction record.

The bundle-Laplacian-direct (vs. scalar-Laplacian-direct) chain comparison **has never been run in production**. Every `output/p7_*` and `output/p8_5_*` file in tree is from `compute_metric_laplacian_spectrum` (scalar), not `solve_harmonic_zero_modes::eigenvalues_full` (bundle).

---

## C — Triple overlap integrals `y_{ij,k̄} = ∫_X ω_i ∧ ω_j ∧ ω̄_k`

### C.1 Where the codebase computes them

| File | Function | Line | What it computes |
|---|---|---|---|
| `src/route34/yukawa_overlap_real.rs` | `compute_yukawa_couplings` | 326+ (entry around line 326–500) | Triple overlap with HYM metric in normalisation, Shiffman-Zelditch quadrature, bootstrap MC error bars, convergence test `n_pts → 2 n_pts → 4 n_pts`. Returns `YukawaResult` (struct around line 127–200) with `couplings: Tensor3` (the 3-tensor `Y_{ijk}`), `couplings_uncertainty: Tensor3Real`, `convergence_ratio`, `quadrature_uniformity_score`. |
| `src/route34/yukawa_overlap_real_gpu.rs` | `compute_yukawa_couplings_gpu` | 75 | GPU adapter (Phase-1 CPU-fallback, Phase-2 NVRTC deferred). Bit-identical to CPU at 1e-10. |
| `src/yukawa_overlap.rs` | (legacy module) | 1795 lines | Legacy single-Cartan implementation. Superseded by `route34::yukawa_overlap_real`. |
| `src/route34/yukawa_pipeline.rs` | `predict_fermion_masses` | 373–520 | End-to-end driver: HYM → harmonic modes → triple overlap → sector assignment → `extract_3x3_from_tensor` → RG → fermion masses + CKM. |

### C.2 Are the integrals stored in a 3-tensor / matrix that can be diagonalized?

**Yes, in `Tensor3`** (`yukawa_overlap_real.rs:127–147`), an `n × n × n` complex tensor with row-major layout `data[i * n² + j * n + k]`. Per-entry uncertainty in a parallel `Tensor3Real`. The size `n` equals the harmonic-mode count from `solve_harmonic_zero_modes` (typically 9 — one per `H^1(V)` direction).

The contraction to a 3×3 SVD-able matrix happens in `yukawa_sectors_real::extract_3x3_from_tensor` (`yukawa_sectors_real.rs:416+`):

```text
    Y_{ij}  =  v_h0  ·  T_{i, j, h_0}            (P8.3-followup-C contraction)
```

— i.e. the Higgs slot `h_0` is **the lowest-eigenvalue phase-class-0 mode** picked from `sectors.higgs[0]` (which is sorted ascending by harmonic eigenvalue inside `assign_sectors_dynamic`). The SVD of this 3×3 matrix gives the fermion masses (one per family).

**Critical defect**: the production sweep `p8_3b_yukawa_production.json` (post-Blocker-1 fix to multi-Higgs contraction, then re-retracted in P8.3-followup-C back to single-`h_0`) shows the 3×3 matrix is **rank-1 for both candidates** — the SVD has 1 non-zero singular value, hence 1 non-zero mass per sector (only `m_t`). This is upstream of the contraction layer; the harmonic modes themselves are functionally near-degenerate (`ψ_α(p) ≈ c_α · ψ_common(p)`) at the current basis size. See `p8_3_yukawa_production_results.md` §6 + Caveats §3.

### C.3 What basis (Wilson-line sectors) are the i, j, k indices over?

`i, j, k` in `Tensor3` are flat indices into the harmonic-mode list returned by `solve_harmonic_zero_modes` — i.e. all `H^1(X, V)` modes regardless of Wilson-line phase. The Wilson-line sector assignment is applied **after** the tensor is computed, by `assign_sectors_dynamic`, which returns `SectorAssignment { up_quark: Vec<usize>, down_quark: Vec<usize>, lepton: Vec<usize>, higgs: Vec<usize> }` — these are **index lists into the harmonic-mode list**.

The 3×3 extraction `extract_3x3_from_tensor(&couplings, &sectors.up_quark, &sectors.up_quark, &sectors.higgs)` (`yukawa_pipeline.rs:423`) then projects out the up-quark × up-quark × Higgs slice. Same for d (`up × down × Higgs`) and e (`lepton × lepton × Higgs`). Note: same Higgs index list reused for all three sectors.

**Defect**: `sectors.higgs` is populated from `H^1(V)` phase-class-0 modes, NOT from `H^1(End V)` — see Audit A §A.5 item 2.

### C.4 Precision

- f64 throughout the integration.
- Bootstrap CI from `n_bootstrap` resamples (default in `YukawaConfig`).
- Convergence test `|Y(N) − Y(N/2)|` reported as `convergence_ratio` (production: TY = 0.127, Schoen = 0.265 — both well below the canonical `1/√2 ≈ 0.71` MC convergence floor, so the integrator IS converged at n = 25k).
- Yukawa per-entry max relative uncertainty (`yukawa_uncertainty_relative`): TY = 3.10, Schoen = 4.93. **These are both `>> 1`** — the bootstrap report says the relative uncertainty is larger than the central value. This is because most entries of the rank-1 tensor are at the floating-point zero floor; their uncertainty estimate divides by an essentially-zero norm.

### C.5 Could the existing infrastructure produce a publication-grade Yukawa matrix on demand?

**No, not without three structural fixes:**

1. **Non-degenerate harmonic modes.** Until `solve_harmonic_zero_modes` produces functionally distinct modes (not `ψ_α ≈ c_α · ψ_common`), the resulting `Tensor3` will collapse to rank-1 at the 3×3 contraction. This is the root cause of the production rank-1 collapse (`p8_3_yukawa_production_results.md` §6). The fix likely involves one of: (a) larger polynomial-seed basis (current cap ≤ 4), (b) genuine bundle-twisted Dirac kernel rather than the polynomial-seed Dolbeault Laplacian approximation, (c) sub-bundle projection using `zero_modes_harmonic_z3xz3` rather than full `H^1(V)` then sector-reassigned.

2. **`H^1(End V)` for the Higgs.** The current `phase-class-0 mode of H^1(V)` proxy is structurally wrong; SU(3)-singlet Higgs doublets live in `End V`, not `V`. Until this is fixed, the Higgs contraction picks a vector-bundle mode, not a true singlet.

3. **Schoen-canonical bundle.** `MonadBundle::schoen_z3xz3_canonical()` currently aliases the AKLP TY-bundle data; the 3-factor `chern_classes()` machinery is unimplemented. Without a genuine Schoen bundle, the BBW count = 0 and the "9 modes" are forced via `kernel_dim_target` fallback, not derived from cohomology.

The infrastructure is ~80% wired; what's missing is enumerated above. The single end-to-end binary that runs the full pipeline (`p8_3_yukawa_production`, plus its rewires `p5_6_yukawa_propagation`, `bayes_discriminate`) reaches **rank-1 / m_t-only** output, not a publication-grade Yukawa matrix.

---

## D — Production sweep status

### D.1 Has any single production run computed all of A + B + C at converged precision?

**No.** The closest is `output/p8_3b_yukawa_production.json` (n_pts = 25k, k = 3, iter_cap = 100, donaldson_tol = 1e-6, seed = 12345 — the canonical settings) which exercises all three pieces end-to-end. But:

- (A) Harmonic modes: produced (cohomology_dim_observed = 9 on both candidates), **but rank-degenerate**.
- (B) Bundle Laplacian eigenvalues: produced (the 9 lowest are the kernel; the rest are NOT mapped to chain positions; the **scalar metric Laplacian** chain comparison is the deprecated path).
- (C) Triple overlaps: produced (`Tensor3`), **but rank-1 after contraction**.

The σ-channel sweeps `p5_10_*` (multiple TY-vs-Schoen 5σ achievement runs) reach converged σ at n_pts = 40k, but compute σ only — not the Yukawa pipeline.

The chain-channel sweeps `p7_11_*` and `p7_8b_*` reach the basis-convergence sweep but every one has `"converged": false`. The chain is not converged at any documented basis size.

### D.2 Output files referenced by the paper

Per `REMEDIATION_PLAN_OPTION_B.md` §1 audit and §2.2 problem table:
- σ-channel: `output/p5_10_*` files (Schoen-vs-TY 6.92σ at Tier 0 strict, per `MEMORY.md` "TY-vs-Schoen at 6.92σ Tier 0" entry). These are the ONLY paper-headline numbers currently from a converged JSON.
- Chain-channel: `output/p7_11_quark_chain.json`, `p7_11_lepton_chain.json`, `p8_5_chain_k4_gpu.json` — all have `"converged": false`. Per the remediation plan, the chain-channel BF that the paper claims to compute is currently coming from these non-converged JSONs through the deprecated `from_chain_match` adapter in `bayes_factor_multichannel.rs:845–879` — the BLOCKER that Phase 2 of `REMEDIATION_PLAN_OPTION_B` exists to fix.
- Yukawa-channel: `output/p8_3b_yukawa_production.json` — the one production end-to-end Yukawa run. **Not currently consumed by `bayes_factor_multichannel`** (per `references/p_chain_matcher_deprecation.md` §1, `from_yukawa_pipeline_prediction` does not exist yet).

### D.3 What's the missing piece?

A **purpose-built Stage 3 catalogue post-processor**. The pipeline parts exist; what's missing is a binary that:

1. Loads converged Donaldson + HYM metrics for Schoen/Z_3×Z_3 (and TY/Z_3 for comparison).
2. Solves `solve_harmonic_zero_modes` for `H^1(X, V)` — already does this.
3. Solves a sister `solve_harmonic_zero_modes_endv` for `H^1(X, End V)` — **does not exist**.
4. (Optional) Solves for `H^1(X, ∧^2 V)`, `H^1(X, ∧^3 V)` for an SO(10) decomposition — **does not exist**.
5. Reports the **full** eigenvalue spectrum from the bundle Laplacian (already in `eigenvalues_full`) and runs a `chain_matcher`-style assignment against the framework's full chain `{0, 11, 17, 4, 13/2, 15, 22.5, 26, 36.5, 34.5, 1072/30, 1132/30}` — **the framework chain itself does not exist as a `ChainType` variant in `chain_matcher.rs`**; only the `Lepton` and `Quark` sub-sets do.
6. Runs `compute_yukawa_couplings` on the full 9 (or larger) harmonic basis, with a Wilson-line sector assignment that addresses **9 characters** rather than 3.
7. Outputs a single `output/p_stage3_catalogue.json` with all of (1)–(6).

---

## E — Sector / index conventions

### E.1 Documented decomposition convention used in the codebase

The codebase has a **multi-layered** convention:

**Layer 1 — `wilson_line_e8.rs` (single Z_3 Wilson line):**
- `E_8 → E_6 × SU(3)` via the canonical `ω_2^∨` SU(3)-coweight (Slansky 1981 Tab. 23).
- Lie-dim 86 = 78 (E_6) + 8 (SU(3)). Tested: `canonical_wilson_line_unbroken_dim_86`.
- `27` of E_6 decomposes to `(3, 3̄) ⊕ (3̄, 3) ⊕ (1, 8) ⊕ (1, 1)` under the SU(3) bundle structure group (per `yukawa_sectors_real.rs:13–15` module doc).

**Layer 2 — `wilson_line_e8_z3xz3.rs` (commuting Z_3 × Z_3 pair, Schoen):**
- Module doc lines 28–47: joint commutant is `SU(3)^4` (trinification), dim 32 = 24 invariant roots + 8 Cartan, rank 8.
- This **supersedes a previous (and now retracted) erratum** that claimed the commutant was `SO(10) × SU(3) × U(1)^2` with dim 41 (line 58–64). The current correct claim is `SU(3)^4`.
- The `(g_1, g_2)` character table (line 80–88) covers only **6 of 9 b_lines** — the diagonal `(0,0), (1,1), (2,2)` for the `(1,0)` block and the conjugate-flipped `(0,0), (1,2), (2,1)` for the `(0,1)` block.

**Layer 3 — `yukawa_sectors_real.rs` (sector assignment of harmonic modes):**
- DHOR 2006 / ALP 2011 minimal-monad dictionary (line 230–233):
  - class 0 → Higgs / singlet
  - class 1 → up-quark (and lepton-doublet under SU(5))
  - class 2 → down-quark (and charged-lepton-singlet)
- Falls back to round-robin equal-split if any sector is empty (line 256–271). This **is currently triggered** in 2-factor production runs because the 9-character Z_3×Z_3 collapses to 3 mod-3 classes.

**The user's task brief invokes "BHOP rank-4 SU(4) bundle, SO(10) × U(1) decomposition".** The codebase's `MonadBundle::anderson_lukas_palti_example()` and `MonadBundle::schoen_z3xz3_canonical()` are both **rank-3 SU(3) AKLP-style** monad bundles — NOT BHOP rank-4 SU(4). The SU(4) → SO(10) × U(1) chain that the framework prediction references is **NOT implemented in the current codebase**. The deployed pipeline is on a rank-3 SU(3) bundle and the SO(10) × U(1) language in the brief reflects what Stage 3 should target, not what the code currently does.

This is consistent with `REMEDIATION_PLAN_OPTION_B.md` §P1.3 ("Structure group of V — settle the SU(3) vs SU(4) vs SU(5) ambiguity") which still flags this as an open Phase 1 item.

### E.2 Per-sector mode counts: predicted vs produced

| Sector | Framework prediction | Codebase produces | Notes |
|---|---|---|---|
| Up-quark family | 3 (one per generation) | `sectors.up_quark.len()` ≈ 3 (post-mod-3 round-robin) | Index list from `assign_sectors_dynamic`. Mode SHAPES are near-degenerate → rank-1 contraction. |
| Down-quark family | 3 | `sectors.down_quark.len()` ≈ 3 | Same. |
| Charged-lepton family | 3 | `sectors.lepton.len()` ≈ 3 | Same. |
| Neutrino family (right-handed) | 3 (in SU(4) → SO(10) decomposition) | **Not produced separately** | The 16-rep of SO(10) embeds the right-handed neutrino; the code doesn't break out neutrino slots. |
| Higgs doublet(s) | 1 (or possibly 2 in MSSM-like) | `sectors.higgs.len()` = full phase-class-0 list (≥ 1; `higgs[0]` is the lowest-eigenvalue mode) | Lives in `H^1(V)` as a proxy; should live in `H^1(End V)`. See A.5 item 2. |

`compute_zero_mode_spectrum` (`zero_modes.rs:1332`) reports `n_27 = 9` on the AKLP TY bundle (one per `H^1(V)` mode upstairs, divided by `|Γ| = 3`); this is consistent with **3 generations × 3 SU(3)-flavour-positions = 9 harmonic modes**, which is what the harmonic solver produces. The framework predicts 3 generations × 1 family-slot each = 3 modes per fermion class **after** SU(3) flavour-summing — this summing is what `extract_3x3_from_tensor` is meant to do but currently fails at because of the rank-1 mode degeneracy.

---

## Final gap list (ranked by effort to close Stage 3)

### Critical path (must close to declare Stage 3 done)

1. **G1 — Non-degenerate harmonic modes on Schoen.** The `solve_harmonic_zero_modes` mode-shape degeneracy is the upstream root cause of the rank-1 Yukawa collapse. **Fix avenue:** larger polynomial-seed basis (Z3XZ3_MAX_TOTAL_DEGREE → 6, currently 4 in production); or migrate to the genuine bundle-twisted Dirac kernel (currently approximated as Bochner Δ on holomorphic monomials). Effort: **2–4 weeks** (1 person; involves a basis-convergence sweep at increasing cap, then a kernel-solver upgrade if cap-up doesn't converge).

2. **G2 — Schoen-canonical monad bundle (real, not aliased AKLP).** Implement `chern_classes()` for 3-factor ambients in `zero_modes.rs`; populate `MonadBundle::schoen_z3xz3_canonical()` with genuine Schoen bundle data (BHOP rank-4 SU(4) per the framework brief, OR the AKLP-faithful rank-3 SU(3) per current code). Effort: **1–2 weeks** (1 person; mostly `chern_classes` 3-factor extension + literature dive on BHOP bundle data).

3. **G3 — `H^1(X, End V)` solver for the Higgs sector.** Implement a `solve_harmonic_zero_modes_endv` companion to the existing `H^1(V)` solver. End-V cohomology requires a double Koszul chase that's currently flagged out of scope. Effort: **2–3 weeks** (1 person; BBW + Koszul double chase, plus new `MonadBundleEndV` struct).

4. **G4 — Bundle-Laplacian eigenvalue → framework-chain matching.** Add the framework's full chain `{0, 11, 17, 4, 13/2, 15, 22.5, 26, 36.5, 34.5, 1072/30, 1132/30}` as a `ChainType` variant in a successor-to-`chain_matcher` module. Plug `solve_harmonic_zero_modes::eigenvalues_full` into it (NOT `compute_metric_laplacian_spectrum`, which is the deprecated scalar-Laplacian path). Effort: **0.5–1 week** (1 person).

5. **G5 — Stage 3 catalogue binary.** Implement `src/bin/p_stage3_catalogue.rs` driving G1–G4 end-to-end and emitting a single `output/p_stage3_catalogue.json` with: harmonic modes per `(p, q)`, full bundle-Laplacian spectrum, chain residuals per chain position, triple-overlap tensor + per-sector 3×3 SVD masses + CKM, with bootstrap CIs. Effort: **0.5–1 week** (1 person).

### Important but not blocking (publication-grade quality)

6. **G6 — Quadrature uniformity on Schoen.** Current 0.0081 score is unacceptable. Likely fix: re-weight the `Cy3MetricResultBackground::from_schoen` adapter (`yukawa_pipeline.rs:313`). Effort: **0.5–1 week**.

7. **G7 — 9-class Wilson-line sector decomposition.** Currently 9 (g_1, g_2) classes collapse to 3 mod-3 classes. Extend `assign_sectors_dynamic` to address all 9 explicitly. Effort: **1 week**.

8. **G8 — `H^1(X, V^*)` (27̄) computation.** Currently asserted = 0; needs LES-via-`monad_h2` programmatic check. Effort: **0.5 week**.

9. **G9 — `H^1(X, ∧^q V)` for `q = 2, 3` if SU(4) bundle.** Required for an SO(10) × U(1) decomposition with the BHOP bundle. Effort: **2 weeks** (only if G2 picks the SU(4) path).

### Nice-to-have

10. **G10 — Multi-seed bootstrap on Stage 3 outputs.** Current production sweeps use single seed = 12345; per `REMEDIATION_PLAN_OPTION_B.md` Phase 2.3, ≥ 100 seeds with bootstrap BCa CIs is the publication-grade target. Effort: **1 week** (mostly compute, parallelisable).

---

## Recommended unblock plan

**Step 1 (parallel):** G1 + G2 in parallel. G1 is a basis-convergence sweep (computational) + possibly a kernel-solver upgrade (mathematical). G2 is mostly literature-driven. Both are 1-person tasks; doing them in parallel is the natural sequencing. **Combined wallclock: 2–4 weeks.**

**Step 2:** G3. Cannot start until G2 lands (need a real Schoen bundle to twist End V around). **+ 2–3 weeks.**

**Step 3 (parallel):** G4 + G5. Both are wiring tasks once G1–G3 produce the right shapes. Stage 3 catalogue binary depends on G4 for chain matching. **+ 1–2 weeks.**

**Step 4:** G6 + G7 + G8 + G10 in parallel for publication polish. **+ 1–2 weeks.**

**Step 5 (only if framework requires SU(4)):** G9. **+ 2 weeks.**

**Total estimate:**
- **Critical path (G1–G5)** to declare Stage 3 "done" with a publication-grade catalogue: **5–9 weeks** (1.25–2.25 person-months) for one engineer working full-time.
- **With publication polish (G6–G8, G10)**: add **1–2 weeks** if running G6/G7/G8/G10 in parallel after the critical path. **6–11 weeks total.**
- **With SU(4) → SO(10) × U(1) extension (G9)**: add **2 weeks**. **8–13 weeks total.**
- Compatible with the `REMEDIATION_PLAN_OPTION_B.md` overall 6–12 person-week estimate (Phases 1, 2, 3 of that plan map cleanly onto G2, G3+G4+G5, and G6+G10 here).

**Critical-path single gap:** **G1 — non-degenerate harmonic modes on Schoen.** Until this is fixed, the rank-1 Yukawa collapse persists, and Stage 3 cannot produce a usable Yukawa matrix regardless of how clean the rest of the wiring is. Every other Stage 3 deliverable degrades to a fancy `m_t`-only diagnostic without it.
