# Audit: RG running module (REM-OPT-B Phase 2.6)

**Status (wave-1, 2026-05-04)**: COMPLETE — documentation-only audit. No
code change in this pass. Defects flagged for user prioritization.

## Module paths

* `src/route34/rg_running.rs` (255 lines) — thin convenience wrapper that
  delegates to the legacy SM RG runner. Provides
  `run_yukawas_to_mz()` and a `top_yukawa_running_ratio()` diagnostic.
  Contains its own 6×6-real-block `top_singular_value()` helper for the
  ratio test.
* `src/pdg.rs` (2371 lines) — actual implementation. Owns the β-functions
  (`beta()`, `beta_two_loop_correction()`, `beta_total()`), the RK4
  integrator (`rk4_step()`), and the staged 6f→5f runner
  (`rg_run_to_mz_with()`).

`route34/rg_running.rs` is a wrapper. All audit content below targets
`pdg.rs:700-1151` (β-functions + integrator + driver) and `pdg.rs:1175-1346`
(post-M_Z analytic running used inside `extract_observables`).

## 1. Two-loop coverage matrix

The runner evolves three Yukawa matrices (Y_u, Y_d, Y_e — each 3×3 complex)
and three SM gauge couplings (g_1, g_2, g_3). All entries of each Yukawa
matrix run; there is no per-flavour selection. Per-coupling loop coverage:

| Coupling                     | 1-loop | 2-loop | Notes |
|------------------------------|:------:|:------:|-------|
| Y_u (entire 3×3 matrix)      | YES    | YES    | β includes pure-Yukawa^4 (3/2 H_u² − 1/4 (H_u H_d + H_d H_u) + 11/4 H_d²), Yukawa²×gauge² cross-terms (223/80 g₁² + 135/16 g₂² + 16 g₃²)·H_u, pure-gauge⁴ diagonal (1187/600 g₁⁴ + 3/20 g₁²g₂² + 19/15 g₁²g₃² − 23/4 g₂⁴ + 9 g₂²g₃² − 108 g₃⁴), and the Y_4(S) trace contribution |
| Y_d (entire 3×3 matrix)      | YES    | YES    | analogous (187/80 g₁², 135/16 g₂², 16 g₃², gauge⁴ −127/600·g₁⁴ + 27/20·g₁²g₂² + 31/15·g₁²g₃² − 23/4·g₂⁴ + 9 g₂²g₃² − 108 g₃⁴) |
| Y_e (entire 3×3 matrix)      | YES    | YES    | analogous (387/80 g₁², 135/16 g₂², no g₃; gauge⁴ 1371/200·g₁⁴ + 27/20·g₁²g₂² − 23/4·g₂⁴) |
| g_1 = √(5/3) g'              | YES    | NO     | β = b₁ g₁³/(16π²), b₁ = 41/10. **Two-loop gauge-coupling back-reaction explicitly skipped** (see `pdg.rs:879-881`). |
| g_2                          | YES    | NO     | β = b₂ g₂³/(16π²), b₂ = −19/6 |
| g_3                          | YES    | NO     | β = b₃ g₃³/(16π²), b₃ = −7 (6-flavour, μ > m_t) or b₃ = −23/3 (5-flavour, μ < m_t) |
| Higgs self-coupling λ        | NO     | NO     | **Not evolved.** The 2-loop Y^4 + gauge cross-terms involving λ are skipped (see `pdg.rs:721-724`). v(M_Z) = 248.401 GeV is hard-coded, not derived. |

The Yukawas inherit two-loop running through the **whole matrix**, not just the
top, bottom, tau components. There is no per-flavour two-loop downgrade.

So the row-by-row breakdown the plan asks for collapses: y_t / y_c / y_u
all live in Y_u and all run at 2-loop together; y_b / y_s / y_d live in
Y_d and all run at 2-loop together; y_τ / y_μ / y_e live in Y_e and all
run at 2-loop together. There is no separate single-flavour β.

**Coverage classification**: PARTIAL. Yukawa matrices are full 2-loop in the
dominant Y⁴ + Y²g² + g⁴ + Y₄(S) trace pieces. Skipped at 2-loop:
* Higgs self-coupling λ contributions to Yukawa β (deferred — λ²≈0.017 vs
  g₃²y_t²≈1.4 in the dominant gauge-Yukawa term, ≈sub-permille over 14
  e-folds; see `pdg.rs:721-723`).
* Sub-leading trace cross-terms beyond the leading MV pattern.
* 2-loop gauge β (kept 1-loop; 2-loop gauge-Yukawa back-reaction is
  sub-percent over 14 e-folds per the module's own justification).
* 3-loop QCD pieces in either Yukawa or gauge β (≈10⁻⁴).

References cited in code: Machacek-Vaughn NPB 222, 236, 249 series
(`pdg.rs:11-13`, `pdg.rs:702-703`); Bednyakov-Pikelner-Velizhanin
arXiv:1212.6829 eqs. 4.1-4.3 (`pdg.rs:703`); Mihaila-Salomon-Steinhauser
arXiv:1208.3357 (`pdg.rs:14-15`).

## 2. Threshold structure

The runner's threshold model is staged + hard-cutoff at the top, plus
post-M_Z analytic α_s threshold matching for the light/charm/bottom
quarks inside `extract_observables`. Specifically:

| Threshold | Implemented? | How | Where |
|-----------|:-----------:|----|-------|
| **m_t = 172.57 GeV** | YES | Hard cutoff at μ = m_t. Phase 1: μ_init → m_t with `n_top = 1`, `b₃ = −7`. Phase 2: m_t → M_Z with `n_top = 0`, `b₃ = −23/3`. Captures `y_t_at_mt` at the threshold for downstream m_t reconstruction. The "n_top = 0" branch additionally drops H_u[2][2] from the Y₂(S) trace (1-loop) and from Y_4(S) (2-loop) — a step-function approximation, not a full diagrammatic decoupling. | `pdg.rs:1085-1126`, `pdg.rs:906-913`, `pdg.rs:751-756` |
| **m_b = m_b(m_b)** | PARTIAL | Used as α_s flavor-threshold inside `extract_observables` (5f→4f matching: `α_s^(4)(m_b) = α_s^(5)(m_b)`) for **light-quark mass running between M_Z and 2 GeV**, NOT as a Yukawa-β threshold. The Yukawa runner does NOT integrate out the bottom or any below-m_t quark. | `pdg.rs:1318-1320` |
| **m_c = m_c(m_c)** | PARTIAL | Same as m_b — α_s threshold matching only, used inside `extract_observables` for charm mass extraction (4f→3f? actually still 4f, see code). | `pdg.rs:1331-1334` |
| **M_W = 80.377 GeV** | NO | Not a separate threshold. SM-style: gauge bosons live in the unbroken phase, the Higgs VEV is treated as a single constant from M_Z down. Electroweak symmetry breaking is **not** dynamically resolved — `v_mz = 248.401 GeV` is hard-coded (`pdg.rs:1138`). | — |
| **M_Z = 91.1880 GeV** | YES | Hard endpoint of the runner. No matching to a 4f or non-SM theory below M_Z. | `pdg.rs:1059, 1064, 1141` |

So the **gauge β** sees one threshold (m_t, 6f→5f). The **Yukawa β** sees
the same threshold (the H_u trace contraction). The **post-M_Z light-quark
mass running** sees additional thresholds at m_b and m_c via the
n_f-dependent QCD anomalous dimension (Tarrach 1981; Buras 1980 —
`pdg.rs:1181-1187`).

Threshold classification: **STEP-FUNCTION**, not full diagrammatic
matching. The `n_top = 0` branch additionally truncates the top entry of
H_u from the Yukawa trace pieces, but no log-corrections / finite
matching coefficients are applied at the threshold crossing. This is
explicitly acknowledged in the module docstring as a 0.1%-level
simplification (`pdg.rs:910-912`).

Castano-Piard-Ramond 1994 and Bardeen-Buras-Duncan-Gaillard 1996 are not
cited in the code. The full pole↔MS-bar matching at m_t is implemented
separately (Marquard-Smirnov-Steinhauser² 2015 4-loop, `pdg.rs:1192-1225`)
but only at the observable-extraction stage (m_t pole reconstruction),
not as a coupling-matching coefficient inside the running.

## 3. Scheme commitment at M_Z

**The RG runner outputs at M_Z in the MS-bar scheme.**

Direct evidence in the code:

1. The β-functions are the standard SM MS-bar Machacek-Vaughn series
   (`pdg.rs:702-703`, `pdg.rs:890-891`). MS-bar is the only
   renormalization scheme in which these β coefficients hold without
   additional finite-matching corrections.
2. The output Yukawa matrices Y_u(M_Z), Y_d(M_Z), Y_e(M_Z) are not
   subjected to any pole-mass / on-shell finite renormalization
   correction at M_Z — they're the integrator's terminal state and
   nothing else.
3. The pole-mass conversion happens **only** for m_t, downstream of the
   runner, via `pole_from_msbar()` (`pdg.rs:1209-1225`) and is applied
   to `y_t_at_mt × v/√2` after the running. m_t pole is reconstructed
   from MS-bar at α_s(m_t) at 4-loop QCD (Marquard et al. 2015).
4. The lepton masses extracted at M_Z are described in code as
   "approximately pole" (`pdg.rs:1285`) — this is a known approximation
   acknowledging that the QED MS-bar→pole matching at the lepton scale
   roughly cancels the bare MS-bar→MS-bar QED running between M_Z and
   m_l. No pole-mass renormalization is performed inside the runner.
5. Light-quark and charm/bottom masses are explicitly MS-bar at their
   respective reference scales (see `pdg.rs:24-26, 254-262`).
6. v(M_Z) = 248.401 GeV is taken from Buttazzo et al. JHEP 2013 — that
   value is the **MS-bar-running** Higgs VEV at M_Z (= the `v` that
   satisfies m_t = y_t v/√2 with y_t the MS-bar Yukawa). The on-shell
   VEV is ≈ 246.22 GeV; the runner uses the MS-bar value
   (`pdg.rs:237, 243, 1138`).

So the scheme commitment is unambiguous: **MS-bar at M_Z, with one
optional downstream pole-mass conversion for m_t only**.

## 4. Defects and gaps

### D1 [HIGH]: sin²θ_W is hard-coded at the MS-bar value, not on-shell

Location: `src/route12/route2.rs:105-106`.

```rust
const SIN2_THETA_W: f64 = 0.23121;
let alpha_2 = ALPHA_EM_MZ / SIN2_THETA_W;
```

`0.23121` is PDG-2024's **MS-bar** value sin²θ̂_W(M_Z) (from the Erler-
Freitas EW review chapter). The on-shell scheme value
1 − m_W²/m_Z² = 1 − (80.377/91.1880)² ≈ 0.22290 is roughly **3.7% lower**.

The plan note flags: "the paper claims sin²θ_W matches on-shell
1 − m_W²/m_Z², NOT MS-bar — flag if RG runner targets MS-bar instead."

Verdict: the runner targets MS-bar. `route2.rs` consumes the
MS-bar value of sin²θ_W to construct α_2(M_Z) = α_em / sin²θ_W. If the
publication's gauge-Yukawa prediction is supposed to be benchmarked
against the **on-shell** Weinberg angle (the geometric definition
1 − m_W²/m_Z²), then `route2.rs:105` is incorrect and the predicted
α_2(M_Z) is off by ≈3.7%. Conversely, if the publication target is
MS-bar Weinberg angle, then `route2.rs` is correct but the publication
claim about "on-shell matching" is mis-stated.

The decision between these two interpretations is the **author-
prioritized scheme commitment** that this audit cannot resolve in
isolation.

### D2 [MEDIUM]: 2-loop gauge-coupling β is missing

Location: `src/pdg.rs:879-881`, `pdg.rs:948-953`.

The module docstring (`pdg.rs:55, 725`) states the omission is
"sub-percent over 14 e-folds." For a publication aiming at >5σ
discrimination on 13 PDG observables this should be quantified rather
than asserted: 2-loop gauge β shifts α_s(M_Z) by O(0.5%) relative to
1-loop running from M_GUT, which propagates into m_b(m_b) and m_t at
the ≈1-2% level via the QCD anomalous dimension (12/(33−2nf)
exponent in `run_msbar_mass`). This is the same order as several
PDG uncertainties.

### D3 [MEDIUM]: Higgs self-coupling λ is not evolved

Location: `pdg.rs:721-724`, `pdg.rs:1138`.

v(M_Z) is hard-coded to the Buttazzo et al. value (248.401 GeV). The
2-loop Yukawa β contributions involving λ are skipped. This couples to
D2: a self-consistent 2-loop SM RG would evolve λ alongside the Yukawas
and gauge couplings, derive v(M_Z) from m_H and λ(M_Z), and feed it
back into the m_t reconstruction. The current pipeline takes v(M_Z) as
external input.

### D4 [LOW]: m_b and m_c are not Yukawa-β thresholds, only α_s thresholds

Location: `pdg.rs:1318-1334`.

The Yukawa runner integrates from μ_init down to M_Z without any
matching at m_b or m_c. Below M_Z the only running performed is α_s
and the 1-loop QCD anomalous dimension on the **MS-bar quark masses**
(`run_msbar_mass`). The Yukawa matrix entries Y_d[1][1], Y_d[2][2],
Y_u[1][1] are simply taken at M_Z without further evolution. For the
PDG observables this is correct (PDG reports m_b(m_b), m_c(m_c),
m_q(2 GeV) — these ARE the post-M_Z analytic running endpoints) but
the plan's mention of "Bardeen et al. 1996" (full diagrammatic
matching) is not implemented. Step-function with 1-loop α_s only.

### D5 [LOW]: top decoupling is approximate (drop H_u[2][2] from Y₂(S))

Location: `pdg.rs:906-913`, `pdg.rs:751-756`.

For non-hierarchical input matrices (e.g. flavour-mixing scenarios where
the top isn't the (3,3) entry of Y_u), this approximation is
incorrect. For all physically sensible heterotic input bases (where
the top IS the dominant singular value and IS aligned to the (3,3)
entry through the bi-unitary SVD pre-RG), the error is ≤0.1%. Code
acknowledges this (`pdg.rs:910-912`). Keep as-is unless the
production pipeline starts feeding non-canonical bases into the runner.

### D6 [LOW]: VEV running between μ_init and M_Z is hard-coded, not integrated

Location: `pdg.rs:1138`.

```rust
// Running VEV at M_Z. We approximate by linear interpolation between
// the tree-level v at mu_init and the PDG v(M_Z) = 248.401 GeV.
let v_mz = 248.401_f64;
```

The "linear interpolation" comment is misleading — the code simply
hard-codes the M_Z value. There is no actual interpolation between
v(μ_init) and v(M_Z). For the purpose of `extract_observables` this
is fine because all masses are reconstructed at M_Z, but the
docstring is stale.

## 5. Recommendation

The scheme commitment is **internally consistent**: MS-bar everywhere,
with the m_t pole conversion isolated to a downstream call site. This
is the right design.

The **external consistency** with the publication's claimed observables
is what's at issue:

* If the publication's reported sin²θ_W comparison uses the on-shell
  scheme 1 − m_W²/m_Z² ≈ 0.22290 then `route2.rs:105` should be
  changed to 0.22290 (or computed from `1.0 - (m_W/m_Z).powi(2)` with
  the PDG W and Z pole masses) and the corresponding chi² entries
  re-evaluated. **This is a 3.7% shift in α_2(M_Z) — not a permille
  effect.** It will move the route-2 chi² contribution noticeably.
* If the publication uses the MS-bar value, the current code is
  correct but the prose in the publication / module docstrings should
  be updated to remove any "on-shell sin²θ_W" claim.

Pick one. The **recommended choice for an MS-bar runner** is to keep
sin²θ̂_W(M_Z)^MS-bar = 0.23121 (consistency with everything else in
the runner) and amend the publication wording. Targeting on-shell
sin²θ_W with MS-bar Yukawas is a mixed-scheme trap that requires
explicit finite matching coefficients which the code does **not** apply.

Other gaps (D2-D6) are non-blocking for the >5σ discrimination claim
established in P5.10 v7 — the discrimination signal lives in
ratios and chain-matching rather than in absolute mass precision —
but the **author should be aware** when quoting absolute mass / mixing
predictions to PDG precision (≤permille on m_b, m_c, sin²θ_W) that the
RG runner has ≈0.5-1% systematic from D2 and ≤3.7% systematic from D1.

Priority for fixes (if pursued):

1. **D1 [HIGH]** — Resolve the on-shell-vs-MS-bar sin²θ_W question.
   Either fix `route2.rs:105` to the on-shell value AND add an
   explicit "this enters MS-bar→on-shell mixed scheme" note, OR
   leave the code as-is and audit the publication wording.
2. **D2 [MEDIUM]** — Add 2-loop gauge β if quoting absolute α_s(M_Z),
   m_b(m_b), or top pole mass to ≤permille precision.
3. **D3 [MEDIUM]** — Add λ evolution if quoting v(M_Z) self-
   consistently or comparing m_H predictions.
4. **D4-D6 [LOW]** — Defer; all are documented in the code and
   have ≤0.1% impact.

This audit is documentation-only. No production code is modified.
