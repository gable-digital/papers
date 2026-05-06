# Stage 3 Fix B — Upstream-code spike (wrap vs bespoke)

**Spike date:** 2026-05-04
**Triggered by:** hostile-review task `a7e1e8d7be27051bf` flagging the G1
agent's 3–4-week bespoke Fix B estimate as too high if a published
heterotic-numerics implementation could be wrapped instead.
**Scope:** 1–2 day research spike — "do we know whether wrapping is
viable", not "ship the wrapper". No `rust_solver/` files touched.
**Spike artifacts:** `python_research/spike_dirac_upstream/`.

## TL;DR

> **Bespoke Fix B is the only path for our specific (Schoen Z₃×Z₃ + BHOP
> rank-4 SU(4) extension bundle) setup.** The cleanest upstream candidate
> (`cymyc`, MIT, arXiv:2410.19728) is excellent code that hard-codes
> exactly the two assumptions that make it a non-fit:
>
> 1. Bundle V ≡ T_X (standard embedding only) — no abstraction for V ≠ T_X.
> 2. Hypersurface / multi-hypersurface CICY ambient only — no fiber-product
>    Schoen encoder.
>
> Effort estimate **does not shrink** by wrapping. A "wrap-and-extend"
> attempt would, after gluing, re-implement ~80 % of M1+M2 from G1's plan
> and inherit a JAX runtime dependency. **Recommended:** proceed with
> G1's bespoke 3–4 week Fix B in `rust_solver/`, optionally cite
> `cymyc`'s spectral-NN harmonic objective as the reference design for
> M1/M2 (not as wrappable code).

---

## Step 1 — Candidate research

### Candidate A: ACLP 2017 ("1707.03442")

**Status: paper ID was wrong.** arXiv:1707.03442 is Cieliebak–Eliashberg
"Weinstein manifolds revisited" (symplectic topology), not heterotic
Yukawa. The Anderson–Constantin–Lukas–Palti / Blesneag papers in the
relevant series are:

- arXiv:1512.05322 "Holomorphic Yukawa Couplings in Heterotic String
  Theory" (Blesneag–Buchbinder–Candelas–Lukas, 2015)
- arXiv:1607.03461 "Holomorphic Yukawa Couplings for Complete
  Intersection Calabi–Yau Manifolds" (2016)
- arXiv:1801.09645 "Matter Field Kähler Metric in Heterotic String
  Theory from Localisation" (2018)

None of these papers ship public numerical code on arXiv ancillary or
GitHub. They are analytic/algebraic methods for *holomorphic* Yukawa
couplings (the easy half — protected by the Kodaira–Spencer map). The
*physical* Yukawa requires the Kähler metric on matter fields, which is
the bundle-twisted-Dirac-kernel / harmonic-(0,1)-form computation Fix B
is meant to address.

**Verdict:** No usable upstream code from this candidate.

### Candidate B: Butbaia 2024 (arXiv:2401.15078) → cymyc (arXiv:2410.19728)

This is the bullseye candidate. Butbaia–Mayorga-Peña–Tan–Berglund–
Hübsch–Jejjala–Mishra "Physical Yukawa Couplings in Heterotic String
Compactifications" (2024) is the direct intellectual precursor to Fix B
M1/M2 — they compute the *physical* Yukawa via spectral neural network
construction of harmonic V-valued (0,1)-forms.

The associated open-source library is **`cymyc`** (Calabi–Yau Metrics,
Yukawas, Curvature), shipped with arXiv:2410.19728:

| Field          | Value                                                    |
|----------------|----------------------------------------------------------|
| GitHub         | https://github.com/Justin-Tan/cymyc                      |
| Docs           | https://justin-tan.github.io/cymyc/                      |
| License        | **MIT** (wrappable, no GPL contagion)                    |
| Language       | Python ≥ 3.10, JAX-based                                 |
| Activity       | 213+ commits, "under active development" as of 2025      |
| Method         | Spectral NN ansatz + ∂̄-correction objective              |
| Output         | Bundle-valued (0,1)-form harmonic representatives + WP   |
| API entrypoint | `cymyc.approx.harmonic.HarmonicFull`                     |

`HarmonicFull.__call__` produces "all harmonic representatives by ∂̄-exact
correction to a representative from the H^{0,1} Dolbeault cohomology,
ξ; η = ξ + ∂̄θ", and exposes companion methods `codifferential_eta`,
`wp_metric_harmonic`, `inner_product_Hodge`. This is *exactly* the M1/M2
machinery G1 specified — at the algorithmic level.

### Candidate C: cymetric / cyjax (broader Anderson–Lukas–Ruehle group)

- **cymetric** (Larfors–Lukas–Ruehle–Schneider): GPL-3.0, TensorFlow,
  CICY/KS-list metrics only, *no* bundle support (forthcoming/asterisked
  in README), *no* harmonic-form computation. GPL-3.0 also creates a
  copyleft-contagion concern if linked into our larger code base.
- **cyjax** (Gerdes–Krippendorf): MIT, JAX, CY *metric* only, no
  bundle-twisted Yukawa machinery.

Neither is a candidate for Fix B because both stop at the Ricci-flat
metric; they don't implement the bundle-valued (0,1)-form / Yukawa
layer.

---

## Step 2 — Compatibility assessment (cymyc vs our setup)

| Axis                     | Our setup                        | cymyc                                      | Match? |
|--------------------------|----------------------------------|--------------------------------------------|--------|
| Manifold encoder         | Schoen = fiber product of two dP₉ surfaces over P¹, with Z₃×Z₃ free quotient | hypersurface / multi-hypersurface CICY in product of P^n's, via `examples/poly_spec.py` | **NO** |
| Bundle V                 | BHOP rank-4 SU(4) extension bundle (BHOP-2005 §6, hep-th/0505041) | hard-coded as the holomorphic tangent bundle T_X (standard embedding) — no abstraction layer for V ≠ T_X (confirmed by reading `cymyc/approx/harmonic.py`: `T_X_section`, `section_network` build sections from ambient coords + pullback matrices) | **NO** |
| Hermitian metric on V    | HYM connection from `hym_hermitian.rs` (Donaldson-balanced) | Fubini–Study restricted (since V = T_X is implicit) | **NO** |
| Discrete quotient        | Z₃ × Z₃ free action + Wilson line breaking E₈ → SO(10) → SU(5) × U(1) → SM | only the upstairs Tian–Yau example uses Z₃-equivariant deformations (`TY_KM_poly_spec.py`); generic-quotient bookkeeping not abstracted | partial |
| Metric source            | Donaldson balanced metric (Rust solver, `rust_solver/`) writing JSON checkpoints in our own format | cymyc consumes `metric_fn: Callable` from a pre-trained JAX pickle; format mismatch | adapter |
| Sample-point convention  | our format from rust_solver        | (400000, 12) real-coord arrays, 12 = 2·6 for X₃₃ in P⁵ | adapter |
| License                  | (we are pre-publication; no constraint) | MIT — no contagion, OK to vendor | **OK** |
| Runtime dep              | Rust + Python NumPy/SciPy        | adds JAX + Python ≥ 3.10 + (optional) GPU | acceptable |

**The two NO rows are structural, not glue-layer**:

1. **Bundle hardcoding.** `HarmonicFull` builds sections of T_X from the
   ambient Jacobian via pullback. To compute sections of an SU(4)
   extension bundle V → X (which is *not* a quotient of an ambient
   tangent bundle), the network's coefficient head, the section ansatz,
   AND the codifferential operator (which currently uses ∂̄ on T_X
   trivialised via ambient coords) must all be re-derived from scratch
   for V's local trivialisations and HYM (0,1)-connection coefficients
   A^I_{ī J}. That's M2 in G1's plan. cymyc does not give it to us.

2. **Manifold encoder.** `examples/poly_spec.py` lays out CY3s as
   monomial+coefficient triples in a product of projective spaces. Our
   Schoen is a fiber product of two dP₉ surfaces over P¹ — the universal
   ambient is P²×P²×P¹ with a co-rank-2 system (BHOP-2005 §6,
   `(3,0,1),(0,3,1)`). cymyc's `monomials/dQdz_monomials/dQdz_coeffs`
   schema can in principle express this, *but* the spectral-NN section
   ansatz uses ambient-coordinate polynomials of degree set by a single
   ambient — it is not currently exercised on multi-ambient
   non-Cartesian-degree CICYs of the Schoen shape. Best case: a
   100–300-line `schoen_poly_spec.py` is needed to even feed the
   library; whether the spectral basis is rich enough to converge on
   the fiber-product geometry is an open empirical question.

3. **Quotient + Wilson line.** Even if (1) and (2) were solved, our
   downstream chain-matcher needs *equivariant* harmonic forms (the
   Wilson-line projection picks SO(10)/SU(5)/SM-rep components from the
   BHOP cohomology). cymyc's TY example does Z₃-equivariance by
   *restricting the deformation basis* — fine for h^{2,1} on the
   tangent bundle, not a substitute for projecting H¹(X, V) onto
   irreducible representations of Z₃ × Z₃.

**Important note** (from arXiv:2401.15078): Butbaia et al. *only*
compute physical Yukawas in the **standard embedding** (V = T_X) on
quintic, X₃₃ bicubic, and Tian-Yau (with a Z₅ × Z₅ Fermat-quintic
quotient as a CFT cross-check). They explicitly do **not** treat
non-standard-embedding rank-4 SU(4) bundles or the Schoen geometry. So
even the *paper* the library was written for does not solve our
problem; cymyc is a precursor implementation for a strictly weaker
case.

---

## Step 3 — Wrap-vs-port feasibility

### Option W1: "Thin wrapper, V = T_X only" (3–5 days)

Wrap cymyc to compute harmonic T_X-valued (0,1)-forms on the *upstairs*
Schoen via a `schoen_poly_spec.py`, ignoring the Z₃ × Z₃ quotient and
the BHOP extension bundle. Produces standard-embedding Yukawas on the
covering space.

- ✅ Doable in 3–5 days.
- ❌ Does **not** solve Stage 3 Fix B. Standard-embedding Yukawas on the
  covering space are not the chain-match observable. The whole point of
  Fix B is the rank-1-collapse fix on the *physical* (BHOP, downstairs)
  bundle.
- **Verdict:** not actionable.

### Option W2: "Wrap + bundle-extend" (≥ 4 weeks)

Wrap cymyc + replace the section network and codifferential with
bundle-V variants implementing M1+M2.

Concretely, this requires writing fresh JAX code for:
- (W2-a) Local trivialisations of V on each ambient chart.
- (W2-b) HYM (0,1)-connection coefficients A^I_{ī J} from our
  `hym_hermitian.rs` output, marshalled into the cymyc data format.
- (W2-c) ∂̄_V s = ∂_z̄ s + A^I_{ī J} s^J replacing cymyc's tangent-bundle
  codifferential (`codifferential_eta`).
- (W2-d) Z₃ × Z₃ equivariant projector composed onto the harmonic basis
  (M4 chain-match prerequisite).
- (W2-e) Schoen fiber-product `poly_spec.py` (M0 in G1's plan
  effectively).
- (W2-f) JSON↔JAX pickle adapter for our Donaldson balanced metric.

This is ~80 % of G1's bespoke M1+M2 effort, *plus* ~5–7 days of JAX
plumbing, *plus* a JAX runtime dependency in CI. The remaining 20 %
saving (we get to reuse `HarmonicFull`'s spectral-NN ansatz architecture
and training loop) does not justify the wrapping overhead.

- ✅ Reuses cymyc's well-tested NN training scaffold and Hodge inner-
  product machinery.
- ❌ Estimate **rises** to 4–5 weeks once JAX adapter + Schoen encoder +
  validation against G1's M3 (Lanczos) cross-check are included.
- ❌ Adds a JAX runtime dependency on top of our Rust solver — the
  rust_solver/ pipeline currently has no Python eigensolver dependency,
  and adding JAX-on-GPU pulls cudatoolkit/jaxlib into the production
  reproducibility set.
- ❌ License is fine (MIT) but the architectural coupling to cymyc's
  internal class hierarchy is high; an upstream refactor breaks us.
- **Verdict:** the savings are illusory.

### Option B: Bespoke (G1's plan) (3–4 weeks)

Stay in `rust_solver/`, implement M1–M4 against AKLP/ACLP 2017
benchmarks as G1 specified.

- ✅ No new runtime dependency.
- ✅ Reuses our Donaldson + HYM Rust pipeline directly.
- ✅ Lanczos eigensolver (M3) works in Rust against existing
  ndarray/faer infrastructure.
- ✅ Validation regression (M4) is unchanged.
- ❌ ~3–4 weeks of focused work — the original G1 estimate.
- **Verdict:** the right path for our specific (Schoen, BHOP rank-4)
  setup.

---

## Step 4 — Recommendation

**Do bespoke Fix B as G1 drafted it, in `rust_solver/`.** Cite cymyc /
Butbaia 2024 / arXiv:2410.19728 as the algorithmic reference for the
spectral-NN ansatz used in M1, but do not introduce JAX into the
runtime.

**Effort estimate:** unchanged — **3–4 weeks** for M1–M4 bespoke. The
hostile-review hypothesis (1-week wrap) is **not viable** for our
specific Schoen-Z₃²+BHOP-rank-4 setup; it would only have been viable
for a standard-embedding quintic / TY redo, which is not the goal.

### Concrete next steps (kept minimal, hand-off to G1's bespoke track)

The spike does **not** authorise starting wrap-and-extend. The next
work item under Stage 3 Fix B is G1's M1 (basis construction in Rust).
The only spike-derived deliverable that should propagate forward:

- M1 reference: when G1 implements the spectral-NN basis or the
  ACLP/Blesneag polynomial-deformation basis, cite the cymyc objective
  function (sum of `(ξ,ξ)_WP`, `(∂̄θ, ∂̄θ)`, codiff_mean, polarisation)
  as the reference correctness criterion. Even though we are not
  using the cymyc *code*, the *test oracle* is established literature.

### Risk flags

- **License:** MIT clean; no risk if we reuse the algorithmic ideas
  with attribution, no risk if we vendor (we are not vendoring).
- **Dependency conflicts:** none (we are not pulling cymyc in).
- **API mismatch cost:** N/A (not using).
- **Reproducibility:** by staying in Rust, the rust_solver/
  reproducibility envelope is preserved; no Python-stack drift.
- **Citation hygiene:** when M1/M2 land, the references file
  (`rust_solver/references/`) MUST add an entry for arXiv:2410.19728
  (cymyc) and arXiv:2401.15078 (Butbaia et al.) crediting the
  spectral-NN harmonic-form approach.

---

## Spike artifacts

| File | Purpose |
|------|---------|
| `python_research/spike_dirac_upstream/__init__.py` | Package marker. |
| `python_research/spike_dirac_upstream/test_compatibility.py` | Non-installing dry-import probe. Confirms cymyc not in active interp; documents structural blockers regardless. Exit codes: 0 = installed + API matches, 1 = not installed (current state), 2 = installed but API drift. |
| `rust_solver/references/p_stage3_fix_b_upstream_spike.md` | This file. |

The compatibility probe was run; cymyc is not installed (as expected
and intended — we deliberately did not pollute any global env). Output
shows the structural-blockers checklist, which is the binding finding
regardless of installation state.

## Sources

- [Justin-Tan/cymyc on GitHub (MIT)](https://github.com/Justin-Tan/cymyc)
- [cymyc API docs — Harmonic module](https://justin-tan.github.io/cymyc/api/harmonic/)
- [Butbaia et al. 2024 — arXiv:2401.15078](https://arxiv.org/abs/2401.15078)
- [Tan et al. 2024 — cymyc paper, arXiv:2410.19728](https://arxiv.org/abs/2410.19728)
- [pythoncymetric/cymetric on GitHub (GPL-3.0)](https://github.com/pythoncymetric/cymetric)
- [ml4physics/cyjax on GitHub (MIT)](https://github.com/ml4physics/cyjax)
- [Blesneag–Buchbinder–Candelas–Lukas — arXiv:1512.05322](https://arxiv.org/abs/1512.05322)
- [Blesneag et al. — arXiv:1607.03461](https://arxiv.org/abs/1607.03461)
- [Anderson–Constantin–Lukas–Palti localisation — arXiv:1801.09645](https://arxiv.org/abs/1801.09645)
