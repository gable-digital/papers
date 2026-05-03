# arXiv:1004.4399 — direct text verification

## Paper identification
- Title: **"Numerical Hermitian Yang-Mills Connections and Vector Bundle Stability in Heterotic Theories"**
- Authors: **Lara B. Anderson, Volker Braun, Robert L. Karp, Burt A. Ovrut**
- arXiv ID: 1004.4399
- Journal: JHEP 06 (2010) 107
- DOI: 10.1007/JHEP06(2010)107
- Source URL used for retrieval: `https://arxiv.org/pdf/1004.4399` (PDF) and `https://ar5iv.labs.arxiv.org/html/1004.4399` (HTML cross-check)
- Retrieval date: 2026-04-29
- Retrieval method: WebFetch (downloaded PDF) → pypdf text extraction (52 pages, 104 075 chars), cross-checked against ar5iv HTML rendering
- Naming clarification: prior project notes called this paper "AKLP" (Anderson-Karp-Lukas-Palti). Authors are actually **Anderson-Braun-Karp-Ovrut (ABKO)**. The "AKLP" abbreviation is a misnomer that has propagated through our internal docs; the paper itself is ABKO. The follow-up Anderson-Karp-Lukas-Palti paper is arXiv:1103.3041 (a different paper).

## Fit constants — VERIFICATION

### 3.51 (leading coefficient, Fermat quintic)
- **Section/Page**: §2.4 "The Quintic Threefold", Figure 4, page 15 (PDF page 16)
- **Surrounding text** (verbatim from PDF extraction; line breaks reflect plot legend layout, "k2" = `k²`, "k3" = `k³`, "k4" = `k⁴`):
  ```
  k = 0 k = 1 k = 2 k = 3 k = 4 k = 5 k = 6 k = 7 k = 8 k = 9
  σk
  σk for ψ = 1/2 , Code1
  σk for ψ = 1/2 e2πi/5, Code1
  σk for ψ = 1/2 , Code2
  σk for ψ = 1/2 e2πi/5, Code2
  Fit for k ≥ 3: σk = 3.51/k2 − 5.19/k3
  σk = 3.51/k2 − 5.12/k3 + − 0.14/k4
  Figure 4: The σk error measures for the quintic threefold (2.24) in P4. Shown
  is the error measure described in Subsection 2.2, evaluated for the
  two values ψ = 1/2 and ψ = i/2 . Code1 and Code2 are associated to
  the implementations of [48, 49] and [50, 43] respectively.
  ```
- Equation reference: legend of **Figure 4** (not in numbered equation form). Two fits are reported on the same plot:
  - 2-term fit (k ≥ 3): `σ_k = 3.51/k² − 5.19/k³`
  - 3-term fit: `σ_k = 3.51/k² − 5.12/k³ − 0.14/k⁴`
- **VERDICT: confirmed** — exact bit-match for `3.51` as leading coefficient.

### −5.19 (sub-leading coefficient, Fermat quintic)
- **Section/Page**: §2.4, Figure 4, page 15 (PDF page 16) — same legend block as above.
- **Verbatim**: `Fit for k ≥ 3: σk = 3.51/k2 − 5.19/k3`
- **VERDICT: confirmed** — exact bit-match for `−5.19` as `1/k³` coefficient.

### σ_9 ≈ 0.043 (paper-stated single value at k=9)
- **VERDICT: not-found (as a literal numeric quote)**
- The string "0.043" does NOT appear anywhere in the PDF text (verified by exhaustive grep on the 104 075-char extraction; only matches for "0.04*" prefix in this document are absent). The σ axis of Figure 4 ranges 0–0.4, with k running 0–9. The k=9 point is **plotted** but not tabulated and not quoted in prose.
- The "≈ 0.043" value in our prior `aklp_2010.md` notes is a **plot-read estimate**, not a paper quote. Direct evaluation of the 2-term fit at k=9 gives `3.51/81 − 5.19/729 = 0.04333 − 0.00712 = 0.0362`, and the 3-term fit gives `0.0362 − 0.14/6561 = 0.0362 − 0.0000213 ≈ 0.0362`. So `σ_9 ≈ 0.036` from the fit; the "0.043" plot-read may correspond to the actual data point (not the fit), which is plausibly higher than the fit at low-k boundary.
- **Bottom line**: 0.043 is NOT a verbatim paper quote. Treat it as a plot-read approximation, not as a direct numeric quote.

## Convention used by the paper

### σ definition (L¹ MAD vs L² RMS vs other) — verbatim from page 11–12
```
The first is σk introduced in [48, 49] and given by
σk = (1/VolCY) ∫_X | 1 − (ωk^d/Volk) / (Ω∧Ω̄/VolCY) | dVolCY    (2.16)
```
- This is **L¹** (integrated absolute value of the volume-form deviation), NOT L² and NOT RMS.
- The integrand is `|1 − ratio|`, where `ratio = (ω_k^d / Vol_k) / (Ω∧Ω̄ / Vol_CY)`. Vanishes iff ω_k is the Calabi-Yau Kähler form.

### Predicted convergence law (eq. 2.17, page 12)
```
σk = a2/k2 + a3/k3 + . . .    (2.17)
```
The fit constants 3.51 and −5.19 are the empirical values of `a2` and `a3` for the Fermat quintic.

### Sample distribution / optimization method
- Algorithm: **Donaldson's T-operator iteration** (eq. 2.10, called "Donaldson's algorithm" throughout). No post-refinement step; balanced metric is the endpoint.
- Sample distribution: not described explicitly in §2.4; the paper cites two prior implementations [48, 49] (Code1) and [50, 43] (Code2). Both are Fubini-Study-induced sampling on the embedded projective variety, per the cited papers.
- Two parallel implementations (Code1 and Code2) were run and shown to agree on Figure 4 — the σk values from both codes overlay closely at every k.

### n_sample_points stated — verbatim from §2.4, page 14 (PDF page 15)
```
The T-map was iterated with 2,000,000 points, and the error measures were
computed with 500,000 points.
```
- Quintic: 2 000 000 T-iteration points, **500 000 σ-evaluation points**.
- (For comparison, K3 in §2.3, page 14: 1 600 000 T-iteration points, 500 000 σ-evaluation points.)
- T-operator iteration: ≤ 10 iterations (referenced in [48, 49] and consistent with this paper's plots; explicit "≤ 10" is from prior literature, not literally quoted here).

## Quartic K3 fit (cross-check)

The quartic K3 fit is reported in **Figure 1** (NOT Figure 4 as our prior `aklp_2010.md` claimed). Verbatim:
```
Fit for k ≥ 3: σk = 0.90/k2 − 0.25/k3
σk = 1.05/k2 − 1.52/k3 + 2.52/k4
σk = 0.31 exp(− 0.42k)
Figure 1: The error measures σk defined in Subsection 2.2. The data shown
is for the Quartic K3 defined as a hypersurface in P3 (2.22). The
complex structure parameter is chosen to be ψ = 1/2 and ψ = i/2.
Shown is data generated using the code developed in [48, 49] (Code1)
and data generated by the implementation in [50, 43] (Code2). The
error measure is fitted to the theoretical convergence given in (2.17).
```
- Quartic K3 2-term fit confirmed: `σ_k = 0.90/k² − 0.25/k³`. Bit-exact match.
- Note also the alternative 3-term fit `1.05/k² − 1.52/k³ + 2.52/k⁴` and exponential fit `0.31·exp(−0.42k)` are reported alongside; paper says (page 14) "the assumption that σk = O(1/k) fits our data better than exponential fall-off".

## Figure-number corrections to existing project notes

Our prior `references/aklp_2010.md` says:
- Quartic K3 fit → "Fig. 4 / Code1" — **WRONG**, actual location is **Figure 1**.
- Fermat quintic fit → "Fig. 6 (Code1)" — **WRONG**, actual location is **Figure 4**.
- "σ_9 ≈ 0.043 (Fermat quintic) — quoted point" — **WRONG**, this is not a verbatim quote; the paper does not state 0.043 in prose. The k=9 data point is only readable from Figure 4's scatter plot. Best estimate from the 2-term fit is `σ_9 ≈ 0.036`.
- "σ_{10} ≈ 0.009 (quartic K3) — quoted point" — also NOT a verbatim quote. Figure 1 plots up to k=10 but no tabulated value.

## Summary — VERDICT

| Constant       | Paper says                                  | Verdict                  |
|---------------:|:--------------------------------------------|:-------------------------|
| 3.51 (a₂, quintic)   | `σk = 3.51/k2 − 5.19/k3` (Fig. 4)     | **confirmed bit-exact**  |
| −5.19 (a₃, quintic)  | `σk = 3.51/k2 − 5.19/k3` (Fig. 4)     | **confirmed bit-exact**  |
| 0.90 (a₂, K3)        | `σk = 0.90/k2 − 0.25/k3` (Fig. 1)     | **confirmed bit-exact**  |
| −0.25 (a₃, K3)       | `σk = 0.90/k2 − 0.25/k3` (Fig. 1)     | **confirmed bit-exact**  |
| σ_9 ≈ 0.043          | not stated in prose                         | **not-found** (plot-read only; fit gives ~0.036) |
| σ_10 ≈ 0.009         | not stated in prose                         | **not-found** (plot-read only)                    |

The headline scientific claim (CY3 σ-functional matching ABKO 2010 fit `σ_k ≈ 3.51/k² − 5.19/k³` for the Fermat quintic, k ≥ 3) is **scientifically valid**: both constants are paper-quoted bit-exact from Figure 4. The 13 % rel-err agreement reported in P3.10 stands.

Caveats for downstream use:
1. The `σ_9 ≈ 0.043` and `σ_{10} ≈ 0.009` values used in `aklp_2010.md` are NOT direct paper quotes — they are estimates read off Figure 4 / Figure 1. Anything that hangs on bit-exact agreement with these per-k values should be re-evaluated against the **fit** (not a quoted point).
2. Figure numbering in our prior notes is wrong (we said Fig. 6 for quintic, actual is Fig. 4; said Fig. 4 for K3, actual is Fig. 1).
3. The 2-term fit `3.51/k² − 5.19/k³` is the paper's primary k ≥ 3 form; a 3-term refinement `3.51/k² − 5.12/k³ − 0.14/k⁴` is also given on the same Figure 4 — note the leading 3.51 is unchanged, but the `1/k³` coefficient is slightly different (−5.12 vs −5.19) when the 4th-order term is included. If our solver computes only the 2-term reduction, target `−5.19`; if it uses the 3-term form, target `−5.12`.
