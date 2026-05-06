# CODATA-2022 / PDG-2024 Reference-Value Snapshot Manifest

**Purpose.** Pin every empirical reference value used by the substrate-framework
discrimination pipeline (`book/paper/substrate_particle_equations.tex`,
`book/scripts/cy3_substrate_discrimination/rust_solver/src/pdg.rs`,
`book/scripts/cy3_substrate_discrimination/empirical_constants.py`,
`book/scripts/cy3_substrate_discrimination/python_research/rational_search_baseline.py`)
to a specific, citation-pinned snapshot. This is the audit-defense answer to the hostile
referee gambit "you used an old PDG value": every comparison the framework makes is
against the values listed here, with explicit citation.

**Snapshot date.** 2026-05-04.

**Primary sources.**

- **PDG 2024**: R.L. Workman et al. (Particle Data Group), *Review of Particle Physics*,
  Prog. Theor. Exp. Phys. 2024, 083C01 (2024) and 2025 update through pdg.lbl.gov,
  https://pdg.lbl.gov/2024/.
- **CODATA 2022**: E. Tiesinga, P.J. Mohr, D.B. Newell, B.N. Taylor, *CODATA Recommended
  Values of the Fundamental Physical Constants: 2022*,
  Rev. Mod. Phys. 97 (2025) 025002, https://pml.nist.gov/cuu/Constants/.
- **Planck 2018**: Planck Collaboration, *Planck 2018 results. VI. Cosmological
  parameters*, A&A 641 (2020) A6, arXiv:1807.06209.
- **NuFIT 5.3** (or 5.2 where indicated): I. Esteban, M.C. Gonzalez-Garcia,
  M. Maltoni, T. Schwetz, A. Zhou, *The fate of hints: updated global analysis of
  three-flavor neutrino oscillations*, JHEP 09 (2020) 178, http://www.nu-fit.org/.
- **KATRIN 2024**: KATRIN Collaboration, *Direct neutrino-mass measurement based on
  259 days of KATRIN data*, Science 388 (2025) 180.
- **Buttazzo et al. 2013**: D. Buttazzo et al., *Investigating the near-criticality
  of the Higgs boson*, JHEP 12 (2013) 089, arXiv:1307.3536 (table 3, running VEV
  v(M_Z)).
- **CKMfitter 2024 update**: J. Charles et al., http://ckmfitter.in2p3.fr/.

---

## 1. Charged-lepton pole masses

| Quantity | Value | Uncertainty (1σ) | Scheme | Source | Citation | Date | Used by |
|---|---|---|---|---|---|---|---|
| m_e | 0.51099895069 MeV | 1.6 × 10⁻¹³ MeV | pole / on-shell | CODATA 2022 | Rev. Mod. Phys. 97 (2025) 025002 | 2024 | `pdg.rs:303`; paper `Derivation 6` |
| m_μ | 105.6583755 MeV | 2.3 × 10⁻⁹ MeV | pole / on-shell | CODATA 2022 | Rev. Mod. Phys. 97 (2025) 025002 | 2024 | `pdg.rs:304`; paper `Derivation 6 / muon` |
| m_τ | 1776.86 MeV | 0.12 MeV | pole / on-shell | PDG 2024 listings | PTEP 2024 083C01 § Lepton listings | 2024 | `pdg.rs:305`; paper Table near 2508 |

**Notes.** The framework uses CODATA m_e and m_μ as fundamental anchors (paper §1019,
§1061, §1914 cite "CODATA m_μ = 105.6583755(23) MeV" and CODATA m_e to 17 ppm via the
m_e/m_Planck identity). m_τ is taken from PDG listings.

---

## 2. Quark masses

| Quantity | Value | Uncertainty (1σ) | Scheme | Source | Citation | Date | Used by |
|---|---|---|---|---|---|---|---|
| m_u(2 GeV) | 2.16 MeV | +0.49 / −0.26 MeV | MS-bar at μ = 2 GeV | PDG 2024 (Quark Mass review ch. 60) | PTEP 2024 083C01 ch. 60 | 2024 | `pdg.rs:308` |
| m_d(2 GeV) | 4.70 MeV | +0.48 / −0.17 MeV | MS-bar at μ = 2 GeV | PDG 2024 | PTEP 2024 083C01 ch. 60 | 2024 | `pdg.rs:309` |
| m_s(2 GeV) | 93.5 MeV | +8.6 / −3.4 MeV | MS-bar at μ = 2 GeV | PDG 2024 | PTEP 2024 083C01 ch. 60 | 2024 | `pdg.rs:310` |
| m_c(m_c) | 1.2730 GeV | 0.0046 GeV | MS-bar at μ = m_c | PDG 2024 | PTEP 2024 083C01 ch. 60 | 2024 | `pdg.rs:313` |
| m_b(m_b) | 4.183 GeV | 0.007 GeV | MS-bar at μ = m_b | PDG 2024 | PTEP 2024 083C01 ch. 60 | 2024 | `pdg.rs:314` |
| m_t (pole) | 172.57 GeV | 0.29 GeV | pole / on-shell | PDG 2024 | PTEP 2024 083C01 ch. Top quark | 2024 | `pdg.rs:317`; paper line 2503 |
| m_t (MS-bar) | 162.5 GeV | 0.7 GeV | MS-bar at μ = m_t | PDG 2024 | PTEP 2024 083C01 ch. Top quark | 2024 | `pdg.rs:318` |

**Pole↔MS-bar conversion** is performed using the 4-loop QCD relation of
Marquard-Smirnov-Steinhauser-Steinhauser, arXiv:1502.01030 (default in
`PoleMsbarOrder::FourLoop`). 1-loop Tarrach (NPB 183, 384, 1981) is selectable.

---

## 3. Gauge bosons and Higgs

| Quantity | Value | Uncertainty (1σ) | Scheme | Source | Citation | Date | Used by |
|---|---|---|---|---|---|---|---|
| m_W | **80.3692 GeV** *(paper)* / **80.377 GeV** *(empirical_constants.py)* | 0.012 GeV | pole / on-shell | PDG 2024 EW review ch. 10 | PTEP 2024 083C01 ch. 10 | 2024 | paper line 903, 927; `empirical_constants.py:81-82` |
| m_Z | **91.1876 GeV** *(paper)* / **91.1880 GeV** *(pdg.rs)* | 0.0020 GeV (pdg.rs); 0.0021 GeV (paper-cited PDG) | pole / on-shell | PDG 2024 EW review ch. 10 | PTEP 2024 083C01 ch. 10 | 2024 | `pdg.rs:294`; paper line 903, 927 |
| m_H (Higgs) | 125.20 GeV | ≈ 0.11 GeV (PDG 2024 combined) | pole / on-shell | PDG 2024 Higgs Boson Physics review | PTEP 2024 083C01 § Higgs | 2024 | paper line 2518 |
| v (Higgs VEV, tree) | 246.21965 GeV | derived from G_F (~250 ppb) | tree-level, μ → 0 | PDG 2024 G_F → v = (√2 G_F)^(−1/2) | PTEP 2024 083C01 § Electroweak | 2024 | `pdg.rs:295`; paper §`Step 2.5` line 809-811 |
| v (Higgs VEV, M_Z) | 248.401 GeV | 0.032 GeV | running, MS-bar at μ = M_Z | Buttazzo et al. 2013 | JHEP 12 (2013) 089 (arXiv:1307.3536) table 3 | 2013 | `pdg.rs:296` |

**Two paper/code inconsistencies flagged in this row** — see §10 below. Both are within
the PDG 2024 listing's central-value uncertainty but the framework should pick one and
pin it.

---

## 4. Couplings and electroweak parameters

| Quantity | Value | Uncertainty (1σ) | Scheme | Source | Citation | Date | Used by |
|---|---|---|---|---|---|---|---|
| α_em(0) (fine-structure) | 7.2973525693 × 10⁻³ ≡ 1/137.035999084 | 1.1 × 10⁻¹² | Q² = 0 (Thomson limit) | CODATA 2018 | Rev. Mod. Phys. 93 (2021) 025010 | 2019 | `empirical_constants.py:99-104` (LEGACY: paper does not use this; should be migrated to CODATA 2022 for consistency) |
| α_em(M_Z)⁻¹ | 127.951 | 0.009 | MS-bar at μ = M_Z | PDG 2024 EW review | PTEP 2024 083C01 ch. 10 | 2024 | implicit in α_s/α_em RG running (`pdg.rs` Yukawa runner) |
| α_s(M_Z) | 0.1180 *(pdg.rs, paper)* / **0.1179** *(empirical_constants.py, rational_search_baseline.py)* | 0.0009 (pdg.rs) / 0.0010 (rational_search) | MS-bar at μ = M_Z | PDG 2024 world average | PTEP 2024 083C01 § QCD ch. 9 | 2024 | `pdg.rs:297`; `empirical_constants.py:107-113`; `rational_search_baseline.py:226` |
| sin²θ_W (on-shell) | 0.22321 (= 1 − m_W²/m_Z² with PDG m_W = 80.3692, m_Z = 91.1876) | 0.00012 | on-shell, Sirlin scheme | PDG 2024 EW review | PTEP 2024 083C01 ch. 10 | 2024 | paper line 903, 927; `rational_search_baseline.py:244` |
| sin²θ_W (MS-bar) | 0.23122 | 0.00004 | MS-bar at μ = M_Z | PDG 2024 EW review | PTEP 2024 083C01 ch. 10 | 2024 | paper line 924, 936; `references/p_sin2thetaW_alphaS_provenance.md` |
| α_W(M_Z) (= g₂²/4π) | 0.0338 | 0.0001 | derived from EW fit | PDG 2024 EW fit | PTEP 2024 083C01 ch. 10 | 2024 | `empirical_constants.py:116-122` |
| g₁(M_Z) (GUT-normalized) | 0.461228 | — | MS-bar GUT, g₁² = (5/3) g'² | derived from PDG α_em(M_Z), sin²θ_W | PTEP 2024 083C01 ch. 10 | 2024 | `pdg.rs:298` |
| g₂(M_Z) | 0.65096 | — | MS-bar at μ = M_Z | derived from PDG | PTEP 2024 083C01 ch. 10 | 2024 | `pdg.rs:299` |
| g₃(M_Z) | 1.2123 | — | MS-bar at μ = M_Z | derived from α_s(M_Z) = 0.1180 | PTEP 2024 083C01 ch. 9 | 2024 | `pdg.rs:300` |
| G_F (Fermi) | 1.1663787 × 10⁻⁵ GeV⁻² | 6 × 10⁻¹² GeV⁻² | μ-decay extraction | PDG 2024 / CODATA 2022 | PTEP 2024 083C01 ch. 10 | 2024 | implicit in v_tree = (√2 G_F)^(−1/2); paper line 809-811 |

---

## 5. CKM matrix and CP-violating phase

| Quantity | Value | Uncertainty (1σ) | Scheme | Source | Citation | Date | Used by |
|---|---|---|---|---|---|---|---|
| \|V_us\| | 0.22501 | 0.00068 | PDG 2024 average | PDG 2024 CKM review ch. 12 | PTEP 2024 083C01 ch. 12 | 2024 | `pdg.rs:321` |
| \|V_cb\| | 0.04182 | +0.00085 / −0.00074 | PDG 2024 (incl. tension) | PDG 2024 CKM review ch. 12 | PTEP 2024 083C01 ch. 12 | 2024 | `pdg.rs:322` |
| \|V_ub\| | 0.00369 | 0.00011 | PDG 2024 average | PDG 2024 CKM review ch. 12 | PTEP 2024 083C01 ch. 12 | 2024 | `pdg.rs:323` |
| Wolfenstein λ | 0.22501 | (≡ \|V_us\|) | PDG 2024 | PTEP 2024 083C01 ch. 12 | 2024 | `pdg.rs:325` |
| Wolfenstein A | 0.826 | — | PDG 2024 CKM fit | PTEP 2024 083C01 ch. 12 | 2024 | `pdg.rs:326` |
| Wolfenstein ρ̄ | 0.159 | — | PDG 2024 CKM fit | PTEP 2024 083C01 ch. 12 | 2024 | `pdg.rs:327` |
| Wolfenstein η̄ | 0.348 | — | PDG 2024 CKM fit | PTEP 2024 083C01 ch. 12 | 2024 | `pdg.rs:328` |
| Jarlskog J | 3.08 × 10⁻⁵ | +0.15 / −0.13 × 10⁻⁵ | PDG 2024 | PTEP 2024 083C01 ch. 12 | 2024 | `pdg.rs:330` |
| δ_CKM (= δ_13, std. parametrization) | 1.144 rad | 0.026 rad | PDG 2024 | PTEP 2024 083C01 ch. 12 | 2024 | paper line 2295-2296 ("PDG 2024 quotes δ_CKM ≈ 1.144(26) rad") |
| γ (= φ_3, UT angle) | 65.9° (1.151 rad) | +3.3 / −3.5 deg | PDG 2024 (LHCb-led world average) | PTEP 2024 083C01 ch. 12 | 2024 | implicit (γ ≈ δ_CKM in std. parametrization) |
| δ_CKM (alt., rational_search_baseline.py value) | 1.196 rad | 0.044 rad | older-PDG / NuFit-derived | rational_search_baseline.py:264-266 | local code | — | INCONSISTENCY — see §10 |

**CKM correlations.** Off-diagonal CKM-sub-block correlations are taken from the
CKMfitter 2024 update (J. Charles et al., http://ckmfitter.in2p3.fr/), wired into
`pdg.rs::ckmfitter_2024_covariance()` for the correlated chi-squared variant.

---

## 6. PMNS (neutrino mixing)

| Quantity | Value | Uncertainty (1σ) | Scheme | Source | Citation | Date | Used by |
|---|---|---|---|---|---|---|---|
| sin²θ₁₂ | 0.307 | +0.013 / −0.012 | NuFIT 5.3 NO best fit | I. Esteban et al., http://nu-fit.org/ | NuFIT 5.3 (2024) | 2024 | implicit in paper §"PMNS" |
| sin²θ₁₃ | 0.02220 | +0.00068 / −0.00062 | NuFIT 5.3 NO | I. Esteban et al., http://nu-fit.org/ | NuFIT 5.3 (2024) | 2024 | implicit |
| sin²θ₂₃ | 0.572 | +0.018 / −0.023 | NuFIT 5.3 NO | I. Esteban et al., http://nu-fit.org/ | NuFIT 5.3 (2024) | 2024 | implicit |
| δ_CP^PMNS | ≈ 1.08 π (≈ 3.4 rad, NO best fit) | ±0.5 rad (1σ) | NuFIT 5.2 NO | I. Esteban et al. | NuFIT 5.2 (2022) | 2022 | `rational_search_baseline.py:278-279`; paper §"falsification" line 1162 ("global fits NuFIT 5.3, T2K/NOvA combined place δ_CP^PMNS within ~1.5σ of 11π/10 = 198°") |
| Δm²_21 | 7.42 × 10⁻⁵ eV² | 0.21 × 10⁻⁵ eV² | NuFIT 5.3 / PDG 2024 | PTEP 2024 083C01 § Neutrino mixing | 2024 | paper line 1592 |
| Δm²_31 (NO) | 2.514 × 10⁻³ eV² | 0.028 × 10⁻³ eV² | NuFIT 5.3 NO | NuFIT 5.3 | 2024 | paper line 1592 (quotes Δm²_32 ≈ 2.5 × 10⁻³ eV²) |
| Δm²_32 (NO) | 2.5 × 10⁻³ eV² (paper-quoted leading digit) | 0.03 × 10⁻³ eV² | NuFIT 5.3 / PDG 2024 | PTEP 2024 083C01 | 2024 | paper line 1592 |
| Σm_ν (cosmological) | < 0.072 eV (DESI Y1+CMB) – < 0.12 eV (Planck-only) | 95% CL upper bound | Planck 2018 + DESI Y1 BAO | A&A 641 (2020) A6; DESI Y1 cosmology paper 2024 | 2024 | paper line 1595 |
| m_νe (kinematic) | < 0.45 eV | 90% CL upper | KATRIN 2024 (Science 388, 180) | KATRIN, Science 388 (2025) 180 | 2024 | paper line 1595 |

---

## 7. Gravitational and atomic constants

| Quantity | Value | Uncertainty (1σ) | Scheme | Source | Citation | Date | Used by |
|---|---|---|---|---|---|---|---|
| G (Newton's constant) | 6.67430 × 10⁻¹¹ m³ kg⁻¹ s⁻² | 1.5 × 10⁻¹⁵ m³ kg⁻¹ s⁻² (22 ppm) | CODATA 2022 | CODATA 2022 | Rev. Mod. Phys. 97 (2025) 025002 | 2024 | paper line 1133 ("Newton's G at 2.5 ppm"), §`Derivation 6`; `pdg.rs` not used |
| m_Planck (= √(ℏc/G)) | 1.220890 × 10¹⁹ GeV/c² | 1.4 × 10¹³ GeV (CODATA G uncertainty propagated) | CODATA 2022 derived | Rev. Mod. Phys. 97 (2025) 025002 | 2024 | paper line 2480, 2549 (`m_Planck = 1.22089 × 10¹⁹ GeV`) |
| ℏ (reduced Planck) | 1.054571817 × 10⁻³⁴ J·s | exact (2019 SI) | SI / CODATA 2022 | Rev. Mod. Phys. 97 (2025) 025002 | 2024 | implicit in m_e/m_Planck identity |
| c (speed of light) | 299 792 458 m/s | exact (2019 SI) | SI definition | — | — | implicit |
| α_em⁻¹ (Q²=0) | 137.035999177 | 0.000000021 | CODATA 2022 | Rev. Mod. Phys. 97 (2025) 025002 | 2024 | implicit (CODATA 2022 supersedes the CODATA 2018 value in `empirical_constants.py:99-104`) |

---

## 8. Cosmology

| Quantity | Value | Uncertainty (1σ) | Scheme | Source | Citation | Date | Used by |
|---|---|---|---|---|---|---|---|
| Ω_DM h² | 0.1200 | 0.0010 | Planck 2018 TT+TE+EE+lowE+lensing | A&A 641 (2020) A6 (table 2) | 2018 (publ. 2020) | 2018 | paper line 2245-2246, eq. 2234 |
| Ω_b h² | 0.02237 | 0.00015 | Planck 2018 | A&A 641 (2020) A6 | 2020 | 2018 | implicit (η = n_B/n_γ derivation) |
| H₀ (CMB-inferred, ΛCDM) | 67.4 km/s/Mpc | 0.5 km/s/Mpc | Planck 2018 base ΛCDM | A&A 641 (2020) A6 | 2020 | 2018 | paper line 1323 ("Planck through ΛCDM H_0 ≈ 67") |
| H₀ (local distance ladder) | 73.0 km/s/Mpc | 1.0 km/s/Mpc | SH0ES Cepheid+SN Ia | Riess et al. 2022, ApJL 934, L7 (arXiv:2112.04510) | 2022 | 2022 | paper line 1322 ("Local distance-ladder H_0 ≈ 73") |
| n_s (CMB scalar tilt) | 0.9665 | 0.0038 | Planck 2018 base ΛCDM | A&A 641 (2020) A6 | 2020 | 2018 | paper line 1289 (framework predicts 58/60 = 0.9667) |
| r (tensor-to-scalar) | < 0.036 | 95% CL upper | BICEP/Keck 2021 + Planck | BICEP/Keck Collaboration, PRL 127, 151301 (arXiv:2110.00483) | 2021 | 2021 | paper line 1319 |
| η = n_B/n_γ (baryon-to-photon) | 6.13 × 10⁻¹⁰ | 0.06 × 10⁻¹⁰ | BBN+CMB joint | Particle Data Group 2024 review (Fields-Olive § BBN); Planck 2018 + BBN | PTEP 2024 083C01 § BBN | 2024 | `empirical_constants.py:35-42`; paper line 1280 (framework predicts 6.115(38) × 10⁻¹⁰) |
| N_eff | 2.99 | 0.17 | Planck 2018 + BBN | PTEP 2024 083C01 § BBN; A&A 641 (2020) A6 | 2024 | 2018-2024 | paper line 1562 |
| D/H (primordial) | 2.527 × 10⁻⁵ | 0.030 × 10⁻⁵ | BBN measurement | PTEP 2024 083C01 § BBN | 2024 | paper line 1565 |
| Y_p (helium-4) | 0.245 | 0.003 | BBN measurement | PTEP 2024 083C01 § BBN | 2024 | paper line 1565 |

---

## 9. Other constants and bounds

| Quantity | Value | Uncertainty / bound | Scheme | Source | Citation | Date | Used by |
|---|---|---|---|---|---|---|---|
| θ_QCD | < 10⁻¹⁰ | 90% CL upper (nEDM-derived) | on-shell | nEDM Collaboration, PRL 124, 081803 (arXiv:2001.11966) | 2020 | 2020 | paper Eq. (2285) "θ_QCD = 0", falsifier line 1633 |
| a_e (electron g−2 anomaly) | 1.15965218073 × 10⁻³ | 2.8 × 10⁻¹³ | Penning trap | Fan, Myers, Sukra, Gabrielse, PRL 130, 071801 (2023) | 2023 | 2023 | paper line 1539 (framework predicts 1.16067 × 10⁻³ at +875 ppm) |
| a_μ (muon g−2 anomaly) | 1.16592059 × 10⁻³ | 2.2 × 10⁻¹⁰ | Fermilab E989 + BNL E821 world average 2023 | Fermilab Muon g−2 Collaboration, PRL 131, 161802 (arXiv:2308.06230) | 2023 | 2023 | paper line 1545 (framework predicts 1.16694 × 10⁻³ at +878 ppm) |
| τ_e (electron lifetime) | > 6.6 × 10²⁸ yr | 90% CL lower | Borexino 2015, PRL 115, 231802 | (PDG 2024 listings) | 2024 | paper line 1509 |
| τ_p (proton lifetime, p → e⁺π⁰) | > 1.6 × 10³⁴ yr | 90% CL lower | Super-K 2020 | (PDG 2024 listings) | 2024 | paper line 1511 |
| Lamb shift (H 2S₁/₂ − 2P₁/₂) | 1057.844 MHz | 0.014 MHz | atomic spectroscopy | PDG 2024 / Hagley-Pipkin 1994 + later updates | 2024 | paper line 1492 |
| ρ_Λ (cosmological-constant density) | ~6 × 10⁻³⁰ g/cm³ | order-of-magnitude | Planck 2018 ΛCDM | A&A 641 (2020) A6 | 2020 | paper line 1496 |
| LZ 2024 dark-matter cross-section bound | < 4 × 10⁻⁴⁸ cm² at m_DM ≈ 30 GeV | 90% CL upper | LZ 2024 | LZ Collaboration, PRL 131, 041002 (arXiv:2207.03764, updated 2024) | 2024 | paper line 1586 |
| N (light neutral neutrino families) | 2.984 | 0.008 | LEP Z-pole | PDG 2024 / ALEPH-DELPHI-L3-OPAL combined | 2024 | paper line 1581 |

---

## 10. INCONSISTENCIES BETWEEN PAPER AND CODE — flagged for review (do NOT auto-fix)

**These are the rough edges a hostile referee will probe. Each needs to be resolved by
picking one canonical value and propagating through paper + code.**

### 10.1 m_W: paper uses 80.3692 GeV; `empirical_constants.py` uses 80.377 GeV

- **Paper** (`substrate_particle_equations.tex` lines 903, 927):
  `m_W = 80.3692 GeV`. This is the PDG 2024 listing's central value.
- **`empirical_constants.py:81-82`**: `M_W_GEV = 80.377; M_W_UNCERT_GEV = 0.012`.
  This appears to be the older PDG 2022 / CDF-pre-correction value. PDG 2024 quotes
  m_W = 80.3692 ± 0.0133 GeV (post-CDF-tension averaged).
- **`pdg.rs`**: `m_W` is **not stored** as a field of `Pdg2024`. Implicit only via
  `sin²θ_W = 1 − m_W²/m_Z²`, which the framework reads from PDG separately.
- **Decision required:** the paper is the canonical statement; pin both code paths to
  PDG 2024 `m_W = 80.3692 ± 0.0133 GeV` and update `empirical_constants.py` accordingly.

### 10.2 m_Z: paper uses 91.1876 GeV; `pdg.rs` uses 91.1880 GeV

- **Paper** (lines 903, 927): `m_Z = 91.1876 GeV` — the LEP 2006 final combined value
  (Phys. Rep. 427, 257) historically listed by PDG through 2022.
- **`pdg.rs:294`**: `Measurement::new(91.1880, 0.0020)` — the slightly revised PDG 2024
  central value reflecting the LHC m_W/m_Z constraint update.
- The shift is 4 × 10⁻⁴ GeV (≈ 4.4 × 10⁻⁶ relative), well inside the 2.0 MeV PDG
  uncertainty, but it does change `1 − m_W²/m_Z²` at the ~10 ppm level.
- **Decision required:** pin to PDG 2024 `m_Z = 91.1880 ± 0.0020 GeV`. Update paper §3
  and §4 references from 91.1876 → 91.1880.

### 10.3 α_s(M_Z): paper / `pdg.rs` use 0.1180; `empirical_constants.py` and `rational_search_baseline.py` use 0.1179

- **`pdg.rs:297`**: `Measurement::new(0.1180, 0.0009)` — PDG 2024 world average.
- **Paper** (eq. 2283, line 2316): `α_s(M_Z) = 849/7192 ≈ 0.11804783`, compared to
  PDG `0.1180(9)`.
- **`empirical_constants.py:107-113`**: `central_value=0.1179, uncertainty=0.0010`.
- **`rational_search_baseline.py:226-227`**: `observed=0.1179, observed_unc=0.0010`.
- The PDG 2024 world average is `α_s(M_Z) = 0.1180 ± 0.0009`. The 0.1179 value comes
  from the PDG 2022 world average. The shift of 0.0001 is within the uncertainty but
  affects the ppm-residual claim of `849/7192` matches.
- **Decision required:** pin all four locations to PDG 2024 `α_s(M_Z) = 0.1180 ± 0.0009`.
  Update `empirical_constants.py` and `rational_search_baseline.py`. Update the paper's
  ppm-residual figure if material.

### 10.4 α_em: `empirical_constants.py` cites CODATA 2018; should be CODATA 2022

- **`empirical_constants.py:99-104`**: `central_value=7.2973525693e-3, uncertainty=1.1e-12,
  source="CODATA 2018"`.
- **CODATA 2022** value: α = 7.2973525643(11) × 10⁻³, equivalently α⁻¹ = 137.035999177(21).
- The shift is at the 10⁻¹⁰ relative level — entirely irrelevant to the framework's
  ppm-level claims, but the audit-defense answer requires consistency with the rest of
  the manifest's CODATA 2022 pinning.
- **Decision required:** update `empirical_constants.py` source string from "CODATA 2018"
  to "CODATA 2022" and update the central value to 7.2973525643 × 10⁻³.

### 10.5 δ_CKM: paper uses 1.144(26) rad; `rational_search_baseline.py` uses 1.196(44) rad

- **Paper** (line 2295-2296): "PDG 2024 quotes δ_CKM ≈ 1.144(26) rad. The framework's
  prediction 11π/30 ≈ 1.152 rad is within 1σ of PDG."
- **`rational_search_baseline.py:264-266`**:
  ```
  observed=1.196 / math.pi,           # 1.196 rad
  observed_unc=0.044 / math.pi,       # 0.044 rad
  ```
  This uses an older PDG / pre-2024 LHCb-combined value.
- **PDG 2024** quotes the UT angle γ = 65.9° = 1.151 rad with +3.3 / −3.5 deg
  uncertainty, equivalent to δ_CKM ≈ 1.151 rad. The paper's "1.144(26)" appears to
  be the CKMfitter or UTfit 2024 indirect-fit central value, distinct from the
  direct-measurement γ.
- **Decision required:** pin `rational_search_baseline.py` to either PDG 2024 direct
  γ = 1.151 ± 0.058 rad (= 65.9° ± 3.4°) or CKMfitter 2024 indirect δ_CKM = 1.144 ±
  0.026 rad. Document the choice with explicit citation.

### 10.6 v_higgs (Higgs VEV): paper uses 246.21965 GeV; `pdg.rs` uses 246.21965; `rational_search_baseline.py:217` uses 246.21965 — CONSISTENT

No inconsistency. Pinned to PDG 2024 G_F → v_tree.

### 10.7 m_t pole and m_t MS-bar: CONSISTENT across paper and `pdg.rs`

`m_t (pole) = 172.57 ± 0.29 GeV` and `m_t (MS-bar) = 162.5 ± 0.7 GeV` match between
paper Table 2503 and `pdg.rs:317-318`.

### 10.8 m_e, m_μ, m_τ: CONSISTENT

CODATA 2022 m_e and m_μ; PDG 2024 m_τ. All match between paper and `pdg.rs`.

### 10.9 Quark MS-bar masses: CONSISTENT

PDG 2024 light quarks at 2 GeV and m_c(m_c), m_b(m_b) match between paper and
`pdg.rs`.

### 10.10 CKM magnitudes (\|V_us\|, \|V_cb\|, \|V_ub\|): CONSISTENT in `pdg.rs`; paper does not separately re-cite these numerically

`pdg.rs:321-323` uses PDG 2024 values. Paper relies on `pdg.rs` for chi-squared
comparison.

### 10.11 sin²θ_W on-shell vs. MS-bar choice: documented and intentional

This is **not** an inconsistency — it is a deliberate scheme choice already documented
in `references/p_sin2thetaW_alphaS_provenance.md`. The framework's `2/9` rational
matches **on-shell** `1 − m_W²/m_Z² = 0.22321`, NOT MS-bar `0.23122`. This is a
semantic commitment that should remain intact; no fix needed.

### 10.12 Ω_DM h² leading prefactor K = 0.12: documented as a fitted Planck-2018 anchor

This is **not** an inconsistency — paper line 2245-2247 explicitly admits K = 0.12 is
"anchored to Planck 2018 (Ω_DM h² = 0.1200 ± 0.0010); it is therefore a fitted
cosmological prefactor, not a structural derivation". No fix needed.

---

## 11. Summary: cataloged reference values

- **Charged leptons**: 3 (m_e, m_μ, m_τ)
- **Quarks**: 7 (m_u, m_d, m_s at 2 GeV; m_c(m_c), m_b(m_b); m_t pole and MS-bar)
- **Gauge + Higgs**: 5 (m_W, m_Z, m_H, v_tree, v(M_Z))
- **Couplings + EW parameters**: 9 (α_em(0), α_em(M_Z), α_s(M_Z), sin²θ_W on-shell,
  sin²θ_W MS-bar, α_W(M_Z), g₁, g₂, g₃; G_F)
- **CKM**: 9 (\|V_us\|, \|V_cb\|, \|V_ub\|; λ, A, ρ̄, η̄; J; δ_CKM/γ)
- **PMNS / neutrinos**: 8 (sin²θ₁₂, sin²θ₁₃, sin²θ₂₃, δ_CP^PMNS, Δm²_21, Δm²_31,
  Σm_ν, m_νe)
- **Gravitational + atomic**: 5 (G, m_Planck, ℏ, c, α_em⁻¹ CODATA 2022)
- **Cosmology**: 11 (Ω_DM h², Ω_b h², H₀ CMB, H₀ local, n_s, r, η_baryon, N_eff,
  D/H, Y_p, ρ_Λ)
- **Other / bounds**: 8 (θ_QCD, a_e, a_μ, τ_e, τ_p, Lamb shift, LZ 2024, N_ν families)

**Total: 65 reference values cataloged.**

**Inconsistencies between paper and code: 5** (§10.1–§10.5).

All other reference values are mutually consistent between paper and code.

---

## 12. Maintenance protocol

When PDG releases a new edition (PDG 2026 expected late 2026):

1. Re-read this manifest top-to-bottom against the new PDG listings.
2. Update each row's **Value** and **Uncertainty** column to reflect the new edition.
3. Update the **Date** column.
4. Update the **Source** column to "PDG 2026" / "PTEP 2026 ..." citations.
5. Update `pdg.rs::Pdg2024::new()` numerical literals, renaming the struct to
   `Pdg2026` if signature-bumping is appropriate.
6. Update `empirical_constants.py` and `rational_search_baseline.py` literals.
7. Update paper text wherever numerical PDG values appear inline (currently lines
   903, 927, 2295-2296 — see Inconsistency table for the canonical list).
8. Re-run the discrimination pipeline; any prediction-vs-observation residual that
   shifts by > 1σ requires a paper update.

When CODATA releases a new adjustment (CODATA 2026 expected ~2027):

1. Update §1, §7 rows.
2. Update `empirical_constants.py:99-104` α_em row (currently flagged in §10.4).
3. Re-run the m_e/m_Planck identity check; update the 17 ppm figure if needed.

---

## 13. Source files referenced

- `book/paper/substrate_particle_equations.tex` — primary paper text.
- `book/scripts/cy3_substrate_discrimination/rust_solver/src/pdg.rs` — frozen
  PDG 2024 snapshot used by the chi-squared discrimination pipeline.
- `book/scripts/cy3_substrate_discrimination/empirical_constants.py` — Python-side
  PDG/CODATA constants used by Route 2/3/4 discrimination scripts.
- `book/scripts/cy3_substrate_discrimination/python_research/rational_search_baseline.py`
  — PDG-comparison reference values for the rational-search baseline (P6.5).
- `book/scripts/cy3_substrate_discrimination/references/p_sin2thetaW_alphaS_provenance.md`
  — companion audit on sin²θ_W and α_s(M_Z) rational closure rules.
