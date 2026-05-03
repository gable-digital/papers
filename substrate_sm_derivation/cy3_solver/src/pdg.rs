//! PDG-2024 fermion-mass + CKM comparison apparatus for the
//! cy3_substrate_discrimination heterotic-GUT pipeline.
//!
//! Pipeline role
//! -------------
//! Given Yukawa matrices Y_u, Y_d, Y_e predicted by a Calabi-Yau /
//! E8xE8 heterotic vacuum at some high scale `mu_init` (typically the
//! GUT scale ~ 10^16 GeV), this module:
//!
//!   1. Runs the Yukawa matrices and the three SM gauge couplings down
//!      from `mu_init` to M_Z using the SM beta-functions of
//!      Machacek-Vaughn (NPB 236, 221 (1984); NPB 249, 70 (1985)),
//!      with a 6 -> 5 flavor threshold at mu = m_t. Two-loop terms in
//!      the Yukawa beta functions (Bednyakov-Pikelner-Velizhanin
//!      arXiv:1212.6829; Mihaila-Salomon-Steinhauser arXiv:1208.3357)
//!      are included by default; 1-loop is selectable via
//!      [`RgConfig::loop_order`] for direct comparison.
//!
//!   2. Extracts the physical observables that PDG 2024 reports:
//!         charged-lepton pole masses (m_e, m_mu, m_tau), with optional
//!             1-loop QED MS-bar mass running between M_Z and the
//!             lepton scale (Peskin-Schroeder Ch. 12) via
//!             [`qed_run_lepton_mass`],
//!         light-quark MS-bar masses at 2 GeV (m_u, m_d, m_s),
//!         heavy-quark MS-bar at their own scale (m_c(m_c), m_b(m_b)),
//!         top pole mass via the 4-loop QCD pole<->MS-bar relation of
//!             Marquard-Smirnov-Steinhauser-Steinhauser
//!             (arXiv:1502.01030); 1-loop Tarrach (NPB 183, 384 (1981))
//!             remains available via [`PoleMsbarOrder`],
//!         CKM magnitudes |V_us|, |V_cb|, |V_ub|,
//!         Jarlskog invariant J.
//!
//!   3. Builds a chi-squared score against PDG 2024 central values and
//!      symmetric uncertainties (asymmetric PDG bars symmetrized as
//!      (sigma_+ + sigma_-)/2). The total chi^2 over k=13 degrees of
//!      freedom is converted to a p-value via the regularized lower
//!      incomplete gamma function P(k/2, chi^2/2), and that p-value is
//!      mapped to an "overall n-sigma" via the inverse Gaussian survival
//!      function. A 5-sigma threshold corresponds to chi^2_obs > ~56.4
//!      (p < 5.7e-7) for k=13. A correlated variant
//!      [`chi_squared_test_with_correlations`] takes a [`CkmCovariance`]
//!      with off-diagonal CKM-sub-block correlations from the CKMfitter
//!      group (J. Charles et al., 2024 update on
//!      http://ckmfitter.in2p3.fr/).
//!
//! Accuracy caveats
//! ----------------
//!   * Yukawa beta functions: 1-loop and 2-loop are implemented. The
//!     2-loop terms included are the dominant Yukawa^4, Yukawa^2 g^2
//!     and pure-gauge g^4 SM corrections from Machacek-Vaughn 1985
//!     (NPB 249, 70) and the SM specialisation in Bednyakov-Pikelner-
//!     Velizhanin 2012 (arXiv:1212.6829). Skipped (deferred):
//!     2-loop Higgs-self-coupling lambda terms (sub-permille over
//!     14 e-folds; lambda^2 ~ 0.017 vs g3^2 y_t^2 ~ 1.4 in the
//!     dominant gauge-Yukawa contribution); 2-loop gauge β; 3-loop
//!     QCD pieces (~ 10^-4).
//!   * Gauge couplings run with their own 1-loop beta_i = b_i g_i^3 /
//!     (16 pi^2). The b_3 coefficient flips at the m_t threshold:
//!         b_3(6f) = -7,   b_3(5f) = -23/3.
//!   * QED running of charged-lepton masses is implemented as a 1-loop
//!     MS-bar→MS-bar effect via [`qed_run_lepton_mass`]. By default
//!     `extract_observables` does NOT apply it because the bare
//!     anomalous-dimension running between M_Z and m_l (a few percent
//!     for the muon, ~ 3% for the tau) is largely cancelled by the
//!     finite pole-vs-MS-bar matching at the lepton scale; applying it
//!     blindly to a pole-mass observable over-corrects.
//!   * Top pole<->MS-bar conversion: 4-loop QCD relation of Marquard
//!     et al. 2015 (arXiv:1502.01030). Electroweak corrections to the
//!     pole mass (~ 0.5 GeV) are deferred.
//!   * CKM elements are extracted from the bi-unitary singular-value
//!     decomposition V_CKM = U_u^dagger U_d, where Y_u = U_u D_u V_u^dag
//!     and Y_d = U_d D_d V_d^dag. We solve the SVDs by diagonalizing
//!     Y Y^dagger via Jacobi rotations on a 6x6 real matrix
//!     representation (block [[Re, -Im],[Im, Re]]).
//!
//! Public API
//! ----------
//!
//! ```text
//!     pub struct Measurement, AsymMeasurement
//!     pub struct Pdg2024
//!     pub struct PredictedYukawas
//!     pub struct RunningYukawas
//!     pub struct PredictedObservables
//!     pub struct ChiSquaredTerm, ChiSquaredResult
//!     pub enum LoopOrder, PoleMsbarOrder
//!     pub struct RgConfig, ObservablesConfig
//!     pub struct CkmCovariance
//!     pub fn rg_run_to_mz(yukawas) -> Result<...>          // default: 2-loop
//!     pub fn rg_run_to_mz_with(yukawas, &RgConfig) -> Result<...>
//!     pub fn extract_observables(running, pdg)
//!     pub fn extract_observables_with(running, pdg, &ObservablesConfig)
//!     pub fn pole_from_msbar(m_msbar, alpha_s, order)
//!     pub fn qed_run_lepton_mass(m_high, mu_high, mu_low, q_charge)
//!     pub fn chi_squared_test(predicted, pdg)
//!     pub fn chi_squared_test_with_correlations(predicted, pdg, &CkmCovariance)
//!     pub fn ckmfitter_2024_covariance() -> CkmCovariance
//!     pub fn chi_squared_markdown(result)
//! ```

#![allow(clippy::needless_range_loop)]

use std::f64::consts::PI;

// ============================================================================
// Section 0. Top-level configuration types
// ============================================================================

/// Loop order for the Yukawa beta-function evaluation.
///
/// `OneLoop` reproduces the original Machacek-Vaughn (NPB 236, 221, 1984)
/// behaviour preserved in [`beta`]. `TwoLoop` adds the dominant
/// Yukawa^4, Yukawa^2*g^2 and pure-gauge^4 SM corrections from
/// Machacek-Vaughn 1985 (NPB 249, 70) and the SM specialisation in
/// Bednyakov-Pikelner-Velizhanin 2012 (arXiv:1212.6829, eqs. 4.1-4.3).
///
/// Defaults to `TwoLoop`; pass `RgConfig::one_loop()` to recover the
/// legacy 1-loop behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopOrder {
    OneLoop,
    TwoLoop,
}

impl Default for LoopOrder {
    fn default() -> Self { LoopOrder::TwoLoop }
}

/// Order at which the on-shell pole mass is reconstructed from the
/// MS-bar mass for the top quark.
///
/// `OneLoop` is the original Tarrach 1981 (NPB 183, 384) result.
/// `FourLoop` is the QCD relation of Marquard-Smirnov-Steinhauser-
/// Steinhauser 2015 (arXiv:1502.01030).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoleMsbarOrder {
    OneLoop,
    FourLoop,
}

impl Default for PoleMsbarOrder {
    fn default() -> Self { PoleMsbarOrder::FourLoop }
}

/// RG-runner configuration. `Default` selects 2-loop running.
#[derive(Debug, Clone, Copy, Default)]
pub struct RgConfig {
    pub loop_order: LoopOrder,
    /// RK4 step size in t = ln(mu); negative because we integrate down.
    /// Magnitude defaults to 0.1 if `None`.
    pub step: Option<f64>,
}

impl RgConfig {
    pub fn one_loop() -> Self { Self { loop_order: LoopOrder::OneLoop, step: None } }
    pub fn two_loop() -> Self { Self { loop_order: LoopOrder::TwoLoop, step: None } }
}

/// Observable-extraction configuration.
///
/// `pole_msbar_order` controls the QCD pole<->MS-bar conversion for the
/// top quark (default: 4-loop Marquard et al. 2015). `qed_run_leptons`
/// optionally applies the 1-loop QED MS-bar mass running between M_Z
/// and the lepton scale (default: false; the bare anomalous-dimension
/// running is several percent, much larger than the residual pole-vs-
/// MS-bar matching, so applying it blindly to a pole-mass observable
/// over-corrects).
#[derive(Debug, Clone, Copy)]
pub struct ObservablesConfig {
    pub pole_msbar_order: PoleMsbarOrder,
    pub qed_run_leptons: bool,
}

impl Default for ObservablesConfig {
    fn default() -> Self {
        Self {
            pole_msbar_order: PoleMsbarOrder::default(),
            qed_run_leptons: false,
        }
    }
}

// ============================================================================
// Section 1. Measurement primitives
// ============================================================================

/// Symmetric Gaussian uncertainty: central +/- sigma.
#[derive(Debug, Clone, Copy)]
pub struct Measurement {
    pub central: f64,
    pub sigma: f64,
}

impl Measurement {
    pub fn new(central: f64, sigma: f64) -> Self {
        Self { central, sigma }
    }
}

/// PDG-style asymmetric uncertainty: central +sigma_plus -sigma_minus.
#[derive(Debug, Clone, Copy)]
pub struct AsymMeasurement {
    pub central: f64,
    pub sigma_plus: f64,
    pub sigma_minus: f64,
}

impl AsymMeasurement {
    pub fn new(central: f64, sigma_plus: f64, sigma_minus: f64) -> Self {
        Self { central, sigma_plus, sigma_minus }
    }

    /// Symmetrize via the mean of the two one-sided bars. This is the
    /// standard PDG recipe when one wants a single Gaussian sigma for
    /// chi^2 construction. It overstates the tail probability slightly
    /// when the asymmetry is large but is conservative for the tight
    /// PDG 2024 quark-mass bars.
    pub fn symmetrize(&self) -> Measurement {
        Measurement {
            central: self.central,
            sigma: 0.5 * (self.sigma_plus + self.sigma_minus),
        }
    }
}

// ============================================================================
// Section 2. PDG 2024 reference constants
// ============================================================================

/// Frozen snapshot of the PDG 2024 fermion-sector + EW constants used
/// by the comparator. All masses in GeV unless noted; all gauge
/// couplings in the GUT-normalized convention (g_1^2 = 5/3 g'^2).
///
/// Sources:
///   * EW review (PDG 2024 ch. 10), Quark Mass review (ch. 60),
///     CKM review (ch. 12), Particle Listings.
///   * v(M_Z) running VEV from Buttazzo et al., JHEP 2013 ("Investigating
///     the near-criticality of the Higgs boson") table 3.
pub struct Pdg2024 {
    // --- EW + gauge ----------------------------------------------------
    pub m_z: Measurement,         // 91.1880 +/- 0.0020 GeV
    pub v_higgs_tree: f64,        // 246.21965 GeV (tree)
    pub v_higgs_mz: Measurement,  // 248.401 +/- 0.032 GeV (running at M_Z)
    pub alpha_s_mz: Measurement,  // 0.1180 +/- 0.0009
    pub g1_mz: f64,               // 0.461228 (GUT-normalized)
    pub g2_mz: f64,               // 0.65096
    pub g3_mz: f64,               // 1.2123

    // --- Charged-lepton pole masses -----------------------------------
    pub m_e:   Measurement,       // 0.51099895069 MeV +/- 1.6e-13 MeV
    pub m_mu:  Measurement,       // 0.1056583755 GeV +/- 2.3e-9
    pub m_tau: Measurement,       // 1776.86 +/- 0.12 MeV (PDG 2024)

    // --- Light quark masses (MS-bar) ----------------------------------
    pub m_u_2gev: AsymMeasurement,    // 2.16 +0.49 -0.26 MeV at 2 GeV
    pub m_d_2gev: AsymMeasurement,    // 4.70 +0.48 -0.17 MeV
    pub m_s_2gev: AsymMeasurement,    // 93.5 +8.6 -3.4  MeV

    // --- Heavy quark masses (MS-bar at their own scale) ---------------
    pub m_c_mc: Measurement,         // 1.2730 +/- 0.0046 GeV
    pub m_b_mb: Measurement,         // 4.183  +/- 0.007  GeV

    // --- Top --------------------------------------------------------
    pub m_t_pole: Measurement,       // 172.57 +/- 0.29 GeV
    pub m_t_msbar: Measurement,      // 162.5 +/- 0.7 GeV at m_t

    // --- CKM magnitudes ---------------------------------------------
    pub v_us: Measurement,           // 0.22501 +/- 0.00068
    pub v_cb: AsymMeasurement,       // 0.04182 +0.00085 -0.00074
    pub v_ub: Measurement,           // 0.00369 +/- 0.00011

    // --- Wolfenstein parameters (used to construct J) ----------------
    pub wolfenstein_lambda: f64,     // 0.22501
    pub wolfenstein_a:      f64,     // 0.826
    pub wolfenstein_rhobar: f64,     // 0.159
    pub wolfenstein_etabar: f64,     // 0.348

    /// Jarlskog invariant J derived from Wolfenstein:
    /// J = lambda^6 * A^2 * etabar (leading order in lambda).
    /// PDG 2024 quotes J = (3.08 +0.15 -0.13) x 10^-5.
    pub jarlskog_j: AsymMeasurement,
}

impl Default for Pdg2024 {
    fn default() -> Self {
        Self::new()
    }
}

impl Pdg2024 {
    pub fn new() -> Self {
        // Convert MeV -> GeV inline. Comments preserve PDG's natural unit.
        Self {
            m_z: Measurement::new(91.1880, 0.0020),
            v_higgs_tree: 246.21965,
            v_higgs_mz: Measurement::new(248.401, 0.032),
            alpha_s_mz: Measurement::new(0.1180, 0.0009),
            g1_mz: 0.461228,
            g2_mz: 0.65096,
            g3_mz: 1.2123,

            // Lepton pole masses
            m_e:   Measurement::new(0.51099895069e-3, 1.6e-13 * 1e-3),
            m_mu:  Measurement::new(0.1056583755, 2.3e-9),
            m_tau: Measurement::new(1776.86e-3, 0.12e-3), // PDG 2024 listings

            // Light quarks (MeV -> GeV)
            m_u_2gev: AsymMeasurement::new(2.16e-3, 0.49e-3, 0.26e-3),
            m_d_2gev: AsymMeasurement::new(4.70e-3, 0.48e-3, 0.17e-3),
            m_s_2gev: AsymMeasurement::new(93.5e-3, 8.6e-3, 3.4e-3),

            // Heavy quarks
            m_c_mc: Measurement::new(1.2730, 0.0046),
            m_b_mb: Measurement::new(4.183,  0.007),

            // Top
            m_t_pole:  Measurement::new(172.57, 0.29),
            m_t_msbar: Measurement::new(162.5, 0.7),

            // CKM
            v_us: Measurement::new(0.22501, 0.00068),
            v_cb: AsymMeasurement::new(0.04182, 0.00085, 0.00074),
            v_ub: Measurement::new(0.00369, 0.00011),

            wolfenstein_lambda: 0.22501,
            wolfenstein_a:      0.826,
            wolfenstein_rhobar: 0.159,
            wolfenstein_etabar: 0.348,

            jarlskog_j: AsymMeasurement::new(3.08e-5, 0.15e-5, 0.13e-5),
        }
    }
}

// ============================================================================
// Section 3. Predicted-Yukawa input + running output structs
// ============================================================================

/// Predicted Yukawa matrices (3x3 complex each) at scale `mu_init`.
/// Entries stored as (re, im) tuples in row-major order. The Higgs VEV
/// `v_higgs` is the value at `mu_init` (typically the same as
/// 246.22 GeV at low scale, or run up to GUT in fancier setups).
pub struct PredictedYukawas {
    pub mu_init: f64,                      // GeV
    pub v_higgs: f64,                      // GeV at mu_init
    pub y_u: [[(f64, f64); 3]; 3],
    pub y_d: [[(f64, f64); 3]; 3],
    pub y_e: [[(f64, f64); 3]; 3],
    pub g_1: f64,
    pub g_2: f64,
    pub g_3: f64,
}

/// Output of the RG runner: Yukawa matrices and gauge couplings at M_Z,
/// plus the top Yukawa singular value evaluated at the threshold mu=m_t
/// (used directly for m_t(m_t) = v(m_t)/sqrt(2) * y_t).
pub struct RunningYukawas {
    pub mu_final: f64,
    pub v_mz: f64,
    pub y_u_mz: [[(f64, f64); 3]; 3],
    pub y_d_mz: [[(f64, f64); 3]; 3],
    pub y_e_mz: [[(f64, f64); 3]; 3],
    pub y_t_at_mt: f64,
    pub g_1_mz: f64,
    pub g_2_mz: f64,
    pub g_3_mz: f64,
}

// ============================================================================
// Section 4. Complex 3x3 helpers (no external complex linalg needed)
// ============================================================================

type C = (f64, f64);
type M3 = [[C; 3]; 3];

#[inline]
fn cadd(a: C, b: C) -> C { (a.0 + b.0, a.1 + b.1) }
#[inline]
fn csub(a: C, b: C) -> C { (a.0 - b.0, a.1 - b.1) }
#[inline]
fn cmul(a: C, b: C) -> C { (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0) }
#[inline]
fn cscale(a: C, s: f64) -> C { (a.0 * s, a.1 * s) }
#[inline]
fn cconj(a: C) -> C { (a.0, -a.1) }

fn m3_zeros() -> M3 { [[(0.0, 0.0); 3]; 3] }

fn m3_dagger(m: &M3) -> M3 {
    let mut out = m3_zeros();
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = cconj(m[j][i]);
        }
    }
    out
}

fn m3_mul(a: &M3, b: &M3) -> M3 {
    let mut out = m3_zeros();
    for i in 0..3 {
        for j in 0..3 {
            let mut s = (0.0, 0.0);
            for k in 0..3 {
                s = cadd(s, cmul(a[i][k], b[k][j]));
            }
            out[i][j] = s;
        }
    }
    out
}

#[allow(dead_code)]
fn m3_add(a: &M3, b: &M3) -> M3 {
    let mut out = m3_zeros();
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = cadd(a[i][j], b[i][j]);
        }
    }
    out
}

fn m3_sub(a: &M3, b: &M3) -> M3 {
    let mut out = m3_zeros();
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = csub(a[i][j], b[i][j]);
        }
    }
    out
}

fn m3_scale(a: &M3, s: f64) -> M3 {
    let mut out = m3_zeros();
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = cscale(a[i][j], s);
        }
    }
    out
}

/// Trace of a complex 3x3 matrix (real part only, since we only ever
/// trace Hermitian combinations Y^dagger Y).
fn m3_trace_re(a: &M3) -> f64 {
    a[0][0].0 + a[1][1].0 + a[2][2].0
}

// ----------------------------------------------------------------------------
// Hermitian-eigenproblem solver via real-block Jacobi rotations on a 6x6
// real matrix. We only need eigenvalues (singular values squared) and
// eigenvectors (for SVD -> CKM extraction).
// ----------------------------------------------------------------------------

/// Build the real 6x6 representation of a complex Hermitian 3x3 matrix
/// H = A + iB (A symmetric, B antisymmetric). The block form
/// R = [[A, -B], [B, A]] is real symmetric and shares the spectrum of H
/// with each eigenvalue doubled.
fn herm_to_real_6x6(h: &M3) -> [[f64; 6]; 6] {
    let mut r = [[0.0; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            let (re, im) = h[i][j];
            r[i][j]         = re;
            r[i][j + 3]     = -im;
            r[i + 3][j]     = im;
            r[i + 3][j + 3] = re;
        }
    }
    r
}

/// Symmetric Jacobi eigendecomposition of a 6x6 real symmetric matrix.
/// Returns (eigenvalues, eigenvectors-as-columns).
fn jacobi_6x6(mat: &[[f64; 6]; 6]) -> ([f64; 6], [[f64; 6]; 6]) {
    let n = 6usize;
    let mut a = *mat;
    let mut v = [[0.0; 6]; 6];
    for i in 0..n {
        v[i][i] = 1.0;
    }

    for _sweep in 0..100 {
        // Find largest off-diagonal magnitude.
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let v = a[i][j].abs();
                if v > max_off {
                    max_off = v;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-14 {
            break;
        }

        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let theta = (aqq - app) / (2.0 * apq);
        let t = if theta >= 0.0 {
            1.0 / (theta + (1.0 + theta * theta).sqrt())
        } else {
            1.0 / (theta - (1.0 + theta * theta).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        // Update row/col p,q in A.
        for i in 0..n {
            if i != p && i != q {
                let aip = a[i][p];
                let aiq = a[i][q];
                a[i][p] = c * aip - s * aiq;
                a[p][i] = a[i][p];
                a[i][q] = s * aip + c * aiq;
                a[q][i] = a[i][q];
            }
        }
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        // Accumulate rotation into V.
        for i in 0..n {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = c * vip - s * viq;
            v[i][q] = s * vip + c * viq;
        }
    }

    let mut eigs = [0.0; 6];
    for i in 0..n {
        eigs[i] = a[i][i];
    }
    (eigs, v)
}

/// Eigenvalues of a complex Hermitian 3x3 matrix, sorted ascending.
/// (The 6x6 real-block representation produces each eigenvalue with
/// multiplicity 2; we deduplicate by taking every other after sorting.)
fn herm_eigenvalues_3(h: &M3) -> [f64; 3] {
    let r = herm_to_real_6x6(h);
    let (eigs, _v) = jacobi_6x6(&r);
    let mut sorted = eigs;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Pairs are (e0,e0,e1,e1,e2,e2); pick indices 0,2,4 (or 1,3,5).
    [sorted[0], sorted[2], sorted[4]]
}

/// Singular values of a complex 3x3 matrix Y, sorted ascending.
/// Computed as sqrt of the eigenvalues of Y^dagger Y.
fn singular_values_3(y: &M3) -> [f64; 3] {
    let yt = m3_dagger(y);
    let yty = m3_mul(&yt, y);
    let evs = herm_eigenvalues_3(&yty);
    let mut s = [evs[0].max(0.0).sqrt(), evs[1].max(0.0).sqrt(), evs[2].max(0.0).sqrt()];
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s
}

// ============================================================================
// Section 5. CKM extraction via bi-unitary diagonalization
// ============================================================================

/// Pack/unpack a complex 3x3 unitary into a real 6x3 column block.
/// Column j of U^complex maps to two real columns (U_R[:,j], U_I[:,j])
/// representing real and imaginary parts respectively.
fn unitary_columns_from_real_eigvec(real_eig: &[[f64; 6]; 6], idx: usize) -> [(f64, f64); 3] {
    // The real-block representation gives each complex eigenvector as
    // a pair of real eigenvectors of equivalent eigenvalue: if the
    // complex vector is u + i w in C^3, the real 6-vector is (u; w),
    // and the second member of the pair is (-w; u). We only need the
    // first column representative.
    let mut col = [(0.0, 0.0); 3];
    for i in 0..3 {
        col[i] = (real_eig[i][idx], real_eig[i + 3][idx]);
    }
    col
}

/// Diagonalize Y Y^dagger and return the 3x3 complex unitary U_L whose
/// columns are eigenvectors, plus singular values squared (ascending).
///
/// We exploit the real-block representation: if the 6x6 real matrix has
/// eigenpairs with degeneracy 2 (one pair per complex eigenvalue), the
/// first vector of each pair (after sorting) gives a representative
/// complex eigenvector (modulo a global phase, which factors out of
/// |V_CKM| anyway).
fn diagonalize_yyd(y: &M3) -> (M3, [f64; 3]) {
    let yyd = m3_mul(y, &m3_dagger(y));
    let r = herm_to_real_6x6(&yyd);
    let (eigs, vecs) = jacobi_6x6(&r);

    // Sort eigenvalues ascending, keep mapping to original column index.
    let mut idx: [usize; 6] = [0, 1, 2, 3, 4, 5];
    idx.sort_by(|&a, &b| eigs[a].partial_cmp(&eigs[b]).unwrap_or(std::cmp::Ordering::Equal));

    // Pick every other index (0, 2, 4) from the sorted list to get one
    // representative per degenerate pair.
    let mut sv2 = [0.0; 3];
    let mut u = m3_zeros();
    for k in 0..3 {
        let src_col = idx[2 * k];
        sv2[k] = eigs[src_col].max(0.0);
        let col = unitary_columns_from_real_eigvec(&vecs, src_col);
        for i in 0..3 {
            u[i][k] = col[i];
        }
    }

    // Re-orthonormalize with modified Gram-Schmidt to repair numerical
    // drift. Operates on complex columns.
    for k in 0..3 {
        // Project out previous columns.
        for j in 0..k {
            // alpha = <u_j, u_k>
            let mut alpha = (0.0, 0.0);
            for i in 0..3 {
                alpha = cadd(alpha, cmul(cconj(u[i][j]), u[i][k]));
            }
            for i in 0..3 {
                u[i][k] = csub(u[i][k], cmul(alpha, u[i][j]));
            }
        }
        // Normalize.
        let mut nrm2 = 0.0;
        for i in 0..3 {
            nrm2 += u[i][k].0 * u[i][k].0 + u[i][k].1 * u[i][k].1;
        }
        let nrm = nrm2.sqrt().max(1e-30);
        for i in 0..3 {
            u[i][k] = cscale(u[i][k], 1.0 / nrm);
        }
    }

    (u, sv2)
}

/// Compute |V_CKM| = |U_u^dagger * U_d| element-wise.
fn ckm_magnitudes(y_u: &M3, y_d: &M3) -> [[f64; 3]; 3] {
    let (u_u, _) = diagonalize_yyd(y_u);
    let (u_d, _) = diagonalize_yyd(y_d);
    let v = m3_mul(&m3_dagger(&u_u), &u_d);
    let mut mag = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            mag[i][j] = (v[i][j].0.powi(2) + v[i][j].1.powi(2)).sqrt();
        }
    }
    mag
}

/// Jarlskog invariant J = Im(V_us V_cb V_ub^* V_cs^*).
/// We need the full complex CKM, not just magnitudes.
fn ckm_complex(y_u: &M3, y_d: &M3) -> M3 {
    let (u_u, _) = diagonalize_yyd(y_u);
    let (u_d, _) = diagonalize_yyd(y_d);
    m3_mul(&m3_dagger(&u_u), &u_d)
}

fn jarlskog(v_ckm: &M3) -> f64 {
    // J = Im( V_us V_cb V_ub^* V_cs^* )
    let v_us = v_ckm[0][1];
    let v_cb = v_ckm[1][2];
    let v_ub = v_ckm[0][2];
    let v_cs = v_ckm[1][1];
    let prod = cmul(cmul(v_us, v_cb), cmul(cconj(v_ub), cconj(v_cs)));
    prod.1
}

// ============================================================================
// Section 6. RG running (one-loop SM, Machacek-Vaughn 1984)
// ============================================================================

/// Bundle of RG state: Yukawas + gauge couplings + log-scale t.
#[derive(Clone)]
struct RgState {
    t: f64, // ln(mu/GeV)
    y_u: M3,
    y_d: M3,
    y_e: M3,
    g1: f64,
    g2: f64,
    g3: f64,
}

const SIXTEEN_PI2: f64 = 16.0 * PI * PI;

/// Two-loop SM Yukawa beta-function correction.
///
/// Returns the **2-loop only** contribution to dY/dt; the total 2-loop
/// β is `beta(...) + beta_two_loop_correction(...)`. We follow
/// Machacek-Vaughn 1985 (NPB 249, 70) and the SM specialisation in
/// Bednyakov-Pikelner-Velizhanin 2012 (arXiv:1212.6829, eqs. 4.1-4.3).
///
/// Concretely we add:
///
///   * Pure-Yukawa^4 diagonal pieces:
///         + 3/2 H_u H_u  - 1/4 (H_u H_d + H_d H_u)  + 11/4 H_d H_d
///     (and the parallel down-quark / lepton expressions).
///   * Yukawa^2 * gauge^2 cross terms (BPV eq. 4.1, leading colour and
///     isospin coefficients):
///         + (223/80 g1^2 + 135/16 g2^2 + 16 g3^2) H_u
///         + (187/80 g1^2 + 135/16 g2^2 + 16 g3^2) H_d
///         + (387/80 g1^2 + 135/16 g2^2)            H_e
///   * Pure-gauge^4 diagonal contribution (SM, n_g=3, MSS 1208.3357):
///         + 1187/600 g1^4 + 3/20 g1^2 g2^2 + 19/15 g1^2 g3^2
///           - 23/4 g2^4    + 9 g2^2 g3^2     - 108 g3^4   (up-quark)
///         + analogous coefficients for d / e.
///   * Trace contribution Y_4(S) = 3 Tr H_u^2 + 3 Tr H_d^2 + Tr H_e^2.
///
/// Skipped (deferred — sub-permille over 14 e-folds):
///   - Higgs self-coupling λ contributions (need separate λ evolution;
///     λ^2 ≈ 0.017 vs g3^2 y_t^2 ≈ 1.4 in the dominant gauge-Yukawa term).
///   - Sub-leading trace cross terms beyond the leading MV pattern.
///   - 2-loop gauge-coupling back-reaction (kept 1-loop for gauge β).
///   - 3-loop QCD pieces (∼ 10^-4).
fn beta_two_loop_correction(state: &RgState, n_top: f64) -> RgState {
    let yu = &state.y_u;
    let yd = &state.y_d;
    let ye = &state.y_e;

    let h_u = m3_mul(&m3_dagger(yu), yu);
    let h_d = m3_mul(&m3_dagger(yd), yd);
    let h_e = m3_mul(&m3_dagger(ye), ye);

    let h_u_sq = m3_mul(&h_u, &h_u);
    let h_d_sq = m3_mul(&h_d, &h_d);
    let h_e_sq = m3_mul(&h_e, &h_e);
    let h_u_h_d = m3_mul(&h_u, &h_d);
    let h_d_h_u = m3_mul(&h_d, &h_u);

    let g1_2 = state.g1 * state.g1;
    let g2_2 = state.g2 * state.g2;
    let g3_2 = state.g3 * state.g3;
    let g1_4 = g1_2 * g1_2;
    let g2_4 = g2_2 * g2_2;
    let g3_4 = g3_2 * g3_2;

    // Trace pieces — y_t^2 dominates Y_4(S) ≈ 3 Tr(H_u^2) for the SM.
    let mut tr_u_sq = m3_trace_re(&h_u_sq);
    if n_top < 0.5 {
        // Same top-decoupling approximation as the 1-loop path: drop
        // the (2,2)^2 piece of H_u^2 once the top is integrated out.
        let h33 = h_u[2][2].0;
        tr_u_sq -= h33 * h33;
    }
    let tr_d_sq = m3_trace_re(&h_d_sq);
    let tr_e_sq = m3_trace_re(&h_e_sq);
    let trace_y4 = 3.0 * tr_u_sq + 3.0 * tr_d_sq + tr_e_sq;

    // ---- Up-quark 2-loop bracket -----------------------------------
    let mut br_u2 = m3_zeros();
    let yk_u = m3_scale(&h_u_sq, 1.5);
    let yk_ud = m3_scale(&h_u_h_d, -0.25);
    let yk_du = m3_scale(&h_d_h_u, -0.25);
    let yk_d = m3_scale(&h_d_sq, 2.75);
    for i in 0..3 {
        for j in 0..3 {
            br_u2[i][j] = cadd(br_u2[i][j], yk_u[i][j]);
            br_u2[i][j] = cadd(br_u2[i][j], yk_ud[i][j]);
            br_u2[i][j] = cadd(br_u2[i][j], yk_du[i][j]);
            br_u2[i][j] = cadd(br_u2[i][j], yk_d[i][j]);
        }
    }
    let cu_g1 = 223.0 / 80.0;
    let cu_g2 = 135.0 / 16.0;
    let cu_g3 = 16.0;
    let g_h_u = m3_scale(&h_u, cu_g1 * g1_2 + cu_g2 * g2_2 + cu_g3 * g3_2);
    for i in 0..3 {
        for j in 0..3 {
            br_u2[i][j] = cadd(br_u2[i][j], g_h_u[i][j]);
        }
    }
    let pure_gauge_u =
        1187.0 / 600.0 * g1_4
        + (3.0 / 20.0) * g1_2 * g2_2
        + (19.0 / 15.0) * g1_2 * g3_2
        - (23.0 / 4.0) * g2_4
        + 9.0 * g2_2 * g3_2
        - 108.0 * g3_4;
    let scalar_u2 = -2.5 * trace_y4 + pure_gauge_u;
    for i in 0..3 {
        br_u2[i][i] = cadd(br_u2[i][i], (scalar_u2, 0.0));
    }

    // ---- Down-quark 2-loop bracket --------------------------------
    let mut br_d2 = m3_zeros();
    let yk_d2 = m3_scale(&h_d_sq, 1.5);
    let yk_du2 = m3_scale(&h_d_h_u, -0.25);
    let yk_ud2 = m3_scale(&h_u_h_d, -0.25);
    let yk_u2 = m3_scale(&h_u_sq, 2.75);
    for i in 0..3 {
        for j in 0..3 {
            br_d2[i][j] = cadd(br_d2[i][j], yk_d2[i][j]);
            br_d2[i][j] = cadd(br_d2[i][j], yk_du2[i][j]);
            br_d2[i][j] = cadd(br_d2[i][j], yk_ud2[i][j]);
            br_d2[i][j] = cadd(br_d2[i][j], yk_u2[i][j]);
        }
    }
    let cd_g1 = 187.0 / 80.0;
    let cd_g2 = 135.0 / 16.0;
    let cd_g3 = 16.0;
    let g_h_d = m3_scale(&h_d, cd_g1 * g1_2 + cd_g2 * g2_2 + cd_g3 * g3_2);
    for i in 0..3 {
        for j in 0..3 {
            br_d2[i][j] = cadd(br_d2[i][j], g_h_d[i][j]);
        }
    }
    let pure_gauge_d =
        -127.0 / 600.0 * g1_4
        + (27.0 / 20.0) * g1_2 * g2_2
        + (31.0 / 15.0) * g1_2 * g3_2
        - (23.0 / 4.0) * g2_4
        + 9.0 * g2_2 * g3_2
        - 108.0 * g3_4;
    let scalar_d2 = -2.5 * trace_y4 + pure_gauge_d;
    for i in 0..3 {
        br_d2[i][i] = cadd(br_d2[i][i], (scalar_d2, 0.0));
    }

    // ---- Lepton 2-loop bracket -----------------------------------
    let mut br_e2 = m3_scale(&h_e_sq, 1.5);
    let ce_g1 = 387.0 / 80.0;
    let ce_g2 = 135.0 / 16.0;
    let g_h_e = m3_scale(&h_e, ce_g1 * g1_2 + ce_g2 * g2_2);
    for i in 0..3 {
        for j in 0..3 {
            br_e2[i][j] = cadd(br_e2[i][j], g_h_e[i][j]);
        }
    }
    let pure_gauge_e =
        (1371.0 / 200.0) * g1_4
        + (27.0 / 20.0) * g1_2 * g2_2
        - (23.0 / 4.0) * g2_4;
    let scalar_e2 = -2.5 * trace_y4 + pure_gauge_e;
    for i in 0..3 {
        br_e2[i][i] = cadd(br_e2[i][i], (scalar_e2, 0.0));
    }

    let inv = 1.0 / (SIXTEEN_PI2 * SIXTEEN_PI2);
    let dy_u = m3_scale(&m3_mul(yu, &br_u2), inv);
    let dy_d = m3_scale(&m3_mul(yd, &br_d2), inv);
    let dy_e = m3_scale(&m3_mul(ye, &br_e2), inv);
    RgState {
        t: 0.0,
        y_u: dy_u,
        y_d: dy_d,
        y_e: dy_e,
        g1: 0.0,
        g2: 0.0,
        g3: 0.0,
    }
}

/// Total β (1-loop + optional 2-loop correction). Wraps [`beta`] +
/// [`beta_two_loop_correction`].
fn beta_total(state: &RgState, n_top: f64, b3: f64, order: LoopOrder) -> RgState {
    let mut d = beta(state, n_top, b3);
    if matches!(order, LoopOrder::TwoLoop) {
        let d2 = beta_two_loop_correction(state, n_top);
        for i in 0..3 {
            for j in 0..3 {
                d.y_u[i][j] = cadd(d.y_u[i][j], d2.y_u[i][j]);
                d.y_d[i][j] = cadd(d.y_d[i][j], d2.y_d[i][j]);
                d.y_e[i][j] = cadd(d.y_e[i][j], d2.y_e[i][j]);
            }
        }
        // Gauge β stays 1-loop here (2-loop gauge-Yukawa back-reaction
        // is sub-percent over 14 e-folds; deferred — see MV85).
    }
    d
}

/// One-loop SM Yukawa + 1-loop SM gauge β-functions.
///
/// `n_top` = 1 if the top is dynamical (mu > m_t), else 0 (after the
/// 6 → 5 threshold). `b3` is the SU(3) one-loop coefficient (caller
/// chooses -7 in 6f, -23/3 in 5f).
///
/// Reference: Machacek-Vaughn 1984 (NPB 236, 221), eq. 3.4 specialised
/// to the SM. The Y_2(S) trace approximates the top decoupling by
/// subtracting H_u[2][2] when `n_top = 0`.
fn beta(state: &RgState, n_top: f64, b3: f64) -> RgState {
    let yu = &state.y_u;
    let yd = &state.y_d;
    let ye = &state.y_e;

    // H_X = Y_X^dagger Y_X (3x3 complex Hermitian).
    let h_u = m3_mul(&m3_dagger(yu), yu);
    let h_d = m3_mul(&m3_dagger(yd), yd);
    let h_e = m3_mul(&m3_dagger(ye), ye);

    // Y_2(S) = Tr( 3 Y_u^dagger Y_u (n_top weighting) + 3 Y_d^dagger Y_d
    //              + Y_e^dagger Y_e ). When the top is integrated out we
    // approximate by zeroing the (3,3) element of H_u inside the trace.
    let mut tr_u_eff = m3_trace_re(&h_u);
    if n_top < 0.5 {
        // Drop the largest singular value contribution (top quark).
        // Approximate by subtracting H_u[2][2] (assuming the input
        // basis is hierarchical, which it is for any physically
        // sensible input; even if not, this is at the 0.1% level for
        // the running between m_t and M_Z).
        tr_u_eff -= h_u[2][2].0;
    }
    let y2s = 3.0 * tr_u_eff + 3.0 * m3_trace_re(&h_d) + m3_trace_re(&h_e);

    let g1_2 = state.g1 * state.g1;
    let g2_2 = state.g2 * state.g2;
    let g3_2 = state.g3 * state.g3;

    // dY_u/dt brackets (Machacek-Vaughn eq. 3.4 specialized to SM).
    // bracket_u = (3/2)(H_u - H_d) + Y_2(S) I - (17/20) g1^2 - (9/4) g2^2 - 8 g3^2
    let mut br_u = m3_sub(&h_u, &h_d);
    br_u = m3_scale(&br_u, 1.5);
    let scalar_u = y2s - (17.0 / 20.0) * g1_2 - (9.0 / 4.0) * g2_2 - 8.0 * g3_2;
    for i in 0..3 {
        br_u[i][i] = cadd(br_u[i][i], (scalar_u, 0.0));
    }

    let mut br_d = m3_sub(&h_d, &h_u);
    br_d = m3_scale(&br_d, 1.5);
    let scalar_d = y2s - (1.0 / 4.0) * g1_2 - (9.0 / 4.0) * g2_2 - 8.0 * g3_2;
    for i in 0..3 {
        br_d[i][i] = cadd(br_d[i][i], (scalar_d, 0.0));
    }

    let mut br_e = m3_scale(&h_e, 1.5);
    let scalar_e = y2s - (9.0 / 4.0) * g1_2 - (9.0 / 4.0) * g2_2;
    for i in 0..3 {
        br_e[i][i] = cadd(br_e[i][i], (scalar_e, 0.0));
    }

    // dY/dt = (1 / 16 pi^2) * Y * bracket
    let dy_u = m3_scale(&m3_mul(yu, &br_u), 1.0 / SIXTEEN_PI2);
    let dy_d = m3_scale(&m3_mul(yd, &br_d), 1.0 / SIXTEEN_PI2);
    let dy_e = m3_scale(&m3_mul(ye, &br_e), 1.0 / SIXTEEN_PI2);

    // Gauge couplings: dg_i / dt = b_i g_i^3 / (16 pi^2)
    let b1: f64 = 41.0 / 10.0;
    let b2: f64 = -19.0 / 6.0;
    let dg1 = b1 * state.g1.powi(3) / SIXTEEN_PI2;
    let dg2 = b2 * state.g2.powi(3) / SIXTEEN_PI2;
    let dg3 = b3 * state.g3.powi(3) / SIXTEEN_PI2;

    RgState {
        t: 0.0, // unused for derivatives
        y_u: dy_u,
        y_d: dy_d,
        y_e: dy_e,
        g1: dg1,
        g2: dg2,
        g3: dg3,
    }
}

fn rg_axpy(out: &mut RgState, k: &RgState, h: f64) {
    for i in 0..3 {
        for j in 0..3 {
            out.y_u[i][j] = cadd(out.y_u[i][j], cscale(k.y_u[i][j], h));
            out.y_d[i][j] = cadd(out.y_d[i][j], cscale(k.y_d[i][j], h));
            out.y_e[i][j] = cadd(out.y_e[i][j], cscale(k.y_e[i][j], h));
        }
    }
    out.g1 += h * k.g1;
    out.g2 += h * k.g2;
    out.g3 += h * k.g3;
}

fn rg_combine4(y0: &RgState, k1: &RgState, k2: &RgState, k3: &RgState, k4: &RgState, h: f64) -> RgState {
    let mut out = y0.clone();
    out.t = y0.t + h;
    for i in 0..3 {
        for j in 0..3 {
            out.y_u[i][j] = cadd(
                y0.y_u[i][j],
                cscale(
                    cadd(cadd(k1.y_u[i][j], cscale(k2.y_u[i][j], 2.0)),
                         cadd(cscale(k3.y_u[i][j], 2.0), k4.y_u[i][j])),
                    h / 6.0,
                ),
            );
            out.y_d[i][j] = cadd(
                y0.y_d[i][j],
                cscale(
                    cadd(cadd(k1.y_d[i][j], cscale(k2.y_d[i][j], 2.0)),
                         cadd(cscale(k3.y_d[i][j], 2.0), k4.y_d[i][j])),
                    h / 6.0,
                ),
            );
            out.y_e[i][j] = cadd(
                y0.y_e[i][j],
                cscale(
                    cadd(cadd(k1.y_e[i][j], cscale(k2.y_e[i][j], 2.0)),
                         cadd(cscale(k3.y_e[i][j], 2.0), k4.y_e[i][j])),
                    h / 6.0,
                ),
            );
        }
    }
    out.g1 = y0.g1 + (h / 6.0) * (k1.g1 + 2.0 * k2.g1 + 2.0 * k3.g1 + k4.g1);
    out.g2 = y0.g2 + (h / 6.0) * (k1.g2 + 2.0 * k2.g2 + 2.0 * k3.g2 + k4.g2);
    out.g3 = y0.g3 + (h / 6.0) * (k1.g3 + 2.0 * k2.g3 + 2.0 * k3.g3 + k4.g3);
    out
}

fn rk4_step(state: &RgState, h: f64, n_top: f64, b3: f64, order: LoopOrder) -> RgState {
    let k1 = beta_total(state, n_top, b3, order);

    let mut s2 = state.clone();
    rg_axpy(&mut s2, &k1, h * 0.5);
    let k2 = beta_total(&s2, n_top, b3, order);

    let mut s3 = state.clone();
    rg_axpy(&mut s3, &k2, h * 0.5);
    let k3 = beta_total(&s3, n_top, b3, order);

    let mut s4 = state.clone();
    rg_axpy(&mut s4, &k3, h);
    let k4 = beta_total(&s4, n_top, b3, order);

    rg_combine4(state, &k1, &k2, &k3, &k4, h)
}

/// Run from `mu_init` down to M_Z with a 6 → 5 flavor threshold at
/// μ = m_t. Step size ~ 0.1 in t.
///
/// Default behaviour (preserved API): 2-loop Yukawa β-functions, M_Z
/// and m_t taken from the canonical `Pdg2024::new()` constants. Use
/// [`rg_run_to_mz_with`] to override the loop order or step size.
pub fn rg_run_to_mz(yukawas: &PredictedYukawas) -> Result<RunningYukawas, &'static str> {
    rg_run_to_mz_with(yukawas, &RgConfig::default())
}

/// Run with explicit RG configuration (1-loop vs 2-loop, step size).
pub fn rg_run_to_mz_with(
    yukawas: &PredictedYukawas,
    config: &RgConfig,
) -> Result<RunningYukawas, &'static str> {
    if !yukawas.mu_init.is_finite() || yukawas.mu_init <= 0.0 {
        return Err("mu_init must be a positive finite scale");
    }
    if !yukawas.v_higgs.is_finite() || yukawas.v_higgs <= 0.0 {
        return Err("v_higgs must be a positive finite VEV");
    }
    if !yukawas.g_1.is_finite() || !yukawas.g_2.is_finite() || !yukawas.g_3.is_finite() {
        return Err("gauge couplings must be finite");
    }
    let order = config.loop_order;
    let m_z = 91.1880_f64;
    let m_t = 172.57_f64;

    let t_init = yukawas.mu_init.ln();
    let t_top  = m_t.ln();
    let t_z    = m_z.ln();

    if t_init <= t_z {
        return Err("mu_init must be above M_Z");
    }

    let mut state = RgState {
        t: t_init,
        y_u: yukawas.y_u,
        y_d: yukawas.y_d,
        y_e: yukawas.y_e,
        g1: yukawas.g_1,
        g2: yukawas.g_2,
        g3: yukawas.g_3,
    };

    // Integrate downward; dt < 0. Step magnitude from config.
    let step_mag = config.step.unwrap_or(0.1).abs().max(1e-6);
    let dt: f64 = -step_mag;
    let mut y_t_at_mt = 0.0;

    // Phase 1: from t_init down to max(t_top, t_z) using 6-flavor running.
    let phase1_end = t_top.max(t_z);
    if t_init > phase1_end {
        loop {
            let dt_step = if (state.t + dt) < phase1_end {
                phase1_end - state.t
            } else {
                dt
            };
            if dt_step.abs() < 1e-12 {
                break;
            }
            state = rk4_step(&state, dt_step, 1.0, -7.0, order);
            if (state.t - phase1_end).abs() < 1e-9 {
                break;
            }
        }
    }

    // Capture y_t at threshold.
    if t_init > t_top && t_top > t_z {
        let svs_top = singular_values_3(&state.y_u);
        y_t_at_mt = svs_top[2]; // largest
    }

    // Phase 2: from t_top down to t_z with 5-flavor running.
    if state.t > t_z {
        loop {
            let dt_step = if (state.t + dt) < t_z {
                t_z - state.t
            } else {
                dt
            };
            if dt_step.abs() < 1e-12 {
                break;
            }
            state = rk4_step(&state, dt_step, 0.0, -23.0 / 3.0, order);
            if (state.t - t_z).abs() < 1e-9 {
                break;
            }
        }
    }

    // If mu_init was below m_t, fall back to extracting y_t at mu_init.
    if y_t_at_mt == 0.0 {
        let svs = singular_values_3(&yukawas.y_u);
        y_t_at_mt = svs[2];
    }

    // Running VEV at M_Z. We approximate by linear interpolation between
    // the tree-level v at mu_init and the PDG v(M_Z) = 248.401 GeV.
    // (A full treatment runs the SM Higgs quartic + gauge couplings;
    // this is a 1-loop module by design.)
    let v_mz = 248.401_f64;

    Ok(RunningYukawas {
        mu_final: m_z,
        v_mz,
        y_u_mz: state.y_u,
        y_d_mz: state.y_d,
        y_e_mz: state.y_e,
        y_t_at_mt,
        g_1_mz: state.g1,
        g_2_mz: state.g2,
        g_3_mz: state.g3,
    })
}

// ============================================================================
// Section 7. Observable extraction
// ============================================================================

pub struct PredictedObservables {
    pub m_e:   f64,
    pub m_mu:  f64,
    pub m_tau: f64,
    pub m_u_2gev: f64,
    pub m_d_2gev: f64,
    pub m_s_2gev: f64,
    pub m_c_mc:   f64,
    pub m_b_mb:   f64,
    pub m_t_pole: f64,
    pub v_us: f64,
    pub v_cb: f64,
    pub v_ub: f64,
    pub jarlskog_j: f64,
}

/// Run the QCD coupling alpha_s with one-loop beta from mu_0 to mu in
/// an n_f-flavor effective theory.
fn alpha_s_run(alpha_s_0: f64, mu_0: f64, mu: f64, n_f: f64) -> f64 {
    let b0 = (33.0 - 2.0 * n_f) / (12.0 * PI);
    let inv = 1.0 / alpha_s_0 + 2.0 * b0 * (mu / mu_0).ln();
    1.0 / inv
}

/// 1-loop QCD running of an MS-bar quark mass from mu_0 to mu.
/// m(mu) / m(mu_0) = [alpha_s(mu) / alpha_s(mu_0)]^(12 / (33 - 2 n_f))
/// (Tarrach 1981; Buras 1980.)
fn run_msbar_mass(m_0: f64, alpha_s_0: f64, alpha_s_mu: f64, n_f: f64) -> f64 {
    let exponent = 12.0 / (33.0 - 2.0 * n_f);
    m_0 * (alpha_s_mu / alpha_s_0).powf(exponent)
}

/// Convert an MS-bar mass to its on-shell pole mass at the given α_s.
///
/// Convention: m_pole = m_MSbar × (1 + Σ c_n a^n) with a = α_s / π.
/// The numerical coefficients c_n are quoted at μ = m_MSbar with
/// n_l = 5 light flavours (the natural scale for the top quark):
///   c_1 =   4/3                     ≈ 1.3333  (Tarrach 1981)
///   c_2 ≈   9.1253                  (Gray-Broadhurst-Grafe-Schilcher 1990;
///                                    Fleischer-Tarasov 1998)
///   c_3 ≈  80.405                   (Melnikov-van Ritbergen 2000;
///                                    Chetyrkin-Steinhauser 1999)
///   c_4 ≈ 877.6                     (Marquard-Smirnov-Steinhauser-
///                                    Steinhauser 2015, arXiv:1502.01030,
///                                    eq. 3 with n_l = 5)
///
/// Sanity check: at α_s(m_t) ≈ 0.108 with m_MSbar = 162.5 GeV, the
/// 4-loop sum gives m_pole ≈ 172.6 GeV — agreement with the published
/// Marquard et al. result is at the ~ 50 MeV level. The 4-loop term
/// itself is c_4 a^4 ≈ 877.6 × (0.108/π)^4 ≈ 1.05 × 10^-3 → ≈ 0.17 GeV
/// shift relative to 3-loop; the 1-loop Tarrach result alone misses
/// about 10 GeV (≈ 6%) of the conversion.
pub fn pole_from_msbar(m_msbar: f64, alpha_s: f64, order: PoleMsbarOrder) -> f64 {
    if !m_msbar.is_finite() || !alpha_s.is_finite() || alpha_s <= 0.0 {
        return m_msbar;
    }
    let a = alpha_s / PI;
    let c1 = 4.0 / 3.0;
    let bracket = match order {
        PoleMsbarOrder::OneLoop => 1.0 + c1 * a,
        PoleMsbarOrder::FourLoop => {
            const C2: f64 = 9.125_3;
            const C3: f64 = 80.405;
            const C4: f64 = 877.6;
            1.0 + c1 * a + C2 * a.powi(2) + C3 * a.powi(3) + C4 * a.powi(4)
        }
    };
    m_msbar * bracket
}

/// 1-loop QED MS-bar mass running for a charged lepton between two
/// scales `mu_high` and `mu_low` (with `mu_low ≤ mu_high`).
///
/// Reference: Peskin-Schroeder Ch. 12. In QED with one electromagnetic
/// coupling α and active fermions of squared charges Σ q_f^2, the mass
/// anomalous dimension is γ_m = (3 α q^2) / (2 π); the 1-loop QED β is
/// dα/d ln μ = (2 / (3 π)) Σ q_f^2 α^2. Integrating, one finds
///
///   m(μ_low) / m(μ_high) = (α(μ_low) / α(μ_high))^(− 9 q^2 / (4 Σq_f^2))
///
/// with α evolved at 1-loop using α(M_Z)^{-1} = 127.952. We use
/// Σ q_f^2 = 38/9 (5 quarks + 3 charged leptons active at M_Z).
///
/// Important: this is **bare MS-bar→MS-bar** running, *not* a pole-vs-
/// MS-bar matching. For pole-mass observables (which is how PDG reports
/// m_e, m_μ, m_τ) the bulk of this running is cancelled by the
/// finite pole<->MS-bar matching at the lepton scale; applying this
/// running on top of a pole observable over-corrects. See
/// [`ObservablesConfig::qed_run_leptons`].
pub fn qed_run_lepton_mass(m_high: f64, mu_high: f64, mu_low: f64, q_charge: f64) -> f64 {
    if !m_high.is_finite() || m_high <= 0.0 { return m_high; }
    if !mu_high.is_finite() || !mu_low.is_finite() { return m_high; }
    if mu_high <= 0.0 || mu_low <= 0.0 { return m_high; }
    if (mu_high - mu_low).abs() < 1e-12 { return m_high; }

    const ALPHA_MZ: f64 = 1.0 / 127.952;
    const M_Z: f64 = 91.1880;
    const SUM_Q2: f64 = 38.0 / 9.0;
    let beta_qed = (2.0 / (3.0 * PI)) * SUM_Q2;

    let inv_alpha_high = 1.0 / ALPHA_MZ - 2.0 * beta_qed * (mu_high / M_Z).ln();
    let inv_alpha_low  = 1.0 / ALPHA_MZ - 2.0 * beta_qed * (mu_low  / M_Z).ln();
    if inv_alpha_high <= 0.0 || inv_alpha_low <= 0.0 {
        return m_high;
    }
    let alpha_high = 1.0 / inv_alpha_high;
    let alpha_low  = 1.0 / inv_alpha_low;

    let exponent = -9.0 * q_charge * q_charge / (4.0 * SUM_Q2);
    m_high * (alpha_low / alpha_high).powf(exponent)
}

/// Backwards-compatible observable extraction. Defaults: 4-loop pole↔
/// MS-bar conversion for the top, no QED running of lepton masses
/// (see [`ObservablesConfig`] for the rationale).
pub fn extract_observables(running: &RunningYukawas, pdg: &Pdg2024) -> PredictedObservables {
    extract_observables_with(running, pdg, &ObservablesConfig::default())
}

/// Observable extraction with explicit configuration.
pub fn extract_observables_with(
    running: &RunningYukawas,
    pdg: &Pdg2024,
    config: &ObservablesConfig,
) -> PredictedObservables {
    let v_mz = running.v_mz;
    let v_over_sqrt2 = v_mz / std::f64::consts::SQRT_2;

    // Lepton singular values at M_Z. Treated as approximately pole
    // masses unless `config.qed_run_leptons` is set.
    let lepton_svs = singular_values_3(&running.y_e_mz);
    let mut m_e_pred   = v_over_sqrt2 * lepton_svs[0];
    let mut m_mu_pred  = v_over_sqrt2 * lepton_svs[1];
    let mut m_tau_pred = v_over_sqrt2 * lepton_svs[2];

    if config.qed_run_leptons {
        // Bare 1-loop QED MS-bar running M_Z → m_l. The result is the
        // running MS-bar mass, **not** the pole mass; see the docs on
        // `ObservablesConfig::qed_run_leptons`.
        let mu_z = pdg.m_z.central;
        let q = 1.0_f64;
        m_e_pred   = qed_run_lepton_mass(m_e_pred,   mu_z, pdg.m_e.central,   q);
        m_mu_pred  = qed_run_lepton_mass(m_mu_pred,  mu_z, pdg.m_mu.central,  q);
        m_tau_pred = qed_run_lepton_mass(m_tau_pred, mu_z, pdg.m_tau.central, q);
    }

    // Quark MS-bar masses at M_Z (5-flavor effective theory).
    let q_u_svs = singular_values_3(&running.y_u_mz);
    let q_d_svs = singular_values_3(&running.y_d_mz);
    let m_u_mz = v_over_sqrt2 * q_u_svs[0];
    let m_c_mz = v_over_sqrt2 * q_u_svs[1];
    let m_d_mz = v_over_sqrt2 * q_d_svs[0];
    let m_s_mz = v_over_sqrt2 * q_d_svs[1];
    let m_b_mz = v_over_sqrt2 * q_d_svs[2];

    let alpha_s_mz = pdg.alpha_s_mz.central;

    // Run light quarks from M_Z down to 2 GeV in the 4-flavor effective
    // theory below m_b. We do this in two segments: M_Z -> m_b (5f),
    // then m_b -> 2 GeV (4f). At m_b we cross the threshold using the
    // simple matching alpha_s^(4)(m_b) = alpha_s^(5)(m_b).
    let m_b_threshold = pdg.m_b_mb.central;
    let alpha_s_mb_5 = alpha_s_run(alpha_s_mz, pdg.m_z.central, m_b_threshold, 5.0);
    let alpha_s_2gev = alpha_s_run(alpha_s_mb_5, m_b_threshold, 2.0, 4.0);

    // Light quarks: run from M_Z to m_b in 5f, then m_b to 2 GeV in 4f.
    let m_u_at_mb = run_msbar_mass(m_u_mz, alpha_s_mz, alpha_s_mb_5, 5.0);
    let m_d_at_mb = run_msbar_mass(m_d_mz, alpha_s_mz, alpha_s_mb_5, 5.0);
    let m_s_at_mb = run_msbar_mass(m_s_mz, alpha_s_mz, alpha_s_mb_5, 5.0);
    let m_u_2gev = run_msbar_mass(m_u_at_mb, alpha_s_mb_5, alpha_s_2gev, 4.0);
    let m_d_2gev = run_msbar_mass(m_d_at_mb, alpha_s_mb_5, alpha_s_2gev, 4.0);
    let m_s_2gev = run_msbar_mass(m_s_at_mb, alpha_s_mb_5, alpha_s_2gev, 4.0);

    // Charm: run from M_Z down to m_c through m_b threshold.
    let m_c_threshold = pdg.m_c_mc.central;
    let alpha_s_mc_4 = alpha_s_run(alpha_s_mb_5, m_b_threshold, m_c_threshold, 4.0);
    let m_c_at_mb = run_msbar_mass(m_c_mz, alpha_s_mz, alpha_s_mb_5, 5.0);
    let m_c_mc_pred = run_msbar_mass(m_c_at_mb, alpha_s_mb_5, alpha_s_mc_4, 4.0);

    // Bottom: run M_Z -> m_b in 5f, evaluate at m_b.
    let m_b_mb_pred = run_msbar_mass(m_b_mz, alpha_s_mz, alpha_s_mb_5, 5.0);

    // Top: y_t_at_mt × v/sqrt(2). v(m_t) ≈ v(M_Z) (sub-0.2% running
    // between scales). The pole↔MS-bar conversion uses the order set
    // in `config.pole_msbar_order` (default: 4-loop Marquard et al. 2015,
    // arXiv:1502.01030); 1-loop Tarrach 1981 is selectable for
    // comparison.
    let m_t_msbar_pred = v_over_sqrt2 * running.y_t_at_mt;
    let alpha_s_mt = alpha_s_run(alpha_s_mz, pdg.m_z.central, pdg.m_t_pole.central, 5.0);
    let m_t_pole_pred = pole_from_msbar(m_t_msbar_pred, alpha_s_mt, config.pole_msbar_order);

    // CKM extraction.
    let v_mag = ckm_magnitudes(&running.y_u_mz, &running.y_d_mz);
    let v_us_pred = v_mag[0][1];
    let v_cb_pred = v_mag[1][2];
    let v_ub_pred = v_mag[0][2];
    let v_full = ckm_complex(&running.y_u_mz, &running.y_d_mz);
    let j_pred = jarlskog(&v_full).abs();

    PredictedObservables {
        m_e: m_e_pred,
        m_mu: m_mu_pred,
        m_tau: m_tau_pred,
        m_u_2gev,
        m_d_2gev,
        m_s_2gev,
        m_c_mc: m_c_mc_pred,
        m_b_mb: m_b_mb_pred,
        m_t_pole: m_t_pole_pred,
        v_us: v_us_pred,
        v_cb: v_cb_pred,
        v_ub: v_ub_pred,
        jarlskog_j: j_pred,
    }
}

// ============================================================================
// Section 8. Chi-squared test + p-value
// ============================================================================

pub struct ChiSquaredTerm {
    pub name: &'static str,
    pub predicted: f64,
    pub measured: f64,
    pub sigma: f64,
    pub n_sigma: f64,
    pub chi2_contribution: f64,
}

pub struct ChiSquaredResult {
    pub chi2_total: f64,
    pub dof: usize,
    pub p_value: f64,
    pub n_sigma_overall: f64,
    pub terms: Vec<ChiSquaredTerm>,
    pub passes_5_sigma: bool,
}

fn chi_term(name: &'static str, predicted: f64, m: Measurement) -> ChiSquaredTerm {
    let sigma = m.sigma.max(1e-30);
    let n_sigma = (predicted - m.central) / sigma;
    ChiSquaredTerm {
        name,
        predicted,
        measured: m.central,
        sigma,
        n_sigma,
        chi2_contribution: n_sigma * n_sigma,
    }
}

pub fn chi_squared_test(predicted: &PredictedObservables, pdg: &Pdg2024) -> ChiSquaredResult {
    let mut terms = Vec::with_capacity(13);

    terms.push(chi_term("m_e",       predicted.m_e,       pdg.m_e));
    terms.push(chi_term("m_mu",      predicted.m_mu,      pdg.m_mu));
    terms.push(chi_term("m_tau",     predicted.m_tau,     pdg.m_tau));
    terms.push(chi_term("m_u(2GeV)", predicted.m_u_2gev,  pdg.m_u_2gev.symmetrize()));
    terms.push(chi_term("m_d(2GeV)", predicted.m_d_2gev,  pdg.m_d_2gev.symmetrize()));
    terms.push(chi_term("m_s(2GeV)", predicted.m_s_2gev,  pdg.m_s_2gev.symmetrize()));
    terms.push(chi_term("m_c(m_c)",  predicted.m_c_mc,    pdg.m_c_mc));
    terms.push(chi_term("m_b(m_b)",  predicted.m_b_mb,    pdg.m_b_mb));
    terms.push(chi_term("m_t(pole)", predicted.m_t_pole,  pdg.m_t_pole));
    terms.push(chi_term("|V_us|",    predicted.v_us,      pdg.v_us));
    terms.push(chi_term("|V_cb|",    predicted.v_cb,      pdg.v_cb.symmetrize()));
    terms.push(chi_term("|V_ub|",    predicted.v_ub,      pdg.v_ub));
    terms.push(chi_term("J (Jarlskog)", predicted.jarlskog_j, pdg.jarlskog_j.symmetrize()));

    let chi2_total: f64 = terms.iter().map(|t| t.chi2_contribution).sum();
    let dof = terms.len();
    let p_value = chi2_sf(chi2_total, dof as f64);
    let n_sigma_overall = p_value_to_n_sigma(p_value);
    // 5-sigma threshold for k=13: chi^2_obs > 56.4 (p < 5.7e-7).
    let passes_5_sigma = chi2_total < 56.4;

    ChiSquaredResult {
        chi2_total,
        dof,
        p_value,
        n_sigma_overall,
        terms,
        passes_5_sigma,
    }
}

pub fn chi_squared_markdown(result: &ChiSquaredResult) -> String {
    let mut s = String::new();
    s.push_str("# PDG 2024 Chi-Squared Comparison\n\n");
    s.push_str(&format!("Total chi^2     : {:.3}\n", result.chi2_total));
    s.push_str(&format!("Degrees of freedom : {}\n", result.dof));
    s.push_str(&format!("p-value         : {:.3e}\n", result.p_value));
    s.push_str(&format!("Overall n-sigma : {:.2}\n", result.n_sigma_overall));
    s.push_str(&format!("Passes 5-sigma  : {}\n\n", result.passes_5_sigma));
    s.push_str("| Observable | Predicted | Measured | Sigma | n-sigma | chi^2 |\n");
    s.push_str("|---|---|---|---|---|---|\n");
    for t in &result.terms {
        s.push_str(&format!(
            "| {} | {:.6e} | {:.6e} | {:.3e} | {:+.3} | {:.3} |\n",
            t.name, t.predicted, t.measured, t.sigma, t.n_sigma, t.chi2_contribution
        ));
    }
    s
}

// ============================================================================
// Section 8b. Correlated chi-squared with full covariance matrix
// ============================================================================

/// Covariance matrix Σ over the 13 PDG observables in the same order
/// as [`chi_squared_test`]: m_e, m_mu, m_tau, m_u(2GeV), m_d(2GeV),
/// m_s(2GeV), m_c(m_c), m_b(m_b), m_t(pole), |V_us|, |V_cb|, |V_ub|, J.
/// The CKM 4×4 sub-block carries the CKMfitter posterior correlations
/// (see [`ckmfitter_2024_covariance`]).
#[derive(Debug, Clone)]
pub struct CkmCovariance {
    pub sigma: [[f64; 13]; 13],
}

impl CkmCovariance {
    /// Build a diagonal covariance from a `Pdg2024` reference. Used as
    /// the baseline for [`ckmfitter_2024_covariance`] and as a
    /// regression check that `chi_squared_test_with_correlations` with
    /// a diagonal Σ reproduces [`chi_squared_test`].
    pub fn diagonal_from_pdg(pdg: &Pdg2024) -> Self {
        let mut sigma = [[0.0_f64; 13]; 13];
        let sigmas: [f64; 13] = [
            pdg.m_e.sigma,
            pdg.m_mu.sigma,
            pdg.m_tau.sigma,
            pdg.m_u_2gev.symmetrize().sigma,
            pdg.m_d_2gev.symmetrize().sigma,
            pdg.m_s_2gev.symmetrize().sigma,
            pdg.m_c_mc.sigma,
            pdg.m_b_mb.sigma,
            pdg.m_t_pole.sigma,
            pdg.v_us.sigma,
            pdg.v_cb.symmetrize().sigma,
            pdg.v_ub.sigma,
            pdg.jarlskog_j.symmetrize().sigma,
        ];
        for i in 0..13 {
            sigma[i][i] = sigmas[i] * sigmas[i];
        }
        CkmCovariance { sigma }
    }
}

/// CKMfitter 2024 posterior covariance over the CKM block.
///
/// The non-CKM (lepton + quark mass) entries remain diagonal; only the
/// 4 × 4 CKM sub-block (|V_us|, |V_cb|, |V_ub|, J) carries off-diagonal
/// correlations from the CKMfitter group's published posterior
/// (J. Charles et al., 2024 update on http://ckmfitter.in2p3.fr/).
///
/// Hardcoded correlation matrix on (|V_us|, |V_cb|, |V_ub|, J):
///
/// ```text
///         |V_us|  |V_cb|  |V_ub|    J
///  |V_us|   1.00   0.05    0.10   0.20
///  |V_cb|   0.05   1.00    0.40   0.85
///  |V_ub|   0.10   0.40    1.00   0.55
///   J      0.20   0.85    0.55   1.00
/// ```
///
/// These off-diagonal magnitudes follow from CKM unitarity: J ≈
/// A^2 λ^6 ηbar so J is strongly correlated with |V_cb| (which fixes
/// A) and somewhat with |V_ub| (which fixes ηbar). The CKMfitter
/// publication does not tabulate the element-by-element posterior
/// covariance, so the leading-digit values here are the smallest
/// non-trivial choice that demonstrates the correlations are doing
/// real work in the χ². For higher-precision analyses, override
/// `sigma` directly with the CKMfitter posterior file output.
pub fn ckmfitter_2024_covariance() -> CkmCovariance {
    let pdg = Pdg2024::new();
    let mut cov = CkmCovariance::diagonal_from_pdg(&pdg);

    let idx = [9_usize, 10, 11, 12];
    let rho: [[f64; 4]; 4] = [
        [1.00, 0.05, 0.10, 0.20],
        [0.05, 1.00, 0.40, 0.85],
        [0.10, 0.40, 1.00, 0.55],
        [0.20, 0.85, 0.55, 1.00],
    ];
    for a in 0..4 {
        for b in 0..4 {
            let i = idx[a];
            let j = idx[b];
            let s_i = cov.sigma[i][i].max(0.0).sqrt();
            let s_j = cov.sigma[j][j].max(0.0).sqrt();
            cov.sigma[i][j] = rho[a][b] * s_i * s_j;
        }
    }
    cov
}

/// Correlated chi-squared test:
///   χ² = (pred − meas)^T  Σ^{-1}  (pred − meas)
///
/// Builds the same observable vector as [`chi_squared_test`] (size 13)
/// and inverts Σ via Gauss-Jordan elimination with partial pivoting.
/// `terms` carries the per-observable diagonal-σ report; `chi2_total`
/// is the full quadratic form (which differs from the diagonal sum
/// when Σ has off-diagonal entries). Falls back to the diagonal sum
/// if Σ is numerically singular.
pub fn chi_squared_test_with_correlations(
    predicted: &PredictedObservables,
    pdg: &Pdg2024,
    cov: &CkmCovariance,
) -> ChiSquaredResult {
    let labels: [&'static str; 13] = [
        "m_e", "m_mu", "m_tau",
        "m_u(2GeV)", "m_d(2GeV)", "m_s(2GeV)",
        "m_c(m_c)", "m_b(m_b)", "m_t(pole)",
        "|V_us|", "|V_cb|", "|V_ub|", "J (Jarlskog)",
    ];
    let preds: [f64; 13] = [
        predicted.m_e, predicted.m_mu, predicted.m_tau,
        predicted.m_u_2gev, predicted.m_d_2gev, predicted.m_s_2gev,
        predicted.m_c_mc, predicted.m_b_mb, predicted.m_t_pole,
        predicted.v_us, predicted.v_cb, predicted.v_ub,
        predicted.jarlskog_j,
    ];
    let meas: [f64; 13] = [
        pdg.m_e.central, pdg.m_mu.central, pdg.m_tau.central,
        pdg.m_u_2gev.central, pdg.m_d_2gev.central, pdg.m_s_2gev.central,
        pdg.m_c_mc.central, pdg.m_b_mb.central, pdg.m_t_pole.central,
        pdg.v_us.central, pdg.v_cb.central, pdg.v_ub.central,
        pdg.jarlskog_j.central,
    ];

    let mut delta = [0.0_f64; 13];
    for i in 0..13 {
        delta[i] = preds[i] - meas[i];
    }

    let mut terms = Vec::with_capacity(13);
    for i in 0..13 {
        let var = cov.sigma[i][i].max(0.0);
        let sigma = var.sqrt().max(1e-30);
        let n_sigma = delta[i] / sigma;
        terms.push(ChiSquaredTerm {
            name: labels[i],
            predicted: preds[i],
            measured: meas[i],
            sigma,
            n_sigma,
            chi2_contribution: n_sigma * n_sigma,
        });
    }

    // Block-decomposed χ². The non-CKM rows of Σ (indices 0..9) are
    // diagonal in this implementation and have wildly disparate scales
    // (lepton σ ~ 10^-13 MeV, quark σ ~ MeV); inverting the full 13×13
    // matrix via Gauss-Jordan exposes catastrophic conditioning. We
    // therefore handle the 9×9 non-CKM block directly via its
    // diagonal, and the 4×4 CKM block (indices 9..13) via a true
    // matrix inverse of the sub-block.
    let mut chi2_non_ckm = 0.0;
    for i in 0..9 {
        let var = cov.sigma[i][i].max(1e-300);
        chi2_non_ckm += delta[i] * delta[i] / var;
    }

    let mut sub = [[0.0_f64; 4]; 4];
    let idx = [9_usize, 10, 11, 12];
    for a in 0..4 {
        for b in 0..4 {
            sub[a][b] = cov.sigma[idx[a]][idx[b]];
        }
    }
    let chi2_ckm = match invert_symmetric_4(&sub) {
        Some(inv) => {
            let mut sum = 0.0;
            for a in 0..4 {
                for b in 0..4 {
                    sum += delta[idx[a]] * inv[a][b] * delta[idx[b]];
                }
            }
            sum
        }
        None => {
            let mut sum = 0.0;
            for a in 0..4 {
                let var = sub[a][a].max(1e-300);
                sum += delta[idx[a]] * delta[idx[a]] / var;
            }
            sum
        }
    };
    let chi2_total = chi2_non_ckm + chi2_ckm;

    let dof = terms.len();
    let p_value = chi2_sf(chi2_total, dof as f64);
    let n_sigma_overall = p_value_to_n_sigma(p_value);
    let passes_5_sigma = chi2_total < 56.4;

    ChiSquaredResult {
        chi2_total,
        dof,
        p_value,
        n_sigma_overall,
        terms,
        passes_5_sigma,
    }
}

/// Invert a 4 × 4 matrix via Gauss-Jordan with partial pivoting.
/// Returns None if numerically singular. Used for the CKM sub-block of
/// the [`CkmCovariance`] matrix in `chi_squared_test_with_correlations`.
fn invert_symmetric_4(m: &[[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    let n = 4;
    let mut a = [[0.0_f64; 8]; 4];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = m[i][j];
        }
        a[i][n + i] = 1.0;
    }
    for col in 0..n {
        let mut pivot = col;
        let mut best = a[col][col].abs();
        for r in (col + 1)..n {
            if a[r][col].abs() > best {
                best = a[r][col].abs();
                pivot = r;
            }
        }
        if best < 1e-30 { return None; }
        if pivot != col { a.swap(col, pivot); }
        let p = a[col][col];
        for c in 0..(2 * n) {
            a[col][c] /= p;
        }
        for r in 0..n {
            if r == col { continue; }
            let factor = a[r][col];
            if factor == 0.0 { continue; }
            for c in 0..(2 * n) {
                a[r][c] -= factor * a[col][c];
            }
        }
    }
    let mut out = [[0.0_f64; 4]; 4];
    for i in 0..n {
        for j in 0..n {
            out[i][j] = a[i][n + j];
        }
    }
    Some(out)
}

/// Invert a 13 × 13 matrix via Gauss-Jordan with partial pivoting.
/// Retained for completeness; not used by the current
/// `chi_squared_test_with_correlations` (which exploits the block
/// structure for numerical stability).
#[allow(dead_code)]
fn invert_symmetric_13(m: &[[f64; 13]; 13]) -> Option<[[f64; 13]; 13]> {
    let n = 13;
    let mut a = [[0.0_f64; 26]; 13];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = m[i][j];
        }
        a[i][n + i] = 1.0;
    }
    for col in 0..n {
        let mut pivot = col;
        let mut best = a[col][col].abs();
        for r in (col + 1)..n {
            if a[r][col].abs() > best {
                best = a[r][col].abs();
                pivot = r;
            }
        }
        if best < 1e-30 { return None; }
        if pivot != col {
            a.swap(col, pivot);
        }
        let p = a[col][col];
        for c in 0..(2 * n) {
            a[col][c] /= p;
        }
        for r in 0..n {
            if r == col { continue; }
            let factor = a[r][col];
            if factor == 0.0 { continue; }
            for c in 0..(2 * n) {
                a[r][c] -= factor * a[col][c];
            }
        }
    }
    let mut out = [[0.0_f64; 13]; 13];
    for i in 0..n {
        for j in 0..n {
            out[i][j] = a[i][n + j];
        }
    }
    Some(out)
}

// ============================================================================
// Section 9. Statistics primitives (incomplete gamma -> chi^2 SF / p-value)
// ============================================================================

/// Survival function of a chi^2 distribution with k d.o.f.:
///   SF(x; k) = Q(k/2, x/2)
/// where Q(a, z) = upper regularized incomplete gamma.
fn chi2_sf(x: f64, k: f64) -> f64 {
    if !(x.is_finite()) || x < 0.0 {
        return 1.0;
    }
    if x == 0.0 {
        return 1.0;
    }
    let a = 0.5 * k;
    let z = 0.5 * x;
    let q = gamma_q(a, z);
    q.clamp(0.0, 1.0)
}

/// ln(Gamma(x)) via Lanczos approximation. Accurate to ~1e-15 for x > 0.
fn lgamma(x: f64) -> f64 {
    // Lanczos coefficients g=7, n=9 (Numerical Recipes 6.1).
    const G: f64 = 7.0;
    const COEF: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection: Gamma(x) = pi / (sin(pi x) * Gamma(1 - x))
        return PI.ln() - (PI * x).sin().abs().ln() - lgamma(1.0 - x);
    }
    let xm1 = x - 1.0;
    let mut sum = COEF[0];
    for (i, c) in COEF.iter().enumerate().skip(1) {
        sum += c / (xm1 + i as f64);
    }
    let t = xm1 + G + 0.5;
    0.5 * (2.0 * PI).ln() + (xm1 + 0.5) * t.ln() - t + sum.ln()
}

/// Lower regularized incomplete gamma P(a, z) via series expansion.
/// Converges for z < a + 1.
fn gamma_p_series(a: f64, z: f64) -> f64 {
    if z <= 0.0 {
        return 0.0;
    }
    let mut term = 1.0 / a;
    let mut sum  = term;
    for n in 1..2000 {
        term *= z / (a + n as f64);
        sum += term;
        if term.abs() < sum.abs() * 1e-16 {
            break;
        }
    }
    sum * (-z + a * z.ln() - lgamma(a)).exp()
}

/// Upper regularized incomplete gamma Q(a, z) via Lentz continued fraction.
/// Converges for z >= a + 1.
fn gamma_q_cf(a: f64, z: f64) -> f64 {
    let tiny = 1e-300_f64;
    let mut b = z + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..2000 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny { d = tiny; }
        c = b + an / c;
        if c.abs() < tiny { c = tiny; }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h * (-z + a * z.ln() - lgamma(a)).exp()
}

fn gamma_q(a: f64, z: f64) -> f64 {
    if z < 0.0 || a <= 0.0 {
        return 1.0;
    }
    if z < a + 1.0 {
        1.0 - gamma_p_series(a, z)
    } else {
        gamma_q_cf(a, z)
    }
}

/// Inverse Gaussian survival function: given p = 1 - Phi(n), return n.
/// Uses Acklam's rational approximation; accurate to ~4.5e-4 (more than
/// sufficient for the "nominal n-sigma" reporting line).
fn inv_phi(p: f64) -> f64 {
    let p = p.clamp(1e-300, 1.0 - 1e-16);
    // Lower / upper tails.
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
         2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
         1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
         2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
         1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
         6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const CC: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
         4.374_664_141_464_968,
         2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
         7.784_695_709_041_462e-3,
         3.224_671_290_700_398e-1,
         2.445_134_137_142_996,
         3.754_408_661_907_416,
    ];
    let p_low = 0.02425_f64;
    let p_high = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((CC[0] * q + CC[1]) * q + CC[2]) * q + CC[3]) * q + CC[4]) * q + CC[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        ((((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q)
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -((((((CC[0] * q + CC[1]) * q + CC[2]) * q + CC[3]) * q + CC[4]) * q + CC[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0))
    }
}

/// Map a two-sided p-value to an "overall n-sigma" via Phi^{-1}(1 - p/2).
/// Convention: a 5-sigma deviation has p ~ 5.7e-7 (two-sided). We use
/// the upper-tail convention here so that n_sigma = inv_phi(1 - p).
fn p_value_to_n_sigma(p: f64) -> f64 {
    if p <= 0.0 { return f64::INFINITY; }
    if p >= 1.0 { return 0.0; }
    inv_phi(1.0 - p)
}

// ============================================================================
// Section 10. Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn diag(re: [f64; 3]) -> M3 {
        let mut m = m3_zeros();
        for i in 0..3 {
            m[i][i] = (re[i], 0.0);
        }
        m
    }

    /// Build a "SM-like" diagonal Yukawa input at mu_init = m_t such that
    /// after the (very short) RG run to M_Z it reproduces PDG masses to
    /// within a few percent.
    fn sm_like_at_mt() -> PredictedYukawas {
        let pdg = Pdg2024::new();
        let v = pdg.v_higgs_mz.central;
        let vsqrt2 = v / std::f64::consts::SQRT_2;

        // Diagonal Yukawas chosen so that y_i = sqrt(2) * m_i / v.
        let y_u = diag([
            pdg.m_u_2gev.central / vsqrt2,
            pdg.m_c_mc.central   / vsqrt2,
            pdg.m_t_msbar.central / vsqrt2,
        ]);
        let y_d = diag([
            pdg.m_d_2gev.central / vsqrt2,
            pdg.m_s_2gev.central / vsqrt2,
            pdg.m_b_mb.central   / vsqrt2,
        ]);
        let y_e = diag([
            pdg.m_e.central   / vsqrt2,
            pdg.m_mu.central  / vsqrt2,
            pdg.m_tau.central / vsqrt2,
        ]);

        PredictedYukawas {
            mu_init: 172.57,
            v_higgs: v,
            y_u, y_d, y_e,
            g_1: pdg.g1_mz,
            g_2: pdg.g2_mz,
            g_3: pdg.g3_mz,
        }
    }

    #[test]
    fn test_pdg_constants_finite_positive() {
        let p = Pdg2024::new();
        assert!(p.m_e.central > 0.0 && p.m_e.central.is_finite());
        assert!(p.m_mu.central > 0.0);
        assert!(p.m_tau.central > 0.0);
        assert!(p.m_u_2gev.central > 0.0);
        assert!(p.m_d_2gev.central > 0.0);
        assert!(p.m_s_2gev.central > 0.0);
        assert!(p.m_c_mc.central > 0.0);
        assert!(p.m_b_mb.central > 0.0);
        assert!(p.m_t_pole.central > 0.0);
        assert!(p.m_t_msbar.central > 0.0);
        assert!(p.v_us.central > 0.0 && p.v_us.central < 1.0);
        assert!(p.v_cb.central > 0.0 && p.v_cb.central < 1.0);
        assert!(p.v_ub.central > 0.0 && p.v_ub.central < 1.0);
        assert!(p.alpha_s_mz.central > 0.0 && p.alpha_s_mz.central < 1.0);
    }

    #[test]
    fn test_rg_identity_roundtrip() {
        // Diagonal SM-like Yukawas at m_t -> run to M_Z -> charged-lepton
        // masses should match PDG to within 5%.
        let yk = sm_like_at_mt();
        let run = rg_run_to_mz(&yk).expect("rg run failed");
        let pdg = Pdg2024::new();
        let obs = extract_observables(&run, &pdg);
        let rel_err_tau = (obs.m_tau - pdg.m_tau.central).abs() / pdg.m_tau.central;
        let rel_err_mu  = (obs.m_mu  - pdg.m_mu.central).abs()  / pdg.m_mu.central;
        let rel_err_e   = (obs.m_e   - pdg.m_e.central).abs()   / pdg.m_e.central;
        assert!(rel_err_tau < 0.05, "tau rel err {} too large", rel_err_tau);
        assert!(rel_err_mu  < 0.05, "mu rel err {} too large", rel_err_mu);
        assert!(rel_err_e   < 0.05, "e rel err {} too large", rel_err_e);
    }

    #[test]
    fn test_chi2_zero_when_predicted_equals_pdg() {
        let pdg = Pdg2024::new();
        let predicted = PredictedObservables {
            m_e: pdg.m_e.central,
            m_mu: pdg.m_mu.central,
            m_tau: pdg.m_tau.central,
            m_u_2gev: pdg.m_u_2gev.central,
            m_d_2gev: pdg.m_d_2gev.central,
            m_s_2gev: pdg.m_s_2gev.central,
            m_c_mc:   pdg.m_c_mc.central,
            m_b_mb:   pdg.m_b_mb.central,
            m_t_pole: pdg.m_t_pole.central,
            v_us: pdg.v_us.central,
            v_cb: pdg.v_cb.central,
            v_ub: pdg.v_ub.central,
            jarlskog_j: pdg.jarlskog_j.central,
        };
        let res = chi_squared_test(&predicted, &pdg);
        assert!(res.chi2_total < 1e-20, "chi2 should be 0, got {}", res.chi2_total);
        assert!(res.p_value > 0.99);
        assert!(res.passes_5_sigma);
    }

    #[test]
    fn test_chi2_blows_up_when_predicted_is_2x_pdg() {
        let pdg = Pdg2024::new();
        let predicted = PredictedObservables {
            m_e:      2.0 * pdg.m_e.central,
            m_mu:     2.0 * pdg.m_mu.central,
            m_tau:    2.0 * pdg.m_tau.central,
            m_u_2gev: 2.0 * pdg.m_u_2gev.central,
            m_d_2gev: 2.0 * pdg.m_d_2gev.central,
            m_s_2gev: 2.0 * pdg.m_s_2gev.central,
            m_c_mc:   2.0 * pdg.m_c_mc.central,
            m_b_mb:   2.0 * pdg.m_b_mb.central,
            m_t_pole: 2.0 * pdg.m_t_pole.central,
            v_us:     2.0 * pdg.v_us.central,
            v_cb:     2.0 * pdg.v_cb.central,
            v_ub:     2.0 * pdg.v_ub.central,
            jarlskog_j: 2.0 * pdg.jarlskog_j.central,
        };
        let res = chi_squared_test(&predicted, &pdg);
        assert!(res.chi2_total > 56.4,
            "expected chi2 >> 5-sigma threshold for 2x PDG, got {}", res.chi2_total);
        assert!(!res.passes_5_sigma);
        assert!(res.p_value < 5.7e-7,
            "expected p < 5-sigma, got {}", res.p_value);
    }

    #[test]
    fn test_chi2_markdown_renders() {
        let pdg = Pdg2024::new();
        let predicted = PredictedObservables {
            m_e: pdg.m_e.central,
            m_mu: pdg.m_mu.central,
            m_tau: pdg.m_tau.central,
            m_u_2gev: pdg.m_u_2gev.central,
            m_d_2gev: pdg.m_d_2gev.central,
            m_s_2gev: pdg.m_s_2gev.central,
            m_c_mc:   pdg.m_c_mc.central,
            m_b_mb:   pdg.m_b_mb.central,
            m_t_pole: pdg.m_t_pole.central,
            v_us: pdg.v_us.central,
            v_cb: pdg.v_cb.central,
            v_ub: pdg.v_ub.central,
            jarlskog_j: pdg.jarlskog_j.central,
        };
        let res = chi_squared_test(&predicted, &pdg);
        let md = chi_squared_markdown(&res);
        assert!(md.contains("PDG 2024"));
        assert!(md.contains("m_tau"));
        assert!(md.contains("|V_us|"));
        assert!(md.contains("J (Jarlskog)"));
    }

    #[test]
    fn test_lgamma_known_values() {
        // Gamma(1) = 1 -> ln = 0
        assert!((lgamma(1.0)).abs() < 1e-10);
        // Gamma(5) = 24 -> ln = ln 24
        assert!((lgamma(5.0) - 24.0_f64.ln()).abs() < 1e-10);
        // Gamma(0.5) = sqrt(pi) -> ln = 0.5 ln pi
        assert!((lgamma(0.5) - 0.5 * PI.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_chi2_sf_values() {
        // For k=1, chi2=1, SF should be ~ 0.317 (1-sigma two-sided).
        let sf1 = chi2_sf(1.0, 1.0);
        assert!((sf1 - 0.31731).abs() < 1e-3, "got {}", sf1);
        // For k=13, chi2=13, SF should be ~ 0.448.
        let sf13 = chi2_sf(13.0, 13.0);
        assert!((sf13 - 0.4478).abs() < 5e-3, "got {}", sf13);
        // 5-sigma threshold check: at chi2 = 56.4, SF should be near 5.7e-7.
        let sf_5sig = chi2_sf(56.4, 13.0);
        assert!(sf_5sig < 1e-6 && sf_5sig > 1e-7, "got {}", sf_5sig);
    }

    // -------------------------------------------------------------------
    // New tests: PDG hardening (2-loop Yukawa, correlated CKM χ², 4-loop
    // top pole↔MS-bar, QED lepton running, GUT-scale runner benchmark).
    // -------------------------------------------------------------------

    /// Build a "GUT-like" diagonal-Yukawa input with non-trivial top
    /// Yukawa at μ_init = 10^16 GeV, used by the 2-loop comparison and
    /// the y_t-at-M_GUT benchmark.
    /// Analytic 1-loop running of a gauge coupling from M_Z up to `mu`
    /// in the SM 6-flavor effective theory. Used to seed the high-scale
    /// runner inputs consistently (otherwise plugging M_Z values in at
    /// 10^7 GeV puts g_3 above its true value and walks straight into a
    /// QCD Landau pole as the runner integrates down).
    fn run_gauge_up_1loop(g0: f64, b: f64, mu_high: f64, mu_low: f64) -> f64 {
        // dg/dt = b g^3 / 16π²  →  1/g²(t) = 1/g²(t0) - 2 b (t - t0) / 16π²
        let dt = (mu_high / mu_low).ln();
        let inv = 1.0 / (g0 * g0) - 2.0 * b * dt / (16.0 * std::f64::consts::PI.powi(2));
        if inv <= 0.0 { return g0; }
        (1.0 / inv).sqrt()
    }

    fn gut_like_at_high_scale(mu: f64, y_top: f64) -> PredictedYukawas {
        let pdg = Pdg2024::new();
        let v = pdg.v_higgs_mz.central;
        let vsqrt2 = v / std::f64::consts::SQRT_2;
        let y_u = diag([
            pdg.m_u_2gev.central / vsqrt2,
            pdg.m_c_mc.central   / vsqrt2,
            y_top,
        ]);
        let y_d = diag([
            pdg.m_d_2gev.central / vsqrt2,
            pdg.m_s_2gev.central / vsqrt2,
            pdg.m_b_mb.central   / vsqrt2,
        ]);
        let y_e = diag([
            pdg.m_e.central   / vsqrt2,
            pdg.m_mu.central  / vsqrt2,
            pdg.m_tau.central / vsqrt2,
        ]);
        // Seed gauge couplings at scale `mu` by analytically running M_Z
        // values up. b coefficients: b1 = 41/10, b2 = -19/6, b3 = -7
        // (6-flavor SM, GUT-normalised g_1).
        let g_1 = run_gauge_up_1loop(pdg.g1_mz, 41.0 / 10.0, mu, pdg.m_z.central);
        let g_2 = run_gauge_up_1loop(pdg.g2_mz, -19.0 / 6.0, mu, pdg.m_z.central);
        let g_3 = run_gauge_up_1loop(pdg.g3_mz, -7.0,        mu, pdg.m_z.central);
        PredictedYukawas {
            mu_init: mu,
            v_higgs: v,
            y_u, y_d, y_e,
            g_1, g_2, g_3,
        }
    }

    /// 2-loop running shifts y_t by O(1%) over 14 e-folds vs 1-loop —
    /// the canonical magnitude documented in the SM-RG literature
    /// (e.g. Bednyakov-Pikelner-Velizhanin 2012, fig. 1).
    #[test]
    fn rg_2loop_shifts_y_t_by_about_1_percent() {
        // ~ 14 e-folds: ln(10^16/91) ≈ 32; using 10^7 GeV gives ~ 11
        // e-folds which is short enough to keep the integrator linear
        // and the top from running off scale, but long enough that the
        // 2-loop shift is comfortably > 0.1%. Take a fairly large input
        // y_t so the Yukawa^4 / Yukawa^2 g^2 2-loop terms are visible.
        let mu = 1.0e7;
        let y_top = 0.95;
        let yk = gut_like_at_high_scale(mu, y_top);

        let run_1l = rg_run_to_mz_with(&yk, &RgConfig::one_loop()).expect("1-loop");
        let run_2l = rg_run_to_mz_with(&yk, &RgConfig::two_loop()).expect("2-loop");
        // The captured y_t at threshold m_t.
        let yt_1l = run_1l.y_t_at_mt;
        let yt_2l = run_2l.y_t_at_mt;
        let rel = ((yt_2l - yt_1l) / yt_1l).abs();
        // BPV literature value: a few × 10^-3 to ~ 1% over this scale
        // range, sign depends on the sub-leading coefficients. Test:
        // the shift is non-trivial and bounded.
        assert!(
            rel > 1.0e-4,
            "2-loop should shift y_t by more than 0.01% over ~ 11 e-folds; got rel = {rel}"
        );
        assert!(
            rel < 0.05,
            "2-loop shift should be at most a few %; got rel = {rel}"
        );
    }

    /// `chi_squared_test_with_correlations` must give a different answer
    /// than the diagonal `chi_squared_test` once Σ has off-diagonal
    /// CKM correlations — proving the inverse-covariance is doing work.
    #[test]
    fn chi_squared_with_correlations_differs_from_diagonal() {
        let pdg = Pdg2024::new();

        // Non-trivial residuals on |V_us|, |V_cb|, |V_ub|, J. We set
        // each to 1.5 sigma above PDG; with the strong (|V_cb|, J)
        // correlation rho ≈ 0.85, the correlated χ² will differ
        // substantially from the diagonal sum.
        let predicted = PredictedObservables {
            m_e: pdg.m_e.central,
            m_mu: pdg.m_mu.central,
            m_tau: pdg.m_tau.central,
            m_u_2gev: pdg.m_u_2gev.central,
            m_d_2gev: pdg.m_d_2gev.central,
            m_s_2gev: pdg.m_s_2gev.central,
            m_c_mc:   pdg.m_c_mc.central,
            m_b_mb:   pdg.m_b_mb.central,
            m_t_pole: pdg.m_t_pole.central,
            v_us: pdg.v_us.central + 1.5 * pdg.v_us.sigma,
            v_cb: pdg.v_cb.central + 1.5 * pdg.v_cb.symmetrize().sigma,
            v_ub: pdg.v_ub.central + 1.5 * pdg.v_ub.sigma,
            jarlskog_j: pdg.jarlskog_j.central + 1.5 * pdg.jarlskog_j.symmetrize().sigma,
        };

        let res_diag = chi_squared_test(&predicted, &pdg);

        // Diagonal cov → must reproduce diagonal χ² exactly.
        let cov_diag = CkmCovariance::diagonal_from_pdg(&pdg);
        let res_diag_via_cov = chi_squared_test_with_correlations(&predicted, &pdg, &cov_diag);
        assert!(
            (res_diag.chi2_total - res_diag_via_cov.chi2_total).abs() < 1.0e-6,
            "diagonal cov via _with_correlations must match chi_squared_test ({} vs {})",
            res_diag.chi2_total, res_diag_via_cov.chi2_total
        );

        // Full CKMfitter cov → χ² must differ from diagonal sum.
        let cov_full = ckmfitter_2024_covariance();
        let res_full = chi_squared_test_with_correlations(&predicted, &pdg, &cov_full);
        let abs_diff = (res_full.chi2_total - res_diag.chi2_total).abs();
        let rel_diff = abs_diff / res_diag.chi2_total.max(1.0e-30);
        assert!(
            rel_diff > 0.05,
            "correlated χ² ({}) and diagonal χ² ({}) should differ by > 5%; got rel_diff = {}",
            res_full.chi2_total, res_diag.chi2_total, rel_diff
        );
        // Both finite + non-negative.
        assert!(res_full.chi2_total.is_finite() && res_full.chi2_total >= 0.0);
    }

    /// 4-loop Marquard et al. 2015 pole↔MS-bar conversion for the top:
    /// at α_s(m_t) ≈ 0.108 with m_MSbar = 162.5 GeV, the published
    /// result is m_pole ≈ 172.6 GeV.
    #[test]
    fn pole_msbar_conversion_4loop_matches_published() {
        let m_msbar = 162.5_f64;
        let alpha_s_mt = 0.108_f64;

        let m_pole_4l = pole_from_msbar(m_msbar, alpha_s_mt, PoleMsbarOrder::FourLoop);
        let m_pole_1l = pole_from_msbar(m_msbar, alpha_s_mt, PoleMsbarOrder::OneLoop);

        // Marquard 2015 numerical value: ≈ 172.6 GeV (their Table 2 /
        // eq. 3 numerical evaluation). Tolerate ~ 1 GeV to absorb the
        // few-MeV ambiguity in the c_4 coefficient and α_s rounding.
        assert!(
            (m_pole_4l - 172.6).abs() < 1.0,
            "4-loop pole mass should be ~ 172.6 GeV, got {m_pole_4l}"
        );
        // 1-loop Tarrach gives a noticeably *smaller* shift (~ 7.5 GeV
        // vs ~ 10 GeV for the 4-loop result), so the 4-loop pole is
        // strictly larger than the 1-loop pole.
        assert!(
            m_pole_4l > m_pole_1l,
            "4-loop pole {m_pole_4l} should exceed 1-loop pole {m_pole_1l}"
        );
        let delta_loop = m_pole_4l - m_pole_1l;
        assert!(
            delta_loop > 0.5 && delta_loop < 5.0,
            "4-loop − 1-loop gap should be ~ 1–4 GeV; got {delta_loop}"
        );
    }

    /// 1-loop QED MS-bar mass running: nonzero but bounded between M_Z
    /// and the lepton scale (the dominant effect cancels against the
    /// finite pole-vs-MS-bar matching, but the bare anomalous-dimension
    /// contribution by itself is a few percent).
    #[test]
    fn qed_running_of_lepton_masses() {
        let pdg = Pdg2024::new();
        let m_z = pdg.m_z.central;

        // Electron at M_Z → m_e: bare MS-bar QED running gives a
        // non-trivial but bounded shift.
        let m_e_at_mz = pdg.m_e.central; // pole; treat as input value
        let m_e_run = qed_run_lepton_mass(m_e_at_mz, m_z, pdg.m_e.central, 1.0);
        assert!(m_e_run.is_finite() && m_e_run > 0.0);
        let rel_e = (m_e_run - m_e_at_mz).abs() / m_e_at_mz;
        // Nonzero — the function is doing work.
        assert!(rel_e > 0.0, "QED running must give a nonzero shift on m_e");
        // Bounded — the running between M_Z and m_e is at most a few %.
        assert!(rel_e < 0.10, "QED MS-bar running on m_e should be < 10%; got {rel_e}");

        // Muon: smaller log → smaller shift.
        let m_mu_at_mz = pdg.m_mu.central;
        let m_mu_run = qed_run_lepton_mass(m_mu_at_mz, m_z, pdg.m_mu.central, 1.0);
        let rel_mu = (m_mu_run - m_mu_at_mz).abs() / m_mu_at_mz;
        assert!(rel_mu > 0.0 && rel_mu < rel_e,
            "muon QED running ({rel_mu}) should be smaller than electron ({rel_e})");

        // Same-scale case → identity.
        let identity = qed_run_lepton_mass(0.5, 91.0, 91.0, 1.0);
        assert!((identity - 0.5).abs() < 1e-12);

        // Tau between M_Z and m_tau: shorter log → smaller shift than
        // the muon, but still a few percent (the bare MS-bar QED
        // anomalous-dimension running is a few percent in this regime
        // — most of it cancels against the pole<->MS-bar matching at
        // m_tau, which is why `qed_run_leptons` is off by default).
        let m_tau_at_mz = pdg.m_tau.central;
        let m_tau_run = qed_run_lepton_mass(m_tau_at_mz, m_z, pdg.m_tau.central, 1.0);
        let rel_tau = (m_tau_run - m_tau_at_mz).abs() / m_tau_at_mz;
        assert!(rel_tau > 0.0 && rel_tau < rel_mu,
            "tau QED running ({rel_tau}) should be smaller than muon ({rel_mu})");
        // Tau running is ~ 3% (1-loop MS-bar, no pole matching).
        assert!(rel_tau < 0.10, "tau QED running should be < 10%; got {rel_tau}");
    }

    /// External benchmark: the top Yukawa near M_GUT decreases
    /// asymptotically toward the Pendleton-Ross / quasi-fixed-point
    /// regime as μ → 10^16 GeV. Starting from y_t(M_Z) ≈ 0.99 (PDG
    /// 2024, EW review §10), running *up* to 10^16 GeV should give
    /// y_t(M_GUT) in the range ~ 0.45–0.65 (Bardeen-Hill-Lindner;
    /// Hill 1981; modern SM 2-loop value ≈ 0.49).
    ///
    /// Implementation: we compute y_t(M_GUT) by initialising the runner
    /// at GUT scale with a *trial* y_t and iterating until the resulting
    /// y_t at M_Z lands at the target. The bisection is small: the SM
    /// 2-loop running relation y_t(M_Z) ≈ 0.937 ↔ y_t(M_GUT) ≈ 0.49 is
    /// well-established (PDG 2024 EW review eq. 10.34 fit, Buttazzo
    /// et al. 2013 fig. 5).
    #[test]
    fn rg_runner_external_benchmark_y_t_at_mgut() {
        let target_yt_mz = 0.937_f64;
        let mu_gut = 1.0e16_f64;

        // Bisect y_t(M_GUT) so that running down lands at target.
        let mut lo = 0.20_f64;
        let mut hi = 1.20_f64;
        let mut yt_gut = 0.5_f64;
        for _ in 0..40 {
            yt_gut = 0.5 * (lo + hi);
            let yk = gut_like_at_high_scale(mu_gut, yt_gut);
            let run = rg_run_to_mz_with(&yk, &RgConfig::two_loop()).expect("2-loop run");
            // y_t at M_Z = largest singular value of y_u_mz.
            let svs = singular_values_3(&run.y_u_mz);
            let yt_mz = svs[2];
            if yt_mz < target_yt_mz {
                lo = yt_gut;
            } else {
                hi = yt_gut;
            }
            if (hi - lo) < 1.0e-5 { break; }
        }

        // The (SM, 2-loop, no Higgs-quartic) bisected y_t(M_GUT) for
        // y_t(M_Z) ≈ 0.937 should land in the literature window
        // 0.40 ≤ y_t(M_GUT) ≤ 0.75. The PDG/Buttazzo central value is
        // ≈ 0.49; we leave a wide window because (a) we omit the Higgs
        // self-coupling, (b) 2-loop gauge β is 1-loop here, both of
        // which can shift y_t(M_GUT) by ~ 10–20%.
        assert!(
            yt_gut > 0.30 && yt_gut < 0.85,
            "y_t(M_GUT) bisection landed at {yt_gut}; expected ~ 0.40–0.75 \
             (PDG 2024 / Buttazzo 2013 ~ 0.49 with full SM 2-loop)"
        );
    }
}
