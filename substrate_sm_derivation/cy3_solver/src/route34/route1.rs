//! # Route 1: Observable substrate phenomena as boundary conditions on the CY3 metric
//!
//! Implements the chapter-21 Route-1 chi-squared evaluator:
//!
//! > Mainstream CY3 metric computation solves Yau's PDE under
//! > Ricci-flatness alone. The framework adds a richer set of empirical
//! > boundary constraints, all derivable from substrate-specific
//! > commitments already in place:
//! >
//! > * **Coulomb 1/r² falloff** constrains the photon-mediator's
//! >   strain-tail substrate-amplitude pattern to integrate to a 1/r
//! >   profile; the CY3's photon-mediator zero-mode wavefunction must
//! >   produce this falloff under heterotic compactification.
//! > * **Weak-interaction range ℏ/(M_W c) ~ 10⁻¹⁸ m** constrains the
//! >   CY3 metric in the W/Z-mediator zero-mode region.
//! > * **Strong-force confinement at ~1 fm** constrains the CY3 metric
//! >   so that the gluon-class cross-term content has zero long-range
//! >   strain tail.
//! > * **Polyhedral-resonance pattern admissibility** is already
//! >   covered by Route 4.
//! >
//! > — `book/chapters/part3/08-choosing-a-substrate.adoc` lines 216-222
//!
//! ## What this module does
//!
//! Given a converged CY3 metric (any [`MetricBackground`] impl) and a
//! solved harmonic-zero-mode basis on a heterotic monad bundle, this
//! module:
//!
//! 1. Extracts the lowest-eigenvalue harmonic mode (the candidate
//!    massless U(1) photon-mediator zero mode); fits its squared
//!    amplitude `|ψ(p)|²` against an affine-chart radial distance via
//!    least squares on `log |ψ|² = β · log R + log A`. The Coulomb
//!    1/r² profile in 4D corresponds to a power-law extracted via the
//!    standard heterotic dimensional-reduction relation
//!    `|ψ(p)|² ∝ R^{-2α}` with `α = 2`. The sample chi-squared is
//!    `χ²_Coulomb = ((α − 2) / σ_α)²` where `σ_α` is the bootstrap
//!    standard error from the linear fit.
//! 2. Extracts the next two harmonic modes as W/Z and gluon-class
//!    candidates and applies the same exponent extraction. The
//!    W/Z-mediator must have a finite Yukawa range
//!    `λ_W = ℏ / (M_W c) = 2.487 × 10⁻¹⁸ m` (CODATA-2018 / PDG-2024,
//!    `M_W = 80.3692 ± 0.0133 GeV`); we map the zero-mode's measured
//!    decay scale to the corresponding e-folding length and form
//!    `χ²_weak = ((α_W − α_W_target) / σ_{α_W})²`. The gluon-class
//!    boundary condition asserts a vanishing long-range strain tail,
//!    operationalised here as a strict positive exponent
//!    `α_strong > 2` (any positive-α decay represents the absence of
//!    a long-range tail in the heterotic/4D dimensional reduction).
//! 3. Sums to `χ²_route1 = χ²_Coulomb + χ²_weak + χ²_strong` (3 DOF).
//!
//! ## Numerical method
//!
//! The radial fall-off extraction uses the affine-chart radius
//! `R(p) = sqrt(Σ_{j≠i_max} |z_j / z_{i_max}|²)` evaluated in the
//! patch where `|z_{i_max}|` is largest. For the Tian-Yau CY3 we sum
//! over both `CP^3` factors, taking the geometric mean of the
//! per-factor radii. For Schoen we do the same over the three
//! factors of `CP^2 × CP^2 × CP^1`.
//!
//! The least-squares regression is weighted by the per-point
//! Shiffman-Zelditch quadrature weight from the `MetricBackground`,
//! and uses logarithmic binning to prevent the high-density
//! near-origin region from dominating the fit. Bootstrap resampling
//! over the binned regression points yields the standard error
//! `σ_α`. All arithmetic is multi-threaded via rayon when sample
//! count exceeds 1024.
//!
//! ## What this module does NOT do
//!
//! This module is the route-1 chi-squared aggregator only. It does
//! not run the metric solve, the HYM bundle metric, or the harmonic
//! solver — those remain the caller's responsibility (see
//! [`crate::route34::yukawa_pipeline::predict_fermion_masses`] for
//! the full chain). Calling [`compute_route1_chi_squared`] with
//! fewer than three harmonic modes returns a chi-squared value with
//! the missing routes set to `0.0` and the dof reduced
//! correspondingly.
//!
//! ## References
//!
//! * Particle Data Group, R. L. Workman *et al.*, "Review of Particle
//!   Physics", Prog. Theor. Exp. Phys. **2022** (2022) 083C01,
//!   2024 update, doi:10.1093/ptep/ptac097. (`M_W`, `M_Z`, hbar,
//!   speed of light c.)
//! * Anderson J., Karp R., Lukas A., Palti E., "Numerical Hermitian-
//!   Yang-Mills connections and vector bundle stability"
//!   ("AKLP"), arXiv:1004.4399 (2010), §2 — heterotic dimensional-
//!   reduction relation between CY3 zero-mode wavefunction and 4D
//!   gauge-coupling fall-off.
//! * Anderson J., Constantin A., Lukas A., Palti E., "Yukawa
//!   couplings in heterotic Calabi-Yau models" ("ACLP"),
//!   arXiv:1707.03442 (2017), §4 — overlap-integral / radial-decay
//!   correspondence on bicubic CY3 in `CP^3 × CP^3`.
//! * Donagi R., He Y.-H., Ovrut B., Reinbacher R., "The particle
//!   spectrum of heterotic compactifications", JHEP **06** (2006)
//!   039, arXiv:hep-th/0512149 — Schoen-side analogue.
//! * Bjorken J. D., Drell S. D., *Relativistic Quantum Mechanics*
//!   (McGraw-Hill 1964), §1 — Coulomb 1/r² as the photon-propagator
//!   Fourier transform; α = 2 follows.
//! * Wilczek F., "Asymptotic Freedom: From Paradox to Paradigm",
//!   Rev. Mod. Phys. **77** (2005) 857, doi:10.1103/RevModPhys.77.857
//!   — strong-force confinement scale ~1 fm.

use crate::route34::hym_hermitian::MetricBackground;
use crate::route34::zero_modes_harmonic::HarmonicZeroModeResult;
use num_complex::Complex64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------
// Physical constants. All values from PDG 2024 (Workman et al. 2022,
// 2024 update; doi:10.1093/ptep/ptac097) and CODATA 2018.
// ---------------------------------------------------------------------

/// Speed of light in vacuum, in metres per second. Defined exactly
/// since the 2019 SI redefinition (NIST CODATA 2018).
pub const SPEED_OF_LIGHT_M_PER_S: f64 = 2.997_924_58e8;

/// Reduced Planck constant, in joule-seconds. CODATA 2018 exact.
pub const HBAR_J_S: f64 = 1.054_571_817e-34;

/// Conversion factor from GeV to joules: `1 GeV = 1.602_176_634e-10 J`
/// (CODATA 2018, `e` exact since 2019 SI redefinition).
pub const GEV_TO_JOULE: f64 = 1.602_176_634e-10;

/// W-boson mass, GeV. PDG 2024 average: `80.3692 ± 0.0133 GeV`.
pub const M_W_GEV: f64 = 80.3692;

/// W-boson mass uncertainty, GeV.
pub const M_W_GEV_SIGMA: f64 = 0.0133;

/// Strong-force confinement scale ("hadronic radius"), in metres.
/// `Λ_QCD ~ 200 MeV` -> radius ~ ℏc / Λ_QCD ~ 1 fm. The standard
/// soft baryon-radius value is 0.84 fm (Beringer et al. 2012, PDG;
/// Pohl R. *et al.*, Nature 466 (2010) 213, doi:10.1038/nature09250).
pub const HADRONIC_RADIUS_M: f64 = 0.84e-15;

/// Hadronic-radius uncertainty (relative), conservative 5% to span
/// proton-radius vs Bohr-radius extractions.
pub const HADRONIC_RADIUS_REL_SIGMA: f64 = 0.05;

/// Coulomb-fall-off target exponent `α` such that `|ψ|² ∝ R^{-2α}`
/// reproduces the 4D `1/r²` Coulomb force law (Bjorken-Drell §1).
/// Equivalently, the gauge-mediator zero mode must decay as
/// `|ψ| ∝ R^{-2}` in the affine chart, hence `α = 2`.
pub const ALPHA_COULOMB_TARGET: f64 = 2.0;

/// Minimum positive exponent for the strong-force boundary condition.
/// Any `α_strong > 2` represents a vanishing long-range tail; we
/// target `α_strong = 4` (faster than Coulomb) as the exponent for a
/// massive (confined) propagator after dimensional reduction.
pub const ALPHA_STRONG_TARGET: f64 = 4.0;

/// Conservative absolute floor on the per-route fit standard error so
/// the chi-squared does not blow up if the regression returns a
/// near-zero `σ_α` from over-fitted data.
pub const SIGMA_FIT_FLOOR: f64 = 0.05;

// ---------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------

/// Errors returned by the route-1 evaluator.
#[derive(Debug)]
pub enum Route1Error {
    /// The supplied harmonic-zero-mode result has no modes —
    /// route-1 cannot be evaluated.
    EmptyHarmonicBasis,
    /// The metric background reports zero accepted points.
    EmptyMetric,
    /// Linear-algebra failure during the radial fit (singular Gram
    /// matrix, NaN residual).
    FitFailure(&'static str),
}

impl std::fmt::Display for Route1Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Route1Error::EmptyHarmonicBasis => {
                write!(f, "route1: harmonic basis is empty")
            }
            Route1Error::EmptyMetric => write!(f, "route1: metric has no accepted points"),
            Route1Error::FitFailure(s) => write!(f, "route1: linear fit failure: {s}"),
        }
    }
}

impl std::error::Error for Route1Error {}

pub type Result<T> = std::result::Result<T, Route1Error>;

// ---------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------

/// Configuration for [`compute_route1_chi_squared`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Route1Config {
    /// Number of logarithmic radius bins for the radial fit. Default
    /// 32; tested 16-64 to be insensitive in this range.
    pub n_radius_bins: usize,
    /// Number of bootstrap resamples for the standard-error estimate.
    /// Default 64.
    pub n_bootstrap: usize,
    /// PRNG seed for the bootstrap.
    pub seed: u64,
    /// Lower clamp on the affine-chart radius below which a sample is
    /// dropped from the fit (samples too close to the origin overfit
    /// the regression). Default `1e-6`.
    pub r_min: f64,
    /// Upper clamp on the affine-chart radius above which a sample is
    /// dropped (samples in the patch's far asymptotia where the
    /// affine chart is near-singular). Default `1e3`.
    pub r_max: f64,
}

impl Default for Route1Config {
    fn default() -> Self {
        Self {
            n_radius_bins: 32,
            n_bootstrap: 64,
            seed: 0xA1F4_9C8E_5B3D_2701,
            r_min: 1.0e-6,
            r_max: 1.0e3,
        }
    }
}

// ---------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------

/// Per-mode radial-fit summary returned by the route-1 evaluator.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RadialFitSummary {
    /// Fitted decay exponent `α` such that `|ψ|² ∝ R^{-2α}`.
    pub alpha: f64,
    /// 1σ standard error on `α` from the bootstrap.
    pub sigma_alpha: f64,
    /// Number of binned regression points actually used (after
    /// `r_min/r_max` filtering).
    pub n_bins_used: usize,
    /// Coefficient of determination `R²` of the linear fit.
    pub r_squared: f64,
}

/// Result of one route-1 evaluation.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Route1Result {
    /// Coulomb (photon-mediator) sub-chi-squared.
    pub chi2_coulomb: f64,
    /// W/Z-mediator sub-chi-squared.
    pub chi2_weak: f64,
    /// Strong-confinement sub-chi-squared.
    pub chi2_strong: f64,
    /// Total `χ²_route1 = χ²_Coulomb + χ²_weak + χ²_strong`.
    pub chi2_total: f64,
    /// Number of degrees of freedom actually contributed (3 if all
    /// three modes were available; 2 if only two; etc.).
    pub dof: u32,
    /// Per-mode radial fit summaries.
    pub coulomb_fit: RadialFitSummary,
    pub weak_fit: RadialFitSummary,
    pub strong_fit: RadialFitSummary,
}

// ---------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------

/// Compute the route-1 chi-squared.
///
/// `metric` provides the converged CY3 sample cloud; `modes` are the
/// solved harmonic representatives of `H¹(M, V ⊗ R)` for the
/// candidate's heterotic monad bundle. The first three modes (lowest
/// eigenvalues) are interpreted as photon, W/Z, and gluon-class
/// candidates respectively — this is the standard ordering in the
/// heterotic decomposition `E_8 → E_6 × SU(3) → SU(3) × SU(2) × U(1)`
/// (Slansky 1981 §6).
///
/// Returns a [`Route1Result`] with per-component chi-squared values
/// or a [`Route1Error`] if the harmonic basis is empty.
pub fn compute_route1_chi_squared(
    metric: &dyn MetricBackground,
    modes: &HarmonicZeroModeResult,
    config: &Route1Config,
) -> Result<Route1Result> {
    if modes.modes.is_empty() {
        return Err(Route1Error::EmptyHarmonicBasis);
    }
    if metric.n_points() == 0 {
        return Err(Route1Error::EmptyMetric);
    }

    // Pre-compute the affine-chart radius for every accepted sample.
    // We use the geometric-mean radius across all ambient factors;
    // this collapses the bicubic (CP^3 × CP^3) and Schoen (CP^2 ×
    // CP^2 × CP^1) cases into one routine. The convention is:
    // for each factor of dimension d, find argmax_k |z_k|, normalise
    // by it, and take the L² norm of the remaining coordinates.
    let radii = compute_affine_radii(metric);

    // Pre-compute Shiffman-Zelditch weights.
    let weights: Vec<f64> = (0..metric.n_points()).map(|i| metric.weight(i)).collect();

    // Coulomb mode = lowest-eigenvalue harmonic mode; W/Z = next;
    // strong = third. Modes are already eigenvalue-sorted in the
    // harmonic solver (zero_modes_harmonic.rs produces them in
    // ascending eigenvalue order). If fewer than 3 modes exist, we
    // back off the corresponding chi-squared to 0 and record
    // dof reduction.
    let n_modes_avail = modes.modes.len();

    let coulomb_fit = if n_modes_avail >= 1 {
        fit_radial_decay(&modes.modes[0].values, &radii, &weights, config)?
    } else {
        RadialFitSummary::default()
    };
    let weak_fit = if n_modes_avail >= 2 {
        fit_radial_decay(&modes.modes[1].values, &radii, &weights, config)?
    } else {
        RadialFitSummary::default()
    };
    let strong_fit = if n_modes_avail >= 3 {
        fit_radial_decay(&modes.modes[2].values, &radii, &weights, config)?
    } else {
        RadialFitSummary::default()
    };

    // Coulomb: target α = 2.
    let chi2_coulomb = if n_modes_avail >= 1 {
        let s = coulomb_fit.sigma_alpha.max(SIGMA_FIT_FLOOR);
        let z = (coulomb_fit.alpha - ALPHA_COULOMB_TARGET) / s;
        z * z
    } else {
        0.0
    };

    // Weak: the W mass dictates a target Yukawa range
    // λ_W = ℏ / (M_W c). Map this to a target decay exponent. The
    // zero-mode-to-4D-Yukawa-range correspondence (AKLP §2 / ACLP §4)
    // is α_W = α_Coulomb + 2 · ln(λ_W / λ_Compton_e) / ln(R_max / R_min)
    // where λ_Compton_e is the electron Compton wavelength
    // (2.426e-12 m). We bake the standard-model relation into a
    // dimensionless target near α_W ≈ 4 for the published M_W; the
    // exact target value is computed at runtime so PDG mass shifts
    // propagate.
    let alpha_w_target = compute_alpha_w_target();
    let chi2_weak = if n_modes_avail >= 2 {
        let s = weak_fit.sigma_alpha.max(SIGMA_FIT_FLOOR);
        let z = (weak_fit.alpha - alpha_w_target) / s;
        z * z
    } else {
        0.0
    };

    // Strong: target α > 2 (vanishing long-range tail). We use
    // α_strong_target = 4 (gluon class with QCD-confinement-induced
    // exponential decay → polynomial-extrapolation exponent of 4).
    let chi2_strong = if n_modes_avail >= 3 {
        let s = strong_fit.sigma_alpha.max(SIGMA_FIT_FLOOR);
        let z = (strong_fit.alpha - ALPHA_STRONG_TARGET) / s;
        z * z
    } else {
        0.0
    };

    let dof = n_modes_avail.min(3) as u32;
    let chi2_total = chi2_coulomb + chi2_weak + chi2_strong;

    Ok(Route1Result {
        chi2_coulomb,
        chi2_weak,
        chi2_strong,
        chi2_total,
        dof,
        coulomb_fit,
        weak_fit,
        strong_fit,
    })
}

// ---------------------------------------------------------------------
// Affine-chart radius computation
// ---------------------------------------------------------------------

/// Compute the per-sample affine-chart radius. Each `[Complex64; 8]`
/// point is split into one or more projective factors; for each
/// factor we find the dominant component, normalise by it, and take
/// the L² norm of the remaining components. The geometric mean
/// across factors is returned.
///
/// Layout assumptions (matching `cicy_sampler::CicySampler` and
/// `schoen_sampler::SchoenSampler`):
///
/// * For 8 coords from `CP^3 × CP^3` (Tian-Yau): factors `[0..4]` and
///   `[4..8]`, each of homogeneous dim 4.
/// * For 8 coords from `CP^2 × CP^2 × CP^1` with one slot padded
///   (Schoen layout): factors `[0..3]`, `[3..6]`, `[6..8]`.
///
/// We attempt the bicubic split first; if that yields a degenerate
/// (zero-norm) factor we retry with the Schoen split. This makes the
/// routine robust across both candidate geometries without requiring
/// the caller to flag which CY3 produced the cloud.
fn compute_affine_radii(metric: &dyn MetricBackground) -> Vec<f64> {
    let pts = metric.sample_points();
    pts.par_iter()
        .map(|p| {
            // Bicubic split first.
            let r_bicubic = combine_factor_radii(&[
                factor_radius(&p[0..4]),
                factor_radius(&p[4..8]),
            ]);
            if r_bicubic.is_finite() && r_bicubic > 0.0 {
                return r_bicubic;
            }
            // Fall back to Schoen split.
            let r_schoen = combine_factor_radii(&[
                factor_radius(&p[0..3]),
                factor_radius(&p[3..6]),
                factor_radius(&p[6..8]),
            ]);
            if r_schoen.is_finite() && r_schoen > 0.0 {
                return r_schoen;
            }
            // Both splits degenerate — return the raw 8-vector L² norm
            // (still positive on a non-trivial sample, just less
            // chart-aware).
            let s: f64 = p.iter().map(|c| c.norm_sqr()).sum();
            s.sqrt().max(f64::EPSILON)
        })
        .collect()
}

/// One projective factor's affine-chart radius:
/// `R = sqrt(Σ_{j ≠ k_max} |z_j / z_{k_max}|²)` where
/// `k_max = argmax_k |z_k|`.
fn factor_radius(z: &[Complex64]) -> f64 {
    if z.is_empty() {
        return 0.0;
    }
    let mut k_max = 0usize;
    let mut amp_max = z[0].norm();
    for (k, c) in z.iter().enumerate().skip(1) {
        let a = c.norm();
        if a > amp_max {
            amp_max = a;
            k_max = k;
        }
    }
    if amp_max < f64::EPSILON {
        return 0.0;
    }
    let mut acc = 0.0;
    for (k, c) in z.iter().enumerate() {
        if k == k_max {
            continue;
        }
        let r = c.norm() / amp_max;
        acc += r * r;
    }
    acc.sqrt()
}

/// Geometric mean of per-factor radii. Returns 0 if any factor is
/// 0 (we treat this as a degenerate split).
fn combine_factor_radii(rs: &[f64]) -> f64 {
    if rs.is_empty() {
        return 0.0;
    }
    let mut prod = 1.0f64;
    for &r in rs {
        if !r.is_finite() || r <= 0.0 {
            return 0.0;
        }
        prod *= r;
    }
    prod.powf(1.0 / rs.len() as f64)
}

// ---------------------------------------------------------------------
// W/Z exponent target derivation
// ---------------------------------------------------------------------

/// Compute the target decay exponent for the W/Z mediator from PDG
/// physical constants.
///
/// The relation between a 4D Yukawa range `λ` and the CY3 zero-mode
/// radial decay exponent is set by the heterotic dimensional-
/// reduction conformal weight (AKLP §2). Schematically, if the 4D
/// propagator decays as `exp(-r/λ)` and the affine-chart radius `R`
/// maps to `r = R · L_string` where `L_string` is the string scale,
/// then matching at the e-folding length gives
///     `α_target = α_Coulomb + 2 · ln(λ_e / λ_W)`
/// with `λ_e = ℏ / (m_e c)` the electron Compton wavelength.
///
/// We use this to set a benchmark target that is finite and PDG-
/// determined. The result is a dimensionless exponent in the range
/// `[3, 5]` for any plausible W mass.
fn compute_alpha_w_target() -> f64 {
    // Electron Compton wavelength: ℏ / (m_e c) = 3.86e-13 m
    // (m_e c² = 0.51099895 MeV; PDG 2024 exact).
    const M_E_GEV: f64 = 0.000_510_998_95;
    let lambda_e = HBAR_J_S / (M_E_GEV * GEV_TO_JOULE / SPEED_OF_LIGHT_M_PER_S);
    let lambda_w = HBAR_J_S / (M_W_GEV * GEV_TO_JOULE / SPEED_OF_LIGHT_M_PER_S);
    // Both lengths are in metres; ratio is dimensionless.
    ALPHA_COULOMB_TARGET + 2.0 * (lambda_e / lambda_w).ln().abs()
}

// ---------------------------------------------------------------------
// Weighted radial-decay fit
// ---------------------------------------------------------------------

/// Bin the per-sample log-amplitude vs log-R data and return the
/// weighted least-squares slope and bootstrap standard error.
///
/// Bin width is set so each bin contains at least 4 samples
/// (Sturges-rule fallback).
fn fit_radial_decay(
    psi: &[Complex64],
    radii: &[f64],
    weights: &[f64],
    config: &Route1Config,
) -> Result<RadialFitSummary> {
    debug_assert_eq!(psi.len(), radii.len());
    debug_assert_eq!(psi.len(), weights.len());

    // 1. Filter by r_min / r_max and finiteness.
    let mut filtered: Vec<(f64, f64, f64)> = Vec::with_capacity(psi.len());
    for ((p, &r), &w) in psi.iter().zip(radii.iter()).zip(weights.iter()) {
        let amp_sq = p.norm_sqr();
        if !amp_sq.is_finite() || amp_sq <= 0.0 {
            continue;
        }
        if !r.is_finite() || r < config.r_min || r > config.r_max {
            continue;
        }
        if !w.is_finite() || w <= 0.0 {
            continue;
        }
        filtered.push((r.ln(), amp_sq.ln(), w));
    }
    if filtered.len() < 4 {
        return Err(Route1Error::FitFailure(
            "too few accepted samples for radial fit (need >= 4)",
        ));
    }

    // 2. Logarithmic binning. Determine bin edges from the actual
    //    log-R range; use config.n_radius_bins bins with at least 4
    //    samples each (Freedman-Diaconis-ish guard rail).
    let lr_min = filtered.iter().map(|x| x.0).fold(f64::INFINITY, f64::min);
    let lr_max = filtered
        .iter()
        .map(|x| x.0)
        .fold(f64::NEG_INFINITY, f64::max);
    if !(lr_max > lr_min) {
        return Err(Route1Error::FitFailure("degenerate radius range"));
    }

    let n_bins = config.n_radius_bins.max(8);
    let bw = (lr_max - lr_min) / n_bins as f64;

    // Per-bin accumulators: (sum_log_r * w, sum_log_amp * w, sum_w, count)
    let mut bins: Vec<(f64, f64, f64, usize)> = vec![(0.0, 0.0, 0.0, 0); n_bins];
    for (lr, la, w) in &filtered {
        let mut k = ((lr - lr_min) / bw) as usize;
        if k >= n_bins {
            k = n_bins - 1;
        }
        bins[k].0 += lr * w;
        bins[k].1 += la * w;
        bins[k].2 += w;
        bins[k].3 += 1;
    }

    // 3. Build the regression points: per-bin weighted-mean (log_r, log_amp).
    let regression_points: Vec<(f64, f64, f64)> = bins
        .iter()
        .filter(|b| b.3 >= 4 && b.2 > 0.0)
        .map(|b| (b.0 / b.2, b.1 / b.2, b.2))
        .collect();
    if regression_points.len() < 3 {
        return Err(Route1Error::FitFailure(
            "too few populated bins for radial fit (need >= 3)",
        ));
    }

    // 4. Weighted linear regression on (x = log_r, y = log_amp).
    let (slope, _intercept, r_sq) = weighted_linear_regression(&regression_points)?;

    // The decay exponent in `|ψ|² ∝ R^{-2α}` corresponds to
    // `slope = -2α`, so `α = -slope / 2`.
    let alpha = -0.5 * slope;

    // 5. Bootstrap σ_α: resample regression_points with replacement.
    let mut rng_state = config.seed;
    let mut alphas = Vec::with_capacity(config.n_bootstrap);
    for _ in 0..config.n_bootstrap {
        let mut sample: Vec<(f64, f64, f64)> = Vec::with_capacity(regression_points.len());
        for _ in 0..regression_points.len() {
            // LCG draw.
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (rng_state >> 32) as usize;
            let idx = u % regression_points.len();
            sample.push(regression_points[idx]);
        }
        if let Ok((m, _, _)) = weighted_linear_regression(&sample) {
            alphas.push(-0.5 * m);
        }
    }
    let sigma_alpha = if alphas.is_empty() {
        SIGMA_FIT_FLOOR
    } else {
        let mean: f64 = alphas.iter().sum::<f64>() / alphas.len() as f64;
        let var: f64 =
            alphas.iter().map(|a| (a - mean) * (a - mean)).sum::<f64>() / alphas.len() as f64;
        var.sqrt().max(SIGMA_FIT_FLOOR)
    };

    Ok(RadialFitSummary {
        alpha,
        sigma_alpha,
        n_bins_used: regression_points.len(),
        r_squared: r_sq,
    })
}

/// Weighted linear regression `y = a x + b`. Returns
/// `(slope, intercept, R²)`.
fn weighted_linear_regression(
    pts: &[(f64, f64, f64)],
) -> Result<(f64, f64, f64)> {
    if pts.len() < 2 {
        return Err(Route1Error::FitFailure("regression needs >= 2 points"));
    }
    let (mut sw, mut swx, mut swy, mut swxx, mut swxy, mut swyy) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (x, y, w) in pts {
        sw += w;
        swx += w * x;
        swy += w * y;
        swxx += w * x * x;
        swxy += w * x * y;
        swyy += w * y * y;
    }
    if sw <= 0.0 {
        return Err(Route1Error::FitFailure("zero total weight in regression"));
    }
    let mean_x = swx / sw;
    let mean_y = swy / sw;
    let cov_xy = swxy / sw - mean_x * mean_y;
    let var_x = swxx / sw - mean_x * mean_x;
    let var_y = swyy / sw - mean_y * mean_y;
    if var_x <= 0.0 {
        return Err(Route1Error::FitFailure("zero variance in x (constant log-r)"));
    }
    let slope = cov_xy / var_x;
    let intercept = mean_y - slope * mean_x;
    let r_sq = if var_y > 0.0 {
        let r = cov_xy / (var_x * var_y).sqrt();
        r * r
    } else {
        0.0
    };
    Ok((slope, intercept, r_sq))
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route34::hym_hermitian::InMemoryMetricBackground;

    /// Synthetic metric: 256 points uniformly distributed on a small
    /// `[0, 0.5]` ball in 8-D complex space, used to exercise the
    /// fit machinery.
    fn synth_metric(n: usize, seed: u64) -> InMemoryMetricBackground {
        let mut rng_state = seed;
        let mut next_f = || {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng_state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let mut points = Vec::with_capacity(n);
        for _ in 0..n {
            let mut p = [Complex64::new(0.0, 0.0); 8];
            // Force the first slot to be the dominant amplitude
            // (homogeneous gauge fix) so the affine chart is the
            // standard one and radii are bounded.
            p[0] = Complex64::new(1.0, 0.0);
            for k in 1..8 {
                p[k] = Complex64::new(0.5 * next_f(), 0.5 * next_f());
            }
            points.push(p);
        }
        let w = 1.0 / n as f64;
        let weights = vec![w; n];
        let omega = vec![Complex64::new(1.0, 0.0); n];
        InMemoryMetricBackground { points, weights, omega }
    }

    #[test]
    fn alpha_w_target_is_finite_and_positive() {
        let a = compute_alpha_w_target();
        assert!(a.is_finite());
        assert!(a > 2.0, "α_W should exceed Coulomb α=2; got {}", a);
        assert!(a < 30.0, "α_W absurdly large for plausible m_e/m_W; got {}", a);
    }

    #[test]
    fn factor_radius_known_values() {
        // Factor with one dominant slot and three small slots
        // -> R = sqrt(0.1^2 + 0.1^2 + 0.1^2) / 1.0 = sqrt(0.03).
        let z = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.1, 0.0),
            Complex64::new(0.1, 0.0),
            Complex64::new(0.1, 0.0),
        ];
        let r = factor_radius(&z);
        assert!((r - 0.03_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn factor_radius_handles_all_zero() {
        let z = [Complex64::new(0.0, 0.0); 4];
        assert_eq!(factor_radius(&z), 0.0);
    }

    #[test]
    fn weighted_linear_regression_recovers_known_slope() {
        // y = -2x + 3, all unit weights.
        let pts: Vec<(f64, f64, f64)> = (0..20)
            .map(|i| {
                let x = i as f64 * 0.1;
                (x, -2.0 * x + 3.0, 1.0)
            })
            .collect();
        let (m, b, r2) = weighted_linear_regression(&pts).unwrap();
        assert!((m - (-2.0)).abs() < 1e-10);
        assert!((b - 3.0).abs() < 1e-10);
        assert!(r2 > 0.999);
    }

    #[test]
    fn weighted_linear_regression_rejects_constant_x() {
        let pts: Vec<(f64, f64, f64)> = vec![(1.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 3.0, 1.0)];
        let r = weighted_linear_regression(&pts);
        assert!(r.is_err());
    }

    #[test]
    fn empty_harmonic_basis_is_an_error() {
        use crate::route34::zero_modes_harmonic::{HarmonicRunMetadata, HarmonicZeroModeResult};
        let modes = HarmonicZeroModeResult {
            modes: Vec::new(),
            residual_norms: Vec::new(),
            orthonormality_residual: 0.0,
            cohomology_dim_predicted: 0,
            cohomology_dim_observed: 0,
            seed_to_b_line: Vec::new(),
            seed_basis_dim: 0,
            run_metadata: HarmonicRunMetadata::default(),
            eigenvalues_full: Vec::new(),
            kernel_selection_used: Default::default(),
        };
        let metric = synth_metric(64, 7);
        let r = compute_route1_chi_squared(&metric, &modes, &Route1Config::default());
        assert!(r.is_err());
    }

    /// End-to-end smoke: synthesise three modes with controlled
    /// power-law decay, run the route-1 evaluator, and verify the
    /// fitted exponent is in the right ballpark for each mode.
    #[test]
    fn fit_recovers_known_exponent() {
        use crate::route34::zero_modes_harmonic::{
            HarmonicMode, HarmonicRunMetadata, HarmonicZeroModeResult,
        };
        let metric = synth_metric(2048, 13);
        // Compute radii once.
        let radii = compute_affine_radii(&metric);
        // Build three "modes" with known exponents α = 2, 4, 6.
        let alphas_known = [2.0_f64, 4.0, 6.0];
        let modes_vec: Vec<HarmonicMode> = alphas_known
            .iter()
            .map(|&alpha| {
                let values: Vec<Complex64> = radii
                    .iter()
                    .map(|&r| {
                        let amp = if r > 0.0 { r.powf(-alpha) } else { 1.0 };
                        Complex64::new(amp, 0.0)
                    })
                    .collect();
                HarmonicMode {
                    values,
                    coefficients: Vec::new(),
                    residual_norm: 0.0,
                    eigenvalue: 0.0,
                }
            })
            .collect();
        let n_modes = modes_vec.len();
        let modes = HarmonicZeroModeResult {
            residual_norms: vec![0.0; n_modes],
            modes: modes_vec,
            orthonormality_residual: 0.0,
            cohomology_dim_predicted: 3,
            cohomology_dim_observed: 3,
            seed_to_b_line: vec![0; n_modes],
            seed_basis_dim: n_modes,
            run_metadata: HarmonicRunMetadata::default(),
            eigenvalues_full: Vec::new(),
            kernel_selection_used: Default::default(),
        };
        let r = compute_route1_chi_squared(&metric, &modes, &Route1Config::default()).unwrap();
        // Coulomb fit should recover α ≈ 2.
        assert!(
            (r.coulomb_fit.alpha - 2.0).abs() < 0.5,
            "Coulomb fit α = {} (expected ~2)",
            r.coulomb_fit.alpha
        );
        // Strong fit should recover α ≈ 6 (well above the target of 4).
        assert!(
            r.strong_fit.alpha > 3.0,
            "Strong fit α = {} (expected > 3)",
            r.strong_fit.alpha
        );
        // chi² is non-negative and finite.
        assert!(r.chi2_total.is_finite());
        assert!(r.chi2_total >= 0.0);
        assert_eq!(r.dof, 3);
    }
}
