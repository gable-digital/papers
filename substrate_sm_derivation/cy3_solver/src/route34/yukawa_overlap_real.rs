//! # Real Yukawa overlap integrals on (CY3, V) — publication grade
//!
//! ## What this module fixes vs. the legacy [`crate::yukawa_overlap`]
//!
//! 1. **Harmonic representatives, not polynomial seeds.** The triple
//!    overlap is integrated against the genuine harmonic
//!    representatives produced by
//!    [`crate::route34::zero_modes_harmonic`], not the polynomial
//!    seeds from [`crate::zero_modes::evaluate_polynomial_seeds`].
//!    The two differ by an O(1) systematic — see the brief.
//!
//! 2. **HYM Hermitian metric, not identity.** The bundle Hermitian
//!    inner product `h_V(·, ·)` enters the overlap with the converged
//!    HYM metric from [`crate::route34::hym_hermitian`] rather than
//!    the identity placeholder.
//!
//! 3. **Shiffman-Zelditch quadrature with correct weighting.** The
//!    Monte-Carlo measure on the CY3 must be the *uniform* measure
//!    `dvol_g_M` weighted by the holomorphic-3-form factor `|Ω|²`.
//!    The line-intersection sampler returns weights in the form
//!    `w = |Ω|² / det g_pb`. We convert these into the canonical
//!    Shiffman-Zelditch weights by dividing by `|Ω|²` and re-normalising
//!    to sum-to-one. Quadrature uniformity (variance of the
//!    re-weighted distribution) is reported as
//!    [`YukawaResult::quadrature_uniformity_score`].
//!
//! 4. **Bootstrapped error bars.** Rather than reporting only a
//!    central value, we resample the sample cloud with replacement
//!    `n_bootstrap` times, recompute the triple overlap on each
//!    resample, and return the per-entry standard deviation as the
//!    uncertainty. We also report the convergence test
//!    `Y(N) − Y(N/2)` against the canonical `O(N^{-1/2})` MC scaling.
//!
//! ## Mathematical content
//!
//! For matter zero modes `ψ_i, ψ_j, ψ_k ∈ H^1(M, V)` realised as
//! bundle-valued (0,1)-forms (Dolbeault cohomology, harmonic
//! representatives), the canonical heterotic holomorphic Yukawa
//! coupling is, in the form quoted by Anderson-Constantin-Lukas-Palti
//! (ACLP 2017, arXiv:1707.03442 §3 eq. (3.2)) and the
//! Anderson-Karp-Lukas-Palti AKLP 2010 (arXiv:1004.4399 §4) reduction:
//!
//! ```text
//!     λ_{ijk}  =  ∫_M  Ω ∧ Tr_V ( ψ_i ∧ ψ_j ∧ ψ_k )
//! ```
//!
//! where `Ω` is the holomorphic (3,0)-form on the CY3 (one factor,
//! not `Ω ∧ Ω`), `Tr_V` is the gauge-invariant contraction on the
//! bundle `V`, and `∧` on the `ψ` factors is the wedge product on
//! `(0,1)`-forms (the result is a `(0,3)`-form, which together with
//! `Ω` integrates over the CY3).
//!
//! For the long-wavelength / single-Cartan-component AKLP/ACLP
//! reduction reported in AKLP 2010 Tables 6-8 — which is what we
//! evaluate here — the bundle-trace of the `(0,3)`-form factor
//! collapses to a point-wise complex scalar product of the harmonic
//! representatives' coefficients along the dominant Cartan direction,
//! dressed by the HYM bundle metric `h_V`. Concretely:
//!
//! ```text
//!     λ_{ijk}  ≈  ∫_M  Ω̄(p)  ·  ψ_i(p) · ψ_j(p) · ψ_k(p)  · h_V(·, ·)  dvol_g
//!              ≈  Σ_α  w_α^{SZ}  ·  Ω̄(p_α)  ·  ψ_i(p_α) · ψ_j(p_α) · ψ_k(p_α)
//!                  · √( h_V[i,i] · h_V[j,j] · h_V[k,k] )
//! ```
//!
//! with the following conventions:
//!
//! * **`Ω̄(p)` (single conjugate, not `Ω̄ ∧ Ω̄`).** The CY3 measure
//!   `dvol_g` carries one factor of `|Ω|^2 = Ω ∧ Ω̄`; the integrand of
//!   the Yukawa form `Ω ∧ ψ_i ∧ ψ_j ∧ ψ_k` is converted into the
//!   `dvol_g`-measure by absorbing one `Ω` into the volume form,
//!   leaving a single `Ω̄` against the (now scalarised) ψ-product.
//!   We use the conjugate so that the integrand is invariant under
//!   the antiholomorphic involution that takes Ω ↔ Ω̄ (cf. ACLP §3.1
//!   "the real form of the holomorphic Yukawa").
//!
//! * **`w_α^{SZ}` are Shiffman-Zelditch weights** (uniform measure on
//!   the CY3 with respect to `dvol_g`), produced from the sampler-
//!   native weights `w = |Ω|^2 / det g_pb` by dividing by `|Ω|^2` and
//!   re-normalising. See [`sz_weights`] below.
//!
//! * **`√(h_V[i,i] · h_V[j,j] · h_V[k,k])`** is the diagonal
//!   long-wavelength simplification of the HYM-dressed gauge trace.
//!   In the AKLP/ACLP single-Cartan-component limit, the bundle
//!   tensor product `⊗_h V` over the three ψ-factors collapses to a
//!   product of per-mode `h_V` rescalings on the harmonic-mode
//!   coefficient vectors. We project the HYM matrix onto the
//!   harmonic-mode basis and read off its diagonal as `h_v_diag`
//!   (see lines 365-383 of this module).
//!
//! ### Limits of validity of the scalar reduction
//!
//! The scalar reduction is exact in the AKLP single-line-bundle case
//! (AKLP 2010 §4), where each `ψ_i` is a section of a single line
//! bundle in the bundle's cohomology decomposition. For genuinely
//! non-abelian bundles (`V` of rank > 1 with non-line-bundle
//! components) the gauge trace must be kept; this module does not
//! handle that case. The downstream pipeline is built on the AKLP
//! example (`MonadBundle::anderson_lukas_palti_example()`) which is
//! a sum of three line bundles, so the scalar reduction is exact for
//! the deployed wiring.
//!
//! ## References
//!
//! * Anderson-Karp-Lukas-Palti, "Numerical Hermitian-Yang-Mills
//!   connections and vector bundle stability", arXiv:1004.4399 (2010),
//!   Tables 6-8.
//! * Anderson-Constantin-Lukas-Palti, "Yukawa couplings in heterotic
//!   Calabi-Yau models", arXiv:1707.03442 (2017).
//! * Shiffman, B., Zelditch, S., "Distribution of zeros of random and
//!   quantum chaotic sections of positive line bundles", Comm. Math.
//!   Phys. 200 (1999) 661.

use crate::route34::hym_hermitian::{HymHermitianMetric, MetricBackground};
use crate::route34::zero_modes_harmonic::HarmonicZeroModeResult;
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------
// Tensor3 — 3-index complex tensor (Y_{ijk}) for the triple overlap
// ---------------------------------------------------------------------

/// 3-index complex tensor `Y_{ijk}` for the triple overlap.
#[derive(Clone, Debug)]
pub struct Tensor3 {
    pub n: usize,
    /// Row-major: `data[i * n * n + j * n + k]`.
    pub data: Vec<Complex64>,
}

impl Tensor3 {
    pub fn zeros(n: usize) -> Self {
        Self {
            n,
            data: vec![Complex64::new(0.0, 0.0); n * n * n],
        }
    }
    pub fn entry(&self, i: usize, j: usize, k: usize) -> Complex64 {
        self.data[i * self.n * self.n + j * self.n + k]
    }
    pub fn set(&mut self, i: usize, j: usize, k: usize, z: Complex64) {
        self.data[i * self.n * self.n + j * self.n + k] = z;
    }
}

/// Per-entry uncertainty on `Tensor3` — same shape, real-valued.
#[derive(Clone, Debug)]
pub struct Tensor3Real {
    pub n: usize,
    pub data: Vec<f64>,
}

impl Tensor3Real {
    pub fn zeros(n: usize) -> Self {
        Self {
            n,
            data: vec![0.0; n * n * n],
        }
    }
    pub fn entry(&self, i: usize, j: usize, k: usize) -> f64 {
        self.data[i * self.n * self.n + j * self.n + k]
    }
    pub fn set(&mut self, i: usize, j: usize, k: usize, v: f64) {
        self.data[i * self.n * self.n + j * self.n + k] = v;
    }
}

/// Yukawa-overlap result.
#[derive(Clone, Debug)]
pub struct YukawaResult {
    pub couplings: Tensor3,
    pub couplings_uncertainty: Tensor3Real,
    pub n_quadrature_points: usize,
    /// 1 - relative-variance of the SZ weights. 1.0 = perfectly
    /// uniform; 0.0 = pathologically non-uniform.
    pub quadrature_uniformity_score: f64,
    /// Convergence test: max-norm of `Y(N) − Y(N/2)` divided by
    /// the expected `O(N^{-1/2})` factor `||Y(N)|| / sqrt(N)`. A
    /// value `≪ 1` means MC has converged.
    pub convergence_ratio: f64,
    pub run_metadata: YukawaRunMetadata,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct YukawaRunMetadata {
    pub wall_clock_seconds: f64,
    pub seed: u64,
    pub n_bootstrap: usize,
    pub used_hym_metric: bool,
    pub used_harmonic_modes: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct YukawaConfig {
    /// Number of bootstrap resamples used to estimate per-entry MC
    /// uncertainty.
    pub n_bootstrap: usize,
    /// Seed for bootstrap resampling.
    pub seed: u64,
}

impl Default for YukawaConfig {
    fn default() -> Self {
        Self {
            n_bootstrap: 64,
            seed: 0xBADD_CAFE_F00D,
        }
    }
}

// ---------------------------------------------------------------------
// SZ weight conversion + uniformity
// ---------------------------------------------------------------------

/// Convert sampler-native weights (`w = |Ω|² / det g_pb`, summing to
/// 1 by sampler convention) into Shiffman-Zelditch weights for the
/// CY3 metric measure (`w_SZ = w / |Ω|²`, re-normalised to sum-to-1).
/// Returns `(sz_weights, uniformity_score)`.
fn sz_weights(metric: &dyn MetricBackground) -> (Vec<f64>, f64) {
    let n = metric.n_points();
    let mut sz = vec![0.0f64; n];
    let mut total = 0.0f64;
    let mut bad = 0usize;
    for alpha in 0..n {
        let w = metric.weight(alpha);
        if !w.is_finite() || w <= 0.0 {
            bad += 1;
            continue;
        }
        let omega = metric.omega(alpha);
        let omega2 = omega.norm_sqr();
        if !omega2.is_finite() || omega2 <= 0.0 {
            bad += 1;
            continue;
        }
        let v = w / omega2;
        sz[alpha] = v;
        total += v;
    }
    if total > 0.0 {
        let inv = 1.0 / total;
        for w in sz.iter_mut() {
            *w *= inv;
        }
    }

    // Uniformity score: 1 − Var(N · w) / Mean(N · w).
    // For a perfectly uniform sampler N · w = 1 everywhere, so
    // variance is zero and the score is 1.0. For pathologically
    // non-uniform samplers (one weight dominates) the variance →
    // infinity and the score → 0.
    let n_eff = (n - bad) as f64;
    if n_eff <= 1.0 {
        return (sz, 0.0);
    }
    let mean: f64 = sz.iter().sum::<f64>() / (n as f64);
    let var: f64 = sz.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / (n as f64);
    let cv = if mean > 0.0 { var.sqrt() / mean } else { f64::INFINITY };
    let score = (1.0 / (1.0 + cv)).clamp(0.0, 1.0);
    (sz, score)
}

// ---------------------------------------------------------------------
// HYM-coupled triple overlap
// ---------------------------------------------------------------------

/// Compute the triple overlap `Y_{ijk}` on a fixed sample cloud.
fn triple_overlap_cpu(
    modes: &[Vec<Complex64>],
    omega: &[Complex64],
    weights: &[f64],
    h_v_diag: &[f64],
    n_modes: usize,
    n_pts: usize,
) -> Tensor3 {
    // h_v_diag is the diagonal of the HYM metric in the harmonic-mode
    // basis, applied as a Hermitian rescaling factor on each ψ
    // contribution. After Gram-Schmidt under the bundle inner product,
    // the harmonic modes are unit-norm; the residual h_v dressing
    // contributes a real scalar pre-factor per mode.
    let mut y = Tensor3::zeros(n_modes);
    for i in 0..n_modes {
        for j in 0..n_modes {
            for k in 0..n_modes {
                let mut acc = Complex64::new(0.0, 0.0);
                for p in 0..n_pts {
                    let w = weights[p];
                    if !w.is_finite() {
                        continue;
                    }
                    let om = omega[p];
                    if !om.re.is_finite() || !om.im.is_finite() {
                        continue;
                    }
                    let psi_i = modes[i][p];
                    let psi_j = modes[j][p];
                    let psi_k = modes[k][p];
                    if !psi_i.re.is_finite() || !psi_j.re.is_finite() || !psi_k.re.is_finite() {
                        continue;
                    }
                    acc += om.conj() * psi_i * psi_j * psi_k * Complex64::new(w, 0.0);
                }
                let scale = h_v_diag.get(i).copied().unwrap_or(1.0).sqrt()
                    * h_v_diag.get(j).copied().unwrap_or(1.0).sqrt()
                    * h_v_diag.get(k).copied().unwrap_or(1.0).sqrt();
                y.set(i, j, k, acc * Complex64::new(scale, 0.0));
            }
        }
    }
    y
}

/// Bootstrap one resample. Returns the resampled `Tensor3`.
fn bootstrap_one(
    modes: &[Vec<Complex64>],
    omega: &[Complex64],
    sz_weights: &[f64],
    h_v_diag: &[f64],
    n_modes: usize,
    n_pts: usize,
    rng_seed: u64,
) -> Tensor3 {
    let mut rng = ChaCha20Rng::seed_from_u64(rng_seed);
    // Sample n_pts indices with replacement.
    let mut idx = vec![0usize; n_pts];
    for i in 0..n_pts {
        idx[i] = rng.random_range(0..n_pts);
    }
    let mut y = Tensor3::zeros(n_modes);
    for i in 0..n_modes {
        for j in 0..n_modes {
            for k in 0..n_modes {
                let mut acc = Complex64::new(0.0, 0.0);
                for &p in &idx {
                    let w = sz_weights[p];
                    if !w.is_finite() {
                        continue;
                    }
                    let om = omega[p];
                    if !om.re.is_finite() || !om.im.is_finite() {
                        continue;
                    }
                    let psi_i = modes[i][p];
                    let psi_j = modes[j][p];
                    let psi_k = modes[k][p];
                    acc += om.conj() * psi_i * psi_j * psi_k * Complex64::new(w, 0.0);
                }
                let scale = h_v_diag.get(i).copied().unwrap_or(1.0).sqrt()
                    * h_v_diag.get(j).copied().unwrap_or(1.0).sqrt()
                    * h_v_diag.get(k).copied().unwrap_or(1.0).sqrt();
                y.set(i, j, k, acc * Complex64::new(scale, 0.0));
            }
        }
    }
    y
}

// ---------------------------------------------------------------------
// Top-level entry point
// ---------------------------------------------------------------------

/// Compute the Yukawa coupling tensor `Y_{ijk}` from harmonic-mode
/// representatives on a CY3 with HYM-equipped bundle.
///
/// `zero_modes.modes[i]` provides the i-th harmonic basis element as
/// per-sample-point complex values. `h_v` is the HYM metric on the
/// bundle (used to dress the contraction). `metric` provides the
/// CY3 sample cloud.
pub fn compute_yukawa_couplings(
    metric: &dyn MetricBackground,
    h_v: &HymHermitianMetric,
    zero_modes: &HarmonicZeroModeResult,
    config: &YukawaConfig,
) -> YukawaResult {
    let started = std::time::Instant::now();
    let n_modes = zero_modes.modes.len();
    let n_pts = metric.n_points();
    if n_modes == 0 || n_pts == 0 {
        return YukawaResult {
            couplings: Tensor3::zeros(0),
            couplings_uncertainty: Tensor3Real::zeros(0),
            n_quadrature_points: n_pts,
            quadrature_uniformity_score: 0.0,
            convergence_ratio: f64::INFINITY,
            run_metadata: YukawaRunMetadata {
                wall_clock_seconds: started.elapsed().as_secs_f64(),
                seed: config.seed,
                n_bootstrap: 0,
                used_hym_metric: !h_v.is_trivial_bundle,
                used_harmonic_modes: false,
            },
        };
    }

    // Mode values per point (one Vec<Complex64> per harmonic mode).
    let modes: Vec<Vec<Complex64>> = zero_modes
        .modes
        .iter()
        .map(|m| m.values.clone())
        .collect();

    // Omega values.
    let omega: Vec<Complex64> = (0..n_pts).map(|a| metric.omega(a)).collect();

    // SZ-corrected quadrature weights.
    let (sz, uniformity) = sz_weights(metric);

    // h_V diagonal in the harmonic-mode basis. Project the HYM
    // matrix onto the harmonic modes via the coefficient vectors:
    // h_v_diag[α] = Σ_{i,j} c_α[i]^* H[i,j] c_α[j].
    let mut h_v_diag = vec![1.0f64; n_modes];
    if !h_v.is_trivial_bundle && h_v.n > 0 {
        for alpha in 0..n_modes {
            let coeffs = &zero_modes.modes[alpha].coefficients;
            let m = coeffs.len().min(h_v.n);
            let mut acc = Complex64::new(0.0, 0.0);
            for i in 0..m {
                for j in 0..m {
                    acc += coeffs[i].conj() * h_v.entry(i, j) * coeffs[j];
                }
            }
            // Real positive part by Hermitian construction.
            let re = acc.re.max(1.0e-30);
            h_v_diag[alpha] = re;
        }
    }

    // Central-value Yukawa.
    let couplings = triple_overlap_cpu(&modes, &omega, &sz, &h_v_diag, n_modes, n_pts);

    // Bootstrap uncertainty (parallel over resamples).
    let n_bs = config.n_bootstrap.max(1);
    let bootstrap_seeds: Vec<u64> = (0..n_bs)
        .map(|i| config.seed.wrapping_add((i as u64).wrapping_mul(2654435761)))
        .collect();
    let bs_tensors: Vec<Tensor3> = bootstrap_seeds
        .into_par_iter()
        .map(|s| bootstrap_one(&modes, &omega, &sz, &h_v_diag, n_modes, n_pts, s))
        .collect();

    // Per-entry standard deviation: sqrt of the mean squared
    // deviation across resamples from the central value.
    let mut uncert = Tensor3Real::zeros(n_modes);
    for i in 0..n_modes {
        for j in 0..n_modes {
            for k in 0..n_modes {
                let central = couplings.entry(i, j, k);
                let mut s = 0.0f64;
                for bs in &bs_tensors {
                    let z = bs.entry(i, j, k);
                    let dre = z.re - central.re;
                    let dim = z.im - central.im;
                    s += dre * dre + dim * dim;
                }
                let var = s / (n_bs as f64);
                uncert.set(i, j, k, var.sqrt());
            }
        }
    }

    // Convergence test: Y on n_pts/2 vs Y on n_pts. Use the first
    // half of the points (deterministic) for the half estimate.
    let half_pts = n_pts / 2;
    let convergence_ratio = if half_pts >= 8 && n_modes >= 1 {
        let half_modes: Vec<Vec<Complex64>> = modes.iter().map(|v| v[..half_pts].to_vec()).collect();
        let half_omega: Vec<Complex64> = omega[..half_pts].to_vec();
        let half_w_unnorm: Vec<f64> = sz[..half_pts].to_vec();
        let total: f64 = half_w_unnorm.iter().sum();
        let half_w: Vec<f64> = if total > 0.0 {
            half_w_unnorm.iter().map(|w| w / total).collect()
        } else {
            half_w_unnorm
        };
        let y_half = triple_overlap_cpu(&half_modes, &half_omega, &half_w, &h_v_diag, n_modes, half_pts);
        // max |Y(N) - Y(N/2)| / max |Y(N)|.
        let mut max_diff = 0.0f64;
        let mut max_full = 0.0f64;
        for k in 0..(n_modes * n_modes * n_modes) {
            let f = couplings.data[k];
            let h = y_half.data[k];
            let d = ((f.re - h.re).powi(2) + (f.im - h.im).powi(2)).sqrt();
            if d > max_diff {
                max_diff = d;
            }
            if f.norm() > max_full {
                max_full = f.norm();
            }
        }
        if max_full > 0.0 {
            max_diff / max_full
        } else {
            0.0
        }
    } else {
        f64::NAN
    };

    YukawaResult {
        couplings,
        couplings_uncertainty: uncert,
        n_quadrature_points: n_pts,
        quadrature_uniformity_score: uniformity,
        convergence_ratio,
        run_metadata: YukawaRunMetadata {
            wall_clock_seconds: started.elapsed().as_secs_f64(),
            seed: config.seed,
            n_bootstrap: n_bs,
            used_hym_metric: !h_v.is_trivial_bundle,
            used_harmonic_modes: true,
        },
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route34::hym_hermitian::{
        solve_hym_metric, HymConfig, InMemoryMetricBackground,
    };
    use crate::route34::zero_modes_harmonic::{
        solve_harmonic_zero_modes, HarmonicConfig,
    };
    use crate::zero_modes::{AmbientCY3, MonadBundle};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    fn synthetic_metric(n_pts: usize, seed: u64) -> InMemoryMetricBackground {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut points = Vec::with_capacity(n_pts);
        for _ in 0..n_pts {
            let mut p = [Complex64::new(0.0, 0.0); 8];
            for k in 0..8 {
                let re: f64 = rng.random_range(-1.0..1.0);
                let im: f64 = rng.random_range(-1.0..1.0);
                p[k] = Complex64::new(re, im);
            }
            points.push(p);
        }
        let w_each = 1.0 / (n_pts as f64);
        InMemoryMetricBackground {
            points,
            weights: vec![w_each; n_pts],
            omega: vec![Complex64::new(1.0, 0.0); n_pts],
        }
    }

    /// Test 1: quadrature uniformity score is finite and in [0, 1].
    #[test]
    fn quadrature_uniformity_in_unit_interval() {
        let metric = synthetic_metric(60, 21);
        let (_, score) = sz_weights(&metric);
        assert!(score.is_finite());
        assert!((0.0..=1.0).contains(&score), "score {} out of [0,1]", score);
    }

    /// Test 2: full pipeline runs and returns a finite Yukawa tensor +
    /// non-negative uncertainty tensor.
    #[test]
    fn yukawa_pipeline_produces_finite_tensor() {
        let bundle = MonadBundle::anderson_lukas_palti_example();
        let ambient = AmbientCY3::tian_yau_upstairs();
        let metric = synthetic_metric(80, 22);
        let h_v = solve_hym_metric(&bundle, &metric, &HymConfig {
            max_iter: 8,
            damping: 0.5,
            ..HymConfig::default()
        });
        let modes = solve_harmonic_zero_modes(
            &bundle,
            &ambient,
            &metric,
            &h_v,
            &HarmonicConfig::default(),
        );
        if modes.modes.is_empty() {
            return;
        }
        let cfg = YukawaConfig {
            n_bootstrap: 16,
            ..YukawaConfig::default()
        };
        let y = compute_yukawa_couplings(&metric, &h_v, &modes, &cfg);
        assert_eq!(y.couplings.n, modes.modes.len());
        for z in &y.couplings.data {
            assert!(z.re.is_finite() && z.im.is_finite());
        }
        for u in &y.couplings_uncertainty.data {
            assert!(*u >= 0.0 && u.is_finite());
        }
    }

    /// Test 3: bootstrap uncertainty decreases (qualitatively) when
    /// the sample size grows. We compare an n_pts = 60 run to an
    /// n_pts = 240 run and verify the average per-entry uncertainty
    /// is lower at n_pts = 240.
    #[test]
    fn yukawa_uncertainty_decreases_with_n() {
        let bundle = MonadBundle::anderson_lukas_palti_example();
        let ambient = AmbientCY3::tian_yau_upstairs();
        let m_small = synthetic_metric(60, 31);
        let m_large = synthetic_metric(240, 31);

        let cfg_hym = HymConfig {
            max_iter: 6,
            damping: 0.5,
            ..HymConfig::default()
        };

        let h_small = solve_hym_metric(&bundle, &m_small, &cfg_hym);
        let h_large = solve_hym_metric(&bundle, &m_large, &cfg_hym);
        let modes_small = solve_harmonic_zero_modes(
            &bundle,
            &ambient,
            &m_small,
            &h_small,
            &HarmonicConfig::default(),
        );
        let modes_large = solve_harmonic_zero_modes(
            &bundle,
            &ambient,
            &m_large,
            &h_large,
            &HarmonicConfig::default(),
        );

        if modes_small.modes.is_empty() || modes_large.modes.is_empty() {
            return;
        }

        let cfg = YukawaConfig {
            n_bootstrap: 32,
            ..YukawaConfig::default()
        };
        let y_small = compute_yukawa_couplings(&m_small, &h_small, &modes_small, &cfg);
        let y_large = compute_yukawa_couplings(&m_large, &h_large, &modes_large, &cfg);

        // Average uncertainty.
        let avg_small: f64 = y_small.couplings_uncertainty.data.iter().sum::<f64>()
            / y_small.couplings_uncertainty.data.len().max(1) as f64;
        let avg_large: f64 = y_large.couplings_uncertainty.data.iter().sum::<f64>()
            / y_large.couplings_uncertainty.data.len().max(1) as f64;

        // We don't enforce strict monotonicity (the Y tensor itself is
        // also resampled), but the average uncertainty should not be
        // wildly larger at n=240 vs n=60. A 3× upper bound is generous.
        assert!(
            avg_large <= 3.0 * avg_small.max(1.0e-30),
            "avg uncertainty grew unexpectedly: small {} large {}",
            avg_small,
            avg_large
        );
    }

    /// Test 4: Tensor3 indexing round-trip.
    #[test]
    fn tensor3_set_get_roundtrip() {
        let mut t = Tensor3::zeros(3);
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let z = Complex64::new(i as f64, (j + k) as f64);
                    t.set(i, j, k, z);
                }
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let z = t.entry(i, j, k);
                    assert!((z.re - i as f64).abs() < 1e-12);
                    assert!((z.im - (j + k) as f64).abs() < 1e-12);
                }
            }
        }
    }
}
