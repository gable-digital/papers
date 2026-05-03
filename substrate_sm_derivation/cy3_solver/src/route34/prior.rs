//! # Bayesian prior distributions over moduli
//!
//! Defines the [`Prior`] trait used by [`crate::route34::nested_sampling`]
//! and the concrete distributions used in the Tian-Yau / Schoen
//! discrimination problem:
//!
//! * [`LogUniformPrior`] — Jeffreys 1946 prior for positive scale
//!   parameters (Kähler-class moduli). Density `pi(theta) = 1/theta`
//!   with normaliser `1 / ln(max/min)` per dimension.
//! * [`DiscreteUniformPrior`] — uniform measure over a finite set
//!   (line-bundle enumeration, Wilson-line embeddings).
//! * [`ProductPrior`] — product measure over (Kähler, bundle, Wilson).
//! * [`UniformPrior`] — uniform on a hyper-rectangle (used by tests).
//! * [`GaussianPrior`] — diagonal multivariate Gaussian (used by tests
//!   to validate the nested sampler against analytic evidence values).
//!
//! ## References
//!
//! * Jeffreys H., "An invariant form for the prior probability in
//!   estimation problems", Proc. Roy. Soc. A 186 (1946) 453,
//!   DOI 10.1098/rspa.1946.0056.
//! * Skilling J., "Nested sampling for general Bayesian computation",
//!   AIP Conf. Proc. 735 (2004) 395, DOI 10.1063/1.1835238.
//! * Berger J.O., "Statistical Decision Theory and Bayesian Analysis"
//!   (Springer 1985), §3.3 (objective priors).

use std::f64::consts::PI;
use std::fmt::Debug;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};

/// A point in the moduli space the prior is defined over.
///
/// `continuous` carries the real-valued moduli coordinates (e.g. Kähler
/// classes); `discrete_indices` carries indices into discrete enumerations
/// (e.g. which line-bundle entry from [`crate::route34::bundle_search`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModuliPoint {
    pub continuous: Vec<f64>,
    pub discrete_indices: Vec<usize>,
}

impl ModuliPoint {
    pub fn new(continuous: Vec<f64>, discrete_indices: Vec<usize>) -> Self {
        Self {
            continuous,
            discrete_indices,
        }
    }

    pub fn continuous_only(continuous: Vec<f64>) -> Self {
        Self {
            continuous,
            discrete_indices: Vec::new(),
        }
    }

    pub fn dimension(&self) -> usize {
        self.continuous.len() + self.discrete_indices.len()
    }
}

/// Trait for a prior distribution over a moduli space.
///
/// Conventions:
///
/// * `log_prior` returns `ln pi(theta)`. For an improper prior (no
///   normalisable density), an additive constant is permissible only
///   when comparing models that share the same prior support.
/// * `sample` draws a single i.i.d. sample using the supplied RNG.
/// * `sample_unit_cube` accepts a uniform sample `u in [0,1]^d` and
///   maps it to a sample from the prior via the inverse-CDF method.
///   This is required by Skilling 2004 nested sampling: the live
///   points are reparameterised onto the unit cube and the
///   constrained-prior proposal sampler proposes new unit-cube points,
///   which the prior maps back to physical coordinates.
pub trait Prior: Send + Sync + Debug {
    fn log_prior(&self, theta: &ModuliPoint) -> f64;
    fn sample(&self, rng: &mut ChaCha20Rng) -> ModuliPoint;
    fn sample_unit_cube(&self, u: &[f64]) -> ModuliPoint;
    fn dimension(&self) -> usize;
    /// Number of continuous (real-valued) dimensions.
    fn continuous_dimension(&self) -> usize;
    /// Number of discrete (integer-valued) dimensions.
    fn discrete_dimension(&self) -> usize {
        self.dimension() - self.continuous_dimension()
    }
    /// Domain check: returns `true` iff `theta` lies in the support.
    fn in_support(&self, theta: &ModuliPoint) -> bool;
}

// ----------------------------------------------------------------------
// LogUniformPrior — Jeffreys 1946 scale-parameter prior.
// ----------------------------------------------------------------------

/// Log-uniform (Jeffreys) prior on a positive-real hyper-rectangle.
///
/// Density:
///     pi(theta_k) = 1 / (theta_k * ln(max_k / min_k))  for theta_k in [min_k, max_k]
///
/// All components are independent, so
///     log pi(theta) = -sum_k ln(theta_k) - sum_k ln(ln(max_k / min_k))
///
/// Reference: Jeffreys 1946 §3.1 — for a positive scale parameter `s`
/// the unique invariant prior under reparameterisation `s -> a s` is
/// `pi(s) ds = ds / s`, equivalent to a uniform measure on `ln s`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogUniformPrior {
    pub min: Vec<f64>,
    pub max: Vec<f64>,
}

impl LogUniformPrior {
    /// Construct. Returns `None` if any bound is non-positive or if
    /// `min >= max` componentwise.
    pub fn new(min: Vec<f64>, max: Vec<f64>) -> Option<Self> {
        if min.len() != max.len() || min.is_empty() {
            return None;
        }
        for k in 0..min.len() {
            if !(min[k].is_finite()) || !(max[k].is_finite()) {
                return None;
            }
            if min[k] <= 0.0 || max[k] <= min[k] {
                return None;
            }
        }
        Some(Self { min, max })
    }

    /// Log of the per-dimension normaliser ln(ln(max/min)).
    fn ln_log_ratio(&self, k: usize) -> f64 {
        (self.max[k].ln() - self.min[k].ln()).ln()
    }
}

impl Prior for LogUniformPrior {
    fn log_prior(&self, theta: &ModuliPoint) -> f64 {
        if theta.continuous.len() != self.min.len() || !theta.discrete_indices.is_empty() {
            return f64::NEG_INFINITY;
        }
        let mut lp = 0.0;
        for k in 0..self.min.len() {
            let x = theta.continuous[k];
            if !(x.is_finite()) || x <= 0.0 || x < self.min[k] || x > self.max[k] {
                return f64::NEG_INFINITY;
            }
            // ln pi_k(x) = -ln(x) - ln(ln(max_k/min_k))
            lp -= x.ln();
            lp -= self.ln_log_ratio(k);
        }
        lp
    }

    fn sample(&self, rng: &mut ChaCha20Rng) -> ModuliPoint {
        let mut x = Vec::with_capacity(self.min.len());
        for k in 0..self.min.len() {
            let u: f64 = rng.random_range(0.0..1.0);
            // Inverse CDF: x = exp(ln(min) + u * ln(max/min))
            let ln_min = self.min[k].ln();
            let ln_max = self.max[k].ln();
            x.push((ln_min + u * (ln_max - ln_min)).exp());
        }
        ModuliPoint::continuous_only(x)
    }

    fn sample_unit_cube(&self, u: &[f64]) -> ModuliPoint {
        debug_assert_eq!(u.len(), self.min.len());
        let mut x = Vec::with_capacity(self.min.len());
        for k in 0..self.min.len() {
            let uk = u[k].clamp(1e-300, 1.0 - 1e-15);
            let ln_min = self.min[k].ln();
            let ln_max = self.max[k].ln();
            x.push((ln_min + uk * (ln_max - ln_min)).exp());
        }
        ModuliPoint::continuous_only(x)
    }

    fn dimension(&self) -> usize {
        self.min.len()
    }

    fn continuous_dimension(&self) -> usize {
        self.min.len()
    }

    fn in_support(&self, theta: &ModuliPoint) -> bool {
        if theta.continuous.len() != self.min.len() || !theta.discrete_indices.is_empty() {
            return false;
        }
        for k in 0..self.min.len() {
            let x = theta.continuous[k];
            if !x.is_finite() || x < self.min[k] || x > self.max[k] {
                return false;
            }
        }
        true
    }
}

// ----------------------------------------------------------------------
// DiscreteUniformPrior — uniform measure over a finite enumeration.
// ----------------------------------------------------------------------

/// Uniform measure over an indexed finite set of size `n`.
///
/// `log_prior(theta) = -ln(n)` if the (single) discrete index is in
/// `[0, n)`, else `-inf`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiscreteUniformPrior {
    pub n: usize,
}

impl DiscreteUniformPrior {
    pub fn new(n: usize) -> Option<Self> {
        if n == 0 {
            None
        } else {
            Some(Self { n })
        }
    }
}

impl Prior for DiscreteUniformPrior {
    fn log_prior(&self, theta: &ModuliPoint) -> f64 {
        if theta.discrete_indices.len() != 1 || !theta.continuous.is_empty() {
            return f64::NEG_INFINITY;
        }
        let idx = theta.discrete_indices[0];
        if idx >= self.n {
            return f64::NEG_INFINITY;
        }
        -(self.n as f64).ln()
    }

    fn sample(&self, rng: &mut ChaCha20Rng) -> ModuliPoint {
        let idx = rng.random_range(0..self.n);
        ModuliPoint::new(Vec::new(), vec![idx])
    }

    fn sample_unit_cube(&self, u: &[f64]) -> ModuliPoint {
        debug_assert_eq!(u.len(), 1);
        let uk = u[0].clamp(0.0, 1.0 - 1e-15);
        let idx = ((uk * self.n as f64) as usize).min(self.n - 1);
        ModuliPoint::new(Vec::new(), vec![idx])
    }

    fn dimension(&self) -> usize {
        1
    }

    fn continuous_dimension(&self) -> usize {
        0
    }

    fn in_support(&self, theta: &ModuliPoint) -> bool {
        theta.discrete_indices.len() == 1
            && theta.continuous.is_empty()
            && theta.discrete_indices[0] < self.n
    }
}

// ----------------------------------------------------------------------
// UniformPrior — uniform on a hyper-rectangle (used by tests).
// ----------------------------------------------------------------------

/// Uniform prior on `prod_k [low_k, high_k]`.
///
/// Density `pi(theta) = 1 / prod_k (high_k - low_k)`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UniformPrior {
    pub low: Vec<f64>,
    pub high: Vec<f64>,
}

impl UniformPrior {
    pub fn new(low: Vec<f64>, high: Vec<f64>) -> Option<Self> {
        if low.len() != high.len() || low.is_empty() {
            return None;
        }
        for k in 0..low.len() {
            if !low[k].is_finite() || !high[k].is_finite() || high[k] <= low[k] {
                return None;
            }
        }
        Some(Self { low, high })
    }

    pub fn log_volume(&self) -> f64 {
        self.low
            .iter()
            .zip(self.high.iter())
            .map(|(a, b)| (b - a).ln())
            .sum()
    }
}

impl Prior for UniformPrior {
    fn log_prior(&self, theta: &ModuliPoint) -> f64 {
        if theta.continuous.len() != self.low.len() || !theta.discrete_indices.is_empty() {
            return f64::NEG_INFINITY;
        }
        for k in 0..self.low.len() {
            let x = theta.continuous[k];
            if !x.is_finite() || x < self.low[k] || x > self.high[k] {
                return f64::NEG_INFINITY;
            }
        }
        -self.log_volume()
    }

    fn sample(&self, rng: &mut ChaCha20Rng) -> ModuliPoint {
        let mut x = Vec::with_capacity(self.low.len());
        for k in 0..self.low.len() {
            let u: f64 = rng.random_range(0.0..1.0);
            x.push(self.low[k] + u * (self.high[k] - self.low[k]));
        }
        ModuliPoint::continuous_only(x)
    }

    fn sample_unit_cube(&self, u: &[f64]) -> ModuliPoint {
        debug_assert_eq!(u.len(), self.low.len());
        let mut x = Vec::with_capacity(self.low.len());
        for k in 0..self.low.len() {
            let uk = u[k].clamp(0.0, 1.0);
            x.push(self.low[k] + uk * (self.high[k] - self.low[k]));
        }
        ModuliPoint::continuous_only(x)
    }

    fn dimension(&self) -> usize {
        self.low.len()
    }

    fn continuous_dimension(&self) -> usize {
        self.low.len()
    }

    fn in_support(&self, theta: &ModuliPoint) -> bool {
        if theta.continuous.len() != self.low.len() || !theta.discrete_indices.is_empty() {
            return false;
        }
        for k in 0..self.low.len() {
            let x = theta.continuous[k];
            if !x.is_finite() || x < self.low[k] || x > self.high[k] {
                return false;
            }
        }
        true
    }
}

// ----------------------------------------------------------------------
// GaussianPrior — diagonal multivariate Gaussian (used by tests).
// ----------------------------------------------------------------------

/// Diagonal multivariate Gaussian prior with mean `mu` and standard
/// deviations `sigma` (componentwise).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GaussianPrior {
    pub mu: Vec<f64>,
    pub sigma: Vec<f64>,
}

impl GaussianPrior {
    pub fn new(mu: Vec<f64>, sigma: Vec<f64>) -> Option<Self> {
        if mu.len() != sigma.len() || mu.is_empty() {
            return None;
        }
        for s in &sigma {
            if !s.is_finite() || *s <= 0.0 {
                return None;
            }
        }
        Some(Self { mu, sigma })
    }
}

/// Inverse standard-normal CDF (Acklam 2003 rational approximation).
/// Accurate to ~4.5e-4. Sufficient for unit-cube reparameterisation;
/// nested sampling does not depend on the prior CDF being invertible to
/// machine precision.
fn inv_phi_acklam(p: f64) -> f64 {
    let p = p.clamp(1e-300, 1.0 - 1e-16);
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

impl Prior for GaussianPrior {
    fn log_prior(&self, theta: &ModuliPoint) -> f64 {
        if theta.continuous.len() != self.mu.len() || !theta.discrete_indices.is_empty() {
            return f64::NEG_INFINITY;
        }
        let mut lp = 0.0;
        for k in 0..self.mu.len() {
            let x = theta.continuous[k];
            if !x.is_finite() {
                return f64::NEG_INFINITY;
            }
            let z = (x - self.mu[k]) / self.sigma[k];
            lp += -0.5 * z * z - self.sigma[k].ln() - 0.5 * (2.0 * PI).ln();
        }
        lp
    }

    fn sample(&self, rng: &mut ChaCha20Rng) -> ModuliPoint {
        // Box-Muller via two uniform draws.
        let mut x = Vec::with_capacity(self.mu.len());
        for k in 0..self.mu.len() {
            let u1: f64 = rng.random_range(1e-15..1.0);
            let u2: f64 = rng.random_range(0.0..1.0);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            x.push(self.mu[k] + self.sigma[k] * z);
        }
        ModuliPoint::continuous_only(x)
    }

    fn sample_unit_cube(&self, u: &[f64]) -> ModuliPoint {
        debug_assert_eq!(u.len(), self.mu.len());
        let mut x = Vec::with_capacity(self.mu.len());
        for k in 0..self.mu.len() {
            let z = inv_phi_acklam(u[k]);
            x.push(self.mu[k] + self.sigma[k] * z);
        }
        ModuliPoint::continuous_only(x)
    }

    fn dimension(&self) -> usize {
        self.mu.len()
    }

    fn continuous_dimension(&self) -> usize {
        self.mu.len()
    }

    fn in_support(&self, theta: &ModuliPoint) -> bool {
        theta.continuous.len() == self.mu.len()
            && theta.discrete_indices.is_empty()
            && theta.continuous.iter().all(|x| x.is_finite())
    }
}

// ----------------------------------------------------------------------
// ProductPrior — product of (Kähler, bundle, Wilson) priors.
// ----------------------------------------------------------------------

/// Product prior over the discrimination problem's three moduli sectors:
///
/// * continuous Kähler-class moduli (log-uniform / Jeffreys)
/// * discrete bundle index (uniform over enumerated catalogue)
/// * discrete Wilson-line index (uniform over canonical embeddings)
///
/// `sample_unit_cube` consumes a `(d_k + 2)`-dim uniform sample where
/// `d_k = kahler.dimension()`; the first `d_k` entries map onto the
/// Kähler block, the next two pick the bundle and Wilson indices.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProductPrior {
    pub kahler: LogUniformPrior,
    pub bundle: DiscreteUniformPrior,
    pub wilson: DiscreteUniformPrior,
}

impl ProductPrior {
    pub fn new(
        kahler: LogUniformPrior,
        bundle: DiscreteUniformPrior,
        wilson: DiscreteUniformPrior,
    ) -> Self {
        Self {
            kahler,
            bundle,
            wilson,
        }
    }
}

impl Prior for ProductPrior {
    fn log_prior(&self, theta: &ModuliPoint) -> f64 {
        if theta.continuous.len() != self.kahler.dimension()
            || theta.discrete_indices.len() != 2
        {
            return f64::NEG_INFINITY;
        }
        let kahler_pt = ModuliPoint::continuous_only(theta.continuous.clone());
        let bundle_pt = ModuliPoint::new(Vec::new(), vec![theta.discrete_indices[0]]);
        let wilson_pt = ModuliPoint::new(Vec::new(), vec![theta.discrete_indices[1]]);
        let lp_k = self.kahler.log_prior(&kahler_pt);
        if !lp_k.is_finite() {
            return f64::NEG_INFINITY;
        }
        let lp_b = self.bundle.log_prior(&bundle_pt);
        if !lp_b.is_finite() {
            return f64::NEG_INFINITY;
        }
        let lp_w = self.wilson.log_prior(&wilson_pt);
        if !lp_w.is_finite() {
            return f64::NEG_INFINITY;
        }
        lp_k + lp_b + lp_w
    }

    fn sample(&self, rng: &mut ChaCha20Rng) -> ModuliPoint {
        let k_pt = self.kahler.sample(rng);
        let b_pt = self.bundle.sample(rng);
        let w_pt = self.wilson.sample(rng);
        ModuliPoint::new(
            k_pt.continuous,
            vec![b_pt.discrete_indices[0], w_pt.discrete_indices[0]],
        )
    }

    fn sample_unit_cube(&self, u: &[f64]) -> ModuliPoint {
        let d_k = self.kahler.dimension();
        debug_assert_eq!(u.len(), d_k + 2);
        let k_pt = self.kahler.sample_unit_cube(&u[..d_k]);
        let b_pt = self.bundle.sample_unit_cube(&[u[d_k]]);
        let w_pt = self.wilson.sample_unit_cube(&[u[d_k + 1]]);
        ModuliPoint::new(
            k_pt.continuous,
            vec![b_pt.discrete_indices[0], w_pt.discrete_indices[0]],
        )
    }

    fn dimension(&self) -> usize {
        self.kahler.dimension() + 2
    }

    fn continuous_dimension(&self) -> usize {
        self.kahler.dimension()
    }

    fn in_support(&self, theta: &ModuliPoint) -> bool {
        if theta.continuous.len() != self.kahler.dimension()
            || theta.discrete_indices.len() != 2
        {
            return false;
        }
        if !self
            .kahler
            .in_support(&ModuliPoint::continuous_only(theta.continuous.clone()))
        {
            return false;
        }
        theta.discrete_indices[0] < self.bundle.n && theta.discrete_indices[1] < self.wilson.n
    }
}

/// Construct a [`ChaCha20Rng`] from a `u64` seed via the crate-canonical
/// path. Provided here so that downstream modules (likelihood,
/// nested_sampling) all agree on the seed-to-RNG mapping.
pub fn rng_from_seed(seed: u64) -> ChaCha20Rng {
    ChaCha20Rng::seed_from_u64(seed)
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_uniform_normalizes_to_unity() {
        // 1-D log-uniform on [0.01, 100]: integrate exp(log_prior) dx.
        // Discrete trapezoid in log-space (where the density is uniform).
        let p = LogUniformPrior::new(vec![0.01], vec![100.0]).unwrap();
        // Use 100k log-spaced points, trapezoid rule in linear x.
        let n = 100_000;
        let ln_min = 0.01_f64.ln();
        let ln_max = 100.0_f64.ln();
        let mut total = 0.0;
        let mut prev_x: f64 = 0.01;
        let mut prev_pdf = (-prev_x.ln() - p.ln_log_ratio(0)).exp();
        for i in 1..=n {
            let t = i as f64 / n as f64;
            let x = (ln_min + t * (ln_max - ln_min)).exp();
            let pdf = (-x.ln() - p.ln_log_ratio(0)).exp();
            total += 0.5 * (prev_pdf + pdf) * (x - prev_x);
            prev_x = x;
            prev_pdf = pdf;
        }
        assert!(
            (total - 1.0).abs() < 1e-3,
            "log-uniform pdf does not normalise: integral = {}",
            total
        );
    }

    #[test]
    fn test_jeffreys_prior_reference() {
        // Jeffreys 1946 §3.1: for a positive scale parameter `s`, the
        // invariant prior is pi(s) ds = ds / s. The cumulative
        //     P(s <= S) = (ln S - ln min) / (ln max - ln min).
        // Equivalently, sampling u ~ Uniform[0,1] and setting
        //     s = exp(ln min + u (ln max - ln min))
        // gives s drawn from this prior. We verify by drawing 100k
        // samples and checking that the empirical CDF matches the
        // analytic one at five quantile points.
        let p = LogUniformPrior::new(vec![1e-3], vec![1e3]).unwrap();
        let mut rng = rng_from_seed(20240426);
        let n = 100_000usize;
        let mut samples: Vec<f64> = (0..n).map(|_| p.sample(&mut rng).continuous[0]).collect();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Quantile points 10%, 25%, 50%, 75%, 90%.
        let qs = [0.1, 0.25, 0.5, 0.75, 0.9];
        let ln_min = 1e-3_f64.ln();
        let ln_max = 1e3_f64.ln();
        for q in &qs {
            let s_emp = samples[((q * n as f64) as usize).min(n - 1)];
            let s_ana = (ln_min + q * (ln_max - ln_min)).exp();
            let rel = ((s_emp - s_ana) / s_ana).abs();
            // Empirical-CDF Monte-Carlo tolerance for 100k samples;
            // Kolmogorov-Smirnov 99.5%-quantile is ~ 1.7 / sqrt(n) ~
            // 5e-3 on the CDF scale. The corresponding *quantile-scale*
            // tolerance depends on the local density and is several
            // percent for the log-uniform near the centre. A 5% relative
            // band passes comfortably.
            assert!(
                rel < 0.05,
                "Jeffreys-prior empirical CDF at q={} differs from analytic by {:.3e} (s_emp={}, s_ana={})",
                q,
                rel,
                s_emp,
                s_ana
            );
        }
    }

    #[test]
    fn test_log_uniform_unit_cube_round_trip() {
        let p = LogUniformPrior::new(vec![0.5, 1e-2], vec![20.0, 1.0]).unwrap();
        // u = 0 -> min; u = 1 -> max (modulo clamp).
        let pt0 = p.sample_unit_cube(&[0.0, 0.0]);
        let pt1 = p.sample_unit_cube(&[1.0, 1.0]);
        assert!((pt0.continuous[0] - 0.5).abs() < 1e-6);
        assert!((pt0.continuous[1] - 1e-2).abs() < 1e-6);
        assert!((pt1.continuous[0] - 20.0).abs() < 1e-3);
        assert!((pt1.continuous[1] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_log_uniform_rejects_out_of_support() {
        let p = LogUniformPrior::new(vec![1.0], vec![10.0]).unwrap();
        let bad = ModuliPoint::continuous_only(vec![0.5]);
        assert_eq!(p.log_prior(&bad), f64::NEG_INFINITY);
        let bad2 = ModuliPoint::continuous_only(vec![100.0]);
        assert_eq!(p.log_prior(&bad2), f64::NEG_INFINITY);
        // Wrong dimension.
        let bad3 = ModuliPoint::continuous_only(vec![5.0, 5.0]);
        assert_eq!(p.log_prior(&bad3), f64::NEG_INFINITY);
        // Discrete-mixed in.
        let bad4 = ModuliPoint::new(vec![5.0], vec![0]);
        assert_eq!(p.log_prior(&bad4), f64::NEG_INFINITY);
    }

    #[test]
    fn test_log_uniform_rejects_nonpositive_bounds() {
        assert!(LogUniformPrior::new(vec![0.0], vec![1.0]).is_none());
        assert!(LogUniformPrior::new(vec![-1.0], vec![1.0]).is_none());
        assert!(LogUniformPrior::new(vec![1.0], vec![1.0]).is_none());
    }

    #[test]
    fn test_discrete_uniform_log_prior() {
        let p = DiscreteUniformPrior::new(7).unwrap();
        for i in 0..7 {
            let pt = ModuliPoint::new(Vec::new(), vec![i]);
            let lp = p.log_prior(&pt);
            assert!((lp - (-(7.0_f64).ln())).abs() < 1e-15);
        }
        let pt_bad = ModuliPoint::new(Vec::new(), vec![7]);
        assert_eq!(p.log_prior(&pt_bad), f64::NEG_INFINITY);
    }

    #[test]
    fn test_discrete_uniform_sample_distribution() {
        // 10k draws into 4 buckets — chi^2 vs uniform should be small.
        let p = DiscreteUniformPrior::new(4).unwrap();
        let mut rng = rng_from_seed(7);
        let n = 10_000;
        let mut counts = [0usize; 4];
        for _ in 0..n {
            let pt = p.sample(&mut rng);
            counts[pt.discrete_indices[0]] += 1;
        }
        let expected = (n as f64) / 4.0;
        let chi2: f64 = counts
            .iter()
            .map(|&c| {
                let d = c as f64 - expected;
                d * d / expected
            })
            .sum();
        // 4 bins => 3 dof; the 99.9-th percentile of chi^2_3 is 16.27.
        assert!(
            chi2 < 16.27,
            "discrete-uniform distribution-of-fit chi^2 = {} (3 dof; should be << 16.27)",
            chi2
        );
    }

    #[test]
    fn test_uniform_prior_log_prior() {
        let p = UniformPrior::new(vec![-1.0, 0.0], vec![1.0, 4.0]).unwrap();
        // log volume = ln 2 + ln 4 = ln 8
        let ln_v = 8.0_f64.ln();
        let pt = ModuliPoint::continuous_only(vec![0.5, 2.0]);
        assert!((p.log_prior(&pt) - (-ln_v)).abs() < 1e-15);
    }

    #[test]
    fn test_gaussian_prior_normalisation() {
        // Diagonal 2D N(0, I): integrate exp(log_prior) over a large box.
        let p = GaussianPrior::new(vec![0.0, 0.0], vec![1.0, 1.0]).unwrap();
        let n = 401;
        let r = 6.0;
        let h = 2.0 * r / (n as f64 - 1.0);
        let mut total = 0.0;
        for i in 0..n {
            for j in 0..n {
                let x = -r + i as f64 * h;
                let y = -r + j as f64 * h;
                let pt = ModuliPoint::continuous_only(vec![x, y]);
                let pdf = p.log_prior(&pt).exp();
                let w = if (i == 0 || i == n - 1) && (j == 0 || j == n - 1) {
                    0.25
                } else if i == 0 || i == n - 1 || j == 0 || j == n - 1 {
                    0.5
                } else {
                    1.0
                };
                total += w * pdf * h * h;
            }
        }
        assert!(
            (total - 1.0).abs() < 5e-3,
            "2D unit Gaussian PDF integral = {} (expected 1)",
            total
        );
    }

    #[test]
    fn test_product_prior_dimension_and_support() {
        let kp = LogUniformPrior::new(vec![1e-2, 1e-2], vec![1e2, 1e2]).unwrap();
        let bp = DiscreteUniformPrior::new(8).unwrap();
        let wp = DiscreteUniformPrior::new(3).unwrap();
        let p = ProductPrior::new(kp, bp, wp);
        assert_eq!(p.dimension(), 4);
        assert_eq!(p.continuous_dimension(), 2);
        assert_eq!(p.discrete_dimension(), 2);

        let mut rng = rng_from_seed(42);
        let pt = p.sample(&mut rng);
        assert_eq!(pt.continuous.len(), 2);
        assert_eq!(pt.discrete_indices.len(), 2);
        assert!(p.in_support(&pt));
    }

    #[test]
    fn test_inv_phi_acklam_matches_known_values() {
        // Phi(0) = 0.5  -> inv_phi(0.5) = 0
        let z0 = inv_phi_acklam(0.5);
        assert!(z0.abs() < 1e-3, "inv_phi(0.5) = {}", z0);
        // Phi(1.96) ~ 0.975
        let z1 = inv_phi_acklam(0.975);
        assert!(
            (z1 - 1.96).abs() < 1e-3,
            "inv_phi(0.975) = {} (expected ~1.96)",
            z1
        );
        // Symmetric.
        let zlo = inv_phi_acklam(0.025);
        assert!((zlo + 1.96).abs() < 1e-3);
    }
}
