//! Bootstrap error bars for residual estimators.
//!
//! Monte-Carlo residuals at finite sample size N have a stochastic
//! component that goes as 1/sqrt(N). For publication-grade reporting
//! we need an error bar on the residual, not just a point estimate.
//! This module implements the standard non-parametric bootstrap:
//!
//!   1. Given a per-point quantity x_p (p = 1..N), compute the point
//!      estimate as (1/N) sum_p x_p (or any statistic of x_p).
//!   2. Resample N points WITH replacement to get a bootstrap sample
//!      x_p^(b); compute the same statistic on this resample.
//!   3. Repeat for B bootstrap iterations (typical B = 100-1000).
//!   4. The distribution of the resampled statistics gives the
//!      standard error and confidence interval.
//!
//! For our pipeline, the per-point quantity is `log|det H_tan|` (the
//! pointwise Monge-Ampère integrand), and the statistic is the
//! variance over points (which is the residual itself).

use crate::LCG;

/// Bootstrap variance estimator. Given per-point values `x` (length N),
/// returns (point_estimate, std_error) where:
///   point_estimate = var(x)
///   std_error      = std-dev of var(x_b) over B bootstrap resamples
pub fn bootstrap_variance(x: &[f64], n_bootstrap: usize, seed: u64) -> (f64, f64) {
    let n = x.len();
    if n == 0 || n_bootstrap == 0 {
        return (0.0, 0.0);
    }
    let mean: f64 = x.iter().sum::<f64>() / n as f64;
    let variance: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;

    let mut rng = LCG::new(seed);
    let mut boot_vars: Vec<f64> = Vec::with_capacity(n_bootstrap);
    for _ in 0..n_bootstrap {
        // Resample N indices with replacement.
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let idx = (rng.next_f64() * n as f64) as usize;
            let idx = idx.min(n - 1);
            let v = x[idx];
            sum += v;
            sum_sq += v * v;
        }
        let mean_b = sum / n as f64;
        let var_b = sum_sq / n as f64 - mean_b * mean_b;
        boot_vars.push(var_b);
    }
    // Standard error of the variance = std-dev of bootstrap-variance values.
    let mean_boot: f64 = boot_vars.iter().sum::<f64>() / n_bootstrap as f64;
    let se = (boot_vars
        .iter()
        .map(|v| (v - mean_boot).powi(2))
        .sum::<f64>()
        / (n_bootstrap as f64 - 1.0).max(1.0))
    .sqrt();
    (variance, se)
}

/// Bootstrap percentile-based confidence interval [lo, hi] at the
/// given confidence level (e.g., 0.95 for 95% CI).
pub fn bootstrap_variance_ci(
    x: &[f64],
    n_bootstrap: usize,
    seed: u64,
    confidence: f64,
) -> (f64, f64, f64) {
    let n = x.len();
    if n == 0 || n_bootstrap == 0 {
        return (0.0, 0.0, 0.0);
    }
    let mean: f64 = x.iter().sum::<f64>() / n as f64;
    let variance: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;

    let mut rng = LCG::new(seed);
    let mut boot_vars: Vec<f64> = Vec::with_capacity(n_bootstrap);
    for _ in 0..n_bootstrap {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let idx = (rng.next_f64() * n as f64) as usize;
            let idx = idx.min(n - 1);
            let v = x[idx];
            sum += v;
            sum_sq += v * v;
        }
        let mean_b = sum / n as f64;
        let var_b = sum_sq / n as f64 - mean_b * mean_b;
        boot_vars.push(var_b);
    }
    boot_vars.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let alpha = 1.0 - confidence;
    let lo_idx = ((alpha / 2.0) * n_bootstrap as f64) as usize;
    let hi_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64) as usize;
    let lo_idx = lo_idx.min(n_bootstrap - 1);
    let hi_idx = hi_idx.min(n_bootstrap - 1);
    (variance, boot_vars[lo_idx], boot_vars[hi_idx])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bootstrap_variance_recovers_known_variance() {
        // Sample of constants: variance = 0, SE = 0.
        let x = vec![5.0; 100];
        let (var, se) = bootstrap_variance(&x, 100, 42);
        assert!(var.abs() < 1e-12);
        assert!(se.abs() < 1e-12);
    }

    #[test]
    fn bootstrap_se_decreases_with_n() {
        // For an iid sample with variance 1, bootstrap SE on var
        // should be roughly proportional to 1/sqrt(n) for large n.
        let mut rng = LCG::new(7);
        let make_sample = |n: usize, rng: &mut LCG| -> Vec<f64> {
            (0..n).map(|_| rng.next_normal()).collect()
        };
        let x_small = make_sample(50, &mut rng);
        let x_large = make_sample(500, &mut rng);
        let (_, se_small) = bootstrap_variance(&x_small, 200, 1);
        let (_, se_large) = bootstrap_variance(&x_large, 200, 2);
        // Roughly SE_large < SE_small, with factor ~ sqrt(10) ~ 3.
        assert!(
            se_large < se_small,
            "expected SE to decrease: small={se_small}, large={se_large}"
        );
    }

    #[test]
    fn bootstrap_ci_contains_estimate() {
        let x: Vec<f64> = (0..100).map(|i| (i as f64) / 50.0 - 1.0).collect();
        let (var, lo, hi) = bootstrap_variance_ci(&x, 200, 11, 0.95);
        assert!(
            lo <= var && var <= hi,
            "var={var} not in CI [{lo}, {hi}]"
        );
    }
}
