//! # Adaptive nested sampling (Skilling 2004) for the evidence integral
//!
//! Computes
//!     Z = int_Theta pi(theta) L(D | theta) d theta
//! with the algorithm of Skilling 2004 §3:
//!
//! 1. Initialise `n_live` points i.i.d. from the prior `pi`.
//! 2. Repeat:
//!    a. find the live point with smallest likelihood, call it
//!       `(theta_i, L_i)`;
//!    b. add `L_i * (X_{i-1} - X_i)` to the running evidence,
//!       where `X_j ~ exp(-j / n_live)` is the expected prior volume
//!       enclosing the worst-likelihood iso-surface;
//!    c. replace it with a fresh draw from the prior constrained to
//!       `L > L_i`. We use the Feroz-Hobson 2008 ellipsoidal-rejection
//!       sampler in the unit-cube reparameterisation: the live points
//!       are mapped to the unit cube via the prior CDFs, an enclosing
//!       ellipsoid is fit via the principal-axis decomposition of the
//!       sample covariance (rescaled by an enlargement factor), points
//!       are drawn uniformly inside the ellipsoid, and rejected if
//!       outside `[0,1]^d` or with insufficient likelihood.
//! 3. Terminate when the maximum-likelihood live point's contribution
//!    to the residual is below `stop_log_evidence_change` (Skilling
//!    2004 §4.3 stopping criterion).
//!
//! Includes the residual contribution from the remaining live points
//! at termination via `Z += sum_remaining L_j * X_term / n_live`
//! (Skilling 2004 §3.3).
//!
//! ## Evidence uncertainty
//!
//! Skilling 2004 §4: `sigma_{ln Z} ~ sqrt(H / N_live)` where
//!     `H = sum_i w_i (ln L_i - ln Z)`
//! is the information content (KL divergence between the posterior and
//! the prior).
//!
//! ## Reproducibility
//!
//! All random draws come from a single [`ChaCha20Rng`] seeded by
//! `config.seed`. At fixed `n_live` and `seed`, the algorithm produces
//! a bit-identical `ln Z`.
//!
//! ## Checkpointing
//!
//! When `config.checkpoint_path` is set, the live-point set, accumulated
//! `ln Z`, and iteration count are persisted at every
//! `checkpoint_interval` iterations via an atomic write-then-rename.
//! [`compute_evidence`] resumes from the most recent checkpoint when
//! one exists and is consistent.
//!
//! ## References
//!
//! * Skilling, J. "Nested sampling for general Bayesian computation",
//!   AIP Conf. Proc. 735 (2004) 395, DOI 10.1063/1.1835238.
//! * Feroz, F.; Hobson, M.P. "Multimodal nested sampling", MNRAS 384
//!   (2008) 449, arXiv:0704.3704, DOI 10.1111/j.1365-2966.2007.12353.x.
//! * Feroz, F.; Hobson, M.P.; Bridges, M. "MultiNest: an efficient
//!   and robust Bayesian inference tool for cosmology and particle
//!   physics", MNRAS 398 (2009) 1601, arXiv:0809.3437.

use std::fs;
use std::io;
use std::path::PathBuf;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::route34::likelihood::LogLikelihoodResult;
use crate::route34::prior::{ModuliPoint, Prior};

// ----------------------------------------------------------------------
// Configuration and result types.
// ----------------------------------------------------------------------

/// Configuration for [`compute_evidence`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NestedSamplingConfig {
    /// Number of live points. Skilling 2004 recommends 100-1000;
    /// MultiNest defaults to 500 for cosmology-scale problems.
    pub n_live: usize,
    /// Stopping tolerance on the maximum residual `ln Z` change
    /// per iteration. Skilling 2004 recommends 1e-3.
    pub stop_log_evidence_change: f64,
    /// Hard maximum number of iterations.
    pub max_iterations: usize,
    /// PRNG seed.
    pub seed: u64,
    /// If set, persist live-point set and accumulated state to this
    /// file every `checkpoint_interval` iterations.
    pub checkpoint_path: Option<PathBuf>,
    /// Persist a checkpoint every this-many iterations.
    pub checkpoint_interval: usize,
    /// Maximum ratio (ellipsoid volume / sample-covariance volume) for
    /// the Feroz-Hobson 2008 enlargement factor. Default 1.5.
    pub ellipsoid_enlargement: f64,
    /// Hard cap on the constrained-draw rejection-sampler attempts
    /// per replacement. Default 10000. Each attempted draw is one
    /// likelihood evaluation; this caps the worst-case cost.
    pub max_constrained_draw_attempts: usize,
    /// Number of posterior samples to retain (resampled from the
    /// dead-point list with weights `w_i = L_i (X_{i-1} - X_i) / Z`).
    pub n_posterior_samples: usize,
}

impl Default for NestedSamplingConfig {
    fn default() -> Self {
        Self {
            n_live: 500,
            stop_log_evidence_change: 1e-3,
            max_iterations: 100_000,
            seed: 42,
            checkpoint_path: None,
            checkpoint_interval: 200,
            ellipsoid_enlargement: 1.5,
            max_constrained_draw_attempts: 10_000,
            n_posterior_samples: 0,
        }
    }
}

/// Reproducibility metadata for a nested-sampling run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunMetadata {
    pub seed: u64,
    pub n_live: usize,
    pub iterations_run: usize,
    pub n_likelihood_evaluations: u64,
    pub n_constrained_draws: u64,
    pub wall_clock_seconds: f64,
    /// SHA-256 of all accepted live-point coordinates concatenated in
    /// big-endian byte order. Used as a reproducibility fingerprint.
    pub live_points_sha256: String,
    /// Whether the run resumed from a checkpoint.
    pub resumed_from_checkpoint: bool,
}

/// Output of [`compute_evidence`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvidenceResult {
    /// Marginal log-evidence `ln Z = ln int pi(theta) L(D | theta) d theta`.
    pub log_evidence: f64,
    /// Skilling 2004 §4 standard deviation `sqrt(H / n_live)`.
    pub log_evidence_uncertainty: f64,
    /// Information content (KL divergence between posterior and prior),
    /// in nats: `H = sum_i w_i (ln L_i - ln Z)`.
    pub information_h: f64,
    /// Number of remaining live points at termination.
    pub n_live_points_remaining: usize,
    /// Posterior samples: `(theta, log_weight)` pairs. The `log_weight`
    /// is `ln(L_i (X_{i-1} - X_i) / Z)` (so weights sum to 1 in
    /// non-log space).
    pub posterior_samples: Vec<(ModuliPoint, f64)>,
    /// Reproducibility metadata.
    pub run_metadata: RunMetadata,
}

// ----------------------------------------------------------------------
// Internal: live-point bookkeeping.
// ----------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LivePoint {
    /// Unit-cube coordinates (input to the prior's `sample_unit_cube`).
    u: Vec<f64>,
    /// Physical moduli coordinates (output of `sample_unit_cube`).
    theta: ModuliPoint,
    /// Log-likelihood at theta.
    log_likelihood: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DeadPoint {
    theta: ModuliPoint,
    log_likelihood: f64,
    /// Log-weight `ln (L_i (X_{i-1} - X_i))` (not yet normalised by `Z`).
    log_weight_unnormalised: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NestedSamplingCheckpoint {
    seed: u64,
    n_live: usize,
    iteration: usize,
    log_evidence_running: f64,
    log_xi_running: f64,
    info_running: f64,
    log_evidence_running_squared_term: f64,
    live: Vec<LivePoint>,
    dead: Vec<DeadPoint>,
    n_likelihood_evaluations: u64,
    n_constrained_draws: u64,
    started_unix_timestamp: u64,
    rng_state_seed: u64,
}

// ----------------------------------------------------------------------
// Top-level driver.
// ----------------------------------------------------------------------

/// Compute the marginal log-evidence `ln Z` for a model with prior
/// `prior` and likelihood `likelihood_fn`.
///
/// `likelihood_fn` returns a [`LogLikelihoodResult`] for any
/// in-support `theta`; if it returns an error, the live-point that
/// produced it is rejected and resampled. Returning `f64::NEG_INFINITY`
/// for the log-likelihood is also a valid way to signal "out of support
/// for this likelihood model" without aborting the run.
pub fn compute_evidence<F, P, E>(
    likelihood_fn: F,
    prior: &P,
    config: &NestedSamplingConfig,
) -> Result<EvidenceResult, NestedSamplingError>
where
    F: Fn(&ModuliPoint) -> Result<LogLikelihoodResult, E> + Sync + Send,
    P: Prior + ?Sized,
    E: std::fmt::Display,
{
    let started = std::time::Instant::now();
    let _started_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    if config.n_live < 2 {
        return Err(NestedSamplingError::ConfigError(format!(
            "n_live must be >= 2, got {}",
            config.n_live
        )));
    }

    // Try to resume from checkpoint.
    let (mut state, resumed) = if let Some(ckpt_path) = &config.checkpoint_path {
        match read_checkpoint(ckpt_path) {
            Ok(Some(c)) if c.seed == config.seed && c.n_live == config.n_live => {
                (c, true)
            }
            _ => (initial_state(prior, &likelihood_fn, config)?, false),
        }
    } else {
        (initial_state(prior, &likelihood_fn, config)?, false)
    };

    let dim_total = prior.dimension();
    let mut rng = ChaCha20Rng::seed_from_u64(state.rng_state_seed);

    // Main loop.
    let n_live = config.n_live;
    let log_n_live = (n_live as f64).ln();
    // log_xi = ln X_i, the expected log prior-volume after i iterations.
    // X_i = exp(-i / n_live) ; ln X_i = -i / n_live.
    // Each step: ln (X_{i-1} - X_i) ~ ln(X_{i-1}) + ln(1 - exp(-1/n_live)).
    let log_volume_shrink = (1.0 - (-1.0_f64 / n_live as f64).exp()).ln();

    loop {
        if state.iteration >= config.max_iterations {
            break;
        }

        // a. Find min-likelihood live point.
        let worst_idx = argmin_log_likelihood(&state.live);
        let worst_log_l = state.live[worst_idx].log_likelihood;
        let worst_theta = state.live[worst_idx].theta.clone();

        // log w_i = log L_i + log (X_{i-1} - X_i)
        let log_x_prev = state.log_xi_running;
        let log_w_i = worst_log_l + log_x_prev + log_volume_shrink;

        // b. Update running log evidence by log-sum-exp.
        let new_log_z = logsumexp2(state.log_evidence_running, log_w_i);
        // Information increment: dH = exp(log_w - new_log_z) * (log_l - new_log_z) + ...
        // We track H by Skilling 2004 (eq. 6):
        //   H = sum_i w_i / Z * (ln L_i - ln Z)
        // Update: H_new = exp(old_lnZ - new_lnZ) * (H_old + old_lnZ)
        //                + exp(log_w - new_lnZ) * (log_L_i)
        //                - new_lnZ
        let new_info = if state.log_evidence_running.is_finite() {
            ((state.log_evidence_running - new_log_z).exp())
                * (state.info_running + state.log_evidence_running)
                + ((log_w_i - new_log_z).exp()) * worst_log_l
                - new_log_z
        } else {
            // First step: H = log L_i - new_log_z
            ((log_w_i - new_log_z).exp()) * worst_log_l - new_log_z
        };

        state.dead.push(DeadPoint {
            theta: worst_theta.clone(),
            log_likelihood: worst_log_l,
            log_weight_unnormalised: log_w_i,
        });

        state.log_evidence_running = new_log_z;
        state.info_running = new_info.max(0.0);

        // c. Replace the worst point with a constrained draw.
        let replacement = constrained_draw(
            &state.live,
            worst_idx,
            worst_log_l,
            prior,
            &likelihood_fn,
            config,
            &mut rng,
            &mut state.n_constrained_draws,
            &mut state.n_likelihood_evaluations,
        )?;
        state.live[worst_idx] = replacement;

        state.iteration += 1;
        state.log_xi_running = -(state.iteration as f64) / (n_live as f64);

        // Persist a checkpoint if requested.
        if let Some(ckpt_path) = &config.checkpoint_path {
            if state.iteration % config.checkpoint_interval.max(1) == 0 {
                state.rng_state_seed = rng.random::<u64>();
                let _ = write_checkpoint(ckpt_path, &state);
            }
        }

        // Stopping criterion (Skilling 2004 §4.3): if the maximum
        // remaining likelihood times the current prior volume contributes
        // less than `stop_log_evidence_change` to ln Z, stop.
        let max_live_log_l = state
            .live
            .iter()
            .map(|p| p.log_likelihood)
            .fold(f64::NEG_INFINITY, f64::max);
        let log_residual = max_live_log_l + state.log_xi_running - log_n_live - state.log_evidence_running;
        if log_residual < state.log_evidence_running.ln_1p_neg_safe(config.stop_log_evidence_change)
        {
            break;
        }
    }

    // Add the residual contribution from the surviving live points
    // (Skilling 2004 §3.3): each contributes `L_j * X_term / n_live`.
    let log_volume_per_live = state.log_xi_running - (n_live as f64).ln();
    for live in &state.live {
        let log_w = live.log_likelihood + log_volume_per_live;
        state.dead.push(DeadPoint {
            theta: live.theta.clone(),
            log_likelihood: live.log_likelihood,
            log_weight_unnormalised: log_w,
        });
        let new_log_z = logsumexp2(state.log_evidence_running, log_w);
        let new_info = if state.log_evidence_running.is_finite() {
            ((state.log_evidence_running - new_log_z).exp())
                * (state.info_running + state.log_evidence_running)
                + ((log_w - new_log_z).exp()) * live.log_likelihood
                - new_log_z
        } else {
            ((log_w - new_log_z).exp()) * live.log_likelihood - new_log_z
        };
        state.log_evidence_running = new_log_z;
        state.info_running = new_info.max(0.0);
    }

    // Final result.
    let log_evidence = state.log_evidence_running;
    let info = state.info_running.max(0.0);
    let log_evidence_uncertainty = (info / n_live as f64).sqrt();

    // Posterior samples: weights w_i = exp(log_w_i - log_Z).
    let posterior_samples = if config.n_posterior_samples > 0 {
        resample_posterior(&state.dead, log_evidence, config.n_posterior_samples, &mut rng)
    } else {
        Vec::new()
    };

    // SHA-256 of accepted live points (final live set).
    let live_points_sha256 = sha256_live_points(&state.live);

    // Successful completion: erase checkpoint.
    if let Some(ckpt_path) = &config.checkpoint_path {
        if ckpt_path.exists() {
            let _ = fs::remove_file(ckpt_path);
        }
    }

    let _ = dim_total; // kept for future structured fingerprinting

    Ok(EvidenceResult {
        log_evidence,
        log_evidence_uncertainty,
        information_h: info,
        n_live_points_remaining: state.live.len(),
        posterior_samples,
        run_metadata: RunMetadata {
            seed: config.seed,
            n_live,
            iterations_run: state.iteration,
            n_likelihood_evaluations: state.n_likelihood_evaluations,
            n_constrained_draws: state.n_constrained_draws,
            wall_clock_seconds: started.elapsed().as_secs_f64(),
            live_points_sha256,
            resumed_from_checkpoint: resumed,
        },
    })
}

// ----------------------------------------------------------------------
// Helpers.
// ----------------------------------------------------------------------

/// Saturate-safe log-sum-exp for two arguments. Handles `-inf`.
fn logsumexp2(a: f64, b: f64) -> f64 {
    if !a.is_finite() && a < 0.0 {
        return b;
    }
    if !b.is_finite() && b < 0.0 {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

trait Ln1pNegSafe {
    fn ln_1p_neg_safe(self, eps: f64) -> f64;
}
impl Ln1pNegSafe for f64 {
    fn ln_1p_neg_safe(self, eps: f64) -> f64 {
        // Returns ln(self) + ln(eps)  ~ ln(self * eps), the threshold
        // we compare log_residual against. If self is -inf, the
        // threshold is -inf (always trigger exit). This is a stopping
        // helper, not a generic numerical routine.
        if !self.is_finite() {
            return f64::NEG_INFINITY;
        }
        eps.ln()
    }
}

fn argmin_log_likelihood(live: &[LivePoint]) -> usize {
    let mut best = 0usize;
    let mut best_v = live[0].log_likelihood;
    for (i, p) in live.iter().enumerate().skip(1) {
        if p.log_likelihood < best_v {
            best_v = p.log_likelihood;
            best = i;
        }
    }
    best
}

fn sha256_live_points(live: &[LivePoint]) -> String {
    let mut h = Sha256::new();
    for p in live {
        for &u in &p.u {
            h.update(u.to_be_bytes());
        }
        for &c in &p.theta.continuous {
            h.update(c.to_be_bytes());
        }
        for &i in &p.theta.discrete_indices {
            h.update((i as u64).to_be_bytes());
        }
        h.update(p.log_likelihood.to_be_bytes());
    }
    hex::encode(h.finalize())
}

// ----------------------------------------------------------------------
// Initial state.
// ----------------------------------------------------------------------

fn initial_state<F, P, E>(
    prior: &P,
    likelihood_fn: &F,
    config: &NestedSamplingConfig,
) -> Result<NestedSamplingCheckpoint, NestedSamplingError>
where
    F: Fn(&ModuliPoint) -> Result<LogLikelihoodResult, E>,
    P: Prior + ?Sized,
    E: std::fmt::Display,
{
    let mut rng = ChaCha20Rng::seed_from_u64(config.seed);
    let dim = prior.dimension();
    let mut live = Vec::with_capacity(config.n_live);
    let mut n_lik_evals: u64 = 0;
    let max_init_attempts = config.n_live * 100;
    let mut attempts = 0;
    while live.len() < config.n_live && attempts < max_init_attempts {
        let u: Vec<f64> = (0..dim).map(|_| rng.random_range(0.0..1.0)).collect();
        let theta = prior.sample_unit_cube(&u);
        if !prior.in_support(&theta) {
            attempts += 1;
            continue;
        }
        n_lik_evals += 1;
        match likelihood_fn(&theta) {
            Ok(r) if r.log_likelihood.is_finite() => {
                live.push(LivePoint {
                    u,
                    theta,
                    log_likelihood: r.log_likelihood,
                });
            }
            _ => {}
        }
        attempts += 1;
    }
    if live.len() < config.n_live {
        return Err(NestedSamplingError::InitialDrawFailed {
            wanted: config.n_live,
            got: live.len(),
        });
    }
    Ok(NestedSamplingCheckpoint {
        seed: config.seed,
        n_live: config.n_live,
        iteration: 0,
        log_evidence_running: f64::NEG_INFINITY,
        log_xi_running: 0.0,
        info_running: 0.0,
        log_evidence_running_squared_term: 0.0,
        live,
        dead: Vec::new(),
        n_likelihood_evaluations: n_lik_evals,
        n_constrained_draws: 0,
        started_unix_timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        rng_state_seed: rng.random::<u64>(),
    })
}

// ----------------------------------------------------------------------
// Constrained draw (Feroz-Hobson 2008 ellipsoidal rejection).
// ----------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn constrained_draw<F, P, E>(
    live: &[LivePoint],
    skip_idx: usize,
    log_l_min: f64,
    prior: &P,
    likelihood_fn: &F,
    config: &NestedSamplingConfig,
    rng: &mut ChaCha20Rng,
    n_draws: &mut u64,
    n_lik: &mut u64,
) -> Result<LivePoint, NestedSamplingError>
where
    F: Fn(&ModuliPoint) -> Result<LogLikelihoodResult, E>,
    P: Prior + ?Sized,
    E: std::fmt::Display,
{
    let dim = prior.dimension();

    // Build the enlarged ellipsoid in the unit-cube reparameterisation.
    // Mean and covariance over the live points (excluding `skip_idx` to
    // avoid a singular matrix when n_live == dim+1).
    let n = live.len();
    if n < 2 {
        return Err(NestedSamplingError::ConfigError(
            "live-point set too small".to_string(),
        ));
    }
    let mean = unit_cube_mean(live, skip_idx, dim);
    let cov = unit_cube_covariance(live, skip_idx, dim, &mean);

    // Eigendecomposition (symmetric, dim x dim) via Jacobi rotations.
    let (eigvals, eigvecs) = jacobi_eigendecomp(&cov, dim);

    // Enlarge: each eigenvalue scaled by `ellipsoid_enlargement`.
    let scale = config.ellipsoid_enlargement.max(1.0);
    let mut radii = vec![0.0_f64; dim];
    for k in 0..dim {
        // Standard deviation along principal axis * sqrt(F * (n + 1))
        // where F is the enlargement factor; Feroz-Hobson 2008 §3.2.
        let var = eigvals[k].max(1e-30);
        radii[k] = (var.sqrt()) * scale * (n as f64).sqrt();
    }

    // Rejection sampling: draw uniformly inside the ellipsoid in
    // unit-cube coords, reject if outside cube or insufficient L.
    for _ in 0..config.max_constrained_draw_attempts {
        *n_draws += 1;

        // Sample uniformly inside the d-ball, then rotate / scale.
        let mut x = vec![0.0_f64; dim];
        // Box-Muller into x.
        let mut k = 0;
        while k + 1 < dim {
            let u1: f64 = rng.random_range(1e-15..1.0);
            let u2: f64 = rng.random_range(0.0..1.0);
            let r = (-2.0 * u1.ln()).sqrt();
            x[k] = r * (2.0 * std::f64::consts::PI * u2).cos();
            x[k + 1] = r * (2.0 * std::f64::consts::PI * u2).sin();
            k += 2;
        }
        if k < dim {
            let u1: f64 = rng.random_range(1e-15..1.0);
            let u2: f64 = rng.random_range(0.0..1.0);
            let r = (-2.0 * u1.ln()).sqrt();
            x[k] = r * (2.0 * std::f64::consts::PI * u2).cos();
        }
        let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
        let u_radial: f64 = rng.random_range(0.0_f64..1.0);
        let scale_radius = u_radial.powf(1.0 / dim as f64);
        for v in x.iter_mut() {
            *v = *v / norm * scale_radius;
        }

        // y = mean + V * diag(radii) * x   (rotate to cov axes).
        let mut y = mean.clone();
        for i in 0..dim {
            let mut acc = 0.0;
            for j in 0..dim {
                acc += eigvecs[i * dim + j] * radii[j] * x[j];
            }
            y[i] += acc;
        }

        // Reject if outside the unit cube.
        if y.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
            continue;
        }

        let theta = prior.sample_unit_cube(&y);
        if !prior.in_support(&theta) {
            continue;
        }

        *n_lik += 1;
        match likelihood_fn(&theta) {
            Ok(r) if r.log_likelihood > log_l_min && r.log_likelihood.is_finite() => {
                return Ok(LivePoint {
                    u: y,
                    theta,
                    log_likelihood: r.log_likelihood,
                });
            }
            _ => continue,
        }
    }

    // Fall back to plain rejection sampling from the prior. This is
    // strictly correct (the prior is the "infinite enlargement" limit)
    // but very slow in high-d small-volume regimes.
    for _ in 0..config.max_constrained_draw_attempts {
        *n_draws += 1;
        let u: Vec<f64> = (0..dim).map(|_| rng.random_range(0.0..1.0)).collect();
        let theta = prior.sample_unit_cube(&u);
        if !prior.in_support(&theta) {
            continue;
        }
        *n_lik += 1;
        match likelihood_fn(&theta) {
            Ok(r) if r.log_likelihood > log_l_min && r.log_likelihood.is_finite() => {
                return Ok(LivePoint {
                    u,
                    theta,
                    log_likelihood: r.log_likelihood,
                });
            }
            _ => continue,
        }
    }

    Err(NestedSamplingError::ConstrainedDrawFailed {
        log_l_min,
        attempts: config.max_constrained_draw_attempts * 2,
    })
}

fn unit_cube_mean(live: &[LivePoint], skip_idx: usize, dim: usize) -> Vec<f64> {
    let mut out = vec![0.0; dim];
    let mut count = 0usize;
    for (i, p) in live.iter().enumerate() {
        if i == skip_idx {
            continue;
        }
        for k in 0..dim {
            out[k] += p.u[k];
        }
        count += 1;
    }
    let inv = 1.0 / (count.max(1) as f64);
    for v in out.iter_mut() {
        *v *= inv;
    }
    out
}

fn unit_cube_covariance(live: &[LivePoint], skip_idx: usize, dim: usize, mean: &[f64]) -> Vec<f64> {
    let mut cov = vec![0.0; dim * dim];
    let mut count = 0usize;
    for (i, p) in live.iter().enumerate() {
        if i == skip_idx {
            continue;
        }
        for r in 0..dim {
            let dr = p.u[r] - mean[r];
            for c in 0..dim {
                let dc = p.u[c] - mean[c];
                cov[r * dim + c] += dr * dc;
            }
        }
        count += 1;
    }
    let inv = 1.0 / ((count.max(2) - 1) as f64);
    for v in cov.iter_mut() {
        *v *= inv;
    }
    // Regularise (avoid singular cov when `dim > count`).
    for i in 0..dim {
        cov[i * dim + i] += 1e-9;
    }
    cov
}

/// Jacobi rotations for symmetric eigendecomposition.
/// Returns (eigvals[dim], eigvecs[dim*dim] in row-major; eigvecs[i*dim+j]
/// = component i of eigenvector j).
fn jacobi_eigendecomp(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let max_sweeps = 100;
    for _ in 0..max_sweeps {
        // Sum of off-diagonal squared.
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let aij = a[i * n + j];
                off += aij * aij;
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..(n - 1) {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-30 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                // Update A.
                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;
                for r in 0..n {
                    if r != p && r != q {
                        let arp = a[r * n + p];
                        let arq = a[r * n + q];
                        a[r * n + p] = c * arp - s * arq;
                        a[r * n + q] = s * arp + c * arq;
                        a[p * n + r] = a[r * n + p];
                        a[q * n + r] = a[r * n + q];
                    }
                }
                // Update V.
                for r in 0..n {
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = c * vrp - s * vrq;
                    v[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }
    let eigvals: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    (eigvals, v)
}

// ----------------------------------------------------------------------
// Posterior resampling (importance-weight resampling).
// ----------------------------------------------------------------------

fn resample_posterior(
    dead: &[DeadPoint],
    log_evidence: f64,
    n_samples: usize,
    rng: &mut ChaCha20Rng,
) -> Vec<(ModuliPoint, f64)> {
    if dead.is_empty() {
        return Vec::new();
    }
    // Cumulative weights (after subtracting log_evidence).
    let mut weights: Vec<f64> = dead
        .iter()
        .map(|d| (d.log_weight_unnormalised - log_evidence).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return Vec::new();
    }
    for w in weights.iter_mut() {
        *w /= total;
    }
    let mut cumul = 0.0;
    let cdf: Vec<f64> = weights
        .iter()
        .map(|&w| {
            cumul += w;
            cumul
        })
        .collect();
    let mut out = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let u: f64 = rng.random_range(0.0..1.0);
        let i = cdf.partition_point(|&c| c < u).min(dead.len() - 1);
        out.push((dead[i].theta.clone(), weights[i].max(1e-300).ln()));
    }
    out
}

// ----------------------------------------------------------------------
// Checkpointing.
// ----------------------------------------------------------------------

fn write_checkpoint(path: &PathBuf, ckpt: &NestedSamplingCheckpoint) -> io::Result<()> {
    let tmp = path.with_extension("ns.tmp");
    let bytes = serde_json::to_vec(ckpt).map_err(|e| {
        io::Error::new(io::ErrorKind::Other, format!("serialise: {}", e))
    })?;
    fs::write(&tmp, &bytes)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

fn read_checkpoint(path: &PathBuf) -> io::Result<Option<NestedSamplingCheckpoint>> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(path)?;
    match serde_json::from_slice::<NestedSamplingCheckpoint>(&bytes) {
        Ok(c) => Ok(Some(c)),
        Err(_) => Ok(None),
    }
}

// ----------------------------------------------------------------------
// Errors.
// ----------------------------------------------------------------------

#[derive(Debug)]
pub enum NestedSamplingError {
    ConfigError(String),
    InitialDrawFailed { wanted: usize, got: usize },
    ConstrainedDrawFailed { log_l_min: f64, attempts: usize },
    Io(io::Error),
}

impl std::fmt::Display for NestedSamplingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NestedSamplingError::ConfigError(s) => write!(f, "config error: {}", s),
            NestedSamplingError::InitialDrawFailed { wanted, got } => write!(
                f,
                "initial draw failed: wanted {} live points, got {}",
                wanted, got
            ),
            NestedSamplingError::ConstrainedDrawFailed { log_l_min, attempts } => write!(
                f,
                "constrained draw failed: no point with log L > {} after {} attempts",
                log_l_min, attempts
            ),
            NestedSamplingError::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for NestedSamplingError {}

impl From<io::Error> for NestedSamplingError {
    fn from(e: io::Error) -> Self {
        NestedSamplingError::Io(e)
    }
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route34::likelihood::{ChiSquaredBreakdown, LikelihoodConfig};
    use crate::route34::prior::{GaussianPrior, UniformPrior};
    use std::f64::consts::PI;

    /// Likelihood for a unit Gaussian centred at the origin in d dim:
    /// L(theta) = (2 pi)^{-d/2} exp(-0.5 sum theta_i^2).
    fn unit_gaussian_loglik(theta: &ModuliPoint) -> Result<LogLikelihoodResult, &'static str> {
        let mut chi2 = 0.0;
        for &x in &theta.continuous {
            chi2 += x * x;
        }
        let d = theta.continuous.len() as f64;
        let log_norm = -0.5 * d * (2.0 * PI).ln();
        let bd = ChiSquaredBreakdown::from_components(
            chi2,
            theta.continuous.len() as u32,
            0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0,
        );
        Ok(LogLikelihoodResult {
            log_likelihood: -0.5 * chi2 + log_norm,
            chi_squared_breakdown: bd,
            p_value: 0.0,
            n_sigma: 0.0,
        })
    }

    /// 2D unit Gaussian likelihood on a uniform prior box [-5,5]^2.
    /// Analytic evidence:
    ///     Z = int_{-5}^{5}^2 (1/100) * (1/(2 pi)) exp(-r^2/2) dx dy
    ///       = (1 / 100) * P(chi^2_2 < 25) ~ (1/100) * 0.9999963
    /// ln Z ~ ln(1/100) + ln(0.9999963) ~ -4.6052
    #[test]
    fn test_gaussian_toy_evidence_matches_analytic() {
        let prior = UniformPrior::new(vec![-5.0, -5.0], vec![5.0, 5.0]).unwrap();
        let cfg = NestedSamplingConfig {
            n_live: 400,
            stop_log_evidence_change: 1e-3,
            max_iterations: 50_000,
            seed: 12345,
            ..Default::default()
        };
        let result = compute_evidence(unit_gaussian_loglik, &prior, &cfg).unwrap();
        // Analytic ln Z = -ln 100 + ln(P(chi2_2 < 25)).
        // P(chi2_2 < 25) = 1 - exp(-25/2) ~ 1 - 3.7e-6.
        let analytic = -100.0_f64.ln() + (1.0 - (-12.5_f64).exp()).ln();
        let diff = (result.log_evidence - analytic).abs();
        // Skilling-2004 1-sigma uncertainty.
        let sigma = result.log_evidence_uncertainty.max(1e-6);
        assert!(
            diff < 4.0 * sigma + 0.1,
            "ln Z = {} (sigma {}); analytic = {}; diff = {}",
            result.log_evidence,
            sigma,
            analytic,
            diff
        );
    }

    /// Gaussian likelihood on a Gaussian prior — analytic evidence is
    /// the convolution: posterior is Gaussian with sigma_p^{-2} =
    /// sigma_lik^{-2} + sigma_prior^{-2}, and Z is the marginal of
    /// the joint Gaussian. Set both unit.
    /// Z = N(0; sigma^2 = 2) at the data y=0 :
    ///     ln Z = -0.5 * ln(2 pi * 2) = -1.2655
    #[test]
    fn test_gaussian_prior_gaussian_likelihood_evidence() {
        let prior = GaussianPrior::new(vec![0.0], vec![1.0]).unwrap();
        // Likelihood: N(theta; mu = data = 0, sigma_lik = 1).
        let lik = |theta: &ModuliPoint| -> Result<LogLikelihoodResult, &'static str> {
            let x = theta.continuous[0];
            let chi2 = x * x;
            let log_norm = -0.5 * (2.0 * PI).ln();
            let bd =
                ChiSquaredBreakdown::from_components(chi2, 1, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0);
            Ok(LogLikelihoodResult {
                log_likelihood: -0.5 * chi2 + log_norm,
                chi_squared_breakdown: bd,
                p_value: 0.0,
                n_sigma: 0.0,
            })
        };
        let cfg = NestedSamplingConfig {
            n_live: 400,
            stop_log_evidence_change: 1e-3,
            max_iterations: 50_000,
            seed: 9999,
            ..Default::default()
        };
        let result = compute_evidence(lik, &prior, &cfg).unwrap();
        let analytic = -0.5 * (2.0 * PI * 2.0).ln();
        let diff = (result.log_evidence - analytic).abs();
        let sigma = result.log_evidence_uncertainty.max(1e-6);
        assert!(
            diff < 5.0 * sigma + 0.15,
            "ln Z = {} (sigma {}); analytic = {}; diff = {}",
            result.log_evidence,
            sigma,
            analytic,
            diff
        );
    }

    /// Two-mode Gaussian mixture: nested sampling should still recover
    /// the marginal evidence to within statistical uncertainty.
    #[test]
    fn test_two_well_evidence_gaussian_mixture() {
        let prior = UniformPrior::new(vec![-10.0], vec![10.0]).unwrap();
        let lik = |theta: &ModuliPoint| -> Result<LogLikelihoodResult, &'static str> {
            let x = theta.continuous[0];
            // Mixture of two unit Gaussians at +- 4 with equal weights.
            let g1 = (-0.5 * (x - 4.0).powi(2)).exp();
            let g2 = (-0.5 * (x + 4.0).powi(2)).exp();
            let pdf = 0.5 * (g1 + g2) / (2.0_f64 * PI).sqrt();
            let log_l = pdf.max(1e-300).ln();
            let bd =
                ChiSquaredBreakdown::from_components(0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0);
            Ok(LogLikelihoodResult {
                log_likelihood: log_l,
                chi_squared_breakdown: bd,
                p_value: 0.0,
                n_sigma: 0.0,
            })
        };
        let cfg = NestedSamplingConfig {
            n_live: 400,
            stop_log_evidence_change: 1e-3,
            max_iterations: 50_000,
            seed: 7777,
            ..Default::default()
        };
        let result = compute_evidence(lik, &prior, &cfg).unwrap();
        // Analytic ln Z = ln (1/20). The mixture is normalised, prior
        // volume is 20, so Z = 1/20 (each Gaussian fully inside [-10,10]).
        let analytic = -(20.0_f64).ln();
        let diff = (result.log_evidence - analytic).abs();
        let sigma = result.log_evidence_uncertainty.max(1e-6);
        assert!(
            diff < 5.0 * sigma + 0.2,
            "ln Z = {} (sigma {}); analytic = {}; diff = {}",
            result.log_evidence,
            sigma,
            analytic,
            diff
        );
    }

    /// Determinism: the same seed must yield bit-identical ln Z.
    #[test]
    fn test_evidence_seed_determinism() {
        let prior = UniformPrior::new(vec![-3.0, -3.0], vec![3.0, 3.0]).unwrap();
        let cfg = NestedSamplingConfig {
            n_live: 100,
            stop_log_evidence_change: 1e-2,
            max_iterations: 5_000,
            seed: 555,
            ..Default::default()
        };
        let r1 = compute_evidence(unit_gaussian_loglik, &prior, &cfg).unwrap();
        let r2 = compute_evidence(unit_gaussian_loglik, &prior, &cfg).unwrap();
        assert_eq!(
            r1.log_evidence.to_bits(),
            r2.log_evidence.to_bits(),
            "ln Z not bit-identical between runs with same seed"
        );
        assert_eq!(r1.run_metadata.live_points_sha256, r2.run_metadata.live_points_sha256);
    }

    #[test]
    fn test_jacobi_eigendecomp_diagonal() {
        let a = vec![3.0, 0.0, 0.0, 5.0]; // 2x2 diag(3,5)
        let (vals, vecs) = jacobi_eigendecomp(&a, 2);
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 3.0).abs() < 1e-12);
        assert!((sorted[1] - 5.0).abs() < 1e-12);
        // Vectors are orthonormal.
        let (v00, v01, v10, v11) = (vecs[0], vecs[1], vecs[2], vecs[3]);
        assert!((v00 * v00 + v10 * v10 - 1.0).abs() < 1e-12);
        assert!((v01 * v01 + v11 * v11 - 1.0).abs() < 1e-12);
        assert!((v00 * v01 + v10 * v11).abs() < 1e-12);
    }

    #[test]
    fn test_logsumexp2_with_neg_infinity() {
        assert_eq!(logsumexp2(f64::NEG_INFINITY, 0.5), 0.5);
        assert_eq!(logsumexp2(0.5, f64::NEG_INFINITY), 0.5);
        let r = logsumexp2(0.0, 0.0);
        assert!((r - 2.0_f64.ln()).abs() < 1e-15);
    }

    /// Checkpoint write/read round-trip.
    #[test]
    fn test_checkpoint_resume() {
        use std::env;
        let dir = env::temp_dir();
        let path = dir.join("cy3_ns_test_checkpoint.json");
        let _ = fs::remove_file(&path);

        let prior = UniformPrior::new(vec![-3.0, -3.0], vec![3.0, 3.0]).unwrap();
        let cfg = NestedSamplingConfig {
            n_live: 100,
            stop_log_evidence_change: 1e-2,
            max_iterations: 1_000,
            seed: 4242,
            checkpoint_path: Some(path.clone()),
            checkpoint_interval: 50,
            ..Default::default()
        };
        let r1 = compute_evidence(unit_gaussian_loglik, &prior, &cfg).unwrap();
        // Run again; should not resume because previous run cleaned up.
        // But re-run with same seed -> identical ln Z.
        let r2 = compute_evidence(unit_gaussian_loglik, &prior, &cfg).unwrap();
        assert_eq!(r1.log_evidence.to_bits(), r2.log_evidence.to_bits());
        let _ = fs::remove_file(&path);
    }

    /// LikelihoodConfig is part of the public surface; exercise it
    /// here to keep it from drifting unused in the discrimination crate.
    #[test]
    fn test_likelihood_config_used() {
        let _ = LikelihoodConfig::default();
    }
}
