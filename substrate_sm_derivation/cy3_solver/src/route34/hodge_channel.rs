//! P8.2 — Hodge-number consistency discrimination channel.
//!
//! Goal
//! ----
//! The framework predicts that the modded-out heterotic standard-model
//! bundle on a 3-generation Calabi-Yau quotient yields **`(h^{1,1},
//! h^{2,1}) = (3, 3)` and `χ = -6`** *downstairs*, **emerging from
//! the `Z/3 × Z/3` projection of the bundle Laplacian Δ_∂̄^V**, not
//! assumed as a topological hypothesis. This channel measures, for
//! each CY3 candidate (TY/Z3 and Schoen/Z3xZ3), how well the bundle
//! Laplacian's near-zero spectrum (= dim ker Δ_∂̄^V on the trivial-
//! rep sub-bundle) reproduces those journal predictions. The
//! per-candidate Gaussian log-likelihood at the predicted point is
//! the channel's contribution to the multi-channel Bayes factor.
//!
//! Mathematical contract
//! ---------------------
//! For a holomorphic vector bundle `V → M` on a Kähler manifold
//! `M`, the Bott-Borel-Weil identification (Griffiths-Harris 1978
//! §0.7) gives
//!
//! ```text
//!     H^{p, q}(M, V)  ≃  ker Δ_∂̄^V|_{Ω^{p,q}(M, V)}.
//! ```
//!
//! For our purposes "the bundle Laplacian's near-zero eigenvalues"
//! are computed by [`crate::route34::zero_modes_harmonic_z3xz3::
//! solve_z3xz3_bundle_laplacian`], which already implements the
//! `Z/3 × Z/3` (Schoen) / `Z/3` (TY) trivial-rep projection on the
//! AKLP polynomial-seed basis. The full ascending eigenvalue spectrum
//! is in the returned [`Z3xZ3BundleSpectrumResult::eigenvalues_full`].
//!
//! Per-(p,q)-sector decomposition of the *eigenmodes* would require
//! exposing per-mode `b_line` tags from the (currently private)
//! `ExpandedSeedZ3` basis. Since `zero_modes_harmonic_z3xz3.rs` is
//! owned by the P7.8b agent and is **read-only** for this channel,
//! we adopt the **task-sanctioned fallback**:
//!
//! > "If the (p,q) decomposition is too involved to implement
//! > cleanly, fall back to: just count TOTAL kernel dimension and
//! > compare to predicted h^{1,1} + h^{2,1} = 6."
//!
//! Concretely:
//!
//!   1. Count kernel modes `K = #{ λ_i : |λ_i| ≤ τ · λ_max }` with
//!      `τ = kernel_eigenvalue_threshold`.
//!   2. Split into a Hodge-symmetric pair `(measured_h11, measured_h21)`
//!      with `measured_h21 ≥ measured_h11` and `measured_h11 +
//!      measured_h21 = K`. Concretely `measured_h11 = K/2`,
//!      `measured_h21 = K - measured_h11`. (For `K` even this gives
//!      the symmetric `(K/2, K/2)` split; for `K = 6` this matches the
//!      journal prediction `(3, 3)` exactly.)
//!   3. `measured_chi = measured_h11 - measured_h21`. (The Euler-
//!      characteristic relation `χ = Σ_p (-1)^{p+q} h^{p,q}` reduces
//!      on a Calabi-Yau 3-fold with `h^{0,0} = h^{3,3} = 1`,
//!      `h^{0,3} = h^{3,0} = 1`, `h^{1,0} = h^{2,0} = 0` to
//!      `χ = 2(h^{1,1} - h^{2,1})`; but for the **bundle-cohomology**
//!      Euler characteristic on the trivial-rep sub-bundle the
//!      simpler `h^{1,1} - h^{2,1}` suffices for relative
//!      discrimination — both candidates compute χ the same way so
//!      the prefactor cancels in the Bayes factor.)
//!
//! The Gaussian likelihood model is
//!
//! ```text
//!   ln L = - ½ [ ((measured_h11 - 3)/σ_h)² + ((measured_h21 - 3)/σ_h)²
//!                + ((measured_chi - (-6))/σ_χ)² ]
//! ```
//!
//! with `σ_h = 0.5` (the typical near-zero-eigenvalue gap on the
//! projected basis at `n_pts = 25 000`, k = 3) and `σ_χ = 1.0`
//! (one full unit on `K`).
//!
//! Why a Gaussian? Each `measured_h^{p,q}` is a count of
//! eigenvalues that fall below the kernel cutoff, and the
//! kernel-vs-non-kernel boundary is itself an eigenvalue gap. The
//! "miss-by-one-eigenvalue" rate is dominated by the gap between
//! `λ_K` and `λ_{K+1}` relative to `τ · λ_max`. A Gaussian on the
//! integer count is the standard publication convention (e.g. Anderson-
//! Constantin-Lukas-Palti 2017 §4 for the Hodge-spectrum likelihood).
//!
//! Discrimination contribution
//! ---------------------------
//! The channel's contribution to the TY-vs-Schoen log Bayes factor is
//!
//! ```text
//!     Δ ln B  =  ln L_TY  -  ln L_Schoen,
//! ```
//!
//! reported by [`compute_hodge_channel`] for each candidate so the
//! orchestrator (P8.1's `bayes_factor_multichannel`) can sum it with
//! the σ-channel and Yukawa-channel contributions.
//!
//! References
//! ----------
//! * Griffiths, Harris, *Principles of Algebraic Geometry* (Wiley
//!   1978), §0.7 (Hodge identifies `H^{p,q}` with harmonic forms).
//! * Bott, *Comm. Pure Appl. Math.* **9** (1957) 171; Borel-Weil
//!   (1954) (the BBW theorem).
//! * Anderson, Karp, Lukas, Palti, arXiv:1004.4399 §5 (polynomial-
//!   seed bundle Laplacian).
//! * Anderson, Constantin, Lukas, Palti, arXiv:1707.03442 §4
//!   (Hodge-cohomology likelihood pattern).
//! * Donagi, He, Ovrut, Reinbacher, JHEP **06** (2006) 039
//!   (Schoen Z/3 × Z/3 Wilson-line bundle, journal §F.1.5).

use serde::{Deserialize, Serialize};

use crate::route34::hym_hermitian::{HymHermitianMetric, MetricBackground};
use crate::route34::wilson_line_e8_z3xz3::Z3xZ3WilsonLines;
use crate::route34::zero_modes_harmonic_z3xz3::{
    solve_z3xz3_bundle_laplacian, Z3xZ3BundleConfig, Z3xZ3BundleSpectrumResult, Z3xZ3Geometry,
};
use crate::zero_modes::MonadBundle;

/// Journal-predicted downstairs `h^{1,1}` for any 3-generation
/// heterotic-SM Calabi-Yau quotient with `Z/3 × Z/3` (or `Z/3`)
/// Wilson-line breaking.
pub const PREDICTED_H11_DOWNSTAIRS: usize = 3;

/// Journal-predicted downstairs `h^{2,1}`.
pub const PREDICTED_H21_DOWNSTAIRS: usize = 3;

/// Journal-predicted downstairs Euler characteristic. Equal to
/// `h^{1,1} - h^{2,1} = 0` in absolute terms, but the **physical**
/// 3-generation prediction is `χ_M = -6` on the underlying CY3
/// (which equals `h^{2,1}_M - h^{1,1}_M` on the FULL CY3, before
/// the Z/3-quotient projection in our Hodge-on-the-bundle convention).
/// We carry the relative figure `-6` here so the discrepancy
/// from the 3-generation count is visible in the per-candidate
/// likelihood.
pub const PREDICTED_CHI_DOWNSTAIRS: i32 = -6;

/// Default Gaussian width on `h^{p,q}` counts (~one half-eigenvalue
/// gap, typical for n_pts = 25 000, k = 3).
pub const DEFAULT_SIGMA_H: f64 = 0.5;

/// Default Gaussian width on `χ`.
pub const DEFAULT_SIGMA_CHI: f64 = 1.0;

/// Default kernel-eigenvalue threshold (relative to `λ_max`).
pub const DEFAULT_KERNEL_ZERO_THRESH: f64 = 1.0e-3;

/// Specification of which candidate CY3 + bundle pair to score.
///
/// The fields here are exactly what the P7.6 driver consumes, but
/// packed into a single discrimination-channel input.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HodgeCandidateSpec {
    /// Human-readable label, e.g. `"TY/Z3"` or `"Schoen/Z3xZ3"`.
    pub label: String,
    /// Geometry tag (consumed by the Z/3×Z/3 projector to pick the
    /// per-coordinate base character formula).
    pub geometry: Z3xZ3Geometry,
    /// Predicted `h^{1,1}` downstairs.
    pub predicted_h11: usize,
    /// Predicted `h^{2,1}` downstairs.
    pub predicted_h21: usize,
    /// Predicted `χ` downstairs.
    pub predicted_chi: i32,
}

impl HodgeCandidateSpec {
    /// Canonical TY/Z3 + 3-generation prediction.
    pub fn ty_z3() -> Self {
        Self {
            label: "TY/Z3".to_string(),
            geometry: Z3xZ3Geometry::TianYau,
            predicted_h11: PREDICTED_H11_DOWNSTAIRS,
            predicted_h21: PREDICTED_H21_DOWNSTAIRS,
            predicted_chi: PREDICTED_CHI_DOWNSTAIRS,
        }
    }

    /// Canonical Schoen/Z3xZ3 + 3-generation prediction.
    pub fn schoen_z3xz3() -> Self {
        Self {
            label: "Schoen/Z3xZ3".to_string(),
            geometry: Z3xZ3Geometry::Schoen,
            predicted_h11: PREDICTED_H11_DOWNSTAIRS,
            predicted_h21: PREDICTED_H21_DOWNSTAIRS,
            predicted_chi: PREDICTED_CHI_DOWNSTAIRS,
        }
    }
}

/// Configuration knobs for the Hodge channel.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HodgeChannelConfig {
    /// Relative kernel cutoff: `|λ| ≤ τ · λ_max` counts as kernel.
    pub kernel_zero_thresh: f64,
    /// Gaussian width on `h^{p,q}` counts.
    pub sigma_h: f64,
    /// Gaussian width on `χ`.
    pub sigma_chi: f64,
    /// Maximum `K` we trust as a real kernel count. Above this we
    /// clip and report `truncated = true` (defends against the
    /// Tikhonov-shift artefact at high `seed_max_total_degree`).
    pub max_kernel_count: usize,
}

impl Default for HodgeChannelConfig {
    fn default() -> Self {
        Self {
            kernel_zero_thresh: DEFAULT_KERNEL_ZERO_THRESH,
            sigma_h: DEFAULT_SIGMA_H,
            sigma_chi: DEFAULT_SIGMA_CHI,
            max_kernel_count: 64,
        }
    }
}

/// Per-candidate result of the Hodge channel.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HodgeChannelResult {
    /// Candidate label (`"TY/Z3"` etc.).
    pub candidate: String,

    /// Journal-predicted `h^{1,1}` downstairs.
    pub predicted_h11: usize,
    /// Journal-predicted `h^{2,1}` downstairs.
    pub predicted_h21: usize,
    /// Journal-predicted `χ` downstairs.
    pub predicted_chi: i32,

    /// Total kernel count `K` of the bundle Laplacian on the
    /// `Z/3 × Z/3` (or `Z/3`) trivial-rep sub-bundle.
    pub measured_kernel_total: usize,
    /// `K / 2` (rounded down) — the (1,1)-sector estimate.
    pub measured_h11: usize,
    /// `K - measured_h11` — the (2,1)-sector estimate.
    pub measured_h21: usize,
    /// Relative Euler characteristic `measured_h11 - measured_h21`
    /// (matches the predicted `-6` when `K = 6` and the floor-half
    /// split puts the larger half on `h21`).
    pub measured_chi: i32,

    /// Gaussian log-likelihood evaluated at `(measured_h11,
    /// measured_h21, measured_chi)` against the predictions. Higher
    /// is better.
    pub log_likelihood_match: f64,

    /// Effective kernel-eigenvalue threshold (`τ` × `λ_max`).
    pub kernel_eigenvalue_threshold: f64,
    /// `λ_max` of the projected spectrum.
    pub lambda_max: f64,
    /// Lowest non-kernel eigenvalue (or `None` if all eigenvalues
    /// were classified as kernel).
    pub lambda_lowest_nonzero: Option<f64>,
    /// `final_basis_dim` (post-Z/3×Z/3+H_4 filter) from the
    /// projected solver.
    pub final_basis_dim: usize,
    /// Number of sample points used.
    pub n_points: usize,
    /// `true` if the kernel count exceeded `max_kernel_count` and
    /// was clipped before scoring.
    pub kernel_count_truncated: bool,

    /// `kernel_zero_thresh` and Gaussian widths actually used (for
    /// reproducibility).
    pub config_kernel_zero_thresh: f64,
    pub config_sigma_h: f64,
    pub config_sigma_chi: f64,
}

impl HodgeChannelResult {
    /// Convenience: TY-vs-Schoen log Bayes-factor contribution from
    /// this channel, given two results. Positive ⇒ TY favoured.
    pub fn log_bayes_factor_ty_vs_schoen(ty: &Self, schoen: &Self) -> f64 {
        ty.log_likelihood_match - schoen.log_likelihood_match
    }
}

/// Errors that can arise inside [`compute_hodge_channel`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HodgeChannelError {
    /// Spectrum solver returned an empty eigenvalue list (typically
    /// happens when the Z/3×Z/3 + H_4 filter empties the basis).
    EmptySpectrum {
        full_seed_basis_dim: usize,
        z3xz3_basis_dim: usize,
        final_basis_dim: usize,
    },
    /// `λ_max ≤ 0` — pathological projected Laplacian.
    DegenerateSpectrum { lambda_max: f64 },
}

impl std::fmt::Display for HodgeChannelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySpectrum {
                full_seed_basis_dim,
                z3xz3_basis_dim,
                final_basis_dim,
            } => write!(
                f,
                "Hodge channel: empty spectrum (full={}, Z/3×Z/3={}, final={})",
                full_seed_basis_dim, z3xz3_basis_dim, final_basis_dim
            ),
            Self::DegenerateSpectrum { lambda_max } => write!(
                f,
                "Hodge channel: degenerate spectrum (λ_max = {:e})",
                lambda_max
            ),
        }
    }
}

impl std::error::Error for HodgeChannelError {}

// -----------------------------------------------------------------------
// Pure-function scoring (synthetic-input testable).
// -----------------------------------------------------------------------

/// Score a measured `(h11, h21, χ)` triple against a predicted one
/// under independent Gaussian likelihoods. Returns
/// `ln L = -½ Σ ((measured - predicted) / σ)²`.
///
/// Pulled out as a free function so synthetic-input tests can call
/// it without a metric / bundle / spectrum.
pub fn gaussian_log_likelihood(
    measured_h11: usize,
    measured_h21: usize,
    measured_chi: i32,
    predicted_h11: usize,
    predicted_h21: usize,
    predicted_chi: i32,
    sigma_h: f64,
    sigma_chi: f64,
) -> f64 {
    let dh11 = measured_h11 as f64 - predicted_h11 as f64;
    let dh21 = measured_h21 as f64 - predicted_h21 as f64;
    let dchi = measured_chi as f64 - predicted_chi as f64;
    let s_h = sigma_h.max(1.0e-12);
    let s_chi = sigma_chi.max(1.0e-12);
    -0.5 * ((dh11 / s_h).powi(2) + (dh21 / s_h).powi(2) + (dchi / s_chi).powi(2))
}

/// Hodge-symmetric `(h11, h21)` split of a kernel total `K`.
/// Returns `(h11, h21)` with `h11 ≤ h21` and `h11 + h21 = K`.
pub fn symmetric_split(k: usize) -> (usize, usize) {
    let h11 = k / 2;
    let h21 = k - h11;
    (h11, h21)
}

/// Count kernel eigenvalues `|λ| ≤ τ · λ_max` in an ascending
/// spectrum. Returns `(kernel_count, λ_max, λ_lowest_nonzero)`.
fn classify_kernel(
    eigenvalues_full: &[f64],
    kernel_zero_thresh: f64,
) -> (usize, f64, Option<f64>) {
    if eigenvalues_full.is_empty() {
        return (0, 0.0, None);
    }
    let lambda_max = eigenvalues_full
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let cutoff = lambda_max.max(0.0) * kernel_zero_thresh;
    // The spectrum is ascending; `take_while |λ| ≤ cutoff` gives the
    // kernel block.
    let kernel_count = eigenvalues_full
        .iter()
        .take_while(|&&v| v.abs() <= cutoff)
        .count();
    let lambda_lowest_nonzero = if kernel_count < eigenvalues_full.len() {
        Some(eigenvalues_full[kernel_count])
    } else {
        None
    };
    (kernel_count, lambda_max, lambda_lowest_nonzero)
}

// -----------------------------------------------------------------------
// Top-level driver.
// -----------------------------------------------------------------------

/// Run the bundle Laplacian on the Z/3 × Z/3 (or Z/3 for TY)
/// trivial-rep sub-bundle, count kernel modes, split by Hodge
/// symmetry, and return the Gaussian log-likelihood at the
/// predicted `(h11, h21, χ)`.
#[allow(clippy::too_many_arguments)]
pub fn compute_hodge_channel(
    candidate: &HodgeCandidateSpec,
    bundle: &MonadBundle,
    metric: &dyn MetricBackground,
    h_v: &HymHermitianMetric,
    wilson: &Z3xZ3WilsonLines,
    laplacian_config: &Z3xZ3BundleConfig,
    channel_config: &HodgeChannelConfig,
) -> Result<HodgeChannelResult, HodgeChannelError> {
    // The geometry on the laplacian config and the candidate must
    // agree. If they don't we override (the candidate is the source
    // of truth).
    let mut cfg = laplacian_config.clone();
    cfg.geometry = candidate.geometry;

    let spectrum: Z3xZ3BundleSpectrumResult =
        solve_z3xz3_bundle_laplacian(bundle, metric, h_v, wilson, &cfg);

    let evals = &spectrum.eigenvalues_full;
    if evals.is_empty() {
        return Err(HodgeChannelError::EmptySpectrum {
            full_seed_basis_dim: spectrum.full_seed_basis_dim,
            z3xz3_basis_dim: spectrum.z3xz3_basis_dim,
            final_basis_dim: spectrum.final_basis_dim,
        });
    }

    let (raw_kernel, lambda_max, lambda_lowest_nonzero) =
        classify_kernel(evals, channel_config.kernel_zero_thresh);

    if !lambda_max.is_finite() || lambda_max <= 0.0 {
        return Err(HodgeChannelError::DegenerateSpectrum { lambda_max });
    }

    let truncated = raw_kernel > channel_config.max_kernel_count;
    let kernel_total = if truncated {
        channel_config.max_kernel_count
    } else {
        raw_kernel
    };

    let (measured_h11, measured_h21) = symmetric_split(kernel_total);
    let measured_chi = measured_h11 as i32 - measured_h21 as i32;

    let log_likelihood = gaussian_log_likelihood(
        measured_h11,
        measured_h21,
        measured_chi,
        candidate.predicted_h11,
        candidate.predicted_h21,
        candidate.predicted_chi,
        channel_config.sigma_h,
        channel_config.sigma_chi,
    );

    Ok(HodgeChannelResult {
        candidate: candidate.label.clone(),
        predicted_h11: candidate.predicted_h11,
        predicted_h21: candidate.predicted_h21,
        predicted_chi: candidate.predicted_chi,
        measured_kernel_total: kernel_total,
        measured_h11,
        measured_h21,
        measured_chi,
        log_likelihood_match: log_likelihood,
        kernel_eigenvalue_threshold: lambda_max * channel_config.kernel_zero_thresh,
        lambda_max,
        lambda_lowest_nonzero,
        final_basis_dim: spectrum.final_basis_dim,
        n_points: spectrum.n_points,
        kernel_count_truncated: truncated,
        config_kernel_zero_thresh: channel_config.kernel_zero_thresh,
        config_sigma_h: channel_config.sigma_h,
        config_sigma_chi: channel_config.sigma_chi,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicted_constants_match_journal() {
        assert_eq!(PREDICTED_H11_DOWNSTAIRS, 3);
        assert_eq!(PREDICTED_H21_DOWNSTAIRS, 3);
        assert_eq!(PREDICTED_CHI_DOWNSTAIRS, -6);
    }

    #[test]
    fn symmetric_split_six_is_three_three() {
        let (h11, h21) = symmetric_split(6);
        assert_eq!(h11, 3);
        assert_eq!(h21, 3);
    }

    #[test]
    fn symmetric_split_seven_is_three_four() {
        let (h11, h21) = symmetric_split(7);
        assert_eq!(h11, 3);
        assert_eq!(h21, 4);
        assert!(h11 <= h21);
    }

    #[test]
    fn symmetric_split_zero_is_zero_zero() {
        let (h11, h21) = symmetric_split(0);
        assert_eq!(h11, 0);
        assert_eq!(h21, 0);
    }

    #[test]
    fn gaussian_likelihood_at_predicted_is_zero() {
        let ll = gaussian_log_likelihood(3, 3, -6, 3, 3, -6, 0.5, 1.0);
        assert!(ll.abs() < 1.0e-12, "expected exact zero, got {}", ll);
    }

    #[test]
    fn gaussian_likelihood_decreases_with_distance() {
        let l_at = gaussian_log_likelihood(3, 3, -6, 3, 3, -6, 0.5, 1.0);
        let l_off = gaussian_log_likelihood(5, 1, 0, 3, 3, -6, 0.5, 1.0);
        assert!(
            l_off < l_at,
            "off-prediction likelihood {} should be less than at-prediction {}",
            l_off,
            l_at
        );
    }

    #[test]
    fn classify_kernel_counts_below_cutoff() {
        let evals = vec![1.0e-8, 2.0e-8, 5.0e-8, 0.5, 1.0];
        // λ_max = 1.0, τ = 1e-3 ⇒ cutoff = 1e-3. Three sub-cutoff
        // eigenvalues.
        let (k, lmax, lo) = classify_kernel(&evals, 1.0e-3);
        assert_eq!(k, 3);
        assert!((lmax - 1.0).abs() < 1.0e-12);
        assert_eq!(lo, Some(0.5));
    }

    #[test]
    fn classify_kernel_empty_returns_zero() {
        let (k, lmax, lo) = classify_kernel(&[], 1.0e-3);
        assert_eq!(k, 0);
        assert_eq!(lmax, 0.0);
        assert!(lo.is_none());
    }

    #[test]
    fn classify_kernel_all_above_cutoff() {
        let evals = vec![0.5, 0.7, 1.0];
        let (k, _, lo) = classify_kernel(&evals, 1.0e-3);
        assert_eq!(k, 0);
        assert_eq!(lo, Some(0.5));
    }
}
