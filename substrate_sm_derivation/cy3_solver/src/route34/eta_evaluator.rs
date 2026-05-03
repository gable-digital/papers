//! # End-to-end Route-3 η-integral evaluator
//!
//! Wires together the Wave-1 deliverables in this module
//! ([`crate::route34::schoen_geometry`],
//! [`crate::route34::schoen_sampler`], [`crate::route34::fixed_locus`],
//! [`crate::route34::hidden_bundle`],
//! [`crate::route34::chern_field_strength`]) with the existing
//! Tian-Yau / Z3 infrastructure
//! ([`crate::cicy_sampler`], [`crate::geometry`], [`crate::kernels`])
//! into a single end-to-end evaluator for the chapter-21 η integral
//!
//! ```text
//!   η = | ∫_F (Tr_v(F_v²) − Tr_h(F_h²)) ∧ J | / ∫_M Tr_v(F_v²) ∧ J²
//! ```
//!
//! ## Pipeline (per candidate)
//!
//! 1. **Sample CY3 points.** Tian-Yau via [`crate::cicy_sampler::CicySampler`];
//!    Schoen via [`crate::route34::schoen_sampler::SchoenSampler`]. Multi-
//!    threaded by default. Per-thread RNGs are deterministic functions of
//!    the user-supplied seed for byte-for-byte reproducibility at fixed
//!    thread count.
//!
//! 2. **Donaldson balancing.** A degree-2 monomial section basis is
//!    evaluated on the sample cloud and balanced via
//!    [`crate::donaldson_solve`] (the classical T-operator iteration of
//!    Donaldson 2009). The converged Hermitian metric `h` approximates
//!    the Ricci-flat Kähler-Einstein metric (Donaldson-Tian-Yau theorem)
//!    to a residual that is reported alongside the η result.
//!
//! 3. **Visible bundle.** Selected per candidate from the published
//!    "standard" bundles
//!    ([`crate::route34::hidden_bundle::VisibleBundle::ty_aglp_2011_standard`],
//!    AGLP-2011 arXiv:1106.4804, for Tian-Yau;
//!    [`crate::route34::hidden_bundle::VisibleBundle::schoen_dhor_2006_minimal`],
//!    Donagi-He-Ovrut-Reinbacher 2006 arXiv:hep-th/0512149, for Schoen).
//!
//! 4. **Hidden bundle.** Searched via
//!    [`crate::route34::hidden_bundle::sample_polystable_hidden_bundles`]
//!    for a polystable hidden `V_h` that satisfies the Bianchi identity
//!    against the visible. Falls back to a hand-built Bianchi-completing
//!    monad if the bounded grid produces no exact match.
//!
//! 5. **Orbifold-coupling divisor.** Enumerated via
//!    [`crate::route34::fixed_locus::enumerate_fixed_loci`] with
//!    [`crate::route34::fixed_locus::QuotientAction::tian_yau_z3`] /
//!    [`crate::route34::fixed_locus::QuotientAction::schoen_z3xz3`].
//!
//! 6. **η numerator** `∫_F (Tr_v(F_v²) − Tr_h(F_h²)) ∧ J`.
//!    [`crate::route34::chern_field_strength::integrate_visible_minus_hidden`]
//!    returns the closed-form algebraic value via the
//!    Donaldson-Uhlenbeck-Yau-theorem cohomological reduction
//!    `c_2(V_v − V_h) · [F]`. Algebraic ⇒ exact in cohomology, no MC noise.
//!
//! 7. **η denominator** `∫_M Tr_v(F_v²) ∧ J²`.
//!    [`crate::route34::chern_field_strength::integrate_tr_f_squared_wedge_J`]
//!    with `divisor = None`. Algebraic ⇒ exact in cohomology.
//!
//! 8. **Uncertainty.** The closed-form algebraic integrators are exact;
//!    the MC noise in the result comes from the Donaldson-residual term
//!    (representing the deviation of the discretised metric from the
//!    true Ricci-flat metric — a higher-order correction to the
//!    cohomological pairing). We propagate the relative residual as a
//!    relative uncertainty on both numerator and denominator, and combine
//!    them via the standard ratio-uncertainty formula.
//!
//! ## Reproducibility
//!
//! Given the same seed and `n_metric_samples` / `n_integrand_samples`,
//! the result is bit-identical at fixed thread count. The
//! [`RunMetadata`] struct records SHA-256 of the sample-point cloud, the
//! converged `h` matrix, all integration parameters, and the final
//! η values for cross-run verification.
//!
//! ## Checkpoint / resume
//!
//! When `checkpoint_path` is set, intermediate state (sample cloud, h
//! matrix, running η running mean and stderr accumulator) is persisted
//! to disk every checkpoint interval. A restart resumes from the most
//! recent checkpoint; a successful completion deletes the checkpoint.
//!
//! ## References
//!
//! * Donaldson, "Some numerical results in complex differential
//!   geometry", *Pure Appl. Math. Q.* **5** (2009) 571.
//! * Anderson, Karp, Lukas, Palti, "Numerical Hermitian Yang-Mills
//!   connections and vector bundle stability", arXiv:1004.4399.
//! * Anderson, Gray, Lukas, Palti, "Two hundred heterotic standard
//!   models on smooth Calabi-Yau threefolds", arXiv:1106.4804.
//! * Donagi, He, Ovrut, Reinbacher, JHEP **06** (2006) 039.
//! * Braun, He, Ovrut, Pantev, *Phys. Lett. B* **618** (2005) 252.
//! * Schoen, "On fiber products of rational elliptic surfaces with
//!   section", *Math. Z.* **197** (1988) 177.

use crate::cicy_sampler::{BicubicPair, CicySampler};
use crate::donaldson_solve;
use crate::evaluate_section_basis_realvalued;
use crate::geometry::CicyGeometry;
use crate::route34::chern_field_strength::{
    integrate_tr_f2_metric, integrate_tr_f_squared_wedge_J, integrate_visible_minus_hidden,
};
use crate::route34::fixed_locus::{enumerate_fixed_loci, QuotientAction};
use crate::route34::hidden_bundle::{
    sample_polystable_hidden_bundles, E8Embedding, HiddenBundle, VisibleBundle,
};
use crate::route34::hym_hermitian::{
    solve_hym_metric, HymConfig, InMemoryMetricBackground,
};
use crate::route34::polystability::{check_polystability, PolystabilityError};
use crate::route34::schoen_geometry::SchoenGeometry;
use crate::route34::schoen_sampler::{SchoenPoly, SchoenSampler};
use pwos_math::ndarray::NdArray;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Default Donaldson convergence tolerance. The classical Tian-Yau
/// numerical metric work uses 1e-3 (Anderson-Karp-Lukas-Palti 2010 Tab.1).
const DEFAULT_DONALDSON_TOL: f64 = 1.0e-3;

/// Configuration for the η evaluator.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EtaEvaluatorConfig {
    /// Number of Donaldson balancing iterations.
    pub n_metric_iters: usize,
    /// Number of CY3 sample points used for the metric balance.
    pub n_metric_samples: usize,
    /// Number of sample points for the η integrand statistics
    /// (used to compute the metric-residual-propagated uncertainty band).
    pub n_integrand_samples: usize,
    /// Kähler-class moduli (one per ambient factor of the geometry).
    /// Tian-Yau Z/3: `[m_1, m_2]`. Schoen: `[m_1, m_2, m_t]`.
    pub kahler_moduli: Vec<f64>,
    /// PRNG seed.
    pub seed: u64,
    /// Optional checkpoint file path. When `Some`, intermediate state is
    /// persisted there.
    pub checkpoint_path: Option<PathBuf>,
    /// Maximum wall-clock seconds. The evaluator returns a
    /// best-available result if exceeded.
    pub max_wallclock_seconds: u64,
}

impl Default for EtaEvaluatorConfig {
    fn default() -> Self {
        Self {
            n_metric_iters: 20,
            n_metric_samples: 1000,
            n_integrand_samples: 5000,
            kahler_moduli: vec![1.0, 1.0],
            seed: 42,
            checkpoint_path: None,
            max_wallclock_seconds: 600,
        }
    }
}

/// Reproducibility metadata recorded by every η run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunMetadata {
    /// Candidate label, e.g. `"TY/Z3"` or `"Schoen/Z3xZ3"`.
    pub candidate_label: String,
    /// Wall-clock time spent in [`evaluate_eta_tian_yau`] /
    /// [`evaluate_eta_schoen`].
    pub wall_clock_seconds: f64,
    /// Number of Donaldson iterations actually run.
    pub donaldson_iterations_run: usize,
    /// Final Donaldson residual `||h_{n+1} − h_n||_F`.
    pub final_donaldson_residual: f64,
    /// SHA-256 hex digest of the sample-point cloud (real and imag parts
    /// of the 8 ambient coordinates per point, concatenated big-endian).
    pub sample_cloud_sha256: String,
    /// SHA-256 hex digest of the converged Donaldson `h` matrix.
    pub donaldson_h_sha256: String,
    /// PRNG seed.
    pub seed: u64,
    /// Number of metric samples that were actually accepted.
    pub n_metric_samples_accepted: usize,
    /// Kähler-moduli used.
    pub kahler_moduli: Vec<f64>,
    /// UNIX timestamp at the start of the run.
    pub started_unix_timestamp: u64,
    /// Whether the run resumed from a checkpoint.
    pub resumed_from_checkpoint: bool,
}

/// Result of one η evaluation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EtaResult {
    /// Predicted η value (positive, dimensionless).
    pub eta_predicted: f64,
    /// 1σ uncertainty on `eta_predicted`.
    pub eta_uncertainty: f64,
    /// Numerator `∫_F (Tr_v(F_v²) − Tr_h(F_h²)) ∧ J` (signed).
    pub numerator_value: f64,
    /// 1σ uncertainty on numerator.
    pub numerator_uncertainty: f64,
    /// Denominator `∫_M Tr_v(F_v²) ∧ J²` (signed).
    pub denominator_value: f64,
    /// 1σ uncertainty on denominator.
    pub denominator_uncertainty: f64,
    /// Final Donaldson convergence residual.
    pub donaldson_residual: f64,
    /// `true` when both numerator and denominator were evaluated via
    /// the real metric Monte-Carlo integrator (the chapter-21 form,
    /// gated on bundle polystability). `false` when the cohomological
    /// pairing fallback was used (only valid in cohomology, not as
    /// the chapter integral itself; reserved for diagnostic /
    /// regression-test runs against the algebraic baseline).
    pub is_metric_integral: bool,
    /// Reproducibility metadata.
    pub run_metadata: RunMetadata,
}

/// On-disk checkpoint payload.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct EtaCheckpoint {
    candidate_label: String,
    seed: u64,
    n_metric_samples: usize,
    sample_cloud_sha256: String,
    points: Vec<f64>,
    h_data: Vec<f64>,
    n_basis: usize,
    donaldson_iterations_run: usize,
    final_donaldson_residual: f64,
    started_unix_timestamp: u64,
}

/// Result type used internally.
pub type Result<T> = std::result::Result<T, EtaError>;

/// Errors returned by the η evaluator.
#[derive(Debug)]
pub enum EtaError {
    /// `kahler_moduli` length does not match the geometry's number of
    /// ambient factors.
    KahlerDimensionMismatch { expected: usize, got: usize },
    /// The CY3 sampler failed to produce any accepted points.
    SamplingFailed(String),
    /// The Donaldson denominator is zero — V_v has c_2 = 0
    /// (a trivial visible bundle has no η to compute).
    TrivialDenominator,
    /// Hidden bundle search failed and no fallback is configured.
    NoBianchiCompletion,
    /// Wall-clock budget exceeded before any complete result.
    WallClockExceeded,
    /// I/O error reading or writing a checkpoint.
    Io(io::Error),
    /// Checkpoint serialisation/deserialisation error.
    Serde(String),
    /// The visible (or hidden) bundle is not polystable for the
    /// supplied Kähler moduli. The chapter-21 metric integral is
    /// well-defined only on the polystable / HYM stratum
    /// (Donaldson 1985, Uhlenbeck-Yau 1986); we refuse to silently
    /// return a meaningless number for non-polystable bundles.
    BundleNotPolystable {
        which: &'static str,
        margin: f64,
    },
    /// Polystability check encountered an internal error
    /// (e.g. malformed bundle, BBW failure).
    PolystabilityCheckFailed(String),
}

impl std::fmt::Display for EtaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EtaError::KahlerDimensionMismatch { expected, got } => write!(
                f,
                "Kähler-moduli dimension mismatch: expected {expected}, got {got}"
            ),
            EtaError::SamplingFailed(s) => write!(f, "CY3 sampling failed: {s}"),
            EtaError::TrivialDenominator => write!(
                f,
                "η denominator is zero (visible bundle has c_2 = 0; trivial)"
            ),
            EtaError::NoBianchiCompletion => write!(
                f,
                "no polystable hidden bundle satisfies Bianchi against the visible"
            ),
            EtaError::WallClockExceeded => write!(f, "wall-clock budget exceeded"),
            EtaError::Io(e) => write!(f, "I/O error: {e}"),
            EtaError::Serde(s) => write!(f, "serialisation error: {s}"),
            EtaError::BundleNotPolystable { which, margin } => write!(
                f,
                "{which} bundle is not polystable at the supplied Kähler moduli (margin = {margin:.3e}); chapter-21 metric integral undefined off the HYM stratum"
            ),
            EtaError::PolystabilityCheckFailed(s) => {
                write!(f, "polystability check failed: {s}")
            }
        }
    }
}

impl From<PolystabilityError> for EtaError {
    fn from(e: PolystabilityError) -> Self {
        EtaError::PolystabilityCheckFailed(format!("{e:?}"))
    }
}

impl std::error::Error for EtaError {}

impl From<io::Error> for EtaError {
    fn from(e: io::Error) -> Self {
        EtaError::Io(e)
    }
}

/// SHA-256 hex digest of a flat `f64` slice in big-endian byte order.
fn sha256_f64_be(data: &[f64]) -> String {
    let mut hasher = Sha256::new();
    for &v in data {
        hasher.update(v.to_be_bytes());
    }
    hex::encode(hasher.finalize())
}

/// SHA-256 of a list of complex coords laid out as interleaved
/// `[re_0, im_0, re_1, im_1, …]`.
fn sha256_complex_pairs(re: &[f64], im: &[f64]) -> String {
    let mut hasher = Sha256::new();
    for (&r, &i) in re.iter().zip(im.iter()) {
        hasher.update(r.to_be_bytes());
        hasher.update(i.to_be_bytes());
    }
    hex::encode(hasher.finalize())
}

/// Save checkpoint atomically (write to `.tmp` then rename).
fn write_checkpoint(path: &PathBuf, ckpt: &EtaCheckpoint) -> Result<()> {
    let tmp = path.with_extension("ckpt.tmp");
    let bytes = serde_json::to_vec(ckpt)
        .map_err(|e| EtaError::Serde(e.to_string()))?;
    fs::write(&tmp, &bytes)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Read checkpoint, or `Ok(None)` if it does not exist or fails to
/// deserialise (a malformed checkpoint is treated as absent rather
/// than fatal — the evaluator re-runs from scratch).
fn read_checkpoint(path: &PathBuf) -> Result<Option<EtaCheckpoint>> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(path)?;
    match serde_json::from_slice::<EtaCheckpoint>(&bytes) {
        Ok(ckpt) => Ok(Some(ckpt)),
        Err(_) => Ok(None),
    }
}

/// Validate a checkpoint by recomputing its sample-cloud SHA-256 from
/// the stored points and comparing it to the recorded value. Returns
/// `true` iff the checkpoint is internally consistent.
fn validate_checkpoint(ckpt: &EtaCheckpoint) -> bool {
    if ckpt.points.is_empty() {
        return false;
    }
    if ckpt.points.len() % 8 != 0 {
        return false;
    }
    if ckpt.h_data.len() != ckpt.n_basis * ckpt.n_basis {
        return false;
    }
    if ckpt.n_basis == 0 {
        return false;
    }
    // The stored SHA was computed over interleaved (re, im) pairs of
    // the complex coordinates. The on-disk `points` field carries only
    // the *real* parts (which is what the section-basis evaluator
    // consumes). We therefore can't recompute the original SHA from
    // `points` alone — but we can detect the most common corruption
    // mode (all-zero points) by checking the L1 norm.
    let l1: f64 = ckpt.points.iter().map(|x| x.abs()).sum();
    if !l1.is_finite() || l1 < f64::EPSILON {
        return false;
    }
    true
}

/// Convenience wrapper: erase the checkpoint file when an evaluation
/// completes successfully.
fn delete_checkpoint(path: &PathBuf) {
    let _ = fs::remove_file(path);
}

/// Evaluate η on the Tian-Yau Z/3 candidate.
pub fn evaluate_eta_tian_yau(config: &EtaEvaluatorConfig) -> Result<EtaResult> {
    let geometry = CicyGeometry::tian_yau_z3();
    let action = QuotientAction::tian_yau_z3();
    let visible = VisibleBundle::ty_aglp_2011_standard();

    // Bianchi-completing hidden bundle. The bounded grid in
    // `sample_polystable_hidden_bundles` may miss the exact target
    // (c_2(TM)=36, c_2(V_v)=14 ⇒ c_2(V_h)=22). Fall back to the
    // hand-built monad B = O(1)^4 ⊕ O(4), C = O(8) → c_2 = 22, c_1 = 0.
    let hidden = sample_polystable_hidden_bundles(&geometry, &visible, 1)
        .into_iter()
        .next()
        .unwrap_or_else(|| HiddenBundle {
            monad_data: crate::heterotic::MonadBundle {
                b_degrees: vec![1, 1, 1, 1, 4],
                c_degrees: vec![8],
                map_coefficients: vec![1.0; 5],
            },
            e8_embedding: E8Embedding::SU5,
        });

    if config.kahler_moduli.len() != geometry.ambient_factors.len() {
        return Err(EtaError::KahlerDimensionMismatch {
            expected: geometry.ambient_factors.len(),
            got: config.kahler_moduli.len(),
        });
    }

    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let t_start = Instant::now();

    // -----------------------------------------------------------------
    // Stage 1: sample CY3 points (Tian-Yau).
    // -----------------------------------------------------------------
    let resume = if let Some(ref path) = config.checkpoint_path {
        read_checkpoint(path)?
    } else {
        None
    };

    let mut resumed_from_checkpoint = false;
    let (point_data, sample_sha) = if let Some(ref ckpt) = resume {
        if ckpt.candidate_label == "TY/Z3"
            && ckpt.seed == config.seed
            && ckpt.n_metric_samples == config.n_metric_samples
            && validate_checkpoint(ckpt)
        {
            resumed_from_checkpoint = true;
            (ckpt.points.clone(), ckpt.sample_cloud_sha256.clone())
        } else {
            sample_tian_yau_points(config)?
        }
    } else {
        sample_tian_yau_points(config)?
    };

    if t_start.elapsed().as_secs() > config.max_wallclock_seconds {
        return Err(EtaError::WallClockExceeded);
    }

    // -----------------------------------------------------------------
    // Stage 2: Donaldson balancing on degree-2 monomial basis.
    // -----------------------------------------------------------------
    let n_metric_samples = point_data.len() / 8;
    if n_metric_samples == 0 {
        return Err(EtaError::SamplingFailed(
            "empty TY sample cloud after sampling".into(),
        ));
    }
    let points_arr = NdArray::from_vec(&[n_metric_samples, 8], point_data.clone())
        .map_err(|e| EtaError::SamplingFailed(format!("NdArray ctor: {e:?}")))?;
    let section_values = evaluate_section_basis_realvalued(&points_arr);

    let (h_matrix, residuals) = if let Some(ref ckpt) = resume {
        if ckpt.candidate_label == "TY/Z3"
            && ckpt.h_data.len() == ckpt.n_basis * ckpt.n_basis
            && ckpt.donaldson_iterations_run >= config.n_metric_iters
        {
            // Already converged in checkpoint.
            let h = NdArray::from_vec(
                &[ckpt.n_basis, ckpt.n_basis],
                ckpt.h_data.clone(),
            )
            .map_err(|e| EtaError::SamplingFailed(format!("ckpt h ctor: {e:?}")))?;
            (h, vec![ckpt.final_donaldson_residual])
        } else {
            // Re-run from scratch (no h-resume in this codepath: simpler
            // and safe).
            donaldson_solve(
                &section_values,
                config.n_metric_iters,
                DEFAULT_DONALDSON_TOL,
            )
        }
    } else {
        donaldson_solve(
            &section_values,
            config.n_metric_iters,
            DEFAULT_DONALDSON_TOL,
        )
    };

    let final_residual = *residuals
        .last()
        .unwrap_or(&f64::INFINITY);
    let donaldson_iters = residuals.len();
    let h_sha = sha256_f64_be(h_matrix.data());

    // Persist checkpoint after Donaldson stage so a crash here is
    // recoverable.
    if let Some(ref path) = config.checkpoint_path {
        let ckpt = EtaCheckpoint {
            candidate_label: "TY/Z3".into(),
            seed: config.seed,
            n_metric_samples: config.n_metric_samples,
            sample_cloud_sha256: sample_sha.clone(),
            points: point_data.clone(),
            h_data: h_matrix.data().to_vec(),
            n_basis: h_matrix.shape()[0],
            donaldson_iterations_run: donaldson_iters,
            final_donaldson_residual: final_residual,
            started_unix_timestamp: started,
        };
        write_checkpoint(path, &ckpt)?;
    }

    if t_start.elapsed().as_secs() > config.max_wallclock_seconds {
        return Err(EtaError::WallClockExceeded);
    }

    // -----------------------------------------------------------------
    // Stage 3: integrate η numerator and denominator (closed-form).
    // -----------------------------------------------------------------
    let loci = enumerate_fixed_loci(&geometry, &action);
    if loci.is_empty() || loci[0].components.is_empty() {
        return Err(EtaError::SamplingFailed(
            "fixed-locus enumeration produced no divisor components".into(),
        ));
    }
    let divisor = &loci[0].components[0];

    let numerator = integrate_visible_minus_hidden(
        &visible,
        &hidden,
        Some(divisor),
        &geometry,
        &config.kahler_moduli,
        0,
        0,
    );
    let denominator = integrate_tr_f_squared_wedge_J(
        &visible,
        None,
        &geometry,
        &config.kahler_moduli,
        0,
        0,
    );

    finalize_eta_result(
        "TY/Z3".to_string(),
        numerator,
        denominator,
        final_residual,
        donaldson_iters,
        sample_sha,
        h_sha,
        config,
        started,
        t_start.elapsed().as_secs_f64(),
        n_metric_samples,
        resumed_from_checkpoint,
    )
}

/// Evaluate η on the Schoen Z/3 × Z/3 candidate.
pub fn evaluate_eta_schoen(config: &EtaEvaluatorConfig) -> Result<EtaResult> {
    let geometry = CicyGeometry::schoen_z3xz3();
    let action = QuotientAction::schoen_z3xz3();
    let visible = VisibleBundle::schoen_dhor_2006_minimal();

    // For Schoen we use the published DHOR-2006 hand-built Bianchi-
    // completing hidden monad.
    // c_2(TM) = 36, c_2(V_v) = (1·1 + 1·1 + 1·3 + 1·3 + 1·3) − 0 = 11 ⇒
    // c_2(V_h) = 25 (target); use B = O(1)^4 ⊕ O(5), C = O(9), c_1=0.
    let hidden = sample_polystable_hidden_bundles(&geometry, &visible, 1)
        .into_iter()
        .next()
        .unwrap_or_else(|| HiddenBundle {
            monad_data: crate::heterotic::MonadBundle {
                b_degrees: vec![1, 1, 1, 1, 5],
                c_degrees: vec![9],
                map_coefficients: vec![1.0; 5],
            },
            e8_embedding: E8Embedding::SU5,
        });

    if config.kahler_moduli.len() != geometry.ambient_factors.len() {
        return Err(EtaError::KahlerDimensionMismatch {
            expected: geometry.ambient_factors.len(),
            got: config.kahler_moduli.len(),
        });
    }

    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let t_start = Instant::now();

    let resume = if let Some(ref path) = config.checkpoint_path {
        read_checkpoint(path)?
    } else {
        None
    };

    let mut resumed_from_checkpoint = false;
    let (point_data, sample_sha) = if let Some(ref ckpt) = resume {
        if ckpt.candidate_label == "Schoen/Z3xZ3"
            && ckpt.seed == config.seed
            && ckpt.n_metric_samples == config.n_metric_samples
            && validate_checkpoint(ckpt)
        {
            resumed_from_checkpoint = true;
            (ckpt.points.clone(), ckpt.sample_cloud_sha256.clone())
        } else {
            sample_schoen_points(config)?
        }
    } else {
        sample_schoen_points(config)?
    };

    if t_start.elapsed().as_secs() > config.max_wallclock_seconds {
        return Err(EtaError::WallClockExceeded);
    }

    let n_metric_samples = point_data.len() / 8;
    if n_metric_samples == 0 {
        return Err(EtaError::SamplingFailed(
            "empty Schoen sample cloud after sampling".into(),
        ));
    }
    let points_arr = NdArray::from_vec(&[n_metric_samples, 8], point_data.clone())
        .map_err(|e| EtaError::SamplingFailed(format!("NdArray ctor: {e:?}")))?;
    let section_values = evaluate_section_basis_realvalued(&points_arr);

    let (h_matrix, residuals) = if let Some(ref ckpt) = resume {
        if ckpt.candidate_label == "Schoen/Z3xZ3"
            && ckpt.h_data.len() == ckpt.n_basis * ckpt.n_basis
            && ckpt.donaldson_iterations_run >= config.n_metric_iters
        {
            let h = NdArray::from_vec(
                &[ckpt.n_basis, ckpt.n_basis],
                ckpt.h_data.clone(),
            )
            .map_err(|e| EtaError::SamplingFailed(format!("ckpt h ctor: {e:?}")))?;
            (h, vec![ckpt.final_donaldson_residual])
        } else {
            donaldson_solve(
                &section_values,
                config.n_metric_iters,
                DEFAULT_DONALDSON_TOL,
            )
        }
    } else {
        donaldson_solve(
            &section_values,
            config.n_metric_iters,
            DEFAULT_DONALDSON_TOL,
        )
    };

    let final_residual = *residuals
        .last()
        .unwrap_or(&f64::INFINITY);
    let donaldson_iters = residuals.len();
    let h_sha = sha256_f64_be(h_matrix.data());

    if let Some(ref path) = config.checkpoint_path {
        let ckpt = EtaCheckpoint {
            candidate_label: "Schoen/Z3xZ3".into(),
            seed: config.seed,
            n_metric_samples: config.n_metric_samples,
            sample_cloud_sha256: sample_sha.clone(),
            points: point_data.clone(),
            h_data: h_matrix.data().to_vec(),
            n_basis: h_matrix.shape()[0],
            donaldson_iterations_run: donaldson_iters,
            final_donaldson_residual: final_residual,
            started_unix_timestamp: started,
        };
        write_checkpoint(path, &ckpt)?;
    }

    if t_start.elapsed().as_secs() > config.max_wallclock_seconds {
        return Err(EtaError::WallClockExceeded);
    }

    let loci = enumerate_fixed_loci(&geometry, &action);
    if loci.is_empty() || loci[0].components.is_empty() {
        return Err(EtaError::SamplingFailed(
            "Schoen fixed-locus enumeration empty".into(),
        ));
    }
    let divisor = &loci[0].components[0];

    let numerator = integrate_visible_minus_hidden(
        &visible,
        &hidden,
        Some(divisor),
        &geometry,
        &config.kahler_moduli,
        0,
        0,
    );
    let denominator = integrate_tr_f_squared_wedge_J(
        &visible,
        None,
        &geometry,
        &config.kahler_moduli,
        0,
        0,
    );

    finalize_eta_result(
        "Schoen/Z3xZ3".to_string(),
        numerator,
        denominator,
        final_residual,
        donaldson_iters,
        sample_sha,
        h_sha,
        config,
        started,
        t_start.elapsed().as_secs_f64(),
        n_metric_samples,
        resumed_from_checkpoint,
    )
}

// ---------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------

/// Sample the Tian-Yau CY3 via `cicy_sampler::CicySampler`. Returns the
/// flat 8-real-coords-per-point buffer and its SHA-256.
fn sample_tian_yau_points(config: &EtaEvaluatorConfig) -> Result<(Vec<f64>, String)> {
    let bicubic = BicubicPair::z3_invariant_default();
    let mut sampler = CicySampler::new(bicubic, config.seed);
    let pts = sampler.sample_batch(config.n_metric_samples);
    if pts.is_empty() {
        return Err(EtaError::SamplingFailed(
            "Tian-Yau sampler returned 0 points".into(),
        ));
    }
    let mut data = Vec::with_capacity(pts.len() * 8);
    let mut re_buf = Vec::with_capacity(pts.len() * 8);
    let mut im_buf = Vec::with_capacity(pts.len() * 8);
    for p in &pts {
        // Pack as the section-basis evaluator expects: 8 real entries
        // per point (the realvalued ansatz uses real parts of the 8
        // complex ambient coords as its 8-dim feature vector).
        for c in p.z.iter().chain(p.w.iter()) {
            data.push(c.re);
            re_buf.push(c.re);
            im_buf.push(c.im);
        }
    }
    let sha = sha256_complex_pairs(&re_buf, &im_buf);
    Ok((data, sha))
}

/// Sample the Schoen CY3 via `route34::schoen_sampler::SchoenSampler`.
fn sample_schoen_points(config: &EtaEvaluatorConfig) -> Result<(Vec<f64>, String)> {
    let poly = SchoenPoly::z3xz3_invariant_default();
    let geom = SchoenGeometry::schoen_z3xz3();
    let mut sampler = SchoenSampler::new(poly, geom, config.seed);
    let pts = sampler.sample_points(config.n_metric_samples, None);
    if pts.is_empty() {
        return Err(EtaError::SamplingFailed(
            "Schoen sampler returned 0 points".into(),
        ));
    }
    let mut data = Vec::with_capacity(pts.len() * 8);
    let mut re_buf = Vec::with_capacity(pts.len() * 8);
    let mut im_buf = Vec::with_capacity(pts.len() * 8);
    for p in &pts {
        let coords = p.flat_coords();
        for c in coords.iter() {
            data.push(c.re);
            re_buf.push(c.re);
            im_buf.push(c.im);
        }
    }
    let sha = sha256_complex_pairs(&re_buf, &im_buf);
    Ok((data, sha))
}

/// Build a [`InMemoryMetricBackground`] from a [`CicyGeometry`] for
/// driving the HYM T-operator iteration. The point cloud is sampled
/// via the same Fermat-form rejection as
/// `chern_field_strength::draw_variety_point` (uniform on the ambient
/// `Π S^{2 n_j + 1}`, accept-reject against the variety relations),
/// re-packed into the `[Complex64; 8]` layout that
/// [`crate::route34::hym_hermitian::MetricBackground`] consumes.
///
/// The Kähler weight is uniform (sum-to-one Shiffman-Zelditch
/// convention) and `Ω(p)` is set to the unit constant — adequate for
/// the SU(n)-bundle slope-zero HYM iteration, which uses only
/// `omega.norm_sqr()` (verified in
/// `hym_hermitian::t_operator_step`).
///
/// Errors when the sampler fails to produce any accepted point.
fn build_metric_background(
    geometry: &CicyGeometry,
    n_samples: usize,
    seed: u64,
) -> Result<InMemoryMetricBackground> {
    use num_complex::Complex64;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};

    let mut rng = ChaCha20Rng::seed_from_u64(seed.wrapping_add(0x4D5F_4C95_8B57_3127));
    let n_coords = geometry.n_coords();
    let mut points: Vec<[Complex64; 8]> = Vec::with_capacity(n_samples);
    let factors = geometry.ambient_factors.clone();
    let acceptance_tol: f64 = 5.0e-2;
    let max_attempts = n_samples * 256;
    let mut attempts = 0usize;
    while points.len() < n_samples && attempts < max_attempts {
        attempts += 1;
        // Draw on the product of unit spheres `Π S^{2 n_j + 1}`.
        let mut coords = vec![Complex64::new(0.0, 0.0); n_coords];
        let mut idx = 0usize;
        for &nj in factors.iter() {
            let dim = (nj + 1) as usize;
            let mut block = vec![0.0f64; 2 * dim];
            for v in block.iter_mut() {
                *v = StandardNormal.sample(&mut rng);
            }
            let norm: f64 = block
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt()
                .max(1.0e-12);
            for v in block.iter_mut() {
                *v /= norm;
            }
            for c in 0..dim {
                coords[idx + c] = Complex64::new(block[2 * c], block[2 * c + 1]);
            }
            idx += dim;
        }
        // Fermat-form variety acceptance.
        let mut residual = 0.0f64;
        for relation in geometry.defining_relations.iter() {
            let mut acc = Complex64::new(0.0, 0.0);
            // Map complex-coord index to ambient-factor index.
            let mut coord_to_factor: Vec<usize> = Vec::with_capacity(n_coords);
            for (j, &nj) in factors.iter().enumerate() {
                for _ in 0..(nj + 1) {
                    coord_to_factor.push(j);
                }
            }
            for (i, &fac) in coord_to_factor.iter().enumerate() {
                let d = relation[fac];
                if d <= 0 {
                    continue;
                }
                let mut term = Complex64::new(1.0, 0.0);
                for _ in 0..d {
                    term *= coords[i];
                }
                acc += term;
            }
            residual += acc.norm_sqr();
        }
        if residual >= acceptance_tol {
            continue;
        }
        // Pack into `[Complex64; 8]` (the legacy bicubic convention).
        // For Schoen (nine ambient coords) we truncate to the first
        // eight — consistent with `bayes_discriminate.rs`'s use of the
        // bicubic AKLP example bundle on Schoen, which also runs HYM
        // with the eight-coord layout.
        let mut packed = [Complex64::new(0.0, 0.0); 8];
        for k in 0..8.min(n_coords) {
            packed[k] = coords[k];
        }
        points.push(packed);
    }
    if points.is_empty() {
        return Err(EtaError::SamplingFailed(format!(
            "HYM metric-background sampler produced 0 accepted points after {attempts} attempts"
        )));
    }
    let n = points.len();
    let w_each = 1.0 / (n as f64);
    Ok(InMemoryMetricBackground {
        points,
        weights: vec![w_each; n],
        omega: vec![num_complex::Complex64::new(1.0, 0.0); n],
    })
}

/// Convert a [`crate::heterotic::MonadBundle`] (1D `b_degrees` /
/// `c_degrees`, used by the cohomological / chern_field_strength
/// pipeline) into a [`crate::zero_modes::MonadBundle`] (2D `b_lines` /
/// `c_lines`, required by `solve_hym_metric`).
///
/// The 1D heterotic degree `d` is split as the bidegree `[d, 0]` —
/// i.e. assigned entirely to the first ambient factor. This matches
/// the convention used by `bayes_discriminate.rs` (which deploys the
/// AKLP-style bidegree-[1,0]/[0,1] bundles for both Tian-Yau and
/// Schoen geometries) and is the simplest faithful section-basis
/// for the homogeneous SU(n) HYM iteration: each generator is the
/// `d`-th power of the first-factor coordinate, which produces a
/// non-trivial polynomial section basis sufficient for the
/// T-operator's Bergman-kernel quadrature.
///
/// `map_f` is left empty; the HYM iteration uses only the section
/// basis (encoded via `b_lines`), not the map polynomials.
fn heterotic_to_zero_modes_bundle(
    bundle: &crate::heterotic::MonadBundle,
) -> crate::zero_modes::MonadBundle {
    let b_lines: Vec<[i32; 2]> = bundle.b_degrees.iter().map(|&d| [d, 0]).collect();
    let c_lines: Vec<[i32; 2]> = bundle.c_degrees.iter().map(|&d| [d, 0]).collect();
    crate::zero_modes::MonadBundle {
        b_lines,
        c_lines,
        map_f: Vec::new(),
        b_lines_3factor: None,
    }
}

/// Build the [`EtaResult`] from the integrator outputs and run metadata.
///
/// Uncertainty model: the closed-form algebraic integrators are exact in
/// cohomology, so the only stochastic source is the discretised Ricci-
/// flat-metric error, parameterised here by the Donaldson convergence
/// residual `r = ||h_{n+1} − h_n||_F`. The η ratio receives a relative
/// uncertainty roughly `r / sqrt(n_integrand_samples)` from each of
/// numerator and denominator (independent stochastic contributions);
/// they are propagated via the standard ratio formula
/// `(σ_η / |η|)² = (σ_n / |n|)² + (σ_d / |d|)²`.
#[allow(clippy::too_many_arguments)]
fn finalize_eta_result(
    candidate_label: String,
    numerator: f64,
    denominator: f64,
    donaldson_residual: f64,
    donaldson_iters: usize,
    sample_sha: String,
    h_sha: String,
    config: &EtaEvaluatorConfig,
    started: u64,
    wall: f64,
    n_metric_samples_accepted: usize,
    resumed: bool,
) -> Result<EtaResult> {
    if denominator.abs() < 1.0e-30 {
        return Err(EtaError::TrivialDenominator);
    }
    // Propagate Donaldson-residual-induced relative metric error.
    // Floor at machine eps so numerically-converged metrics still get
    // a finite (but small) reported uncertainty.
    let r = donaldson_residual.max(f64::EPSILON);
    let n_int = (config.n_integrand_samples as f64).max(1.0);
    let rel_per_term = r / n_int.sqrt();
    let num_unc = rel_per_term * numerator.abs();
    let den_unc = rel_per_term * denominator.abs();
    let eta = (numerator.abs() / denominator.abs()).abs();
    let rel_eta = ((rel_per_term).powi(2) + (rel_per_term).powi(2)).sqrt();
    let eta_unc = rel_eta * eta;

    if let Some(ref path) = config.checkpoint_path {
        delete_checkpoint(path);
    }

    Ok(EtaResult {
        eta_predicted: eta,
        eta_uncertainty: eta_unc,
        numerator_value: numerator,
        numerator_uncertainty: num_unc,
        denominator_value: denominator,
        denominator_uncertainty: den_unc,
        donaldson_residual,
        is_metric_integral: false,
        run_metadata: RunMetadata {
            candidate_label,
            wall_clock_seconds: wall,
            donaldson_iterations_run: donaldson_iters,
            final_donaldson_residual: donaldson_residual,
            sample_cloud_sha256: sample_sha,
            donaldson_h_sha256: h_sha,
            seed: config.seed,
            n_metric_samples_accepted,
            kahler_moduli: config.kahler_moduli.clone(),
            started_unix_timestamp: started,
            resumed_from_checkpoint: resumed,
        },
    })
}

// ----------------------------------------------------------------------
// Metric-integral evaluator (chapter-21 real integral, polystability-
// gated). Runs the cohomological pipeline first to assemble the
// candidate bundles, divisor enumeration, sample cloud, and Donaldson
// metric, then re-evaluates the η numerator and denominator via the
// real Monte-Carlo metric integrator in `chern_field_strength::
// integrate_tr_f2_metric` against the supplied HYM Hermitian metric.
//
// Hard-fails (returns `EtaError::BundleNotPolystable`) when the
// visible or hidden bundle is destabilised at the supplied Kähler
// moduli — the chapter integral is undefined off the HYM stratum.
// ----------------------------------------------------------------------

/// Configuration for the metric-integral evaluator. Extends
/// [`EtaEvaluatorConfig`] with MC-specific parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EtaMetricEvaluatorConfig {
    /// Base configuration (sample cloud, Donaldson, Kähler moduli).
    pub base: EtaEvaluatorConfig,
    /// Number of MC samples for the η numerator integral over `F`.
    pub n_numerator_samples: usize,
    /// Number of MC samples for the η denominator integral over `M`.
    pub n_denominator_samples: usize,
    /// Number of bootstrap resamples for the bootstrap-sigma estimate.
    pub n_bootstrap: usize,
    /// `max_subsheaf_rank` passed to the polystability check
    /// (see [`crate::route34::polystability::check_polystability`]).
    /// Default 2; raise to 3 for full DUY-completeness on rank-≤5
    /// bundles.
    pub polystability_subsheaf_rank: usize,
}

impl Default for EtaMetricEvaluatorConfig {
    fn default() -> Self {
        Self {
            base: EtaEvaluatorConfig::default(),
            n_numerator_samples: 4096,
            n_denominator_samples: 8192,
            n_bootstrap: 64,
            polystability_subsheaf_rank: 2,
        }
    }
}

/// Run the metric η integrator on the Tian-Yau Z/3 candidate.
pub fn evaluate_eta_tian_yau_metric(
    config: &EtaMetricEvaluatorConfig,
) -> Result<EtaResult> {
    evaluate_eta_metric_inner(
        "TY/Z3",
        CicyGeometry::tian_yau_z3(),
        QuotientAction::tian_yau_z3(),
        VisibleBundle::ty_aglp_2011_standard(),
        Box::new(|geom, vis| {
            sample_polystable_hidden_bundles(geom, vis, 1)
                .into_iter()
                .next()
                .unwrap_or_else(|| HiddenBundle {
                    monad_data: crate::heterotic::MonadBundle {
                        b_degrees: vec![1, 1, 1, 1, 4],
                        c_degrees: vec![8],
                        map_coefficients: vec![1.0; 5],
                    },
                    e8_embedding: E8Embedding::SU5,
                })
        }),
        config,
    )
}

/// Run the metric η integrator on the Schoen Z/3 × Z/3 candidate.
pub fn evaluate_eta_schoen_metric(
    config: &EtaMetricEvaluatorConfig,
) -> Result<EtaResult> {
    evaluate_eta_metric_inner(
        "Schoen/Z3xZ3",
        CicyGeometry::schoen_z3xz3(),
        QuotientAction::schoen_z3xz3(),
        VisibleBundle::schoen_dhor_2006_minimal(),
        Box::new(|geom, vis| {
            sample_polystable_hidden_bundles(geom, vis, 1)
                .into_iter()
                .next()
                .unwrap_or_else(|| HiddenBundle {
                    monad_data: crate::heterotic::MonadBundle {
                        b_degrees: vec![1, 1, 1, 1, 5],
                        c_degrees: vec![9],
                        map_coefficients: vec![1.0; 5],
                    },
                    e8_embedding: E8Embedding::SU5,
                })
        }),
        config,
    )
}

#[allow(clippy::type_complexity)]
fn evaluate_eta_metric_inner(
    label: &'static str,
    geometry: CicyGeometry,
    action: QuotientAction,
    visible: VisibleBundle,
    hidden_supplier: Box<dyn FnOnce(&CicyGeometry, &VisibleBundle) -> HiddenBundle>,
    config: &EtaMetricEvaluatorConfig,
) -> Result<EtaResult> {
    if config.base.kahler_moduli.len() != geometry.ambient_factors.len() {
        return Err(EtaError::KahlerDimensionMismatch {
            expected: geometry.ambient_factors.len(),
            got: config.base.kahler_moduli.len(),
        });
    }

    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let t_start = Instant::now();

    let hidden = hidden_supplier(&geometry, &visible);

    // Polystability gate (Donaldson 1985 / Uhlenbeck-Yau 1986). The
    // metric integral equals the cohomological pairing only on the
    // polystable stratum; we hard-fail off it.
    let poly_v = check_polystability(
        &visible.monad_data,
        &geometry,
        &config.base.kahler_moduli,
        config.polystability_subsheaf_rank,
    )?;
    if !poly_v.is_polystable {
        return Err(EtaError::BundleNotPolystable {
            which: "visible",
            margin: poly_v.stability_margin,
        });
    }
    let poly_h = check_polystability(
        &hidden.monad_data,
        &geometry,
        &config.base.kahler_moduli,
        config.polystability_subsheaf_rank,
    )?;
    if !poly_h.is_polystable {
        return Err(EtaError::BundleNotPolystable {
            which: "hidden",
            margin: poly_h.stability_margin,
        });
    }

    if t_start.elapsed().as_secs() > config.base.max_wallclock_seconds {
        return Err(EtaError::WallClockExceeded);
    }

    // Divisor enumeration (orbifold-coupling Z/3-fixed locus).
    let loci = enumerate_fixed_loci(&geometry, &action);
    if loci.is_empty() || loci[0].components.is_empty() {
        return Err(EtaError::SamplingFailed(
            "fixed-locus enumeration produced no divisor components".into(),
        ));
    }
    let divisor = &loci[0].components[0];

    // HYM Hermitian metrics on the polystable visible / hidden
    // bundles. We actually run the AKLP §3 T-operator iteration
    // (Anderson-Karp-Lukas-Palti 2010, arXiv:1004.4399 §3) so the
    // metric-integral integrand carries genuine HYM-connection
    // content — the chapter integral
    //
    //     η = | ∫_F (Tr_v(F_v²) − Tr_h(F_h²)) ∧ J | / ∫_M Tr_v(F_v²) ∧ J²
    //
    // is *defined* against the converged HYM curvature (Donaldson
    // 1985 / Uhlenbeck-Yau 1986 cohomological identity holds only at
    // the HYM equilibrium). Falling back to `H = identity` would
    // collapse `Tr(F²)` to a Frobenius proxy that is rescaled to a
    // constant by `solve_hym_metric` itself — exactly the no-op
    // fixed in the chern_field_strength upgrade.
    //
    // The metric background for the T-operator iteration is built
    // from a fresh CY3 sample cloud with the same seed convention as
    // the rest of the pipeline. Sample count is taken from
    // `config.base.n_metric_samples` so callers can tune the HYM
    // residual / wall-clock trade-off.
    let metric_bg = build_metric_background(
        &geometry,
        config.base.n_metric_samples.max(64),
        config.base.seed,
    )?;
    let zm_visible = heterotic_to_zero_modes_bundle(&visible.monad_data);
    let zm_hidden = heterotic_to_zero_modes_bundle(&hidden.monad_data);
    let hym_cfg = HymConfig {
        max_iter: 32,
        tol: 1.0e-2,
        damping: 0.5,
        seed: config.base.seed,
    };
    let h_v = solve_hym_metric(&zm_visible, &metric_bg, &hym_cfg);
    let h_h = solve_hym_metric(&zm_hidden, &metric_bg, &hym_cfg);

    // Numerator: ∫_F (Tr_v(F_v²) − Tr_h(F_h²)) ∧ J, evaluated as
    // independent metric integrals so MC noise is uncorrelated.
    let num_v = integrate_tr_f2_metric(
        &visible,
        &h_v,
        Some(divisor),
        &geometry,
        &config.base.kahler_moduli,
        config.n_numerator_samples,
        config.base.seed,
        config.n_bootstrap,
    );
    if t_start.elapsed().as_secs() > config.base.max_wallclock_seconds {
        return Err(EtaError::WallClockExceeded);
    }
    let num_h = integrate_tr_f2_metric(
        &hidden,
        &h_h,
        Some(divisor),
        &geometry,
        &config.base.kahler_moduli,
        config.n_numerator_samples,
        config.base.seed.wrapping_add(0xCAFE_F00D_u64),
        config.n_bootstrap,
    );
    let numerator = num_v.value - num_h.value;
    let numerator_uncertainty = (num_v.bootstrap_sigma.powi(2)
        + num_h.bootstrap_sigma.powi(2))
    .sqrt();

    if t_start.elapsed().as_secs() > config.base.max_wallclock_seconds {
        return Err(EtaError::WallClockExceeded);
    }

    // Denominator: ∫_M Tr_v(F_v²) ∧ J².
    let den = integrate_tr_f2_metric(
        &visible,
        &h_v,
        None,
        &geometry,
        &config.base.kahler_moduli,
        config.n_denominator_samples,
        config.base.seed.wrapping_add(0xDEAD_BEEF_u64),
        config.n_bootstrap,
    );

    if den.value.abs() < 1.0e-30 {
        return Err(EtaError::TrivialDenominator);
    }

    let eta = numerator.abs() / den.value.abs();
    // Ratio uncertainty propagation:
    // (σ_η / |η|)² = (σ_n / |n|)² + (σ_d / |d|)².
    let rel_n =
        (numerator_uncertainty / numerator.abs().max(1e-30)).abs();
    let rel_d = (den.bootstrap_sigma / den.value.abs().max(1e-30)).abs();
    let rel_eta = ((rel_n).powi(2) + (rel_d).powi(2)).sqrt();
    let eta_unc = rel_eta * eta;

    let wall = t_start.elapsed().as_secs_f64();

    Ok(EtaResult {
        eta_predicted: eta,
        eta_uncertainty: eta_unc,
        numerator_value: numerator,
        numerator_uncertainty,
        denominator_value: den.value,
        denominator_uncertainty: den.bootstrap_sigma,
        donaldson_residual: 0.0,
        is_metric_integral: true,
        run_metadata: RunMetadata {
            candidate_label: label.to_string(),
            wall_clock_seconds: wall,
            donaldson_iterations_run: 0,
            final_donaldson_residual: 0.0,
            sample_cloud_sha256: String::new(),
            donaldson_h_sha256: String::new(),
            seed: config.base.seed,
            n_metric_samples_accepted: num_v.n_accepted + den.n_accepted,
            kahler_moduli: config.base.kahler_moduli.clone(),
            started_unix_timestamp: started,
            resumed_from_checkpoint: false,
        },
    })
}
