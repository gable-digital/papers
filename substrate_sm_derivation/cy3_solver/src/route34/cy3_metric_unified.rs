//! Unified trait + thin polymorphic API over the Tian-Yau and Schoen
//! Calabi-Yau metric solvers, so downstream code (Yukawa overlaps,
//! Bayesian discrimination, η-integral evaluator) can call
//! `solver.solve_metric(candidate)` without branching on the concrete
//! variety.
//!
//! Both implementations share the same algorithmic contract:
//!
//!   1. Sample `n_sample` points on the actual sub-variety (Newton
//!      projection from a parametric ambient line, patch rescale,
//!      `|Ω|^2 / det g_pb` weighting).
//!   2. Build the `Z/k`- (or `Z/3 × Z/3`-) invariant section basis
//!      modulo the defining ideal at the requested degree.
//!   3. Run weighted Donaldson balancing to a fixed Frobenius
//!      tolerance with explicit Hermitian symmetrisation + trace
//!      normalisation each iteration.
//!   4. Report `(n_basis, balanced_h, sigma_residual, donaldson_residual,
//!      iterations_run)`, plus reproducibility metadata.
//!
//! The trait packs that contract into a small object-safe interface;
//! see [`Cy3MetricSolver`] and the two impls below.

use crate::route34::schoen_metric::{
    self, SchoenMetricConfig, SchoenMetricResult,
};
use crate::route34::ty_metric::{self, TyMetricConfig, TyMetricResult};

// ---------------------------------------------------------------------------
// Unified spec / result
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Cy3MetricSpec {
    TianYau {
        k: u32,
        n_sample: usize,
        max_iter: usize,
        donaldson_tol: f64,
        seed: u64,
    },
    Schoen {
        d_x: u32,
        d_y: u32,
        d_t: u32,
        n_sample: usize,
        max_iter: usize,
        donaldson_tol: f64,
        seed: u64,
    },
}

/// Optional Adam σ-refinement override that callers can attach to a
/// [`Cy3MetricSpec`] dispatch to enable post-Donaldson Adam descent.
/// Stored separately from the spec enum to keep all existing literal
/// initialisers compatible.
#[derive(Debug, Clone, Default)]
pub struct Cy3AdamOverride {
    pub adam_refine: Option<crate::route34::schoen_metric::AdamRefineConfig>,
    /// P7.11 — propagate the Donaldson-loop GPU flag into the
    /// Schoen / TY config when dispatching through the unified
    /// solver. Independent of the σ-refinement GPU flag carried
    /// by `adam_refine`.
    pub use_gpu_donaldson: bool,
}

impl Cy3MetricSpec {
    pub fn ty_publication_default() -> Self {
        Self::TianYau {
            k: 4,
            n_sample: 2000,
            max_iter: 50,
            donaldson_tol: 1.0e-3,
            seed: 42,
        }
    }
    pub fn schoen_publication_default() -> Self {
        Self::Schoen {
            d_x: 4,
            d_y: 4,
            d_t: 2,
            n_sample: 2000,
            max_iter: 50,
            donaldson_tol: 1.0e-3,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Cy3MetricResultKind {
    TianYau(Box<TyMetricResult>),
    Schoen(Box<SchoenMetricResult>),
}

impl Cy3MetricResultKind {
    pub fn summary(&self) -> Cy3MetricSummary {
        match self {
            Self::TianYau(r) => Cy3MetricSummary {
                variety: "TianYau",
                n_basis: r.n_basis,
                n_points: r.sample_points.len(),
                iterations_run: r.iterations_run,
                final_sigma_residual: r.final_sigma_residual,
                sigma_fs_identity: r.sigma_fs_identity,
                final_donaldson_residual: r.final_donaldson_residual,
                balanced_h_sha256: r.run_metadata.balanced_h_sha256.clone(),
                sample_cloud_sha256: r.run_metadata.sample_cloud_sha256.clone(),
                git_sha: r.run_metadata.git_sha.clone(),
                wall_clock_seconds: r.run_metadata.wall_clock_seconds,
            },
            Self::Schoen(r) => Cy3MetricSummary {
                variety: "Schoen",
                n_basis: r.n_basis,
                n_points: r.sample_points.len(),
                iterations_run: r.iterations_run,
                final_sigma_residual: r.final_sigma_residual,
                sigma_fs_identity: r.sigma_fs_identity,
                final_donaldson_residual: r.final_donaldson_residual,
                balanced_h_sha256: r.run_metadata.balanced_h_sha256.clone(),
                sample_cloud_sha256: r.run_metadata.sample_cloud_sha256.clone(),
                git_sha: r.run_metadata.git_sha.clone(),
                wall_clock_seconds: r.run_metadata.wall_clock_seconds,
            },
        }
    }
    pub fn final_donaldson_residual(&self) -> f64 {
        match self {
            Self::TianYau(r) => r.final_donaldson_residual,
            Self::Schoen(r) => r.final_donaldson_residual,
        }
    }
    pub fn final_sigma_residual(&self) -> f64 {
        match self {
            Self::TianYau(r) => r.final_sigma_residual,
            Self::Schoen(r) => r.final_sigma_residual,
        }
    }
    pub fn iterations_run(&self) -> usize {
        match self {
            Self::TianYau(r) => r.iterations_run,
            Self::Schoen(r) => r.iterations_run,
        }
    }
    /// σ measured at the FS-Gram identity (`h = I`) BEFORE any Donaldson
    /// iteration. P5.5d-introduced field on each underlying result type.
    /// Used by P5.10 (and its hostile-review balanced-vs-early-bail
    /// re-analysis) to compare a seed's final σ against its starting
    /// FS-identity σ — early-bail seeds whose final σ sits within a few
    /// percent of `sigma_fs_identity` are evidence of the regression
    /// guard restoring a near-FS snapshot rather than a balanced metric.
    pub fn sigma_fs_identity(&self) -> f64 {
        match self {
            Self::TianYau(r) => r.sigma_fs_identity,
            Self::Schoen(r) => r.sigma_fs_identity,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Cy3MetricSummary {
    pub variety: &'static str,
    pub n_basis: usize,
    pub n_points: usize,
    pub iterations_run: usize,
    pub final_sigma_residual: f64,
    /// σ at FS-identity, before any Donaldson iteration. See
    /// `Cy3MetricResultKind::sigma_fs_identity` for context.
    pub sigma_fs_identity: f64,
    pub final_donaldson_residual: f64,
    pub balanced_h_sha256: String,
    pub sample_cloud_sha256: String,
    pub git_sha: String,
    pub wall_clock_seconds: f64,
}

// ---------------------------------------------------------------------------
// Trait + impls
// ---------------------------------------------------------------------------

pub trait Cy3MetricSolver: Send + Sync {
    fn label(&self) -> &'static str;
    fn solve_metric(&self, spec: &Cy3MetricSpec) -> Result<Cy3MetricResultKind, Cy3MetricError>;
    /// Solve with an optional Adam σ-refinement override. Default
    /// implementation forwards to `solve_metric` (no Adam).
    fn solve_metric_with_adam(
        &self,
        spec: &Cy3MetricSpec,
        _adam: &Cy3AdamOverride,
    ) -> Result<Cy3MetricResultKind, Cy3MetricError> {
        self.solve_metric(spec)
    }
}

#[derive(Debug)]
pub enum Cy3MetricError {
    SpecMismatch { expected: &'static str, got: &'static str },
    TianYau(ty_metric::TyMetricError),
    Schoen(schoen_metric::SchoenMetricError),
}

impl std::fmt::Display for Cy3MetricError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SpecMismatch { expected, got } => write!(
                f,
                "Cy3MetricSpec mismatch: solver expected {expected} spec, got {got}"
            ),
            Self::TianYau(e) => write!(f, "{e}"),
            Self::Schoen(e) => write!(f, "{e}"),
        }
    }
}
impl std::error::Error for Cy3MetricError {}

impl From<ty_metric::TyMetricError> for Cy3MetricError {
    fn from(e: ty_metric::TyMetricError) -> Self {
        Self::TianYau(e)
    }
}
impl From<schoen_metric::SchoenMetricError> for Cy3MetricError {
    fn from(e: schoen_metric::SchoenMetricError) -> Self {
        Self::Schoen(e)
    }
}

pub struct TianYauSolver;

impl Cy3MetricSolver for TianYauSolver {
    fn label(&self) -> &'static str {
        "Tian-Yau Z/3"
    }
    fn solve_metric(&self, spec: &Cy3MetricSpec) -> Result<Cy3MetricResultKind, Cy3MetricError> {
        self.solve_metric_with_adam(spec, &Cy3AdamOverride::default())
    }

    fn solve_metric_with_adam(
        &self,
        spec: &Cy3MetricSpec,
        adam: &Cy3AdamOverride,
    ) -> Result<Cy3MetricResultKind, Cy3MetricError> {
        match spec {
            Cy3MetricSpec::TianYau {
                k,
                n_sample,
                max_iter,
                donaldson_tol,
                seed,
            } => {
                let cfg = TyMetricConfig {
                    k_degree: *k,
                    n_sample: *n_sample,
                    max_iter: *max_iter,
                    donaldson_tol: *donaldson_tol,
                    seed: *seed,
                    checkpoint_path: None,
                    apply_z3_quotient: true,
                    adam_refine: adam.adam_refine.clone(),
                    use_gpu: adam.use_gpu_donaldson,
                    donaldson_damping: None,
                    donaldson_tikhonov_shift: None,
                };
                let r = ty_metric::solve_ty_metric(cfg)?;
                Ok(Cy3MetricResultKind::TianYau(Box::new(r)))
            }
            Cy3MetricSpec::Schoen { .. } => Err(Cy3MetricError::SpecMismatch {
                expected: "TianYau",
                got: "Schoen",
            }),
        }
    }
}

pub struct SchoenSolver;

impl Cy3MetricSolver for SchoenSolver {
    fn label(&self) -> &'static str {
        "Schoen Z/3 × Z/3"
    }
    fn solve_metric(&self, spec: &Cy3MetricSpec) -> Result<Cy3MetricResultKind, Cy3MetricError> {
        self.solve_metric_with_adam(spec, &Cy3AdamOverride::default())
    }
    fn solve_metric_with_adam(
        &self,
        spec: &Cy3MetricSpec,
        adam: &Cy3AdamOverride,
    ) -> Result<Cy3MetricResultKind, Cy3MetricError> {
        match spec {
            Cy3MetricSpec::Schoen {
                d_x,
                d_y,
                d_t,
                n_sample,
                max_iter,
                donaldson_tol,
                seed,
            } => {
                let cfg = SchoenMetricConfig {
                    d_x: *d_x,
                    d_y: *d_y,
                    d_t: *d_t,
                    n_sample: *n_sample,
                    max_iter: *max_iter,
                    donaldson_tol: *donaldson_tol,
                    seed: *seed,
                    checkpoint_path: None,
                    apply_z3xz3_quotient: true,
                    adam_refine: adam.adam_refine.clone(),
                    use_gpu: adam.use_gpu_donaldson,
                    donaldson_damping: None,
                    donaldson_tikhonov_shift: None,
                };
                let r = schoen_metric::solve_schoen_metric(cfg)?;
                Ok(Cy3MetricResultKind::Schoen(Box::new(r)))
            }
            Cy3MetricSpec::TianYau { .. } => Err(Cy3MetricError::SpecMismatch {
                expected: "Schoen",
                got: "TianYau",
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ty_solver_dispatches_correctly() {
        let solver = TianYauSolver;
        let spec = Cy3MetricSpec::TianYau {
            k: 2,
            n_sample: 60,
            max_iter: 4,
            donaldson_tol: 1.0e-3,
            seed: 11,
        };
        let r = solver.solve_metric(&spec).expect("solve");
        let s = r.summary();
        assert_eq!(s.variety, "TianYau");
        assert!(s.n_basis > 0);
        assert!(r.iterations_run() > 0);
    }

    #[test]
    fn schoen_solver_dispatches_correctly() {
        let solver = SchoenSolver;
        let spec = Cy3MetricSpec::Schoen {
            d_x: 3,
            d_y: 3,
            d_t: 1,
            n_sample: 60,
            max_iter: 4,
            donaldson_tol: 1.0e-3,
            seed: 11,
        };
        let r = solver.solve_metric(&spec).expect("solve");
        let s = r.summary();
        assert_eq!(s.variety, "Schoen");
        assert!(s.n_basis > 0);
        assert!(r.iterations_run() > 0);
    }

    #[test]
    fn spec_mismatch_returns_error() {
        let solver = TianYauSolver;
        let bad_spec = Cy3MetricSpec::Schoen {
            d_x: 3,
            d_y: 3,
            d_t: 1,
            n_sample: 60,
            max_iter: 4,
            donaldson_tol: 1.0e-3,
            seed: 11,
        };
        assert!(matches!(
            solver.solve_metric(&bad_spec),
            Err(Cy3MetricError::SpecMismatch { .. })
        ));
    }

    #[test]
    fn polymorphic_dispatch_via_trait_object() {
        let solvers: Vec<Box<dyn Cy3MetricSolver>> =
            vec![Box::new(TianYauSolver), Box::new(SchoenSolver)];
        let labels: Vec<_> = solvers.iter().map(|s| s.label()).collect();
        assert!(labels.contains(&"Tian-Yau Z/3"));
        assert!(labels.contains(&"Schoen Z/3 × Z/3"));
    }
}
