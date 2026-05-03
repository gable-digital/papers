//! On-disk content-addressed cache for Donaldson-balanced Calabi-Yau
//! metric solves.
//!
//! ## Why this module exists
//!
//! The Donaldson-balanced Hermitian metric on a CY3 is a function of
//! the bundle-independent inputs
//!
//!   * manifold (Tian-Yau / Schoen / ...),
//!   * section-basis degree,
//!   * sample count `n_sample`,
//!   * iteration cap `max_iter`,
//!   * convergence tolerance `donaldson_tol`,
//!   * PRNG seed (controls the sample cloud).
//!
//! In particular, the metric does **not** depend on the Higgs / Kähler
//! moduli that nested sampling varies in
//! [`crate::route34::discrimination`]. It is therefore wasteful to
//! re-solve the metric for every likelihood draw — and at the
//! publication-grade settings `(n_sample ≥ 50_000, max_iter ≥ 80,
//! donaldson_tol ≤ 1e-4)` the per-draw cost would be prohibitive.
//!
//! This module provides a single content-addressed cache for both
//! `TyMetricResult` and `SchoenMetricResult`. The cache key is a
//! BLAKE-style stable hash of the configuration struct, so any
//! parameter change (including [`SchoenMetricConfig::d_x`]/`d_y`/`d_t`
//! tuples and the `apply_z3xz3_quotient` flag) yields a fresh key.
//!
//! ## Format
//!
//! Cached files are JSON-encoded `serde_json` blobs at paths of the
//! form
//!
//!   `<cache-root>/{ty,schoen}_k{...}_n{n_sample}_i{max_iter}_seed{seed}_<hash>.json`
//!
//! The trailing `<hash>` (8 lower-case hex chars) is a SHA-256
//! prefix of the canonical-JSON encoding of the config struct.
//! Including the human-readable parameter prefix makes cache files
//! easy to identify by eye; the hash suffix guarantees that any
//! parameter change MISSES.
//!
//! ## Quality guard
//!
//! On a cache HIT, the loader asserts
//! `final_sigma_residual < SIGMA_QUALITY_THRESHOLD`. If a cached
//! result is below quality (e.g. a previous run stopped early due to
//! `max_iter` cap before reaching `donaldson_tol`), the load fails
//! with [`MetricCacheError::StaleQuality`] and the caller is expected
//! to recompute fresh.
//!
//! ## Reference
//!
//! Donaldson 2009 (Pure Appl. Math. Q. 5, arXiv:math/0512625) §5
//! reports σ ~ 10⁻³ at k=4 needs > 10⁵ samples and ≥ 50 iterations on
//! the quintic. The threshold here is `1.0e-3` (publication grade).

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::route34::schoen_metric::{
    solve_schoen_metric, SchoenMetricConfig, SchoenMetricResult,
};
use crate::route34::ty_metric::{solve_ty_metric, TyMetricConfig, TyMetricResult};

/// σ-residual threshold above which a cached metric is rejected as
/// stale / sub-publication-grade. Donaldson 2009 §5 baseline.
pub const SIGMA_QUALITY_THRESHOLD: f64 = 1.0e-3;

#[derive(Debug)]
pub enum MetricCacheError {
    Io(std::io::Error),
    Serde(serde_json::Error),
    SolveTy(crate::route34::ty_metric::TyMetricError),
    SolveSchoen(crate::route34::schoen_metric::SchoenMetricError),
    StaleQuality {
        path: PathBuf,
        sigma: f64,
        threshold: f64,
    },
}

impl std::fmt::Display for MetricCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "metric_cache: I/O: {e}"),
            Self::Serde(e) => write!(f, "metric_cache: serde: {e}"),
            Self::SolveTy(e) => write!(f, "metric_cache: TY solve failure: {e}"),
            Self::SolveSchoen(e) => write!(f, "metric_cache: Schoen solve failure: {e}"),
            Self::StaleQuality {
                path,
                sigma,
                threshold,
            } => write!(
                f,
                "metric_cache: cache file {} has σ = {:.3e} ≥ threshold {:.3e}; \
                 publication-grade quality not satisfied; recompute fresh",
                path.display(),
                sigma,
                threshold
            ),
        }
    }
}

impl std::error::Error for MetricCacheError {}

impl From<std::io::Error> for MetricCacheError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for MetricCacheError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serde(e)
    }
}

pub type Result<T> = std::result::Result<T, MetricCacheError>;

// ---------------------------------------------------------------------------
// Cache-key serialisation forms
//
// We do NOT directly hash `TyMetricConfig` / `SchoenMetricConfig` because
// they may grow new fields over time. Instead, we hash a stable
// canonicalised struct whose schema we control here.
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct TyKeyMaterial<'a> {
    schema_version: u32,
    manifold: &'a str,
    k_degree: u32,
    n_sample: usize,
    max_iter: usize,
    donaldson_tol_bits: u64,
    seed: u64,
    apply_z3_quotient: bool,
}

#[derive(Serialize)]
struct SchoenKeyMaterial<'a> {
    schema_version: u32,
    manifold: &'a str,
    d_x: u32,
    d_y: u32,
    d_t: u32,
    n_sample: usize,
    max_iter: usize,
    donaldson_tol_bits: u64,
    seed: u64,
    apply_z3xz3_quotient: bool,
    // Schoen-specific bundle-S5-flag (toggled to invalidate the cache
    // when the parallel S5 fix lands). For now this is `false` to
    // match `bayes_discriminate.rs`'s placeholder bundle path; flip
    // to `true` when the Schoen branch is wired to its genuine bundle.
    schoen_bundle_canonical: bool,
}

/// Stable 8-hex-char hash prefix of the canonical-JSON of `key`.
fn hash_prefix<S: Serialize>(key: &S) -> Result<String> {
    let buf = serde_json::to_vec(key)?;
    let mut hasher = Sha256::new();
    hasher.update(&buf);
    let digest = hasher.finalize();
    let hex_full = hex::encode(digest);
    Ok(hex_full[..16].to_string())
}

/// Build cache file path for a TY config under `cache_root`.
pub fn ty_cache_path(cache_root: &Path, cfg: &TyMetricConfig) -> Result<PathBuf> {
    let key = TyKeyMaterial {
        schema_version: 1,
        manifold: "tian_yau_z3",
        k_degree: cfg.k_degree,
        n_sample: cfg.n_sample,
        max_iter: cfg.max_iter,
        donaldson_tol_bits: cfg.donaldson_tol.to_bits(),
        seed: cfg.seed,
        apply_z3_quotient: cfg.apply_z3_quotient,
    };
    let prefix = hash_prefix(&key)?;
    let fname = format!(
        "ty_k{}_n{}_i{}_seed{}_{}.json",
        cfg.k_degree, cfg.n_sample, cfg.max_iter, cfg.seed, prefix
    );
    Ok(cache_root.join(fname))
}

/// Build cache file path for a Schoen config under `cache_root`.
pub fn schoen_cache_path(cache_root: &Path, cfg: &SchoenMetricConfig) -> Result<PathBuf> {
    // S5-canonical-bundle flag: while a parallel agent is fixing the
    // Schoen branch (currently using the TY ambient as a placeholder
    // — see `bayes_discriminate.rs:585-591`), keep this `false`. When
    // S5 lands, flip to `true` here AND in `bayes_discriminate.rs` so
    // pre-S5 caches MISS. This is sized at the cache-key level, NOT
    // the metric solver — the metric itself is independent of the
    // bundle, but we pessimistically invalidate to avoid a future
    // operator stepping on a stale-bundle cache.
    let key = SchoenKeyMaterial {
        schema_version: 1,
        manifold: "schoen_z3xz3",
        d_x: cfg.d_x,
        d_y: cfg.d_y,
        d_t: cfg.d_t,
        n_sample: cfg.n_sample,
        max_iter: cfg.max_iter,
        donaldson_tol_bits: cfg.donaldson_tol.to_bits(),
        seed: cfg.seed,
        apply_z3xz3_quotient: cfg.apply_z3xz3_quotient,
        schoen_bundle_canonical: false,
    };
    let prefix = hash_prefix(&key)?;
    let fname = format!(
        "schoen_dx{}_dy{}_dt{}_n{}_i{}_seed{}_{}.json",
        cfg.d_x, cfg.d_y, cfg.d_t, cfg.n_sample, cfg.max_iter, cfg.seed, prefix
    );
    Ok(cache_root.join(fname))
}

/// Outcome of a `load_or_solve_*` call. Used for diagnostic reporting.
#[derive(Debug, Clone, Copy)]
pub enum CacheOutcome {
    Hit,
    Miss,
    ForcedRecompute,
}

/// Load a `TyMetricResult` from `cache_root`, or solve fresh and write
/// the cache if missing (or `force_recompute`).
///
/// Quality guard: if `strict_quality` is `true`, rejects results whose
/// `final_sigma_residual` is at or above [`SIGMA_QUALITY_THRESHOLD`]
/// (the Donaldson 2009 publication-grade target). If
/// `strict_quality` is `false`, the result is still cached / loaded,
/// but a strong warning is emitted on stderr — the underlying solver
/// has not reached publication grade. This permits forward progress
/// in the harness while making the quality gap visible.
pub fn load_or_solve_ty(
    cache_root: &Path,
    cfg: &TyMetricConfig,
    force_recompute: bool,
    strict_quality: bool,
) -> Result<(TyMetricResult, CacheOutcome, PathBuf, std::time::Duration)> {
    fs::create_dir_all(cache_root)?;
    let path = ty_cache_path(cache_root, cfg)?;
    let started = std::time::Instant::now();

    if !force_recompute && path.exists() {
        let bytes = fs::read(&path)?;
        let result: TyMetricResult = serde_json::from_slice(&bytes)?;
        let elapsed = started.elapsed();
        check_sigma_quality(&path, result.final_sigma_residual, strict_quality)?;
        return Ok((result, CacheOutcome::Hit, path, elapsed));
    }

    let outcome = if force_recompute {
        CacheOutcome::ForcedRecompute
    } else {
        CacheOutcome::Miss
    };

    let result = solve_ty_metric(cfg.clone()).map_err(MetricCacheError::SolveTy)?;
    check_sigma_quality(&path, result.final_sigma_residual, strict_quality)?;
    let bytes = serde_json::to_vec(&result)?;
    // Atomic write: write to tmp then rename.
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, &bytes)?;
    fs::rename(&tmp, &path)?;
    let elapsed = started.elapsed();
    Ok((result, outcome, path, elapsed))
}

/// Same as [`load_or_solve_ty`] for the Schoen branch.
pub fn load_or_solve_schoen(
    cache_root: &Path,
    cfg: &SchoenMetricConfig,
    force_recompute: bool,
    strict_quality: bool,
) -> Result<(SchoenMetricResult, CacheOutcome, PathBuf, std::time::Duration)> {
    fs::create_dir_all(cache_root)?;
    let path = schoen_cache_path(cache_root, cfg)?;
    let started = std::time::Instant::now();

    if !force_recompute && path.exists() {
        let bytes = fs::read(&path)?;
        let result: SchoenMetricResult = serde_json::from_slice(&bytes)?;
        let elapsed = started.elapsed();
        check_sigma_quality(&path, result.final_sigma_residual, strict_quality)?;
        return Ok((result, CacheOutcome::Hit, path, elapsed));
    }

    let outcome = if force_recompute {
        CacheOutcome::ForcedRecompute
    } else {
        CacheOutcome::Miss
    };

    let result = solve_schoen_metric(cfg.clone()).map_err(MetricCacheError::SolveSchoen)?;
    check_sigma_quality(&path, result.final_sigma_residual, strict_quality)?;
    let bytes = serde_json::to_vec(&result)?;
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, &bytes)?;
    fs::rename(&tmp, &path)?;
    let elapsed = started.elapsed();
    Ok((result, outcome, path, elapsed))
}

/// σ-residual quality gate. In strict mode, return `Err(StaleQuality)`
/// if the residual is non-finite or at-or-above the publication
/// threshold; in non-strict mode, emit a stderr warning and proceed.
fn check_sigma_quality(path: &Path, sigma: f64, strict: bool) -> Result<()> {
    let bad = !sigma.is_finite() || sigma >= SIGMA_QUALITY_THRESHOLD;
    if !bad {
        return Ok(());
    }
    if strict {
        return Err(MetricCacheError::StaleQuality {
            path: path.to_path_buf(),
            sigma,
            threshold: SIGMA_QUALITY_THRESHOLD,
        });
    }
    eprintln!(
        "WARNING: metric_cache: {} has σ = {:.3e} ≥ Donaldson-2009 publication \
         threshold {:.3e}. Cached / loaded anyway because strict_quality=false. \
         Re-run with --strict-metric-quality to escalate this to a hard error.",
        path.display(),
        sigma,
        SIGMA_QUALITY_THRESHOLD,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ty_path_includes_all_key_fields() {
        let cfg_a = TyMetricConfig {
            k_degree: 4,
            n_sample: 1000,
            max_iter: 20,
            donaldson_tol: 1.0e-3,
            seed: 42,
            checkpoint_path: None,
            apply_z3_quotient: true,
            adam_refine: None,
            use_gpu: false,
            donaldson_damping: None,
            donaldson_tikhonov_shift: None,
        };
        let mut cfg_b = cfg_a.clone();
        cfg_b.seed = 43;
        let mut cfg_c = cfg_a.clone();
        cfg_c.k_degree = 5;
        let mut cfg_d = cfg_a.clone();
        cfg_d.donaldson_tol = 1.0e-4;

        let root = std::path::Path::new("target/test_metric_cache");
        let p_a = ty_cache_path(root, &cfg_a).unwrap();
        let p_b = ty_cache_path(root, &cfg_b).unwrap();
        let p_c = ty_cache_path(root, &cfg_c).unwrap();
        let p_d = ty_cache_path(root, &cfg_d).unwrap();
        assert_ne!(p_a, p_b, "different seed must yield different cache key");
        assert_ne!(p_a, p_c, "different k_degree must yield different cache key");
        assert_ne!(p_a, p_d, "different tol must yield different cache key");
    }

    #[test]
    fn schoen_path_includes_all_key_fields() {
        let cfg_a = SchoenMetricConfig {
            d_x: 4,
            d_y: 4,
            d_t: 2,
            n_sample: 1000,
            max_iter: 20,
            donaldson_tol: 1.0e-3,
            seed: 42,
            checkpoint_path: None,
            apply_z3xz3_quotient: true,
            adam_refine: None,
            use_gpu: false,
            donaldson_damping: None,
            donaldson_tikhonov_shift: None,
        };
        let mut cfg_b = cfg_a.clone();
        cfg_b.d_x = 5;
        let mut cfg_c = cfg_a.clone();
        cfg_c.apply_z3xz3_quotient = false;

        let root = std::path::Path::new("target/test_metric_cache");
        let p_a = schoen_cache_path(root, &cfg_a).unwrap();
        let p_b = schoen_cache_path(root, &cfg_b).unwrap();
        let p_c = schoen_cache_path(root, &cfg_c).unwrap();
        assert_ne!(p_a, p_b);
        assert_ne!(p_a, p_c);
    }

    #[test]
    fn quality_threshold_constant_matches_donaldson_2009() {
        assert_eq!(SIGMA_QUALITY_THRESHOLD, 1.0e-3);
    }
}
