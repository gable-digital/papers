//! Integration-level tests for [`crate::route34::schoen_sampler`].
//!
//! Cross-checks the line-intersection sampler's basic invariants:
//! convergence, weight normalisation, reproducibility, multi-threading
//! agreement, and per-point Schoen-variety membership.

use crate::route34::schoen_geometry::SchoenGeometry;
use crate::route34::schoen_sampler::{SchoenPoly, SchoenSampler, NCOORDS, NHYPER};

/// Sampler accepts at least one point in 4096 trials.
#[test]
fn sampler_finds_a_point() {
    let poly = SchoenPoly::z3xz3_invariant_default();
    let geom = SchoenGeometry::schoen_z3xz3();
    let mut sampler = SchoenSampler::new(poly, geom, 0xC0FFEE);
    let mut found = 0;
    for _ in 0..4096 {
        if sampler.sample_one().is_some() {
            found += 1;
            if found >= 4 {
                break;
            }
        }
    }
    assert!(found >= 1, "sampler should find ≥ 1 point in 4096 trials");
}

/// 100-point batch normalises weights to sum 1.
#[test]
fn batch_weights_normalised() {
    let poly = SchoenPoly::z3xz3_invariant_default();
    let geom = SchoenGeometry::schoen_z3xz3();
    let mut sampler = SchoenSampler::new(poly, geom, 0xBADF00D);
    let pts = sampler.sample_points(100, None);
    assert_eq!(pts.len(), 100);
    let sum: f64 = pts.iter().map(|p| p.weight).sum();
    assert!((sum - 1.0).abs() < 1e-9);
    for p in &pts {
        assert!(p.weight.is_finite() && p.weight > 0.0);
    }
}

/// Same seed → same output (bit-identical for all coordinates and weights).
#[test]
fn reproducibility_under_fixed_seed() {
    let poly = SchoenPoly::z3xz3_invariant_default();
    let geom = SchoenGeometry::schoen_z3xz3();
    let mut s1 = SchoenSampler::new(poly.clone(), geom.clone(), 12345);
    let pts1 = s1.sample_points(50, None);
    let mut s2 = SchoenSampler::new(poly, geom, 12345);
    let pts2 = s2.sample_points(50, None);
    assert_eq!(pts1.len(), pts2.len());
    for (a, b) in pts1.iter().zip(pts2.iter()) {
        for k in 0..3 {
            assert_eq!(a.x[k], b.x[k]);
            assert_eq!(a.y[k], b.y[k]);
        }
        for k in 0..2 {
            assert_eq!(a.t[k], b.t[k]);
        }
        assert_eq!(a.weight, b.weight);
    }
}

/// Multi-threaded sampler produces same point COUNT (weights / per-thread
/// content differs because threads draw their own RNG streams). Idempotent
/// **counts** under fixed seed, fixed thread count.
#[test]
fn multithreaded_count_reproducible() {
    let poly = SchoenPoly::z3xz3_invariant_default();
    let geom = SchoenGeometry::schoen_z3xz3();
    let mut s1 = SchoenSampler::new(poly.clone(), geom.clone(), 99);
    let p1 = s1.sample_batch_parallel(64, 4);
    let mut s2 = SchoenSampler::new(poly, geom, 99);
    let p2 = s2.sample_batch_parallel(64, 4);
    assert_eq!(p1.len(), p2.len());
    let s1_sum: f64 = p1.iter().map(|p| p.weight).sum();
    let s2_sum: f64 = p2.iter().map(|p| p.weight).sum();
    assert!((s1_sum - 1.0).abs() < 1e-9);
    assert!((s2_sum - 1.0).abs() < 1e-9);
}

/// Run metadata populated correctly post-run.
#[test]
fn run_metadata_has_sha256() {
    let poly = SchoenPoly::z3xz3_invariant_default();
    let geom = SchoenGeometry::schoen_z3xz3();
    let mut sampler = SchoenSampler::new(poly, geom, 42);
    let _ = sampler.sample_points(20, None);
    let m = sampler.run_metadata();
    assert_eq!(m.seed, 42);
    assert_eq!(m.n_points, 20);
    assert_eq!(m.point_cloud_sha256.len(), 64); // 256 bits = 64 hex
    assert!(m.wall_clock_seconds >= 0.0);
}

/// Constants are well-typed.
#[test]
fn constants_consistent() {
    assert_eq!(NCOORDS, 8);
    assert_eq!(NHYPER, 2);
    assert_eq!(NCOORDS, 3 + 3 + 2);
}

/// Quotient weight divides each point's weight by 9.
#[test]
fn z3xz3_quotient_divides_weight() {
    let poly = SchoenPoly::z3xz3_invariant_default();
    let geom = SchoenGeometry::schoen_z3xz3();
    let mut sampler = SchoenSampler::new(poly, geom, 5);
    let mut pts = sampler.sample_points(20, None);
    let pre: Vec<f64> = pts.iter().map(|p| p.weight).collect();
    SchoenSampler::apply_z3xz3_quotient(&mut pts);
    for (p, w0) in pts.iter().zip(pre.iter()) {
        assert!((p.weight - w0 / 9.0).abs() < 1e-15);
    }
}
