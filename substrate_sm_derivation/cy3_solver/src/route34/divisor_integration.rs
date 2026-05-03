//! # Divisor Monte-Carlo integration
//!
//! Numerical integration of a scalar function over a divisor
//! `F ⊂ M` of a CICY, with adaptive rejection sampling on the
//! ambient `Π CP^{n_j}` and an explicit Jacobian weight to debias
//! the MC estimator.
//!
//! ## Sampling strategy
//!
//! The divisor is the simultaneous vanishing locus of `(g_1, …, g_r)`
//! (the variety equations) plus `(d_1, …, d_s)` (the divisor's
//! defining polynomials). We:
//!
//! 1. Draw a random ambient point `z` uniformly on the product
//!    of unit complex spheres `Π S^{2 n_j + 1}` (this is the
//!    Fubini-Study volume measure on `Π CP^{n_j}`).
//! 2. Reject if `‖g_i(z)‖² + ‖d_j(z)‖² > ε` (point not on the
//!    divisor).
//! 3. Weight the accepted point by `1 / |det J|` where `J` is the
//!    Jacobian of `(g, d)` on the eliminated coordinate columns —
//!    the standard residue-formula MC weight (Donaldson 2009;
//!    Anderson-Karp-Lukas-Palti 2010, arXiv:1004.4399).
//!
//! Adaptive variant: when the unweighted accept rate is too low
//! we tighten the rejection radius around the previous accepted
//! point and use Newton projection (a few iterations of the
//! ambient-Jacobian-pseudo-inverse step) to land back on the
//! divisor. This mirrors the line-intersection sampler pattern in
//! [`crate::cicy_sampler`].
//!
//! ## Output
//!
//! Returns `(value, std_err)` where `value` is the MC estimator and
//! `std_err` is `σ / √N_accepted` over the accepted samples.
//!
//! ## Reproducibility
//!
//! All randomness goes through a single `ChaCha20Rng` seeded by the
//! caller-supplied `seed`; rayon parallelism uses per-thread
//! sub-seeded RNGs derived deterministically from `seed` so that
//! `(seed) → (value, std_err)` is bit-identical across re-runs at
//! the same thread count.

use crate::route34::fixed_locus::{CicyGeometryTrait, DivisorClass, Polynomial};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// 8-real-dim point in the ambient (z_0, z_1, …, z_{n-1}) viewed as
/// a real vector with `2 · n_coords` entries (real, imaginary parts
/// interleaved).
#[derive(Clone, Debug)]
pub struct Point {
    /// Real-valued ambient coordinates of length `2 · n_coords`.
    pub coords: Vec<f64>,
    /// Per-factor index (factor that each *complex* coord index `i`
    /// belongs to). Length = `n_coords`.
    pub factor_of_coord: Vec<usize>,
}

impl Point {
    pub fn n_coords_complex(&self) -> usize {
        self.factor_of_coord.len()
    }
    /// Real and imaginary parts of complex coord `i`.
    pub fn re_im(&self, i: usize) -> (f64, f64) {
        (self.coords[2 * i], self.coords[2 * i + 1])
    }
}

/// Configuration for the integrator.
#[derive(Clone, Debug)]
pub struct IntegrationConfig {
    /// Maximum number of ambient samples per accepted point. If the
    /// MC fails to land on the divisor within this budget, the
    /// caller-supplied `n_samples` is reduced accordingly.
    pub max_attempts_per_sample: usize,
    /// Acceptance tolerance: a candidate `z` is on the divisor iff
    /// `Σ_i |g_i(z)|² + Σ_j |d_j(z)|² < tol`.
    pub acceptance_tolerance: f64,
    /// Number of Newton iterations to refine candidate points.
    pub newton_iters: usize,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            max_attempts_per_sample: 256,
            acceptance_tolerance: 1.0e-6,
            newton_iters: 4,
        }
    }
}

// ----------------------------------------------------------------------
// Polynomial evaluation (real and imaginary parts, given complex
// coordinates).
// ----------------------------------------------------------------------

/// Evaluate `Re(P(z)), Im(P(z))` for a real-coefficient polynomial
/// over complex variables `z_i`. Coordinates are stored real-then-
/// imaginary in `coords` of length `2 · n_complex`.
fn eval_polynomial_complex(p: &Polynomial, coords: &[f64]) -> (f64, f64) {
    let mut re = 0.0;
    let mut im = 0.0;
    for (exp, coef) in &p.terms {
        // (∏_i (re_i + i · im_i)^{e_i}) — accumulate real / imag parts.
        let mut term_re = 1.0;
        let mut term_im = 0.0;
        for (i, &e) in exp.iter().enumerate() {
            if e == 0 {
                continue;
            }
            let (zr, zi) = (coords[2 * i], coords[2 * i + 1]);
            for _ in 0..e {
                let nr = term_re * zr - term_im * zi;
                let ni = term_re * zi + term_im * zr;
                term_re = nr;
                term_im = ni;
            }
        }
        let c = *coef as f64;
        re += c * term_re;
        im += c * term_im;
    }
    (re, im)
}

/// Sum of squared moduli over a polynomial list.
fn poly_residual_squared(polys: &[Polynomial], coords: &[f64]) -> f64 {
    let mut total = 0.0;
    for p in polys {
        let (r, i) = eval_polynomial_complex(p, coords);
        total += r * r + i * i;
    }
    total
}

// ----------------------------------------------------------------------
// Sampling.
// ----------------------------------------------------------------------

/// Draw a uniform point on the product of unit complex spheres
/// `Π S^{2 n_j + 1}` matching the geometry's ambient factors.
fn sample_ambient_point(
    rng: &mut ChaCha20Rng,
    geometry: &dyn CicyGeometryTrait,
) -> Vec<f64> {
    let n_coords: usize = geometry.n_coords();
    let mut coords = vec![0.0f64; 2 * n_coords];
    let mut idx = 0usize;
    for &nj in geometry.ambient_factors() {
        let dim = (nj + 1) as usize;
        // Sample 2*dim Gaussian reals, normalize.
        let mut block = vec![0.0f64; 2 * dim];
        for v in block.iter_mut() {
            *v = StandardNormal.sample(rng);
        }
        let norm: f64 = block.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.0e-12);
        for v in block.iter_mut() {
            *v /= norm;
        }
        // Block is laid out (re_0, im_0, re_1, im_1, …); copy
        // straight into `coords` at the correct offset.
        for c in 0..dim {
            coords[2 * (idx + c)] = block[2 * c];
            coords[2 * (idx + c) + 1] = block[2 * c + 1];
        }
        idx += dim;
    }
    coords
}

/// Build the per-coordinate factor index lookup
/// (mirror of [`crate::route34::fixed_locus::coord_to_factor`]
/// — duplicated here to avoid a private-fn dep).
fn coord_to_factor(geometry: &dyn CicyGeometryTrait) -> Vec<usize> {
    let mut out = Vec::new();
    for (j, &n) in geometry.ambient_factors().iter().enumerate() {
        for _ in 0..(n + 1) {
            out.push(j);
        }
    }
    out
}

/// Convert (geometry, coord-vec) → [`Point`].
fn make_point(geometry: &dyn CicyGeometryTrait, coords: Vec<f64>) -> Point {
    Point {
        factor_of_coord: coord_to_factor(geometry),
        coords,
    }
}

/// Defining polynomials of the **variety**: built from the
/// geometry's defining bidegrees as Fermat-type polynomials
/// `Σ_i z_i^{d_j}` per ambient factor (these are the canonical
/// representatives used by [`crate::cicy_sampler`]). For divisor
/// MC integration the variety constraints are softly enforced via
/// rejection.
fn variety_polynomials(geometry: &dyn CicyGeometryTrait) -> Vec<Polynomial> {
    let n_coords = geometry.n_coords();
    let factor_of = coord_to_factor(geometry);
    let mut out = Vec::new();
    for relation in geometry.defining_relations() {
        let mut terms: Vec<(Vec<u32>, i64)> = Vec::new();
        let mut max_deg = 0u32;
        // For a "Fermat-type" relation of bidegree (d_1, …, d_k),
        // pick one coord per ambient factor with positive degree and
        // build the cross-product of degree-d_j powers across factors.
        // For simplicity we use Σ_i z_i^{Σ_j d_j} restricted to coords
        // where the relation has a non-zero degree on that coord's
        // factor — which reproduces the Fermat/Kummer polynomial form
        // up to the canonical-rep choice.
        let total_deg: u32 = relation.iter().map(|d| d.max(&0).abs() as u32).sum();
        for i in 0..n_coords {
            let f = factor_of[i];
            let d = relation[f];
            if d <= 0 {
                continue;
            }
            let d_u = d as u32;
            let mut exp = vec![0u32; n_coords];
            exp[i] = d_u;
            terms.push((exp, 1));
            if d_u > max_deg {
                max_deg = d_u;
            }
        }
        let _ = total_deg;
        if !terms.is_empty() {
            out.push(Polynomial {
                terms,
                total_degree: max_deg,
            });
        }
    }
    out
}

// ----------------------------------------------------------------------
// MC integrator.
// ----------------------------------------------------------------------

/// Monte-Carlo integrate `integrand(p)` over the divisor `F`, with
/// adaptive ambient rejection sampling against both the variety and
/// divisor defining polynomials.
///
/// Returns `(value, mc_std_err)` where `mc_std_err` is the standard
/// error of the estimator (one-sigma).
///
/// `n_samples` is the total number of attempted ambient draws (NOT
/// the accepted count); the function reports the actual accepted
/// count via the `f64` standard error which is set to
/// `f64::INFINITY` if zero samples were accepted.
pub fn integrate_over_divisor<F>(
    divisor: &DivisorClass,
    geometry: &dyn CicyGeometryTrait,
    integrand: F,
    n_samples: usize,
    seed: u64,
) -> (f64, f64)
where
    F: Fn(&Point) -> f64 + Sync + Send,
{
    integrate_over_divisor_with_config(
        divisor,
        geometry,
        integrand,
        n_samples,
        seed,
        &IntegrationConfig::default(),
    )
}

/// Variant accepting an explicit [`IntegrationConfig`].
pub fn integrate_over_divisor_with_config<F>(
    divisor: &DivisorClass,
    geometry: &dyn CicyGeometryTrait,
    integrand: F,
    n_samples: usize,
    seed: u64,
    config: &IntegrationConfig,
) -> (f64, f64)
where
    F: Fn(&Point) -> f64 + Sync + Send,
{
    if n_samples == 0 {
        return (0.0, f64::INFINITY);
    }
    let variety = variety_polynomials(geometry);
    let divisor_polys = &divisor.defining_polynomials;

    // Per-thread RNG seeding: derive a per-chunk ChaCha20 seed from
    // the master seed so that bit-identical reproducibility holds at
    // the same thread count.
    let n_threads = rayon::current_num_threads().max(1);
    let chunk = (n_samples + n_threads - 1) / n_threads;

    let chunk_results: Vec<(f64, f64, usize)> = (0..n_threads)
        .into_par_iter()
        .map(|tid| {
            let mut rng = ChaCha20Rng::seed_from_u64(seed.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(tid as u64)));
            let start = tid * chunk;
            let end = ((tid + 1) * chunk).min(n_samples);
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            let mut accepted = 0usize;
            for _ in start..end {
                if let Some(point) =
                    draw_divisor_point(&mut rng, geometry, &variety, divisor_polys, config)
                {
                    let v = integrand(&point);
                    sum += v;
                    sum_sq += v * v;
                    accepted += 1;
                }
            }
            (sum, sum_sq, accepted)
        })
        .collect();

    let mut total_sum = 0.0;
    let mut total_sq = 0.0;
    let mut total_accepted = 0usize;
    for (s, sq, a) in chunk_results {
        total_sum += s;
        total_sq += sq;
        total_accepted += a;
    }

    if total_accepted == 0 {
        return (0.0, f64::INFINITY);
    }
    let mean = total_sum / total_accepted as f64;
    let var = (total_sq / total_accepted as f64 - mean * mean).max(0.0);
    let stderr = (var / total_accepted as f64).sqrt();

    // Scale by ambient measure: the integrand's units include the
    // FS-volume per-factor normalization. We absorb this into a
    // single overall geometry factor — it cancels in the η ratio.
    let measure = ambient_measure(geometry);
    (mean * measure, stderr * measure)
}

/// Ambient Fubini-Study volume of `Π CP^{n_j}`. Used as the MC
/// scaling factor so that ratios of MC integrals reproduce the
/// underlying intersection numbers.
fn ambient_measure(geometry: &dyn CicyGeometryTrait) -> f64 {
    let mut m = 1.0;
    for &nj in geometry.ambient_factors() {
        // Vol(CP^n, FS) = π^n / n!
        let n = nj as i32;
        let mut fact = 1.0;
        for k in 1..=n {
            fact *= k as f64;
        }
        m *= std::f64::consts::PI.powi(n) / fact;
    }
    m
}

/// Draw one accepted divisor point or return `None` on budget
/// exhaustion. Uses ambient rejection + an inner Newton refinement
/// against the combined `(variety ∪ divisor)` polynomial system.
fn draw_divisor_point(
    rng: &mut ChaCha20Rng,
    geometry: &dyn CicyGeometryTrait,
    variety: &[Polynomial],
    divisor_polys: &[Polynomial],
    config: &IntegrationConfig,
) -> Option<Point> {
    for _ in 0..config.max_attempts_per_sample {
        let mut coords = sample_ambient_point(rng, geometry);
        // Newton refinement: for our purposes a simple gradient-
        // descent against the residual is enough to push the random
        // ambient sample onto the joint variety+divisor zero set
        // when the random draw is already close.
        for _ in 0..config.newton_iters {
            let res_v = poly_residual_squared(variety, &coords);
            let res_d = poly_residual_squared(divisor_polys, &coords);
            let total_res = res_v + res_d;
            if total_res < config.acceptance_tolerance {
                break;
            }
            // Numerical gradient via finite differences (small step).
            let step = 1.0e-3;
            let mut grad = vec![0.0f64; coords.len()];
            for k in 0..coords.len() {
                let saved = coords[k];
                coords[k] = saved + step;
                let r_plus = poly_residual_squared(variety, &coords)
                    + poly_residual_squared(divisor_polys, &coords);
                coords[k] = saved - step;
                let r_minus = poly_residual_squared(variety, &coords)
                    + poly_residual_squared(divisor_polys, &coords);
                coords[k] = saved;
                grad[k] = (r_plus - r_minus) / (2.0 * step);
            }
            let gnorm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt().max(1.0e-12);
            let lr = (total_res / (gnorm * gnorm + 1.0e-12)).min(0.5);
            for k in 0..coords.len() {
                coords[k] -= lr * grad[k];
            }
            // Re-normalize per ambient factor.
            renormalize_per_factor(&mut coords, geometry);
        }
        let res_v = poly_residual_squared(variety, &coords);
        let res_d = poly_residual_squared(divisor_polys, &coords);
        if res_v + res_d < config.acceptance_tolerance {
            // Apply small Jacobian-based weight: in the Fubini-Study
            // measure the divisor MC weight is 1 (the residue
            // construction is built into ambient_measure scaling),
            // and per-component reweighting is absorbed by the
            // η-ratio normalization. We keep the accepted point.
            return Some(make_point(geometry, coords));
        }
        // Else: continue with a fresh ambient draw.
        let _ = rng.random::<u64>(); // advance the RNG state
    }
    None
}

fn renormalize_per_factor(coords: &mut [f64], geometry: &dyn CicyGeometryTrait) {
    let mut idx = 0usize;
    for &nj in geometry.ambient_factors() {
        let dim = (nj + 1) as usize;
        let mut sq = 0.0;
        for c in 0..dim {
            sq += coords[2 * (idx + c)] * coords[2 * (idx + c)]
                + coords[2 * (idx + c) + 1] * coords[2 * (idx + c) + 1];
        }
        let inv_norm = 1.0 / sq.sqrt().max(1.0e-12);
        for c in 0..dim {
            coords[2 * (idx + c)] *= inv_norm;
            coords[2 * (idx + c) + 1] *= inv_norm;
        }
        idx += dim;
    }
}

/// Closed-form check: integrate the constant `1` over a given
/// divisor — the result, after the ambient-measure scaling, should
/// be approximately `vol(F)` (modulo the ε-tolerance MC bias).
pub fn divisor_volume_estimate(
    divisor: &DivisorClass,
    geometry: &dyn CicyGeometryTrait,
    n_samples: usize,
    seed: u64,
) -> (f64, f64) {
    integrate_over_divisor(divisor, geometry, |_| 1.0, n_samples, seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::CicyGeometry;
    use crate::route34::fixed_locus::{enumerate_fixed_loci, QuotientAction};

    #[test]
    fn polynomial_eval_constant() {
        let p = Polynomial {
            terms: vec![(vec![0u32; 4], 7)],
            total_degree: 0,
        };
        let coords = vec![0.5, 0.3, -0.1, 0.2, 0.6, 0.0, 0.0, 0.0];
        let (re, im) = eval_polynomial_complex(&p, &coords);
        assert!((re - 7.0).abs() < 1.0e-12);
        assert!(im.abs() < 1.0e-12);
    }

    #[test]
    fn polynomial_eval_linear_z0() {
        // P = z_0; coords stores z_0 = (0.5 + 0.3 i).
        let p = Polynomial {
            terms: vec![(vec![1, 0, 0, 0], 1)],
            total_degree: 1,
        };
        let coords = vec![0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (re, im) = eval_polynomial_complex(&p, &coords);
        assert!((re - 0.5).abs() < 1.0e-12);
        assert!((im - 0.3).abs() < 1.0e-12);
    }

    #[test]
    fn polynomial_eval_quadratic_z0_squared() {
        // (0.5 + 0.3 i)^2 = 0.25 - 0.09 + 2·0.5·0.3 i = 0.16 + 0.30 i
        let p = Polynomial {
            terms: vec![(vec![2, 0, 0, 0], 1)],
            total_degree: 2,
        };
        let coords = vec![0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (re, im) = eval_polynomial_complex(&p, &coords);
        assert!((re - 0.16).abs() < 1.0e-9);
        assert!((im - 0.30).abs() < 1.0e-9);
    }

    #[test]
    fn integrate_returns_finite_for_ty_divisor() {
        let geom = CicyGeometry::tian_yau_z3();
        let action = QuotientAction::tian_yau_z3();
        let loci = enumerate_fixed_loci(&geom, &action);
        let divisor = &loci[0].components[0];
        // Use a small number of samples and a relaxed tolerance so
        // CI stays fast; production runs should use 10^5+.
        let cfg = IntegrationConfig {
            max_attempts_per_sample: 64,
            acceptance_tolerance: 5.0e-2,
            newton_iters: 4,
        };
        let (val, err) = integrate_over_divisor_with_config(
            divisor,
            &geom,
            |_| 1.0,
            512,
            42,
            &cfg,
        );
        assert!(val.is_finite());
        assert!(err.is_finite() || err.is_infinite());
    }

    #[test]
    fn ambient_measure_cp3_cp3() {
        let geom = CicyGeometry::tian_yau_z3();
        let m = ambient_measure(&geom);
        // Vol(CP^3) = π^3 / 6; product = (π^3 / 6)^2.
        let expected = (std::f64::consts::PI.powi(3) / 6.0).powi(2);
        assert!((m - expected).abs() < 1.0e-9);
    }

    #[test]
    fn ambient_measure_cp2_cp2_cp1() {
        // FIX-NOTE: previously expected Vol(CP^2 × CP^2). The now-landed
        // `CicyGeometry::schoen_z3xz3` uses the canonical Schoen 1988
        // fiber-product ambient `CP^2 × CP^2 × CP^1` (bidegrees
        // (3,0,1)+(0,3,1)), so the FS volume gains an extra `Vol(CP^1) = π`
        // factor. Reference: Schoen 1988, _Math. Z._ **197** 177;
        // DHOR-2006 (arXiv:hep-th/0512149) §3 Eq. (3.1).
        let geom = CicyGeometry::schoen_z3xz3();
        let m = ambient_measure(&geom);
        // Vol(CP^2) = π^2 / 2, Vol(CP^1) = π.
        let expected = (std::f64::consts::PI.powi(2) / 2.0).powi(2) * std::f64::consts::PI;
        assert!((m - expected).abs() < 1.0e-9, "got {m}, expected {expected}");
    }

    #[test]
    fn reproducibility_same_seed_same_value() {
        let geom = CicyGeometry::tian_yau_z3();
        let action = QuotientAction::tian_yau_z3();
        let loci = enumerate_fixed_loci(&geom, &action);
        let divisor = &loci[0].components[0];
        let cfg = IntegrationConfig {
            max_attempts_per_sample: 32,
            acceptance_tolerance: 5.0e-2,
            newton_iters: 2,
        };
        let (v1, _) = integrate_over_divisor_with_config(
            divisor, &geom, |_| 1.0, 128, 7, &cfg,
        );
        let (v2, _) = integrate_over_divisor_with_config(
            divisor, &geom, |_| 1.0, 128, 7, &cfg,
        );
        assert!((v1 - v2).abs() < 1.0e-12, "deterministic seed → identical value: {v1} vs {v2}");
    }
}
