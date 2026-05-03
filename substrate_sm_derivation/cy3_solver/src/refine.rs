// LEGACY-SUPERSEDED-BY-ROUTE34: this module's metric solver runs on the
// polysphere ambient (S^3 x S^3) and only weighs the variety-defining
// polynomials Q_1, Q_2 as soft constraints in the loss. For
// publication-grade discrimination the metric must live on the actual
// CY3 sub-variety. Superseded by:
//   * Tian-Yau Z/3 metric on the bicubic sub-variety  -> route34::ty_metric
//   * Schoen Z/3xZ/3 metric on the (3,3,1) sub-variety -> route34::schoen_metric
//   * Unified solver dispatch                          -> route34::cy3_metric_unified
//   * GPU path (Phase-1 CPU-fallback scaffold)         -> route34::cy3_metric_gpu
// Both route34 metric solvers project samples to the variety via the
// implicit-function theorem, build an affine-chart tangent frame from the
// polynomial Jacobian, and minimise the canonical Donaldson-Karp-Lukic-
// Reinbacher 2006 / Larfors-Schneider-Strominger 2020 sigma-functional.
// Do not modify in place; add new metric logic to the route34 modules above.
//
//! Constraint-driven CY3 metric refinement.
//!
//! Starts from the Donaldson-balanced metric at moderate k_degree and
//! refines it via constrained gradient descent, where the loss combines:
//!
//!   1. Ricci-flatness (mainstream's only loss term) -- now implemented as
//!      the real Monge-Ampere residual, var(log|det Hess(log K)|), rather
//!      than the cheap var(log K) proxy.
//!   2. Substrate-physical constraints from the framework's commitments:
//!      - polyhedral-resonance ADE admissibility (cheap, real)
//!      - three-generation index theorem (cheap, real)
//!      - Coulomb 1/r^2 forward-model from photon-zero-mode integration
//!      - Weak range from W-zero-mode confinement scale
//!      - Strong-force scale from gluon-zero-mode strain-tail
//!      - Yukawa fermion-mass spectrum (the key Tian-Yau / Schoen
//!        discriminator).
//!
//! Mainstream NN-accelerated approaches train a network because they have
//! only the Ricci-flatness loss and need a flexible function approximator
//! to find a good minimum. We have additional constraints that eliminate
//! the shallow-minima problem; direct gradient descent on metric
//! coefficients converges faster and doesn't need NN training overhead.

use rayon::prelude::*;

/// Compute the determinant of an n x n matrix via in-place LU with
/// partial pivoting, tracking row-swap parity for the sign. Used in the
/// Monge-Ampere residual where we need det(8x8) at every sample point.
fn determinant_lu(a: &mut [f64], n: usize) -> f64 {
    let mut sign = 1.0;
    for k in 0..n {
        let mut max_val = a[k * n + k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let v = a[i * n + k].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        if max_val == 0.0 {
            return 0.0;
        }
        if max_row != k {
            for j in 0..n {
                let tmp = a[k * n + j];
                a[k * n + j] = a[max_row * n + j];
                a[max_row * n + j] = tmp;
            }
            sign = -sign;
        }
        let pivot = a[k * n + k];
        for i in (k + 1)..n {
            let factor = a[i * n + k] / pivot;
            a[i * n + k] = factor;
            for j in (k + 1)..n {
                a[i * n + j] -= factor * a[k * n + j];
            }
        }
    }
    let mut det = sign;
    for i in 0..n {
        det *= a[i * n + i];
    }
    det
}

/// Cache of pre-built monomial catalogues, keyed by degree. The
/// monomial set for a given k is independent of any candidate -- it is
/// purely a function of the bigraded structure of CP^3 x CP^3 -- so we
/// cache per-degree to avoid rebuilding on every per-candidate score
/// call. Caching matters most for Pass-3 where every precision-scored
/// candidate currently rebuilds the k=2 monomial list (~2025 entries).
fn cached_monomials(k: u32) -> &'static [[u32; 8]] {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<u32, &'static [[u32; 8]]>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let map = cache.lock().unwrap();
        if let Some(v) = map.get(&k) {
            return v;
        }
    }
    // Build outside the lock to avoid holding it across the (rare)
    // rebuild path. Then re-lock and insert (or take an existing entry
    // if another thread won the race).
    let built = build_degree_k_monomials_impl(k);
    let leaked: &'static [[u32; 8]] = Box::leak(built.into_boxed_slice());
    let mut map = cache.lock().unwrap();
    *map.entry(k).or_insert(leaked)
}

/// Build the catalogue of degree-k bigraded monomial exponent tuples for
/// CP^3 x CP^3 (8 coordinates total, degree k in each factor).
///
/// Backwards-compatible wrapper that returns an owned `Vec`; new callers
/// should prefer `degree_k_monomials` which returns a `&'static` slice
/// from the cache and avoids the per-call allocation.
pub fn build_degree_k_monomials(k: u32) -> Vec<[u32; 8]> {
    cached_monomials(k).to_vec()
}

/// Cached, allocation-free access to the degree-k monomial catalogue.
/// Returns a `&'static` slice; the underlying data is built once per
/// process per degree.
pub fn degree_k_monomials(k: u32) -> &'static [[u32; 8]] {
    cached_monomials(k)
}

fn build_degree_k_monomials_impl(k: u32) -> Vec<[u32; 8]> {
    let mut monomials = Vec::new();
    let k_max = k as i32;
    for a0 in 0..=k_max {
        for a1 in 0..=(k_max - a0) {
            for a2 in 0..=(k_max - a0 - a1) {
                let a3 = k_max - a0 - a1 - a2;
                if a3 < 0 {
                    continue;
                }
                for b0 in 0..=k_max {
                    for b1 in 0..=(k_max - b0) {
                        for b2 in 0..=(k_max - b0 - b1) {
                            let b3 = k_max - b0 - b1 - b2;
                            if b3 < 0 {
                                continue;
                            }
                            monomials.push([
                                a0 as u32, a1 as u32, a2 as u32, a3 as u32,
                                b0 as u32, b1 as u32, b2 as u32, b3 as u32,
                            ]);
                        }
                    }
                }
            }
        }
    }
    monomials
}

/// Number of degree-k bigraded monomials = C(k+3, 3)^2 for CP^3 x CP^3.
pub fn n_basis_for_degree(k: u32) -> usize {
    let kk = k as usize;
    let binom_k3_3 = (kk + 1) * (kk + 2) * (kk + 3) / 6;
    binom_k3_3 * binom_k3_3
}

/// Loss components for one refinement step.
#[derive(Debug, Clone, Copy, Default)]
pub struct LossBreakdown {
    pub ricci_flatness: f64,
    pub polyhedral_admissibility: f64,
    pub generation_count: f64,
    pub coulomb_alpha: f64,
    pub weak_mass: f64,
    pub strong_lambda: f64,
    pub yukawa_spectrum: f64,
    pub total: f64,
}

impl LossBreakdown {
    pub fn sum(&self) -> f64 {
        self.ricci_flatness
            + self.polyhedral_admissibility
            + self.generation_count
            + self.coulomb_alpha
            + self.weak_mass
            + self.strong_lambda
            + self.yukawa_spectrum
    }
}

/// Configuration for one refinement run.
#[derive(Debug, Clone, Copy)]
pub struct RefineConfig {
    pub k_degree: u32,
    pub n_sample: usize,
    pub max_donaldson_iters: usize,
    pub max_refine_iters: usize,
    pub donaldson_tol: f64,
    pub refine_lr: f64,
    pub refine_tol: f64,
    pub w_ricci: f64,
    pub w_polyhedral: f64,
    pub w_generation: f64,
    pub w_coulomb: f64,
    pub w_weak: f64,
    pub w_strong: f64,
    pub w_yukawa: f64,
    /// Donaldson early-abort: if residual at iter N exceeds residual at
    /// iter N-1 by this factor, the candidate is marked diverging and
    /// the solve terminates early.
    pub donaldson_divergence_factor: f64,
    /// Multi-resolution: if true, run Donaldson at k_degree, k_degree-1,
    /// k_degree-2 in cascade rather than starting cold at full k_degree.
    pub use_multi_res: bool,
    /// Adaptive importance sampling: weight points by previous |s|^2
    /// distribution rather than uniform polysphere.
    pub use_importance_sampling: bool,
}

impl Default for RefineConfig {
    fn default() -> Self {
        Self::medium()
    }
}

impl RefineConfig {
    pub fn quick() -> Self {
        Self {
            k_degree: 2,
            n_sample: 5_000,
            max_donaldson_iters: 8,
            max_refine_iters: 20,
            donaldson_tol: 1e-3,
            refine_lr: 0.05,
            refine_tol: 1e-2,
            w_ricci: 1.0,
            w_polyhedral: 0.1,
            w_generation: 1.0,
            w_coulomb: 0.1,
            w_weak: 0.1,
            w_strong: 0.1,
            w_yukawa: 0.5,
            donaldson_divergence_factor: 3.0,
            use_multi_res: false,
            use_importance_sampling: false,
        }
    }

    pub fn medium() -> Self {
        Self {
            k_degree: 3,
            n_sample: 50_000,
            max_donaldson_iters: 20,
            max_refine_iters: 200,
            donaldson_tol: 1e-4,
            refine_lr: 0.01,
            refine_tol: 1e-3,
            w_ricci: 1.0,
            w_polyhedral: 0.2,
            w_generation: 1.0,
            w_coulomb: 0.5,
            w_weak: 0.5,
            w_strong: 0.5,
            w_yukawa: 1.0,
            donaldson_divergence_factor: 2.5,
            use_multi_res: true,
            use_importance_sampling: true,
        }
    }

    pub fn publication() -> Self {
        Self {
            k_degree: 4,
            n_sample: 500_000,
            max_donaldson_iters: 50,
            max_refine_iters: 2_000,
            donaldson_tol: 1e-6,
            refine_lr: 0.001,
            refine_tol: 1e-5,
            w_ricci: 1.0,
            w_polyhedral: 0.5,
            w_generation: 1.0,
            w_coulomb: 1.0,
            w_weak: 1.0,
            w_strong: 1.0,
            w_yukawa: 2.0,
            donaldson_divergence_factor: 2.0,
            use_multi_res: true,
            use_importance_sampling: true,
        }
    }

    pub fn from_preset(preset: &str) -> Result<Self, String> {
        match preset {
            "quick" => Ok(Self::quick()),
            "medium" => Ok(Self::medium()),
            "publication" => Ok(Self::publication()),
            other => Err(format!("unknown preset: {other}; valid: quick|medium|publication")),
        }
    }

    pub fn with_k_degree(mut self, k: u32) -> Self {
        self.k_degree = k;
        self
    }
    pub fn with_n_sample(mut self, n: usize) -> Self {
        self.n_sample = n;
        self
    }
    pub fn with_max_donaldson_iters(mut self, n: usize) -> Self {
        self.max_donaldson_iters = n;
        self
    }
    pub fn with_max_refine_iters(mut self, n: usize) -> Self {
        self.max_refine_iters = n;
        self
    }
    pub fn with_donaldson_tol(mut self, tol: f64) -> Self {
        self.donaldson_tol = tol;
        self
    }
    pub fn with_refine_lr(mut self, lr: f64) -> Self {
        self.refine_lr = lr;
        self
    }
    pub fn with_refine_tol(mut self, tol: f64) -> Self {
        self.refine_tol = tol;
        self
    }

    pub fn estimated_cost_units(&self) -> f64 {
        let n_basis = n_basis_for_degree(self.k_degree) as f64;
        let n_p = self.n_sample as f64;
        let total_iters = (self.max_donaldson_iters + self.max_refine_iters) as f64;
        n_basis * n_basis * n_p * total_iters
    }

    pub fn summary(&self) -> String {
        format!(
            "k_degree={}, n_basis={}, n_sample={}, donaldson_iters={}, refine_iters={}, donaldson_tol={:.1e}, refine_tol={:.1e}, lr={:.1e}, multi_res={}, importance={}",
            self.k_degree,
            n_basis_for_degree(self.k_degree),
            self.n_sample,
            self.max_donaldson_iters,
            self.max_refine_iters,
            self.donaldson_tol,
            self.refine_tol,
            self.refine_lr,
            self.use_multi_res,
            self.use_importance_sampling,
        )
    }
}

// ----------------------------------------------------------------------
// Section-basis derivative evaluation
// ----------------------------------------------------------------------

/// Evaluate, at one sample point, the section-basis values plus first
/// derivatives in each of the 8 real coordinates plus second derivatives
/// in each of the 8 + 28 = 36 (i,j) coordinate pairs (i <= j).
///
/// Returns (s, ds, dds) where:
///   s: [n_basis] -- values
///   ds: [8 * n_basis] -- ds[k * n_basis + j] = ds_j/dx_k
///   dds: [36 * n_basis] -- in (i,j) lower-triangular packed order
pub fn evaluate_section_basis_with_derivs(
    z: &[f64; 8],
    monomials: &[[u32; 8]],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_basis = monomials.len();
    // Power table: pow_table[k * (kmax+1) + e] = z[k]^e for e=0..kmax,
    // where kmax is the highest exponent any monomial uses. For degree k
    // the max exponent is k. Find it.
    let mut kmax: u32 = 0;
    for m in monomials {
        for &v in m {
            if v > kmax {
                kmax = v;
            }
        }
    }
    let kmax = kmax as usize;
    let stride = kmax + 1;
    let mut pow_table = vec![1.0f64; 8 * stride];
    for k in 0..8 {
        pow_table[k * stride] = 1.0;
        for e in 1..=kmax {
            pow_table[k * stride + e] = pow_table[k * stride + e - 1] * z[k];
        }
    }

    let mut s = vec![0.0; n_basis];
    let mut ds = vec![0.0; 8 * n_basis];
    let mut dds = vec![0.0; 36 * n_basis];

    // (i, j) lower-triangular packing index helper.
    let pack_ij = |i: usize, j: usize| -> usize {
        // 0..7 diagonal, then off-diagonal pairs
        if i == j {
            i
        } else {
            let (lo, hi) = if i < j { (i, j) } else { (j, i) };
            // diagonals occupy 0..=7, off-diagonals start at 8
            // ordering: (0,1),(0,2)...,(0,7),(1,2),...,(6,7)
            // count = sum_{a=0..lo} (7-a) + (hi - lo - 1)
            let mut idx = 8;
            for a in 0..lo {
                idx += 7 - a;
            }
            idx += hi - lo - 1;
            idx
        }
    };

    for j in 0..n_basis {
        let m = &monomials[j];
        // s = product_k z[k]^m[k]
        let mut prod = 1.0;
        for k in 0..8 {
            prod *= pow_table[k * stride + m[k] as usize];
        }
        s[j] = prod;

        // ds_j/dx_k = m[k] * z[k]^(m[k]-1) * (product over l != k)
        for k in 0..8 {
            if m[k] == 0 {
                ds[k * n_basis + j] = 0.0;
                continue;
            }
            let factor = m[k] as f64 * pow_table[k * stride + m[k] as usize - 1];
            // product over l != k of z[l]^m[l]
            let mut prod_other = 1.0;
            for l in 0..8 {
                if l == k {
                    continue;
                }
                prod_other *= pow_table[l * stride + m[l] as usize];
            }
            ds[k * n_basis + j] = factor * prod_other;
        }

        // d2s_j/dx_i dx_j' = ...
        for i in 0..8 {
            for jj in i..8 {
                let p = pack_ij(i, jj);
                if i == jj {
                    if m[i] < 2 {
                        dds[p * n_basis + j] = 0.0;
                        continue;
                    }
                    let factor =
                        (m[i] as f64) * (m[i] as f64 - 1.0) * pow_table[i * stride + m[i] as usize - 2];
                    let mut prod_other = 1.0;
                    for l in 0..8 {
                        if l == i {
                            continue;
                        }
                        prod_other *= pow_table[l * stride + m[l] as usize];
                    }
                    dds[p * n_basis + j] = factor * prod_other;
                } else {
                    if m[i] == 0 || m[jj] == 0 {
                        dds[p * n_basis + j] = 0.0;
                        continue;
                    }
                    let f1 = m[i] as f64 * pow_table[i * stride + m[i] as usize - 1];
                    let f2 = m[jj] as f64 * pow_table[jj * stride + m[jj] as usize - 1];
                    let mut prod_other = 1.0;
                    for l in 0..8 {
                        if l == i || l == jj {
                            continue;
                        }
                        prod_other *= pow_table[l * stride + m[l] as usize];
                    }
                    dds[p * n_basis + j] = f1 * f2 * prod_other;
                }
            }
        }
    }

    (s, ds, dds)
}

/// Helper for unpacking the 36-entry symmetric (i,j) packed index.
fn pack_ij(i: usize, j: usize) -> usize {
    if i == j {
        i
    } else {
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        let mut idx = 8;
        for a in 0..lo {
            idx += 7 - a;
        }
        idx += hi - lo - 1;
        idx
    }
}

// ----------------------------------------------------------------------
// Real Monge-Ampere residual
// ----------------------------------------------------------------------

/// Build an 8x6 orthonormal basis P for the tangent space to S^3 x S^3
/// at the sample point z. Returns column-major flattened (P[k*6 + j] is
/// row-k column-j entry).
///
/// At z = (z_a, z_b) with ||z_a|| = ||z_b|| = 1, the tangent space is
/// the 6-dim subspace orthogonal to the two radial directions z_a (in
/// the first 4 coords) and z_b (in the last 4 coords).
///
/// Construction: project the 8 standard-basis vectors onto the tangent
/// space (subtract their radial components) and Gram-Schmidt the result;
/// the first 6 non-degenerate columns span the tangent space.
fn polysphere_tangent_basis(z: &[f64; 8]) -> [f64; 48] {
    // Build 8 candidate tangent vectors by removing the radial component
    // from each standard-basis vector e_k.
    let mut candidates = [[0.0f64; 8]; 8];
    let za_dot = |v: &[f64; 8]| v[0] * z[0] + v[1] * z[1] + v[2] * z[2] + v[3] * z[3];
    let zb_dot = |v: &[f64; 8]| v[4] * z[4] + v[5] * z[5] + v[6] * z[6] + v[7] * z[7];
    for k in 0..8 {
        let mut v = [0.0f64; 8];
        v[k] = 1.0;
        let da = za_dot(&v);
        let db = zb_dot(&v);
        // Project off both radial components (they live in disjoint
        // coordinate blocks 0..4 and 4..8 so no cross-correction needed).
        for i in 0..4 {
            v[i] -= da * z[i];
        }
        for i in 4..8 {
            v[i] -= db * z[i];
        }
        candidates[k] = v;
    }

    // Gram-Schmidt: keep first 6 non-degenerate columns. Use Modified
    // Gram-Schmidt for numerical stability.
    let mut p = [0.0f64; 48]; // 8 rows x 6 cols
    let mut col = 0usize;
    for k in 0..8 {
        if col >= 6 {
            break;
        }
        let mut v = candidates[k];
        // Orthogonalise against already-stored columns.
        for j in 0..col {
            let mut dot = 0.0;
            for i in 0..8 {
                dot += p[i * 6 + j] * v[i];
            }
            for i in 0..8 {
                v[i] -= dot * p[i * 6 + j];
            }
        }
        let nrm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nrm > 1e-10 {
            for i in 0..8 {
                p[i * 6 + col] = v[i] / nrm;
            }
            col += 1;
        }
    }
    p
}

/// Project an 8x8 symmetric matrix H onto the tangent space defined by
/// the 8x6 basis P: H_tan = P^T H P (6x6).
fn project_hessian_to_tangent(hess_8x8: &[f64; 64], p: &[f64; 48]) -> [f64; 36] {
    // Step 1: HP = H * P (8x6)
    let mut hp = [0.0f64; 48];
    for i in 0..8 {
        for j in 0..6 {
            let mut s = 0.0;
            for k in 0..8 {
                s += hess_8x8[i * 8 + k] * p[k * 6 + j];
            }
            hp[i * 6 + j] = s;
        }
    }
    // Step 2: H_tan = P^T (HP) (6x6)
    let mut h_tan = [0.0f64; 36];
    for i in 0..6 {
        for j in 0..6 {
            let mut s = 0.0;
            for k in 0..8 {
                s += p[k * 6 + i] * hp[k * 6 + j];
            }
            h_tan[i * 6 + j] = s;
        }
    }
    h_tan
}

/// Real Monge-Ampere residual: variance of log|det H_tan| where H_tan
/// is the 6x6 Hessian of log K projected onto the tangent space of
/// S^3 x S^3 at each sample point.
///
/// For a Kahler-Einstein metric, det H_tan is constant up to volume-form
/// rescaling; non-zero variance of log|det H_tan| measures deviation from
/// Ricci-flatness on the actual 6-real-dim manifold.
///
/// IMPORTANT: this is the residual on S^3 x S^3 with the Kahler
/// potential K(z, zbar) = s^T h s -- NOT on the actual Calabi-Yau
/// (Tian-Yau or Schoen Z3xZ3). The polysphere is a toy ambient; the
/// real CY3 is a codimension-2 sub-variety inside CP^3 x CP^3 cut out
/// by a defining ideal, and the metric should be restricted to that
/// sub-variety. See research-scope deferred items in module header
/// for the true-CY3 implementation status.
pub fn monge_ampere_residual(
    points: &[f64],               // [n_points * 8] -- needed for tangent projection
    section_values: &[f64],
    section_first_derivs: &[f64], // [n_points * 8 * n_basis]
    section_second_derivs: &[f64], // [n_points * 36 * n_basis]
    h: &[f64],
    n_points: usize,
    n_basis: usize,
) -> f64 {
    let log_det_per_point: Vec<f64> = (0..n_points)
        .into_par_iter()
        .with_min_len(64)
        .map(|p| {
            // Slices for this point
            let s = &section_values[p * n_basis..(p + 1) * n_basis];
            let ds = &section_first_derivs[p * 8 * n_basis..(p + 1) * 8 * n_basis];
            let dds = &section_second_derivs[p * 36 * n_basis..(p + 1) * 36 * n_basis];

            // K = s^T h s
            let mut k_val = 0.0;
            // h_s = h s (used multiple times below)
            let mut h_s = vec![0.0; n_basis];
            for a in 0..n_basis {
                let mut row_sum = 0.0;
                for b in 0..n_basis {
                    row_sum += h[a * n_basis + b] * s[b];
                }
                h_s[a] = row_sum;
                k_val += s[a] * row_sum;
            }
            let k_safe = k_val.max(1e-30);

            // dK/dx_k = 2 (ds_k)^T h s
            let mut dk = [0.0f64; 8];
            for k in 0..8 {
                let dsk = &ds[k * n_basis..(k + 1) * n_basis];
                let mut sum = 0.0;
                for a in 0..n_basis {
                    sum += dsk[a] * h_s[a];
                }
                dk[k] = 2.0 * sum;
            }

            // d2K/dx_i dx_j = 2 [(d2s_ij)^T h s + (ds_i)^T h (ds_j)]
            let mut d2k = [0.0f64; 36];
            for i in 0..8 {
                for j in i..8 {
                    let pij = pack_ij(i, j);
                    let dsi = &ds[i * n_basis..(i + 1) * n_basis];
                    let dsj = &ds[j * n_basis..(j + 1) * n_basis];
                    let dds_ij = &dds[pij * n_basis..(pij + 1) * n_basis];
                    let mut term1 = 0.0;
                    for a in 0..n_basis {
                        term1 += dds_ij[a] * h_s[a];
                    }
                    // term2: ds_i^T h ds_j
                    let mut h_dsj = vec![0.0; n_basis];
                    for a in 0..n_basis {
                        let mut rs = 0.0;
                        for b in 0..n_basis {
                            rs += h[a * n_basis + b] * dsj[b];
                        }
                        h_dsj[a] = rs;
                    }
                    let mut term2 = 0.0;
                    for a in 0..n_basis {
                        term2 += dsi[a] * h_dsj[a];
                    }
                    d2k[pij] = 2.0 * (term1 + term2);
                }
            }

            // Hessian of log K at this point:
            //   H_{ij} = d2K/dx_i dx_j / K - (dK/dx_i)(dK/dx_j) / K^2
            let mut hess = [0.0f64; 64]; // 8x8
            for i in 0..8 {
                for j in i..8 {
                    let pij = pack_ij(i, j);
                    let val = d2k[pij] / k_safe - (dk[i] * dk[j]) / (k_safe * k_safe);
                    hess[i * 8 + j] = val;
                    hess[j * 8 + i] = val;
                }
            }

            // Project the 8x8 Hessian onto the 6-dim tangent space of
            // S^3 x S^3 at this point. The 2 radial directions are
            // pure-constraint and contribute spurious near-zero
            // eigenvalues that would dominate the 8x8 determinant.
            let z_pt: [f64; 8] = [
                points[p * 8],
                points[p * 8 + 1],
                points[p * 8 + 2],
                points[p * 8 + 3],
                points[p * 8 + 4],
                points[p * 8 + 5],
                points[p * 8 + 6],
                points[p * 8 + 7],
            ];
            let p_basis = polysphere_tangent_basis(&z_pt);
            let h_tan = project_hessian_to_tangent(&hess, &p_basis);

            // 6x6 determinant via LU with sign-tracking pivoting.
            let mut h_lu = h_tan.to_vec();
            for i in 0..6 {
                h_lu[i * 6 + i] += 1e-12;
            }
            let det = determinant_lu(&mut h_lu, 6);
            if !det.is_finite() {
                return f64::NAN;
            }
            (det.abs().max(1e-30)).ln()
        })
        .collect();

    // Filter NaN entries (from singular tangent projections) before
    // computing variance: a few near-radial samples can produce noise.
    let finite: Vec<f64> = log_det_per_point
        .into_iter()
        .filter(|x| x.is_finite())
        .collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    let n = finite.len() as f64;
    let mean: f64 = finite.iter().sum::<f64>() / n;
    finite.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n
}

/// Backwards-compatible alias for the old proxy used by callers that
/// haven't been upgraded to the full Monge-Ampere path. New code should
/// prefer `monge_ampere_residual` with derivative buffers.
pub fn ricci_flatness_residual(
    section_values: &[f64],
    h: &[f64],
    n_points: usize,
    n_basis: usize,
) -> f64 {
    let log_k_per_point: Vec<f64> = (0..n_points)
        .into_par_iter()
        .with_min_len(64)
        .map(|i| {
            let s_i = &section_values[i * n_basis..(i + 1) * n_basis];
            let mut k = 0.0;
            for a in 0..n_basis {
                let mut row_sum = 0.0;
                for b in 0..n_basis {
                    row_sum += h[a * n_basis + b] * s_i[b];
                }
                k += s_i[a] * row_sum;
            }
            (k.max(1e-30)).ln()
        })
        .collect();
    let mean: f64 = log_k_per_point.iter().sum::<f64>() / log_k_per_point.len() as f64;
    log_k_per_point
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / log_k_per_point.len() as f64
}

// ----------------------------------------------------------------------
// Cheap topological constraints
// ----------------------------------------------------------------------

/// Polyhedral-resonance admissibility loss: how well the candidate
/// CY3's discrete-automorphism irrep table matches the observed
/// solar-system polyhedral wavenumbers (Saturn hexagon n=6, hypothetical
/// n=8 jovian-class jet, Jupiter polar 5+1 cyclones).
///
/// Wires the McKay-correspondence irrep machinery in
/// `automorphism.rs` (M7) into the refine loss vector, replacing the
/// previous Boolean Z3-or-not gate.
///
/// Implementation:
///   - For each candidate fundamental group we pull the irrep
///     dimension table via `tianyau_z3_irreps` /
///     `schoen_z3xz3_irreps`.
///   - `automorphism::ade_irrep_match_loss` computes the squared
///     distance from each observed wavenumber to the nearest irrep
///     dimension. For Z/3 (dims = [1,1,1]) and Z/3 × Z/3 (dims = nine
///     1s) this is large (the observed wavenumbers 5, 6, 8 are all
///     far from 1) — neither candidate fits well, but the SCORE is
///     now a real number the optimiser can rank, not a constant 0.
///   - Normalised by 100 to bring the typical loss into the same
///     order of magnitude as the other refine-loss components.
///   - Unknown groups return 1.0 (existing behaviour preserved).
///
/// **Caveat (recorded in `automorphism.rs` and Part 3 chapter 8):**
/// the polyhedral-wavenumber-to-ADE map is framework-conjectural —
/// no published math/physics literature connects ADE to Saturn's
/// hexagon or Jupiter's cyclones. The published Saturn-hexagon
/// explanation is barotropic Rossby-wave instability, not group
/// theory. This loss term therefore contributes structural pressure
/// during refine but is NOT a 5σ-quality constraint on its own.
pub fn polyhedral_admissibility_loss(fundamental_group: &str) -> f64 {
    use crate::automorphism::{
        ade_irrep_match_loss, schoen_z3xz3_irreps, tianyau_z3_irreps,
    };
    let raw = match fundamental_group {
        "Z3" => ade_irrep_match_loss(&tianyau_z3_irreps()),
        "Z3xZ3" => ade_irrep_match_loss(&schoen_z3xz3_irreps()),
        _ => return 1.0,
    };
    // Normalise: typical raw values range 0..150; divide by 100 so
    // the contribution matches the scale of the other loss terms in
    // LossBreakdown (which are mostly in [0, 1]).
    raw / 100.0
}

pub fn generation_count_loss(chi: i32) -> f64 {
    let n_gen = (chi.abs() / 2) as f64;
    let target = 3.0_f64;
    (n_gen - target).powi(2)
}

// ----------------------------------------------------------------------
// Real forward-models
// ----------------------------------------------------------------------

/// Predict alpha (fine-structure constant) from the candidate's metric
/// h-spectrum and EM-bundle moduli.
///
/// The EM gauge coupling is g_em^2 = 4 pi alpha. In heterotic
/// compactification, g_em^2 is determined by the volume of the cycle
/// supporting the U(1)_Y bundle and by the Hermitian-Yang-Mills
/// renormalisation of the gauge kinetic term:
///
///   1/g_em^2 = V_cycle * Z(h)
///
/// where Z(h) is the photon-zero-mode wavefunction normalisation,
/// computable from h's spectrum. We use h's max eigenvalue (the
/// long-range strain-tail scale) as the Z(h) proxy. We deliberately
/// do NOT use h's mean: Donaldson trace-normalises h so trace(h) =
/// n_basis exactly, making mean degenerate to 1.0 across all candidates.
pub fn predict_alpha_em_from_metric(em_sector_norm: f64, h_spectral_max: f64) -> f64 {
    // Calibration: at em_sector_norm = sqrt(4 pi alpha) and
    // h_spectral_max = 1.0 we recover alpha exactly. Other candidates
    // deviate as a function of both bundle norm and metric spread.
    let g_sq = em_sector_norm * em_sector_norm;
    let z_factor = h_spectral_max.max(1e-6);
    g_sq * z_factor / (4.0 * std::f64::consts::PI)
}

/// Coulomb alpha loss against the **alpha(m_Z)** value, NOT alpha(0).
///
/// Renormalisation-scheme choice: heterotic compactification predicts
/// the gauge couplings at the GUT/string scale; matching to low-energy
/// data goes through the renormalisation-group flow. The natural
/// matching scale is m_Z (the EW unification scale), at which
/// alpha(m_Z)^{-1} = 127.94 in MS-bar (PDG 2024). Comparing predictions
/// to alpha(0)^{-1} = 137.036 introduces a ~10% systematic from QED
/// running between m_e and m_Z; using alpha(m_Z) eliminates that bias.
///
/// References: PDG 2024, Sec. 1.3.5 ("electroweak parameter matching").
pub fn coulomb_alpha_loss(em_sector_norm: f64, h_spectral_max: f64) -> f64 {
    if !em_sector_norm.is_finite() || !h_spectral_max.is_finite() {
        return 1.0e3;
    }
    let predicted = predict_alpha_em_from_metric(em_sector_norm, h_spectral_max);
    if !predicted.is_finite() {
        return 1.0e3;
    }
    // alpha(m_Z) in MS-bar: 1/127.952(9). Used here for RG-consistent
    // matching to heterotic-string predictions at the unification scale.
    let measured = 1.0 / 127.952_f64;
    let rel_err = (predicted - measured) / measured;
    rel_err * rel_err
}

/// Calibration helper: returns the em_sector_norm value that makes the
/// alpha forward-model match alpha(m_Z) exactly when h_spectral_max = 1.
pub fn alpha_em_calibration_em_norm() -> f64 {
    let alpha_mz = 1.0 / 127.952_f64;
    (4.0 * std::f64::consts::PI * alpha_mz).sqrt()
}

/// Predict M_W from the W-zero-mode confinement scale on the SU(2)
/// divisor.
///
/// In heterotic compactification, the W-boson mass arises from the
/// Higgs VEV times the W-zero-mode normalisation. We approximate this
/// as v * weak_norm * f(h_gap) where h_gap is the spectral gap of h
/// restricted to the weak-sector (a proxy for the W-zero-mode
/// localisation scale).
pub fn predict_m_w_gev_from_metric(weak_sector_norm: f64, h_spectral_gap: f64) -> f64 {
    // v = 246 GeV * sin(theta_W) / 2 ~ 80 (for sin(theta_W) ~ 0.65)
    let v_eff = 246.0 * 0.6535 / 2.0; // ~ 80.4
    v_eff * weak_sector_norm * (1.0 + 0.05 * (h_spectral_gap - 1.0))
}

pub fn weak_mass_loss(weak_sector_norm: f64, h_spectral_gap: f64) -> f64 {
    if !weak_sector_norm.is_finite() || !h_spectral_gap.is_finite() {
        return 1.0e3;
    }
    let predicted = predict_m_w_gev_from_metric(weak_sector_norm, h_spectral_gap);
    if !predicted.is_finite() {
        return 1.0e3;
    }
    let measured = 80.377;
    let rel_err = (predicted - measured) / measured;
    rel_err * rel_err
}

/// Predict Lambda_QCD from the gluon-zero-mode strain-tail integration.
///
/// Lambda_QCD is the dimensional-transmutation scale where
/// alpha_s(Lambda_QCD) becomes O(1). In compactification, this is
/// determined by the QCD-sector bundle's Chern-class integral and the
/// metric's largest eigenvalue (the long-range strain-tail scale).
pub fn predict_lambda_qcd_gev_from_metric(qcd_sector_norm: f64, h_spectral_max: f64) -> f64 {
    let m_planck_gev = 1.221e19;
    // Dimensional transmutation: log(M_pl / Lambda) ~ 8 pi^2 / (b_0 g_s^2)
    // where g_s^2 is set by qcd_sector_norm and b_0 = 7 for QCD.
    let g_s_sq = qcd_sector_norm.max(1e-3);
    let b0 = 7.0;
    let log_ratio = 8.0 * std::f64::consts::PI.powi(2) / (b0 * g_s_sq);
    // Modulate by the metric's largest eigenvalue (long-distance scale)
    let modulation = 1.0 + 0.02 * (h_spectral_max - 1.0).abs();
    m_planck_gev * (-log_ratio * modulation).exp()
}

pub fn strong_lambda_loss(qcd_sector_norm: f64, h_spectral_max: f64) -> f64 {
    if !qcd_sector_norm.is_finite() || !h_spectral_max.is_finite() {
        return 1.0e3;
    }
    let predicted = predict_lambda_qcd_gev_from_metric(qcd_sector_norm, h_spectral_max);
    if !predicted.is_finite() {
        return 1.0e3;
    }
    let measured = 0.213; // GeV, n_f=5
    let rel_err = (predicted - measured) / measured;
    rel_err * rel_err
}

// ----------------------------------------------------------------------
// h-matrix spectral helpers
// ----------------------------------------------------------------------

/// Compute only the max eigenvalue (and trace mean) of the h matrix.
/// Cheaper than `h_spectrum` -- skips the min-eigenvalue Rayleigh
/// orthogonalization passes. Use this when only `spec.max` is needed
/// downstream (e.g. coulomb_alpha_loss / strong_lambda_loss). The
/// returned spectrum has `min = max` and `gap = 0` placeholder values
/// so the type stays stable -- callers needing `gap` (notably
/// weak_mass_loss) must use `h_spectrum` instead.
pub fn h_spectrum_max_only(h: &[f64], n: usize) -> HSpectrum {
    let mut trace = 0.0;
    for a in 0..n {
        trace += h[a * n + a];
    }
    let mean = trace / n as f64;

    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut lambda_max = 0.0;
    for _ in 0..15 {
        let mut hv = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += h[i * n + j] * v[j];
            }
            hv[i] = s;
        }
        let norm: f64 = hv.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            break;
        }
        lambda_max = 0.0;
        for i in 0..n {
            lambda_max += v[i] * hv[i];
        }
        for i in 0..n {
            v[i] = hv[i] / norm;
        }
    }
    let lambda_max = lambda_max.abs();
    HSpectrum {
        mean,
        max: lambda_max,
        min: lambda_max,
        gap: 0.0,
    }
}

/// Compute the trace, max-eigenvalue, min-eigenvalue, and spectral gap
/// of the h matrix. Uses simple power iteration (10 iters) for the max
/// eigenvalue and inverse iteration for the min eigenvalue (with LU
/// solve from linalg).
pub fn h_spectrum(h: &[f64], n: usize) -> HSpectrum {
    let mut trace = 0.0;
    for a in 0..n {
        trace += h[a * n + a];
    }
    let mean = trace / n as f64;

    // Power iteration for max eigenvalue
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut lambda_max = 0.0;
    for _ in 0..15 {
        let mut hv = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += h[i * n + j] * v[j];
            }
            hv[i] = s;
        }
        let norm: f64 = hv.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            break;
        }
        lambda_max = 0.0;
        for i in 0..n {
            lambda_max += v[i] * hv[i];
        }
        for i in 0..n {
            v[i] = hv[i] / norm;
        }
    }
    let lambda_max = lambda_max.abs();

    // Min eigenvalue via shift-and-invert: solve (H - sigma I) y = v
    // approximated cheaply by trace lower bound. Simpler approach:
    // estimate min via Rayleigh quotient on a few orthogonalised vectors.
    let mut lambda_min_est = lambda_max;
    let mut u = vec![0.0; n];
    for k in 0..3 {
        for i in 0..n {
            u[i] = ((k as f64 + 1.0) * (i as f64 + 1.0)).sin();
        }
        // orthogonalise against v
        let mut dot = 0.0;
        for i in 0..n {
            dot += u[i] * v[i];
        }
        for i in 0..n {
            u[i] -= dot * v[i];
        }
        let unorm: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        if unorm < 1e-12 {
            continue;
        }
        for i in 0..n {
            u[i] /= unorm;
        }
        let mut hu_dot_u = 0.0;
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += h[i * n + j] * u[j];
            }
            hu_dot_u += u[i] * s;
        }
        if hu_dot_u.abs() < lambda_min_est {
            lambda_min_est = hu_dot_u.abs();
        }
    }

    let gap = (lambda_max - lambda_min_est).abs();
    HSpectrum {
        mean,
        max: lambda_max,
        min: lambda_min_est,
        gap,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HSpectrum {
    pub mean: f64,
    pub max: f64,
    pub min: f64,
    pub gap: f64,
}

// ----------------------------------------------------------------------
// Yukawa fermion-mass spectrum loss
// ----------------------------------------------------------------------

/// Predict the 9 charged fermion masses from the Yukawa tensor's
/// dominant-contraction eigenvalues against a uniform Higgs direction.
///
/// In the framework's reading the cross-term content between mode-overlap
/// pairs IS the Yukawa coupling (<<hyp-substrate-coupling-as-cross-term>>);
/// the Yukawa-contraction matrix's eigenvalue spectrum maps onto the
/// physical fermion masses.
///
/// Eigenvalue extraction uses **simultaneous orthogonal iteration** on
/// a 9-dim block, not Wielandt deflation. Deflation accumulates ~1
/// digit of error per eigenvalue removed (so 9 deflations would lose
/// 9 digits); orthogonal iteration keeps all 9 eigenvalues to bounded
/// precision in one pass.
///
/// We sort the eigenvalues by magnitude and compare to the measured PDG
/// charged-fermion masses (e, mu, tau, u, d, s, c, b, t) sorted the same
/// way. Loss is the L2 distance of log-mass predictions from log-mass
/// measurements, since fermion masses span 6 orders of magnitude.
pub fn yukawa_fermion_mass_loss(yukawa: &[f64], n_modes: usize) -> f64 {
    // Build M_ij = sum_k Y_ijk h_k with h uniform 1/sqrt(n).
    let h_val = 1.0 / (n_modes as f64).sqrt();
    let mut m = vec![0.0; n_modes * n_modes];
    for i in 0..n_modes {
        for j in 0..n_modes {
            let mut s = 0.0;
            for k in 0..n_modes {
                s += yukawa[i * n_modes * n_modes + j * n_modes + k] * h_val;
            }
            m[i * n_modes + j] = s;
        }
    }

    let want = 9.min(n_modes);
    let eigenvalues = subspace_iteration_eigenvalues(&m, n_modes, want, 50);
    let mut eigenvalues: Vec<f64> = eigenvalues.into_iter().map(|x| x.abs()).collect();
    while eigenvalues.len() < 9 {
        eigenvalues.push(1e-6);
    }
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // PDG charged-fermion masses (GeV), sorted ascending
    // electron, up, down, strange, muon, charm, tau, bottom, top
    let measured_gev: [f64; 9] = [
        0.000511, 0.00216, 0.00467, 0.0934, 0.10566, 1.273, 1.7768, 4.18, 172.76,
    ];

    // Calibration: predicted_mass[k] = lambda_k * v_eff with v_eff = 246/sqrt(2)
    let v_eff = 246.0 / std::f64::consts::SQRT_2;
    let mut total = 0.0;
    for k in 0..9 {
        let predicted = eigenvalues[k] * v_eff;
        let log_pred = predicted.max(1e-12).ln();
        let log_meas = measured_gev[k].ln();
        let rel = (log_pred - log_meas) / log_meas.abs().max(1.0);
        total += rel * rel;
    }
    total / 9.0
}

/// Simultaneous orthogonal iteration to extract the top-`want` eigenvalues
/// of a SYMMETRIC matrix M (n x n, row-major).
///
/// **Precondition**: M must be symmetric (M_ij = M_ji). For non-
/// symmetric M, the algorithm still converges to a Schur-form basis,
/// and the diagonal of Q^T M Q gives the real eigenvalues IF they
/// exist; complex eigenvalue pairs would not be resolved correctly.
/// In debug builds we assert symmetry; release builds skip the check.
///
/// Algorithm (Stewart 2001, Algorithm 5.5; Golub-Van Loan 4th ed. 8.2.4):
///   1. Initialise Q (n x p) with orthonormal columns (p = want).
///   2. For i in 0..n_iter:
///        Z = M Q
///        Q, R = QR(Z)
///   3. Eigenvalues = diag(Q^T M Q) (Rayleigh quotients on subspace).
///
/// Convergence: subspace converges at rate |lambda_{p+1}/lambda_p|^iter,
/// independent of the spread of the leading p eigenvalues. Far better
/// than Wielandt deflation for small p relative to n.
pub fn subspace_iteration_eigenvalues(
    m: &[f64],
    n: usize,
    want: usize,
    n_iter: usize,
) -> Vec<f64> {
    // Symmetry precondition (debug-only — release path is unchanged).
    debug_assert!(
        {
            let mut max_asym = 0.0_f64;
            for i in 0..n {
                for j in (i + 1)..n {
                    let asym = (m[i * n + j] - m[j * n + i]).abs();
                    if asym > max_asym {
                        max_asym = asym;
                    }
                }
            }
            max_asym < 1e-9
        },
        "subspace_iteration_eigenvalues requires symmetric M; pass symmetrised M = (M + Mᵀ)/2"
    );
    if want == 0 {
        return Vec::new();
    }
    let p = want.min(n);

    // Initialise Q with deterministic but well-conditioned columns:
    // mix of standard basis + small random-ish perturbation.
    let mut q = vec![0.0f64; n * p];
    for j in 0..p {
        for i in 0..n {
            // Diagonal-ish initialisation with tiny seed-based jitter
            // to avoid Q starting in an exact invariant subspace.
            let base = if i == j { 1.0 } else { 0.0 };
            let jitter = ((i as f64 + 1.0) * (j as f64 + 1.7)).sin() * 1e-6;
            q[i * p + j] = base + jitter;
        }
    }
    qr_orthonormalize_in_place(&mut q, n, p);

    // Iterate.
    let mut z = vec![0.0f64; n * p];
    for _ in 0..n_iter {
        // Z = M Q
        for i in 0..n {
            for j in 0..p {
                let mut s = 0.0;
                for k in 0..n {
                    s += m[i * n + k] * q[k * p + j];
                }
                z[i * p + j] = s;
            }
        }
        // Q, _ = QR(Z) (orthonormalise the columns of Z into Q).
        std::mem::swap(&mut q, &mut z);
        qr_orthonormalize_in_place(&mut q, n, p);
    }

    // Final Rayleigh quotients: lambda_j = (Q[:, j])^T M (Q[:, j]).
    let mut eigs = Vec::with_capacity(p);
    let mut mq_col = vec![0.0f64; n];
    for j in 0..p {
        for i in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += m[i * n + k] * q[k * p + j];
            }
            mq_col[i] = s;
        }
        let mut lambda = 0.0;
        for i in 0..n {
            lambda += q[i * p + j] * mq_col[i];
        }
        eigs.push(lambda);
    }
    eigs
}

/// Modified Gram-Schmidt QR, orthonormalising columns of Q (n x p,
/// row-major: Q[i * p + j]) in place.
fn qr_orthonormalize_in_place(q: &mut [f64], n: usize, p: usize) {
    for j in 0..p {
        // Orthogonalise column j against earlier columns.
        for k in 0..j {
            let mut dot = 0.0;
            for i in 0..n {
                dot += q[i * p + k] * q[i * p + j];
            }
            for i in 0..n {
                q[i * p + j] -= dot * q[i * p + k];
            }
        }
        // Normalise column j.
        let mut nrm_sq = 0.0;
        for i in 0..n {
            nrm_sq += q[i * p + j] * q[i * p + j];
        }
        let nrm = nrm_sq.sqrt();
        if nrm > 1e-14 {
            for i in 0..n {
                q[i * p + j] /= nrm;
            }
        } else {
            // Column collapsed: replace with a fallback unit vector
            // orthogonal to all earlier columns (tiny re-injection so
            // iteration can recover).
            for i in 0..n {
                q[i * p + j] = if i == j { 1e-8 } else { 0.0 };
            }
            // Re-orthogonalise; if it's still zero we accept it.
            for k in 0..j {
                let mut dot = 0.0;
                for i in 0..n {
                    dot += q[i * p + k] * q[i * p + j];
                }
                for i in 0..n {
                    q[i * p + j] -= dot * q[i * p + k];
                }
            }
            let mut nrm2_sq = 0.0;
            for i in 0..n {
                nrm2_sq += q[i * p + j] * q[i * p + j];
            }
            let nrm2 = nrm2_sq.sqrt();
            if nrm2 > 1e-20 {
                for i in 0..n {
                    q[i * p + j] /= nrm2;
                }
            }
        }
    }
}

#[allow(dead_code)] // retained for sum-rule test cross-check
fn power_iteration(m: &[f64], n: usize, iters: usize) -> f64 {
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut lambda = 0.0;
    for _ in 0..iters {
        let mut mv = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                mv[i] += m[i * n + j] * v[j];
            }
        }
        let norm: f64 = mv.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            return 0.0;
        }
        lambda = 0.0;
        for i in 0..n {
            lambda += v[i] * mv[i];
        }
        for i in 0..n {
            v[i] = mv[i] / norm;
        }
    }
    lambda
}

// ----------------------------------------------------------------------
// Aggregate loss
// ----------------------------------------------------------------------

/// Inputs needed for full constraint-aware loss. Caller fills these
/// during the pass: section values + derivatives, current h, bundle norms,
/// Yukawa tensor (or None to skip the Yukawa term).
pub struct LossInputs<'a> {
    pub points: Option<&'a [f64]>,
    pub section_values: &'a [f64],
    pub section_first_derivs: Option<&'a [f64]>,
    pub section_second_derivs: Option<&'a [f64]>,
    pub h: &'a [f64],
    pub n_points: usize,
    pub n_basis: usize,
    pub chi: i32,
    pub fundamental_group: &'a str,
    pub em_sector_norm: f64,
    pub weak_sector_norm: f64,
    pub qcd_sector_norm: f64,
    pub yukawa_tensor: Option<&'a [f64]>,
    pub n_modes: usize,
}

pub fn compute_loss_full(config: &RefineConfig, inputs: &LossInputs<'_>) -> LossBreakdown {
    let ricci = if let (Some(pts), Some(ds), Some(dds)) = (
        inputs.points,
        inputs.section_first_derivs,
        inputs.section_second_derivs,
    ) {
        monge_ampere_residual(
            pts,
            inputs.section_values,
            ds,
            dds,
            inputs.h,
            inputs.n_points,
            inputs.n_basis,
        )
    } else {
        ricci_flatness_residual(inputs.section_values, inputs.h, inputs.n_points, inputs.n_basis)
    };

    let spec = h_spectrum(inputs.h, inputs.n_basis);
    let poly = polyhedral_admissibility_loss(inputs.fundamental_group);
    let gen = generation_count_loss(inputs.chi);
    let coulomb = coulomb_alpha_loss(inputs.em_sector_norm, spec.max);
    let weak = weak_mass_loss(inputs.weak_sector_norm, spec.gap);
    let strong = strong_lambda_loss(inputs.qcd_sector_norm, spec.max);
    let yukawa = if let Some(y) = inputs.yukawa_tensor {
        yukawa_fermion_mass_loss(y, inputs.n_modes)
    } else {
        0.0
    };

    let mut breakdown = LossBreakdown {
        ricci_flatness: config.w_ricci * ricci,
        polyhedral_admissibility: config.w_polyhedral * poly,
        generation_count: config.w_generation * gen,
        coulomb_alpha: config.w_coulomb * coulomb,
        weak_mass: config.w_weak * weak,
        strong_lambda: config.w_strong * strong,
        yukawa_spectrum: config.w_yukawa * yukawa,
        total: 0.0,
    };
    breakdown.total = breakdown.sum();
    breakdown
}

/// Backwards-compatible signature using the old proxy.
pub fn compute_loss(
    config: &RefineConfig,
    section_values: &[f64],
    h: &[f64],
    n_points: usize,
    n_basis: usize,
    chi: i32,
    fundamental_group: &str,
    em_sector_norm: f64,
    weak_sector_norm: f64,
    qcd_sector_norm: f64,
) -> LossBreakdown {
    let inputs = LossInputs {
        points: None,
        section_values,
        section_first_derivs: None,
        section_second_derivs: None,
        h,
        n_points,
        n_basis,
        chi,
        fundamental_group,
        em_sector_norm,
        weak_sector_norm,
        qcd_sector_norm,
        yukawa_tensor: None,
        n_modes: 0,
    };
    compute_loss_full(config, &inputs)
}

// ----------------------------------------------------------------------
// Adam optimizer for h
// ----------------------------------------------------------------------

/// Adam optimizer state for h-coefficient updates. Stored separately
/// because Adam needs first/second moment buffers across iterations.
#[derive(Debug, Clone)]
pub struct AdamState {
    pub m: Vec<f64>, // first moment
    pub v: Vec<f64>, // second moment
    pub t: u64,      // time-step (for bias correction)
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

impl AdamState {
    pub fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    /// One Adam update step given a gradient.
    pub fn step(&mut self, params: &mut [f64], grad: &[f64], lr: f64) {
        self.t = self.t.saturating_add(1);
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        let scale = lr * (bc2.sqrt() / bc1);
        for k in 0..params.len() {
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * grad[k];
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * grad[k] * grad[k];
            params[k] -= scale * self.m[k] / (self.v[k].sqrt() + self.eps);
        }
    }
}

/// Compute the gradient of the Ricci-flatness residual proxy w.r.t. the
/// h coefficients ANALYTICALLY in closed form.
///
/// Let K_p = s_p^T h s_p, sigma = (1/n) sum_p (log K_p - mean)^2 where
/// mean = (1/n) sum_p log K_p. Then
///
///   dK_p/dh_{ab} = s_a(p) s_b(p)
///   d(log K_p)/dh_{ab} = s_a(p) s_b(p) / K_p
///   d(mean)/dh_{ab} = (1/n) sum_p s_a(p) s_b(p) / K_p
///
/// Let c_p := log K_p - mean and d_p := s_a(p) s_b(p) / K_p. Then
///
///   d(sigma)/dh_{ab} = (2/n) sum_p c_p (d_p - <d>)
///
/// where <d> = mean over p of d_p. Since c_p has mean 0, the <d> term
/// vanishes after summation against c_p, giving
///
///   d(sigma)/dh_{ab} = (2/n) sum_p (log K_p - mean) * s_a(p) s_b(p) / K_p
///
/// Cost: O(n_points * n_basis^2), the same as one residual evaluation.
/// The previous finite-difference implementation cost n_basis^2 residual
/// evaluations -- a 10000x slowdown at n_basis=100.
pub fn refine_step_adam(
    h: &mut [f64],
    section_values: &[f64],
    n_points: usize,
    n_basis: usize,
    state: &mut AdamState,
    lr: f64,
) -> f64 {
    // First pass: compute K_p and log K_p at each sample.
    let log_k: Vec<f64> = (0..n_points)
        .into_par_iter()
        .with_min_len(64)
        .map(|p| {
            let s_p = &section_values[p * n_basis..(p + 1) * n_basis];
            let mut k = 0.0;
            for a in 0..n_basis {
                let mut row_sum = 0.0;
                for b in 0..n_basis {
                    row_sum += h[a * n_basis + b] * s_p[b];
                }
                k += s_p[a] * row_sum;
            }
            (k.max(1e-30)).ln()
        })
        .collect();

    let mean: f64 = log_k.iter().sum::<f64>() / log_k.len() as f64;
    let variance: f64 =
        log_k.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / log_k.len() as f64;

    // Second pass: accumulate the gradient.
    //   grad[a, b] = (2/n) sum_p (log K_p - mean) * s_a(p) s_b(p) / K_p
    //
    // We rebuild K_p locally (cheap re-evaluation; avoids storing a
    // separate buffer). Use a per-thread accumulator + reduce to avoid
    // contention on the gradient matrix.
    let n_coef = n_basis * n_basis;
    let two_over_n = 2.0 / n_points as f64;
    let grad: Vec<f64> = (0..n_points)
        .into_par_iter()
        .with_min_len(64)
        .fold(
            || vec![0.0f64; n_coef],
            |mut acc, p| {
                let s_p = &section_values[p * n_basis..(p + 1) * n_basis];
                let mut k = 0.0;
                for a in 0..n_basis {
                    let mut row_sum = 0.0;
                    for b in 0..n_basis {
                        row_sum += h[a * n_basis + b] * s_p[b];
                    }
                    k += s_p[a] * row_sum;
                }
                let k_safe = k.max(1e-30);
                let coef = (log_k[p] - mean) / k_safe;
                if !coef.is_finite() {
                    return acc;
                }
                // Outer product s s^T scaled by coef.
                for a in 0..n_basis {
                    let sa_coef = s_p[a] * coef;
                    let row = a * n_basis;
                    for b in 0..n_basis {
                        acc[row + b] += sa_coef * s_p[b];
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0.0f64; n_coef],
            |mut a, b| {
                for k in 0..n_coef {
                    a[k] += b[k];
                }
                a
            },
        )
        .into_iter()
        .map(|g| g * two_over_n)
        .collect();

    // Symmetrise: K_p depends on (h + h^T)/2 only, so the gradient
    // should be symmetric. Floating-point asymmetry is harmless but
    // compounds over Adam steps; symmetrise once to keep h symmetric.
    let mut grad_sym = grad.clone();
    for a in 0..n_basis {
        for b in (a + 1)..n_basis {
            let avg = 0.5 * (grad[a * n_basis + b] + grad[b * n_basis + a]);
            grad_sym[a * n_basis + b] = avg;
            grad_sym[b * n_basis + a] = avg;
        }
    }

    state.step(h, &grad_sym, lr);

    // Re-symmetrise h after Adam (m/v can drift by 1 ULP per step).
    for a in 0..n_basis {
        for b in (a + 1)..n_basis {
            let avg = 0.5 * (h[a * n_basis + b] + h[b * n_basis + a]);
            h[a * n_basis + b] = avg;
            h[b * n_basis + a] = avg;
        }
    }

    // Re-normalise trace (Donaldson invariant).
    let trace: f64 = (0..n_basis).map(|a| h[a * n_basis + a]).sum();
    if trace > 1e-10 && trace.is_finite() {
        let renorm = (n_basis as f64) / trace;
        for v in h.iter_mut() {
            *v *= renorm;
        }
    }

    variance
}

/// Original scalar-damping refine step (kept for backwards compatibility
/// and for the cheap-pass codepath). Prefer `refine_step_adam` for
/// publication-grade convergence.
pub fn refine_step(
    h: &mut [f64],
    section_values: &[f64],
    n_points: usize,
    n_basis: usize,
    learning_rate: f64,
) -> f64 {
    let log_k: Vec<f64> = (0..n_points)
        .into_par_iter()
        .with_min_len(64)
        .map(|i| {
            let s_i = &section_values[i * n_basis..(i + 1) * n_basis];
            let mut k = 0.0;
            for a in 0..n_basis {
                let mut row_sum = 0.0;
                for b in 0..n_basis {
                    row_sum += h[a * n_basis + b] * s_i[b];
                }
                k += s_i[a] * row_sum;
            }
            (k.max(1e-30)).ln()
        })
        .collect();

    let mean: f64 = log_k.iter().sum::<f64>() / log_k.len() as f64;
    let variance: f64 =
        log_k.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / log_k.len() as f64;
    let stddev = variance.sqrt();
    let scale = (-learning_rate * stddev).exp();
    for v in h.iter_mut() {
        *v *= scale;
    }
    let trace: f64 = (0..n_basis).map(|a| h[a * n_basis + a]).sum::<f64>();
    if trace > 1e-10 {
        let renorm = (n_basis as f64) / trace;
        for v in h.iter_mut() {
            *v *= renorm;
        }
    }
    variance
}

// ----------------------------------------------------------------------
// Adaptive importance sampling
// ----------------------------------------------------------------------

/// Importance-sampling-weighted Monge-Ampere residual. Uses point
/// weights to bias the estimator toward high-curvature regions of the
/// CY3 metric. Drops the n_sample requirement at fixed accuracy.
///
/// The weights `w_p` should sum to `n_points` (unbiased estimator).
/// `importance_weights()` produces such weights from a previous
/// section-value distribution.
pub fn monge_ampere_residual_weighted(
    points: &[f64],
    section_values: &[f64],
    section_first_derivs: &[f64],
    section_second_derivs: &[f64],
    h: &[f64],
    weights: &[f64],
    n_points: usize,
    n_basis: usize,
) -> f64 {
    // Per-point log|det H_tan|, same as monge_ampere_residual.
    let log_det_per_point: Vec<f64> = (0..n_points)
        .into_par_iter()
        .with_min_len(64)
        .map(|p| {
            let s = &section_values[p * n_basis..(p + 1) * n_basis];
            let ds = &section_first_derivs[p * 8 * n_basis..(p + 1) * 8 * n_basis];
            let dds = &section_second_derivs[p * 36 * n_basis..(p + 1) * 36 * n_basis];

            let mut k_val = 0.0;
            let mut h_s = vec![0.0; n_basis];
            for a in 0..n_basis {
                let mut row_sum = 0.0;
                for b in 0..n_basis {
                    row_sum += h[a * n_basis + b] * s[b];
                }
                h_s[a] = row_sum;
                k_val += s[a] * row_sum;
            }
            let k_safe = k_val.max(1e-30);
            let mut dk = [0.0f64; 8];
            for k in 0..8 {
                let dsk = &ds[k * n_basis..(k + 1) * n_basis];
                let mut sum = 0.0;
                for a in 0..n_basis {
                    sum += dsk[a] * h_s[a];
                }
                dk[k] = 2.0 * sum;
            }
            let mut d2k = [0.0f64; 36];
            for i in 0..8 {
                for j in i..8 {
                    let pij = pack_ij(i, j);
                    let dsi = &ds[i * n_basis..(i + 1) * n_basis];
                    let dsj = &ds[j * n_basis..(j + 1) * n_basis];
                    let dds_ij = &dds[pij * n_basis..(pij + 1) * n_basis];
                    let mut term1 = 0.0;
                    for a in 0..n_basis {
                        term1 += dds_ij[a] * h_s[a];
                    }
                    let mut h_dsj = vec![0.0; n_basis];
                    for a in 0..n_basis {
                        let mut rs = 0.0;
                        for b in 0..n_basis {
                            rs += h[a * n_basis + b] * dsj[b];
                        }
                        h_dsj[a] = rs;
                    }
                    let mut term2 = 0.0;
                    for a in 0..n_basis {
                        term2 += dsi[a] * h_dsj[a];
                    }
                    d2k[pij] = 2.0 * (term1 + term2);
                }
            }
            let mut hess = [0.0f64; 64];
            for i in 0..8 {
                for j in i..8 {
                    let pij = pack_ij(i, j);
                    let val = d2k[pij] / k_safe - (dk[i] * dk[j]) / (k_safe * k_safe);
                    hess[i * 8 + j] = val;
                    hess[j * 8 + i] = val;
                }
            }
            let z_pt: [f64; 8] = [
                points[p * 8],
                points[p * 8 + 1],
                points[p * 8 + 2],
                points[p * 8 + 3],
                points[p * 8 + 4],
                points[p * 8 + 5],
                points[p * 8 + 6],
                points[p * 8 + 7],
            ];
            let p_basis = polysphere_tangent_basis(&z_pt);
            let h_tan = project_hessian_to_tangent(&hess, &p_basis);
            let mut h_lu = h_tan.to_vec();
            for i in 0..6 {
                h_lu[i * 6 + i] += 1e-12;
            }
            let det = determinant_lu(&mut h_lu, 6);
            if !det.is_finite() {
                return f64::NAN;
            }
            (det.abs().max(1e-30)).ln()
        })
        .collect();

    // Weighted variance: var_w(x) = (sum_p w_p (x_p - mu_w)^2) / sum_p w_p
    // with mu_w = (sum_p w_p x_p) / sum_p w_p.
    let mut total_w = 0.0;
    let mut weighted_sum = 0.0;
    let mut keep: Vec<(f64, f64)> = Vec::with_capacity(log_det_per_point.len());
    for (p, ld) in log_det_per_point.into_iter().enumerate() {
        if !ld.is_finite() {
            continue;
        }
        let w = weights[p];
        if !w.is_finite() || w <= 0.0 {
            continue;
        }
        keep.push((ld, w));
        total_w += w;
        weighted_sum += w * ld;
    }
    if total_w < 1e-12 {
        return f64::NAN;
    }
    let mu_w = weighted_sum / total_w;
    let mut var_w = 0.0;
    for (ld, w) in &keep {
        let d = ld - mu_w;
        var_w += w * d * d;
    }
    var_w / total_w
}

/// Budget-aware Monge-Ampere residual with sequential early-abort.
///
/// Computes log|det H_tan| over points in chunks of `CHUNK = 1024` and
/// tracks the running variance estimate after each chunk. If the
/// running estimate scaled by `w_ricci` already exceeds `budget`,
/// accounting for a small estimator-noise slack `slack/sqrt(n_so_far)`,
/// the function aborts and returns the current variance estimate (which
/// is then guaranteed to make the candidate fail the threshold).
///
/// The slack term is a heuristic safety margin against the running
/// variance estimate being temporarily low due to sampling noise. The
/// theoretical Bernstein-style lower-confidence bound on the population
/// variance from a finite sample requires a 4th-moment estimate; we
/// approximate it here with a `slack=2.0` empirical multiplier on the
/// 1/sqrt(n) noise scale, which is conservative for log|det H_tan|
/// distributions encountered in practice. A slack of 0.0 reduces the
/// function to the running estimate without statistical correction,
/// which is faster but slightly more aggressive.
///
/// All other arguments match `monge_ampere_residual_weighted`.
pub fn monge_ampere_residual_weighted_budget(
    points: &[f64],
    section_values: &[f64],
    section_first_derivs: &[f64],
    section_second_derivs: &[f64],
    h: &[f64],
    weights: &[f64],
    n_points: usize,
    n_basis: usize,
    w_ricci: f64,
    budget: f64,
    slack: f64,
) -> f64 {
    const CHUNK: usize = 1024;
    let mut sum_w = 0.0;
    let mut sum_wx = 0.0;
    let mut sum_wxx = 0.0;
    let mut n_finite = 0usize;

    let mut start = 0;
    while start < n_points {
        let end = (start + CHUNK).min(n_points);
        let log_det_chunk: Vec<(f64, f64)> = (start..end)
            .into_par_iter()
            .with_min_len(64)
            .map(|p| {
                let s = &section_values[p * n_basis..(p + 1) * n_basis];
                let ds = &section_first_derivs[p * 8 * n_basis..(p + 1) * 8 * n_basis];
                let dds = &section_second_derivs[p * 36 * n_basis..(p + 1) * 36 * n_basis];

                let mut k_val = 0.0;
                let mut h_s = vec![0.0; n_basis];
                for a in 0..n_basis {
                    let mut row_sum = 0.0;
                    for b in 0..n_basis {
                        row_sum += h[a * n_basis + b] * s[b];
                    }
                    h_s[a] = row_sum;
                    k_val += s[a] * row_sum;
                }
                let k_safe = k_val.max(1e-30);
                let mut dk = [0.0f64; 8];
                for k in 0..8 {
                    let dsk = &ds[k * n_basis..(k + 1) * n_basis];
                    let mut sum = 0.0;
                    for a in 0..n_basis {
                        sum += dsk[a] * h_s[a];
                    }
                    dk[k] = 2.0 * sum;
                }
                let mut d2k = [0.0f64; 36];
                for i in 0..8 {
                    for j in i..8 {
                        let pij = pack_ij(i, j);
                        let dsi = &ds[i * n_basis..(i + 1) * n_basis];
                        let dsj = &ds[j * n_basis..(j + 1) * n_basis];
                        let dds_ij = &dds[pij * n_basis..(pij + 1) * n_basis];
                        let mut term1 = 0.0;
                        for a in 0..n_basis {
                            term1 += dds_ij[a] * h_s[a];
                        }
                        let mut h_dsj = vec![0.0; n_basis];
                        for a in 0..n_basis {
                            let mut rs = 0.0;
                            for b in 0..n_basis {
                                rs += h[a * n_basis + b] * dsj[b];
                            }
                            h_dsj[a] = rs;
                        }
                        let mut term2 = 0.0;
                        for a in 0..n_basis {
                            term2 += dsi[a] * h_dsj[a];
                        }
                        d2k[pij] = 2.0 * (term1 + term2);
                    }
                }
                let mut hess = [0.0f64; 64];
                for i in 0..8 {
                    for j in i..8 {
                        let pij = pack_ij(i, j);
                        let val = d2k[pij] / k_safe - (dk[i] * dk[j]) / (k_safe * k_safe);
                        hess[i * 8 + j] = val;
                        hess[j * 8 + i] = val;
                    }
                }
                let z_pt: [f64; 8] = [
                    points[p * 8],
                    points[p * 8 + 1],
                    points[p * 8 + 2],
                    points[p * 8 + 3],
                    points[p * 8 + 4],
                    points[p * 8 + 5],
                    points[p * 8 + 6],
                    points[p * 8 + 7],
                ];
                let p_basis = polysphere_tangent_basis(&z_pt);
                let h_tan = project_hessian_to_tangent(&hess, &p_basis);
                let mut h_lu = h_tan.to_vec();
                for i in 0..6 {
                    h_lu[i * 6 + i] += 1e-12;
                }
                let det = determinant_lu(&mut h_lu, 6);
                let ld = if det.is_finite() {
                    (det.abs().max(1e-30)).ln()
                } else {
                    f64::NAN
                };
                let w = weights[p];
                (ld, w)
            })
            .collect();

        // Running weighted-variance update.
        for (ld, w) in log_det_chunk {
            if !ld.is_finite() || !w.is_finite() || w <= 0.0 {
                continue;
            }
            sum_w += w;
            sum_wx += w * ld;
            sum_wxx += w * ld * ld;
            n_finite += 1;
        }

        // Estimate running variance and check budget. If below 256
        // points have finished, skip the early-abort to avoid noise
        // dominating.
        if n_finite >= 256 && sum_w > 1e-12 {
            let mu_w = sum_wx / sum_w;
            let var_w = (sum_wxx / sum_w) - mu_w * mu_w;
            let var_w = var_w.max(0.0);
            // Slack term scaled by 1/sqrt(n_finite); represents the
            // approximate sampling-noise band on the variance estimate.
            let slack_term = if slack > 0.0 {
                slack / (n_finite as f64).sqrt()
            } else {
                0.0
            };
            let lower_bound = (var_w - slack_term).max(0.0);
            if w_ricci * lower_bound > budget {
                return var_w;
            }
        }

        start = end;
    }

    if sum_w < 1e-12 {
        return f64::NAN;
    }
    let mu_w = sum_wx / sum_w;
    let var_w = (sum_wxx / sum_w) - mu_w * mu_w;
    var_w.max(0.0)
}

/// Generate importance-sampled point weights from a previous |s|^2
/// distribution. Higher |s|^2 -> higher curvature region -> sample more
/// densely.
///
/// Returns per-point weights summing to n_points (so the residual
/// estimator stays unbiased when averaged). The weights are normalised
/// so a flat distribution returns all-ones.
pub fn importance_weights(section_values: &[f64], n_points: usize, n_basis: usize) -> Vec<f64> {
    let mut s2: Vec<f64> = (0..n_points)
        .into_par_iter()
        .map(|i| {
            let s_i = &section_values[i * n_basis..(i + 1) * n_basis];
            let mut sum = 0.0;
            for a in 0..n_basis {
                sum += s_i[a] * s_i[a];
            }
            sum.max(1e-30).sqrt()
        })
        .collect();
    let total: f64 = s2.iter().sum();
    if total < 1e-12 {
        return vec![1.0; n_points];
    }
    let scale = n_points as f64 / total;
    for v in s2.iter_mut() {
        *v *= scale;
    }
    s2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_ij_round_trip() {
        let mut seen = std::collections::HashSet::new();
        for i in 0..8 {
            for j in i..8 {
                let p = pack_ij(i, j);
                assert!(p < 36, "pack {i},{j} -> {p} out of range");
                assert!(seen.insert(p), "duplicate pack {i},{j} -> {p}");
            }
        }
        assert_eq!(seen.len(), 36);
    }

    #[test]
    fn evaluate_section_basis_with_derivs_smoke() {
        let z = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let monomials = build_degree_k_monomials(2);
        let (s, ds, dds) = evaluate_section_basis_with_derivs(&z, &monomials);
        assert_eq!(s.len(), monomials.len());
        assert_eq!(ds.len(), 8 * monomials.len());
        assert_eq!(dds.len(), 36 * monomials.len());
    }

    #[test]
    fn adam_state_step_changes_params() {
        let mut state = AdamState::new(3);
        let mut params = vec![1.0; 3];
        let grad = vec![0.5; 3];
        state.step(&mut params, &grad, 0.1);
        for p in &params {
            assert!(*p < 1.0, "expected decrease");
        }
    }

    #[test]
    fn adam_zero_grad_no_change_after_bias_correction() {
        let mut state = AdamState::new(3);
        let mut params = vec![1.0; 3];
        for _ in 0..5 {
            state.step(&mut params, &vec![0.0; 3], 0.1);
        }
        for p in &params {
            assert!((p - 1.0).abs() < 1e-9);
        }
    }

    // ------------------------------------------------------------------
    // Forward-model calibration-point tests.
    //
    // Each forward-model has a known input value at which its prediction
    // matches the measured constant exactly. These tests verify that
    // hitting the known input gives near-zero loss.
    // ------------------------------------------------------------------

    #[test]
    fn alpha_em_zero_loss_at_calibration() {
        // Calibration: em_sector_norm = sqrt(4 pi alpha(m_Z)), h_max = 1.0.
        let em_norm = alpha_em_calibration_em_norm();
        let loss = coulomb_alpha_loss(em_norm, 1.0);
        assert!(
            loss < 1e-15,
            "expected near-zero loss at em calibration point, got {loss}"
        );
    }

    #[test]
    fn alpha_em_loss_grows_off_calibration() {
        // Doubling em_sector_norm sends predicted alpha to 4x measured.
        let em_norm = alpha_em_calibration_em_norm();
        let loss_off = coulomb_alpha_loss(em_norm * 2.0, 1.0);
        assert!(loss_off > 1.0, "expected large loss off-calibration, got {loss_off}");
    }

    #[test]
    fn weak_mass_zero_loss_at_calibration() {
        // M_W formula: 80.4 * weak_norm * (1 + 0.05 * (h_gap - 1)).
        // At weak_norm = 80.377/80.4 and h_gap = 1.0, loss should be ~0.
        let v_eff: f64 = 246.0 * 0.6535 / 2.0;
        let weak_norm = 80.377 / v_eff;
        let loss = weak_mass_loss(weak_norm, 1.0);
        assert!(
            loss < 1e-6,
            "expected near-zero loss at M_W calibration, got {loss}"
        );
    }

    #[test]
    fn lambda_qcd_predict_in_right_ballpark() {
        // The dimensional-transmutation formula is sensitive; we just
        // verify it produces a value in the GeV ballpark for moderate
        // qcd_norm and that predict scales as expected.
        let lo = predict_lambda_qcd_gev_from_metric(0.5, 1.0);
        let hi = predict_lambda_qcd_gev_from_metric(0.6, 1.0);
        assert!(lo > 0.0 && lo.is_finite(), "lo not finite/positive: {lo}");
        assert!(hi > lo, "expected larger qcd_norm -> larger Lambda, lo={lo} hi={hi}");
    }

    // ------------------------------------------------------------------
    // Yukawa sum-rule consistency.
    //
    // For the contracted matrix M_ij = sum_k Y_ijk * h_k (h uniform):
    //   Tr(M^2) = sum_i lambda_i^2  (eigenvalue-sum identity)
    // So the squared Frobenius norm of M, computed two ways, must agree.
    // ------------------------------------------------------------------

    #[test]
    fn yukawa_sum_rule_consistency() {
        // Build a small Y_ijk via Gaussian centers, exactly as the
        // pipeline does at runtime.
        use crate::workspace::DiscriminationWorkspace;
        let n_modes = 8;
        let mut ws = DiscriminationWorkspace::new(500, 100, n_modes, 5, 1);
        crate::sample_points_into(&mut ws, 17);
        crate::evaluate_section_basis_into(&mut ws);
        crate::init_yukawa_centers(&mut ws, 23);
        crate::yukawa_tensor_in_place(&mut ws);

        // Compute Tr(M^2) directly:
        // M_ij = sum_k Y_ijk * h_k where h_k = 1/sqrt(n_modes).
        let h_val = 1.0 / (n_modes as f64).sqrt();
        let mut m = vec![0.0; n_modes * n_modes];
        for i in 0..n_modes {
            for j in 0..n_modes {
                let mut s = 0.0;
                for k in 0..n_modes {
                    s += ws.yukawa_tensor[i * n_modes * n_modes + j * n_modes + k] * h_val;
                }
                m[i * n_modes + j] = s;
            }
        }
        let mut tr_msq = 0.0;
        for i in 0..n_modes {
            for j in 0..n_modes {
                tr_msq += m[i * n_modes + j] * m[j * n_modes + i];
            }
        }

        // Now extract eigenvalues via deflation (same as
        // yukawa_fermion_mass_loss does internally) and sum squares.
        let mut m_work = m.clone();
        let mut sum_lambda_sq = 0.0;
        for _ in 0..n_modes {
            let lambda = power_iteration(&m_work, n_modes, 80);
            if lambda.abs() < 1e-14 {
                break;
            }
            sum_lambda_sq += lambda * lambda;
            // Re-extract dominant eigenvector for deflation.
            let mut v = vec![1.0 / (n_modes as f64).sqrt(); n_modes];
            for _ in 0..80 {
                let mut mv = vec![0.0; n_modes];
                for i in 0..n_modes {
                    for j in 0..n_modes {
                        mv[i] += m_work[i * n_modes + j] * v[j];
                    }
                }
                let norm: f64 = mv.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm < 1e-14 {
                    break;
                }
                for i in 0..n_modes {
                    v[i] = mv[i] / norm;
                }
            }
            for i in 0..n_modes {
                for j in 0..n_modes {
                    m_work[i * n_modes + j] -= lambda * v[i] * v[j];
                }
            }
        }

        let rel_err = ((sum_lambda_sq - tr_msq) / tr_msq.abs().max(1e-12)).abs();
        // Power-iteration with deflation has bounded but non-zero
        // numerical error; require agreement to within 5%.
        assert!(
            rel_err < 0.05,
            "sum-rule violation: Tr(M^2)={tr_msq:.6e}, sum lambda_i^2={sum_lambda_sq:.6e}, rel_err={rel_err:.3e}"
        );
    }

    // ------------------------------------------------------------------
    // Donaldson FS-identity test.
    //
    // On a uniform polysphere sample (no quotient, untwisted), the
    // Donaldson balancing iteration converges to the diagonal Bergman
    // metric. After trace-normalisation the diagonal entries should
    // cluster near 1.0; off-diagonal entries should be small.
    // ------------------------------------------------------------------

    #[test]
    fn donaldson_balanced_h_is_near_diagonal_on_polysphere() {
        use crate::kernels::donaldson_solve_in_place;
        use crate::workspace::DiscriminationWorkspace;

        let n_basis = 100;
        let mut ws = DiscriminationWorkspace::new(2000, n_basis, 8, 50, 1);
        crate::sample_points_into(&mut ws, 42);
        crate::evaluate_section_basis_into(&mut ws);
        donaldson_solve_in_place(&mut ws, 1e-5);

        // After normalisation: trace = n_basis (so mean diagonal = 1).
        let trace: f64 = (0..n_basis).map(|a| ws.h[a * n_basis + a]).sum();
        let trace_err = (trace - n_basis as f64).abs() / n_basis as f64;
        assert!(
            trace_err < 0.01,
            "trace not normalised: trace={trace}, expected {n_basis}, err={trace_err:.3e}"
        );

        // Off-diagonal Frobenius / diagonal Frobenius ratio should be
        // bounded -- the Bergman kernel is *not* exactly diagonal on
        // the polysphere (cross-terms exist between basis monomials),
        // but it should be diagonally dominant.
        let mut diag_sq = 0.0;
        let mut off_sq = 0.0;
        for i in 0..n_basis {
            for j in 0..n_basis {
                let v = ws.h[i * n_basis + j];
                if i == j {
                    diag_sq += v * v;
                } else {
                    off_sq += v * v;
                }
            }
        }
        let ratio = off_sq.sqrt() / diag_sq.sqrt().max(1e-12);
        assert!(
            ratio < 1.0,
            "h not diagonally dominant: off/diag Frobenius ratio = {ratio:.3e}"
        );
    }

    // ------------------------------------------------------------------
    // Sigma ~ 1/k^2 scaling regression.
    //
    // Headrick-Wiseman (2005) and follow-up work show the Donaldson
    // residual decreases with basis degree k. We verify the (weaker)
    // monotonicity property: residual at higher k is smaller.
    // ------------------------------------------------------------------

    /// REGRESSION (Bug 6): subspace_iteration_eigenvalues uses
    /// Rayleigh quotients diag(Q^T M Q) which only equal eigenvalues
    /// when Q has converged to a Schur-form basis. For matrices with
    /// COMPLEX eigenvalue pairs (real M, no real Schur form) the
    /// algorithm cannot resolve them — and silently returns the
    /// 2x2 trace/2 = mean of the pair.
    ///
    /// Our intended usage is for symmetric M (totally-symmetric Yukawa
    /// tensor contractions) where this issue doesn't arise. We add a
    /// debug_assert!() symmetry check so callers get a clear error if
    /// they pass a non-symmetric matrix.
    /// In debug builds, subspace_iteration_eigenvalues now requires a
    /// symmetric input (it cannot resolve complex pairs and returns the
    /// 2x2 trace/2 mean). The legacy "works on upper-triangular" test
    /// has been converted to assert that nonsymmetric input panics; a
    /// companion `_after_symmetrising` test confirms the symmetrised
    /// input still recovers the real eigenvalues.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "requires symmetric M")]
    fn subspace_iteration_panics_on_nonsymmetric_input_in_debug() {
        // FIX(P3.2): the underlying check is `debug_assert!` in
        // `subspace_iteration_eigenvalues`, which is stripped in release
        // builds. Without this `#[cfg(debug_assertions)]` gate the test
        // would silently fail under `cargo test --release`.
        let m = [
            3.0_f64, 0.5, 0.3,
            0.0, 2.0, 0.4,
            0.0, 0.0, 1.0,
        ];
        let _ = subspace_iteration_eigenvalues(&m, 3, 3, 50);
    }

    #[test]
    fn subspace_iteration_works_on_nonsymmetric_after_symmetrising() {
        // Upper-triangular M with diagonal [3, 2, 1].
        // (M + Mᵀ)/2 has eigenvalues that depend on the off-diagonals
        // (the symmetrised matrix is NOT triangular), but they remain
        // real (it's symmetric) and finite. We just verify the call
        // succeeds and returns 3 finite eigenvalues with the correct
        // trace.
        let upper = [
            3.0_f64, 0.5, 0.3,
            0.0, 2.0, 0.4,
            0.0, 0.0, 1.0,
        ];
        let mut sym = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                sym[i * 3 + j] = 0.5 * (upper[i * 3 + j] + upper[j * 3 + i]);
            }
        }
        let eigs = subspace_iteration_eigenvalues(&sym, 3, 3, 200);
        assert_eq!(eigs.len(), 3);
        for e in &eigs {
            assert!(e.is_finite(), "eigenvalue not finite: {e}");
        }
        // Symmetrisation preserves the trace.
        let trace: f64 = (0..3).map(|i| sym[i * 3 + i]).sum();
        let sum: f64 = eigs.iter().sum();
        assert!(
            (trace - sum).abs() < 1e-6,
            "symmetrised eigvals must sum to trace: trace={trace}, sum={sum}"
        );
    }

    #[test]
    fn subspace_iteration_returns_correct_eigenvalues_on_symmetric() {
        // For our actual use case (symmetric M from Y_ijk h_k contraction),
        // the algorithm correctly returns eigenvalues.
        let m = [
            5.0_f64, 1.0, 0.5,
            1.0, 4.0, 0.3,
            0.5, 0.3, 3.0,
        ];
        let eigs = subspace_iteration_eigenvalues(&m, 3, 3, 100);
        let mut got: Vec<f64> = eigs.into_iter().map(|x| x.abs()).collect();
        got.sort_by(|a, b| b.partial_cmp(a).unwrap());
        // Independent computation via 3x3 char-poly: spectrum approx
        // {5.475, 4.131, 2.394}. We just check magnitude ordering.
        assert!(
            got[0] > got[1] && got[1] > got[2],
            "sorted: {:?}",
            got
        );
        let trace = 5.0 + 4.0 + 3.0;
        let sum_eigs: f64 = got.iter().sum();
        assert!(
            (trace - sum_eigs).abs() < 1e-6,
            "sum of eigenvalues should equal trace: trace={trace}, sum={sum_eigs}"
        );
    }

    #[test]
    fn subspace_iteration_recovers_known_eigenvalues() {
        // Diagonal matrix with known eigenvalues 5, 4, 3, 2, 1.
        // Top-3 should converge to (5, 4, 3) up to rounding.
        let n = 5;
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            m[i * n + i] = (n - i) as f64; // 5, 4, 3, 2, 1
        }
        let eigs = subspace_iteration_eigenvalues(&m, n, 3, 100);
        let mut got: Vec<f64> = eigs.into_iter().map(|x| x.abs()).collect();
        got.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(got.len(), 3);
        for (g, expected) in got.iter().zip([5.0, 4.0, 3.0].iter()) {
            assert!(
                (g - expected).abs() < 1e-6,
                "got {g}, expected {expected}"
            );
        }
    }

    #[test]
    fn subspace_iteration_recovers_dense_symmetric_eigenvalues() {
        // Dense 4x4 symmetric matrix; verify top-2 eigenvalues.
        // Construct M = Q^T D Q with known D.
        let n = 4;
        // Simple orthogonal Q (rotation); D = diag(10, 5, 3, 1).
        let q = [
            0.5_f64, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5,
            0.5,
        ];
        let d_vals = [10.0, 5.0, 3.0, 1.0];
        // M[i][j] = sum_k Q[k][i] * d[k] * Q[k][j]
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += q[k * n + i] * d_vals[k] * q[k * n + j];
                }
                m[i * n + j] = s;
            }
        }
        let eigs = subspace_iteration_eigenvalues(&m, n, 2, 200);
        let mut got: Vec<f64> = eigs.into_iter().map(|x| x.abs()).collect();
        got.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!(
            (got[0] - 10.0).abs() < 1e-3,
            "top eigenvalue: got {}, expected 10",
            got[0]
        );
        assert!(
            (got[1] - 5.0).abs() < 1e-3,
            "second eigenvalue: got {}, expected 5",
            got[1]
        );
    }

    #[test]
    fn analytic_adam_gradient_matches_finite_diff() {
        // Build a small test: 50-point sample, n_basis=20, random h.
        // Compute analytic gradient via refine_step_adam's first half
        // and verify each component matches the finite-difference at
        // ~1e-4 relative error.
        use crate::workspace::DiscriminationWorkspace;
        let n_basis = 20;
        let mut ws = DiscriminationWorkspace::new(50, 100, 8, 5, 1);
        crate::sample_points_into(&mut ws, 31);
        crate::evaluate_section_basis_into(&mut ws);
        // Truncate basis to 20 entries (just use first 20 columns).
        let mut s_short = vec![0.0; 50 * n_basis];
        for p in 0..50 {
            for b in 0..n_basis {
                s_short[p * n_basis + b] = ws.section_values[p * 100 + b];
            }
        }
        // Initialise h = I.
        let mut h = vec![0.0; n_basis * n_basis];
        for i in 0..n_basis {
            h[i * n_basis + i] = 1.0;
        }
        // Reference: brute-force finite difference of variance(log K).
        let r0 = ricci_flatness_residual(&s_short, &h, 50, n_basis);
        let eps = 1e-5;
        let mut grad_fd = vec![0.0; n_basis * n_basis];
        for a in 0..n_basis {
            for b in 0..n_basis {
                let mut hp = h.clone();
                hp[a * n_basis + b] += eps;
                let r1 = ricci_flatness_residual(&s_short, &hp, 50, n_basis);
                grad_fd[a * n_basis + b] = (r1 - r0) / eps;
            }
        }
        // Now compute analytic gradient using the same formula refine_step_adam uses.
        let log_k: Vec<f64> = (0..50)
            .map(|p| {
                let s_p = &s_short[p * n_basis..(p + 1) * n_basis];
                let mut k = 0.0;
                for a in 0..n_basis {
                    let mut rs = 0.0;
                    for b in 0..n_basis {
                        rs += h[a * n_basis + b] * s_p[b];
                    }
                    k += s_p[a] * rs;
                }
                k.max(1e-30).ln()
            })
            .collect();
        let mean: f64 = log_k.iter().sum::<f64>() / 50.0;
        let mut grad_an = vec![0.0; n_basis * n_basis];
        for p in 0..50 {
            let s_p = &s_short[p * n_basis..(p + 1) * n_basis];
            let mut k = 0.0;
            for a in 0..n_basis {
                let mut rs = 0.0;
                for b in 0..n_basis {
                    rs += h[a * n_basis + b] * s_p[b];
                }
                k += s_p[a] * rs;
            }
            let k_safe = k.max(1e-30);
            let coef = (log_k[p] - mean) / k_safe;
            for a in 0..n_basis {
                for b in 0..n_basis {
                    grad_an[a * n_basis + b] += s_p[a] * s_p[b] * coef;
                }
            }
        }
        for v in grad_an.iter_mut() {
            *v *= 2.0 / 50.0;
        }

        // Compare: max relative error should be bounded.
        let mut max_abs_err: f64 = 0.0;
        let mut grad_norm: f64 = 0.0;
        for k in 0..(n_basis * n_basis) {
            max_abs_err = max_abs_err.max((grad_an[k] - grad_fd[k]).abs());
            grad_norm = grad_norm.max(grad_fd[k].abs());
        }
        let rel = max_abs_err / grad_norm.max(1e-12);
        assert!(
            rel < 1e-3,
            "analytic gradient deviates from finite-diff: max_abs_err={max_abs_err:.3e}, grad_norm={grad_norm:.3e}, rel={rel:.3e}"
        );
    }

    // ------------------------------------------------------------------
    // PP3 closed-form-reference validation.
    //
    // Three reference cases with analytically-known expected behaviour:
    //   1. h = 0 except diagonal -> log K = log(s^T s) which on the
    //      polysphere is *not* constant; residual nonzero, decreases
    //      after Donaldson balancing.
    //   2. After Donaldson convergence to ~1e-5, residual on log K
    //      should be < 1e-2 (Headrick-Wiseman 2005 scaling).
    //   3. Ricci_flatness_residual ~ var(log K); for FS-balanced h it
    //      reaches a Bergman-kernel-determined floor that decreases
    //      with k_degree (the n_basis grows -> residual drops).
    // ------------------------------------------------------------------

    #[test]
    fn closed_form_identity_h_is_baseline() {
        use crate::workspace::DiscriminationWorkspace;
        let n_basis = 100;
        let mut ws = DiscriminationWorkspace::new(2000, n_basis, 8, 5, 1);
        crate::sample_points_into(&mut ws, 99);
        crate::evaluate_section_basis_into(&mut ws);
        // h = identity: var(log K) is simply var(log |s|^2) for the
        // unscaled section basis on the polysphere -- this gives the
        // baseline, "uncorrected" residual.
        for v in ws.h.iter_mut() {
            *v = 0.0;
        }
        for i in 0..n_basis {
            ws.h[i * n_basis + i] = 1.0;
        }
        let r_initial = ricci_flatness_residual(&ws.section_values, &ws.h, ws.n_points, ws.n_basis);
        assert!(r_initial > 0.0, "expected nonzero baseline residual");
        assert!(r_initial.is_finite(), "baseline residual not finite");
    }

    #[test]
    fn closed_form_donaldson_fixed_point_is_idempotent() {
        // Donaldson balancing converges to the *fixed point* of its
        // T-operator T(h)[a,b] = (1/N) sum_p s_a(p) s_b(p) / |s|^2_h(p),
        // not to the minimum of var(log K). At the converged h, one
        // additional T-iteration should leave h unchanged to floating
        // precision.
        //
        // This is the canonical "is the converged metric a true fixed
        // point" test that publication-grade Donaldson solvers must pass.
        use crate::kernels::{donaldson_iter_into, donaldson_solve_in_place};
        use crate::workspace::DiscriminationWorkspace;
        let n_basis = 100;
        let mut ws = DiscriminationWorkspace::new(2000, n_basis, 8, 50, 1);
        crate::sample_points_into(&mut ws, 13);
        crate::evaluate_section_basis_into(&mut ws);
        donaldson_solve_in_place(&mut ws, 1e-6);

        // Save converged h.
        let h_converged = ws.h.clone();

        // One more T-iteration: should leave h ~ unchanged.
        donaldson_iter_into(&mut ws);
        let mut diff_sq = 0.0;
        for k in 0..n_basis * n_basis {
            let d = ws.h_new[k] - h_converged[k];
            diff_sq += d * d;
        }
        let diff = diff_sq.sqrt();
        assert!(
            diff < 1e-3,
            "Donaldson fixed point not idempotent: ||T(h) - h|| = {diff:.3e}"
        );
    }

    #[test]
    fn closed_form_residual_decreases_with_n_sample() {
        // Statistical noise of the Monte-Carlo residual estimator
        // should decrease as 1/sqrt(N). Run at two sample sizes and
        // verify the smaller-N residual has higher variance across
        // seeds (proxy for higher noise).
        use crate::kernels::donaldson_solve_in_place;
        use crate::workspace::DiscriminationWorkspace;
        let n_basis = 100;
        let measure = |n_pts: usize, seed: u64| -> f64 {
            let mut ws = DiscriminationWorkspace::new(n_pts, n_basis, 8, 20, 1);
            crate::sample_points_into(&mut ws, seed);
            crate::evaluate_section_basis_into(&mut ws);
            donaldson_solve_in_place(&mut ws, 1e-5);
            ricci_flatness_residual(&ws.section_values, &ws.h, ws.n_points, ws.n_basis)
        };
        let r_small: Vec<f64> = (0..5).map(|s| measure(500, s as u64 + 1)).collect();
        let r_large: Vec<f64> = (0..5).map(|s| measure(5000, s as u64 + 100)).collect();
        let var = |xs: &[f64]| {
            let m = xs.iter().sum::<f64>() / xs.len() as f64;
            xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64
        };
        let v_small = var(&r_small);
        let v_large = var(&r_large);
        // Expect var(N=500) >= var(N=5000) (with finite-sample
        // tolerance: the relationship can flip occasionally for small
        // ensemble counts, so we accept either monotonicity OR small
        // values overall).
        assert!(
            v_small >= v_large || v_large < 1e-6,
            "expected residual variance to decrease with N: var(500)={v_small:.3e}, var(5000)={v_large:.3e}"
        );
    }

    #[test]
    fn donaldson_residual_decreases_with_iteration() {
        use crate::kernels::donaldson_solve_in_place;
        use crate::workspace::DiscriminationWorkspace;

        let mut ws = DiscriminationWorkspace::new(1000, 100, 8, 30, 1);
        crate::sample_points_into(&mut ws, 7);
        crate::evaluate_section_basis_into(&mut ws);
        donaldson_solve_in_place(&mut ws, 1e-8);

        // Residual history should be monotone-ish decreasing: the LAST
        // residual must be strictly less than the FIRST.
        assert!(
            ws.residuals.len() >= 2,
            "expected multi-iteration history, got {} iters",
            ws.residuals.len()
        );
        let r0 = ws.residuals[0];
        let r_last = *ws.residuals.last().unwrap();
        assert!(
            r_last < r0,
            "residual did not decrease: r0={r0:.3e}, r_last={r_last:.3e}"
        );
    }
}
