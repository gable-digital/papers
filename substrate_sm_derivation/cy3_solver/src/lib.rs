//! Rust port of the CY3 substrate-discrimination heavy numerical kernels,
//! built on pwos-math's SIMD-accelerated NdArray primitives.
//!
//! Mirrors the algorithms in
//! `book/scripts/cy3_substrate_discrimination/tier_bc/solvers/{ricci_flat_metric,yukawa_overlap}.py`
//! using `pwos_math::ndarray::NdArray<f64>` for storage and pwos-math's
//! matmul / inverse / matvec primitives for the heavy linear algebra.
//!
//! The complex-valued aspects of the algorithms are represented as
//! 2N x 2N real-valued block-Hermitian matrices: a complex matrix C
//! decomposes as C = A + iB (A symmetric, B antisymmetric), and the
//! 2x2 block matrix [[A, -B], [B, A]] preserves Hermitian structure
//! under all the matmul / inverse / outer-product operations the
//! Donaldson algorithm requires.
//!
//! ## Publication-grade scope statement
//!
//! What this codebase IS:
//!  - A multi-pass discrimination pipeline (broad → refine → precision)
//!    with crash-safe checkpointing and reproducibility metadata.
//!  - A Donaldson balancing solver on the polysphere ansatz S^3 × S^3,
//!    with verified fixed-point idempotency (||T(h) - h|| < 1e-3 at
//!    convergence).
//!  - A real-Monge-Ampère residual on tangent-space-projected Hessians
//!    (8x8 → 6x6 to remove polysphere radial constraint directions).
//!  - Analytic Adam optimizer for h-coefficient refinement (10000x
//!    faster than the finite-difference baseline at n_basis=100).
//!  - Subspace-iteration eigenvalue extraction for Yukawa mass spectrum
//!    (replaces Wielandt deflation; preserves precision across all
//!    extracted eigenvalues, verified against known eigenvalues).
//!  - Stage-1 topology filters AND real heterotic-bundle structure
//!    (monad-bundle Chern classes, c_2(V)=c_2(TM) Bianchi anomaly
//!    cancellation, polystability slope inequality, E_8 Wilson lines
//!    in 8-dim Cartan with Z/3 quantization, E_6 x SU(3) breaking
//!    pattern, bundle Z/3 invariance).
//!  - Forward-models for α(m_Z) / M_W / Λ_QCD with calibration-point
//!    tests; α now matches at the EW unification scale (PDG MS-bar)
//!    rather than α(0).
//!  - Separated u/d/lepton Yukawa sectors with full SVD-based
//!    diagonalisation, CKM rotation extraction (V_CKM = V_uL V_dL†),
//!    and PDG-mass + CKM-magnitude losses.
//!  - Z/3-quotient projector for sections; restricts the basis to
//!    Z/3-invariant sub-space.
//!  - Bootstrap error bars (mean ± SE, percentile CI) for residual
//!    estimators.
//!  - **Chapter-8 five-route discrimination pipeline** wired into
//!    [`crate::pipeline::compute_5sigma_score_for_candidate`]
//!    through plumbed [`crate::pipeline::FiveSigmaBreakdown`] fields:
//!      - Route 1: empirical-observable boundary penalties on the
//!        CY3 metric ([`crate::route12::route1`]; predictors deferred
//!        until the metric-derived predictor pipeline lands —
//!        currently surfaces the API only).
//!      - Route 2: gauge-Yukawa magnitudes from empirical force
//!        constants via the cross-term-as-coupling identity
//!        ([`crate::route12::route2`]; chi^2 contribution
//!        `loss_route2_gauge`, always-on).
//!      - Route 3: η-integral on the candidate geometry vs the
//!        chapter's `η_obs = (6.115 ± 0.038) × 10⁻¹⁰`
//!        ([`crate::route34::eta_evaluator`]; chi^2 contribution
//!        `loss_eta_chi2`, opt-in via `compute_eta_chi2`).
//!      - Route 4: polyhedral-resonance ADE wavenumber discrimination
//!        at Saturn / Jupiter polar critical-boundaries
//!        ([`crate::route34::route4_predictor`]; chi^2 contribution
//!        `loss_route4_chi2`, always-on).
//!      - Route 5: scalar spectral index `n_s = 58/60` from
//!        `E_8 × E_8` Coxeter geometry vs Planck 2018
//!        `n_s = 0.9649 ± 0.0042` ([`crate::route5::spectral_index`];
//!        chi^2 contribution `loss_route5_ns`, always-on; closed-
//!        form, no sampling required).
//!    Top-level [`crate::pipeline::sweep_candidates`] aggregates
//!    `chi2_pdg + ckm_unitarity_residual + loss_route2_gauge +
//!    loss_eta_chi2 + loss_route4_chi2 + loss_route5_ns` into a
//!    single ranking key.
//!
//! What this codebase IS NOT (research-scope deferred items):
//!
//!  - **P1 (Dirac zero-modes)**: True fermion zero-modes from solving
//!    the twisted Dirac equation D_V psi = 0 on the CY3 with the
//!    bundle's connection. The current pipeline uses **polynomial
//!    seeds + lite L²-residue harmonic projection**
//!    ([`crate::zero_modes::evaluate_polynomial_seeds`] +
//!    [`crate::zero_modes::project_to_harmonic`]) in place of the
//!    full ∂̄_E* projection. The lite version unit-normalises modes
//!    under the CY measure but does not enforce co-closedness against
//!    the bundle Hermite-Einstein metric. Full Dirac-zero-mode
//!    solving is the canonical "30 years of effort" item that
//!    mainstream literature (Anderson-Karp-Lukas-Palti 2010,
//!    Ashmore-Lukas 2020, Halverson-Tian 2018, Larfors-Schneider-
//!    Strominger 2022) has only recently begun to crack via NN-
//!    accelerated methods to ~few-percent precision.
//!  - **P2 (real Yukawa overlap)**: Triple overlap integrals
//!    ∫_M ψ_i ∧ ψ_j ∧ ψ_k ∧ Ω̄ on the actual CY3 using zero-modes
//!    from P1. The current implementation
//!    ([`crate::yukawa_overlap::compute_yukawa_spectrum`]) reduces
//!    the bundle ε-tensor cup product to a per-point scalar product
//!    and uses a placeholder bundle metric H = identity; both are
//!    documented inline as caveats. Putting these on a real footing
//!    depends on P1 + bundle Hermite-Einstein balancing (P3).
//!  - **P3 (real bundle Hermite-Einstein metric H)**: Donaldson
//!    balancing on the gauge bundle V (analogous to the existing
//!    Donaldson balancing on the CY3 metric in [`crate::quintic`]).
//!    Without this, Yukawa magnitudes are wrong by O(1) factors that
//!    don't cancel in the χ². The "5σ scoring" claim depends on this
//!    being implemented.
//!  - **P6 (derived Higgs VEV)**: The mu-problem in heterotic;
//!    requires moduli stabilisation analysis (Anderson-Constantin-
//!    Lukas-Palti 2017+).
//!  - **P9 (real baryogenesis)**: CP-violating reheating + sphaleron
//!    dynamics is research-scope and unrelated to CY3 metrics.
//!
//! What this codebase IS that an earlier revision of this preamble
//! claimed it WAS NOT (recently-built items):
//!
//!  - **M4 (CY3 ideal sampling)**: real Newton-projection sampling on
//!    the CY3 ideal, both for the Tian-Yau Z/3 triple
//!    ([`crate::cicy_sampler`], NHYPER=3 line-intersection on
//!    `CP^3 × CP^3`) and the Schoen Z/3 × Z/3 fiber product
//!    ([`crate::route34::schoen_sampler`], NHYPER=2 on
//!    `CP^2 × CP^2 × CP^1`). Both have CPU implementations with
//!    rayon parallelism and GPU-accelerated paths
//!    ([`crate::gpu_sampler`] / `crate::route34::schoen_sampler_gpu`)
//!    behind the `gpu` feature.
//!  - **M7 (Killing-vector spectrum)**: real Killing-equation solver
//!    on the CY3 metric via the Lichnerowicz operator
//!    (`crate::route34::{lichnerowicz, killing_solver,
//!    isometry_subgroups}`), with continuous-isometry-group
//!    classification used by Route 4 (polyhedral-resonance
//!    discrimination).
//!  - **PP2 (literature reproduction)**: published h-coefficients for
//!    Headrick-Wiseman 2005 (quintic) and Anderson-Karp-Lukas-Palti
//!    2010 (Tian-Yau) are encoded with provenance tags in
//!    [`crate::reference_metrics`]; the per-k uncertainty schedule
//!    comes from the published residual figures.
//!
//! For a journal submission, the deferred items above would need
//! plumbed-through implementations OR explicit citation of existing
//! implementations as black-box prior work.
//!
//! ## Conventions
//!
//! Coordinate convention: 8 real ambient coords (z_0..z_3, z_4..z_7),
//! with two unit-norm constraints ||z_a||_2 = ||z_b||_2 = 1 placing the
//! sample on S^3 × S^3 ⊂ R^8. The 6-real-dim CY tangent at each point
//! is computed by removing the two radial directions.
//!
//! Trace normalisation: Donaldson balancing fixes trace(h) = n_basis
//! at each iteration, so the mean diagonal of h is identically 1.0.
//! Spectral statistics that vary across candidates are `max`, `min`,
//! and `gap = max - min`.
//!
//! Sign / scale conventions for forward-models (calibration values):
//!  - α at calibration: em_sector_norm = √(4πα), h_max = 1.0 → α exactly.
//!  - M_W at calibration: weak_sector_norm = M_W / v_eff with
//!    v_eff = 246 GeV × sin(θ_W) / 2 ≈ 80.4 GeV.
//!  - Λ_QCD: dimensional transmutation with b_0 = 7.

extern crate pwos_math;
extern crate rayon;

use pwos_math::ndarray::NdArray;
use rayon::prelude::*;

pub mod linalg;
pub mod workspace;
pub mod kernels;
pub mod refine;
pub mod pipeline;
pub mod topology_filters;
pub mod heterotic;
pub mod quotient;
pub mod yukawa_sectors;
pub mod bootstrap;
pub mod quintic;
pub mod calabi_metric;
pub mod geometry;
pub mod cicy_sampler;
pub mod automorphism;
pub mod reference_metrics;
pub mod pdg;
pub mod zero_modes;
pub mod gpu_adam;
pub mod yukawa_overlap;
pub mod simd_kernels;
pub mod route12;
pub mod route34;
pub mod route5;
pub mod nn_metric;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "gpu")]
pub mod orchestrator;

#[cfg(feature = "gpu")]
pub mod gpu_quintic;

#[cfg(feature = "gpu")]
pub mod gpu_yukawa;

#[cfg(feature = "gpu")]
pub mod gpu_omega;

#[cfg(feature = "gpu")]
pub mod gpu_polynomial_seeds;

#[cfg(feature = "gpu")]
pub mod gpu_sampler;

#[cfg(feature = "gpu")]
pub mod gpu_harmonic;

pub use kernels::{
    discriminate_in_place, donaldson_iter_into, donaldson_solve_in_place,
    dominant_eigenvalue_in_place, evaluate_section_basis_into,
    init_yukawa_centers, sample_points_into, yukawa_tensor_in_place,
};
pub use workspace::{DiscriminationWorkspace, build_degree2_monomials, N_BASIS_DEGREE2};

/// Simple deterministic RNG: linear congruential. Matches the seed in the
/// Python reference for cross-language reproducibility tests.
pub struct LCG(pub u64);

impl LCG {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1))
    }

    pub fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = ((self.0 >> 11) & ((1u64 << 53) - 1)) as f64;
        bits / ((1u64 << 53) as f64)
    }

    /// Box-Muller normal sample.
    pub fn next_normal(&mut self) -> f64 {
        let u1 = (self.next_f64() + 1e-12).max(1e-12);
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        r * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Sample N points (8 real coordinates each) on the polysphere
/// approximating CP^3 x CP^3 / Tian-Yau quotient.
pub fn sample_points(n: usize, seed: u64) -> NdArray<f64> {
    let mut rng = LCG::new(seed);
    let mut data = Vec::with_capacity(n * 8);
    for _ in 0..n {
        let mut z = [0.0f64; 8];
        for k in 0..8 {
            z[k] = rng.next_normal();
        }
        // Normalise the two CP^3 factors
        let n1: f64 = (z[0] * z[0] + z[1] * z[1] + z[2] * z[2] + z[3] * z[3]).sqrt();
        let n2: f64 = (z[4] * z[4] + z[5] * z[5] + z[6] * z[6] + z[7] * z[7]).sqrt();
        for k in 0..4 {
            z[k] /= n1.max(1e-10);
        }
        for k in 4..8 {
            z[k] /= n2.max(1e-10);
        }
        for v in z {
            data.push(v);
        }
    }
    NdArray::from_vec(&[n, 8], data).unwrap()
}

/// Build a degree-2 monomial-section evaluation matrix at the given points.
/// Returns shape (n_points, n_basis) array of real-valued section values
/// (for benchmarking purposes; the discrimination calculation in the
/// Python reference uses complex values, but the magnitude structure
/// reproduces under real-valued ansatz for the benchmark).
pub fn evaluate_section_basis_realvalued(points: &NdArray<f64>) -> NdArray<f64> {
    let n_points = points.shape()[0];
    // Degree-2 bigraded monomials in 4+4 real coordinates: a0+a1+a2+a3 = 2
    // and b0+b1+b2+b3 = 2. Number of bigraded degree-2 monomials per
    // factor = C(5,3) = 10; total = 100.
    let mut monomials: Vec<[u32; 8]> = Vec::new();
    for a0 in 0..=2 {
        for a1 in 0..=(2 - a0) {
            for a2 in 0..=(2 - a0 - a1) {
                let a3 = 2 - a0 - a1 - a2;
                for b0 in 0..=2 {
                    for b1 in 0..=(2 - b0) {
                        for b2 in 0..=(2 - b0 - b1) {
                            let b3 = 2 - b0 - b1 - b2;
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
    let n_basis = monomials.len();

    let pts = points.data();
    // Parallelise per-row monomial evaluation. Inside each row, we
    // pre-tabulate the 8 coordinates' powers (z[k]^0, z[k]^1, z[k]^2)
    // once, then look up each monomial as a product of 8 entries from
    // that table. This avoids the f64::powi(2) hot-loop calls that
    // dominated the previous implementation.
    let mut data = vec![0.0; n_points * n_basis];
    data.par_chunks_mut(n_basis)
        .with_min_len(32)
        .enumerate()
        .for_each(|(i, row)| {
            let z = &pts[i * 8..(i + 1) * 8];
            // Power table: pow_table[k][e] = z[k]^e for e in 0..=2.
            // Stored flat as [k * 3 + e] for cache locality.
            let mut pow_table = [1.0f64; 24];
            for k in 0..8 {
                pow_table[k * 3] = 1.0;
                pow_table[k * 3 + 1] = z[k];
                pow_table[k * 3 + 2] = z[k] * z[k];
            }
            for j in 0..n_basis {
                let m = &monomials[j];
                row[j] = pow_table[m[0] as usize]
                    * pow_table[3 + m[1] as usize]
                    * pow_table[6 + m[2] as usize]
                    * pow_table[9 + m[3] as usize]
                    * pow_table[12 + m[4] as usize]
                    * pow_table[15 + m[5] as usize]
                    * pow_table[18 + m[6] as usize]
                    * pow_table[21 + m[7] as usize];
            }
        });
    NdArray::from_vec(&[n_points, n_basis], data).unwrap()
}

/// Parallel row-chunked matmul: splits the rows of `a` into chunks, runs
/// pwos-math's cache-blocked GEMM on each chunk in parallel against `b`,
/// and concatenates the results back into a single output array. This
/// gives us multi-threaded scaling on top of pwos-math's single-threaded
/// GEMM microkernels, with each chunk still benefiting from cache-blocked
/// packing.
pub fn parallel_matmul_against_small_rhs(a: &NdArray<f64>, b: &NdArray<f64>) -> NdArray<f64> {
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    let n_threads = rayon::current_num_threads().max(1);
    // Aim for roughly one chunk per thread, with a floor of 64 rows per chunk
    // to keep each pwos-math GEMM call fed with enough work for its panel
    // packing to amortise.
    let chunk_rows = ((m + n_threads - 1) / n_threads).max(64);

    let chunks: Vec<(usize, usize)> = (0..m)
        .step_by(chunk_rows)
        .map(|start| (start, (start + chunk_rows).min(m)))
        .collect();

    // Build chunk NdArrays in parallel and run pwos-math matmul on each.
    let chunk_results: Vec<(usize, NdArray<f64>)> = chunks
        .into_par_iter()
        .map(|(start, end)| {
            let chunk_data: Vec<f64> = a.data()[start * k..end * k].to_vec();
            let chunk = NdArray::from_vec(&[end - start, k], chunk_data).unwrap();
            (start, chunk.matmul(b).unwrap())
        })
        .collect();

    let mut output = vec![0.0; m * n];
    for (start, result) in chunk_results {
        let result_data = result.data();
        let rows = result.shape()[0];
        let dst_start = start * n;
        let dst_end = dst_start + rows * n;
        output[dst_start..dst_end].copy_from_slice(&result_data[..rows * n]);
    }
    NdArray::from_vec(&[m, n], output).unwrap()
}

/// Pre-allocated workspace for the Donaldson balancing solver. Buffers
/// are sized once at construction; subsequent iterations reuse them.
pub struct DonaldsonWorkspace {
    pub n_points: usize,
    pub n_basis: usize,
    pub weights: Vec<f64>,
    pub sw_buffer: Vec<f64>,
    pub residuals: Vec<f64>,
}

impl DonaldsonWorkspace {
    pub fn new(n_points: usize, n_basis: usize, max_iter: usize) -> Self {
        Self {
            n_points,
            n_basis,
            weights: vec![0.0; n_points],
            sw_buffer: vec![0.0; n_points * n_basis],
            residuals: Vec::with_capacity(max_iter),
        }
    }
}

/// One Donaldson balancing iteration on the section-basis Gram matrix h.
///
///   h_new[a,b] = (1/n_points) sum_i s_a(z_i) s_b(z_i) / |s|_h^2(z_i)
///
/// where |s|_h^2(z_i) = sum_{cd} h^{cd} s_c(z_i) s_d(z_i).
///
/// All operations are performed via pwos-math's NdArray primitives.
pub fn donaldson_iteration(
    section_values: &NdArray<f64>,
    h: &NdArray<f64>,
    ws: &mut DonaldsonWorkspace,
) -> NdArray<f64> {
    let n_points = section_values.shape()[0];
    let n_basis = section_values.shape()[1];
    debug_assert_eq!(ws.n_points, n_points);
    debug_assert_eq!(ws.n_basis, n_basis);

    // Compute h_inv via pwos-math matrix inverse
    let h_inv = h.inverse().expect("Donaldson: h must be invertible");

    // Per-point weights w_i = sum_{ab} h_inv[a,b] s_a(z_i) s_b(z_i).
    //
    // Restructured as two cache-friendly stages:
    //   1. T = S @ h_inv   (parallel cache-blocked GEMM via row-chunked
    //                       pwos-math matmul calls in rayon)
    //   2. w_i = T[i] . S[i]   (per-row dot, sequential cache access)
    let t_matrix = parallel_matmul_against_small_rhs(section_values, &h_inv);
    let s_data = section_values.data();
    let t_data = t_matrix.data();
    // Reuse pre-allocated weights buffer
    ws.weights
        .par_iter_mut()
        .with_min_len(32)
        .enumerate()
        .for_each(|(i, w_out)| {
            let t_i = &t_data[i * n_basis..(i + 1) * n_basis];
            let s_i = &s_data[i * n_basis..(i + 1) * n_basis];
            let mut w = 0.0;
            for a in 0..n_basis {
                w += t_i[a] * s_i[a];
            }
            *w_out = w.max(1e-12);
        });

    // h_new[a,b] = (1/n) sum_i s_a(z_i) s_b(z_i) / w_i
    // = Sw^T @ Sw / n where Sw[i, a] = s_a(z_i) / sqrt(w_i).
    // Reuse pre-allocated sw_buffer
    let weights = &ws.weights;
    ws.sw_buffer
        .par_chunks_mut(n_basis)
        .with_min_len(32)
        .enumerate()
        .for_each(|(i, row)| {
            let inv_sqrt_w = 1.0 / weights[i].sqrt();
            let s_i = &s_data[i * n_basis..(i + 1) * n_basis];
            for a in 0..n_basis {
                row[a] = s_i[a] * inv_sqrt_w;
            }
        });
    // The NdArray::from_vec call below requires owning the Vec. We move
    // sw_buffer out, do the matmul, and put a fresh empty buffer back.
    // Subsequent iterations will reallocate this buffer; that's the
    // unavoidable cost of pwos-math's owning-Vec API.
    let sw_data = std::mem::replace(&mut ws.sw_buffer, vec![0.0; n_points * n_basis]);
    let sw = NdArray::from_vec(&[n_points, n_basis], sw_data).unwrap();
    let sw_t = sw.transpose_2d().unwrap();

    let mut h_new = sw_t.matmul(&sw).unwrap();

    // Normalise to fixed trace
    let mut data: Vec<f64> = h_new.data().to_vec();
    let trace: f64 = (0..n_basis).map(|a| data[a * n_basis + a]).sum();
    if trace > 1e-10 {
        let scale = (n_basis as f64) / trace;
        for v in &mut data {
            *v *= scale;
        }
        h_new = NdArray::from_vec(&[n_basis, n_basis], data).unwrap();
    }

    h_new
}

/// Run the Donaldson balancing algorithm to convergence (or max_iter)
/// using a pre-allocated workspace.
pub fn donaldson_solve_with_workspace(
    section_values: &NdArray<f64>,
    max_iter: usize,
    tol: f64,
    ws: &mut DonaldsonWorkspace,
) -> (NdArray<f64>, Vec<f64>) {
    let n_basis = section_values.shape()[1];
    // Initialise h = I
    let mut h_data = vec![0.0; n_basis * n_basis];
    for a in 0..n_basis {
        h_data[a * n_basis + a] = 1.0;
    }
    let mut h = NdArray::from_vec(&[n_basis, n_basis], h_data).unwrap();
    ws.residuals.clear();
    for _ in 0..max_iter {
        let h_new = donaldson_iteration(section_values, &h, ws);
        let h_data_ref = h.data();
        let h_new_data = h_new.data();
        let mut diff_sq = 0.0;
        for k in 0..h_data_ref.len() {
            let d = h_new_data[k] - h_data_ref[k];
            diff_sq += d * d;
        }
        let residual = diff_sq.sqrt();
        ws.residuals.push(residual);
        h = h_new;
        if residual < tol {
            break;
        }
    }
    let residuals = ws.residuals.clone();
    (h, residuals)
}

/// Convenience wrapper that allocates a workspace per call.
pub fn donaldson_solve(
    section_values: &NdArray<f64>,
    max_iter: usize,
    tol: f64,
) -> (NdArray<f64>, Vec<f64>) {
    let n_points = section_values.shape()[0];
    let n_basis = section_values.shape()[1];
    let mut ws = DonaldsonWorkspace::new(n_points, n_basis, max_iter);
    donaldson_solve_with_workspace(section_values, max_iter, tol, &mut ws)
}

/// Yukawa-tensor triple-overlap computation. For benchmark purposes we
/// represent zero-mode wavefunctions as Gaussian profiles centred at
/// bundle-moduli-derived points; the actual contraction structure is
/// what dominates the timing.
///
/// Returns Y_ijk where i,j,k index n_modes zero-modes.
pub fn yukawa_tensor(
    points: &NdArray<f64>,
    centers: &NdArray<f64>,
) -> Vec<f64> {
    let n_points = points.shape()[0];
    let n_modes = centers.shape()[0];
    let dim = points.shape()[1].min(centers.shape()[1]);
    let pts = points.data();
    let cts = centers.data();
    let centers_stride = centers.shape()[1];
    let tensor_size = n_modes * n_modes * n_modes;

    // Parallel per-point reduction: each thread builds a partial tensor for
    // its slice of points, then we reduce by element-wise summation.
    // Use with_min_len to give each thread a meaningful work-chunk before
    // the reduction overhead dominates. Empirically, chunks of >= 64 points
    // amortise the per-thread tensor allocation well.
    let mut y: Vec<f64> = (0..n_points)
        .into_par_iter()
        .with_min_len(64)
        .fold(
            || vec![0.0; tensor_size],
            |mut acc, p| {
                let pt = &pts[p * 8..p * 8 + dim];
                let mut phi = vec![0.0; n_modes];
                for i in 0..n_modes {
                    let c = &cts[i * centers_stride..(i + 1) * centers_stride];
                    let mut r2 = 0.0;
                    for d in 0..dim {
                        let diff = pt[d] - c[d];
                        r2 += diff * diff;
                    }
                    phi[i] = (-0.5 * r2).exp();
                }
                for i in 0..n_modes {
                    for j in 0..n_modes {
                        let pi_pj = phi[i] * phi[j];
                        let base = i * n_modes * n_modes + j * n_modes;
                        for k in 0..n_modes {
                            acc[base + k] += pi_pj * phi[k];
                        }
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0.0; tensor_size],
            |mut a, b| {
                for k in 0..tensor_size {
                    a[k] += b[k];
                }
                a
            },
        );

    let inv_n = 1.0 / n_points as f64;
    for v in &mut y {
        *v *= inv_n;
    }
    y
}

/// Contract Yukawa tensor against a uniform Higgs-direction vector and
/// return the largest absolute eigenvalue of the resulting n_modes x n_modes
/// matrix M_ij = sum_k Y_ijk h_k. Uses simple power iteration.
pub fn dominant_eigenvalue(yukawa: &[f64], n_modes: usize, n_iter: usize) -> f64 {
    // Higgs direction: uniform 1/sqrt(n)
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

    // Power iteration for dominant eigenvalue
    let mut v = vec![1.0 / (n_modes as f64).sqrt(); n_modes];
    let mut lambda = 0.0;
    for _ in 0..n_iter {
        let mut mv = vec![0.0; n_modes];
        for i in 0..n_modes {
            for j in 0..n_modes {
                mv[i] += m[i * n_modes + j] * v[j];
            }
        }
        let norm: f64 = mv.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            break;
        }
        // Rayleigh quotient
        lambda = 0.0;
        for i in 0..n_modes {
            lambda += v[i] * mv[i];
        }
        for i in 0..n_modes {
            v[i] = mv[i] / norm;
        }
    }
    lambda.abs()
}
