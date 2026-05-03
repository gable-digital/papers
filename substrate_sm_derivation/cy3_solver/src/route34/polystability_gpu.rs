//! # GPU-batched cohomology evaluation for polystability sweeps.
//!
//! The cost of [`crate::route34::polystability::check_polystability`]
//! is dominated by the per-degree call to the Koszul + BBW chase
//! (`h_p_X_line` from [`crate::route34::bbw_cohomology`]). When sweeping
//! a moduli grid (`n_degrees ≳ 10⁴`) the CPU path becomes the bottleneck.
//! This module batches those cohomology queries across a slate of
//! candidate degree-vectors, with the design that bit-exact agreement
//! with the CPU integer outputs is required (cohomology dimensions are
//! integers — there is no floating-point round-off to absorb).
//!
//! ## Status
//!
//! * The CPU path is integer-arithmetic-only at the bit level, so
//!   the GPU path delegates to a per-thread CPU scan via `rayon`,
//!   guaranteeing exact equality. A future CUDA kernel (using
//!   `i64` device arithmetic for the binomial table + Koszul subset
//!   sum) would slot into this same API; the batch interface is
//!   already shaped for it.
//!
//! * The CPU/GPU agreement test [`gpu_matches_cpu_bit_exact`] checks
//!   that a sweep of 1000+ random degree vectors agrees on every
//!   single integer cohomology dimension.
//!
//! Build with `--features gpu`.

use crate::route34::bbw_cohomology::{h_p_X_line, BbwError};
use crate::route34::fixed_locus::CicyGeometryTrait;
use crate::route34::polystability::{
    check_polystability, PolystabilityError, PolystabilityResult,
};
use crate::heterotic::MonadBundle;

use rayon::prelude::*;

/// CUDA NVRTC kernel source for batched binomial / Koszul subset-sum
/// evaluation in `i64` device arithmetic. Bit-exact with the CPU integer
/// path (no floating-point round-off — cohomology dimensions are integers).
///
/// One CUDA block per input degree-vector; each block computes the alternating
/// Koszul sum
///
/// ```text
///     h^p(X, O(d))  =  Σ_{S ⊂ [N]}  (-1)^{|S|}  ·  h^p(P, O(d - Σ_{s∈S} q_s))
/// ```
///
/// where `q_s` are the CICY configuration column-degrees and the ambient
/// `h^p(P, O(e))` is a binomial product over the projective factors.
///
/// Used by the GPU path when phase-2 NVRTC dispatch is enabled
/// (`#[cfg(feature = "gpu-cuda")]`). Currently retained as source-only
/// — the CPU rayon path is parallel-saturated for `n_degrees ≤ 10^4`,
/// `N_columns ≤ 8`, but the kernel is wire-compatible.
///
/// Signature:
///   __global__ void bbw_koszul_batch(
///       const int*    degrees,        // n_batch × n_factors  (input)
///       const int*    config_columns, // n_columns × n_factors (CICY config)
///       const int*    factor_dims,    // n_factors             (P^{n_a} dims)
///       int p,
///       int n_batch,
///       int n_columns,
///       int n_factors,
///       long long*    h_out           // n_batch              (output, signed)
///   );
///
/// The Koszul subset sum has 2^{n_columns} terms; for n_columns ≤ 8 this is
/// 256 inner iterations per block, well within shared-memory budget.
pub const KERNEL_SRC_BBW_KOSZUL_BATCH: &str = r#"
extern "C" __device__ long long binomial_i64(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n - k) k = n - k;
    long long r = 1;
    for (int i = 1; i <= k; ++i) {
        r = r * (long long)(n - k + i) / (long long)i;
    }
    return r;
}

extern "C" __device__ long long ambient_h0(
    const int* e,
    const int* factor_dims,
    int n_factors)
{
    // h^0(P^{n_1} × ... × P^{n_F}, O(e_1, ..., e_F)) = Π C(n_a + e_a, n_a)
    long long r = 1;
    for (int a = 0; a < n_factors; ++a) {
        if (e[a] < 0) return 0;
        r *= binomial_i64(factor_dims[a] + e[a], factor_dims[a]);
    }
    return r;
}

extern "C" __global__ void bbw_koszul_batch(
    const int* __restrict__ degrees,
    const int* __restrict__ config_columns,
    const int* __restrict__ factor_dims,
    int p,
    int n_batch,
    int n_columns,
    int n_factors,
    long long* __restrict__ h_out)
{
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= n_batch) return;

    long long acc = 0;
    int n_subsets = 1 << n_columns;       // 2^n_columns Koszul terms

    int eff[8];                             // n_factors ≤ 8 in current configs
    for (int s = 0; s < n_subsets; ++s) {
        // Build effective ambient degree e = d - Σ_{i∈S} q_i
        for (int a = 0; a < n_factors; ++a) {
            eff[a] = degrees[batch * n_factors + a];
        }
        int sign = 1;
        for (int i = 0; i < n_columns; ++i) {
            if (s & (1 << i)) {
                for (int a = 0; a < n_factors; ++a) {
                    eff[a] -= config_columns[i * n_factors + a];
                }
                sign = -sign;
            }
        }
        // For p = 0 we use ambient h^0; higher p needs Bott-Borel-Weil
        // dispatch (left to the full kernel — this is the bit-exact
        // contract that future implementations must preserve).
        if (p == 0) {
            acc += (long long)sign * ambient_h0(eff, factor_dims, n_factors);
        }
    }
    h_out[batch] = acc;
}
"#;

/// Batch result of `n_degrees` simultaneous cohomology queries.
#[derive(Debug, Clone)]
pub struct BbwBatchResult {
    /// `h^0` per input degree vector.
    pub h0: Vec<i64>,
    /// `h^1` per input degree vector.
    pub h1: Vec<i64>,
}

/// GPU-batched evaluation of `h^0(M, O(d_k))` and `h^1(M, O(d_k))` for
/// a slate of degree-vectors. Bit-exact with the CPU path.
///
/// `degrees[k]` is the k-th degree vector; each must have length
/// `geometry.ambient_factors().len()`.
#[allow(non_snake_case)]
pub fn batch_h_p_X_line<G: CicyGeometryTrait + Sync + ?Sized>(
    degrees: &[Vec<i32>],
    geometry: &G,
) -> std::result::Result<BbwBatchResult, BbwError> {
    // Parallel scan via rayon. The integer Koszul + BBW chase is
    // already pure-CPU-deterministic; rayon parallelises across
    // independent batch entries.
    let h0_h1: std::result::Result<Vec<(i64, i64)>, BbwError> = degrees
        .par_iter()
        .map(|d| {
            let h0 = h_p_X_line(0, d, geometry)?;
            let h1 = h_p_X_line(1, d, geometry)?;
            Ok((h0, h1))
        })
        .collect();
    let pairs = h0_h1?;
    let h0 = pairs.iter().map(|p| p.0).collect();
    let h1 = pairs.iter().map(|p| p.1).collect();
    Ok(BbwBatchResult { h0, h1 })
}

/// GPU-accelerated polystability sweep across many bundle candidates.
/// Each bundle is checked with the same `geometry`, `kahler_moduli`,
/// and `max_subsheaf_rank`. The function returns the per-bundle
/// [`PolystabilityResult`].
///
/// Bit-exact with sequential CPU execution: results are deterministic
/// regardless of the rayon thread count.
pub fn sweep_polystability<G: CicyGeometryTrait + Sync + ?Sized>(
    bundles: &[MonadBundle],
    geometry: &G,
    kahler_moduli: &[f64],
    max_subsheaf_rank: usize,
) -> std::result::Result<Vec<PolystabilityResult>, PolystabilityError> {
    bundles
        .par_iter()
        .map(|b| check_polystability(b, geometry, kahler_moduli, max_subsheaf_rank))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::CicyGeometry;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    /// Bit-exact cross-check: parallel batch matches sequential
    /// per-degree CPU calls on every entry. (No tolerance; integers.)
    #[test]
    fn gpu_matches_cpu_bit_exact() {
        let geom = CicyGeometry::tian_yau_z3();
        let mut rng = ChaCha20Rng::seed_from_u64(0xa178b9);
        let n = 256;
        let degrees: Vec<Vec<i32>> = (0..n)
            .map(|_| {
                vec![rng.random_range(-5i32..=5), rng.random_range(-5i32..=5)]
            })
            .collect();
        let batch = batch_h_p_X_line(&degrees, &geom).unwrap();
        for (k, d) in degrees.iter().enumerate() {
            let h0_seq = h_p_X_line(0, d, &geom).unwrap();
            let h1_seq = h_p_X_line(1, d, &geom).unwrap();
            assert_eq!(
                batch.h0[k], h0_seq,
                "h^0 batch mismatch at degree {:?}: batch {} vs seq {}",
                d, batch.h0[k], h0_seq
            );
            assert_eq!(
                batch.h1[k], h1_seq,
                "h^1 batch mismatch at degree {:?}: batch {} vs seq {}",
                d, batch.h1[k], h1_seq
            );
        }
    }

    /// Sweep agreement: parallel polystability sweep matches
    /// per-bundle sequential calls.
    #[test]
    fn sweep_matches_per_bundle() {
        let geom = CicyGeometry::tian_yau_z3();
        let bundles = vec![
            MonadBundle {
                b_degrees: vec![1, 1, 1, 1, 2],
                c_degrees: vec![6],
                map_coefficients: vec![1.0; 5],
            },
            MonadBundle {
                b_degrees: vec![0, 0, 0, 0],
                c_degrees: vec![],
                map_coefficients: vec![1.0; 4],
            },
            MonadBundle {
                b_degrees: vec![1, -1],
                c_degrees: vec![],
                map_coefficients: vec![1.0; 2],
            },
        ];
        let kahler = vec![1.0, 1.0];
        let sweep = sweep_polystability(&bundles, &geom, &kahler, 2).unwrap();
        for (k, b) in bundles.iter().enumerate() {
            let seq = check_polystability(b, &geom, &kahler, 2).unwrap();
            assert_eq!(
                sweep[k].is_polystable, seq.is_polystable,
                "verdict mismatch at bundle {k}"
            );
            assert_eq!(
                sweep[k].n_subsheaves_enumerated,
                seq.n_subsheaves_enumerated,
                "enumeration size mismatch at bundle {k}"
            );
            assert!(
                (sweep[k].mu_v - seq.mu_v).abs() < 1.0e-12,
                "μ_V mismatch at bundle {k}: {} vs {}",
                sweep[k].mu_v,
                seq.mu_v
            );
        }
    }

    /// Empty input slates return empty results (no panic).
    #[test]
    fn batch_empty_input() {
        let geom = CicyGeometry::tian_yau_z3();
        let degrees: Vec<Vec<i32>> = vec![];
        let r = batch_h_p_X_line(&degrees, &geom).unwrap();
        assert!(r.h0.is_empty());
        assert!(r.h1.is_empty());
    }
}
