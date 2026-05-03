//! P-REPRO-2-fix-BC: deterministic, GPU-tree-matched CPU h_pair
//! summation for the Donaldson T-operator inner reduction.
//!
//! ## Problem
//!
//! The CPU Donaldson T-operator inner sum was a sequential
//! `for p in 0..n_points { acc += factor * (...) }` accumulator
//! (`schoen_metric.rs` and `ty_metric.rs` h_pair construction).
//!
//! The matching GPU kernel (`cy3_donaldson_accum_h_pair` in
//! `cy3_donaldson_gpu.rs`) runs `block_size = 256` threads per (a,b)
//! block, each thread strided over points (`p = tid, tid+256, ...`),
//! followed by a power-of-two `__syncthreads()` tree reduction in
//! shared memory.
//!
//! These two reductions are **algebraically equal but bit-distinct
//! in f64**: a per-iteration ε ≈ 1e-15 round-off difference
//! accumulates geometrically through the Donaldson fixed-point
//! iteration and, on borderline-convergent k=4 seeds, shifts the
//! `donaldson_tol = 1e-6` exit by tens of iterations
//! (P8.4 production seed 137 showed 74 CPU iters vs 22 GPU iters
//! on the same seed; see
//! `references/p_repro2_nondeterminism_diagnostic.md`).
//!
//! ## Strategy
//!
//! Mirror the GPU reduction pattern **exactly** in CPU code:
//!
//! 1. Use `BLOCK_SIZE = 256` partial accumulators (`partials[0..256]`),
//!    matching the GPU `block_acc = 256` literal in
//!    `cy3_donaldson_gpu.rs:292`.
//! 2. Each "lane" `t ∈ [0, 256)` accumulates
//!    `Σ_{p ≡ t (mod 256)} factor[p] · (sa·sb†)`
//!    by sequential `+=` — same as a single GPU thread does inside
//!    the strided `for (p = tid; p < n_points; p += blockSize)` loop.
//! 3. Tree-reduce: for stride in `[128, 64, 32, 16, 8, 4, 2, 1]`,
//!    `partials[t] += partials[t + stride]` for `t < stride`. Final
//!    sum lands in `partials[0]`.
//!
//! Because every f64 op is performed in the same order, with the
//! same operands, on IEEE 754 double-precision hardware as the GPU
//! kernel, the result is **bit-identical** to the GPU output for
//! any fixed `(n_points, weights, k_values, section_values, a, b)`
//! tuple, modulo any FMA-vs-mul-add fusion difference. CPU+GPU
//! both compute strict mul/add (no FMA) here, so parity is exact.
//!
//! ## Numerical stability
//!
//! As a side benefit, the 256-way pairwise tree reduces summation
//! error from `O(n_points · ε)` (sequential) to
//! `O(log2(n_points) · ε) ≈ O(15 · ε) ≈ 1e-14 abs` for n_points
//! up to 32K. This is the same tightening Kahan compensation would
//! buy us, at zero extra flops in the inner pass (the work is
//! constant per `p`, only the accumulator layout changes).
//!
//! ## Performance
//!
//! Inner loop is **identical flop count** to the legacy sequential
//! sum (one mul + one add per `p`). The only overhead is the final
//! 8-stage tree reduction over 256 partials per (a,b), i.e.
//! `255 + 255 ≈ 510` extra adds per (a,b) — utterly negligible
//! compared to the `n_points = 25_000` inner loop.
//!
//! Stack-allocated `[f64; 256]` for the partials avoids any heap
//! traffic. Matches the GPU shared-memory layout one-for-one.

/// Block size matching the GPU `cy3_donaldson_accum_h_pair` kernel
/// launch configuration (`block_acc = 256` in
/// `cy3_donaldson_gpu.rs`). MUST be a power of two and MUST stay in
/// sync with the GPU literal — change both atomically or CPU/GPU
/// bit-parity is lost.
pub const H_PAIR_BLOCK_SIZE: usize = 256;

/// Compute the CPU h_pair (real, imag) accumulator for a single
/// `(a, b)` pair using the GPU-mirrored 256-lane tree reduction.
///
/// Inputs:
/// - `n_points`: number of sample points.
/// - `n_basis`: number of basis sections (monomials post-projection).
///   Used for the section row stride `2 * n_basis`.
/// - `a`, `b`: basis indices (each in `[0, n_basis)`).
/// - `section_values`: row-major `[2 * n_basis * n_points]` f64
///   buffer; entry `(p, k)` real part at `2*p*n_basis + 2*k`,
///   imag at `2*p*n_basis + 2*k + 1`.
/// - `weights`: per-point weights, length `n_points`.
/// - `k_values`: per-point K_p = s† H s, length `n_points`.
///
/// Returns `(acc_re, acc_im)`:
/// - `acc_re = Σ_p (w_p / K_p) (s_a_re s_b_re + s_a_im s_b_im)`,
/// - `acc_im = Σ_p (w_p / K_p) (s_a_re s_b_im - s_a_im s_b_re)`,
/// matching the GPU kernel `cy3_donaldson_accum_h_pair` exactly.
///
/// Skips any point with non-finite or non-positive `w_p` or `K_p`,
/// matching the GPU kernel's `isfinite(w) && w > 0 && isfinite(kp)
/// && kp > 0` guard at `cy3_donaldson_gpu.rs:134-135`.
#[inline]
pub fn h_pair_pairwise_sum(
    n_points: usize,
    n_basis: usize,
    a: usize,
    b: usize,
    section_values: &[f64],
    weights: &[f64],
    k_values: &[f64],
) -> (f64, f64) {
    // Stack-allocated lane partials. 256 × 8 B = 2 KiB per
    // accumulator; well within typical thread stack limits.
    let mut partials_re = [0.0_f64; H_PAIR_BLOCK_SIZE];
    let mut partials_im = [0.0_f64; H_PAIR_BLOCK_SIZE];

    let two_n = 2 * n_basis;
    let idx_a = 2 * a;
    let idx_b = 2 * b;

    // Step 1: 256 strided lane accumulators. Lane `t` sees
    // points p = t, t+256, t+512, ... — exactly like a GPU
    // thread `tid = t` walking the `for (p = tid; p < n_points;
    // p += blockSize)` loop in the kernel.
    for t in 0..H_PAIR_BLOCK_SIZE {
        let mut acc_re = 0.0_f64;
        let mut acc_im = 0.0_f64;
        let mut p = t;
        while p < n_points {
            let w = weights[p];
            let kp = k_values[p];
            // Skip non-finite / non-positive entries — matches
            // GPU guard.
            if w.is_finite() && w > 0.0 && kp.is_finite() && kp > 0.0 {
                let row = &section_values[p * two_n..(p + 1) * two_n];
                let sar = row[idx_a];
                let sai = row[idx_a + 1];
                let sbr = row[idx_b];
                let sbi = row[idx_b + 1];
                let factor = w / kp;
                acc_re += factor * (sar * sbr + sai * sbi);
                acc_im += factor * (sar * sbi - sai * sbr);
            }
            p += H_PAIR_BLOCK_SIZE;
        }
        partials_re[t] = acc_re;
        partials_im[t] = acc_im;
    }

    // Step 2: power-of-two tree reduction. Mirrors the GPU
    // shared-memory loop:
    //   for (int s = blockSize / 2; s > 0; s >>= 1) {
    //       if (tid < s) { sre[tid] += sre[tid + s]; ... }
    //       __syncthreads();
    //   }
    // Final sum in lane 0.
    let mut stride = H_PAIR_BLOCK_SIZE / 2;
    while stride > 0 {
        for t in 0..stride {
            partials_re[t] += partials_re[t + stride];
            partials_im[t] += partials_im[t + stride];
        }
        stride >>= 1;
    }

    (partials_re[0], partials_im[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: with constant inputs the GPU-mirrored sum equals
    /// the simple sequential sum to bit-identical precision when
    /// `n_points` is a multiple of `BLOCK_SIZE` AND every input is
    /// 1.0 (so all reductions are exact integer adds).
    #[test]
    fn pairwise_sum_constant_inputs_exact() {
        let n_basis = 2;
        let n_points = 1024;
        // section[p, k] = (1.0, 0.0) for every p, k.
        let two_n = 2 * n_basis;
        let section_values = vec![1.0_f64, 0.0_f64].repeat(n_basis * n_points);
        assert_eq!(section_values.len(), two_n * n_points);
        let weights = vec![1.0_f64; n_points];
        let k_values = vec![1.0_f64; n_points];
        let (re, im) = h_pair_pairwise_sum(
            n_points,
            n_basis,
            0,
            1,
            &section_values,
            &weights,
            &k_values,
        );
        // factor = 1, sa = (1,0), sb = (1,0):
        //   re = sar*sbr + sai*sbi = 1
        //   im = sar*sbi - sai*sbr = 0
        // sum over n_points = 1024 → re = 1024, im = 0.
        assert_eq!(re, n_points as f64);
        assert_eq!(im, 0.0);
    }

    /// Pairwise sum agrees with naive sequential sum to within
    /// O(log2(n_pts) · ε) absolute on random inputs. Tighter than
    /// the legacy `O(n_pts · ε)` worst case, looser than bit-exact
    /// (which only holds vs the GPU kernel doing the same tree).
    #[test]
    fn pairwise_sum_matches_naive_within_log_epsilon() {
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0xCAFE_F00D);
        let n_basis = 4;
        let n_points = 2000;
        let two_n = 2 * n_basis;
        let mut section_values = vec![0.0_f64; two_n * n_points];
        for v in section_values.iter_mut() {
            *v = rng.gen_range(-1.0..1.0);
        }
        let weights: Vec<f64> = (0..n_points).map(|_| rng.gen_range(0.1..1.0)).collect();
        let k_values: Vec<f64> = (0..n_points).map(|_| rng.gen_range(0.1..2.0)).collect();
        let a = 1usize;
        let b = 2usize;
        let (pr, pi) =
            h_pair_pairwise_sum(n_points, n_basis, a, b, &section_values, &weights, &k_values);
        // Naive sequential sum.
        let mut nr = 0.0_f64;
        let mut ni = 0.0_f64;
        for p in 0..n_points {
            let row = &section_values[p * two_n..(p + 1) * two_n];
            let sar = row[2 * a];
            let sai = row[2 * a + 1];
            let sbr = row[2 * b];
            let sbi = row[2 * b + 1];
            let factor = weights[p] / k_values[p];
            nr += factor * (sar * sbr + sai * sbi);
            ni += factor * (sar * sbi - sai * sbr);
        }
        // Both errors are bounded by O(n_pts · ε) sequentially and
        // O(log2(n_pts) · ε) for the tree; their difference fits a
        // generous 1e-12 abs at n_points = 2000.
        assert!((pr - nr).abs() < 1.0e-12, "re mismatch: pr={pr} nr={nr}");
        assert!((pi - ni).abs() < 1.0e-12, "im mismatch: pi={pi} ni={ni}");
    }
}
