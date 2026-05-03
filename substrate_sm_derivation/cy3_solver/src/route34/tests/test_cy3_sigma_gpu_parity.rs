//! P7.10 — CPU↔GPU parity test for the σ-evaluator at NCOORDS=8.
//!
//! Builds a small synthetic CY3 σ-eval input (random points, random
//! sections, random Hermitian H), evaluates σ on both CPU and GPU,
//! and asserts agreement to 1e-10 relative.
//!
//! The synthetic input does not require the full Schoen/TY samplers;
//! we hand-craft section_values, section_derivs, and chart frames so
//! that:
//!   1. K = s†Hs is positive (H is SPD).
//!   2. The chart frame `T` is a non-degenerate 3×8 complex matrix
//!      such that the projected g_tan is non-singular.
//!   3. Per-point η is finite and positive at every sample point.
//!
//! Two test functions cover:
//!   * Schoen-style tangent frame (block-diagonal-ish) at seed=42, k=3.
//!   * TY-style tangent frame (dense Wilson-line mixed) at seed=42, k=3.
//!
//! Both must pass at 1e-10 relative before the production sweep
//! is run.

use crate::route34::cy3_sigma_gpu::{
    cpu_sigma_reference, NCOORDS, NFOLD,
};
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const FRAME_SIZE: usize = 2 * NFOLD * NCOORDS;

fn build_synthetic_inputs(
    seed: u64,
    n_pts: usize,
    n_basis: usize,
    schoen_style_frames: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let two_n = 2 * n_basis;
    let stride_sd = NCOORDS * two_n;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    let mut section_values = vec![0.0_f64; n_pts * two_n];
    for v in section_values.iter_mut() {
        *v = rng.random::<f64>() * 2.0 - 1.0;
    }
    let mut section_derivs = vec![0.0_f64; n_pts * stride_sd];
    for v in section_derivs.iter_mut() {
        *v = rng.random::<f64>() * 0.5 - 0.25;
    }

    let mut frames = vec![0.0_f64; n_pts * FRAME_SIZE];
    for p in 0..n_pts {
        let block = &mut frames[p * FRAME_SIZE..(p + 1) * FRAME_SIZE];
        if schoen_style_frames {
            // Schoen-like: 3 free coords {0, 3, 6} carry unit entry,
            // with eliminated entries in the other 5 slots.
            let free = [0_usize, 3, 6];
            for (a, &fa) in free.iter().enumerate() {
                // T_a[fa] = 1
                block[2 * (a * NCOORDS + fa)] = 1.0;
                block[2 * (a * NCOORDS + fa) + 1] = 0.0;
                // 5 other (eliminated) slots get small complex entries
                for i in 0..NCOORDS {
                    if i == fa {
                        continue;
                    }
                    block[2 * (a * NCOORDS + i)] =
                        (rng.random::<f64>() * 2.0 - 1.0) * 0.3;
                    block[2 * (a * NCOORDS + i) + 1] =
                        (rng.random::<f64>() * 2.0 - 1.0) * 0.3;
                }
            }
        } else {
            // TY-style: dense complex frame.
            for v in block.iter_mut() {
                *v = (rng.random::<f64>() * 2.0 - 1.0) * 0.7;
            }
            // Then add rank-3 dominant on the diagonal-ish to ensure
            // the 3 rows are linearly independent.
            for a in 0..NFOLD {
                let i = a * 2; // 0, 2, 4
                block[2 * (a * NCOORDS + i)] += 1.5;
            }
        }
    }

    let weights: Vec<f64> = (0..n_pts)
        .map(|_| rng.random::<f64>() * 0.5 + 0.5)
        .collect();
    let omega_sq: Vec<f64> = (0..n_pts)
        .map(|_| rng.random::<f64>() * 1.5 + 0.5)
        .collect();

    // SPD H = A^T A + n_basis * I; this guarantees strictly positive
    // K = s†Hs at every point.
    let n_basis_sq = n_basis * n_basis;
    let mut a_re = vec![0.0_f64; n_basis_sq];
    let mut a_im = vec![0.0_f64; n_basis_sq];
    for v in a_re.iter_mut() {
        *v = rng.random::<f64>() * 2.0 - 1.0;
    }
    for v in a_im.iter_mut() {
        *v = rng.random::<f64>() * 2.0 - 1.0;
    }
    let mut h_re = vec![0.0_f64; n_basis_sq];
    let mut h_im = vec![0.0_f64; n_basis_sq];
    for i in 0..n_basis {
        for j in 0..n_basis {
            let mut sre = 0.0_f64;
            let mut sim = 0.0_f64;
            for k in 0..n_basis {
                // (A^H)_ki = conj(A)_ik = (A_re_ik - i A_im_ik); (A^H A)_ij = sum_k conj(A_ki) A_kj
                let ar = a_re[k * n_basis + i];
                let ai = -a_im[k * n_basis + i];
                let br = a_re[k * n_basis + j];
                let bi = a_im[k * n_basis + j];
                sre += ar * br - ai * bi;
                sim += ar * bi + ai * br;
            }
            h_re[i * n_basis + j] = sre;
            h_im[i * n_basis + j] = sim;
            if i == j {
                h_re[i * n_basis + j] += n_basis as f64;
            }
        }
    }
    // Hermitian symmetrise
    for i in 0..n_basis {
        for j in (i + 1)..n_basis {
            let avg_re = 0.5 * (h_re[i * n_basis + j] + h_re[j * n_basis + i]);
            let avg_im = 0.5 * (h_im[i * n_basis + j] - h_im[j * n_basis + i]);
            h_re[i * n_basis + j] = avg_re;
            h_re[j * n_basis + i] = avg_re;
            h_im[i * n_basis + j] = avg_im;
            h_im[j * n_basis + i] = -avg_im;
        }
        h_im[i * n_basis + i] = 0.0;
    }

    (section_values, section_derivs, frames, omega_sq, weights, h_re, h_im)
}

#[test]
fn cy3_sigma_cpu_self_consistent_seed42_k3_schoen() {
    // Sanity: CPU reference is deterministic and finite.
    let n_pts = 200;
    let n_basis = 12;
    let (s, d, fr, om, w, h_re, h_im) =
        build_synthetic_inputs(42, n_pts, n_basis, true);
    let sigma1 =
        cpu_sigma_reference(&s, &d, &fr, &om, &w, &h_re, &h_im, n_pts, n_basis);
    let sigma2 =
        cpu_sigma_reference(&s, &d, &fr, &om, &w, &h_re, &h_im, n_pts, n_basis);
    assert!(sigma1.is_finite(), "sigma1 not finite: {}", sigma1);
    assert!(sigma1 > 0.0, "sigma1 not positive: {}", sigma1);
    assert_eq!(sigma1, sigma2, "CPU σ not deterministic");
}

#[test]
#[cfg(feature = "gpu")]
#[ignore]
fn cy3_sigma_cpu_gpu_parity_seed42_k3_schoen_style() {
    use crate::route34::cy3_sigma_gpu::Cy3SigmaGpu;
    let n_pts = 1024;
    let n_basis = 12;
    let (s, d, fr, om, w, h_re, h_im) =
        build_synthetic_inputs(42, n_pts, n_basis, true);

    let sigma_cpu = cpu_sigma_reference(&s, &d, &fr, &om, &w, &h_re, &h_im, n_pts, n_basis);
    assert!(sigma_cpu.is_finite() && sigma_cpu > 0.0,
        "CPU σ not finite/positive: {}", sigma_cpu);

    let mut gpu = Cy3SigmaGpu::new(n_pts, n_basis).expect("Cy3SigmaGpu::new");
    gpu.upload_static(&s, &d, &fr, &om, &w).expect("upload_static");
    let sigma_gpu = gpu.compute_sigma(&h_re, &h_im).expect("gpu sigma");

    let rel = (sigma_cpu - sigma_gpu).abs() / sigma_cpu.abs().max(1e-30);
    println!(
        "[CPU↔GPU Schoen-style parity] σ_cpu={:.15e} σ_gpu={:.15e} rel_diff={:.3e}",
        sigma_cpu, sigma_gpu, rel
    );
    assert!(
        rel < 1.0e-10,
        "CPU/GPU σ parity violated (Schoen-style frames): cpu={} gpu={} rel={:.3e}",
        sigma_cpu,
        sigma_gpu,
        rel
    );
}

#[test]
#[cfg(feature = "gpu")]
#[ignore]
fn cy3_sigma_cpu_gpu_parity_seed42_k3_ty_style() {
    use crate::route34::cy3_sigma_gpu::Cy3SigmaGpu;
    let n_pts = 1024;
    let n_basis = 12;
    let (s, d, fr, om, w, h_re, h_im) =
        build_synthetic_inputs(42, n_pts, n_basis, false);

    let sigma_cpu = cpu_sigma_reference(&s, &d, &fr, &om, &w, &h_re, &h_im, n_pts, n_basis);
    assert!(sigma_cpu.is_finite() && sigma_cpu > 0.0,
        "CPU σ not finite/positive: {}", sigma_cpu);

    let mut gpu = Cy3SigmaGpu::new(n_pts, n_basis).expect("Cy3SigmaGpu::new");
    gpu.upload_static(&s, &d, &fr, &om, &w).expect("upload_static");
    let sigma_gpu = gpu.compute_sigma(&h_re, &h_im).expect("gpu sigma");

    let rel = (sigma_cpu - sigma_gpu).abs() / sigma_cpu.abs().max(1e-30);
    println!(
        "[CPU↔GPU TY-style parity] σ_cpu={:.15e} σ_gpu={:.15e} rel_diff={:.3e}",
        sigma_cpu, sigma_gpu, rel
    );
    assert!(
        rel < 1.0e-10,
        "CPU/GPU σ parity violated (TY-style frames): cpu={} gpu={} rel={:.3e}",
        sigma_cpu,
        sigma_gpu,
        rel
    );
}

#[test]
#[cfg(feature = "gpu")]
#[ignore]
fn cy3_sigma_cpu_gpu_parity_n_basis_70() {
    // Production-scale n_basis. This is the n_basis that k=4 reaches in
    // the orthogonal Z₃×Z₃ basis. Smaller n_pts (256) to keep the test
    // fast — parity should hold at any size.
    use crate::route34::cy3_sigma_gpu::Cy3SigmaGpu;
    let n_pts = 256;
    let n_basis = 70;
    let (s, d, fr, om, w, h_re, h_im) =
        build_synthetic_inputs(42, n_pts, n_basis, true);

    let sigma_cpu = cpu_sigma_reference(&s, &d, &fr, &om, &w, &h_re, &h_im, n_pts, n_basis);
    assert!(sigma_cpu.is_finite() && sigma_cpu > 0.0,
        "CPU σ not finite/positive: {}", sigma_cpu);

    let mut gpu = Cy3SigmaGpu::new(n_pts, n_basis).expect("Cy3SigmaGpu::new");
    gpu.upload_static(&s, &d, &fr, &om, &w).expect("upload_static");
    let sigma_gpu = gpu.compute_sigma(&h_re, &h_im).expect("gpu sigma");

    let rel = (sigma_cpu - sigma_gpu).abs() / sigma_cpu.abs().max(1e-30);
    println!(
        "[CPU↔GPU n_basis=70] σ_cpu={:.15e} σ_gpu={:.15e} rel_diff={:.3e}",
        sigma_cpu, sigma_gpu, rel
    );
    assert!(
        rel < 1.0e-10,
        "CPU/GPU σ parity violated at n_basis=70: cpu={} gpu={} rel={:.3e}",
        sigma_cpu,
        sigma_gpu,
        rel
    );
}

#[test]
#[cfg(feature = "gpu")]
#[ignore]
fn cy3_sigma_gpu_speedup_benchmark() {
    use crate::route34::cy3_sigma_gpu::Cy3SigmaGpu;
    use std::time::Instant;
    let n_pts = 25000;
    let n_basis = 70;
    println!("Building synthetic input: n_pts={}, n_basis={}...", n_pts, n_basis);
    let (s, d, fr, om, w, h_re, h_im) =
        build_synthetic_inputs(42, n_pts, n_basis, true);

    let t0 = Instant::now();
    let sigma_cpu = cpu_sigma_reference(&s, &d, &fr, &om, &w, &h_re, &h_im, n_pts, n_basis);
    let t_cpu = t0.elapsed().as_secs_f64();

    let mut gpu = Cy3SigmaGpu::new(n_pts, n_basis).expect("Cy3SigmaGpu::new");
    gpu.upload_static(&s, &d, &fr, &om, &w).expect("upload_static");
    // First call includes JIT warmup; throw away.
    let _ = gpu.compute_sigma(&h_re, &h_im).expect("warmup");

    let n_repeats = 10;
    let t1 = Instant::now();
    let mut sigma_gpu = 0.0;
    for _ in 0..n_repeats {
        sigma_gpu = gpu.compute_sigma(&h_re, &h_im).expect("gpu sigma");
    }
    let t_gpu_total = t1.elapsed().as_secs_f64();
    let t_gpu = t_gpu_total / n_repeats as f64;

    let speedup = t_cpu / t_gpu;
    let rel = (sigma_cpu - sigma_gpu).abs() / sigma_cpu.abs().max(1e-30);
    println!(
        "[GPU speedup] σ_cpu={:.10e} σ_gpu={:.10e} rel={:.3e} | CPU={:.3} s GPU={:.3} ms speedup={:.1}×",
        sigma_cpu, sigma_gpu, rel, t_cpu, t_gpu * 1000.0, speedup
    );
    assert!(rel < 1.0e-10, "Parity broke at production scale");
}
