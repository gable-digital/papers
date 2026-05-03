//! P7.11 — CPU↔GPU parity for the Donaldson T-operator (NCOORDS=8,
//! Schoen + TY).
//!
//! Two layers of test:
//!
//! 1. Synthetic-input T-operator parity (`cpu_t_operator_reference`
//!    vs `Cy3DonaldsonGpu::t_operator_raw`). Hand-built section
//!    values + weights; SPD H. Bit-for-bit FP agreement isn't
//!    expected because the GPU uses tree reduction over points
//!    (different summation order); 1e-12 relative is the bar at
//!    n_pts ≤ 5000.
//!
//! 2. End-to-end Schoen / TY solver parity: `solve_schoen_metric` /
//!    `solve_ty_metric` with `use_gpu = true` vs `use_gpu = false`,
//!    same seed and sample budget, asserts agreement on
//!    `final_sigma_residual` to ≤ 1e-10 relative.

#[cfg(feature = "gpu")]
use crate::route34::cy3_donaldson_gpu::{cpu_t_operator_reference, Cy3DonaldsonGpu};
#[cfg(feature = "gpu")]
use rand::{Rng, SeedableRng};
#[cfg(feature = "gpu")]
use rand_chacha::ChaCha20Rng;

#[cfg(feature = "gpu")]
fn build_synthetic_donaldson_inputs(
    seed: u64,
    n_pts: usize,
    n_basis: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let two_n = 2 * n_basis;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    let mut section_values = vec![0.0_f64; n_pts * two_n];
    for v in section_values.iter_mut() {
        *v = rng.random::<f64>() * 2.0 - 1.0;
    }
    let weights: Vec<f64> = (0..n_pts)
        .map(|_| rng.random::<f64>() * 0.5 + 0.5)
        .collect();
    // SPD H = A^T A + n * I
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
    (section_values, weights, h_re, h_im)
}

#[test]
#[cfg(feature = "gpu")]
#[ignore]
fn cy3_donaldson_t_operator_cpu_gpu_parity_synthetic() {
    let n_pts = 2048;
    let n_basis = 16;
    let (s, w, h_re, h_im) = build_synthetic_donaldson_inputs(42, n_pts, n_basis);

    let (cpu_re, cpu_im) =
        cpu_t_operator_reference(&s, &w, &h_re, &h_im, n_pts, n_basis);

    let mut gpu = Cy3DonaldsonGpu::new(n_pts, n_basis).expect("Cy3DonaldsonGpu::new");
    gpu.upload_static(&s, &w).expect("upload_static");
    let (gpu_re, gpu_im) = gpu.t_operator_raw(&h_re, &h_im).expect("t_operator_raw");

    let n_basis_sq = n_basis * n_basis;
    let mut max_rel_diff = 0.0_f64;
    for i in 0..n_basis_sq {
        let dr = (cpu_re[i] - gpu_re[i]).abs();
        let di = (cpu_im[i] - gpu_im[i]).abs();
        let mag = (cpu_re[i].abs().max(cpu_im[i].abs())).max(1.0e-30);
        let rel = (dr.max(di)) / mag;
        if rel > max_rel_diff {
            max_rel_diff = rel;
        }
    }
    println!(
        "[T-operator CPU↔GPU synthetic n_pts={} n_basis={}] max_rel_diff = {:.3e}",
        n_pts, n_basis, max_rel_diff
    );
    assert!(
        max_rel_diff < 1.0e-12,
        "T-operator CPU/GPU parity violation: max rel = {:.3e}",
        max_rel_diff
    );
}

#[test]
#[cfg(feature = "gpu")]
#[ignore]
fn schoen_donaldson_cpu_gpu_parity_seed42_k3() {
    use crate::route34::schoen_metric::{solve_schoen_metric, SchoenMetricConfig};
    let base_cfg = || SchoenMetricConfig {
        d_x: 3,
        d_y: 3,
        d_t: 1,
        n_sample: 5000,
        max_iter: 30,
        donaldson_tol: 1.0e-3,
        seed: 42,
        checkpoint_path: None,
        apply_z3xz3_quotient: true,
        adam_refine: None,
        use_gpu: false,
        donaldson_damping: None,
        donaldson_tikhonov_shift: None,
    };
    let cpu = solve_schoen_metric(base_cfg()).expect("CPU solve");
    let mut gpu_cfg = base_cfg();
    gpu_cfg.use_gpu = true;
    let gpu = solve_schoen_metric(gpu_cfg).expect("GPU solve");

    let denom = cpu.final_sigma_residual.abs().max(1.0e-30);
    let rel = (cpu.final_sigma_residual - gpu.final_sigma_residual).abs() / denom;
    println!(
        "[Schoen Donaldson CPU↔GPU parity seed=42 k=3 n_pts=5000] σ_cpu={:.15e} σ_gpu={:.15e} rel={:.3e}",
        cpu.final_sigma_residual, gpu.final_sigma_residual, rel
    );
    assert_eq!(
        cpu.iterations_run, gpu.iterations_run,
        "Donaldson iter count differs: cpu={} gpu={}",
        cpu.iterations_run, gpu.iterations_run
    );
    assert!(
        rel < 1.0e-10,
        "Schoen Donaldson CPU/GPU parity violated: cpu={} gpu={} rel={:.3e}",
        cpu.final_sigma_residual,
        gpu.final_sigma_residual,
        rel
    );
}

#[test]
#[cfg(feature = "gpu")]
#[ignore]
fn ty_donaldson_cpu_gpu_parity_seed42_k3() {
    use crate::route34::ty_metric::{solve_ty_metric, TyMetricConfig};
    let base_cfg = || TyMetricConfig {
        k_degree: 3,
        n_sample: 5000,
        max_iter: 30,
        donaldson_tol: 1.0e-3,
        seed: 42,
        checkpoint_path: None,
        apply_z3_quotient: true,
        adam_refine: None,
        use_gpu: false,
        donaldson_damping: None,
        donaldson_tikhonov_shift: None,
    };
    let cpu = solve_ty_metric(base_cfg()).expect("CPU solve");
    let mut gpu_cfg = base_cfg();
    gpu_cfg.use_gpu = true;
    let gpu = solve_ty_metric(gpu_cfg).expect("GPU solve");

    let denom = cpu.final_sigma_residual.abs().max(1.0e-30);
    let rel = (cpu.final_sigma_residual - gpu.final_sigma_residual).abs() / denom;
    println!(
        "[TY Donaldson CPU↔GPU parity seed=42 k=3 n_pts=5000] σ_cpu={:.15e} σ_gpu={:.15e} rel={:.3e}",
        cpu.final_sigma_residual, gpu.final_sigma_residual, rel
    );
    assert_eq!(
        cpu.iterations_run, gpu.iterations_run,
        "Donaldson iter count differs: cpu={} gpu={}",
        cpu.iterations_run, gpu.iterations_run
    );
    assert!(
        rel < 1.0e-10,
        "TY Donaldson CPU/GPU parity violated: cpu={} gpu={} rel={:.3e}",
        cpu.final_sigma_residual,
        gpu.final_sigma_residual,
        rel
    );
}

#[test]
#[cfg(feature = "gpu")]
#[ignore]
fn cy3_donaldson_gpu_speedup_benchmark() {
    use std::time::Instant;
    let n_pts = 25000;
    let n_basis = 70;
    println!(
        "Building synthetic Donaldson input: n_pts={}, n_basis={}...",
        n_pts, n_basis
    );
    let (s, w, h_re, h_im) = build_synthetic_donaldson_inputs(42, n_pts, n_basis);

    let n_repeats = 5;
    // CPU
    let t0 = Instant::now();
    for _ in 0..n_repeats {
        let _ = cpu_t_operator_reference(&s, &w, &h_re, &h_im, n_pts, n_basis);
    }
    let t_cpu = t0.elapsed().as_secs_f64() / (n_repeats as f64);

    // GPU
    let mut gpu = Cy3DonaldsonGpu::new(n_pts, n_basis).expect("Cy3DonaldsonGpu::new");
    gpu.upload_static(&s, &w).expect("upload_static");
    // Warmup
    let _ = gpu.t_operator_raw(&h_re, &h_im).expect("warmup");
    let t1 = Instant::now();
    for _ in 0..n_repeats {
        let _ = gpu.t_operator_raw(&h_re, &h_im).expect("t_operator_raw");
    }
    let t_gpu = t1.elapsed().as_secs_f64() / (n_repeats as f64);

    let speedup = t_cpu / t_gpu;
    println!(
        "[Donaldson GPU benchmark] n_pts={} n_basis={}\n  CPU: {:.3} s/iter\n  GPU: {:.3} s/iter\n  Speedup: {:.1}×",
        n_pts, n_basis, t_cpu, t_gpu, speedup
    );
}
