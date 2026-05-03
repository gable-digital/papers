//! GPU benchmark: runs the discrimination pipeline on the RTX 4090
//! (or any CUDA device) and reports timing for direct comparison
//! against the CPU pipeline.

extern crate cy3_rust_solver;

use std::time::Instant;

use cy3_rust_solver::gpu::{gpu_discriminate, GpuDiscriminationWorkspace};
use cy3_rust_solver::N_BASIS_DEGREE2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem_sizes: &[(usize, usize, usize, &str)] = &[
        (500, 16, 12, "medium"),
        (2000, 16, 12, "large"),
        (5000, 16, 12, "xlarge"),
        (10000, 16, 12, "xxlarge"),
        (20000, 16, 12, "xxxlarge"),
    ];

    println!("{{");
    println!("  \"benchmark\": \"gpu_discrimination\",");
    println!("  \"device\": \"CUDA (default device 0)\",");
    println!("  \"runs\": [");

    let mut first = true;
    for (n_sample, n_modes, max_iter, label) in problem_sizes {
        let comma = if first { "" } else { "," };
        first = false;

        let t_setup = Instant::now();
        let mut ws = GpuDiscriminationWorkspace::new(
            *n_sample,
            N_BASIS_DEGREE2,
            *n_modes,
            *max_iter,
        )?;
        let setup_ns = t_setup.elapsed().as_nanos() as u64;
        let device_bytes = ws.total_device_bytes();

        // Warmup
        let _ = gpu_discriminate(&mut ws, 42, 12345, 1e-3, 100)?;

        // Timed runs
        let n_timed = 5;
        let mut best_total = u64::MAX;
        let mut best_iters = 0usize;
        let mut best_lambda = 0.0f64;
        for run_idx in 0..n_timed {
            let t = Instant::now();
            let (iters, lambda) = gpu_discriminate(
                &mut ws,
                42 + run_idx as u64,
                12345,
                1e-3,
                100,
            )?;
            let elapsed_ns = t.elapsed().as_nanos() as u64;
            if elapsed_ns < best_total {
                best_total = elapsed_ns;
                best_iters = iters;
                best_lambda = lambda;
            }
        }

        print!("{comma}");
        println!("    {{");
        println!("      \"problem\": \"{label}\",");
        println!("      \"n_sample\": {n_sample},");
        println!("      \"n_modes\": {n_modes},");
        println!("      \"max_iter\": {max_iter},");
        println!("      \"setup_ns\": {setup_ns},");
        println!("      \"device_bytes\": {device_bytes},");
        println!("      \"best_total_ns\": {best_total},");
        println!("      \"donaldson_iters\": {best_iters},");
        println!("      \"yukawa_dominant_eigenvalue\": {best_lambda}");
        print!("    }}");
    }

    println!();
    println!("  ]");
    println!("}}");
    Ok(())
}
