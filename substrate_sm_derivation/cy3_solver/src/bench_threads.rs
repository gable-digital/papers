//! Thread-count sweep with pass-level pre-allocated workspace + custom
//! in-place GEMM/LU. Zero per-iteration heap allocations.

extern crate cy3_rust_solver;
extern crate rayon;

use std::time::Instant;

use cy3_rust_solver::{discriminate_in_place, DiscriminationWorkspace, N_BASIS_DEGREE2};

fn run_one_pass(ws: &mut DiscriminationWorkspace, sample_seed: u64) -> (u64, u64) {
    let t_total = Instant::now();
    let _ = discriminate_in_place(ws, sample_seed, 12345, 1e-3, 100);
    let total_ns = t_total.elapsed().as_nanos() as u64;
    (total_ns, ws.residuals.len() as u64)
}

fn main() {
    let problem_sizes: &[(usize, usize, usize, &str)] = &[
        (500, 16, 12, "medium"),
        (2000, 16, 12, "large"),
        (5000, 16, 12, "xlarge"),
    ];

    let max_threads: usize = std::env::var("MAX_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(36);

    let n_warmup = 1;
    let n_timed = 3;

    println!("{{");
    println!("  \"benchmark\": \"thread_count_sweep_v4_workspace\",");
    println!("  \"n_warmup_runs\": {n_warmup},");
    println!("  \"n_timed_runs\": {n_timed},");
    println!("  \"results\": [");

    let mut first = true;
    for (n_sample, n_modes, max_iter, label) in problem_sizes {
        for n_threads in [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36] {
            if n_threads > max_threads {
                break;
            }

            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .expect("rayon pool build");

            // Pre-allocate workspace once for this thread count
            let mut ws = DiscriminationWorkspace::new(
                *n_sample,
                N_BASIS_DEGREE2,
                *n_modes,
                *max_iter,
                n_threads,
            );
            let bytes = ws.total_bytes();

            // Warmup
            for _ in 0..n_warmup {
                pool.install(|| run_one_pass(&mut ws, 42));
            }

            // Timed runs
            let mut best_total = u64::MAX;
            let mut best_iters = 0u64;
            for run_idx in 0..n_timed {
                let (total_ns, iters) =
                    pool.install(|| run_one_pass(&mut ws, 42 + run_idx as u64));
                if total_ns < best_total {
                    best_total = total_ns;
                    best_iters = iters;
                }
            }

            let comma = if first { "" } else { "," };
            first = false;
            print!("{comma}");
            println!("    {{");
            println!("      \"problem\": \"{label}\",");
            println!("      \"n_sample\": {n_sample},");
            println!("      \"n_modes\": {n_modes},");
            println!("      \"max_iter\": {max_iter},");
            println!("      \"n_threads\": {n_threads},");
            println!("      \"workspace_bytes\": {bytes},");
            println!("      \"best_total_ns\": {best_total},");
            println!("      \"donaldson_iters\": {best_iters}");
            print!("    }}");
        }
    }

    println!();
    println!("  ]");
    println!("}}");
}
