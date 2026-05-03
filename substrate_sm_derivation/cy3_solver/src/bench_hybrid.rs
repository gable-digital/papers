//! Hybrid GPU+CPU pipeline benchmark.
//!
//! Sweeps over configurations:
//!   - GPU-only with 1 stream
//!   - GPU-only with 2 streams (Option C: pipeline parallelism)
//!   - GPU 2 streams + CPU worker (Option B+C combined)
//!   - CPU-only baseline
//!
//! Reports total wall-clock and throughput per configuration.

extern crate cy3_rust_solver;

use cy3_rust_solver::orchestrator::{run_hybrid_pipeline, summarise, HybridConfig};

fn main() {
    let problem_sizes = [
        (5000usize, 16usize, 12usize, "xlarge"),
        (10000usize, 16usize, 12usize, "xxlarge"),
    ];

    let n_tasks = 100;

    println!("{{");
    println!("  \"benchmark\": \"hybrid_gpu_cpu_orchestrator\",");
    println!("  \"n_tasks_per_run\": {n_tasks},");
    println!("  \"results\": [");

    let mut first = true;
    for (n_sample, n_modes, max_iter, label) in problem_sizes {
        let configs = [
            ("gpu_1stream",   1, 0, false),
            ("gpu_2streams",  2, 0, false),
            ("gpu_2s_cpu1",   2, 12, true),
            ("gpu_2s_cpu2",   2, 18, true),
            ("cpu_only_12t",  0, 12, true),
        ];

        for (cfg_label, n_gpu, n_cpu_threads, use_cpu) in configs {
            // Skip GPU-only when n_gpu=0
            if n_gpu == 0 && !use_cpu {
                continue;
            }

            let config = HybridConfig {
                n_sample,
                n_modes,
                max_iter,
                n_gpu_streams: n_gpu,
                n_cpu_threads,
                use_cpu_worker: use_cpu,
                donaldson_tol: 1e-3,
                eigenvalue_iters: 100,
            };

            // Warmup
            let _ = run_hybrid_pipeline(config, 5);

            // Timed
            let (results, total_ns) = run_hybrid_pipeline(config, n_tasks);

            let summary = summarise(&results, total_ns);
            let throughput = results.len() as f64 / (total_ns as f64 / 1e9);
            let avg_per_task_ms = total_ns as f64 / 1e6 / results.len() as f64;

            // Per-processor counts
            let mut gpu_count = 0;
            let mut cpu_count = 0;
            for r in &results {
                match r.processor {
                    cy3_rust_solver::orchestrator::ProcessorKind::GpuStream(_) => gpu_count += 1,
                    cy3_rust_solver::orchestrator::ProcessorKind::Cpu => cpu_count += 1,
                }
            }

            let comma = if first { "" } else { "," };
            first = false;
            print!("{comma}");
            println!("    {{");
            println!("      \"problem\": \"{label}\",");
            println!("      \"config\": \"{cfg_label}\",");
            println!("      \"n_sample\": {n_sample},");
            println!("      \"n_modes\": {n_modes},");
            println!("      \"n_gpu_streams\": {n_gpu},");
            println!("      \"n_cpu_threads\": {n_cpu_threads},");
            println!("      \"use_cpu_worker\": {use_cpu},");
            println!("      \"n_tasks\": {n_tasks},");
            println!("      \"total_ns\": {total_ns},");
            println!("      \"throughput_per_sec\": {throughput:.2},");
            println!("      \"avg_per_task_ms\": {avg_per_task_ms:.3},");
            println!("      \"gpu_tasks\": {gpu_count},");
            println!("      \"cpu_tasks\": {cpu_count},");
            println!("      \"summary\": \"{}\"",
                     summary.replace('\n', " | ").replace('\"', "'"));
            print!("    }}");
        }
    }

    println!();
    println!("  ]");
    println!("}}");
}
