//! Hybrid GPU+CPU discrimination orchestrator.
//!
//! Splits work across:
//!   - N GPU workers, each owning its own GpuDiscriminationWorkspace and
//!     CudaStream (Option C: pipeline parallelism — multiple GPU streams
//!     run concurrently on the device, overlapping kernel launches).
//!   - 1 CPU worker, owning a DiscriminationWorkspace and using rayon
//!     internally for per-pass parallelism (Option B: across-pass split).
//!
//! Tasks are dispatched through a crossbeam-channel work queue; results
//! come back through a results channel. The orchestrator records timing
//! and processor-attribution per result.

use std::time::Instant;

use crossbeam_channel::{bounded, Receiver, Sender};

use crate::kernels::discriminate_in_place;
use crate::workspace::DiscriminationWorkspace;
use crate::N_BASIS_DEGREE2;

#[cfg(feature = "gpu")]
use crate::gpu::{gpu_discriminate, GpuDiscriminationWorkspace};

#[derive(Debug, Clone, Copy)]
pub struct DiscriminationTask {
    pub task_id: u64,
    pub sample_seed: u64,
    pub centers_seed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessorKind {
    GpuStream(u32),
    Cpu,
}

#[derive(Debug, Clone, Copy)]
pub struct DiscriminationResult {
    pub task_id: u64,
    pub iters: usize,
    pub lambda: f64,
    pub elapsed_ns: u64,
    pub processor: ProcessorKind,
}

#[derive(Debug, Clone, Copy)]
pub struct HybridConfig {
    pub n_sample: usize,
    pub n_modes: usize,
    pub max_iter: usize,
    pub n_gpu_streams: usize,
    pub n_cpu_threads: usize,
    pub use_cpu_worker: bool,
    pub donaldson_tol: f64,
    pub eigenvalue_iters: usize,
}

#[cfg(feature = "gpu")]
fn gpu_worker_loop(
    stream_id: u32,
    config: HybridConfig,
    task_rx: Receiver<DiscriminationTask>,
    result_tx: Sender<DiscriminationResult>,
) {
    let mut ws = GpuDiscriminationWorkspace::new(
        config.n_sample,
        N_BASIS_DEGREE2,
        config.n_modes,
        config.max_iter,
    )
    .expect("GPU workspace allocation failed");

    while let Ok(task) = task_rx.recv() {
        let t0 = Instant::now();
        match gpu_discriminate(
            &mut ws,
            task.sample_seed,
            task.centers_seed,
            config.donaldson_tol,
            config.eigenvalue_iters,
        ) {
            Ok((iters, lambda)) => {
                let elapsed = t0.elapsed().as_nanos() as u64;
                let _ = result_tx.send(DiscriminationResult {
                    task_id: task.task_id,
                    iters,
                    lambda,
                    elapsed_ns: elapsed,
                    processor: ProcessorKind::GpuStream(stream_id),
                });
            }
            Err(e) => {
                eprintln!(
                    "[gpu_worker stream={stream_id}] task {} failed: {}",
                    task.task_id, e
                );
            }
        }
    }
}

fn cpu_worker_loop(
    config: HybridConfig,
    pool: rayon::ThreadPool,
    task_rx: Receiver<DiscriminationTask>,
    result_tx: Sender<DiscriminationResult>,
) {
    let mut ws = DiscriminationWorkspace::new(
        config.n_sample,
        N_BASIS_DEGREE2,
        config.n_modes,
        config.max_iter,
        config.n_cpu_threads,
    );

    while let Ok(task) = task_rx.recv() {
        let t0 = Instant::now();
        let (iters, lambda) = pool.install(|| {
            discriminate_in_place(
                &mut ws,
                task.sample_seed,
                task.centers_seed,
                config.donaldson_tol,
                config.eigenvalue_iters,
            )
        });
        let elapsed = t0.elapsed().as_nanos() as u64;
        let _ = result_tx.send(DiscriminationResult {
            task_id: task.task_id,
            iters,
            lambda,
            elapsed_ns: elapsed,
            processor: ProcessorKind::Cpu,
        });
    }
}

#[cfg(feature = "gpu")]
pub fn run_hybrid_pipeline(
    config: HybridConfig,
    n_tasks: usize,
) -> (Vec<DiscriminationResult>, u64) {
    let (task_tx, task_rx) = bounded::<DiscriminationTask>(n_tasks.max(1));
    let (result_tx, result_rx) = bounded::<DiscriminationResult>(n_tasks.max(1));

    // Fill task queue
    for i in 0..n_tasks {
        task_tx
            .send(DiscriminationTask {
                task_id: i as u64,
                sample_seed: 42 + i as u64,
                centers_seed: 12345 + (i as u64),
            })
            .unwrap();
    }
    drop(task_tx); // close so workers can exit when queue empty

    let t_total = Instant::now();
    let mut handles = Vec::new();

    // Spawn GPU workers (Option C: each gets its own CUDA stream + workspace)
    for stream_id in 0..config.n_gpu_streams {
        let task_rx_clone = task_rx.clone();
        let result_tx_clone = result_tx.clone();
        let cfg = config;
        handles.push(std::thread::spawn(move || {
            gpu_worker_loop(stream_id as u32, cfg, task_rx_clone, result_tx_clone);
        }));
    }

    // Spawn CPU worker (Option B: across-pass split)
    if config.use_cpu_worker {
        let task_rx_clone = task_rx.clone();
        let result_tx_clone = result_tx.clone();
        let cfg = config;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.n_cpu_threads)
            .build()
            .expect("rayon pool build");
        handles.push(std::thread::spawn(move || {
            cpu_worker_loop(cfg, pool, task_rx_clone, result_tx_clone);
        }));
    }

    drop(task_rx);
    drop(result_tx);

    // Collect results
    let mut results: Vec<DiscriminationResult> = result_rx.iter().collect();
    let total_ns = t_total.elapsed().as_nanos() as u64;

    for h in handles {
        let _ = h.join();
    }

    results.sort_by_key(|r| r.task_id);
    (results, total_ns)
}

/// Summarise a result list: counts per processor, total throughput.
pub fn summarise(results: &[DiscriminationResult], total_ns: u64) -> String {
    let total_secs = total_ns as f64 / 1e9;
    let throughput = results.len() as f64 / total_secs.max(1e-9);

    let n_total = results.len();
    let mut gpu_counts = std::collections::HashMap::new();
    let mut cpu_count = 0;
    let mut gpu_time_sum = 0u64;
    let mut cpu_time_sum = 0u64;
    for r in results {
        match r.processor {
            ProcessorKind::GpuStream(id) => {
                *gpu_counts.entry(id).or_insert(0u64) += 1;
                gpu_time_sum += r.elapsed_ns;
            }
            ProcessorKind::Cpu => {
                cpu_count += 1;
                cpu_time_sum += r.elapsed_ns;
            }
        }
    }

    let mut lines = Vec::new();
    lines.push(format!(
        "Total tasks: {}, wall-clock: {:.2}s, throughput: {:.1} tasks/s",
        n_total, total_secs, throughput
    ));
    let mut gpu_total = 0u64;
    for (id, count) in &gpu_counts {
        gpu_total += count;
        lines.push(format!("  GPU stream {} processed {} tasks", id, count));
    }
    if cpu_count > 0 {
        lines.push(format!("  CPU processed {} tasks", cpu_count));
    }

    if gpu_total > 0 {
        let avg_gpu_per_task = gpu_time_sum as f64 / gpu_total as f64 / 1e6;
        lines.push(format!(
            "  Avg per-task GPU time: {:.2} ms",
            avg_gpu_per_task
        ));
    }
    if cpu_count > 0 {
        let avg_cpu_per_task = cpu_time_sum as f64 / cpu_count as f64 / 1e6;
        lines.push(format!(
            "  Avg per-task CPU time: {:.2} ms",
            avg_cpu_per_task
        ));
    }

    lines.join("\n")
}
