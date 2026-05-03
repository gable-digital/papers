//! Benchmark Donaldson Ricci-flat metric balancing iteration.
//! Outputs JSON with timing for direct comparison to the Python pipeline.

extern crate cy3_rust_solver;
extern crate pwos_math;

use std::time::Instant;

use cy3_rust_solver::{donaldson_solve, evaluate_section_basis_realvalued, sample_points};

fn main() {
    let problem_sizes: &[(usize, usize, usize)] = &[
        // (n_sample, max_iter, label-id)
        (50, 8, 0),
        (200, 8, 1),
        (500, 8, 2),
        (1000, 12, 3),
    ];

    println!("{{");
    println!("  \"benchmark\": \"donaldson_balancing_iteration\",");
    println!("  \"runs\": [");

    for (idx, (n_sample, max_iter, _label)) in problem_sizes.iter().enumerate() {
        let comma = if idx + 1 < problem_sizes.len() { "," } else { "" };

        // Sample points (cheap, not benched)
        let points = sample_points(*n_sample, 42);

        // Build section-basis evaluation matrix (cheap-ish, but matters)
        let t_basis = Instant::now();
        let basis_values = evaluate_section_basis_realvalued(&points);
        let basis_ns = t_basis.elapsed().as_nanos() as u64;

        let n_basis = basis_values.shape()[1];

        // Donaldson iteration (the heavy work)
        let t_iter = Instant::now();
        let (h, residuals) = donaldson_solve(&basis_values, *max_iter, 1e-3);
        let iter_ns = t_iter.elapsed().as_nanos() as u64;

        let final_residual = residuals.last().copied().unwrap_or(f64::INFINITY);
        let n_iters_run = residuals.len();
        let h_trace: f64 = (0..n_basis).map(|a| h.data()[a * n_basis + a]).sum();

        println!("    {{");
        println!("      \"n_sample\": {n_sample},");
        println!("      \"max_iter\": {max_iter},");
        println!("      \"n_basis\": {n_basis},");
        println!("      \"basis_construction_ns\": {basis_ns},");
        println!("      \"donaldson_total_ns\": {iter_ns},");
        println!("      \"iterations_run\": {n_iters_run},");
        println!("      \"final_residual\": {final_residual},");
        println!("      \"h_trace_check\": {h_trace}");
        println!("    }}{comma}");
    }

    println!("  ]");
    println!("}}");
}
