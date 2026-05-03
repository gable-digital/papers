//! Benchmark Yukawa overlap-tensor + dominant-eigenvalue extraction.

extern crate cy3_rust_solver;
extern crate pwos_math;

use std::time::Instant;

use cy3_rust_solver::{
    dominant_eigenvalue, sample_points, yukawa_tensor, LCG,
};
use pwos_math::ndarray::NdArray;

fn random_centers(n_modes: usize, dim: usize, seed: u64) -> NdArray<f64> {
    let mut rng = LCG::new(seed);
    let mut data = vec![0.0; n_modes * dim];
    for v in &mut data {
        *v = rng.next_normal() * 0.5;
    }
    NdArray::from_vec(&[n_modes, dim], data).unwrap()
}

fn main() {
    let problem_sizes: &[(usize, usize, u32)] = &[
        // (n_sample, n_modes, label-id)
        (30, 5, 0),
        (50, 9, 1),
        (200, 9, 2),
        (500, 12, 3),
        (1000, 16, 4),
    ];

    println!("{{");
    println!("  \"benchmark\": \"yukawa_overlap_and_eigenvalue\",");
    println!("  \"runs\": [");

    for (idx, (n_sample, n_modes, _label)) in problem_sizes.iter().enumerate() {
        let comma = if idx + 1 < problem_sizes.len() { "," } else { "" };

        let points = sample_points(*n_sample, 42);
        let centers = random_centers(*n_modes, 8, 12345);

        let t_y = Instant::now();
        let y = yukawa_tensor(&points, &centers);
        let y_ns = t_y.elapsed().as_nanos() as u64;

        let t_e = Instant::now();
        let lambda_max = dominant_eigenvalue(&y, *n_modes, 100);
        let e_ns = t_e.elapsed().as_nanos() as u64;

        let total_ns = y_ns + e_ns;
        let frob: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();

        println!("    {{");
        println!("      \"n_sample\": {n_sample},");
        println!("      \"n_modes\": {n_modes},");
        println!("      \"yukawa_tensor_ns\": {y_ns},");
        println!("      \"eigenvalue_ns\": {e_ns},");
        println!("      \"total_ns\": {total_ns},");
        println!("      \"yukawa_frobenius_norm\": {frob},");
        println!("      \"dominant_eigenvalue\": {lambda_max}");
        println!("    }}{comma}");
    }

    println!("  ]");
    println!("}}");
}
