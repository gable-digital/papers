//! Pass-level pre-allocated workspace. Holds every buffer needed for a
//! complete discrimination pass (sample -> basis -> Donaldson -> Yukawa
//! -> eigenvalue) with zero per-iteration allocations.

use rayon::prelude::*;

/// All buffers needed for one discrimination pass at given problem size.
/// Construct once, reuse across iterations.
pub struct DiscriminationWorkspace {
    pub n_points: usize,
    pub n_basis: usize,
    pub n_modes: usize,
    pub max_iter: usize,
    pub n_threads: usize,

    // Sample points
    pub points: Vec<f64>, // n_points * 8

    // Section-basis monomials (computed once)
    pub monomials: Vec<[u32; 8]>,

    // Section-basis evaluation
    pub section_values: Vec<f64>, // n_points * n_basis

    // Donaldson per-iteration buffers
    pub h: Vec<f64>,         // n_basis^2
    pub h_new: Vec<f64>,     // n_basis^2
    pub h_inv: Vec<f64>,     // n_basis^2
    pub h_lu: Vec<f64>,      // n_basis^2 (LU workspace)
    pub h_lu_perm: Vec<usize>, // n_basis
    pub h_lu_col: Vec<f64>,  // n_basis (per-column solve buffer)
    pub t_matrix: Vec<f64>,  // n_points * n_basis (S @ h_inv)
    pub sw_buffer: Vec<f64>, // n_points * n_basis
    pub sw_t_buffer: Vec<f64>, // n_basis * n_points (transposed)
    pub weights: Vec<f64>,   // n_points
    pub residuals: Vec<f64>, // max_iter

    // Yukawa buffers
    pub yukawa_centers: Vec<f64>,         // n_modes * 8
    pub yukawa_tensor: Vec<f64>,          // n_modes^3
    pub yukawa_thread_tensors: Vec<Vec<f64>>, // [n_threads][n_modes^3]

    // Mass-eigenvalue buffers
    pub m_matrix: Vec<f64>, // n_modes^2
    pub eig_v: Vec<f64>,    // n_modes
    pub eig_mv: Vec<f64>,   // n_modes
}

impl DiscriminationWorkspace {
    pub fn new(
        n_points: usize,
        n_basis: usize,
        n_modes: usize,
        max_iter: usize,
        n_threads: usize,
    ) -> Self {
        let monomials = build_degree2_monomials();
        debug_assert_eq!(monomials.len(), n_basis);

        Self {
            n_points,
            n_basis,
            n_modes,
            max_iter,
            n_threads,

            points: vec![0.0; n_points * 8],

            monomials,

            section_values: vec![0.0; n_points * n_basis],

            h: vec![0.0; n_basis * n_basis],
            h_new: vec![0.0; n_basis * n_basis],
            h_inv: vec![0.0; n_basis * n_basis],
            h_lu: vec![0.0; n_basis * n_basis],
            h_lu_perm: vec![0; n_basis],
            h_lu_col: vec![0.0; n_basis],
            t_matrix: vec![0.0; n_points * n_basis],
            sw_buffer: vec![0.0; n_points * n_basis],
            sw_t_buffer: vec![0.0; n_basis * n_points],
            weights: vec![0.0; n_points],
            residuals: Vec::with_capacity(max_iter),

            yukawa_centers: vec![0.0; n_modes * 8],
            yukawa_tensor: vec![0.0; n_modes * n_modes * n_modes],
            yukawa_thread_tensors: (0..n_threads)
                .map(|_| vec![0.0; n_modes * n_modes * n_modes])
                .collect(),

            m_matrix: vec![0.0; n_modes * n_modes],
            eig_v: vec![0.0; n_modes],
            eig_mv: vec![0.0; n_modes],
        }
    }

    /// Total bytes pre-allocated by this workspace.
    pub fn total_bytes(&self) -> usize {
        self.points.len() * 8
            + self.section_values.len() * 8
            + self.h.len() * 8
            + self.h_new.len() * 8
            + self.h_inv.len() * 8
            + self.h_lu.len() * 8
            + self.h_lu_perm.len() * std::mem::size_of::<usize>()
            + self.h_lu_col.len() * 8
            + self.t_matrix.len() * 8
            + self.sw_buffer.len() * 8
            + self.sw_t_buffer.len() * 8
            + self.weights.len() * 8
            + self.yukawa_centers.len() * 8
            + self.yukawa_tensor.len() * 8
            + self
                .yukawa_thread_tensors
                .iter()
                .map(|t| t.len() * 8)
                .sum::<usize>()
            + self.m_matrix.len() * 8
            + self.eig_v.len() * 8
            + self.eig_mv.len() * 8
    }

    /// Reset Yukawa thread-tensors to zero (called before each Yukawa run).
    pub fn reset_yukawa_thread_tensors(&mut self) {
        self.yukawa_thread_tensors.par_iter_mut().for_each(|t| {
            for v in t.iter_mut() {
                *v = 0.0;
            }
        });
        for v in self.yukawa_tensor.iter_mut() {
            *v = 0.0;
        }
    }

    /// Initialise h to identity (called at start of Donaldson solve).
    pub fn reset_h_to_identity(&mut self) {
        let n = self.n_basis;
        for v in self.h.iter_mut() {
            *v = 0.0;
        }
        for i in 0..n {
            self.h[i * n + i] = 1.0;
        }
        self.residuals.clear();
    }
}

/// Build the catalogue of degree-2 bigraded monomial exponent tuples for
/// CP^3 x CP^3 (8 coordinates total, degree 2 in each factor).
pub fn build_degree2_monomials() -> Vec<[u32; 8]> {
    let mut monomials = Vec::new();
    for a0 in 0..=2 {
        for a1 in 0..=(2 - a0) {
            for a2 in 0..=(2 - a0 - a1) {
                let a3 = 2 - a0 - a1 - a2;
                for b0 in 0..=2 {
                    for b1 in 0..=(2 - b0) {
                        for b2 in 0..=(2 - b0 - b1) {
                            let b3 = 2 - b0 - b1 - b2;
                            monomials.push([
                                a0 as u32, a1 as u32, a2 as u32, a3 as u32,
                                b0 as u32, b1 as u32, b2 as u32, b3 as u32,
                            ]);
                        }
                    }
                }
            }
        }
    }
    monomials
}

/// Number of degree-2 bigraded monomials = 100.
pub const N_BASIS_DEGREE2: usize = 100;
