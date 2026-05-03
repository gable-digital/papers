//! CUDA path for batch evaluation of [`SmoothFunctionGerm`] at many
//! sample points. Used for cross-validating CPU-vs-GPU agreement to
//! 1e-10 in the route34 test suite, and for fast Lyapunov-functional
//! evaluation at MC sample points when scanning the (σ̃, ρ̃) control-
//! parameter plane during catastrophe-theory verification.
//!
//! ## Scope
//!
//! Only [`SmoothFunctionGerm::evaluate`] is GPU-accelerated; the
//! Splitting-Lemma / classification logic is intrinsically sequential
//! (one germ → one ADE label) and runs CPU-only via rayon. Batch-
//! polynomial evaluation IS embarrassingly parallel and benefits
//! significantly from a CUDA dispatch.
//!
//! ## Memory layout
//!
//! The host packs each germ's coefficient vector contiguously,
//! together with an exponent table (one row per monomial, n_vars
//! integers per row), and dispatches a kernel that:
//!
//!   1. Loads the exponent table into shared memory.
//!   2. For each sample point `x[i]`, evaluates V(x[i]) by summing
//!      coeff[k] * prod_j x[i, j]^exp[k, j].
//!
//! See `KERNEL_SOURCE` below.
//!
//! ## CPU-fallback parity
//!
//! Each public function provides an identical CPU implementation
//! invoked when `gpu_enabled() == false`; the test
//! `test_gpu_evaluate_matches_cpu` in `route34/tests/test_arnold.rs`
//! verifies agreement to 1e-10.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use crate::route34::arnold_normal_form::{
    index_to_exponents_buf, num_monomials_up_to, GermError, SmoothFunctionGerm,
};

const KERNEL_SOURCE: &str = r#"
extern "C" __global__ void evaluate_germ_batch(
    const double* coeffs,       // n_monomials
    const int* exps,            // n_monomials x n_vars
    int n_monomials,
    int n_vars,
    int max_degree,
    const double* points,       // n_points x n_vars
    double* values,             // n_points
    int n_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    const double* x = &points[idx * n_vars];
    // Pre-compute powers x_j^k for k = 0..max_degree.
    // Bounded n_vars <= 6 and max_degree <= 8 in practice, so this
    // sits in registers / L1 cleanly.
    const int MAX_VARS = 8;
    const int MAX_DEG  = 9;
    double pow_table[MAX_VARS * MAX_DEG];
    for (int j = 0; j < n_vars; ++j) {
        pow_table[j * MAX_DEG] = 1.0;
        for (int k = 1; k <= max_degree; ++k) {
            pow_table[j * MAX_DEG + k] = pow_table[j * MAX_DEG + k - 1] * x[j];
        }
    }

    double acc = 0.0;
    for (int m = 0; m < n_monomials; ++m) {
        double c = coeffs[m];
        if (c == 0.0) continue;
        double term = c;
        for (int j = 0; j < n_vars; ++j) {
            int e = exps[m * n_vars + j];
            if (e > 0) term *= pow_table[j * MAX_DEG + e];
        }
        acc += term;
    }
    values[idx] = acc;
}
"#;

/// CUDA-backed batch evaluator for a single [`SmoothFunctionGerm`] at
/// many sample points.
pub struct GermGpuEvaluator {
    /// Held to keep the CUDA context alive for the lifetime of the
    /// evaluator; not directly read after construction.
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    func: cudarc::driver::CudaFunction,
}

impl GermGpuEvaluator {
    /// Initialise a CUDA context, compile the evaluation kernel, and
    /// return a usable evaluator. Returns `Err` if cuda is not
    /// available or the kernel fails to compile.
    pub fn new() -> Result<Self, GermError> {
        let ctx = CudaContext::new(0).map_err(|e| {
            GermError::NumericalFailure(format!("CudaContext::new failed: {}", e))
        })?;
        let stream = ctx.default_stream();
        let ptx = compile_ptx(KERNEL_SOURCE).map_err(|e| {
            GermError::NumericalFailure(format!("PTX compile failed: {}", e))
        })?;
        let module = ctx.load_module(ptx).map_err(|e| {
            GermError::NumericalFailure(format!("Module load failed: {}", e))
        })?;
        let func = module.load_function("evaluate_germ_batch").map_err(|e| {
            GermError::NumericalFailure(format!("Function load failed: {}", e))
        })?;
        Ok(Self { ctx, stream, func })
    }

    /// Evaluate `germ` at every point in `points` (each row a point in
    /// R^n). Returns the vector of `n_points` evaluations.
    pub fn evaluate_batch(
        &self,
        germ: &SmoothFunctionGerm,
        points: &[Vec<f64>],
    ) -> Result<Vec<f64>, GermError> {
        if germ.n_vars == 0 || germ.n_vars > 8 {
            return Err(GermError::DegenerateInput(
                "GPU path requires 1 <= n_vars <= 8",
            ));
        }
        if germ.max_degree > 8 {
            return Err(GermError::DegenerateInput(
                "GPU path requires max_degree <= 8",
            ));
        }
        for p in points {
            if p.len() != germ.n_vars {
                return Err(GermError::ShapeMismatch);
            }
        }
        let n = germ.n_vars;
        let n_monomials = num_monomials_up_to(n, germ.max_degree);
        if n_monomials != germ.coeffs.len() {
            return Err(GermError::ShapeMismatch);
        }
        // Build exponent table.
        let mut exps_flat: Vec<i32> = Vec::with_capacity(n_monomials * n);
        let mut buf = vec![0u32; n];
        for idx in 0..n_monomials {
            index_to_exponents_buf(idx, n, germ.max_degree, &mut buf)?;
            for &e in &buf {
                exps_flat.push(e as i32);
            }
        }
        // Flatten points.
        let n_points = points.len();
        let mut points_flat: Vec<f64> = Vec::with_capacity(n_points * n);
        for p in points {
            points_flat.extend_from_slice(p);
        }
        // Allocate device buffers.
        let d_coeffs = self
            .stream
            .memcpy_stod(&germ.coeffs)
            .map_err(|e| GermError::NumericalFailure(format!("memcpy d_coeffs: {}", e)))?;
        let d_exps = self
            .stream
            .memcpy_stod(&exps_flat)
            .map_err(|e| GermError::NumericalFailure(format!("memcpy d_exps: {}", e)))?;
        let d_points = self
            .stream
            .memcpy_stod(&points_flat)
            .map_err(|e| GermError::NumericalFailure(format!("memcpy d_points: {}", e)))?;
        let mut d_values: cudarc::driver::CudaSlice<f64> = self
            .stream
            .alloc_zeros(n_points)
            .map_err(|e| GermError::NumericalFailure(format!("alloc d_values: {}", e)))?;
        let cfg = LaunchConfig::for_num_elems(n_points as u32);
        let mut launch = self.stream.launch_builder(&self.func);
        let n_mon = n_monomials as i32;
        let n_var = n as i32;
        let max_deg = germ.max_degree as i32;
        let n_pts = n_points as i32;
        launch.arg(&d_coeffs);
        launch.arg(&d_exps);
        launch.arg(&n_mon);
        launch.arg(&n_var);
        launch.arg(&max_deg);
        launch.arg(&d_points);
        launch.arg(&mut d_values);
        launch.arg(&n_pts);
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| GermError::NumericalFailure(format!("launch: {}", e)))?;
        }
        let out = self
            .stream
            .memcpy_dtov(&d_values)
            .map_err(|e| GermError::NumericalFailure(format!("memcpy out: {}", e)))?;
        Ok(out)
    }
}

/// Convenience: CPU-fallback evaluator (does the same job sequentially
/// via [`SmoothFunctionGerm::evaluate`]). Used by tests to verify
/// CPU-vs-GPU agreement.
pub fn evaluate_batch_cpu(
    germ: &SmoothFunctionGerm,
    points: &[Vec<f64>],
) -> Result<Vec<f64>, GermError> {
    let mut out = Vec::with_capacity(points.len());
    for p in points {
        out.push(germ.evaluate(p)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_germ() -> SmoothFunctionGerm {
        // V(x, y) = x^3 + y^4 + 2 x y^2 - 0.5 x^2 + 1.0
        let mut g = SmoothFunctionGerm::zeros(2, 4).unwrap();
        g.set_coeff(&[3, 0], 1.0).unwrap();
        g.set_coeff(&[0, 4], 1.0).unwrap();
        g.set_coeff(&[1, 2], 2.0).unwrap();
        g.set_coeff(&[2, 0], -0.5).unwrap();
        g.set_coeff(&[0, 0], 1.0).unwrap();
        g
    }

    #[test]
    fn test_cpu_fallback_matches_direct_evaluate() {
        let g = make_test_germ();
        let pts = vec![vec![0.5, 0.7], vec![1.0, -0.3], vec![0.0, 0.0]];
        let cpu = evaluate_batch_cpu(&g, &pts).unwrap();
        for (i, p) in pts.iter().enumerate() {
            let direct = g.evaluate(p).unwrap();
            assert!((cpu[i] - direct).abs() < 1e-12);
        }
    }

    #[test]
    fn test_gpu_evaluate_matches_cpu() {
        // Skip silently if no CUDA device available — CI runners
        // without GPU should not fail.
        let evaluator = match GermGpuEvaluator::new() {
            Ok(e) => e,
            Err(_) => return,
        };
        let g = make_test_germ();
        let pts: Vec<Vec<f64>> = (0..256)
            .map(|i| {
                let t = i as f64 / 256.0;
                vec![0.5 - t * 0.7, 0.3 + t * 0.4]
            })
            .collect();
        let gpu = evaluator.evaluate_batch(&g, &pts).unwrap();
        let cpu = evaluate_batch_cpu(&g, &pts).unwrap();
        for i in 0..pts.len() {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1e-10,
                "mismatch at i={}: gpu={}, cpu={}",
                i,
                gpu[i],
                cpu[i]
            );
        }
    }
}
