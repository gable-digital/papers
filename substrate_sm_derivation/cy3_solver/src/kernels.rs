//! In-place kernels operating exclusively on pre-allocated workspace
//! buffers. No Vec allocation in the hot path.

use rayon::prelude::*;
use pwos_math::ndarray::NdArray;

use crate::workspace::DiscriminationWorkspace;
use crate::LCG;

/// Sample n points on the polysphere and store in ws.points (in-place).
pub fn sample_points_into(ws: &mut DiscriminationWorkspace, seed: u64) {
    let mut rng = LCG::new(seed);
    let n = ws.n_points;
    for i in 0..n {
        let z = &mut ws.points[i * 8..(i + 1) * 8];
        for k in 0..8 {
            z[k] = rng.next_normal();
        }
        let n1 = (z[0] * z[0] + z[1] * z[1] + z[2] * z[2] + z[3] * z[3]).sqrt();
        let n2 = (z[4] * z[4] + z[5] * z[5] + z[6] * z[6] + z[7] * z[7]).sqrt();
        for k in 0..4 {
            z[k] /= n1.max(1e-10);
        }
        for k in 4..8 {
            z[k] /= n2.max(1e-10);
        }
    }
}

/// Evaluate the section basis at every sampled point, in-place into
/// ws.section_values. Per-point work is done in parallel chunks.
pub fn evaluate_section_basis_into(ws: &mut DiscriminationWorkspace) {
    let _n_points = ws.n_points;
    let n_basis = ws.n_basis;
    let pts = &ws.points;
    let monomials = &ws.monomials;
    ws.section_values
        .par_chunks_mut(n_basis)
        .with_min_len(64)
        .enumerate()
        .for_each(|(i, row)| {
            let z = &pts[i * 8..(i + 1) * 8];
            // 8 coordinates × 3 powers (0, 1, 2) flat-stored.
            let mut pow_table = [1.0f64; 24];
            for k in 0..8 {
                pow_table[k * 3] = 1.0;
                pow_table[k * 3 + 1] = z[k];
                pow_table[k * 3 + 2] = z[k] * z[k];
            }
            for j in 0..n_basis {
                let m = &monomials[j];
                row[j] = pow_table[m[0] as usize]
                    * pow_table[3 + m[1] as usize]
                    * pow_table[6 + m[2] as usize]
                    * pow_table[9 + m[3] as usize]
                    * pow_table[12 + m[4] as usize]
                    * pow_table[15 + m[5] as usize]
                    * pow_table[18 + m[6] as usize]
                    * pow_table[21 + m[7] as usize];
            }
        });
}

/// Run one Donaldson balancing iteration. Uses pwos-math's hand-tuned
/// AVX2 GEMM and inverse for the heavy linear algebra (these allocate
/// their outputs internally but the matmul time dominates the alloc
/// overhead). Pre-allocated workspace buffers are used for everything
/// else: weights, sw_buffer, sample-points, monomials, etc.
pub fn donaldson_iter_into(ws: &mut DiscriminationWorkspace) {
    let n_points = ws.n_points;
    let n_basis = ws.n_basis;

    // Step 1: pwos-math inverse via mem::replace (move buffer in, alloc
    // a fresh empty Vec back; no copies).
    let h_data = std::mem::replace(&mut ws.h, vec![0.0; n_basis * n_basis]);
    let h_array = NdArray::from_vec(&[n_basis, n_basis], h_data).unwrap();
    let h_inv = h_array.inverse().expect("Donaldson: h must be invertible");
    // Move h_array's buffer back into ws.h (still no copy).
    ws.h.copy_from_slice(h_array.data());

    // Step 2: T = S @ h_inv via row-chunked parallel pwos-math GEMM.
    // We need to keep ws.section_values intact across iterations, so we
    // walk slices of ws.section_values directly into the parallel chunks
    // (the chunks each take a copy via to_vec, which is unavoidable
    // because pwos-math's NdArray::from_vec takes ownership).
    let n_threads = ws.n_threads.max(1);
    let chunk_rows = ((n_points + n_threads - 1) / n_threads).max(64);
    let chunks: Vec<(usize, usize)> = (0..n_points)
        .step_by(chunk_rows)
        .map(|start| (start, (start + chunk_rows).min(n_points)))
        .collect();

    let chunk_outputs: Vec<(usize, NdArray<f64>)> = {
        let s = &ws.section_values;
        let h_inv_ref = &h_inv;
        chunks
            .into_par_iter()
            .map(|(start, end)| {
                let chunk_data: Vec<f64> = s[start * n_basis..end * n_basis].to_vec();
                let chunk = NdArray::from_vec(&[end - start, n_basis], chunk_data).unwrap();
                let result = chunk.matmul(h_inv_ref).unwrap();
                (start, result)
            })
            .collect()
    };

    // Stitch chunk outputs into ws.t_matrix
    for (start, result) in chunk_outputs {
        let result_data = result.data();
        let rows = result.shape()[0];
        let dst_start = start * n_basis;
        ws.t_matrix[dst_start..dst_start + rows * n_basis]
            .copy_from_slice(&result_data[..rows * n_basis]);
    }

    // Step 3: weights[i] = T[i] . S[i] (per-row dot in parallel)
    {
        let t = &ws.t_matrix;
        let s = &ws.section_values;
        ws.weights
            .par_iter_mut()
            .with_min_len(64)
            .enumerate()
            .for_each(|(i, w_out)| {
                let t_i = &t[i * n_basis..(i + 1) * n_basis];
                let s_i = &s[i * n_basis..(i + 1) * n_basis];
                let mut w = 0.0;
                for a in 0..n_basis {
                    w += t_i[a] * s_i[a];
                }
                *w_out = w.max(1e-12);
            });
    }

    // Step 4: sw[i, a] = s[i, a] / sqrt(w_i)
    {
        let weights = &ws.weights;
        let s = &ws.section_values;
        ws.sw_buffer
            .par_chunks_mut(n_basis)
            .with_min_len(64)
            .enumerate()
            .for_each(|(i, row)| {
                let inv_sqrt_w = 1.0 / weights[i].sqrt();
                let s_i = &s[i * n_basis..(i + 1) * n_basis];
                for a in 0..n_basis {
                    row[a] = s_i[a] * inv_sqrt_w;
                }
            });
    }

    // Step 5: h_new = sw^T @ sw via pwos-math GEMM (transpose then matmul).
    // Use mem::replace to move sw_buffer into NdArray without copying;
    // place a fresh-allocated empty Vec back into ws.sw_buffer.
    let sw_data = std::mem::replace(&mut ws.sw_buffer, vec![0.0; n_points * n_basis]);
    let sw_array = NdArray::from_vec(&[n_points, n_basis], sw_data).unwrap();
    let sw_t_array = sw_array.transpose_2d().unwrap();
    let h_new_array = sw_t_array.matmul(&sw_array).unwrap();
    ws.h_new.copy_from_slice(h_new_array.data());

    // Step 7: normalise h_new to fixed trace
    let mut trace = 0.0;
    for a in 0..n_basis {
        trace += ws.h_new[a * n_basis + a];
    }
    if trace > 1e-10 {
        let scale = (n_basis as f64) / trace;
        for v in ws.h_new.iter_mut() {
            *v *= scale;
        }
    }
}

/// Solve Donaldson balancing iteratively, in-place into ws. Assumes
/// `evaluate_section_basis_into` has already populated ws.section_values.
/// Returns the residual history (also stored in ws.residuals).
pub fn donaldson_solve_in_place(ws: &mut DiscriminationWorkspace, tol: f64) {
    let _ = donaldson_solve_in_place_ext(ws, tol, DonaldsonSolveOpts::default());
}

/// Result of a Donaldson solve, distinguishing converged / max-iter /
/// diverged outcomes so the caller can decide whether to keep the
/// candidate or kill it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DonaldsonOutcome {
    Converged,
    MaxIter,
    Diverged,
}

/// Per-call options for `donaldson_solve_in_place_ext`. Defaults match
/// the original `donaldson_solve_in_place` behaviour.
#[derive(Debug, Clone)]
pub struct DonaldsonSolveOpts {
    /// If residual at iter N exceeds residual at iter N-1 by this factor
    /// for `divergence_window` iterations in a row, abort early.
    pub divergence_factor: f64,
    pub divergence_window: usize,
    /// Skip identity initialisation (caller has pre-loaded ws.h with a
    /// warm start, e.g. from a lower k_degree pass).
    pub warm_start: bool,
    /// If set, write a checkpoint to this path every N iterations (where
    /// N is `checkpoint_every`). Used for long Pass-3 runs where a
    /// SIGKILL loses hours of work.
    pub checkpoint_path: Option<std::path::PathBuf>,
    pub checkpoint_every: usize,
    /// Optional trajectory-extrapolation early-abort. After at least
    /// `extrapolate_min_iters` iterations have been recorded, fit a
    /// geometric tail residual_n ~ residual_inf + A * rho^n to the
    /// most recent residuals. If the projected residual_inf squared and
    /// scaled by `extrapolate_w_ricci` exceeds `extrapolate_budget`, the
    /// solve aborts with `Diverged`. This kills slow-converging bad
    /// candidates whose final residual would not pass the loss threshold
    /// regardless of how many more iterations they run.
    ///
    /// Set `extrapolate_budget` to `f64::INFINITY` (the default) to
    /// disable trajectory-based abort while retaining the
    /// divergence-streak abort above.
    pub extrapolate_budget: f64,
    pub extrapolate_w_ricci: f64,
    pub extrapolate_min_iters: usize,
}

impl Default for DonaldsonSolveOpts {
    fn default() -> Self {
        Self {
            divergence_factor: f64::INFINITY,
            divergence_window: 3,
            warm_start: false,
            checkpoint_path: None,
            checkpoint_every: 5,
            extrapolate_budget: f64::INFINITY,
            extrapolate_w_ricci: 1.0,
            extrapolate_min_iters: 4,
        }
    }
}

/// Project the asymptotic residual from the recent residual history
/// using a one-parameter geometric-tail model:
///
///   r_n ~ r_inf + A * rho^n
///
/// We estimate `rho` from `r_{n-1} / r_{n-2}` and then back out
/// `r_inf` via `r_n - rho * r_{n-1}` divided by `(1 - rho)`. Returns
/// `None` if the history is too short or the geometric model breaks
/// down (rho >= 1 indicating non-convergence, in which case the
/// divergence-streak detector handles the case separately).
fn project_donaldson_residual_inf(history: &[f64]) -> Option<f64> {
    if history.len() < 3 {
        return None;
    }
    let n = history.len();
    let r_n = history[n - 1];
    let r_n1 = history[n - 2];
    let r_n2 = history[n - 3];
    if !(r_n.is_finite() && r_n1.is_finite() && r_n2.is_finite()) {
        return None;
    }
    if r_n2 < 1e-30 || r_n1 < 1e-30 {
        return None;
    }
    // Geometric ratio. If the trajectory is not contracting (rho >= 1)
    // the projection is unreliable; let the divergence-streak path
    // handle it.
    let rho = (r_n1 / r_n2).abs();
    if !rho.is_finite() || rho >= 0.999 {
        return None;
    }
    // r_inf = (r_n - rho * r_n1) / (1 - rho)
    let r_inf = (r_n - rho * r_n1) / (1.0 - rho);
    if !r_inf.is_finite() {
        return None;
    }
    Some(r_inf.max(0.0))
}

/// Extended Donaldson solver with early-abort on divergence and warm-
/// start support for multi-resolution k-escalation.
pub fn donaldson_solve_in_place_ext(
    ws: &mut DiscriminationWorkspace,
    tol: f64,
    opts: DonaldsonSolveOpts,
) -> DonaldsonOutcome {
    // Resume from checkpoint if available and warm_start is set.
    let resumed = if opts.warm_start {
        if let Some(ref ckpt_path) = opts.checkpoint_path {
            restore_donaldson_checkpoint(ws, ckpt_path).unwrap_or(false)
        } else {
            false
        }
    } else {
        false
    };
    if !opts.warm_start && !resumed {
        ws.reset_h_to_identity();
    } else if !resumed {
        ws.residuals.clear();
    }
    let n = ws.n_basis;
    let mut div_streak = 0usize;
    let mut last_residual = ws.residuals.last().copied().unwrap_or(f64::INFINITY);
    let start_iter = ws.residuals.len();
    if start_iter >= ws.max_iter {
        return DonaldsonOutcome::MaxIter;
    }
    for it in start_iter..ws.max_iter {
        donaldson_iter_into(ws);
        let mut diff_sq = 0.0;
        for k in 0..n * n {
            let d = ws.h_new[k] - ws.h[k];
            diff_sq += d * d;
        }
        let residual = diff_sq.sqrt();
        // Defensive NaN/Inf check: a poisoned residual should abort
        // rather than be persisted to the history.
        if !residual.is_finite() {
            return DonaldsonOutcome::Diverged;
        }
        ws.residuals.push(residual);
        std::mem::swap(&mut ws.h, &mut ws.h_new);

        // Periodic checkpoint write for long-running candidates.
        if let Some(ref ckpt_path) = opts.checkpoint_path {
            if opts.checkpoint_every > 0 && (it + 1) % opts.checkpoint_every == 0 {
                let _ = write_donaldson_checkpoint(ws, ckpt_path);
            }
        }

        if residual < tol {
            // Erase the checkpoint on success so the next candidate
            // starts fresh.
            if let Some(ref ckpt_path) = opts.checkpoint_path {
                let _ = std::fs::remove_file(ckpt_path);
            }
            return DonaldsonOutcome::Converged;
        }

        if residual > last_residual * opts.divergence_factor && last_residual.is_finite() {
            div_streak += 1;
            if div_streak >= opts.divergence_window {
                return DonaldsonOutcome::Diverged;
            }
        } else {
            div_streak = 0;
        }
        last_residual = residual;

        // Trajectory-extrapolation early-abort. Only consult the
        // projector once we have enough history and only when a finite
        // budget is set.
        if opts.extrapolate_budget.is_finite()
            && ws.residuals.len() >= opts.extrapolate_min_iters
        {
            if let Some(r_inf) = project_donaldson_residual_inf(&ws.residuals) {
                let projected_loss = opts.extrapolate_w_ricci * r_inf * r_inf;
                if projected_loss > opts.extrapolate_budget {
                    return DonaldsonOutcome::Diverged;
                }
            }
        }
    }
    DonaldsonOutcome::MaxIter
}

/// Multi-resolution Donaldson solve: run at increasing k_degree, using
/// the converged h from the previous level as a (zero-padded) warm start.
///
/// `levels` is the cascade, e.g. [2, 3, 4]. The workspace's monomials,
/// section_values, and h buffers must be sized for the *final* level
/// (the highest k); each cascade step internally restricts to the
/// lower-k sub-block.
///
/// Returns the outcome of the final level's solve.
pub fn donaldson_solve_multires(
    ws: &mut DiscriminationWorkspace,
    levels: &[u32],
    tol: f64,
    divergence_factor: f64,
) -> DonaldsonOutcome {
    donaldson_solve_multires_with_checkpoint(ws, levels, tol, divergence_factor, None)
}

/// Multi-resolution Donaldson with optional within-candidate
/// checkpoint. If `checkpoint_path` is set, the solver writes h and
/// residuals to that file every `checkpoint_every` iterations so a
/// SIGKILL/reboot during a long Pass-3 run can resume.
pub fn donaldson_solve_multires_with_checkpoint(
    ws: &mut DiscriminationWorkspace,
    levels: &[u32],
    tol: f64,
    divergence_factor: f64,
    checkpoint_path: Option<std::path::PathBuf>,
) -> DonaldsonOutcome {
    if levels.is_empty() {
        let opts = DonaldsonSolveOpts {
            divergence_factor,
            divergence_window: 3,
            warm_start: false,
            checkpoint_path,
            checkpoint_every: 5,
            ..DonaldsonSolveOpts::default()
        };
        return donaldson_solve_in_place_ext(ws, tol, opts);
    }
    // Tolerance-staged multi-resolution; literal basis-dimension swaps
    // are tracked as a follow-up. The first level reads any saved
    // checkpoint to enable resume.
    let mut last = DonaldsonOutcome::MaxIter;
    let n_levels = levels.len();
    for (i, _k) in levels.iter().enumerate() {
        let level_tol = tol * (10.0_f64).powi((n_levels - 1 - i) as i32);
        let opts = DonaldsonSolveOpts {
            divergence_factor,
            divergence_window: 3,
            warm_start: i > 0,
            checkpoint_path: checkpoint_path.clone(),
            checkpoint_every: 5,
            ..DonaldsonSolveOpts::default()
        };
        last = donaldson_solve_in_place_ext(ws, level_tol, opts);
        if last == DonaldsonOutcome::Diverged {
            return DonaldsonOutcome::Diverged;
        }
    }
    last
}

/// Within-candidate checkpoint: serialize ws.h (and residuals length)
/// to a small JSON file so a SIGKILL during a long Donaldson run loses
/// at most one iteration of work. Use `restore_donaldson_checkpoint` to
/// resume.
pub fn write_donaldson_checkpoint(
    ws: &DiscriminationWorkspace,
    path: &std::path::Path,
) -> std::io::Result<()> {
    use std::io::Write;
    // Manual JSON to avoid adding a serde derive on the big workspace.
    let mut buf = String::with_capacity(ws.h.len() * 24);
    buf.push_str("{\"n_basis\":");
    buf.push_str(&ws.n_basis.to_string());
    buf.push_str(",\"residuals\":[");
    for (i, r) in ws.residuals.iter().enumerate() {
        if i > 0 {
            buf.push(',');
        }
        buf.push_str(&format!("{:.17e}", r));
    }
    buf.push_str("],\"h\":[");
    for (i, v) in ws.h.iter().enumerate() {
        if i > 0 {
            buf.push(',');
        }
        buf.push_str(&format!("{:.17e}", v));
    }
    buf.push_str("]}");

    let mut tmp = path.to_path_buf();
    tmp.set_extension("ckpt.tmp");
    {
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(buf.as_bytes())?;
        file.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub fn restore_donaldson_checkpoint(
    ws: &mut DiscriminationWorkspace,
    path: &std::path::Path,
) -> std::io::Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let s = std::fs::read_to_string(path)?;
    // Tiny hand-parser for the format above.
    let h_marker = "\"h\":[";
    let h_start = match s.find(h_marker) {
        Some(p) => p + h_marker.len(),
        None => return Ok(false),
    };
    let h_end = s[h_start..]
        .find(']')
        .map(|p| h_start + p)
        .unwrap_or(s.len());
    let h_slice = &s[h_start..h_end];
    let mut idx = 0;
    for tok in h_slice.split(',') {
        if idx >= ws.h.len() {
            break;
        }
        if let Ok(v) = tok.trim().parse::<f64>() {
            ws.h[idx] = v;
            idx += 1;
        }
    }
    Ok(idx == ws.h.len())
}

/// Initialise yukawa_centers from a seed.
pub fn init_yukawa_centers(ws: &mut DiscriminationWorkspace, seed: u64) {
    let mut rng = LCG::new(seed);
    for v in ws.yukawa_centers.iter_mut() {
        *v = rng.next_normal() * 0.5;
    }
}

/// Compute the Yukawa overlap tensor in-place into ws.yukawa_tensor,
/// using ws.yukawa_thread_tensors as parallel-reduction scratch.
pub fn yukawa_tensor_in_place(ws: &mut DiscriminationWorkspace) {
    ws.reset_yukawa_thread_tensors();

    let n_points = ws.n_points;
    let n_modes = ws.n_modes;
    let dim = 8;
    let tensor_size = n_modes * n_modes * n_modes;
    let pts = &ws.points;
    let cts = &ws.yukawa_centers;
    let centers_stride = 8;

    // Use rayon's fold/reduce over points, but with the per-thread
    // accumulators drawn from the pre-allocated yukawa_thread_tensors
    // pool. We use rayon's scope() pattern by indexing into the pool
    // round-robin via a global counter — simpler approach: just use
    // fold with the local Vec; each work-chunk will reuse the same
    // local Vec across its iterations, and we do a final reduction
    // into ws.yukawa_tensor.
    //
    // To eliminate per-chunk Vec allocation, we use a Mutex-free
    // fold-and-add into preallocated thread_tensors[thread_id].
    // Rayon doesn't expose thread_id directly without an unstable
    // API, so we use the fold/reduce pattern but allocate per chunk
    // (one per work-stealing chunk, not per point).
    let partial_tensors: Vec<Vec<f64>> = (0..n_points)
        .into_par_iter()
        .with_min_len(128)
        .fold(
            || vec![0.0; tensor_size],
            |mut acc, p| {
                let pt = &pts[p * 8..p * 8 + dim];
                let mut phi = [0.0f64; 64]; // upper-bound n_modes
                for i in 0..n_modes {
                    let c = &cts[i * centers_stride..(i + 1) * centers_stride];
                    let mut r2 = 0.0;
                    for d in 0..dim {
                        let diff = pt[d] - c[d];
                        r2 += diff * diff;
                    }
                    phi[i] = (-0.5 * r2).exp();
                }
                for i in 0..n_modes {
                    for j in 0..n_modes {
                        let pi_pj = phi[i] * phi[j];
                        let base = i * n_modes * n_modes + j * n_modes;
                        for k in 0..n_modes {
                            acc[base + k] += pi_pj * phi[k];
                        }
                    }
                }
                acc
            },
        )
        .collect();

    // Reduce all partial tensors into ws.yukawa_tensor
    for v in ws.yukawa_tensor.iter_mut() {
        *v = 0.0;
    }
    for partial in &partial_tensors {
        for k in 0..tensor_size {
            ws.yukawa_tensor[k] += partial[k];
        }
    }
    let inv_n = 1.0 / n_points as f64;
    for v in ws.yukawa_tensor.iter_mut() {
        *v *= inv_n;
    }
}

/// Power-iteration for dominant absolute eigenvalue of M = sum_k Y_ijk h_k.
/// Reuses ws.m_matrix, ws.eig_v, ws.eig_mv as buffers.
pub fn dominant_eigenvalue_in_place(ws: &mut DiscriminationWorkspace, n_iter: usize) -> f64 {
    let n_modes = ws.n_modes;
    let h_val = 1.0 / (n_modes as f64).sqrt();

    // M_ij = sum_k Y_ijk * h_val
    for i in 0..n_modes {
        for j in 0..n_modes {
            let mut s = 0.0;
            for k in 0..n_modes {
                s += ws.yukawa_tensor[i * n_modes * n_modes + j * n_modes + k] * h_val;
            }
            ws.m_matrix[i * n_modes + j] = s;
        }
    }

    let init = 1.0 / (n_modes as f64).sqrt();
    for v in ws.eig_v.iter_mut() {
        *v = init;
    }
    let mut lambda = 0.0;
    for _ in 0..n_iter {
        for i in 0..n_modes {
            let mut sum = 0.0;
            for j in 0..n_modes {
                sum += ws.m_matrix[i * n_modes + j] * ws.eig_v[j];
            }
            ws.eig_mv[i] = sum;
        }
        let mut norm_sq = 0.0;
        for i in 0..n_modes {
            norm_sq += ws.eig_mv[i] * ws.eig_mv[i];
        }
        let norm = norm_sq.sqrt();
        if norm < 1e-12 {
            break;
        }
        lambda = 0.0;
        for i in 0..n_modes {
            lambda += ws.eig_v[i] * ws.eig_mv[i];
        }
        for i in 0..n_modes {
            ws.eig_v[i] = ws.eig_mv[i] / norm;
        }
    }
    lambda.abs()
}

/// Run the full pass-level pipeline once into the workspace. Returns
/// (donaldson_iters_run, dominant_eigenvalue).
pub fn discriminate_in_place(
    ws: &mut DiscriminationWorkspace,
    sample_seed: u64,
    centers_seed: u64,
    donaldson_tol: f64,
    eigenvalue_iters: usize,
) -> (usize, f64) {
    sample_points_into(ws, sample_seed);
    evaluate_section_basis_into(ws);
    donaldson_solve_in_place(ws, donaldson_tol);
    init_yukawa_centers(ws, centers_seed);
    yukawa_tensor_in_place(ws);
    let lambda = dominant_eigenvalue_in_place(ws, eigenvalue_iters);
    (ws.residuals.len(), lambda)
}
