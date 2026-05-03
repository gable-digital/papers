//! P8.3-followup-A — Zero-mode rank diagnostic.
//!
//! Hostile-review hypothesis: `solve_harmonic_zero_modes` produces
//! rank-1 modes (all ψ_α proportional to a single common ψ_common).
//! That would explain why P8.3b's full Yukawa pipeline at k=3 yielded
//! 8/9 bit-identical-zero mass predictions on BOTH TY and Schoen
//! candidates: extract_3x3 contracts a rank-1 T_{ijk} → rank-1 Y →
//! 8 zero singular values.
//!
//! This binary does NOT modify production code. It calls the same
//! `solve_harmonic_zero_modes` entry point used by the production
//! pipeline, then reports:
//!   * Pairwise L²(M, ω³/3!) inner-product matrix ⟨ψ_α | ψ_β⟩.
//!   * Rank of the n_pts × n_modes evaluation matrix Ψ via SVD.
//!   * Singular values of Ψ.
//!   * Coefficient-vector pairwise overlaps |c_α · c̄_β| (basis-space
//!     view, independent of sample-cloud weighting).
//!   * Mode count vs. BBW prediction.
//!
//! The "rank" reported here is the rank of the harmonic-zero
//! subspace itself, regardless of orthonormalisation: ψ_β = c · ψ_α
//! with |c|=1 still passes Gram-Schmidt orthonormalisation as a
//! distinct unit vector, but the SVD on raw point values reveals
//! the true geometric rank.
//!
//! Run: `cargo run --release --bin p8_3_followup_a_zero_mode_diag`

use cy3_rust_solver::route34::cy3_metric_unified::{
    Cy3MetricResultKind, Cy3MetricSolver, Cy3MetricSpec, TianYauSolver,
};
use cy3_rust_solver::route34::hym_hermitian::{solve_hym_metric, HymConfig};
use cy3_rust_solver::route34::yukawa_pipeline::Cy3MetricResultBackground;
use cy3_rust_solver::route34::zero_modes_harmonic::{
    solve_harmonic_zero_modes, HarmonicConfig,
};
use cy3_rust_solver::zero_modes::{AmbientCY3, MonadBundle};
use num_complex::Complex64;

const N_PTS: usize = 200;
const SEED: u64 = 42;
const K: u32 = 3;
const MAX_ITER: usize = 50;
const DONALDSON_TOL: f64 = 1.0e-9;

fn main() {
    eprintln!("=== P8.3-followup-A zero-mode rank diagnostic ===");
    eprintln!("TY k={}, n_pts={}, seed={}, max_iter={}", K, N_PTS, SEED, MAX_ITER);
    eprintln!();

    let bundle = MonadBundle::anderson_lukas_palti_example();
    let ambient = AmbientCY3::tian_yau_upstairs();

    // 1. TY metric solve.
    let solver = TianYauSolver;
    let spec = Cy3MetricSpec::TianYau {
        k: K,
        n_sample: N_PTS,
        max_iter: MAX_ITER,
        donaldson_tol: DONALDSON_TOL,
        seed: SEED,
    };
    let r = solver.solve_metric(&spec).expect("TY metric solve");
    let summary = r.summary();
    eprintln!(
        "TY metric: σ={:.6e}  iters={}",
        summary.final_sigma_residual, summary.iterations_run
    );

    let bg = match &r {
        Cy3MetricResultKind::TianYau(t) => Cy3MetricResultBackground::from_ty(t.as_ref()),
        Cy3MetricResultKind::Schoen(_) => panic!("expected TY result"),
    };

    // 2. HYM bundle metric.
    let hym_cfg = HymConfig {
        max_iter: 8,
        damping: 0.5,
        ..HymConfig::default()
    };
    let h_v = solve_hym_metric(&bundle, &bg, &hym_cfg);
    eprintln!("HYM solved (h_V dim {})", h_v.n);

    // 3. Harmonic zero modes. Try DEFAULT first (legacy 1e-3 ratio).
    //    If that returns 0 modes (the empty-kernel-basis path), retry
    //    with `kernel_dim_target = Some(9)` — the same fallback the
    //    production pipeline applies in `predict_fermion_masses_with_overrides`.
    let cfg_default = HarmonicConfig::default();
    let res_default = solve_harmonic_zero_modes(&bundle, &ambient, &bg, &h_v, &cfg_default);
    eprintln!(
        "Default config: n_modes = {} (predicted = {})",
        res_default.modes.len(),
        res_default.cohomology_dim_predicted
    );
    let res = if res_default.modes.is_empty() {
        eprintln!("Default returned 0 modes — retrying with kernel_dim_target=Some(9)");
        let cfg_fallback = HarmonicConfig {
            auto_use_predicted_dim: false,
            kernel_dim_target: Some(9),
            ..HarmonicConfig::default()
        };
        solve_harmonic_zero_modes(&bundle, &ambient, &bg, &h_v, &cfg_fallback)
    } else {
        res_default
    };

    let n_modes = res.modes.len();
    eprintln!();
    eprintln!("--- HarmonicZeroModeResult summary ---");
    eprintln!("seed_basis_dim         : {}", res.seed_basis_dim);
    eprintln!("cohomology_dim_predicted: {}", res.cohomology_dim_predicted);
    eprintln!("cohomology_dim_observed: {}", res.cohomology_dim_observed);
    eprintln!("n_modes returned       : {}", n_modes);
    eprintln!("orthonormality_residual: {:.3e}", res.orthonormality_residual);
    eprintln!("kernel_selection_used  : {:?}", res.kernel_selection_used);

    // Print full eigenvalue spectrum.
    eprintln!();
    eprintln!("--- Full L = D*D eigenvalue spectrum (ascending) ---");
    for (i, lam) in res.eigenvalues_full.iter().enumerate() {
        eprintln!("  λ[{:>2}] = {:.6e}", i, lam);
    }

    if n_modes == 0 {
        eprintln!("\nNo modes returned — diagnostic cannot proceed further.");
        return;
    }

    // ========================================================
    // (A) Pairwise L² inner-product matrix on raw mode values.
    // ========================================================
    //
    // We use a UNIFORM weighting (no ω, no h_V) here so the rank
    // diagnostic isn't masked by inner-product reweighting that
    // could rescale linearly dependent modes into apparent
    // orthonormality. This is the "geometric rank of the function
    // space spanned by ψ_α at the sample cloud", not the
    // orthonormality test (which is already reported above).
    let n_pts = res.modes[0].values.len();
    eprintln!();
    eprintln!("--- Pairwise raw-L² inner products ⟨ψ_α | ψ_β⟩ (uniform weight) ---");
    eprintln!("(Diagonal = ||ψ_α||²; off-diagonal magnitude near √(diag_α·diag_β) signals proportionality.)");
    let mut g_raw = vec![Complex64::new(0.0, 0.0); n_modes * n_modes];
    for a in 0..n_modes {
        for b in 0..n_modes {
            let mut acc = Complex64::new(0.0, 0.0);
            for p in 0..n_pts {
                acc += res.modes[a].values[p].conj() * res.modes[b].values[p];
            }
            acc /= n_pts as f64;
            g_raw[a * n_modes + b] = acc;
        }
    }
    eprint!("       ");
    for b in 0..n_modes {
        eprint!("    β={:<2}     ", b);
    }
    eprintln!();
    for a in 0..n_modes {
        eprint!("α={:<2} : ", a);
        for b in 0..n_modes {
            let z = g_raw[a * n_modes + b];
            eprint!("{:+.3e} ", z.norm());
        }
        eprintln!();
    }

    // Normalised correlation matrix |⟨α|β⟩| / √(⟨α|α⟩⟨β|β⟩).
    eprintln!();
    eprintln!("--- Normalised correlation |⟨ψ_α|ψ_β⟩|/√(||ψ_α||²·||ψ_β||²) ---");
    eprintln!("(1.0 = perfectly proportional, 0.0 = orthogonal.)");
    eprint!("       ");
    for b in 0..n_modes {
        eprint!("    β={:<2}     ", b);
    }
    eprintln!();
    for a in 0..n_modes {
        eprint!("α={:<2} : ", a);
        for b in 0..n_modes {
            let num = g_raw[a * n_modes + b].norm();
            let da = g_raw[a * n_modes + a].re.max(0.0).sqrt();
            let db = g_raw[b * n_modes + b].re.max(0.0).sqrt();
            let den = (da * db).max(1.0e-30);
            eprint!("{:+.3e} ", num / den);
        }
        eprintln!();
    }

    // ========================================================
    // (B) SVD of mode-evaluation matrix Ψ : n_pts × n_modes.
    // ========================================================
    //
    // Compute Ψ† Ψ (n_modes × n_modes Hermitian) and diagonalise.
    // Singular values of Ψ are √eigenvalues of Ψ†Ψ. Rank = #(σ above
    // floor).
    //
    // We re-use the same psi†psi pattern (identical structure to
    // g_raw above) — they're the same matrix mathematically.
    eprintln!();
    eprintln!("--- Singular values of Ψ (n_pts × n_modes), via Ψ†Ψ eigenvalues ---");
    let psi_dag_psi = g_raw.clone();
    let (sv2, _v) = hermitian_eig(&psi_dag_psi, n_modes);
    let mut sv: Vec<f64> = sv2.iter().map(|x| x.max(0.0).sqrt()).collect();
    sv.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let s_max = sv.first().copied().unwrap_or(0.0).max(1.0e-300);
    eprintln!("(σ_i / σ_max — values << 1 are linearly dependent directions.)");
    for (i, s) in sv.iter().enumerate() {
        eprintln!(
            "  σ[{:>2}] = {:.6e}   σ/σ_max = {:.6e}",
            i,
            s,
            s / s_max
        );
    }
    let floor = 1.0e-10;
    let rank: usize = sv.iter().filter(|&&s| s / s_max > floor).count();
    eprintln!();
    eprintln!("Rank(Ψ) at floor σ/σ_max > {:.0e}: {}/{}", floor, rank, n_modes);

    // ========================================================
    // (C) Coefficient-vector overlaps (basis-space view).
    // ========================================================
    //
    // Each mode has a coefficient vector c_α ∈ ℂ^n_seeds. The
    // eigenvectors of a Hermitian matrix are orthonormal in the
    // standard inner product, so |c_α · c̄_β| = δ_{αβ} is expected
    // BY CONSTRUCTION of the Jacobi solve. The interesting test is
    // whether c_α has support on multiple b-line classes (i.e.
    // whether the mode mixes B-summands) or collapses onto one.
    eprintln!();
    eprintln!("--- Coefficient-vector overlaps c_α · c̄_β (Jacobi orthogonality check) ---");
    let n_seeds = res.modes[0].coefficients.len();
    eprintln!("n_seeds = {}", n_seeds);
    let mut max_off = 0.0_f64;
    for a in 0..n_modes {
        for b in 0..n_modes {
            let mut acc = Complex64::new(0.0, 0.0);
            for k in 0..n_seeds {
                acc += res.modes[a].coefficients[k].conj() * res.modes[b].coefficients[k];
            }
            if a != b && acc.norm() > max_off {
                max_off = acc.norm();
            }
        }
    }
    eprintln!("max off-diagonal |c_α · c̄_β| (a≠b) = {:.3e}", max_off);

    // Per-mode: which b-line(s) carry significant weight?
    eprintln!();
    eprintln!("--- Per-mode B-line support (|c|² aggregated by b_line) ---");
    eprintln!("(If every mode has ALL its weight on the same b_line, that's ");
    eprintln!(" the rank-1-collapse signature.)");
    let n_b = bundle.b_lines.len();
    for (m_idx, m) in res.modes.iter().enumerate() {
        let mut by_bline = vec![0.0_f64; n_b];
        let mut total = 0.0_f64;
        for k in 0..n_seeds {
            let w = m.coefficients[k].norm_sqr();
            let b = res.seed_to_b_line[k];
            by_bline[b] += w;
            total += w;
        }
        eprint!("mode {:>2} : λ={:+.3e}  ", m_idx, m.eigenvalue);
        for (b, w) in by_bline.iter().enumerate() {
            let frac = if total > 0.0 { w / total } else { 0.0 };
            eprint!("b{}:{:.2} ", b, frac);
        }
        eprintln!();
    }

    // ========================================================
    // (D) Final verdict.
    // ========================================================
    eprintln!();
    eprintln!("=== Verdict ===");
    if n_modes >= 2 {
        // Examine off-diagonal of normalised correlation. If average
        // off-diagonal correlation > 0.9, this is rank-1 collapse.
        let mut sum_off = 0.0_f64;
        let mut count_off = 0usize;
        for a in 0..n_modes {
            for b in 0..n_modes {
                if a == b {
                    continue;
                }
                let num = g_raw[a * n_modes + b].norm();
                let da = g_raw[a * n_modes + a].re.max(0.0).sqrt();
                let db = g_raw[b * n_modes + b].re.max(0.0).sqrt();
                let den = (da * db).max(1.0e-30);
                sum_off += num / den;
                count_off += 1;
            }
        }
        let avg_off = if count_off > 0 {
            sum_off / count_off as f64
        } else {
            0.0
        };
        eprintln!("Avg normalised |⟨ψ_α|ψ_β⟩| (a≠b) = {:.3e}", avg_off);
        eprintln!("Rank(Ψ) = {} / n_modes = {}", rank, n_modes);
        if rank <= 1 && n_modes >= 2 {
            eprintln!("VERDICT: rank-1 collapse CONFIRMED — modes are ~proportional.");
        } else if avg_off > 0.5 {
            eprintln!(
                "VERDICT: severe linear dependence (avg off-diag {:.2}).",
                avg_off
            );
        } else if rank == n_modes {
            eprintln!("VERDICT: full rank — modes are linearly independent.");
        } else {
            eprintln!(
                "VERDICT: partial rank {}/{} — some dependence present.",
                rank, n_modes
            );
        }
    } else {
        eprintln!("VERDICT: only {} mode(s) returned — cannot test rank.", n_modes);
    }
}

// -------- tiny n×n complex Hermitian eigensolver (real eigenvalues).
//
// Local copy of the same Jacobi rotation pattern used inside
// `zero_modes_harmonic.rs`. Kept independent so this diagnostic
// does NOT exercise any internal helpers that may change in the
// production solver.
fn hermitian_eig(a_in: &[Complex64], n: usize) -> (Vec<f64>, Vec<Complex64>) {
    let mut a = a_in.to_vec();
    let mut v = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        v[i * n + i] = Complex64::new(1.0, 0.0);
    }
    let max_sweeps = 256;
    let tol = 1.0e-14;
    for _sweep in 0..max_sweeps {
        let mut off = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off += a[i * n + j].norm_sqr();
            }
        }
        if off.sqrt() < tol {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let app = a[p * n + p].re;
                let aqq = a[q * n + q].re;
                let apq = a[p * n + q];
                if apq.norm() < 1.0e-18 {
                    continue;
                }
                // Build 2×2 Hermitian, solve for unitary that diagonalises.
                // The standard trick: for Hermitian A, find phase φ
                // such that e^{iφ} apq is real, then real-symmetric Jacobi.
                let phase = if apq.norm() > 0.0 {
                    apq / Complex64::new(apq.norm(), 0.0)
                } else {
                    Complex64::new(1.0, 0.0)
                };
                let r_apq = apq.norm();
                let theta = (aqq - app) / (2.0 * r_apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                // Update rows/cols p, q.
                for i in 0..n {
                    let aip = a[i * n + p];
                    let aiq = a[i * n + q];
                    a[i * n + p] = Complex64::new(c, 0.0) * aip
                        - Complex64::new(s, 0.0) * phase.conj() * aiq;
                    a[i * n + q] = Complex64::new(s, 0.0) * phase * aip
                        + Complex64::new(c, 0.0) * aiq;
                }
                for j in 0..n {
                    let apj = a[p * n + j];
                    let aqj = a[q * n + j];
                    a[p * n + j] = Complex64::new(c, 0.0) * apj
                        - Complex64::new(s, 0.0) * phase * aqj;
                    a[q * n + j] = Complex64::new(s, 0.0) * phase.conj() * apj
                        + Complex64::new(c, 0.0) * aqj;
                }
                // Update eigenvector accumulator.
                for i in 0..n {
                    let vip = v[i * n + p];
                    let viq = v[i * n + q];
                    v[i * n + p] = Complex64::new(c, 0.0) * vip
                        - Complex64::new(s, 0.0) * phase.conj() * viq;
                    v[i * n + q] = Complex64::new(s, 0.0) * phase * vip
                        + Complex64::new(c, 0.0) * viq;
                }
            }
        }
    }
    let mut eigs: Vec<f64> = (0..n).map(|i| a[i * n + i].re).collect();
    // Sort ascending for cleanliness.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| eigs[i].partial_cmp(&eigs[j]).unwrap_or(std::cmp::Ordering::Equal));
    let eigs_sorted: Vec<f64> = idx.iter().map(|&i| eigs[i]).collect();
    let mut v_sorted = vec![Complex64::new(0.0, 0.0); n * n];
    for (new_col, &old_col) in idx.iter().enumerate() {
        for r in 0..n {
            v_sorted[r * n + new_col] = v[r * n + old_col];
        }
    }
    eigs.copy_from_slice(&eigs_sorted);
    (eigs, v_sorted)
}
