//! P8.3-followup-A2 — T-tensor sparsity diagnostic.
//!
//! P8.3-followup-A established that the harmonic zero modes ψ_α are
//! full rank (9/9 at machine precision). The 8/9 zero-mass collapse
//! must therefore originate downstream of the harmonic basis. Two
//! candidate causes remain:
//!
//!   (a) **Geometric (T-sparse).** `compute_yukawa_couplings` itself
//!       produces a sparse / low-rank T_{ijk} (e.g. only diagonal
//!       entries non-zero, or only one Higgs slice non-trivial),
//!       which is a property of the bundle/metric and the triple
//!       overlap that no choice of sector assignment can fix.
//!
//!   (b) **Assignment-driven.** T_{ijk} has many non-zero entries,
//!       but `assign_sectors_dynamic`'s round-robin sector buckets
//!       (forced when only 2 of the expected 3 Wilson Z/3 phase
//!       classes appear) pull from disjoint mode pools at the
//!       chosen Higgs slice h_0, zeroing 8/9 entries by simple
//!       index mismatch.
//!
//! This binary localises the cause by:
//!
//!   1. Solving the metric (TY k=3 then Schoen d=(3,3,1)).
//!   2. Solving HYM + harmonic zero modes via the same call chain
//!      used by the production pipeline.
//!   3. Calling `compute_yukawa_couplings` to obtain T_{ijk} (full
//!      9 × 9 × 9 tensor).
//!   4. Dumping all 729 entries to a CSV alongside summary stats:
//!      sparsity ratio, max, min nonzero, mean nonzero, per-Higgs-
//!      slice rank.
//!   5. Calling `assign_sectors_dynamic` and cross-referencing
//!      which (i, j) buckets get filled with the (i, j) coordinates
//!      that are non-zero at T_{i, j, h_0}.
//!
//! This binary does NOT modify production code.
//!
//! Run:
//! ```bash
//! cargo run --release --features "gpu precision-bigfloat" \
//!   --bin p8_3_followup_a2_tensor_sparsity_diag
//! ```

use cy3_rust_solver::route34::cy3_metric_unified::{
    Cy3MetricResultKind, Cy3MetricSolver, Cy3MetricSpec, SchoenSolver, TianYauSolver,
};
use cy3_rust_solver::route34::hym_hermitian::{
    solve_hym_metric, HymConfig, MetricBackground,
};
use cy3_rust_solver::route34::wilson_line_e8::WilsonLineE8;
use cy3_rust_solver::route34::yukawa_overlap_real::{
    compute_yukawa_couplings, Tensor3, YukawaConfig,
};
use cy3_rust_solver::route34::yukawa_pipeline::Cy3MetricResultBackground;
use cy3_rust_solver::route34::yukawa_sectors_real::assign_sectors_dynamic;
use cy3_rust_solver::route34::zero_modes_harmonic::{
    solve_harmonic_zero_modes, HarmonicConfig,
};
use cy3_rust_solver::zero_modes::{AmbientCY3, MonadBundle};
use num_complex::Complex64;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::Path;

const N_PTS: usize = 200;
const SEED: u64 = 42;
const K: u32 = 3;
const MAX_ITER: usize = 50;
const DONALDSON_TOL: f64 = 1.0e-9;
const MAG_FLOOR: f64 = 1.0e-10;

fn main() {
    eprintln!("=== P8.3-followup-A2 T-tensor sparsity diagnostic ===");
    eprintln!(
        "n_pts={}, seed={}, k_or_dx={}, max_iter={}, donaldson_tol={:.0e}",
        N_PTS, SEED, K, MAX_ITER, DONALDSON_TOL
    );
    eprintln!("MAG_FLOOR (treat |T| <= floor as zero) = {:.0e}", MAG_FLOOR);
    eprintln!();

    let out_dir = Path::new("output");
    if !out_dir.exists() {
        create_dir_all(out_dir).expect("create output dir");
    }

    // Run TY first.
    eprintln!(
        "=================================================================="
    );
    eprintln!("TY/Z3 (k={})", K);
    eprintln!(
        "=================================================================="
    );
    let ty_summary = run_one(
        "TY/Z3",
        Cy3MetricSpec::TianYau {
            k: K,
            n_sample: N_PTS,
            max_iter: MAX_ITER,
            donaldson_tol: DONALDSON_TOL,
            seed: SEED,
        },
        Box::new(TianYauSolver),
        MonadBundle::anderson_lukas_palti_example(),
        AmbientCY3::tian_yau_upstairs(),
        WilsonLineE8::canonical_e8_to_e6_su3(3),
        9,
        out_dir.join("p8_3_followup_a2_tensor_TY.csv"),
    );

    // Run Schoen with current AKLP-aliased bundle.
    eprintln!();
    eprintln!(
        "=================================================================="
    );
    eprintln!("Schoen Z/3xZ/3 (d=(3,3,1))");
    eprintln!(
        "=================================================================="
    );
    let schoen_summary = run_one(
        "Schoen/Z3xZ3",
        Cy3MetricSpec::Schoen {
            d_x: 3,
            d_y: 3,
            d_t: 1,
            n_sample: N_PTS,
            max_iter: MAX_ITER,
            donaldson_tol: DONALDSON_TOL,
            seed: SEED,
        },
        Box::new(SchoenSolver),
        MonadBundle::schoen_z3xz3_canonical(),
        AmbientCY3::schoen_z3xz3_upstairs(),
        WilsonLineE8::canonical_e8_to_e6_su3(9),
        9,
        out_dir.join("p8_3_followup_a2_tensor_Schoen.csv"),
    );

    eprintln!();
    eprintln!(
        "=================================================================="
    );
    eprintln!("=== TY vs Schoen comparison ===");
    eprintln!(
        "=================================================================="
    );
    let header = format!(
        "{:<14} {:>8} {:>8} {:>10} {:>12} {:>12} {:>12}",
        "label", "n_modes", "n_nz", "sparsity", "max|T|", "min_nz|T|", "mean_nz|T|"
    );
    eprintln!("{}", header);
    for s in [&ty_summary, &schoen_summary] {
        eprintln!(
            "{:<14} {:>8} {:>8} {:>10.4} {:>12.4e} {:>12.4e} {:>12.4e}",
            s.label,
            s.n_modes,
            s.n_nonzero,
            s.sparsity_ratio,
            s.max_abs,
            s.min_nz_abs,
            s.mean_nz_abs,
        );
    }
    eprintln!();
    eprintln!("CSV outputs:");
    eprintln!("  output/p8_3_followup_a2_tensor_TY.csv");
    eprintln!("  output/p8_3_followup_a2_tensor_Schoen.csv");
}

/// Per-candidate summary statistics for the final comparison.
struct Summary {
    label: String,
    n_modes: usize,
    n_nonzero: usize,
    sparsity_ratio: f64,
    max_abs: f64,
    min_nz_abs: f64,
    mean_nz_abs: f64,
}

#[allow(clippy::too_many_arguments)]
fn run_one(
    label: &str,
    spec: Cy3MetricSpec,
    solver: Box<dyn Cy3MetricSolver>,
    bundle: MonadBundle,
    ambient: AmbientCY3,
    wilson: WilsonLineE8,
    fallback_kernel_dim: usize,
    csv_path: std::path::PathBuf,
) -> Summary {
    let r = solver.solve_metric(&spec).expect("metric solve");
    let summary = r.summary();
    eprintln!(
        "metric: σ={:.6e}  iters={}  n_basis={}",
        summary.final_sigma_residual, summary.iterations_run, summary.n_basis
    );

    let bg = match &r {
        Cy3MetricResultKind::TianYau(t) => Cy3MetricResultBackground::from_ty(t.as_ref()),
        Cy3MetricResultKind::Schoen(s) => Cy3MetricResultBackground::from_schoen(s.as_ref()),
    };
    let n_pts_accepted: usize = bg.n_points();
    eprintln!("metric background: {} accepted points", n_pts_accepted);

    let hym_cfg = HymConfig {
        max_iter: 8,
        damping: 0.5,
        ..HymConfig::default()
    };
    let h_v = solve_hym_metric(&bundle, &bg, &hym_cfg);
    eprintln!("HYM solved (h_V dim = {})", h_v.n);

    // Try BBW-correct path first; if zero modes, fall back to
    // kernel_dim_target.
    let cfg_default = HarmonicConfig {
        auto_use_predicted_dim: true,
        ..HarmonicConfig::default()
    };
    let modes_default = solve_harmonic_zero_modes(&bundle, &ambient, &bg, &h_v, &cfg_default);
    let modes = if modes_default.modes.is_empty() {
        eprintln!(
            "BBW-correct kernel returned 0 modes — falling back to kernel_dim_target=Some({})",
            fallback_kernel_dim
        );
        let cfg_fallback = HarmonicConfig {
            auto_use_predicted_dim: false,
            kernel_dim_target: Some(fallback_kernel_dim),
            ..HarmonicConfig::default()
        };
        solve_harmonic_zero_modes(&bundle, &ambient, &bg, &h_v, &cfg_fallback)
    } else {
        modes_default
    };

    let n_modes = modes.modes.len();
    eprintln!(
        "harmonic modes: returned={}, predicted={}, observed={}",
        n_modes, modes.cohomology_dim_predicted, modes.cohomology_dim_observed
    );
    if n_modes == 0 {
        eprintln!("zero modes returned; cannot continue diagnostic for {}", label);
        return Summary {
            label: label.to_string(),
            n_modes: 0,
            n_nonzero: 0,
            sparsity_ratio: 1.0,
            max_abs: 0.0,
            min_nz_abs: 0.0,
            mean_nz_abs: 0.0,
        };
    }

    // Compute T_{ijk}.
    let yres = compute_yukawa_couplings(
        &bg,
        &h_v,
        &modes,
        &YukawaConfig {
            n_bootstrap: 4,
            seed: SEED,
        },
    );
    let t = yres.couplings;
    let n = t.n;
    let total = n * n * n;
    eprintln!("Yukawa T tensor: n={}, total entries={}", n, total);

    // CSV dump of all entries.
    {
        let f = File::create(&csv_path).expect("open csv");
        let mut w = BufWriter::new(f);
        writeln!(w, "i,j,k,abs,arg,re,im").ok();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let z = t.entry(i, j, k);
                    writeln!(
                        w,
                        "{},{},{},{:.6e},{:.6e},{:.6e},{:.6e}",
                        i,
                        j,
                        k,
                        z.norm(),
                        z.arg(),
                        z.re,
                        z.im
                    )
                    .ok();
                }
            }
        }
        eprintln!("CSV written: {}", csv_path.display());
    }

    // Aggregate sparsity stats.
    let mut n_nz = 0usize;
    let mut max_abs = 0.0f64;
    let mut sum_nz = 0.0f64;
    let mut min_nz = f64::INFINITY;
    for k_idx in 0..total {
        let a = t.data[k_idx].norm();
        if a > MAG_FLOOR {
            n_nz += 1;
            sum_nz += a;
            if a > max_abs {
                max_abs = a;
            }
            if a < min_nz {
                min_nz = a;
            }
        }
    }
    let mean_nz = if n_nz > 0 { sum_nz / n_nz as f64 } else { 0.0 };
    let min_nz_out = if n_nz > 0 { min_nz } else { 0.0 };
    let sparsity_ratio = (total - n_nz) as f64 / total as f64;
    eprintln!();
    eprintln!("--- Sparsity statistics ({}) ---", label);
    eprintln!("Total entries           : {}", total);
    eprintln!("Non-zero entries        : {}", n_nz);
    eprintln!("Sparsity ratio (zeros/N): {:.4}", sparsity_ratio);
    eprintln!("max |T|                 : {:.6e}", max_abs);
    eprintln!("min nonzero |T|         : {:.6e}", min_nz_out);
    eprintln!("mean nonzero |T|        : {:.6e}", mean_nz);

    // Per-Higgs-slice rank.
    eprintln!();
    eprintln!("--- Per-Higgs-slice rank (9x9 (i,j) submatrix at fixed k) ---");
    eprintln!("{:>3}  {:>6}  {:>6}  {:>14}  {:>14}", "k", "rank", "n_nz", "max|T|", "fro_norm");
    let mut slice_nz_coords: Vec<Vec<(usize, usize)>> = Vec::with_capacity(n);
    for k in 0..n {
        let (rank, slice_nz_count, max_slice, fro_norm, coords) = slice_stats(&t, k);
        slice_nz_coords.push(coords);
        eprintln!(
            "{:>3}  {:>6}  {:>6}  {:>14.6e}  {:>14.6e}",
            k, rank, slice_nz_count, max_slice, fro_norm
        );
    }

    // Identify h_0 the way assign_sectors_dynamic does (sorted ascending
    // by harmonic eigenvalue, then take first).
    let sectors = assign_sectors_dynamic(&bundle, &modes, &wilson);
    eprintln!();
    eprintln!("--- Sector assignment ---");
    eprintln!("up_quark  : {:?}", sectors.up_quark);
    eprintln!("down_quark: {:?}", sectors.down_quark);
    eprintln!("lepton    : {:?}", sectors.lepton);
    eprintln!("higgs     : {:?}", sectors.higgs);

    let h0 = sectors
        .higgs
        .iter()
        .copied()
        .find(|&h| h < n)
        .unwrap_or(0);
    eprintln!("h_0 (lowest-eigenvalue Higgs index)    : {}", h0);

    // Print eigenvalue list to clarify the sort order.
    eprintln!();
    eprintln!("--- Mode eigenvalues (ascending order from solver) ---");
    for (i, m) in modes.modes.iter().enumerate() {
        eprintln!("  λ[{:>2}] = {:.6e}", i, m.eigenvalue);
    }

    // Print extracted 3x3 buckets that the pipeline would actually
    // populate at h_0 (cartesian products of left x right index lists).
    eprintln!();
    eprintln!("--- Sector x sector buckets evaluated at h_0={} ---", h0);
    println_3x3_at_h0("Y_u (up x up)", &t, &sectors.up_quark, &sectors.up_quark, h0);
    println_3x3_at_h0(
        "Y_d (up x down)",
        &t,
        &sectors.up_quark,
        &sectors.down_quark,
        h0,
    );
    println_3x3_at_h0(
        "Y_e (lepton x lepton)",
        &t,
        &sectors.lepton,
        &sectors.lepton,
        h0,
    );

    // Cross-reference (assignment vs sparsity) at the h_0 slice.
    //
    // For each of Y_u, Y_d, Y_e the pipeline evaluates a 3x3
    // (li, rj, h_0) sub-block. Count how many of those nine entries
    // land on coordinates that are above the MAG_FLOOR on the full
    // T_{i,j,h_0} slice.
    eprintln!();
    eprintln!("--- Assignment vs sparsity cross-reference ---");
    eprintln!("(For each sector pair: of the 9 (li,rj) buckets evaluated at h_0,");
    eprintln!(" how many are above the MAG_FLOOR on the T_{{i,j,h_0}} slice?)");
    let h0_slice_nz: &[(usize, usize)] = if h0 < slice_nz_coords.len() {
        &slice_nz_coords[h0]
    } else {
        &[]
    };
    let h0_slice_nz_set: std::collections::HashSet<(usize, usize)> =
        h0_slice_nz.iter().copied().collect();
    eprintln!(
        "h_0 slice has {} non-zero (i,j) entries (out of {})",
        h0_slice_nz.len(),
        n * n
    );
    bucket_overlap(
        "Y_u (up x up)",
        &sectors.up_quark,
        &sectors.up_quark,
        &h0_slice_nz_set,
    );
    bucket_overlap(
        "Y_d (up x down)",
        &sectors.up_quark,
        &sectors.down_quark,
        &h0_slice_nz_set,
    );
    bucket_overlap(
        "Y_e (lepton x lepton)",
        &sectors.lepton,
        &sectors.lepton,
        &h0_slice_nz_set,
    );

    // Verdict.
    //
    // Cause (b) (assignment-driven) — diagnostic signature:
    //   * h_0 slice has substantially more than 3 non-zero (i,j)
    //     entries (i.e. T is NOT geometrically sparse at h_0).
    //   * Yet the 9 buckets that the round-robin assignment evaluates
    //     are mostly disjoint from those non-zero coordinates.
    //
    // Cause (a) (geometric) — diagnostic signature:
    //   * h_0 slice itself has only ~1-3 non-zero (i,j) entries.
    //     Then no reassignment of sector buckets can produce a
    //     full-rank Y_ij at h_0.
    eprintln!();
    eprintln!("--- Verdict ({}) ---", label);
    let h0_nz_count = h0_slice_nz.len();
    let h0_rank = if h0 < slice_nz_coords.len() {
        slice_rank(&t, h0)
    } else {
        0
    };
    eprintln!("Non-zero (i,j) entries at h_0 = {}", h0_nz_count);
    eprintln!("Rank of T_{{:,:,h_0}} 9x9 slice = {}", h0_rank);
    if h0_rank <= 1 || h0_nz_count <= 3 {
        eprintln!(
            "VERDICT: cause (a) GEOMETRIC sparsity — T_{{i,j,h_0}} is itself rank ≤ 1 ({} nz)",
            h0_nz_count
        );
        eprintln!(
            "→ Even relaxing the sector bucket assignment cannot lift Y to rank > 1."
        );
        eprintln!(
            "→ Recommend P8.3-followup-D: investigate why T-tensor is geometrically sparse"
        );
        eprintln!(
            "  (likely upstream representation-theory / harmonic-mode-overlap issue)."
        );
    } else {
        eprintln!(
            "VERDICT: cause (b) ASSIGNMENT-DRIVEN — T has {} non-zero (i,j) entries at h_0,",
            h0_nz_count
        );
        eprintln!(
            "         rank {}, but bucket overlap with sector indices is sparse.",
            h0_rank
        );
        eprintln!(
            "→ Recommend P8.3-followup-B (real Schoen 3-factor bundle providing 3 Wilson"
        );
        eprintln!(
            "  Z/3 phase classes). Once 3 phase classes exist, the round-robin fallback in"
        );
        eprintln!(
            "  assign_sectors_dynamic disengages and bucket assignment becomes consistent."
        );
    }

    Summary {
        label: label.to_string(),
        n_modes,
        n_nonzero: n_nz,
        sparsity_ratio,
        max_abs,
        min_nz_abs: min_nz_out,
        mean_nz_abs: mean_nz,
    }
}

/// Print a 3×3 sector × sector slice at fixed h_0.
fn println_3x3_at_h0(
    name: &str,
    t: &Tensor3,
    left: &[usize],
    right: &[usize],
    h0: usize,
) {
    eprintln!("{}:", name);
    if t.n == 0 {
        eprintln!("  (empty tensor)");
        return;
    }
    // **P8.3-followup-B2.** Mirror the new zero-pad semantics in the
    // matrix display: under-sized sector slots show a `--` placeholder
    // rather than a duplicated-index entry, so the printed matrix
    // matches what `extract_3x3_from_tensor` would actually return.
    let lookup = |v: &[usize], i: usize| -> Option<usize> {
        v.get(i).copied().filter(|&idx| idx < t.n)
    };
    eprint!("       ");
    for j in 0..3 {
        match lookup(right, j) {
            Some(rj) => eprint!("    rj={:<2}     ", rj),
            None => eprint!("    rj=--     "),
        }
    }
    eprintln!();
    for i in 0..3 {
        match lookup(left, i) {
            Some(li) => eprint!("li={:<2} : ", li),
            None => eprint!("li=-- : "),
        }
        for j in 0..3 {
            match (lookup(left, i), lookup(right, j)) {
                (Some(li), Some(rj)) => {
                    let z = t.entry(li, rj, h0);
                    eprint!("{:+.3e} ", z.norm());
                }
                _ => eprint!("    0     "),
            }
        }
        eprintln!();
    }
}

/// Compute (rank, nonzero_count, max_abs, fro_norm, nz_coords) for a
/// 9×9 (i, j) submatrix at fixed k. Rank is via SVD ≈ #(σ above
/// floor) on the Hermitian Gram M = A† A using Jacobi rotations.
fn slice_stats(
    t: &Tensor3,
    k: usize,
) -> (usize, usize, f64, f64, Vec<(usize, usize)>) {
    let n = t.n;
    if n == 0 {
        return (0, 0, 0.0, 0.0, vec![]);
    }
    let mut nz_count = 0usize;
    let mut max_abs = 0.0f64;
    let mut fro2 = 0.0f64;
    let mut nz_coords = Vec::new();
    let mut a = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let z = t.entry(i, j, k);
            a[i * n + j] = z;
            let m = z.norm();
            if m > MAG_FLOOR {
                nz_count += 1;
                nz_coords.push((i, j));
                if m > max_abs {
                    max_abs = m;
                }
            }
            fro2 += z.norm_sqr();
        }
    }
    let r = matrix_rank(&a, n);
    (r, nz_count, max_abs, fro2.sqrt(), nz_coords)
}

fn slice_rank(t: &Tensor3, k: usize) -> usize {
    let (r, _, _, _, _) = slice_stats(t, k);
    r
}

/// Cross-reference a sector pair against the non-zero coords of the
/// h_0 slice.
fn bucket_overlap(
    name: &str,
    left: &[usize],
    right: &[usize],
    h0_nz: &std::collections::HashSet<(usize, usize)>,
) {
    // **P8.3-followup-B2.** Diagnostic mirrors the new
    // `extract_3x3_from_tensor` zero-pad semantics. Under-sized
    // sectors leave the (i, j) cell empty (recorded as `None`)
    // instead of duplicate-padding with the last available index.
    // The earlier duplicate-pad mode produced spurious `(li, rj)`
    // repeats and inflated/deflated the bucket-hit count.
    let lookup = |v: &[usize], i: usize| -> Option<usize> {
        v.get(i).copied()
    };
    let mut hit = 0usize;
    let mut buckets: Vec<Option<(usize, usize)>> = Vec::with_capacity(9);
    for i in 0..3 {
        for j in 0..3 {
            let bucket = match (lookup(left, i), lookup(right, j)) {
                (Some(li), Some(rj)) => Some((li, rj)),
                _ => None,
            };
            if let Some((li, rj)) = bucket {
                if h0_nz.contains(&(li, rj)) {
                    hit += 1;
                }
            }
            buckets.push(bucket);
        }
    }
    eprintln!(
        "{:<22} buckets={:?}  hit={}/9",
        name, buckets, hit
    );
}

/// Rank of an n×n complex matrix via Hermitian Gram diagonalisation:
/// rank(A) = rank(A† A) = #(eigenvalues > floor * λ_max).
fn matrix_rank(a: &[Complex64], n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    // Build M = A† A.
    let mut m = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = Complex64::new(0.0, 0.0);
            for k in 0..n {
                // (A†)[i,k] = conj(A[k,i])
                acc += a[k * n + i].conj() * a[k * n + j];
            }
            m[i * n + j] = acc;
        }
    }
    let (evs, _) = hermitian_eig(&m, n);
    let max = evs.iter().cloned().fold(0.0f64, f64::max).max(1.0e-300);
    let floor = 1.0e-10;
    evs.iter().filter(|&&e| e / max > floor).count()
}

// -------- tiny n×n complex Hermitian eigensolver (real eigenvalues).
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
