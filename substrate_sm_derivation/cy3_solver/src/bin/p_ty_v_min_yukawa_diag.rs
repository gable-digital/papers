//! Cycle 4 — TY V_min Yukawa-pipeline diagnostic.
//!
//! Wires the H2-cycle V_min monad
//!
//! ```text
//!   B = O(0,0)² ⊕ O(1,0) ⊕ O(0,1)        (rank 4)
//!   C = O(1,1)                            (rank 1)
//!   V = ker(B → C),  rank V = 3,  c_1(V) = 0,  ∫c_3(V) = -18
//! ```
//!
//! into the existing Yukawa pipeline (TY/Z3 metric, Wilson Z/3) and
//! reports T-tensor sparsity, sector assignment, and per-sector
//! bucket-hit counts at the lowest-eigenvalue Higgs slice h_0. Mirrors
//! the layout of `p8_3_followup_a2_tensor_sparsity_diag.rs` so the
//! V_min vs AKLP comparison can be read side-by-side.
//!
//! Hypothesis (cycle 4):  V_min produces a balanced 3-generation
//! Yukawa pipeline with non-degenerate Y_u, Y_d, Y_e bucket-hits
//! comparable to Schoen's 3-factor bundle. Falsification: bucket-hits
//! remain degenerate (≤ 2/9 across all sectors) or numerical
//! instability prevents convergence.
//!
//! This binary does NOT modify production code.
//!
//! Run:
//! ```bash
//! cargo run --release --features "gpu precision-bigfloat" \
//!   --bin p_ty_v_min_yukawa_diag
//! ```

use cy3_rust_solver::route34::cy3_metric_unified::{
    Cy3MetricResultKind, Cy3MetricSolver, Cy3MetricSpec, TianYauSolver,
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
    eprintln!("=== Cycle 4 TY V_min Yukawa diagnostic ===");
    eprintln!(
        "n_pts={}, seed={}, k={}, max_iter={}, donaldson_tol={:.0e}",
        N_PTS, SEED, K, MAX_ITER, DONALDSON_TOL
    );
    eprintln!("MAG_FLOOR (treat |T| <= floor as zero) = {:.0e}", MAG_FLOOR);
    eprintln!();

    let out_dir = Path::new("output");
    if !out_dir.exists() {
        create_dir_all(out_dir).expect("create output dir");
    }

    eprintln!("==================================================================");
    eprintln!("TY/Z3 + V_min (k={})", K);
    eprintln!("==================================================================");
    let v_min_summary = run_one(
        "TY/Z3+V_min",
        Cy3MetricSpec::TianYau {
            k: K,
            n_sample: N_PTS,
            max_iter: MAX_ITER,
            donaldson_tol: DONALDSON_TOL,
            seed: SEED,
        },
        Box::new(TianYauSolver),
        MonadBundle::tian_yau_z3_v_min(),
        AmbientCY3::tian_yau_upstairs(),
        WilsonLineE8::canonical_e8_to_e6_su3(3),
        9,
        out_dir.join("p_ty_v_min_tensor_TY.csv"),
    );

    eprintln!();
    eprintln!("==================================================================");
    eprintln!("TY/Z3 + V_min2 (cycle 6, k={})", K);
    eprintln!("==================================================================");
    let v_min2_summary = run_one(
        "TY/Z3+V_min2",
        Cy3MetricSpec::TianYau {
            k: K,
            n_sample: N_PTS,
            max_iter: MAX_ITER,
            donaldson_tol: DONALDSON_TOL,
            seed: SEED,
        },
        Box::new(TianYauSolver),
        MonadBundle::tian_yau_z3_v_min2(),
        AmbientCY3::tian_yau_upstairs(),
        WilsonLineE8::canonical_e8_to_e6_su3(3),
        9,
        out_dir.join("p_ty_v_min_tensor_V_min2.csv"),
    );

    eprintln!();
    eprintln!("==================================================================");
    eprintln!("TY/Z3 + AKLP (baseline, k={})", K);
    eprintln!("==================================================================");
    let aklp_summary = run_one(
        "TY/Z3+AKLP",
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
        out_dir.join("p_ty_v_min_tensor_AKLP.csv"),
    );

    eprintln!();
    eprintln!("==================================================================");
    eprintln!("=== Cycle 6: V_min vs V_min2 vs AKLP comparison (TY/Z3) ===");
    eprintln!("==================================================================");
    let header = format!(
        "{:<14} {:>8} {:>8} {:>10} {:>12} {:>12} {:>12}",
        "label", "n_modes", "n_nz", "sparsity", "max|T|", "min_nz|T|", "mean_nz|T|"
    );
    eprintln!("{}", header);
    for s in [&v_min_summary, &v_min2_summary, &aklp_summary] {
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
    eprintln!("Bucket hit summary at h_0 (V_min vs V_min2 vs AKLP):");
    eprintln!(
        "  Bundle  | Y_u | Y_d | Y_e | Total"
    );
    eprintln!(
        "  --------|-----|-----|-----|------"
    );
    eprintln!(
        "  AKLP    | {}/9 | {}/9 | {}/9 | {}/27",
        aklp_summary.y_u_hits,
        aklp_summary.y_d_hits,
        aklp_summary.y_e_hits,
        aklp_summary.y_u_hits + aklp_summary.y_d_hits + aklp_summary.y_e_hits
    );
    eprintln!(
        "  V_min   | {}/9 | {}/9 | {}/9 | {}/27",
        v_min_summary.y_u_hits,
        v_min_summary.y_d_hits,
        v_min_summary.y_e_hits,
        v_min_summary.y_u_hits + v_min_summary.y_d_hits + v_min_summary.y_e_hits
    );
    eprintln!(
        "  V_min2  | {}/9 | {}/9 | {}/9 | {}/27",
        v_min2_summary.y_u_hits,
        v_min2_summary.y_d_hits,
        v_min2_summary.y_e_hits,
        v_min2_summary.y_u_hits + v_min2_summary.y_d_hits + v_min2_summary.y_e_hits
    );
    eprintln!();
    let v_min2_total =
        v_min2_summary.y_u_hits + v_min2_summary.y_d_hits + v_min2_summary.y_e_hits;
    let v_min2_balanced = v_min2_summary.y_u_hits >= 3
        && v_min2_summary.y_d_hits >= 3
        && v_min2_summary.y_e_hits >= 3;
    if v_min2_balanced && v_min2_total >= 9 {
        eprintln!(
            "CYCLE 6 VERDICT: V_min2 hypothesis SUPPORTED — bucket-hits {}/27 \
             with each sector ≥ 3/9. V_min2 viable as third BF channel.",
            v_min2_total
        );
    } else {
        eprintln!(
            "CYCLE 6 VERDICT: V_min2 hypothesis FALSIFIED — bucket-hits {}/27 \
             (Y_u={}/9, Y_d={}/9, Y_e={}/9). Per-sector ≥3 floor not cleared.",
            v_min2_total,
            v_min2_summary.y_u_hits,
            v_min2_summary.y_d_hits,
            v_min2_summary.y_e_hits
        );
    }
    eprintln!();
    eprintln!("CSV outputs:");
    eprintln!("  output/p_ty_v_min_tensor_TY.csv");
    eprintln!("  output/p_ty_v_min_tensor_V_min2.csv");
    eprintln!("  output/p_ty_v_min_tensor_AKLP.csv");
}

struct Summary {
    label: String,
    n_modes: usize,
    n_nonzero: usize,
    sparsity_ratio: f64,
    max_abs: f64,
    min_nz_abs: f64,
    mean_nz_abs: f64,
    y_u_hits: usize,
    y_d_hits: usize,
    y_e_hits: usize,
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
            y_u_hits: 0,
            y_d_hits: 0,
            y_e_hits: 0,
        };
    }

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

    eprintln!();
    eprintln!("--- Per-Higgs-slice rank (n×n (i,j) submatrix at fixed k) ---");
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

    eprintln!();
    eprintln!("--- Mode eigenvalues (ascending order from solver) ---");
    for (i, m) in modes.modes.iter().enumerate() {
        eprintln!("  λ[{:>2}] = {:.6e}", i, m.eigenvalue);
    }

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

    eprintln!();
    eprintln!("--- Assignment vs sparsity cross-reference ---");
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
    let y_u_hits = bucket_overlap(
        "Y_u (up x up)",
        &sectors.up_quark,
        &sectors.up_quark,
        &h0_slice_nz_set,
    );
    let y_d_hits = bucket_overlap(
        "Y_d (up x down)",
        &sectors.up_quark,
        &sectors.down_quark,
        &h0_slice_nz_set,
    );
    let y_e_hits = bucket_overlap(
        "Y_e (lepton x lepton)",
        &sectors.lepton,
        &sectors.lepton,
        &h0_slice_nz_set,
    );

    eprintln!();
    eprintln!("--- Verdict ({}) ---", label);
    let h0_nz_count = h0_slice_nz.len();
    let h0_rank = if h0 < slice_nz_coords.len() {
        slice_rank(&t, h0)
    } else {
        0
    };
    eprintln!("Non-zero (i,j) entries at h_0 = {}", h0_nz_count);
    eprintln!("Rank of T_{{:,:,h_0}} {}x{} slice = {}", n, n, h0_rank);
    eprintln!(
        "Bucket hits: Y_u={}/9, Y_d={}/9, Y_e={}/9",
        y_u_hits, y_d_hits, y_e_hits
    );
    if h0_rank <= 1 || h0_nz_count <= 3 {
        eprintln!(
            "VERDICT: cause (a) GEOMETRIC sparsity at h_0 — even a perfect bucket assignment \
             cannot lift Y above rank ≤ 1."
        );
    } else if y_u_hits + y_d_hits + y_e_hits >= 18 {
        eprintln!(
            "VERDICT: BALANCED — bucket hits totalling {} ≥ 18/27 with non-degenerate sparsity. \
             V_min Yukawa channel viable.",
            y_u_hits + y_d_hits + y_e_hits
        );
    } else {
        eprintln!(
            "VERDICT: PARTIAL — bucket hits totalling {}/27. V_min did not produce balanced \
             3-generation fills.",
            y_u_hits + y_d_hits + y_e_hits
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
        y_u_hits,
        y_d_hits,
        y_e_hits,
    }
}

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

fn bucket_overlap(
    name: &str,
    left: &[usize],
    right: &[usize],
    h0_nz: &std::collections::HashSet<(usize, usize)>,
) -> usize {
    let lookup = |v: &[usize], i: usize| -> Option<usize> { v.get(i).copied() };
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
    eprintln!("{:<22} buckets={:?}  hit={}/9", name, buckets, hit);
    hit
}

fn matrix_rank(a: &[Complex64], n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut m = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = Complex64::new(0.0, 0.0);
            for k in 0..n {
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
