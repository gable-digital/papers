//! P8.4-followup — Donaldson trajectory diagnostic (THROWAWAY).
//!
//! Runs ONE Schoen seed at a chosen k and dumps the per-iteration
//! Donaldson residual + σ trajectory as JSONL. Used to identify whether
//! k=4 stalled seeds (P8.4) hit a smooth slow stall, oscillation, or
//! catastrophic-rebound-then-restore.
//!
//! Production code is NOT modified — this just calls the existing
//! `SchoenSolver` API and reads back `donaldson_history` + `sigma_history`
//! that are already populated for every run.
//!
//! Usage:
//!   p8_4_donaldson_stall_diag --seed 4242 --k 4 --n-pts 15000 \
//!       --output output/p8_4_diag_seed4242_k4.jsonl

use clap::Parser;
use cy3_rust_solver::route34::cy3_metric_unified::{
    Cy3AdamOverride, Cy3MetricResultKind, Cy3MetricSolver, Cy3MetricSpec, SchoenSolver,
};
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "P8.4 Donaldson stall trajectory diagnostic")]
struct Cli {
    #[arg(long)]
    seed: u64,

    #[arg(long, default_value_t = 4)]
    k: u32,

    #[arg(long, default_value_t = 15_000)]
    n_pts: usize,

    #[arg(long, default_value_t = 100)]
    donaldson_iters: usize,

    #[arg(long, default_value_t = 1.0e-6)]
    donaldson_tol: f64,

    #[arg(long, default_value_t = false)]
    use_gpu: bool,

    /// Write per-iter JSONL trajectory here.
    #[arg(long)]
    output: PathBuf,
}

#[derive(Serialize)]
struct IterRecord {
    iter: usize,
    /// L2 step size ‖h_new - h_old‖_F as recorded by the solver. This
    /// is the SAME quantity used for Donaldson convergence.
    residual: f64,
    sigma: f64,
    /// Ratio of this iter's residual to the running min, for spotting
    /// a regression. >1 means we just got worse.
    resid_over_running_min: f64,
}

#[derive(Serialize)]
struct Summary {
    seed: u64,
    k: u32,
    n_pts: usize,
    n_basis: usize,
    iterations_run: usize,
    final_donaldson_residual: f64,
    sigma_final: f64,
    sigma_fs_identity: f64,
    elapsed_s: f64,
    /// argmin index in donaldson_history.
    min_residual_iter: usize,
    min_residual: f64,
    /// peak rebound residual after min. Diagnoses catastrophic blow-up.
    peak_after_min: f64,
    /// returned-via-guard? heuristic: final residual equals min residual
    /// (snapshot was restored) AND iterations_run < requested cap AND
    /// peak after min > 100×min.
    likely_guard_restored: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let spec = Cy3MetricSpec::Schoen {
        d_x: 3,
        d_y: 3,
        d_t: 1,
        n_sample: cli.n_pts,
        max_iter: cli.donaldson_iters,
        donaldson_tol: cli.donaldson_tol,
        seed: cli.seed,
    };

    let adam = Cy3AdamOverride {
        adam_refine: None,
        use_gpu_donaldson: cli.use_gpu,
    };

    eprintln!(
        "[diag] seed={} k={} n_pts={} iters={} tol={} gpu={}",
        cli.seed, cli.k, cli.n_pts, cli.donaldson_iters, cli.donaldson_tol, cli.use_gpu
    );

    let _ = cli.k; // k is fixed at the Schoen bidegree (3,3,1) — kept on
                   // CLI for the caller's bookkeeping; bidegree below is
                   // independent of k. (Schoen's "k" indexes the basis
                   // size implicitly through n_sample budgeting in the
                   // calling P5.10 binary; this diagnostic just uses the
                   // canonical bidegree.)

    let solver = SchoenSolver;
    let t0 = Instant::now();
    let r = solver.solve_metric_with_adam(&spec, &adam)?;
    let elapsed_s = t0.elapsed().as_secs_f64();

    let res = match r {
        Cy3MetricResultKind::Schoen(b) => *b,
        _ => return Err("unexpected solver result kind".into()),
    };

    // Open JSONL writer
    if let Some(parent) = cli.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut f = File::create(&cli.output)?;

    let mut running_min = f64::INFINITY;
    let mut min_idx = 0usize;
    for (i, &resid) in res.donaldson_history.iter().enumerate() {
        if resid < running_min {
            running_min = resid;
            min_idx = i;
        }
        let sig = res.sigma_history.get(i).copied().unwrap_or(f64::NAN);
        let ratio = if running_min > 0.0 && running_min.is_finite() {
            resid / running_min
        } else {
            f64::NAN
        };
        let rec = IterRecord {
            iter: i,
            residual: resid,
            sigma: sig,
            resid_over_running_min: ratio,
        };
        writeln!(f, "{}", serde_json::to_string(&rec)?)?;
    }

    // Compute peak after argmin.
    let mut peak_after_min = 0.0_f64;
    for &r in res.donaldson_history.iter().skip(min_idx + 1) {
        if r.is_finite() && r > peak_after_min {
            peak_after_min = r;
        }
    }

    let summary = Summary {
        seed: cli.seed,
        k: cli.k,
        n_pts: cli.n_pts,
        n_basis: res.n_basis,
        iterations_run: res.iterations_run,
        final_donaldson_residual: res.final_donaldson_residual,
        sigma_final: res.final_sigma_residual,
        sigma_fs_identity: res.sigma_fs_identity,
        elapsed_s,
        min_residual_iter: min_idx,
        min_residual: running_min,
        peak_after_min,
        likely_guard_restored: res.iterations_run < cli.donaldson_iters
            && (res.final_donaldson_residual - running_min).abs()
                < 1.0e-12 * running_min.max(1.0)
            && peak_after_min > 100.0 * running_min,
    };

    let summary_path = cli.output.with_extension("summary.json");
    let mut sf = File::create(&summary_path)?;
    sf.write_all(serde_json::to_string_pretty(&summary)?.as_bytes())?;

    eprintln!(
        "[diag] done: iters_run={} min_resid={:.3e} (iter {}) peak_after_min={:.3e} guard_restored={} elapsed={:.1}s",
        summary.iterations_run,
        summary.min_residual,
        summary.min_residual_iter,
        summary.peak_after_min,
        summary.likely_guard_restored,
        summary.elapsed_s
    );
    eprintln!("[diag] trajectory: {}", cli.output.display());
    eprintln!("[diag] summary:    {}", summary_path.display());

    Ok(())
}
