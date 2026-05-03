//! P_repro2 — within-process Schoen reproducibility diagnostic.
//!
//! Runs the SAME Schoen seed twice in one binary invocation and
//! diff's the per-iter Donaldson + σ trajectories plus the sample-
//! cloud SHA-256. If the two runs are bit-exact within the same
//! process, the non-determinism is between-process (env-keyed RNG,
//! cross-thread-count rayon work-split, etc.). If they differ even
//! within one process, the non-determinism is within-iteration
//! (par_iter sums, GPU atomics, etc.).
//!
//! Diagnostic only. Does NOT modify production code.

use clap::Parser;
use cy3_rust_solver::route34::cy3_metric_unified::{
    Cy3AdamOverride, Cy3MetricResultKind, Cy3MetricSolver, Cy3MetricSpec, SchoenSolver,
};

#[derive(Parser, Debug)]
#[command(about = "P_repro2 within-process Schoen reproducibility")]
struct Cli {
    #[arg(long, default_value_t = 137)]
    seed: u64,

    #[arg(long, default_value_t = 4_000)]
    n_pts: usize,

    #[arg(long, default_value_t = 30)]
    donaldson_iters: usize,

    #[arg(long, default_value_t = 1.0e-6)]
    donaldson_tol: f64,

    #[arg(long, default_value_t = false)]
    use_gpu: bool,
}

fn run_one(cli: &Cli) -> (String, Vec<f64>, Vec<f64>, usize) {
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
    let solver = SchoenSolver;
    let r = solver
        .solve_metric_with_adam(&spec, &adam)
        .expect("solver failed");
    let res = match r {
        Cy3MetricResultKind::Schoen(b) => *b,
        _ => panic!("unexpected solver result kind"),
    };
    (
        res.run_metadata.sample_cloud_sha256.clone(),
        res.donaldson_history.clone(),
        res.sigma_history.clone(),
        res.iterations_run,
    )
}

fn main() {
    let cli = Cli::parse();
    eprintln!(
        "[repro2] seed={} n_pts={} iters={} tol={} gpu={}",
        cli.seed, cli.n_pts, cli.donaldson_iters, cli.donaldson_tol, cli.use_gpu
    );

    let (s1, d1, sg1, it1) = run_one(&cli);
    eprintln!(
        "[repro2] run1: cloud_sha={} iters_run={} donaldson_history.len={}",
        &s1[..16.min(s1.len())],
        it1,
        d1.len()
    );

    let (s2, d2, sg2, it2) = run_one(&cli);
    eprintln!(
        "[repro2] run2: cloud_sha={} iters_run={} donaldson_history.len={}",
        &s2[..16.min(s2.len())],
        it2,
        d2.len()
    );

    let cloud_match = s1 == s2;
    let iters_match = it1 == it2;
    let dlen_match = d1.len() == d2.len();
    let mut donaldson_bit_exact = dlen_match;
    let mut sigma_bit_exact = sg1.len() == sg2.len();
    let mut max_d_diff = 0.0_f64;
    let mut max_s_diff = 0.0_f64;
    if dlen_match {
        for (a, b) in d1.iter().zip(d2.iter()) {
            if a.to_bits() != b.to_bits() {
                donaldson_bit_exact = false;
            }
            let dd = (a - b).abs();
            if dd.is_finite() && dd > max_d_diff {
                max_d_diff = dd;
            }
        }
    }
    if sg1.len() == sg2.len() {
        for (a, b) in sg1.iter().zip(sg2.iter()) {
            if a.to_bits() != b.to_bits() {
                sigma_bit_exact = false;
            }
            let dd = (a - b).abs();
            if dd.is_finite() && dd > max_s_diff {
                max_s_diff = dd;
            }
        }
    }

    println!("=== P_repro2 within-process determinism summary ===");
    println!("seed                      = {}", cli.seed);
    println!("n_pts                     = {}", cli.n_pts);
    println!("max_iter                  = {}", cli.donaldson_iters);
    println!("donaldson_tol             = {}", cli.donaldson_tol);
    println!("use_gpu                   = {}", cli.use_gpu);
    println!("sample_cloud_sha matches  = {}", cloud_match);
    println!("iterations_run matches    = {} ({} vs {})", iters_match, it1, it2);
    println!("donaldson_history len eq  = {}", dlen_match);
    println!("donaldson_history bit-exact = {}", donaldson_bit_exact);
    println!("sigma_history bit-exact   = {}", sigma_bit_exact);
    println!("max |Δ residual|          = {:.6e}", max_d_diff);
    println!("max |Δ sigma|             = {:.6e}", max_s_diff);

    if dlen_match {
        let n = d1.len();
        let show = n.min(8);
        println!("first {} iter residuals (run1 / run2):", show);
        for i in 0..show {
            println!(
                "  iter {:>3}  {:.10e}  {:.10e}  diff={:.3e}",
                i,
                d1[i],
                d2[i],
                (d1[i] - d2[i]).abs()
            );
        }
        if n > show {
            println!("  ... ({} more)", n - show);
        }
    } else {
        println!(
            "donaldson_history lengths differ: run1={} run2={}",
            d1.len(),
            d2.len()
        );
    }
}
