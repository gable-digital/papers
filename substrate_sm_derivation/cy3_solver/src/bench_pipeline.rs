//! Multi-pass discrimination pipeline orchestrator.
//!
//! Runs three passes in sequence:
//!   1. broad: low-compute sweep over millions of candidates with cheap
//!      topology filters (Chern, DUY slope, Wilson-line, eta, ADE).
//!   2. refine: medium-compute refinement with multi-resolution Donaldson
//!      and importance-sampled points.
//!   3. precision: publication-grade refinement with the real
//!      Monge-Ampere residual + Adam optimizer + Yukawa fermion-mass
//!      loss. Optionally GPU-accelerated when built with --features gpu.
//!
//! Usage:
//!
//!   cargo run --release --bin bench_pipeline -- run-all \
//!       --output-dir ./pipeline_run --candidate TY/Z3 \
//!       --n-broad 1000000 --top-after-broad 10000 --top-after-refine 100
//!
//!   # Sweep the topology family instead of fixing one candidate:
//!   cargo run --release --bin bench_pipeline -- run-all \
//!       --output-dir ./pipeline_run --candidate all
//!
//!   # Resume a crashed run (broad pass only):
//!   cargo run --release --bin bench_pipeline -- broad \
//!       --output-dir ./pipeline_run

extern crate cy3_rust_solver;

use std::path::PathBuf;
use std::time::Duration;

use cy3_rust_solver::heterotic::{heterotic_bundle_loss, CY3TopologicalData};
use cy3_rust_solver::pipeline::{
    iter_broad_sweep_candidates_in_range, promote_to_next_pass, read_pass_results,
    select_for_next_pass, Candidate, ModuliRanges, PassKind, PassRunner, ScoreResult,
    SelectionStrategy,
};
use cy3_rust_solver::quotient::bundle_z3_invariance_loss;
use cy3_rust_solver::topology_filters::{stage1_topological_loss_with_budget, TopologyFamily};
use cy3_rust_solver::yukawa_sectors::yukawa_sector_loss;

#[derive(Debug, Clone)]
struct CliArgs {
    command: String,
    output_dir: PathBuf,
    candidate_short: String,
    n_broad: usize,
    top_after_broad: usize,
    top_after_refine: usize,
    sync_secs: u64,
    h11: Option<usize>,
    h21: Option<usize>,
    n_bundle: usize,
    start_id: u64,
    end_id: Option<u64>,
    use_gpu: bool,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            command: "help".to_string(),
            output_dir: PathBuf::from("./pipeline_run"),
            candidate_short: "TY/Z3".to_string(),
            n_broad: 1_000_000,
            top_after_broad: 10_000,
            top_after_refine: 100,
            sync_secs: 60,
            h11: None,
            h21: None,
            n_bundle: 30,
            start_id: 0,
            end_id: None,
            use_gpu: false,
        }
    }
}

fn parse_args() -> CliArgs {
    let mut args = CliArgs::default();
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.is_empty() {
        args.command = "help".to_string();
        return args;
    }
    args.command = raw[0].clone();
    let mut i = 1;
    while i < raw.len() {
        let arg = &raw[i];
        let next = || -> String { raw[i + 1].clone() };
        match arg.as_str() {
            "--output-dir" => {
                args.output_dir = PathBuf::from(next());
                i += 2;
            }
            "--candidate" => {
                args.candidate_short = next();
                i += 2;
            }
            "--n-broad" => {
                args.n_broad = next().parse().expect("n-broad");
                i += 2;
            }
            "--top-after-broad" => {
                args.top_after_broad = next().parse().expect("top-after-broad");
                i += 2;
            }
            "--top-after-refine" => {
                args.top_after_refine = next().parse().expect("top-after-refine");
                i += 2;
            }
            "--sync-secs" => {
                args.sync_secs = next().parse().expect("sync-secs");
                i += 2;
            }
            "--h11" => {
                args.h11 = Some(next().parse().expect("h11"));
                i += 2;
            }
            "--h21" => {
                args.h21 = Some(next().parse().expect("h21"));
                i += 2;
            }
            "--n-bundle" => {
                args.n_bundle = next().parse().expect("n-bundle");
                i += 2;
            }
            "--start-id" => {
                args.start_id = next().parse().expect("start-id");
                i += 2;
            }
            "--end-id" => {
                args.end_id = Some(next().parse().expect("end-id"));
                i += 2;
            }
            "--gpu" => {
                args.use_gpu = true;
                i += 1;
            }
            "--help" | "-h" => {
                args.command = "help".to_string();
                return args;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    args
}

fn print_help() {
    println!("Multi-pass discrimination pipeline");
    println!();
    println!("Commands:");
    println!("  run-all       Run all three passes (broad -> refine -> precision)");
    println!("  broad         Run only the broad-sweep pass (Pass 1)");
    println!("  refine        Run only the refinement pass (Pass 2)");
    println!("  precision     Run only the precision pass (Pass 3)");
    println!();
    println!("Common flags:");
    println!("  --output-dir DIR        Where to write JSONL + checkpoints (default ./pipeline_run)");
    println!("  --candidate NAME        TY/Z3 | Schoen/Z3xZ3 | all (default TY/Z3)");
    println!("  --n-broad N             Pass-1 candidates per family (default 1,000,000)");
    println!("  --top-after-broad N     Top survivors -> Pass 2 (default 10,000)");
    println!("  --top-after-refine N    Top survivors -> Pass 3 (default 100)");
    println!("  --sync-secs N           Checkpoint sync interval (default 60s)");
    println!("  --start-id N            ID range start (resume support, default 0)");
    println!("  --end-id N              ID range end (default start + n-broad)");
    println!("  --h11 N                 Override Kahler moduli count");
    println!("  --h21 N                 Override complex moduli count");
    println!("  --n-bundle N            Bundle moduli count (default 30)");
    println!("  --gpu                   Enable GPU path on Stage 3 (requires --features gpu build)");
    println!();
    println!("Crash resume: each pass writes JSONL + .checkpoint sibling.");
    println!("Re-run the same command to pick up where it left off.");
}

fn families_from_arg(name: &str) -> Vec<TopologyFamily> {
    if name == "all" {
        TopologyFamily::all().to_vec()
    } else if let Some(f) = TopologyFamily::from_short_name(name) {
        vec![f]
    } else {
        eprintln!("unknown candidate: {name}; valid: TY/Z3 | Schoen/Z3xZ3 | all");
        std::process::exit(1);
    }
}

/// Topology data for a candidate from its fundamental_group label.
///
/// Returns one of two precomputed static instances. CY3TopologicalData
/// is `Copy`, and there are only two relevant fundamental groups in the
/// pipeline (Z3, Z3xZ3), so caching is trivial: a single static OnceLock
/// per group label that lazily fills from the constructor.
fn cy3_topo_for(c: &Candidate) -> CY3TopologicalData {
    use std::sync::OnceLock;
    static TIAN_YAU_Z3: OnceLock<CY3TopologicalData> = OnceLock::new();
    static SCHOEN_Z3XZ3: OnceLock<CY3TopologicalData> = OnceLock::new();
    match c.fundamental_group.as_str() {
        "Z3xZ3" => *SCHOEN_Z3XZ3.get_or_init(CY3TopologicalData::schoen_z3xz3),
        _ => *TIAN_YAU_Z3.get_or_init(CY3TopologicalData::tian_yau_z3),
    }
}

/// Pass-1 broad-sweep scoring: cheap topology filters PLUS the real
/// heterotic-bundle structure constraints (c_1=0, c_3 generations,
/// c_2(V)=c_2(TM) anomaly, polystability slope, E_8 Wilson Z/3
/// quantization, E_6 x SU(3) breaking pattern, Z/3 quotient invariance).
///
/// Budget-aware: cumulative loss is tracked across cost-ordered groups
/// of channels; once the partial sum exceeds `budget` the remaining
/// channels are filled with a large sentinel (1.0) to ensure the
/// candidate fails the filter. All loss components are non-negative,
/// so the short-circuit is sound.
fn score_broad_sweep(c: &Candidate, budget: f64) -> ScoreResult {
    use cy3_rust_solver::refine::{generation_count_loss, polyhedral_admissibility_loss};

    // Sentinel value to fill skipped channels: large enough to cross
    // any reasonable broad-sweep threshold (typical = 2.0).
    const SKIPPED: f64 = 1.0;

    let mut out = ScoreResult {
        loss_ricci: SKIPPED,
        loss_polyhedral: SKIPPED,
        loss_generation: SKIPPED,
        loss_coulomb: SKIPPED,
        loss_weak: SKIPPED,
        loss_strong: SKIPPED,
        loss_pdg_chi2: 0.0,
        loss_ckm_unitarity: 0.0,
    };
    let mut acc = 0.0;

    // Group A: cheapest discriminative filters (Wilson, Chern, eta,
    // ADE, slope) via the budget-aware Stage-1 walker. The walker
    // itself is budget-pruning; passing the full budget lets it walk
    // through the full filter chain only when needed.
    let topo = stage1_topological_loss_with_budget(c, 3, budget);

    // Group B: O(1) topological + O(generation) integer math --
    // generation_count_loss + polyhedral_admissibility_loss. These are
    // essentially free and load loss_polyhedral / loss_generation
    // sub-channels.
    let poly_admiss = polyhedral_admissibility_loss(&c.fundamental_group);
    let gen_count = generation_count_loss(c.euler_characteristic);

    out.loss_polyhedral = poly_admiss + topo.ade;
    acc += out.loss_polyhedral;
    if acc > budget {
        return out;
    }

    out.loss_weak = topo.wilson_line; // Wilson-line only (no z3_inv yet)
    acc += out.loss_weak;
    if acc > budget {
        return out;
    }

    // Group C: heterotic bundle structure -- monad Chern classes,
    // anomaly cancellation, polystability slope, E_8 Wilson + breaking.
    // These are O(n_b^3) at worst (c_3 triple sum) but small constants.
    let cy3 = cy3_topo_for(c);
    let bundle = heterotic_bundle_loss(&c.bundle_moduli, &cy3);

    out.loss_generation = gen_count + bundle.c1_zero + bundle.c3_three_gen;
    acc += out.loss_generation;
    if acc > budget {
        return out;
    }

    out.loss_ricci = bundle.anomaly;
    acc += out.loss_ricci;
    if acc > budget {
        return out;
    }

    out.loss_coulomb = bundle.wilson_z3 + bundle.wilson_breaking;
    acc += out.loss_coulomb;
    if acc > budget {
        return out;
    }

    // Group D: Z/3 invariance + slope (loosest filters; run last).
    let z3_inv = bundle_z3_invariance_loss(&c.bundle_moduli);
    out.loss_weak = topo.wilson_line + z3_inv;
    out.loss_strong = bundle.polystability + topo.slope;

    out
}

/// Pass-2 refine scoring: multi-resolution Donaldson with early-abort,
/// the real Monge-Ampere residual, and metric-aware forward-models.
///
/// Budget-aware: O(1) polyhedral + generation losses are computed first
/// and the cumulative sum is checked. If those alone exceed the budget,
/// the entire Donaldson solve and metric-aware forward-models are
/// skipped (large sentinel filled in remaining channels). For 1M-row
/// broad-sweep survivors many of which fail the cheap filters, this
/// avoids burning ~ms-per-candidate of Donaldson balancing on
/// candidates that cannot pass anyway.
fn score_refine(c: &Candidate, budget: f64) -> ScoreResult {
    use cy3_rust_solver::kernels::{
        donaldson_solve_in_place_ext, DonaldsonOutcome, DonaldsonSolveOpts,
    };
    use cy3_rust_solver::refine::{
        coulomb_alpha_loss, generation_count_loss, h_spectrum, h_spectrum_max_only,
        polyhedral_admissibility_loss, strong_lambda_loss, weak_mass_loss,
    };
    use cy3_rust_solver::workspace::{DiscriminationWorkspace, N_BASIS_DEGREE2};

    const SKIPPED: f64 = 100.0;
    let mut out = ScoreResult {
        loss_ricci: SKIPPED,
        loss_polyhedral: SKIPPED,
        loss_generation: SKIPPED,
        loss_coulomb: SKIPPED,
        loss_weak: SKIPPED,
        loss_strong: SKIPPED,
        loss_pdg_chi2: 0.0,
        loss_ckm_unitarity: 0.0,
    };
    let mut acc = 0.0;

    // Group A: O(1) topology lookups -- polyhedral admissibility +
    // generation count. Bail before allocating any workspace if these
    // alone exceed the budget.
    out.loss_polyhedral = polyhedral_admissibility_loss(&c.fundamental_group);
    out.loss_generation = generation_count_loss(c.euler_characteristic);
    acc += out.loss_polyhedral + out.loss_generation;
    if acc > budget {
        return out;
    }

    // Group B: heavy Donaldson solve. Only enter this path if Group A
    // did not already exhaust the budget. The remaining budget seeds
    // the Donaldson trajectory-extrapolation early-abort: if the
    // projected asymptotic residual already exceeds what would let the
    // candidate pass the threshold, the solve quits.
    let mut ws = DiscriminationWorkspace::new(2_000, N_BASIS_DEGREE2, 16, 20, 8);
    cy3_rust_solver::sample_points_into(&mut ws, c.id);
    cy3_rust_solver::evaluate_section_basis_into(&mut ws);

    let donaldson_budget = (budget - acc).max(0.0);
    let outcome = donaldson_solve_in_place_ext(
        &mut ws,
        1e-3,
        DonaldsonSolveOpts {
            divergence_factor: 2.5,
            divergence_window: 3,
            warm_start: false,
            checkpoint_path: None,
            checkpoint_every: 5,
            extrapolate_budget: donaldson_budget,
            extrapolate_w_ricci: 1.0,
            extrapolate_min_iters: 4,
        },
    );
    let ricci_residual = ws.residuals.last().copied().unwrap_or(1.0);
    let diverged_penalty = if outcome == DonaldsonOutcome::Diverged {
        10.0
    } else {
        0.0
    };
    out.loss_ricci = ricci_residual * ricci_residual + diverged_penalty;
    acc += out.loss_ricci;
    if acc > budget {
        return out;
    }

    // Group C: metric-aware forward-models for gauge-coupling channels.
    // Compute only the max eigenvalue first (cheap power iteration);
    // the full spectrum (with min + gap) is required only for weak_mass.
    let spec_max = h_spectrum_max_only(&ws.h, ws.n_basis);

    let em_sector_norm = bundle_norm_slice(c, 5, 5);
    let weak_sector_norm = bundle_norm_slice(c, 10, 5);
    let qcd_sector_norm = bundle_norm_slice(c, 15, 5);

    out.loss_coulomb = coulomb_alpha_loss(em_sector_norm, spec_max.max);
    acc += out.loss_coulomb;
    if acc > budget {
        return out;
    }

    out.loss_strong = strong_lambda_loss(qcd_sector_norm, spec_max.max);
    acc += out.loss_strong;
    if acc > budget {
        return out;
    }

    // Only compute the full spectrum (min + gap) if weak_mass channel
    // is still in play.
    let spec_full = h_spectrum(&ws.h, ws.n_basis);
    out.loss_weak = weak_mass_loss(weak_sector_norm, spec_full.gap);
    out
}

/// Pass-3 precision scoring: real Monge-Ampere residual + Yukawa loss.
/// Optionally GPU-accelerated for the Donaldson balancing phase. Long
/// runs are checkpointed per-candidate so a SIGKILL doesn't lose work.
///
/// Budget-aware: O(1) topology lookups are computed first; if their sum
/// already exceeds the budget, the heavy Donaldson + Monge-Ampere +
/// Yukawa pipeline is skipped entirely. For Pass-3 the threshold is
/// typically 0.05, so a candidate whose polyhedral + generation losses
/// alone hit that budget is dropped before any expensive computation.
fn score_precision_with_checkpoint_dir(
    c: &Candidate,
    use_gpu: bool,
    checkpoint_dir: Option<&std::path::Path>,
    budget: f64,
) -> ScoreResult {
    use cy3_rust_solver::kernels::{
        donaldson_solve_multires_with_checkpoint, init_yukawa_centers, yukawa_tensor_in_place,
        DonaldsonOutcome,
    };
    use cy3_rust_solver::refine::{
        coulomb_alpha_loss, degree_k_monomials, evaluate_section_basis_with_derivs,
        generation_count_loss, h_spectrum, importance_weights,
        monge_ampere_residual_weighted_budget, polyhedral_admissibility_loss,
        strong_lambda_loss, weak_mass_loss, yukawa_fermion_mass_loss,
    };
    use cy3_rust_solver::workspace::{DiscriminationWorkspace, N_BASIS_DEGREE2};

    // Group A: O(1) topology checks. Bail before any allocation if the
    // candidate already fails on these alone.
    let poly_loss = polyhedral_admissibility_loss(&c.fundamental_group);
    let gen_loss = generation_count_loss(c.euler_characteristic);
    if poly_loss + gen_loss > budget {
        const SKIPPED: f64 = 100.0;
        return ScoreResult {
            loss_ricci: SKIPPED,
            loss_polyhedral: poly_loss,
            loss_generation: gen_loss,
            loss_coulomb: SKIPPED,
            loss_weak: SKIPPED,
            loss_strong: SKIPPED,
            loss_pdg_chi2: 0.0,
            loss_ckm_unitarity: 0.0,
        };
    }

    let mut ws = DiscriminationWorkspace::new(50_000, N_BASIS_DEGREE2, 16, 50, 12);
    cy3_rust_solver::sample_points_into(&mut ws, c.id);
    cy3_rust_solver::evaluate_section_basis_into(&mut ws);
    init_yukawa_centers(&mut ws, c.id.wrapping_add(7));

    // GPU path: when enabled, route the heavy Donaldson balancing to a
    // thread-local GPU workspace, copy the balanced h back to CPU, and
    // continue with CPU-side Monge-Ampere + Yukawa fermion-mass loss.
    let gpu_ok = if use_gpu {
        gpu_donaldson_for_candidate(c.id, &mut ws)
    } else {
        false
    };

    let outcome = if gpu_ok {
        DonaldsonOutcome::Converged
    } else {
        let ckpt = checkpoint_dir.map(|d| d.join(format!("p3_{}.ckpt", c.id)));
        donaldson_solve_multires_with_checkpoint(&mut ws, &[2, 3], 1e-5, 2.0, ckpt)
    };
    if outcome == DonaldsonOutcome::Diverged {
        return ScoreResult {
            loss_ricci: 100.0,
            loss_polyhedral: 0.0,
            loss_generation: 0.0,
            loss_coulomb: 0.0,
            loss_weak: 0.0,
            loss_strong: 0.0,
            loss_pdg_chi2: 0.0,
            loss_ckm_unitarity: 0.0,
        };
    }

    // Real Monge-Ampere residual: build per-point derivatives at a
    // smaller subsample (residual computation is O(n*36*n_basis^2)
    // and dominates wall-clock past ~5K points).
    let n_subsample = ws.n_points.min(5_000);
    let monomials = degree_k_monomials(2); // cached &'static, no per-call build
    let mut subsample_section = vec![0.0; n_subsample * ws.n_basis];
    let mut subsample_first = vec![0.0; n_subsample * 8 * ws.n_basis];
    let mut subsample_second = vec![0.0; n_subsample * 36 * ws.n_basis];
    for p in 0..n_subsample {
        let z = [
            ws.points[p * 8],
            ws.points[p * 8 + 1],
            ws.points[p * 8 + 2],
            ws.points[p * 8 + 3],
            ws.points[p * 8 + 4],
            ws.points[p * 8 + 5],
            ws.points[p * 8 + 6],
            ws.points[p * 8 + 7],
        ];
        let (s, ds, dds) = evaluate_section_basis_with_derivs(&z, monomials);
        subsample_section[p * ws.n_basis..(p + 1) * ws.n_basis].copy_from_slice(&s);
        subsample_first[p * 8 * ws.n_basis..(p + 1) * 8 * ws.n_basis].copy_from_slice(&ds);
        subsample_second[p * 36 * ws.n_basis..(p + 1) * 36 * ws.n_basis].copy_from_slice(&dds);
    }
    // Take only the first n_subsample point rows from ws.points.
    let subsample_points = &ws.points[..n_subsample * 8];
    // Importance weights from the section distribution: high-|s|^2
    // points get more weight, concentrating sample effort where the
    // metric varies fastest.
    let weights = importance_weights(&subsample_section, n_subsample, ws.n_basis);

    // Budget-aware Monge-Ampere with sequential early-abort. The
    // budget passed here is the precision-pass filter threshold; the
    // residual-computation function itself decides when to short-circuit.
    // w_ricci = 1.0 here because score_precision packs ricci + 0.5*yukawa
    // into loss_ricci (we leave headroom for the yukawa addend by using
    // 0.7 * budget for the Monge-Ampere abort threshold).
    let ricci = monge_ampere_residual_weighted_budget(
        subsample_points,
        &subsample_section,
        &subsample_first,
        &subsample_second,
        &ws.h,
        &weights,
        n_subsample,
        ws.n_basis,
        1.0,            // w_ricci: residual lands in loss_ricci with weight 1
        budget * 0.7,   // leave 30% of budget for downstream yukawa + other channels
        2.0,            // slack: heuristic 2-sigma sampling-noise band
    );

    // Yukawa fermion-mass loss: separated u/d/lepton sectors + CKM
    // mixing-angle loss (requires n_modes >= 9 for sector separation).
    yukawa_tensor_in_place(&mut ws);
    let yukawa_lump = yukawa_fermion_mass_loss(&ws.yukawa_tensor, ws.n_modes);
    let yukawa_sectors = yukawa_sector_loss(&ws.yukawa_tensor, ws.n_modes);
    let yukawa = yukawa_lump + yukawa_sectors.total();

    let spec = h_spectrum(&ws.h, ws.n_basis);
    let em_sector_norm = bundle_norm_slice(c, 5, 5);
    let weak_sector_norm = bundle_norm_slice(c, 10, 5);
    let qcd_sector_norm = bundle_norm_slice(c, 15, 5);

    // Pack Yukawa loss into the previously-unused `loss_strong` extra
    // slot? No -- ScoreResult is fixed shape. Add it into loss_ricci
    // as a weighted addend (the score-result struct doesn't carry a
    // dedicated yukawa channel). Long-term TODO: extend ScoreResult.
    ScoreResult {
        loss_ricci: ricci + 0.5 * yukawa,
        loss_polyhedral: polyhedral_admissibility_loss(&c.fundamental_group),
        loss_generation: generation_count_loss(c.euler_characteristic),
        loss_coulomb: coulomb_alpha_loss(em_sector_norm, spec.max),
        loss_weak: weak_mass_loss(weak_sector_norm, spec.gap),
        loss_strong: strong_lambda_loss(qcd_sector_norm, spec.max),
        loss_pdg_chi2: 0.0,
        loss_ckm_unitarity: 0.0,
    }
}

/// GPU-accelerated Donaldson balancing for one candidate. Returns true
/// if the GPU path was used and `ws.h` is now populated with the
/// balanced metric; false to fall back to CPU.
#[cfg(feature = "gpu")]
fn gpu_donaldson_for_candidate(
    candidate_id: u64,
    ws: &mut cy3_rust_solver::workspace::DiscriminationWorkspace,
) -> bool {
    use cy3_rust_solver::gpu::{gpu_discriminate, GpuDiscriminationWorkspace};
    use cy3_rust_solver::workspace::N_BASIS_DEGREE2;
    use std::cell::RefCell;
    thread_local! {
        static GPU_WS: RefCell<Option<GpuDiscriminationWorkspace>> = const { RefCell::new(None) };
    }
    let ok = GPU_WS.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            match GpuDiscriminationWorkspace::new(50_000, N_BASIS_DEGREE2, 16, 50) {
                Ok(g) => *slot = Some(g),
                Err(e) => {
                    eprintln!("[gpu] init failed, falling back to CPU: {}", e);
                    return false;
                }
            }
        }
        let g = slot.as_mut().unwrap();
        match gpu_discriminate(g, candidate_id, candidate_id.wrapping_add(7), 1e-5, 30) {
            Ok(_) => {
                // Copy balanced h from device back into the CPU workspace.
                let stream = g.stream.clone();
                match stream.memcpy_dtov(&g.d_h) {
                    Ok(h_host) => {
                        ws.h.copy_from_slice(&h_host);
                        true
                    }
                    Err(e) => {
                        eprintln!("[gpu] dtoh failed, falling back: {}", e);
                        false
                    }
                }
            }
            Err(e) => {
                eprintln!("[gpu] discriminate failed, falling back: {}", e);
                false
            }
        }
    });
    ok
}

#[cfg(not(feature = "gpu"))]
fn gpu_donaldson_for_candidate(
    _candidate_id: u64,
    _ws: &mut cy3_rust_solver::workspace::DiscriminationWorkspace,
) -> bool {
    false
}

fn bundle_norm_slice(c: &Candidate, start: usize, count: usize) -> f64 {
    c.bundle_moduli
        .iter()
        .skip(start)
        .take(count)
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
}

fn run_broad(args: &CliArgs) -> std::io::Result<()> {
    std::fs::create_dir_all(&args.output_dir)?;
    let output_path = args.output_dir.join("pass1_broad.jsonl");

    let families = families_from_arg(&args.candidate_short);
    let end_id = args
        .end_id
        .unwrap_or(args.start_id + args.n_broad as u64 - 1);

    // Streaming candidate generation: chain one iterator per family,
    // each yielding candidates lazily. Candidates are materialised
    // per-rayon-task and dropped after scoring -- no upfront Vec<Candidate>
    // allocation across the entire 1M-candidate broad sweep.
    let n_per_family = (end_id + 1 - args.start_id) as u64;
    let total_hint = n_per_family * families.len() as u64;
    println!(
        "[broad] streaming {} candidate(s) per family across {} families ({} total)",
        n_per_family,
        families.len(),
        total_hint,
    );

    let n_bundle = args.n_bundle;
    let start_id = args.start_id;
    let h11_override = args.h11;
    let h21_override = args.h21;

    // Build a chained streaming iterator across families. Each family
    // contributes its own `iter_broad_sweep_candidates_in_range`
    // iterator (now taking owned arguments), and `flat_map` concatenates
    // them lazily.
    let families_vec: Vec<TopologyFamily> = families.clone();
    let stream: Box<dyn Iterator<Item = Candidate> + Send> = Box::new(
        families_vec
            .into_iter()
            .enumerate()
            .flat_map(move |(idx, family)| {
                let id_offset = idx as u64 * n_per_family;
                let h11 = h11_override.unwrap_or(family.h11());
                let h21 = h21_override.unwrap_or(family.h21());
                iter_broad_sweep_candidates_in_range(
                    family.short_name().to_string(),
                    family.euler_characteristic(),
                    family.fundamental_group().to_string(),
                    h11,
                    h21,
                    n_bundle,
                    start_id + id_offset,
                    end_id + id_offset,
                    ModuliRanges::default(),
                    42,
                )
            }),
    );

    let runner = PassRunner::new(
        PassKind::BroadSweep,
        &output_path,
        Duration::from_secs(args.sync_secs),
        2.0,
    );
    let report =
        runner.run_budget_streaming(stream, score_broad_sweep, total_hint)?;
    println!(
        "[broad] complete: {}/{} processed, {} passed filter, {:.2}s, output: {}",
        report.completed,
        report.total_candidates,
        report.passed_filter,
        report.elapsed_secs,
        report.output_path.display()
    );
    Ok(())
}

fn run_refine(args: &CliArgs) -> std::io::Result<()> {
    let input_path = args.output_dir.join("pass1_broad.jsonl");
    let output_path = args.output_dir.join("pass2_refine.jsonl");

    println!("[refine] reading {} ...", input_path.display());
    let pass1_results = read_pass_results(&input_path)?;
    println!("[refine] read {} pass-1 results", pass1_results.len());

    let strategy = SelectionStrategy::TopK(args.top_after_broad);
    let selected = select_for_next_pass(&pass1_results, &strategy);
    println!("[refine] selected {} survivors ({})", selected.len(), strategy.describe());

    let candidates = promote_to_next_pass(&selected, 0);

    let runner = PassRunner::new(
        PassKind::Refine,
        &output_path,
        Duration::from_secs(args.sync_secs),
        0.5,
    );
    let report = runner.run_budget(candidates, score_refine)?;
    println!(
        "[refine] complete: {}/{} processed, {} passed filter, {:.2}s, output: {}",
        report.completed,
        report.total_candidates,
        report.passed_filter,
        report.elapsed_secs,
        report.output_path.display()
    );
    Ok(())
}

fn run_precision(args: &CliArgs) -> std::io::Result<()> {
    let input_path = args.output_dir.join("pass2_refine.jsonl");
    let output_path = args.output_dir.join("pass3_precision.jsonl");

    println!("[precision] reading {} ...", input_path.display());
    let pass2_results = read_pass_results(&input_path)?;
    println!("[precision] read {} pass-2 results", pass2_results.len());

    let strategy = SelectionStrategy::TopK(args.top_after_refine);
    let selected = select_for_next_pass(&pass2_results, &strategy);
    println!(
        "[precision] selected {} survivors ({})",
        selected.len(),
        strategy.describe()
    );

    let candidates = promote_to_next_pass(&selected, 0);

    let runner = PassRunner::new(
        PassKind::Precision,
        &output_path,
        Duration::from_secs(args.sync_secs),
        0.05,
    );
    let use_gpu = args.use_gpu;
    if use_gpu {
        println!("[precision] --gpu enabled (Stage 3 will run with GPU-accelerated Donaldson where available)");
    }
    let ckpt_dir = args.output_dir.join("p3_ckpts");
    let _ = std::fs::create_dir_all(&ckpt_dir);
    let report = runner.run_budget(candidates, move |c, budget| {
        score_precision_with_checkpoint_dir(c, use_gpu, Some(&ckpt_dir), budget)
    })?;
    println!(
        "[precision] complete: {}/{} processed, {} passed filter, {:.2}s, output: {}",
        report.completed,
        report.total_candidates,
        report.passed_filter,
        report.elapsed_secs,
        report.output_path.display()
    );
    Ok(())
}

fn run_all(args: &CliArgs) -> std::io::Result<()> {
    run_broad(args)?;
    run_refine(args)?;
    run_precision(args)?;
    Ok(())
}

fn main() {
    let args = parse_args();
    match args.command.as_str() {
        "help" => print_help(),
        "broad" => run_broad(&args).unwrap(),
        "refine" => run_refine(&args).unwrap(),
        "precision" => run_precision(&args).unwrap(),
        "run-all" => run_all(&args).unwrap(),
        other => {
            eprintln!("unknown command: {other}");
            print_help();
            std::process::exit(1);
        }
    }
}
