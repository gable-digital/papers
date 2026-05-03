//! Determinism smoke test for P5.10's core Donaldson loop.
//!
//! Goal: catch non-determinism regressions (e.g. a future PR introducing a
//! HashMap iteration in a hot loop, or rayon thread-count sensitivity)
//! before the headline 6.921σ result moves underneath us.
//!
//! Strategy: run the Donaldson solve at the smallest reasonable setting
//! for both candidates (TY at k=3, Schoen at k=3) at a single seed, and
//! assert that `(σ_final, σ_fs_identity, final_donaldson_residual,
//! iterations_run)` match a checked-in golden file BIT-EXACTLY (`f64 ==
//! f64`, no tolerance).
//!
//! Wallclock budget: <30s on a typical dev box. We use n_pts=200 and
//! donaldson_iters=5 to keep the test fast — the bootstrap loop is not
//! exercised here (its determinism is covered by `pwos-math` tests).
//!
//! Golden generation: the FIRST run, if `tests/data/p5_10_determinism_golden.json`
//! is missing, writes the golden and prints a notice. Subsequent runs
//! compare against the file. Commit the golden after generating it.

use std::path::PathBuf;

use cy3_rust_solver::route34::cy3_metric_unified::{
    Cy3MetricSolver, Cy3MetricSpec, SchoenSolver, TianYauSolver,
};
use serde::{Deserialize, Serialize};

/// Golden file format. One record per (candidate, seed) pair; the field
/// names match P5.10's `PerSeedRecord` to make spot-comparison easy.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
struct DeterminismRecord {
    candidate: String,
    seed: u64,
    k: u32,
    n_pts: usize,
    donaldson_iters: usize,
    donaldson_tol: f64,
    n_basis: usize,
    iterations_run: usize,
    sigma_fs_identity_bits: u64,
    sigma_final_bits: u64,
    final_donaldson_residual_bits: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
struct DeterminismGolden {
    /// Set on the first generation; informational only.
    note: String,
    records: Vec<DeterminismRecord>,
}

fn run_ty(
    seed: u64,
    n_pts: usize,
    iters: usize,
    tol: f64,
) -> DeterminismRecord {
    let solver = TianYauSolver;
    let spec = Cy3MetricSpec::TianYau {
        k: 3,
        n_sample: n_pts,
        max_iter: iters,
        donaldson_tol: tol,
        seed,
    };
    let r = solver
        .solve_metric(&spec)
        .expect("TY solve must succeed at minimal settings");
    let s = r.summary();
    let sigma_fs_identity = r.sigma_fs_identity();
    DeterminismRecord {
        candidate: "TY".to_string(),
        seed,
        k: 3,
        n_pts,
        donaldson_iters: iters,
        donaldson_tol: tol,
        n_basis: s.n_basis,
        iterations_run: s.iterations_run,
        sigma_fs_identity_bits: sigma_fs_identity.to_bits(),
        sigma_final_bits: s.final_sigma_residual.to_bits(),
        final_donaldson_residual_bits: s.final_donaldson_residual.to_bits(),
    }
}

fn run_schoen(
    seed: u64,
    n_pts: usize,
    iters: usize,
    tol: f64,
) -> DeterminismRecord {
    // (3,3,1) for k=3 — same mapping as P5.10's `schoen_tuple_for_k`.
    let solver = SchoenSolver;
    let spec = Cy3MetricSpec::Schoen {
        d_x: 3,
        d_y: 3,
        d_t: 1,
        n_sample: n_pts,
        max_iter: iters,
        donaldson_tol: tol,
        seed,
    };
    let r = solver
        .solve_metric(&spec)
        .expect("Schoen solve must succeed at minimal settings");
    let s = r.summary();
    let sigma_fs_identity = r.sigma_fs_identity();
    DeterminismRecord {
        candidate: "Schoen".to_string(),
        seed,
        k: 3,
        n_pts,
        donaldson_iters: iters,
        donaldson_tol: tol,
        n_basis: s.n_basis,
        iterations_run: s.iterations_run,
        sigma_fs_identity_bits: sigma_fs_identity.to_bits(),
        sigma_final_bits: s.final_sigma_residual.to_bits(),
        final_donaldson_residual_bits: s.final_donaldson_residual.to_bits(),
    }
}

/// Resolve the golden file path relative to this test source.
fn golden_path() -> PathBuf {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir.join("tests").join("data").join("p5_10_determinism_golden.json")
}

#[test]
fn p5_10_determinism_is_bit_exact() {
    // Minimal-cost configuration. n_pts=200 keeps each solve under a
    // second on a typical dev box; iters=5 caps total work even if the
    // tolerance is never reached.
    let n_pts: usize = 200;
    let iters: usize = 5;
    let tol: f64 = 1.0e-6;
    // Seed 42 is the head of P5.7/P5.10's SEEDS_20; using it ties this
    // test directly to one of the seeds in the production ensemble.
    let seed: u64 = 42;

    let observed = vec![
        run_ty(seed, n_pts, iters, tol),
        run_schoen(seed, n_pts, iters, tol),
    ];

    let path = golden_path();
    if !path.exists() {
        // First-time generation: write the golden and notify the dev.
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create tests/data");
        }
        let golden = DeterminismGolden {
            note:
                "Auto-generated by test_p5_10_determinism::p5_10_determinism_is_bit_exact. \
                 Bit-exact f64 representations of (sigma_fs_identity, sigma_final, \
                 final_donaldson_residual). Commit after generation."
                    .to_string(),
            records: observed.clone(),
        };
        let json = serde_json::to_string_pretty(&golden)
            .expect("serialise golden");
        std::fs::write(&path, json).expect("write golden");
        eprintln!(
            "\n*** Generated {} on first run. Commit it so future runs can verify determinism. ***\n",
            path.display()
        );
        return;
    }

    // Compare against the checked-in golden.
    let raw = std::fs::read_to_string(&path).expect("read golden");
    let golden: DeterminismGolden =
        serde_json::from_str(&raw).expect("parse golden");

    assert_eq!(
        observed.len(),
        golden.records.len(),
        "record count mismatch — golden has {}, observed {}; regenerate after deliberate change",
        golden.records.len(),
        observed.len()
    );
    for (i, (o, g)) in observed.iter().zip(golden.records.iter()).enumerate() {
        assert_eq!(
            o, g,
            "DETERMINISM REGRESSION at record {} ({} seed={}): observed={:#?} golden={:#?}",
            i, o.candidate, o.seed, o, g
        );
    }
}
