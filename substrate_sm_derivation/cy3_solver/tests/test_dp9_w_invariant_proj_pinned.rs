//! Regression pin for `p_dp9_w_invariant_proj` — ensures the
//! Z/3×Z/3 invariant projection of the W-twisted Ext¹ shadow on the
//! BHOP-2005 §6 SU(4) extension bundle remains at zero per the
//! definitive (a)-verdict of DP9-W-INVARIANT-PROJ.
//!
//! This test pins the JSON-output schema and the **bottom-line
//! verdict invariants**:
//!
//! 1. Total invariant H¹ lower bound = 0 (lower-bound robustness).
//! 2. Total invariant H¹ upper bound = 0 (SES-split robustness).
//! 3. Verdict starts with "(a)".
//! 4. Per-summand invariant character is at β = 2 (not β = 0), which
//!    is the geometric reason verdict (a) is robust under
//!    BHOP-canonical (g_α=0, g_β=0) lift.
//! 5. Bundle-lift sensitivity at `g_β = 1` does light up Summand A
//!    (showing the answer is sensitive to the BHOP-paper-fixed lift,
//!    NOT to numerical noise) — confirming the verdict is geometric,
//!    not arithmetic-rounding artefact.
//!
//! Failure of any pin requires rederivation of the verdict and the
//! downstream framework-falsification chain.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

fn rust_solver_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

fn run_binary_with_output() -> serde_json::Value {
    // Use a per-test unique output path under the system temp dir so
    // parallel tests don't race and we don't pollute `output/`.
    let tmp_root = std::env::temp_dir();
    std::fs::create_dir_all(&tmp_root).ok();
    let nonce = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let output_path = tmp_root.join(format!(
        "p_dp9_w_invariant_proj_pid{}_n{}.json",
        std::process::id(),
        nonce
    ));

    // Locate the binary. Prefer a pre-built `target-dp9-w-invariant`
    // release binary if present; otherwise fall back to `cargo run
    // --release`.
    let candidate_dirs = [
        rust_solver_root().join("target-dp9-w-invariant").join("release"),
        rust_solver_root().join("target").join("release"),
    ];
    let mut bin_path: Option<PathBuf> = None;
    for d in &candidate_dirs {
        for ext in &["", ".exe"] {
            let p = d.join(format!("p_dp9_w_invariant_proj{}", ext));
            if p.exists() {
                bin_path = Some(p);
                break;
            }
        }
        if bin_path.is_some() {
            break;
        }
    }

    let status = if let Some(bin) = bin_path {
        Command::new(bin)
            .args(["--output", output_path.to_str().unwrap()])
            .current_dir(rust_solver_root())
            .status()
            .expect("run pre-built binary")
    } else {
        Command::new("cargo")
            .args([
                "run",
                "--release",
                "--bin",
                "p_dp9_w_invariant_proj",
                "--",
                "--output",
                output_path.to_str().unwrap(),
            ])
            .current_dir(rust_solver_root())
            .status()
            .expect("cargo run")
    };
    assert!(status.success(), "binary exited non-zero");

    let s = std::fs::read_to_string(&output_path).expect("read output");
    serde_json::from_str(&s).expect("parse output")
}

#[test]
fn dp9_w_invariant_proj_total_lower_bound_is_zero() {
    let v = run_binary_with_output();
    let total_lower = v["total_invariant_h1_dim"].as_i64().expect("i64");
    assert_eq!(
        total_lower, 0,
        "PINNED: total invariant H¹ (lower bound) must be 0 for verdict (a)"
    );
}

#[test]
fn dp9_w_invariant_proj_total_upper_bound_is_zero() {
    let v = run_binary_with_output();
    let total_upper = v["total_invariant_h1_dim_upper"].as_i64().expect("i64");
    assert_eq!(
        total_upper, 0,
        "PINNED: total invariant H¹ (upper bound) must be 0 for ROBUST verdict (a)"
    );
}

#[test]
fn dp9_w_invariant_proj_verdict_is_a() {
    let v = run_binary_with_output();
    let verdict = v["verdict"].as_str().expect("verdict string");
    assert!(
        verdict.starts_with("(a)"),
        "PINNED: verdict must be (a); got: {verdict}"
    );
}

#[test]
fn dp9_w_invariant_proj_summand_a_h1_concentrated_at_beta_two() {
    let v = run_binary_with_output();
    let summands = v["summands"].as_array().expect("summands array");
    let summand_a = &summands[0];
    let chi_h1_l_minus_2f = summand_a["chi_h1_l_minus_2f"]["counts"]
        .as_array()
        .expect("counts array");
    // Layout: counts[3*α + β]. β=0 column = indices 0, 3, 6.
    // β=2 column = indices 2, 5, 8.
    let beta_2_total: i64 = (0..3)
        .map(|alpha| {
            chi_h1_l_minus_2f[3 * alpha + 2]
                .as_i64()
                .expect("entry")
        })
        .sum();
    let beta_0_total: i64 = (0..3)
        .map(|alpha| chi_h1_l_minus_2f[3 * alpha].as_i64().expect("entry"))
        .sum();
    assert_eq!(beta_2_total, 3, "Summand A: H¹ should have 3 modes at β=2");
    assert_eq!(beta_0_total, 0, "Summand A: H¹ must have 0 modes at β=0");
}

#[test]
fn dp9_w_invariant_proj_sensitivity_lights_up_at_g_beta_one() {
    // g_β = 1 shifts β=2 → β=0, lighting up the invariant subspace.
    // This is sensitivity to BHOP-paper-fixed equivariant lift, NOT to
    // numerical noise — the answer is geometric.
    let v = run_binary_with_output();
    let sens = v["sensitivity"].as_array().expect("sensitivity array");
    let summand_a_table = sens[0]["invariant_h1_dim_table"]
        .as_array()
        .expect("table");
    // table[g_α][g_β] — with g_α = 0, g_β = 1: should be ≥ 1.
    let g0_g1 = summand_a_table[0].as_array().expect("row")[1]
        .as_i64()
        .expect("entry");
    assert!(
        g0_g1 >= 1,
        "Summand A sensitivity table at (g_α=0, g_β=1) should be ≥ 1 (β=2 → β=0 shift); got {g0_g1}"
    );
}
