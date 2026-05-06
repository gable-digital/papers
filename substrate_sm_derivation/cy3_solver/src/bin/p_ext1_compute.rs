//! # EXT1-ENGAGEMENT — Serre-dual computation of `Ext¹(V_2, V_1)`
//!
//! Off-diagonal SU(4) extension data for the BHOP-2005 §6 rank-4 bundle on the
//! Schoen `Z/3 × Z/3` cover `X̃ ⊂ CP² × CP² × CP¹`. By Serre duality on a
//! Calabi-Yau (`K_X = O`):
//!
//! ```text
//!   Ext¹(V_2, V_1)  ≅  H¹(X̃, V_1 ⊗ V_2*)
//! ```
//!
//! A non-trivial `Ext¹` is the harmonic representative that, when wired into
//! [`BhopMonadAdapter::a01_su4_extension`] via a non-zero `κ`, mixes the V_1
//! ↔ V_2 blocks of the SU(4) connection and lets the rank-4 H¹ pipeline
//! redistribute the `(10, +1)` SU(5) rep across {Q, u^c, e^c}. A vanishing
//! `Ext¹` says the BHOP framework's bundle is **split** (V = V_1 ⊕ V_2) and
//! `κ = 0` IS the published physical content — the M3 lift's missing
//! `e^c, ν^c` is then a structural feature, not a deferred wiring task.
//!
//! ## Bundle construction (shadow truncation)
//!
//! BHOP §6.1 Eq. 86 gives `V_1`, `V_2` as rank-2 bundles on `X̃`:
//!
//! ```text
//!   V_1 = O ⊕ O ⊕ O(-τ_1+τ_2) ⊕ O(-τ_1+τ_2)        (rank 4 in line-bundle expansion)
//!   V_2 = O(τ_1-τ_2) ⊗ π_2*(W)                      (rank 2; W rank 2 on dP9)
//! ```
//!
//! The `BhopMonadAdapter` shadow keeps the leading line-bundle summands of the
//! Whitney expansion and truncates one redundant `O(-1,+1)` from V_1 and the
//! `π_2*(W)` factor from V_2:
//!
//! ```text
//!   V_1 (shadow) = O ⊕ O ⊕ O(-1,+1)                 (rank 3)
//!   V_2 (shadow) = O(+1,-1)                         (rank 1)
//! ```
//!
//! Therefore in the shadow:
//!
//! ```text
//!   V_2*       = O(-1,+1)
//!   V_1 ⊗ V_2* = O(-1,+1) ⊕ O(-1,+1) ⊕ O(-2,+2)     (rank 3, c_1 = (-4,+4))
//! ```
//!
//! Note: `c_1(V_1 ⊗ V_2*) ≠ 0`; this is `Hom(V_2, V_1)`, a sum of line bundles
//! whose H¹ on the Schoen cover we compute independently and sum.
//!
//! ## Why BBW instead of the Bergman / Donaldson pipeline?
//!
//! The numerical `compute_h1_twisted_with_method` pipeline (used by
//! `p_lagrangian_eigenmodes`) feeds through `evaluate_chern_curvature`, which
//! is Bergman-kernel-based and requires `H⁰(O(b)) ≠ 0` to ground the fibre
//! Hermitian metric. For each summand of `V_1 ⊗ V_2*` (every one of which has
//! a negative entry in its bidegree) the canonical FS section evaluates to
//! identically zero on every sample, the curvature evaluator returns
//! `AllSamplesDegenerate`, and the H¹ pipeline cannot run at all. This is a
//! structural limitation of the section-basis Bergman approach, not a bug.
//!
//! Bott-Borel-Weil + Koszul on the Schoen CICY (already implemented in
//! [`crate::route34::bbw_cohomology`]) gives the line-bundle cohomology in
//! **closed form** with no numerical sampling, no curvature dependence, no
//! H⁰ ≠ 0 prerequisite. This is the correct tool for Ext¹ here.
//!
//! ## Schoen `Z/3 × Z/3` quotient
//!
//! BBW returns `h^p(X̃, O(L))` on the **upstairs** Schoen cover. The downstairs
//! `Ext¹(V_2, V_1)` on `X̃ / Γ` is the `Γ-invariant subspace`:
//!
//! ```text
//!   Ext¹_X̃/Γ = H¹(X̃, V_1 ⊗ V_2*)^Γ  ⊆  H¹(X̃, V_1 ⊗ V_2*).
//! ```
//!
//! If the upstairs h¹ is `0`, the invariant subspace is also `0` (split). If
//! the upstairs h¹ is positive, the invariant subspace is **at most** that
//! dimension; the closed-form character decomposition would refine it further
//! but is outside the scope of this binary.
//!
//! ## Pipeline
//!
//! 1. Construct `SchoenGeometry::schoen_z3xz3()` (CY3 in `CP² × CP² × CP¹`,
//!    cut by the canonical bidegree-`(3,0,1)` and `(0,3,1)` hypersurfaces).
//! 2. For each of the three `V_1 ⊗ V_2*` summands `O(d_1, d_2, 0)`, call
//!    [`bbw_cohomology::h_star_X_line`] to get `[h⁰, h¹, h², h³]`.
//! 3. Aggregate `Σ h¹` and report.
//!
//! ## Interpretation
//!
//! * `total_h1 == 0`: shadow `Ext¹(V_2, V_1) = 0`. The published BHOP rank-4
//!   bundle's line-bundle shadow is split; `κ = 0` is the framework's correct
//!   physical setting; e^c, ν^c absence is structural at the shadow level.
//! * `total_h1 > 0`: there are non-trivial extension classes upstairs.
//!   `Ext¹` (after Γ-invariance) is at most this. Numerical κ derivation
//!   would require building the harmonic representative and projecting it
//!   into the off-diagonal A^{(0,1)} block — out of scope.
//!
//! ## CLI
//!
//! ```text
//!   p_ext1_compute --output output/p_ext1_compute.json
//! ```
//!
//! There are no sampling parameters: BBW is closed-form. Polynomial-degree,
//! n_pts, seed, etc. are absent by design — the result is fully determined
//! by the BHOP shadow b_lines and the Schoen CICY defining bidegrees.

use clap::Parser;
use cy3_rust_solver::route34::bbw_cohomology::h_star_X_line;
use cy3_rust_solver::route34::bhop_monad_adapter::BhopMonadAdapter;
use cy3_rust_solver::route34::fixed_locus::CicyGeometryTrait;
use cy3_rust_solver::route34::repro::{
    PerSeedEvent, ReplogEvent, ReplogWriter, ReproManifest,
};
use cy3_rust_solver::route34::schoen_geometry::SchoenGeometry;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// One-shot `CicyGeometryTrait` facade over a `SchoenGeometry`. Mirrors the
/// `SchoenFacade` defined inside `bbw_cohomology::h0_line_bundle_schoen` so
/// that we can call `h_star_X_line` directly (it returns the full
/// `[h⁰, h¹, h², h³]` cohomology vector, while the existing public wrapper
/// only returns `h⁰`).
struct SchoenFacade<'a> {
    amb: &'a [u32],
    rels: &'a [Vec<i32>],
}

impl<'a> CicyGeometryTrait for SchoenFacade<'a> {
    fn name(&self) -> &str {
        "Schoen Z/3 × Z/3 (BBW; EXT1-ENGAGEMENT)"
    }
    fn n_coords(&self) -> usize {
        self.amb.iter().map(|&n| (n + 1) as usize).sum()
    }
    fn n_fold(&self) -> usize {
        self.amb.iter().map(|&n| n as usize).sum::<usize>() - self.rels.len()
    }
    fn ambient_factors(&self) -> &[u32] {
        self.amb
    }
    fn defining_relations(&self) -> &[Vec<i32>] {
        self.rels
    }
    fn quotient_label(&self) -> &str {
        "Z3xZ3"
    }
    fn quotient_order(&self) -> u32 {
        9
    }
    fn triple_intersection(&self, _a: &[i32], _b: &[i32], _c: &[i32]) -> i64 {
        0 // Not used by the BBW chase.
    }
    fn intersection_number(&self, _exponents: &[u32]) -> i64 {
        0
    }
}

#[derive(Parser, Debug)]
#[command(about = "EXT1-ENGAGEMENT: H¹(X̃, V_1 ⊗ V_2*) for BHOP §6 SU(4) extension via BBW")]
struct Cli {
    /// Output JSON path. The `.replog` sidecar is written next to it.
    #[arg(long, default_value = "output/p_ext1_compute.json")]
    output: PathBuf,
}

/// Per-summand BBW result.
#[derive(Debug, Serialize, Deserialize, Clone)]
struct SummandResult {
    /// Index in the shadow `V_1 ⊗ V_2*` decomposition.
    summand_idx: usize,
    /// 3-factor bidegree on `CP² × CP² × CP¹` (third entry zero by shadow
    /// convention).
    bidegree: [i32; 3],
    /// Provenance label, e.g. "O(0,0) ⊗ O(+1,-1)*  [V_1 frame 0 ⊗ V_2 frame 3*]".
    provenance: String,
    /// `[h⁰, h¹, h², h³](X̃, O(bidegree))` from BBW + Koszul-Schoen.
    h_star: [i64; 4],
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Ext1ComputeOutput {
    manifest: ReproManifest,
    config: serde_json::Value,
    build_id: String,
    /// Shadow b_lines of `V_1` per BhopMonadAdapter (`[O, O, O(-1,+1)]`).
    v1_shadow_b_lines: Vec<[i32; 2]>,
    /// Shadow b_lines of `V_2` per BhopMonadAdapter (`[O(+1,-1)]`).
    v2_shadow_b_lines: Vec<[i32; 2]>,
    /// `V_1 ⊗ V_2*` summand b_lines (= `[O(-1,+1), O(-1,+1), O(-2,+2)]`).
    v1_tensor_v2_dual_b_lines: Vec<[i32; 2]>,
    /// `c_1(V_1 ⊗ V_2*)` summed over summands.
    c1_v1_tensor_v2_dual: [i32; 2],
    /// Schoen CICY ambient: `[2, 2, 1]` (CP² × CP² × CP¹).
    schoen_ambient_factors: Vec<u32>,
    /// Schoen defining-bidegree relations: `[(3,0,1), (0,3,1)]`.
    schoen_defining_bidegrees: Vec<[i32; 3]>,
    /// Schoen Z/3 × Z/3 quotient order.
    quotient_order: u32,
    /// Per-summand BBW results.
    per_summand: Vec<SummandResult>,
    /// `Σ_i h⁰(X̃, O(b_i))` (informational; should be 0 — sanity).
    total_h0: i64,
    /// `Σ_i h¹(X̃, O(b_i))` — the upstairs Ext¹(V_2, V_1) shadow value.
    total_h1: i64,
    /// `Σ_i h²(X̃, O(b_i))` — informational.
    total_h2: i64,
    /// `Σ_i h³(X̃, O(b_i))` — informational; by Serre dual to h⁰.
    total_h3: i64,
    /// Interpretation banner.
    interpretation: String,
    /// Final SHA-256 chain hash of the `.replog` sidecar.
    replog_final_chain_sha256: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let manifest = ReproManifest::collect();
    let git_short = manifest
        .git_revision
        .as_deref()
        .map(|s| s.chars().take(8).collect::<String>())
        .unwrap_or_else(|| "nogit".to_string());
    let build_id = format!("{}_ext1_bbw", git_short);

    eprintln!("[EXT1-ENGAGEMENT] starting Serre-dual H¹(V_1 ⊗ V_2*) BBW computation");
    eprintln!("  build_id           = {}", build_id);
    eprintln!("  output             = {}", cli.output.display());

    // Read BHOP shadow bundles via the canonical adapter (lookup-don't-hardcode).
    let adapter = BhopMonadAdapter::published();
    let shadow = adapter.shadow_monad();
    // Shadow b_lines = [O, O, O(-1,+1), O(+1,-1)].
    // V_1 frames = [0, 1, 2]; V_2 frame = [3] (per `bhop_monad_adapter`
    // module docstring + the comment chain in `BhopMonadAdapter::published`).
    let v1_b_lines: Vec<[i32; 2]> = shadow.b_lines[0..3].to_vec();
    let v2_b_lines: Vec<[i32; 2]> = shadow.b_lines[3..4].to_vec();
    eprintln!(
        "  v1_shadow_b_lines  = {:?}  (V_1 = O ⊕ O ⊕ O(-1,+1))",
        v1_b_lines
    );
    eprintln!(
        "  v2_shadow_b_lines  = {:?}  (V_2 = O(+1,-1))",
        v2_b_lines
    );

    // Construct V_1 ⊗ V_2* summands. V_2 is rank-1 in the shadow so V_2* is
    // a single line bundle = O(-v2_b). The full tensor = V_1 ⊗ O(-v2_b).
    assert_eq!(v2_b_lines.len(), 1, "V_2 shadow must be rank 1");
    let v2_dual: [i32; 2] = [-v2_b_lines[0][0], -v2_b_lines[0][1]];
    let tensor_b_lines: Vec<[i32; 2]> = v1_b_lines
        .iter()
        .map(|&v1| [v1[0] + v2_dual[0], v1[1] + v2_dual[1]])
        .collect();
    eprintln!(
        "  v1_tensor_v2_dual  = {:?}  (V_1 ⊗ V_2* = O(-1,+1) ⊕ O(-1,+1) ⊕ O(-2,+2))",
        tensor_b_lines
    );
    let c1_sum: [i32; 2] = tensor_b_lines.iter().fold([0, 0], |acc, b| {
        [acc[0] + b[0], acc[1] + b[1]]
    });
    eprintln!(
        "  c_1(V_1 ⊗ V_2*)    = ({}, {})",
        c1_sum[0], c1_sum[1]
    );

    // Schoen geometry + BBW facade.
    let geometry = SchoenGeometry::schoen_z3xz3();
    eprintln!(
        "  schoen_ambient     = {:?}  (CP^2 × CP^2 × CP^1)",
        geometry.ambient_factors
    );
    eprintln!(
        "  schoen_relations   = {:?}  (canonical (3,0,1)+(0,3,1))",
        geometry.defining_bidegrees
    );
    let amb_vec: Vec<u32> = geometry.ambient_factors.to_vec();
    let rels_vec: Vec<Vec<i32>> = geometry
        .defining_bidegrees
        .iter()
        .map(|r: &[i32; 3]| r.to_vec())
        .collect();
    let facade = SchoenFacade {
        amb: &amb_vec,
        rels: &rels_vec,
    };

    // BBW-Koszul cohomology per summand. Lift 2-factor bidegree to the
    // 3-factor `(d_1, d_2, 0)` lift (the Schoen ambient has 3 CP^* factors).
    let provenances = [
        "O(0,0) ⊗ O(+1,-1)*  [V_1 frame 0 ⊗ V_2 frame 3*]",
        "O(0,0) ⊗ O(+1,-1)*  [V_1 frame 1 ⊗ V_2 frame 3*]",
        "O(-1,+1) ⊗ O(+1,-1)*  [V_1 frame 2 ⊗ V_2 frame 3*]",
    ];
    let mut per_summand: Vec<SummandResult> = Vec::with_capacity(3);
    eprintln!();
    eprintln!("  [BBW] line-bundle cohomology on Schoen X̃:");
    let t_start = Instant::now();
    for (i, b) in tensor_b_lines.iter().enumerate() {
        let bidegree3: [i32; 3] = [b[0], b[1], 0];
        let line_3factor: Vec<i32> = bidegree3.to_vec();
        let h_star = h_star_X_line(&line_3factor, &facade)
            .map_err(|e| format!("BBW failed for summand {} bidegree {:?}: {}", i, bidegree3, e))?;
        eprintln!(
            "    summand {}  O({:>3},{:>3}, 0)  : h^* = [{}, {}, {}, {}]   ({})",
            i, b[0], b[1], h_star[0], h_star[1], h_star[2], h_star[3], provenances[i]
        );
        per_summand.push(SummandResult {
            summand_idx: i,
            bidegree: bidegree3,
            provenance: provenances[i].to_string(),
            h_star,
        });
    }
    let t_bbw_secs = t_start.elapsed().as_secs_f64();
    eprintln!("  [BBW] elapsed = {:.6}s", t_bbw_secs);

    let total_h0: i64 = per_summand.iter().map(|s| s.h_star[0]).sum();
    let total_h1: i64 = per_summand.iter().map(|s| s.h_star[1]).sum();
    let total_h2: i64 = per_summand.iter().map(|s| s.h_star[2]).sum();
    let total_h3: i64 = per_summand.iter().map(|s| s.h_star[3]).sum();

    let interpretation = if total_h1 == 0 {
        "split-shadow: H¹(X̃, V_1 ⊗ V_2*) = 0 upstairs. Ext¹(V_2, V_1) shadow \
         vanishes; the BHOP §6 line-bundle shadow extension is SPLIT \
         (V_shadow = V_1_shadow ⊕ V_2_shadow). κ = 0 IS the published BHOP \
         physical setting; e^c, ν^c absence on the rank-4 shadow is \
         STRUCTURAL, not a deferred wiring task. Caveat: the published \
         BHOP V_2 = O(τ_1-τ_2) ⊗ π_2*(W) carries a rank-2 dP9 bundle W not \
         captured in the line-bundle shadow; a non-trivial Ext¹ on the \
         full BHOP bundle would require H¹ on V_1 ⊗ V_2*-with-W, which \
         requires dP9-side cohomology beyond this BBW computation."
            .to_string()
    } else {
        format!(
            "non-trivial-extension-upstairs: Σ h¹(X̃, O(b_i)) = {} > 0. The \
             upstairs (pre-Z/3×Z/3-quotient) Ext¹(V_2, V_1) shadow is \
             non-zero. Ext¹ on X̃/Γ is at most this dimension; full \
             determination requires Γ-invariant character decomposition. \
             Each non-zero summand contributes a candidate harmonic \
             representative; numerical κ derivation requires building the \
             representative and projecting into the off-diagonal A^(0,1) \
             block (out of scope here).",
            total_h1
        )
    };

    eprintln!();
    eprintln!("[EXT1-ENGAGEMENT] === RESULTS ===");
    eprintln!("  total h^0 (informational, expect 0)      = {}", total_h0);
    eprintln!("  total h^1 (Ext¹ upstairs shadow)         = {}", total_h1);
    eprintln!("  total h^2 (informational)                = {}", total_h2);
    eprintln!("  total h^3 (Serre dual to h^0; expect 0)  = {}", total_h3);
    eprintln!();
    eprintln!("  interpretation: {}", interpretation);

    // Replog stream.
    let config_json = serde_json::json!({
        "output": cli.output.to_string_lossy(),
        "method": "bbw_koszul_schoen_z3xz3_cover",
    });
    let mut replog = ReplogWriter::new(8);
    replog.push(ReplogEvent::RunStart {
        binary: "p_ext1_compute".to_string(),
        manifest: manifest.clone(),
        config_json: config_json.clone(),
    });
    for s in &per_summand {
        replog.push(ReplogEvent::PerSeed(PerSeedEvent {
            seed: 0,
            candidate: format!(
                "ext1_summand_{}_O({},{},{})",
                s.summand_idx, s.bidegree[0], s.bidegree[1], s.bidegree[2]
            ),
            k: 0,
            iters_run: 0,
            final_residual: 0.0,
            sigma_fs_identity: 0.0,
            sigma_final: 0.0,
            n_basis: s.h_star.iter().sum::<i64>().max(0) as usize,
            elapsed_ms: 1000.0 * t_bbw_secs / 3.0,
        }));
    }
    let summary_json = serde_json::json!({
        "total_h0": total_h0,
        "total_h1": total_h1,
        "total_h2": total_h2,
        "total_h3": total_h3,
        "interpretation": interpretation,
        "build_id": build_id.clone(),
    });
    replog.push(ReplogEvent::RunEnd {
        summary: summary_json,
        total_elapsed_s: t_bbw_secs,
    });

    let output = Ext1ComputeOutput {
        manifest,
        config: config_json,
        build_id: build_id.clone(),
        v1_shadow_b_lines: v1_b_lines,
        v2_shadow_b_lines: v2_b_lines,
        v1_tensor_v2_dual_b_lines: tensor_b_lines,
        c1_v1_tensor_v2_dual: c1_sum,
        schoen_ambient_factors: geometry.ambient_factors.to_vec(),
        schoen_defining_bidegrees: geometry
            .defining_bidegrees
            .iter()
            .map(|r| *r)
            .collect(),
        quotient_order: 9,
        per_summand,
        total_h0,
        total_h1,
        total_h2,
        total_h3,
        interpretation,
        replog_final_chain_sha256: replog.final_chain_hex(),
    };

    if let Some(parent) = cli.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let json_bytes = serde_json::to_vec_pretty(&output)?;
    fs::write(&cli.output, &json_bytes)?;

    let replog_path = cli.output.with_extension("replog");
    replog.write_to_path(&replog_path)?;

    eprintln!();
    eprintln!("[EXT1-ENGAGEMENT] wrote JSON  : {}", cli.output.display());
    eprintln!("[EXT1-ENGAGEMENT] wrote replog: {}", replog_path.display());
    eprintln!(
        "[EXT1-ENGAGEMENT] replog_final_chain_sha256 = {}",
        output.replog_final_chain_sha256
    );

    Ok(())
}
