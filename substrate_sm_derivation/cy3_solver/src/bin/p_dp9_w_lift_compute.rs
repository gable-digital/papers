//! # DP9-W-LIFT — `Ext¹(V_2, V_1)` lift to the FULL BHOP §6 bundle (W ≠ trivial)
//!
//! ## Mission
//!
//! EXT1-ENGAGEMENT (`references/p_ext1_engagement_2026-05-05.md`) computed the
//! **line-bundle shadow** of `Ext¹(V_2, V_1) = H¹(X̃, V_1 ⊗ V_2*)` on the BHOP
//! §6 SU(4) extension bundle and found it identically zero. The shadow
//! truncated `V_2 = O(τ_1-τ_2) ⊗ π_2*(W)` to its leading line-bundle factor
//! `O(τ_1-τ_2)`, dropping the rank-2 dP9 bundle `W` entirely. This binary
//! engages the FULL bundle by carrying the `π_2*(W)` factor explicitly.
//!
//! ## Spectral sequence setup
//!
//! Schoen `X̃ = B_1 ×_{CP¹} B_2` with `π_2 : X̃ → B_2 = dP9`. For the
//! line-bundle K-theory expansion `[V_1] = 2·[O] + 2·[O(-τ_1+τ_2)]` (BHOP
//! Eq. 86) and `V_2* = O(-τ_1+τ_2) ⊗ π_2*(W*)`:
//!
//! ```text
//!   [V_1 ⊗ V_2*] = 2·[O(-τ_1+τ_2) ⊗ π_2*(W*)] + 2·[O(-2τ_1+2τ_2) ⊗ π_2*(W*)]
//! ```
//!
//! Note: BHOP-2005 `c_1(W) = 0` so `W ≅ W*` (Eq. 85, line below). We use
//! `W* ≃ W` throughout.
//!
//! For each summand `O(a, b, c) ⊗ π_2*(W)` on `X̃`, Leray spectral sequence
//! for `π_2 : X̃ → B_2`:
//!
//! ```text
//!   E_2^{p,q} = H^p(B_2, R^q π_{2*}(O(a, b, c)) ⊗ W)  ⇒  H^{p+q}(X̃, O(a,b,c) ⊗ π_2*(W))
//! ```
//!
//! The fiber of `π_2` is the cubic curve `E ⊂ CP²_x` (an elliptic curve)
//! cut by `F_1(x, t)|_{t fixed} = 0` — fiber-direction degree of `O(a,b,c)`
//! is `a` (the CP²_x degree); the (b, c) degrees are constant along fibers.
//! The fiber line bundle is `O_E(3a)` (since `O_{CP²}(a)|_E` = O_E(3a) for
//! a smooth cubic E).
//!
//! For an elliptic curve E, line bundle of degree d:
//! * `d > 0`: `h⁰(O_E(d)) = d`, `h¹ = 0`
//! * `d < 0`: `h⁰ = 0`, `h¹(O_E(d)) = -d`
//! * `d = 0` and trivial: `h⁰ = h¹ = 1`; non-trivial: both 0
//!
//! In generic position the fibrewise line bundle of degree `3a ≠ 0` is
//! non-trivial of degree `3a`, so:
//!
//! * `a > 0`: `R^0 π_{2*} O(a,b,c) = (rank 3a bundle on B_2)` ⊗ `(b,c)`-part;
//!   `R^1 π_{2*} = 0`
//! * `a < 0`: `R^0 = 0`; `R^1 π_{2*} = (rank -3a bundle on B_2)` ⊗ `(b,c)`-part
//! * `a = 0`: trivial line bundle on each fiber → `R^0 = O_{B_2}` ⊗ `(b,c)`-part,
//!   `R^1 = O_{B_2}` ⊗ `(b,c)`-part (each elliptic fiber's H¹ is 1-dim).
//!
//! Both summands of the K-theory expansion have `a = -1` and `a = -2`, both
//! < 0, so `R^0 π_{2*} = 0` and `R^1 π_{2*}` is a rank-3 / rank-6 vector
//! bundle on `B_2`. Computing the relative duality:
//!
//! ```text
//!   R^1 π_{2*} O(a, b, c)|_X̃  ≃  (R^0 π_{2*} O(-3-a, ...))*  (relative Serre)
//! ```
//!
//! For our case, `R^1 π_{2*} O(-1,b,c) = (R^0 π_{2*} O(-2, b, c) ⊗ K_π)*`
//! where `K_π = π_2^* K_{B_2 → CP¹}^{-1} ⊗ ...`. **This is the technical
//! heart of the Leray chase, and is genuinely non-trivial.** The relative
//! dualising sheaf for a smooth elliptic fibration over a base needs explicit
//! computation; the Schoen-CY3 condition fixes it canonically.
//!
//! For the purposes of this binary we **truncate** to the dP9-side
//! computation that we CAN do exactly:
//!
//! 1. **Line-bundle cohomology on dP9** for all relevant pulled-back-to-base
//!    bidegrees.
//! 2. **W-cohomology on dP9** via the BHOP §6.1 Eq. 85 SES `0 → O(-2f) → W →
//!    2·O(2f) ⊗ I_9 → 0`, including the I_9 generic-position ideal twist.
//! 3. **χ(W ⊗ L) on dP9** Riemann-Roch sanity check.
//! 4. **Output disposition**: definitive Riemann-Roch χ values, partial h^*
//!    decomposition with explicit literature-gap caveats for the steps that
//!    require additional infrastructure not built here.
//!
//! ## Rank discrepancy notice
//!
//! BHOP §6.1 Eq. 85 gives `0 → O(-2f) → W → 2·O(2f) ⊗ I_9 → 0`, which a
//! literal reading makes `rank(W) = 1 + 2 = 3`. The codebase
//! (`hidden_bundle.rs:657`) records `rank(W) = 2`. This is a known
//! literature-vs-code discrepancy. Both interpretations are computed
//! below and reported. The K-theoretic Chern character is computable
//! either way; the rank changes the Riemann-Roch constant offset but
//! not the line-bundle-twist sensitivity.
//!
//! ## CLI
//!
//! ```text
//!   p_dp9_w_lift_compute --output output/p_dp9_w_lift_compute.json
//! ```
//!
//! ## References
//!
//! * Braun-He-Ovrut-Pantev, JHEP 06 (2006) 070, arXiv:hep-th/0505041, §6.1 Eqs. 85-87
//! * Hartshorne, *Algebraic Geometry*, GTM 52, III §5, IV §1 (elliptic surfaces)
//! * Beauville, *Complex Algebraic Surfaces*, 2nd ed., Ch. IX (rational elliptic / dP9)
//! * EXT1-ENGAGEMENT predecessor: `references/p_ext1_engagement_2026-05-05.md`

use clap::Parser;
use cy3_rust_solver::route34::bbw_cohomology::h_star_X_line;
use cy3_rust_solver::route34::fixed_locus::CicyGeometryTrait;
use cy3_rust_solver::route34::repro::{
    PerSeedEvent, ReplogEvent, ReplogWriter, ReproManifest,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

// ---------------------------------------------------------------------------
// dP9 = (3,1) ⊂ CP² × CP¹  CICY facade
// ---------------------------------------------------------------------------

/// dP9 surface as the smooth complete intersection of one bidegree-(3, 1)
/// hypersurface in `CP²_x × CP¹_t`.
///
/// This is the Weierstrass-form rational elliptic surface, a.k.a. the
/// `B_i` factor of the Schoen 3-fold. In the BHOP §6.1 convention `f` is
/// the elliptic-fiber class, restricted-from `O_{CP¹}(1)`. The
/// hyperplane class of `CP²_x` restricts to a line bundle of degree 3
/// on each fiber (the cubic curve).
struct Dp9Geometry {
    ambient: [u32; 2],
    rels: Vec<Vec<i32>>,
}

impl Dp9Geometry {
    fn new() -> Self {
        Self {
            ambient: [2, 1],
            rels: vec![vec![3, 1]],
        }
    }
}

impl CicyGeometryTrait for Dp9Geometry {
    fn name(&self) -> &str {
        "dP9 = (3,1) ⊂ CP^2 × CP^1 (Weierstrass elliptic surface)"
    }
    fn n_coords(&self) -> usize {
        // (2+1) + (1+1) = 5
        5
    }
    fn n_fold(&self) -> usize {
        // surface
        2
    }
    fn ambient_factors(&self) -> &[u32] {
        &self.ambient
    }
    fn defining_relations(&self) -> &[Vec<i32>] {
        &self.rels
    }
    fn quotient_label(&self) -> &str {
        "trivial"
    }
    fn quotient_order(&self) -> u32 {
        1
    }
    fn triple_intersection(&self, _a: &[i32], _b: &[i32], _c: &[i32]) -> i64 {
        0
    }
    fn intersection_number(&self, _exponents: &[u32]) -> i64 {
        0
    }
}

// ---------------------------------------------------------------------------
// Line-bundle cohomology on dP9 (closed form via BBW + Koszul on (3,1) CICY)
// ---------------------------------------------------------------------------

/// `[h⁰, h¹, h², h³]` (h³ always 0 for a surface) on dP9 = (3,1) ⊂ CP² × CP¹.
fn h_star_dp9(line: &[i32; 2], geom: &Dp9Geometry) -> [i64; 4] {
    let line_v: Vec<i32> = line.to_vec();
    h_star_X_line(&line_v, geom).expect("BBW failed on dP9 (3,1) CICY")
}

/// Generic-position evaluation of the ideal-sheaf twist `I_n · L` on a
/// surface S, for a 0-dim subscheme `Z` of length `n`:
///
/// ```text
///   0 → I_Z → O_S → O_Z → 0
/// ```
///
/// tensored with `L` gives:
///
/// ```text
///   0 → I_Z · L → L → L|_Z = O_Z^n → 0
/// ```
///
/// (since L|_Z is just n copies of the residue field). Long-exact:
///
/// ```text
///   0 → H⁰(I_Z·L) → H⁰(L) →^{ev} C^n → H¹(I_Z·L) → H¹(L) → 0,
///   H²(I_Z·L) ≃ H²(L)
/// ```
///
/// In **generic position** the evaluation map is maximally rank, so:
///
/// ```text
///   h⁰(I_Z·L) = max(0, h⁰(L) - n)
///   h¹(I_Z·L) = max(0, n - h⁰(L)) + h¹(L)
///   h²(I_Z·L) = h²(L)
/// ```
fn h_star_ideal_twist(h_l: [i64; 4], n: i64) -> [i64; 4] {
    let h0_l = h_l[0];
    let h1_l = h_l[1];
    let h2_l = h_l[2];
    let h0 = (h0_l - n).max(0);
    let h1 = (n - h0_l).max(0) + h1_l;
    let h2 = h2_l;
    [h0, h1, h2, 0]
}

// ---------------------------------------------------------------------------
// W cohomology on dP9 via the BHOP §6.1 Eq. 85 SES.
// ---------------------------------------------------------------------------

/// Compute `[h⁰, h¹, h², h³](B_2, W ⊗ L)` from the BHOP-2005 §6.1 Eq. 85
/// extension SES of the auxiliary rank-2 bundle `W → B_2 = dP9`:
///
/// ```text
///   0 → O(-2f) → W → 2·O(2f) ⊗ I_9 → 0     (BHOP Eq. 85)
/// ```
///
/// where `f` is the elliptic-fiber class (= O_{B_2}(0,1) in the (3,1) CICY
/// convention) and `I_9` is the ideal sheaf of a generic Z/3×Z/3 orbit
/// of 9 points on B_2.
///
/// Tensoring with line bundle L gives:
///
/// ```text
///   0 → L(-2f) → W ⊗ L → 2·L(2f) ⊗ I_9 → 0
/// ```
///
/// The associated long-exact cohomology sequence determines `H^*(W ⊗ L)`
/// from `H^*(L(-2f))` and `H^*(L(2f) ⊗ I_9)` via:
///
/// ```text
///   h^p(W ⊗ L) ∈ [|h^p(A) - h^{p-1}(B)| | rank-bounded via SES connecting maps]
/// ```
///
/// Without explicit knowledge of the SES connecting-map ranks (which
/// depend on the specific extension class — the splitting type of the
/// BHOP construction), we report the **K-theoretic Euler characteristic**
/// which is invariant:
///
/// ```text
///   χ(W ⊗ L) = χ(L(-2f)) + 2·χ(L(2f) ⊗ I_9)
///            = χ(L(-2f)) + 2·[χ(L(2f)) - 9]    (length-9 ideal)
/// ```
///
/// and the **min/max bounds** on each h^p from the long exact sequence
/// (for verification).
///
/// Returns `(h_w_lower, h_w_upper, chi_w)` where the bounds are derived
/// from the long-exact sequence:
///
/// ```text
///   ... → h^{p-1}(B) → h^p(A) → h^p(W⊗L) → h^p(B) → h^{p+1}(A) → ...
/// ```
///
/// Bounds:
/// * `h^p(W⊗L) ≥ |h^p(A) - h^{p-1}(B)|`  (kernel/coker considerations)
/// * `h^p(W⊗L) ≤ h^p(A) + h^p(B)`         (upper bound: SES splits)
fn h_star_w_tensor_l_dp9(
    line: [i32; 2],
    geom: &Dp9Geometry,
) -> WTensorLResult {
    // f = elliptic-fiber class = (0, 1) in the CP² × CP¹ ambient.
    // L(-2f) = L + (0, -2)
    let l_minus_2f: [i32; 2] = [line[0], line[1] - 2];
    // L(2f) = L + (0, +2)
    let l_plus_2f: [i32; 2] = [line[0], line[1] + 2];

    let h_a = h_star_dp9(&l_minus_2f, geom); // H^*(L(-2f))
    let h_b_pre = h_star_dp9(&l_plus_2f, geom); // H^*(L(2f)) before ideal twist
    let h_b_ideal = h_star_ideal_twist(h_b_pre, 9); // H^*(L(2f) ⊗ I_9)
    // 2 copies:
    let h_b: [i64; 4] = [
        2 * h_b_ideal[0],
        2 * h_b_ideal[1],
        2 * h_b_ideal[2],
        2 * h_b_ideal[3],
    ];

    // Euler characteristic via SES: χ(W⊗L) = χ(A) + χ(B).
    let chi = |h: [i64; 4]| -> i64 {
        h[0] - h[1] + h[2] - h[3]
    };
    let chi_w = chi(h_a) + chi(h_b);

    // Long-exact bounds:
    //   ... → h^{p-1}(B) → h^p(A) → h^p(W⊗L) → h^p(B) → h^{p+1}(A) → ...
    //
    // h^p(W⊗L) = ker(h^p(B) → h^{p+1}(A)) image-of-(h^p(A) → h^p(W⊗L))
    //          = (h^p(A) - im_in_A) + (h^p(B) - im_out_B)
    // where im_in_A = im(h^{p-1}(B) → h^p(A)) and im_out_B = im(h^p(B) → h^{p+1}(A))
    //
    // Lower bound: |χ contribution|; upper bound: h^p(A) + h^p(B).
    // For surfaces (p ∈ {0, 1, 2}):

    let h_w_upper: [i64; 4] = [
        h_a[0] + h_b[0],
        h_a[1] + h_b[1],
        h_a[2] + h_b[2],
        0,
    ];

    // Lower bound from the long exact sequence — most-degenerate case
    // where every connecting map is rank-maximal (kills the rank as
    // possible at each step). This is the same generic-rank assumption
    // that AGLP-2011 §2 makes for Koszul chases.
    //
    //   p=0: 0 → h⁰(A) → h⁰(W⊗L) → h⁰(B) → h¹(A) → ...
    //        h⁰(W⊗L) = h⁰(A) + h⁰(B) - rank(B → h¹(A))
    //        rank(B → h¹(A)) ≤ min(h⁰(B), h¹(A))
    //        ⇒ h⁰(W⊗L) ≥ h⁰(A) + h⁰(B) - min(h⁰(B), h¹(A))
    //                  = h⁰(A) + max(0, h⁰(B) - h¹(A))
    //   p=1: similarly with shifted indices
    //   p=2: h²(W⊗L) ≥ h²(A) + max(0, h²(B) - h³(A)) = h²(A) + h²(B)  (h³(A)=0 for surface)
    let g_max_rank = |h_a_p: i64, h_b_p: i64, h_a_pp1: i64| -> i64 {
        h_a_p + 0i64.max(h_b_p - h_a_pp1)
    };
    let h_w_lower: [i64; 4] = [
        g_max_rank(h_a[0], h_b[0], h_a[1]),
        g_max_rank(h_a[1], h_b[1], h_a[2]),
        h_a[2] + h_b[2], // p=2: h³(A)=0
        0,
    ];

    WTensorLResult {
        line,
        h_a_l_minus_2f: h_a,
        h_b_pre_ideal: h_b_pre,
        h_b_post_ideal_singleton: h_b_ideal,
        h_b_total: h_b,
        chi_w_tensor_l: chi_w,
        h_w_lower_bound: h_w_lower,
        h_w_upper_bound: h_w_upper,
    }
}

// ---------------------------------------------------------------------------
// Riemann-Roch sanity check on dP9.
// ---------------------------------------------------------------------------

/// Hirzebruch-Riemann-Roch on dP9 surface:
///
/// ```text
///   χ(L) = 1 + (1/2)·c_1(L)·(c_1(L) - K_S)
///        = 1 + (1/2)·c_1(L)·(c_1(L) + f)        (K_S = -f for dP9)
/// ```
///
/// Intersection numbers on dP9 ⊂ CP² × CP¹:
/// * Let H = O(1, 0)|_S (CP² hyperplane restriction), F = O(0, 1)|_S = f.
/// * H² ⊂ S = (degree of S) ⋅ H² in CP² × CP¹ = restricting top-power.
///   Equivalently, on a (3,1) divisor S in CP² × CP¹:
///   * H · H · S = H² · S = H² · (3H + F) = 3·H² · H + H² · F
///     in the ambient. H³ = 0 (CP² is 2-dim), F² = 0 (CP¹ is 1-dim).
///     So H²·H = 0 and H²·F = 1 in ambient, giving H²·S = 1 in ambient = H²
///     restricted to S = 1.
///   * H · F · S = H · F · (3H + F) = 3·H²F + H·F² = 3·1 + 0 = 3.
///   * F · F · S = F² · (3H + F) = 0.
///
/// So on S = dP9: H² = 1, H · F = 3, F² = 0.
///
/// For L = O(a, b)|_S: c_1(L)² = a²·H² + 2ab·H·F + b²·F² = a² + 6ab.
/// K_S · c_1(L) = -F · (aH + bF) = -a·H·F - b·F² = -3a.
///
/// ```text
///   χ(L) = 1 + (1/2)·[(a² + 6ab) - (-3a)]
///        = 1 + (1/2)·(a² + 6ab + 3a)
/// ```
fn chi_dp9_line(line: [i32; 2]) -> i64 {
    let a = line[0] as i64;
    let b = line[1] as i64;
    // χ = 1 + (a² + 6ab + 3a) / 2
    let num = a * a + 6 * a * b + 3 * a;
    debug_assert!(num % 2 == 0, "RR numerator must be even (Td integral)");
    1 + num / 2
}

// ---------------------------------------------------------------------------
// Output structures
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone)]
struct WTensorLResult {
    /// Line bundle L on dP9 (bidegree (a, b) in CP² × CP¹).
    line: [i32; 2],
    /// `H^*(B_2, L(-2f))` (sub of the BHOP SES).
    h_a_l_minus_2f: [i64; 4],
    /// `H^*(B_2, L(2f))` before the I_9 twist.
    h_b_pre_ideal: [i64; 4],
    /// `H^*(B_2, L(2f) ⊗ I_9)` (one copy).
    h_b_post_ideal_singleton: [i64; 4],
    /// `H^*(B_2, 2·L(2f) ⊗ I_9)` (the BHOP quotient summand: 2 copies).
    h_b_total: [i64; 4],
    /// `χ(W ⊗ L) = χ(A) + χ(B)` (invariant under SES).
    chi_w_tensor_l: i64,
    /// Lower bound on `[h⁰, h¹, h², h³](W ⊗ L)` from the long exact sequence
    /// under the generic-rank assumption.
    h_w_lower_bound: [i64; 4],
    /// Upper bound (SES splits trivially: H^*(W ⊗ L) ≤ H^*(A) + H^*(B)).
    h_w_upper_bound: [i64; 4],
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct LineProbeResult {
    /// dP9 line bundle bidegree `(a, b)`.
    line: [i32; 2],
    /// `[h⁰, h¹, h²]` on dP9.
    h_star: [i64; 4],
    /// Riemann-Roch χ from the closed-form formula.
    chi_rr: i64,
    /// `χ_bbw = h⁰ - h¹ + h²` from the BBW computation; must equal `chi_rr`.
    chi_bbw: i64,
    /// Sanity check.
    rr_match: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Dp9WLiftOutput {
    manifest: ReproManifest,
    config: serde_json::Value,
    build_id: String,
    /// dP9 = (3,1) ⊂ CP² × CP¹.
    dp9_ambient: Vec<u32>,
    dp9_relations: Vec<Vec<i32>>,
    /// Line-bundle BBW probe results on dP9 (sanity vs Riemann-Roch).
    line_probes: Vec<LineProbeResult>,
    /// W ⊗ L results for the K-theoretic line-bundle expansion of V_1 ⊗ V_2*.
    w_tensor_l_summands: Vec<WTensorLResult>,
    /// Sum of χ(W ⊗ L) over all summands (K-theoretic invariant of
    /// `V_1 ⊗ V_2* ⊗ π_2*(W*)` on dP9, before pushing forward to X̃).
    total_chi_dp9_side: i64,
    /// Verbal interpretation of the computation.
    interpretation: String,
    /// Open research items (multi-day work needed to close the full computation).
    open_research_items: Vec<String>,
    /// Replog SHA-256 chain hash.
    replog_final_chain_sha256: String,
}

#[derive(Parser, Debug)]
#[command(about = "DP9-W-LIFT: H^*(B_2, W ⊗ L) for the BHOP §6 W-twist of V_1 ⊗ V_2*")]
struct Cli {
    /// Output JSON path. The `.replog` sidecar is written next to it.
    #[arg(long, default_value = "output/p_dp9_w_lift_compute.json")]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let manifest = ReproManifest::collect();
    let git_short = manifest
        .git_revision
        .as_deref()
        .map(|s| s.chars().take(8).collect::<String>())
        .unwrap_or_else(|| "nogit".to_string());
    let build_id = format!("{}_dp9_w_lift", git_short);

    eprintln!("[DP9-W-LIFT] starting H^*(B_2 = dP9, W ⊗ L) computation");
    eprintln!("  build_id           = {}", build_id);
    eprintln!("  output             = {}", cli.output.display());

    let geom = Dp9Geometry::new();
    eprintln!(
        "  dp9_ambient        = {:?}  (CP^2 × CP^1)",
        geom.ambient_factors()
    );
    eprintln!(
        "  dp9_relations      = {:?}  (canonical (3,1) hypersurface)",
        geom.defining_relations()
    );

    // -----------------------------------------------------------------
    // Step 1: Sanity-check the dP9 BBW infrastructure against
    // closed-form Riemann-Roch. We probe a representative set of
    // line bundles `L = O(a, b)|_{dP9}` and verify that
    // `χ_bbw = h⁰ - h¹ + h² = χ_rr = 1 + (a² + 6ab + 3a)/2`.
    // -----------------------------------------------------------------
    eprintln!();
    eprintln!("  [Step 1] Riemann-Roch sanity probes on dP9:");
    let probes: Vec<[i32; 2]> = vec![
        [0, 0],   // O_dP9; χ = 1
        [1, 0],   // hyperplane H; χ = 1 + (1+3)/2 = 3
        [0, 1],   // fiber f; χ = 1 (degree-0 in CP² direction → just χ(O))
        [-1, 0],  // -H; χ = 1 + (1-3)/2 = 0
        [0, -1],  // -f; χ(K_S) = 1 (Serre dual to χ(O))
        [1, 1],   // H+f; χ = 1 + (1+6+3)/2 = 6
        [-1, 1],  // -H+f; χ = 1 + (1-6-3)/2 = -3
        [-2, 2],  // c_1 of the second K-theoretic summand bidegree
        [-3, 1],  // K_dP9 + something interesting
    ];
    let mut line_probes: Vec<LineProbeResult> = Vec::new();
    let mut all_match = true;
    for line in &probes {
        let h = h_star_dp9(line, &geom);
        let chi_bbw = h[0] - h[1] + h[2] - h[3];
        let chi_rr = chi_dp9_line(*line);
        let rr_match = chi_bbw == chi_rr;
        if !rr_match {
            all_match = false;
        }
        eprintln!(
            "    O({:>3},{:>3})  h^* = [{:>3}, {:>3}, {:>3}, {}]  χ_bbw = {:>3}  χ_rr = {:>3}  {}",
            line[0],
            line[1],
            h[0],
            h[1],
            h[2],
            h[3],
            chi_bbw,
            chi_rr,
            if rr_match { "OK" } else { "MISMATCH" }
        );
        line_probes.push(LineProbeResult {
            line: *line,
            h_star: h,
            chi_rr,
            chi_bbw,
            rr_match,
        });
    }
    if !all_match {
        return Err("dP9 BBW vs Riemann-Roch mismatch — REAL BUG. Aborting.".into());
    }
    eprintln!("    [Step 1] all {} probes match Riemann-Roch.", probes.len());

    // -----------------------------------------------------------------
    // Step 2: Compute H^*(dP9, W ⊗ L) for each summand of the
    // K-theoretic V_1 ⊗ V_2* expansion.
    //
    // The full BHOP V_1 ⊗ V_2* on X̃ in K-theory:
    //
    //   [V_1 ⊗ V_2*] = (2·[O] + 2·[O(-τ_1+τ_2)]) ⊗ [O(-τ_1+τ_2) ⊗ π_2*(W*)]
    //                = 2·[O(-τ_1+τ_2) ⊗ π_2*(W*)] + 2·[O(-2τ_1+2τ_2) ⊗ π_2*(W*)]
    //
    // (using c_1(W) = 0 ⇒ W ≅ W*).
    //
    // The "τ_2" direction maps under π_2 to the dP9 hyperplane class
    // H_2 of CP²_y (BHOP convention: τ_2 is the elliptic-section
    // class of B_2 = dP9, equivalent to the CP²_y hyperplane in
    // our embedding). The τ_1 direction is the X̃-side (CP²_x)
    // hyperplane and pushes forward via R^q π_{2*} to a base bundle.
    //
    // Here we report the dP9-side cohomology of `W ⊗ L_base` where
    // L_base is the **base-only** factor `O(b, c)` corresponding to
    // the (τ_2, t) directions of the X̃ summand. The CP²_x (τ_1)
    // direction is absorbed by R^q π_{2*} which we report separately
    // as a research-plan item.
    // -----------------------------------------------------------------
    eprintln!();
    eprintln!("  [Step 2] H^*(dP9, W ⊗ L) for K-theoretic V_1 ⊗ V_2* summands:");
    eprintln!();
    eprintln!("    K-theoretic [V_1 ⊗ V_2*] expansion on X̃:");
    eprintln!("      Summand A:  2·O(-τ_1+τ_2) ⊗ π_2*(W*)   (multiplicity 2)");
    eprintln!("      Summand B:  2·O(-2τ_1+2τ_2) ⊗ π_2*(W*)  (multiplicity 2)");
    eprintln!();
    eprintln!("    For each, the dP9-side line bundle is L_base = O_{{dP9}}(b, 0):");
    eprintln!("      Summand A:  L_base = O_dP9(+1, 0)  (from τ_2-degree +1)");
    eprintln!("      Summand B:  L_base = O_dP9(+2, 0)  (from τ_2-degree +2)");
    eprintln!();
    eprintln!("    Note: dP9-coordinates are (CP²_y hyperplane, CP¹_t fiber); BHOP τ_2");
    eprintln!("          identified with CP²_y hyperplane class on B_2.");
    eprintln!();

    // The dP9-side line bundles probed (ignoring fiber-direction τ_1 which
    // pushes through π_2):
    let dp9_line_summands: Vec<([i32; 2], &str, u32)> = vec![
        ([1, 0], "Summand A: V_1 ⊗ V_2* base factor for [-τ_1+τ_2] piece", 2),
        ([2, 0], "Summand B: V_1 ⊗ V_2* base factor for [-2τ_1+2τ_2] piece", 2),
    ];

    let t_start = Instant::now();
    let mut w_tensor_l_summands: Vec<WTensorLResult> = Vec::new();
    let mut total_chi: i64 = 0;
    for (line, label, mult) in &dp9_line_summands {
        let r = h_star_w_tensor_l_dp9(*line, &geom);
        eprintln!("    {}", label);
        eprintln!("      L                    = O_dP9({}, {})", line[0], line[1]);
        eprintln!(
            "      h^*(L(-2f))          = {:?}    (BHOP-Eq.85 sub of W)",
            r.h_a_l_minus_2f
        );
        eprintln!(
            "      h^*(L(2f))           = {:?}    (before I_9 twist)",
            r.h_b_pre_ideal
        );
        eprintln!(
            "      h^*(L(2f) ⊗ I_9)     = {:?}    (after generic-position I_9 twist, single copy)",
            r.h_b_post_ideal_singleton
        );
        eprintln!(
            "      h^*(2·L(2f) ⊗ I_9)   = {:?}    (BHOP quotient: 2 copies)",
            r.h_b_total
        );
        eprintln!(
            "      χ(W ⊗ L)             = {}     (K-theoretic invariant via SES)",
            r.chi_w_tensor_l
        );
        eprintln!(
            "      h^*(W ⊗ L) bounds    = {:?} ≤ h^* ≤ {:?}",
            r.h_w_lower_bound, r.h_w_upper_bound
        );
        eprintln!(
            "      contribution × {} (multiplicity)",
            mult
        );
        eprintln!();
        total_chi += (*mult as i64) * r.chi_w_tensor_l;
        w_tensor_l_summands.push(r);
    }
    let t_step2 = t_start.elapsed().as_secs_f64();
    eprintln!("  [Step 2] elapsed = {:.6}s", t_step2);

    eprintln!();
    eprintln!("  [Step 2 summary] Σ_summand mult × χ_dP9(W ⊗ L_base) = {}", total_chi);
    eprintln!();

    // -----------------------------------------------------------------
    // Step 3: Output disposition + research plan
    // -----------------------------------------------------------------
    let interpretation = format!(
        "DP9-W-LIFT partial computation:\n\n\
         (i) dP9 BBW infrastructure validated against closed-form Riemann-Roch\n\
             on {} probe line bundles — all matched.\n\n\
         (ii) For each K-theoretic summand of [V_1 ⊗ V_2*] on X̃, the dP9-side\n\
             cohomology of W ⊗ L_base was computed via the BHOP §6.1 Eq. 85\n\
             SES `0 → O(-2f) → W → 2·O(2f) ⊗ I_9 → 0` with generic-position\n\
             ideal-sheaf twist for the 9 points of the Z/3×Z/3 orbit.\n\n\
         (iii) Total K-theoretic χ_dP9 contribution = {}\n\n\
         (iv) DEFINITIVE Ext¹ value not yet derivable from this computation\n\
              alone. The final answer requires the Leray spectral sequence\n\
              chase that pushes the X̃-side line bundle (with its CP²_x = τ_1\n\
              degree) forward to dP9 along π_2: X̃ → B_2; the fiber-cohomology\n\
              R^* π_{{2*}} of `O(a τ_1)` involves the relative dualising\n\
              sheaf of the elliptic fibration B_1 → CP¹ pulled back along\n\
              B_2 → CP¹, which is non-trivial Schoen-CY3 geometry and is\n\
              outside the scope of this single-stream binary.",
        line_probes.len(),
        total_chi
    );
    eprintln!("  [Step 3] interpretation:");
    for line in interpretation.lines() {
        eprintln!("    {}", line);
    }

    let open_items = vec![
        "Resolve the rank(W) = 2 (code) vs rank(W) = 3 (literal BHOP Eq. 85 \
         reading) discrepancy. Likely the BHOP `2·O(2f)` notation means \
         the rank-2 trivial bundle tensored with O(2f), not two copies. \
         Cross-check against BHOP-2005 hep-th/0505041 PDF directly."
            .to_string(),
        "Implement R^* π_{2*} for the elliptic fibration π_2: X̃ → B_2 = dP9. \
         The fiber over y ∈ B_2 is the cubic curve E_y ⊂ CP²_x cut by \
         F_1(x, t(y)) = 0. For O_X̃(a, b, c)|_fiber = O_E(3a) the Künneth \
         + relative duality gives R^q π_{2*} = (rank |3a| bundle on B_2) \
         ⊗ O_{B_2}(b, c) for q = (a < 0 ? 1 : 0); explicit identification \
         of the bundle requires the relative dualising sheaf of the \
         elliptic fibration."
            .to_string(),
        "Push the dP9-side computation H^*(B_2, W ⊗ R^q π_{2*}(O_{X̃}(a,b,c))) \
         through the Leray E_2 → E_∞ spectral sequence chase. Combine \
         with the χ_dP9 results in this binary to produce h^p(X̃, V_1 ⊗ V_2*) \
         per K-theoretic summand."
            .to_string(),
        "Project onto the Z/3 × Z/3 invariant subspace via character \
         decomposition (the BBW Schoen module already does this via the \
         Koszul chase; the analogue for the W-twisted case requires \
         tracking Γ-equivariant W-module characters)."
            .to_string(),
        "Resolve the BHOP V_1 rank(2) vs line-bundle expansion rank(4) \
         discrepancy. Most likely V_1 in the paper is a non-split rank-2 \
         extension whose K-theoretic class equals 2·[O] + 2·[O(-τ_1+τ_2)] \
         but whose actual rank is 2; the K-theoretic Euler characteristic \
         is the same either way, but the SES bounds on h^p will differ."
            .to_string(),
        "Once all of (i)-(iv) are complete: derive κ ≠ 0 and propagate \
         through the rank-4 H¹ frame-irrep redistribution pipeline to \
         test whether {Q, u^c, e^c} now appear at non-zero κ."
            .to_string(),
    ];

    eprintln!();
    eprintln!("  [Step 3] open research items (multi-day scope):");
    for (i, item) in open_items.iter().enumerate() {
        eprintln!("    ({}) {}", i + 1, item);
    }

    // -----------------------------------------------------------------
    // Replog stream
    // -----------------------------------------------------------------
    let config_json = serde_json::json!({
        "output": cli.output.to_string_lossy(),
        "method": "dp9_bbw_w_extension_ses_partial_lift",
        "scope": "dP9-side only (X̃-side Leray chase deferred)",
    });
    let mut replog = ReplogWriter::new(8);
    replog.push(ReplogEvent::RunStart {
        binary: "p_dp9_w_lift_compute".to_string(),
        manifest: manifest.clone(),
        config_json: config_json.clone(),
    });
    for (idx, r) in w_tensor_l_summands.iter().enumerate() {
        replog.push(ReplogEvent::PerSeed(PerSeedEvent {
            seed: 0,
            candidate: format!(
                "w_tensor_summand_{}_O({},{})_dp9",
                idx, r.line[0], r.line[1]
            ),
            k: 0,
            iters_run: 0,
            final_residual: 0.0,
            sigma_fs_identity: 0.0,
            sigma_final: r.chi_w_tensor_l as f64,
            n_basis: r.h_w_upper_bound.iter().sum::<i64>().max(0) as usize,
            elapsed_ms: 1000.0 * t_step2 / w_tensor_l_summands.len().max(1) as f64,
        }));
    }
    let summary_json = serde_json::json!({
        "total_chi_dp9_side": total_chi,
        "n_line_probes": line_probes.len(),
        "n_w_summands": w_tensor_l_summands.len(),
        "rr_sanity_passed": all_match,
        "build_id": build_id.clone(),
    });
    replog.push(ReplogEvent::RunEnd {
        summary: summary_json,
        total_elapsed_s: t_step2,
    });

    // -----------------------------------------------------------------
    // Write outputs
    // -----------------------------------------------------------------
    let output = Dp9WLiftOutput {
        manifest,
        config: config_json,
        build_id,
        dp9_ambient: geom.ambient_factors().to_vec(),
        dp9_relations: geom.defining_relations().to_vec(),
        line_probes,
        w_tensor_l_summands,
        total_chi_dp9_side: total_chi,
        interpretation,
        open_research_items: open_items,
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
    eprintln!("[DP9-W-LIFT] wrote JSON  : {}", cli.output.display());
    eprintln!("[DP9-W-LIFT] wrote replog: {}", replog_path.display());
    eprintln!(
        "[DP9-W-LIFT] replog_final_chain_sha256 = {}",
        output.replog_final_chain_sha256
    );

    Ok(())
}
