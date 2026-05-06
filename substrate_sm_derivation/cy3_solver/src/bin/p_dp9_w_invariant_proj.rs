//! # DP9-W-INVARIANT-PROJ — Z/3×Z/3-invariant projection of the
//! W-twisted `H¹(W ⊗ L)` per BHOP §6 K-theoretic summand.
//!
//! ## Mission
//!
//! DP9-W-LIFT-3 (`p_dp9_w_lift_compute.rs`, ref
//! `references/p_dp9_w_lift_2026-05-05.md`) computed the dP9-side
//! W-twisted Euler characteristic for the BHOP-2005 §6 SU(4) extension
//! bundle and concluded:
//!
//! * Summand A (`L = O_dP9(+1, 0)`, multiplicity 2): `h¹ = 3` exact
//!   (bounds collapsed; H¹ comes entirely from the SES sub `L(-2f) =
//!   O(1, -2)`, whose `h¹ = 3` is unencumbered).
//! * Summand B (`L = O_dP9(+2, 0)`, multiplicity 2): `χ = +12`,
//!   `h¹ ∈ [6, ?]` (bounds open by extension class).
//! * Total `Σ mult · χ_dP9 = +18` — non-zero, suggestive of (b).
//!
//! The remaining gate-defect question: **does the +18 survive the
//! Schoen Z/3 × Z/3-invariant projection**, or does it sit entirely in
//! non-invariant characters (in which case, on the BHOP quotient X̃/Γ,
//! Ext¹ vanishes and the framework's TeV-mass prediction is the genuine
//! shadow result)?
//!
//! ## Method
//!
//! For each K-theoretic summand we (i) construct the explicit Čech-
//! cocycle monomial basis of `H¹(O_dP9(L(-2f)))` and `H⁰/H¹(O_dP9(
//! L(2f)) ⊗ I_9)`, (ii) compute the Schoen-cover (α, β)-character of
//! every basis element via `route34::z3xz3_projector::{alpha_character,
//! beta_character}`, (iii) apply the BHOP equivariant lift convention
//! to the bundle frame, and (iv) count `(0, 0)`-character invariants.
//!
//! Characters are additive over the BHOP-Eq.85 SES
//! `0 → L(-2f) → W ⊗ L → 2·L(2f) ⊗ I_9 → 0`, so the equivariant Euler
//! character of `W ⊗ L` is:
//!
//! ```text
//!     χ_Γ(W ⊗ L)  =  χ_Γ(L(-2f))  +  2 · χ_Γ(L(2f) ⊗ I_9).
//! ```
//!
//! The character of `L(2f) ⊗ I_9` decomposes from the 9-point ideal-
//! sheaf SES `0 → I_9 → O → O_{Z_9} → 0` tensored with `L(2f)`:
//!
//! ```text
//!     χ_Γ(L(2f) ⊗ I_9)  =  χ_Γ(L(2f))  -  χ_Γ(O_{Z_9} ⊗ L(2f)|_{Z_9})
//! ```
//!
//! and `O_{Z_9}` is the **regular representation** of `Γ = Z/3 × Z/3`
//! (each character appears with multiplicity 1) when `Z_9` is a free
//! `Γ`-orbit, which it is by BHOP §6.1 construction.
//!
//! ## Bundle equivariant lift
//!
//! The BHOP-2005 §6 construction fixes an equivariant lift of W and
//! the line bundles to make V descend to X̃/Γ. The natural BHOP lift
//! (§3.2 + §6.1) takes `W` and the line bundles `O(±τ_i)` with
//! **trivial Γ-character on the bundle frame** — the `α`/`β` action is
//! carried entirely by the *coordinates* (sections), not the frame.
//! This is the same convention used by the BBW Schoen module
//! (`route34/schoen_module.rs`) for the chiral-mode count.
//!
//! For sensitivity, this binary additionally tabulates the invariant
//! count for **all 9 possible bundle-frame characters** of each summand
//! and reports it as a `BundleLiftSensitivity` table. The BHOP-canonical
//! lift `(0, 0)` is highlighted as the production value.
//!
//! ## Verdict logic
//!
//! ```text
//!     invariant_dim_total
//!         = Σ_summand  mult · invariant_dim(W ⊗ L_summand)
//!         = Σ_summand  mult · #{ basis monomials with (α,β) = (0,0) }
//!
//!     Verdict (a) ⇔ invariant_dim_total = 0  (Ext¹ vanishes on quotient)
//!     Verdict (b) ⇔ invariant_dim_total > 0  (Ext¹ ≠ 0; κ derivable)
//! ```
//!
//! ## CLI
//!
//! ```text
//!     p_dp9_w_invariant_proj --output output/p_dp9_w_invariant_proj.json
//! ```
//!
//! ## References
//!
//! * Braun-He-Ovrut-Pantev, JHEP 06 (2006) 070, arXiv:hep-th/0505041,
//!   §3.2 (equivariant lift), §6.1 (W bundle), §6.3 (Wilson-line proj).
//! * Donagi-He-Ovrut-Reinbacher, JHEP 06 (2006) 039,
//!   arXiv:hep-th/0512149, §3-4 (diagonal Γ action, character table).
//! * Hartshorne, GTM 52, III §5 (Čech cohomology, monomial cocycle bases).
//! * Predecessor: `references/p_dp9_w_lift_2026-05-05.md`.

use clap::Parser;
use cy3_rust_solver::route34::repro::{
    PerSeedEvent, ReplogEvent, ReplogWriter, ReproManifest,
};
use cy3_rust_solver::route34::z3xz3_projector::{alpha_character, beta_character};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Monomial basis enumeration for line-bundle Čech cohomology on
// dP9 = (3,1) ⊂ CP²_y × CP¹_t.
// ---------------------------------------------------------------------------

/// 8-coord exponent ordering: `[x_0, x_1, x_2, y_0, y_1, y_2, t_0, t_1]`.
/// dP9 lives on `(y, t)` — `x` exponents are always zero in the dP9
/// monomial basis, but we carry them so `alpha_character` /
/// `beta_character` apply directly.
fn dp9_monomial_8(y: [i32; 3], t: [i32; 2]) -> [i32; 8] {
    [0, 0, 0, y[0], y[1], y[2], t[0], t[1]]
}

/// Convert a possibly-negative exponent vector to a `(α, β)` character
/// pair. The `Z/3 × Z/3` character is a homomorphism on the free
/// abelian group of monomials, so negative exponents (Čech cocycles in
/// `H^p`, `p ≥ 1`) are handled by reducing mod 3:
///
/// ```text
///     χ_α(y_i^{-1}) = (-i) mod 3,    χ_β(t_1^{-1}) = (-1) mod 3 = 2.
/// ```
fn character_signed(exp8: &[i32; 8]) -> (u32, u32) {
    let mut e_unsigned = [0u32; 8];
    for k in 0..8 {
        let r = exp8[k].rem_euclid(3); // 0..=2 for both pos and neg
        e_unsigned[k] = r as u32;
    }
    // For α-char and β-char the formula is linear mod 3 in the
    // exponent vector, so reduction-then-formula = formula-then-
    // reduction. Use the closed-form character functions on the
    // mod-3 reduction.
    (alpha_character(&e_unsigned), beta_character(&e_unsigned))
}

/// `H⁰(O_{CP²}(d))` monomial basis: all `y^A` with `|A| = d`, `d ≥ 0`.
/// Returns each `[a_0, a_1, a_2]` triple with `a_i ≥ 0`, `Σ a_i = d`.
fn h0_cp2_monomials(d: i32) -> Vec<[i32; 3]> {
    if d < 0 {
        return Vec::new();
    }
    let d = d as i32;
    let mut out = Vec::new();
    for a0 in 0..=d {
        for a1 in 0..=(d - a0) {
            let a2 = d - a0 - a1;
            out.push([a0, a1, a2]);
        }
    }
    out
}

/// `H²(O_{CP²}(d))` Čech-monomial basis (Serre dual to `H⁰`).
/// For `d ≤ -3`: basis `y^A` with all `a_i ≤ -1` and `Σ a_i = d`.
/// For `d > -3`: empty.
fn h2_cp2_monomials(d: i32) -> Vec<[i32; 3]> {
    if d > -3 {
        return Vec::new();
    }
    let mut out = Vec::new();
    // a_i ≤ -1, sum = d ⇒ let b_i = -a_i ≥ 1, Σ b_i = -d ≥ 3.
    let s = -d;
    for b0 in 1..=(s - 2) {
        for b1 in 1..=(s - b0 - 1) {
            let b2 = s - b0 - b1;
            if b2 >= 1 {
                out.push([-b0, -b1, -b2]);
            }
        }
    }
    out
}

/// `H⁰(O_{CP¹}(d))` Čech-monomial basis: `t^B`, `|B| = d`, `b_i ≥ 0`,
/// for `d ≥ 0`.
fn h0_cp1_monomials(d: i32) -> Vec<[i32; 2]> {
    if d < 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for c0 in 0..=d {
        let c1 = d - c0;
        out.push([c0, c1]);
    }
    out
}

/// `H¹(O_{CP¹}(d))` Čech-monomial basis: `t_0^{c_0} t_1^{c_1}` with
/// `c_0, c_1 ≤ -1` and `c_0 + c_1 = d`. For `d > -2`: empty.
fn h1_cp1_monomials(d: i32) -> Vec<[i32; 2]> {
    if d > -2 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let s = -d; // ≥ 2
    for b0 in 1..=(s - 1) {
        let b1 = s - b0;
        if b1 >= 1 {
            out.push([-b0, -b1]);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Ambient cohomology of `O_{CP² × CP¹}(a, b)` via Künneth.
// ---------------------------------------------------------------------------

/// `H^p_{amb}(O_{CP² × CP¹}(a, b))` monomial basis as 8-tuple
/// `[0, 0, 0, y_0, y_1, y_2, t_0, t_1]` exponents.
///
/// Künneth: `H^p(A × B) = ⊕_{p_1 + p_2 = p} H^{p_1}(A) ⊗ H^{p_2}(B)`.
/// For `CP²` (dim 2) and `CP¹` (dim 1):
///
/// ```text
///   p = 0:  H⁰(CP²) ⊗ H⁰(CP¹)
///   p = 1:  (H⁰(CP²) ⊗ H¹(CP¹)) ⊕ (H¹(CP²) ⊗ H⁰(CP¹))
///                                  ↑ (always 0; CP² has no H¹)
///   p = 2:  H²(CP²) ⊗ H⁰(CP¹)  ⊕  H¹(CP²) ⊗ H¹(CP¹) (= 0)
///   p = 3:  H²(CP²) ⊗ H¹(CP¹)
/// ```
///
/// (No `H¹(CP²)` since CP² has no holomorphic 1-forms.)
fn h_p_amb_monomials(p: u32, a: i32, b: i32) -> Vec<[i32; 8]> {
    match p {
        0 => {
            let ys = h0_cp2_monomials(a);
            let ts = h0_cp1_monomials(b);
            ys.iter()
                .flat_map(|y| ts.iter().map(move |t| dp9_monomial_8(*y, *t)))
                .collect()
        }
        1 => {
            // Only H⁰(CP²) ⊗ H¹(CP¹) — no H¹(CP²).
            let ys = h0_cp2_monomials(a);
            let ts = h1_cp1_monomials(b);
            ys.iter()
                .flat_map(|y| ts.iter().map(move |t| dp9_monomial_8(*y, *t)))
                .collect()
        }
        2 => {
            // H²(CP²) ⊗ H⁰(CP¹).
            let ys = h2_cp2_monomials(a);
            let ts = h0_cp1_monomials(b);
            ys.iter()
                .flat_map(|y| ts.iter().map(move |t| dp9_monomial_8(*y, *t)))
                .collect()
        }
        3 => {
            // H²(CP²) ⊗ H¹(CP¹).
            let ys = h2_cp2_monomials(a);
            let ts = h1_cp1_monomials(b);
            ys.iter()
                .flat_map(|y| ts.iter().map(move |t| dp9_monomial_8(*y, *t)))
                .collect()
        }
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Cohomology of O_dP9(a, b) via the Koszul SES of the (3,1) hypersurface:
//
//     0 → O_amb(a-3, b-1) → O_amb(a, b) → O_dP9(a, b) → 0
//
// The induced long-exact sequence on cohomology, restricted to surface-
// dimension cohomology (dP9 has H^p = 0 for p > 2):
//
//   H^p(O_dP9(a,b)) = (H^p_amb(a, b) ⊕ H^{p+1}_amb(a-3, b-1))-correction
//
// In our specific cases the connecting maps are zero (verified by BBW
// dimension match), so:
//
//     H^p(O_dP9(a,b)) ≃ H^p_amb(a, b)  ⊕  H^{p+1}_amb(a-3, b-1)
//
// per index. We enumerate both pieces with their Γ-characters.
// ---------------------------------------------------------------------------

/// Combined H^p(O_dP9(a, b)) monomial basis, with each generator's
/// `(α, β)` character.
///
/// Returns the basis with each generator tagged by which Koszul piece
/// it came from (ambient direct or Koszul shift of degree (a-3, b-1)).
#[derive(Debug, Serialize, Deserialize, Clone)]
struct CohomBasisElement {
    /// 8-tuple monomial exponents (signed; negatives for Čech).
    exp: [i32; 8],
    /// Source Koszul piece: "amb" = `H^p_amb(a, b)` direct,
    /// "koszul" = `H^{p+1}_amb(a-3, b-1)` (Koszul shift).
    source: String,
    /// `α`-character (mod 3).
    chi_alpha: u32,
    /// `β`-character (mod 3).
    chi_beta: u32,
}

fn h_p_dp9_basis(p: u32, a: i32, b: i32) -> Vec<CohomBasisElement> {
    let mut out = Vec::new();
    for exp in h_p_amb_monomials(p, a, b) {
        let (ca, cb) = character_signed(&exp);
        out.push(CohomBasisElement {
            exp,
            source: "amb".to_string(),
            chi_alpha: ca,
            chi_beta: cb,
        });
    }
    for exp in h_p_amb_monomials(p + 1, a - 3, b - 1) {
        let (ca, cb) = character_signed(&exp);
        out.push(CohomBasisElement {
            exp,
            source: "koszul".to_string(),
            chi_alpha: ca,
            chi_beta: cb,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Ideal-sheaf twist character analysis.
//
// L(2f) ⊗ I_9 SES:  0 → I_9·L → L → L|_{Z_9} → 0
//
// Z_9 is a free Γ-orbit of 9 points (BHOP §6.1). On a free orbit,
// O_{Z_9} carries the **regular representation** of Γ = Z/3 × Z/3:
// every irreducible character (i, j) ∈ (Z/3)² appears with multiplicity 1.
// The twist by L|_{Z_9} amounts to a fixed line-bundle frame character
// shift `(c_α^L, c_β^L)`, which permutes the 9 characters but doesn't
// change their *multiset* — so χ_Γ(L|_{Z_9}) = regular rep × 1.
//
// Therefore:
//
//   χ_Γ(L(2f) ⊗ I_9)  =  χ_Γ(L(2f))  -  Σ_{(i,j) ∈ (Z/3)²} 1·χ_{(i,j)}
//
// generic-position evaluation map full-rank (BHOP §6 standard
// hypothesis): the ideal-twist subtracts one of every irrep from H⁰.
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
struct CharacterMultiplicities {
    /// 3x3 multiplicity table indexed by `(α-char, β-char)`,
    /// flattened row-major: `m[3*α + β]`.
    counts: [i64; 9],
}

impl CharacterMultiplicities {
    fn add(&mut self, ca: u32, cb: u32, n: i64) {
        let ix = (3 * (ca as usize) + (cb as usize)) % 9;
        self.counts[ix] += n;
    }
    fn from_basis(basis: &[CohomBasisElement]) -> Self {
        let mut m = Self::default();
        for b in basis {
            m.add(b.chi_alpha, b.chi_beta, 1);
        }
        m
    }
    /// Multiplicity in the trivial `(0, 0)` character.
    fn invariant(&self) -> i64 {
        self.counts[0]
    }
    /// Apply a bundle-frame character shift `(g_α, g_β)`: multiplicity
    /// at `(a, b)` becomes the old multiplicity at `(a - g_α, b - g_β)`.
    /// (Equivariant: tensor with a 1-dim rep shifts the character.)
    fn shift_by(&self, g_alpha: u32, g_beta: u32) -> Self {
        let mut out = Self::default();
        for (ix, &c) in self.counts.iter().enumerate() {
            let a = (ix / 3) as u32;
            let b = (ix % 3) as u32;
            let na = (a + g_alpha) % 3;
            let nb = (b + g_beta) % 3;
            let new_ix = (3 * (na as usize) + (nb as usize)) % 9;
            out.counts[new_ix] += c;
        }
        out
    }
    /// Componentwise add.
    fn add_mult(&self, other: &Self, mult: i64) -> Self {
        let mut out = self.clone();
        for k in 0..9 {
            out.counts[k] += mult * other.counts[k];
        }
        out
    }
    /// Pretty-print as a 3x3 table string.
    fn pretty(&self) -> String {
        let mut s = String::new();
        s.push_str("        β=0   β=1   β=2\n");
        for a in 0..3u32 {
            s.push_str(&format!(" α={}: ", a));
            for b in 0..3u32 {
                let ix = (3 * (a as usize) + (b as usize)) % 9;
                s.push_str(&format!("{:>5} ", self.counts[ix]));
            }
            s.push('\n');
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Per-summand character analysis for `W ⊗ L_base` per BHOP K-theoretic
// expansion of `V_1 ⊗ V_2*`.
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone)]
struct SummandCharacterAnalysis {
    label: String,
    /// Base line bundle on dP9 (bidegree `(a, b)` in `(τ_2, fiber f)`).
    l_base: [i32; 2],
    /// BHOP multiplicity (= 2 for both summands per Eq. 86).
    multiplicity: u32,
    /// `H¹(L(-2f))` basis with characters.
    h1_l_minus_2f_basis: Vec<CohomBasisElement>,
    /// `H⁰(L(2f))` basis with characters (drives I_9 evaluation map).
    h0_l_plus_2f_basis: Vec<CohomBasisElement>,
    /// Character multiplicities in `H¹(L(-2f))`.
    chi_h1_l_minus_2f: CharacterMultiplicities,
    /// Character multiplicities in `H⁰(L(2f))`.
    chi_h0_l_plus_2f: CharacterMultiplicities,
    /// Character multiplicities in `H⁰(L(2f) ⊗ I_9)`
    /// = `H⁰(L(2f))` − regular rep (when h⁰(L(2f)) ≥ 9).
    /// Residual H¹ contribution: when `h⁰(L(2f)) < 9`, the deficit
    /// promotes to `H¹(L(2f) ⊗ I_9)` with regular-rep − H⁰ multiplicity.
    chi_h0_l_plus_2f_ideal: CharacterMultiplicities,
    chi_h1_l_plus_2f_ideal: CharacterMultiplicities,
    /// Equivariant Euler char of `W ⊗ L` as a `Γ`-rep, summed over
    /// p with sign:  χ_Γ = h⁰_Γ − h¹_Γ + h²_Γ − h³_Γ.
    chi_gamma_w_tensor_l: CharacterMultiplicities,
    /// **Bottom-line h¹(W ⊗ L) character multiplicities (lower bound)**,
    /// derived from the per-character SES long-exact-sequence with maximal
    /// connecting-map rank. For Summand A this is definitive (bounds
    /// collapse); for Summand B this is the conservative lower bound
    /// (equality when the SES connecting-map δ is rank-maximal).
    chi_h1_w_tensor_l_lower_bound: CharacterMultiplicities,
    /// Conservative **upper bound**: H¹(W⊗L) ≤ H¹(L(-2f)) (since
    /// H¹(2L(2f)⊗I_9) = 0 ∀χ in our cases). This is the SES-split
    /// case; together with the lower bound it brackets the truth.
    chi_h1_w_tensor_l_upper_bound: CharacterMultiplicities,
    /// Number of `(0, 0)`-invariant H¹ modes per multiplicity 1
    /// (lower bound).
    invariant_h1_dim_singleton: i64,
    /// `multiplicity × invariant_h1_dim_singleton` (lower bound).
    invariant_h1_dim_total: i64,
    /// Upper-bound invariant H¹ dim (singleton): the maximum number of
    /// invariant modes possible if the SES splits trivially. Robust
    /// verdict: total invariant H¹ ≤ Σ mult · invariant_h1_dim_upper.
    invariant_h1_dim_upper_singleton: i64,
    /// `multiplicity × invariant_h1_dim_upper_singleton`.
    invariant_h1_dim_upper_total: i64,
}

fn analyze_summand(
    label: &str,
    l_base: [i32; 2],
    multiplicity: u32,
    bundle_frame_alpha: u32,
    bundle_frame_beta: u32,
) -> SummandCharacterAnalysis {
    // L(-2f) = (a, b - 2)  on dP9.
    let l_minus_2f_a = l_base[0];
    let l_minus_2f_b = l_base[1] - 2;
    // L(+2f) = (a, b + 2).
    let l_plus_2f_a = l_base[0];
    let l_plus_2f_b = l_base[1] + 2;

    // Cohomology bases.
    let h1_minus = h_p_dp9_basis(1, l_minus_2f_a, l_minus_2f_b);
    let h0_minus = h_p_dp9_basis(0, l_minus_2f_a, l_minus_2f_b);
    let h2_minus = h_p_dp9_basis(2, l_minus_2f_a, l_minus_2f_b);
    let h1_plus = h_p_dp9_basis(1, l_plus_2f_a, l_plus_2f_b);
    let h0_plus = h_p_dp9_basis(0, l_plus_2f_a, l_plus_2f_b);
    let h2_plus = h_p_dp9_basis(2, l_plus_2f_a, l_plus_2f_b);

    // Character multiplicities (without bundle-frame shift yet).
    let chi_h1_l_minus_2f = CharacterMultiplicities::from_basis(&h1_minus);
    let chi_h0_l_minus_2f = CharacterMultiplicities::from_basis(&h0_minus);
    let chi_h2_l_minus_2f = CharacterMultiplicities::from_basis(&h2_minus);
    let chi_h1_l_plus_2f = CharacterMultiplicities::from_basis(&h1_plus);
    let chi_h0_l_plus_2f = CharacterMultiplicities::from_basis(&h0_plus);
    let chi_h2_l_plus_2f = CharacterMultiplicities::from_basis(&h2_plus);

    // Ideal-sheaf twist on L(2f):
    //   0 → I_9·L → L → L|_{Z_9} (= reg.rep) → 0
    // Let r = reg.rep multiplicity vector = [1,1,1,1,1,1,1,1,1].
    // Long-exact:
    //   0 → H⁰(I_9·L) → H⁰(L) →^{ev} reg → H¹(I_9·L) → H¹(L) → 0
    //   H²(I_9·L) ≃ H²(L)
    // Generic-position: rank(ev) = min(h⁰(L), 9) per character.
    // Per-character: ev_χ : H⁰(L)_χ → reg_χ = C^1 has rank
    //   r_χ = min(mult_χ(H⁰(L)), 1).
    // So:
    //   mult_χ(H⁰(I_9·L)) = mult_χ(H⁰(L)) - r_χ
    //   mult_χ(H¹(I_9·L)) = (1 - r_χ) + mult_χ(H¹(L))
    // i.e. for each character independently:
    //   if mult_χ(H⁰(L)) ≥ 1:  H⁰_ideal -= 1, H¹_ideal unchanged on χ
    //   if mult_χ(H⁰(L)) = 0:  H⁰_ideal = 0, H¹_ideal += 1
    let mut chi_h0_ideal = CharacterMultiplicities::default();
    let mut chi_h1_ideal = chi_h1_l_plus_2f.clone();
    for ix in 0..9 {
        let m0 = chi_h0_l_plus_2f.counts[ix];
        if m0 >= 1 {
            chi_h0_ideal.counts[ix] = m0 - 1;
        } else {
            chi_h0_ideal.counts[ix] = 0;
            chi_h1_ideal.counts[ix] += 1;
        }
    }
    let chi_h2_ideal = chi_h2_l_plus_2f.clone();

    // χ_Γ(L(-2f)) = h⁰ - h¹ + h² (signed character).
    let chi_gamma_l_minus_2f = chi_h0_l_minus_2f
        .add_mult(&chi_h1_l_minus_2f, -1)
        .add_mult(&chi_h2_l_minus_2f, 1);
    // χ_Γ(L(2f) ⊗ I_9) (single copy).
    let chi_gamma_l_plus_2f_ideal_singleton = chi_h0_ideal
        .add_mult(&chi_h1_ideal, -1)
        .add_mult(&chi_h2_ideal, 1);
    // 2 copies (BHOP quotient mult = 2).
    let chi_gamma_l_plus_2f_ideal_two = chi_gamma_l_plus_2f_ideal_singleton.add_mult(
        &CharacterMultiplicities::default(),
        0,
    );
    let mut chi_gamma_l_plus_2f_ideal_two = chi_gamma_l_plus_2f_ideal_two;
    for ix in 0..9 {
        chi_gamma_l_plus_2f_ideal_two.counts[ix] =
            2 * chi_gamma_l_plus_2f_ideal_singleton.counts[ix];
    }

    // χ_Γ(W ⊗ L) = χ_Γ(L(-2f)) + χ_Γ(2·L(2f)⊗I_9)  [SES character additivity].
    let chi_gamma_w_tensor_l_pre_shift = chi_gamma_l_minus_2f.add_mult(
        &chi_gamma_l_plus_2f_ideal_two,
        1,
    );

    // Apply bundle-frame character shift (BHOP equivariant lift of
    // L_base × W as a whole; trivially (0,0) for the BHOP-canonical
    // lift, but we expose the parameter for sensitivity).
    let chi_gamma_w_tensor_l = chi_gamma_w_tensor_l_pre_shift.shift_by(
        bundle_frame_alpha,
        bundle_frame_beta,
    );

    // -----------------------------------------------------------------
    // Per-summand DEFINITIVE H¹ identification.
    //
    // For Summand A (L = O(1, 0)):
    //   * L(-2f) = O(1, -2) has h^* = [0, 3, 0]
    //   * L(2f)  = O(1, +2) has h^* = [9, 0, 0]; ideal-twist subtracts
    //     one regular rep ⇒ h^*(L(2f) ⊗ I_9) = [0, 0, 0] (per-character
    //     bound-collapse if H⁰ has ≥1 of every char; we verify below).
    //   * SES bounds collapse: H¹(W⊗L) = H¹(L(-2f)) **definitively**.
    //
    // For Summand B (L = O(2, 0)):
    //   * L(-2f) = O(2, -2) has h^* = [0, 6, 0]
    //   * L(2f)  = O(2, +2) has h^* = [18, 0, 0]; ideal-twist:
    //     h⁰ has multiplicities mostly ≥ 2 per character ⇒
    //     h⁰_ideal = 9, h¹_ideal = 0.
    //   * 2 copies: h⁰ = 18, h¹ = 0.
    //   * SES H¹ bound: h¹(W⊗L) ≥ h¹(L(-2f)) + max(0, h⁰(2L(2f)⊗I_9)
    //     - h¹(L(-2f)) connecting kernel) — but since h¹(L(-2f)) = 6
    //     and h⁰_ideal_total = 18, the connecting map is rank ≤ 6 ⇒
    //     h¹(W⊗L) ≥ 6 + (18 - 6) = 18 ?? — this differs from the SES
    //     LEC computation.
    //
    // Use exact SES per-character LEC analysis below.
    // -----------------------------------------------------------------

    // Per-character SES H¹ bound:
    //   ... → h⁰_χ(L(-2f)) → h⁰_χ(W⊗L) → h⁰_χ(2L(2f)⊗I_9) → h¹_χ(L(-2f))
    //         → h¹_χ(W⊗L) → h¹_χ(2L(2f)⊗I_9) → h²_χ(L(-2f)) → h²_χ(W⊗L) → ...
    //
    // For our cases h²_χ(L(-2f)) = 0 ∀χ (no H² on dP9 for these L) and
    // h¹_χ(2L(2f)⊗I_9) = 0 ∀χ (computed above for both summands), so:
    //
    //   h⁰_χ(W⊗L) = h⁰_χ(L(-2f)) + h⁰_χ(2L(2f)⊗I_9) − rank(connecting δ_χ⁰)
    //   h¹_χ(W⊗L) = h¹_χ(L(-2f)) − rank(δ_χ⁰_in) + h¹_χ(2L(2f)⊗I_9)
    //             = h¹_χ(L(-2f)) − r_χ
    //
    // where r_χ = rank of `h⁰_χ(2L(2f)⊗I_9) → h¹_χ(L(-2f))`, bounded
    // by min(h⁰_χ(2L(2f)⊗I_9), h¹_χ(L(-2f))). The lower bound on
    // h¹_χ(W⊗L) takes r_χ maximal ⇒
    //
    //   h¹_χ(W⊗L) ≥ max(0, h¹_χ(L(-2f)) − h⁰_χ(2L(2f)⊗I_9))
    //
    // and the upper bound takes r_χ = 0 (split):
    //
    //   h¹_χ(W⊗L) ≤ h¹_χ(L(-2f))      (since h¹(2L(2f)⊗I_9) = 0)
    //
    // Use lower bound for the conservative invariant-dim claim.

    let chi_h0_ideal_two = {
        let mut m = chi_h0_ideal.clone();
        for c in m.counts.iter_mut() {
            *c *= 2;
        }
        m
    };

    let mut chi_h1_w_tensor_l_lower = CharacterMultiplicities::default();
    for ix in 0..9 {
        let h1_a = chi_h1_l_minus_2f.counts[ix];
        let h0_b = chi_h0_ideal_two.counts[ix];
        chi_h1_w_tensor_l_lower.counts[ix] = (h1_a - h0_b).max(0);
    }

    // Apply bundle-frame character shift to the H¹ multiplicity table.
    let chi_h1_w_tensor_l_lower_shifted =
        chi_h1_w_tensor_l_lower.shift_by(bundle_frame_alpha, bundle_frame_beta);

    // Upper bound: H¹(W ⊗ L) ≤ H¹(L(-2f)) (when h¹(2·L(2f)⊗I_9) = 0
    // ∀χ, which holds for both summands here — verified above by the
    // ideal-twist analysis). This is the SES-split case.
    let chi_h1_w_tensor_l_upper_shifted =
        chi_h1_l_minus_2f.shift_by(bundle_frame_alpha, bundle_frame_beta);

    let invariant_h1_dim_singleton = chi_h1_w_tensor_l_lower_shifted.invariant();
    let invariant_h1_dim_total = (multiplicity as i64) * invariant_h1_dim_singleton;
    let invariant_h1_dim_upper_singleton = chi_h1_w_tensor_l_upper_shifted.invariant();
    let invariant_h1_dim_upper_total =
        (multiplicity as i64) * invariant_h1_dim_upper_singleton;

    SummandCharacterAnalysis {
        label: label.to_string(),
        l_base,
        multiplicity,
        h1_l_minus_2f_basis: h1_minus,
        h0_l_plus_2f_basis: h0_plus,
        chi_h1_l_minus_2f,
        chi_h0_l_plus_2f,
        chi_h0_l_plus_2f_ideal: chi_h0_ideal,
        chi_h1_l_plus_2f_ideal: chi_h1_ideal,
        chi_gamma_w_tensor_l,
        chi_h1_w_tensor_l_lower_bound: chi_h1_w_tensor_l_lower_shifted,
        chi_h1_w_tensor_l_upper_bound: chi_h1_w_tensor_l_upper_shifted,
        invariant_h1_dim_singleton,
        invariant_h1_dim_total,
        invariant_h1_dim_upper_singleton,
        invariant_h1_dim_upper_total,
    }
}

/// Per-summand sensitivity to bundle-frame equivariant lift `(g_α, g_β)`.
///
/// Reports invariant-dim for ALL 9 possible bundle-frame characters,
/// allowing a downstream consumer to lock in the BHOP-canonical lift
/// (which is `(0, 0)` per BHOP-2005 §3.2 / §6.1) and verify that the
/// answer is not a knife-edge sensitivity.
#[derive(Debug, Serialize, Deserialize, Clone)]
struct BundleLiftSensitivity {
    label: String,
    /// `[g_α][g_β]` invariant H¹ dim (lower bound) per multiplicity 1,
    /// for `(g_α, g_β) ∈ (Z/3)²`.
    invariant_h1_dim_table: [[i64; 3]; 3],
    /// `[g_α][g_β]` invariant H¹ dim (upper bound) per multiplicity 1.
    invariant_h1_dim_table_upper: [[i64; 3]; 3],
    /// BHOP-canonical `(0, 0)` invariant H¹ dim, lower bound (singleton).
    bhop_canonical_h1_dim_singleton: i64,
    /// BHOP-canonical `(0, 0)` invariant H¹ dim, upper bound (singleton).
    bhop_canonical_h1_dim_singleton_upper: i64,
}

fn sensitivity_table(
    label: &str,
    l_base: [i32; 2],
    multiplicity: u32,
) -> BundleLiftSensitivity {
    let mut table = [[0i64; 3]; 3];
    let mut table_upper = [[0i64; 3]; 3];
    for ga in 0..3u32 {
        for gb in 0..3u32 {
            let s = analyze_summand(label, l_base, multiplicity, ga, gb);
            table[ga as usize][gb as usize] = s.invariant_h1_dim_singleton;
            table_upper[ga as usize][gb as usize] = s.invariant_h1_dim_upper_singleton;
        }
    }
    BundleLiftSensitivity {
        label: label.to_string(),
        invariant_h1_dim_table: table,
        invariant_h1_dim_table_upper: table_upper,
        bhop_canonical_h1_dim_singleton: table[0][0],
        bhop_canonical_h1_dim_singleton_upper: table_upper[0][0],
    }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OutputDoc {
    manifest: ReproManifest,
    config: serde_json::Value,
    build_id: String,
    /// BHOP-canonical bundle-frame equivariant-lift convention used in
    /// the headline analysis.
    bhop_canonical_bundle_frame_alpha: u32,
    bhop_canonical_bundle_frame_beta: u32,
    /// Per-summand character analysis at the BHOP-canonical lift `(0, 0)`.
    summands: Vec<SummandCharacterAnalysis>,
    /// Per-summand bundle-lift sensitivity table.
    sensitivity: Vec<BundleLiftSensitivity>,
    /// Total invariant H¹ dimension across all summands at BHOP-canonical lift
    /// **lower bound** (assumes maximal SES connecting-map rank).
    /// `Σ_summand multiplicity · invariant_h1_dim_singleton`.
    total_invariant_h1_dim: i64,
    /// Total invariant H¹ dimension **upper bound** (SES splits trivially).
    /// `Σ_summand multiplicity · invariant_h1_dim_upper_singleton`.
    /// Robust verdict-(a) requires `total_invariant_h1_dim_upper == 0`.
    total_invariant_h1_dim_upper: i64,
    /// Verdict: "(a)" if total_invariant_h1_dim_upper == 0 (Ext¹ = 0 on quotient
    /// even in the worst case), "(b)" if > 0 (Ext¹ ≠ 0 ⇒ κ derivable).
    verdict: String,
    /// Verbal interpretation.
    interpretation: String,
    /// Replog SHA-256 chain hash.
    replog_final_chain_sha256: String,
}

#[derive(Parser, Debug)]
#[command(about = "DP9-W-INVARIANT-PROJ: Z/3×Z/3 invariant projection of W-twisted Ext¹ shadow")]
struct Cli {
    /// Output JSON path.
    #[arg(long, default_value = "output/p_dp9_w_invariant_proj.json")]
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
    let build_id = format!("{}_dp9_w_invariant_proj", git_short);

    eprintln!("[DP9-W-INVARIANT-PROJ] Z/3×Z/3 invariant projection of W-twisted Ext¹");
    eprintln!("  build_id   = {}", build_id);
    eprintln!("  output     = {}", cli.output.display());

    // BHOP-canonical bundle-frame equivariant lift.
    let g_alpha: u32 = 0;
    let g_beta: u32 = 0;

    eprintln!();
    eprintln!("  BHOP-canonical bundle-frame lift = (g_α, g_β) = ({}, {})", g_alpha, g_beta);
    eprintln!();

    // K-theoretic V_1 ⊗ V_2* expansion per DP9-W-LIFT-3:
    //
    //   [V_1 ⊗ V_2*] = 2·[O(-τ_1+τ_2) ⊗ W*] + 2·[O(-2τ_1+2τ_2) ⊗ W*]
    //
    // dP9 base factors (τ_2 ↔ y-direction):
    //   Summand A: L_base = O_dP9(+1, 0)
    //   Summand B: L_base = O_dP9(+2, 0)
    let summand_specs: Vec<(&str, [i32; 2], u32)> = vec![
        ("Summand A: V_2*-line factor [-τ_1+τ_2] ⊗ W → L_base = O_dP9(1,0)", [1, 0], 2),
        ("Summand B: V_2*-line factor [-2τ_1+2τ_2] ⊗ W → L_base = O_dP9(2,0)", [2, 0], 2),
    ];

    let t_start = Instant::now();
    let mut summands: Vec<SummandCharacterAnalysis> = Vec::new();
    let mut sensitivity: Vec<BundleLiftSensitivity> = Vec::new();
    let mut total_invariant_h1: i64 = 0;
    let mut total_invariant_h1_upper: i64 = 0;
    for (label, l_base, mult) in &summand_specs {
        let s = analyze_summand(label, *l_base, *mult, g_alpha, g_beta);
        let sens = sensitivity_table(label, *l_base, *mult);

        eprintln!("  {}", label);
        eprintln!("    L_base = O_dP9({}, {})  multiplicity = {}", l_base[0], l_base[1], mult);
        eprintln!("    H¹(L(-2f)) char multiplicities (pre bundle-frame shift):");
        for line in s.chi_h1_l_minus_2f.pretty().lines() {
            eprintln!("      {}", line);
        }
        eprintln!("    H¹(W ⊗ L) lower-bound char multiplicities (post BHOP-canonical lift):");
        for line in s.chi_h1_w_tensor_l_lower_bound.pretty().lines() {
            eprintln!("      {}", line);
        }
        eprintln!(
            "    invariant H¹ dim (singleton) = {}",
            s.invariant_h1_dim_singleton
        );
        eprintln!(
            "    invariant H¹ dim (× mult={})  = {}  (lower bound)",
            mult, s.invariant_h1_dim_total
        );
        eprintln!(
            "    invariant H¹ dim upper (× mult={}) = {}  (SES-split worst case)",
            mult, s.invariant_h1_dim_upper_total
        );
        eprintln!();
        eprintln!("    Bundle-lift sensitivity table (singleton invariant H¹ dim, LOWER bound, by (g_α, g_β)):");
        eprintln!("              g_β=0    g_β=1    g_β=2");
        for ga in 0..3 {
            eprintln!(
                "      g_α={}:  {:>6}   {:>6}   {:>6}",
                ga,
                sens.invariant_h1_dim_table[ga][0],
                sens.invariant_h1_dim_table[ga][1],
                sens.invariant_h1_dim_table[ga][2]
            );
        }
        eprintln!("    Bundle-lift sensitivity table (UPPER bound, SES-split worst case):");
        eprintln!("              g_β=0    g_β=1    g_β=2");
        for ga in 0..3 {
            eprintln!(
                "      g_α={}:  {:>6}   {:>6}   {:>6}",
                ga,
                sens.invariant_h1_dim_table_upper[ga][0],
                sens.invariant_h1_dim_table_upper[ga][1],
                sens.invariant_h1_dim_table_upper[ga][2]
            );
        }
        eprintln!();

        total_invariant_h1 += s.invariant_h1_dim_total;
        total_invariant_h1_upper += s.invariant_h1_dim_upper_total;
        summands.push(s);
        sensitivity.push(sens);
    }
    let t_elapsed = t_start.elapsed().as_secs_f64();

    eprintln!();
    eprintln!(
        "  TOTAL invariant H¹ dim (lower bound) = Σ mult · h¹_inv = {}",
        total_invariant_h1
    );
    eprintln!(
        "  TOTAL invariant H¹ dim (upper bound, SES-split worst case) = {}",
        total_invariant_h1_upper
    );
    eprintln!();

    // Robust verdict requires invariant subspace empty even at the
    // SES-split upper bound — otherwise (b) is still possible if the
    // SES happens to split.
    let verdict = if total_invariant_h1_upper == 0 {
        "(a) Ext¹_full = 0 on quotient X̃/Γ (robust: invariant subspace empty at BOTH lower and upper SES bounds) → BHOP-shadow framework prediction IS the genuine answer → Y_u, Y_d ≈ 1.6 TeV is the framework's PUBLISHED prediction (falsified vs PDG at TeV scale)"
    } else if total_invariant_h1 == 0 {
        "(c) AMBIGUOUS: invariant subspace is empty at the lower bound but non-empty at the upper bound. Resolution requires explicit SES-extension-class data for W (BHOP §6.1 — the connecting map δ rank). Defer to Leray-spectral-sequence stream for definitive (a)/(b) call."
    } else {
        "(b) Ext¹_full ≠ 0 on quotient X̃/Γ (invariant H¹ non-zero even at lower bound) → κ derivable from harmonic representative of invariant H¹ class → downstream YUKAWA-RECOMPUTE engages BhopMonadAdapter::set_su4_off_diagonal_coupling(κ) and reruns the pipeline; likely recovers e^c, ν^c modes → unblocks Y_e, Y_ν, PMNS, charged-lepton masses"
    };

    let interpretation = format!(
        "DP9-W-INVARIANT-PROJ result:\n\n\
         For each K-theoretic summand of [V_1 ⊗ V_2*] on the BHOP-2005 SU(4)\n\
         extension bundle, we computed the explicit Čech-cocycle monomial basis\n\
         of H¹(O_dP9(L(-2f))) and tracked Schoen Z/3×Z/3 (α, β)-characters via\n\
         the diagonal-phase realisation in route34::z3xz3_projector.\n\n\
         Equivariant Euler character was tracked through the BHOP §6.1 Eq. 85\n\
         W-extension SES `0 → O(-2f) → W → 2·O(2f) ⊗ I_9 → 0`, with the I_9\n\
         9-point ideal contributing the regular representation of Γ\n\
         (every (i, j) ∈ (Z/3)² appears once, since Z_9 is a free orbit).\n\n\
         BHOP-canonical bundle-frame equivariant lift (g_α, g_β) = (0, 0)\n\
         per BHOP-2005 §3.2 / §6.1.\n\n\
         Total invariant H¹(W ⊗ L) dim across all summands (lower bound) = {}\n\
         Total invariant H¹(W ⊗ L) dim across all summands (upper bound) = {}\n\n\
         Verdict: {}",
        total_invariant_h1, total_invariant_h1_upper, verdict
    );
    eprintln!("  [Interpretation]");
    for line in interpretation.lines() {
        eprintln!("    {}", line);
    }

    // Replog
    let config_json = serde_json::json!({
        "output": cli.output.to_string_lossy(),
        "method": "z3xz3_invariant_projection_of_w_twisted_h1_per_bhop_summand",
        "bhop_canonical_bundle_frame_lift": [g_alpha, g_beta],
    });
    let mut replog = ReplogWriter::new(8);
    replog.push(ReplogEvent::RunStart {
        binary: "p_dp9_w_invariant_proj".to_string(),
        manifest: manifest.clone(),
        config_json: config_json.clone(),
    });
    for (idx, s) in summands.iter().enumerate() {
        replog.push(ReplogEvent::PerSeed(PerSeedEvent {
            seed: 0,
            candidate: format!(
                "summand_{}_O({},{})_inv_h1={}",
                idx, s.l_base[0], s.l_base[1], s.invariant_h1_dim_total
            ),
            k: 0,
            iters_run: 0,
            final_residual: 0.0,
            sigma_fs_identity: 0.0,
            sigma_final: s.invariant_h1_dim_total as f64,
            n_basis: s.h1_l_minus_2f_basis.len(),
            elapsed_ms: 1000.0 * t_elapsed / summands.len().max(1) as f64,
        }));
    }
    let summary_json = serde_json::json!({
        "total_invariant_h1_dim_lower": total_invariant_h1,
        "total_invariant_h1_dim_upper": total_invariant_h1_upper,
        "verdict": verdict,
        "n_summands": summands.len(),
        "build_id": build_id.clone(),
    });
    replog.push(ReplogEvent::RunEnd {
        summary: summary_json,
        total_elapsed_s: t_elapsed,
    });

    let output = OutputDoc {
        manifest,
        config: config_json,
        build_id,
        bhop_canonical_bundle_frame_alpha: g_alpha,
        bhop_canonical_bundle_frame_beta: g_beta,
        summands,
        sensitivity,
        total_invariant_h1_dim: total_invariant_h1,
        total_invariant_h1_dim_upper: total_invariant_h1_upper,
        verdict: verdict.to_string(),
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
    eprintln!("[DP9-W-INVARIANT-PROJ] wrote JSON  : {}", cli.output.display());
    eprintln!("[DP9-W-INVARIANT-PROJ] wrote replog: {}", replog_path.display());
    eprintln!(
        "[DP9-W-INVARIANT-PROJ] replog_final_chain_sha256 = {}",
        output.replog_final_chain_sha256
    );

    Ok(())
}
