//! # Integration tests for [`crate::route34::wilson_line_e8`]
//!
//! Cross-checks the structured E_8 Wilson-line element against
//! the canonical E_8 root system and the `E_8 → E_6 × SU(3)`
//! breaking pattern from Slansky 1981 / Anderson-Gray-Lukas-Palti
//! 2011 / Braun-He-Ovrut-Pantev 2005.
//!
//! ## References used in this test file
//!
//! * Slansky, "Group Theory for Unified Model Building",
//!   *Phys. Rep.* **79** (1981) 1, §6 and Table 16.
//! * Anderson-Gray-Lukas-Palti 2011, arXiv:1106.4804 (the
//!   AGLP-2011 Wilson-line catalogue for `Z/3` quotients).
//! * Braun-He-Ovrut-Pantev 2005, arXiv:hep-th/0501070
//!   (the BHOP-2005 Wilson-line catalogue for `Z/3 × Z/3`).
//! * Bourbaki, *Lie Groups and Lie Algebras* Ch. VI §4 Plate VII.

use crate::route34::bundle_search::LineBundleDegrees;
use crate::route34::wilson_line_e8::{
    e8_roots, in_e8_root_lattice, WilsonLineE8, E8_SIMPLE_ROOTS,
};

// ----------------------------------------------------------------------
// Test 1: E_8 root system structural correctness.
// ----------------------------------------------------------------------

/// E_8 has exactly 240 roots (`dim E_8 - rank E_8 = 248 - 8`).
#[test]
fn test_e8_has_240_roots() {
    let r = e8_roots();
    assert_eq!(r.len(), 240);
}

/// Every E_8 root has Euclidean norm² = 2 (standard normalisation
/// in which simple roots are length √2).
#[test]
fn test_e8_roots_norm_squared_two() {
    for r in e8_roots() {
        let n2: f64 = r.iter().map(|&x| x * x).sum();
        assert!(
            (n2 - 2.0).abs() < 1.0e-9,
            "root norm² should be 2, got {n2} for {r:?}"
        );
    }
}

/// All 8 simple roots are in the E_8 root system. (They had
/// better be — they generate it.)
#[test]
fn test_simple_roots_are_roots() {
    let all_roots = e8_roots();
    for s in E8_SIMPLE_ROOTS.iter() {
        let found = all_roots.iter().any(|r| {
            (0..8).all(|k| (r[k] - s[k]).abs() < 1.0e-9)
        });
        assert!(found, "simple root {s:?} should appear in e8_roots()");
    }
}

/// E_8 roots are closed under negation.
#[test]
fn test_root_system_closed_under_negation() {
    let roots = e8_roots();
    for r in &roots {
        let neg = [
            -r[0], -r[1], -r[2], -r[3], -r[4], -r[5], -r[6], -r[7],
        ];
        let found = roots.iter().any(|q| {
            (0..8).all(|k| (q[k] - neg[k]).abs() < 1.0e-9)
        });
        assert!(found, "negative of {r:?} not in root system");
    }
}

// ----------------------------------------------------------------------
// Test 2: Z/3 Wilson line at canonical α leaves E_6 × SU(3) unbroken.
// ----------------------------------------------------------------------

/// **Slansky 1981 Table 23 / AGLP-2011 / BHOP-2005:** the
/// canonical `Z/3` Wilson line
/// `α = (1/3) · (2, -1, -1, 0, 0, 0, 0, 0)`
/// breaks `E_8 → E_6 × SU(3)` exactly. The unbroken subalgebra
/// has dimension `78 (E_6) + 8 (SU(3)) = 86`.
///
/// Verification (matches Slansky 1981 §6):
///
/// * `D_8` roots `±e_i ± e_j` with `i, j ∈ {3, …, 7}`: 40
///   invariant.
/// * `D_8` roots `±(e_0 - e_1)`, `±(e_0 - e_2)`, `±(e_1 - e_2)`:
///   6 invariant.
/// * Spinor roots with `(s_0, s_1, s_2) ∈ {(+,+,+), (-,-,-)}`
///   and even total minus-count: 32 invariant.
///
/// Total: `40 + 6 + 32 = 78` roots, plus the `8`-dim Cartan,
/// gives `86 = dim(E_6 × SU(3))`.
#[test]
fn test_z3_wilson_line_unbroken_e6_su3() {
    let w = WilsonLineE8::canonical_e8_to_e6_su3(3);
    let h = w.unbroken_subalgebra();
    assert_eq!(
        h.lie_dimension(),
        86,
        "E_6 × SU(3) has dim 78 + 8 = 86; got {}",
        h.lie_dimension()
    );
    assert_eq!(h.invariant_root_count, 78);
    assert!(h.is_e6_times_su3());
    assert!(!h.is_full_e8());
}

/// **Quantization:** at the canonical α, the Wilson line satisfies
/// `W^3 = 1` exactly (residual = 0).
#[test]
fn test_quantization_residual_zero_at_canonical() {
    let w = WilsonLineE8::canonical_e8_to_e6_su3(3);
    let q = w.quantization_residual();
    assert!(
        q < 1.0e-9,
        "canonical Z/3 Wilson line: quantization residual should be 0, got {q}"
    );
}

/// At α = 0, no breaking — full E_8 unbroken.
#[test]
fn test_zero_alpha_full_e8() {
    let w = WilsonLineE8::new([0.0; 8], 3);
    let h = w.unbroken_subalgebra();
    assert!(h.is_full_e8());
    assert_eq!(h.invariant_root_count, 240);
    assert_eq!(h.lie_dimension(), 248);
}

/// **Generic non-canonical α:** picking a "random" Cartan vector
/// that does NOT lie on the canonical coweight should produce a
/// non-zero quantization residual, indicating `W^3 ≠ 1`.
#[test]
fn test_generic_alpha_violates_quantization() {
    let w = WilsonLineE8::new(
        [0.1, 0.2, 0.3, 0.05, 0.4, -0.15, 0.07, 0.22],
        3,
    );
    let q = w.quantization_residual();
    assert!(
        q > 1.0e-3,
        "generic α should not lie on Z/3 quantization manifold, got q = {q}"
    );
}

// ----------------------------------------------------------------------
// Test 3: Wilson-line / bundle-holonomy compatibility.
// ----------------------------------------------------------------------

/// **Embedding compatibility:** the canonical Z/3 Wilson line
/// commutes with any rank-≥1 SU(n) bundle (c_1 = 0). This is the
/// content of `WilsonLineE8::embeds_for_se`.
#[test]
fn test_invariance_under_bundle_holonomy() {
    let w = WilsonLineE8::canonical_e8_to_e6_su3(3);
    // SU(5) AGLP shape: B = (1,1,1,1,2), C = (6). c_1 = 0.
    let lb = LineBundleDegrees::new(vec![1, 1, 1, 1, 2], vec![6]);
    assert_eq!(lb.derived_chern().c1, 0);
    assert!(
        w.embeds_for_se(&lb),
        "canonical Wilson line must embed compatibly with c_1 = 0 SU(n) bundle"
    );
}

/// **Embedding incompatibility:** a bundle with c_1 ≠ 0 cannot be
/// the structure-group commutant of the canonical Wilson line.
#[test]
fn test_bundle_with_nonzero_c1_rejected() {
    let w = WilsonLineE8::canonical_e8_to_e6_su3(3);
    // c_1 ≠ 0 example.
    let lb = LineBundleDegrees::new(vec![1, 1, 1, 1, 2], vec![5]);
    assert_ne!(lb.derived_chern().c1, 0);
    assert!(
        !w.embeds_for_se(&lb),
        "c_1 ≠ 0 bundle should be rejected by embeds_for_se"
    );
}

/// **Embedding incompatibility:** a Wilson line with a non-zero
/// quantization residual cannot embed.
#[test]
fn test_unquantized_wilson_rejected() {
    // Take canonical α and add 1e-2 to the last component, breaking quantization.
    let w = WilsonLineE8::new(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0 / 3.0, 2.0 / 3.0 + 0.01],
        3,
    );
    let lb = LineBundleDegrees::new(vec![1, 1, 1, 1, 2], vec![6]);
    assert!(
        !w.embeds_for_se(&lb),
        "non-quantized Wilson line should fail embeds_for_se"
    );
}

/// **Empty bundle:** rank ≤ 0 is rejected.
#[test]
fn test_empty_bundle_rejected() {
    let w = WilsonLineE8::canonical_e8_to_e6_su3(3);
    let lb = LineBundleDegrees::new(vec![], vec![]);
    assert!(!w.embeds_for_se(&lb));
}

// ----------------------------------------------------------------------
// Test 4: root lattice membership.
// ----------------------------------------------------------------------

/// `3 · α_canonical = (2, -1, -1, 0, 0, 0, 0, 0)` must lie in the
/// E_8 root lattice (integer coordinates with even sum;
/// 2 + (-1) + (-1) = 0, even, hence in the D_8 sublattice).
#[test]
fn test_three_alpha_canonical_in_root_lattice() {
    let v = [2.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert!(
        in_e8_root_lattice(&v),
        "3 α_canonical must lie in Λ_root(E_8)"
    );
}

/// A vector with odd integer sum is NOT in the E_8 lattice
/// (D_8 sublattice has even sum).
#[test]
fn test_odd_sum_not_in_lattice() {
    let v = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert!(!in_e8_root_lattice(&v));
}

/// Spinor lattice vector `(1/2, 1/2, ..., 1/2)` (zero −1/2 signs)
/// is in E_8.
#[test]
fn test_spinor_all_plus_in_lattice() {
    let v = [0.5; 8];
    assert!(in_e8_root_lattice(&v));
}

// ----------------------------------------------------------------------
// Test 5: serde round-trip.
// ----------------------------------------------------------------------

#[test]
fn test_wilson_line_serde_roundtrip() {
    let w = WilsonLineE8::canonical_e8_to_e6_su3(3);
    let json = serde_json::to_string(&w).expect("ser");
    let w2: WilsonLineE8 = serde_json::from_str(&json).expect("de");
    assert_eq!(w, w2);
}

// ----------------------------------------------------------------------
// Test 6: cross-check against AGLP-2011 catalogue choice.
// ----------------------------------------------------------------------

/// Slansky-1981 / AGLP-2011 canonical Wilson-line representative
/// `α = (1/3) · (2, -1, -1, 0, 0, 0, 0, 0)`.
#[test]
fn test_aglp_2011_canonical_choice() {
    let w = WilsonLineE8::canonical_e8_to_e6_su3(3);
    let expected = [
        2.0 / 3.0,
        -1.0 / 3.0,
        -1.0 / 3.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    for k in 0..8 {
        assert!(
            (w.cartan_phases[k] - expected[k]).abs() < 1.0e-12,
            "cartan_phases[{k}] = {}, expected {}",
            w.cartan_phases[k],
            expected[k]
        );
    }
    assert_eq!(w.quotient_order, 3);
}

/// **Schoen extension to `Z/3 × Z/3`:** the same canonical α
/// works, with quotient_order = 9. The integrality condition is
/// `9 · α ∈ Λ_root`, which becomes `(0, ..., -6, 6)` — still a
/// D_8 lattice element.
#[test]
fn test_schoen_z9_canonical_quantization() {
    let w = WilsonLineE8::canonical_e8_to_e6_su3(9);
    let q = w.quantization_residual();
    assert!(
        q < 1.0e-9,
        "Z/9 canonical Wilson line should be quantized, got q = {q}"
    );
    assert_eq!(w.quotient_order, 9);
    let h = w.unbroken_subalgebra();
    // The unbroken subgroup depends on the integrality condition
    // ⟨β, α⟩ ∈ Z. α is the same vector regardless of quotient_order,
    // so the unbroken subgroup is also E_6 × SU(3).
    assert!(h.is_e6_times_su3());
}
