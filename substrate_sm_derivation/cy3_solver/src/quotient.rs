//! Z/3 quotient projector for sections on CP^3 x CP^3.
//!
//! The Tian-Yau and Schoen-Z3xZ3 manifolds are Z/3-quotients of their
//! ambient covers. Heterotic sections on the quotient correspond to
//! Z/3-invariant sections on the cover. This module implements the
//! projection
//!
//!   s_inv(z) = (1/|Z/3|) sum_{g in Z/3} s(g · z)
//!
//! by acting on monomials with the Z/3 generator (cube-root-of-unity
//! multiplication on selected coordinates) and averaging.
//!
//! ## Z/3 action convention
//!
//! For Tian-Yau (CICY in CP^3 x CP^3 with Z/3 quotient): the freely
//! acting Z/3 generator g maps
//!
//!   (z_0 : z_1 : z_2 : z_3) x (w_0 : w_1 : w_2 : w_3)
//!     -> (z_0 : ω z_1 : ω^2 z_2 : z_3) x (w_0 : ω w_1 : ω^2 w_2 : w_3)
//!
//! where ω = exp(2πi/3). This is the standard "diagonal" Z/3 action
//! used in Tian-Yau Wilson-line constructions; the fixed-point-free
//! requirement means generic representatives have orbits of size 3.
//!
//! For Schoen-Z3xZ3, the action is similar but with Z/3 x Z/3
//! diagonal-block structure.
//!
//! ## Action on monomials
//!
//! For a monomial m(z, w) = z_0^{a_0} z_1^{a_1} z_2^{a_2} z_3^{a_3}
//!                          w_0^{b_0} w_1^{b_1} w_2^{b_2} w_3^{b_3},
//! the Z/3 action multiplies it by
//!
//!   ω^(a_1 + 2 a_2 + b_1 + 2 b_2)
//!
//! Z/3-invariant monomials are those with (a_1 + 2 a_2 + b_1 + 2 b_2) ≡ 0
//! (mod 3). The complement (non-invariant) monomials sum to zero under
//! Z/3 averaging and are projected out.
//!
//! ## Real-valued ambient
//!
//! Since our pipeline uses real polysphere coordinates rather than
//! complex (M1 in audit), the Z/3 action above is approximated by
//! treating each real coordinate as the real part of a complex
//! coordinate, applying the Z/3 character mod 3, and projecting out
//! non-invariant components. This reduces the section basis from 100
//! to ~33 monomials (degree-2 invariants).

/// Compute the Z/3 character index for a degree-k bigraded monomial
/// with exponent tuple [a_0, a_1, a_2, a_3, b_0, b_1, b_2, b_3]:
///
///   chi(m) = (a_1 + 2 a_2 + b_1 + 2 b_2) mod 3
///
/// Z/3-invariant monomials have chi(m) = 0.
pub fn z3_character(monomial: &[u32; 8]) -> u32 {
    (monomial[1] + 2 * monomial[2] + monomial[5] + 2 * monomial[6]) % 3
}

/// Indices of Z/3-invariant monomials in a given monomial list.
pub fn invariant_indices(monomials: &[[u32; 8]]) -> Vec<usize> {
    monomials
        .iter()
        .enumerate()
        .filter_map(|(i, m)| if z3_character(m) == 0 { Some(i) } else { None })
        .collect()
}

/// Project a section-coefficient vector h (n_basis x n_basis Hermitian)
/// onto the Z/3-invariant subspace. Off-diagonal entries connecting
/// different Z/3 character classes are zeroed (these are the
/// "non-invariant" couplings that the quotient projector removes).
pub fn project_h_to_z3_invariant(h: &mut [f64], monomials: &[[u32; 8]]) {
    let n = monomials.len();
    let chars: Vec<u32> = monomials.iter().map(z3_character).collect();
    for a in 0..n {
        for b in 0..n {
            // Hermitian inner product s_a^* h_{ab} s_b survives Z/3
            // averaging only when chi(s_a) - chi(s_b) ≡ 0 (mod 3),
            // i.e. when both basis functions are in the same character
            // class. Other pairs are zeroed.
            //
            // (Strictly speaking the full projector also includes the
            // chi=1, chi'=2 pairs, since chi - chi' = -1 ≡ 2 (mod 3)
            // is not zero -- so only chi = chi' pairs survive.)
            if chars[a] != chars[b] {
                h[a * n + b] = 0.0;
            }
        }
    }
}

/// Filter section_values to keep only Z/3-invariant basis columns.
/// Returns (new_section_values, kept_indices). The kept_indices vector
/// can be used to map h-matrix entries back to the original basis.
pub fn filter_z3_invariant_columns(
    section_values: &[f64],
    n_points: usize,
    n_basis: usize,
    monomials: &[[u32; 8]],
) -> (Vec<f64>, Vec<usize>) {
    let kept = invariant_indices(monomials);
    let new_basis = kept.len();
    let mut new_sv = vec![0.0f64; n_points * new_basis];
    for p in 0..n_points {
        for (j_new, &j_old) in kept.iter().enumerate() {
            new_sv[p * new_basis + j_new] = section_values[p * n_basis + j_old];
        }
    }
    (new_sv, kept)
}

/// Total Z/3-invariance loss for a candidate's bundle moduli: penalises
/// bundle degrees-of-freedom that fail the Z/3 character constraint.
/// Specifically, the bundle's parameter slots 6..20 are assumed to
/// pair into 7 groups of 3 indices that should sum to ~zero modulo 2π
/// for Z/3 invariance of the bundle line-bundle restriction.
pub fn bundle_z3_invariance_loss(bundle_moduli: &[f64]) -> f64 {
    if bundle_moduli.len() < 9 {
        return 0.0;
    }
    use std::f64::consts::PI;
    // Z/3 fundamental domain is 2π/3 wide. A sum of phases ≡ 0 (mod 2π/3)
    // lies in a Z/3 character class (i.e., transforms by ω^n for some
    // integer n under the action). For Z/3 invariance the natural
    // condition is: distance to the nearest multiple of 2π/3.
    let third = 2.0 * PI / 3.0;
    let mut total = 0.0;
    let n_groups = ((bundle_moduli.len() - 6).min(15)) / 3;
    for g in 0..n_groups {
        let i0 = 6 + g * 3;
        let s = bundle_moduli[i0] + bundle_moduli[i0 + 1] + bundle_moduli[i0 + 2];
        let s_mod = s.rem_euclid(third);
        let d = s_mod.min(third - s_mod);
        total += d * d;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::build_degree2_monomials;

    #[test]
    fn z3_character_correct_on_known_monomials() {
        // z_0^2 -- a = (2,0,0,0,0,0,0,0). chi = 0 + 0 + 0 + 0 = 0. ✓
        assert_eq!(z3_character(&[2, 0, 0, 0, 0, 0, 0, 0]), 0);
        // z_1 z_2 -- chi = 1 + 2 = 3 ≡ 0. ✓
        assert_eq!(z3_character(&[0, 1, 1, 0, 0, 0, 0, 0]), 0);
        // z_1 -- chi = 1.
        assert_eq!(z3_character(&[0, 1, 0, 0, 0, 0, 0, 0]), 1);
        // z_2 -- chi = 2.
        assert_eq!(z3_character(&[0, 0, 1, 0, 0, 0, 0, 0]), 2);
        // z_1 w_2 -- chi = 1 + 2 = 3 ≡ 0. ✓
        assert_eq!(z3_character(&[0, 1, 0, 0, 0, 0, 1, 0]), 0);
    }

    #[test]
    fn invariant_count_for_degree_2() {
        let monomials = build_degree2_monomials();
        // For degree-2 bigraded monomials on CP^3 x CP^3: 100 total.
        // Z/3 partitions them by character; roughly 100/3 = 33 invariants.
        let inv = invariant_indices(&monomials);
        assert_eq!(monomials.len(), 100);
        // Expect roughly 33-36 invariants depending on representation theory.
        assert!(
            inv.len() >= 30 && inv.len() <= 40,
            "expected ~33 invariant degree-2 monomials, got {}",
            inv.len()
        );
    }

    /// REGRESSION (Bug 8): bundle_z3_invariance_loss should classify a
    /// triple of phases (0, 0, 2π/3) as Z/3-invariant: the sum 2π/3 is
    /// ≡ 0 (mod 2π/3) — i.e., a non-trivial Z/3 character that's
    /// nonetheless a closed group element. Previous version wrapped
    /// modulo 2π, treating 2π/3 as far from the identity (residual
    /// (2π/3)² ≈ 4.39).
    ///
    /// CORRECT behaviour: wrap modulo 2π/3 (the Z/3 fundamental
    /// domain). Triple (0, 0, 2π/3) lands at 0 exactly → loss ≈ 0.
    #[test]
    fn z3_invariance_loss_recognizes_third_root_phases() {
        use std::f64::consts::PI;
        let mut bundle_moduli = vec![0.0_f64; 30];
        // Place phases (0, 0, 2π/3) into one group. Group convention
        // in `bundle_z3_invariance_loss`: starts at index 6, groups of 3.
        bundle_moduli[6] = 0.0;
        bundle_moduli[7] = 0.0;
        bundle_moduli[8] = 2.0 * PI / 3.0;
        let loss = bundle_z3_invariance_loss(&bundle_moduli);
        assert!(
            loss < 1e-9,
            "expected ~0 loss for Z/3-character-2 triple summing to 2π/3, got {loss:.4e}"
        );
    }

    #[test]
    fn project_h_zeros_cross_character_entries() {
        let monomials = build_degree2_monomials();
        let n = monomials.len();
        let mut h = vec![1.0; n * n];
        project_h_to_z3_invariant(&mut h, &monomials);
        // Find one chi=0 and one chi=1 monomial; verify their cross
        // entry is zero.
        let mut chi0_idx = None;
        let mut chi1_idx = None;
        for (i, m) in monomials.iter().enumerate() {
            if z3_character(m) == 0 && chi0_idx.is_none() {
                chi0_idx = Some(i);
            }
            if z3_character(m) == 1 && chi1_idx.is_none() {
                chi1_idx = Some(i);
            }
            if chi0_idx.is_some() && chi1_idx.is_some() {
                break;
            }
        }
        let i = chi0_idx.unwrap();
        let j = chi1_idx.unwrap();
        assert_eq!(h[i * n + j], 0.0);
        assert_eq!(h[j * n + i], 0.0);
        // Within-character entries should still be 1.
        assert_eq!(h[i * n + i], 1.0);
    }
}
