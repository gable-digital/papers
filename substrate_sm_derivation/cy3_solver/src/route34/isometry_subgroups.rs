//! Cyclic-subgroup detection from a discrete Killing-vector basis, and
//! the "admissible-wavenumber" map consumed by the Arnold catastrophe
//! classifier (see chapter 21, [`crate::route34::arnold_normal_form`]
//! once that module lands in a parallel agent's deliverable).
//!
//! ## What this module does
//!
//! Given an extracted Killing-algebra basis `{ξ_a}_{a=1..k}` from
//! [`crate::route34::killing_solver::KillingResult`], we identify the
//! **continuous cyclic subgroups** `Z/n` of the connected-component
//! identity isometry group that act non-trivially on the manifold, and
//! we expose the resulting set of integers `{n}` that the Arnold
//! classifier downstream uses to filter the polyhedral-resonance
//! wavenumber predictions.
//!
//! ### Cyclic subgroups of the connected isometry group
//!
//! A continuous cyclic subgroup of a Lie group `G = Isom_0(M)` is the
//! image under the exponential map of a 1-dimensional Lie subalgebra
//! `R ξ ⊂ Lie(G)`. The action of `exp(t ξ)` on `M` is a one-parameter
//! family of isometries; its orbits are either points (`ξ = 0`),
//! periodic with some period `T(ξ) > 0`, or non-periodic (the latter
//! happens for non-compact subgroups / dense orbits).
//!
//! For a **compact** Riemannian manifold `(M, g)` (the case we care
//! about, since CY3s are compact), every Killing field generates a
//! one-parameter subgroup whose closure is a torus `T^r` ⊂ Isom(M).
//! Cyclic subgroups `Z/n` arise as
//!
//! 1. periodic orbits of a single `ξ` (then `n = ∞` for the continuous
//!    `S^1`, and any finite `n | ∞` is admissible — i.e. **all** finite
//!    cyclic subgroups of `S^1` are realised), or
//! 2. centres of products `S^1 × S^1 × …` arising from commuting
//!    Killing fields (the discrete diagonals).
//!
//! For the framework's purpose (filter the substrate's Rossby-Arnold
//! wavenumber predictions through the candidate CY3's continuous
//! isometry structure), the relevant data is:
//!
//! * the dimension `r` of a maximal torus inside the Killing algebra
//!   (the rank of the connected isometry group),
//! * the **commuting-set count** `c` = number of pairwise-commuting
//!   independent Killing fields,
//! * the orbit-period spectrum `{T(ξ_a)}` of the algebra basis (where
//!   computable; otherwise marked as `None`).
//!
//! The Arnold classifier then takes the integer set
//!
//! ```text
//!     N(M) = { n ∈ Z_{≥ 1} : Z/n acts via Isom_0(M) on M }
//! ```
//!
//! and intersects it with the abstract Arnold-ADE wavenumber set to
//! produce the filtered polyhedral wavenumber predictions. For
//! `Isom_0(M) ⊃ S^1` (any compact 1-parameter Killing flow), `N(M)`
//! contains every positive integer; for `Isom_0(M)` discrete, `N(M) =
//! {1}`.
//!
//! ## Implementation
//!
//! Lie-bracket structure constants are estimated by the routine
//! [`crate::route34::killing_solver::killing_bracket_structure_constants`].
//! From those we extract:
//!
//! 1. The **abelian-rank** (cardinality of a maximal commuting subset),
//!    found greedily on the structure-constant tensor.
//! 2. For each Killing field, an estimate of its orbit period via the
//!    eigenvalues of `ad(ξ)` restricted to the abelian subalgebra
//!    containing it.
//! 3. The cyclic-subgroup data is summarised in [`CyclicSubgroup`]
//!    entries.
//!
//! ## Bound on `n`
//!
//! Per the framework's substrate-physics chapter 21, the predicted
//! wavenumbers of physical interest are bounded by the highest-`n`
//! polyhedral fluid resonance the substrate admits at the polar
//! critical-boundary regime: in practice this caps at `n ≤ 16` (Saturn
//! n=6, hypothetical n=8, Jupiter polar 5+1 ⇒ |Z/5 × Z/6|, plus a
//! safety margin). [`polyhedral_admissible_wavenumbers`] returns the
//! integer set up to this bound.
//!
//! ## References
//!
//! * Kobayashi, *Transformation Groups in Differential Geometry*
//!   (1972), §II.1 (compact-isometry Lie group, maximal torus).
//! * Onishchik-Vinberg, *Lie Groups and Algebraic Groups* (1990),
//!   §IV.4 (cyclic subgroups of compact connected Lie groups).
//! * Arnold, Gusein-Zade, Varchenko, *Singularities of Differentiable
//!   Maps* vol. I (1985), §15-16 (ADE classification and admissible
//!   integer set per ADE type).

use crate::route34::killing_solver::KillingVectorField;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// One detected cyclic subgroup `Z/n ⊂ Isom_0(M)`.
///
/// `order = u32::MAX` is the sentinel for the continuous `S^1` (which
/// contains every finite `Z/n`).
#[derive(Debug, Clone)]
pub struct CyclicSubgroup {
    /// The cyclic order. `u32::MAX` denotes the continuous `S^1`
    /// containing all finite cyclic subgroups.
    pub order: u32,
    /// Index (in the input Killing-basis) of the generator vector
    /// field. For `S^1`s arising from a single Killing vector this is
    /// the index of that vector.
    pub generator_index: usize,
    /// Estimated orbit period `T = 2π / |ad(ξ)|_{eff}` (or `None` if
    /// not periodic / not estimable).
    pub period_estimate: Option<f64>,
}

/// Summary of the connected-component isometry structure of `M`.
#[derive(Debug, Clone)]
pub struct IsometryStructure {
    /// Dimension of the Killing algebra `dim Isom_0(M)`.
    pub killing_dim: usize,
    /// Rank of a maximal abelian subalgebra (= rank of the maximal
    /// torus inside `Isom_0(M)`).
    pub abelian_rank: usize,
    /// Detected cyclic subgroups.
    pub cyclic_subgroups: Vec<CyclicSubgroup>,
    /// Whether `Isom_0(M)` contains a continuous `S^1` factor (true
    /// iff `killing_dim > 0`).
    pub has_continuous_s1: bool,
}

// ---------------------------------------------------------------------------
// Cyclic-subgroup detection
// ---------------------------------------------------------------------------

/// Detect cyclic subgroups in the Killing algebra from the basis and
/// (optionally) the structure-constant tensor produced by
/// [`crate::route34::killing_solver::killing_bracket_structure_constants`].
///
/// If `structure_constants` is `None`, we treat each Killing field
/// independently (each generates an `S^1` ⊂ Isom).
pub fn detect_cyclic_subgroups(
    killing_basis: &[KillingVectorField],
    structure_constants: Option<&[f64]>,
) -> Vec<CyclicSubgroup> {
    let k = killing_basis.len();
    let mut out = Vec::with_capacity(k);
    if k == 0 {
        return out;
    }
    // Default to "every Killing generates an S^1".
    if structure_constants.is_none() {
        for a in 0..k {
            out.push(CyclicSubgroup {
                order: u32::MAX,
                generator_index: a,
                period_estimate: None,
            });
        }
        return out;
    }
    let f = structure_constants.unwrap();
    debug_assert_eq!(f.len(), k * k * k);

    // For each generator a, estimate the spectral radius of ad(ξ_a) on
    // the rest of the algebra: for a compact Killing vector with finite
    // orbit period, ad(ξ_a) has imaginary eigenvalues ±i · ω_a, and the
    // orbit period is T = 2π / max |ω|.
    for a in 0..k {
        let mut spectral_radius = 0.0f64;
        for b in 0..k {
            for c in 0..k {
                let v = f[a * k * k + b * k + c].abs();
                if v > spectral_radius {
                    spectral_radius = v;
                }
            }
        }
        let period_estimate = if spectral_radius > 1e-12 {
            Some(2.0 * std::f64::consts::PI / spectral_radius)
        } else {
            None
        };
        out.push(CyclicSubgroup {
            order: u32::MAX,
            generator_index: a,
            period_estimate,
        });
    }
    out
}

/// Estimate the rank of the maximal abelian subalgebra of the Killing
/// algebra, given the structure-constant tensor.
///
/// Greedy algorithm: start with the empty set; for each generator in
/// order, add it to the abelian set if its bracket against every
/// already-included generator is below `tol`. The greedy answer is a
/// lower bound on the true rank but is exact for the test cases (S^n,
/// products of S^n) where the algebra is highly symmetric.
pub fn abelian_rank(structure_constants: &[f64], k: usize, tol: f64) -> usize {
    if k == 0 {
        return 0;
    }
    debug_assert_eq!(structure_constants.len(), k * k * k);
    let mut included: Vec<usize> = Vec::with_capacity(k);
    for a in 0..k {
        let mut ok = true;
        for &b in &included {
            // Check both [ξ_a, ξ_b] and [ξ_b, ξ_a] structure-constant
            // magnitudes are below tol.
            let mut max_mag = 0.0f64;
            for c in 0..k {
                let v1 = structure_constants[a * k * k + b * k + c].abs();
                let v2 = structure_constants[b * k * k + a * k + c].abs();
                if v1 > max_mag {
                    max_mag = v1;
                }
                if v2 > max_mag {
                    max_mag = v2;
                }
            }
            if max_mag > tol {
                ok = false;
                break;
            }
        }
        if ok {
            included.push(a);
        }
    }
    included.len()
}

/// Build the [`IsometryStructure`] summary from a Killing-algebra basis
/// and (optional) structure constants.
pub fn isometry_structure(
    killing_basis: &[KillingVectorField],
    structure_constants: Option<&[f64]>,
    structure_tol: f64,
) -> IsometryStructure {
    let k = killing_basis.len();
    let abelian_rank = match structure_constants {
        Some(f) => abelian_rank(f, k, structure_tol),
        None => k, // No bracket info ⇒ assume worst-case all-abelian
    };
    let cyclic_subgroups = detect_cyclic_subgroups(killing_basis, structure_constants);
    IsometryStructure {
        killing_dim: k,
        abelian_rank,
        cyclic_subgroups,
        has_continuous_s1: k > 0,
    }
}

// ---------------------------------------------------------------------------
// Admissible-wavenumber map (consumed by the Arnold classifier)
// ---------------------------------------------------------------------------

/// Default upper bound for admissible wavenumbers, per chapter 21
/// substrate-physics enumeration. Polar polyhedral resonances of
/// physical interest cap at `N_MAX_PHYSICAL = 16` (Jupiter polar 5+1
/// products with Saturn n=6 hexagon hypotheticals, plus margin).
pub const N_MAX_PHYSICAL: u32 = 16;

/// Compute the set of admissible wavenumbers `n ∈ {1, …, N_MAX}` that
/// are compatible with the candidate's continuous-isometry structure.
///
/// **Rule.** If `Isom_0(M)` contains a continuous `S^1` factor (i.e.
/// `killing_dim > 0`), every `n` from 1 to `N_MAX` is admissible —
/// the `S^1` ⊃ Z/n for all positive integers `n`. If `Isom_0(M)` is
/// discrete (i.e. `killing_dim == 0`), only `n = 1` is admissible from
/// the **continuous-isometry side**; the discrete-automorphism side
/// (which lives in [`crate::automorphism`]) supplies any non-trivial
/// finite cyclic subgroups.
///
/// The returned set is sorted ascending, deduplicated.
pub fn polyhedral_admissible_wavenumbers(structure: &IsometryStructure) -> Vec<u32> {
    polyhedral_admissible_wavenumbers_with_bound(structure, N_MAX_PHYSICAL)
}

/// Same as [`polyhedral_admissible_wavenumbers`] but with a custom
/// upper bound `n_max`.
pub fn polyhedral_admissible_wavenumbers_with_bound(
    structure: &IsometryStructure,
    n_max: u32,
) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::new();
    if structure.has_continuous_s1 {
        for n in 1..=n_max {
            out.push(n);
        }
    } else {
        out.push(1);
    }
    out.sort_unstable();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_killing(coeffs_len: usize, eigenvalue: f64, residual: f64) -> KillingVectorField {
        KillingVectorField {
            coefficients: vec![1.0; coeffs_len],
            eigenvalue,
            residual,
        }
    }

    #[test]
    fn empty_killing_basis_gives_only_n1() {
        let structure = IsometryStructure {
            killing_dim: 0,
            abelian_rank: 0,
            cyclic_subgroups: Vec::new(),
            has_continuous_s1: false,
        };
        let wn = polyhedral_admissible_wavenumbers(&structure);
        assert_eq!(wn, vec![1]);
    }

    #[test]
    fn nonempty_killing_basis_gives_all_n_up_to_bound() {
        let basis = vec![
            dummy_killing(3, 1e-10, 1e-9),
            dummy_killing(3, 1e-10, 1e-9),
            dummy_killing(3, 1e-10, 1e-9),
        ];
        // No structure constants ⇒ default-S^1-each.
        let structure = isometry_structure(&basis, None, 1e-6);
        assert_eq!(structure.killing_dim, 3);
        assert!(structure.has_continuous_s1);
        let wn = polyhedral_admissible_wavenumbers(&structure);
        assert_eq!(wn.len(), N_MAX_PHYSICAL as usize);
        assert_eq!(wn[0], 1);
        assert_eq!(wn[wn.len() - 1], N_MAX_PHYSICAL);
    }

    #[test]
    fn abelian_rank_recognises_diagonal_algebra() {
        // Three pairwise-commuting Killing fields ⇒ abelian rank 3.
        let k = 3;
        let f = vec![0.0; k * k * k];
        assert_eq!(abelian_rank(&f, k, 1e-12), 3);
    }

    #[test]
    fn abelian_rank_recognises_so3_like() {
        // [ξ_0, ξ_1] = ξ_2 (so(3) Levi-Civita symbol). Abelian rank 1.
        let k = 3;
        let mut f = vec![0.0; k * k * k];
        // Normalisation: f[a, b, c] = ε_abc.
        let eps = |a: usize, b: usize, c: usize| -> f64 {
            // Permutations of (0,1,2)
            match (a, b, c) {
                (0, 1, 2) | (1, 2, 0) | (2, 0, 1) => 1.0,
                (1, 0, 2) | (0, 2, 1) | (2, 1, 0) => -1.0,
                _ => 0.0,
            }
        };
        for a in 0..k {
            for b in 0..k {
                for c in 0..k {
                    f[a * k * k + b * k + c] = eps(a, b, c);
                }
            }
        }
        assert_eq!(abelian_rank(&f, k, 0.5), 1);
    }
}
