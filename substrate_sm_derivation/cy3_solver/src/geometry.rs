//! Calabi-Yau 3-fold geometry as a *first-class candidate-discrimination
//! parameter*.
//!
//! Why this module exists
//! ----------------------
//!
//! The substrate-physics book commits to "the substrate-mathematical-
//! object selects a CY3 + bundle" but leaves the **specific** manifold
//! as the discrimination target — that is literally the program name
//! (`cy3_substrate_discrimination`). Before this module existed,
//! [`crate::cicy_sampler`] hardcoded the canonical Tian-Yau triple
//! `(3,0)+(0,3)+(1,1)` while [`crate::zero_modes`] hardcoded a
//! Schoen-class bicubic `(3,1)+(1,3)`. Both are valid CY3 constructions
//! but they're **different manifolds**, so the two pipeline stages were
//! computing on incompatible geometries — `compute_5sigma_score` mixed
//! them and produced an incoherent result.
//!
//! This module gives both stages a single source of truth: the
//! [`CicyGeometry`] descriptor, which carries
//!
//!   * the ambient projective factors `Π CP^{n_i}`,
//!   * the defining-relation multi-bidegrees `d_1, …, d_N` with
//!     `Σ d_i = (n_1+1, …, n_k+1)` (the Calabi-Yau condition),
//!   * the discrete-quotient action data (order `|Γ|` and a label;
//!     the explicit action is consumed by [`crate::automorphism`]),
//!   * Hodge numbers `h^{1,1}`, `h^{2,1}` and `χ(X)` upstairs (i.e.
//!     before quotient).
//!
//! Both [`crate::zero_modes`] and [`crate::cicy_sampler`] read from a
//! [`CicyGeometry`] passed by the caller; the [`crate::pipeline`]
//! reads the geometry off the [`crate::pipeline::Candidate`] being
//! scored.
//!
//! ## Built-in candidates
//!
//! Two published Calabi-Yau 3-folds are wired up today:
//!
//!   * [`CicyGeometry::tian_yau_z3`] — three defining relations of
//!     bidegrees `(3,0), (0,3), (1,1)` on `CP^3 × CP^3` with a free
//!     `Z/3` action; `χ_upstairs = -18`, downstairs `χ = -6`
//!     (3 generations) [Tian-Yau 1986].
//!   * [`CicyGeometry::schoen_z3xz3`] — two defining relations of
//!     bidegrees `(3,0,1) + (0,3,1)` on `CP^2 × CP^2 × CP^1`, the
//!     canonical Schoen 1988 fiber-product `B_1 ×_{CP^1} B_2` of two
//!     rational elliptic surfaces (DOI 10.1007/BF01215653). With the
//!     free `Z/3 × Z/3` action of Braun-He-Ovrut-Pantev 2005
//!     (arXiv:hep-th/0501070), `χ_upstairs = 0` and three generations
//!     emerge after Wilson-line `SU(5) → SM` breaking.
//!
//! New candidate geometries can be added by appending another constructor;
//! the data-shape is generic over `N` ambient factors and `M` defining
//! relations.

use serde::{Deserialize, Serialize};

/// Zero-action label for the trivial discrete quotient (no quotient).
pub const QUOTIENT_TRIVIAL: &str = "trivial";

/// First-class CY3 descriptor consumed by both the line-intersection
/// sampler (`cicy_sampler`) and the Koszul/BBW cohomology pipeline
/// (`zero_modes`). Two stages reading from the same [`CicyGeometry`]
/// is the architectural invariant that makes the
/// `compute_5sigma_score` integral coherent.
///
/// All vectors are `Vec` rather than fixed-size arrays so that future
/// candidates with different ambient-factor counts or relation counts
/// can be added without touching the type.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CicyGeometry {
    /// Human-readable label, e.g. `"Tian-Yau Z/3"`.
    pub name: String,

    /// Ambient projective factor dimensions.
    /// For `CP^3 × CP^3` this is `vec![3, 3]`; for `CP^4` alone, `vec![4]`.
    /// The total number of homogeneous coordinates is
    /// `Σ (n_i + 1)`, returned by [`Self::n_coords`].
    pub ambient_factors: Vec<u32>,

    /// Defining-relation multi-bidegrees. Each inner vector has
    /// length `ambient_factors.len()`. The Calabi-Yau condition is
    /// `Σ_i defining_relations[i][j] = ambient_factors[j] + 1` for
    /// every `j` (sum of all relation degrees on factor `j` equals
    /// the canonical-bundle degree of `CP^{n_j}`).
    pub defining_relations: Vec<Vec<i32>>,

    /// Discrete free quotient identifier (`"trivial"`, `"Z3"`,
    /// `"Z3xZ3"`, …). Consumed by [`crate::automorphism`].
    pub quotient_label: String,

    /// `|Γ|`, the order of the discrete free quotient. `1` for the
    /// trivial case.
    pub quotient_order: u32,

    /// `h^{1,1}(X)` upstairs (before quotient).
    pub h11_upstairs: u32,

    /// `h^{2,1}(X)` upstairs.
    pub h21_upstairs: u32,

    /// `χ(X)` upstairs. After the free `Γ` quotient,
    /// `χ_downstairs = χ_upstairs / |Γ|`.
    pub chi_upstairs: i32,
}

impl CicyGeometry {
    /// Total number of homogeneous ambient coordinates,
    /// `Σ (n_i + 1)`. For `CP^3 × CP^3` this is `8`.
    #[inline]
    pub fn n_coords(&self) -> usize {
        self.ambient_factors.iter().map(|&n| (n + 1) as usize).sum()
    }

    /// Number of defining relations cutting out the CY in the ambient.
    #[inline]
    pub fn n_relations(&self) -> usize {
        self.defining_relations.len()
    }

    /// Complex dimension of the CICY: `Σ n_i − N_relations`.
    #[inline]
    pub fn n_fold(&self) -> usize {
        self.ambient_factors.iter().map(|&n| n as usize).sum::<usize>() - self.n_relations()
    }

    /// `χ(X) / |Γ|`, the downstairs Euler characteristic. For an
    /// E_6 GUT compactification `χ_downstairs / 2 = N_generations`.
    #[inline]
    pub fn chi_downstairs(&self) -> i32 {
        self.chi_upstairs / (self.quotient_order as i32).max(1)
    }

    /// Compute `∫_X J_1^{e_1} ⋯ J_k^{e_k}` — the intersection number
    /// of products of hyperplane classes from each ambient `CP^{n_j}`
    /// factor — on the CICY `X`.
    ///
    /// Algorithm: `[X] ∈ H^*( Π CP^{n_j})` is the cup-product of the
    /// relation classes `[d_i] = Σ_j d_{ij} J_j`. The intersection
    /// number is the coefficient of the top class `Π J_j^{n_j}` in the
    /// product
    ///
    /// ```text
    ///     [X] · J_1^{e_1} ⋯ J_k^{e_k}  =  Π_i (Σ_j d_{ij} J_j) · Π_j J_j^{e_j}
    /// ```
    ///
    /// expanded multilinearly using the projective truncation
    /// `J_j^{n_j+1} = 0`. Returns `0` when no top-class term appears
    /// (e.g. when `Σ e_j ≠ n_fold`).
    ///
    /// `exponents` must have length `ambient_factors.len()`.
    ///
    /// **Examples**
    ///
    /// * Tian-Yau Z/3 `(3,0)+(0,3)+(1,1)` on `CP^3 × CP^3`:
    ///   `[X] = (3 J_1)(3 J_2)(J_1+J_2) = 9 (J_1^2 J_2 + J_1 J_2^2)`,
    ///   so `∫_X J_1^2 J_2 = 9`, `∫_X J_1 J_2^2 = 9`.
    /// * Schoen `(3,3)` on `CP^2 × CP^2`:
    ///   `[X] = 3 J_1 + 3 J_2`, so `∫_X J_1^2 J_2 = ∫_X J_1 J_2^2 = 3`.
    pub fn intersection_number(&self, exponents: &[u32]) -> i64 {
        debug_assert_eq!(exponents.len(), self.ambient_factors.len());
        let nf = self.ambient_factors.len();
        // Polynomial in the per-factor hyperplane classes, represented
        // as a flat map from exponent-tuple → coefficient. The
        // exponent-tuple is encoded as a Vec<u32>.
        // Start with the seed `Π J_j^{e_j}` (single term).
        let mut poly: std::collections::HashMap<Vec<u32>, i64> =
            std::collections::HashMap::new();
        poly.insert(exponents.to_vec(), 1);

        // Multiply by each relation class [d_i] = Σ_j d_{ij} J_j.
        for relation in &self.defining_relations {
            let mut next: std::collections::HashMap<Vec<u32>, i64> =
                std::collections::HashMap::new();
            for (exp, coef) in poly.iter() {
                for j in 0..nf {
                    let d_ij = relation[j] as i64;
                    if d_ij == 0 {
                        continue;
                    }
                    let mut new_exp = exp.clone();
                    new_exp[j] = new_exp[j].saturating_add(1);
                    // Truncate: J_j^{n_j + 1} = 0.
                    if new_exp[j] > self.ambient_factors[j] {
                        continue;
                    }
                    *next.entry(new_exp).or_insert(0) += coef * d_ij;
                }
            }
            poly = next;
            if poly.is_empty() {
                return 0;
            }
        }

        // The coefficient of the top class Π J_j^{n_j} is the
        // intersection number.
        let top: Vec<u32> = self.ambient_factors.clone();
        *poly.get(&top).unwrap_or(&0)
    }

    /// Convenience: `∫_X (Σ_j a_j J_j)(Σ_j b_j J_j)(Σ_j c_j J_j)` —
    /// the intersection of three divisor classes on the 3-fold.
    /// Used by Chern-class integration.
    ///
    /// Panics in debug builds if the geometry is not 3-dimensional.
    pub fn triple_intersection(&self, a: &[i32], b: &[i32], c: &[i32]) -> i64 {
        debug_assert_eq!(self.n_fold(), 3);
        debug_assert_eq!(a.len(), self.ambient_factors.len());
        debug_assert_eq!(b.len(), self.ambient_factors.len());
        debug_assert_eq!(c.len(), self.ambient_factors.len());
        let nf = self.ambient_factors.len();
        // Expand (Σ a_i J_i)(Σ b_j J_j)(Σ c_k J_k) and integrate
        // each monomial via `intersection_number`.
        let mut total: i64 = 0;
        for i in 0..nf {
            if a[i] == 0 {
                continue;
            }
            for j in 0..nf {
                if b[j] == 0 {
                    continue;
                }
                for k in 0..nf {
                    if c[k] == 0 {
                        continue;
                    }
                    let mut exp = vec![0u32; nf];
                    exp[i] += 1;
                    exp[j] += 1;
                    exp[k] += 1;
                    let weight = (a[i] as i64) * (b[j] as i64) * (c[k] as i64);
                    total = total.saturating_add(weight.saturating_mul(
                        self.intersection_number(&exp),
                    ));
                }
            }
        }
        total
    }

    /// Returns `true` iff the canonical-bundle (CY) condition
    /// `Σ d_i = (n_1+1, …, n_k+1)` is satisfied.
    pub fn satisfies_calabi_yau_condition(&self) -> bool {
        let nf = self.ambient_factors.len();
        if self.defining_relations.iter().any(|d| d.len() != nf) {
            return false;
        }
        for j in 0..nf {
            let sum: i32 = self.defining_relations.iter().map(|d| d[j]).sum();
            let expected = self.ambient_factors[j] as i32 + 1;
            if sum != expected {
                return false;
            }
        }
        true
    }

    /// Canonical Tian-Yau `Z/3` quotient.
    ///
    /// Cover `K_0 ⊂ CP^3 × CP^3` cut by three relations of bidegrees
    /// `(3,0), (0,3), (1,1)`. The free `Z/3` action
    /// `(z_0, α^2 z_1, α z_2, α z_3) × (w_0, α w_1, α^2 w_2, α^2 w_3)`
    /// (with `α = e^{2πi/3}`) descends to a smooth quotient with
    /// `χ_downstairs = -6` (3 generations).
    ///
    /// References:
    /// * Tian-Yau, *Three-dimensional algebraic manifolds with `c_1 = 0`
    ///   and `χ = -6`* (1986).
    /// * Anderson-Lukas-Palti, arXiv:1106.4804 (heterotic SM context).
    /// * Davies thesis (CERN Inspire) for the explicit `Z/3` action.
    pub fn tian_yau_z3() -> Self {
        let g = Self {
            name: "Tian-Yau Z/3".to_string(),
            ambient_factors: vec![3, 3],
            defining_relations: vec![vec![3, 0], vec![0, 3], vec![1, 1]],
            quotient_label: "Z3".to_string(),
            quotient_order: 3,
            h11_upstairs: 14,
            h21_upstairs: 23,
            chi_upstairs: -18,
        };
        debug_assert!(g.satisfies_calabi_yau_condition());
        debug_assert_eq!(g.n_fold(), 3);
        g
    }

    /// Schoen `Z/3 × Z/3` Calabi-Yau three-fold (fiber-product
    /// construction).
    ///
    /// Cover `X̃ ⊂ CP^2 × CP^2 × CP^1` cut by **two** divisors:
    ///
    /// ```text
    ///     F_1(x, t)  =  p_1(x) · t_0  +  p_2(x) · t_1   bidegree (3, 0, 1)
    ///     F_2(y, t)  =  q_1(y) · t_0  +  q_2(y) · t_1   bidegree (0, 3, 1)
    /// ```
    ///
    /// Sum of bidegrees = `(3, 0, 1) + (0, 3, 1) = (3, 3, 2)` =
    /// `(n_1 + 1, n_2 + 1, n_3 + 1)`, so the canonical-bundle (CY)
    /// condition is satisfied. Complex dimension `(2 + 2 + 1) − 2 = 3` ✓.
    ///
    /// This is the **Schoen 1988 fiber-product** construction
    /// (DOI 10.1007/BF01215653) — `X̃ = B_1 ×_{CP^1} B_2` where
    /// `B_i → CP^1` are rational elliptic surfaces meeting transversely
    /// over `CP^1`. The free `Z/3 × Z/3` action acts independently as
    /// cube roots of unity on the affine coordinates of each `CP^2`
    /// factor (preserving the `t_0, t_1` of the `CP^1`); the quotient
    /// `X = X̃ / (Z/3 × Z/3)` is the standard-model heterotic candidate
    /// of Braun-He-Ovrut-Pantev 2005 (arXiv:hep-th/0501070), with
    /// `χ_downstairs = 0` and three fermion generations after
    /// Wilson-line `SU(5) → SM` breaking around the two `Z/3` factors.
    ///
    /// **Earlier revisions of this entry** used a single bidegree-`(3,3)`
    /// hypersurface in `CP^2 × CP^2` — that produces a different CY3
    /// (one of Schoen's other examples) and is **not** the Schoen Z/3×Z/3
    /// candidate the chapter-8 substrate-physics argument references.
    /// Fixed to the canonical fiber-product form above so the
    /// `Z/3 × Z/3` quotient and the `χ_downstairs = 0` count line up
    /// with [`crate::route34::schoen_geometry`]. Hodge data
    /// `(h^{1,1}, h^{2,1}) = (19, 19)` are the canonical pre-Wilson-
    /// breaking values of the cover (Donagi-He-Ovrut-Reinbacher 2006,
    /// arXiv:hep-th/0512149).
    pub fn schoen_z3xz3() -> Self {
        let g = Self {
            name: "Schoen Z/3 × Z/3 fiber-product".to_string(),
            ambient_factors: vec![2, 2, 1],
            defining_relations: vec![vec![3, 0, 1], vec![0, 3, 1]],
            quotient_label: "Z3xZ3".to_string(),
            quotient_order: 9,
            h11_upstairs: 19,
            h21_upstairs: 19,
            chi_upstairs: 0,
        };
        debug_assert!(g.satisfies_calabi_yau_condition());
        debug_assert_eq!(g.n_fold(), 3);
        g
    }
}

impl Default for CicyGeometry {
    /// Default to the Tian-Yau `Z/3` quotient — that is the line of
    /// candidates the substrate-physics book singles out (`χ = -6`,
    /// 3 generations).
    fn default() -> Self {
        Self::tian_yau_z3()
    }
}

/// Enumerate every subset `S ⊂ {0, …, n−1}` of size `k`, in
/// lexicographic order, calling `f(S)` once per subset. Used by the
/// generalized Koszul chase in [`crate::zero_modes`] to walk the
/// `∧^k B` term of the Koszul resolution.
pub fn for_each_subset_of_size<F: FnMut(&[usize])>(n: usize, k: usize, mut f: F) {
    if k > n {
        return;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        f(&idx);
        if k == 0 {
            return;
        }
        // Find the rightmost index that can be incremented.
        let mut i = k;
        while i > 0 {
            i -= 1;
            if idx[i] < n - k + i {
                idx[i] += 1;
                for j in (i + 1)..k {
                    idx[j] = idx[j - 1] + 1;
                }
                break;
            }
            if i == 0 {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tian_yau_z3_satisfies_cy_condition() {
        let g = CicyGeometry::tian_yau_z3();
        assert!(g.satisfies_calabi_yau_condition());
        assert_eq!(g.n_coords(), 8);
        assert_eq!(g.n_relations(), 3);
        assert_eq!(g.n_fold(), 3);
        assert_eq!(g.chi_downstairs(), -6);
    }

    #[test]
    fn schoen_z3xz3_satisfies_cy_condition() {
        let g = CicyGeometry::schoen_z3xz3();
        assert!(g.satisfies_calabi_yau_condition());
        // Schoen fiber product: CP^2 × CP^2 × CP^1
        // → coords = (2+1) + (2+1) + (1+1) = 8
        assert_eq!(g.n_coords(), 8);
        // Two defining relations: (3,0,1) + (0,3,1)
        assert_eq!(g.n_relations(), 2);
        // Complex dim: (2 + 2 + 1) − 2 = 3
        assert_eq!(g.n_fold(), 3);
        assert_eq!(g.chi_downstairs(), 0);
    }

    #[test]
    fn default_is_tian_yau() {
        assert_eq!(CicyGeometry::default(), CicyGeometry::tian_yau_z3());
    }

    #[test]
    fn tian_yau_intersection_numbers() {
        let g = CicyGeometry::tian_yau_z3();
        // [X] = (3 J_1)(3 J_2)(J_1+J_2) = 9 (J_1^2 J_2 + J_1 J_2^2)
        assert_eq!(g.intersection_number(&[2, 1]), 9);
        assert_eq!(g.intersection_number(&[1, 2]), 9);
        assert_eq!(g.intersection_number(&[3, 0]), 0);
        assert_eq!(g.intersection_number(&[0, 3]), 0);
    }

    #[test]
    fn schoen_intersection_numbers() {
        let g = CicyGeometry::schoen_z3xz3();
        // [X̃] = (3 J_1 + J_t)(3 J_2 + J_t)
        //     = 9 J_1 J_2 + 3 J_1 J_t + 3 J_2 J_t  (J_t^2 = 0 on CP^1)
        // Top class on the ambient CP^2 × CP^2 × CP^1 is J_1^2 J_2^2 J_t = 1.
        // So for any (a, b, c) with a+b+c = 3, ∫_X J_1^a J_2^b J_t^c
        // = (coefficient of J_1^{2-a} J_2^{2-b} J_t^{1-c} in [X̃]) read off
        // via the three nonzero combinations.
        assert_eq!(g.intersection_number(&[1, 1, 1]), 9); // J_1 J_2 J_t
        assert_eq!(g.intersection_number(&[2, 1, 0]), 3); // J_1^2 J_2
        assert_eq!(g.intersection_number(&[1, 2, 0]), 3); // J_1 J_2^2
        // Vanishing combinations.
        assert_eq!(g.intersection_number(&[3, 0, 0]), 0);
        assert_eq!(g.intersection_number(&[0, 3, 0]), 0);
        assert_eq!(g.intersection_number(&[0, 0, 3]), 0); // J_t^3 = 0
    }

    #[test]
    fn triple_intersection_matches_intersection_number() {
        // ∫_X J_1 · J_1 · J_2 should equal ∫_X J_1^2 J_2.
        let g = CicyGeometry::tian_yau_z3();
        let j1 = vec![1, 0];
        let j2 = vec![0, 1];
        assert_eq!(
            g.triple_intersection(&j1, &j1, &j2),
            g.intersection_number(&[2, 1])
        );
    }

    #[test]
    fn cy_condition_rejects_invalid_bidegrees() {
        let mut bad = CicyGeometry::tian_yau_z3();
        bad.defining_relations[0][0] = 2; // (2,0) instead of (3,0)
        assert!(!bad.satisfies_calabi_yau_condition());
    }

    #[test]
    fn subset_enumeration_is_complete_and_ordered() {
        let mut got: Vec<Vec<usize>> = Vec::new();
        for_each_subset_of_size(4, 2, |s| got.push(s.to_vec()));
        assert_eq!(
            got,
            vec![
                vec![0, 1],
                vec![0, 2],
                vec![0, 3],
                vec![1, 2],
                vec![1, 3],
                vec![2, 3],
            ]
        );
    }

    #[test]
    fn subset_enumeration_size_zero_yields_one_empty() {
        let mut count = 0;
        for_each_subset_of_size(5, 0, |s| {
            assert!(s.is_empty());
            count += 1;
        });
        assert_eq!(count, 1);
    }

    #[test]
    fn subset_enumeration_size_equals_n_yields_one_full() {
        let mut count = 0;
        for_each_subset_of_size(3, 3, |s| {
            assert_eq!(s, &[0, 1, 2]);
            count += 1;
        });
        assert_eq!(count, 1);
    }
}
